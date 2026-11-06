//! Auto-labelling (`blue-onyx-prism train autolabel`, the Review and Training pages' Auto-label
//! buttons; docs/TRAINING.md "Auto-label"): proposals for the frames of the dataset folder,
//! written into their label files as pending `source: auto` boxes the reviewer accepts or
//! rejects. Two modes:
//!
//! - **Labelers** (default): `scripts/autolabel.py` in the training environment runs
//!   MM-Grounding-DINO or, in its seat, SAM 3 ([`AutolabelRequest::sam3`], gated) (open vocabulary,
//!   prompts from `collect.classes` plus `training.synonyms`), RF-DETR L (COCO classes) and fuses
//!   them (WBF), then asks Qwen3-VL "is there a real <label>" about each box above a score and ranks animal species
//!   with SigLIP 2. The service triages every box ([`triage`]): `accept` only when both labelers
//!   agree at a score of at least 0.6 and no check objects; night person / animal boxes and SAM 3
//!   boxes always `review`.
//! - **A model** (`--with-model <id>`, any model of [`models`]: configured, a local file, or a
//!   catalog model the run downloads or exports first and adds to the config disabled): the
//!   service's own inference path (the benchmark pipeline: decode, preprocess, infer,
//!   postprocess) labels the frames on the model's device or the run's `device`; its labels are
//!   mapped onto the class list ([`triage::map_label`]); its boxes are `review`, or `accept` at a
//!   pre-accept score (`model_accept_score`; night person / animal boxes stay `review`). By
//!   default it labels unreviewed frames (first-pass labelling); with `status=any` it is the
//!   self-training loop: frames whose reviewed labels it disagrees with (a new box, a missed
//!   box, a class change at IoU 0.5) go back to review with a `disagreement` note, which the
//!   Review queue sorts first. With `missing_only` it looks for real objects without a label on
//!   reviewed frames and nothing else: frames the model trained on are left out
//!   ([`trained_frames`]), only a new object reopens a frame ([`triage::apply_missing`]), no
//!   `auto_run` marker is written (several models can run in turn, and a rerun re-detects).
//!
//! The service owns every label-file write ([`crate::collect::labels::merge_proposals`],
//! [`triage::apply_missing`]); the Python side only returns proposals as JSONL. Nothing is accepted
//! automatically, and hard negatives come from the reviewer: `apply_missing` removes rejected boxes
//! only when no reviewer judged them.
//!
//! - [`catalog`]: the class catalog of the Review page's picker, [`catalog::coco_class`] and
//!   the classes the enabled models report.
//! - [`triage`]: pure: the triage rule, proposals to label boxes, the re-label disagreement.
//! - [`models`]: the models a run can use (`GET /v1/autolabel/models`) and how a chosen one
//!   becomes a (disabled) config entry.
//! - [`relabel`]: detection with a model of the config in this process.
//! - [`job`]: the stage machine ([`AutolabelJob`], a [`super::task::Task`]).
//! - [`preview`]: what a run would do, from its own first stage ([`preview_autolabel`]).

pub mod catalog;
pub mod job;
pub mod models;
pub mod preview;
pub mod relabel;
pub mod triage;

pub use job::AutolabelJob;

use super::capability::Capability;
use crate::collect::labels::queue::Filters;
use crate::config::Config;
use crate::resources::catalog::{self as res_catalog, Platform, Resource};
use crate::resources::export::train::{self as env, TrainPaths};
use crate::resources::hf_token::{self, LoginCommand, Lookup};
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Duration;

/// Default most frames of one run.
pub const DEFAULT_LIMIT: usize = 200;
/// Most frames of one run.
pub const MAX_LIMIT: usize = 20_000;
/// Threshold of a `missing_only` run that leaves it at [`models::DEFAULT_THRESHOLD`]: every box it
/// proposes reopens a reviewed frame, so it should be a sure one.
pub const MISSING_THRESHOLD: f32 = 0.5;

/// Prompt phrases of the default classes (`training.synonyms` adds or replaces entries).
pub const DEFAULT_SYNONYMS: [(&str, &[&str]); 14] = [
    ("person", &["person", "pedestrian"]),
    ("car", &["car", "van", "pickup truck"]),
    ("truck", &["truck", "lorry"]),
    ("bicycle", &["bicycle"]),
    ("motorcycle", &["motorcycle", "scooter"]),
    ("dog", &["dog"]),
    ("cat", &["cat"]),
    ("raccoon", &["raccoon"]),
    ("opossum", &["opossum", "Virginia opossum"]),
    ("skunk", &["skunk"]),
    ("fox", &["fox", "red fox"]),
    ("deer", &["deer", "white-tailed deer"]),
    ("squirrel", &["squirrel"]),
    ("bird", &["bird"]),
];

/// Vehicle class names (the VLM check expects "vehicle").
const VEHICLES: [&str; 9] = [
    "car",
    "truck",
    "bus",
    "bicycle",
    "motorcycle",
    "train",
    "boat",
    "airplane",
    "van",
];

/// Animal class names (the VLM check expects "animal"; SigLIP ranks them; at night always
/// reviewed).
const ANIMALS: [&str; 33] = [
    "dog", "cat", "raccoon", "opossum", "skunk", "fox", "deer", "squirrel", "bird", "horse",
    "sheep", "cow", "bear", "coyote", "rabbit", "hare", "badger", "boar", "wolf", "elephant",
    "zebra", "giraffe", "animal", "mouse", "rat", "goat", "chicken", "duck", "goose", "owl",
    "bobcat", "cougar", "moose",
];

/// What the VLM answer of a class is checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Category {
    Person,
    Vehicle,
    Animal,
    /// Any other class: no VLM check, no species ranking.
    Object,
}

impl Category {
    pub fn of(label: &str) -> Self {
        let l = label.trim().to_ascii_lowercase();
        if l == "person" {
            Category::Person
        } else if VEHICLES.contains(&l.as_str()) {
            Category::Vehicle
        } else if ANIMALS.contains(&l.as_str()) {
            Category::Animal
        } else {
            Category::Object
        }
    }
}

/// The prompt phrases of `classes`: `overrides` (`training.synonyms`) first, else the defaults,
/// else the class name itself.
pub fn synonyms(
    classes: &[String],
    overrides: &BTreeMap<String, Vec<String>>,
) -> BTreeMap<String, Vec<String>> {
    classes
        .iter()
        .map(|c| {
            let v = overrides
                .get(c)
                .filter(|v| !v.is_empty())
                .cloned()
                .or_else(|| {
                    DEFAULT_SYNONYMS
                        .iter()
                        .find(|(k, _)| k.eq_ignore_ascii_case(c))
                        .map(|(_, v)| v.iter().map(|s| (*s).to_string()).collect())
                })
                .unwrap_or_else(|| vec![c.clone()]);
            (c.clone(), v)
        })
        .collect()
}

/// MM-Grounding-DINO checkpoint size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GdinoSize {
    Tiny,
    Base,
    Large,
}

impl GdinoSize {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "tiny" | "t" | "swin-t" => Some(GdinoSize::Tiny),
            "base" | "b" | "swin-b" => Some(GdinoSize::Base),
            "large" | "l" | "swin-l" => Some(GdinoSize::Large),
            _ => None,
        }
    }
    pub const fn key(self) -> &'static str {
        match self {
            GdinoSize::Tiny => "mm-gdino-tiny",
            GdinoSize::Base => "mm-gdino-base",
            GdinoSize::Large => "mm-gdino-large",
        }
    }
}

/// Qwen3-VL checkpoint size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VlmSize {
    #[serde(rename = "qwen3-vl-2b")]
    Qwen2b,
    #[serde(rename = "qwen3-vl-4b")]
    Qwen4b,
    #[serde(rename = "qwen3-vl-8b")]
    Qwen8b,
}

impl VlmSize {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "2b" | "qwen3-vl-2b" => Some(VlmSize::Qwen2b),
            "4b" | "qwen3-vl-4b" => Some(VlmSize::Qwen4b),
            "8b" | "qwen3-vl-8b" => Some(VlmSize::Qwen8b),
            _ => None,
        }
    }
    pub const fn key(self) -> &'static str {
        match self {
            VlmSize::Qwen2b => "qwen3-vl-2b",
            VlmSize::Qwen4b => "qwen3-vl-4b",
            VlmSize::Qwen8b => "qwen3-vl-8b",
        }
    }
}

/// Which frames a run labels: the Review queue's filters (`bucket`, `scene`, `camera`,
/// `label` (any of several classes, separated by `|`: `label=person|car`), `status`, `source`,
/// `auto`; `auto=overlaps` and `auto=both` need [`Filters::wanted`], which [`AutolabelJob`] sets
/// from the config as the Review page does), `import=<id>|any` (imported frames), `again=1` (also
/// frames these labelers already looked at, see [`crate::collect::labels::AutoRun`]) and `limit`
/// (most frames, in queue order). Written `key=value` pairs separated by commas
/// (`scene=ir,status=unreviewed,limit=20`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameFilter {
    pub query: BTreeMap<String, String>,
    pub limit: usize,
}

impl FrameFilter {
    /// Parse `text` (empty = defaults: every frame, unreviewed). `default_status` is used when
    /// `status` is not given.
    pub fn parse(text: &str, default_status: &str) -> Result<Self> {
        let mut query = BTreeMap::new();
        let mut limit = DEFAULT_LIMIT;
        for part in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (k, v) = part
                .split_once('=')
                .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
                .ok_or_else(|| anyhow::anyhow!("'{part}' is not key=value"))?;
            match k.as_str() {
                "limit" => {
                    limit = v
                        .parse()
                        .map_err(|_| anyhow::anyhow!("limit: '{v}' is not a number"))?;
                }
                "again" => {
                    if !matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes") {
                        bail!(
                            "again: '{v}' is not 1, true or yes (leave it out to skip frames \
                               these labelers already looked at)"
                        );
                    }
                    query.insert(k, "1".into());
                }
                "bucket" | "scene" | "camera" | "label" | "status" | "source" | "auto"
                | "import" => {
                    query.insert(k, v);
                }
                other => bail!(
                    "unknown frame filter '{other}' (bucket, scene, camera, label, status, \
                     source, auto, import, again, limit)"
                ),
            }
        }
        query.entry("source".into()).or_insert_with(|| "all".into());
        query
            .entry("status".into())
            .or_insert_with(|| default_status.into());
        if !(1..=MAX_LIMIT).contains(&limit) {
            bail!("limit must be between 1 and {MAX_LIMIT}");
        }
        let f = Self { query, limit };
        f.filters()?;
        Ok(f)
    }

    /// The Review queue filters.
    pub fn filters(&self) -> Result<Filters> {
        let mut q = self.query.clone();
        q.remove("import");
        q.remove("again");
        // `|` separates the classes here (a comma separates the filters); the queue takes a
        // comma list.
        if let Some(l) = q.get_mut("label") {
            *l = l.replace('|', ",");
        }
        Filters::from_query(&q).map_err(anyhow::Error::msg)
    }

    /// `again=1`: label frames these labelers already looked at too.
    pub fn again(&self) -> bool {
        self.query.contains_key("again")
    }

    /// `import=` value: `any` or an import id.
    pub fn import(&self) -> Option<&str> {
        self.query.get("import").map(String::as_str)
    }

    /// Whether a frame with `source` (`import:<id>` or none) passes the `import` filter.
    pub fn accepts_source(&self, source: Option<&str>) -> bool {
        match self.import() {
            None => true,
            Some("any") => source.is_some_and(|s| s.starts_with("import:")),
            Some(id) => source.and_then(|s| s.strip_prefix("import:")) == Some(id),
        }
    }

    /// `scene=ir, status=unreviewed, limit 20`.
    pub fn describe(&self) -> String {
        let mut s: Vec<String> = self.query.iter().map(|(k, v)| format!("{k}={v}")).collect();
        s.push(format!("limit {}", self.limit));
        s.join(", ")
    }
}

impl Default for FrameFilter {
    fn default() -> Self {
        Self {
            query: [
                ("source".to_string(), "all".to_string()),
                ("status".to_string(), "unreviewed".to_string()),
            ]
            .into(),
            limit: DEFAULT_LIMIT,
        }
    }
}

/// What to label how.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutolabelRequest {
    pub frames: FrameFilter,
    /// MM-Grounding-DINO (None = off).
    pub gdino: Option<GdinoSize>,
    /// SAM 3 in MM-Grounding-DINO's seat (`gdino` is None then). Its boxes never triage `accept`.
    #[serde(default)]
    pub sam3: bool,
    /// RF-DETR L for the COCO classes.
    pub rfdetr: bool,
    /// Qwen3-VL crop check (None = off).
    pub vlm: Option<VlmSize>,
    /// SigLIP 2 species ranking.
    pub siglip: bool,
    /// Label with this model instead of the labelers: a configured model's name, `local:<file>`
    /// or `resource:<catalog id>` ([`models::resolve`]).
    #[serde(default)]
    pub with_model: Option<String>,
    /// Score threshold of `with_model`'s boxes.
    pub threshold: f32,
    /// Device of `with_model` for this run only (a device spec; None = the model's device).
    #[serde(default)]
    pub device: Option<String>,
    /// GPU of the labelers (`cuda:N` of [`Capability::cuda_options`]; None = the automatic
    /// pick). [`plan_autolabel`] stores the device it resolved to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    /// Pre-accept `with_model`'s boxes at this score or more (None = every box `review`).
    #[serde(default)]
    pub model_accept_score: Option<f32>,
    /// Only look for real objects with no label on reviewed frames, with `with_model`
    /// ([`triage::apply_missing`], [`plan_autolabel`] sets the frames it needs).
    #[serde(default)]
    pub missing_only: bool,
    /// The user confirmed a restrictive (non-commercial / personal use) licence of a catalog
    /// model the run installs or adds.
    #[serde(default)]
    pub confirm_license: bool,
    pub gdino_threshold: f32,
    /// SAM 3's own score threshold, before fusion: higher means fewer boxes and missed faint
    /// objects. Not part of the frames' marker, so a rerun at another value needs `again=1`.
    #[serde(default = "default_sam3_threshold")]
    pub sam3_threshold: f32,
    pub rfdetr_threshold: f32,
    /// Triage: both labelers agree at this fused score or more to `accept`.
    pub accept_score: f32,
    /// The class list (default `collect.classes`).
    #[serde(default)]
    pub classes: Option<Vec<String>>,
    #[serde(default)]
    pub allow_large: bool,
    /// A cancel while the frames are labelled keeps the finished ones: they are triaged and
    /// written like a finished run's (with the `auto_run` marker, so a rerun of the same
    /// labelers resumes after them) and the run ends with a "stopped by Cancel" summary. Off: a
    /// cancel discards everything.
    #[serde(default = "yes")]
    pub keep_on_cancel: bool,
    /// Ask Qwen3-VL about a frame's boxes in batches ([`VLM_BATCH`] per call) instead of one
    /// by one.
    #[serde(default = "yes")]
    pub vlm_batch: bool,
    /// Check only the boxes with a fused score of at least this with Qwen3-VL (None = every box;
    /// a request stored before this default keeps None). [`VLM_SAMPLE`] of the boxes below it are
    /// asked anyway, so the cutoff can be audited.
    #[serde(default)]
    pub vlm_min_score: Option<f32>,
    /// Load and decode the next frame on a background thread while the GPU labels this one.
    #[serde(default = "yes")]
    pub prefetch: bool,
}

/// Boxes per Qwen3-VL call when [`AutolabelRequest::vlm_batch`] is on.
pub const VLM_BATCH: usize = 8;

/// Default of [`AutolabelRequest::vlm_min_score`]: below it boxes were 5% real on public sets and
/// none on a fixed camera (docs/TRAINING.md, "Measuring auto-label").
pub const SUGGESTED_VLM_MIN_SCORE: f32 = 0.15;

/// Share of the boxes below [`AutolabelRequest::vlm_min_score`] the VLM is asked about anyway.
pub const VLM_SAMPLE: f32 = 0.02;

/// Default of [`AutolabelRequest::sam3_threshold`].
pub const SAM3_THRESHOLD: f32 = 0.3;

const fn yes() -> bool {
    true
}

const fn default_sam3_threshold() -> f32 {
    SAM3_THRESHOLD
}

impl AutolabelRequest {
    /// The defaults: MM-Grounding-DINO tiny, RF-DETR L, Qwen3-VL 2B, SigLIP 2.
    pub fn new(frames: FrameFilter) -> Self {
        Self {
            frames,
            gdino: Some(GdinoSize::Tiny),
            sam3: false,
            rfdetr: true,
            vlm: Some(VlmSize::Qwen2b),
            siglip: true,
            with_model: None,
            threshold: models::DEFAULT_THRESHOLD,
            device: None,
            gpu: None,
            model_accept_score: None,
            missing_only: false,
            confirm_license: false,
            gdino_threshold: 0.2,
            sam3_threshold: SAM3_THRESHOLD,
            rfdetr_threshold: 0.3,
            accept_score: triage::ACCEPT_SCORE,
            classes: None,
            allow_large: false,
            keep_on_cancel: true,
            vlm_batch: true,
            vlm_min_score: Some(SUGGESTED_VLM_MIN_SCORE),
            prefetch: true,
        }
    }

    pub fn validate(&self) -> Result<()> {
        for (k, v) in [
            ("threshold", self.threshold),
            ("gdino threshold", self.gdino_threshold),
            ("sam3 threshold", self.sam3_threshold),
            ("rfdetr threshold", self.rfdetr_threshold),
            ("accept score", self.accept_score),
        ] {
            if !(v.is_finite() && (0.01..=1.0).contains(&v)) {
                bail!("{k} must be between 0.01 and 1, got {v}");
            }
        }
        if self.with_model.is_none() && self.gdino.is_none() && !self.sam3 && !self.rfdetr {
            bail!("no labeler: enable MM-Grounding-DINO, SAM 3 or RF-DETR, or pass --with-model");
        }
        if self.sam3 && self.with_model.is_some() {
            bail!(
                "SAM 3 is one of the labelers, and a run with a model (--with-model) uses that \
                 model alone: remove --sam3 or --with-model"
            );
        }
        if self.sam3 && self.gdino.is_some() {
            bail!("SAM 3 takes MM-Grounding-DINO's place: choose one of them");
        }
        if self.missing_only && self.with_model.is_none() {
            bail!(
                "finding missing labels needs a trained model to look with: add --with-model \
                 <name> (Training page: Label with a model)"
            );
        }
        if self.missing_only && self.model_accept_score.is_some() {
            bail!(
                "finding missing labels pre-accepts nothing, you check every box it proposes: \
                 remove --accept-score-model (on the pages: leave Pre-accept empty)"
            );
        }
        if let Some(v) = self.model_accept_score {
            if self.with_model.is_none() {
                bail!("the model accept score applies to a run with a model (--with-model)");
            }
            if !(v.is_finite() && (0.01..=1.0).contains(&v)) {
                bail!("model accept score must be between 0.01 and 1, got {v}");
            }
            if v < self.threshold {
                bail!(
                    "model accept score {v} is below the threshold {}: raise it to the threshold                      or more (boxes under the threshold are never proposed)",
                    self.threshold
                );
            }
        }
        if let Some(d) = &self.device {
            if self.with_model.is_none() {
                bail!(
                    "a device applies to a run with a model (--with-model); the labelers run on                      the training GPU"
                );
            }
            models::canonical_device(d)?;
        }
        if let Some(g) = super::capability::gpu_arg(self.gpu.as_deref())?
            && self.with_model.is_some()
        {
            bail!(
                "a GPU ({g}) applies to the labelers; a run with a model (--with-model) takes a \
                 device instead (--device ort:cuda:N, openvino:gpu.N, ...)"
            );
        }
        if let Some(v) = self.vlm_min_score
            && !(v.is_finite() && v > 0.0 && v <= 1.0)
        {
            bail!("VLM minimum score must be above 0 and at most 1, got {v}");
        }
        Ok(())
    }

    /// The checkpoints the labelers need (none for `with_model`).
    pub fn resources(&self) -> Vec<&'static Resource> {
        if self.with_model.is_some() {
            return Vec::new();
        }
        let mut v = Vec::new();
        if let Some(g) = self
            .gdino
            .and_then(|g| res_catalog::autolabel_weights(g.key()))
        {
            v.push(g);
        }
        if self.sam3
            && let Some(s) = hf_token::sam3_resource()
        {
            v.push(s);
        }
        if self.rfdetr
            && let Some(r) = res_catalog::base_weights(res_catalog::TrainFamily::Rfdetr, "large")
        {
            v.push(r);
        }
        if let Some(q) = self
            .vlm
            .and_then(|q| res_catalog::autolabel_weights(q.key()))
        {
            v.push(q);
        }
        if self.siglip
            && let Some(s) = res_catalog::autolabel_weights("siglip2-base")
        {
            v.push(s);
        }
        v
    }
}

/// Why a SAM 3 run could not get its weights, with what to do; None when it can: every file is on
/// disk (`parts_on_disk`: adopted by hash, nothing is downloaded), or there is a token to try.
/// Whether that token has access only the download can tell.
fn sam3_problem(parts_on_disk: bool, lookup: &Lookup, login: &LoginCommand) -> Option<String> {
    if parts_on_disk || lookup.token().is_some() {
        return None;
    }
    let why = lookup
        .problem()
        .unwrap_or_else(|| "No Hugging Face token is set up".to_string());
    let gated = res_catalog::gated(res_catalog::SAM3_ID)?;
    Some(format!(
        "SAM 3 is not installed, and its download needs a Hugging Face login. {why}. Request access \
         on {} ({}), then sign in: {} ({}). Then rerun with --allow-large, or download it from the \
         Training page's SAM 3 panel",
        gated.repo_url, gated.license, login.command, login.note,
    ))
}

/// The part of a plan that decides a run's frames, and what the preview shares with the run: no
/// GPU, login or download gate applies. A `missing_only` run looks at every status and every frame
/// of the folder (`status=any`, `source=all`: the job keeps [`triage::missing_scope`]; the default
/// `unreviewed`, or the Review page's `source=queue`, would miss the reviewed frames), and the
/// default limit and threshold become [`MAX_LIMIT`] and [`MISSING_THRESHOLD`] (any other value is
/// used as given). Fails on a bad filter or an unknown class or model.
fn scoped_job(config: &Config, mut request: AutolabelRequest) -> Result<AutolabelJob> {
    request.validate()?;
    if request.missing_only {
        let query = &mut request.frames.query;
        query.insert("status".into(), "any".into());
        query.insert("source".into(), "all".into());
        if request.frames.limit == DEFAULT_LIMIT {
            request.frames.limit = MAX_LIMIT;
        }
        if (request.threshold - models::DEFAULT_THRESHOLD).abs() < f32::EPSILON {
            request.threshold = MISSING_THRESHOLD;
        }
    }
    let classes = without_hidden(
        run_classes(&config.collect.classes, request.classes.as_deref())?,
        &config.collect.hidden_classes,
    )?;
    let mut model = None;
    if let Some(m) = &request.with_model {
        if let Some(d) = &request.device {
            request.device = Some(models::canonical_device(d)?);
        }
        model = Some(models::resolve(config, m, request.device.as_deref())?);
    }
    let id = super::service::new_run_id();
    Ok(AutolabelJob {
        work: super::training_dir(config).join("autolabel").join(&id),
        id,
        dataset_dir: config.collect.resolved_review_dir(&config.data_root()),
        synonyms: synonyms(&classes, &config.training.synonyms),
        classes,
        request,
        paths: TrainPaths::for_config(config),
        platform: Platform::current(),
        accelerator: None,
        timeout: Duration::from_secs(u64::from(config.training.timeout_hours) * 3600),
        config: Box::new(config.clone()),
        model,
        detector: None,
        config_writer: None,
        after: None,
        finish: None,
    })
}

/// What a run of `request` on `config` would do: the frames it takes and leaves out, a rough time
/// ([`preview::AutolabelPreview`]). Reads every sidecar of the folder and writes nothing. Unlike
/// [`plan_autolabel`] it does not care whether this machine can run it.
pub fn preview_autolabel(
    config: &Config,
    request: AutolabelRequest,
) -> Result<preview::AutolabelPreview> {
    scoped_job(config, request)?.preview()
}

/// Plan an auto-label run of `request` on `config`, the config file at `config_path` (fails fast
/// on a bad filter, an unknown model, a missing GPU, SAM 3 with no way to get it, or a large
/// download that was not allowed; [`scoped_job`] says what a `missing_only` run changes).
pub fn plan_autolabel(
    config: &Config,
    config_path: &Path,
    request: AutolabelRequest,
    cap: &Capability,
) -> Result<AutolabelJob> {
    let mut job = scoped_job(config, request)?;
    let allow_large = job.request.allow_large || config.allow_large_downloads;
    job.accelerator.clone_from(&cap.accelerator);
    if let Some(plan) = &job.model {
        check_model_install(config, plan, allow_large, job.request.confirm_license)?;
    } else {
        let paths = &job.paths;
        let request = &mut job.request;
        // A chosen GPU: its device and its lane's environment.
        let Some(a) = &cap.resolve_gpu(request.gpu.as_deref())? else {
            bail!(
                "the auto-labelers need an NVIDIA GPU (Windows / Linux) or Apple silicon: {}; \
                 re-label with a trained model instead (--with-model <name>)",
                cap.summary()
            );
        };
        if request.sam3
            && let Some(problem) = sam3_problem(
                hf_token::sam3_ready(paths) || hf_token::sam3_parts_present(paths),
                &hf_token::lookup(&config.data_root()),
                &hf_token::login_command_for(config_path),
            )
        {
            bail!(problem);
        }
        if !env::train_env_ready(paths, job.platform, a.lane()) && !allow_large {
            let (dl, _) = a.lane().estimate();
            bail!(
                "the auto-labelers run in the training environment (about {} download); rerun \
                 with --allow-large (Training page: tick the download confirmation)",
                res_catalog::format_size(dl)
            );
        }
        let list: Vec<(&'static Resource, std::path::PathBuf)> = request
            .resources()
            .into_iter()
            .map(|r| (r, super::pyenv::weights_dir(paths, r)))
            .collect();
        super::pyenv::check_large(&list, allow_large)?;
        request.gpu = super::capability::gpu_arg(request.gpu.as_deref())?.map(|_| a.device_arg());
        job.accelerator = Some(a.clone());
    }
    job.request.allow_large = allow_large;
    Ok(job)
}

/// The frame of a `train.txt` line (`./images/s0-day.back.2026-10-25.<stem>.jpg`): the file name
/// [`crate::training::dataset`] gives it without its `s<N>-` source prefix, as an export's flat
/// name. None: not a frame of a dataset folder (a public image).
fn trained_frame(line: &str) -> Option<String> {
    let name = line.trim().rsplit('/').next()?;
    let (source, flat) = name.strip_prefix('s')?.split_once('-')?;
    if source.is_empty() || !source.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(crate::collect::labels::export::parse_flat_name(flat)?.jpg())
}

/// The frames `model` trained on (`<scene>/<camera>/<date>/<stem>.jpg`, each once, however often
/// balancing repeated it): its `<stem>.provenance.json` beside the model file names the run, whose
/// `<training dir>/<run_id>/data/train.txt` lists them. Validation and held-out frames are not in
/// that list. A model without the record (a catalog or plain local model) trained on nothing of
/// here as far as anyone can tell: a warning, no frames. A record that does not read, or a run
/// whose list is gone, is an error: the frames cannot be left out, and the model would be asked
/// about what it memorised.
pub fn trained_frames(
    config: &Config,
    model: &crate::config::ModelConfig,
    warnings: &mut Vec<String>,
) -> Result<BTreeSet<String>> {
    #[derive(Deserialize)]
    struct Record {
        run_id: String,
    }
    let name = model.effective_name();
    let path = config.data_path(&model.path);
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned());
    let record = path.with_file_name(format!("{}.provenance.json", stem.unwrap_or_default()));
    let text = match std::fs::read_to_string(&record) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            warnings.push(no_training_record(&name));
            return Ok(BTreeSet::new());
        }
        t => t.with_context(|| format!("reading {}", record.display()))?,
    };
    let run = serde_json::from_str::<Record>(&text)
        .ok()
        .map(|r| r.run_id)
        .filter(|id| models::plain_file(id))
        .ok_or_else(|| {
            anyhow!(
                "{} names no training run, so the frames {name} trained on cannot be left out; \
                 restore the file, or choose another model",
                record.display()
            )
        })?;
    let list = super::training_dir(config)
        .join(&run)
        .join("data")
        .join("train.txt");
    let text = std::fs::read_to_string(&list).map_err(|e| {
        anyhow!(
            "{name} was trained in run {run}, but its frame list {} does not read ({e}), so the \
             frames it trained on cannot be left out; restore that run folder, or choose a model \
             whose run folder still exists",
            list.display()
        )
    })?;
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let frames: BTreeSet<String> = lines.iter().filter_map(|l| trained_frame(l)).collect();
    let other = lines.iter().filter(|l| trained_frame(l).is_none()).count();
    if other > 0 {
        warnings.push(format!(
            "{other} of the {} lines of {} name no frame of a dataset folder (public images, or a \
             name cut short); a frame among them is not left out",
            lines.len(),
            list.display()
        ));
    }
    Ok(frames)
}

/// The warning of a model without a training record ([`trained_frames`]).
pub(crate) fn no_training_record(name: &str) -> String {
    format!(
        "{name} has no training record (<model>.provenance.json), so no frame is left out as \
         trained on; if it did train on these frames, expect few real finds"
    )
}

/// [`super::task::Task::skip_reason`] of a run after an import that wrote no frame.
pub const NO_NEW_FRAMES: &str =
    "not run: the import wrote no new frames (all were imported before)";

/// Make `job` (planned by [`plan_autolabel`]) label exactly the frames of `import`, after it:
/// that import's folder, its frames whatever their review status (`import=<id>, status=any,
/// source=all`), at most as many as it imports (capped at [`MAX_LIMIT`]). The labelers' marker
/// still skips frames they looked at already (`again` is not set).
pub fn follow_import(job: &mut AutolabelJob, import: &super::import::job::ImportJob) {
    job.after = Some(import.id.clone());
    job.dataset_dir = import.dataset_dir.clone();
    job.request.frames = FrameFilter {
        query: [
            ("import".to_string(), import.id.clone()),
            ("status".to_string(), "any".to_string()),
            ("source".to_string(), "all".to_string()),
        ]
        .into(),
        limit: import.request.frames.clamp(1, MAX_LIMIT),
    };
}

/// The gates of a catalog model a run installs or adds: an export this platform cannot run, a
/// large download not allowed, a restrictive licence not confirmed.
fn check_model_install(
    config: &Config,
    plan: &models::ModelPlan,
    allow_large: bool,
    confirm_license: bool,
) -> Result<()> {
    let Some(res) = plan.resource() else {
        return Ok(());
    };
    let platform = Platform::current();
    if let Some(lic) = &plan.restricted
        && !confirm_license
    {
        bail!(
            "{} is licensed {lic}; confirm that your use is covered (Training page: tick the \
             licence confirmation; CLI: --confirm-license)",
            res.title
        );
    }
    if plan.install == Some(models::Install::Export)
        && let Some(why) = crate::resources::export::unsupported_reason(platform.os, platform.arch)
    {
        bail!("{} has to be exported on this machine: {why}", res.title);
    }
    if plan.install.is_some() && !allow_large {
        let ready = crate::resources::export::toolchain_state(
            &crate::resources::export::Paths::for_config(config),
            platform,
        )
        .ready();
        let toolchain = (plan.install == Some(models::Install::Export) && !ready)
            .then(|| res_catalog::toolchain_estimate(platform.os, platform.arch))
            .flatten();
        if res.is_large() || (plan.install == Some(models::Install::Export) && !ready) {
            bail!(
                "installing {} downloads {}{}; allow the large download (Training page: tick \
                 the download confirmation; CLI: --allow-large)",
                res.title,
                res_catalog::format_size(res.size()),
                toolchain.map_or_else(String::new, |t| format!(
                    " plus the export toolchain ({}, once)",
                    res_catalog::format_size(t.download)
                ))
            );
        }
    }
    Ok(())
}

/// The classes of a run: `requested` (None or empty = every class of `review_classes`), each
/// matched case-insensitively to a class of `review_classes` (`collect.classes`) and given its
/// spelling there, once each, in `review_classes` order. A class outside the list is refused: the
/// reviewer could not accept its boxes.
pub fn run_classes(review_classes: &[String], requested: Option<&[String]>) -> Result<Vec<String>> {
    if review_classes.is_empty() {
        bail!("no classes: set collect.classes (Review page, Classes)");
    }
    let Some(req) = requested.filter(|r| r.iter().any(|c| !c.trim().is_empty())) else {
        return Ok(review_classes.to_vec());
    };
    let unknown: Vec<&str> = req
        .iter()
        .map(|c| c.trim())
        .filter(|c| !c.is_empty() && !review_classes.iter().any(|r| r.eq_ignore_ascii_case(c)))
        .collect();
    if !unknown.is_empty() {
        bail!(
            "class(es) {} are not in collect.classes ({}); add them to the class list first \
             (Review page, Classes)",
            unknown.join(", "),
            review_classes.join(", ")
        );
    }
    Ok(review_classes
        .iter()
        .filter(|r| req.iter().any(|c| c.trim().eq_ignore_ascii_case(r)))
        .cloned()
        .collect())
}

/// A run's `classes` ([`run_classes`]) without those of `hidden` (`collect.hidden_classes`: the
/// reviewer does not want their boxes, so a run neither looks for them nor writes them). An
/// error when nothing is left.
pub fn without_hidden(classes: Vec<String>, hidden: &[String]) -> Result<Vec<String>> {
    let (gone, kept): (Vec<String>, Vec<String>) = classes
        .into_iter()
        .partition(|c| crate::collect::labels::is_hidden(c, hidden));
    if kept.is_empty() {
        bail!(
            "class(es) {} are marked \"don't want\" in Review (collect.hidden_classes); untick it \
             for a class to label (Review page, Classes) or pick another class",
            gone.join(", ")
        );
    }
    if !gone.is_empty() {
        tracing::info!(
            classes = %gone.join(", "),
            "auto-label: unwanted (hidden) classes left out of the run"
        );
    }
    Ok(kept)
}

/// The status filter of a run that does not name one: `unreviewed` for the labelers and for a
/// model (first-pass labelling); self-training over reviewed frames is an explicit `status=any`
/// (the Training page's preset).
pub fn default_status(_with_model: bool) -> &'static str {
    "unreviewed"
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    use std::path::Path;

    struct Tmp(std::path::PathBuf);

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tmp(tag: &str) -> Tmp {
        let d = std::env::temp_dir().join(format!("bop-autolabel-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        Tmp(d)
    }

    #[test]
    fn frame_filters_parse_with_defaults() {
        let f = FrameFilter::parse("", "unreviewed").unwrap();
        assert_eq!(f, FrameFilter::default());
        let f = FrameFilter::parse("scene=ir, bucket=empty,limit=20,import=any", "any").unwrap();
        assert_eq!(f.limit, 20);
        assert_eq!(f.import(), Some("any"));
        assert_eq!(f.query["status"], "any");
        let q = f.filters().unwrap();
        assert_eq!(q.scene.as_deref(), Some("ir"));
        assert!(f.accepts_source(Some("import:x")) && !f.accepts_source(None));
        let one = FrameFilter::parse("import=20261022-1", "unreviewed").unwrap();
        assert!(one.accepts_source(Some("import:20261022-1")));
        assert!(!one.accepts_source(Some("import:other")));
        for bad in ["scene", "scene=dusk", "colour=red", "limit=0", "limit=x"] {
            assert!(FrameFilter::parse(bad, "any").is_err(), "{bad}");
        }
        let l = FrameFilter::parse("label=person", "unreviewed").unwrap();
        assert_eq!(l.filters().unwrap().label, ["person".to_string()].into());
        let any = FrameFilter::parse("label=person| car |,status=any", "unreviewed").unwrap();
        assert_eq!(
            any.filters().unwrap().label,
            ["person".to_string(), "car".into()].into()
        );
        assert!(FrameFilter::parse("label=ok|a\u{7}b", "any").is_err());
        assert!(!l.again());
        let a = FrameFilter::parse("auto=none,again=Yes", "unreviewed").unwrap();
        assert!(a.again());
        assert_eq!(a.query["again"], "1");
        assert_eq!(
            a.filters().unwrap().auto,
            Some(crate::collect::labels::queue::AutoFilter::Never)
        );
        for bad in ["again=0", "again=", "auto=maybe"] {
            assert!(FrameFilter::parse(bad, "any").is_err(), "{bad}");
        }
        assert!(f.describe().contains("scene=ir"));
    }

    #[test]
    fn run_options_default_on_validate_and_parse_from_old_requests() {
        let r = AutolabelRequest::new(FrameFilter::default());
        assert!(r.keep_on_cancel && r.vlm_batch && r.prefetch);
        assert_eq!(r.vlm_min_score, Some(SUGGESTED_VLM_MIN_SCORE));
        // A request saved before the options existed (tasks.json) parses with the defaults.
        let mut old = serde_json::to_value(&r).unwrap();
        let obj = old.as_object_mut().unwrap();
        for k in ["keep_on_cancel", "vlm_batch", "vlm_min_score", "prefetch"] {
            assert!(obj.remove(k).is_some(), "{k}");
        }
        let mut back: AutolabelRequest = serde_json::from_value(old).unwrap();
        // Queued before the cutoff was the default: it keeps checking every box.
        assert_eq!(back.vlm_min_score, None);
        back.vlm_min_score = r.vlm_min_score;
        assert_eq!(back, r);
        let mut off = r.clone();
        (off.keep_on_cancel, off.vlm_batch, off.prefetch) = (false, false, false);
        off.vlm_min_score = Some(0.2);
        let back: AutolabelRequest =
            serde_json::from_str(&serde_json::to_string(&off).unwrap()).unwrap();
        assert_eq!(back, off);
        for ok in [0.01, 0.2, 1.0] {
            let mut m = r.clone();
            m.vlm_min_score = Some(ok);
            assert!(m.validate().is_ok(), "{ok}");
        }
        for bad in [0.0, -0.1, 1.01, f32::NAN, f32::INFINITY] {
            let mut m = r.clone();
            m.vlm_min_score = Some(bad);
            let e = m.validate().unwrap_err().to_string();
            assert!(e.contains("VLM minimum score"), "{bad}: {e}");
        }
    }

    #[test]
    fn missing_only_needs_a_model_and_refuses_a_pre_accept() {
        let mut r = AutolabelRequest::new(FrameFilter::default());
        assert!(!r.missing_only);
        // Saved before the option existed (tasks.json).
        let mut old = serde_json::to_value(&r).unwrap();
        assert!(
            old.as_object_mut()
                .unwrap()
                .remove("missing_only")
                .is_some()
        );
        assert_eq!(serde_json::from_value::<AutolabelRequest>(old).unwrap(), r);
        r.missing_only = true;
        let e = r.validate().unwrap_err().to_string();
        assert!(e.contains("--with-model"), "{e}");
        r.with_model = Some("m".into());
        assert!(r.validate().is_ok());
        r.model_accept_score = Some(0.8);
        let e = r.validate().unwrap_err().to_string();
        assert!(e.contains("--accept-score-model"), "{e}");
    }

    #[test]
    fn train_lines_map_to_frames() {
        assert_eq!(
            trained_frame("./images/s0-day.back.2026-10-25.20261026T003155612Z_2e444695.jpg\n")
                .as_deref(),
            Some("day/back/2026-10-25/20261026T003155612Z_2e444695.jpg")
        );
        assert_eq!(
            trained_frame("./images/s12-ir.a-b_c.2026-01-02.x-1.jpg").as_deref(),
            Some("ir/a-b_c/2026-01-02/x-1.jpg")
        );
        for other in [
            "./images/s0-img_001.jpg",
            "./images/sx-day.back.2026-10-25.a.jpg",
            "./images/s-day.back.2026-10-25.a.jpg",
            "./images/day.back.2026-10-25.a.jpg",
            "./images/s0-day.back.2026-10-25.a",
            "",
        ] {
            assert_eq!(trained_frame(other), None, "{other}");
        }
    }

    /// `Config` with a model `night` under `root` and its provenance, whose run is `run_id`.
    fn trained_setup(root: &Path, run_id: &str) -> (Config, crate::config::ModelConfig) {
        let config = Config {
            download_dir: Some(root.to_path_buf()),
            ..Config::default()
        };
        let entry = crate::config::ModelConfig {
            name: Some("night".into()),
            path: "models/night.onnx".into(),
            ..crate::config::ModelConfig::default()
        };
        let models = root.join("models");
        std::fs::create_dir_all(&models).unwrap();
        // Only `run_id` is read: a record of another version still works.
        std::fs::write(
            models.join("night.provenance.json"),
            serde_json::json!({"version": 9, "run_id": run_id}).to_string(),
        )
        .unwrap();
        (config, entry)
    }

    #[test]
    fn trained_frames_come_from_the_runs_list_each_once() {
        let t = tmp("trained");
        let (config, entry) = trained_setup(&t.0, "20261031-1");
        let data = crate::training::training_dir(&config)
            .join("20261031-1")
            .join("data");
        std::fs::create_dir_all(&data).unwrap();
        // Repeats from balancing, two sources, a public image.
        std::fs::write(
            data.join("train.txt"),
            "./images/s0-day.back.2026-10-25.a.jpg\n\
             ./images/s0-day.back.2026-10-25.a.jpg\n\
             ./images/s1-ir.gate.2026-10-26.b.jpg\n\
             ./images/s1-ir.gate.2026-10-26.b.jpg\n\
             ./images/s2-coco_0001.jpg\n",
        )
        .unwrap();
        // valid.txt is not read: those frames stay eligible.
        std::fs::write(
            data.join("valid.txt"),
            "./images/s0-day.back.2026-10-25.v.jpg\n",
        )
        .unwrap();
        let mut warnings = Vec::new();
        let got = trained_frames(&config, &entry, &mut warnings).unwrap();
        let want: BTreeSet<String> = [
            "day/back/2026-10-25/a.jpg".to_string(),
            "ir/gate/2026-10-26/b.jpg".to_string(),
        ]
        .into();
        assert_eq!(got, want);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].starts_with("1 of the 5 lines"), "{warnings:?}");
    }

    #[test]
    fn no_record_warns_and_a_lost_list_or_a_bad_record_refuses() {
        let t = tmp("trained-gone");
        let (config, entry) = trained_setup(&t.0, "20261031-1");
        // Provenance, no train.txt.
        let mut warnings = Vec::new();
        let e = trained_frames(&config, &entry, &mut warnings)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("20261031-1") && e.contains("train.txt") && e.contains("restore"),
            "{e}"
        );
        assert!(warnings.is_empty());
        // A run id that leaves the training folder, or no run id.
        for bad in [
            serde_json::json!({"run_id": "../../secrets"}).to_string(),
            serde_json::json!({"run_id": ""}).to_string(),
            serde_json::json!({"name": "x"}).to_string(),
            "not json".to_string(),
        ] {
            std::fs::write(t.0.join("models/night.provenance.json"), &bad).unwrap();
            let e = trained_frames(&config, &entry, &mut warnings)
                .unwrap_err()
                .to_string();
            assert!(e.contains("names no training run"), "{bad}: {e}");
        }
        // No record at all.
        std::fs::remove_file(t.0.join("models/night.provenance.json")).unwrap();
        let got = trained_frames(&config, &entry, &mut warnings).unwrap();
        assert!(got.is_empty());
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("no training record"), "{warnings:?}");
    }

    #[test]
    fn run_classes_are_listed_classes_in_list_order() {
        let list: Vec<String> = ["person", "car", "Raccoon"].map(String::from).to_vec();
        assert_eq!(run_classes(&list, None).unwrap(), list);
        assert_eq!(run_classes(&list, Some(&[])).unwrap(), list);
        assert_eq!(run_classes(&list, Some(&[" ".into()])).unwrap(), list);
        let req: Vec<String> = ["raccoon", " CAR ", "car"].map(String::from).to_vec();
        assert_eq!(run_classes(&list, Some(&req)).unwrap(), ["car", "Raccoon"]);
        let e = run_classes(&list, Some(&["dog".into(), "car".into()]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("dog") && e.contains("collect.classes"), "{e}");
        assert!(run_classes(&[], None).is_err());
    }

    #[test]
    fn runs_leave_out_hidden_classes() {
        let list: Vec<String> = ["person", "car", "Raccoon"].map(String::from).to_vec();
        let hidden = vec!["raccoon".to_string()];
        assert_eq!(
            without_hidden(list.clone(), &hidden).unwrap(),
            ["person", "car"]
        );
        assert_eq!(without_hidden(list.clone(), &[]).unwrap(), list);
        let e = without_hidden(vec!["Raccoon".into()], &hidden)
            .unwrap_err()
            .to_string();
        assert!(e.contains("Raccoon") && e.contains("hidden_classes"), "{e}");
    }

    #[test]
    fn synonyms_categories_and_resources() {
        let classes = vec!["person".to_string(), "raccoon".into(), "package".into()];
        let over: BTreeMap<String, Vec<String>> = [(
            "raccoon".to_string(),
            vec!["raccoon".to_string(), "trash panda".into()],
        )]
        .into();
        let s = synonyms(&classes, &over);
        assert_eq!(s["person"], ["person", "pedestrian"]);
        assert_eq!(s["raccoon"], ["raccoon", "trash panda"]);
        assert_eq!(s["package"], ["package"]);
        assert_eq!(Category::of("Person"), Category::Person);
        assert_eq!(Category::of("truck"), Category::Vehicle);
        assert_eq!(Category::of("opossum"), Category::Animal);
        assert_eq!(Category::of("package"), Category::Object);
        let r = AutolabelRequest::new(FrameFilter::default());
        let ids: Vec<&str> = r.resources().iter().map(|r| r.id).collect();
        assert_eq!(
            ids,
            [
                "autolabel:mm-gdino-tiny",
                "train-base:rfdetr-large",
                "autolabel:qwen3-vl-2b",
                "autolabel:siglip2-base"
            ]
        );
        let mut m = r.clone();
        m.with_model = Some("x".into());
        assert!(m.resources().is_empty());
        let mut none = r;
        none.gdino = None;
        none.rfdetr = false;
        assert!(none.validate().is_err());
        assert_eq!(GdinoSize::parse("Swin-B"), Some(GdinoSize::Base));
        assert_eq!(VlmSize::parse("8b"), Some(VlmSize::Qwen8b));
    }

    #[test]
    fn requests_saved_before_sam3_still_parse() {
        // The fields a request had before SAM 3.
        let old = serde_json::json!({
            "frames": FrameFilter::default(), "gdino": "tiny", "rfdetr": true, "vlm": "qwen3-vl-2b",
            "siglip": true, "threshold": 0.3, "gdino_threshold": 0.2, "rfdetr_threshold": 0.3,
            "accept_score": 0.6,
        });
        let r: AutolabelRequest = serde_json::from_value(old).unwrap();
        assert!(!r.sam3);
        assert_eq!(r.gdino, Some(GdinoSize::Tiny));
        assert!(
            (r.sam3_threshold - 0.3).abs() < f32::EPSILON,
            "not serde's 0.0"
        );
        r.validate().unwrap();
    }

    #[test]
    fn sam3_takes_the_gdino_seat_and_only_that() {
        let sam3 = || {
            let mut r = AutolabelRequest::new(FrameFilter::default());
            r.gdino = None;
            r.sam3 = true;
            r
        };
        sam3().validate().unwrap();
        // Enough on its own.
        let mut alone = sam3();
        alone.rfdetr = false;
        alone.validate().unwrap();
        let ids: Vec<&str> = sam3().resources().iter().map(|r| r.id).collect();
        assert_eq!(
            ids,
            [
                res_catalog::SAM3_ID,
                "train-base:rfdetr-large",
                "autolabel:qwen3-vl-2b",
                "autolabel:siglip2-base"
            ]
        );
        let ids: Vec<&'static str> = AutolabelRequest::new(FrameFilter::default())
            .resources()
            .iter()
            .map(|r| r.id)
            .collect();
        assert!(!ids.contains(&res_catalog::SAM3_ID), "{ids:?}");

        let mut both = sam3();
        both.gdino = Some(GdinoSize::Tiny);
        let e = both.validate().unwrap_err().to_string();
        assert!(e.contains("choose one"), "{e}");

        let mut model = sam3();
        model.with_model = Some("x".into());
        let e = model.validate().unwrap_err().to_string();
        assert!(e.contains("--sam3") && e.contains("--with-model"), "{e}");
        assert!(model.resources().is_empty());

        for bad in [0.0, 1.5, f32::NAN] {
            let mut t = sam3();
            t.sam3_threshold = bad;
            let e = t.validate().unwrap_err().to_string();
            assert!(e.contains("sam3 threshold"), "{e}");
        }
    }

    #[test]
    fn sam3_with_no_way_to_get_it_names_the_login() {
        let login = LoginCommand {
            command: "bop --config c.json hf-login".into(),
            note: "Run it in PowerShell.".into(),
        };
        let found =
            Lookup::Found(hf_token::HfToken::new("hf_x", hf_token::TokenSource::Saved).unwrap());
        // Files on disk are adopted, a token is worth a try.
        assert_eq!(sam3_problem(true, &Lookup::Missing, &login), None);
        assert_eq!(sam3_problem(false, &found, &login), None);

        let e = sam3_problem(false, &Lookup::Missing, &login).unwrap();
        for want in [
            "No Hugging Face token",
            "https://huggingface.co/facebook/sam3",
            "SAM License",
            "bop --config c.json hf-login",
            "Run it in PowerShell.",
            "--allow-large",
        ] {
            assert!(e.contains(want), "{want}: {e}");
        }
        // A login that exists but cannot be read says that instead.
        let unreadable = Lookup::Unreadable {
            source: hf_token::TokenSource::Saved,
            path: Some(Path::new("t").to_path_buf()),
            reason: "access denied".into(),
        };
        let e = sam3_problem(false, &unreadable, &login).unwrap();
        assert!(
            e.contains("access denied") && !e.contains("No Hugging Face"),
            "{e}"
        );
        assert!(e.contains("hf-login"), "{e}");
    }
}
