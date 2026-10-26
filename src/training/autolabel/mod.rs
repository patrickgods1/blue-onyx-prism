//! Auto-labelling (`blue-onyx-prism train autolabel`, the Review and Training pages' Auto-label
//! buttons; docs/TRAINING.md "Auto-label"): proposals for the frames of the dataset folder,
//! written into their label files as pending `source: auto` boxes the reviewer accepts or
//! rejects. Two modes:
//!
//! - **Labelers** (default): `scripts/autolabel.py` in the training environment runs
//!   MM-Grounding-DINO (open vocabulary, prompts from `collect.classes` plus
//!   `training.synonyms`), RF-DETR L (COCO classes) and fuses them (WBF), then checks person and
//!   single-labeler boxes with Qwen3-VL and ranks animal species with SigLIP 2. The service
//!   triages every box ([`triage`]): `accept` only when both labelers agree at a score of at least
//!   0.6 and no check objects; night person / animal boxes always `review`.
//! - **A model** (`--with-model <id>`, any model of [`models`]: configured, a local file, or a
//!   catalog model the run downloads or exports first and adds to the config disabled): the
//!   service's own inference path (the benchmark pipeline: decode, preprocess, infer,
//!   postprocess) labels the frames on the model's device or the run's `device`; its labels are
//!   mapped onto the class list ([`triage::map_label`]); its boxes are `review`, or `accept` at a
//!   pre-accept score (`model_accept_score`; night person / animal boxes stay `review`). By
//!   default it labels unreviewed frames (first-pass labelling); with `status=any` it is the
//!   self-training loop: frames whose reviewed labels it disagrees with (a new box, a missed
//!   box, a class change at IoU 0.5) go back to review with a `disagreement` note, which the
//!   Review queue sorts first.
//!
//! The service owns every label-file write ([`crate::collect::labels::merge_proposals`]); the
//! Python side only returns proposals as JSONL. Nothing is deleted or accepted automatically:
//! hard negatives come from the reviewer only.
//!
//! - [`catalog`]: the class catalog of the Review page's picker, [`catalog::coco_class`] and
//!   the classes the enabled models report.
//! - [`triage`]: pure: the triage rule, proposals to label boxes, the re-label disagreement.
//! - [`models`]: the models a run can use (`GET /v1/autolabel/models`) and how a chosen one
//!   becomes a (disabled) config entry.
//! - [`relabel`]: detection with a model of the config in this process.
//! - [`job`]: the stage machine ([`AutolabelJob`], a [`super::task::Task`]).

pub mod catalog;
pub mod job;
pub mod models;
pub mod relabel;
pub mod triage;

pub use job::AutolabelJob;

use super::capability::Capability;
use crate::collect::labels::queue::Filters;
use crate::config::Config;
use crate::resources::catalog::{self as res_catalog, Platform, Resource};
use crate::resources::export::train::{self as env, TrainPaths};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;

/// Default most frames of one run.
pub const DEFAULT_LIMIT: usize = 200;
/// Most frames of one run.
pub const MAX_LIMIT: usize = 20_000;

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
/// `auto`), `import=<id>|any` (imported frames), `again=1` (also
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
    /// The user confirmed a restrictive (non-commercial / personal use) licence of a catalog
    /// model the run installs or adds.
    #[serde(default)]
    pub confirm_license: bool,
    pub gdino_threshold: f32,
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
    /// Check only the boxes with a fused score of at least this with Qwen3-VL (None = every
    /// eligible box).
    #[serde(default)]
    pub vlm_min_score: Option<f32>,
    /// Load and decode the next frame on a background thread while the GPU labels this one.
    #[serde(default = "yes")]
    pub prefetch: bool,
}

/// Boxes per Qwen3-VL call when [`AutolabelRequest::vlm_batch`] is on.
pub const VLM_BATCH: usize = 8;

/// Default of [`AutolabelRequest::vlm_min_score`] the Training page suggests.
pub const SUGGESTED_VLM_MIN_SCORE: f32 = 0.2;

const fn yes() -> bool {
    true
}

impl AutolabelRequest {
    /// The defaults: MM-Grounding-DINO tiny, RF-DETR L, Qwen3-VL 2B, SigLIP 2.
    pub fn new(frames: FrameFilter) -> Self {
        Self {
            frames,
            gdino: Some(GdinoSize::Tiny),
            rfdetr: true,
            vlm: Some(VlmSize::Qwen2b),
            siglip: true,
            with_model: None,
            threshold: models::DEFAULT_THRESHOLD,
            device: None,
            gpu: None,
            model_accept_score: None,
            confirm_license: false,
            gdino_threshold: 0.2,
            rfdetr_threshold: 0.3,
            accept_score: triage::ACCEPT_SCORE,
            classes: None,
            allow_large: false,
            keep_on_cancel: true,
            vlm_batch: true,
            vlm_min_score: None,
            prefetch: true,
        }
    }

    pub fn validate(&self) -> Result<()> {
        for (k, v) in [
            ("threshold", self.threshold),
            ("gdino threshold", self.gdino_threshold),
            ("rfdetr threshold", self.rfdetr_threshold),
            ("accept score", self.accept_score),
        ] {
            if !(v.is_finite() && (0.01..=1.0).contains(&v)) {
                bail!("{k} must be between 0.01 and 1, got {v}");
            }
        }
        if self.with_model.is_none() && self.gdino.is_none() && !self.rfdetr {
            bail!("no labeler: enable MM-Grounding-DINO or RF-DETR, or pass --with-model");
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

/// Plan an auto-label run of `request` on `config` (fails fast on a bad filter, an unknown
/// model, a missing GPU or a large download that was not allowed).
pub fn plan_autolabel(
    config: &Config,
    mut request: AutolabelRequest,
    cap: &Capability,
) -> Result<AutolabelJob> {
    request.validate()?;
    let classes = run_classes(&config.collect.classes, request.classes.as_deref())?;
    let paths = TrainPaths::for_config(config);
    let platform = Platform::current();
    let allow_large = request.allow_large || config.allow_large_downloads;
    let mut model = None;
    let mut accelerator = cap.accelerator.clone();
    if let Some(m) = &request.with_model {
        if let Some(d) = &request.device {
            request.device = Some(models::canonical_device(d)?);
        }
        let plan = models::resolve(config, m, request.device.as_deref())?;
        check_model_install(config, &plan, allow_large, request.confirm_license)?;
        model = Some(plan);
    } else {
        // A chosen GPU: its device and its lane's environment.
        let Some(a) = &cap.resolve_gpu(request.gpu.as_deref())? else {
            bail!(
                "the auto-labelers need an NVIDIA GPU (Windows / Linux) or Apple silicon: {}; \
                 re-label with a trained model instead (--with-model <name>)",
                cap.summary()
            );
        };
        if !env::train_env_ready(&paths, platform, a.lane()) && !allow_large {
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
            .map(|r| (r, super::pyenv::weights_dir(&paths, r)))
            .collect();
        super::pyenv::check_large(&list, allow_large)?;
        request.gpu = super::capability::gpu_arg(request.gpu.as_deref())?.map(|_| a.device_arg());
        accelerator = Some(a.clone());
    }
    request.allow_large = allow_large;
    let id = super::service::new_run_id();
    Ok(AutolabelJob {
        work: super::training_dir(config).join("autolabel").join(&id),
        id,
        dataset_dir: config.collect.resolved_review_dir(&config.data_root()),
        synonyms: synonyms(&classes, &config.training.synonyms),
        classes,
        request,
        paths,
        platform,
        accelerator,
        timeout: Duration::from_secs(u64::from(config.training.timeout_hours) * 3600),
        config: Box::new(config.clone()),
        model,
        detector: None,
        config_writer: None,
        after: None,
        finish: None,
    })
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
        assert_eq!(r.vlm_min_score, None);
        // A request saved before the options existed (tasks.json) parses with the defaults.
        let mut old = serde_json::to_value(&r).unwrap();
        let obj = old.as_object_mut().unwrap();
        for k in ["keep_on_cancel", "vlm_batch", "vlm_min_score", "prefetch"] {
            assert!(obj.remove(k).is_some(), "{k}");
        }
        let back: AutolabelRequest = serde_json::from_value(old).unwrap();
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
}
