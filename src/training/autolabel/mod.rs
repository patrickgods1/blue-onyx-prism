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
//! - **Self-training** (`--with-model <trained model>`): the service's own inference path (the
//!   benchmark pipeline: decode, preprocess, infer, postprocess) re-labels the frames with a
//!   trained model; frames whose reviewed labels it disagrees with (a new box, a missed box, a
//!   class change at IoU 0.5) go back to review with a `disagreement` note, which the Review
//!   queue sorts first.
//!
//! The service owns every label-file write ([`crate::collect::labels::merge_proposals`]); the
//! Python side only returns proposals as JSONL. Nothing is deleted or accepted automatically:
//! hard negatives come from the reviewer only.
//!
//! - [`triage`]: pure: the triage rule, proposals to label boxes, the re-label disagreement.
//! - [`relabel`]: detection with a configured model in this process.
//! - [`job`]: the stage machine ([`AutolabelJob`], a [`super::task::Task`]).

pub mod job;
pub mod relabel;
pub mod triage;

pub use job::AutolabelJob;

use super::capability::Capability;
use crate::collect::labels::queue::Filters;
use crate::config::Config;
use crate::resources::catalog::{self, Platform, Resource};
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
/// `status`, `source`), `import=<id>|any` (imported frames) and `limit` (most frames, in queue
/// order). Written `key=value` pairs separated by commas (`scene=ir,status=unreviewed,limit=20`).
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
                "bucket" | "scene" | "camera" | "status" | "source" | "import" => {
                    query.insert(k, v);
                }
                other => bail!(
                    "unknown frame filter '{other}' (bucket, scene, camera, status, source, \
                     import, limit)"
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
        Filters::from_query(&q).map_err(anyhow::Error::msg)
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
    /// Self-training: re-label with this configured model instead of the labelers.
    #[serde(default)]
    pub with_model: Option<String>,
    /// Score threshold of `with_model`'s boxes.
    pub threshold: f32,
    pub gdino_threshold: f32,
    pub rfdetr_threshold: f32,
    /// Triage: both labelers agree at this fused score or more to `accept`.
    pub accept_score: f32,
    /// The class list (default `collect.classes`).
    #[serde(default)]
    pub classes: Option<Vec<String>>,
    #[serde(default)]
    pub allow_large: bool,
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
            threshold: 0.3,
            gdino_threshold: 0.2,
            rfdetr_threshold: 0.3,
            accept_score: triage::ACCEPT_SCORE,
            classes: None,
            allow_large: false,
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
        Ok(())
    }

    /// The checkpoints the labelers need (none for `with_model`).
    pub fn resources(&self) -> Vec<&'static Resource> {
        if self.with_model.is_some() {
            return Vec::new();
        }
        let mut v = Vec::new();
        if let Some(g) = self.gdino.and_then(|g| catalog::autolabel_weights(g.key())) {
            v.push(g);
        }
        if self.rfdetr
            && let Some(r) = catalog::base_weights(catalog::TrainFamily::Rfdetr, "large")
        {
            v.push(r);
        }
        if let Some(q) = self.vlm.and_then(|q| catalog::autolabel_weights(q.key())) {
            v.push(q);
        }
        if self.siglip
            && let Some(s) = catalog::autolabel_weights("siglip2-base")
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
    let classes = request
        .classes
        .clone()
        .filter(|c| !c.is_empty())
        .unwrap_or_else(|| config.collect.classes.clone());
    if classes.is_empty() {
        bail!("no classes: set collect.classes (Review page, Classes)");
    }
    let paths = TrainPaths::for_config(config);
    let platform = Platform::current();
    let allow_large = request.allow_large || config.allow_large_downloads;
    if let Some(m) = &request.with_model {
        if !config
            .models
            .iter()
            .any(|x| x.effective_name().eq_ignore_ascii_case(m))
        {
            bail!(
                "no configured model named {m}: a training run adds its model to the config (or \
                 use add-model)"
            );
        }
    } else {
        let Some(a) = &cap.accelerator else {
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
                catalog::format_size(dl)
            );
        }
        let list: Vec<(&'static Resource, std::path::PathBuf)> = request
            .resources()
            .into_iter()
            .map(|r| (r, super::pyenv::weights_dir(&paths, r)))
            .collect();
        super::pyenv::check_large(&list, allow_large)?;
    }
    request.allow_large = allow_large;
    let id = super::service::new_run_id();
    Ok(AutolabelJob {
        work: super::training_dir(config).join("autolabel").join(&id),
        id,
        dataset_dir: crate::collect::CollectSettings::for_config(config).dir,
        synonyms: synonyms(&classes, &config.training.synonyms),
        classes,
        request,
        paths,
        platform,
        accelerator: cap.accelerator.clone(),
        timeout: Duration::from_secs(u64::from(config.training.timeout_hours) * 3600),
        config: Box::new(config.clone()),
        detector: None,
    })
}

/// Whether a status filter would list reviewed frames (the self-training default).
pub fn default_status(with_model: bool) -> &'static str {
    if with_model { "any" } else { "unreviewed" }
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
        assert!(f.describe().contains("scene=ir"));
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
