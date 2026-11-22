//! Pipeline variants (`benchmark --variants`): the same model run with production options the
//! plain benchmark leaves off, image enhancement ([`crate::model::enhance`]) and the night / IR
//! filter rules ([`crate::model::filters`]), so their effect on accuracy, false positives and
//! speed can be compared on the same images. A variant's results are named
//! `<model>+<variant>` (e.g. `IPcam-general+pre:clahe`), so the report, `benchmark.json` and the
//! run history treat it as one more model. The run applies a variant with the functions the
//! service calls (scene statistics on the decoded frame, `enhance::apply`, `filters::apply`,
//! the verifier's `plan` / `outcomes`; [`super::served`]), so the benchmark and the service
//! cannot drift apart, and scores what the pipeline returns at each threshold of the sweep (B2:
//! overlap evidence and verifier candidates are the boxes at that threshold).
//!
//! Syntax: a comma list of variants; each is `none` (the plain pipeline) or `+`-joined parts:
//! `filters`, `pre:<mode>` (every frame), `night-pre:<mode>` (night / IR frames only),
//! `verify:<model>` (the verifier cascade with another configured model), where `<mode>` is a
//! [`Preprocess`] string (`clahe:2,8` keeps its comma). `filters` are the model's configured
//! night / IR rules ([`crate::worker::filter_rules`]); when those are all off, the built-in
//! [`night_default_filters`] preset. `verify:<model>` uses the model's configured `verify`
//! settings when they name `<model>`, else the defaults of [`VerifyConfig`]; the verifier is
//! compiled in-process on the device the model runs on and checks the candidates with the
//! service's [`crate::verify`] functions (after the filters, like the service).
//!
//! **As served** (`benchmark --served`, `benchmark.served`, the page's "As served"; refused
//! together with `--variants`): one variant per model, [`PipelineVariant::served`], named
//! `<model>+served` ([`served_name`]), built from the model's config entry through the functions
//! the worker builds its [`crate::worker::LiveSettings`] with, so nothing is re-implemented. One
//! image runs what one request does in the service, in the worker's order:
//!
//! 1. decode, then the frame's [`crate::image_stats::Scene`] (`prep.rs`);
//! 2. the model's enhancement for that scene ([`crate::config::Config::preprocess_modes`]:
//!    `night_preprocess` on night / IR frames, else `preprocess`) and the letterbox / resize;
//! 3. a night / IR frame of a model with a `night_model` goes to the night model
//!    ([`NightRoute::Model`], `worker::run::route`): the request has already paid for the
//!    model's own enhancement and resize (the prep pool runs before routing, so they count in
//!    its preprocess time), then the night model prepares the original frame with its own
//!    enhancement and runs steps 4-6 with its own settings (threshold `threshold_for_scene` on
//!    the device, its filters, its object filter, its verifier). A forwarded frame is never
//!    forwarded again. When the night model is not a served model (missing, disabled, the model
//!    itself, its file missing) or fails to compile, night frames run on the model itself, as
//!    the service's fallback does, and the result records why ([`NightRoute::Fallback`],
//!    `RunResult::served`);
//! 4. inference and the family's post-processing at the frame's threshold:
//!    `LiveSettings::threshold_for_scene` (the model's `night_threshold` on night / IR frames
//!    when set, else the device's served threshold);
//! 5. the filter rules ([`crate::worker::filter_rules`], evidence from the boxes at that
//!    threshold only), then the object filter;
//! 6. the verifier ([`crate::worker::verify_settings`]) on that answer, fail-open: a verifier
//!    that is not served or cannot compile, or a crop it cannot check, leaves the candidates
//!    unverified (kept).
//!
//! Static-box memory is left out (per-camera history of a live stream). Timing counts steps
//! 1-6 as the request costs them (enhancement in preprocess, filters and verifier in
//! postprocess, a night frame's inference is the night model's, compiled on the same device with
//! its worker's tuning). The evaluation scores what is returned at each threshold of the sweep
//! exactly ([`super::served`]: day frames follow the swept threshold, night / IR frames with a
//! `night_threshold` and frames the night model answers keep theirs). The results carry the
//! fingerprint of these settings ([`served_fingerprint`]); `GET /v1/benchmark` marks a result
//! `stale_rules` when the config's fingerprint differs now. `auto` device ordering, the device
//! recommendation and Apply read the bare `<model>` results only (a `+served` name is no
//! configured model).

use crate::config::{Config, ModelConfig};
use crate::model::enhance::{Preprocess, PreprocessModes};
use crate::model::filters::{FilterRules, OverlapRule};
use crate::verify::VerifyConfig;
use std::collections::BTreeMap;

/// What a benchmark job runs on top of the plain pipeline.
#[derive(Debug, Clone, Default)]
pub struct PipelineVariant {
    /// The variant as typed (`filters`, `pre:clahe`, ...), the suffix of the result name.
    pub name: String,
    /// Enhancement of every frame (None = none).
    pub preprocess: Option<Preprocess>,
    /// Enhancement of night / IR frames, replacing `preprocess` there.
    pub night_preprocess: Option<Preprocess>,
    /// Filter rules applied to the detections before scoring.
    pub filters: Option<FilterRules>,
    /// How `filters` were chosen (for the printed summary).
    pub filters_source: Option<&'static str>,
    /// The verifier cascade (`verify:<model>`).
    pub verify: Option<VerifyVariant>,
    /// As served: the rest of the model's serving settings (None for `--variants`).
    pub served: Option<ServedRoute>,
}

/// What an As served variant adds ([`PipelineVariant::served`]).
#[derive(Debug, Clone)]
pub struct ServedRoute {
    /// `night_threshold`: the threshold of night / IR frames run on the model itself.
    pub night_threshold: Option<f32>,
    /// `night_model` (None: the model has none).
    pub night: Option<NightRoute>,
    /// What could not be set up as configured (a verifier that is not served: fail-open).
    pub notes: Vec<String>,
    /// [`served_fingerprint`] of the settings this variant was built from.
    pub fingerprint: String,
}

/// Where the night / IR frames of an As served model go.
#[derive(Debug, Clone)]
pub enum NightRoute {
    /// To this night model, with its own As served pipeline (never routed again).
    Model {
        job: Box<super::Job>,
        variant: Box<PipelineVariant>,
    },
    /// `night_model` names a model that cannot take them: they run on the model itself.
    Fallback { model: String, reason: String },
}

impl NightRoute {
    /// The configured night model's name.
    pub fn model(&self) -> &str {
        match self {
            NightRoute::Model { job, .. } => &job.name,
            NightRoute::Fallback { model, .. } => model,
        }
    }
}

/// The result-name suffix of the As served variant (`<model>+served`).
pub const SERVED: &str = "served";

/// The result name of `model` benchmarked as served.
pub fn served_name(model: &str) -> String {
    format!("{model}+{SERVED}")
}

/// Whether the result `name` is an As served one (`<model>+served`).
pub fn is_served(name: &str) -> bool {
    base_name(name).len() != name.len()
}

/// The model a result name belongs to: `<model>` of `<model>+served`, else `name`.
pub fn base_name(name: &str) -> &str {
    let suffix = SERVED.len() + 1;
    match name.len().checked_sub(suffix) {
        Some(at)
            if at > 0
                && name.is_char_boundary(at)
                && name
                    .get(at..)
                    .is_some_and(|s| s.eq_ignore_ascii_case("+served")) =>
        {
            name.get(..at).unwrap_or(name)
        }
        _ => name,
    }
}

/// The verifier of a `verify:<model>` variant.
#[derive(Debug, Clone)]
pub struct VerifyVariant {
    pub config: VerifyConfig,
    /// The verifier model (its device is the one the verified model runs on).
    pub job: Box<super::Job>,
    /// The verifier's own image enhancement (as in the service).
    pub modes: PreprocessModes,
    /// How `config` was chosen (for the printed summary).
    pub source: &'static str,
}

impl PipelineVariant {
    pub fn modes(&self) -> PreprocessModes {
        PreprocessModes {
            preprocess: self.preprocess.unwrap_or_default(),
            night_preprocess: self.night_preprocess,
        }
    }

    /// One line: what this variant runs.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if let Some(p) = self.preprocess {
            parts.push(format!("preprocess {p}"));
        }
        if let Some(p) = self.night_preprocess {
            parts.push(format!("night/IR preprocess {p}"));
        }
        if let Some(f) = &self.filters {
            parts.push(format!(
                "filters ({}): {}",
                self.filters_source.unwrap_or("configured"),
                describe_rules(f)
            ));
        }
        if let Some(v) = &self.verify {
            parts.push(format!("verify ({}): {}", v.source, v.config.describe()));
        }
        if let Some(s) = &self.served {
            if let Some(t) = s.night_threshold {
                parts.push(format!("night / IR threshold {t}"));
            }
            match &s.night {
                Some(NightRoute::Model { job, .. }) => {
                    parts.push(format!("night / IR frames to {}", job.name))
                }
                Some(NightRoute::Fallback { model, reason }) => parts.push(format!(
                    "night_model {model} unused ({reason}): night / IR frames run here"
                )),
                None => {}
            }
            parts.extend(s.notes.iter().cloned());
            if parts.is_empty() {
                parts.push("no serving rules: the bare pipeline".into());
            }
        }
        parts.join("; ")
    }

    /// Model `m` (its entry; `name` is its effective name) as it serves under `config` (see
    /// the module docs). `profile`: the hardware profile the night model and verifier compile
    /// with (None: this machine's under the config). Never fails: what cannot be set up as
    /// configured falls back as the service does and is noted.
    pub fn served(
        config: &Config,
        m: &ModelConfig,
        profile: Option<crate::profile::Profile>,
    ) -> PipelineVariant {
        Self::served_with(config, m, profile, true)
    }

    fn served_with(
        config: &Config,
        m: &ModelConfig,
        profile: Option<crate::profile::Profile>,
        route: bool,
    ) -> PipelineVariant {
        let live = crate::worker::LiveSettings::for_model(config, m);
        let mut notes = Vec::new();
        let verify =
            live.verify
                .as_ref()
                .and_then(|v| match served_verifier(config, v, profile.clone()) {
                    Ok(vv) => Some(vv),
                    Err(why) => {
                        notes.push(format!(
                            "verifier {} {why}: candidates stay unverified (fail-open, as served)",
                            v.model
                        ));
                        None
                    }
                });
        let night = if route {
            live.night_model
                .as_deref()
                .map(|n| night_route(config, m, n, profile.clone()))
        } else {
            None
        };
        let modes = live.preprocess;
        PipelineVariant {
            name: SERVED.to_string(),
            preprocess: (modes.preprocess != Preprocess::None).then_some(modes.preprocess),
            night_preprocess: modes.night_preprocess,
            filters: (!live.filters.is_noop()).then_some(live.filters),
            filters_source: Some("configured"),
            verify,
            served: Some(ServedRoute {
                night_threshold: live.night_threshold,
                night,
                notes,
                fingerprint: served_fingerprint(config, m),
            }),
        }
    }
}

/// The verifier `v` names, as its worker would serve it; Err says why it would not be.
fn served_verifier(
    config: &Config,
    v: &VerifyConfig,
    profile: Option<crate::profile::Profile>,
) -> Result<VerifyVariant, String> {
    let entry = super::find_model(config, v.model.trim())
        .ok_or_else(|| "is not a configured model".to_string())?;
    if !entry.enabled {
        return Err("is disabled".into());
    }
    let job = super::job_as_served(config, entry, None, profile).map_err(|e| format!("{e:#}"))?;
    Ok(VerifyVariant {
        config: v.clone(),
        job: Box::new(job),
        modes: config.preprocess_modes(entry),
        source: "configured",
    })
}

/// Where the night / IR frames of `m` go when its `night_model` is `night` (`worker::run`'s
/// `night_target`: a served model other than itself).
fn night_route(
    config: &Config,
    m: &ModelConfig,
    night: &str,
    profile: Option<crate::profile::Profile>,
) -> NightRoute {
    use crate::registry::normalize_name;
    let fallback = |reason: String| NightRoute::Fallback {
        model: night.to_string(),
        reason,
    };
    let Some(entry) = super::find_model(config, night) else {
        return fallback("not a configured model".into());
    };
    if normalize_name(&entry.effective_name()) == normalize_name(&m.effective_name()) {
        return fallback("it is the model itself".into());
    }
    if !entry.enabled {
        return fallback("it is disabled".into());
    }
    match super::job_as_served(config, entry, None, profile.clone()) {
        Ok(job) => NightRoute::Model {
            job: Box::new(job),
            variant: Box::new(PipelineVariant::served_with(config, entry, profile, false)),
        },
        Err(e) => fallback(format!("{e:#}")),
    }
}

/// A fingerprint (16 hex digits of a SHA-256) of the serving settings an As served result of
/// `m` depends on: its enhancement, filter rules, object filter, `night_threshold` and verifier
/// (settings and the verifier's file), and its night model (name, file, and that model's
/// enhancement, filter rules, object filter, thresholds and verifier). The model's own day
/// thresholds (`confidence_threshold`, `device_thresholds`) are left out: they are what the
/// sweep varies, and `GET /v1/benchmark` reads any of them off the stored predictions
/// ([`super::current`]), so applying a pick does not make the result stale. A file is its path,
/// size and modification time.
pub fn served_fingerprint(config: &Config, m: &ModelConfig) -> String {
    use sha2::{Digest, Sha256};
    let file = |e: &ModelConfig| {
        let path = config.data_path(&e.path);
        let meta = std::fs::metadata(&path).ok();
        let modified = meta
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs());
        format!(
            "{}|{:?}|{modified:?}",
            path.display(),
            meta.map(|m| m.len())
        )
    };
    let verifier = |live: &crate::worker::LiveSettings| match &live.verify {
        Some(v) => format!(
            "{v:?}|{}",
            super::find_model(config, v.model.trim())
                .map(|e| format!("{}|{}", e.enabled, file(e)))
                .unwrap_or_default()
        ),
        None => String::new(),
    };
    let own = |live: &crate::worker::LiveSettings| {
        format!(
            "{:?}|{:?}|{:?}|{:?}|{}",
            live.preprocess,
            live.filters,
            live.object_filter,
            live.night_threshold,
            verifier(live)
        )
    };
    let live = crate::worker::LiveSettings::for_model(config, m);
    let night = live
        .night_model
        .as_deref()
        .map(|n| match super::find_model(config, n) {
            Some(e) => {
                let nl = crate::worker::LiveSettings::for_model(config, e);
                format!(
                    "{}|{}|{}|{}|{:?}|{}",
                    e.effective_name(),
                    e.enabled,
                    file(e),
                    nl.confidence_threshold,
                    nl.device_thresholds,
                    own(&nl)
                )
            }
            None => format!("{n}|missing"),
        });
    let text = format!("v1|{}|{night:?}", own(&live));
    let digest = Sha256::digest(text.as_bytes());
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn describe_rules(f: &FilterRules) -> String {
    let mut out = Vec::new();
    if let Some(t) = f.night_confidence_threshold {
        out.push(format!("night threshold {t}"));
    }
    for (k, v) in &f.label_thresholds {
        out.push(format!("{k} >= {v}"));
    }
    for (k, v) in &f.night_label_thresholds {
        out.push(format!("night {k} >= {v}"));
    }
    if let Some([a, b]) = f.person_aspect {
        out.push(format!("person aspect {a}..{b}"));
    }
    if let Some(a) = f.min_box_area_frac {
        out.push(format!("min area {a}"));
    }
    for r in &f.suppress_overlaps {
        out.push(format!(
            "drop {} on {} (IoU {}{})",
            r.drop,
            r.if_overlaps,
            r.iou,
            if r.night_only { ", night" } else { "" }
        ));
    }
    out.join(", ")
}

/// One `--variants` entry as typed (filters not yet resolved against a model).
#[derive(Debug, Clone, PartialEq)]
pub struct VariantSpec {
    pub name: String,
    pub preprocess: Option<Preprocess>,
    pub night_preprocess: Option<Preprocess>,
    pub filters: bool,
    /// The verifier model of `verify:<model>`.
    pub verify: Option<String>,
}

impl VariantSpec {
    /// The plain pipeline (`none`).
    pub fn is_plain(&self) -> bool {
        self.preprocess.is_none()
            && self.night_preprocess.is_none()
            && !self.filters
            && self.verify.is_none()
    }

    /// The variant for model entry `m` (None: a `--model` file, which gets the global rules)
    /// under `config`; None for the plain pipeline. Fails when a `verify:<model>` verifier is
    /// not a configured model with its file present, or is the model itself (`name` is the
    /// verified model's name).
    pub fn resolve(
        &self,
        config: &Config,
        m: Option<&ModelConfig>,
        name: &str,
    ) -> anyhow::Result<Option<PipelineVariant>> {
        if self.is_plain() {
            return Ok(None);
        }
        let verify = match &self.verify {
            Some(v) => Some(resolve_verifier(config, m, name, v)?),
            None => None,
        };
        let (filters, filters_source) = if self.filters {
            let rules = crate::worker::filter_rules(config, m.unwrap_or(&ModelConfig::default()));
            if rules.is_noop() {
                (Some(night_default_filters()), Some("night-default preset"))
            } else {
                (Some(rules), Some("configured"))
            }
        } else {
            (None, None)
        };
        Ok(Some(PipelineVariant {
            name: self.name.clone(),
            preprocess: self.preprocess,
            night_preprocess: self.night_preprocess,
            filters,
            filters_source,
            verify,
            served: None,
        }))
    }
}

/// The verifier `verifier` of model `name` (entry `m`): its config entry as a job, and the
/// verify settings of `m` (else the global ones) when they name it, else the defaults.
fn resolve_verifier(
    config: &Config,
    m: Option<&ModelConfig>,
    name: &str,
    verifier: &str,
) -> anyhow::Result<VerifyVariant> {
    use crate::registry::normalize_name;
    let key = normalize_name(verifier);
    if key == normalize_name(name) {
        anyhow::bail!("verify:{verifier}: a model cannot verify itself; name another model");
    }
    let entry = super::find_model(config, verifier).ok_or_else(|| {
        anyhow::anyhow!(
            "verify:{verifier}: not a configured model; add it to `models` in the config (it may \
             be disabled)"
        )
    })?;
    let job = super::job_for_config(config, entry, None)
        .map_err(|e| anyhow::anyhow!("verify:{verifier}: {e:#}"))?;
    let configured = m
        .and_then(|m| m.verify.as_ref())
        .or(config.verify.as_ref())
        .filter(|v| v.enabled && normalize_name(&v.model) == key);
    let (config_v, source) = match configured {
        Some(v) => (v.clone(), "configured"),
        None => (VerifyConfig::with_model(entry.effective_name()), "defaults"),
    };
    Ok(VerifyVariant {
        config: config_v,
        job: Box::new(job),
        modes: config.preprocess_modes(entry),
        source,
    })
}

/// A parsed `--variants` list (one clap value).
#[derive(Debug, Clone, PartialEq)]
pub struct VariantList(pub Vec<VariantSpec>);

/// The `filters` preset used when the model has no night / IR rules configured: night / IR
/// confidence floor 0.5, night / IR `person` floor 0.6, `person` width / height within
/// `[0.2, 0.9]` (all frames), and drop a `person` box on a `potted plant` box (IoU 0.5 or 80%
/// covered, all frames). The same values as the Config page's suggested Night / IR settings.
pub fn night_default_filters() -> FilterRules {
    FilterRules {
        night_confidence_threshold: Some(0.5),
        label_thresholds: BTreeMap::new(),
        night_label_thresholds: BTreeMap::from([("person".to_string(), 0.6)]),
        person_aspect: Some([0.2, 0.9]),
        min_box_area_frac: None,
        suppress_overlaps: vec![OverlapRule {
            drop: "person".into(),
            if_overlaps: "potted plant".into(),
            iou: crate::model::filters::DEFAULT_OVERLAP_IOU,
            night_only: false,
        }],
    }
}

/// Split a comma list, re-joining a piece that starts with a digit to the one before it
/// (`pre:clahe:2,8` keeps its comma).
fn split_list(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for piece in s.split(',') {
        let p = piece.trim();
        match out.last_mut() {
            Some(last) if p.starts_with(|c: char| c.is_ascii_digit()) => {
                last.push(',');
                last.push_str(p);
            }
            _ => out.push(p.to_string()),
        }
    }
    out
}

/// Parse one variant (see the module docs).
pub fn parse_variant(text: &str) -> Result<VariantSpec, String> {
    let name = text.trim().to_ascii_lowercase();
    let mut v = VariantSpec {
        name: name.clone(),
        preprocess: None,
        night_preprocess: None,
        filters: false,
        verify: None,
    };
    if name.is_empty() {
        return Err(
            "an empty variant; use none, filters, pre:<mode>, night-pre:<mode> or verify:<model>"
                .into(),
        );
    }
    if name == "none" {
        return Ok(v);
    }
    for part in name.split('+') {
        let part = part.trim();
        if part == "filters" {
            if v.filters {
                return Err(format!("variant '{text}': filters given twice"));
            }
            v.filters = true;
        } else if let Some(mode) = part.strip_prefix("night-pre:") {
            if v.night_preprocess.is_some() {
                return Err(format!("variant '{text}': night-pre given twice"));
            }
            v.night_preprocess = Some(mode.parse()?);
        } else if let Some(model) = part.strip_prefix("verify:") {
            if v.verify.is_some() {
                return Err(format!("variant '{text}': verify given twice"));
            }
            let model = model.trim();
            if model.is_empty() {
                return Err(format!(
                    "variant '{text}': verify needs a model name, e.g. verify:yolo26s"
                ));
            }
            v.verify = Some(model.to_string());
        } else if let Some(mode) = part.strip_prefix("pre:") {
            if v.preprocess.is_some() {
                return Err(format!("variant '{text}': pre given twice"));
            }
            v.preprocess = Some(mode.parse()?);
        } else {
            return Err(format!(
                "variant '{text}': unknown part '{part}'; use none, filters, pre:<mode>, \
                 night-pre:<mode> or verify:<model> (joined with +), with <mode> one of: {}",
                crate::model::enhance::PREPROCESS_FORMS
            ));
        }
    }
    Ok(v)
}

/// Parse a `--variants` list; names must be unique.
pub fn parse_variants(s: &str) -> Result<Vec<VariantSpec>, String> {
    let mut out: Vec<VariantSpec> = Vec::new();
    for piece in split_list(s) {
        let v = parse_variant(&piece)?;
        if out.iter().any(|o| o.name == v.name) {
            return Err(format!("variant '{}' is listed twice", v.name));
        }
        out.push(v);
    }
    if out.is_empty() {
        return Err("no variants given; e.g. --variants none,filters,pre:clahe".into());
    }
    Ok(out)
}

/// The result name of `model` with `variant`.
pub fn variant_name(model: &str, variant: &PipelineVariant) -> String {
    format!("{model}+{}", variant.name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lists_with_mode_commas() {
        let v =
            parse_variants("none, filters,pre:clahe,pre:gamma:0.6,night-pre:clahe:3,4").unwrap();
        let names: Vec<&str> = v.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "none",
                "filters",
                "pre:clahe",
                "pre:gamma:0.6",
                "night-pre:clahe:3,4"
            ]
        );
        assert!(v[0].is_plain());
        assert!(v[1].filters && v[1].preprocess.is_none());
        assert_eq!(v[2].preprocess, Some(Preprocess::CLAHE));
        assert_eq!(v[3].preprocess, Some(Preprocess::Gamma(0.6)));
        assert_eq!(
            v[4].night_preprocess,
            Some(Preprocess::Clahe {
                clip: 3.0,
                tiles: 4
            })
        );
        let combo = parse_variant("filters+night-pre:grayscale").unwrap();
        assert!(combo.filters);
        assert_eq!(combo.night_preprocess, Some(Preprocess::Grayscale));
    }

    #[test]
    fn rejects_bad_variants() {
        for bad in [
            "",
            "pre:sharpen",
            "blur",
            "filters,filters",
            "filters+filters",
            "pre:clahe+pre:none",
            "pre:gamma:0",
        ] {
            assert!(parse_variants(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn filters_resolve_from_the_config_else_the_preset() {
        let config = Config::default();
        let m = ModelConfig::default();
        let spec = parse_variant("filters").unwrap();
        let v = spec.resolve(&config, Some(&m), "m").unwrap().unwrap();
        assert_eq!(v.filters, Some(night_default_filters()));
        assert_eq!(v.filters_source, Some("night-default preset"));
        assert!(
            v.describe().contains("night threshold 0.5"),
            "{}",
            v.describe()
        );

        let config = Config {
            night_confidence_threshold: Some(0.7),
            ..Config::default()
        };
        let v = spec.resolve(&config, Some(&m), "m").unwrap().unwrap();
        assert_eq!(v.filters.unwrap().night_confidence_threshold, Some(0.7));
        assert_eq!(v.filters_source, Some("configured"));
        // A --model file gets the global rules.
        let v = spec.resolve(&config, None, "m").unwrap().unwrap();
        assert_eq!(v.filters.unwrap().night_confidence_threshold, Some(0.7));

        assert!(
            parse_variant("none")
                .unwrap()
                .resolve(&config, None, "m")
                .unwrap()
                .is_none()
        );
        let pre = parse_variant("night-pre:clahe")
            .unwrap()
            .resolve(&config, None, "m")
            .unwrap()
            .unwrap();
        assert!(pre.filters.is_none());
        assert_eq!(pre.modes().preprocess, Preprocess::None);
        assert_eq!(pre.modes().night_preprocess, Some(Preprocess::CLAHE));
        assert_eq!(
            variant_name("IPcam-general", &pre),
            "IPcam-general+night-pre:clahe"
        );
    }

    #[test]
    fn verify_variants_parse_and_resolve() {
        let v = parse_variants("verify:yolo26s,filters+verify:YOLO26S").unwrap();
        assert_eq!(v[0].verify.as_deref(), Some("yolo26s"));
        assert!(!v[0].is_plain() && !v[0].filters);
        assert!(v[1].filters);
        assert_eq!(v[1].name, "filters+verify:yolo26s");
        assert!(parse_variant("verify:").is_err());
        assert!(parse_variant("verify:a+verify:b").is_err());

        let dir = std::env::temp_dir().join(format!("bop-variant-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["a.onnx", "b.onnx"] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        let entry = |n: &str| ModelConfig {
            name: Some(n.into()),
            path: dir.join(format!("{n}.onnx")),
            ..ModelConfig::default()
        };
        let mut a = entry("a");
        a.verify = Some(VerifyConfig {
            min_confidence: 0.6,
            ..VerifyConfig::with_model("B")
        });
        let config = Config {
            models: vec![a.clone(), entry("b")],
            ..Config::default()
        };
        let spec = parse_variant("verify:b").unwrap();
        let r = spec.resolve(&config, Some(&a), "a").unwrap().unwrap();
        let vv = r.verify.as_ref().unwrap();
        assert_eq!(vv.config.min_confidence, 0.6);
        assert_eq!(vv.source, "configured");
        assert_eq!(vv.job.name, "b");
        assert!(r.filters.is_none());
        assert!(
            r.describe()
                .contains("verify (configured): B verifies person"),
            "{}",
            r.describe()
        );
        // A --model file (no entry): the defaults.
        let r = spec.resolve(&config, None, "x").unwrap().unwrap();
        let vv = r.verify.unwrap();
        assert_eq!(vv.config, VerifyConfig::with_model("b"));
        assert_eq!(vv.source, "defaults");
        // Itself, unknown.
        let e = parse_variant("verify:a")
            .unwrap()
            .resolve(&config, Some(&a), "a")
            .unwrap_err()
            .to_string();
        assert!(e.contains("cannot verify itself"), "{e}");
        let e = parse_variant("verify:zzz")
            .unwrap()
            .resolve(&config, Some(&a), "a")
            .unwrap_err()
            .to_string();
        assert!(e.contains("not a configured model"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod served_tests {
    use super::*;

    /// As served is built from the model's live settings; a verifier that is not served is
    /// left out with a note (fail-open, as the service), a model without rules says so.
    #[test]
    fn served_variants_follow_the_live_settings() {
        let dir = std::env::temp_dir().join(format!("bop-served-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.onnx"), b"x").unwrap();
        let a = ModelConfig {
            name: Some("a".into()),
            path: dir.join("a.onnx"),
            ..ModelConfig::default()
        };
        let plain = PipelineVariant::served(&Config::default(), &a, None);
        assert_eq!(plain.name, SERVED);
        assert!(plain.filters.is_none() && plain.verify.is_none() && plain.preprocess.is_none());
        assert_eq!(plain.describe(), "no serving rules: the bare pipeline");

        let mut m = a.clone();
        m.night_threshold = Some(0.3);
        m.night_preprocess = Some(Preprocess::CLAHE);
        m.label_thresholds = Some(BTreeMap::from([("person".to_string(), 0.6)]));
        m.verify = Some(VerifyConfig::with_model("ghost"));
        let config = Config {
            models: vec![m.clone()],
            ..Config::default()
        };
        let v = PipelineVariant::served(&config, &m, None);
        let s = v.served.as_ref().unwrap();
        assert_eq!(s.night_threshold, Some(0.3));
        assert!(s.night.is_none());
        assert_eq!(v.night_preprocess, Some(Preprocess::CLAHE));
        assert_eq!(
            v.filters.as_ref().unwrap().label_thresholds.get("person"),
            Some(&0.6)
        );
        assert!(v.verify.is_none());
        assert!(
            s.notes[0].contains("verifier ghost is not a configured model"),
            "{:?}",
            s.notes
        );
        let text = v.describe();
        assert!(text.contains("night / IR threshold 0.3"), "{text}");
        assert!(text.contains("unverified (fail-open"), "{text}");
        assert_eq!(s.fingerprint, served_fingerprint(&config, &m));
        assert_eq!(s.fingerprint.len(), 16);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
