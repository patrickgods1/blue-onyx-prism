//! Pipeline variants (`benchmark --variants`): the same model run with production options the
//! plain benchmark leaves off, image enhancement ([`crate::model::enhance`]) and the night / IR
//! filter rules ([`crate::model::filters`]), so their effect on accuracy, false positives and
//! speed can be compared on the same images. A variant's results are named
//! `<model>+<variant>` (e.g. `IPcam-general+pre:clahe`), so the report, `benchmark.json` and the
//! run history treat it as one more model. `Pipeline::run` applies a variant with the functions
//! the service calls (scene statistics on the decoded frame, `enhance::apply`, `filters::apply`
//! before scoring), so the benchmark and the service cannot drift apart.
//!
//! Syntax: a comma list of variants; each is `none` (the plain pipeline) or `+`-joined parts:
//! `filters`, `pre:<mode>` (every frame), `night-pre:<mode>` (night / IR frames only), where
//! `<mode>` is a [`Preprocess`] string (`clahe:2,8` keeps its comma). `filters` are the model's
//! configured night / IR rules ([`crate::worker::filter_rules`]); when those are all off, the
//! built-in [`night_default_filters`] preset.

use crate::config::{Config, ModelConfig};
use crate::model::enhance::{Preprocess, PreprocessModes};
use crate::model::filters::{FilterRules, OverlapRule};
use std::collections::BTreeMap;

/// What a benchmark job runs on top of the plain pipeline.
#[derive(Debug, Clone, PartialEq, Default)]
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
        parts.join("; ")
    }
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
}

impl VariantSpec {
    /// The plain pipeline (`none`).
    pub fn is_plain(&self) -> bool {
        self.preprocess.is_none() && self.night_preprocess.is_none() && !self.filters
    }

    /// The variant for model entry `m` (None: a `--model` file, which gets the global rules)
    /// under `config`; None for the plain pipeline.
    pub fn resolve(&self, config: &Config, m: Option<&ModelConfig>) -> Option<PipelineVariant> {
        if self.is_plain() {
            return None;
        }
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
        Some(PipelineVariant {
            name: self.name.clone(),
            preprocess: self.preprocess,
            night_preprocess: self.night_preprocess,
            filters,
            filters_source,
        })
    }
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
    };
    if name.is_empty() {
        return Err("an empty variant; use none, filters, pre:<mode> or night-pre:<mode>".into());
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
        } else if let Some(mode) = part.strip_prefix("pre:") {
            if v.preprocess.is_some() {
                return Err(format!("variant '{text}': pre given twice"));
            }
            v.preprocess = Some(mode.parse()?);
        } else {
            return Err(format!(
                "variant '{text}': unknown part '{part}'; use none, filters, pre:<mode> or \
                 night-pre:<mode> (joined with +), with <mode> one of: {}",
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
        let v = spec.resolve(&config, Some(&m)).unwrap();
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
        let v = spec.resolve(&config, Some(&m)).unwrap();
        assert_eq!(v.filters.unwrap().night_confidence_threshold, Some(0.7));
        assert_eq!(v.filters_source, Some("configured"));
        // A --model file gets the global rules.
        let v = spec.resolve(&config, None).unwrap();
        assert_eq!(v.filters.unwrap().night_confidence_threshold, Some(0.7));

        assert!(
            parse_variant("none")
                .unwrap()
                .resolve(&config, None)
                .is_none()
        );
        let pre = parse_variant("night-pre:clahe")
            .unwrap()
            .resolve(&config, None)
            .unwrap();
        assert!(pre.filters.is_none());
        assert_eq!(pre.modes().preprocess, Preprocess::None);
        assert_eq!(pre.modes().night_preprocess, Some(Preprocess::CLAHE));
        assert_eq!(
            variant_name("IPcam-general", &pre),
            "IPcam-general+night-pre:clahe"
        );
    }
}
