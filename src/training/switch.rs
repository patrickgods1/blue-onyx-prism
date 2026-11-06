//! A safe switch of the default model, pure (no I/O; `crate::server` feeds it the facts): what
//! changes between the current default and a trained model ([`diff`]), which per-model settings
//! carry over ([`copy_settings`]), the threshold the held-out benchmark suggests ([`suggest`]),
//! the switch itself with its [`PreviousDefault`] record ([`apply_switch`]), and Revert
//! ([`revert`]), which writes the opposite record, so a second Revert redoes the switch.
//!
//! Classes and the device belong to the model. Device profiles and every other threshold are
//! not the same kind of thing: profiles are measurements of the old model and stay behind, while
//! the thresholds the owner set by hand carry over, because they say how strict the cameras
//! should be. Settings naming a class the new model does not have are narrowed or left out.

use crate::benchmark::report::ModelResult;
use crate::config::{Config, ModelConfig, PreviousDefault, ThresholdSource};
use crate::config_merge::field_label;
use crate::registry::normalize_name;
use crate::training::autolabel::catalog::same_class;
use anyhow::{Result, anyhow, bail};
use serde::Serialize;
use std::collections::BTreeMap;

/// The fields a switch can change on a model entry, with the helpers that work on them by name.
macro_rules! switch_fields {
    ($($field:ident),* $(,)?) => {
        /// The per-model settings a switch can change.
        pub const SWITCH_FIELDS: &[&str] = &[$(stringify!($field)),*];

        fn swap_field(a: &mut ModelConfig, b: &mut ModelConfig, name: &str) {
            match name {
                $(stringify!($field) => std::mem::swap(&mut a.$field, &mut b.$field),)*
                _ => {}
            }
        }

        fn field_differs(a: &ModelConfig, b: &ModelConfig, name: &str) -> bool {
            match name {
                $(stringify!($field) => a.$field != b.$field,)*
                _ => false,
            }
        }
    };
}

switch_fields!(
    enabled,
    confidence_threshold,
    device_thresholds,
    object_filter,
    night_confidence_threshold,
    label_thresholds,
    night_label_thresholds,
    person_aspect,
    min_box_area_frac,
    suppress_overlaps,
    static_suppression,
    preprocess,
    night_preprocess,
    night_model,
    night_threshold,
    verify,
);

/// `models`' entry named `name` (spelled as the registry does).
fn entry<'a>(models: &'a [ModelConfig], name: &str) -> Option<&'a ModelConfig> {
    let key = normalize_name(name);
    models
        .iter()
        .find(|m| normalize_name(&m.effective_name()) == key)
}

fn entry_mut<'a>(models: &'a mut [ModelConfig], name: &str) -> Option<&'a mut ModelConfig> {
    let key = normalize_name(name);
    models
        .iter_mut()
        .find(|m| normalize_name(&m.effective_name()) == key)
}

fn same_name(a: &str, b: &str) -> bool {
    normalize_name(a) == normalize_name(b)
}

/// The model serving `/v1/vision/detection` now (the name of the entry), if any.
pub fn current_default(c: &Config) -> Option<String> {
    c.effective_default_index()
        .and_then(|i| c.models.get(i))
        .map(ModelConfig::effective_name)
}

/// The classes a model reports: `classes` narrowed by its own `object_filter`, else `global`.
/// An empty filter keeps every class.
pub fn reported(classes: &[String], own: Option<&[String]>, global: &[String]) -> Vec<String> {
    let filter = own.unwrap_or(global);
    if filter.is_empty() {
        return classes.to_vec();
    }
    classes
        .iter()
        .filter(|c| filter.iter().any(|f| f.eq_ignore_ascii_case(c)))
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Copying settings
// ---------------------------------------------------------------------------------------------

/// A setting that carried over, with the value it has now.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Copied {
    pub label: String,
    pub value: String,
}

/// A setting that did not carry over, and why.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NotCopied {
    pub label: String,
    pub why: String,
}

/// What [`copy_settings`] did.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct CopyReport {
    pub copied: Vec<Copied>,
    pub not_copied: Vec<NotCopied>,
}

impl CopyReport {
    fn yes(&mut self, label: &str, value: String) {
        self.copied.push(Copied {
            label: label.to_string(),
            value,
        });
    }

    fn no(&mut self, label: &str, why: impl Into<String>) {
        self.not_copied.push(NotCopied {
            label: label.to_string(),
            why: why.into(),
        });
    }
}

/// Whether the class list has `name`; an unknown list (empty) has everything.
fn has_class(classes: &[String], name: &str) -> bool {
    classes.is_empty() || classes.iter().any(|c| same_class(c, name))
}

/// `name` in the spelling of `classes` (a filter's `motorbike` on a `motorcycle` model).
fn spelled(classes: &[String], name: &str) -> String {
    classes
        .iter()
        .find(|c| same_class(c, name))
        .cloned()
        .unwrap_or_else(|| name.to_string())
}

/// The part of a per-label map the model can use: (kept, dropped labels).
fn narrow_map(
    map: &BTreeMap<String, f32>,
    classes: &[String],
) -> (BTreeMap<String, f32>, Vec<String>) {
    let mut kept = BTreeMap::new();
    let mut dropped = Vec::new();
    for (k, v) in map {
        if has_class(classes, k) {
            kept.insert(spelled(classes, k), *v);
        } else {
            dropped.push(k.clone());
        }
    }
    (kept, dropped)
}

/// Copy `from`'s behaviour settings onto `to`, the entry of the model that takes over, limited to
/// what still makes sense for it. `classes` are `to`'s classes (empty = unknown, nothing is
/// narrowed); `models` the configured entries. Settings `from` does not set leave `to` as it is.
pub fn copy_settings(
    from: &ModelConfig,
    to: &mut ModelConfig,
    classes: &[String],
    models: &[ModelConfig],
) -> CopyReport {
    let mut r = CopyReport::default();
    let source = from.effective_name();
    let name = to.effective_name();
    r.no(
        "Classes",
        "they belong to the model: it reports what it was trained on",
    );
    r.no(
        "Device",
        "it stays where this model runs; the machine may suit it differently",
    );
    if !from.device_thresholds.is_empty() {
        r.no(
            "Confidence per device",
            format!("measured for {source}, not for this model; use the suggestion instead"),
        );
    }
    if let Some(t) = from.confidence_threshold {
        to.confidence_threshold = Some(t);
        r.yes("Confidence threshold", format!("{t:.2}"));
    }
    if let Some(f) = &from.object_filter {
        let (kept, dropped): (Vec<&String>, Vec<&String>) =
            f.iter().partition(|l| has_class(classes, l));
        if kept.is_empty() && !f.is_empty() {
            r.no(
                "Object filter",
                format!("it names none of this model's classes ({})", join(f)),
            );
        } else {
            let list: Vec<String> = kept.iter().map(|l| spelled(classes, l)).collect();
            r.yes("Object filter", filter_text(&list));
            to.object_filter = Some(list);
            if !dropped.is_empty() {
                r.no(
                    "Object filter entries",
                    format!(
                        "this model has no such class: {}",
                        dropped
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                );
            }
        }
    }
    if let Some(n) = &from.night_model {
        let used_by = models.iter().find(|m| {
            m.night_model
                .as_deref()
                .is_some_and(|x| same_name(x, &name))
        });
        if same_name(n, &name) {
            r.no("Night model", "that is this model itself");
        } else if let Some(u) = used_by {
            r.no(
                "Night model",
                format!(
                    "{} uses this model as its night model, and night models do not chain",
                    u.effective_name()
                ),
            );
        } else {
            to.night_model = Some(n.clone());
            r.yes("Night model", n.clone());
        }
    }
    if let Some(t) = from.night_threshold {
        to.night_threshold = Some(t);
        r.yes("Night threshold (same model)", format!("{t:.2}"));
    }
    if let Some(t) = from.night_confidence_threshold {
        to.night_confidence_threshold = Some(t);
        r.yes("Night minimum confidence", format!("{t:.2}"));
    }
    let maps = [
        ("Label thresholds", &from.label_thresholds, 0),
        ("Night label thresholds", &from.night_label_thresholds, 1),
    ];
    for (label, map, which) in maps {
        let Some(map) = map else { continue };
        let (kept, dropped) = narrow_map(map, classes);
        if !dropped.is_empty() {
            r.no(
                label,
                format!("this model has no such class: {}", dropped.join(", ")),
            );
        }
        if !kept.is_empty() {
            r.yes(label, map_text(&kept));
            let slot = if which == 0 {
                &mut to.label_thresholds
            } else {
                &mut to.night_label_thresholds
            };
            *slot = Some(kept);
        }
    }
    if let Some(a) = from.person_aspect {
        if has_class(classes, "person") {
            to.person_aspect = Some(a);
            r.yes("Person aspect", aspect_text(Some(a)));
        } else {
            r.no("Person aspect", "this model has no person class");
        }
    }
    if let Some(a) = from.min_box_area_frac {
        to.min_box_area_frac = Some(a);
        r.yes("Min box area", area_text(Some(a)));
    }
    if let Some(rules) = &from.suppress_overlaps {
        let (kept, dropped): (Vec<_>, Vec<_>) = rules
            .iter()
            .cloned()
            .partition(|o| has_class(classes, &o.drop) && has_class(classes, &o.if_overlaps));
        if !dropped.is_empty() {
            r.no(
                "Overlap rules",
                format!("{} name a class this model does not have", dropped.len()),
            );
        }
        if !kept.is_empty() || rules.is_empty() {
            r.yes("Overlap rules", overlap_text(&kept));
            to.suppress_overlaps = Some(kept);
        }
    }
    if let Some(s) = &from.static_suppression {
        to.static_suppression = Some(s.clone());
        r.yes("Static-box suppression", static_text(s));
    }
    if let Some(p) = from.preprocess {
        to.preprocess = Some(p);
        r.yes("Image enhancement", preprocess_text(Some(p)));
    }
    if let Some(p) = from.night_preprocess {
        to.night_preprocess = Some(p);
        r.yes("Night image enhancement", preprocess_text(Some(p)));
    }
    if let Some(v) = &from.verify {
        let some_label = v.labels.is_empty() || v.labels.iter().any(|l| has_class(classes, l));
        if v.enabled && same_name(&v.model, &name) {
            r.no("Verifier", "that is this model itself");
        } else if !some_label {
            r.no(
                "Verifier",
                format!(
                    "it checks {}, which this model does not report",
                    v.labels.join(", ")
                ),
            );
        } else {
            to.verify = Some(v.clone());
            r.yes("Verifier", verify_text(Some(v)));
        }
    }
    r
}

// ---------------------------------------------------------------------------------------------
// The words of the diff
// ---------------------------------------------------------------------------------------------

fn join(list: &[String]) -> String {
    list.join(", ")
}

/// A class list, shortened: the page's table stays readable with a COCO head.
fn class_text(list: &[String]) -> String {
    const SHOWN: usize = 12;
    match list.len() {
        0 => "none".to_string(),
        n if n <= SHOWN => join(list),
        n => format!(
            "{} and {} more",
            list.iter()
                .take(SHOWN)
                .cloned()
                .collect::<Vec<_>>()
                .join(", "),
            n - SHOWN
        ),
    }
}

fn filter_text(list: &[String]) -> String {
    if list.is_empty() {
        "every class of the model".to_string()
    } else {
        class_text(list)
    }
}

fn map_text(m: &BTreeMap<String, f32>) -> String {
    if m.is_empty() {
        return "none".to_string();
    }
    m.iter()
        .map(|(k, v)| format!("{k} {v:.2}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn off_or(t: Option<f32>) -> String {
    t.map_or_else(|| "off".to_string(), |t| format!("{t:.2}"))
}

fn aspect_text(a: Option<[f32; 2]>) -> String {
    a.map_or_else(
        || "off".to_string(),
        |[lo, hi]| format!("width/height {lo:.2} to {hi:.2}"),
    )
}

fn area_text(a: Option<f32>) -> String {
    a.map_or_else(
        || "off".to_string(),
        |a| format!("{:.2}% of the image", a * 100.0),
    )
}

fn overlap_text(rules: &[crate::model::filters::OverlapRule]) -> String {
    if rules.is_empty() {
        return "none".to_string();
    }
    rules
        .iter()
        .map(|r| {
            let night = if r.night_only { " (night)" } else { "" };
            format!("drop {} over {}{night}", r.drop, r.if_overlaps)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn static_text(s: &crate::static_memory::StaticSuppression) -> String {
    if !s.enabled {
        return "off".to_string();
    }
    format!(
        "on ({} after {} repeats within {} min)",
        if s.labels.is_empty() {
            "every label".to_string()
        } else {
            s.labels.join(", ")
        },
        s.hits,
        s.window_minutes
    )
}

fn preprocess_text(p: Option<crate::model::enhance::Preprocess>) -> String {
    p.map_or_else(|| "none".to_string(), String::from)
}

fn verify_text(v: Option<&crate::verify::VerifyConfig>) -> String {
    match v.filter(|v| v.enabled) {
        Some(v) => format!(
            "{} checks {}{}",
            v.model,
            if v.labels.is_empty() {
                "every label".to_string()
            } else {
                v.labels.join(", ")
            },
            if v.night_only { " at night" } else { "" }
        ),
        None => "off".to_string(),
    }
}

/// `global` with `own`'s entries over it (the merge the filters use).
fn merged(global: &BTreeMap<String, f32>, own: Option<&BTreeMap<String, f32>>) -> String {
    let mut m = global.clone();
    if let Some(own) = own {
        m.extend(own.iter().map(|(k, v)| (k.clone(), *v)));
    }
    map_text(&m)
}

/// The value of every compared setting of `m` as it would run (`(key, label, text)`): its own
/// setting, else the global one.
fn setting_texts(c: &Config, m: &ModelConfig, device: &str) -> Vec<(&'static str, String, String)> {
    let (t, source) = m.threshold_on(Some(device), c.confidence_threshold);
    let source = match source {
        ThresholdSource::Device(k) => format!("profile for {k}"),
        ThresholdSource::Model => "this model's own".to_string(),
        _ => "global default".to_string(),
    };
    let filter = m.object_filter.as_deref().unwrap_or(&c.object_filter);
    let row = |key: &'static str, label: &str, text: String| (key, label.to_string(), text);
    vec![
        row(
            "confidence_threshold",
            "Confidence threshold",
            format!("{t:.2} ({source})"),
        ),
        row("object_filter", "Object filter", filter_text(filter)),
        row(
            "night_model",
            "Night model",
            m.night_model.clone().unwrap_or_else(|| "none".to_string()),
        ),
        row(
            "night_threshold",
            "Night threshold (same model)",
            off_or(m.night_threshold),
        ),
        row(
            "night_confidence_threshold",
            "Night minimum confidence",
            off_or(
                m.night_confidence_threshold
                    .or(c.night_confidence_threshold),
            ),
        ),
        row(
            "label_thresholds",
            "Label thresholds",
            merged(&c.label_thresholds, m.label_thresholds.as_ref()),
        ),
        row(
            "night_label_thresholds",
            "Night label thresholds",
            merged(&c.night_label_thresholds, m.night_label_thresholds.as_ref()),
        ),
        row(
            "person_aspect",
            "Person aspect",
            aspect_text(m.person_aspect.or(c.person_aspect)),
        ),
        row(
            "min_box_area_frac",
            "Min box area",
            area_text(m.min_box_area_frac.or(c.min_box_area_frac)),
        ),
        row(
            "suppress_overlaps",
            "Overlap rules",
            overlap_text(
                m.suppress_overlaps
                    .as_deref()
                    .unwrap_or(&c.suppress_overlaps),
            ),
        ),
        row(
            "static_suppression",
            "Static-box suppression",
            static_text(
                m.static_suppression
                    .as_ref()
                    .unwrap_or(&c.static_suppression),
            ),
        ),
        row(
            "preprocess",
            "Image enhancement",
            preprocess_text(m.preprocess.or(c.preprocess)),
        ),
        row(
            "night_preprocess",
            "Night image enhancement",
            preprocess_text(m.night_preprocess.or(c.night_preprocess)),
        ),
        row(
            "verify",
            "Verifier",
            verify_text(m.verify.as_ref().or(c.verify.as_ref())),
        ),
    ]
}

/// One line of the diff: the setting now, on the new model as it stands, and on the new model
/// with the settings copied (the page's checkbox picks one).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DiffRow {
    pub key: String,
    pub label: String,
    pub current: String,
    pub new: String,
    pub with_copy: String,
}

/// The classes on either side of a switch.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct ClassDiff {
    pub current: String,
    pub new: String,
    /// Reported today, not after the switch: the loud part, Blue Iris may filter on labels.
    pub lost: Vec<String>,
    /// Reported only after the switch, on the new model as it stands.
    pub gained: Vec<String>,
    /// [`ClassDiff::gained`] with the settings copied (the copied filter narrows it).
    pub gained_with_copy: Vec<String>,
    /// Why the classes could not be compared.
    pub unknown: Option<String>,
}

/// What differs between the current default and the new model.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct SwitchDiff {
    pub current: String,
    pub new: String,
    pub rows: Vec<DiffRow>,
    pub classes: ClassDiff,
    pub copy: CopyReport,
}

/// One side of [`diff`].
#[derive(Debug, Clone, Copy)]
pub struct Side<'a> {
    /// The model's classes; empty = not known.
    pub classes: &'a [String],
    /// Why they are not known.
    pub unknown: Option<&'a str>,
    /// The device it runs on, or would.
    pub device: &'a str,
}

fn minus(a: &[String], b: &[String]) -> Vec<String> {
    a.iter()
        .filter(|x| !b.iter().any(|y| same_class(x, y)))
        .cloned()
        .collect()
}

/// The diff between the entries `current` and `new` of `c` (the model must be in `c`). `rows`
/// holds every compared setting with a difference in one of the two outcomes, and the threshold
/// always (the page's accepted suggestion lands in that row).
pub fn diff(
    c: &Config,
    current: &ModelConfig,
    new: &ModelConfig,
    cur_side: Side<'_>,
    new_side: Side<'_>,
) -> SwitchDiff {
    let mut copied = new.clone();
    let copy = copy_settings(current, &mut copied, new_side.classes, &c.models);
    let a = setting_texts(c, current, cur_side.device);
    let b = setting_texts(c, new, new_side.device);
    let d = setting_texts(c, &copied, new_side.device);
    let mut rows: Vec<DiffRow> = a
        .into_iter()
        .zip(b)
        .zip(d)
        .filter(|(((key, _, x), (_, _, y)), (_, _, z))| {
            *key == "confidence_threshold" || x != y || x != z
        })
        .map(
            |(((key, label, current), (_, _, new)), (_, _, with_copy))| DiffRow {
                key: key.to_string(),
                label,
                current,
                new,
                with_copy,
            },
        )
        .collect();
    if cur_side.device != new_side.device {
        rows.push(DiffRow {
            key: "device".to_string(),
            label: "Device".to_string(),
            current: cur_side.device.to_string(),
            new: new_side.device.to_string(),
            with_copy: new_side.device.to_string(),
        });
    }
    let side_reported = |side: Side<'_>, m: &ModelConfig| {
        reported(side.classes, m.object_filter.as_deref(), &c.object_filter)
    };
    let old = side_reported(cur_side, current);
    let now = side_reported(new_side, new);
    let then = side_reported(new_side, &copied);
    let known = !cur_side.classes.is_empty() && !new_side.classes.is_empty();
    let unknown = cur_side
        .unknown
        .or(new_side.unknown)
        .map(str::to_string)
        .or_else(|| (!known).then(|| "the class list of a model could not be read".to_string()));
    let classes = ClassDiff {
        current: class_text(&old),
        new: class_text(&now),
        lost: if known { minus(&old, &now) } else { vec![] },
        gained: if known { minus(&now, &old) } else { vec![] },
        gained_with_copy: if known { minus(&then, &old) } else { vec![] },
        unknown,
    };
    SwitchDiff {
        current: current.effective_name(),
        new: new.effective_name(),
        rows,
        classes,
        copy,
    }
}

// ---------------------------------------------------------------------------------------------
// The suggested threshold
// ---------------------------------------------------------------------------------------------

/// A threshold from a held-out benchmark, in words.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Suggestion {
    #[serde(serialize_with = "crate::config::thr::ser")]
    pub threshold: f32,
    /// The device whose confidence profile takes it; None = the model's own threshold, for a
    /// device the benchmark did not run on.
    pub device: Option<String>,
    /// What the number means.
    pub words: String,
    /// What to keep in mind (other sets mixed in, its own training frames, another device).
    pub notes: Vec<String>,
}

fn pct(v: f64) -> String {
    format!("{:.0}%", v * 100.0)
}

/// The threshold [`ModelResult::threshold`] suggests for `model` running on `device`. The result
/// must cover the model's held-out frames. Err says why there is none.
pub fn suggest(
    r: &ModelResult,
    model: &str,
    device: Option<&str>,
) -> std::result::Result<Suggestion, String> {
    let held = format!("heldout-{model}");
    if !r.datasets.iter().any(|d| d == &held) {
        return Err(format!(
            "the latest benchmark of {model} did not use its held-out frames ({}), so no \
             threshold can be suggested from it",
            r.datasets.join(", ")
        ));
    }
    let advice = r
        .threshold
        .as_ref()
        .ok_or_else(|| format!("the benchmark of {model} has no threshold result"))?;
    if advice.relative {
        return Err(format!(
            "the benchmark of {model} had no human labels to measure a threshold against"
        ));
    }
    let on_device = device.and_then(|d| {
        let t = advice.apply_value_on(d)?;
        let key = crate::config::device_threshold_key(d).ok()?;
        let row = advice
            .by_device
            .iter()
            .find(|p| p.device == d)
            .or_else(|| {
                advice
                    .by_device
                    .iter()
                    .find(|p| crate::config::same_device(&p.device, d))
            })?;
        Some((t, Some(key), row.best.as_ref()?.point))
    });
    let (threshold, profile, point) = match on_device {
        Some(x) => x,
        None => {
            let t = advice.apply_value().ok_or_else(|| {
                format!(
                    "the benchmark of {model} found no threshold that meets the goal \
                     ({}); change the goal on the Benchmark page, or keep the threshold",
                    r.threshold_objective.describe()
                )
            })?;
            let best = advice
                .best
                .as_ref()
                .ok_or_else(|| format!("the benchmark of {model} has no best threshold"))?;
            (t, None, best.point)
        }
    };
    let (Some(recall), Some(precision)) = (point.recall, point.precision) else {
        return Err(format!(
            "the benchmark of {model} has no finds or correct figures at its best threshold"
        ));
    };
    let mut notes = Vec::new();
    if let (None, Some(d)) = (&profile, device) {
        notes.push(format!(
            "Measured on {}, not on {d}, where this model will run: saved as the model's own \
             threshold.",
            advice.device
        ));
    }
    if r.datasets.len() > 1 {
        notes.push(format!(
            "The figures also cover {}.",
            r.datasets
                .iter()
                .filter(|d| *d != &held)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if let Some(chip) = r
        .trained_on_share
        .and_then(crate::training::bench::trained_chip)
    {
        notes.push(format!("{chip}: the real result will be lower."));
    }
    let frames = r.images;
    Ok(Suggestion {
        threshold,
        device: profile,
        words: format!(
            "At {threshold:.2} this model found {} of the labelled objects on {frames} frames it \
             did not train on, and {} of what it reported was right. A higher number means \
             fewer false alerts and more missed objects; a lower number, the opposite.",
            pct(recall),
            pct(precision),
        ),
        notes,
    })
}

// ---------------------------------------------------------------------------------------------
// Switching and reverting
// ---------------------------------------------------------------------------------------------

/// How a switch runs. The new model's entry must already be in the config and enabled
/// (`crate::server` adds it through the trained-model code).
#[derive(Debug, Clone, Default)]
pub struct SwitchOptions {
    /// The current default, read before the new model was added or enabled.
    pub from: Option<String>,
    /// The new model's entry before it was added or enabled; None = as it stands.
    pub before: Option<ModelConfig>,
    /// Copy the settings of `from` ([`copy_settings`]).
    pub copy: bool,
    /// The new model's classes.
    pub classes: Vec<String>,
    /// A suggested threshold the owner accepted ([`Suggestion`]).
    pub threshold: Option<Accepted>,
    /// The time of the switch (RFC 3339).
    pub at: String,
}

/// A threshold to write: `device`'s confidence profile, else the model's own threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct Accepted {
    pub threshold: f32,
    pub device: Option<String>,
}

impl From<&Suggestion> for Accepted {
    fn from(s: &Suggestion) -> Self {
        Self {
            threshold: s.threshold,
            device: s.device.clone(),
        }
    }
}

/// What [`apply_switch`] did.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SwitchOutcome {
    /// The new model was the default already (only an accepted threshold was written).
    pub already_default: bool,
    pub from: Option<String>,
    /// Labels of the settings of the new model that changed.
    pub changed: Vec<String>,
    pub copy: CopyReport,
}

impl SwitchOutcome {
    /// One sentence for the page.
    pub fn message(&self, name: &str) -> String {
        if self.already_default {
            return if self.changed.is_empty() {
                format!("{name} is the default model already; nothing changed.")
            } else {
                format!("{name} stays the default model; its threshold was saved.")
            };
        }
        let mut s = format!("{name} is now the default model.");
        if let Some(from) = &self.from {
            s.push_str(&format!(" {from} stays loaded; Revert goes back to it."));
        }
        if !self.copy.copied.is_empty() {
            let list: Vec<String> = self
                .copy
                .copied
                .iter()
                .map(|c| format!("{} {}", c.label.to_lowercase(), c.value))
                .collect();
            s.push_str(&format!(" Copied: {}.", list.join("; ")));
        }
        s
    }
}

fn write_threshold(c: &mut Config, name: &str, a: &Accepted) -> Result<()> {
    match &a.device {
        Some(d) => {
            crate::config_form::apply_device_thresholds(
                c,
                &[(name.to_string(), d.clone(), Some(a.threshold))],
            )?;
        }
        None => {
            if let Some(m) = entry_mut(&mut c.models, name) {
                m.confidence_threshold = Some(crate::config::round_threshold(a.threshold));
            }
        }
    }
    Ok(())
}

/// Make `name` the default model and write the [`PreviousDefault`] record (see the module docs).
/// Switching to the model that is the default already writes no record, only `opts.threshold`:
/// Revert keeps meaning the last real switch.
pub fn apply_switch(c: &mut Config, name: &str, opts: SwitchOptions) -> Result<SwitchOutcome> {
    let before = opts
        .before
        .clone()
        .or_else(|| entry(&c.models, name).cloned())
        .ok_or_else(|| anyhow!("no configured model named {name}"))?;
    if opts.from.as_deref().is_some_and(|f| same_name(f, name)) {
        if let Some(a) = &opts.threshold {
            write_threshold(c, name, a)?;
        }
        let changed = entry(&c.models, name)
            .map(|m| changed_fields(&before, m))
            .unwrap_or_default();
        return Ok(SwitchOutcome {
            already_default: true,
            from: opts.from,
            changed: changed.iter().map(|f| field_label(f)).collect(),
            copy: CopyReport::default(),
        });
    }
    let raw = c.default_model.clone();
    let mut copy = CopyReport::default();
    if opts.copy
        && let Some(from) = opts
            .from
            .as_deref()
            .and_then(|f| entry(&c.models, f))
            .cloned()
    {
        let models = c.models.clone();
        let target = entry_mut(&mut c.models, name)
            .ok_or_else(|| anyhow!("no configured model named {name}"))?;
        copy = copy_settings(&from, target, &opts.classes, &models);
    }
    if let Some(a) = &opts.threshold {
        write_threshold(c, name, a)?;
    }
    let after = entry(&c.models, name)
        .ok_or_else(|| anyhow!("no configured model named {name}"))?
        .clone();
    let changed = changed_fields(&before, &after);
    c.default_model = Some(after.effective_name());
    let labels = changed.iter().map(|f| field_label(f)).collect();
    if let Some(from) = &opts.from {
        c.previous_default = Some(PreviousDefault {
            model: from.clone(),
            default_model: raw,
            switched_to: after.effective_name(),
            settings: (!changed.is_empty()).then_some(before),
            changed,
            at: opts.at,
        });
    }
    Ok(SwitchOutcome {
        already_default: false,
        from: opts.from,
        changed: labels,
        copy,
    })
}

fn changed_fields(a: &ModelConfig, b: &ModelConfig) -> Vec<String> {
    SWITCH_FIELDS
        .iter()
        .filter(|f| field_differs(a, b, f))
        .map(|f| (*f).to_string())
        .collect()
}

/// Why a Revert cannot run, for a record that still applies.
fn revert_blocked(c: &Config, p: &PreviousDefault) -> Option<String> {
    entry(&c.models, &p.model).is_none().then(|| {
        format!(
            "{} is no longer in the config, so there is no model to go back to. Add it again \
             (Config page, Add a model by path), or forget this record.",
            p.model
        )
    })
}

/// Whether `p` still describes the default: a default changed by hand since makes it stale.
fn applies(c: &Config, p: &PreviousDefault) -> bool {
    current_default(c).is_some_and(|d| same_name(&d, &p.switched_to))
}

/// What the page shows about a [`PreviousDefault`] record.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RevertView {
    /// The model a Revert goes back to.
    pub to: String,
    /// The default now.
    pub from: String,
    /// The day of the switch (or empty).
    pub at: String,
    /// Labels of the settings a Revert restores.
    pub restores: String,
    /// Why it cannot run; the page offers to forget the record instead.
    pub blocked: Option<String>,
}

/// The view of `c`'s record; None without one or when it is stale.
pub fn revert_view(c: &Config) -> Option<RevertView> {
    let p = c.previous_default.as_ref().filter(|p| applies(c, p))?;
    Some(RevertView {
        to: p.model.clone(),
        from: p.switched_to.clone(),
        at: p.at.chars().take(10).collect(),
        restores: p
            .changed
            .iter()
            .map(|f| field_label(f))
            .collect::<Vec<_>>()
            .join(", "),
        blocked: revert_blocked(c, p),
    })
}

/// What [`revert`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct RevertOutcome {
    pub to: String,
    pub from: String,
    /// Labels of the settings restored.
    pub restored: Vec<String>,
}

impl RevertOutcome {
    pub fn message(&self) -> String {
        let mut s = format!(
            "Back to {} as the default model. Revert again to return to {}.",
            self.to, self.from
        );
        if !self.restored.is_empty() {
            s.push_str(&format!(" Restored: {}.", self.restored.join(", ")));
        }
        s
    }
}

/// Undo the last switch: `default_model` and the fields the switch changed go back, a setting
/// the switch did not touch stays as it is now. The record becomes its opposite (the values
/// replaced here and the other model), so reverting twice ends where it began. Refused, with
/// `c` unchanged, when the model to go back to is gone or the default changed by hand since.
pub fn revert(c: &mut Config, at: &str) -> Result<RevertOutcome> {
    let p = c
        .previous_default
        .clone()
        .ok_or_else(|| anyhow!("no default switch is recorded, so there is nothing to revert"))?;
    if !applies(c, &p) {
        bail!(
            "the default model is now {}, not {}: it was changed after the switch, so going back \
             to {} would undo that change too. Choose the default on the Config page (Models)",
            current_default(c).unwrap_or_else(|| "none".to_string()),
            p.switched_to,
            p.model
        );
    }
    if let Some(why) = revert_blocked(c, &p) {
        bail!("{why}");
    }
    let mut n = c.clone();
    let raw = n.default_model.clone();
    let mut settings = p.settings.clone();
    if let Some(saved) = &mut settings {
        // Gone since the switch: nothing to restore on it, and nothing to give back later.
        match entry_mut(&mut n.models, &saved.effective_name()) {
            Some(live) => {
                for f in &p.changed {
                    swap_field(live, saved, f);
                }
            }
            None => settings = None,
        }
    }
    let changed = if settings.is_some() {
        p.changed.clone()
    } else {
        Vec::new()
    };
    n.default_model = p.default_model.clone();
    if !current_default(&n).is_some_and(|d| same_name(&d, &p.model)) {
        n.default_model = Some(p.model.clone());
    }
    if !current_default(&n).is_some_and(|d| same_name(&d, &p.model)) {
        bail!(
            "{} is not enabled, so it cannot serve as the default model. Tick Load for it on the \
             Models card (Config page), then revert",
            p.model
        );
    }
    let restored = changed.iter().map(|f| field_label(f)).collect();
    n.previous_default = Some(PreviousDefault {
        model: p.switched_to.clone(),
        default_model: raw,
        switched_to: p.model.clone(),
        changed,
        settings,
        at: at.to_string(),
    });
    *c = n;
    Ok(RevertOutcome {
        to: p.model,
        from: p.switched_to,
        restored,
    })
}

// ---------------------------------------------------------------------------------------------
// The verifier
// ---------------------------------------------------------------------------------------------

/// The verifier `target` uses now: its own, else the global one (an enabled one only).
pub fn current_verifier(c: &Config, target: &ModelConfig) -> Option<String> {
    target
        .verify
        .as_ref()
        .or(c.verify.as_ref())
        .filter(|v| v.enabled && !v.model.trim().is_empty())
        .map(|v| v.model.clone())
}

/// Refuse to replace `target`'s verifier by `new` unless `confirmed` names the one replaced, so
/// a page opened before someone changed it cannot confirm something else.
pub fn check_verifier(
    c: &Config,
    target: &ModelConfig,
    new: &str,
    confirmed: Option<&str>,
) -> Result<()> {
    let Some(existing) = current_verifier(c, target).filter(|v| !same_name(v, new)) else {
        return Ok(());
    };
    if confirmed.is_some_and(|x| same_name(x, &existing)) {
        return Ok(());
    }
    bail!(
        "{existing} already verifies {}, and a model has one verifier. To replace it (and its \
         verifier settings), press Apply again and confirm; to keep it, choose another model",
        target.effective_name()
    )
}

/// `{model name: verifier}` of every model with one (the page's confirm names it).
pub fn verifiers(c: &Config) -> BTreeMap<String, String> {
    c.models
        .iter()
        .filter_map(|m| Some((m.effective_name(), current_verifier(c, m)?)))
        .collect()
}

#[cfg(test)]
mod tests;
