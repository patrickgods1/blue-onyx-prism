//! Field-level view of the config for the web UI: flatten a [`Config`] into canonical
//! `key -> JSON value` maps, describe what changed between two configs (the change log), say
//! which changes apply without a restart, and compute what a restart would still apply.
//!
//! Keys:
//! - `port`, `device`, ... : top-level config fields (`benchmark.<field>` for the benchmark section);
//! - `model::<name>`: the model exists (value `true`);
//! - `model:<field>:<name>`: a field of a model entry (`device`, `confidence_threshold`, ...);
//! - `default`: the model that serves `/v1/vision/detection` (the Models card's radio);
//! - `models:order`: the models' order (normalized names).
//!
//! `<name>` is the model's normalized name ([`normalize_name`]), so entries are identified by
//! name, not by position.
//!
//! Saving is last write wins: a form posts the section it edits and the server applies it as
//! is (the Config page refreshes the fields the user has not touched from `GET /v1/config`).

use crate::config::{Config, ModelConfig};
use crate::registry::normalize_name;
use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};

/// Canonical field values (see the module docs for the keys).
pub type Fields = BTreeMap<String, Value>;

pub const ORDER_KEY: &str = "models:order";
pub const DEFAULT_KEY: &str = "default";

pub fn model_key(field: &str, name: &str) -> String {
    format!("model:{field}:{}", normalize_name(name))
}

pub fn presence_key(name: &str) -> String {
    format!("model::{}", normalize_name(name))
}

/// A parsed key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key<'a> {
    Global(&'a str),
    Bench(&'a str),
    Order,
    Default,
    Presence(&'a str),
    Model { field: &'a str, name: &'a str },
}

pub fn parse_key(k: &str) -> Key<'_> {
    if k == ORDER_KEY {
        return Key::Order;
    }
    if k == DEFAULT_KEY {
        return Key::Default;
    }
    if let Some(rest) = k.strip_prefix("model:")
        && let Some((field, name)) = rest.split_once(':')
    {
        return if field.is_empty() {
            Key::Presence(name)
        } else {
            Key::Model { field, name }
        };
    }
    if let Some(f) = k.strip_prefix("benchmark.") {
        return Key::Bench(f);
    }
    Key::Global(k)
}

/// The config as a JSON object (thresholds rounded, see [`crate::config::thr`]).
fn config_object(c: &Config) -> Map<String, Value> {
    let mut n = c.clone();
    n.normalize();
    match serde_json::to_value(&n) {
        Ok(Value::Object(m)) => m,
        _ => Map::new(),
    }
}

fn model_object(m: &ModelConfig) -> Map<String, Value> {
    match serde_json::to_value(m) {
        Ok(Value::Object(o)) => o,
        _ => Map::new(),
    }
}

/// Normalized name and JSON object of every model, in config order (names filled in).
fn models_of(c: &Config) -> Vec<(String, Map<String, Value>)> {
    let mut n = c.clone();
    n.normalize();
    n.models
        .iter()
        .map(|m| (normalize_name(&m.effective_name()), model_object(m)))
        .collect()
}

/// Every field of `c`: globals, `benchmark.*`, `default_model`, models (presence, fields, order).
pub fn flatten_all(c: &Config) -> Fields {
    let mut out = Fields::new();
    for (k, v) in config_object(c) {
        match (k.as_str(), v) {
            ("models", _) => {}
            ("benchmark", Value::Object(b)) => {
                for (bk, bv) in b {
                    out.insert(format!("benchmark.{bk}"), bv);
                }
            }
            (_, v) => {
                out.insert(k, v);
            }
        }
    }
    add_models(&mut out, c);
    out
}

fn add_models(out: &mut Fields, c: &Config) {
    let models = models_of(c);
    out.insert(
        ORDER_KEY.to_string(),
        Value::Array(
            models
                .iter()
                .map(|(n, _)| Value::String(n.clone()))
                .collect(),
        ),
    );
    for (name, obj) in models {
        out.insert(format!("model::{name}"), Value::Bool(true));
        for (f, v) in obj {
            out.insert(format!("model:{f}:{name}"), v);
        }
    }
}

/// Display names of the models mentioned in `maps` (normalized name -> name as configured).
fn display_names(maps: &[&Fields]) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for m in maps {
        for (k, v) in m.iter() {
            if let Key::Model {
                field: "name",
                name,
            } = parse_key(k)
                && let Some(s) = v.as_str()
            {
                out.entry(name.to_string()).or_insert_with(|| s.to_string());
            }
        }
    }
    out
}

fn field_label(field: &str) -> String {
    match field {
        "confidence_threshold" => "threshold".into(),
        "enabled" => "load".into(),
        "gpu_precision" => "GPU precision".into(),
        "object_filter" => "object filter".into(),
        "night_confidence_threshold" => "night threshold".into(),
        "label_thresholds" => "label thresholds".into(),
        "night_label_thresholds" => "night label thresholds".into(),
        "person_aspect" => "person aspect".into(),
        "min_box_area_frac" => "min box area".into(),
        "suppress_overlaps" => "overlap suppression".into(),
        "static_suppression" => "static-box suppression".into(),
        "preprocess" => "preprocess".into(),
        "night_preprocess" => "night preprocess".into(),
        "night_model" => "night model".into(),
        "verify" => "verifier".into(),
        other => other.replace('_', " "),
    }
}

fn global_label(k: &str) -> String {
    match k {
        "port" => "Port".into(),
        "bind_address" => "Bind address".into(),
        "allow_remote_admin" => "Allow remote administration".into(),
        "request_timeout_secs" => "Request timeout".into(),
        "worker_queue_size" => "Worker queue size".into(),
        "device" => "Device".into(),
        "gpu_index" => "GPU index".into(),
        "force_cpu" => "Force CPU".into(),
        "cache_dir" => "Cache dir".into(),
        "confidence_threshold" => "Confidence threshold".into(),
        "nms_iou" => "NMS IoU".into(),
        "object_filter" => "Object filter".into(),
        "log_level" => "Log level".into(),
        "log_path" => "Log directory".into(),
        "log_timezone" => "Log timestamp time zone".into(),
        "log_format" => "Log file format".into(),
        "save_image_path" => "Save annotated images to".into(),
        "save_ref_image" => "Save the unannotated image".into(),
        "intra_threads" => "CPU threads".into(),
        "save_image_max_age_days" => "Saved images: max age (days)".into(),
        "save_image_max_mb" => "Saved images: max size (MB)".into(),
        "preprocess_threads" => "Preprocess threads".into(),
        "inflight" => "Inferences in flight".into(),
        "ort_sessions" => "ONNX Runtime sessions".into(),
        "hardware_profile" => "Hardware profile".into(),
        "auto_download" => "Auto download".into(),
        "allow_large_downloads" => "Allow large downloads".into(),
        "default_model" => "Default model".into(),
        "max_image_pixels" => "Max image pixels".into(),
        "night_confidence_threshold" => "Night / IR confidence threshold".into(),
        "label_thresholds" => "Label thresholds".into(),
        "night_label_thresholds" => "Night / IR label thresholds".into(),
        "person_aspect" => "Person aspect (width/height)".into(),
        "min_box_area_frac" => "Min box area (share of image)".into(),
        "suppress_overlaps" => "Overlap suppression".into(),
        "static_suppression" => "Static-box suppression".into(),
        "preprocess" => "Image preprocessing".into(),
        "night_preprocess" => "Night / IR image preprocessing".into(),
        "verify" => "Verifier cascade".into(),
        "collect" => "Dataset collector".into(),
        other => other.replace('_', " "),
    }
}

/// Human label of a key, with model display names from `names`.
pub fn label(key: &str, names: &BTreeMap<String, String>) -> String {
    let show = |n: &str| names.get(n).cloned().unwrap_or_else(|| n.to_string());
    match parse_key(key) {
        Key::Global(k) => global_label(k),
        Key::Bench(k) => format!("Benchmark {}", k.replace('_', " ")),
        Key::Order => "Models order".into(),
        Key::Default => "Default model".into(),
        Key::Presence(n) => format!("Model {}", show(n)),
        Key::Model { field, name } => format!("{} \u{b7} {}", show(name), field_label(field)),
    }
}

/// Display text of a value of `key` (None = absent).
pub fn value_text(key: &str, v: Option<&Value>) -> String {
    let (field, presence) = match parse_key(key) {
        // A per-model rule that is absent follows the global value.
        Key::Model { field, .. } if v.is_none() && NIGHT_MODEL_FIELDS.contains(&field) => {
            return "global".into();
        }
        Key::Model { field, .. } => (field, false),
        Key::Presence(_) => ("", true),
        Key::Global(k) => (k, false),
        Key::Default => ("default", false),
        _ => ("", false),
    };
    match v {
        None if presence => "removed".into(),
        None => "(none)".into(),
        Some(Value::Bool(true)) if presence => "present".into(),
        Some(Value::Null) => match field {
            "device" | "confidence_threshold" | "object_filter" => "global".into(),
            "default" => "first enabled".into(),
            _ => "(none)".into(),
        },
        Some(Value::String(s)) if s.is_empty() => "(empty)".into(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(b)) => if *b { "on" } else { "off" }.into(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Array(a)) if a.iter().all(Value::is_string) => {
            if a.is_empty() {
                "(none)".into()
            } else {
                a.iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        }
        Some(v) => v.to_string(),
    }
}

/// One changed field, for the change log.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FieldChange {
    pub key: String,
    pub label: String,
    pub from: String,
    pub to: String,
}

/// Fields that differ between `old` and `new`, and a one-line summary grouped by model:
/// "IPcam-general device → openvino:cpu, threshold → 0.35; Port → 4000".
pub fn describe_changes(old: &Config, new: &Config) -> (Vec<FieldChange>, String) {
    let a = flatten_all(old);
    let b = flatten_all(new);
    let names = display_names(&[&b, &a]);
    let keys: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
    let mut changes = Vec::new();
    // Summary parts: per model (in first-seen order) and globals.
    let mut model_parts: Vec<(String, Vec<(usize, String)>)> = Vec::new();
    // Model fields in the order the UI shows them.
    let rank = |f: &str| {
        [
            "name",
            "path",
            "family",
            "enabled",
            "device",
            "confidence_threshold",
        ]
        .iter()
        .position(|x| *x == f)
        .unwrap_or(99)
    };
    let mut global_parts: Vec<String> = Vec::new();
    let mut model_part = |name: &str, field: &str, text: String| {
        let display = names.get(name).cloned().unwrap_or_else(|| name.to_string());
        let item = (rank(field), text);
        match model_parts.iter_mut().find(|(n, _)| *n == display) {
            Some((_, v)) => v.push(item),
            None => model_parts.push((display, vec![item])),
        }
    };
    for k in keys {
        let (x, y) = (a.get(k), b.get(k));
        if x == y {
            continue;
        }
        let from = value_text(k, x);
        let to = value_text(k, y);
        changes.push(FieldChange {
            key: k.clone(),
            label: label(k, &names),
            from: from.clone(),
            to: to.clone(),
        });
        match parse_key(k) {
            Key::Presence(n) => {
                model_part(n, "", if y.is_some() { "added" } else { "removed" }.into());
            }
            // Fields of an added/removed model are covered by "added"/"removed" (an optional
            // field set or cleared on a model present in both is a change like any other).
            Key::Model { name, .. }
                if !(a.contains_key(&presence_key(name))
                    && b.contains_key(&presence_key(name))) => {}
            Key::Model { field, name } => {
                model_part(name, field, format!("{} \u{2192} {to}", field_label(field)));
            }
            Key::Order => global_parts.push("models reordered".into()),
            _ => global_parts.push(format!("{} \u{2192} {to}", label(k, &names))),
        }
    }
    let mut parts: Vec<String> = model_parts
        .into_iter()
        .map(|(n, mut v)| {
            v.sort_by_key(|(r, _)| *r);
            let v: Vec<String> = v.into_iter().map(|(_, t)| t).collect();
            format!("{n} {}", v.join(", "))
        })
        .collect();
    parts.extend(global_parts);
    (changes, parts.join("; "))
}

// ---------------------------------------------------------------------------------------------
// Pending restart

/// What a restart would change: differences between the config this generation runs with and
/// the current config.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Pending {
    /// Server-wide reasons ("Port: 32168 → 4000", "default model: a → b").
    pub global: Vec<String>,
    /// Reasons per model (normalized name), e.g. "device: auto → openvino:cpu".
    pub models: BTreeMap<String, Vec<String>>,
    /// Changes only a process restart applies (`port`, `bind_address`, `log_path`,
    /// `log_format`).
    pub process: Vec<String>,
}

impl Pending {
    pub fn count(&self) -> usize {
        self.global.len() + self.models.values().map(Vec::len).sum::<usize>() + self.process.len()
    }

    pub fn restart_needed(&self) -> bool {
        self.count() > 0
    }

    pub fn for_model(&self, name: &str) -> &[String] {
        self.models
            .get(&normalize_name(name))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

/// Globals a restart does not need: the log level and `allow_remote_admin` apply immediately,
/// the per-request settings go into every worker's live settings
/// ([`crate::worker::LiveSettings`], through `ModelRegistry::apply_live`), `default_model`
/// switches the registry's default at once, and the benchmark section is read when a
/// benchmark starts. `max_image_pixels` is the process-wide decode ceiling, set again after
/// every config change. Every other global is runtime-level and needs a restart.
pub fn live_global(k: &str) -> bool {
    matches!(
        k,
        "log_level"
            | "log_timezone"
            | "default_model"
            | "allow_remote_admin"
            | "request_timeout_secs"
            | "confidence_threshold"
            | "nms_iou"
            | "object_filter"
            | "save_image_path"
            | "save_ref_image"
            | "save_image_max_age_days"
            | "save_image_max_mb"
            | "max_image_pixels"
            | "night_confidence_threshold"
            | "label_thresholds"
            | "night_label_thresholds"
            | "person_aspect"
            | "min_box_area_frac"
            | "suppress_overlaps"
            | "static_suppression"
            | "preprocess"
            | "night_preprocess"
            | "verify"
            | "collect"
    ) || k.starts_with("benchmark.")
}

/// Model fields applied live (the worker reads them per request).
pub const LIVE_MODEL_FIELDS: &[&str] = &[
    "confidence_threshold",
    "object_filter",
    "night_model",
    "night_confidence_threshold",
    "label_thresholds",
    "night_label_thresholds",
    "person_aspect",
    "min_box_area_frac",
    "suppress_overlaps",
    "static_suppression",
    "preprocess",
    "night_preprocess",
    "verify",
];

/// The per-model night / IR rule fields (serialized only when set; absent = the global value).
pub const NIGHT_MODEL_FIELDS: &[&str] = &[
    "night_confidence_threshold",
    "label_thresholds",
    "night_label_thresholds",
    "person_aspect",
    "min_box_area_frac",
    "suppress_overlaps",
    "static_suppression",
    "preprocess",
    "night_preprocess",
    "verify",
];

/// Labels of the changed fields that applied without a restart (live globals, the default
/// model, per-model thresholds and object filters; the benchmark section is not listed: it is
/// only read when a benchmark starts).
pub fn applied_live(changes: &[FieldChange]) -> Vec<String> {
    changes
        .iter()
        .filter(|c| match parse_key(&c.key) {
            Key::Global(g) => live_global(g),
            Key::Default => true,
            Key::Model { field, .. } => LIVE_MODEL_FIELDS.contains(&field),
            _ => false,
        })
        .map(|c| c.label.clone())
        .collect()
}

/// `current` with the runtime-level globals (everything [`live_global`] does not list, e.g.
/// `port`, `device`, `force_cpu`, `cache_dir`) of `running`: what this generation can apply
/// without a restart. Models, live globals and the benchmark section come from `current`.
pub fn runtime_overlay(running: &Config, current: &Config) -> Config {
    let mut obj = config_object(current);
    for (k, v) in config_object(running) {
        if !matches!(k.as_str(), "models" | "benchmark" | "default_model") && !live_global(&k) {
            obj.insert(k, v);
        }
    }
    match serde_json::from_value::<Config>(Value::Object(obj)) {
        Ok(mut c) => {
            c.normalize();
            c
        }
        Err(_) => current.clone(),
    }
}

/// The enabled entry of model `key` (normalized name).
fn enabled_model<'a>(c: &'a Config, key: &str) -> Option<&'a ModelConfig> {
    c.models
        .iter()
        .find(|m| m.enabled && normalize_name(&m.effective_name()) == key)
}

/// The models whose worker has to be replaced, started or stopped to go from `running` to
/// `next` (same runtime-level globals, see [`runtime_overlay`]): enabled in one and not the
/// other, or enabled in both with a field other than the live ones ([`LIVE_MODEL_FIELDS`])
/// changed. Names as `next` spells them (else `running`), in `next` order.
pub fn model_reloads(running: &Config, next: &Config) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    for m in next.models.iter().chain(running.models.iter()) {
        let name = m.effective_name();
        let key = normalize_name(&name);
        if !seen.insert(key.clone()) {
            continue;
        }
        let changed = match (enabled_model(running, &key), enabled_model(next, &key)) {
            (None, None) => false,
            (Some(r), Some(n)) => {
                let strip = |m: &ModelConfig| {
                    let mut o = model_object(m);
                    for f in LIVE_MODEL_FIELDS {
                        o.remove(*f);
                    }
                    o
                };
                strip(r) != strip(n) || running.device_for(r) != next.device_for(n)
            }
            _ => true,
        };
        if changed {
            out.push(name);
        }
    }
    out
}

/// Put `source`'s entry of model `name` into `target` (replace it, add it at the end, or
/// remove it when `source` has none): a model whose reload did not happen keeps showing what
/// runs.
pub fn keep_model_from(target: &mut Config, source: &Config, name: &str) {
    let key = normalize_name(name);
    let is = |m: &ModelConfig| normalize_name(&m.effective_name()) == key;
    let from = source.models.iter().find(|m| is(m)).cloned();
    match (target.models.iter().position(is), from) {
        (Some(i), Some(m)) => {
            if let Some(slot) = target.models.get_mut(i) {
                *slot = m;
            }
        }
        (Some(i), None) => {
            target.models.remove(i);
        }
        (None, Some(m)) => target.models.push(m),
        (None, None) => {}
    }
}

/// What a restart would apply, comparing the running config with the current one.
pub fn pending(running: &Config, current: &Config) -> Pending {
    let mut out = Pending::default();
    let (a, b) = (flatten_all(running), flatten_all(current));
    let names = display_names(&[&b, &a]);
    let keys: BTreeSet<&String> = a.keys().chain(b.keys()).collect();
    for k in keys {
        let Key::Global(g) = parse_key(k) else {
            continue;
        };
        let (x, y) = (a.get(k), b.get(k));
        if x == y || live_global(g) {
            continue;
        }
        let text = format!(
            "{}: {} \u{2192} {}",
            label(k, &names),
            value_text(k, x),
            value_text(k, y)
        );
        if matches!(g, "port" | "bind_address" | "log_path" | "log_format") {
            out.process
                .push(format!("{text} (needs a process restart)"));
        } else {
            out.global.push(text);
        }
    }
    // The default model applies live (a default that is not loaded yet shows as that model's
    // "enabled; not loaded yet").
    let find = |c: &'_ Config, key: &str| -> Option<ModelConfig> {
        c.models
            .iter()
            .find(|m| normalize_name(&m.effective_name()) == key)
            .cloned()
    };
    let mut all: Vec<String> = Vec::new();
    for m in running.models.iter().chain(current.models.iter()) {
        let k = normalize_name(&m.effective_name());
        if !all.contains(&k) {
            all.push(k);
        }
    }
    for key in all {
        let (r, c) = (find(running, &key), find(current, &key));
        let mut reasons = Vec::new();
        match (r.filter(|m| m.enabled), c) {
            (None, Some(c)) if c.enabled => reasons.push("enabled; not loaded yet".to_string()),
            (None, _) => {}
            (Some(_), None) => reasons.push("removed from the config; still loaded".to_string()),
            (Some(_), Some(c)) if !c.enabled => reasons.push("disabled; still loaded".to_string()),
            (Some(r), Some(c)) => {
                let (dr, dc) = (running.device_for(&r), current.device_for(&c));
                if dr != dc {
                    reasons.push(format!("device: {dr} \u{2192} {dc}"));
                }
                // Threshold and object filter apply live. Both sides' keys: a field serialized
                // only when set (`inflight`) is absent from one side when removed (CO-09).
                let (or, oc) = (model_object(&r), model_object(&c));
                let fields: BTreeSet<&String> = or.keys().chain(oc.keys()).collect();
                for f in fields {
                    if matches!(f.as_str(), "name" | "device" | "enabled")
                        || LIVE_MODEL_FIELDS.contains(&f.as_str())
                    {
                        continue;
                    }
                    if or.get(f) != oc.get(f) {
                        reasons.push(format!("{} changed", field_label(f)));
                    }
                }
            }
        }
        if !reasons.is_empty() {
            out.models.insert(key, reasons);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        let mut c = Config {
            models: ["IPcam-general", "yolo26s"]
                .iter()
                .map(|n| ModelConfig {
                    name: Some(n.to_string()),
                    path: format!("models/{n}.onnx").into(),
                    ..Default::default()
                })
                .collect(),
            default_model: Some("IPcam-general".into()),
            ..Default::default()
        };
        c.normalize();
        c
    }

    #[test]
    fn keys_round_trip() {
        assert_eq!(parse_key("port"), Key::Global("port"));
        assert_eq!(parse_key("benchmark.warmup"), Key::Bench("warmup"));
        assert_eq!(parse_key("model::a.b:c"), Key::Presence("a.b:c"));
        assert_eq!(
            parse_key("model:device:a.b:c"),
            Key::Model {
                field: "device",
                name: "a.b:c"
            }
        );
        assert_eq!(
            model_key("device", "IPcam-general.onnx"),
            "model:device:ipcam-general"
        );
    }

    #[test]
    fn thresholds_flatten_as_two_decimals() {
        let mut c = cfg();
        c.models[0].confidence_threshold = Some(0.35);
        c.confidence_threshold = 0.4;
        let f = flatten_all(&c);
        assert_eq!(
            f["model:confidence_threshold:ipcam-general"].to_string(),
            "0.35"
        );
        assert_eq!(f["confidence_threshold"].to_string(), "0.4");
        assert_eq!(f["nms_iou"].to_string(), "0.5");
    }

    #[test]
    fn change_summaries() {
        let a = cfg();
        let mut b = a.clone();
        b.models[0].device = Some("openvino:cpu".into());
        b.models[0].confidence_threshold = Some(0.35);
        b.port = 4000;
        let (changes, summary) = describe_changes(&a, &b);
        assert_eq!(changes.len(), 3);
        assert_eq!(
            summary,
            "IPcam-general device \u{2192} openvino:cpu, threshold \u{2192} 0.35; Port \u{2192} 4000"
        );
        b.models.push(ModelConfig {
            name: Some("x".into()),
            path: "x.onnx".into(),
            ..Default::default()
        });
        let (_, summary) = describe_changes(&a, &b);
        assert!(summary.contains("x added"), "{summary}");
    }

    #[test]
    fn applied_live_lists_live_fields_only() {
        let a = cfg();
        let mut b = a.clone();
        b.models[0].confidence_threshold = Some(0.35);
        b.models[1].object_filter = Some(vec!["person".into()]);
        b.models[1].lazy = true;
        b.nms_iou = 0.6;
        b.request_timeout_secs = 5;
        b.port = 4000;
        b.default_model = Some("yolo26s".into());
        b.benchmark.warmup = 9;
        let (changes, _) = describe_changes(&a, &b);
        let live = applied_live(&changes);
        for l in [
            "IPcam-general \u{b7} threshold",
            "yolo26s \u{b7} object filter",
            "NMS IoU",
            "Request timeout",
            "Default model",
        ] {
            assert!(live.contains(&l.to_string()), "{l}: {live:?}");
        }
        assert_eq!(live.len(), 5, "{live:?}");
    }

    #[test]
    fn overlay_keeps_runtime_level_globals() {
        let running = cfg();
        let mut current = running.clone();
        current.port = 4000;
        current.device = "openvino:cpu".into();
        current.force_cpu = true;
        current.worker_queue_size = 3;
        current.confidence_threshold = 0.3;
        current.request_timeout_secs = 7;
        current.default_model = Some("yolo26s".into());
        current.models[1].lazy = true;
        current.benchmark.warmup = 9;
        let o = runtime_overlay(&running, &current);
        assert_eq!(
            (o.port, o.device.as_str(), o.force_cpu, o.worker_queue_size),
            (running.port, running.device.as_str(), false, 0)
        );
        assert_eq!(o.confidence_threshold, 0.3);
        assert_eq!(o.request_timeout_secs, 7);
        assert_eq!(o.default_model.as_deref(), Some("yolo26s"));
        assert!(o.models[1].lazy);
        assert_eq!(o.benchmark.warmup, 9);
        // What stays pending is exactly the runtime-level part (the global device also shows
        // on each model it moves).
        let p = pending(&o, &current);
        assert_eq!(p.global.len(), 3, "{p:?}");
        assert_eq!(p.process.len(), 1, "{p:?}");
        assert_eq!(p.for_model("yolo26s"), ["device: auto \u{2192} CPU"]);
    }

    #[test]
    fn reloads_for_structural_model_changes_only() {
        let running = cfg();
        let mut next = running.clone();
        assert!(model_reloads(&running, &next).is_empty());
        // Live fields: no reload.
        next.models[0].confidence_threshold = Some(0.35);
        next.models[0].object_filter = Some(vec!["car".into()]);
        assert!(model_reloads(&running, &next).is_empty());
        // Device, lazy, a new model, a disabled one.
        next.models[0].device = Some("openvino:cpu".into());
        next.models[1].enabled = false;
        next.models.push(ModelConfig {
            name: Some("extra".into()),
            path: "models/extra.onnx".into(),
            ..Default::default()
        });
        next.models.push(ModelConfig {
            name: Some("off".into()),
            path: "models/off.onnx".into(),
            enabled: false,
            ..Default::default()
        });
        assert_eq!(
            model_reloads(&running, &next),
            ["IPcam-general", "yolo26s", "extra"]
        );
        // Removed from the config.
        let mut gone = running.clone();
        gone.models.remove(1);
        assert_eq!(model_reloads(&running, &gone), ["yolo26s"]);
        // A model whose reload did not happen keeps its running entry.
        keep_model_from(&mut next, &running, "extra");
        keep_model_from(&mut next, &running, "IPcam-general");
        assert_eq!(model_reloads(&running, &next), ["yolo26s"]);
        let mut back = gone.clone();
        keep_model_from(&mut back, &running, "yolo26s");
        assert!(model_reloads(&running, &back).is_empty());
    }

    type Edit = Box<dyn Fn(&mut Config)>;

    /// Phase 3 classification: live globals and per-model live fields are never pending;
    /// every runtime-level global is, one reason each.
    #[test]
    fn pending_lists_only_runtime_level_fields() {
        let running = cfg();
        let live: Vec<Edit> = vec![
            Box::new(|c| c.confidence_threshold = 0.3),
            Box::new(|c| c.nms_iou = 0.6),
            Box::new(|c| c.object_filter = vec!["person".into()]),
            Box::new(|c| c.save_image_path = Some("shots".into())),
            Box::new(|c| c.save_ref_image = true),
            Box::new(|c| c.save_image_max_age_days = 7),
            Box::new(|c| c.save_image_max_mb = 100),
            Box::new(|c| c.request_timeout_secs = 3),
            Box::new(|c| c.log_level = crate::config::LogLevel::Trace),
            Box::new(|c| c.log_timezone = "America/Los_Angeles".parse().unwrap()),
            Box::new(|c| c.allow_remote_admin = true),
            Box::new(|c| c.max_image_pixels = 2_000_000),
            Box::new(|c| c.default_model = Some("yolo26s".into())),
            Box::new(|c| c.benchmark.repeat_per_image = 7),
            Box::new(|c| c.models[0].confidence_threshold = Some(0.2)),
            Box::new(|c| c.models[1].object_filter = Some(vec!["car".into()])),
        ];
        for (i, edit) in live.iter().enumerate() {
            let mut c = running.clone();
            edit(&mut c);
            c.normalize();
            assert_ne!(c, running, "edit {i} changes something");
            let p = pending(&running, &c);
            assert!(!p.restart_needed(), "edit {i} is live: {p:?}");
        }
        let runtime: Vec<(&str, Edit)> = vec![
            ("port", Box::new(|c| c.port = 4000)),
            (
                "bind_address",
                Box::new(|c| c.bind_address = "127.0.0.1".parse().unwrap()),
            ),
            ("log_path", Box::new(|c| c.log_path = Some("logs".into()))),
            (
                "log_format",
                Box::new(|c| c.log_format = crate::config::LogFormat::Json),
            ),
            (
                "openvino_dir",
                Box::new(|c| c.openvino_dir = Some("ov".into())),
            ),
            (
                "onnxruntime_dir",
                Box::new(|c| c.onnxruntime_dir = Some("ort".into())),
            ),
            ("cache_dir", Box::new(|c| c.cache_dir = "c2".into())),
            ("intra_threads", Box::new(|c| c.intra_threads = 2)),
            (
                "download_dir",
                Box::new(|c| c.download_dir = Some("dl".into())),
            ),
            ("models_dir", Box::new(|c| c.models_dir = "m2".into())),
            ("worker_queue_size", Box::new(|c| c.worker_queue_size = 4)),
            ("preprocess_threads", Box::new(|c| c.preprocess_threads = 2)),
            ("inflight", Box::new(|c| c.inflight = 2)),
            ("ort_sessions", Box::new(|c| c.ort_sessions = 2)),
            (
                "auto_download",
                Box::new(|c| c.auto_download = !c.auto_download),
            ),
            (
                "allow_large_downloads",
                Box::new(|c| c.allow_large_downloads = !c.allow_large_downloads),
            ),
            ("gpu_index", Box::new(|c| c.gpu_index = 1)),
        ];
        for (k, edit) in &runtime {
            assert!(!live_global(k), "{k}");
            let mut c = running.clone();
            edit(&mut c);
            let p = pending(&running, &c);
            assert_eq!(p.global.len() + p.process.len(), 1, "{k}: {p:?}");
            assert!(p.models.is_empty(), "{k}: {p:?}");
        }
        // The global device and force_cpu also show on every model they move.
        for edit in [
            Box::new(|c: &mut Config| c.device = "openvino:cpu".into()) as Edit,
            Box::new(|c: &mut Config| c.force_cpu = true),
        ] {
            let mut c = running.clone();
            edit(&mut c);
            let p = pending(&running, &c);
            assert_eq!(p.global.len(), 1, "{p:?}");
            assert_eq!(p.models.len(), 2, "{p:?}");
        }
        // Per-model structural changes stay pending until the model reloads.
        let mut c = running.clone();
        c.models[1].lazy = true;
        c.models[0].enabled = false;
        let p = pending(&running, &c);
        assert_eq!(p.for_model("yolo26s"), ["lazy changed"]);
        assert_eq!(p.for_model("IPcam-general"), ["disabled; still loaded"]);
    }

    #[test]
    fn pending_restart_reasons() {
        let running = cfg();
        assert!(!pending(&running, &running).restart_needed());
        let mut c = running.clone();
        c.models[0].device = Some("openvino:cpu".into());
        c.models[0].confidence_threshold = Some(0.35);
        c.log_level = crate::config::LogLevel::Debug;
        c.benchmark.warmup = 9;
        let p = pending(&running, &c);
        // The threshold applies live.
        assert_eq!(
            p.for_model("IPcam-general"),
            ["device: auto \u{2192} openvino:cpu"]
        );
        assert!(p.global.is_empty(), "{:?}", p.global);
        assert_eq!(p.count(), 1);
        // Global changes and enable/disable.
        c.port = 4000;
        c.models[1].enabled = false;
        c.log_path = Some("logs".into());
        c.bind_address = "127.0.0.1".parse().unwrap();
        // Applies at once: never pending.
        c.allow_remote_admin = true;
        let p = pending(&running, &c);
        assert!(p.global.is_empty(), "{:?}", p.global);
        assert_eq!(p.for_model("yolo26s"), ["disabled; still loaded"]);
        assert_eq!(p.process.len(), 3, "{:?}", p.process);
        assert!(
            p.process
                .contains(&"Port: 32168 \u{2192} 4000 (needs a process restart)".to_string()),
            "{:?}",
            p.process
        );
        // A threshold that only differs below 0.01 is not pending.
        let mut d = running.clone();
        d.confidence_threshold = 0.5000001;
        assert!(!pending(&running, &d).restart_needed());
    }

    #[test]
    fn pending_reports_a_removed_or_added_optional_model_field() {
        let mut running = cfg();
        running.models[0].inflight = Some(2);
        let current = cfg();
        // `inflight` is not serialized when None: removing it must still show.
        let p = pending(&running, &current);
        assert_eq!(p.for_model("IPcam-general"), ["inflight changed"], "{p:?}");
        let p = pending(&current, &running);
        assert_eq!(p.for_model("IPcam-general"), ["inflight changed"], "{p:?}");
        assert!(!pending(&running, &running).restart_needed());
    }

    #[test]
    fn log_timezone_is_live() {
        let running = cfg();
        let mut c = running.clone();
        c.log_timezone = "+08:00".parse().unwrap();
        assert!(!pending(&running, &c).restart_needed());
        assert!(live_global("log_timezone"));
    }

    #[test]
    fn log_format_needs_a_process_restart() {
        let running = cfg();
        let mut c = running.clone();
        c.log_format = crate::config::LogFormat::Json;
        let p = pending(&running, &c);
        assert_eq!(
            p.process,
            ["Log file format: text \u{2192} json (needs a process restart)"]
        );
        assert!(!live_global("log_format"));
    }

    #[test]
    fn night_rules_apply_live_without_a_reload() {
        let old = cfg();
        let mut new = old.clone();
        new.night_confidence_threshold = Some(0.6);
        new.static_suppression.enabled = true;
        new.models[0].person_aspect = Some([0.2, 0.9]);
        new.models[1].label_thresholds = Some(std::collections::BTreeMap::from([(
            "person".to_string(),
            0.7,
        )]));
        assert!(model_reloads(&old, &new).is_empty());
        let p = pending(&old, &new);
        assert!(p.global.is_empty() && p.process.is_empty(), "{p:?}");
        assert!(p.models.is_empty(), "{p:?}");
        let (changes, _) = describe_changes(&old, &new);
        let live = applied_live(&changes);
        assert_eq!(live.len(), 4, "{changes:?}");
        assert!(
            live.contains(&"Night / IR confidence threshold".to_string()),
            "{live:?}"
        );
        assert!(
            live.contains(&"IPcam-general \u{b7} person aspect".to_string()),
            "{live:?}"
        );
        let c = changes
            .iter()
            .find(|c| c.label == "IPcam-general \u{b7} person aspect")
            .unwrap();
        assert_eq!((c.from.as_str(), c.to.as_str()), ("global", "[0.2,0.9]"));
    }

    #[test]
    fn night_model_applies_live_without_a_reload() {
        let old = cfg();
        let mut new = old.clone();
        new.models[0].night_model = Some("yolo26s".into());
        assert!(model_reloads(&old, &new).is_empty());
        let p = pending(&old, &new);
        assert!(p.models.is_empty() && p.global.is_empty(), "{p:?}");
        let (changes, _) = describe_changes(&old, &new);
        assert_eq!(
            applied_live(&changes),
            ["IPcam-general \u{b7} night model"],
            "{changes:?}"
        );
    }

    #[test]
    fn verify_applies_live_without_a_reload() {
        let old = cfg();
        let mut new = old.clone();
        new.models[0].verify = Some(crate::verify::VerifyConfig::with_model("yolo26s"));
        new.verify = Some(crate::verify::VerifyConfig::with_model("yolo26s"));
        assert!(model_reloads(&old, &new).is_empty());
        let p = pending(&old, &new);
        assert!(p.models.is_empty() && p.global.is_empty(), "{p:?}");
        let (changes, _) = describe_changes(&old, &new);
        let live = applied_live(&changes);
        assert!(
            live.contains(&"Verifier cascade".to_string())
                && live.contains(&"IPcam-general \u{b7} verifier".to_string()),
            "{live:?}"
        );
        let c = changes
            .iter()
            .find(|c| c.label == "IPcam-general \u{b7} verifier")
            .unwrap();
        assert_eq!(c.from, "global");
    }

    #[test]
    fn collect_applies_live_without_a_restart() {
        let old = cfg();
        let mut new = old.clone();
        new.collect.enabled = true;
        new.collect.buckets.empty.enabled = false;
        assert!(model_reloads(&old, &new).is_empty());
        let p = pending(&old, &new);
        assert!(!p.restart_needed(), "{p:?}");
        let (changes, _) = describe_changes(&old, &new);
        assert!(
            applied_live(&changes).contains(&"Dataset collector".to_string()),
            "{changes:?}"
        );
    }

    #[test]
    fn preprocess_applies_live_and_absent_means_global() {
        use crate::model::enhance::Preprocess;
        let old = cfg();
        let mut new = old.clone();
        new.preprocess = Some(Preprocess::Gamma(0.6));
        new.models[0].night_preprocess = Some(Preprocess::CLAHE);
        assert!(model_reloads(&old, &new).is_empty());
        let p = pending(&old, &new);
        assert!(p.models.is_empty() && p.global.is_empty(), "{p:?}");
        let (changes, _) = describe_changes(&old, &new);
        let live = applied_live(&changes);
        assert!(
            live.contains(&"Image preprocessing".to_string())
                && live.contains(&"IPcam-general \u{b7} night preprocess".to_string()),
            "{live:?}"
        );
        let c = changes
            .iter()
            .find(|c| c.label == "IPcam-general \u{b7} night preprocess")
            .unwrap();
        assert_eq!((c.from.as_str(), c.to.as_str()), ("global", "clahe"));
    }
}
