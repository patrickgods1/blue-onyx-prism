//! Applying the web UI's forms to a [`Config`]: the `/config` page form, the Models card
//! selection, and the per-model device and threshold picks the benchmark page and CLI apply.
//! All appliers are atomic (the config is untouched on error) and validate through
//! [`crate::config::validate_config`] / [`crate::config::validate_models`].

use crate::config::{
    Config, check_threshold, round_threshold, thr, validate_config, validate_dir_setting,
};
use anyhow::Result;
use serde::Serialize;
use std::path::PathBuf;
use tracing::warn;

/// Fields edited by the main `/config` web form, besides `models_json` (the `models` array as
/// JSON). `default_model` and the per-model `enabled` flags are owned by the Models card
/// (`POST /config/models`, see [`apply_models_selection`]); `apply_config_form` still accepts a
/// `default_model` field for API clients.
/// Checkboxes (`force_cpu`, `save_ref_image`, `allow_remote_admin`, `auto_download`,
/// `allow_large_downloads`) are true when present with any value but `false`/`off`/`0`, and
/// false when absent (browsers omit unchecked boxes); each is read only when its form's marker
/// field (`inference_settings`, `image_settings`, `access_settings`, `download_settings`) is
/// posted too, so a form that does not show the box never clears it.
pub const FORM_FIELDS: &[&str] = &[
    "port",
    "bind_address",
    "allow_remote_admin",
    "request_timeout_secs",
    "worker_queue_size",
    "device",
    "gpu_index",
    "force_cpu",
    "cache_dir",
    "download_dir",
    "models_dir",
    "onnxruntime_dir",
    "openvino_dir",
    "confidence_threshold",
    "nms_iou",
    "object_filter",
    "log_level",
    "log_timezone",
    "log_path",
    "save_image_path",
    "save_ref_image",
    "save_image_max_age_days",
    "save_image_max_mb",
    "intra_threads",
    "auto_download",
    "allow_large_downloads",
    "night_confidence_threshold",
    "night_person_threshold",
    "person_aspect_min",
    "person_aspect_max",
    "drop_person_on_plant",
    "static_suppression",
];

/// The overlap rule behind the Config page's "drop person on potted plant" checkbox.
pub const PLANT_RULE: (&str, &str) = ("person", "potted plant");

/// The config has the [`PLANT_RULE`] overlap rule (any IoU, any night gating).
pub fn has_plant_rule(c: &Config) -> bool {
    c.suppress_overlaps.iter().any(is_plant_rule)
}

fn is_plant_rule(r: &crate::model::filters::OverlapRule) -> bool {
    r.drop.trim().eq_ignore_ascii_case(PLANT_RULE.0)
        && r.if_overlaps.trim().eq_ignore_ascii_case(PLANT_RULE.1)
}

/// The global `night_label_thresholds` entry of `person` (any case).
pub fn night_person_threshold(c: &Config) -> Option<f32> {
    c.night_label_thresholds
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(crate::model::filters::PERSON))
        .map(|(_, v)| *v)
}

/// Apply a submitted `/config` form to `config`. All fields are validated first; on any error
/// `config` is left untouched. Text fields that are absent keep their current value; empty
/// optional paths / `default_model` become `None`. The directory settings
/// (`save_image_path`, `log_path`, `cache_dir`, `download_dir`, `models_dir`,
/// `onnxruntime_dir`, `openvino_dir`) go through
/// [`crate::config::validate_dir_setting`] (stored absolute; relative ones under the data
/// root), and a change of one is logged at warn level: it moves where the service writes.
pub fn apply_config_form(
    config: &mut Config,
    form: &std::collections::HashMap<String, String>,
) -> Result<()> {
    let mut c = config.clone();
    parse_config_form(&mut c, form)?;
    validate_config(&c)?;
    let show = |p: &Option<PathBuf>| {
        p.as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none)".into())
    };
    for (key, from, to) in [
        (
            "save_image_path",
            show(&config.save_image_path),
            show(&c.save_image_path),
        ),
        ("log_path", show(&config.log_path), show(&c.log_path)),
        ("cache_dir", config.cache_dir.clone(), c.cache_dir.clone()),
        (
            "download_dir",
            show(&config.download_dir),
            show(&c.download_dir),
        ),
        (
            "models_dir",
            config.models_dir.display().to_string(),
            c.models_dir.display().to_string(),
        ),
        (
            "onnxruntime_dir",
            show(&config.onnxruntime_dir),
            show(&c.onnxruntime_dir),
        ),
        (
            "openvino_dir",
            show(&config.openvino_dir),
            show(&c.openvino_dir),
        ),
    ] {
        if from != to {
            warn!(key, %from, %to, "directory setting changed by a config form");
        }
    }
    *config = c;
    Ok(())
}

/// The field-by-field part of [`apply_config_form`]: parse and range-check every submitted
/// field into `c` (absent fields keep their value). Cross-field checks are
/// [`validate_config`]'s. On error `c` may be partly modified (callers pass a copy).
pub fn parse_config_form(
    c: &mut Config,
    form: &std::collections::HashMap<String, String>,
) -> Result<()> {
    use anyhow::bail;
    let get = |k: &str| form.get(k).map(|v| v.trim());
    fn num<T: std::str::FromStr>(field: &str, v: &str) -> Result<T> {
        v.parse::<T>()
            .map_err(|_| anyhow::anyhow!("{field}: '{v}' is not a valid number"))
    }
    let checkbox = |k: &str| {
        get(k).is_some_and(|v| !matches!(v.to_ascii_lowercase().as_str(), "false" | "off" | "0"))
    };
    // Directory settings: syntax-checked, relative ones resolved under the data root, system
    // directories refused (SV-03). `download_dir` (the data root itself) comes first, so the
    // other relative settings of the same post resolve under the new root.
    parse_runtime_dirs(c, form)?;
    let data_root = c.data_root();
    let dir = |key: &str, v: &str| validate_dir_setting(key, v, &data_root);

    if let Some(v) = get("port") {
        c.port = num("port", v)?;
        if c.port == 0 {
            bail!("port: must be 1-65535");
        }
    }
    if let Some(v) = get("bind_address") {
        c.bind_address = v.parse().map_err(|_| {
            anyhow::anyhow!("bind_address: '{v}' is not an IP address (e.g. 0.0.0.0 or 127.0.0.1)")
        })?;
    }
    // A checkbox, changed only by forms that show it (hidden `access_settings` field), so other
    // form posts never turn remote administration off or on by omission.
    if form.contains_key("access_settings") {
        c.allow_remote_admin = checkbox("allow_remote_admin");
    }
    if let Some(v) = get("request_timeout_secs") {
        c.request_timeout_secs = num("request_timeout_secs", v)?;
        if c.request_timeout_secs == 0 {
            bail!("request_timeout_secs: must be at least 1");
        }
    }
    if let Some(v) = get("worker_queue_size") {
        c.worker_queue_size = num("worker_queue_size", v)?;
    }
    if let Some(v) = get("device") {
        crate::backend::spec::parse(v).map_err(|e| anyhow::anyhow!("device: {e}"))?;
        c.device = v.to_string();
    }
    if let Some(v) = get("gpu_index") {
        c.gpu_index = num("gpu_index", v)?;
    }
    // Checkboxes changed only by forms that show them (hidden `inference_settings` /
    // `image_settings` markers), so a post of other fields never clears them by omission.
    if form.contains_key("inference_settings") {
        c.force_cpu = checkbox("force_cpu");
    }
    if let Some(v) = get("cache_dir") {
        c.cache_dir = dir("cache_dir", v)?
            .map(|p| p.display().to_string())
            .unwrap_or_default();
    }
    for (field, slot) in [
        ("confidence_threshold", &mut c.confidence_threshold),
        ("nms_iou", &mut c.nms_iou),
    ] {
        if let Some(v) = get(field) {
            let x: f32 = num(field, v)?;
            check_threshold(field, x)?;
            *slot = x;
        }
    }
    if let Some(v) = get("object_filter") {
        c.object_filter = v
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
    }
    if let Some(v) = get("log_level") {
        c.log_level = v.parse().map_err(|e| anyhow::anyhow!("log_level: {e}"))?;
    }
    if let Some(v) = get("log_path") {
        c.log_path = dir("log_path", v)?;
    }
    if let Some(v) = get("log_timezone") {
        c.log_timezone = v.parse()?;
    }
    // Not on the Config page (not in `FORM_FIELDS`); accepted from API clients.
    if let Some(v) = get("log_format") {
        c.log_format = v.parse()?;
    }
    if let Some(v) = get("save_image_path") {
        c.save_image_path = dir("save_image_path", v)?;
    }
    if let Some(v) = get("save_image_max_age_days") {
        c.save_image_max_age_days = num("save_image_max_age_days", v)?;
    }
    if let Some(v) = get("save_image_max_mb") {
        c.save_image_max_mb = num("save_image_max_mb", v)?;
    }
    if form.contains_key("image_settings") {
        c.save_ref_image = checkbox("save_ref_image");
    }
    // Download settings are checkboxes too, but only forms that show them (marked with the
    // hidden `download_settings` field) change them, so older form posts keep their values.
    if form.contains_key("download_settings") {
        c.auto_download = checkbox("auto_download");
        c.allow_large_downloads = checkbox("allow_large_downloads");
    }
    if let Some(v) = get("intra_threads") {
        c.intra_threads = num("intra_threads", v)?;
    }
    parse_night_fields(c, form)?;
    if let Some(v) = form.get("models_json") {
        c.models = serde_json::from_str(v).map_err(|e| anyhow::anyhow!("models: {e}"))?;
    }
    if let Some(v) = get("default_model") {
        c.default_model = (!v.is_empty()).then(|| v.to_string());
    }
    c.normalize();
    Ok(())
}

/// The "Folders" fieldset: `download_dir` (the data root), `models_dir`, `onnxruntime_dir` and
/// `openvino_dir`, each through [`validate_dir_setting`]. `download_dir`, `onnxruntime_dir`
/// and `openvino_dir` resolve against the exe dir (like [`crate::resolve_path`]), `models_dir`
/// under the (possibly new) data root. Empty = the default (`None`; `models_dir`: `models`).
/// A value equal to the current one, as the page shows it, is kept as stored, so saving the
/// form unchanged never turns a relative setting into an absolute one (a spurious restart).
fn parse_runtime_dirs(
    c: &mut Config,
    form: &std::collections::HashMap<String, String>,
) -> Result<()> {
    let get = |k: &str| form.get(k).map(|v| v.trim());
    let exe_dir = crate::exe_dir();
    let unchanged = |v: &str, cur: &Option<PathBuf>| match cur {
        Some(p) => p.display().to_string() == v,
        None => v.is_empty(),
    };
    for (key, slot) in [
        ("download_dir", &mut c.download_dir),
        ("onnxruntime_dir", &mut c.onnxruntime_dir),
        ("openvino_dir", &mut c.openvino_dir),
    ] {
        if let Some(v) = get(key)
            && !unchanged(v, slot)
        {
            *slot = validate_dir_setting(key, v, &exe_dir)?;
        }
    }
    if let Some(v) = get("models_dir")
        && v != c.models_dir.display().to_string()
    {
        c.models_dir = if v.is_empty() {
            Config::default().models_dir
        } else {
            let root = c.data_root();
            validate_dir_setting("models_dir", v, &root)?.unwrap_or_else(|| root.join("models"))
        };
    }
    Ok(())
}

/// A model file path (`path` or `classes`) of a models list that the saved config does not
/// have yet: new or changed. See [`new_model_paths`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewModelPath {
    /// Effective model name.
    pub model: String,
    /// `path` or `classes`.
    pub field: &'static str,
    /// The path as configured (relative ones are under the data root).
    pub path: PathBuf,
}

/// The `path` / `classes` values of `after`'s models that no model of `before` uses (in either
/// field): what a models JSON edit added or changed. Unchanged entries (and paths only moved
/// between entries) are not listed, so existing configs are never re-checked.
pub fn new_model_paths(before: &Config, after: &Config) -> Vec<NewModelPath> {
    let known: std::collections::HashSet<&std::path::Path> = before
        .models
        .iter()
        .flat_map(|m| std::iter::once(m.path.as_path()).chain(m.classes.as_deref()))
        .collect();
    let mut out = Vec::new();
    for m in &after.models {
        for (field, p) in [("path", Some(&m.path)), ("classes", m.classes.as_ref())] {
            if let Some(p) = p
                && !known.contains(p.as_path())
            {
                out.push(NewModelPath {
                    model: m.effective_name(),
                    field,
                    path: p.clone(),
                });
            }
        }
    }
    out
}

/// The "Night / IR" fieldset: `night_confidence_threshold` and `night_person_threshold`
/// (empty = off), `person_aspect_min` / `person_aspect_max` (both or neither), and the
/// `drop_person_on_plant` / `static_suppression` checkboxes, read only with the fieldset's
/// `night_settings` marker. Only the parts the fieldset shows are changed: other label
/// thresholds and overlap rules stay as configured.
fn parse_night_fields(
    c: &mut Config,
    form: &std::collections::HashMap<String, String>,
) -> Result<()> {
    use anyhow::bail;
    let get = |k: &str| form.get(k).map(|v| v.trim());
    let opt = |field: &str| -> Result<Option<f32>> {
        match get(field) {
            None | Some("") => Ok(None),
            Some(v) => {
                let x: f32 = v
                    .parse()
                    .map_err(|_| anyhow::anyhow!("{field}: '{v}' is not a valid number"))?;
                Ok(Some(x))
            }
        }
    };
    if form.contains_key("night_confidence_threshold") {
        let t = opt("night_confidence_threshold")?;
        if let Some(t) = t {
            check_threshold("night_confidence_threshold", t)?;
        }
        c.night_confidence_threshold = t;
    }
    if form.contains_key("night_person_threshold") {
        let t = opt("night_person_threshold")?;
        c.night_label_thresholds
            .retain(|k, _| !k.eq_ignore_ascii_case(crate::model::filters::PERSON));
        if let Some(t) = t {
            check_threshold("night_label_thresholds.person", t)?;
            c.night_label_thresholds
                .insert(crate::model::filters::PERSON.to_string(), t);
        }
    }
    if form.contains_key("person_aspect_min") || form.contains_key("person_aspect_max") {
        c.person_aspect = match (opt("person_aspect_min")?, opt("person_aspect_max")?) {
            (None, None) => None,
            (Some(lo), Some(hi)) => Some([lo, hi]),
            _ => bail!(
                "person_aspect: set both the minimum and the maximum width/height (e.g. 0.2 and \
                 0.9), or clear both to turn the rule off"
            ),
        };
    }
    if form.contains_key("night_settings") {
        let checkbox = |k: &str| {
            get(k)
                .is_some_and(|v| !matches!(v.to_ascii_lowercase().as_str(), "false" | "off" | "0"))
        };
        let want = checkbox("drop_person_on_plant");
        if want && !has_plant_rule(c) {
            c.suppress_overlaps
                .push(crate::model::filters::OverlapRule {
                    drop: PLANT_RULE.0.to_string(),
                    if_overlaps: PLANT_RULE.1.to_string(),
                    iou: crate::model::filters::DEFAULT_OVERLAP_IOU,
                    night_only: false,
                });
        } else if !want {
            c.suppress_overlaps.retain(|r| !is_plant_rule(r));
        }
        c.static_suppression.enabled = checkbox("static_suppression");
    }
    Ok(())
}

/// Apply the Models card of the `/config` page: exactly the models named in `enabled` (effective
/// names, matched like `/v1/vision/custom/{model}`) are enabled, the rest disabled, and
/// `default_model` becomes `default` (an empty/absent `default` clears it). A chosen default that
/// is not checked is enabled too. Fails without modifying `config` when a name is unknown or no
/// model ends up enabled.
pub fn apply_models_selection(
    config: &mut Config,
    enabled: &[String],
    default: Option<&str>,
) -> Result<()> {
    use crate::registry::normalize_name;
    use anyhow::bail;
    let mut c = config.clone();
    let index_of = |name: &str| {
        let key = normalize_name(name);
        c.models
            .iter()
            .position(|m| normalize_name(&m.effective_name()) == key)
    };
    let mut on = vec![false; c.models.len()];
    for name in enabled {
        match index_of(name).and_then(|i| on.get_mut(i)) {
            Some(slot) => *slot = true,
            None => bail!("'{name}' is not one of the configured models"),
        }
    }
    let default = default.map(str::trim).filter(|d| !d.is_empty());
    let default_idx = match default {
        Some(d) => match index_of(d) {
            Some(i) => Some(i),
            None => bail!("default model '{d}' is not one of the configured models"),
        },
        None => None,
    };
    if let Some(slot) = default_idx.and_then(|i| on.get_mut(i)) {
        *slot = true;
    }
    if !on.iter().any(|&b| b) {
        bail!("select at least one model to load");
    }
    for (m, e) in c.models.iter_mut().zip(on) {
        m.enabled = e;
    }
    c.default_model = default_idx
        .and_then(|i| c.models.get(i))
        .map(|m| m.effective_name());
    *config = c;
    Ok(())
}

/// A per-model `device` change made by [`apply_model_devices`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceChange {
    pub model: String,
    /// Previous per-model device (None = global).
    pub from: Option<String>,
    /// New per-model device (None = global).
    pub to: Option<String>,
}

impl DeviceChange {
    /// "IPcam-general: global -> openvino:cpu".
    pub fn describe(&self) -> String {
        let show = |d: &Option<String>| d.clone().unwrap_or_else(|| "global".to_string());
        format!("{}: {} -> {}", self.model, show(&self.from), show(&self.to))
    }
}

/// Set the per-model `device` of the named models (matched like `/v1/vision/custom/{model}`);
/// `None` or an empty/"global" value clears it (the model follows the global device). Every
/// name and device is validated first; on error `config` is unchanged. Returns the entries that
/// actually changed.
pub fn apply_model_devices(
    config: &mut Config,
    devices: &[(String, Option<String>)],
) -> Result<Vec<DeviceChange>> {
    use crate::registry::normalize_name;
    let mut c = config.clone();
    let mut changes = Vec::new();
    for (name, device) in devices {
        let key = normalize_name(name);
        let Some(m) = c
            .models
            .iter_mut()
            .find(|m| normalize_name(&m.effective_name()) == key)
        else {
            anyhow::bail!("'{name}' is not one of the configured models");
        };
        let device = device
            .as_deref()
            .map(str::trim)
            .filter(|d| !d.is_empty() && !d.eq_ignore_ascii_case("global"));
        let new = match device {
            Some(d) => {
                crate::backend::spec::parse(d)
                    .map_err(|e| anyhow::anyhow!("model '{name}' device: {e}"))?;
                Some(d.to_string())
            }
            None => None,
        };
        if m.device != new {
            changes.push(DeviceChange {
                model: m.effective_name(),
                from: m.device.clone(),
                to: new.clone(),
            });
            m.device = new;
        }
    }
    *config = c;
    Ok(changes)
}

/// A per-model `confidence_threshold` change made by [`apply_model_thresholds`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ThresholdChange {
    pub model: String,
    /// Previous per-model threshold (None = global).
    #[serde(serialize_with = "thr::ser_opt")]
    pub from: Option<f32>,
    #[serde(serialize_with = "thr::ser")]
    pub to: f32,
    /// The global `confidence_threshold`.
    #[serde(serialize_with = "thr::ser")]
    pub global: f32,
}

impl ThresholdChange {
    /// "IPcam-general: 0.50 (global) -> 0.35".
    pub fn describe(&self) -> String {
        let from = match self.from {
            Some(t) => format!("{t:.2}"),
            None => format!("{:.2} (global)", self.global),
        };
        format!("{}: {from} -> {:.2}", self.model, self.to)
    }
}

/// Set the per-model `confidence_threshold` of the named models (matched like
/// `/v1/vision/custom/{model}`). A model whose effective threshold (its own, else the global
/// one) already equals the value is left alone. Every name and value (0 < t <= 1) is validated
/// first; on error `config` is unchanged. Returns the entries that changed.
pub fn apply_model_thresholds(
    config: &mut Config,
    thresholds: &[(String, f32)],
) -> Result<Vec<ThresholdChange>> {
    use crate::registry::normalize_name;
    let mut c = config.clone();
    let global = c.confidence_threshold;
    let mut changes = Vec::new();
    for (name, t) in thresholds {
        // The shared range ([`check_threshold`]); a benchmark pick of 0 (report every box)
        // is never a useful recommendation, so this applier also wants it above 0.
        check_threshold(&format!("model '{name}': confidence_threshold"), *t)?;
        let t = &round_threshold(*t);
        if *t <= 0.0 {
            anyhow::bail!("model '{name}': confidence threshold rounds to 0");
        }
        let key = normalize_name(name);
        let Some(m) = c
            .models
            .iter_mut()
            .find(|m| normalize_name(&m.effective_name()) == key)
        else {
            anyhow::bail!("'{name}' is not one of the configured models");
        };
        let effective = m.confidence_threshold.unwrap_or(global);
        if (effective - t).abs() > 1e-6 {
            changes.push(ThresholdChange {
                model: m.effective_name(),
                from: m.confidence_threshold,
                to: *t,
                global,
            });
            m.confidence_threshold = Some(*t);
        }
    }
    *config = c;
    Ok(changes)
}

/// A confidence profile change ([`crate::config::ModelConfig::device_thresholds`]) made by
/// [`apply_device_thresholds`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProfileChange {
    pub model: String,
    /// Canonical device spec ([`crate::config::device_threshold_key`]).
    pub device: String,
    /// Previous profile value (None = no profile: the device used the model's default).
    #[serde(serialize_with = "thr::ser_opt")]
    pub from: Option<f32>,
    /// New profile value (None = removed).
    #[serde(serialize_with = "thr::ser_opt")]
    pub to: Option<f32>,
}

impl ProfileChange {
    /// "rfdetr-nano on ort:coreml: 0.38 -> 0.30", "... : none -> 0.34" (new),
    /// "... : 0.30 -> none" (removed).
    pub fn describe(&self) -> String {
        let show = |t: Option<f32>| t.map_or_else(|| "none".to_string(), |t| format!("{t:.2}"));
        format!(
            "{} on {}: {} -> {}",
            self.model,
            self.device,
            show(self.from),
            show(self.to)
        )
    }
}

/// Set or remove confidence profiles: each entry is `(model, device, value)`, the model matched
/// like `/v1/vision/custom/{model}`, the device any spec naming one device (stored under its
/// canonical key, [`crate::config::device_threshold_key`]; an existing entry for the same device
/// under another spelling, e.g. `openvino:gpu.0` for `openvino:gpu`, is replaced), and `None`
/// removing the profile. Values must pass [`check_threshold`] and not round to 0. Every entry
/// is validated first; on error `config` is unchanged. A later entry for the same model and
/// device wins. Returns the profiles that actually changed, in first-touched order; the model's
/// `confidence_threshold` is never changed.
pub fn apply_device_thresholds(
    config: &mut Config,
    entries: &[(String, String, Option<f32>)],
) -> Result<Vec<ProfileChange>> {
    use crate::config::{device_threshold_key, lookup_device_threshold, same_device};
    use crate::registry::normalize_name;
    let mut c = config.clone();
    // (model index, canonical device) in first-touched order.
    let mut touched: Vec<(usize, String)> = Vec::new();
    for (name, device, value) in entries {
        let key = device_threshold_key(device)
            .map_err(|e| anyhow::anyhow!("model '{name}' confidence profile: {e}"))?;
        let value = match value {
            Some(t) => {
                check_threshold(&format!("model '{name}': confidence profile for {key}"), *t)?;
                let t = round_threshold(*t);
                if t <= 0.0 {
                    anyhow::bail!("model '{name}': confidence profile for {key} rounds to 0");
                }
                Some(t)
            }
            None => None,
        };
        let wanted = normalize_name(name);
        let Some((i, m)) = c
            .models
            .iter_mut()
            .enumerate()
            .find(|(_, m)| normalize_name(&m.effective_name()) == wanted)
        else {
            anyhow::bail!("'{name}' is not one of the configured models");
        };
        let map = &mut m.device_thresholds;
        if same_value(lookup_device_threshold(map, &key), value) {
            continue;
        }
        map.retain(|k, _| !(k == &key || same_device(k, &key)));
        if let Some(t) = value {
            map.insert(key.clone(), t);
        }
        if !touched.iter().any(|(j, k)| *j == i && *k == key) {
            touched.push((i, key));
        }
    }
    // Models are only edited, never added or removed, so `i` indexes both configs.
    let changes = touched
        .into_iter()
        .filter_map(|(i, key)| {
            let (old, new) = (config.models.get(i)?, c.models.get(i)?);
            let from = lookup_device_threshold(&old.device_thresholds, &key);
            let to = new.device_thresholds.get(&key).copied();
            (!same_value(from, to)).then(|| ProfileChange {
                model: new.effective_name(),
                device: key,
                from,
                to,
            })
        })
        .collect();
    *config = c;
    Ok(changes)
}

/// Two optional thresholds are equal (within `f32` noise).
fn same_value(a: Option<f32>, b: Option<f32>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => (a - b).abs() <= 1e-6,
        (None, None) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{LogLevel, ModelConfig};
    use crate::model::ModelFamilyKind;
    use std::collections::HashMap;

    fn three_models() -> Config {
        Config {
            models: ["a", "b", "c"]
                .iter()
                .map(|n| ModelConfig {
                    path: format!("models/{n}.onnx").into(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn apply_thresholds_writes_per_model_values() {
        use super::apply_model_thresholds;
        let m = |n: &str, t: Option<f32>| ModelConfig {
            name: Some(n.into()),
            path: format!("models/{n}.onnx").into(),
            confidence_threshold: t,
            ..Default::default()
        };
        let mut c = Config {
            confidence_threshold: 0.5,
            models: vec![m("a", None), m("b", Some(0.4)), m("c", None)],
            ..Default::default()
        };
        let changes = apply_model_thresholds(
            &mut c,
            &[("A".into(), 0.35), ("b".into(), 0.4), ("c".into(), 0.5)],
        )
        .unwrap();
        // b and c already have these effective thresholds.
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].describe(), "a: 0.50 (global) -> 0.35");
        assert_eq!(c.models[0].confidence_threshold, Some(0.35));
        assert_eq!(c.models[2].confidence_threshold, None);
        let changes = apply_model_thresholds(&mut c, &[("b".into(), 0.6)]).unwrap();
        assert_eq!(changes[0].describe(), "b: 0.40 -> 0.60");
        // Bad input changes nothing.
        let before = c.clone();
        for bad in [("zzz", 0.3), ("a", 0.0), ("a", 1.5), ("a", f32::NAN)] {
            assert!(
                apply_model_thresholds(&mut c, &[("c".into(), 0.3), (bad.0.into(), bad.1)])
                    .is_err()
            );
            assert_eq!(c, before);
        }
        // The value round-trips through the config file as written.
        let text = serde_json::to_string(&c).unwrap();
        assert!(text.contains(r#""confidence_threshold":0.35"#), "{text}");
    }

    #[test]
    fn device_thresholds_set_remove_skip_and_stay_atomic() {
        let mut c = three_models();
        c.fill_model_names();
        c.models[1].confidence_threshold = Some(0.38);
        c.models[1]
            .device_thresholds
            .insert("ort:coreml".into(), 0.38);
        let e = |m: &str, d: &str, t: Option<f32>| (m.to_string(), d.to_string(), t);
        let changes = apply_device_thresholds(
            &mut c,
            &[
                e("B", "ort:coreml", Some(0.30)),
                e("b", "CPU", Some(0.344)),
                // No profile to remove, and an unchanged value: skipped.
                e("a", "ort:cpu", None),
                e("c", "ort:cpu", None),
            ],
        )
        .unwrap();
        let text: Vec<String> = changes.iter().map(ProfileChange::describe).collect();
        assert_eq!(
            text,
            [
                "b on ort:coreml: 0.38 -> 0.30",
                "b on openvino:cpu: none -> 0.34"
            ]
        );
        let p = &c.models[1].device_thresholds;
        assert_eq!(p.get("ort:coreml"), Some(&0.3));
        assert_eq!(p.get("openvino:cpu"), Some(&0.34));
        // The model default is left alone.
        assert_eq!(c.models[1].confidence_threshold, Some(0.38));
        assert!(c.models[0].device_thresholds.is_empty());
        let json = serde_json::to_value(&changes[1]).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"model": "b", "device": "openvino:cpu", "from": null, "to": 0.34})
        );

        // The same value again: no change. Removing: from -> none.
        assert!(
            apply_device_thresholds(&mut c, &[e("b", "ort:coreml", Some(0.3))])
                .unwrap()
                .is_empty()
        );
        let changes = apply_device_thresholds(&mut c, &[e("b", "ort:coreml", None)]).unwrap();
        assert_eq!(changes[0].describe(), "b on ort:coreml: 0.30 -> none");
        assert!(!c.models[1].device_thresholds.contains_key("ort:coreml"));

        // Another spelling of a saved device replaces that entry (gpu.0 == gpu).
        c.models[2]
            .device_thresholds
            .insert("openvino:gpu.0".into(), 0.2);
        let changes = apply_device_thresholds(&mut c, &[e("c", "GPU", Some(0.25))]).unwrap();
        assert_eq!(changes[0].describe(), "c on openvino:gpu: 0.20 -> 0.25");
        let keys: Vec<&String> = c.models[2].device_thresholds.keys().collect();
        assert_eq!(keys, ["openvino:gpu"]);
        // A later entry for the same device wins; set and back is no change.
        let changes = apply_device_thresholds(
            &mut c,
            &[e("c", "openvino:gpu", Some(0.4)), e("c", "GPU", Some(0.25))],
        )
        .unwrap();
        assert!(changes.is_empty(), "{changes:?}");
        assert!(c.validate().is_ok());

        // Bad input changes nothing.
        let before = c.clone();
        for bad in [
            e("zzz", "ort:cpu", Some(0.3)),
            e("a", "auto", Some(0.3)),
            e("a", "nonsense", Some(0.3)),
            e("a", "ort:cpu", Some(0.0)),
            e("a", "ort:cpu", Some(0.004)),
            e("a", "ort:cpu", Some(1.5)),
            e("a", "ort:cpu", Some(f32::NAN)),
        ] {
            let r = apply_device_thresholds(&mut c, &[e("a", "ort:cpu", Some(0.3)), bad.clone()]);
            assert!(r.is_err(), "{bad:?}");
            assert_eq!(c, before, "{bad:?}");
        }
        let err = apply_device_thresholds(&mut c, &[e("a", "auto", Some(0.3))]).unwrap_err();
        assert!(format!("{err:#}").contains("one device"), "{err:#}");
    }

    #[test]
    fn model_devices_are_applied_atomically() {
        let mut c = super::Config {
            models: ["IPcam-general", "dfine-s"]
                .iter()
                .map(|n| ModelConfig {
                    path: format!("models/{n}.onnx").into(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        };
        let set = |pairs: &[(&str, Option<&str>)]| -> Vec<(String, Option<String>)> {
            pairs
                .iter()
                .map(|(n, d)| (n.to_string(), d.map(str::to_string)))
                .collect()
        };
        let ch = super::apply_model_devices(
            &mut c,
            &set(&[
                ("ipcam-general.onnx", Some("openvino:cpu")),
                ("DFINE-S", Some(" ort:coreml ")),
            ]),
        )
        .unwrap();
        assert_eq!(ch.len(), 2);
        assert_eq!(ch[0].describe(), "IPcam-general: global -> openvino:cpu");
        assert_eq!(c.models[1].device.as_deref(), Some("ort:coreml"));
        // Unchanged values are not reported; "" and "global" clear the override.
        let ch = super::apply_model_devices(
            &mut c,
            &set(&[
                ("IPcam-general", Some("openvino:cpu")),
                ("dfine-s", Some("global")),
            ]),
        )
        .unwrap();
        assert_eq!(ch.len(), 1);
        assert_eq!(ch[0].to, None);
        assert_eq!(c.models[1].device, None);
        // Bad device or unknown model: error, nothing changed.
        let before = c.clone();
        for bad in [
            set(&[
                ("dfine-s", Some("ort:coreml")),
                ("IPcam-general", Some("vulkan")),
            ]),
            set(&[("dfine-s", Some("ort:coreml")), ("zzz", Some("ort:cpu"))]),
        ] {
            assert!(super::apply_model_devices(&mut c, &bad).is_err());
            assert_eq!(c, before);
        }
    }

    fn form(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn with_model() -> Config {
        Config {
            models: vec![ModelConfig {
                path: "models/IPcam-general.onnx".into(),
                family: ModelFamilyKind::Yolo5,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn checkboxes_change_only_with_their_marker() {
        let mut c = with_model();
        c.force_cpu = true;
        c.save_ref_image = true;
        // A post without the markers (e.g. the models JSON form) leaves both boxes alone.
        apply_config_form(&mut c, &form(&[("port", "4000")])).unwrap();
        assert!(c.force_cpu && c.save_ref_image);
        // With the markers an absent box means unchecked.
        apply_config_form(
            &mut c,
            &form(&[("inference_settings", "1"), ("image_settings", "1")]),
        )
        .unwrap();
        assert!(!c.force_cpu && !c.save_ref_image);
    }

    #[test]
    fn save_image_retention_fields_apply() {
        let mut c = with_model();
        apply_config_form(
            &mut c,
            &form(&[
                ("save_image_max_age_days", "0"),
                ("save_image_max_mb", "512"),
            ]),
        )
        .unwrap();
        assert_eq!((c.save_image_max_age_days, c.save_image_max_mb), (0, 512));
    }

    #[test]
    fn config_form_applies_all_fields() {
        let mut c = with_model();
        let models = r#"[{"name":"a","path":"m/a.xml","family":"yolo26"},
                         {"path":"m/IPcam-general.onnx","family":"yolo5","lazy":true}]"#;
        let f = form(&[
            ("port", " 4000 "),
            ("request_timeout_secs", "20"),
            ("worker_queue_size", "8"),
            ("device", "GPU.1"),
            ("gpu_index", "1"),
            ("inference_settings", "1"),
            ("force_cpu", "on"),
            ("cache_dir", ""),
            ("confidence_threshold", "0.35"),
            ("nms_iou", "0.6"),
            ("object_filter", " person, car ,,dog "),
            ("log_level", "DEBUG"),
            ("log_path", "logs"),
            ("save_image_path", ""),
            ("image_settings", "1"),
            ("save_ref_image", "true"),
            ("intra_threads", "4"),
            ("default_model", "ipcam-general.onnx"),
            ("models_json", models),
        ]);
        apply_config_form(&mut c, &f).unwrap();
        assert_eq!(c.port, 4000);
        assert_eq!(c.request_timeout_secs, 20);
        assert_eq!(c.worker_queue_size, 8);
        assert_eq!(c.device, "GPU.1");
        assert_eq!(c.gpu_index, 1);
        assert!(c.force_cpu);
        assert_eq!(c.cache_dir, "");
        assert_eq!(c.confidence_threshold, 0.35);
        assert_eq!(c.nms_iou, 0.6);
        assert_eq!(c.object_filter, vec!["person", "car", "dog"]);
        assert_eq!(c.log_level, LogLevel::Debug);
        // Relative directory settings are stored resolved under the data root.
        assert_eq!(c.log_path, Some(c.data_root().join("logs")));
        assert_eq!(c.save_image_path, None);
        assert!(c.save_ref_image);
        assert_eq!(c.intra_threads, 4);
        assert_eq!(c.models.len(), 2);
        assert!(c.models[1].lazy);
        assert_eq!(c.default_model_index(), Some(1));

        // Unchecked boxes are absent from a form that shows them (marker fields); absent text
        // fields keep their values.
        apply_config_form(
            &mut c,
            &form(&[
                ("default_model", ""),
                ("inference_settings", "1"),
                ("image_settings", "1"),
            ]),
        )
        .unwrap();
        assert!(!c.force_cpu && !c.save_ref_image);
        assert_eq!(c.port, 4000);
        assert_eq!(c.default_model, None);
        assert_eq!(c.models.len(), 2);
    }

    #[test]
    fn access_settings() {
        let d = Config::default();
        assert_eq!(d.bind_address.to_string(), "0.0.0.0");
        assert!(!d.allow_remote_admin);
        // Old files without the fields get the defaults; the values round-trip.
        let c: Config = serde_json::from_str(r#"{"port": 1234}"#).unwrap();
        assert_eq!(
            (c.bind_address, c.allow_remote_admin),
            (d.bind_address, false)
        );
        let c: Config =
            serde_json::from_str(r#"{"bind_address": "::1", "allow_remote_admin": true}"#).unwrap();
        assert_eq!(c.bind_address.to_string(), "::1");
        assert!(c.allow_remote_admin);
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["bind_address"], "::1");

        let mut c = with_model();
        apply_config_form(
            &mut c,
            &form(&[
                ("bind_address", "127.0.0.1"),
                ("access_settings", "1"),
                ("allow_remote_admin", "on"),
            ]),
        )
        .unwrap();
        assert_eq!(c.bind_address.to_string(), "127.0.0.1");
        assert!(c.allow_remote_admin);
        // A form without the access section leaves the flag alone.
        apply_config_form(&mut c, &form(&[("port", "4000")])).unwrap();
        assert!(c.allow_remote_admin);
        // The access section without the checkbox turns it off.
        apply_config_form(&mut c, &form(&[("access_settings", "1")])).unwrap();
        assert!(!c.allow_remote_admin);
    }

    #[test]
    fn config_form_rejects_invalid_input_atomically() {
        let original = with_model();
        let cases: &[(&str, &str, &str)] = &[
            ("port", "0", "port"),
            ("port", "70000", "port"),
            ("port", "abc", "port"),
            ("request_timeout_secs", "0", "request_timeout_secs"),
            ("save_image_max_age_days", "3651", "save_image_max_age_days"),
            ("save_image_max_mb", "x", "save_image_max_mb"),
            ("worker_queue_size", "-1", "worker_queue_size"),
            ("confidence_threshold", "1.5", "confidence_threshold"),
            ("nms_iou", "x", "nms_iou"),
            ("log_level", "loud", "log_level"),
            ("device", "G P U", "device"),
            ("models_json", "[{not json", "models"),
            ("models_json", "[]", "at least one"),
            ("models_json", r#"{"path":"a.xml"}"#, "models"),
            (
                "models_json",
                r#"[{"path":"a/x.onnx"},{"path":"b/X.xml"}]"#,
                "duplicate",
            ),
            ("models_json", r#"[{"name":"x"}]"#, "path"),
            (
                "models_json",
                r#"[{"path":"a.xml","enabled":false}]"#,
                "must be enabled",
            ),
            ("default_model", "nope", "default_model"),
            ("bind_address", "localhost", "bind_address"),
            ("save_image_path", "/etc/cron.d", "system directory"),
            (
                "save_image_path",
                "C:\\Windows\\System32",
                "system directory",
            ),
            ("log_path", "../../x", "'..'"),
            ("log_path", "\\\\server\\share", "UNC"),
            ("cache_dir", "C:cache", "drive-relative"),
            ("cache_dir", "a\u{0}b", "NUL"),
        ];
        for (field, value, needle) in cases {
            let mut c = original.clone();
            let f = form(&[(field, value), ("force_cpu", "on"), ("port", "1234")]);
            let f = if *field == "port" {
                form(&[(field, value), ("force_cpu", "on")])
            } else {
                f
            };
            let err = apply_config_form(&mut c, &f)
                .err()
                .unwrap_or_else(|| panic!("{field}={value} must fail"));
            assert!(
                format!("{err:#}").contains(needle),
                "{field}={value}: {err:#}"
            );
            assert_eq!(c, original, "{field}={value} must not modify the config");
        }
    }

    // Unix path layout (absolute `/...` paths, lib64 / ld.so.conf dirs).
    #[cfg(unix)]
    #[test]
    fn directory_settings_are_resolved_under_the_data_root() {
        let mut c = with_model();
        c.download_dir = Some(PathBuf::from("/srv/prism"));
        apply_config_form(
            &mut c,
            &form(&[
                ("cache_dir", "cache"),
                ("log_path", " \"logs\" "),
                ("save_image_path", "/var/lib/prism/shots/"),
            ]),
        )
        .unwrap();
        let root = PathBuf::from("/srv/prism");
        assert_eq!(c.cache_dir, root.join("cache").display().to_string());
        assert_eq!(c.log_path, Some(root.join("logs")));
        assert_eq!(
            c.save_image_path,
            Some(PathBuf::from("/var/lib/prism/shots/"))
        );
        // Empty clears the optional ones and disables the cache.
        apply_config_form(
            &mut c,
            &form(&[("cache_dir", ""), ("log_path", ""), ("save_image_path", "")]),
        )
        .unwrap();
        assert_eq!(c.cache_dir, "");
        assert_eq!((c.log_path, c.save_image_path), (None, None));
    }

    #[test]
    fn models_selection() {
        let mut c = three_models();
        apply_models_selection(&mut c, &["B".into(), "c.onnx".into()], Some("c")).unwrap();
        let on: Vec<bool> = c.models.iter().map(|m| m.enabled).collect();
        assert_eq!(on, [false, true, true]);
        assert_eq!(c.default_model.as_deref(), Some("c"));

        // A default that is not checked gets enabled.
        apply_models_selection(&mut c, &["b".into()], Some("a")).unwrap();
        let on: Vec<bool> = c.models.iter().map(|m| m.enabled).collect();
        assert_eq!(on, [true, true, false]);
        assert_eq!(c.default_model.as_deref(), Some("a"));

        // No default chosen clears it.
        apply_models_selection(&mut c, &["c".into()], None).unwrap();
        assert_eq!(c.default_model, None);
        assert_eq!(c.effective_default_index(), Some(2));

        let before = c.clone();
        for (enabled, default, needle) in [
            (vec![], None, "at least one"),
            (vec![], Some(" "), "at least one"),
            (vec!["zzz".to_string()], None, "zzz"),
            (vec!["a".to_string()], Some("zzz"), "zzz"),
        ] {
            let err = apply_models_selection(&mut c, &enabled, default).unwrap_err();
            assert!(format!("{err:#}").contains(needle), "{err:#}");
            assert_eq!(c, before);
        }
    }

    #[test]
    fn config_form_device_specs() {
        for ok in ["auto", "openvino:gpu.1", "ort:cuda:0", "cpu", "NPU"] {
            let mut c = with_model();
            apply_config_form(&mut c, &form(&[("device", ok)])).unwrap();
            assert_eq!(c.device, ok);
        }
        let mut c = with_model();
        let err = apply_config_form(&mut c, &form(&[("device", "ort:gpu")])).unwrap_err();
        assert!(format!("{err:#}").starts_with("device:"), "{err:#}");
        let models = r#"[{"path":"a.onnx","device":"XPU"}]"#;
        let err = apply_config_form(&mut c, &form(&[("models_json", models)])).unwrap_err();
        assert!(format!("{err:#}").contains("device"), "{err:#}");
    }

    #[test]
    fn log_timezone_is_parsed_from_the_form() {
        let mut c = Config::default();
        parse_config_form(&mut c, &form(&[("log_timezone", " America/New_York ")])).unwrap();
        assert_eq!(c.log_timezone.to_string(), "America/New_York");
        let err = parse_config_form(&mut c, &form(&[("log_timezone", "mars")])).unwrap_err();
        assert!(format!("{err:#}").contains("log_timezone"), "{err:#}");
    }

    #[test]
    fn log_format_is_accepted_from_api_forms() {
        let mut c = Config::default();
        parse_config_form(&mut c, &form(&[("log_format", "json")])).unwrap();
        assert_eq!(c.log_format, crate::config::LogFormat::Json);
        let err = parse_config_form(&mut c, &form(&[("log_format", "xml")])).unwrap_err();
        assert!(format!("{err:#}").contains("log_format"), "{err:#}");
    }

    #[test]
    fn night_fieldset_sets_and_clears_only_its_parts() {
        let mut c = with_model();
        c.night_label_thresholds.insert("car".into(), 0.8);
        c.suppress_overlaps
            .push(crate::model::filters::OverlapRule {
                drop: "dog".into(),
                if_overlaps: "cat".into(),
                iou: 0.5,
                night_only: false,
            });
        let on = form(&[
            ("night_settings", "1"),
            ("night_confidence_threshold", "0.6"),
            ("night_person_threshold", "0.7"),
            ("person_aspect_min", "0.2"),
            ("person_aspect_max", "0.9"),
            ("drop_person_on_plant", "on"),
            ("static_suppression", "on"),
        ]);
        apply_config_form(&mut c, &on).unwrap();
        assert_eq!(c.night_confidence_threshold, Some(0.6));
        assert_eq!(night_person_threshold(&c), Some(0.7));
        assert_eq!(c.night_label_thresholds.get("car"), Some(&0.8));
        assert_eq!(c.person_aspect, Some([0.2, 0.9]));
        assert!(has_plant_rule(&c));
        assert_eq!(c.suppress_overlaps.len(), 2);
        assert!(c.static_suppression.enabled);
        // Posting it again adds no second plant rule.
        apply_config_form(&mut c, &on).unwrap();
        assert_eq!(c.suppress_overlaps.len(), 2);
        // A form without the marker (another form, an API client) leaves the checkboxes.
        apply_config_form(&mut c, &form(&[("port", "4000")])).unwrap();
        assert!(has_plant_rule(&c) && c.static_suppression.enabled);
        // Cleared fields and unchecked boxes turn the parts off; the rest stays.
        let off = form(&[
            ("night_settings", "1"),
            ("night_confidence_threshold", ""),
            ("night_person_threshold", ""),
            ("person_aspect_min", ""),
            ("person_aspect_max", ""),
        ]);
        apply_config_form(&mut c, &off).unwrap();
        assert_eq!(c.night_confidence_threshold, None);
        assert_eq!(night_person_threshold(&c), None);
        assert_eq!(c.night_label_thresholds.get("car"), Some(&0.8));
        assert_eq!(c.person_aspect, None);
        assert!(!has_plant_rule(&c));
        assert_eq!(c.suppress_overlaps.len(), 1, "the dog/cat rule stays");
        assert!(!c.static_suppression.enabled);
        // Errors leave the config untouched.
        let before = c.clone();
        for (k, v, want) in [
            (
                "night_confidence_threshold",
                "2",
                "night_confidence_threshold",
            ),
            ("night_person_threshold", "x", "night_person_threshold"),
            ("person_aspect_min", "0.2", "person_aspect"),
            ("person_aspect_max", "0.1", "person_aspect"),
        ] {
            let mut f = form(&[(k, v)]);
            if k == "person_aspect_max" {
                f.insert("person_aspect_min".into(), "0.5".into());
            }
            let e = format!("{:#}", apply_config_form(&mut c, &f).unwrap_err());
            assert!(e.contains(want), "{k}: {e}");
            assert_eq!(c, before);
        }
    }

    /// The Folders fieldset: relative runtime dirs resolve under the exe dir, `models_dir`
    /// under the (new) data root; empty = the default; an unchanged value keeps its stored
    /// (possibly relative) form; bad values are refused without touching the config.
    #[test]
    fn folder_fields_parse_default_and_validate() {
        let exe = crate::exe_dir();
        let base = std::env::temp_dir().join("bop-folders");
        let dl = base.join("data");
        let mut c = with_model();
        apply_config_form(
            &mut c,
            &form(&[
                ("download_dir", &dl.display().to_string()),
                ("models_dir", "my-models"),
                ("onnxruntime_dir", "rt/ort"),
                ("openvino_dir", " \"ov\" "),
            ]),
        )
        .unwrap();
        assert_eq!(c.download_dir, Some(dl.clone()));
        assert_eq!(c.data_root(), dl);
        assert_eq!(c.models_dir, dl.join("my-models"));
        assert_eq!(c.onnxruntime_dir, Some(exe.join("rt/ort")));
        assert_eq!(c.openvino_dir, Some(exe.join("ov")));

        // Empty = the default.
        let f = form(&[
            ("download_dir", ""),
            ("models_dir", " "),
            ("onnxruntime_dir", ""),
            ("openvino_dir", ""),
        ]);
        apply_config_form(&mut c, &f).unwrap();
        let d = Config::default();
        assert_eq!(
            (
                &c.download_dir,
                &c.models_dir,
                &c.onnxruntime_dir,
                &c.openvino_dir
            ),
            (
                &d.download_dir,
                &d.models_dir,
                &d.onnxruntime_dir,
                &d.openvino_dir
            )
        );

        // The page posts the shown values back: a relative stored value stays as it is.
        let mut c = with_model();
        c.models_dir = "models".into();
        c.openvino_dir = Some("ov".into());
        c.normalize();
        let before = c.clone();
        apply_config_form(
            &mut c,
            &form(&[
                ("download_dir", ""),
                ("models_dir", "models"),
                ("onnxruntime_dir", ""),
                ("openvino_dir", "ov"),
            ]),
        )
        .unwrap();
        assert_eq!(c, before);

        for (field, value, needle) in [
            ("download_dir", "/etc/prism", "system directory"),
            ("models_dir", "/usr/share/models", "system directory"),
            (
                "onnxruntime_dir",
                "C:\\Windows\\System32",
                "system directory",
            ),
            ("openvino_dir", "../ov", "'..'"),
            ("download_dir", "\\\\nas\\share", "UNC"),
            ("models_dir", "C:models", "drive-relative"),
        ] {
            let mut c = with_model();
            let e = format!(
                "{:#}",
                apply_config_form(&mut c, &form(&[(field, value)])).unwrap_err()
            );
            assert!(
                e.contains(field) && e.contains(needle),
                "{field}={value}: {e}"
            );
            assert_eq!(c, with_model());
        }
    }

    #[test]
    fn new_model_paths_lists_only_new_or_changed_files() {
        let before = Config {
            models: vec![
                ModelConfig {
                    path: "/far/away/a.onnx".into(),
                    classes: Some("/far/away/a.yaml".into()),
                    ..Default::default()
                },
                ModelConfig {
                    path: "models/b.onnx".into(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        // Unchanged (also with other fields edited, or reordered): nothing to check.
        let mut after = before.clone();
        after.models[0].lazy = true;
        after.models.swap(0, 1);
        assert!(new_model_paths(&before, &after).is_empty());
        // A changed path, a new classes file and a new entry are listed.
        after.models[0].path = "/elsewhere/b.onnx".into();
        after.models[1].classes = Some("/far/away/other.yaml".into());
        after.models.push(ModelConfig {
            name: Some("c".into()),
            path: "/x/c.onnx".into(),
            ..Default::default()
        });
        let got: Vec<(String, &str, PathBuf)> = new_model_paths(&before, &after)
            .into_iter()
            .map(|n| (n.model, n.field, n.path))
            .collect();
        assert_eq!(
            got,
            vec![
                (
                    after.models[0].effective_name(),
                    "path",
                    "/elsewhere/b.onnx".into()
                ),
                (
                    after.models[1].effective_name(),
                    "classes",
                    "/far/away/other.yaml".into()
                ),
                ("c".to_string(), "path", "/x/c.onnx".into()),
            ]
        );
    }
}
