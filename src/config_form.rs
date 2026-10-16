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
];

/// Apply a submitted `/config` form to `config`. All fields are validated first; on any error
/// `config` is left untouched. Text fields that are absent keep their current value; empty
/// optional paths / `default_model` become `None`. The directory settings
/// (`save_image_path`, `log_path`, `cache_dir`) go through
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
    // directories refused (SV-03). `download_dir` is not a form field, so the root is fixed.
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
    if let Some(v) = form.get("models_json") {
        c.models = serde_json::from_str(v).map_err(|e| anyhow::anyhow!("models: {e}"))?;
    }
    if let Some(v) = get("default_model") {
        c.default_model = (!v.is_empty()).then(|| v.to_string());
    }
    c.normalize();
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
}
