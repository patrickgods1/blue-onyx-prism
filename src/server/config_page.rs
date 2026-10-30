//! The config page and its writers (form, models, restart, log level).

use super::apply::{apply_config_locked, sync_config};
use super::devices::{DevicesSnapshot, devices_snapshot, force_cpu_note};
use super::pages::level_choices;
use super::resources::{FamilyChoice, family_choices};
use super::{AppState, blocking, live, render, wants_html};
use crate::backend::spec::{self, DeviceSpec};
use crate::config::{Config, LogLevel};
use crate::config_form::FORM_FIELDS;
use crate::sync::LockExt;
use askama::Template;
use axum::Json;
use axum::extract::rejection::{FormRejection, QueryRejection};
use axum::extract::{Form, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

/// Delay between answering `POST /config/restart` and stopping the server, so the reply gets out.
const RESTART_DELAY: Duration = Duration::from_millis(250);

// ---------------------------------------------------------------------------------------------
// Config page

/// Form values as strings, so a rejected submission can be shown again as typed (also the
/// `forms.server.view` of `GET /v1/config`, which the page compares its inputs with).
#[derive(serde::Serialize)]
pub(super) struct ConfigView {
    port: String,
    bind_address: String,
    allow_remote_admin: bool,
    request_timeout_secs: String,
    worker_queue_size: String,
    device: String,
    gpu_index: String,
    force_cpu: bool,
    cache_dir: String,
    download_dir: String,
    models_dir: String,
    onnxruntime_dir: String,
    openvino_dir: String,
    confidence_threshold: String,
    nms_iou: String,
    object_filter: String,
    log_level: String,
    log_timezone: String,
    log_path: String,
    save_image_path: String,
    save_ref_image: bool,
    save_image_max_age_days: String,
    save_image_max_mb: String,
    intra_threads: String,
    models_json: String,
    auto_download: bool,
    allow_large_downloads: bool,
    night_confidence_threshold: String,
    night_person_threshold: String,
    person_aspect_min: String,
    person_aspect_max: String,
    drop_person_on_plant: bool,
    static_suppression: bool,
}

impl ConfigView {
    pub(super) fn from_config(c: &Config) -> Self {
        let path = |p: &Option<PathBuf>| {
            p.as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        };
        let opt = |v: Option<f32>| {
            v.map(|x| crate::config::clean_f32(x).to_string())
                .unwrap_or_default()
        };
        Self {
            port: c.port.to_string(),
            bind_address: c.bind_address.to_string(),
            allow_remote_admin: c.allow_remote_admin,
            request_timeout_secs: c.request_timeout_secs.to_string(),
            worker_queue_size: c.worker_queue_size.to_string(),
            device: c.device.clone(),
            gpu_index: c.gpu_index.to_string(),
            force_cpu: c.force_cpu,
            cache_dir: c.cache_dir.clone(),
            download_dir: path(&c.download_dir),
            models_dir: c.models_dir.display().to_string(),
            onnxruntime_dir: path(&c.onnxruntime_dir),
            openvino_dir: path(&c.openvino_dir),
            confidence_threshold: crate::config::round_threshold_f64(c.confidence_threshold)
                .to_string(),
            nms_iou: crate::config::clean_f32(c.nms_iou).to_string(),
            object_filter: c.object_filter.join(", "),
            log_level: c.log_level.as_str().to_string(),
            log_timezone: c.log_timezone.to_string(),
            log_path: path(&c.log_path),
            save_image_path: path(&c.save_image_path),
            save_ref_image: c.save_ref_image,
            save_image_max_age_days: c.save_image_max_age_days.to_string(),
            save_image_max_mb: c.save_image_max_mb.to_string(),
            intra_threads: c.intra_threads.to_string(),
            models_json: serde_json::to_string_pretty(&c.models).unwrap_or_default(),
            auto_download: c.auto_download,
            allow_large_downloads: c.allow_large_downloads,
            night_confidence_threshold: opt(c.night_confidence_threshold),
            night_person_threshold: opt(crate::config_form::night_person_threshold(c)),
            person_aspect_min: opt(c.person_aspect.map(|[lo, _]| lo)),
            person_aspect_max: opt(c.person_aspect.map(|[_, hi]| hi)),
            drop_person_on_plant: crate::config_form::has_plant_rule(c),
            static_suppression: c.static_suppression.enabled,
        }
    }

    /// Replace the values with what the user submitted.
    fn overlay(&mut self, form: &HashMap<String, String>) {
        for &key in FORM_FIELDS {
            let v = form.get(key).cloned();
            match key {
                "force_cpu" if !form.contains_key("inference_settings") => {}
                "save_ref_image" if !form.contains_key("image_settings") => {}
                "force_cpu" => self.force_cpu = v.is_some(),
                "save_ref_image" => self.save_ref_image = v.is_some(),
                "auto_download" | "allow_large_downloads"
                    if !form.contains_key("download_settings") => {}
                "allow_remote_admin" if !form.contains_key("access_settings") => {}
                "allow_remote_admin" => self.allow_remote_admin = v.is_some(),
                "auto_download" => self.auto_download = v.is_some(),
                "drop_person_on_plant" | "static_suppression"
                    if !form.contains_key("night_settings") => {}
                "drop_person_on_plant" => self.drop_person_on_plant = v.is_some(),
                "static_suppression" => self.static_suppression = v.is_some(),
                "allow_large_downloads" => self.allow_large_downloads = v.is_some(),
                _ => {
                    let Some(v) = v else { continue };
                    let slot = match key {
                        "port" => &mut self.port,
                        "bind_address" => &mut self.bind_address,
                        "request_timeout_secs" => &mut self.request_timeout_secs,
                        "worker_queue_size" => &mut self.worker_queue_size,
                        "device" => &mut self.device,
                        "gpu_index" => &mut self.gpu_index,
                        "cache_dir" => &mut self.cache_dir,
                        "download_dir" => &mut self.download_dir,
                        "models_dir" => &mut self.models_dir,
                        "onnxruntime_dir" => &mut self.onnxruntime_dir,
                        "openvino_dir" => &mut self.openvino_dir,
                        "confidence_threshold" => &mut self.confidence_threshold,
                        "nms_iou" => &mut self.nms_iou,
                        "object_filter" => &mut self.object_filter,
                        "log_level" => &mut self.log_level,
                        "log_timezone" => &mut self.log_timezone,
                        "log_path" => &mut self.log_path,
                        "save_image_path" => &mut self.save_image_path,
                        "save_image_max_age_days" => &mut self.save_image_max_age_days,
                        "save_image_max_mb" => &mut self.save_image_max_mb,
                        "intra_threads" => &mut self.intra_threads,
                        "night_confidence_threshold" => &mut self.night_confidence_threshold,
                        "night_person_threshold" => &mut self.night_person_threshold,
                        "person_aspect_min" => &mut self.person_aspect_min,
                        "person_aspect_max" => &mut self.person_aspect_max,
                        _ => continue,
                    };
                    *slot = v;
                }
            }
        }
        if let Some(v) = form.get("models_json") {
            self.models_json = v.clone();
        }
    }
}

/// One row of the Models card.
struct ModelSelectRow {
    name: String,
    /// Normalized name (the row's merge keys are `model:<field>:<key>`).
    key: String,
    family: String,
    path: String,
    /// The last path component, shown in the table (the full path is its tooltip).
    file_name: String,
    /// The model file exists (resolved against the exe dir).
    exists: bool,
    enabled: bool,
    is_default: bool,
    /// Per-model Device select: "Global (...)" first, then every device option.
    devices: Vec<DeviceChoice>,
    /// The Threshold input: the profile for `threshold_key` ("" = none: the device uses
    /// the fallback); with no key, the model's own threshold ("" = global).
    threshold: String,
    /// The device the Threshold input is for (`threshold_key.<name>`; "" = not known).
    threshold_key: String,
    /// The input's placeholder: what a device without a profile uses ("0.38 model default").
    threshold_placeholder: String,
    /// The fallback value alone ("0.38") and its source ("model default" / "global").
    threshold_fallback: String,
    threshold_fallback_source: &'static str,
    /// The model's own threshold, the default for devices without a profile ("" = global).
    default_threshold: String,
    /// The model's confidence profiles: (device, "0.30").
    profiles: Vec<(String, String)>,
    /// The threshold the loaded worker uses ("" = not loaded).
    running_threshold: String,
    /// Night model select: "none" first, "same model" ([`NIGHT_SELF`]), then the other models
    /// it may name.
    night_models: Vec<DeviceChoice>,
    /// The model's `night_threshold` ("" = none), the field beside "same model".
    night_threshold: String,
    /// "same model" is chosen (the night threshold field shows).
    night_self: bool,
    /// The hidden `object_filter.<name>` field: `__global__`, empty (all) or a comma list.
    object_filter: String,
    /// What the Classes cell shows before the script adds the counts: "global", "all", "person, car".
    object_filter_text: String,
    /// What the running worker uses ("OpenVINO CPU · threshold 0.50", "not loaded", ...).
    running: String,
    running_class: &'static str,
    /// Why the model failed, under "Failed" on one truncated line (full text on open).
    running_detail: Option<String>,
    /// What a restart would change for this model (empty = nothing).
    pending: Vec<String>,
    /// Latest benchmark recommendation for this model.
    bench: Option<BenchHint>,
    /// Best confidence threshold of the latest benchmark on `threshold_key`'s device (with no
    /// key: the overall best): (value "0.35", objective "f1", "in use" / "set" when the device
    /// already uses it).
    best_threshold: Option<(String, String, Option<&'static str>)>,
    /// The file is missing (or outdated) and made by exporting this resource (`model:yolo26s`).
    needs_export: Option<String>,
}

/// "benchmark: ort:coreml" link on the Models card.
struct BenchHint {
    text: String,
    title: String,
    /// `/benchmark#m-<name>` (the Benchmark page's id for the model's card).
    href: String,
}

/// JavaScript `encodeURIComponent` (the Benchmark page builds its card ids with it).
pub(super) fn encode_uri_component(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z'
            | b'a'..=b'z'
            | b'0'..=b'9'
            | b'-'
            | b'_'
            | b'.'
            | b'!'
            | b'~'
            | b'*'
            | b'\''
            | b'('
            | b')' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// Display text of the global device: "auto (ort:coreml)", "openvino:gpu.1", "Force CPU".
pub(super) fn global_device_label(c: &Config, snap: &DevicesSnapshot) -> String {
    if c.force_cpu {
        return "Force CPU".to_string();
    }
    match spec::parse(&c.device) {
        Ok(DeviceSpec::Auto) => match snap.selection.auto_pick() {
            Some(o) => format!("auto \u{2192} {}", o.spec),
            None => "auto".to_string(),
        },
        Ok(d) => d.with_default_index(c.gpu_index).to_string(),
        Err(_) => c.device.trim().to_string(),
    }
}

/// Options of the Night model select of `models[me]`: "none", then every other model that has
/// no night model of its own (night models do not chain; a disabled one is marked: the switch
/// falls back to the model itself while it is off). A model this one already names is always
/// listed (as "missing" when it is not configured).
/// The Night model select's value for "the same model with its own night threshold"
/// (`night_threshold`, no `night_model`); a model of that name cannot be chosen as a night
/// model on the card (the JSON editor still can).
pub(super) const NIGHT_SELF: &str = "@self";

/// The value of the Models card's `object_filter.<name>` field for "no per-model filter: use the
/// global Object filter" (`ModelConfig.object_filter` = `None`); empty = every class.
pub(super) const OBJECT_FILTER_GLOBAL: &str = "__global__";

/// `(field value, summary text)` of a model's Classes cell.
fn object_filter_cell(filter: Option<&[String]>) -> (String, String) {
    match filter {
        None => (OBJECT_FILTER_GLOBAL.to_string(), "global".to_string()),
        Some([]) => (String::new(), "all".to_string()),
        Some(v) => (v.join(","), v.join(", ")),
    }
}

fn night_model_choices(models: &[crate::config::ModelConfig], me: usize) -> Vec<DeviceChoice> {
    use crate::registry::normalize_name;
    let Some(m) = models.get(me) else {
        return Vec::new();
    };
    let current = m.night_model.as_deref().map(normalize_name);
    let same = current.is_none() && m.night_threshold.is_some();
    let mut v = vec![
        DeviceChoice {
            value: String::new(),
            label: "none".to_string(),
            selected: current.is_none() && !same,
            disabled: false,
            ..Default::default()
        },
        DeviceChoice {
            value: NIGHT_SELF.to_string(),
            label: "same model, night threshold".to_string(),
            selected: same,
            disabled: false,
            ..Default::default()
        },
    ];
    let mut found = false;
    for (i, o) in models.iter().enumerate() {
        let name = o.effective_name();
        let selected = current.as_deref() == Some(normalize_name(&name).as_str());
        if i == me || (o.night_model.is_some() && !selected) {
            continue;
        }
        found |= selected;
        v.push(DeviceChoice {
            label: if o.enabled {
                name.clone()
            } else {
                format!("{name} (disabled)")
            },
            value: name,
            selected,
            disabled: false,
            ..Default::default()
        });
    }
    if let (Some(n), false) = (m.night_model.as_deref(), found) {
        v.push(DeviceChoice {
            value: n.to_string(),
            label: format!("{n} (missing)"),
            selected: true,
            disabled: false,
            ..Default::default()
        });
    }
    v
}

/// Options of a model's Device select. `current` None = follows the global device.
fn model_device_choices(
    snap: &DevicesSnapshot,
    current: Option<&str>,
    global: &str,
) -> Vec<DeviceChoice> {
    let mut v = device_choices(snap, current.unwrap_or("auto"), "0");
    if current.is_none() {
        for c in &mut v {
            c.selected = false;
        }
    }
    v.insert(
        0,
        DeviceChoice {
            value: String::new(),
            label: format!("Global ({global})"),
            selected: current.is_none(),
            disabled: false,
            ..Default::default()
        },
    );
    v
}

/// Fills in the per-model facts of a model's Device options (see [`DeviceChoice`]): the device
/// each option resolves to for model `m` (`sel` = its selection; the selected option resolves to
/// `running`, the device the model loaded on, when it is loaded), the confidence profile and the
/// benchmark facts there, and labels with the benchmark's speed and best threshold
/// ("ort:coreml — CoreML · 24 ms · best conf 0.30").
fn annotate_model_devices(
    choices: &mut [DeviceChoice],
    c: &Config,
    m: &crate::config::ModelConfig,
    sel: &crate::backend::select::Selection,
    running: Option<&str>,
    facts: Option<&live::BenchFacts>,
) {
    let global_auto = !c.force_cpu && matches!(spec::parse(&c.device), Ok(DeviceSpec::Auto));
    for o in choices.iter_mut() {
        let mut probe = m.clone();
        probe.device = (!o.value.is_empty()).then(|| o.value.clone());
        let resolved = match running {
            Some(r) if o.selected => Some(r.to_string()),
            _ => crate::benchmark::configured_device(c, &probe, sel),
        };
        let key = live::device_key(resolved.as_deref());
        let cell = live::ThresholdCell::new(m, c.confidence_threshold, &key);
        o.thr = cell.profile.map(live::fmt_thr).unwrap_or_default();
        o.src = cell.source_text();
        let bench = facts.and_then(|f| (!key.is_empty()).then(|| f.best_on(&key)).flatten());
        o.bench = bench.map(live::fmt_thr).unwrap_or_default();
        if !key.is_empty() {
            if o.value == "auto" {
                o.label = format!("Auto \u{2014} currently: {key}");
            } else if o.value.is_empty() && global_auto {
                o.label = format!("Global (auto \u{2192} {key})");
            }
            let p50 = facts
                .and_then(|f| f.device_fact(&key))
                .and_then(|d| d.p50_ms);
            if let Some(ms) = p50 {
                if ms >= 10.0 {
                    o.label.push_str(&format!(" \u{b7} {ms:.0} ms"));
                } else {
                    o.label.push_str(&format!(" \u{b7} {ms:.1} ms"));
                }
            }
            if let Some(t) = bench {
                o.label
                    .push_str(&format!(" \u{b7} best conf {}", live::fmt_thr(t)));
            }
        }
        o.key = key;
    }
}

fn model_choices(
    c: &Config,
    snap: &DevicesSnapshot,
    state: &AppState,
    bench: Option<&live::BenchSummary>,
) -> Vec<ModelSelectRow> {
    let default = c.effective_default_index();
    let global = global_device_label(c, snap);
    let pending = crate::config_merge::pending(&state.running(), c);
    let thr = crate::config::round_threshold_f64;
    c.models
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let name = m.effective_name();
            let key = crate::registry::normalize_name(&name);
            let facts = bench.and_then(|b| b.models.get(&key));
            let run = live::running_view(state, &name);
            let sel = snap.model_selection(state, c, m);
            let configured = crate::benchmark::configured_device(c, m, &sel);
            let hint = facts.map(|r| {
                let text = match &r.recommended {
                    Some(d) if configured.as_deref() == Some(d.as_str()) => {
                        if run.device.as_deref() == Some(d.as_str()) {
                            format!("benchmark: {d} (in use)")
                        } else {
                            format!("benchmark: {d} (configured; restart to use)")
                        }
                    }
                    Some(d) => format!("benchmark recommends {d}"),
                    None => "benchmark: no recommendation".to_string(),
                };
                BenchHint {
                    text,
                    title: r.recommendation_now(
                        m,
                        c.confidence_threshold,
                        run.threshold,
                        configured.clone(),
                    ),
                    href: format!("/benchmark#m-{}", encode_uri_component(&r.model)),
                }
            });
            let mut devices = model_device_choices(snap, m.device.as_deref(), &global);
            annotate_model_devices(&mut devices, c, m, &sel, run.device.as_deref(), facts);
            // The cell follows the selected option's device.
            let threshold_key = devices
                .iter()
                .find(|d| d.selected)
                .map(|d| d.key.clone())
                .unwrap_or_default();
            let cell = live::ThresholdCell::new(m, c.confidence_threshold, &threshold_key);
            let effective = thr(cell.effective());
            let best_threshold = facts.and_then(|f| {
                let t = if threshold_key.is_empty() {
                    f.best_threshold?
                } else {
                    f.best_on(&threshold_key)?
                };
                let t = thr(t);
                let state = if t != effective {
                    None
                } else if run.threshold == Some(t) {
                    Some("in use")
                } else {
                    Some("set")
                };
                Some((format!("{t:.2}"), f.objective.clone(), state))
            });
            let own = m
                .confidence_threshold
                .map(live::fmt_thr)
                .unwrap_or_default();
            let threshold = if threshold_key.is_empty() {
                own.clone()
            } else {
                cell.profile.map(live::fmt_thr).unwrap_or_default()
            };
            let exists = c.data_path(&m.path).is_file();
            // Missing, or from another version (the old RF-DETR large download).
            let needs_export = crate::resources::export::pending_export(&c.data_path(&m.path))
                .map(|r| r.id.to_string());
            ModelSelectRow {
                devices,
                night_models: night_model_choices(&c.models, i),
                night_threshold: m.night_threshold.map(live::fmt_thr).unwrap_or_default(),
                night_self: m.night_model.is_none() && m.night_threshold.is_some(),
                object_filter: object_filter_cell(m.object_filter.as_deref()).0,
                object_filter_text: object_filter_cell(m.object_filter.as_deref()).1,
                key,
                family: m.family.to_string(),
                path: m.path.display().to_string(),
                file_name: m
                    .path
                    .file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                    .unwrap_or_else(|| m.path.display().to_string()),
                exists,
                needs_export,
                enabled: m.enabled,
                is_default: default == Some(i),
                threshold,
                threshold_placeholder: cell.placeholder(),
                threshold_fallback: live::fmt_thr(cell.fallback),
                threshold_fallback_source: cell.fallback_source,
                threshold_key,
                default_threshold: own,
                profiles: m
                    .device_thresholds
                    .iter()
                    .map(|(d, t)| (d.clone(), live::fmt_thr(*t)))
                    .collect(),
                running_threshold: run.threshold.map(|t| format!("{t:.2}")).unwrap_or_default(),
                running_detail: run.detail.clone().filter(|_| run.state_class == "fail"),
                running: run.text,
                running_class: run.state_class,
                pending: pending.for_model(&name).to_vec(),
                bench: hint,
                best_threshold,
                name,
            }
        })
        .collect()
}

pub(super) struct LevelChoice {
    pub(super) value: &'static str,
    pub(super) selected: bool,
}

/// An option of the "Log timestamp time zone" dropdown.
struct TimezoneChoice {
    value: String,
    label: String,
    selected: bool,
}

/// Time zones offered by the dropdown: `(log_timezone value, label)`.
const LOG_TIMEZONES: &[(&str, &str)] = &[
    ("utc", "UTC"),
    ("local", "Local (this computer's time zone)"),
    ("America/Los_Angeles", "Pacific (Los Angeles)"),
    ("America/Denver", "Mountain (Denver)"),
    ("America/Phoenix", "Arizona (Phoenix, no DST)"),
    ("America/Chicago", "Central (Chicago)"),
    ("America/New_York", "Eastern (New York)"),
    ("America/Halifax", "Atlantic (Halifax)"),
    ("America/Anchorage", "Alaska (Anchorage)"),
    ("Pacific/Honolulu", "Hawaii (Honolulu)"),
    ("America/Sao_Paulo", "Brazil (Sao Paulo)"),
    ("Europe/London", "UK (London)"),
    ("Europe/Berlin", "Central Europe (Berlin)"),
    ("Europe/Helsinki", "Eastern Europe (Helsinki)"),
    ("Asia/Kolkata", "India (Kolkata)"),
    ("Asia/Shanghai", "China (Shanghai)"),
    ("Asia/Tokyo", "Japan (Tokyo)"),
    ("Australia/Sydney", "Australia East (Sydney)"),
    ("Pacific/Auckland", "New Zealand (Auckland)"),
];

/// The dropdown's options with `current` selected. A value not in [`LOG_TIMEZONES`] (an offset or
/// another IANA name set in the config file, or a rejected submission) is kept as a first
/// "Custom" option so saving the page does not change it.
fn timezone_choices(current: &str) -> Vec<TimezoneChoice> {
    let current = current.trim();
    let current = if current.is_empty() { "utc" } else { current };
    let mut out: Vec<TimezoneChoice> = LOG_TIMEZONES
        .iter()
        .map(|&(value, label)| TimezoneChoice {
            value: value.to_string(),
            label: label.to_string(),
            selected: value.eq_ignore_ascii_case(current),
        })
        .collect();
    if !out.iter().any(|t| t.selected) {
        out.insert(
            0,
            TimezoneChoice {
                value: current.to_string(),
                label: format!("Custom: {current}"),
                selected: true,
            },
        );
    }
    out
}

#[derive(Template)]
#[template(path = "config.html")]
pub(super) struct ConfigTemplate {
    nav: &'static str,
    version: &'static str,
    config_path: String,
    c: ConfigView,
    models: Vec<ModelSelectRow>,
    log_levels: Vec<LevelChoice>,
    log_timezones: Vec<TimezoneChoice>,
    device_choices: Vec<DeviceChoice>,
    message: Option<String>,
    error: Option<String>,
    /// Resources card, grouped (Runtimes, GPU libraries, Models by family).
    pub(super) resource_groups: Vec<crate::resources::status::ResourceGroup>,
    /// Model files in `models_dir` that no config entry uses.
    pub(super) local_models: Vec<crate::resources::status::LocalModel>,
    /// Families of the "Add a model by path" form.
    family_choices: Vec<FamilyChoice>,
    /// Resolved data root (relative model paths are under it).
    data_root: String,
    /// The program's directory (the default data root; relative runtime dirs are under it).
    exe_dir: String,
    /// Placeholder of the model path field: a file under `models_dir` with this OS's separators
    /// (where Browse starts).
    model_path_example: String,
    resources_enabled: bool,
    /// Resolved `models_dir` (where exported models go).
    models_dir: String,
    /// Where the model export toolchain lives (`<data root>/tools`).
    tools_dir: String,
    /// License notices of the export recipes, shown with the export group.
    export_notice: String,
    force_cpu_note: Option<String>,
    /// One line about the `benchmark` config section and the last results.
    bench_summary: String,
    /// Config revision the page was rendered from.
    revision: u64,
    /// Global confidence threshold (placeholder of the per-model inputs).
    global_threshold: String,
    /// What a restart would apply (all reasons).
    pending: Vec<String>,
    /// Where `message` / `error` are shown: "" (top), "models", "server", "resources" or
    /// "resources-row" (in the row of resource `at_row`). The forms post to their section's
    /// `#fragment`, so a no-JS answer lands there.
    pub(super) at: &'static str,
    pub(super) at_row: String,
}

/// When a change of a global config field takes effect, as the Server tab's badges say it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ApplyWhen {
    /// Live: [`crate::config_merge::live_global`].
    Now,
    /// A server restart (a new registry generation) applies it.
    Restart,
    /// Only a restart of the process applies it (the listener, the log file).
    Process,
}

impl ApplyWhen {
    /// The global config fields only a process restart applies; the same set
    /// [`crate::config_merge::pending`] reports as "needs a process restart" (a test checks it).
    const PROCESS: &[&str] = &["port", "bind_address", "log_path", "log_format"];

    pub(super) fn of(key: &str) -> Self {
        if crate::config_merge::live_global(key) {
            Self::Now
        } else if Self::PROCESS.contains(&key) {
            Self::Process
        } else {
            Self::Restart
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Self::Now => "<span class=\"tag now\">applies now</span>",
            Self::Restart => "<span class=\"tag restart\">needs restart</span>",
            Self::Process => "<span class=\"tag restart\">process restart</span>",
        }
    }
}

/// The badge of comma-separated global config `keys` when they all take effect alike, else
/// "" (a group of mixed fields shows a badge on each field instead).
pub(super) fn when_badge(keys: &str) -> &'static str {
    let mut it = keys.split(',').map(str::trim).filter(|k| !k.is_empty());
    let Some(first) = it.next().map(ApplyWhen::of) else {
        return "";
    };
    if it.all(|k| ApplyWhen::of(k) == first) {
        first.tag()
    } else {
        ""
    }
}

impl ConfigTemplate {
    /// Server tab badge of a group or a field (static HTML; see [`when_badge`]).
    fn badge(&self, keys: &str) -> &'static str {
        when_badge(keys)
    }
}

/// One `<option>` of the Device select.
#[derive(Debug, Clone, Default)]
struct DeviceChoice {
    value: String,
    label: String,
    selected: bool,
    disabled: bool,
    /// Per-model Device select only (else empty): the canonical device the option resolves to
    /// for this model (`data-key`; "" = none), its confidence profile (`data-thr`, "" = none),
    /// where the threshold on it comes from (`data-src`: "profile", "model default 0.38",
    /// "global 0.50") and the benchmark's best threshold on it (`data-bench`, "" = none).
    key: String,
    thr: String,
    src: String,
    bench: String,
}

/// Options for the Device select: `auto` first, then every runnable option, then the ones that
/// cannot run (disabled, with the reason). The current value is always present; when it is not
/// runnable it stays enabled so saving the form does not silently change it.
fn device_choices(snap: &DevicesSnapshot, current: &str, gpu_index: &str) -> Vec<DeviceChoice> {
    let parsed = spec::parse(current).ok();
    let gpu_index = gpu_index.trim().parse::<u32>().unwrap_or(0);
    let is_current = |d: &crate::backend::spec::Device| match parsed {
        Some(DeviceSpec::Device(p)) => p == *d || p.with_default_index(gpu_index) == *d,
        _ => false,
    };
    let mut out = vec![DeviceChoice {
        value: "auto".to_string(),
        label: format!("Auto \u{2014} currently: {}", snap.auto_pick_label()),
        selected: matches!(parsed, Some(DeviceSpec::Auto)),
        disabled: false,
        ..Default::default()
    }];
    let (ok, bad): (Vec<_>, Vec<_>) = snap.selection.options.iter().partition(|o| o.runnable);
    for o in ok {
        out.push(DeviceChoice {
            value: o.spec.to_string(),
            label: format!("{} \u{2014} {}", o.spec, o.label),
            selected: is_current(&o.spec),
            disabled: false,
            ..Default::default()
        });
    }
    // Downloadable options are selectable (the download itself is started from the Resources
    // card / `fetch`; until it is installed the model falls back as usual).
    let (dl, bad): (Vec<_>, Vec<_>) = bad.into_iter().partition(|o| o.is_downloadable());
    for o in dl {
        let summary = o.download.as_ref().map_or("", |d| d.summary.as_str());
        out.push(DeviceChoice {
            value: o.spec.to_string(),
            label: format!("{} \u{2014} {summary}", o.spec),
            selected: is_current(&o.spec),
            disabled: false,
            ..Default::default()
        });
    }
    for o in bad {
        let selected = is_current(&o.spec);
        out.push(DeviceChoice {
            value: o.spec.to_string(),
            label: format!(
                "{} \u{2014} not available: {}",
                o.spec,
                o.reason.as_deref().unwrap_or("not runnable")
            ),
            selected,
            disabled: !selected,
            ..Default::default()
        });
    }
    if !out.iter().any(|c| c.selected) {
        out.push(DeviceChoice {
            value: current.trim().to_string(),
            label: format!("{} \u{2014} not a known option", current.trim()),
            selected: true,
            disabled: false,
            ..Default::default()
        });
    }
    out
}

/// The "About exports" notice, derived from the catalog so a new recipe or a restrictive model
/// license cannot be left out: the notice of every [`ExportRecipe`] an exportable model uses
/// (catalog order), then each non-commercial / personal-use license with the models under it.
///
/// [`ExportRecipe`]: crate::resources::catalog::ExportRecipe
pub(super) fn export_notice() -> String {
    use crate::resources::catalog;
    let mut recipes: Vec<catalog::ExportRecipe> = Vec::new();
    let mut restricted: Vec<(&'static str, Vec<&'static str>)> = Vec::new();
    for res in catalog::all() {
        let Some(recipe) = catalog::export_recipe(res) else {
            continue;
        };
        if !recipes.contains(&recipe) {
            recipes.push(recipe);
        }
        let license = res.license.to_ascii_lowercase();
        if license.contains("non-commercial") || license.contains("personal use") {
            let name = res.model_name().unwrap_or(res.title);
            match restricted.iter_mut().find(|(l, _)| *l == res.license) {
                Some((_, names)) => names.push(name),
                None => restricted.push((res.license, vec![name])),
            }
        }
    }
    let mut parts: Vec<String> = recipes.iter().map(|r| r.notice().to_string()).collect();
    for (license, names) in restricted {
        parts.push(format!("{}: {license}.", names.join(", ")));
    }
    parts.join(" ")
}

pub(super) fn config_template(
    state: &AppState,
    message: Option<String>,
    error: Option<String>,
    submitted: Option<&HashMap<String, String>>,
) -> ConfigTemplate {
    sync_config(state);
    let snap = devices_snapshot(state);
    let bench = live::bench_summary(&crate::benchmark::results_path(&state.config_path));
    let (revision, cfg) = state.config.snapshot();
    let mut c = ConfigView::from_config(&cfg);
    let models = model_choices(&cfg, &snap, state, (*bench).as_ref());
    if let Some(form) = submitted {
        c.overlay(form);
    }
    let log_levels = level_choices(&c.log_level);
    let log_timezones = timezone_choices(&c.log_timezone);
    let device_choices = device_choices(&snap, &c.device, &c.gpu_index);
    let resource_groups = crate::resources::status::grouped(
        crate::resources::status::rows(state.resources.as_ref(), &cfg),
        crate::resources::status::export_group_note(&cfg),
    );
    let local_models = crate::resources::status::local_models(state.resources.as_ref(), &cfg);
    let models_path = cfg.data_path(&cfg.models_dir);
    let model_path_example = models_path
        .join("my-rfdetr")
        .join("inference_model.onnx")
        .display()
        .to_string();
    let models_dir = models_path.display().to_string();
    let force_cpu_note = force_cpu_note(&cfg, &snap);
    let tools_dir = cfg
        .data_root()
        .join(crate::resources::export::TOOLS_DIR)
        .display()
        .to_string();
    let bench_summary = {
        let b = &cfg.benchmark;
        let datasets: Vec<String> = b.datasets.iter().map(|d| d.label()).collect();
        let plural = |n: usize, one: &str, many: &str| {
            if n == 1 {
                format!("1 {one}")
            } else {
                format!("{n} {many}")
            }
        };
        let last = match (*bench).as_ref() {
            Some(r) => {
                let when = chrono::DateTime::parse_from_rfc3339(&r.timestamp)
                    .map(|t| t.format("%Y-%m-%d %H:%M UTC").to_string())
                    .unwrap_or_else(|_| r.timestamp.clone());
                format!(
                    "Last benchmark: {}, {when}.",
                    plural(r.count, "model", "models")
                )
            }
            None => "No benchmark results yet.".to_string(),
        };
        let images = if b.max_images_per_dataset == 0 {
            "all images".to_string()
        } else {
            plural(b.max_images_per_dataset, "image", "images")
        };
        format!(
            "{last} Defaults: {}; {images} per dataset; {} per image; accuracy is {:.0}% of the grade.",
            if datasets.is_empty() {
                "no datasets".to_string()
            } else {
                datasets.join(", ")
            },
            plural(b.repeat_per_image, "timed run", "timed runs"),
            b.weights.accuracy_share() * 100.0
        )
    };
    let pending = {
        let p = crate::config_merge::pending(&state.running(), &cfg);
        let mut all = p.global.clone();
        for m in &cfg.models {
            for r in p.for_model(&m.effective_name()) {
                all.push(format!("{}: {r}", m.effective_name()));
            }
        }
        all.extend(p.process.iter().cloned());
        all
    };
    ConfigTemplate {
        bench_summary,
        resource_groups,
        local_models,
        family_choices: family_choices(),
        data_root: cfg.data_root().display().to_string(),
        exe_dir: crate::exe_dir().display().to_string(),
        model_path_example,
        models_dir,
        tools_dir,
        export_notice: export_notice(),
        force_cpu_note,
        resources_enabled: state.resources.is_some(),
        nav: "config",
        version: crate::VERSION,
        config_path: state.config_path.display().to_string(),
        c,
        models,
        log_levels,
        log_timezones,
        device_choices,
        message,
        error,
        revision,
        global_threshold: format!(
            "{:.2}",
            crate::config::round_threshold_f64(cfg.confidence_threshold)
        ),
        pending,
        at: "",
        at_row: String::new(),
    }
}

/// [`config_template`] with the message shown in section `at` (see [`ConfigTemplate::at`]).
fn config_template_at(
    state: &AppState,
    message: Option<String>,
    error: Option<String>,
    submitted: Option<&HashMap<String, String>>,
    at: &'static str,
) -> ConfigTemplate {
    let mut t = config_template(state, message, error, submitted);
    t.at = at;
    t
}

pub(super) async fn config_page(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || render(&config_template(&state, None, None, None))).await
}

/// Apply `edit` to the current config through the config store (one writer for everything:
/// it bumps the revision and logs the change under `source`), write the file and apply what
/// can be applied now ([`apply_config`](super::apply::apply_config): live settings, per-model reloads, the log level).
/// Returns the message for the page ("Applied now: ...", "reloading x on y", "restart to
/// apply ...").
pub(super) fn save_config(
    state: &AppState,
    source: &str,
    edit: impl FnOnce(&mut Config) -> anyhow::Result<()>,
) -> anyhow::Result<String> {
    // Write and apply under the apply lock: another applier (a page's config sync, the runner
    // after a restart) could otherwise apply this edit in between, and the message below would
    // miss what it reloaded ("Saved." without "Loading x on y; stopping z.").
    let (up, applied) = {
        let held = state.apply_lock.lock_recover();
        let up = state.config.update(source, edit)?;
        // Also applies a change picked up from the file before the edit.
        (up, apply_config_locked(state, &held))
    };
    if !up.changed {
        return Ok(format!(
            "No changes: {} already has these values.",
            state.config_path.display()
        ));
    }
    let level = state.config_read().log_level;
    let level = (level != up.before.log_level).then_some(level);
    info!(path = %state.config_path.display(), revision = up.revision, source, "config saved from the web UI");
    let mut msg = format!("Saved to {}.", state.config_path.display());
    let live: Vec<String> = up
        .record
        .as_ref()
        .map(|r| {
            r.live
                .iter()
                .filter(|l| *l != "Log level")
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    if !live.is_empty() {
        msg.push_str(&format!(" Applied now: {}.", live.join(", ")));
    }
    if let Some(text) = applied.describe() {
        msg.push(' ');
        msg.push_str(&text);
    }
    let pending = crate::config_merge::pending(&state.running(), &state.config_read());
    if pending.restart_needed() {
        msg.push_str(" Restart the server to apply the other changes.");
    }
    let tz = state.config_read().log_timezone;
    if tz != up.before.log_timezone
        && let Some(h) = &state.log_reload
    {
        h.set_timezone(tz);
    }
    if let Some(level) = level {
        match apply_log_level(state, level) {
            Ok(()) => msg.push_str(&format!(
                " The log level ({}) was applied immediately.",
                level.as_str()
            )),
            Err(e) => msg.push_str(&format!(" Log level not applied: {e:#}.")),
        }
    }
    Ok(msg)
}

/// Validate and save the form, as submitted (last write wins; the change log records what
/// changed). Changes apply at once where they can ([`apply_config`](super::apply::apply_config): live settings, per-model
/// reloads, the log level); runtime-level settings on restart. `default_model` and the
/// `enabled` flags are only changed when the form carries them (the page's main form does
/// not; the Models card posts to `/config/models`). Clients that send
/// `Accept: application/json` get JSON.
pub(super) async fn config_submit(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    blocking(move || {
        let json = live::wants_json(&headers);
        let pairs = match form {
            Ok(Form(f)) => f,
            Err(e) => {
                let msg = format!(
                    "Not saved: the form submission could not be read ({e}); the page is \
                     probably out of date, so reload it and save again."
                );
                if json {
                    return live::error_json(StatusCode::BAD_REQUEST, msg);
                }
                return render(&config_template(&state, None, Some(msg), None));
            }
        };
        let form: HashMap<String, String> = pairs.into_iter().collect();
        let edit = |c: &mut Config| {
            let before = c.clone();
            live::server_form_edit(c, &form)?;
            check_new_model_paths(&before, c, super::browse::allowed_or_missing_inside_roots)
        };
        match save_config(&state, "config page", edit) {
            Ok(msg) => {
                // A device change to an option that needs a download was queued by
                // `apply_config`: the model serves on what it can meanwhile; an installed runtime
                // starts a new generation (runtime libraries load once per generation).
                if json {
                    return live::saved_json(&state, msg, false);
                }
                render(&config_template_at(&state, Some(msg), None, None, "server"))
            }
            Err(e) => {
                warn!("rejected config form: {e:#}");
                let msg = format!("Not saved: {e:#}. Correct that value and save again.");
                if json {
                    return live::error_json(StatusCode::BAD_REQUEST, msg);
                }
                render(&config_template_at(
                    &state,
                    None,
                    Some(msg),
                    Some(&form),
                    "server",
                ))
            }
        }
    })
    .await
}

/// Model files typed into the models JSON editor: every `path` / `classes` that is new or
/// changed against `before` ([`crate::config_form::new_model_paths`]) must pass `allowed`
/// (the server's [`super::browse::allowed_or_missing_inside_roots`]: inside the browsable
/// folders, even when not downloaded yet, or covered by a recent native pick), like the
/// "Add a model by path" form. It is checked against the roots of the saved config and of the
/// edited one (a `download_dir` changed in the same post), and resolved under the edited
/// config's data root. Unchanged entries are never re-checked.
fn check_new_model_paths<E>(
    before: &Config,
    after: &Config,
    mut allowed: impl FnMut(&Config, &str) -> Result<PathBuf, E>,
) -> anyhow::Result<()> {
    for n in crate::config_form::new_model_paths(before, after) {
        let full = after.data_path(&n.path).display().to_string();
        if allowed(before, &full).is_err() && allowed(after, &full).is_err() {
            anyhow::bail!(
                "models: the {} of model '{}' ({}) is outside the folders this page may use \
                 (the data root {}, the models folder, the download folder and your home \
                 folder); choose the file with Browse... under Add a model by path (or move \
                 it into the models folder) and save again",
                n.field,
                n.model,
                n.path.display(),
                after.data_root().display()
            );
        }
    }
    Ok(())
}

/// The Models card: `enabled` (repeated, one per checked model), `default_model` (radio),
/// `device.<name>` / `threshold.<name>` ("" = global) and `action` (`save` or `restart`). Saves like `config_submit` (a changed model reloads on its
/// own: "reloading <model> on <device>"); `restart` then restarts like `POST /config/restart`.
pub(super) async fn config_models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    blocking(move || {
        let json = live::wants_json(&headers);
        let form = match form {
            Ok(Form(f)) => f,
            Err(e) => {
                let msg = format!(
                    "Not saved: the form submission could not be read ({e}); the page is \
                     probably out of date, so reload it and save again."
                );
                if json {
                    return live::error_json(StatusCode::BAD_REQUEST, msg);
                }
                return render(&config_template(&state, None, Some(msg), None));
            }
        };
        let restart = form.iter().any(|(k, a)| k == "action" && a == "restart");
        let result = save_config(&state, "config page (Models card)", |c| {
            live::models_card_edit(c, &form)
        });
        match result {
            Ok(msg) if restart => {
                if json {
                    schedule_restart(&state);
                    return live::saved_json(&state, msg, true);
                }
                restart_response(&state, "models")
            }
            Ok(msg) if json => live::saved_json(&state, msg, false),
            Ok(msg) => render(&config_template_at(&state, Some(msg), None, None, "models")),
            Err(e) => {
                warn!("rejected models selection: {e:#}");
                let msg = format!("Models not saved: {e:#}. Correct that entry and save again.");
                if json {
                    return live::error_json(StatusCode::BAD_REQUEST, msg);
                }
                render(&config_template_at(&state, None, Some(msg), None, "models"))
            }
        }
    })
    .await
}

/// `POST /config/models/remove` (`name`): the Models card's Remove. Drops the config entry of a
/// disabled model, saves like the other writers, then deletes its files when they are ours
/// (inside the models folder, not a catalog model, not used by another entry).
/// 404 for an unknown name, 409 for an enabled model. JSON unless the client asks for HTML
/// (then the config page at the Models card).
pub(super) async fn config_models_remove(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    blocking(move || {
        use crate::resources::actions::{ActionError, remove_config_entry};
        let html = wants_html(&headers);
        let name = form
            .ok()
            .and_then(|Form(f)| f.into_iter().find(|(k, _)| k == "name").map(|(_, v)| v))
            .map(|v| v.trim().to_string())
            .unwrap_or_default();
        let result = if name.is_empty() {
            Err(ActionError::BadRequest(
                "no model name given (field `name`); reload the page and try again".into(),
            ))
        } else {
            let mut outcome = None;
            save_config(&state, "config page (remove model entry)", |c| {
                outcome = Some(remove_config_entry(c, &name).map_err(|e| anyhow::anyhow!(e))?);
                Ok(())
            })
            .map(|msg| {
                // Files go only once the entry is saved out of the config.
                let done = outcome
                    .as_ref()
                    .map(|o| o.delete_files())
                    .unwrap_or_default();
                format!("{done}. {msg}")
            })
            .map_err(|e| match e.downcast::<ActionError>() {
                Ok(a) => a,
                Err(e) => ActionError::Failed(format!("{e:#}")),
            })
        };
        match result {
            Ok(msg) if html => render(&config_template_at(&state, Some(msg), None, None, "models")),
            Ok(msg) => live::saved_json(&state, msg, false),
            Err(e) => {
                warn!("model entry not removed: {e}");
                let msg = format!("Not removed: {e}.");
                if html {
                    return render(&config_template_at(&state, None, Some(msg), None, "models"));
                }
                live::error_json(
                    StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    msg,
                )
            }
        }
    })
    .await
}

fn apply_log_level(state: &AppState, level: LogLevel) -> anyhow::Result<()> {
    match &state.log_reload {
        Some(h) => h.set_level(level),
        None => anyhow::bail!(
            "runtime log level control is not available in this process (logging was set up \
             elsewhere); set log_level in the config file and restart"
        ),
    }
}

#[derive(Template)]
#[template(path = "message.html")]
pub(super) struct MessageTemplate {
    pub(super) nav: &'static str,
    pub(super) version: &'static str,
    pub(super) heading: String,
    pub(super) message: String,
    pub(super) refresh_secs: u32,
    pub(super) refresh_url: String,
    /// The server is restarting: poll until it answers, then go to `refresh_url` (the meta
    /// refresh stays as the no-JS fallback).
    pub(super) restarting: bool,
    /// Config-store generation when the restart was requested; the page waits until
    /// `/v1/config` reports another one (the listener stays up across a restart).
    pub(super) generation: u64,
}

/// Ends this generation: the run loop (`runner`) reloads the config file, starts the next
/// generation and swaps it in; the listener stays up and the old workers stop once their
/// requests have drained.
/// `back` (query): the Config page section to return to (`models`, `server`, `restart`).
pub(super) async fn config_restart(
    State(state): State<Arc<AppState>>,
    query: Result<Query<HashMap<String, String>>, QueryRejection>,
) -> Response {
    let back = query
        .ok()
        .and_then(|Query(q)| q.get("back").cloned())
        .unwrap_or_default();
    restart_response(&state, &back)
}

/// Cancel this generation's restart token shortly (so the reply gets out) and render the
/// "Restarting" page, which returns to the Config page's section `back` (when known).
fn restart_response(state: &AppState, back: &str) -> Response {
    let mut url = "/config".to_string();
    if matches!(back, "models" | "server" | "restart" | "resources") {
        url.push_str(&format!("#at-{back}"));
        if back == "resources" {
            url = "/config#resources".into();
        }
    }
    restart_response_to(
        state,
        "The server is reloading its config and recompiling the enabled models. This page \
         reloads as soon as the server answers again."
            .into(),
        url,
    )
}

/// Cancel this generation's restart token after [`RESTART_DELAY`] (so the reply gets out).
pub(super) fn schedule_restart(state: &AppState) {
    info!("restart requested from the web UI");
    let token = state.restart.clone();
    tokio::spawn(async move {
        tokio::time::sleep(RESTART_DELAY).await;
        token.cancel();
    });
}

/// Restart, showing `message`, then go to `url`.
fn restart_response_to(state: &AppState, message: String, url: String) -> Response {
    schedule_restart(state);
    render(&MessageTemplate {
        nav: "config",
        version: crate::VERSION,
        heading: "Restarting\u{2026}".into(),
        message,
        refresh_secs: 5,
        refresh_url: url,
        restarting: true,
        generation: state.config.generation(),
    })
}

/// `level` from the urlencoded body or the query string. Applied immediately through the
/// tracing reload handle and persisted as `log_level` in the config file. Browsers (Accept:
/// text/html) get the config page back, other clients JSON.
pub(super) async fn config_loglevel(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    query: Result<Query<HashMap<String, String>>, QueryRejection>,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    blocking(move || {
        let level = form
            .ok()
            .and_then(|Form(f)| f.get("level").cloned())
            .or_else(|| query.ok().and_then(|Query(q)| q.get("level").cloned()));
        let result = (|| -> anyhow::Result<LogLevel> {
            let raw = level.ok_or_else(|| {
                anyhow::anyhow!(
                    "the request has no 'level' field (the page is probably out of date); \
                     reload the page and pick a level again"
                )
            })?;
            let level: LogLevel = raw.trim().parse()?;
            apply_log_level(&state, level)?;
            state.config.update("log level", |c| {
                c.log_level = level;
                Ok(())
            })?;
            Ok(level)
        })();
        let html = wants_html(&headers);
        match result {
            Ok(level) => {
                info!(level = level.as_str(), "log level changed from the web UI");
                let msg = format!("Log level set to {} and saved.", level.as_str());
                if html {
                    render(&config_template(&state, Some(msg), None, None))
                } else {
                    Json(
                    serde_json::json!({ "success": true, "message": msg, "level": level.as_str() }),
                )
                .into_response()
                }
            }
            Err(e) => {
                let msg = format!("Log level not changed: {e:#}");
                if html {
                    render(&config_template(&state, None, Some(msg), None))
                } else {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({ "success": false, "message": msg })),
                    )
                        .into_response()
                }
            }
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::detect::HardwareInfo;
    use crate::backend::select::{RuntimeProbe, select};
    use crate::server::test_util::*;

    use crate::config::Config;

    use axum::http::StatusCode;

    use std::time::Duration;

    use axum::body::Body;
    use axum::http::Request;

    #[test]
    fn timezone_choices_select_the_current_value() {
        let selected = |cur: &str| -> Vec<String> {
            timezone_choices(cur)
                .into_iter()
                .filter(|t| t.selected)
                .map(|t| format!("{}|{}", t.value, t.label))
                .collect()
        };
        assert_eq!(selected("utc"), ["utc|UTC"]);
        assert_eq!(selected(""), ["utc|UTC"]);
        assert_eq!(
            selected("local"),
            ["local|Local (this computer's time zone)"]
        );
        assert_eq!(
            selected("America/Los_Angeles"),
            ["America/Los_Angeles|Pacific (Los Angeles)"]
        );
        // Values set in the config file stay selected as a custom option.
        let custom = timezone_choices("+08:00");
        assert_eq!(custom[0].value, "+08:00");
        assert_eq!(custom[0].label, "Custom: +08:00");
        assert!(custom[0].selected);
        assert_eq!(custom.iter().filter(|t| t.selected).count(), 1);
        // Every offered value is accepted by the config parser.
        for (value, _) in LOG_TIMEZONES {
            value
                .parse::<crate::config::LogTimezone>()
                .unwrap_or_else(|e| panic!("{value}: {e:#}"));
        }
    }

    fn snapshot(devices: &[&str]) -> DevicesSnapshot {
        use crate::backend::detect::{GpuAdapter, GpuVendor};
        let hardware = HardwareInfo::new(
            "linux",
            "x86_64",
            vec![
                GpuAdapter {
                    vendor: GpuVendor::Intel,
                    name: "UHD 630".into(),
                    vram_mb: 0,
                    index: 0,
                    discrete: false,
                    cuda: None,
                },
                GpuAdapter {
                    vendor: GpuVendor::Nvidia,
                    name: "RTX 3060".into(),
                    vram_mb: 12288,
                    index: 1,
                    discrete: true,
                    cuda: None,
                },
            ],
        );
        let names: Vec<String> = devices.iter().map(|d| d.to_string()).collect();
        let probe = RuntimeProbe::openvino_only(&names);
        let selection = select(&hardware, &probe, true);
        DevicesSnapshot {
            hardware,
            selection,
            probe,
            openvino_version: "test".into(),
            openvino_devices: names,
            onnxruntime: None,
        }
    }

    #[test]
    fn device_choices_list_auto_first_and_grey_out_unrunnable() {
        let snap = snapshot(&["CPU", "GPU.0"]);
        let c = device_choices(&snap, "auto", "0");
        assert_eq!(c[0].value, "auto");
        assert!(
            c[0].label.starts_with("Auto \u{2014} currently: "),
            "{}",
            c[0].label
        );
        assert!(c[0].selected && !c[0].disabled);
        assert_eq!(c.iter().filter(|d| d.selected).count(), 1);
        let gpu = c
            .iter()
            .find(|d| d.value == "openvino:gpu.0")
            .expect("gpu option");
        assert!(!gpu.disabled);
        // ONNX Runtime is not in this probe: its options are listed, disabled, with a reason.
        let cuda = c
            .iter()
            .find(|d| d.value.starts_with("ort:cuda"))
            .expect("cuda option");
        assert!(cuda.disabled && !cuda.selected && cuda.label.contains("not available"));
        // Runnable options come before the disabled ones.
        let first_disabled = c.iter().position(|d| d.disabled).unwrap();
        assert!(c[..first_disabled].iter().all(|d| !d.disabled));
        assert!(c[first_disabled..].iter().all(|d| d.disabled));
    }

    #[test]
    fn device_choices_select_current_value() {
        let snap = snapshot(&["CPU", "GPU.0"]);
        // Legacy spellings map to their option.
        for cur in ["GPU", "openvino:gpu", " gpu "] {
            let c = device_choices(&snap, cur, "0");
            let sel: Vec<_> = c.iter().filter(|d| d.selected).collect();
            assert_eq!(sel.len(), 1, "{cur}");
            assert_eq!(sel[0].value, "openvino:gpu.0", "{cur}");
        }
        // A current value that cannot run stays selected and enabled (so saving keeps it).
        let c = device_choices(&snap, "ort:cuda", "0");
        let sel: Vec<_> = c.iter().filter(|d| d.selected).collect();
        assert_eq!(sel.len(), 1);
        assert!(!sel[0].disabled && sel[0].label.contains("not available"));
        // An unparseable value is kept as an extra option.
        let c = device_choices(&snap, "vulkan", "0");
        let last = c.last().unwrap();
        assert!(last.selected && last.value == "vulkan");
    }

    #[test]
    fn model_device_options_carry_the_threshold_per_device() {
        let snap = snapshot(&["CPU", "GPU.0"]);
        let c = Config {
            device: "auto".into(),
            ..Default::default()
        };
        let m = crate::config::ModelConfig {
            name: Some("m".into()),
            path: "models/m.onnx".into(),
            device: Some("openvino:gpu.0".into()),
            confidence_threshold: Some(0.38),
            device_thresholds: [("openvino:gpu".to_string(), 0.3)].into(),
            ..Default::default()
        };
        let facts = live::BenchFacts {
            devices: vec![
                live::DeviceFact {
                    device: "openvino:cpu".into(),
                    p50_ms: Some(24.4),
                    best: Some(0.34),
                },
                live::DeviceFact {
                    device: "openvino:gpu.0".into(),
                    p50_ms: Some(8.25),
                    best: None,
                },
            ],
            ..Default::default()
        };
        let mut v = model_device_choices(&snap, m.device.as_deref(), "auto");
        annotate_model_devices(&mut v, &c, &m, &snap.selection, None, Some(&facts));
        let by = |value: &str| v.iter().find(|d| d.value == value).unwrap();
        let cpu = by("openvino:cpu");
        assert_eq!(cpu.key, "openvino:cpu");
        assert_eq!((cpu.thr.as_str(), cpu.bench.as_str()), ("", "0.34"));
        assert_eq!(cpu.src, "model default 0.38");
        assert!(
            cpu.label.ends_with(" \u{b7} 24 ms \u{b7} best conf 0.34"),
            "{}",
            cpu.label
        );
        // gpu.0 matches the `openvino:gpu` profile; a sub-10 ms speed keeps a decimal.
        let gpu = by("openvino:gpu.0");
        assert!(gpu.selected);
        assert_eq!((gpu.thr.as_str(), gpu.src.as_str()), ("0.30", "profile"));
        assert_eq!(gpu.bench, "");
        assert!(gpu.label.ends_with(" \u{b7} 8.2 ms"), "{}", gpu.label);
        // Auto and Global name the device they resolve to for this model.
        let auto = by("auto");
        assert!(!auto.key.is_empty());
        assert!(
            auto.label
                .starts_with(&format!("Auto \u{2014} currently: {}", auto.key))
        );
        let global = by("");
        assert_eq!(global.key, auto.key);
        assert!(
            global
                .label
                .starts_with(&format!("Global (auto \u{2192} {})", global.key)),
            "{}",
            global.label
        );
        // Loaded: the selected option is the device the model runs on (a fallback here).
        let mut v = model_device_choices(&snap, m.device.as_deref(), "auto");
        annotate_model_devices(
            &mut v,
            &c,
            &m,
            &snap.selection,
            Some("openvino:cpu"),
            Some(&facts),
        );
        let gpu = v.iter().find(|d| d.selected).unwrap();
        assert_eq!(
            (gpu.key.as_str(), gpu.bench.as_str()),
            ("openvino:cpu", "0.34")
        );
        // A value that is not a device resolves to nothing.
        let mut v = model_device_choices(&snap, Some("vulkan"), "auto");
        annotate_model_devices(&mut v, &c, &m, &snap.selection, None, None);
        let bad = v.iter().find(|d| d.selected).unwrap();
        assert_eq!((bad.key.as_str(), bad.thr.as_str()), ("", ""));
    }

    #[tokio::test]
    async fn models_card_threshold_follows_the_selected_device() {
        let state = test_state_with(|c| {
            let m = &mut c.models[0];
            m.device = Some("openvino:cpu".into());
            m.confidence_threshold = Some(0.38);
            m.device_thresholds = [
                ("openvino:cpu".to_string(), 0.3),
                ("ort:coreml".to_string(), 0.25),
            ]
            .into();
        });
        live::tests::write_profiled_bench(&state);
        let (_, html) = get_text(&state, "/config").await;
        // The options carry the device, its profile, the threshold source and the benchmark's best.
        assert!(
            html.contains(
                "<option value=\"openvino:cpu\" selected data-key=\"openvino:cpu\" data-thr=\"0.30\" data-src=\"profile\" data-bench=\"0.34\">"
            ),
            "{html}"
        );
        assert!(html.contains("best conf 0.34</option>"), "{html}");
        // The field shows the profile of the selected device; the placeholder the fallback.
        assert!(
            html.contains(
                "name=\"threshold.ipcam-general\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0.30\" placeholder=\"0.38 model default\""
            ),
            "{html}"
        );
        assert!(
            html.contains(
                "<input type=\"hidden\" class=\"thr-key\" name=\"threshold_key.ipcam-general\" value=\"openvino:cpu\">"
            ),
            "{html}"
        );
        assert!(html.contains("for <code>openvino:cpu</code>"), "{html}");
        assert!(html.contains("benchmark best 0.34"), "{html}");
        assert!(html.contains("data-use-threshold=\"0.34\""), "{html}");
        // Every profile with a remove box, and the default for other devices.
        assert!(
            html.contains(
                "<code>ort:coreml</code> 0.25 <label class=\"check small\"><input type=\"checkbox\" name=\"profile_remove.ipcam-general\" value=\"ort:coreml\"> remove</label>"
            ),
            "{html}"
        );
        assert!(html.contains("Confidence per device (2)"), "{html}");
        assert!(
            html.contains("name=\"threshold_default.ipcam-general\" min=\"0\" max=\"1\" step=\"0.01\" value=\"0.38\""),
            "{html}"
        );

        // Saving the card for another device writes that device's profile only.
        let body = "enabled=ipcam-general&enabled=%3Cb%3Eevil%3C%2Fb%3E&device.ipcam-general=openvino%3Acpu\
                    &threshold.ipcam-general=0.34&threshold_key.ipcam-general=openvino%3Acpu\
                    &threshold_default.ipcam-general=0.38&profile_remove.ipcam-general=ort%3Acoreml&action=save";
        let (s, _, b) = call(&state, post_form("/config/models", body, false)).await;
        assert_eq!(s, StatusCode::OK, "{}", String::from_utf8_lossy(&b));
        let saved = Config::load(&state.config_path).unwrap();
        assert_eq!(
            saved.models[0].device_thresholds,
            [("openvino:cpu".to_string(), 0.34)].into()
        );
        assert_eq!(saved.models[0].confidence_threshold, Some(0.38));
        let (_, html) = get_text(&state, "/config").await;
        assert!(html.contains("Confidence per device (1)"), "{html}");
        // The device's profile now equals the benchmark's best: no "use" button.
        assert!(!html.contains("data-use-threshold=\"0.34\""), "{html}");
        assert!(
            html.contains("benchmark best 0.34</span> <span class=\"muted\">set</span>"),
            "{html}"
        );
    }

    #[tokio::test]
    async fn config_save_and_validation() {
        let state = test_state();
        let path = state.config_path.clone();

        let bad = format!("port=4000&models_json={}", urlencode("[{not json"));
        let (s, _, b) = call(&state, post_form("/config", &bad, true)).await;
        let page = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::OK);
        assert!(
            page.contains("class=\"error\">Not saved: models:"),
            "{page}"
        );
        // The rejected submission is shown as typed; nothing changed or written.
        assert!(page.contains("[{not json</textarea>"));
        assert!(!path.exists());
        assert_eq!(state.config_read().port, crate::DEFAULT_PORT);

        let models =
            r#"[{"name":"ipcam-general","path":"models/ipcam-general.onnx","family":"yolo5"}]"#;
        let good = format!(
            "port=4000&request_timeout_secs=20&device=CPU&inference_settings=1&force_cpu=on&confidence_threshold=0.4\
             &object_filter={}&log_level=info&default_model=ipcam-general&models_json={}",
            urlencode("person, car"),
            urlencode(models)
        );
        let (_, _, b) = call(&state, post_form("/config", &good, true)).await;
        let page = String::from_utf8(b).unwrap();
        assert!(page.contains("Restart the server to apply"), "{page}");
        let saved = Config::load(&path).unwrap();
        assert_eq!(saved.port, 4000);
        assert!(saved.force_cpu);
        assert_eq!(saved.object_filter, vec!["person", "car"]);
        assert_eq!(saved.models.len(), 1);
        assert_eq!(*state.config_read(), saved);

        // A models list with every entry disabled is rejected; file and memory unchanged.
        let off =
            r#"[{"name":"ipcam-general","path":"models/ipcam-general.onnx","enabled":false}]"#;
        let body = format!("port=4002&models_json={}", urlencode(off));
        let (s, _, b) = call(&state, post_form("/config", &body, true)).await;
        let page = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::OK);
        assert!(
            page.contains("Not saved: models: at least one model must be enabled"),
            "{page}"
        );
        assert_eq!(Config::load(&path).unwrap(), saved);
        assert_eq!(*state.config_read(), saved);
    }

    /// `POST /config/models/remove`: a disabled entry goes, an enabled one and an
    /// unknown name are refused, and only loopback peers (same-site) may call it.
    #[tokio::test]
    async fn config_models_remove_entry() {
        let state = test_state_with(|c| {
            c.models.push(crate::config::ModelConfig {
                name: Some("dfine-m".into()),
                path: "models/dfine-m.onnx".into(),
                enabled: false,
                ..Default::default()
            });
            c.default_model = Some("dfine-m".into());
        });
        let path = state.config_path.clone();
        let (_, html) = get_text(&state, "/config").await;
        assert!(
            html.contains("action=\"/config/models/remove#at-models\" id=\"model-remove\""),
            "{html}"
        );
        // Enabled rows: a disabled button with the reason; the disabled row can be removed.
        assert!(html.contains(
            "form=\"model-remove\" name=\"name\" value=\"ipcam-general\" data-confirm=\"Remove ipcam-general from the config and delete its files in the models folder? Files outside it and catalog models stay.\" disabled title=\"Enabled: disable it first"
        ), "{html}");
        assert!(html.contains(
            "value=\"dfine-m\" data-confirm=\"Remove dfine-m from the config and delete its files in the models folder? Files outside it and catalog models stay.\" title=\"Remove this entry"
        ), "{html}");

        // A remote peer and a cross-site page are refused; nothing changes.
        let remote = from_peer(
            post_form("/config/models/remove", "name=dfine-m", false),
            REMOTE,
        );
        let (s, _, _) = call(&state, remote).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let mut cross = post_form("/config/models/remove", "name=dfine-m", false);
        cross
            .headers_mut()
            .insert("sec-fetch-site", "cross-site".parse().unwrap());
        let (s, _, _) = call(&state, cross).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        assert_eq!(state.config_read().models.len(), 3);

        let post = |body: &'static str| {
            let state = state.clone();
            async move {
                let (s, _, b) = call(&state, post_form("/config/models/remove", body, false)).await;
                (s, serde_json::from_slice::<serde_json::Value>(&b).unwrap())
            }
        };
        let (s, b) = post("name=nope").await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{b}");
        let (s, b) = post("name=").await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{b}");
        let (s, b) = post("name=ipcam-general").await;
        assert_eq!(s, StatusCode::CONFLICT, "{b}");
        assert!(
            b["message"].as_str().unwrap().contains("disable it first"),
            "{b}"
        );
        assert_eq!(state.config_read().models.len(), 3);

        let (s, b) = post("name=DFINE-M").await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert_eq!(b["success"], true);
        let msg = b["message"].as_str().unwrap();
        assert!(
            msg.starts_with("removed 'dfine-m' from the config. Saved to"),
            "{msg}"
        );
        let saved = Config::load(&path).unwrap();
        let names: Vec<String> = saved.models.iter().map(|m| m.effective_name()).collect();
        assert_eq!(names, ["ipcam-general", "<b>evil</b>"]);
        assert_eq!(saved.default_model, None, "the dangling default is cleared");
        assert_eq!(*state.config_read(), saved);
        let last = state.config.changes(1).pop().expect("logged");
        assert_eq!(last.source, "config page (remove model entry)");

        // Browsers get the page at the Models card.
        let (s, _, b) = call(
            &state,
            post_form("/config/models/remove", "name=dfine-m", true),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let page = String::from_utf8(b).unwrap();
        assert!(
            page.contains("class=\"error\">Not removed: no model named &#39;dfine-m&#39;"),
            "{page}"
        );
    }

    #[tokio::test]
    async fn config_models_selection() {
        let state = test_state();
        let path = state.config_path.clone();
        let evil = urlencode("<b>evil</b>");

        let (_, cfg) = get_text(&state, "/config").await;
        assert!(cfg.contains("action=\"/config/models#at-models\""), "{cfg}");
        assert!(cfg.contains(
            "name=\"enabled\" value=\"ipcam-general\" aria-label=\"Load ipcam-general\" checked"
        ));
        assert!(cfg.contains(
            "name=\"default_model\" value=\"ipcam-general\" aria-label=\"Default ipcam-general\" checked"
        ));
        assert!(cfg.contains("(missing)"));
        assert!(!cfg.contains("<b>evil</b>"));
        // The main form no longer carries default_model.
        assert!(!cfg.contains("name=\"default_model\" value=\"\""));

        // Enable a subset with an explicit default.
        let body = format!("enabled={evil}&default_model={evil}&action=save");
        let (s, _, b) = call(&state, post_form("/config/models", &body, true)).await;
        let page = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::OK);
        assert!(page.contains("Restart the server to apply"), "{page}");
        let saved = Config::load(&path).unwrap();
        let on: Vec<bool> = saved.models.iter().map(|m| m.enabled).collect();
        assert_eq!(on, [false, true]);
        assert_eq!(saved.default_model.as_deref(), Some("<b>evil</b>"));
        assert_eq!(*state.config_read(), saved);
        assert!(page.contains(
            "name=\"enabled\" value=\"ipcam-general\" aria-label=\"Load ipcam-general\">"
        ));

        // A default that is not checked gets enabled.
        let body = format!("enabled={evil}&default_model=ipcam-general&action=save");
        call(&state, post_form("/config/models", &body, true)).await;
        let saved = Config::load(&path).unwrap();
        assert!(saved.models.iter().all(|m| m.enabled));
        assert_eq!(saved.default_model.as_deref(), Some("ipcam-general"));

        // The main form keeps default_model and the enabled flags.
        let body = "enabled=ipcam-general&default_model=ipcam-general&action=save";
        call(&state, post_form("/config/models", body, true)).await;
        let models_json = serde_json::to_string(&state.config_read().models).unwrap();
        let main = format!("port=4001&models_json={}", urlencode(&models_json));
        call(&state, post_form("/config", &main, true)).await;
        let saved = Config::load(&path).unwrap();
        assert_eq!(saved.port, 4001);
        assert_eq!(saved.default_model.as_deref(), Some("ipcam-general"));
        let on: Vec<bool> = saved.models.iter().map(|m| m.enabled).collect();
        assert_eq!(on, [true, false]);

        // Zero enabled models are rejected and nothing changes.
        let (s, _, b) = call(&state, post_form("/config/models", "action=save", true)).await;
        let page = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::OK);
        assert!(
            page.contains("class=\"error\">Models not saved: select at least one model"),
            "{page}"
        );
        assert_eq!(Config::load(&path).unwrap(), saved);
        assert_eq!(*state.config_read(), saved);
        let (_, _, b) = call(
            &state,
            post_form("/config/models", "enabled=zzz&action=save", true),
        )
        .await;
        assert!(String::from_utf8(b).unwrap().contains("Models not saved"));
        assert!(!state.restart.is_cancelled());

        // Save and restart.
        let (s, _, b) = call(
            &state,
            post_form(
                "/config/models",
                &format!("enabled=ipcam-general&enabled={evil}&action=restart"),
                true,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert!(String::from_utf8(b).unwrap().contains("Restarting"));
        let saved = Config::load(&path).unwrap();
        assert!(saved.models.iter().all(|m| m.enabled));
        assert_eq!(saved.default_model, None);
        tokio::time::timeout(Duration::from_secs(5), state.restart.cancelled())
            .await
            .expect("restart token cancelled");
    }

    #[tokio::test]
    async fn loglevel_and_restart() {
        let state = test_state();
        // No reload handle in tests: the change is refused and nothing is persisted.
        let (s, _, b) = call(&state, post_form("/config/loglevel", "level=debug", false)).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert!(String::from_utf8(b).unwrap().contains("not available"));
        let (s, _, _) = call(&state, post_form("/config/loglevel", "level=loud", false)).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _, b) = call(&state, post_form("/config/loglevel", "", true)).await;
        assert_eq!(s, StatusCode::OK);
        let page = String::from_utf8(b).unwrap();
        assert!(
            page.contains("level") && page.contains("reload the page"),
            "{page}"
        );
        assert!(!state.config_path.exists());

        assert!(!state.restart.is_cancelled());
        let (s, _, b) = call(
            &state,
            Request::post("/config/restart")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert!(String::from_utf8(b).unwrap().contains("Restarting"));
        tokio::time::timeout(Duration::from_secs(5), state.restart.cancelled())
            .await
            .expect("restart token cancelled");
    }

    /// Models JSON paths: a new or changed file outside the browsable folders is refused
    /// (naming the model and the field), an unchanged one is kept, and a missing file inside
    /// the data root (a download still to come) is accepted.
    #[test]
    fn models_json_paths_must_stay_inside_the_roots() {
        use super::super::browse::allowed_or_missing_inside_roots as allowed;
        let root = std::env::temp_dir().join(format!("bop-cfg-roots-{}", uuid::Uuid::new_v4()));
        // Not canonicalized: a Windows verbatim (`\\?\`) path is refused as typed input.
        std::fs::create_dir_all(root.join("models")).unwrap();
        let outside = std::env::temp_dir().join(format!("bop-cfg-out-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&outside).unwrap();
        let outside = outside.canonicalize().unwrap();
        let model = |name: &str, path: PathBuf| crate::config::ModelConfig {
            name: Some(name.into()),
            path,
            ..Default::default()
        };
        let before = Config {
            download_dir: Some(root.clone()),
            models: vec![model("old", outside.join("old.onnx"))],
            ..Default::default()
        };
        // Unchanged entry outside every root: kept.
        let mut after = before.clone();
        after.models[0].lazy = true;
        check_new_model_paths(&before, &after, allowed).unwrap();
        // New entry inside the data root, not downloaded yet: accepted (relative or absolute).
        after
            .models
            .push(model("later", "models/later.onnx".into()));
        after
            .models
            .push(model("abs", root.join("models/sub/abs.onnx")));
        check_new_model_paths(&before, &after, allowed).unwrap();
        // A changed path or classes file outside: refused, naming the model and the field.
        let in_home = ["HOME", "USERPROFILE"].iter().any(|v| {
            std::env::var_os(v)
                .and_then(|h| PathBuf::from(h).canonicalize().ok())
                .is_some_and(|h| outside.starts_with(h))
        });
        if !in_home {
            let mut moved = after.clone();
            moved.models[1].path = outside.join("later.onnx");
            let e = check_new_model_paths(&before, &moved, allowed)
                .unwrap_err()
                .to_string();
            assert!(
                e.contains("path of model 'later'") && e.contains("Browse"),
                "{e}"
            );
            let mut classes = after.clone();
            classes.models[0].classes = Some(outside.join("c.yaml"));
            let e = check_new_model_paths(&before, &classes, allowed)
                .unwrap_err()
                .to_string();
            assert!(e.contains("classes of model 'old'"), "{e}");
        }
        // The rule with a stand-in check: only the new paths are asked about.
        let mut asked = Vec::new();
        let _ = check_new_model_paths(&before, &after, |_, p: &str| {
            asked.push(p.to_string());
            Ok::<PathBuf, ()>(PathBuf::new())
        });
        assert_eq!(asked.len(), 2, "{asked:?}");
        assert!(
            asked
                .iter()
                .all(|p| p.starts_with(&root.display().to_string()))
        );
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn server_tab_badges_follow_the_merge_rules() {
        use crate::config_merge::{flatten_all, pending};
        // Process restart: exactly what `pending` reports as needing a process restart.
        let base = Config::default();
        let mut c = base.clone();
        c.port = base.port.wrapping_add(1);
        c.bind_address = "127.0.0.1".parse().unwrap();
        c.log_path = Some("logs-elsewhere".into());
        c.log_format = match base.log_format {
            crate::config::LogFormat::Text => crate::config::LogFormat::Json,
            crate::config::LogFormat::Json => crate::config::LogFormat::Text,
        };
        let p = pending(&base, &c);
        assert_eq!(p.process.len(), ApplyWhen::PROCESS.len(), "{:?}", p.process);
        assert!(p.global.is_empty(), "{:?}", p.global);
        for k in ApplyWhen::PROCESS {
            assert_eq!(ApplyWhen::of(k), ApplyWhen::Process, "{k}");
        }
        // A runtime-level field: a server restart; a live one: now.
        let mut c = base.clone();
        c.worker_queue_size += 3;
        let p = pending(&base, &c);
        assert_eq!((p.global.len(), p.process.len()), (1, 0), "{p:?}");
        assert_eq!(ApplyWhen::of("worker_queue_size"), ApplyWhen::Restart);
        assert_eq!(ApplyWhen::of("request_timeout_secs"), ApplyWhen::Now);
        // Groups: one badge when the fields agree, none when they differ.
        assert!(when_badge("confidence_threshold,nms_iou").contains("applies now"));
        assert!(when_badge("device,cache_dir").contains("needs restart"));
        assert_eq!(when_badge("port,request_timeout_secs"), "");
        assert!(when_badge("port").contains("process restart"));
        // Every key the template asks about is a real global config field (serialized, or one
        // serialized only when set).
        let known = flatten_all(&base);
        let source = include_str!("../config.rs");
        let tpl = include_str!("../../templates/config.html");
        let mut n = 0;
        for part in tpl.split("self.badge(\"").skip(1) {
            let keys = &part[..part.find('"').unwrap()];
            for k in keys.split(',') {
                n += 1;
                assert!(
                    known.contains_key(k) || source.contains(&format!("    pub {k}: ")),
                    "unknown config key {k} in a Server tab badge"
                );
            }
        }
        assert!(n > 20, "{n}");
    }

    #[tokio::test]
    async fn config_page_renders_and_saves_the_per_model_classes() {
        let state = test_state_with(|c| {
            c.models[1].object_filter = Some(vec![]);
        });
        let (s, html) = get_text(&state, "/config").await;
        assert_eq!(s, StatusCode::OK);
        // No per-model filter = global; an empty one = all; both rendered in the hidden field.
        assert!(
            html.contains(
                "class=\"of-value\" name=\"object_filter.ipcam-general\" value=\"__global__\""
            ),
            "{html}"
        );
        assert!(html.contains("class=\"of-summary\" data-model=\"ipcam-general\">global<"));
        assert!(
            html.contains("value=\"\"><span class=\"of-summary\""),
            "{html}"
        );
        assert!(html.contains("id=\"of-global-edit\""));
        assert!(html.contains("id=\"class-dialog\""));
        let (picker, cfg) = (
            html.find("/static/class-picker.js?v=").unwrap(),
            html.find("/static/config.js?v=").unwrap(),
        );
        assert!(picker < cfg);

        // A list, then empty, then global again round-trip into ModelConfig.object_filter.
        for (value, want) in [
            ("person%2C+car", Some(vec!["person", "car"])),
            ("", Some(vec![])),
            ("__global__", None),
        ] {
            let body = format!(
                "enabled=ipcam-general&enabled=%3Cb%3Eevil%3C%2Fb%3E&object_filter.ipcam-general={value}&action=save"
            );
            let (s, _, b) = call(&state, post_form("/config/models", &body, false)).await;
            assert_eq!(s, StatusCode::OK, "{}", String::from_utf8_lossy(&b));
            let saved = Config::load(&state.config_path).unwrap();
            let want: Option<Vec<String>> = want.map(|v| v.into_iter().map(String::from).collect());
            assert_eq!(saved.models[0].object_filter, want, "{value}");
        }
        // A list that names no class is refused; the saved value stays.
        let body = "enabled=ipcam-general&object_filter.ipcam-general=%2C+%2C&action=save";
        let (_, _, b) = call(&state, post_form("/config/models", body, false)).await;
        let page = String::from_utf8(b).unwrap();
        assert!(page.contains("Models not saved"), "{page}");
        assert!(page.contains("names no class"), "{page}");
        assert_eq!(
            Config::load(&state.config_path).unwrap().models[0].object_filter,
            None
        );

        // The saved list shows in the cell.
        let body = "enabled=ipcam-general&object_filter.ipcam-general=person%2Ccar&action=save";
        call(&state, post_form("/config/models", body, false)).await;
        let (_, html) = get_text(&state, "/config").await;
        assert!(html.contains("value=\"person,car\""), "{html}");
        assert!(html.contains(">person, car</span>"), "{html}");
    }

    #[tokio::test]
    async fn config_page_has_browse_buttons() {
        let state = test_state();
        let (s, html) = get_text(&state, "/config").await;
        assert_eq!(s, StatusCode::OK);
        assert!(html.contains("data-browse=\"model\" data-target=\"path\""));
        assert!(html.contains("data-browse=\"classes\" data-target=\"classes\""));
        assert!(html.contains("id=\"browse-dialog\""));
        // The folder settings of the Server tab browse folders; the shared picker script
        // (native dialog, else the in-page browser) is loaded after config.js.
        for target in [
            "cache_dir",
            "log_path",
            "save_image_path",
            "download_dir",
            "models_dir",
            "onnxruntime_dir",
            "openvino_dir",
        ] {
            assert!(
                html.contains(&format!("data-browse=\"folder\" data-target=\"{target}\"")),
                "{target}"
            );
        }
        let (cfg_js, browse_js) = (
            html.find("/static/config.js?v=").unwrap(),
            html.find("/static/browse.js?v=").unwrap(),
        );
        assert!(cfg_js < browse_js);
        assert!(
            super::super::pages::static_file("config.js")
                .is_some_and(|js| !js.contains("/v1/fs/browse"))
        );
        // The path hint is a file under the models dir in this OS's form (where Browse starts).
        let cfg = state.config_read().clone();
        let example = cfg
            .data_path(&cfg.models_dir)
            .join("my-rfdetr")
            .join("inference_model.onnx");
        let escaped = askama::filters::escape(example.display(), askama::filters::Html)
            .unwrap()
            .to_string();
        assert!(
            html.contains(&format!("placeholder=\"{escaped}\"")),
            "{html}"
        );
        let (s, body) = get_text(&state, "/v1/fs/browse?kind=nope").await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert!(body.contains("\"success\":false"));
    }

    #[tokio::test]
    async fn config_page_renders_the_resources_card() {
        let (state, root) = resources_state();
        let (s, html) = get_text(&state, "/config").await;
        assert_eq!(s, StatusCode::OK);
        assert!(html.contains("<h2>Resources</h2>"), "{html}");
        // One collapsible group per kind with its counts, one filter box above them, one
        // column layout.
        assert!(
            html.contains("<details class=\"res-group\" id=\"res-"),
            "{html}"
        );
        assert!(html.contains("id=\"res-filter\""));
        assert!(
            html.contains("<col class=\"c-size\"><col class=\"c-state\"><col class=\"c-action\">")
        );
        assert!(html.contains(" resources") || html.contains("1 resource"));
        assert!(html.contains("model:fixture-model"));
        assert!(html.contains("action=\"/v1/resources/download#resource-"));
        assert!(
            html.contains("Download 629 MB (large)"),
            "large button shows the size"
        );
        assert!(html.contains("export_model.py"));
        assert!(html.contains("name=\"auto_download\" checked"));
        assert!(html.contains("name=\"allow_large_downloads\">"));
        // A browser download goes back to the page with a notice.
        let (s, _, b) = call(
            &state,
            post_form("/v1/resources/download", "id=model%3Abig-model", true),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let b = String::from_utf8(b).unwrap();
        assert!(b.contains("confirm the large download"), "{b}");
        // ... shown in the resource's row (the form posts to that row's #fragment), not at the
        // top of the page.
        let row = b
            .find("id=\"resource-model:big-model\"")
            .expect("row anchor");
        let note = b.find("confirm the large download").unwrap();
        let next_row = b[row + 1..].find("<tr ").map_or(b.len(), |i| row + 1 + i);
        assert!(row < note && note < next_row, "message in the row");
        assert!(b.contains("action=\"/v1/resources/download#resource-model:big-model\""));
        assert!(b.contains("data-ajax"));
        // Without a manager (plain test state) the card says so and downloads are refused.
        let plain = test_state();
        let (_, html) = get_text(&plain, "/config").await;
        assert!(html.contains("has no download manager"));
        let (s, _) = post_action(&plain, "/v1/resources/download", "id=openvino-runtime").await;
        assert_eq!(s, StatusCode::INTERNAL_SERVER_ERROR);
        let _ = std::fs::remove_dir_all(&root);
    }
    #[tokio::test]
    async fn no_js_answers_land_in_their_section() {
        let state = test_state();
        let (_, html) = get_text(&state, "/config").await;
        for a in [
            "action=\"/config/models#at-models\"",
            "action=\"/config#at-server\"",
            "action=\"/config/restart?back=restart\"",
            "formaction=\"/config/restart?back=models\"",
        ] {
            assert!(html.contains(a), "{a}");
        }
        let between = |b: &str, msg: &str, from: &str, to: &str| {
            let m = b.find(msg).unwrap_or_else(|| panic!("{msg} missing: {b}"));
            b.find(from).unwrap() < m && m < b.find(to).unwrap()
        };
        // Server form save: the notice right above the Server form.
        let body = format!(
            "base_revision=0&port={}&request_timeout_secs=15&worker_queue_size=0&device=auto&gpu_index=0&confidence_threshold=0.5&nms_iou=0.4&intra_threads=0&log_level=info",
            state.config_read().port
        );
        let (s, _, b) = call(&state, post_form("/config", &body, true)).await;
        assert_eq!(s, StatusCode::OK);
        let b = String::from_utf8(b).unwrap();
        assert!(
            between(&b, "class=\"notice\"", "id=\"at-server\"", "id=\"server\""),
            "{b}"
        );
        // Models card save: above the card.
        let (s, _, b) = call(
            &state,
            post_form(
                "/config/models",
                "base_revision=0&enabled=ipcam-general&action=save",
                true,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let b = String::from_utf8(b).unwrap();
        let m = b.find("id=\"at-models\"").unwrap();
        let card = b.find("id=\"models\"").unwrap();
        assert!(
            b[m..card].contains("class=\"notice\"") || b[m..card].contains("class=\"error\""),
            "{b}"
        );
        // Restart returns to the section it came from.
        let (_, _, b) = call(&state, post_form("/config/restart?back=models", "", true)).await;
        let b = String::from_utf8(b).unwrap();
        assert!(b.contains("/config#at-models"), "{b}");
    }

    /// The export notice covers every recipe and every restrictive license in the catalog.
    #[test]
    fn export_notice_is_derived_from_the_catalog() {
        use crate::resources::catalog;
        let n = super::export_notice();
        let mut seen = std::collections::HashSet::new();
        for res in catalog::all() {
            let Some(r) = catalog::export_recipe(res) else {
                continue;
            };
            if seen.insert(r) {
                assert!(n.contains(r.notice()), "{r:?} notice missing: {n}");
            }
            if res.license.to_ascii_lowercase().contains("non-commercial") {
                let name = res.model_name().unwrap();
                assert!(n.contains(name), "{name} missing: {n}");
                assert!(n.contains(res.license), "{} missing: {n}", res.license);
            }
        }
        // The recipes added with the night work.
        assert!(seen.contains(&catalog::ExportRecipe::RtDetrV2), "{seen:?}");
        assert!(n.contains("MegaDetector v6 RT-DETR"), "{n}");
        assert!(n.contains("exdark-rfdetr-small"), "{n}");
        assert!(n.contains("trained on ExDark"), "{n}");
        // Each notice once, however many models share it.
        assert_eq!(n.matches("Ultralytics YOLO weights and").count(), 1, "{n}");
    }
}
