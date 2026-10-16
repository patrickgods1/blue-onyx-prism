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
}

impl ConfigView {
    pub(super) fn from_config(c: &Config) -> Self {
        let path = |p: &Option<PathBuf>| {
            p.as_ref()
                .map(|p| p.display().to_string())
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
    /// Per-model threshold as typed in the input ("" = global).
    threshold: String,
    /// What the running worker uses ("OpenVINO CPU · threshold 0.50", "not loaded", ...).
    running: String,
    running_class: &'static str,
    /// What a restart would change for this model (empty = nothing).
    pending: Vec<String>,
    /// Latest benchmark recommendation for this model.
    bench: Option<BenchHint>,
    /// Best confidence threshold of the latest benchmark: (value "0.35", objective "f1",
    /// "in use" / "configured; restart to use" when it is the configured threshold).
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
        },
    );
    v
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
            let hint = facts.map(|r| {
                let sel = snap.model_selection(state, c, m);
                let configured = crate::benchmark::configured_device(c, m, &sel);
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
                        m.confidence_threshold.unwrap_or(c.confidence_threshold),
                        run.threshold,
                    ),
                    href: format!("/benchmark#m-{}", encode_uri_component(&r.model)),
                }
            });
            let effective = thr(m.confidence_threshold.unwrap_or(c.confidence_threshold));
            let best_threshold = facts.and_then(|f| {
                let t = thr(f.best_threshold?);
                let state = if t != effective {
                    None
                } else if run.threshold == Some(t) {
                    Some("in use")
                } else {
                    Some("configured; restart to use")
                };
                Some((format!("{t:.2}"), f.objective.clone(), state))
            });
            let exists = c.data_path(&m.path).is_file();
            // Missing, or from another version (the old RF-DETR large download).
            let needs_export = crate::resources::export::pending_export(&c.data_path(&m.path))
                .map(|r| r.id.to_string());
            ModelSelectRow {
                devices: model_device_choices(snap, m.device.as_deref(), &global),
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
                threshold: m
                    .confidence_threshold
                    .map(|t| thr(t).to_string())
                    .unwrap_or_default(),
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

#[derive(Template)]
#[template(path = "config.html")]
pub(super) struct ConfigTemplate {
    nav: &'static str,
    version: &'static str,
    config_path: String,
    c: ConfigView,
    models: Vec<ModelSelectRow>,
    log_levels: Vec<LevelChoice>,
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

/// One `<option>` of the Device select.
struct DeviceChoice {
    value: String,
    label: String,
    selected: bool,
    disabled: bool,
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
    }];
    let (ok, bad): (Vec<_>, Vec<_>) = snap.selection.options.iter().partition(|o| o.runnable);
    for o in ok {
        out.push(DeviceChoice {
            value: o.spec.to_string(),
            label: format!("{} \u{2014} {}", o.spec, o.label),
            selected: is_current(&o.spec),
            disabled: false,
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
        });
    }
    if !out.iter().any(|c| c.selected) {
        out.push(DeviceChoice {
            value: current.trim().to_string(),
            label: format!("{} \u{2014} not a known option", current.trim()),
            selected: true,
            disabled: false,
        });
    }
    out
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
    let device_choices = device_choices(&snap, &c.device, &c.gpu_index);
    let resource_groups = crate::resources::status::grouped(crate::resources::status::rows(
        state.resources.as_ref(),
        &cfg,
    ));
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
        model_path_example,
        models_dir,
        tools_dir,
        export_notice: [
            crate::resources::catalog::ExportRecipe::Ultralytics,
            crate::resources::catalog::ExportRecipe::RfDetr,
            crate::resources::catalog::ExportRecipe::DeimV2,
        ]
        .iter()
        .map(|r| r.notice())
        .collect::<Vec<_>>()
        .join(" "),
        force_cpu_note,
        resources_enabled: state.resources.is_some(),
        nav: "config",
        version: crate::VERSION,
        config_path: state.config_path.display().to_string(),
        c,
        models,
        log_levels,
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
        match save_config(&state, "config page", |c| live::server_form_edit(c, &form)) {
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

    #[tokio::test]
    async fn config_page_has_browse_buttons() {
        let state = test_state();
        let (s, html) = get_text(&state, "/config").await;
        assert_eq!(s, StatusCode::OK);
        assert!(html.contains("data-browse=\"model\" data-target=\"path\""));
        assert!(html.contains("data-browse=\"classes\" data-target=\"classes\""));
        assert!(html.contains("id=\"browse-dialog\""));
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
}
