//! What the web UI shows about resources (phase 7.5): the rows of `GET /v1/resources` and the
//! config page's Resources card. The actions on them live in [`super::actions`]:
//!
//! - **Download** queues the resource on the shared download manager. Resources over the
//!   large-download threshold need `confirm_large` (the button that shows the size supplies it)
//!   unless `allow_large_downloads` is set.
//! - **Remove** deletes an install we made (a `.installed.json` manifest names it). It refuses
//!   while the resource downloads, when this process has the runtime loaded (OpenVINO, the
//!   loaded ONNX Runtime flavor, preloaded CUDA libraries: "restart required"), and when an
//!   enabled model of the config uses the model files.
//! - **Add to config** appends an installed catalog model to `models` with the same rules as
//!   `download-models --add-to-config`, or a model file found in `models_dir` that no config
//!   entry references (id `local:<file>`, see [`local_models`]; family `auto`).
//!
//! - **Export** (Ultralytics YOLO, RF-DETR L, DEIMv2; [`super::actions::export`]): queues a
//!   user-confirmed export on the [`super::export`] exporter (the toolchain download needs
//!   `confirm_large` unless `allow_large_downloads`); **Cancel** stops it. Remove deletes the
//!   exported files; removing `tool:uv` removes the whole export toolchain (including the DEIMv2
//!   source, which has no row of its own).
//!
//! Rows carry a `group` ([`group_of`]) so the UI shows Runtimes, GPU libraries and Models
//! (by family) as separate tables ([`grouped`]).

use super::catalog::{self, Flavor, Platform, Resource, ResourceKind};
use super::export::{self, ExportState};
use super::manager::{self, State, Status};
use super::provision::{Provision, Provisioner};
use super::resolve::{self, Installed, Resolution};
use crate::config::Config;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// What the HTTP layer of one generation knows about resources.
#[derive(Clone)]
pub struct ResourcesCtx {
    /// Owns the download manager for the whole run.
    pub provisioner: Arc<Provisioner>,
    /// This generation's plan (what models wait for).
    pub provision: Arc<Provision>,
    /// OpenVINO is loaded in this process (its files are in use).
    pub openvino_loaded: bool,
    /// ONNX Runtime flavor loaded in this process.
    pub ort_loaded: Option<Flavor>,
}

impl std::fmt::Debug for ResourcesCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourcesCtx")
            .field("openvino_loaded", &self.openvino_loaded)
            .field("ort_loaded", &self.ort_loaded)
            .finish_non_exhaustive()
    }
}

/// One resource as shown by `/v1/resources` and the Resources card.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceRow {
    pub id: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    /// `openvino-runtime`, `onnx-runtime`, `cuda-libs`, `model`.
    pub kind: &'static str,
    /// UI section: `runtimes`, `gpu-libraries`, `models-yolov5`, `models-rt-detr`,
    /// `models-d-fine`, `models-rf-detr` (see [`group_of`]).
    pub group: &'static str,
    pub version: &'static str,
    pub license: &'static str,
    pub size: u64,
    /// "29 MB".
    pub size_text: String,
    /// Over the large-download threshold (needs confirmation / `allow_large_downloads`).
    pub large: bool,
    /// `installed`, `downloading`, `queued`, `verifying`, `extracting`, `failed`, `needed`,
    /// `optional` (needed by a better device, gated by `allow_large_downloads`), `available`.
    pub state: &'static str,
    /// Human-readable state, e.g. "downloading 42% (18/44 MB)".
    pub state_text: String,
    /// Queued, downloading, verifying or extracting.
    pub busy: bool,
    /// Download progress in percent while downloading.
    pub progress: Option<u8>,
    /// Last error of a failed download.
    pub error: Option<String>,
    /// Seconds until the next automatic retry of a failed download.
    pub retry_in: Option<u64>,
    /// Device options / model it provides (`ort:coreml`, `model:IPcam-general`).
    pub provides: Vec<String>,
    /// Enabled models that wait for it.
    pub blocking: Vec<String>,
    /// The ONNX Runtime flavor marked active (`active.txt`) or loaded.
    pub active: bool,
    /// Loaded by this process right now (cannot be removed before a restart).
    pub loaded: bool,
    /// Remove is possible now.
    pub removable: bool,
    /// Why Remove is not possible (when installed).
    pub remove_blocked: Option<String>,
    /// A model resource that can be appended to `models` (installed, not in the config yet).
    pub can_add_to_config: bool,
    /// Where it installs.
    pub target: String,
    /// Which buttons the UI shows: `download` (catalog downloads), `export` (exported models), `toolchain`
    /// (`tool:uv`: Remove deletes the whole export toolchain), `none` (not possible here).
    pub action: &'static str,
    /// A running export or download can be cancelled.
    pub cancellable: bool,
    /// Text of the confirmation before the action (exports: size and license).
    pub confirm: Option<String>,
    /// Second size line, e.g. "+ export toolchain ~350 MB (once)".
    pub size_note: Option<String>,
    /// Last lines of output of a running or failed export.
    pub log: Vec<String>,
    /// A model whose size the hardware profile recommends for this machine
    /// ([`crate::profile::Profile::recommends`]).
    pub recommended: bool,
    /// Installed in another version than the catalog's: the catalog version Download installs
    /// (the UI labels the button Update).
    pub update_available: Option<&'static str>,
}

/// True for a model resource whose size `profile` recommends (the IPcam YOLOv5 models count
/// as `s`).
pub fn recommended_for(profile: &crate::profile::Profile, r: &Resource) -> bool {
    match r.provides {
        catalog::Provides::Model { name, family } => {
            profile.recommends(name, family == crate::model::ModelFamilyKind::Yolo5)
        }
        _ => false,
    }
}

fn kind_str(k: ResourceKind) -> &'static str {
    match k {
        ResourceKind::OpenVinoRuntime => "openvino-runtime",
        ResourceKind::OnnxRuntime(_) => "onnx-runtime",
        ResourceKind::CudaLibs => "cuda-libs",
        ResourceKind::Model => "model",
        ResourceKind::BenchImages => "bench-images",
        ResourceKind::ExportModel => "export-model",
        ResourceKind::Tool => "tool",
    }
}

/// Group keys in display order with their titles.
pub const GROUPS: &[(&str, &str)] = &[
    ("runtimes", "Runtimes"),
    ("gpu-libraries", "GPU libraries"),
    ("models-yolov5", "Models \u{b7} YOLOv5 (IPcam and custom)"),
    ("models-rt-detr", "Models \u{b7} RT-DETRv2"),
    ("models-d-fine", "Models \u{b7} D-FINE"),
    ("models-rf-detr", "Models \u{b7} RF-DETR"),
    (
        "models-export",
        "Models \u{b7} exported on this machine (Ultralytics YOLO, RF-DETR L, DEIMv2)",
    ),
    ("bench-images", "Benchmark images"),
];

/// Group key of the local (unconfigured) model files.
pub const LOCAL_GROUP: &str = "local-models";

/// UI section of a resource: runtimes (OpenVINO, ONNX Runtime flavors incl. DirectML), GPU
/// libraries (NVIDIA CUDA/cuDNN), and models by family, keyed on the id prefix.
pub fn group_of(r: &Resource) -> &'static str {
    match r.kind {
        ResourceKind::OpenVinoRuntime | ResourceKind::OnnxRuntime(_) => "runtimes",
        ResourceKind::CudaLibs => "gpu-libraries",
        ResourceKind::Model => model_group(r.id),
        ResourceKind::BenchImages => "bench-images",
        ResourceKind::ExportModel | ResourceKind::Tool => EXPORT_GROUP,
    }
}

/// Group key of the exported models and their toolchain.
pub const EXPORT_GROUP: &str = "models-export";

fn model_group(id: &str) -> &'static str {
    let id = id.to_ascii_lowercase();
    let name = id.strip_prefix(catalog::MODEL_ID_PREFIX).unwrap_or(&id);
    if name.starts_with("rt-detr") {
        "models-rt-detr"
    } else if name.starts_with("dfine") {
        "models-d-fine"
    } else if name.starts_with("rfdetr") {
        "models-rf-detr"
    } else {
        "models-yolov5"
    }
}

/// Rows of one UI section.
#[derive(Debug, Clone)]
pub struct ResourceGroup {
    pub key: &'static str,
    pub title: &'static str,
    pub rows: Vec<ResourceRow>,
}

/// Split `rows` into the non-empty [`GROUPS`], in display order.
pub fn grouped(rows: Vec<ResourceRow>) -> Vec<ResourceGroup> {
    let mut out: Vec<ResourceGroup> = GROUPS
        .iter()
        .map(|&(key, title)| ResourceGroup {
            key,
            title,
            rows: Vec::new(),
        })
        .collect();
    for r in rows {
        if let Some(g) = out.iter_mut().find(|g| g.key == r.group) {
            g.rows.push(r);
        }
    }
    out.retain(|g| !g.rows.is_empty());
    out
}

/// Id prefix of local model files (`local:yolo26s.onnx`).
pub const LOCAL_ID_PREFIX: &str = "local:";

/// A model file in `models_dir` that no config entry references and that is not a catalog
/// download (those have their own rows).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LocalModel {
    /// `local:<file name>`.
    pub id: String,
    /// File name, e.g. `yolo26s.onnx` (the `.onnx` when both formats exist).
    pub file: String,
    /// Model name it gets when added (the file stem).
    pub name: String,
    /// `onnx` or `openvino-ir`.
    pub format: &'static str,
    /// The other format of the same stem when both exist (`yolo26s.xml`).
    pub also: Option<String>,
    pub size: u64,
    pub size_text: String,
    /// Always `auto` (picked from the model's outputs at load).
    pub family: &'static str,
    /// Full path.
    pub path: String,
    /// What it can run on.
    pub note: &'static str,
    pub group: &'static str,
}

pub(super) fn model_ext(p: &Path) -> Option<&'static str> {
    match p
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("onnx") => Some("onnx"),
        Some("xml") => Some("xml"),
        _ => None,
    }
}

pub(super) fn same_file(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

/// Model files (`.onnx`, `.xml`) in `models_dir` not referenced by any config entry (by stem:
/// a configured `x.xml` hides `x.onnx` too) and not part of a catalog resource. When both
/// `x.xml` and `x.onnx` exist, the `.onnx` is offered (it runs on both runtimes and CoreML).
pub fn local_models(ctx: Option<&ResourcesCtx>, config: &Config) -> Vec<LocalModel> {
    let dir = config.data_path(&config.models_dir);
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let catalog_files: Vec<String> = platform_resources(ctx)
        .into_iter()
        .chain(catalog::MODELS.iter())
        .filter(|r| r.kind == ResourceKind::Model)
        .flat_map(|r| r.parts.iter().map(|p| p.file_name.to_ascii_lowercase()))
        // Exported models have their own rows (with Add to config).
        .chain(
            catalog::EXPORT_MODELS
                .iter()
                .filter_map(catalog::export_files)
                .map(|(onnx, _)| onnx),
        )
        .collect();
    let configured: Vec<PathBuf> = config
        .models
        .iter()
        .map(|m| config.data_path(&m.path))
        .collect();
    // stem -> (onnx, xml)
    let mut stems: std::collections::BTreeMap<String, (Option<PathBuf>, Option<PathBuf>)> =
        Default::default();
    for e in rd.flatten() {
        let p = e.path();
        if !p.is_file() {
            continue;
        }
        let Some(ext) = model_ext(&p) else { continue };
        let Some(stem) = p.file_stem().and_then(|s| s.to_str()).map(str::to_string) else {
            continue;
        };
        let slot = stems.entry(stem).or_default();
        if ext == "onnx" {
            slot.0 = Some(p);
        } else {
            slot.1 = Some(p);
        }
    }
    let mut out = Vec::new();
    for (stem, (onnx, xml)) in stems {
        let files: Vec<&PathBuf> = onnx.iter().chain(xml.iter()).collect();
        let is_catalog = files.iter().any(|f| {
            f.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| catalog_files.contains(&n.to_ascii_lowercase()))
        });
        let is_configured = files
            .iter()
            .any(|f| configured.iter().any(|c| same_file(c, f)));
        if is_catalog || is_configured {
            continue;
        }
        let (main, other) = match (&onnx, &xml) {
            (Some(o), x) => (o, x.as_ref()),
            (None, Some(x)) => (x, None),
            (None, None) => continue,
        };
        let file = main
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let is_onnx = model_ext(main) == Some("onnx");
        let mut size = std::fs::metadata(main).map_or(0, |m| m.len());
        if !is_onnx {
            size += std::fs::metadata(main.with_extension("bin")).map_or(0, |m| m.len());
        }
        out.push(LocalModel {
            id: format!("{LOCAL_ID_PREFIX}{file}"),
            name: stem,
            format: if is_onnx { "onnx" } else { "openvino-ir" },
            also: other.and_then(|o| o.file_name().map(|n| n.to_string_lossy().to_string())),
            size,
            size_text: catalog::format_size(size),
            family: "auto",
            path: main.display().to_string(),
            note: if is_onnx {
                "ONNX: runs on OpenVINO and ONNX Runtime (CoreML, CUDA, DirectML, CPU)"
            } else {
                "OpenVINO IR: OpenVINO devices only (export the .onnx for ONNX Runtime / CoreML)"
            },
            group: LOCAL_GROUP,
            file,
        });
    }
    out
}

/// Every resource of this platform, plus the provisioner's extra (test) models.
pub fn platform_resources(ctx: Option<&ResourcesCtx>) -> Vec<&'static Resource> {
    let hw = crate::backend::detect::hardware();
    let mut out: Vec<&'static Resource> = ctx
        .map(|c| c.provisioner.extra_models().to_vec())
        .unwrap_or_default();
    out.extend(catalog::for_platform(&hw.os, &hw.arch));
    out
}

/// Find a resource by id (case-insensitive) among [`platform_resources`].
pub fn find(ctx: Option<&ResourcesCtx>, id: &str) -> Option<&'static Resource> {
    let id = id.trim();
    platform_resources(ctx)
        .into_iter()
        .find(|r| r.id.eq_ignore_ascii_case(id))
}

/// Install directory of `res` for `config`: models in `models_dir`, runtimes under the data
/// root.
pub fn target_dir(config: &Config, res: &Resource) -> PathBuf {
    match res.kind {
        ResourceKind::Model | ResourceKind::ExportModel => config.data_path(&config.models_dir),
        _ => config.data_root().join(res.dest),
    }
}

pub(super) fn installed_snapshot(ctx: Option<&ResourcesCtx>, config: &Config) -> Installed {
    let mut installed = resolve::detect_installed(config, &config.data_root(), None);
    if let Some(c) = ctx {
        installed.extra_models = c.provisioner.extra_models().to_vec();
        if c.ort_loaded.is_some() {
            installed.active_ort = c.ort_loaded;
        }
    }
    installed
}

pub(super) fn model_installed(res: &Resource, dir: &Path) -> bool {
    manager::is_installed(res, dir) || res.parts.iter().all(|p| dir.join(p.file_name).is_file())
}

pub(super) fn is_installed(res: &Resource, installed: &Installed, config: &Config) -> bool {
    match res.kind {
        ResourceKind::Model => model_installed(res, &target_dir(config, res)),
        ResourceKind::ExportModel => export::is_exported(res, &target_dir(config, res)),
        ResourceKind::BenchImages | ResourceKind::Tool => {
            manager::is_installed(res, &target_dir(config, res))
        }
        _ => installed.has(res.id),
    }
}

/// The model files of `res` in the models directory (exports: the `.onnx` and `.yaml` they write).
pub(super) fn model_paths(res: &Resource, config: &Config) -> Vec<PathBuf> {
    let dir = target_dir(config, res);
    match catalog::export_files(res) {
        Some((onnx, yaml)) => vec![dir.join(onnx), dir.join(yaml)],
        None => res.parts.iter().map(|p| dir.join(p.file_name)).collect(),
    }
}

/// Our install of `res` (manifest names it), not a user-managed directory.
fn ours(res: &Resource, config: &Config) -> bool {
    let dir = target_dir(config, res);
    manager::read_manifest(&manager::manifest_path(res, &dir)).is_some_and(|m| m.id == res.id)
}

/// Enabled config models whose file is part of `res`.
fn models_using(res: &Resource, config: &Config) -> Vec<String> {
    let files = model_paths(res, config);
    config
        .enabled_models()
        .filter(|m| files.contains(&config.data_path(&m.path)))
        .map(|m| m.effective_name())
        .collect()
}

/// Why `res` cannot be removed right now (None = it can).
pub(super) fn remove_blocked(
    ctx: Option<&ResourcesCtx>,
    res: &Resource,
    config: &Config,
    status: Option<&Status>,
) -> Option<String> {
    if status.is_some_and(|s| !s.state.is_final()) {
        return Some("downloading; wait until it finishes".into());
    }
    let restart = "loaded by this process; restart required (stop the service, then remove it)";
    let exporting = ctx
        .and_then(|c| c.provisioner.exporter())
        .is_some_and(|e| e.busy());
    match res.kind {
        ResourceKind::Tool if exporting => {
            return Some("an export is running; cancel it or wait until it finishes".into());
        }
        ResourceKind::ExportModel
            if ctx
                .and_then(|c| c.provisioner.exporter())
                .and_then(|e| e.status(res.id))
                .is_some_and(|s| s.state.is_busy()) =>
        {
            return Some("being exported; cancel it first".into());
        }
        ResourceKind::Model | ResourceKind::ExportModel => {
            let users = models_using(res, config);
            if !users.is_empty() {
                return Some(format!(
                    "used by enabled model(s) {}; disable them first",
                    users.join(", ")
                ));
            }
        }
        ResourceKind::OpenVinoRuntime if ctx.is_some_and(|c| c.openvino_loaded) => {
            return Some(restart.into());
        }
        ResourceKind::OnnxRuntime(f) if ctx.is_some_and(|c| c.ort_loaded == Some(f)) => {
            return Some(restart.into());
        }
        ResourceKind::CudaLibs
            if crate::backend::libs::cuda_libs_preloaded()
                .is_some_and(|d| d == target_dir(config, res)) =>
        {
            return Some(restart.into());
        }
        _ => {}
    }
    if !res.kind.is_model() && res.kind != ResourceKind::Tool && !ours(res, config) {
        return Some("not installed by Blue Onyx Prism (user-managed); remove it by hand".into());
    }
    None
}

/// The latest manager status for resource `id` (any target).
pub(super) fn status_of<'a>(statuses: &'a [Status], id: &str) -> Option<&'a Status> {
    // Prefer an active job over a finished one.
    statuses
        .iter()
        .filter(|s| s.id == id)
        .min_by_key(|s| s.state.is_final())
}

/// Rows for `/v1/resources` and the Resources card.
pub fn rows(ctx: Option<&ResourcesCtx>, config: &Config) -> Vec<ResourceRow> {
    let hw = crate::backend::detect::hardware();
    let statuses: Vec<Status> = ctx
        .and_then(|c| c.provisioner.manager())
        .map(|m| m.statuses())
        .unwrap_or_default();
    let installed = installed_snapshot(ctx, config);
    let res: Resolution = resolve::needed(config, hw, &installed);
    // Once per call, not per export row (RS-14).
    let platform = Platform::current();
    let paths = export::Paths::for_config(config);
    let export_env = ExportEnv {
        toolchain: export::toolchain_state(&paths, platform),
        unsupported: export::unsupported_reason(platform.os, platform.arch),
        paths,
        platform,
    };
    let ort_root = config.data_root().join(crate::backend::libs::ORT_DIR_NAME);
    let active_flavor = crate::backend::libs::read_active_flavor(&ort_root);
    let configured: Vec<PathBuf> = config
        .models
        .iter()
        .map(|m| config.data_path(&m.path))
        .collect();
    let profile = config.profile();

    platform_resources(ctx)
        .into_iter()
        // The DEIMv2 source and the Windows Visual C++ runtime are part of the export toolchain
        // (the `tool:uv` row).
        .filter(|r| r.id != catalog::DEIMV2_SRC_ID && r.id != catalog::VC_RUNTIME_ID)
        .map(|r| {
            if matches!(r.kind, ResourceKind::ExportModel | ResourceKind::Tool) {
                let mut row = export_row(ctx, config, r, &res, &statuses, &configured, &export_env);
                row.recommended = recommended_for(&profile, r);
                return row;
            }
            let status = status_of(&statuses, r.id);
            let on_disk = is_installed(r, &installed, config);
            let update = on_disk && installed.outdated(r);
            let need = res.needs.iter().find(|n| n.id() == r.id);
            let optional = res.optional.iter().any(|n| n.id() == r.id);
            let (state, progress, error, retry_in) = match status.map(|s| &s.state) {
                Some(State::Downloading { bytes, total }) => (
                    "downloading",
                    Some(super::manager::percent(*bytes, *total)),
                    None,
                    None,
                ),
                Some(State::Queued) => ("queued", None, None, None),
                Some(State::Verifying) => ("verifying", None, None, None),
                Some(State::Extracting) => ("extracting", None, None, None),
                Some(State::Failed {
                    error, retry_at, ..
                }) if !on_disk => {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::SystemTime::UNIX_EPOCH)
                        .map_or(0, |d| d.as_secs());
                    (
                        "failed",
                        None,
                        Some(error.clone()),
                        retry_at.map(|t| t.saturating_sub(now)),
                    )
                }
                _ if on_disk => ("installed", None, None, None),
                _ if need.is_some() => ("needed", None, None, None),
                _ if optional => ("optional", None, None, None),
                _ => ("available", None, None, None),
            };
            let flavor = r.flavor();
            let loaded = match r.kind {
                ResourceKind::OpenVinoRuntime => on_disk && ctx.is_some_and(|c| c.openvino_loaded),
                ResourceKind::OnnxRuntime(f) => ctx.is_some_and(|c| c.ort_loaded == Some(f)),
                ResourceKind::CudaLibs => crate::backend::libs::cuda_libs_preloaded().is_some(),
                ResourceKind::Model
                | ResourceKind::BenchImages
                | ResourceKind::ExportModel
                | ResourceKind::Tool => false,
            };
            let state_text = match (status, state) {
                (Some(s), "downloading" | "queued" | "verifying" | "extracting" | "failed") => {
                    s.state.describe()
                }
                (_, "needed") => format!(
                    "needed: {}",
                    need.map_or(String::new(), |n| n.reason.clone())
                ),
                (_, "optional") => "would enable a better device (large download)".into(),
                (_, "installed") if update => format!(
                    "installed {}; {} available (Update replaces it{})",
                    installed.installed_version(r.id).unwrap_or("?"),
                    r.version,
                    match need {
                        Some(n) if loaded => format!(
                            "; needed by {}, used after a restart",
                            n.blocking_models.join(", ")
                        ),
                        Some(n) => format!("; needed by {}", n.blocking_models.join(", ")),
                        None if loaded => "; used after a restart".to_string(),
                        None => String::new(),
                    }
                ),
                (_, s) => s.to_string(),
            };
            let blocked = if on_disk {
                remove_blocked(ctx, r, config, status)
            } else {
                Some("not installed".into())
            };
            let can_add = r.kind == ResourceKind::Model
                && on_disk
                && !r
                    .parts
                    .iter()
                    .any(|p| configured.contains(&target_dir(config, r).join(p.file_name)));
            ResourceRow {
                id: r.id,
                title: r.title,
                description: r.description,
                kind: kind_str(r.kind),
                group: group_of(r),
                version: r.version,
                license: r.license,
                size: r.size(),
                size_text: catalog::format_size(r.size()),
                large: r.is_large(),
                state,
                state_text,
                busy: matches!(state, "downloading" | "queued" | "verifying" | "extracting"),
                progress,
                error,
                retry_in,
                provides: r.provides.labels(),
                blocking: need.map(|n| n.blocking_models.clone()).unwrap_or_default(),
                active: flavor.is_some()
                    && (flavor.map(|f| f.as_str()) == active_flavor.as_deref()
                        || flavor == ctx.and_then(|c| c.ort_loaded)),
                loaded,
                removable: on_disk && blocked.is_none(),
                remove_blocked: if on_disk { blocked } else { None },
                can_add_to_config: can_add,
                target: target_dir(config, r).display().to_string(),
                action: "download",
                cancellable: false,
                confirm: r.is_large().then(|| {
                    format!(
                        "{} is {} ({}). Download it?",
                        r.title,
                        catalog::format_size(r.size()),
                        r.license
                    )
                }),
                size_note: None,
                log: Vec::new(),
                recommended: recommended_for(&profile, r),
                update_available: update.then_some(r.version),
            }
        })
        .collect()
}

/// "~350 MB download, ~1.1 GB on disk".
pub(super) fn toolchain_cost(p: Platform) -> Option<String> {
    catalog::toolchain_estimate(p.os, p.arch).map(|e| {
        format!(
            "~{} download, ~{} on disk",
            catalog::format_size(e.download),
            catalog::format_size(e.disk)
        )
    })
}

/// Confirmation text before exporting `res`.
pub fn export_confirm(res: &Resource, toolchain_ready: bool, p: Platform) -> String {
    let recipe = catalog::export_recipe(res);
    let extra: u64 = catalog::export_extras(res).iter().map(|r| r.size()).sum();
    let size = catalog::format_size(res.size() + extra);
    let what = if toolchain_ready {
        format!("Downloads the {} weights ({size}).", res.title)
    } else {
        format!(
            "Sets up the export toolchain once (uv, Python 3.11, PyTorch CPU, Ultralytics and \
             transformers: {}) and downloads the {} weights ({size}).",
            toolchain_cost(p).unwrap_or_else(|| "large".into()),
            res.title,
        )
    };
    format!(
        "Export {}? {what} It runs {} locally (a few minutes). {}",
        res.title,
        recipe.map_or("the exporter", |r| r.exporter()),
        recipe.map_or("", |r| r.notice())
    )
}

/// What every export row shares, computed once per [`rows`] call.
struct ExportEnv {
    platform: Platform,
    paths: export::Paths,
    toolchain: export::ToolchainState,
    unsupported: Option<String>,
}

/// Rows of the export group: exportable models and the `tool:uv` toolchain row.
fn export_row(
    ctx: Option<&ResourcesCtx>,
    config: &Config,
    r: &'static Resource,
    resolution: &Resolution,
    statuses: &[Status],
    configured: &[PathBuf],
    env: &ExportEnv,
) -> ResourceRow {
    let platform = env.platform;
    let paths = &env.paths;
    let toolchain = &env.toolchain;
    let unsupported = env.unsupported.clone();
    let exporter = ctx.and_then(|c| c.provisioner.exporter());
    let mut row = ResourceRow {
        id: r.id,
        title: r.title,
        description: r.description,
        kind: kind_str(r.kind),
        group: group_of(r),
        version: r.version,
        license: r.license,
        size: r.size(),
        size_text: catalog::format_size(r.size()),
        large: false,
        state: "available",
        state_text: String::new(),
        busy: false,
        progress: None,
        error: None,
        retry_in: None,
        provides: r.provides.labels(),
        blocking: Vec::new(),
        active: false,
        loaded: false,
        removable: false,
        remove_blocked: None,
        can_add_to_config: false,
        target: target_dir(config, r).display().to_string(),
        action: if unsupported.is_some() {
            "none"
        } else if r.kind == ResourceKind::Tool {
            "toolchain"
        } else {
            "export"
        },
        cancellable: false,
        confirm: None,
        size_note: None,
        log: Vec::new(),
        recommended: false,
        update_available: None,
    };
    if r.kind == ResourceKind::Tool {
        let dl = status_of(statuses, r.id).filter(|s| !s.state.is_final());
        let exporting = exporter.as_ref().is_some_and(|e| e.busy());
        row.target = paths.tools.display().to_string();
        row.size_note = toolchain_cost(platform).map(|c| format!("with Python env: {c}"));
        if let Some(s) = dl {
            row.state = "downloading";
            row.busy = true;
            row.state_text = s.state.describe();
            if let State::Downloading { bytes, total } = s.state {
                row.progress = Some(super::manager::percent(bytes, total));
            }
        } else if toolchain.present {
            row.state = "installed";
            row.state_text = match (toolchain.uv, toolchain.env_ready) {
                (true, true) => format!(
                    "installed: uv {}, Python {} environment in {}",
                    catalog::UV_VERSION,
                    export::PYTHON_VERSION,
                    paths.env.display()
                ),
                (true, false) => {
                    "uv installed; the Python environment is set up by the first export".into()
                }
                _ => "partly installed (an export was interrupted); the next export completes it"
                    .into(),
            };
            let blocked = exporting.then(|| "an export is running".to_string());
            row.removable = blocked.is_none();
            row.remove_blocked = blocked;
            row.confirm = Some(format!(
                "Remove the model export toolchain ({})? Exported models stay; a later export \
                 downloads it again.",
                paths.tools.display()
            ));
        } else {
            row.state_text = "not installed; the first model export downloads it".into();
        }
        if let Some(why) = unsupported {
            row.state_text = why;
        }
        return row;
    }

    // An exportable model.
    let on_disk = export::is_exported(r, &target_dir(config, r));
    let st = exporter.as_ref().and_then(|e| e.status(r.id));
    let needed_by: Vec<String> = resolution
        .models
        .iter()
        .filter(|m| m.needs_export == Some(r.id))
        .map(|m| m.model.clone())
        .collect();
    row.blocking = needed_by.clone();
    row.size_note = Some(if toolchain.ready() {
        "weights; export toolchain installed".to_string()
    } else {
        format!(
            "weights + export toolchain once ({})",
            toolchain_cost(platform).unwrap_or_else(|| "large".into())
        )
    });
    row.large = !toolchain.ready();
    row.confirm = Some(export_confirm(r, toolchain.ready(), platform));
    match st.as_ref().map(|s| &s.state) {
        Some(state @ ExportState::Running { percent, .. }) => {
            row.state = "exporting";
            row.busy = true;
            row.progress = Some(*percent);
            row.state_text = state.describe();
        }
        Some(ExportState::Queued) => {
            row.state = "queued";
            row.busy = true;
            row.state_text = "export queued (one export at a time)".into();
        }
        Some(state @ ExportState::Failed { error, .. }) if !on_disk => {
            row.state = "failed";
            row.error = Some(error.clone());
            row.state_text = state.describe();
        }
        _ if on_disk => {
            row.state = "installed";
            row.state_text = if export::manifest(r, &target_dir(config, r)).is_some() {
                "exported".into()
            } else {
                "exported (by hand)".into()
            };
        }
        _ if !needed_by.is_empty() => {
            row.state = "needs-export";
            row.state_text = format!(
                "needs export: used by {} in the config; click Export",
                needed_by.join(", ")
            );
        }
        Some(ExportState::Cancelled) => row.state_text = "export cancelled".into(),
        _ => row.state_text = "not exported".into(),
    }
    if let Some(s) = &st
        && (row.busy || row.state == "failed")
    {
        row.log = s.log.iter().rev().take(6).rev().cloned().collect();
    }
    row.cancellable = row.busy;
    if on_disk {
        let blocked = remove_blocked(ctx, r, config, None);
        row.removable = blocked.is_none();
        row.remove_blocked = blocked;
        row.can_add_to_config = !model_paths(r, config)
            .first()
            .is_some_and(|onnx| configured.contains(onnx));
    }
    if let Some(why) = unsupported
        && !on_disk
    {
        row.state = "unavailable";
        row.state_text = why;
    }
    row
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_rows_marked_recommended_by_size() {
        use crate::profile::HardwareProfile;
        let dir = std::env::temp_dir().join(format!("bo_rec_{}", uuid::Uuid::new_v4()));
        let mut config = Config {
            download_dir: Some(dir.clone()),
            hardware_profile: HardwareProfile::Low,
            ..Config::default()
        };
        let rec = |config: &Config, id: &str| {
            rows(None, config)
                .into_iter()
                .find(|r| r.id == id)
                .map(|r| r.recommended)
        };
        // Low: n/s (IPcam YOLOv5 counts as s); runtimes are never marked.
        assert_eq!(rec(&config, "model:IPcam-general"), Some(true));
        assert_eq!(rec(&config, "model:rt-detrv2-s"), Some(true));
        assert_eq!(rec(&config, "model:rt-detrv2-l"), Some(false));
        assert!(
            rows(None, &config)
                .iter()
                .filter(|r| !matches!(r.kind, "model" | "export-model"))
                .all(|r| !r.recommended)
        );
        config.hardware_profile = HardwareProfile::High;
        assert_eq!(rec(&config, "model:IPcam-general"), Some(false));
        assert_eq!(rec(&config, "model:rt-detrv2-l"), Some(true));
        std::fs::remove_dir_all(&dir).ok();
    }
}
