//! What the web UI and `add-model` do with resources (phase 7.5): the Download / Remove /
//! Add-to-config / Export / Cancel actions behind `POST /v1/resources/*`, the "Add a model by
//! path" form and `add-model`. The rows they act on are built in `super::status`, whose module
//! docs describe when each action is refused.

use super::catalog::{self, Platform, Resource, ResourceKind};
use super::export;
use super::manager::{self, Job, Status};
use super::resolve;
use super::status::{
    LOCAL_ID_PREFIX, ResourcesCtx, config_entries_using, find, installed_snapshot, is_installed,
    model_ext, model_installed, model_paths, remove_blocked, same_file, status_of, target_dir,
    toolchain_cost,
};
use crate::config::Config;
use crate::model::ModelFamilyKind;
use anyhow::Context as _;
use std::path::{Path, PathBuf};

/// Why an action was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionError {
    NotFound(String),
    /// Refused in the current state (409): large without confirmation, in use, downloading.
    Conflict(String),
    Failed(String),
    /// Invalid input (400): empty or malformed fields.
    BadRequest(String),
}

impl ActionError {
    pub fn message(&self) -> &str {
        match self {
            ActionError::NotFound(m)
            | ActionError::Conflict(m)
            | ActionError::Failed(m)
            | ActionError::BadRequest(m) => m,
        }
    }

    /// HTTP status code.
    pub fn status(&self) -> u16 {
        match self {
            ActionError::NotFound(_) => 404,
            ActionError::Conflict(_) => 409,
            ActionError::Failed(_) => 500,
            ActionError::BadRequest(_) => 400,
        }
    }
}

impl std::error::Error for ActionError {}

impl std::fmt::Display for ActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

/// Queue an export of `id` (an exportable model). Without a ready toolchain it needs `confirm` unless
/// `allow_large_downloads` is set. `on_done` runs after a successful export (e.g. add it to the
/// config).
pub fn export(
    ctx: &ResourcesCtx,
    config: &Config,
    id: &str,
    confirm: bool,
    on_done: Option<export::OnDone>,
) -> Result<String, ActionError> {
    let res = find(Some(ctx), id)
        .ok_or_else(|| ActionError::NotFound(format!("unknown resource '{id}'")))?;
    if res.kind != ResourceKind::ExportModel {
        return Err(ActionError::Conflict(format!(
            "{} is not exported locally; use Download",
            res.title
        )));
    }
    let platform = Platform::current();
    if let Some(why) = export::unsupported_reason(platform.os, platform.arch) {
        return Err(ActionError::Conflict(why));
    }
    let paths = export::Paths::for_config(config);
    if export::is_exported(res, &paths.models) {
        return Ok(format!(
            "{} is already exported in {}",
            res.title,
            paths.models.display()
        ));
    }
    let toolchain = export::toolchain_state(&paths, platform);
    if !toolchain.ready() && !confirm && !config.allow_large_downloads {
        return Err(ActionError::Conflict(format!(
            "exporting {} sets up the export toolchain first ({}) and runs {} ({}); confirm it",
            res.title,
            toolchain_cost(platform).unwrap_or_else(|| "large".into()),
            catalog::export_recipe(res).map_or("the exporter", |r| r.exporter()),
            res.license
        )));
    }
    let exporter = ctx.provisioner.exporter_for(config);
    exporter.forget(res.id);
    let mut job = export::ExportJob::new(res, paths);
    job.platform = platform;
    exporter
        .enqueue(export::ExportRequest { job, on_done })
        .map_err(ActionError::Conflict)?;
    Ok(format!(
        "exporting {} ({}): progress is shown below and in the Logs",
        res.title, res.license
    ))
}

/// Cancel a queued or running export of `id`.
pub fn cancel_export(ctx: &ResourcesCtx, id: &str) -> Result<String, ActionError> {
    let res = find(Some(ctx), id)
        .ok_or_else(|| ActionError::NotFound(format!("unknown resource '{id}'")))?;
    match ctx.provisioner.exporter() {
        Some(e) if e.cancel(res.id) => Ok(format!("cancelling the export of {}", res.title)),
        _ => Err(ActionError::Conflict(format!(
            "no export of {} is running",
            res.title
        ))),
    }
}

/// Queue `id` for download. Large resources need `confirm_large` unless the config allows them.
/// Returns a message for the user.
pub fn download(
    ctx: &ResourcesCtx,
    config: &Config,
    id: &str,
    confirm_large: bool,
) -> Result<String, ActionError> {
    let res = find(Some(ctx), id)
        .ok_or_else(|| ActionError::NotFound(format!("unknown resource '{id}'")))?;
    if res.kind == ResourceKind::ExportModel {
        return Err(ActionError::Conflict(format!(
            "{} cannot be downloaded ({}); use Export",
            res.title, res.license
        )));
    }
    let installed = installed_snapshot(Some(ctx), config);
    let update = installed.outdated(res);
    if is_installed(res, &installed, config) && !update {
        return Ok(format!("{} is already installed", res.title));
    }
    if res.is_large() && !confirm_large && !config.allow_large_downloads {
        return Err(ActionError::Conflict(format!(
            "{} is {} ({}); confirm the large download",
            res.title,
            catalog::format_size(res.size()),
            res.license
        )));
    }
    let mut job = match res.kind {
        ResourceKind::Model => Job::into_dir(res, target_dir(config, res)),
        _ => Job::new(res, &config.data_root()),
    };
    // A new flavor becomes active only when none is; an update keeps the active one active.
    job.activate_ort = res.flavor().is_some()
        && !installed.ort_pinned
        && (installed.active_ort.is_none() || installed.active_ort == res.flavor());
    let manager = ctx.provisioner.manager_for(config);
    manager.forget(&job.key());
    manager.enqueue(job);
    Ok(format!(
        "downloading {} ({})",
        res.describe(),
        match (res.kind, update) {
            (ResourceKind::Model, _) => "add it to the config when it is installed".to_string(),
            (_, true) => format!(
                "replaces the installed {}; used after the next restart",
                installed.installed_version(res.id).unwrap_or("version")
            ),
            (_, false) => "used after the next restart".to_string(),
        }
    ))
}

/// What [`remove`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Removed {
    /// "removed <title> from <dir>".
    pub message: String,
    /// Names of the (disabled) config entries whose model file was removed: the caller drops
    /// them with [`drop_config_entries`] through its config writer and tells the user with
    /// [`Removed::describe`]. Enabled entries block the removal instead.
    pub config_entries: Vec<String>,
}

impl Removed {
    fn files(message: String) -> Self {
        Removed {
            message,
            config_entries: Vec::new(),
        }
    }

    /// The answer after `dropped` (from [`drop_config_entries`]) left the config:
    /// "removed X from <dir> and its config entry 'x'".
    pub fn describe(&self, dropped: &[String]) -> String {
        match dropped {
            [] => self.message.clone(),
            [one] => format!("{} and its config entry '{one}'", self.message),
            many => format!(
                "{} and its config entries {}",
                self.message,
                many.iter()
                    .map(|n| format!("'{n}'"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

/// Drop the config entries named in `names` (normalized names) that are not enabled; a
/// `default_model` naming one of them is cleared (the first enabled model serves
/// `/v1/vision/detection`, as before: a disabled entry never did). Returns the dropped names.
pub fn drop_config_entries(config: &mut Config, names: &[String]) -> Vec<String> {
    let wanted: Vec<String> = names
        .iter()
        .map(|n| crate::registry::normalize_name(n))
        .collect();
    let mut dropped = Vec::new();
    config.models.retain(|m| {
        let name = m.effective_name();
        let go = !m.enabled && wanted.contains(&crate::registry::normalize_name(&name));
        if go {
            dropped.push(name);
        }
        !go
    });
    if let Some(d) = &config.default_model
        && dropped
            .iter()
            .any(|n| crate::registry::normalize_name(n) == crate::registry::normalize_name(d))
    {
        config.default_model = None;
    }
    dropped
}

/// What [`remove_config_entry`] took out of the config. The files go only after the config is
/// saved ([`EntryRemoved::delete_files`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryRemoved {
    pub name: String,
    /// Files of the entry that are ours to delete: inside the models folder, not a catalog
    /// download or export, not used by another entry.
    pub files: Vec<PathBuf>,
    /// Why the model file stays, when it exists and is not in `files`.
    pub kept: Option<String>,
}

impl EntryRemoved {
    /// Delete [`EntryRemoved::files`] and describe the outcome: "removed 'x' from the config
    /// and deleted ...". A file that cannot be deleted is reported, not an error (the entry is
    /// already gone).
    pub fn delete_files(&self) -> String {
        let mut deleted = Vec::new();
        let mut failed = Vec::new();
        for f in &self.files {
            match std::fs::remove_file(f) {
                Ok(()) => deleted.push(f.display().to_string()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    tracing::warn!("deleting {}: {e}", f.display());
                    failed.push(format!("{} ({e})", f.display()));
                }
            }
        }
        let mut msg = format!("removed '{}' from the config", self.name);
        if !deleted.is_empty() {
            msg.push_str(&format!(" and deleted {}", deleted.join(", ")));
        }
        if !failed.is_empty() {
            msg.push_str(&format!(
                "; could not delete {} (still in use? delete it by hand)",
                failed.join(", ")
            ));
        }
        if let Some(why) = &self.kept {
            msg.push_str(&format!("; {why}"));
        }
        msg
    }
}

/// The files of entry `m` that its Remove may delete (see [`EntryRemoved::files`]), or why the
/// model file stays. `others` are the entries left in the config.
fn entry_files(
    config: &Config,
    m: &crate::config::ModelConfig,
    others: &[crate::config::ModelConfig],
) -> (Vec<PathBuf>, Option<String>) {
    let model = config.data_path(&m.path);
    if !model.is_file() {
        return (Vec::new(), None);
    }
    let models_dir = config.data_path(&config.models_dir);
    if !under_root(&model, &models_dir) {
        return (
            Vec::new(),
            Some(format!(
                "its file {} is outside the models folder {}, so it stays",
                model.display(),
                models_dir.display()
            )),
        );
    }
    let file_name = model
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if super::status::catalog_model_files(None).contains(&file_name) {
        return (
            Vec::new(),
            Some(format!(
                "its file {} is a catalog model and stays (remove it under Resources)",
                model.display()
            )),
        );
    }
    let in_use: Vec<PathBuf> = others
        .iter()
        .flat_map(|o| {
            let p = config.data_path(&o.path);
            let mut v = vec![p.with_extension("bin"), p.with_extension("yaml"), p];
            v.extend(o.classes.as_ref().map(|c| config.data_path(c)));
            v
        })
        .collect();
    if in_use.iter().any(|u| same_file(u, &model)) {
        return (
            Vec::new(),
            Some(format!(
                "its file {} stays (another config entry uses it)",
                model.display()
            )),
        );
    }
    let stem = model
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let mut files = vec![model.clone()];
    if model_ext(&model) == Some("xml") {
        files.push(model.with_extension("bin"));
    }
    files.push(model.with_extension("yaml"));
    files.push(model.with_file_name(format!("{stem}.provenance.json")));
    if let Some(c) = &m.classes {
        let c = config.data_path(c);
        if under_root(&c, &models_dir) && model_ext(&c).is_none() {
            files.push(c);
        }
    }
    files.retain(|f| f.is_file() && !in_use.iter().any(|u| same_file(u, f)));
    files.dedup_by(|a, b| same_file(a, b));
    (files, None)
}

/// Remove the config entry `name` (the Models card's Remove): only a disabled entry. Its files
/// are deleted by [`EntryRemoved::delete_files`] once the config is saved.
pub fn remove_config_entry(config: &mut Config, name: &str) -> Result<EntryRemoved, ActionError> {
    let norm = crate::registry::normalize_name(name.trim());
    let Some(m) = config
        .models
        .iter()
        .find(|m| crate::registry::normalize_name(&m.effective_name()) == norm)
    else {
        return Err(ActionError::NotFound(format!(
            "no model named '{}' in the config; reload the page",
            name.trim()
        )));
    };
    let shown = m.effective_name();
    if m.enabled {
        return Err(ActionError::Conflict(format!(
            "'{shown}' is enabled; disable it first (untick Load and save), then remove it"
        )));
    }
    let entry = m.clone();
    let dropped = drop_config_entries(config, std::slice::from_ref(&shown));
    debug_assert_eq!(dropped.len(), 1);
    let (files, kept) = entry_files(config, &entry, &config.models);
    Ok(EntryRemoved {
        name: shown,
        files,
        kept,
    })
}

/// Delete our install of `id` (see the module docs for when it is refused). For a model, the
/// disabled config entries using its files are returned in [`Removed::config_entries`].
pub fn remove(
    ctx: Option<&ResourcesCtx>,
    config: &Config,
    id: &str,
) -> Result<Removed, ActionError> {
    let res =
        find(ctx, id).ok_or_else(|| ActionError::NotFound(format!("unknown resource '{id}'")))?;
    let installed = installed_snapshot(ctx, config);
    let platform = Platform::current();
    let paths = export::Paths::for_config(config);
    let cache_dir = config.cache_dir_path();
    let present = match res.kind {
        ResourceKind::Tool => export::toolchain_state(&paths, platform).present,
        // Also an outdated export or what an earlier export left behind.
        ResourceKind::ExportModel => {
            !export::export_leftovers(res, &paths, cache_dir.as_deref()).is_empty()
        }
        _ => is_installed(res, &installed, config),
    };
    if !present {
        return Err(ActionError::Conflict(format!(
            "{} is not installed",
            res.title
        )));
    }
    let statuses: Vec<Status> = ctx
        .and_then(|c| c.provisioner.manager())
        .map(|m| m.statuses())
        .unwrap_or_default();
    if let Some(why) = remove_blocked(ctx, res, config, status_of(&statuses, res.id)) {
        return Err(ActionError::Conflict(format!(
            "cannot remove {}: {why}",
            res.title
        )));
    }
    // Before the files go (the comparison also resolves other spellings of existing files).
    let config_entries: Vec<String> = config_entries_using(res, config)
        .into_iter()
        .filter(|(_, enabled)| !enabled)
        .map(|(n, _)| n)
        .collect();
    let dir = target_dir(config, res);
    let fail =
        |e: std::io::Error, p: &Path| ActionError::Failed(format!("removing {}: {e}", p.display()));
    match res.kind {
        ResourceKind::ExportModel => {
            export::remove_export(res, &paths, cache_dir.as_deref())
                .map_err(|e| ActionError::Failed(format!("{e:#}")))?;
            if let Some(e) = ctx.and_then(|c| c.provisioner.exporter()) {
                e.forget(res.id);
            }
        }
        ResourceKind::Tool => {
            export::remove_toolchain(&paths).map_err(|e| ActionError::Failed(format!("{e:#}")))?;
            // The manager remembers the uv and weights downloads as installed.
            if let Some(m) = ctx.and_then(|c| c.provisioner.manager()) {
                for s in m.statuses().iter().filter(|s| {
                    s.id == res.id
                        || s.id == catalog::DEIMV2_SRC_ID
                        || s.id == catalog::VC_RUNTIME_ID
                        || catalog::export_model(s.id).is_some()
                }) {
                    m.forget(&s.key);
                }
            }
            return Ok(Removed::files(format!(
                "removed the model export toolchain from {}",
                paths.tools.display()
            )));
        }
        ResourceKind::Model => {
            for p in res.parts {
                let f = dir.join(p.file_name);
                if f.exists() {
                    std::fs::remove_file(&f).map_err(|e| fail(e, &f))?;
                }
            }
            let m = manager::manifest_path(res, &dir);
            let _ = std::fs::remove_file(m);
        }
        _ => {
            std::fs::remove_dir_all(&dir).map_err(|e| fail(e, &dir))?;
            if let (Some(f), Some(parent)) = (res.flavor(), dir.parent())
                && crate::backend::libs::read_active_flavor(parent).as_deref() == Some(f.as_str())
            {
                let _ = std::fs::remove_file(parent.join(crate::backend::libs::ORT_ACTIVE_FILE));
            }
        }
    }
    if let Some(m) = ctx.and_then(|c| c.provisioner.manager()) {
        for s in m.statuses().iter().filter(|s| s.id == res.id) {
            m.forget(&s.key);
        }
    }
    Ok(Removed {
        message: format!("removed {} from {}", res.title, dir.display()),
        config_entries,
    })
}

/// Append installed catalog model `id` to `config.models` (`download-models --add-to-config`
/// rules: disabled when another model is enabled). Returns the outcome text.
pub fn add_to_config(
    ctx: Option<&ResourcesCtx>,
    config: &mut Config,
    id: &str,
) -> Result<String, ActionError> {
    if let Some(file) = catalog::strip_prefix_ci(id.trim(), LOCAL_ID_PREFIX) {
        return add_local_to_config(config, file);
    }
    let res =
        find(ctx, id).ok_or_else(|| ActionError::NotFound(format!("unknown resource '{id}'")))?;
    if res.kind == ResourceKind::ExportModel {
        return add_export_to_config(config, res);
    }
    if res.kind != ResourceKind::Model {
        return Err(ActionError::Conflict(format!(
            "{} is not a model",
            res.title
        )));
    }
    let dir = target_dir(config, res);
    if !model_installed(res, &dir) {
        return Err(ActionError::Conflict(format!(
            "{} is not installed; download it first",
            res.title
        )));
    }
    let files: Vec<PathBuf> = res.parts.iter().map(|p| dir.join(p.file_name)).collect();
    let mut models = crate::download::model_configs(&files, &config.data_root());
    if models.is_empty()
        && let catalog::Provides::Model { name, family } = res.provides
    {
        // Not in the static catalog (e.g. an extra test model): same shape by hand.
        let Some(first) = files.first() else {
            return Err(ActionError::Conflict(format!("{} has no files", res.title)));
        };
        let rel = |p: &Path| {
            p.strip_prefix(config.data_root())
                .map(Path::to_path_buf)
                .unwrap_or_else(|_| p.to_path_buf())
        };
        models.push(crate::config::ModelConfig {
            name: Some(name.to_string()),
            path: rel(first),
            family,
            classes: files.get(1).map(|p| rel(p)),
            ..Default::default()
        });
    }
    let out = crate::download::add_to_config(config, models);
    Ok(describe_outcomes(&out))
}

/// Append an exported model: `models/<name>.onnx` with its class file and the catalog family.
fn add_export_to_config(config: &mut Config, res: &Resource) -> Result<String, ActionError> {
    let dir = target_dir(config, res);
    if !export::is_exported(res, &dir) {
        return Err(ActionError::Conflict(format!(
            "{} is not exported yet; export it first",
            res.title
        )));
    }
    let files = model_paths(res, config);
    let Some(first) = files.first() else {
        return Err(ActionError::Conflict(format!("{} has no files", res.title)));
    };
    let rel = |p: &Path| {
        p.strip_prefix(config.data_root())
            .map(Path::to_path_buf)
            .unwrap_or_else(|_| p.to_path_buf())
    };
    let catalog::Provides::Model { name, family } = res.provides else {
        return Err(ActionError::Conflict(format!(
            "{} is not a model",
            res.title
        )));
    };
    let out = crate::download::add_to_config(
        config,
        vec![crate::config::ModelConfig {
            name: Some(name.to_string()),
            path: rel(first),
            family,
            classes: files.get(1).filter(|p| p.is_file()).map(|p| rel(p)),
            ..Default::default()
        }],
    );
    Ok(describe_outcomes(&out))
}

/// Append model file `file` (a plain file name in `models_dir`) with family `auto`.
fn add_local_to_config(config: &mut Config, file: &str) -> Result<String, ActionError> {
    let file = file.trim();
    let plain = !file.is_empty()
        && !file.contains(['/', '\\'])
        && file != "."
        && file != ".."
        && Path::new(file).file_name().and_then(|n| n.to_str()) == Some(file);
    if !plain || model_ext(Path::new(file)).is_none() {
        return Err(ActionError::NotFound(format!(
            "'{file}' is not a model file name (.onnx or .xml in the models directory)"
        )));
    }
    let models_dir = config.data_path(&config.models_dir);
    let full = models_dir.join(file);
    if !full.is_file() {
        return Err(ActionError::NotFound(format!(
            "{} does not exist",
            full.display()
        )));
    }
    // Relative to the data root like downloads (`models/x.onnx`), absolute otherwise.
    let path = full
        .strip_prefix(config.data_root())
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| full.clone());
    let name = Path::new(file)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| file.to_string());
    let out = crate::download::add_to_config(
        config,
        vec![crate::config::ModelConfig {
            name: Some(name),
            path,
            family: crate::model::ModelFamilyKind::Auto,
            ..Default::default()
        }],
    );
    Ok(describe_outcomes(&out))
}

/// Row id of the "Add a model by path" form (notices land next to it).
pub const CUSTOM_MODEL_ROW: &str = "custom-model";

/// A custom model to add to the config (the form / `add-model` fields).
#[derive(Debug, Clone, Default)]
pub struct CustomModel {
    /// A `.onnx` / `.xml` file or the directory holding one.
    pub path: String,
    pub name: Option<String>,
    pub family: ModelFamilyKind,
    /// YAML with the class names.
    pub classes: Option<String>,
}

/// Trim and strip one pair of surrounding quotes ("Copy as path" on Windows).
fn unquote(s: &str) -> &str {
    let s = s.trim();
    for q in ['"', '\''] {
        if s.len() >= 2 && s.starts_with(q) && s.ends_with(q) {
            return s[1..s.len() - 1].trim();
        }
    }
    s
}

fn model_files_in(dir: &Path) -> Vec<(String, PathBuf)> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && model_ext(p).is_some())
        .collect();
    files.sort();
    // One entry per stem, `.onnx` preferred over `.xml`.
    let mut by_stem: Vec<(String, PathBuf)> = Vec::new();
    for f in files {
        let stem = f
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        match by_stem.iter_mut().find(|(s, _)| *s == stem) {
            Some(e) => {
                if model_ext(&f) == Some("onnx") {
                    e.1 = f;
                }
            }
            None => by_stem.push((stem, f)),
        }
    }
    by_stem
}

/// Largest class-names YAML "Add a model by path" parses (RS-16: the path is admin input).
pub const MAX_CLASSES_FILE_BYTES: u64 = 1024 * 1024;

/// Whether `p` is under `root` (either spelling: as given or canonical).
fn under_root(p: &Path, root: &Path) -> bool {
    if p.starts_with(root) {
        return true;
    }
    match (p.canonicalize(), root.canonicalize()) {
        (Ok(a), Ok(b)) => a.starts_with(b),
        _ => false,
    }
}

/// `load_class_names` behind a size cap, so the form cannot make the service parse an arbitrary
/// large file as YAML.
fn load_classes_capped(p: &Path) -> anyhow::Result<Vec<String>> {
    let len = std::fs::metadata(p)
        .with_context(|| format!("reading {}", p.display()))?
        .len();
    if len > MAX_CLASSES_FILE_BYTES {
        anyhow::bail!(
            "classes file {} is {len} bytes; the limit is {MAX_CLASSES_FILE_BYTES}",
            p.display()
        );
    }
    crate::model::classes::load_class_names(p)
}

/// The single model file in `dir` (or its `onnx/` subdirectory). The names found are only
/// echoed when `dir` is under the data root (RS-16: no directory listing of arbitrary paths
/// in error messages).
fn pick_model_in_dir(dir: &Path, list_names: bool) -> Result<PathBuf, ActionError> {
    let mut found = model_files_in(dir);
    if found.is_empty() {
        found = model_files_in(&dir.join("onnx"));
    }
    match found.len() {
        0 => Err(ActionError::NotFound(format!(
            "no .onnx or .xml model in {}",
            dir.display()
        ))),
        1 => Ok(found.remove(0).1),
        n => {
            let names: Vec<String> = found
                .iter()
                .filter_map(|(_, p)| p.file_name().map(|n| n.to_string_lossy().to_string()))
                .collect();
            let which = if list_names {
                format!(" ({})", names.join(", "))
            } else {
                String::new()
            };
            Err(ActionError::Conflict(format!(
                "{} holds {n} model files{which}; give the path of the model file instead",
                dir.display()
            )))
        }
    }
}

/// Add a custom model by file or directory (the config page's "Add a model by path" form and
/// `add-model`). Duplicates by name or file are errors, not skips.
pub fn add_custom_model(config: &mut Config, req: &CustomModel) -> Result<String, ActionError> {
    let raw = unquote(&req.path);
    if raw.is_empty() {
        return Err(ActionError::BadRequest("the model path is empty".into()));
    }
    let given = config.data_path(Path::new(raw));
    let file = if given.is_dir() {
        pick_model_in_dir(&given, under_root(&given, &config.data_root()))?
    } else if given.is_file() {
        if model_ext(&given).is_none() {
            return Err(ActionError::BadRequest(format!(
                "{} is not a model file (.onnx or .xml)",
                given.display()
            )));
        }
        given
    } else {
        return Err(ActionError::NotFound(format!(
            "{} does not exist",
            given.display()
        )));
    };
    if model_ext(&file) == Some("xml") && !file.with_extension("bin").is_file() {
        return Err(ActionError::NotFound(format!(
            "{} needs its weights file {} beside it",
            file.display(),
            file.with_extension("bin").display()
        )));
    }
    let stem = file
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    let name = req
        .name
        .as_deref()
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| stem.clone());
    if name.contains(['/', '\\', '?', '#']) || name.chars().any(char::is_control) {
        return Err(ActionError::BadRequest(format!(
            "invalid model name '{name}': it is part of a URL, so it must not contain / \\ ? # or control characters"
        )));
    }

    // Classes: explicit file, else a `<stem>.yaml` sidecar (found at load), else a well-known
    // file beside the model.
    let classes = match req
        .classes
        .as_deref()
        .map(unquote)
        .filter(|c| !c.is_empty())
    {
        Some(c) => {
            let p = config.data_path(Path::new(c));
            if !p.is_file() {
                return Err(ActionError::NotFound(format!(
                    "classes file {} does not exist",
                    p.display()
                )));
            }
            load_classes_capped(&p).map_err(|e| ActionError::BadRequest(format!("{e:#}")))?;
            Some(p)
        }
        None => {
            let dir = file.parent().unwrap_or_else(|| Path::new("."));
            if file.with_extension("yaml").is_file() {
                None
            } else {
                let ok: Vec<PathBuf> = ["classes.yaml", "classes.yml", "data.yaml", "names.yaml"]
                    .iter()
                    .map(|n| dir.join(n))
                    .filter(|p| p.is_file() && load_classes_capped(p).is_ok())
                    .collect();
                match ok.as_slice() {
                    [only] => Some(only.clone()),
                    _ => None,
                }
            }
        }
    };

    // Duplicates are errors here (add_to_config would skip them silently).
    let norm = crate::registry::normalize_name(&name);
    for m in &config.models {
        if crate::registry::normalize_name(&m.effective_name()) == norm {
            return Err(ActionError::Conflict(format!(
                "a model named '{name}' is already in the config"
            )));
        }
        if same_file(&config.data_path(&m.path), &file) {
            return Err(ActionError::Conflict(format!(
                "{} is already in the config as '{}'",
                file.display(),
                m.effective_name()
            )));
        }
    }

    let root = config.data_root();
    let root_canon = root.canonicalize().ok();
    let rel = |p: &Path| -> PathBuf {
        for base in [Some(root.as_path()), root_canon.as_deref()]
            .into_iter()
            .flatten()
        {
            if let Ok(r) = p.strip_prefix(base) {
                return r.to_path_buf();
            }
        }
        p.to_path_buf()
    };
    let out = crate::download::add_to_config(
        config,
        vec![crate::config::ModelConfig {
            name: Some(name),
            path: rel(&file),
            family: req.family,
            classes: classes.map(|c| rel(&c)),
            ..Default::default()
        }],
    );
    let mut msg = describe_outcomes(&out);
    if req.family != ModelFamilyKind::Auto {
        msg.push_str(&format!(" (family {})", req.family));
    }
    Ok(msg)
}

fn describe_outcomes(out: &[crate::download::AddOutcome]) -> String {
    out.iter()
        .map(|o| match o {
            crate::download::AddOutcome::Added {
                name,
                enabled: true,
            } => {
                format!("added '{name}' to the config (enabled; restart to load it)")
            }
            crate::download::AddOutcome::Added { name, .. } => {
                format!("added '{name}' to the config (disabled; enable it in the Models card)")
            }
            crate::download::AddOutcome::Skipped { name } => {
                format!("'{name}' is already in the config")
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Download what `config` needs (respecting `allow_large_downloads`): used after the config
/// page changes a device to a downloadable option. Returns the queued resource titles.
pub fn queue_needs(ctx: &ResourcesCtx, config: &Config) -> Vec<String> {
    let hw = crate::backend::detect::hardware();
    let installed = installed_snapshot(Some(ctx), config);
    let r = resolve::needed(config, hw, &installed);
    if r.needs.is_empty() {
        return Vec::new();
    }
    let manager = ctx.provisioner.manager_for(config);
    manager::jobs_for(&r, &r.needs, &config.data_root())
        .into_iter()
        .map(|j| {
            let title = j.resource.describe();
            manager.forget(&j.key());
            manager.enqueue(j);
            title
        })
        .collect()
}

#[cfg(test)]
mod custom_model_tests {
    use super::*;
    use crate::config::ModelConfig;

    struct Dirs {
        root: PathBuf,
        outside: PathBuf,
    }

    impl Drop for Dirs {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
            let _ = std::fs::remove_dir_all(&self.outside);
        }
    }

    fn setup() -> (Dirs, Config) {
        let base = std::env::temp_dir().join(format!("bop-custom-{}", uuid::Uuid::new_v4()));
        let root = base.join("root");
        let outside = base.join("outside");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let config = Config {
            download_dir: Some(root.clone()),
            ..Default::default()
        };
        (
            Dirs {
                root: base,
                outside,
            },
            config,
        )
    }

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"x").unwrap();
    }

    fn req(path: &Path) -> CustomModel {
        CustomModel {
            path: path.to_string_lossy().to_string(),
            ..Default::default()
        }
    }

    const YAML: &str = "names:\n- a\n- b\n";

    fn entry(name: &str, path: &Path, enabled: bool) -> ModelConfig {
        ModelConfig {
            name: Some(name.into()),
            path: path.to_path_buf(),
            enabled,
            ..Default::default()
        }
    }

    /// The Models card's Remove deletes a trained model's files in the models folder (after the
    /// save), keeps files outside it, catalog models and files another entry uses.
    #[test]
    fn remove_entry_deletes_owned_files() {
        let (d, mut c) = setup();
        let models = c.data_path(&c.models_dir);
        let mine = models.join("mine.onnx");
        let sidecars = [
            models.join("mine.yaml"),
            models.join("mine.provenance.json"),
        ];
        let shared = models.join("classes.yaml");
        let other = models.join("other.onnx");
        let outside = d.outside.join("ext.onnx");
        for f in [&mine, &shared, &other, &outside]
            .into_iter()
            .chain(sidecars.iter())
        {
            touch(f);
        }
        c.models = vec![
            entry("keep", &other, true),
            ModelConfig {
                classes: Some(shared.clone()),
                ..entry("mine", &mine, false)
            },
            ModelConfig {
                classes: Some(shared.clone()),
                ..entry("other2", &models.join("x.onnx"), false)
            },
            entry("ext", &outside, false),
        ];

        let r = remove_config_entry(&mut c, "mine").unwrap();
        assert!(mine.exists(), "nothing is deleted before the save");
        let mut want = vec![mine.clone()];
        want.extend(sidecars.iter().cloned());
        assert_eq!(r.files, want, "the shared classes file stays");
        let msg = r.delete_files();
        assert!(
            msg.starts_with("removed 'mine' from the config and deleted "),
            "{msg}"
        );
        assert!(!mine.exists() && sidecars.iter().all(|f| !f.exists()));
        assert!(shared.exists() && other.exists());

        let r = remove_config_entry(&mut c, "ext").unwrap();
        assert!(r.files.is_empty());
        assert!(r.delete_files().contains("outside the models folder"));
        assert!(outside.exists());

        let cat = models.join(catalog::MODELS[0].parts[0].file_name);
        touch(&cat);
        c.models.push(entry("cat", &cat, false));
        let r = remove_config_entry(&mut c, "cat").unwrap();
        assert!(r.files.is_empty() && r.kept.unwrap().contains("catalog model"));
        assert!(cat.exists());
    }

    #[test]
    fn file_path_and_family() {
        let (d, mut c) = setup();
        let f = d.outside.join("mine.onnx");
        touch(&f);
        let r = CustomModel {
            family: ModelFamilyKind::RfDetr,
            ..req(&f)
        };
        let msg = add_custom_model(&mut c, &r).unwrap();
        assert!(
            msg.contains("added 'mine'") && msg.contains("family rfdetr"),
            "{msg}"
        );
        assert_eq!(c.models[0].path, f);
        assert_eq!(c.models[0].family, ModelFamilyKind::RfDetr);
        assert_eq!(c.models[0].classes, None);
    }

    #[test]
    fn dir_with_one_model_and_onnx_preferred() {
        let (d, mut c) = setup();
        touch(&d.outside.join("x.onnx"));
        touch(&d.outside.join("x.xml"));
        touch(&d.outside.join("x.bin"));
        add_custom_model(&mut c, &req(&d.outside)).unwrap();
        assert_eq!(c.models[0].path, d.outside.join("x.onnx"));
    }

    #[test]
    fn dir_with_two_stems_conflicts() {
        let (d, mut c) = setup();
        // Outside the data root the names are not listed (RS-16), only the count.
        touch(&d.outside.join("a.onnx"));
        touch(&d.outside.join("b.onnx"));
        let e = add_custom_model(&mut c, &req(&d.outside)).unwrap_err();
        assert!(matches!(e, ActionError::Conflict(_)), "{e}");
        assert!(!e.message().contains("a.onnx"), "{e}");
        assert!(e.message().contains("2 model files"), "{e}");
        // Under the data root they are.
        let inside = c.data_root().join("custom");
        touch(&inside.join("a.onnx"));
        touch(&inside.join("b.onnx"));
        let e = add_custom_model(&mut c, &req(&inside)).unwrap_err();
        assert!(matches!(e, ActionError::Conflict(_)), "{e}");
        assert!(
            e.message().contains("a.onnx") && e.message().contains("file"),
            "{e}"
        );
        let e = add_custom_model(&mut c, &req(Path::new("custom"))).unwrap_err();
        assert!(e.message().contains("b.onnx"), "{e}");
    }

    #[test]
    fn classes_file_size_is_capped() {
        let (d, mut c) = setup();
        let f = d.outside.join("m.onnx");
        touch(&f);
        let big = d.outside.join("big.yaml");
        let mut text = String::from("names:\n");
        while (text.len() as u64) <= MAX_CLASSES_FILE_BYTES {
            text.push_str("- aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n");
        }
        std::fs::write(&big, &text).unwrap();
        let r = CustomModel {
            classes: Some(big.to_string_lossy().to_string()),
            ..req(&f)
        };
        let e = add_custom_model(&mut c, &r).unwrap_err();
        assert_eq!(e.status(), 400);
        assert!(e.message().contains("limit"), "{e}");
        // An oversized sidecar is ignored instead of adopted.
        std::fs::write(d.outside.join("data.yaml"), &text).unwrap();
        add_custom_model(&mut c, &req(&f)).unwrap();
        assert_eq!(c.models[0].classes, None);
    }

    #[test]
    fn onnx_subdir_is_scanned() {
        let (d, mut c) = setup();
        touch(&d.outside.join("onnx/model.onnx"));
        add_custom_model(&mut c, &req(&d.outside)).unwrap();
        assert_eq!(c.models[0].path, d.outside.join("onnx/model.onnx"));
        assert_eq!(c.models[0].effective_name(), "model");
    }

    #[test]
    fn missing_and_empty_dir_not_found() {
        let (d, mut c) = setup();
        let e = add_custom_model(&mut c, &req(&d.outside.join("nope.onnx"))).unwrap_err();
        assert!(matches!(e, ActionError::NotFound(_)));
        let e = add_custom_model(&mut c, &req(&d.outside)).unwrap_err();
        assert!(matches!(e, ActionError::NotFound(_)));
        assert!(e.message().contains("no .onnx or .xml model"));
        let e = add_custom_model(&mut c, &req(Path::new("  "))).unwrap_err();
        assert_eq!(e.status(), 400);
    }

    #[test]
    fn xml_needs_bin() {
        let (d, mut c) = setup();
        let f = d.outside.join("ir.xml");
        touch(&f);
        let e = add_custom_model(&mut c, &req(&f)).unwrap_err();
        assert!(matches!(e, ActionError::NotFound(_)), "{e}");
        touch(&d.outside.join("ir.bin"));
        add_custom_model(&mut c, &req(&f)).unwrap();
    }

    #[test]
    fn wrong_extension_rejected() {
        let (d, mut c) = setup();
        let f = d.outside.join("notes.txt");
        touch(&f);
        assert_eq!(
            add_custom_model(&mut c, &req(&f)).unwrap_err().status(),
            400
        );
    }

    #[test]
    fn quoted_path() {
        let (d, mut c) = setup();
        let f = d.outside.join("q.onnx");
        touch(&f);
        for q in ['"', '\''] {
            let mut c2 = c.clone();
            let r = CustomModel {
                path: format!("  {q}{}{q} ", f.display()),
                ..Default::default()
            };
            add_custom_model(&mut c2, &r).unwrap();
        }
        add_custom_model(&mut c, &req(&f)).unwrap();
    }

    #[test]
    fn duplicates_conflict() {
        let (d, mut c) = setup();
        let f = d.outside.join("one.onnx");
        let g = d.outside.join("sub/two.onnx");
        touch(&f);
        touch(&g);
        add_custom_model(&mut c, &req(&f)).unwrap();
        // Same name, other file.
        let r = CustomModel {
            name: Some("ONE".into()),
            ..req(&g)
        };
        let e = add_custom_model(&mut c, &r).unwrap_err();
        assert!(matches!(e, ActionError::Conflict(_)));
        assert!(
            e.message().contains("a model named 'ONE' is already"),
            "{e}"
        );
        // Same file, other name.
        let r = CustomModel {
            name: Some("other".into()),
            ..req(&f)
        };
        let e = add_custom_model(&mut c, &r).unwrap_err();
        assert!(matches!(e, ActionError::Conflict(_)));
        assert!(
            e.message().contains("already in the config as 'one'"),
            "{e}"
        );
        assert_eq!(c.models.len(), 1);
    }

    #[test]
    fn classes_validation_and_discovery() {
        let (d, mut c) = setup();
        let f = d.outside.join("m.onnx");
        touch(&f);
        let bad = d.outside.join("bad.yaml");
        std::fs::write(&bad, "foo: 1\n").unwrap();
        let r = CustomModel {
            classes: Some(bad.to_string_lossy().to_string()),
            ..req(&f)
        };
        let e = add_custom_model(&mut c, &r).unwrap_err();
        assert_eq!(e.status(), 400);
        assert!(
            e.message().contains("NAMES") || e.message().contains("names"),
            "{e}"
        );
        assert!(c.models.is_empty());

        std::fs::write(d.outside.join("data.yaml"), YAML).unwrap();
        add_custom_model(&mut c, &req(&f)).unwrap();
        assert_eq!(c.models[0].classes, Some(d.outside.join("data.yaml")));
    }

    #[test]
    fn sidecar_wins_over_data_yaml() {
        let (d, mut c) = setup();
        let f = d.outside.join("m.onnx");
        touch(&f);
        std::fs::write(d.outside.join("m.yaml"), YAML).unwrap();
        std::fs::write(d.outside.join("data.yaml"), YAML).unwrap();
        add_custom_model(&mut c, &req(&f)).unwrap();
        assert_eq!(c.models[0].classes, None);
    }

    #[test]
    fn paths_relative_to_data_root() {
        let (d, mut c) = setup();
        touch(&d.root.join("root/models/in.onnx"));
        touch(&d.outside.join("out.onnx"));
        std::fs::write(d.root.join("root/models/in.yaml"), YAML).unwrap();
        let r = CustomModel {
            classes: Some("models/in.yaml".into()),
            ..req(Path::new("models/in.onnx"))
        };
        add_custom_model(&mut c, &r).unwrap();
        assert_eq!(c.models[0].path, PathBuf::from("models/in.onnx"));
        assert_eq!(c.models[0].classes, Some(PathBuf::from("models/in.yaml")));
        add_custom_model(&mut c, &req(&d.outside.join("out.onnx"))).unwrap();
        assert_eq!(c.models[1].path, d.outside.join("out.onnx"));
        let _: &ModelConfig = &c.models[1];
    }

    #[test]
    fn invalid_names_rejected() {
        let (d, mut c) = setup();
        let f = d.outside.join("n.onnx");
        touch(&f);
        for bad in ["a/b", "a\\b", "a?b", "a#b", "a\nb"] {
            let r = CustomModel {
                name: Some(bad.into()),
                ..req(&f)
            };
            let e = add_custom_model(&mut c, &r).unwrap_err();
            assert_eq!(e.status(), 400, "{bad}");
        }
        assert!(c.models.is_empty());
    }
}

#[cfg(test)]
mod remove_tests {
    use super::*;
    use crate::config::ModelConfig;

    /// A data root with `download_dir` pointing at it; removed on drop.
    struct Root(PathBuf);

    impl Drop for Root {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn setup(models: Vec<ModelConfig>) -> (Root, Config) {
        let root = std::env::temp_dir().join(format!("bop-remove-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("models")).unwrap();
        let config = Config {
            download_dir: Some(root.clone()),
            models,
            ..Default::default()
        };
        (Root(root), config)
    }

    fn entry(name: &str, path: &str, enabled: bool) -> ModelConfig {
        ModelConfig {
            name: Some(name.into()),
            path: path.into(),
            enabled,
            ..Default::default()
        }
    }

    fn names(c: &Config) -> Vec<String> {
        c.models.iter().map(|m| m.effective_name()).collect()
    }

    fn dfine_m() -> &'static Resource {
        catalog::model("dfine-m").expect("dfine-m is in the catalog")
    }

    fn install(root: &Path, res: &Resource) {
        for p in res.parts {
            std::fs::write(root.join("models").join(p.file_name), b"x").unwrap();
        }
    }

    #[test]
    fn remove_returns_the_disabled_entries_of_its_files() {
        let res = dfine_m();
        let file = format!("models/{}", res.parts[0].file_name);
        let (root, mut c) = setup(vec![
            entry("ipcam", "models/ipcam.onnx", true),
            entry("dfine-m", &file, false),
            entry("unrelated", "models/unrelated.onnx", false),
        ]);
        c.default_model = Some("DFINE-M".into());
        install(&root.0, res);
        let removed = remove(None, &c, res.id).unwrap();
        assert_eq!(removed.config_entries, ["dfine-m"]);
        assert!(!root.0.join(&file).exists());

        let dropped = drop_config_entries(&mut c, &removed.config_entries);
        assert_eq!(dropped, ["dfine-m"]);
        assert_eq!(names(&c), ["ipcam", "unrelated"]);
        // The default named the dropped entry: cleared, the first enabled model serves.
        assert_eq!(c.default_model, None);
        assert_eq!(c.effective_default_index(), Some(0));
        crate::config::validate_config(&c).unwrap();
        assert!(
            removed
                .describe(&dropped)
                .ends_with("and its config entry 'dfine-m'"),
            "{}",
            removed.describe(&dropped)
        );
    }

    #[test]
    fn an_enabled_entry_blocks_remove() {
        let res = dfine_m();
        let file = format!("models/{}", res.parts[0].file_name);
        let (root, c) = setup(vec![entry("dfine-m", &file, true)]);
        install(&root.0, res);
        let e = remove(None, &c, res.id).unwrap_err();
        assert_eq!(e.status(), 409, "{e}");
        assert!(e.message().contains("disable them first"), "{e}");
        assert!(root.0.join(&file).is_file(), "nothing removed");
    }

    #[test]
    fn remove_without_config_entries_names_none() {
        let res = dfine_m();
        let (root, c) = setup(vec![entry("ipcam", "models/ipcam.onnx", true)]);
        install(&root.0, res);
        let removed = remove(None, &c, res.id).unwrap();
        assert!(removed.config_entries.is_empty());
        assert_eq!(removed.describe(&[]), removed.message);
        assert!(
            removed.message.starts_with("removed "),
            "{}",
            removed.message
        );
    }

    #[test]
    fn leftover_ir_of_an_export_takes_its_entry() {
        let Some(res) = find(None, "model:yolo26n") else {
            return; // no exports on this platform
        };
        let (root, mut c) = setup(vec![
            entry("ipcam", "models/ipcam.onnx", true),
            entry("yolo26n", "models/yolo26n.xml", false),
        ]);
        // Only the IR of an earlier export is left.
        for f in ["yolo26n.xml", "yolo26n.bin"] {
            std::fs::write(root.0.join("models").join(f), b"x").unwrap();
        }
        let removed = remove(None, &c, res.id).unwrap();
        assert_eq!(removed.config_entries, ["yolo26n"]);
        assert!(!root.0.join("models/yolo26n.xml").exists());
        assert_eq!(
            drop_config_entries(&mut c, &removed.config_entries),
            ["yolo26n"]
        );
        assert_eq!(names(&c), ["ipcam"]);
    }

    #[test]
    fn drop_config_entries_keeps_enabled_and_unnamed() {
        let (_root, mut c) = setup(vec![
            entry("a", "models/a.onnx", true),
            entry("b", "models/b.onnx", false),
            entry("c", "models/c.onnx", false),
        ]);
        c.default_model = Some("a".into());
        let dropped = drop_config_entries(&mut c, &["A".into(), "B".into()]);
        assert_eq!(dropped, ["b"]);
        assert_eq!(names(&c), ["a", "c"]);
        assert_eq!(c.default_model.as_deref(), Some("a"));
    }

    #[test]
    fn remove_config_entry_rules() {
        let (root, mut c) = setup(vec![
            entry("a", "models/a.onnx", true),
            entry("Gone", "models/gone.onnx", false),
            entry("kept", "models/kept.onnx", false),
        ]);
        std::fs::write(root.0.join("models/kept.onnx"), b"x").unwrap();
        c.default_model = Some("gone".into());

        let e = remove_config_entry(&mut c, "nope").unwrap_err();
        assert_eq!(e.status(), 404, "{e}");
        let e = remove_config_entry(&mut c, "A").unwrap_err();
        assert_eq!(e.status(), 409, "{e}");
        assert!(e.message().contains("disable it first"), "{e}");
        assert_eq!(c.models.len(), 3);

        // Missing file: no word about files; the default naming it is cleared.
        let msg = remove_config_entry(&mut c, " gone ")
            .unwrap()
            .delete_files();
        assert_eq!(msg, "removed 'Gone' from the config");
        assert_eq!(c.default_model, None);
        // An existing file in the models folder is deleted and the answer says so.
        let msg = remove_config_entry(&mut c, "kept").unwrap().delete_files();
        assert!(
            msg.contains("and deleted ") && msg.contains("kept.onnx"),
            "{msg}"
        );
        assert!(!root.0.join("models/kept.onnx").exists());
        assert_eq!(names(&c), ["a"]);
    }
}
