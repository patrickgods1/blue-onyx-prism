//! The models a run with a model can label with (`GET /v1/autolabel/models`, `train autolabel
//! --with-model <id>`; docs/TRAINING.md "Label with any model"), and how a chosen one becomes a
//! config entry:
//!
//! - **config**: every entry of `models`, enabled or not (id = its name);
//! - **local**: a model file in `models_dir` no entry uses ([`status::local_models`], the Config
//!   page's "Local models" list; id `local:<file>`), family `auto` (detected from the model's
//!   outputs when it loads);
//! - **catalog**: a model resource of this platform no entry uses (downloaded, or exported on
//!   this machine; id `resource:<catalog id>`, e.g. `resource:model:rfdetr-large`), with its
//!   size, licence, export need and the large-download gate.
//!
//! [`resolve`] turns an id into a [`ModelPlan`] (pure apart from reading the models folder); a
//! local or catalog model is added to the config **disabled** ([`add_disabled`], through a
//! [`ConfigWriter`]: the server's `save_config`, the CLI's config file), so it is never loaded
//! for serving. Nothing here downloads: the job installs a catalog model in its own stage, only
//! for a run the user started.

use super::catalog::{ModelClasses, coco_class, model_classes};
use crate::backend::spec;
use crate::config::{Config, ModelConfig};
use crate::model::ModelFamilyKind;
use crate::resources::catalog::{self as res_catalog, Platform, Resource, ResourceKind};
use crate::resources::status::{self, LocalModel};
use crate::resources::{export, manager};
use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use std::path::{Path, PathBuf};

/// Id prefix of a local model file (`local:yolo26s.onnx`).
pub const LOCAL_PREFIX: &str = status::LOCAL_ID_PREFIX;
/// Id prefix of a catalog model (`resource:model:rfdetr-large`).
pub const RESOURCE_PREFIX: &str = "resource:";
/// Default score threshold of a run with a model.
pub const DEFAULT_THRESHOLD: f32 = 0.3;

/// Where a model comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    Config,
    Local,
    Catalog,
}

/// One model a run can use (`GET /v1/autolabel/models` `models[]`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelOption {
    /// What `with_model` takes: a config name, `local:<file>` or `resource:<catalog id>`.
    pub id: String,
    pub source: Source,
    pub name: String,
    /// The model files are on disk.
    pub installed: bool,
    pub in_config: bool,
    /// Loaded for serving (config entries only).
    pub enabled: bool,
    /// Its family (`auto` = detected from the outputs at load).
    pub family: String,
    /// Number of classes (None: not known before it is downloaded).
    pub classes: Option<usize>,
    /// Every class is a COCO class (aliases applied): RF-DETR-style COCO labels.
    pub coco: bool,
    /// File size (config / local) or download size (catalog: the model, or the weights an
    /// export downloads).
    pub size_bytes: u64,
    /// A catalog model exported on this machine first (the export toolchain).
    pub needs_export: bool,
    /// Installing it is a large download (> 500 MB, the export toolchain included): needs
    /// `allow_large` / `allow_large_downloads`.
    pub large: bool,
    pub license: String,
    /// What to know before using it (an export's notice, a file's runtime note, why it is
    /// missing).
    pub note: String,
    /// The device it runs on by default (its config device, else the config's).
    pub device: String,
    /// A run with it needs `confirm_license` ([`restricted_license`] of a catalog model; the
    /// page asks exactly when this is set).
    pub confirm_license: bool,
}

/// A restrictive licence (non-commercial / personal use): a run that installs or adds the model
/// needs `confirm_license`.
pub fn restricted_license(license: &str) -> bool {
    let l = license.to_ascii_lowercase();
    l.contains("non-commercial") || l.contains("personal use")
}

/// Whether every name of `classes` is a COCO class.
fn all_coco(classes: &[String]) -> bool {
    !classes.is_empty() && classes.iter().all(|c| coco_class(c).is_some())
}

/// The default device of `m` as a canonical spec (`auto`, `openvino:gpu`), else the raw text.
fn device_of(config: &Config, m: &ModelConfig) -> String {
    config
        .device_spec_for(m)
        .map_or_else(|_| config.device_for(m), |d| d.to_string())
}

/// Size of a model file (an OpenVINO IR with its `.bin`).
fn file_size(p: &Path) -> u64 {
    let mut n = std::fs::metadata(p).map_or(0, |m| m.len());
    if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("xml")) {
        n += std::fs::metadata(p.with_extension("bin")).map_or(0, |m| m.len());
    }
    n
}

/// A config entry as an option (pure apart from the file size).
fn config_option(config: &Config, m: &ModelConfig, classes: &ModelClasses) -> ModelOption {
    let path = config.data_path(&m.path);
    let installed = path.is_file();
    let note = match (&classes.error, installed) {
        (_, false) => format!(
            "{} is not there yet: download or export it on the Config page (Resources)",
            path.display()
        ),
        (Some(e), true) => e.clone(),
        (None, true) => String::new(),
    };
    ModelOption {
        id: m.effective_name(),
        source: Source::Config,
        name: m.effective_name(),
        installed,
        in_config: true,
        enabled: m.enabled,
        family: m.family.to_string(),
        classes: (classes.error.is_none()).then_some(classes.classes.len()),
        coco: all_coco(&classes.classes),
        size_bytes: if installed { file_size(&path) } else { 0 },
        needs_export: false,
        large: false,
        license: String::new(),
        note,
        device: device_of(config, m),
        confirm_license: false,
    }
}

/// A local model file as an option.
fn local_option(l: &LocalModel, classes: Option<&[String]>, device: &str) -> ModelOption {
    ModelOption {
        id: l.id.clone(),
        source: Source::Local,
        name: l.name.clone(),
        installed: true,
        in_config: false,
        enabled: false,
        family: ModelFamilyKind::Auto.to_string(),
        classes: classes.map(<[String]>::len),
        coco: classes.is_some_and(all_coco),
        size_bytes: l.size,
        needs_export: false,
        large: false,
        license: String::new(),
        note: format!(
            "{}; added to the config disabled, family auto (detected from its outputs)",
            l.note
        ),
        device: device.to_string(),
        confirm_license: false,
    }
}

/// What a catalog model's install needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogFacts {
    /// The model files are in the models folder (downloaded or exported, current version).
    pub installed: bool,
    /// The export toolchain is set up (exports only).
    pub toolchain_ready: bool,
    /// Bytes the toolchain setup downloads (exports, when not ready).
    pub toolchain_download: u64,
}

/// A catalog model as an option (pure).
pub fn catalog_option(res: &'static Resource, facts: CatalogFacts, device: &str) -> ModelOption {
    let export = res.kind == ResourceKind::ExportModel;
    let name = res.model_name().unwrap_or(res.id).to_string();
    let family = match res.provides {
        res_catalog::Provides::Model { family, .. } => family,
        _ => ModelFamilyKind::Auto,
    };
    let (classes, coco) = if export {
        match res_catalog::export_class_names(res) {
            Some(names) => {
                let names: Vec<String> = names.iter().map(|s| (*s).to_string()).collect();
                (Some(names.len()), all_coco(&names))
            }
            None => (Some(res_catalog::export_class_count(res)), true),
        }
    } else if res.parts.iter().any(|p| p.file_name.ends_with(".yaml"))
        && !matches!(family, ModelFamilyKind::RtDetr)
    {
        // Its own class file (IPcam-general and the other YOLOv5 models): known once downloaded.
        (None, false)
    } else {
        // DETR / RF-DETR use COCO-80; RT-DETRv2's YAML is the COCO list.
        (Some(80), true)
    };
    let needs_export = export && !facts.installed;
    let toolchain = if needs_export && !facts.toolchain_ready {
        facts.toolchain_download
    } else {
        0
    };
    let large = !facts.installed
        && (res.is_large() || (needs_export && (!facts.toolchain_ready || toolchain > 0)));
    let mut note = if export {
        res_catalog::export_recipe(res)
            .map(|r| r.notice().to_string())
            .unwrap_or_default()
    } else {
        res.description.to_string()
    };
    if needs_export && !facts.toolchain_ready {
        note.push_str(&format!(
            " The first export sets up the export toolchain once ({}).",
            if toolchain > 0 {
                res_catalog::format_size(toolchain)
            } else {
                "large".to_string()
            }
        ));
    }
    if restricted_license(res.license) {
        note.push_str(&format!(" {}: confirm the licence to use it.", res.license));
    }
    ModelOption {
        id: format!("{RESOURCE_PREFIX}{}", res.id),
        source: Source::Catalog,
        name,
        installed: facts.installed,
        in_config: false,
        enabled: false,
        family: family.to_string(),
        classes,
        coco,
        size_bytes: res.size(),
        needs_export,
        large,
        license: res.license.to_string(),
        note: note.trim().to_string(),
        device: device.to_string(),
        confirm_license: restricted_license(res.license),
    }
}

/// The order of the list: config (enabled first, then config order), local (by name), catalog
/// (largest first: usually the more accurate).
pub fn sort_options(v: &mut [ModelOption]) {
    let rank = |o: &ModelOption| match o.source {
        Source::Config => u8::from(!o.enabled),
        Source::Local => 2,
        Source::Catalog => 3,
    };
    v.sort_by(|a, b| {
        let r = rank(a).cmp(&rank(b));
        match (r, a.source) {
            (std::cmp::Ordering::Equal, Source::Local) => a.name.cmp(&b.name),
            (std::cmp::Ordering::Equal, Source::Catalog) => b
                .size_bytes
                .cmp(&a.size_bytes)
                .then_with(|| a.name.cmp(&b.name)),
            (r, _) => r,
        }
    });
}

/// The model files a catalog model installs into the models folder (exports: their `.onnx` and
/// `.yaml`).
fn catalog_files(config: &Config, res: &Resource) -> Vec<PathBuf> {
    let dir = status::target_dir(config, res);
    match res_catalog::export_files(res) {
        Some((onnx, yaml)) => vec![dir.join(onnx), dir.join(yaml)],
        None => res.parts.iter().map(|p| dir.join(p.file_name)).collect(),
    }
}

fn same_file(a: &Path, b: &Path) -> bool {
    a == b
        || matches!(
            (a.canonicalize(), b.canonicalize()),
            (Ok(x), Ok(y)) if x == y
        )
}

/// The config entry using `res`'s model files (first one), or named like it.
fn config_entry_for<'a>(config: &'a Config, res: &Resource) -> Option<&'a ModelConfig> {
    let files = catalog_files(config, res);
    let name = res
        .model_name()
        .map(crate::registry::normalize_name)
        .unwrap_or_default();
    config
        .models
        .iter()
        .find(|m| {
            let p = config.data_path(&m.path);
            files.iter().any(|f| same_file(f, &p))
        })
        .or_else(|| {
            config
                .models
                .iter()
                .find(|m| crate::registry::normalize_name(&m.effective_name()) == name)
        })
}

/// Whether `res`'s model files are installed (exports: the current version).
pub fn catalog_installed(config: &Config, res: &Resource) -> bool {
    let dir = status::target_dir(config, res);
    match res.kind {
        ResourceKind::ExportModel => export::is_exported(res, &dir),
        _ => {
            manager::is_installed(res, &dir)
                || res.parts.iter().all(|p| dir.join(p.file_name).is_file())
        }
    }
}

/// The install facts of `res` for `config` on `platform`.
fn catalog_facts(config: &Config, res: &Resource, platform: Platform) -> CatalogFacts {
    let installed = catalog_installed(config, res);
    let (toolchain_ready, toolchain_download) = if res.kind == ResourceKind::ExportModel {
        let paths = export::Paths::for_config(config);
        (
            export::toolchain_state(&paths, platform).ready(),
            res_catalog::toolchain_estimate(platform.os, platform.arch).map_or(0, |e| e.download),
        )
    } else {
        (true, 0)
    };
    CatalogFacts {
        installed,
        toolchain_ready,
        toolchain_download,
    }
}

/// Every model a run can use (see the module docs), sorted ([`sort_options`]).
pub fn model_options(config: &Config) -> Vec<ModelOption> {
    let platform = Platform::current();
    let new_device = device_of(config, &ModelConfig::default());
    let classes = model_classes(config);
    let mut out: Vec<ModelOption> = config
        .models
        .iter()
        .zip(&classes)
        .map(|(m, c)| config_option(config, m, c))
        .collect();
    for l in status::local_models(None, config) {
        let names = crate::model::classes::resolve_class_names(Path::new(&l.path), None).ok();
        out.push(local_option(&l, names.as_deref(), &new_device));
    }
    let exportable = export::unsupported_reason(platform.os, platform.arch).is_none();
    for res in status::platform_resources(None) {
        if !res.kind.is_model()
            || (res.kind == ResourceKind::ExportModel && !exportable)
            || config_entry_for(config, res).is_some()
        {
            continue;
        }
        out.push(catalog_option(
            res,
            catalog_facts(config, res, platform),
            &new_device,
        ));
    }
    sort_options(&mut out);
    out
}

/// What a chosen model is.
#[derive(Debug, Clone, PartialEq)]
pub enum Target {
    /// An entry of the config (its effective name).
    Config { name: String },
    /// A model file in `models_dir` (a plain file name) no entry uses.
    Local { file: String, path: PathBuf },
    /// A catalog model no entry uses.
    Catalog { res: &'static Resource },
}

/// How a catalog model is installed by the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Install {
    Download,
    Export,
}

/// A resolved `with_model` ([`resolve`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ModelPlan {
    /// As given.
    pub id: String,
    pub target: Target,
    /// The model's name (the boxes' labeler, the config entry's name).
    pub name: String,
    /// The device the run asks for, canonical: the run's `device`, else the model's (an entry
    /// added by the run gets the config's).
    pub device: String,
    /// A catalog model that is not installed yet.
    pub install: Option<Install>,
    /// The model's licence when it is restricted (non-commercial / personal use).
    pub restricted: Option<String>,
}

impl ModelPlan {
    /// The marker's labeler: `model:<name>@<device>` (a later run of the same model on the
    /// same device skips the frames unless `again=1`).
    pub fn labeler(&self) -> String {
        format!("model:{}@{}", self.name, self.device)
    }

    /// The catalog resource, if any.
    pub fn resource(&self) -> Option<&'static Resource> {
        match self.target {
            Target::Catalog { res } => Some(res),
            _ => None,
        }
    }
}

/// A device spec, canonical (`GPU` -> `openvino:gpu`); errors name the field.
pub fn canonical_device(s: &str) -> Result<String> {
    spec::parse(s)
        .map(|d| d.to_string())
        .context("device of the run")
}

/// The entry named `name` (matched like `/v1/vision/custom/{model}`).
fn find_entry<'a>(config: &'a Config, name: &str) -> Option<&'a ModelConfig> {
    let key = crate::registry::normalize_name(name);
    config
        .models
        .iter()
        .find(|m| crate::registry::normalize_name(&m.effective_name()) == key)
}

/// A plain file name (no folder part).
fn plain_file(file: &str) -> bool {
    !file.is_empty()
        && file != "."
        && file != ".."
        && !file.contains(['/', '\\'])
        && Path::new(file).file_name().and_then(|n| n.to_str()) == Some(file)
}

/// Resolve `id` (see [`ModelOption::id`]; a catalog model's plain name also works) for
/// `config`, with the run's `device` (None: the model's). A local file or catalog model an
/// entry already uses resolves to that entry.
pub fn resolve(config: &Config, id: &str, device: Option<&str>) -> Result<ModelPlan> {
    let id = id.trim();
    if id.is_empty() {
        bail!("choose a model (GET /v1/autolabel/models lists them)");
    }
    let device_for = |m: &ModelConfig| -> Result<String> {
        match device {
            Some(d) => canonical_device(d),
            None => Ok(device_of(config, m)),
        }
    };
    let from_entry = |m: &ModelConfig| -> Result<ModelPlan> {
        Ok(ModelPlan {
            id: id.to_string(),
            target: Target::Config {
                name: m.effective_name(),
            },
            name: m.effective_name(),
            device: device_for(m)?,
            install: None,
            restricted: None,
        })
    };
    let from_resource = |res: &'static Resource| -> Result<ModelPlan> {
        if !res.kind.is_model() {
            bail!("{} is not a model", res.id);
        }
        if let Some(m) = config_entry_for(config, res) {
            return from_entry(m);
        }
        let installed = catalog_installed(config, res);
        Ok(ModelPlan {
            id: id.to_string(),
            target: Target::Catalog { res },
            name: res.model_name().unwrap_or(res.id).to_string(),
            device: device_for(&ModelConfig::default())?,
            install: (!installed).then_some(if res.kind == ResourceKind::ExportModel {
                Install::Export
            } else {
                Install::Download
            }),
            restricted: restricted_license(res.license).then(|| res.license.to_string()),
        })
    };
    if let Some(rid) = res_catalog::strip_prefix_ci(id, RESOURCE_PREFIX) {
        let res = status::find(None, rid).ok_or_else(|| {
            anyhow!(
                "no catalog model '{rid}' for this platform (`blue-onyx-prism list-resources` \
                 lists them)"
            )
        })?;
        return from_resource(res);
    }
    if let Some(file) = res_catalog::strip_prefix_ci(id, LOCAL_PREFIX) {
        let file = file.trim();
        if !plain_file(file) {
            bail!(
                "'{file}' is not a model file name in the models folder (local:<file>.onnx or \
                 .xml)"
            );
        }
        let path = config.data_path(&config.models_dir).join(file);
        if !path.is_file() {
            bail!("{} does not exist", path.display());
        }
        if let Some(m) = config
            .models
            .iter()
            .find(|m| same_file(&config.data_path(&m.path), &path))
        {
            return from_entry(m);
        }
        let name = Path::new(file)
            .file_stem()
            .map_or_else(|| file.to_string(), |s| s.to_string_lossy().to_string());
        return Ok(ModelPlan {
            id: id.to_string(),
            target: Target::Local {
                file: file.to_string(),
                path,
            },
            name,
            device: device_for(&ModelConfig::default())?,
            install: None,
            restricted: None,
        });
    }
    if let Some(m) = find_entry(config, id) {
        return from_entry(m);
    }
    let platform = Platform::current();
    if let Some(res) = res_catalog::model(id)
        .or_else(|| res_catalog::export_model(id))
        .filter(|r| r.available_on(platform.os, platform.arch))
    {
        return from_resource(res);
    }
    bail!(
        "no model '{id}': not a configured model, local:<file> or resource:<catalog id> \
         (GET /v1/autolabel/models, the Training page's model list, shows every choice)"
    )
}

/// Saves config edits through the normal path (the server's `save_config`, the CLI's file).
pub trait ConfigWriter: Send + Sync {
    /// Apply `edit` to the saved config and save it; returns the config after it and a message.
    fn update(&self, edit: &mut dyn FnMut(&mut Config) -> Result<()>) -> Result<(Config, String)>;
}

/// The CLI's [`ConfigWriter`]: the config file at `path` (a missing file is the default
/// config).
#[derive(Debug, Clone)]
pub struct FileConfigWriter {
    pub path: PathBuf,
}

impl ConfigWriter for FileConfigWriter {
    fn update(&self, edit: &mut dyn FnMut(&mut Config) -> Result<()>) -> Result<(Config, String)> {
        let mut c = if self.path.exists() {
            Config::load(&self.path)
                .context("fix the config file, or pass --config <another file>")?
        } else {
            Config::default()
        };
        edit(&mut c)?;
        c.save(&self.path)?;
        Ok((c, format!("saved {}", self.path.display())))
    }
}

/// The entry of `config` a plan's model is (after [`add_disabled`]): by name for a config
/// model, by file for a local or catalog one.
pub fn entry_of<'a>(config: &'a Config, plan: &ModelPlan) -> Option<&'a ModelConfig> {
    match &plan.target {
        Target::Config { name } => find_entry(config, name),
        Target::Local { path, .. } => config
            .models
            .iter()
            .find(|m| same_file(&config.data_path(&m.path), path)),
        Target::Catalog { res } => {
            let files = catalog_files(config, res);
            config.models.iter().find(|m| {
                let p = config.data_path(&m.path);
                files.iter().any(|f| same_file(f, &p))
            })
        }
    }
}

/// Add a local or catalog plan's model to `config` **disabled** (the Config page's Add to config
/// rules otherwise: relative paths, class file, family), unless an entry uses its file already.
/// Returns what happened. Fails when another entry has its name (rename that one first).
pub fn add_disabled(config: &mut Config, plan: &ModelPlan) -> Result<String> {
    if entry_of(config, plan).is_some() {
        return Ok(format!("'{}' is in the config already", plan.name));
    }
    let id = match &plan.target {
        Target::Config { name } => return Ok(format!("'{name}' is in the config already")),
        Target::Local { file, .. } => format!("{LOCAL_PREFIX}{file}"),
        Target::Catalog { res } => res.id.to_string(),
    };
    let before = config.models.len();
    crate::resources::actions::add_to_config(None, config, &id)
        .map_err(|e| anyhow!("adding {} to the config: {}", plan.name, e.message()))?;
    for m in config.models.iter_mut().skip(before) {
        m.enabled = false;
    }
    if entry_of(config, plan).is_none() {
        bail!(
            "another config model is named {}; rename or remove it on the Config page (Models \
             card), then run again",
            plan.name
        );
    }
    Ok(format!(
        "added '{}' to the config, disabled (used for auto-labelling only; enable it in the \
         Models card to serve it)",
        plan.name
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    struct Tmp(PathBuf);
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn setup() -> (Tmp, Config) {
        let d = std::env::temp_dir().join(format!("bop-al-models-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(d.join("models")).unwrap();
        let mut c = Config {
            download_dir: Some(d.clone()),
            ..Config::default()
        };
        c.seed_default_model();
        (Tmp(d), c)
    }

    fn opt(source: Source, name: &str, enabled: bool, size: u64) -> ModelOption {
        ModelOption {
            id: name.into(),
            source,
            name: name.into(),
            installed: true,
            in_config: source == Source::Config,
            enabled,
            family: "auto".into(),
            classes: None,
            coco: false,
            size_bytes: size,
            needs_export: false,
            large: false,
            license: String::new(),
            note: String::new(),
            device: "auto".into(),
            confirm_license: false,
        }
    }

    #[test]
    fn options_sort_config_enabled_first_then_local_then_catalog_by_size() {
        let mut v = vec![
            opt(Source::Catalog, "small", false, 10),
            opt(Source::Local, "zeta", false, 1),
            opt(Source::Config, "off", false, 0),
            opt(Source::Catalog, "big", false, 100),
            opt(Source::Local, "alpha", false, 1),
            opt(Source::Config, "on", true, 0),
            opt(Source::Config, "on2", true, 0),
        ];
        sort_options(&mut v);
        let names: Vec<&str> = v.iter().map(|o| o.name.as_str()).collect();
        assert_eq!(names, ["on", "on2", "off", "alpha", "zeta", "big", "small"]);
    }

    #[test]
    fn catalog_options_carry_size_licence_export_and_the_large_gate() {
        let ready = CatalogFacts {
            installed: false,
            toolchain_ready: true,
            toolchain_download: 0,
        };
        let rf = res_catalog::model("rfdetr-base").unwrap();
        let o = catalog_option(rf, ready, "auto");
        assert_eq!(o.id, "resource:model:rfdetr-base");
        assert_eq!(
            (o.source, o.name.as_str()),
            (Source::Catalog, "rfdetr-base")
        );
        assert_eq!(
            (o.classes, o.coco, o.family.as_str()),
            (Some(80), true, "rfdetr")
        );
        assert_eq!(o.size_bytes, rf.size());
        assert!(!o.needs_export && !o.large && !o.installed);
        assert_eq!(o.license, "Apache-2.0");
        // An export: the toolchain not set up makes it large; its notice is the note.
        let y = res_catalog::export_model("yolo26n").unwrap();
        let fresh = CatalogFacts {
            installed: false,
            toolchain_ready: false,
            toolchain_download: 400_000_000,
        };
        let o = catalog_option(y, fresh, "openvino:cpu");
        assert!(o.needs_export && o.large, "{o:?}");
        assert!(o.note.contains("400 MB"), "{}", o.note);
        assert_eq!(o.device, "openvino:cpu");
        let o = catalog_option(y, ready, "auto");
        assert!(o.needs_export && !o.large);
        let done = CatalogFacts {
            installed: true,
            ..fresh
        };
        let o = catalog_option(y, done, "auto");
        assert!(!o.needs_export && !o.large && o.installed);
        // A non-commercial model says so.
        let d = res_catalog::export_model("deimv2-s").unwrap();
        let o = catalog_option(d, ready, "auto");
        assert!(restricted_license(&o.license), "{}", o.license);
        assert!(o.note.contains("confirm the licence"), "{}", o.note);
        // MegaDetector: three classes, not COCO.
        let md = res_catalog::EXPORT_MODELS
            .iter()
            .find(|r| r.model_name().is_some_and(|n| n.starts_with("mdv6-")))
            .unwrap();
        let o = catalog_option(md, ready, "auto");
        assert_eq!((o.classes, o.coco), (Some(3), false));
        // The JSON contract.
        let v = serde_json::to_value(&o).unwrap();
        for k in [
            "id",
            "source",
            "name",
            "installed",
            "in_config",
            "enabled",
            "family",
            "classes",
            "coco",
            "size_bytes",
            "needs_export",
            "large",
            "license",
            "note",
            "device",
        ] {
            assert!(v.get(k).is_some(), "{k}: {v}");
        }
        assert_eq!(v["source"], "catalog");
    }

    #[test]
    fn the_list_has_config_local_and_catalog_models() {
        let (t, mut c) = setup();
        let models = t.0.join("models");
        std::fs::write(models.join("mine.onnx"), b"x").unwrap();
        std::fs::write(models.join("mine.yaml"), "names: [person, motorbike]\n").unwrap();
        c.models.push(ModelConfig {
            name: Some("off".into()),
            path: models.join("gone.onnx"),
            enabled: false,
            ..ModelConfig::default()
        });
        let v = model_options(&c);
        let first = &v[0];
        assert_eq!(
            (first.source, first.name.as_str(), first.enabled),
            (Source::Config, "IPcam-general", true)
        );
        assert!(!first.installed && first.note.contains("not there yet"));
        assert_eq!(v[1].name, "off");
        let local = v.iter().find(|o| o.source == Source::Local).unwrap();
        assert_eq!(local.id, "local:mine.onnx");
        assert_eq!((local.classes, local.coco), (Some(2), true));
        assert_eq!(local.family, "auto");
        // The seeded IPcam-general is in the config: not offered again from the catalog.
        assert!(
            !v.iter()
                .any(|o| o.source == Source::Catalog && o.name == "IPcam-general")
        );
        let cat: Vec<&ModelOption> = v.iter().filter(|o| o.source == Source::Catalog).collect();
        assert!(cat.iter().any(|o| o.name == "rfdetr-base"), "{cat:?}");
        assert!(cat.windows(2).all(|w| w[0].size_bytes >= w[1].size_bytes));
    }

    #[test]
    fn ids_resolve_to_config_local_and_catalog_models() {
        let (t, c) = setup();
        let models = t.0.join("models");
        std::fs::write(models.join("mine.onnx"), b"x").unwrap();
        let p = resolve(&c, "ipcam-general", None).unwrap();
        assert_eq!(
            p.target,
            Target::Config {
                name: "IPcam-general".into()
            }
        );
        assert_eq!(p.labeler(), "model:IPcam-general@auto");
        let p = resolve(&c, "IPcam-general", Some("CPU")).unwrap();
        assert_eq!(p.device, "openvino:cpu");
        assert!(resolve(&c, "IPcam-general", Some("tpu")).is_err());
        let p = resolve(&c, "local:mine.onnx", None).unwrap();
        assert_eq!(p.name, "mine");
        assert!(matches!(p.target, Target::Local { .. }));
        for bad in ["local:../x.onnx", "local:nope.onnx", "local:"] {
            assert!(resolve(&c, bad, None).is_err(), "{bad}");
        }
        let p = resolve(&c, "resource:model:rfdetr-base", None).unwrap();
        assert_eq!(p.install, Some(Install::Download));
        assert_eq!(p.labeler(), "model:rfdetr-base@auto");
        let p = resolve(&c, "resource:model:rfdetr-large", None).unwrap();
        assert_eq!(p.install, Some(Install::Export));
        // A plain catalog name works too; an export needs one.
        let p = resolve(&c, "yolo26n", None).unwrap();
        assert_eq!(p.install, Some(Install::Export));
        let p = resolve(&c, "resource:model:deimv2-s", None).unwrap();
        assert!(p.restricted.is_some());
        // The catalog id of a configured model is that entry.
        let p = resolve(&c, "resource:model:IPcam-general", None).unwrap();
        assert!(matches!(p.target, Target::Config { .. }), "{p:?}");
        let e = resolve(&c, "unicorn", None).unwrap_err().to_string();
        assert!(e.contains("/v1/autolabel/models"), "{e}");
        assert!(resolve(&c, "resource:openvino-runtime", None).is_err());
    }

    #[test]
    fn local_and_catalog_models_are_added_disabled() {
        let (t, mut c) = setup();
        let models = t.0.join("models");
        std::fs::write(models.join("mine.onnx"), b"x").unwrap();
        let p = resolve(&c, "local:mine.onnx", None).unwrap();
        let msg = add_disabled(&mut c, &p).unwrap();
        assert!(msg.contains("disabled"), "{msg}");
        let m = entry_of(&c, &p).unwrap();
        assert!(!m.enabled);
        assert_eq!(m.family, ModelFamilyKind::Auto);
        assert_eq!(c.models.len(), 2);
        // Once added, the same id is the config entry.
        assert!(matches!(
            resolve(&c, "local:mine.onnx", None).unwrap().target,
            Target::Config { .. }
        ));
        assert!(add_disabled(&mut c, &p).unwrap().contains("already"));
        // Also when no model is enabled (Add to config would enable the first one).
        let mut empty = Config {
            download_dir: Some(t.0.clone()),
            ..Config::default()
        };
        let p = resolve(&empty, "local:mine.onnx", None).unwrap();
        add_disabled(&mut empty, &p).unwrap();
        assert!(!empty.models[0].enabled);
        // A catalog model (installed by the run first; here by hand).
        let rf = res_catalog::model("rfdetr-base").unwrap();
        let p = resolve(&c, "resource:model:rfdetr-base", None).unwrap();
        assert!(add_disabled(&mut c, &p).is_err(), "not installed yet");
        for part in rf.parts {
            std::fs::write(models.join(part.file_name), b"x").unwrap();
        }
        add_disabled(&mut c, &p).unwrap();
        let m = entry_of(&c, &p).unwrap();
        assert!(!m.enabled);
        assert_eq!(m.family, ModelFamilyKind::RfDetr);
        // Another entry with the name: refused with the fix.
        let mut clash = c.clone();
        clash.models.retain(|m| m.effective_name() != "mine");
        clash.models.push(ModelConfig {
            name: Some("mine".into()),
            path: models.join("other.onnx"),
            ..ModelConfig::default()
        });
        let p = Plan::local(&models, "mine.onnx");
        let e = add_disabled(&mut clash, &p).unwrap_err().to_string();
        assert!(e.contains("rename"), "{e}");
    }

    /// Test plans built by hand.
    struct Plan;
    impl Plan {
        fn local(models: &Path, file: &str) -> ModelPlan {
            ModelPlan {
                id: format!("local:{file}"),
                target: Target::Local {
                    file: file.into(),
                    path: models.join(file),
                },
                name: Path::new(file)
                    .file_stem()
                    .unwrap()
                    .to_string_lossy()
                    .to_string(),
                device: "auto".into(),
                install: None,
                restricted: None,
            }
        }
    }

    #[test]
    fn the_file_writer_saves_the_edit() {
        let (t, _) = setup();
        let path = t.0.join("cfg.json");
        let w = FileConfigWriter { path: path.clone() };
        let (c, msg) = w
            .update(&mut |c| {
                c.collect.classes = vec!["person".into()];
                Ok(())
            })
            .unwrap();
        assert!(msg.contains("cfg.json"));
        assert_eq!(c.collect.classes, ["person"]);
        assert_eq!(Config::load(&path).unwrap().collect.classes, ["person"]);
    }
}
