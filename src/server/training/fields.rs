//! The queue form's readers (`POST /training/queue`, the balance preview, the class list): the
//! dataset side ([`data_fields`]: datasets, scenes, classes, merges, excluded mode, balance), one
//! [`TrainRequest`] per ticked model ([`requests_from_form`]), the folder checks
//! ([`checked_datasets`], [`checked_folder`]) and the Datasets list's edit ([`dataset_edit`]).
//! Errors name the field. The field helpers are [`crate::server::form`]'s.

use super::EXTRA_SETS;
use crate::server::form::{all, checked, field, num};
use crate::training::capability::Capability;
use crate::training::dataset::{Balance, Excluded, SceneFilter};
use crate::training::job::TrainRequest;
use crate::training::recipe::{self, Augment};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The dataset side of a queue form, shared with the balance preview ([`data_fields`]).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DataFields {
    /// Ticked `dataset`s, then the typed `dataset_path`, as typed (see [`checked_datasets`]).
    pub datasets: Vec<PathBuf>,
    pub include_public: bool,
    pub scenes: Vec<SceneFilter>,
    /// Empty = every class of the datasets.
    pub classes: Vec<String>,
    pub excluded: Excluded,
    pub balance: Balance,
    /// Class merges (`from` -> `to`; an empty `to` drops the label), see [`rename_of`].
    pub rename: BTreeMap<String, String>,
}

/// The datasets, scenes ([`scenes_of`]), classes ([`classes_of`]), excluded mode, public mix
/// and balance ([`balance_of`]) of a queue or preview form. Errors name the field.
pub(crate) fn data_fields(
    f: &[(String, String)],
    training: &crate::training::config::TrainingConfig,
) -> anyhow::Result<DataFields> {
    let mut datasets: Vec<PathBuf> = all(f, "dataset").into_iter().map(PathBuf::from).collect();
    if let Some(extra) = field(f, "dataset_path") {
        datasets.push(PathBuf::from(extra));
    }
    let excluded = match field(f, "excluded") {
        Some(v) => Excluded::parse(v).ok_or_else(|| {
            anyhow::anyhow!("objects of unticked classes: '{v}' is not background or drop-frame")
        })?,
        None => Excluded::Background,
    };
    Ok(DataFields {
        datasets,
        include_public: checked(f, "public"),
        scenes: scenes_of(f)?,
        classes: classes_of(f),
        excluded,
        balance: balance_of(f, training)?,
        rename: rename_of(f)?,
    })
}

/// The class merges of a form or query: repeated `rename` fields, each one `from=to` or several
/// joined by commas (the Train tab's Merge classes rows, the older text field); an empty `to`
/// drops the label, a later duplicate `from` wins. Like the CLI's `--rename`, nothing else is
/// checked here (the class list must name the target).
pub(crate) fn rename_of(f: &[(String, String)]) -> anyhow::Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for piece in all(f, "rename")
        .into_iter()
        .flat_map(|v| v.split(','))
        .map(str::trim)
        .filter(|p| !p.is_empty())
    {
        let (from, to) = crate::training::dataset::parse_rename(piece)
            .map_err(|_| anyhow::anyhow!("class merges: '{piece}' is not from=to"))?;
        out.insert(from, to);
    }
    Ok(out)
}

/// The Balance fieldset of a form: `balance` (repeated axis checkboxes and / or comma lists of
/// `class`, `scene`, `camera`, `camera-day`), `balance_strength` (0..=1), `balance_max_repeat`
/// (1..=10) and `balance_cap` (x the median group size), which applies only with the
/// `balance_cap_on` checkbox (a blank cap = [`crate::training::dataset::balance::DEFAULT_CAP`]).
/// A form without any `balance*` field (a script, a bookmark) gets `training.balance`, else
/// off; no axis = off whatever the other fields say. Errors name the field.
pub(crate) fn balance_of(
    f: &[(String, String)],
    training: &crate::training::config::TrainingConfig,
) -> anyhow::Result<Balance> {
    use crate::training::dataset::balance::DEFAULT_CAP;
    if !f.iter().any(|(k, _)| k.starts_with("balance")) {
        return Ok(training.balance.clone().unwrap_or_default());
    }
    let mut axes = std::collections::BTreeSet::new();
    for v in all(f, "balance") {
        axes.extend(Balance::parse_axes(v)?);
    }
    let d = Balance::default();
    let strength: Option<f32> = num(f, "balance_strength", "balance strength")?;
    let max_repeat: Option<u32> = num(f, "balance_max_repeat", "balance max repeat")?;
    let cap: Option<f32> = num(f, "balance_cap", "balance cap")?;
    let b = Balance {
        axes,
        strength: strength.unwrap_or(d.strength),
        max_repeat: max_repeat.unwrap_or(d.max_repeat),
        cap_dominant: checked(f, "balance_cap_on").then(|| cap.unwrap_or(DEFAULT_CAP)),
    };
    if b.is_off() {
        return Ok(Balance::default());
    }
    b.validate()?;
    Ok(b)
}

/// The runs of a queue form: one [`TrainRequest`] per ticked model (`model=yolo26n`, repeated),
/// sharing the dataset and options fields (`gpu`: `cuda:N`, empty / `auto` = the automatic pick;
/// the suggested batches are the chosen GPU's; blank `epochs` / `patience` = `training`'s value,
/// else the family's recommendation, per model). Errors name the field. The paths are taken as
/// typed: [`queue`](super::routes::queue) checks them ([`checked_datasets`], [`checked_folder`]).
pub(crate) fn requests_from_form(
    f: &[(String, String)],
    cap: &Capability,
    training: &crate::training::config::TrainingConfig,
) -> anyhow::Result<(Vec<TrainRequest>, Option<String>)> {
    let models = all(f, "model");
    if models.is_empty() {
        anyhow::bail!("no model ticked: choose at least one family and size");
    }
    let gpu = crate::training::capability::gpu_arg(field(f, "gpu"))?;
    let cap = &cap.for_gpu(gpu.as_deref())?;
    let data = data_fields(f, training)?;
    let scenes = &data.scenes;
    let night = checked(f, "night_aug");
    let base = if night {
        Augment::NIGHT
    } else {
        Augment::default()
    };
    let augment = Augment {
        dark: num(f, "aug_dark", "darkness")?.unwrap_or(base.dark),
        noise: num(f, "aug_noise", "noise")?.unwrap_or(base.noise),
        gray: num(f, "aug_gray", "grayscale")?.unwrap_or(base.gray),
    };
    let epochs: Option<u32> = num(f, "epochs", "epochs")?;
    let patience: Option<u32> = num(f, "patience", "patience")?;
    let batch: Option<u32> = num(f, "batch", "batch")?;
    let imgsz: Option<u32> = num(f, "imgsz", "input size")?;
    let lr: Option<f32> = num(f, "lr", "learning rate")?;
    let lr_schedule = match field(f, "lr_schedule").map(str::trim) {
        None | Some("" | "default") => None,
        Some(v) => Some(recipe::LrSchedule::parse(v).ok_or_else(|| {
            anyhow::anyhow!("LR schedule: '{v}' is not default, cosine, linear or step")
        })?),
    };
    let lr_min_factor: Option<f32> = num(f, "lr_min_factor", "final LR factor")?;
    let select_metric = match field(f, "select_metric").map(str::trim) {
        None | Some("") => None,
        Some(v) => Some(recipe::SelectMetric::parse(v).ok_or_else(|| {
            anyhow::anyhow!("select best by: {}", recipe::select_metric_error(v))
        })?),
    };
    let multi_scale = match field(f, "multi_scale") {
        Some(v) => recipe::MultiScale::parse(v)
            .ok_or_else(|| anyhow::anyhow!("multi-scale: '{v}' is not auto, on or off"))?,
        None => recipe::MultiScale::Auto,
    };
    let seeds: u64 = match num::<u64>(f, "seeds", "seeds")? {
        None => 1,
        Some(n @ 1..=3) => n,
        Some(n) => anyhow::bail!("seeds: {n} is not 1 to 3"),
    };
    let name = field(f, "name").map(str::to_string);
    // A typed name works with one model x scene; its repeats get `-s<i>` (see `queue`).
    if name.is_some() && models.len() * scenes.len() > 1 {
        anyhow::bail!(
            "a model name works with one run only (one ticked model, one scene); clear it or \
             tick one of each"
        );
    }
    // Several scenes or seeds: one comparison group for every run (same held-out frames, one
    // table).
    let compare_group = (scenes.len() as u64 * seeds > 1).then(|| {
        format!(
            "cmp-{}",
            crate::clock::name_stamp(std::time::SystemTime::now())
        )
    });
    let sets: Vec<String> = all(f, "benchmark_set")
        .into_iter()
        .filter(|s| EXTRA_SETS.contains(s))
        .map(str::to_string)
        .collect();
    // The Benchmark devices fieldset (marked by `benchmark_devices_form`: none ticked = the device
    // the model would load on); a form without it (a script, a bookmark) gets the saved setting.
    let bench_devices =
        crate::training::config::benchmark_devices(
            &if field(f, "benchmark_devices_form").is_some() {
                all(f, "benchmark_device")
            } else {
                training
                    .benchmark_devices
                    .iter()
                    .map(String::as_str)
                    .collect()
            },
        )
        .map_err(|e| anyhow::anyhow!("benchmark devices: {e}"))?;
    let mut out = Vec::new();
    for (m, scene) in models
        .iter()
        .flat_map(|m| scenes.iter().map(move |s| (*m, *s)))
    {
        let (family, size) =
            recipe::parse_label(m).ok_or_else(|| anyhow::anyhow!("model: unknown '{m}'"))?;
        let r = recipe::build_recipe(
            family,
            size,
            &recipe::RecipeInputs {
                epochs,
                patience,
                imgsz,
                batch,
                lr,
                lr_schedule,
                lr_min_factor,
                select_metric,
                augment,
                multi_scale,
            },
            training,
            cap,
        )?;
        let request = TrainRequest {
            recipe: r,
            datasets: data.datasets.clone(),
            include_public: data.include_public,
            scene,
            classes: (!data.classes.is_empty()).then(|| data.classes.clone()),
            rename: data.rename.clone(),
            benchmark_sets: sets.clone(),
            benchmark_devices: bench_devices.clone(),
            allow_large: checked(f, "confirm_large"),
            install_dir: field(f, "install_dir").map(PathBuf::from),
            gpu: gpu.clone(),
            excluded: data.excluded,
            balance: data.balance.clone(),
            compare_group: compare_group.clone(),
            train_seed: None,
            ignore_memory_check: checked(f, "ignore_memory_check"),
        };
        // Repeats train on the same frames (the assembly seed is the config's) with their own
        // trainer seed.
        for i in 0..seeds {
            let mut req = request.clone();
            req.train_seed = (seeds > 1).then_some(i);
            out.push(req);
        }
    }
    Ok((out, name))
}

/// The scene filters of a queue form: every `scene` field (repeated checkboxes, each `all` or
/// scenes joined by `+`), deduplicated by meaning in form order; none = all frames.
pub(crate) fn scenes_of(f: &[(String, String)]) -> anyhow::Result<Vec<SceneFilter>> {
    let mut out: Vec<SceneFilter> = Vec::new();
    for s in all(f, "scene") {
        let v = SceneFilter::parse(s).ok_or_else(|| {
            anyhow::anyhow!("scene: '{s}' is not all or scenes joined by + (day, night, ir)")
        })?;
        if !out.contains(&v) {
            out.push(v);
        }
    }
    if out.is_empty() {
        out.push(SceneFilter::ALL_SCENES);
    }
    Ok(out)
}

/// The classes of a queue form, in order: repeated `classes` fields (the class picker) and
/// comma-separated ones (the older text field, bookmarks), without duplicates; empty = every
/// class of the datasets.
pub(crate) fn classes_of(f: &[(String, String)]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in all(f, "classes").into_iter().flat_map(|v| v.split(',')) {
        let c = c.trim();
        if !c.is_empty() && !out.iter().any(|o| o == c) {
            out.push(c.to_string());
        }
    }
    out
}

/// `base`, or `base-2`, `base-3`, ... when `taken` (a default model name already used by a busy
/// run, a configured model or an installed file: two queues within the same minute).
pub(crate) fn unique_name(base: &str, taken: impl Fn(&str) -> bool) -> String {
    if !taken(base) {
        return base.to_string();
    }
    (2..10_000u32)
        .map(|i| format!("{base}-{i}"))
        .find(|n| !taken(n))
        .unwrap_or_else(|| base.to_string())
}

/// The dataset folders of a queue form: a ticked `dataset` is taken as is when it is one the
/// page offers (`offered`: `training.datasets` and the Review tab's exports), anything else and
/// the typed `dataset_path` only when `allow` accepts it ([`crate::server::browse::allowed_path`]: inside
/// the browsable folders or picked in the native dialog), as the path it returns. Errors name the
/// field.
pub(crate) fn checked_datasets(
    f: &[(String, String)],
    offered: &[PathBuf],
    allow: &dyn Fn(&str) -> Result<PathBuf, String>,
) -> anyhow::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for d in all(f, "dataset") {
        let p = PathBuf::from(d);
        if offered.contains(&p) {
            out.push(p);
        } else {
            out.push(allow(d).map_err(|e| anyhow::anyhow!("datasets: {e}"))?);
        }
    }
    if let Some(extra) = field(f, "dataset_path") {
        out.push(allow(extra).map_err(|e| anyhow::anyhow!("another folder: {e}"))?);
    }
    Ok(out)
}

/// The folders the Datasets list offers: `training.datasets`, then the Review tab's exports.
pub(super) fn offered_datasets(config: &crate::config::Config) -> Vec<PathBuf> {
    let mut v = config.training.datasets.clone();
    v.extend(crate::training::discover_exports(config));
    v
}

/// [`crate::server::browse::allowed_path`] as a message.
pub(super) fn allow_in(
    config: &crate::config::Config,
) -> impl Fn(&str) -> Result<PathBuf, String> + '_ {
    move |raw| crate::server::browse::allowed_path(config, raw).map_err(|e| e.into_parts().1)
}

/// A folder setting of a form (`label` names the field): [`crate::config::validate_dir_setting`]
/// (no system folder, no UNC path), then `allow` (an existing folder the page may use). None
/// for an empty value.
pub(crate) fn checked_folder(
    label: &str,
    raw: Option<&str>,
    data_root: &std::path::Path,
    allow: &dyn Fn(&str) -> Result<PathBuf, String>,
) -> anyhow::Result<Option<PathBuf>> {
    let Some(raw) = raw.map(str::trim).filter(|r| !r.is_empty()) else {
        return Ok(None);
    };
    let Some(resolved) = crate::config::validate_dir_setting(label, raw, data_root)? else {
        return Ok(None);
    };
    let p = allow(&resolved.to_string_lossy()).map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
    if !p.is_dir() {
        anyhow::bail!("{label}: {} is not a folder", p.display());
    }
    Ok(Some(p))
}

/// A change of the saved export folders.
#[derive(Debug, PartialEq)]
pub(crate) enum DatasetEdit {
    Add(PathBuf),
    Remove(PathBuf),
}

/// The edit of a `POST /training/datasets` form: `remove` = path (a folder's Remove button)
/// wins; else `add` = path, or the queue form's "Another folder" (`dataset_path`, its Save
/// folder button). The queue form's other fields (ticked folders, models, options) are ignored.
pub(crate) fn dataset_edit(f: &[(String, String)]) -> anyhow::Result<DatasetEdit> {
    if let Some(p) = field(f, "remove") {
        return Ok(DatasetEdit::Remove(PathBuf::from(p)));
    }
    field(f, "add")
        .or_else(|| field(f, "dataset_path"))
        .map(|p| DatasetEdit::Add(PathBuf::from(p)))
        .ok_or_else(|| anyhow::anyhow!("another folder: type or browse to a review export folder"))
}
