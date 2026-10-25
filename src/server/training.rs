//! The Training page (`/training`, admin only, docs/TRAINING.md): what this machine can train
//! (capability, environment state), the dataset picker (review export folders), the family /
//! size picker with estimates and hardware fit, the run queue with live progress
//! (`assets/training.js` polls `GET /v1/training`), finished runs with their benchmark and the
//! "Use as default / night model / verify model" actions (written with [`save_config`]).
//!
//! The queue ([`TrainingService`]) is one per process, started on first use with the download
//! manager of the resources context; its [`Hooks`] add a finished model to the config through
//! [`save_config`] (the `add-model` code) and benchmark it on its held-out frames with the
//! generation's benchmark runner, both on the generation current at that moment.

use super::{AppState, Shared, blocking, render, save_config};
use crate::resources::catalog::{self, Platform};
use crate::resources::export::train::{self as env, TrainPaths};
use crate::training::capability::Capability;
use crate::training::dataset::SceneFilter;
use crate::training::job::{BenchSummary, Hooks, TrainOutput, TrainRequest};
use crate::training::recipe::{self, Augment, TrainRecipe};
use crate::training::service::{RunRecord, RunState, TrainingService};
use crate::training::{bench, estimate};
use askama::Template;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};

/// Built-in benchmark sets offered besides the held-out frames.
const EXTRA_SETS: [&str; 3] = ["nir-night", "exdark-ir", "exdark-night"];

static SERVICE: OnceLock<Arc<TrainingService>> = OnceLock::new();

/// Serializes the tests that queue jobs or move the process's queue ([`SERVICE`]).
#[cfg(test)]
pub(super) static QUEUE_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// The process's training queue, started on first use.
pub(super) fn service(shared: &Shared) -> Arc<TrainingService> {
    let svc = SERVICE
        .get_or_init(|| {
            let state = shared.current();
            let config = state.config_read().clone();
            let fetcher: Arc<dyn crate::resources::export::Fetcher> = match &state.resources {
                Some(ctx) => Arc::new(crate::resources::export::ManagerFetcher::new(
                    ctx.provisioner.manager_for(&config),
                )),
                None => Arc::new(crate::resources::export::cli_fetcher(&config.data_root())),
            };
            TrainingService::start(
                Arc::new(crate::resources::export::SystemRunner),
                fetcher,
                Arc::new(ServerHooks {
                    shared: shared.clone(),
                }),
                Some(crate::training::runs_file(&config)),
            )
        })
        .clone();
    let want = crate::training::runs_file(&shared.current().config_read());
    sync_store(&svc, &want);
    svc
}

/// Keep the started queue on the configured training folder (`training.dir` saved on this page,
/// or edited on disk: then at the next queue, import, auto-label or remove): it moves when idle
/// ([`TrainingService::set_store`]); while a job is queued or running it stays with the job and
/// moves on a later call. Reads (the page, `GET /v1/training`) never move it.
fn sync_store(svc: &TrainingService, want: &std::path::Path) {
    if svc.store().as_deref() != Some(want)
        && let Err(e) = svc.set_store(want)
    {
        tracing::debug!("training folder {} not used yet: {e}", want.display());
    }
}

/// The import and auto-label jobs (the queue's, else `tasks.json`).
pub(super) fn tasks(state: &AppState) -> Vec<crate::training::task::TaskRecord> {
    match SERVICE.get() {
        Some(s) => s.tasks(),
        None => crate::training::task::load_records(&crate::training::task::tasks_file(
            &crate::training::runs_file(&state.config_read()),
        )),
    }
}

/// The queue if it was started (the page shows `runs.json` otherwise).
fn runs(state: &AppState) -> Vec<RunRecord> {
    match SERVICE.get() {
        Some(s) => s.runs(),
        None => crate::training::service::load_records(&crate::training::runs_file(
            &state.config_read(),
        )),
    }
}

struct ServerHooks {
    shared: Shared,
}

impl Hooks for ServerHooks {
    fn register(&self, out: &TrainOutput) -> anyhow::Result<String> {
        let state = self.shared.current();
        let mut msg = String::new();
        let saved = save_config(&state, "training", |c| {
            msg = crate::training::cli::add_trained_model(c, out)?;
            Ok(())
        })?;
        Ok(format!("{msg} {saved}"))
    }

    fn benchmark(
        &self,
        out: &TrainOutput,
        extra: &[String],
        cancel: &AtomicBool,
    ) -> anyhow::Result<Option<BenchSummary>> {
        let state = self.shared.current();
        let shared = self.shared.clone();
        let ctx = move || {
            let st = shared.current();
            crate::benchmark::service::RunContext {
                config: st.config_read().clone(),
                results_path: crate::benchmark::results_path(&st.config_path),
                runtimes: st.registry.runtimes().cloned(),
                loaded: super::benchmark::loaded_devices(&st),
                resources: st.resources.clone(),
            }
        };
        bench::run(
            &state.benchmark,
            &ctx,
            &out.name,
            bench::datasets(&out.name, out.heldout.as_deref(), extra),
            cancel,
        )
    }
}

// ---------------------------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------------------------

struct SizeCell {
    label: String,
    size: &'static str,
    fits: bool,
    /// The full estimate (the cell's tooltip).
    estimate: String,
    /// The compact one shown in the cell (`5.4 GB · batch 32 · 20 s`).
    short: String,
    /// `light` | `medium` | `heavy` (the legend explains them).
    weight: &'static str,
    latency: &'static str,
    why: String,
}

struct FamilyRow {
    title: &'static str,
    license: &'static str,
    available: bool,
    why: String,
    sizes: Vec<SizeCell>,
    /// Empty cells after the sizes, so every row of the grid has as many cells.
    pad: usize,
}

/// A licence notice shared by some families (shown once under the size grid).
struct NoticeRow {
    families: String,
    notice: &'static str,
}

/// One inference-latency class of the legend (`light`, its explanation).
struct LatencyRow {
    weight: &'static str,
    text: &'static str,
}

struct RunRow {
    id: String,
    name: String,
    model: String,
    state: &'static str,
    busy: bool,
    percent: u8,
    describe: String,
    dataset: String,
    log_tail: String,
}

struct ResultRow {
    id: String,
    name: String,
    family: String,
    license: String,
    classes: String,
    benchmark: String,
    finished: String,
    in_config: bool,
}

struct DatasetRow {
    path: String,
    exists: bool,
    /// In `training.datasets` (else found under `<training dir>/exports`).
    saved: bool,
}

#[derive(Template)]
#[template(path = "training.html")]
pub(super) struct TrainingTemplate {
    nav: &'static str,
    version: &'static str,
    message: Option<String>,
    error: Option<String>,
    available: bool,
    summary: String,
    reasons: Vec<String>,
    env_ready: bool,
    env_note: String,
    /// "the training environment (about 450 MB)": what the Train confirmation names.
    env_download: String,
    allow_large: bool,
    /// What an import may need over the large-download gate (`None` = the confirmation is not
    /// shown: nothing large is missing, large downloads are allowed or there is no GPU).
    import_large: Option<String>,
    /// The same for the auto-labelers.
    autolabel_large: Option<String>,
    datasets: Vec<DatasetRow>,
    families: Vec<FamilyRow>,
    /// Most sizes of a family (the grid's size columns).
    size_cols: usize,
    notices: Vec<NoticeRow>,
    latencies: Vec<LatencyRow>,
    epochs: u32,
    scenes: Vec<(&'static str, &'static str)>,
    public_available: bool,
    /// The public night subset is downloaded (else mixing it in needs the large-download
    /// confirmation).
    public_installed: bool,
    /// Its download size (`926 MB`).
    public_size: String,
    extra_sets: Vec<&'static str>,
    runs: Vec<RunRow>,
    results: Vec<ResultRow>,
    models: Vec<String>,
    /// The models folder (where a trained model goes by default).
    models_dir: String,
    default_model: String,
    training_dir: String,
    /// `training.dir` as configured (empty = the default folder).
    training_dir_setting: String,
    /// The default training folder (`<data root>/training`).
    training_dir_default: String,
    /// Import and auto-label jobs.
    tasks: Vec<super::training_tasks::TaskRow>,
    /// A job is queued or running (its table starts open).
    tasks_busy: bool,
    /// A run or task is queued or running (the training folder cannot change).
    jobs_busy: bool,
    footage: Vec<super::training_tasks::FootageRow>,
    dataset_dir: String,
    /// Imports listed in the dataset folder.
    imports: Vec<crate::training::import::write::ImportLine>,
}

fn state_str(s: RunState) -> &'static str {
    match s {
        RunState::Queued => "queued",
        RunState::Running => "running",
        RunState::Done => "done",
        RunState::Failed => "failed",
        RunState::Cancelled => "cancelled",
    }
}

fn bench_text(b: Option<&BenchSummary>, err: Option<&str>) -> String {
    match (b, err) {
        (Some(b), _) => format!(
            "AP50 {} \u{b7} {} on {} \u{b7} {} images ({})",
            b.ap50.map_or_else(|| "n/a".into(), |a| format!("{a:.3}")),
            b.ms_p50
                .map_or_else(|| "?".into(), |m| format!("{m:.1} ms")),
            b.device.as_deref().unwrap_or("?"),
            b.images,
            b.datasets.join(", ")
        ),
        (None, Some(e)) => format!("benchmark failed: {e}"),
        (None, None) => "not benchmarked".into(),
    }
}

/// `light: runs on a CPU ...` -> (`light`, `runs on a CPU ...`).
fn latency_parts(hint: &'static str) -> (&'static str, &'static str) {
    hint.split_once(": ").unwrap_or(("", hint))
}

pub(super) fn view(
    state: &AppState,
    message: Option<String>,
    error: Option<String>,
) -> TrainingTemplate {
    let config = state.config_read().clone();
    let dataset_dir = crate::collect::CollectSettings::for_config(&config).dir;
    let cap: Capability = crate::training::capability_now();
    let paths = TrainPaths::for_config(&config);
    let platform = Platform::current();
    let (env_ready, env_note) = match &cap.accelerator {
        Some(a) => {
            let ready = env::train_env_ready(&paths, platform, a.lane());
            let (dl, disk) = a.lane().estimate();
            let note = if ready {
                format!(
                    "ready (PyTorch {}, {})",
                    a.lane().as_str(),
                    paths.export.env.display()
                )
            } else {
                format!(
                    "not set up: the first run downloads about {} (PyTorch {} and the trainers, \
                     {} on disk under {})",
                    catalog::format_size(dl),
                    a.lane().as_str(),
                    catalog::format_size(disk),
                    paths.export.tools.display()
                )
            };
            (ready, note)
        }
        None => (false, "not available on this machine".into()),
    };
    let env_download = cap.accelerator.as_ref().map_or_else(String::new, |a| {
        format!(
            "the training environment (about {})",
            catalog::format_size(a.lane().estimate().0)
        )
    });
    let size_cols = cap
        .families
        .iter()
        .map(|f| f.sizes.len())
        .max()
        .unwrap_or(0);
    let families: Vec<FamilyRow> = cap
        .families
        .iter()
        .map(|f| FamilyRow {
            title: f.title,
            license: f.license,
            available: f.available,
            why: f.why.clone().unwrap_or_default(),
            pad: size_cols.saturating_sub(f.sizes.len()),
            sizes: f
                .sizes
                .iter()
                .map(|z| {
                    let (estimate, short) = match (z.vram_mb, z.epoch_s_per_100, z.suggested_batch)
                    {
                        (Some(v), Some(t), Some(b)) => {
                            let gb = v as f64 / 1024.0;
                            let t = estimate::format_duration(t);
                            (
                                format!("~{gb:.1} GB at batch {b}, ~{t} / epoch / 100 images"),
                                format!("{gb:.1} GB \u{b7} batch {b} \u{b7} {t}"),
                            )
                        }
                        _ => (String::new(), String::new()),
                    };
                    SizeCell {
                        label: f.family.base_name(z.size),
                        size: z.size,
                        fits: z.fits,
                        estimate,
                        short,
                        weight: latency_parts(z.latency).0,
                        latency: z.latency,
                        why: z.why.clone().unwrap_or_default(),
                    }
                })
                .collect(),
        })
        .collect();
    let mut notices: Vec<NoticeRow> = Vec::new();
    for f in cap.families.iter().filter(|f| f.available) {
        let n = f.family.notice();
        match notices.iter_mut().find(|r| r.notice == n) {
            Some(r) => {
                r.families.push_str(", ");
                r.families.push_str(f.title);
            }
            None => notices.push(NoticeRow {
                families: f.title.to_string(),
                notice: n,
            }),
        }
    }
    let mut latencies: Vec<LatencyRow> = Vec::new();
    for z in cap.families.iter().flat_map(|f| f.sizes.iter()) {
        let (weight, text) = latency_parts(z.latency);
        if !latencies.iter().any(|l| l.weight == weight) {
            latencies.push(LatencyRow { weight, text });
        }
    }
    latencies.sort_by_key(|l| match l.weight {
        "light" => 0,
        "medium" => 1,
        _ => 2,
    });
    let large = |parts: Vec<String>| (!parts.is_empty()).then(|| parts.join("; "));
    let (import_large, autolabel_large) = match (&cap.accelerator, config.allow_large_downloads) {
        (Some(_), false) => {
            let env_part = (!env_ready).then(|| env_download.clone());
            let missing = |keys: &[&str]| {
                let list: Vec<_> = keys
                    .iter()
                    .filter_map(|k| catalog::autolabel_weights(k))
                    .chain(
                        keys.contains(&"rfdetr-large")
                            .then(|| catalog::base_weights(catalog::TrainFamily::Rfdetr, "large"))
                            .flatten(),
                    )
                    .map(|r| (r, crate::training::pyenv::weights_dir(&paths, r)))
                    .collect();
                crate::training::pyenv::missing_bytes(&list)
            };
            let mut import = env_part.iter().cloned().collect::<Vec<_>>();
            let siglip = missing(&["siglip2-base"]);
            if siglip > 0 {
                import.push(format!(
                    "SigLIP 2 ({}) for the SigLIP selection",
                    catalog::format_size(siglip)
                ));
            }
            let mut label = env_part.into_iter().collect::<Vec<_>>();
            let defaults = missing(&[
                "mm-gdino-tiny",
                "rfdetr-large",
                "qwen3-vl-2b",
                "siglip2-base",
            ]);
            let others = missing(&[
                "mm-gdino-base",
                "mm-gdino-large",
                "qwen3-vl-4b",
                "qwen3-vl-8b",
            ]);
            if defaults > 0 {
                label.push(format!(
                    "the labeler weights ({} for the default set, more for larger sizes)",
                    catalog::format_size(defaults)
                ));
            } else if others > 0 {
                label.push("the weights of a larger labeler size if you pick one".into());
            }
            (large(import), large(label))
        }
        _ => (None, None),
    };
    let all = runs(state);
    let run_rows = all
        .iter()
        .rev()
        .filter(|r| r.state != RunState::Done)
        .map(|r| RunRow {
            id: r.id.clone(),
            name: r.name.clone(),
            model: r.request.recipe.label(),
            state: state_str(r.state),
            busy: r.state.is_busy(),
            percent: r.percent,
            describe: r.describe(),
            dataset: r.dataset.as_ref().map_or_else(String::new, |d| {
                format!(
                    "{} train / {} val / {} held out",
                    d.train_images, d.valid_images, d.heldout_images
                )
            }),
            log_tail: r
                .log
                .iter()
                .rev()
                .take(6)
                .rev()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n"),
        })
        .collect();
    let names: Vec<String> = config.models.iter().map(|m| m.effective_name()).collect();
    let results = all
        .iter()
        .rev()
        .filter(|r| r.state == RunState::Done)
        .filter_map(|r| {
            let o = r.output.as_ref()?;
            Some(ResultRow {
                id: r.id.clone(),
                name: o.name.clone(),
                family: format!("{} ({})", r.request.recipe.label(), o.family),
                license: o.license.clone(),
                classes: o.classes.join(", "),
                benchmark: bench_text(r.benchmark.as_ref(), r.benchmark_error.as_deref()),
                finished: r.finished.clone().unwrap_or_default(),
                in_config: names.iter().any(|n| n.eq_ignore_ascii_case(&o.name)),
            })
        })
        .collect();
    let default_model = config
        .effective_default_index()
        .and_then(|i| config.models.get(i))
        .map(|m| m.effective_name())
        .unwrap_or_default();
    let task_rows = super::training_tasks::task_rows(&tasks(state));
    let tasks_busy = task_rows.iter().any(|t| t.busy);
    let jobs_busy = tasks_busy || all.iter().any(|r| r.state.is_busy());
    TrainingTemplate {
        nav: "training",
        version: crate::VERSION,
        message,
        error,
        available: cap.available,
        summary: cap.summary(),
        reasons: cap.reasons.clone(),
        env_ready,
        env_note,
        env_download,
        allow_large: config.allow_large_downloads,
        import_large,
        autolabel_large,
        datasets: {
            let mut v: Vec<DatasetRow> = config
                .training
                .datasets
                .iter()
                .map(|d| DatasetRow {
                    path: d.display().to_string(),
                    exists: config.data_path(d).is_dir(),
                    saved: true,
                })
                .collect();
            for d in crate::training::discover_exports(&config) {
                let path = d.display().to_string();
                if !v.iter().any(|r| r.path == path) {
                    v.push(DatasetRow {
                        path,
                        exists: true,
                        saved: false,
                    });
                }
            }
            v
        },
        families,
        size_cols,
        notices,
        latencies,
        epochs: config.training.epochs,
        scenes: SceneFilter::ALL
            .iter()
            .map(|s| {
                (
                    s.as_str(),
                    match s {
                        SceneFilter::All => "all frames (a general custom model)",
                        SceneFilter::Day => "day frames only",
                        SceneFilter::NightIr => "night + IR frames (a night model)",
                    },
                )
            })
            .collect(),
        public_available: catalog::lila_night().is_some(),
        public_installed: crate::training::public_subset_installed(&config.data_root()),
        public_size: catalog::lila_night()
            .map(|r| catalog::format_size(r.size()))
            .unwrap_or_default(),
        extra_sets: EXTRA_SETS.to_vec(),
        runs: run_rows,
        results,
        models: names,
        default_model,
        training_dir: crate::training::training_dir(&config).display().to_string(),
        training_dir_setting: config
            .training
            .dir
            .as_ref()
            .map(|d| d.display().to_string())
            .unwrap_or_default(),
        training_dir_default: crate::training::config::TrainingConfig::default()
            .dir_in(&config.data_root())
            .display()
            .to_string(),
        models_dir: TrainPaths::for_config(&config)
            .export
            .models
            .display()
            .to_string(),
        tasks: task_rows,
        tasks_busy,
        jobs_busy,
        footage: super::training_tasks::footage(),
        imports: {
            let mut v = crate::training::import::write::read_imports(&dataset_dir);
            v.reverse();
            v.truncate(10);
            v
        },
        dataset_dir: dataset_dir.display().to_string(),
    }
}

// ---------------------------------------------------------------------------------------------
// Forms
// ---------------------------------------------------------------------------------------------

pub(super) fn field<'a>(f: &'a [(String, String)], k: &str) -> Option<&'a str> {
    f.iter()
        .find(|(n, _)| n == k)
        .map(|(_, v)| v.trim())
        .filter(|v| !v.is_empty())
}

pub(super) fn all<'a>(f: &'a [(String, String)], k: &str) -> Vec<&'a str> {
    f.iter()
        .filter(|(n, _)| n == k)
        .map(|(_, v)| v.trim())
        .filter(|v| !v.is_empty())
        .collect()
}

pub(super) fn num<T: std::str::FromStr>(
    f: &[(String, String)],
    k: &str,
    what: &str,
) -> anyhow::Result<Option<T>> {
    field(f, k)
        .map(|v| {
            v.parse::<T>()
                .map_err(|_| anyhow::anyhow!("{what}: '{v}' is not a number"))
        })
        .transpose()
}

/// The runs of a queue form: one [`TrainRequest`] per ticked model (`model=yolo26n`, repeated),
/// sharing the dataset and options fields. Errors name the field. The paths are taken as typed:
/// [`queue`] checks them ([`checked_datasets`], [`checked_folder`]).
pub(crate) fn requests_from_form(
    f: &[(String, String)],
    cap: &Capability,
    default_epochs: u32,
) -> anyhow::Result<(Vec<TrainRequest>, Option<String>)> {
    let models = all(f, "model");
    if models.is_empty() {
        anyhow::bail!("no model ticked: choose at least one family and size");
    }
    let mut datasets: Vec<PathBuf> = all(f, "dataset").into_iter().map(PathBuf::from).collect();
    if let Some(extra) = field(f, "dataset_path") {
        datasets.push(PathBuf::from(extra));
    }
    let scene = match field(f, "scene") {
        Some(s) => SceneFilter::parse(s)
            .ok_or_else(|| anyhow::anyhow!("scene: '{s}' is not all, day or night+ir"))?,
        None => SceneFilter::All,
    };
    let classes: Vec<String> = field(f, "classes")
        .map(|c| {
            c.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let night = field(f, "night_aug").is_some();
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
    let epochs = num(f, "epochs", "epochs")?.unwrap_or(default_epochs);
    let batch: Option<u32> = num(f, "batch", "batch")?;
    let imgsz: Option<u32> = num(f, "imgsz", "input size")?;
    let lr: Option<f32> = num(f, "lr", "learning rate")?;
    let name = field(f, "name").map(str::to_string);
    if name.is_some() && models.len() > 1 {
        anyhow::bail!("a model name works with one ticked model only; clear it or tick one");
    }
    let sets: Vec<String> = all(f, "benchmark_set")
        .into_iter()
        .filter(|s| EXTRA_SETS.contains(s))
        .map(str::to_string)
        .collect();
    let mut out = Vec::new();
    for m in models {
        let (family, size) =
            recipe::parse_label(m).ok_or_else(|| anyhow::anyhow!("model: unknown '{m}'"))?;
        let mut r = TrainRecipe::new(family, size, epochs)?;
        if let Some(i) = imgsz {
            r.imgsz = i;
        }
        r.batch = batch.unwrap_or_else(|| {
            cap.family(family)
                .and_then(|x| x.sizes.iter().find(|s| s.size == size))
                .and_then(|s| s.suggested_batch)
                .unwrap_or(r.batch)
                .min(16)
        });
        r.lr = lr;
        r.augment = augment;
        out.push(TrainRequest {
            recipe: r,
            datasets: datasets.clone(),
            include_public: field(f, "public").is_some(),
            scene,
            classes: (!classes.is_empty()).then(|| classes.clone()),
            rename: Default::default(),
            benchmark_sets: sets.clone(),
            allow_large: field(f, "confirm_large").is_some(),
            install_dir: field(f, "install_dir").map(PathBuf::from),
        });
    }
    Ok((out, name))
}

/// The dataset folders of a queue form: a ticked `dataset` is taken as is when it is one the
/// page offers (`offered`: `training.datasets` and the Review page's exports), anything else and
/// the typed `dataset_path` only when `allow` accepts it ([`super::browse::allowed_path`]: inside
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

/// The folders the Datasets list offers: `training.datasets`, then the Review page's exports.
fn offered_datasets(config: &crate::config::Config) -> Vec<PathBuf> {
    let mut v = config.training.datasets.clone();
    v.extend(crate::training::discover_exports(config));
    v
}

/// [`super::browse::allowed_path`] as a message.
fn allow_in(config: &crate::config::Config) -> impl Fn(&str) -> Result<PathBuf, String> + '_ {
    move |raw| super::browse::allowed_path(config, raw).map_err(|e| e.into_parts().1)
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

/// `GET /training`.
pub(super) async fn page(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || render(&view(&state, None, None))).await
}

/// `GET /v1/training`: capability and runs as JSON (the page's poller).
pub(super) async fn json(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        let cap = crate::training::capability_now();
        Json(serde_json::json!({
            "capability": cap,
            "busy": SERVICE.get().is_some_and(|s| s.busy()),
            "runs": runs(&state),
            "tasks": tasks(&state),
        }))
        .into_response()
    })
    .await
}

/// `POST /training/queue`: queue one run per ticked model.
pub(super) async fn queue(State(shared): State<Shared>, body: axum::body::Bytes) -> Response {
    blocking(move || {
        let state = shared.current();
        let f = super::dataset::parse_form(&body);
        let config = state.config_read().clone();
        let cap = crate::training::capability_now();
        let result = (|| -> anyhow::Result<String> {
            let (mut reqs, name) = requests_from_form(&f, &cap, config.training.epochs)?;
            // Only folders the page offers, or ones the file browser may show / picked natively.
            let allow = allow_in(&config);
            let datasets = checked_datasets(&f, &offered_datasets(&config), &allow)?;
            let install_dir = checked_folder(
                "install the model in",
                field(&f, "install_dir"),
                &config.data_root(),
                &allow,
            )?;
            for r in &mut reqs {
                r.datasets.clone_from(&datasets);
                r.install_dir.clone_from(&install_dir);
            }
            if let Some(a) = &cap.accelerator {
                let paths = TrainPaths::for_config(&config);
                if !env::train_env_ready(&paths, Platform::current(), a.lane())
                    && !config.allow_large_downloads
                    && field(&f, "confirm_large").is_none()
                {
                    let (dl, _) = a.lane().estimate();
                    anyhow::bail!(
                        "the training environment is a large download (about {}): tick \"Allow \
                         downloads over 500 MB\" to confirm, or enable allow_large_downloads",
                        catalog::format_size(dl)
                    );
                }
            }
            let svc = service(&shared);
            let mut queued = Vec::new();
            for r in reqs {
                let job = crate::training::plan_job(&config, r, &cap, name.clone())?;
                let label = format!("{} ({})", job.name, job.request.recipe.label());
                svc.enqueue(job).map_err(anyhow::Error::msg)?;
                queued.push(label);
            }
            Ok(format!("Queued: {}.", queued.join(", ")))
        })();
        match result {
            Ok(m) => render(&view(&state, Some(m), None)),
            Err(e) => render(&view(&state, None, Some(format!("Not queued: {e:#}")))),
        }
    })
    .await
}

/// `POST /training/cancel` (`id`).
pub(super) async fn cancel(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let f = super::dataset::parse_form(&body);
        let id = field(&f, "id").unwrap_or_default().to_string();
        let ok = SERVICE.get().is_some_and(|s| s.cancel(&id));
        if ok {
            render(&view(&state, Some(format!("Cancelling run {id}.")), None))
        } else {
            render(&view(
                &state,
                None,
                Some(format!("No queued or running run {id}.")),
            ))
        }
    })
    .await
}

/// `POST /training/remove` (`id`): forget a finished run (its files stay).
pub(super) async fn remove(State(shared): State<Shared>, body: axum::body::Bytes) -> Response {
    blocking(move || {
        let state = shared.current();
        let f = super::dataset::parse_form(&body);
        let id = field(&f, "id").unwrap_or_default().to_string();
        match service(&shared).remove(&id) {
            Ok(()) => render(&view(
                &state,
                Some(format!("Removed run {id} from the list.")),
                None,
            )),
            Err(e) => render(&view(&state, None, Some(e))),
        }
    })
    .await
}

/// `POST /training/use` (`id`, `role` = `default` | `night` | `verify`, `target`): use a
/// trained model as the default model, or as the night model / verifier of `target`.
pub(super) async fn use_model(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let f = super::dataset::parse_form(&body);
        let result = (|| -> anyhow::Result<String> {
            let id = field(&f, "id").ok_or_else(|| anyhow::anyhow!("no run id"))?;
            let rec = runs(&state)
                .into_iter()
                .find(|r| r.id == id)
                .ok_or_else(|| anyhow::anyhow!("no run {id}"))?;
            let out = rec
                .output
                .ok_or_else(|| anyhow::anyhow!("run {id} has no installed model"))?;
            let role = field(&f, "role").unwrap_or("default").to_string();
            let target = field(&f, "target").map(str::to_string);
            save_config(&state, "training page", |c| {
                apply_use(c, &out, &role, target.as_deref())
            })
        })();
        match result {
            Ok(m) => render(&view(&state, Some(m), None)),
            Err(e) => render(&view(&state, None, Some(format!("Not saved: {e:#}")))),
        }
    })
    .await
}

/// Apply a "Use as" action to `c` (the trained model is added first when missing).
pub(crate) fn apply_use(
    c: &mut crate::config::Config,
    out: &TrainOutput,
    role: &str,
    target: Option<&str>,
) -> anyhow::Result<()> {
    let idx = |c: &crate::config::Config, n: &str| {
        c.models
            .iter()
            .position(|m| m.effective_name().eq_ignore_ascii_case(n))
    };
    if idx(c, &out.name).is_none() {
        crate::training::cli::add_trained_model(c, out)?;
    }
    if let Some(i) = idx(c, &out.name)
        && let Some(m) = c.models.get_mut(i)
    {
        m.enabled = true;
    }
    match role {
        "default" => c.default_model = Some(out.name.clone()),
        "night" | "verify" => {
            let t = match target {
                Some(t) => t.to_string(),
                None => c
                    .effective_default_index()
                    .and_then(|i| c.models.get(i))
                    .map(|m| m.effective_name())
                    .ok_or_else(|| anyhow::anyhow!("no model to attach it to"))?,
            };
            if t.eq_ignore_ascii_case(&out.name) {
                anyhow::bail!("{} cannot be its own {role} model", out.name);
            }
            let i = idx(c, &t).ok_or_else(|| anyhow::anyhow!("no configured model named {t}"))?;
            let m = c
                .models
                .get_mut(i)
                .ok_or_else(|| anyhow::anyhow!("no configured model named {t}"))?;
            if role == "night" {
                m.night_model = Some(out.name.clone());
            } else {
                m.verify = Some(crate::verify::VerifyConfig::with_model(out.name.clone()));
            }
        }
        other => anyhow::bail!("role: '{other}' is not default, night or verify"),
    }
    Ok(())
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

/// `POST /training/datasets` (see [`dataset_edit`]): the export folders offered.
pub(super) async fn datasets(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let f = super::dataset::parse_form(&body);
        let config = state.config_read().clone();
        let result = dataset_edit(&f)
            .and_then(|edit| match edit {
                // The same check as the queue: a browsable folder or one picked natively.
                DatasetEdit::Add(p) => allow_in(&config)(&p.to_string_lossy())
                    .map(DatasetEdit::Add)
                    .map_err(|e| anyhow::anyhow!("another folder: {e}")),
                remove => Ok(remove),
            })
            .and_then(|edit| {
                save_config(&state, "training page", |c| {
                    match &edit {
                        DatasetEdit::Add(p) => {
                            let dir = c.data_path(p);
                            crate::training::dataset::read_classes(&dir)?;
                            if !c.training.datasets.contains(p) {
                                c.training.datasets.push(p.clone());
                            }
                        }
                        DatasetEdit::Remove(p) => c.training.datasets.retain(|d| d != p),
                    }
                    c.training.validate()
                })
            });
        match result {
            Ok(m) => render(&view(&state, Some(m), None)),
            Err(e) => render(&view(&state, None, Some(format!("Not saved: {e:#}")))),
        }
    })
    .await
}

/// `POST /training/dir` (`training_dir`; empty = the default `<data root>/training`): the
/// training folder (`training.dir`), saved with [`save_config`]. The folder must exist and pass
/// [`checked_folder`]. Refused (409) while a run or task is queued or running; otherwise the
/// queue moves to the new folder at once ([`sync_store`]): runs, imports and exports in the old
/// folder stay there.
pub(super) async fn set_dir(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        if SERVICE.get().is_some_and(|s| s.busy()) {
            return (
                StatusCode::CONFLICT,
                render(&view(
                    &state,
                    None,
                    Some(
                        "Not saved: a training run, import or auto-label job is queued or \
                         running; wait for it or cancel it, then change the training folder."
                            .into(),
                    ),
                )),
            )
                .into_response();
        }
        let f = super::dataset::parse_form(&body);
        let config = state.config_read().clone();
        let result = (|| -> anyhow::Result<String> {
            let root = config.data_root();
            let new = checked_folder(
                "training folder",
                field(&f, "training_dir"),
                &root,
                &allow_in(&config),
            )?;
            // The default folder is kept as the default (it follows the data root).
            let default = crate::training::config::TrainingConfig::default().dir_in(&root);
            let new = new.filter(|p| default.canonicalize().ok().as_ref() != Some(p));
            let old = crate::training::training_dir(&config);
            let saved = save_config(&state, "training page", |c| {
                c.training.dir.clone_from(&new);
                Ok(())
            })?;
            let now = crate::training::training_dir(&state.config_read());
            if let Some(svc) = SERVICE.get() {
                sync_store(svc, &now.join(crate::training::service::RUNS_FILE));
            }
            Ok(if now == old {
                saved
            } else {
                format!(
                    "Training folder: {}. Runs, imports and exports in {} stay there (not moved). \
                     {saved}",
                    now.display(),
                    old.display()
                )
            })
        })();
        match result {
            Ok(m) => render(&view(&state, Some(m), None)),
            Err(e) => render(&view(&state, None, Some(format!("Not saved: {e:#}")))),
        }
    })
    .await
}

/// `POST /training/remove-env`: delete the training environment (refused while a run is busy).
pub(super) async fn remove_env(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        if SERVICE.get().is_some_and(|s| s.busy()) {
            return (
                StatusCode::CONFLICT,
                render(&view(
                    &state,
                    None,
                    Some("A run is queued or running; cancel it first.".into()),
                )),
            )
                .into_response();
        }
        let paths = TrainPaths::for_config(&state.config_read());
        match env::remove_train_env(&paths) {
            Ok(()) => render(&view(
                &state,
                Some("Removed the training environment.".into()),
                None,
            )),
            Err(e) => render(&view(&state, None, Some(format!("Not removed: {e:#}")))),
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::detect::{HardwareInfo, apple_gpu};
    use crate::config::Config;
    use crate::model::ModelFamilyKind;

    fn form(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn mac() -> Capability {
        crate::training::capability::capability(
            &HardwareInfo::new("macos", "aarch64", vec![apple_gpu("Apple M1")]),
            Some(16.0),
        )
    }

    #[test]
    fn the_queue_form_makes_one_request_per_model() {
        let f = form(&[
            ("model", "yolo26n"),
            ("model", "rfdetr-nano"),
            ("dataset", "/exports/a"),
            ("dataset_path", "/exports/b"),
            ("scene", "night+ir"),
            ("classes", "cat, raccoon"),
            ("night_aug", "on"),
            ("aug_noise", "0.1"),
            ("epochs", "3"),
            ("benchmark_set", "nir-night"),
            ("benchmark_set", "evil"),
        ]);
        let (reqs, name) = requests_from_form(&f, &mac(), 50).unwrap();
        assert_eq!(name, None);
        assert_eq!(reqs.len(), 2);
        let r = &reqs[1];
        assert_eq!(r.recipe.label(), "rfdetr-nano");
        assert_eq!(r.recipe.epochs, 3);
        assert_eq!(
            r.recipe.augment,
            Augment {
                dark: 0.5,
                noise: 0.1,
                gray: 0.5
            }
        );
        assert_eq!(
            r.datasets,
            [PathBuf::from("/exports/a"), PathBuf::from("/exports/b")]
        );
        assert_eq!(r.scene, SceneFilter::NightIr);
        assert_eq!(
            r.classes.as_deref(),
            Some(&["cat".to_string(), "raccoon".to_string()][..])
        );
        assert_eq!(r.benchmark_sets, ["nir-night"]);
        assert!(r.recipe.batch >= 1 && r.recipe.batch <= 16);
        assert!(!r.include_public && !r.allow_large);
        // The download confirmation also allows the public night subset.
        let (reqs, _) = requests_from_form(
            &form(&[
                ("model", "yolo26n"),
                ("dataset", "/x"),
                ("public", "on"),
                ("confirm_large", "on"),
            ]),
            &mac(),
            50,
        )
        .unwrap();
        assert!(reqs[0].include_public && reqs[0].allow_large);
        assert_eq!(reqs[0].install_dir, None);
        let (reqs, _) = requests_from_form(
            &form(&[("model", "yolo26n"), ("install_dir", " /srv/models ")]),
            &mac(),
            50,
        )
        .unwrap();
        assert_eq!(reqs[0].install_dir, Some(PathBuf::from("/srv/models")));
        let e = requests_from_form(&form(&[("dataset", "/x")]), &mac(), 50).unwrap_err();
        assert!(e.to_string().contains("no model ticked"));
        let e = requests_from_form(
            &form(&[("model", "yolo26n"), ("epochs", "lots")]),
            &mac(),
            50,
        )
        .unwrap_err();
        assert!(e.to_string().contains("epochs"));
        let e = requests_from_form(
            &form(&[("model", "yolo26n"), ("model", "yolo26s"), ("name", "x")]),
            &mac(),
            50,
        )
        .unwrap_err();
        assert!(e.to_string().contains("one ticked model"));
    }

    #[test]
    fn dataset_folders_are_offered_or_allowed() {
        let offered = [PathBuf::from("/exports/a"), PathBuf::from("exports/rel")];
        let allow = |raw: &str| -> Result<PathBuf, String> {
            if raw.starts_with("/ok/") {
                Ok(PathBuf::from(format!("/canon{raw}")))
            } else {
                Err(format!("{raw} is outside the browsable folders"))
            }
        };
        // Offered ones as they are (a relative saved entry too), others through the check.
        let f = form(&[
            ("dataset", "/exports/a"),
            ("dataset", "exports/rel"),
            ("dataset", "/ok/b"),
            ("dataset_path", "/ok/c"),
        ]);
        assert_eq!(
            checked_datasets(&f, &offered, &allow).unwrap(),
            [
                PathBuf::from("/exports/a"),
                PathBuf::from("exports/rel"),
                PathBuf::from("/canon/ok/b"),
                PathBuf::from("/canon/ok/c")
            ]
        );
        // A ticked value the page did not offer.
        let e = checked_datasets(&form(&[("dataset", "/etc/x")]), &offered, &allow).unwrap_err();
        assert!(
            e.to_string().starts_with("datasets: /etc/x is outside"),
            "{e}"
        );
        // The typed folder is always checked, even when it is an offered one.
        let e = checked_datasets(&form(&[("dataset_path", "/exports/a")]), &offered, &allow)
            .unwrap_err();
        assert!(e.to_string().starts_with("another folder:"), "{e}");
        assert!(
            checked_datasets(&form(&[("dataset_path", " ")]), &offered, &allow)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn folder_settings_are_checked() {
        let dir = std::env::temp_dir().join(format!("bop-train-folder-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("f.txt");
        std::fs::write(&file, b"x").unwrap();
        let allow = |raw: &str| -> Result<PathBuf, String> {
            std::path::Path::new(raw)
                .canonicalize()
                .map_err(|_| format!("{raw} does not exist"))
        };
        let root = dir.as_path();
        assert_eq!(checked_folder("x", None, root, &allow).unwrap(), None);
        assert_eq!(checked_folder("x", Some("  "), root, &allow).unwrap(), None);
        assert_eq!(
            checked_folder("x", Some(&dir.display().to_string()), root, &allow).unwrap(),
            Some(dir.canonicalize().unwrap())
        );
        // Relative: under the data root.
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        assert_eq!(
            checked_folder("x", Some("sub"), root, &allow).unwrap(),
            Some(dir.join("sub").canonicalize().unwrap())
        );
        let e = checked_folder("install the model in", Some("nope"), root, &allow).unwrap_err();
        assert!(e.to_string().starts_with("install the model in: "), "{e}");
        let e = checked_folder("x", Some(&file.display().to_string()), root, &allow).unwrap_err();
        assert!(e.to_string().contains("is not a folder"), "{e}");
        let e = checked_folder("training folder", Some("/usr/lib"), root, &allow).unwrap_err();
        assert!(e.to_string().contains("system directory"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_datasets_form_reads_the_queue_forms_buttons() {
        // A saved folder's Remove button inside the queue form: only that folder.
        let f = form(&[
            ("dataset", "/exports/a"),
            ("dataset_path", "/exports/new"),
            ("model", "yolo26n"),
            ("epochs", "50"),
            ("remove", "/exports/a"),
        ]);
        assert_eq!(
            dataset_edit(&f).unwrap(),
            DatasetEdit::Remove(PathBuf::from("/exports/a"))
        );
        // Save folder: the "Another folder" field.
        let f = form(&[
            ("dataset", "/exports/a"),
            ("dataset_path", " /exports/new "),
        ]);
        assert_eq!(
            dataset_edit(&f).unwrap(),
            DatasetEdit::Add(PathBuf::from("/exports/new"))
        );
        // The older form's `add`.
        assert_eq!(
            dataset_edit(&form(&[("add", "/x")])).unwrap(),
            DatasetEdit::Add(PathBuf::from("/x"))
        );
        let e = dataset_edit(&form(&[("dataset_path", ""), ("model", "yolo26n")])).unwrap_err();
        assert!(e.to_string().contains("another folder"), "{e}");
    }

    #[test]
    fn latency_hints_split_into_class_and_text() {
        assert_eq!(
            latency_parts("light: runs anywhere"),
            ("light", "runs anywhere")
        );
        assert_eq!(latency_parts("odd"), ("", "odd"));
    }

    #[test]
    fn use_as_writes_the_default_night_and_verify_models() {
        let dir = std::env::temp_dir().join(format!("bop-train-use-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let onnx = dir.join("night-x.onnx");
        std::fs::write(&onnx, vec![1u8; 2048]).unwrap();
        let yaml = dir.join("night-x.yaml");
        std::fs::write(&yaml, "NAMES:\n  - cat\n").unwrap();
        let out = TrainOutput {
            name: "night-x".into(),
            onnx,
            yaml,
            provenance: dir.join("night-x.provenance.json"),
            family: ModelFamilyKind::Yolo26,
            classes: vec!["cat".into()],
            license: "AGPL-3.0".into(),
            heldout: None,
            device: "mps".into(),
            train_seconds: 1.0,
        };
        let mut c = Config::default();
        c.seed_default_model();
        apply_use(&mut c, &out, "night", None).unwrap();
        assert_eq!(c.models[0].night_model.as_deref(), Some("night-x"));
        assert!(
            c.models
                .iter()
                .any(|m| m.effective_name() == "night-x" && m.enabled)
        );
        apply_use(&mut c, &out, "verify", Some("IPcam-general")).unwrap();
        assert_eq!(c.models[0].verify.as_ref().unwrap().model, "night-x");
        apply_use(&mut c, &out, "default", None).unwrap();
        assert_eq!(c.default_model.as_deref(), Some("night-x"));
        assert!(apply_use(&mut c, &out, "night", Some("night-x")).is_err());
        assert!(apply_use(&mut c, &out, "boss", None).is_err());
        assert!(apply_use(&mut c, &out, "night", Some("nope")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn the_page_and_json_render_and_bad_forms_say_why() {
        use crate::server::test_util::{call, get_text, post_form, test_state_with};
        let _queue = QUEUE_TEST_LOCK.lock().await;
        let root = std::env::temp_dir().join(format!("bop-train-page-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let r2 = root.clone();
        let state = test_state_with(move |c| c.download_dir = Some(r2));
        let (s, page) = get_text(&state, "/training").await;
        assert_eq!(s, StatusCode::OK);
        assert!(
            page.contains("class=\"active\" aria-current=\"page\">Training</a>"),
            "nav"
        );
        assert!(page.contains("This machine"));
        assert!(page.contains("/static/training.js"));
        let (s, json) = get_text(&state, "/v1/training").await;
        assert_eq!(s, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(
            v["capability"]["families"].as_array().unwrap().len() == 6,
            "{v}"
        );
        assert!(v["runs"].as_array().unwrap().is_empty());
        // No model ticked: refused before anything is queued.
        let (_, _, b) = call(&state, post_form("/training/queue", "dataset=%2Fx", true)).await;
        let html = String::from_utf8(b).unwrap();
        assert!(
            html.contains("Not queued") || html.contains("not available"),
            "{html}"
        );
        let (_, _, b) = call(
            &state,
            post_form("/training/use", "id=nope&role=night", true),
        )
        .await;
        assert!(String::from_utf8(b).unwrap().contains("no run nope"));
        let (_, _, b) = call(
            &state,
            post_form("/training/datasets", "add=%2Fno%2Fsuch%2Fdir", true),
        )
        .await;
        assert!(String::from_utf8(b).unwrap().contains("Not saved"));
        // Save folder from the queue form, with its other fields: the same check.
        let (_, _, b) = call(
            &state,
            post_form(
                "/training/datasets",
                "dataset_path=%2Fno%2Fsuch%2Fdir&model=yolo26n&epochs=50&save=1",
                true,
            ),
        )
        .await;
        assert!(String::from_utf8(b).unwrap().contains("Not saved"));
        assert!(state.config_read().training.datasets.is_empty());
        // A real export folder is saved, then removed by its button.
        let export = root.join("exports/review-1");
        std::fs::create_dir_all(&export).unwrap();
        std::fs::write(export.join("classes.txt"), "person\n").unwrap();
        let enc = super::super::encode_uri_component(&export.display().to_string());
        let (_, _, b) = call(
            &state,
            post_form(
                "/training/datasets",
                &format!("dataset_path={enc}&model=yolo26n"),
                true,
            ),
        )
        .await;
        let html = String::from_utf8(b).unwrap();
        assert!(!html.contains("Not saved"), "{html}");
        assert_eq!(
            state.config_read().training.datasets,
            [export.canonicalize().unwrap()]
        );
        let enc = super::super::encode_uri_component(
            &export.canonicalize().unwrap().display().to_string(),
        );
        assert!(html.contains("formaction=\"/training/datasets\" formnovalidate name=\"remove\""));
        let (_, _, _) = call(
            &state,
            post_form(
                "/training/datasets",
                &format!("dataset={enc}&dataset_path=&remove={enc}"),
                true,
            ),
        )
        .await;
        assert!(state.config_read().training.datasets.is_empty());
        // Queue: a dataset the page did not offer, outside the browsable folders, is refused
        // and named; so is an install folder outside them.
        let (_, _, b) = call(
            &state,
            post_form(
                "/training/queue",
                "model=yolo26n&dataset=%2Fno%2Fsuch%2Fexport",
                true,
            ),
        )
        .await;
        let html = String::from_utf8(b).unwrap();
        // Refused either way: outside the roots, or (Windows: drive-relative, so under the
        // data root) not found.
        assert!(
            html.contains("Not queued: datasets: ") && html.contains("/no/such/export"),
            "{html}"
        );
        let (_, _, b) = call(
            &state,
            post_form(
                "/training/queue",
                &format!("model=yolo26n&dataset={enc}&install_dir=%2Fno%2Fsuch%2Fdir"),
                true,
            ),
        )
        .await;
        let html = String::from_utf8(b).unwrap();
        assert!(
            html.contains("Not queued: install the model in: "),
            "{html}"
        );
        // The training folder: shown, set to a folder under the data root, back to the default
        // (once the process's queue is idle: another test's job may still run).
        assert!(page.contains("action=\"/training/dir\""));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        while SERVICE.get().is_some_and(|s| s.busy()) {
            assert!(std::time::Instant::now() < deadline, "the queue stays busy");
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let runs = root.join("my-runs");
        std::fs::create_dir_all(&runs).unwrap();
        let (s, _, b) = call(
            &state,
            post_form("/training/dir", "training_dir=my-runs", true),
        )
        .await;
        let html = String::from_utf8(b).unwrap();
        assert_eq!(s, StatusCode::OK);
        assert!(html.contains("stay there (not moved)"), "{html}");
        assert_eq!(
            state.config_read().training.dir,
            Some(runs.canonicalize().unwrap())
        );
        assert_eq!(
            crate::training::training_dir(&state.config_read()),
            runs.canonicalize().unwrap()
        );
        let (_, _, b) = call(
            &state,
            post_form("/training/dir", "training_dir=%2Fno%2Fsuch%2Fdir", true),
        )
        .await;
        assert!(
            String::from_utf8(b)
                .unwrap()
                .contains("Not saved: training folder: ")
        );
        assert!(state.config_read().training.dir.is_some());
        let (_, _, _) = call(&state, post_form("/training/dir", "training_dir=", true)).await;
        assert_eq!(state.config_read().training.dir, None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn bench_text_reads_well() {
        let b = BenchSummary {
            datasets: vec!["heldout-x".into()],
            device: Some("openvino:cpu".into()),
            ms_p50: Some(12.34),
            ap50: Some(0.5),
            ap50_95: None,
            images: 20,
        };
        assert_eq!(
            bench_text(Some(&b), None),
            "AP50 0.500 \u{b7} 12.3 ms on openvino:cpu \u{b7} 20 images (heldout-x)"
        );
        assert_eq!(bench_text(None, Some("x")), "benchmark failed: x");
    }
}
