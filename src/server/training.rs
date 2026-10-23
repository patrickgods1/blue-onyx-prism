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

/// The process's training queue, started on first use.
pub(super) fn service(shared: &Shared) -> Arc<TrainingService> {
    SERVICE
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
        .clone()
}

/// The import and auto-label jobs (the queue's, else `tasks.json`).
fn tasks(state: &AppState) -> Vec<crate::training::task::TaskRecord> {
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
    estimate: String,
    latency: &'static str,
    why: String,
}

struct FamilyRow {
    title: &'static str,
    license: &'static str,
    available: bool,
    why: String,
    notice: &'static str,
    sizes: Vec<SizeCell>,
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
    allow_large: bool,
    datasets: Vec<DatasetRow>,
    families: Vec<FamilyRow>,
    epochs: u32,
    scenes: Vec<(&'static str, &'static str)>,
    public_available: bool,
    extra_sets: Vec<&'static str>,
    runs: Vec<RunRow>,
    results: Vec<ResultRow>,
    models: Vec<String>,
    default_model: String,
    training_dir: String,
    /// Import and auto-label jobs.
    tasks: Vec<super::training_tasks::TaskRow>,
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
    let families = cap
        .families
        .iter()
        .map(|f| FamilyRow {
            title: f.title,
            license: f.license,
            available: f.available,
            why: f.why.clone().unwrap_or_default(),
            notice: f.family.notice(),
            sizes: f
                .sizes
                .iter()
                .map(|z| SizeCell {
                    label: f.family.base_name(z.size),
                    size: z.size,
                    fits: z.fits,
                    estimate: match (z.vram_mb, z.epoch_s_per_100, z.suggested_batch) {
                        (Some(v), Some(t), Some(b)) => format!(
                            "~{:.1} GB at batch {b}, ~{} / epoch / 100 images",
                            v as f64 / 1024.0,
                            estimate::format_duration(t)
                        ),
                        _ => String::new(),
                    },
                    latency: z.latency,
                    why: z.why.clone().unwrap_or_default(),
                })
                .collect(),
        })
        .collect();
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
        allow_large: config.allow_large_downloads,
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
        extra_sets: EXTRA_SETS.to_vec(),
        runs: run_rows,
        results,
        models: names,
        default_model,
        training_dir: crate::training::training_dir(&config).display().to_string(),
        tasks: super::training_tasks::task_rows(&tasks(state)),
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
/// sharing the dataset and options fields. Errors name the field.
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
        });
    }
    Ok((out, name))
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
            let (reqs, name) = requests_from_form(&f, &cap, config.training.epochs)?;
            if let Some(a) = &cap.accelerator {
                let paths = TrainPaths::for_config(&config);
                if !env::train_env_ready(&paths, Platform::current(), a.lane())
                    && !config.allow_large_downloads
                    && field(&f, "confirm_large").is_none()
                {
                    let (dl, _) = a.lane().estimate();
                    anyhow::bail!(
                        "the training environment is a large download (about {}): tick \"Download \
                         the training environment\" to confirm, or enable allow_large_downloads",
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

/// `POST /training/datasets` (`add` = path | `remove` = path): the export folders offered.
pub(super) async fn datasets(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let f = super::dataset::parse_form(&body);
        let add = field(&f, "add").map(PathBuf::from);
        let rm = field(&f, "remove").map(PathBuf::from);
        let result = save_config(&state, "training page", |c| {
            if let Some(p) = &add {
                let dir = c.data_path(p);
                crate::training::dataset::read_classes(&dir)?;
                if !c.training.datasets.contains(p) {
                    c.training.datasets.push(p.clone());
                }
            }
            if let Some(p) = &rm {
                c.training.datasets.retain(|d| d != p);
            }
            c.training.validate()
        });
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
        assert!(state.config_read().training.datasets.is_empty());
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
