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
use crate::training::compare::{self, CompareRun, Comparison};
use crate::training::dataset::{Balance, BalanceAxis, Excluded, SceneFilter};
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
    /// The family's starting point (`100 ep · patience 15`, [`recipe::recommended`]): what a
    /// blank Epochs / Patience field means.
    recommended: String,
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
    /// Finish now applies: running (or paused) in its training stage, not stopped yet.
    can_finish: bool,
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

/// One GPU of the page's GPU selects (a PC with more than one usable NVIDIA GPU).
struct GpuChoice {
    /// `cuda:N` (the form's `gpu`).
    value: String,
    /// `NVIDIA GeForce RTX 3090 (cuda:1, 24 GB, PyTorch cu126)`.
    label: String,
}

/// One checkbox of the Train form's Runs fieldset.
struct SceneChoice {
    /// The canonical filter (`all`, `day`, `night+ir`, `night`, `ir`; the form's `scene`).
    value: &'static str,
    title: &'static str,
    describe: &'static str,
    /// The model-name prefix (`custom`, `day`, `night`, ...), for the runs line.
    prefix: &'static str,
    /// Shown at once (else under "Other").
    preset: bool,
}

/// The Runs checkboxes: the presets (mixed, day, night + IR), then night only and IR only.
const SCENE_CHOICES: [(SceneFilter, &str, bool); 5] = [
    (SceneFilter::ALL_SCENES, "All scenes (mixed)", true),
    (SceneFilter::DAY, "Day", true),
    (SceneFilter::NIGHT_IR, "Night + IR", true),
    (SceneFilter::new(false, true, false), "Night only", false),
    (SceneFilter::new(false, false, true), "IR only", false),
];

/// One cell of a comparison table (AP50-95, AP50 beneath).
struct CompareCell {
    ap: String,
    ap50: String,
    images: usize,
    best: bool,
}

struct CompareRowView {
    name: String,
    model: String,
    scene: String,
    routed: bool,
    /// The run's state while it has no benchmark (`running`, `failed`); empty when benchmarked.
    state: String,
    /// `best 12 / stopped 27`.
    epochs: String,
    /// Day, night, IR, all.
    cells: Vec<CompareCell>,
    in_scope: String,
    ms: String,
    ms_best: bool,
}

/// The Comparison card of one `compare_group`.
struct CompareCard {
    group: String,
    /// The latest group (and the latest with a result) starts open; older ones collapsed.
    open: bool,
    comparable: bool,
    warnings: Vec<String>,
    rows: Vec<CompareRowView>,
}

/// One axis checkbox of the Train form's Balance fieldset.
struct BalanceChoice {
    /// `class`, `scene`, `camera`, `camera-day` (the form's `balance`).
    value: &'static str,
    title: &'static str,
    hint: &'static str,
    checked: bool,
}

/// The Balance fieldset's defaults: `training.balance` when set, else off.
struct BalanceView {
    axes: Vec<BalanceChoice>,
    on: bool,
    gentle: bool,
    full: bool,
    /// A configured strength other than 0.5 and 1 (a third radio, checked).
    custom: Option<String>,
    max_repeat: u32,
    most: u32,
    cap_on: bool,
    cap: String,
}

impl BalanceView {
    fn of(b: Option<&Balance>) -> Self {
        use crate::training::dataset::balance::{DEFAULT_CAP, MAX_REPEAT};
        let d = Balance::default();
        let b = b.unwrap_or(&d);
        let s = b.strength;
        Self {
            axes: BalanceAxis::ALL
                .iter()
                .map(|&a| BalanceChoice {
                    value: a.as_str(),
                    title: match a {
                        BalanceAxis::Class => "Class",
                        BalanceAxis::Scene => "Scene",
                        BalanceAxis::Camera => "Camera",
                        BalanceAxis::CameraDay => "Camera + day",
                    },
                    hint: a.describe(),
                    checked: b.axes.contains(&a),
                })
                .collect(),
            on: !b.is_off(),
            gentle: s == 0.5,
            full: s == 1.0,
            custom: (s != 0.5 && s != 1.0).then(|| crate::config::clean_f32(s).to_string()),
            max_repeat: b.max_repeat,
            most: MAX_REPEAT,
            cap_on: b.cap_dominant.is_some(),
            cap: crate::config::clean_f32(b.cap_dominant.unwrap_or(DEFAULT_CAP)).to_string(),
        }
    }
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
    /// Where the open-vocabulary labelers run (PyTorch on Metal or CUDA), or why they cannot.
    labeler_device: String,
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
    /// The GPUs the PyTorch jobs may run on, the automatic pick first; empty = nothing to choose
    /// (one GPU, Apple silicon, none): the selects are not shown.
    gpu_choices: Vec<GpuChoice>,
    /// What a GPU on another PyTorch build than the automatic pick's would download
    /// (`PyTorch cu130, about 3.1 GB, for NVIDIA GeForce RTX 5090`), when not set up.
    other_env: Option<String>,
    datasets: Vec<DatasetRow>,
    families: Vec<FamilyRow>,
    /// Most sizes of a family (the grid's size columns).
    size_cols: usize,
    notices: Vec<NoticeRow>,
    latencies: Vec<LatencyRow>,
    /// `training.epochs`, blank = each family's recommendation.
    epochs: String,
    /// `training.patience`, blank = each family's recommendation.
    patience: String,
    scenes: Vec<SceneChoice>,
    balance: BalanceView,
    /// Day / night / mixed comparisons of runs queued together, the latest first.
    comparisons: Vec<CompareCard>,
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
    /// Imports listed in the folder auto-label runs on (`collect.resolved_review_dir`, the
    /// Review page's folder), so their Auto-label buttons find the frames.
    imports: Vec<crate::training::import::write::ImportLine>,
    /// That folder.
    imports_dir: String,
}

/// Finish now applies to `r`: running or paused in its training stage, not stopped and not
/// exporting yet ([`TrainingService::finish`] checks the same).
fn can_finish(r: &RunRecord) -> bool {
    r.state.is_active()
        && r.stage == Some(crate::training::job::TrainStage::Train)
        && r.progress
            .as_ref()
            .is_none_or(|p| p.stopped.is_none() && p.phase.is_none())
}

fn state_str(s: RunState) -> &'static str {
    match s {
        RunState::Queued => "queued",
        RunState::Running => "running",
        RunState::Paused => "paused",
        RunState::Done => "done",
        RunState::Failed => "failed",
        RunState::Cancelled => "cancelled",
    }
}

fn bench_text(b: Option<&BenchSummary>, err: Option<&str>) -> String {
    match (b, err) {
        (Some(b), _) => {
            let mut s = format!(
                "AP50 {} \u{b7} {} on {} \u{b7} {} images ({})",
                b.ap50.map_or_else(|| "n/a".into(), |a| format!("{a:.3}")),
                b.ms_p50
                    .map_or_else(|| "?".into(), |m| format!("{m:.1} ms")),
                b.device.as_deref().unwrap_or("?"),
                b.images,
                b.datasets.join(", ")
            );
            // AP50-95 per scene of the held-out frames, and over the scenes it trained on.
            let scenes: Vec<String> = ["day", "night", "ir"]
                .iter()
                .filter_map(|k| {
                    let a = b.by_scene.get(*k).filter(|a| a.images > 0)?;
                    Some(format!("{} {}", scene_name(k), ap_text(a.ap50_95)))
                })
                .collect();
            if !scenes.is_empty() {
                s.push_str(&format!(" \u{b7} AP50-95 {}", scenes.join(" / ")));
            }
            if let Some(a) = b.in_scope.as_ref().and_then(|a| a.ap50_95) {
                s.push_str(&format!(" \u{b7} in scope {a:.3}"));
            }
            s
        }
        (None, Some(e)) => format!("benchmark failed: {e}"),
        (None, None) => "not benchmarked".into(),
    }
}

/// `day`, `night`, `IR`.
fn scene_name(k: &str) -> &str {
    if k == "ir" { "IR" } else { k }
}

/// An AP as shown (`0.512`), a dash when unknown.
fn ap_text(v: Option<f64>) -> String {
    v.filter(|v| v.is_finite())
        .map_or_else(|| "\u{2014}".into(), |v| format!("{v:.3}"))
}

/// The runs of each `compare_group` in `records` (queue order), groups by their latest run,
/// the newest group first; only groups of two runs or more.
fn compare_groups(records: &[RunRecord]) -> Vec<(String, Vec<&RunRecord>)> {
    let mut groups: Vec<(String, Vec<&RunRecord>, usize)> = Vec::new();
    for (i, r) in records.iter().enumerate() {
        let Some(g) = r.request.compare_group.as_deref() else {
            continue;
        };
        match groups.iter_mut().find(|(k, _, _)| k == g) {
            Some((_, v, last)) => {
                v.push(r);
                *last = i;
            }
            None => groups.push((g.to_string(), vec![r], i)),
        }
    }
    groups.retain(|(_, v, _)| v.len() >= 2);
    groups.sort_by_key(|g| std::cmp::Reverse(g.2));
    groups.into_iter().map(|(g, v, _)| (g, v)).collect()
}

/// A run record as a row of [`compare::compare`].
fn compare_run(r: &RunRecord) -> CompareRun {
    let p = r.progress.as_ref();
    let stop = p.and_then(|p| p.stopped.as_ref());
    CompareRun {
        name: r.name.clone(),
        model: r.request.recipe.label(),
        scene: r.request.scene,
        classes: r.request.classes.clone().unwrap_or_default(),
        excluded: r.request.excluded,
        balance: r.request.balance.clone(),
        best_epoch: stop
            .map(|s| s.best_epoch)
            .or_else(|| p.and_then(|p| p.best.as_ref()).map(|b| b.epoch)),
        // A run that trained to its end stopped at its last epoch.
        stopped_epoch: stop.map(|s| s.epoch).or_else(|| {
            p.filter(|p| p.epoch > 0 && (r.state == RunState::Done || p.phase.is_some()))
                .map(|p| p.epoch)
        }),
        bench: r.benchmark.clone(),
    }
}

/// The comparison of `group` in `records`: its runs and the table; None when no run has it.
fn group_comparison<'a>(
    records: &'a [RunRecord],
    group: &str,
) -> Option<(Vec<&'a RunRecord>, Comparison)> {
    let runs: Vec<&RunRecord> = records
        .iter()
        .filter(|r| r.request.compare_group.as_deref() == Some(group))
        .collect();
    if runs.is_empty() {
        return None;
    }
    let c = compare::compare(&runs.iter().map(|r| compare_run(r)).collect::<Vec<_>>());
    Some((runs, c))
}

/// The Comparison cards of `records` (see [`compare_groups`]); older groups start collapsed.
fn compare_cards(records: &[RunRecord]) -> Vec<CompareCard> {
    let groups = compare_groups(records);
    // Open: the newest group, and the newest one with a result when that one has none yet.
    let with_results = groups
        .iter()
        .position(|(_, runs)| runs.iter().any(|r| r.benchmark.is_some()));
    groups
        .into_iter()
        .enumerate()
        .map(|(i, (group, runs))| {
            let c = compare::compare(&runs.iter().map(|r| compare_run(r)).collect::<Vec<_>>());
            let best = |row: usize, col: &str| c.best_per_column.get(col) == Some(&row);
            let rows = c
                .rows
                .iter()
                .enumerate()
                .map(|(k, row)| {
                    let state = runs
                        .get(k)
                        .filter(|r| !row.routed && r.benchmark.is_none())
                        .map(|r| {
                            if r.state == RunState::Done {
                                "not benchmarked".to_string()
                            } else {
                                state_str(r.state).to_string()
                            }
                        })
                        .unwrap_or_default();
                    let cells = [
                        ("day", &row.day),
                        ("night", &row.night),
                        ("ir", &row.ir),
                        ("all", &row.all),
                    ]
                    .into_iter()
                    .map(|(col, cell)| CompareCell {
                        ap: ap_text(cell.ap50_95),
                        ap50: cell.ap50.map_or_else(String::new, |a| format!("{a:.3}")),
                        images: cell.images,
                        best: !row.routed && best(k, col),
                    })
                    .collect();
                    let epochs = match (row.best_epoch, row.stopped_epoch) {
                        (Some(b), Some(s)) => format!("best {b} / stopped {s}"),
                        (Some(b), None) => format!("best {b}"),
                        (None, Some(s)) => format!("stopped {s}"),
                        (None, None) => "\u{2014}".into(),
                    };
                    CompareRowView {
                        name: row.name.clone(),
                        model: row.model.clone(),
                        scene: row.scene.clone(),
                        routed: row.routed,
                        state,
                        epochs,
                        cells,
                        in_scope: ap_text(row.in_scope),
                        ms: row
                            .ms_p50
                            .map_or_else(|| "\u{2014}".into(), |m| format!("{m:.1}")),
                        ms_best: !row.routed && best(k, "ms_p50"),
                    }
                })
                .collect();
            CompareCard {
                group,
                open: i == 0 || Some(i) == with_results,
                comparable: c.comparable,
                warnings: c.warnings,
                rows,
            }
        })
        .collect()
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
    view_with(state, message, error, runs(state))
}

/// [`view`] of the run records `all` (tests pass their own).
fn view_with(
    state: &AppState,
    message: Option<String>,
    error: Option<String>,
    all: Vec<RunRecord>,
) -> TrainingTemplate {
    let config = state.config_read().clone();
    let dataset_dir = crate::collect::CollectSettings::for_config(&config).dir;
    let imports_dir = config.collect.resolved_review_dir(&config.data_root());
    let cap: Capability = crate::training::capability_with_runs(&all);
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
    // A chosen GPU on another lane sets up that lane's environment (it replaces the other one).
    let gpu_choices: Vec<GpuChoice> = if cap.has_gpu_choice() {
        cap.cuda_options
            .iter()
            .map(|a| GpuChoice {
                value: a.device_arg(),
                label: a.choice_label(),
            })
            .collect()
    } else {
        Vec::new()
    };
    let other_env = {
        let auto_lane = cap.accelerator.as_ref().map(|a| a.lane());
        let mut parts: Vec<String> = Vec::new();
        for a in cap.cuda_options.iter().skip(1) {
            let lane = a.lane();
            if Some(lane) == auto_lane {
                continue;
            }
            let name = a.name();
            let tag = format!("PyTorch {}", lane.as_str());
            match parts.iter_mut().find(|p| p.starts_with(&tag)) {
                Some(p) => {
                    p.push_str(", ");
                    p.push_str(name);
                }
                None if !env::train_env_ready(&paths, platform, lane) => parts.push(format!(
                    "{tag}, about {}, for {name}",
                    catalog::format_size(lane.estimate().0)
                )),
                None => {}
            }
        }
        (!parts.is_empty()).then(|| parts.join("; "))
    };
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
            recommended: {
                let r = recipe::recommended(f.family);
                format!("{} ep \u{b7} patience {}", r.epochs, r.patience)
            },
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
            if let Some(o) = &other_env {
                label.push(format!(
                    "the training environment of another GPU if you choose it ({o})"
                ));
            }
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
            can_finish: can_finish(r),
            percent: r.percent,
            describe: r.describe(),
            dataset: r.dataset.as_ref().map_or_else(String::new, |d| {
                let mut t = format!(
                    "{} train / {} val / {} held out",
                    d.train_images, d.valid_images, d.heldout_images
                );
                if let Some(b) = &d.balance_summary {
                    t.push_str(&format!(" \u{b7} balanced: {b}"));
                }
                t
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
        labeler_device: cap.accelerator.as_ref().map_or_else(
            || "not available: needs an NVIDIA GPU or Apple silicon".to_string(),
            |a| format!("PyTorch on {}", a.torch_backend()),
        ),
        reasons: cap.reasons.clone(),
        env_ready,
        env_note,
        env_download,
        gpu_choices,
        other_env,
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
        epochs: config
            .training
            .epochs
            .map(|e| e.to_string())
            .unwrap_or_default(),
        patience: config
            .training
            .patience
            .map(|e| e.to_string())
            .unwrap_or_default(),
        scenes: SCENE_CHOICES
            .iter()
            .map(|(s, title, preset)| SceneChoice {
                value: s.as_str(),
                title,
                describe: s.describe(),
                prefix: s.label(),
                preset: *preset,
            })
            .collect(),
        balance: BalanceView::of(config.training.balance.as_ref()),
        comparisons: compare_cards(&all),
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
            let mut v = crate::training::import::write::read_imports(&imports_dir);
            v.reverse();
            v.truncate(10);
            v
        },
        imports_dir: imports_dir.display().to_string(),
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
        include_public: field(f, "public").is_some(),
        scenes: scenes_of(f)?,
        classes: classes_of(f),
        excluded,
        balance: balance_of(f, training)?,
    })
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
        cap_dominant: field(f, "balance_cap_on")
            .is_some()
            .then(|| cap.unwrap_or(DEFAULT_CAP)),
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
/// typed: [`queue`] checks them ([`checked_datasets`], [`checked_folder`]).
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
    let epochs: Option<u32> = num(f, "epochs", "epochs")?;
    let patience: Option<u32> = num(f, "patience", "patience")?;
    let batch: Option<u32> = num(f, "batch", "batch")?;
    let imgsz: Option<u32> = num(f, "imgsz", "input size")?;
    let lr: Option<f32> = num(f, "lr", "learning rate")?;
    let multi_scale = match field(f, "multi_scale") {
        Some(v) => recipe::MultiScale::parse(v)
            .ok_or_else(|| anyhow::anyhow!("multi-scale: '{v}' is not auto, on or off"))?,
        None => recipe::MultiScale::Auto,
    };
    let name = field(f, "name").map(str::to_string);
    if name.is_some() && models.len() * scenes.len() > 1 {
        anyhow::bail!(
            "a model name works with one run only (one ticked model, one scene); clear it or \
             tick one of each"
        );
    }
    // Several scenes: one comparison group for every run (same held-out frames, one table).
    let compare_group =
        (scenes.len() > 1).then(|| format!("cmp-{}", chrono::Local::now().format("%Y%m%d-%H%M%S")));
    let sets: Vec<String> = all(f, "benchmark_set")
        .into_iter()
        .filter(|s| EXTRA_SETS.contains(s))
        .map(str::to_string)
        .collect();
    let mut out = Vec::new();
    for (m, scene) in models
        .iter()
        .flat_map(|m| scenes.iter().map(move |s| (*m, *s)))
    {
        let (family, size) =
            recipe::parse_label(m).ok_or_else(|| anyhow::anyhow!("model: unknown '{m}'"))?;
        let mut r = TrainRecipe::new(family, size)?;
        r.epochs = epochs.unwrap_or_else(|| training.epochs_for(family));
        // A typed patience is checked against the epochs; a default one is capped by them.
        r.patience = patience.unwrap_or_else(|| training.patience_for(family).min(r.epochs));
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
        r.multi_scale = multi_scale;
        out.push(TrainRequest {
            recipe: r,
            datasets: data.datasets.clone(),
            include_public: data.include_public,
            scene,
            classes: (!data.classes.is_empty()).then(|| data.classes.clone()),
            rename: Default::default(),
            benchmark_sets: sets.clone(),
            allow_large: field(f, "confirm_large").is_some(),
            install_dir: field(f, "install_dir").map(PathBuf::from),
            gpu: gpu.clone(),
            excluded: data.excluded,
            balance: data.balance.clone(),
            compare_group: compare_group.clone(),
        });
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
        let cap = crate::training::capability_with_runs(&runs(&state));
        let columns = state.config_read().training.effective_metric_columns();
        Json(serde_json::json!({
            "capability": cap,
            "busy": SERVICE.get().is_some_and(|s| s.busy()),
            "runs": runs(&state),
            "tasks": tasks(&state),
            "metric_columns": columns,
            "metrics": crate::training::progress::METRICS,
        }))
        .into_response()
    })
    .await
}

/// The class inventory of `datasets` (query values: offered folders as they are, others through
/// `allow`; see [`checked_datasets`]): every class with its boxes per scene
/// ([`crate::training::dataset::inventory`]), the classes in list order (`classes.txt` of each
/// folder, then the rest), frames per scene, and one error per folder that could not be used.
pub(crate) fn dataset_inventory(
    datasets: &[&str],
    offered: &[PathBuf],
    data_path: &dyn Fn(&std::path::Path) -> PathBuf,
    allow: &dyn Fn(&str) -> Result<PathBuf, String>,
) -> serde_json::Value {
    use crate::training::dataset;
    let mut sources = Vec::new();
    let mut errors = Vec::new();
    for d in datasets {
        let p = PathBuf::from(d);
        let dir = if offered.contains(&p) {
            Ok(data_path(&p))
        } else {
            allow(d)
        };
        match dir.and_then(|dir| dataset::read_export(&dir).map_err(|e| format!("{e:#}"))) {
            Ok(s) => sources.push(s),
            Err(e) => errors.push(serde_json::json!({ "dataset": d, "message": e })),
        }
    }
    let classes = dataset::inventory(&sources, &Default::default());
    let mut order: Vec<String> = Vec::new();
    for c in sources
        .iter()
        .flat_map(|s| s.classes.iter())
        .chain(classes.iter().map(|c| &c.name))
    {
        if !order.contains(c) {
            order.push(c.clone());
        }
    }
    let mut frames = dataset::SceneCounts::default();
    for f in sources.iter().flat_map(|s| s.frames.iter()) {
        *match f.scene.as_deref() {
            Some("day") => &mut frames.day,
            Some("night") => &mut frames.night,
            Some("ir") => &mut frames.ir,
            _ => &mut frames.unknown,
        } += 1;
    }
    serde_json::json!({
        "success": true,
        "classes": classes,
        "order": order,
        "frames": frames,
        "errors": errors,
    })
}

/// `GET /v1/training/dataset-classes?dataset=<path>&dataset=...`: the class picker's classes
/// with their box counts per scene ([`dataset_inventory`]). A folder that cannot be used is
/// listed in `errors` (200: the page asks while folders are ticked or typed). A browser request
/// marked `Sec-Fetch-Site: cross-site` is 403.
pub(super) async fn dataset_classes(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(q): axum::extract::Query<Vec<(String, String)>>,
) -> Response {
    if super::guard::fetch_site_cross(&headers) {
        return json_fail(
            StatusCode::FORBIDDEN,
            "Cross-site request refused: the class list answers this server's own pages only.",
        );
    }
    blocking(move || {
        let config = state.config_read().clone();
        let datasets = all(&q, "dataset");
        Json(dataset_inventory(
            &datasets,
            &offered_datasets(&config),
            &|p| config.data_path(p),
            &allow_in(&config),
        ))
        .into_response()
    })
    .await
}

/// The balance preview of a queue form ([`data_fields`]; the models are not needed): one run
/// per scene filter, each [`crate::training::preview_plan`] (the job's sources and options,
/// nothing written) as `{scene, title, report, warnings, classes, train_frames, train_images,
/// valid_frames, heldout_frames, dropped, error}` (`report` null when balancing is off; `error`
/// set when that run cannot be planned). The folders are checked like the queue's
/// ([`checked_datasets`]); a bad field or folder is an error naming it.
pub(crate) fn balance_preview_json(
    f: &[(String, String)],
    config: &crate::config::Config,
) -> anyhow::Result<serde_json::Value> {
    let data = data_fields(f, &config.training)?;
    let datasets = checked_datasets(f, &offered_datasets(config), &allow_in(config))?;
    if datasets.is_empty() && config.training.datasets.is_empty() {
        anyhow::bail!("no dataset: tick a review export folder or type one under Another folder");
    }
    // A placeholder model: the plan depends on the data fields only.
    let recipe = TrainRecipe::new(catalog::TrainFamily::Yolo26, "n")?;
    let runs: Vec<serde_json::Value> = data
        .scenes
        .iter()
        .map(|&scene| {
            let title = SCENE_CHOICES
                .iter()
                .find(|(s, _, _)| *s == scene)
                .map_or_else(
                    || scene.describe().to_string(),
                    |(_, t, _)| (*t).to_string(),
                );
            let request = TrainRequest {
                recipe: recipe.clone(),
                datasets: datasets.clone(),
                include_public: data.include_public,
                scene,
                classes: (!data.classes.is_empty()).then(|| data.classes.clone()),
                rename: Default::default(),
                benchmark_sets: Vec::new(),
                allow_large: false,
                install_dir: None,
                gpu: None,
                excluded: data.excluded,
                balance: data.balance.clone(),
                compare_group: None,
            };
            match crate::training::preview_plan(config, &request) {
                Ok(p) => serde_json::json!({
                    "scene": scene.as_str(),
                    "title": title,
                    "report": p.balance,
                    "warnings": p.warnings,
                    "classes": p.classes,
                    "train_frames": p.train.len(),
                    "train_images": p.train.iter().map(|(_, n)| *n as usize).sum::<usize>(),
                    "valid_frames": p.valid.len(),
                    "heldout_frames": p.heldout.len(),
                    "dropped": p.stats.balance_dropped,
                    "error": null,
                }),
                Err(e) => serde_json::json!({
                    "scene": scene.as_str(),
                    "title": title,
                    "report": null,
                    "warnings": [],
                    "error": format!("{e:#}"),
                }),
            }
        })
        .collect();
    Ok(serde_json::json!({
        "success": true,
        "balance": data.balance,
        "runs": runs,
    }))
}

/// `POST /v1/training/balance-preview` with the queue form's fields (urlencoded): what
/// balancing would do to each run's training frames ([`balance_preview_json`]). 400 with
/// `{success: false, message}` for a bad field or folder; a browser request marked
/// `Sec-Fetch-Site: cross-site` is 403.
pub(super) async fn balance_preview(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    if super::guard::fetch_site_cross(&headers) {
        return json_fail(
            StatusCode::FORBIDDEN,
            "Cross-site request refused: the balance preview answers this server's own pages \
             only.",
        );
    }
    blocking(move || {
        let f = super::dataset::parse_form(&body);
        let config = state.config_read().clone();
        match balance_preview_json(&f, &config) {
            Ok(v) => Json(v).into_response(),
            Err(e) => json_fail(StatusCode::BAD_REQUEST, format!("{e:#}")),
        }
    })
    .await
}

/// `GET /v1/training/compare?group=<id>`: the day / night / mixed comparison of the runs queued
/// together as `group` ([`compare::compare`]): `runs` (id, name, state), `comparison` (rows,
/// best per column, comparable, warnings) and `text` (the CLI's table). 400 without a group,
/// 404 for an unknown one.
pub(super) async fn compare_json(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(q): axum::extract::Query<Vec<(String, String)>>,
) -> Response {
    blocking(move || {
        let Some(group) = field(&q, "group").map(str::to_string) else {
            return json_fail(
                StatusCode::BAD_REQUEST,
                "give a comparison group (group=...)",
            );
        };
        let records = runs(&state);
        match group_comparison(&records, &group) {
            Some((rs, c)) => Json(serde_json::json!({
                "success": true,
                "group": group,
                "runs": rs
                    .iter()
                    .map(|r| serde_json::json!({
                        "id": r.id, "name": r.name, "state": state_str(r.state)
                    }))
                    .collect::<Vec<_>>(),
                "text": c.render_text(),
                "comparison": c,
            }))
            .into_response(),
            None => json_fail(
                StatusCode::NOT_FOUND,
                format!("no comparison group {group} in the run list"),
            ),
        }
    })
    .await
}

/// Body of `POST /v1/training/metric-columns`.
#[derive(serde::Deserialize)]
struct MetricColumnsBody {
    columns: Vec<String>,
}

/// A JSON error answer (`{success: false, message}`).
fn json_fail(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(serde_json::json!({ "success": false, "message": message.into() })),
    )
        .into_response()
}

/// `POST /v1/training/metric-columns` with `{columns: ["map", "f2", ...]}`: saves the per-epoch
/// table's columns (`training.metric_columns`, [`crate::training::config::
/// validate_metric_columns`]; the default list is stored as empty) and answers the effective
/// list. 415 without a JSON body, 400 on a bad list.
pub(super) async fn metric_columns(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let is_json = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("application/json"));
    if !is_json {
        return json_fail(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "send the body as JSON (Content-Type: application/json)",
        );
    }
    let b: MetricColumnsBody = match serde_json::from_slice(&body) {
        Ok(b) => b,
        Err(e) => return json_fail(StatusCode::BAD_REQUEST, format!("invalid JSON body: {e}")),
    };
    blocking(move || {
        use crate::training::config::{DEFAULT_METRIC_COLUMNS, validate_metric_columns};
        let cols: Vec<String> = b.columns.iter().map(|c| c.trim().to_string()).collect();
        if cols.is_empty() {
            return json_fail(
                StatusCode::BAD_REQUEST,
                "Not saved: choose at least one column",
            );
        }
        if let Err(e) = validate_metric_columns(&cols) {
            return json_fail(StatusCode::BAD_REQUEST, format!("Not saved: {e:#}"));
        }
        let stored = if cols == DEFAULT_METRIC_COLUMNS {
            Vec::new()
        } else {
            cols.clone()
        };
        let result = save_config(&state, "training page", |c| {
            c.training.metric_columns.clone_from(&stored);
            c.training.validate()
        });
        match result {
            Ok(msg) => Json(serde_json::json!({
                "success": true,
                "message": msg,
                "columns": cols,
            }))
            .into_response(),
            Err(e) => json_fail(StatusCode::BAD_REQUEST, format!("Not saved: {e:#}")),
        }
    })
    .await
}

/// `POST /training/queue`: queue one run per ticked model.
pub(super) async fn queue(State(shared): State<Shared>, body: axum::body::Bytes) -> Response {
    blocking(move || {
        let state = shared.current();
        let f = super::dataset::parse_form(&body);
        let config = state.config_read().clone();
        let cap = crate::training::capability_with_runs(&runs(&state));
        let result = (|| -> anyhow::Result<String> {
            let (mut reqs, name) = requests_from_form(&f, &cap, &config.training)?;
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
            // The chosen GPU's lane (another GPU may need the other PyTorch build).
            if let Some(a) = &cap.resolve_gpu(field(&f, "gpu"))? {
                let paths = TrainPaths::for_config(&config);
                if !env::train_env_ready(&paths, Platform::current(), a.lane())
                    && !config.allow_large_downloads
                    && field(&f, "confirm_large").is_none()
                {
                    let (dl, _) = a.lane().estimate();
                    anyhow::bail!(
                        "the training environment (PyTorch {}) is a large download (about {}): \
                         tick \"Allow downloads over 500 MB\" to confirm, or enable \
                         allow_large_downloads",
                        a.lane().as_str(),
                        catalog::format_size(dl)
                    );
                }
            }
            let svc = service(&shared);
            // Default names, made unique here: a comparison queued again within the minute
            // (the stamp's resolution) gets `-2`, ... instead of a refusal.
            let stamp = chrono::Local::now().format("%Y%m%d-%H%M").to_string();
            let busy: Vec<String> = svc
                .runs()
                .into_iter()
                .filter(|r| r.state.is_busy())
                .map(|r| r.name)
                .collect();
            let models_dir = TrainPaths::for_config(&config).export.models;
            let mut names: Vec<String> = Vec::new();
            let mut jobs = Vec::new();
            for r in reqs {
                let name = name.clone().unwrap_or_else(|| {
                    let base = r.recipe.default_name(r.scene.label(), &stamp);
                    let dir = r.install_dir.as_deref().unwrap_or(&models_dir);
                    unique_name(&base, |n| {
                        names.iter().chain(&busy).any(|o| o.eq_ignore_ascii_case(n))
                            || config
                                .models
                                .iter()
                                .any(|m| m.effective_name().eq_ignore_ascii_case(n))
                            || dir.join(format!("{n}.onnx")).exists()
                    })
                });
                names.push(name.clone());
                jobs.push(crate::training::plan_job(&config, r, &cap, Some(name))?);
            }
            let mut queued = Vec::new();
            for job in jobs {
                let on = job
                    .request
                    .gpu
                    .as_ref()
                    .map(|_| format!(" on {}", job.accelerator.device_arg()))
                    .unwrap_or_default();
                let label = format!("{} ({}{on})", job.name, job.request.recipe.label());
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

/// `POST /training/finish` (`id`): stop the running run after its current epoch and export its
/// best epoch ([`TrainingService::finish`]).
pub(super) async fn finish(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let f = super::dataset::parse_form(&body);
        let id = field(&f, "id").unwrap_or_default().to_string();
        let r = SERVICE
            .get()
            .ok_or_else(|| format!("no run {id}"))
            .and_then(|s| s.finish(&id));
        match r {
            Ok(m) => render(&view(&state, Some(m), None)),
            Err(e) => render(&view(&state, None, Some(format!("Not finished: {e}.")))),
        }
    })
    .await
}

/// `POST /training/pause` (`id`): suspend the running run or job's helper processes where they
/// are (models stay loaded, GPU memory stays held); the queue keeps its place.
pub(super) async fn pause(State(state): State<Arc<AppState>>, body: axum::body::Bytes) -> Response {
    pause_or_resume(state, body, true).await
}

/// `POST /training/resume` (`id`): continue a paused run or job.
pub(super) async fn resume(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    pause_or_resume(state, body, false).await
}

async fn pause_or_resume(state: Arc<AppState>, body: axum::body::Bytes, pause: bool) -> Response {
    blocking(move || {
        let f = super::dataset::parse_form(&body);
        let id = field(&f, "id").unwrap_or_default().to_string();
        let ok = SERVICE
            .get()
            .is_some_and(|s| if pause { s.pause(&id) } else { s.resume(&id) });
        match (ok, pause) {
            (true, true) => render(&view(
                &state,
                Some(format!(
                    "Paused {id}: its processes are suspended (models stay loaded and GPU memory \
                     stays in use); nothing else starts until you resume or cancel it."
                )),
                None,
            )),
            (true, false) => render(&view(&state, Some(format!("Resumed {id}.")), None)),
            (false, true) => render(&view(&state, None, Some(format!("No running job {id}.")))),
            (false, false) => render(&view(&state, None, Some(format!("No paused job {id}.")))),
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
    fn the_queue_form_takes_a_gpu_and_suggests_its_batch() {
        use crate::backend::detect::{ComputeCapability, CudaInfo, GpuAdapter, GpuVendor};
        let gpu = |name: &str, vram_mb: u64, ordinal: u32| GpuAdapter {
            vendor: GpuVendor::Nvidia,
            name: name.into(),
            vram_mb,
            index: ordinal,
            discrete: true,
            cuda: Some(CudaInfo {
                ordinal,
                compute_capability: ComputeCapability::new(8, 6),
            }),
        };
        let cap = crate::training::capability::capability(
            &HardwareInfo::new(
                "linux",
                "x86_64",
                vec![gpu("RTX 3090", 24576, 0), gpu("RTX 3050", 4096, 1)],
            )
            .with_cuda_driver(12, 8),
            None,
        );
        let post = |g: &str| {
            requests_from_form(
                &form(&[("model", "rfdetr-medium"), ("dataset", "/x"), ("gpu", g)]),
                &cap,
                &Default::default(),
            )
            .map(|(r, _)| r.into_iter().next().unwrap())
        };
        let auto = post("").unwrap();
        assert_eq!(auto.gpu, None);
        let small = post("cuda:1").unwrap();
        assert_eq!(small.gpu.as_deref(), Some("cuda:1"));
        assert!(
            small.recipe.batch < auto.recipe.batch,
            "4 GB suggests a smaller batch: {} vs {}",
            small.recipe.batch,
            auto.recipe.batch
        );
        let e = post("cuda:5").unwrap_err().to_string();
        assert!(
            e.contains("no such CUDA device") && e.contains("RTX 3050"),
            "{e}"
        );
        let e = requests_from_form(
            &form(&[("model", "yolo26n"), ("dataset", "/x"), ("gpu", "cuda:0")]),
            &mac(),
            &Default::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("no GPU to choose"), "{e}");
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
        let (reqs, name) = requests_from_form(&f, &mac(), &Default::default()).unwrap();
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
        assert_eq!(r.scene, SceneFilter::NIGHT_IR);
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
            &Default::default(),
        )
        .unwrap();
        assert!(reqs[0].include_public && reqs[0].allow_large);
        assert_eq!(reqs[0].install_dir, None);
        let (reqs, _) = requests_from_form(
            &form(&[("model", "yolo26n"), ("install_dir", " /srv/models ")]),
            &mac(),
            &Default::default(),
        )
        .unwrap();
        assert_eq!(reqs[0].install_dir, Some(PathBuf::from("/srv/models")));
        let e = requests_from_form(&form(&[("dataset", "/x")]), &mac(), &Default::default())
            .unwrap_err();
        assert!(e.to_string().contains("no model ticked"));
        let e = requests_from_form(
            &form(&[("model", "yolo26n"), ("epochs", "lots")]),
            &mac(),
            &Default::default(),
        )
        .unwrap_err();
        assert!(e.to_string().contains("epochs"));
        let e = requests_from_form(
            &form(&[("model", "yolo26n"), ("model", "yolo26s"), ("name", "x")]),
            &mac(),
            &Default::default(),
        )
        .unwrap_err();
        assert!(e.to_string().contains("one ticked model"));
    }

    #[test]
    fn form_epochs_and_patience_fall_back_to_the_config_then_the_family() {
        let post = |extra: &[(&str, &str)], t: &crate::training::config::TrainingConfig| {
            let mut f = vec![("model", "yolo26n"), ("model", "dfine-s")];
            f.extend_from_slice(extra);
            let (reqs, _) = requests_from_form(&form(&f), &mac(), t).unwrap();
            reqs.iter()
                .map(|r| (r.recipe.epochs, r.recipe.patience))
                .collect::<Vec<_>>()
        };
        let none = crate::training::config::TrainingConfig::default();
        assert_eq!(post(&[], &none), [(100, 15), (80, 10)]);
        assert_eq!(
            post(&[("epochs", ""), ("patience", "")], &none),
            [(100, 15), (80, 10)]
        );
        let set = crate::training::config::TrainingConfig {
            epochs: Some(12),
            patience: Some(4),
            ..none.clone()
        };
        assert_eq!(post(&[], &set), [(12, 4), (12, 4)]);
        assert_eq!(
            post(&[("epochs", "6"), ("patience", "0")], &set),
            [(6, 0), (6, 0)]
        );
        // A recommended patience above typed epochs is capped by them.
        assert_eq!(post(&[("epochs", "5")], &none), [(5, 5), (5, 5)]);
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
            [super::super::browse::plain(&export.canonicalize().unwrap())]
        );
        let enc = super::super::encode_uri_component(
            &super::super::browse::plain(&export.canonicalize().unwrap())
                .display()
                .to_string(),
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
        let runs = super::super::browse::plain(&runs.canonicalize().unwrap());
        assert_eq!(state.config_read().training.dir, Some(runs.clone()));
        assert_eq!(crate::training::training_dir(&state.config_read()), runs);
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

    /// A running job shows Pause and Cancel, a paused one Resume and Cancel; the JSON says
    /// `paused`; pausing a job that is not on the queue's thread is refused.
    #[tokio::test]
    async fn metric_columns_are_listed_saved_and_checked() {
        use crate::server::test_util::{REMOTE, call, from_peer, get_text, test_state_with};
        use axum::body::Body;
        use axum::http::{Request, header};
        let root = std::env::temp_dir().join(format!("bop-train-cols-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let r2 = root.clone();
        let state = test_state_with(move |c| c.download_dir = Some(r2));
        let post = |body: &str| {
            Request::post("/v1/training/metric-columns")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };
        let json = |b: Vec<u8>| -> serde_json::Value { serde_json::from_slice(&b).unwrap() };
        let (s, text) = get_text(&state, "/v1/training").await;
        assert_eq!(s, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            v["metric_columns"],
            serde_json::json!(["map", "map50", "f1", "f2", "precision", "recall", "loss"])
        );
        let metrics = v["metrics"].as_array().unwrap();
        assert_eq!(metrics.len(), crate::training::progress::METRICS.len());
        let map = metrics.iter().find(|m| m["id"] == "map").unwrap();
        assert_eq!(map["label"], "mAP 50-95");
        assert_eq!(map["short"], "mAP");
        assert_eq!(map["higher_better"], true);
        assert_eq!(map["format"], "fixed3");

        // Saved (trimmed), raw names allowed, answered and served.
        let (s, _, b) = call(
            &state,
            post(r#"{"columns": [" f2 ", "map", "val/AP/raccoon"]}"#),
        )
        .await;
        let v = json(b);
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["success"], true);
        assert_eq!(
            v["columns"],
            serde_json::json!(["f2", "map", "val/AP/raccoon"])
        );
        assert_eq!(
            state.config_read().training.metric_columns,
            ["f2", "map", "val/AP/raccoon"]
        );
        let (_, text) = get_text(&state, "/v1/training").await;
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            v["metric_columns"],
            serde_json::json!(["f2", "map", "val/AP/raccoon"])
        );

        // Bad lists: 400, nothing changes.
        let too_many: Vec<String> = (0..21).map(|i| format!("m{i}")).collect();
        for bad in [
            r#"{"columns": []}"#.to_string(),
            r#"{"columns": ["map", ""]}"#.to_string(),
            r#"{"columns": ["map", "map"]}"#.to_string(),
            format!(r#"{{"columns": ["{}"]}}"#, "x".repeat(65)),
            serde_json::json!({ "columns": too_many }).to_string(),
            r#"{"columns": "map"}"#.to_string(),
        ] {
            let (s, _, b) = call(&state, post(&bad)).await;
            let v = json(b);
            assert_eq!(s, StatusCode::BAD_REQUEST, "{bad}: {v}");
            assert_eq!(v["success"], false);
            assert!(v["message"].as_str().is_some_and(|m| !m.is_empty()));
        }
        assert_eq!(
            state.config_read().training.metric_columns,
            ["f2", "map", "val/AP/raccoon"]
        );
        // Not JSON: 415.
        let form = Request::post("/v1/training/metric-columns")
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
            .body(Body::from("columns=map"))
            .unwrap();
        let (s, _, _) = call(&state, form).await;
        assert_eq!(s, StatusCode::UNSUPPORTED_MEDIA_TYPE);
        // Admin only: a remote peer is refused.
        let (s, _, _) = call(&state, from_peer(post(r#"{"columns": ["map"]}"#), REMOTE)).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        // The default list is stored as empty (not written to the config file).
        let (s, _, _) = call(
            &state,
            post(r#"{"columns": ["map", "map50", "f1", "f2", "precision", "recall", "loss"]}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert!(state.config_read().training.metric_columns.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn busy_jobs_show_pause_resume_and_cancel() {
        use crate::server::test_util::{call, get_text, post_form, test_state_with};
        use crate::training::task::{TaskKind, TaskRecord};
        let _queue = QUEUE_TEST_LOCK.lock().await;
        let root = std::env::temp_dir().join(format!("bop-train-pause-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let r2 = root.clone();
        let state = test_state_with(move |c| c.download_dir = Some(r2));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        while SERVICE.get().is_some_and(|s| s.busy()) {
            assert!(std::time::Instant::now() < deadline, "the queue stays busy");
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        let svc = service(&Shared::from(state.clone()));
        let rec = |id: &str, st: RunState| TaskRecord {
            id: id.into(),
            kind: TaskKind::Autolabel,
            title: format!("auto-label {id}"),
            created: String::new(),
            state: st,
            stage: Some("labelling".into()),
            step: 2,
            steps: 3,
            percent: 40,
            detail: "5 of 9 frames".into(),
            log: vec![],
            summary: None,
            result: None,
            error: None,
            finished: None,
            after: None,
        };
        let (run, paused) = ("pause-test-running", "pause-test-paused");
        svc.put_task_record(rec(run, RunState::Running));
        svc.put_task_record(rec(paused, RunState::Paused));
        let (s, page) = get_text(&state, "/training").await;
        assert_eq!(s, StatusCode::OK);
        let row = |id: &str| {
            let i = page
                .find(&format!("data-task=\"{id}\""))
                .expect("the job's row");
            let j = i + page[i..].find("</tr>").unwrap();
            page[i..j].to_string()
        };
        let (r, p) = (row(run), row(paused));
        let shown = |form: &str| {
            format!("action=\"/training/{form}\" class=\"inline\" data-field=\"{form}\">")
        };
        let hidden = |form: &str| {
            format!("action=\"/training/{form}\" class=\"inline\" data-field=\"{form}\" hidden>")
        };
        assert!(
            r.contains(&shown("pause")) && r.contains(&hidden("resume")),
            "{r}"
        );
        assert!(
            p.contains(&hidden("pause")) && p.contains(&shown("resume")),
            "{p}"
        );
        assert!(r.contains("/training/cancel") && p.contains("/training/cancel"));
        assert!(p.contains("paused (labelling 2/3): 5 of 9 frames"), "{p}");
        assert!(r.contains("GPU memory stays in use"), "the tooltip");
        let (_, json) = get_text(&state, "/v1/training").await;
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        let st = v["tasks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["id"] == paused)
            .map(|t| t["state"].clone());
        assert_eq!(st, Some(serde_json::json!("paused")));
        // Not the job on the queue's thread: nothing to pause or resume.
        let (s, _, b) = call(
            &state,
            post_form("/training/pause", &format!("id={run}"), true),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert!(String::from_utf8(b).unwrap().contains("No running job"));
        let (_, _, b) = call(
            &state,
            post_form("/training/resume", &format!("id={paused}"), true),
        )
        .await;
        assert!(String::from_utf8(b).unwrap().contains("No paused job"));
        // Finish now: only a training run on the queue's thread.
        for id in [run, "no-such-run"] {
            let (s, _, b) = call(
                &state,
                post_form("/training/finish", &format!("id={id}"), true),
            )
            .await;
            assert_eq!(s, StatusCode::OK);
            let b = String::from_utf8(b).unwrap();
            assert!(b.contains("Not finished:"), "{id}");
        }
        for id in [run, paused] {
            svc.put_task_record(rec(id, RunState::Cancelled));
            svc.remove(id).unwrap();
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn scenes_times_models_queue_a_comparison_group() {
        let f = form(&[
            ("model", "yolo26n"),
            ("model", "rfdetr-nano"),
            ("scene", "day"),
            ("scene", "night+ir"),
            ("scene", "all"),
            ("scene", "ir+night"),
            ("scene", ""),
        ]);
        let (reqs, name) = requests_from_form(&f, &mac(), &Default::default()).unwrap();
        assert_eq!(name, None);
        let got: Vec<(String, &str)> = reqs
            .iter()
            .map(|r| (r.recipe.label(), r.scene.as_str()))
            .collect();
        assert_eq!(
            got,
            [
                ("yolo26n".to_string(), "day"),
                ("yolo26n".to_string(), "night+ir"),
                ("yolo26n".to_string(), "all"),
                ("rfdetr-nano".to_string(), "day"),
                ("rfdetr-nano".to_string(), "night+ir"),
                ("rfdetr-nano".to_string(), "all"),
            ]
        );
        let group = reqs[0].compare_group.clone().unwrap();
        assert!(group.starts_with("cmp-") && group.len() == "cmp-20261027-120000".len());
        assert!(
            reqs.iter()
                .all(|r| r.compare_group.as_ref() == Some(&group))
        );
        // Every run gets its own default name within the same stamp.
        let mut names: Vec<String> = reqs
            .iter()
            .map(|r| r.recipe.default_name(r.scene.label(), "20261027-1200"))
            .collect();
        assert_eq!(
            names[..3],
            [
                "day-yolo26n-20261027-1200",
                "night-yolo26n-20261027-1200",
                "custom-yolo26n-20261027-1200"
            ]
        );
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 6);
        // One scene: no group, even with several models; none ticked = all.
        let (reqs, _) = requests_from_form(
            &form(&[
                ("model", "yolo26n"),
                ("model", "yolo26s"),
                ("scene", "night"),
            ]),
            &mac(),
            &Default::default(),
        )
        .unwrap();
        assert!(reqs.iter().all(|r| r.compare_group.is_none()));
        assert_eq!(reqs[0].scene, SceneFilter::new(false, true, false));
        let (reqs, _) =
            requests_from_form(&form(&[("model", "yolo26n")]), &mac(), &Default::default())
                .unwrap();
        assert_eq!(reqs[0].scene, SceneFilter::ALL_SCENES);
        assert_eq!(reqs[0].compare_group, None);
        // A typed name: one run only, scenes included.
        let e = requests_from_form(
            &form(&[
                ("model", "yolo26n"),
                ("scene", "day"),
                ("scene", "all"),
                ("name", "x"),
            ]),
            &mac(),
            &Default::default(),
        )
        .unwrap_err();
        assert!(e.to_string().contains("one run only"), "{e}");
        let (reqs, name) = requests_from_form(
            &form(&[("model", "yolo26n"), ("scene", "day"), ("name", "x")]),
            &mac(),
            &Default::default(),
        )
        .unwrap();
        assert_eq!((reqs.len(), name.as_deref()), (1, Some("x")));
        let e = requests_from_form(
            &form(&[("model", "yolo26n"), ("scene", "dusk")]),
            &mac(),
            &Default::default(),
        )
        .unwrap_err();
        assert!(e.to_string().starts_with("scene: 'dusk'"), "{e}");
    }

    #[test]
    fn classes_repeat_or_comma_and_excluded_parses() {
        let (reqs, _) = requests_from_form(
            &form(&[
                ("model", "yolo26n"),
                ("classes", "person"),
                ("classes", "car, dog"),
                ("classes", " person "),
                ("classes", ""),
                ("excluded", "drop-frame"),
            ]),
            &mac(),
            &Default::default(),
        )
        .unwrap();
        assert_eq!(
            reqs[0].classes.as_deref(),
            Some(&["person".to_string(), "car".into(), "dog".into()][..])
        );
        assert_eq!(reqs[0].excluded, Excluded::DropFrame);
        let (reqs, _) =
            requests_from_form(&form(&[("model", "yolo26n")]), &mac(), &Default::default())
                .unwrap();
        assert_eq!(reqs[0].classes, None);
        assert_eq!(reqs[0].excluded, Excluded::Background);
        let e = requests_from_form(
            &form(&[("model", "yolo26n"), ("excluded", "maybe")]),
            &mac(),
            &Default::default(),
        )
        .unwrap_err();
        assert!(
            e.to_string()
                .starts_with("objects of unticked classes: 'maybe'"),
            "{e}"
        );
    }

    #[test]
    fn default_names_get_a_suffix_when_taken() {
        let taken = ["day-yolo26n-1", "day-yolo26n-1-2"];
        let t = |n: &str| taken.contains(&n);
        assert_eq!(unique_name("night-yolo26n-1", t), "night-yolo26n-1");
        assert_eq!(unique_name("day-yolo26n-1", t), "day-yolo26n-1-3");
        assert!(recipe::valid_model_name(&unique_name("day-yolo26n-1", t)).is_ok());
    }

    /// A tiny export: `person` boxes in day and night frames, `car` in one IR frame.
    fn tiny_export(dir: &std::path::Path) {
        std::fs::create_dir_all(dir.join("images")).unwrap();
        std::fs::create_dir_all(dir.join("labels")).unwrap();
        std::fs::write(dir.join("classes.txt"), "person\ncar\nbus\n").unwrap();
        for (stem, labels) in [
            (
                "day.cam1.2026-10-01.a",
                "0 0.5 0.5 0.2 0.4\n0 0.2 0.2 0.1 0.2\n",
            ),
            ("night.cam1.2026-10-02.b", "0 0.5 0.5 0.2 0.4\n"),
            ("ir.cam2.2026-10-02.c", "1 0.5 0.5 0.3 0.2\n"),
            ("plain", ""),
        ] {
            std::fs::write(dir.join("images").join(format!("{stem}.jpg")), b"x").unwrap();
            std::fs::write(dir.join("labels").join(format!("{stem}.txt")), labels).unwrap();
        }
    }

    #[tokio::test]
    async fn dataset_classes_lists_counts_per_scene_and_refuses_other_folders() {
        use crate::server::test_util::{REMOTE, call, from_peer, get_text, test_state_with};
        let root = std::env::temp_dir().join(format!("bop-train-cls-{}", uuid::Uuid::new_v4()));
        let export = root.join("exports/review-1");
        tiny_export(&export);
        let (r2, e2) = (root.clone(), export.clone());
        let state = test_state_with(move |c| {
            c.download_dir = Some(r2);
            c.training.datasets = vec![e2];
        });
        let enc = super::super::encode_uri_component(&export.display().to_string());
        let (s, text) = get_text(
            &state,
            &format!("/v1/training/dataset-classes?dataset={enc}&dataset=%2Fno%2Fsuch%2Fexport"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{text}");
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["success"], true);
        let classes = v["classes"].as_array().unwrap();
        assert_eq!(classes[0]["name"], "person");
        assert_eq!(classes[0]["boxes"], 3);
        assert_eq!(classes[0]["frames"], 2);
        assert_eq!(
            classes[0]["by_scene"],
            serde_json::json!({"day": 2, "night": 1, "ir": 0, "unknown": 0})
        );
        assert_eq!(classes[1]["name"], "car");
        assert_eq!(classes[1]["by_scene"]["ir"], 1);
        // A listed class without a box is still offered; the order is the class list's.
        assert_eq!(classes[2]["name"], "bus");
        assert_eq!(classes[2]["boxes"], 0);
        assert_eq!(v["order"], serde_json::json!(["person", "car", "bus"]));
        assert_eq!(
            v["frames"],
            serde_json::json!({"day": 1, "night": 1, "ir": 1, "unknown": 1})
        );
        // The folder outside the offered and browsable ones is refused and named.
        let errors = v["errors"].as_array().unwrap();
        assert_eq!(errors.len(), 1, "{v}");
        assert_eq!(errors[0]["dataset"], "/no/such/export");
        assert!(errors[0]["message"].as_str().is_some_and(|m| !m.is_empty()));
        // Nothing ticked: nothing listed.
        let (_, text) = get_text(&state, "/v1/training/dataset-classes").await;
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(v["classes"].as_array().unwrap().is_empty());
        // Admin only.
        let req = axum::http::Request::get("/v1/training/dataset-classes")
            .body(axum::body::Body::empty())
            .unwrap();
        let (s, _, _) = call(&state, from_peer(req, REMOTE)).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let _ = std::fs::remove_dir_all(&root);
    }

    fn scene_ap(v: f64) -> crate::training::job::SceneAp {
        crate::training::job::SceneAp {
            ap50: Some(v + 0.2),
            ap50_95: Some(v),
            images: 10,
        }
    }

    /// A run record of comparison `group` (benchmarked with day / night / IR AP when `ap`).
    fn record(
        id: &str,
        scene: SceneFilter,
        group: Option<&str>,
        st: RunState,
        ap: Option<(f64, f64, f64)>,
    ) -> RunRecord {
        use crate::resources::catalog::TrainFamily;
        let request = TrainRequest {
            recipe: TrainRecipe::new(TrainFamily::Yolo26, "n").unwrap(),
            datasets: vec!["/exports/a".into()],
            include_public: false,
            scene,
            classes: Some(vec!["person".into()]),
            rename: Default::default(),
            benchmark_sets: Vec::new(),
            allow_large: false,
            install_dir: None,
            gpu: None,
            excluded: Excluded::Background,
            balance: Default::default(),
            compare_group: group.map(str::to_string),
        };
        let benchmark = ap.map(|(d, n, i)| BenchSummary {
            datasets: vec![format!("heldout-{id}")],
            device: Some("ort:coreml".into()),
            ms_p50: Some(10.0 + d),
            ap50: Some(d + 0.2),
            ap50_95: Some(d),
            images: 30,
            by_scene: [
                ("day", d),
                ("night", n),
                ("ir", i),
                ("all", (d + n + i) / 3.0),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), scene_ap(v)))
            .collect(),
            in_scope: Some(scene_ap(d.max(n))),
            heldout_fingerprint: Some("f1".into()),
        });
        RunRecord {
            id: id.into(),
            name: format!("{}-yolo26n-{id}", scene.label()),
            created: String::new(),
            request,
            device: "mps".into(),
            state: st,
            stage: None,
            step: 0,
            steps: 8,
            percent: 0,
            detail: String::new(),
            progress: Some(crate::training::progress::TrainProgress {
                epoch: 27,
                epochs: 100,
                patience: 15,
                stopped: Some(crate::training::progress::Stop {
                    reason: "converged".into(),
                    epoch: 27,
                    best_epoch: 12,
                }),
                ..Default::default()
            }),
            log: Vec::new(),
            dataset: None,
            output: None,
            registered: None,
            benchmark,
            benchmark_error: None,
            error: None,
            finished: None,
            run_dir: PathBuf::from("/runs").join(id),
        }
    }

    fn group_records() -> Vec<RunRecord> {
        let mut running = record("r9", SceneFilter::DAY, None, RunState::Running, None);
        running.stage = Some(crate::training::job::TrainStage::Train);
        running.progress = Some(crate::training::progress::TrainProgress {
            epoch: 4,
            epochs: 100,
            patience: 15,
            since_best: 1,
            ..Default::default()
        });
        let mut queued = record("r8", SceneFilter::DAY, None, RunState::Queued, None);
        queued.progress = None;
        vec![
            record(
                "old1",
                SceneFilter::DAY,
                Some("cmp-old"),
                RunState::Done,
                Some((0.5, 0.1, 0.1)),
            ),
            record(
                "old2",
                SceneFilter::ALL_SCENES,
                Some("cmp-old"),
                RunState::Done,
                Some((0.4, 0.3, 0.2)),
            ),
            record(
                "a",
                SceneFilter::DAY,
                Some("cmp-new"),
                RunState::Done,
                Some((0.6, 0.2, 0.1)),
            ),
            record(
                "b",
                SceneFilter::NIGHT_IR,
                Some("cmp-new"),
                RunState::Done,
                Some((0.3, 0.5, 0.45)),
            ),
            record(
                "c",
                SceneFilter::ALL_SCENES,
                Some("cmp-new"),
                RunState::Running,
                None,
            ),
            record(
                "solo",
                SceneFilter::DAY,
                Some("cmp-solo"),
                RunState::Done,
                Some((0.1, 0.1, 0.1)),
            ),
            queued,
            running,
        ]
    }

    #[test]
    fn comparison_groups_rows_and_json() {
        let recs = group_records();
        let groups = compare_groups(&recs);
        let names: Vec<&str> = groups.iter().map(|(g, _)| g.as_str()).collect();
        assert_eq!(
            names,
            ["cmp-new", "cmp-old"],
            "newest first, single runs left out"
        );
        let (runs, c) = group_comparison(&recs, "cmp-new").unwrap();
        assert_eq!(runs.len(), 3);
        assert!(c.comparable);
        assert_eq!(c.rows.len(), 4, "3 runs and the routed row");
        assert!(c.rows[3].routed);
        assert_eq!(c.best_per_column["day"], 0);
        assert_eq!(c.best_per_column["night"], 1);
        assert_eq!(c.rows[0].best_epoch, Some(12));
        assert_eq!(c.rows[0].stopped_epoch, Some(27));
        assert_eq!(c.rows[2].day.ap50_95, None, "the running run has no AP yet");
        assert!(group_comparison(&recs, "cmp-none").is_none());
        let cards = compare_cards(&recs);
        assert_eq!(cards.len(), 2);
        assert!(cards[0].open && !cards[1].open);
        let r = &cards[0].rows;
        assert!(r[0].cells[0].best && !r[1].cells[0].best);
        assert_eq!(r[0].cells[0].ap, "0.600");
        assert_eq!(r[0].cells[0].ap50, "0.800");
        assert_eq!(r[0].epochs, "best 12 / stopped 27");
        assert_eq!(r[2].cells[0].ap, "\u{2014}");
        assert_eq!(r[2].state, "running");
        assert!(r[3].routed && !r[3].cells[1].best);
        // A different held-out set: not comparable, said so.
        let mut recs = recs;
        recs[3].benchmark.as_mut().unwrap().heldout_fingerprint = Some("f2".into());
        let cards = compare_cards(&recs);
        assert!(!cards[0].comparable);
        assert!(!cards[0].warnings.is_empty());
        // The JSON shape.
        let (_, c) = group_comparison(&recs, "cmp-old").unwrap();
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["comparable"], true);
        assert_eq!(v["rows"][0]["day"]["ap50_95"], 0.5);
        assert_eq!(v["best_per_column"]["night"], 1);
    }

    #[tokio::test]
    async fn the_page_shows_runs_comparisons_finish_and_the_new_fields() {
        use crate::server::test_util::{get_text, test_state_with};
        use askama::Template;
        let root = std::env::temp_dir().join(format!("bop-train-cmp-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let r2 = root.clone();
        let state = test_state_with(move |c| c.download_dir = Some(r2));
        let mut recs = group_records();
        recs[2].output = Some(TrainOutput {
            name: recs[2].name.clone(),
            onnx: root.join("a.onnx"),
            yaml: root.join("a.yaml"),
            provenance: root.join("a.provenance.json"),
            family: ModelFamilyKind::Yolo26,
            classes: vec!["person".into()],
            license: "AGPL-3.0".into(),
            heldout: None,
            device: "mps".into(),
            train_seconds: 1.0,
        });
        let r9 = recs.iter_mut().find(|r| r.id == "r9").unwrap();
        r9.dataset = Some(crate::training::service::DatasetCounts {
            train_images: 412,
            balance_summary: Some("209 → 412 images/epoch; raccoon 32→118 boxes".into()),
            ..Default::default()
        });
        let page = view_with(&state, None, None, recs).render().unwrap();
        // The run's dataset line carries its balance summary.
        let i = page.find("<tr data-run=\"r9\"").unwrap();
        let j = i + page[i..].find("</tr>").unwrap();
        assert!(
            page[i..j]
                .contains("412 train / 0 val / 0 held out \u{b7} balanced: 209 → 412 images/epoch"),
            "{}",
            &page[i..j]
        );
        // The Balance fieldset (collapsed, off by default) and the preview's holder.
        assert!(page.contains("<details class=\"tr-more tr-balance\" id=\"train-balance\">"));
        for a in ["class", "scene", "camera", "camera-day"] {
            assert!(
                page.contains(&format!("name=\"balance\" value=\"{a}\">")),
                "axis {a} unticked"
            );
        }
        assert!(page.contains("evens out day / night / IR frame counts"));
        assert!(page.contains("name=\"balance_strength\" value=\"0.5\" checked>"));
        assert!(page.contains("name=\"balance_strength\" value=\"1\">"));
        assert!(
            page.contains(
                "name=\"balance_max_repeat\" min=\"1\" max=\"10\" step=\"1\" value=\"4\">"
            )
        );
        assert!(page.contains("name=\"balance_cap_on\" id=\"train-bal-cap-on\">"));
        assert!(page.contains("name=\"balance_cap\" id=\"train-bal-cap\""));
        assert!(page.contains("id=\"train-balance-preview\""));
        // Finish now: shown for the running run in its training stage only.
        let row = |id: &str| {
            let i = page
                .find(&format!("<tr data-run=\"{id}\""))
                .unwrap_or_else(|| panic!("row {id}"));
            let j = i + page[i..].find("</tr>").unwrap();
            page[i..j].to_string()
        };
        let shown = "action=\"/training/finish\" class=\"inline\" data-field=\"finish\">";
        let hidden = "action=\"/training/finish\" class=\"inline\" data-field=\"finish\" hidden>";
        let (running, queued) = (row("r9"), row("r8"));
        assert!(running.contains(shown), "{running}");
        assert!(running.contains("Stop after this epoch and keep the best checkpoint"));
        assert!(
            queued.contains(hidden) && !queued.contains(shown),
            "{queued}"
        );
        let c = row("c");
        assert!(c.contains(hidden), "running but not training: {c}");
        assert_eq!(page.matches(shown).count(), 1);
        // The Comparison card: the newest group open, the routed row, the best cells.
        let i = page.find("id=\"compare\"").expect("the Comparison card");
        let card = &page[i..i + page[i..].find("id=\"use\"").unwrap()];
        assert!(
            card.contains("<details class=\"tr-compare\" open>"),
            "{card}"
        );
        assert!(card.find("cmp-new").unwrap() < card.find("cmp-old").unwrap());
        assert!(card.contains("day-yolo26n-a"));
        assert!(card.contains("class=\"tr-routed\""));
        assert!(card.contains("day + night+ir routed"));
        assert!(card.contains("class=\"num best-val\""));
        assert!(!card.contains("cmp-solo"));
        // The Runs checkboxes, Patience, the recommendation and the class picker's fields.
        assert!(page.contains("name=\"scene\" value=\"all\" data-prefix=\"custom\" checked>"));
        assert!(page.contains("name=\"scene\" value=\"night+ir\" data-prefix=\"night\">"));
        assert!(page.contains("name=\"scene\" value=\"ir\" data-prefix=\"ir\">"));
        assert!(page.contains("name=\"patience\""));
        assert!(page.contains(
            "name=\"epochs\" min=\"1\" max=\"1000\" value=\"\" placeholder=\"recommended\""
        ));
        assert!(page.contains("100 ep \u{b7} patience 15"));
        assert!(page.contains("80 ep \u{b7} patience 10"));
        assert!(page.contains("id=\"train-picker\""));
        assert!(page.contains("name=\"excluded\" value=\"drop-frame\""));
        // The per-run summary of a finished run: AP50-95 per scene and in scope.
        assert!(
            page.contains("AP50-95 day 0.600 / night 0.200 / IR 0.100 \u{b7} in scope 0.600"),
            "the result row"
        );
        // The live page renders too, and the compare JSON answers 400 / 404.
        let (s, _) = get_text(&state, "/training").await;
        assert_eq!(s, StatusCode::OK);
        let (s, text) = get_text(&state, "/v1/training/compare").await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{text}");
        let (s, text) = get_text(&state, "/v1/training/compare?group=cmp-nope").await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{text}");
        assert!(text.contains("cmp-nope"));
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
            ..BenchSummary::default()
        };
        assert_eq!(
            bench_text(Some(&b), None),
            "AP50 0.500 \u{b7} 12.3 ms on openvino:cpu \u{b7} 20 images (heldout-x)"
        );
        assert_eq!(bench_text(None, Some("x")), "benchmark failed: x");
        // Per scene (only scenes with held-out frames) and in scope.
        let mut b = b;
        b.by_scene.insert("day".into(), scene_ap(0.61));
        b.by_scene.insert("ir".into(), scene_ap(0.3));
        b.by_scene.insert(
            "night".into(),
            crate::training::job::SceneAp {
                images: 0,
                ..scene_ap(0.1)
            },
        );
        b.in_scope = Some(scene_ap(0.55));
        assert!(
            bench_text(Some(&b), None)
                .ends_with(" \u{b7} AP50-95 day 0.610 / IR 0.300 \u{b7} in scope 0.550"),
            "{}",
            bench_text(Some(&b), None)
        );
    }

    #[test]
    fn balance_fields_parse_default_from_the_config_and_name_bad_values() {
        use crate::training::config::TrainingConfig;
        use std::collections::BTreeSet;
        let t = TrainingConfig::default();
        let axes = |v: &[BalanceAxis]| v.iter().copied().collect::<BTreeSet<_>>();
        // Repeated checkboxes and a comma list, strength, max repeat and the gated cap.
        let b = balance_of(
            &form(&[
                ("balance", "class"),
                ("balance", "scene, camera-day"),
                ("balance_strength", "1"),
                ("balance_max_repeat", "6"),
                ("balance_cap_on", "on"),
                ("balance_cap", "2.5"),
            ]),
            &t,
        )
        .unwrap();
        assert_eq!(
            b.axes,
            axes(&[
                BalanceAxis::Class,
                BalanceAxis::Scene,
                BalanceAxis::CameraDay
            ])
        );
        assert_eq!((b.strength, b.max_repeat), (1.0, 6));
        assert_eq!(b.cap_dominant, Some(2.5));
        // The cap applies only with its box; a blank cap is the default 3.
        let b = balance_of(&form(&[("balance", "camera"), ("balance_cap", "2")]), &t).unwrap();
        assert_eq!(b.cap_dominant, None);
        assert_eq!((b.strength, b.max_repeat), (0.5, 4), "defaults");
        let b = balance_of(
            &form(&[
                ("balance", "camera"),
                ("balance_cap_on", "on"),
                ("balance_cap", ""),
            ]),
            &t,
        )
        .unwrap();
        assert_eq!(b.cap_dominant, Some(3.0));
        // No balance field at all: training.balance, else off.
        let configured = TrainingConfig {
            balance: Some(Balance {
                axes: axes(&[BalanceAxis::Class]),
                strength: 1.0,
                ..Balance::default()
            }),
            ..TrainingConfig::default()
        };
        let none = form(&[("dataset", "/x")]);
        assert_eq!(
            &balance_of(&none, &configured).unwrap(),
            configured.balance.as_ref().unwrap()
        );
        assert!(balance_of(&none, &t).unwrap().is_off());
        // A form with the fieldset but no axis ticked: off, whatever the config or the other
        // fields say.
        let off = form(&[("balance_strength", "7"), ("balance_max_repeat", "4")]);
        assert_eq!(balance_of(&off, &configured).unwrap(), Balance::default());
        // Bad values name the field.
        let err = |pairs: &[(&str, &str)]| {
            let mut v = vec![("balance", "class")];
            v.extend_from_slice(pairs);
            format!("{:#}", balance_of(&form(&v), &t).unwrap_err())
        };
        let e = format!(
            "{:#}",
            balance_of(&form(&[("balance", "class,colour")]), &t).unwrap_err()
        );
        assert!(e.contains("balance") && e.contains("'colour'"), "{e}");
        let e = err(&[("balance_strength", "lots")]);
        assert!(
            e.contains("balance strength") && e.contains("'lots'"),
            "{e}"
        );
        assert!(err(&[("balance_strength", "1.5")]).contains("balance.strength"));
        assert!(err(&[("balance_strength", "NaN")]).contains("balance.strength"));
        assert!(err(&[("balance_max_repeat", "0")]).contains("balance.max_repeat"));
        assert!(err(&[("balance_max_repeat", "11")]).contains("balance.max_repeat"));
        assert!(err(&[("balance_max_repeat", "-1")]).contains("balance max repeat"));
        assert!(err(&[("balance_cap_on", "on"), ("balance_cap", "0.5")]).contains("cap_dominant"));
        // Every run of a scenes x models batch gets the same balance.
        let (reqs, _) = requests_from_form(
            &form(&[
                ("model", "yolo26n"),
                ("model", "yolo26s"),
                ("scene", "day"),
                ("scene", "night+ir"),
                ("balance", "class,scene"),
                ("balance_strength", "0.5"),
                ("balance_max_repeat", "5"),
            ]),
            &mac(),
            &t,
        )
        .unwrap();
        assert_eq!(reqs.len(), 4);
        for r in &reqs {
            assert_eq!(
                r.balance.axes,
                axes(&[BalanceAxis::Class, BalanceAxis::Scene])
            );
            assert_eq!(r.balance.max_repeat, 5);
        }
        // A bad balance refuses the queue form.
        let e = requests_from_form(
            &form(&[
                ("model", "yolo26n"),
                ("balance", "class"),
                ("balance_max_repeat", "0"),
            ]),
            &mac(),
            &t,
        )
        .unwrap_err();
        assert!(format!("{e:#}").contains("balance.max_repeat"), "{e:#}");
    }

    /// A review export of 3 cameras x (9 day + 3 IR frames), every fourth one empty, a
    /// raccoon in the first IR frame of each camera, persons elsewhere.
    fn skewed_export(dir: &std::path::Path) {
        std::fs::create_dir_all(dir.join("images")).unwrap();
        std::fs::create_dir_all(dir.join("labels")).unwrap();
        std::fs::write(dir.join("classes.txt"), "person\nraccoon\n").unwrap();
        for cam in ["a", "b", "c"] {
            for (scene, date, n) in [("day", "2026-10-01", 9), ("ir", "2026-10-02", 3)] {
                for i in 0..n {
                    let stem = format!("{scene}.{cam}.{date}.f{i}");
                    std::fs::write(dir.join("images").join(format!("{stem}.jpg")), b"x").unwrap();
                    let label = match (scene, i % 4) {
                        (_, 3) => "",
                        ("ir", 0) => "1 0.5 0.5 0.2 0.2\n",
                        _ => "0 0.5 0.5 0.2 0.4\n",
                    };
                    std::fs::write(dir.join("labels").join(format!("{stem}.txt")), label).unwrap();
                }
            }
        }
    }

    #[tokio::test]
    async fn the_balance_preview_shows_before_and_after_per_run_and_checks_folders() {
        use crate::server::test_util::{REMOTE, call, from_peer, post_form, test_state_with};
        let root = std::env::temp_dir().join(format!("bop-train-bal-{}", uuid::Uuid::new_v4()));
        let export = root.join("exports/review-1");
        skewed_export(&export);
        let (r2, e2) = (root.clone(), export.clone());
        let state = test_state_with(move |c| {
            c.download_dir = Some(r2);
            c.training.datasets = vec![e2];
        });
        let enc = super::super::encode_uri_component(&export.display().to_string());
        let post = |body: String| post_form("/v1/training/balance-preview", &body, false);
        let json = |b: &[u8]| -> serde_json::Value { serde_json::from_slice(b).unwrap() };
        // Class balance over two runs (all scenes, day): the raccoon frames repeat.
        let (s, _, b) = call(
            &state,
            post(format!(
                "dataset={enc}&scene=all&scene=day&balance=class&balance_strength=1&\
                 balance_max_repeat=4"
            )),
        )
        .await;
        let v = json(&b);
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["success"], true);
        assert_eq!(v["balance"]["axes"], serde_json::json!(["class"]));
        let runs = v["runs"].as_array().unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0]["scene"], "all");
        assert_eq!(runs[0]["title"], "All scenes (mixed)");
        assert_eq!(runs[1]["scene"], "day");
        let r = &runs[0];
        assert!(r["error"].is_null(), "{r}");
        let rep = &r["report"];
        let before = rep["before"]["boxes_per_class"]["raccoon"]
            .as_u64()
            .unwrap();
        let after = rep["after"]["boxes_per_class"]["raccoon"].as_u64().unwrap();
        assert!(before > 0 && after > before, "{rep}");
        assert_eq!(
            rep["before"]["boxes_per_class"]["person"],
            rep["after"]["boxes_per_class"]["person"]
        );
        assert!(rep["after"]["images"].as_u64() > rep["before"]["images"].as_u64());
        assert_eq!(r["train_images"], rep["after"]["images"]);
        assert!(r["train_frames"].as_u64().unwrap() > 0);
        assert!(r["heldout_frames"].as_u64().unwrap() > 0, "{r}");
        assert!(r["warnings"].is_array());
        // Balance off: no report, the counts still.
        let (s, _, b) = call(&state, post(format!("dataset={enc}&balance_strength=0.5"))).await;
        let v = json(&b);
        assert_eq!(s, StatusCode::OK, "{v}");
        let r = &v["runs"][0];
        assert!(r["report"].is_null(), "{r}");
        assert_eq!(r["train_images"], r["train_frames"]);
        // A folder outside the offered and browsable ones, a bad field: 400 naming it.
        let (s, _, b) = call(
            &state,
            post("dataset_path=%2Fno%2Fsuch%2Fexport&balance=class".into()),
        )
        .await;
        let v = json(&b);
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        assert_eq!(v["success"], false);
        assert!(
            v["message"].as_str().unwrap().contains("another folder"),
            "{v}"
        );
        let (s, _, b) = call(&state, post(format!("dataset={enc}&balance=colour"))).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert!(json(&b)["message"].as_str().unwrap().contains("'colour'"));
        // A class the datasets do not have: that run's error.
        let (_, _, b) = call(
            &state,
            post(format!("dataset={enc}&classes=zebra&balance=class")),
        )
        .await;
        let v = json(&b);
        assert!(
            v["runs"][0]["error"]
                .as_str()
                .is_some_and(|e| !e.is_empty()),
            "{v}"
        );
        // Cross-site browser requests and remote peers are refused.
        let mut req = post(format!("dataset={enc}&balance=class"));
        req.headers_mut()
            .insert("sec-fetch-site", "cross-site".parse().unwrap());
        let (s, _, _) = call(&state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let (s, _, _) = call(&state, from_peer(post(format!("dataset={enc}")), REMOTE)).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_balance_fieldset_starts_from_training_balance() {
        use askama::Template;
        let state = crate::server::test_util::test_state_with(|c| {
            c.training.balance = Some(Balance {
                axes: Balance::parse_axes("scene,camera-day").unwrap(),
                strength: 0.7,
                max_repeat: 6,
                cap_dominant: Some(2.5),
            });
        });
        let page = view_with(&state, None, None, Vec::new()).render().unwrap();
        assert!(page.contains("<details class=\"tr-more tr-balance\" id=\"train-balance\" open>"));
        assert!(page.contains("name=\"balance\" value=\"scene\" checked>"));
        assert!(page.contains("name=\"balance\" value=\"camera-day\" checked>"));
        assert!(page.contains("name=\"balance\" value=\"class\">"));
        // A configured strength that is neither preset: its own radio, checked.
        assert!(page.contains("name=\"balance_strength\" value=\"0.5\">"));
        assert!(page.contains("name=\"balance_strength\" value=\"0.7\" checked> Configured (0.7)"));
        assert!(page.contains("value=\"6\">"));
        assert!(page.contains("name=\"balance_cap_on\" id=\"train-bal-cap-on\" checked>"));
        assert!(page.contains("value=\"2.5\""));
    }
}
