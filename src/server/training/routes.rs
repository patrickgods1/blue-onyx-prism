//! The handlers of the Import, Auto-label, Train and Models tabs (`GET /train/...`), the run
//! queue's actions (`POST /training/*`: queue, cancel, finish, pause, resume, retry export,
//! remove, use as, datasets, training folder, remove environment) and the page's JSON
//! (`GET /v1/training`, the class list, the balance preview, the comparison, the metric
//! columns). Forms are read with [`crate::server::form`] and [`super::fields`]; pages are
//! [`view`].

use super::fields::{
    DatasetEdit, allow_in, checked_datasets, checked_folder, data_fields, dataset_edit,
    offered_datasets, rename_of, requests_from_form, unique_name,
};
use super::view::{SCENE_CHOICES, group_comparison, view};
use super::{poll, records, runs, service, shown, started_queue, sync_store, tasks};
use crate::resources::catalog::{self, Platform};
use crate::resources::export::train::{self as env, TrainPaths};
use crate::server::answer::json_fail;
use crate::server::form::{all, checked, field, num};
use crate::server::train_area::{AUTOLABEL, IMPORT, MODELS, TRAIN, jobs_tab};
use crate::server::{AppState, Shared, blocking, render, save_config};
use crate::training::job::{TrainOutput, TrainRequest};
use crate::training::recipe::{self, Augment, TrainRecipe};
use crate::training::service::{RunRecord, TrainingService};
use crate::training::{preflight, retry};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

/// `GET /train/import`.
pub(in crate::server) async fn import_tab(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || render(&view(&state, IMPORT, None, None))).await
}

/// `GET /train/autolabel`.
pub(in crate::server) async fn autolabel_tab(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || render(&view(&state, AUTOLABEL, None, None))).await
}

/// `GET /train/train`.
pub(in crate::server) async fn train_tab(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || render(&view(&state, TRAIN, None, None))).await
}

/// `GET /train/models`.
pub(in crate::server) async fn models_tab(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || render(&view(&state, MODELS, None, None))).await
}

/// Query of `GET /v1/training`: `poll=1` asks for the light answer of the page's tick, `have`
/// names the history it holds ([`poll`]).
#[derive(Debug, Default, serde::Deserialize)]
pub(in crate::server) struct TrainingQuery {
    #[serde(default)]
    poll: Option<String>,
    #[serde(default)]
    have: Option<String>,
}

/// `GET /v1/training`: capability and runs as JSON. With `poll=1` the page's poller gets only
/// the epochs it does not hold, and none of what its tick never reads ([`poll`]).
pub(in crate::server) async fn json(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(q): axum::extract::Query<TrainingQuery>,
) -> Response {
    blocking(move || {
        let columns = state.config_read().training.effective_metric_columns();
        let busy = started_queue().is_some_and(|s| s.busy());
        let tasks = tasks(&state);
        if q.poll.as_deref().is_some_and(|p| !p.is_empty() && p != "0") {
            let have = q.have.as_deref().map(poll::parse_have);
            let v = poll::answer(
                shown(&state),
                &tasks,
                busy,
                serde_json::json!(columns),
                serde_json::json!(crate::training::progress::METRICS),
                have.as_ref(),
            );
            return Json(v).into_response();
        }
        let all = runs(&state);
        let cap = crate::training::capability_with_runs(&all);
        // What a run of each recipe trained here before would get now (none without a GPU).
        let forced = state.config_read().training.workers;
        let workers_learned = cap.accelerator.as_ref().map_or_else(Vec::new, |a| {
            crate::training::workers::learned_now(a, forced, &all)
        });
        Json(serde_json::json!({
            "capability": cap,
            "workers_learned": workers_learned,
            "busy": busy,
            "runs": all,
            "tasks": tasks,
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
    rename: &BTreeMap<String, String>,
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
    let classes = dataset::inventory(&sources, rename);
    // The merged names, as `plan` lists them: each class renamed, a dropped one skipped.
    let mut order: Vec<String> = Vec::new();
    for c in sources
        .iter()
        .flat_map(|s| s.classes.iter())
        .map(|c| rename.get(c).unwrap_or(c))
        .chain(classes.iter().map(|c| &c.name))
    {
        if !c.is_empty() && !order.contains(c) {
            order.push(c.clone());
        }
    }
    // The names as the exports have them, with their boxes: the page's "from" choices.
    let mut source_classes: Vec<(String, u64)> = Vec::new();
    for c in sources.iter().flat_map(|s| s.classes.iter()) {
        if !source_classes.iter().any(|(n, _)| n == c) {
            source_classes.push((c.clone(), 0));
        }
    }
    for b in sources
        .iter()
        .flat_map(|s| s.frames.iter())
        .flat_map(|f| f.boxes.iter())
    {
        match source_classes.iter_mut().find(|(n, _)| *n == b.class) {
            Some(e) => e.1 += 1,
            None => source_classes.push((b.class.clone(), 1)),
        }
    }
    let source_classes: Vec<serde_json::Value> = source_classes
        .into_iter()
        .map(|(name, boxes)| serde_json::json!({ "name": name, "boxes": boxes }))
        .collect();
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
        "source_classes": source_classes,
        "frames": frames,
        "errors": errors,
    })
}

/// `GET /v1/training/dataset-classes?dataset=<path>&dataset=...`: the class picker's classes
/// with their box counts per scene ([`dataset_inventory`]). A folder that cannot be used is
/// listed in `errors` (200: the page asks while folders are ticked or typed). A browser request
/// marked `Sec-Fetch-Site: cross-site` is 403.
pub(in crate::server) async fn dataset_classes(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    axum::extract::Query(q): axum::extract::Query<Vec<(String, String)>>,
) -> Response {
    if crate::server::guard::fetch_site_cross(&headers) {
        return json_fail(
            StatusCode::FORBIDDEN,
            "Cross-site request refused: the class list answers this server's own pages only.",
        );
    }
    blocking(move || {
        let config = state.config_read().clone();
        let datasets = all(&q, "dataset");
        let rename = match rename_of(&q) {
            Ok(r) => r,
            Err(e) => return json_fail(StatusCode::BAD_REQUEST, format!("{e:#}")),
        };
        Json(dataset_inventory(
            &datasets,
            &offered_datasets(&config),
            &|p| config.data_path(p),
            &allow_in(&config),
            &rename,
        ))
        .into_response()
    })
    .await
}

/// How long the form's preview waits for a first count of the folder under review.
const UNREVIEWED_WAIT: std::time::Duration = std::time::Duration::from_millis(200);

/// Frames of the folder under review not reviewed yet, for the pre-flight report: the Train strip's
/// count ([`crate::server::train_area::tally_of`], kept, walked on its own thread, so the form never waits
/// for a walk). Its `to_review` is [`crate::training::unreviewed_frames`] but for a frame whose
/// sidecar does not read: that walk reads no sidecar, so such a frame counts here. None = not known:
/// no such folder, or nothing counted yet.
pub(super) fn unreviewed_now(config: &crate::config::Config) -> Option<u64> {
    let dir = config.collect.resolved_review_dir(&config.data_root());
    if !dir.is_dir() {
        return None;
    }
    crate::server::train_area::tally_of(&dir, UNREVIEWED_WAIT).map(|t| t.to_review)
}

/// The models the form ticks, a small one when none is yet: the disk estimate needs one.
fn preview_recipes(f: &[(String, String)]) -> anyhow::Result<Vec<TrainRecipe>> {
    let mut out = Vec::new();
    for m in all(f, "model") {
        if let Some((family, size)) = recipe::parse_label(m) {
            out.push(TrainRecipe::new(family, size)?);
        }
    }
    if out.is_empty() {
        out.push(TrainRecipe::new(catalog::TrainFamily::Yolo26, "n")?);
    }
    Ok(out)
}

/// The preview of a queue form ([`data_fields`]; models, `seeds` and `install_dir` only size the
/// disk estimate): one run per scene filter, each [`crate::training::preview_parts`] (the job's
/// sources and options, nothing written) as `{scene, title, report, warnings, classes,
/// train_frames, train_images, valid_frames, heldout_frames, dropped, preflight, error}` (`report`
/// null when balancing is off; `preflight`: [`preflight::Report`]; `error` set when that run cannot
/// be planned), `disk` `{lines, refusal}`, what every ticked run needs together, `memory`
/// `{refusal, override_label}` when the memory commit check would refuse a run started now (null
/// otherwise; [`preflight::memory`]), and `workers`: per ticked model the data loader workers it
/// would get now ([`preflight::workers_line`], learned from `past`, the queue's records). The
/// folders are checked like the queue's ([`checked_datasets`]); a bad field or folder is an error
/// naming it.
pub(crate) fn balance_preview_json(
    f: &[(String, String)],
    config: &crate::config::Config,
    past: &[RunRecord],
) -> anyhow::Result<serde_json::Value> {
    let data = data_fields(f, &config.training)?;
    let datasets = checked_datasets(f, &offered_datasets(config), &allow_in(config))?;
    if datasets.is_empty() && config.training.datasets.is_empty() {
        anyhow::bail!("no dataset: tick a review export folder or type one under Another folder");
    }
    // The plan depends on the data fields only, the disk estimate on the models too.
    let recipes = preview_recipes(f)?;
    let recipe = recipes
        .first()
        .cloned()
        .unwrap_or(TrainRecipe::new(catalog::TrainFamily::Yolo26, "n")?);
    let seeds = num::<usize>(f, "seeds", "seeds")?.unwrap_or(1).clamp(1, 3);
    let run_dir = crate::training::training_dir(config).join("preview");
    let models_dir = match field(f, "install_dir") {
        Some(d) => config.data_path(std::path::Path::new(d)),
        None => TrainPaths::for_config(config).export.models,
    };
    let unreviewed = unreviewed_now(config);
    // Free GPU memory against each ticked model, as the job checks it at its start.
    let accelerator = crate::training::capability_now().accelerator;
    let gpu: Vec<String> = accelerator
        .as_ref()
        .map(|a| {
            recipes
                .iter()
                .filter_map(|r| preflight::gpu_warning(r, a))
                .collect()
        })
        .unwrap_or_default();
    let workers = accelerator
        .as_ref()
        .map(|a| worker_lines(f, config, a, past))
        .unwrap_or_default();
    let mut needs: Vec<preflight::VolumeNeed> = Vec::new();
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
                rename: data.rename.clone(),
                benchmark_sets: Vec::new(),
                benchmark_devices: Vec::new(),
                allow_large: false,
                install_dir: None,
                gpu: None,
                excluded: data.excluded,
                balance: data.balance.clone(),
                compare_group: None,
                train_seed: None,
                ignore_memory_check: false,
            };
            match crate::training::preview_parts(config, &request) {
                Ok(parts) => {
                    let p = &parts.plan;
                    for r in &recipes {
                        for _ in 0..seeds {
                            if let Ok(n) = preflight::run_needs(
                                p,
                                &parts.sources,
                                r,
                                &run_dir,
                                &models_dir,
                            ) {
                                needs.extend(n);
                            }
                        }
                    }
                    serde_json::json!({
                        "scene": scene.as_str(),
                        "title": title,
                        "report": p.balance,
                        "warnings": p.warnings.iter().chain(&gpu).collect::<Vec<_>>(),
                        "classes": p.classes,
                        "train_frames": p.train.len(),
                        "train_images": p.train.iter().map(|(_, n)| *n as usize).sum::<usize>(),
                        "valid_frames": p.valid.len(),
                        "heldout_frames": p.heldout.len(),
                        "dropped": p.stats.balance_dropped,
                        "preflight": preflight::report(p, &parts.sources, &parts.options, unreviewed),
                        "error": null,
                    })
                }
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
    let lane = crate::training::capability_now()
        .accelerator
        .as_ref()
        .map(crate::training::Accelerator::lane);
    needs.extend(crate::training::tools_needs(config, &recipes, lane));
    let volumes: Vec<(
        preflight::VolumeNeed,
        Option<crate::training::import::disk::Space>,
    )> = preflight::combine(needs)
        .into_iter()
        .map(|n| {
            let s = crate::training::import::disk::space(&n.dir);
            (n, s)
        })
        .collect();
    // The memory commit check each run meets at its start (Windows), with the checkbox's words.
    let memory = preflight::memory(
        &recipes,
        config.training.workers.map_or(
            crate::training::job::Workers::Auto,
            crate::training::job::Workers::Fixed,
        ),
        crate::training::commit::now().as_ref(),
    );
    Ok(serde_json::json!({
        "success": true,
        "balance": data.balance,
        "runs": runs,
        "memory": memory,
        "disk": {
            "lines": volumes.iter().map(|(n, s)| preflight::disk_line(n, *s)).collect::<Vec<_>>(),
            "refusal": preflight::refusal(&volumes),
        },
        "workers": workers,
    }))
}

/// One [`preflight::workers_line`] per model the form ticks (none ticked: none), each recipe built
/// like the queue builds it ([`recipe::build_recipe`]: the typed batch and input size, else the
/// GPU's suggestion), prefixed with the model when there are several. A model whose recipe does
/// not build gets no line (the queue names the error).
fn worker_lines(
    f: &[(String, String)],
    config: &crate::config::Config,
    accel: &crate::training::Accelerator,
    past: &[RunRecord],
) -> Vec<String> {
    let models = all(f, "model");
    if models.is_empty() {
        return Vec::new();
    }
    let typed = recipe::RecipeInputs {
        epochs: None,
        patience: None,
        imgsz: num(f, "imgsz", "input size").ok().flatten(),
        batch: num(f, "batch", "batch").ok().flatten(),
        lr: None,
        lr_schedule: None,
        lr_min_factor: None,
        select_metric: None,
        augment: Augment::default(),
        multi_scale: recipe::MultiScale::Auto,
    };
    let cap = crate::training::capability_with_runs(past);
    let several = models.len() > 1;
    models
        .iter()
        .filter_map(|m| {
            let (family, size) = recipe::parse_label(m)?;
            let r = recipe::build_recipe(family, size, &typed, &config.training, &cap).ok()?;
            let line = preflight::workers_line(&r, accel, config.training.workers, past);
            Some(if several {
                format!("{}: {line}", r.label())
            } else {
                line
            })
        })
        .collect()
}

/// `POST /v1/training/balance-preview` with the queue form's fields (urlencoded): what
/// balancing would do to each run's training frames ([`balance_preview_json`]). 400 with
/// `{success: false, message}` for a bad field or folder; a browser request marked
/// `Sec-Fetch-Site: cross-site` is 403.
pub(in crate::server) async fn balance_preview(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    if crate::server::guard::fetch_site_cross(&headers) {
        return json_fail(
            StatusCode::FORBIDDEN,
            "Cross-site request refused: the balance preview answers this server's own pages \
             only.",
        );
    }
    blocking(move || {
        let f = crate::server::form::parse(&body);
        let config = state.config_read().clone();
        let past = records(&state);
        match balance_preview_json(&f, &config, &past) {
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
pub(in crate::server) async fn compare_json(
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
        let records = records(&state);
        match group_comparison(&records, &group) {
            Some((rs, c)) => Json(serde_json::json!({
                "success": true,
                "group": group,
                "runs": rs
                    .iter()
                    .map(|r| serde_json::json!({
                        "id": r.id, "name": r.name, "state": r.state.as_str()
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

/// `POST /v1/training/metric-columns` with `{columns: ["map", "f2", ...]}`: saves the per-epoch
/// table's columns (`training.metric_columns`, [`crate::training::config::
/// validate_metric_columns`]; the default list is stored as empty) and answers the effective
/// list. 415 without a JSON body, 400 on a bad list.
pub(in crate::server) async fn metric_columns(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    if !crate::server::answer::is_json(&headers) {
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
pub(in crate::server) async fn queue(
    State(shared): State<Shared>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let state = shared.current();
        let f = crate::server::form::parse(&body);
        let config = state.config_read().clone();
        let cap = crate::training::capability_with_runs(&runs(&state));
        let result = (|| -> anyhow::Result<String> {
            let (mut reqs, name) = requests_from_form(&f, &cap, &config.training)?;
            let bench_devices = reqs
                .first()
                .map(|r| r.benchmark_devices.clone())
                .unwrap_or_default();
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
                    && !checked(&f, "confirm_large")
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
            let stamp = crate::clock::zone()
                .wall(std::time::SystemTime::now())
                .format("%Y%m%d-%H%M")
                .to_string();
            let busy: Vec<String> = svc
                .records()
                .into_iter()
                .filter(|r| r.state.is_busy())
                .map(|r| r.name)
                .collect();
            let models_dir = TrainPaths::for_config(&config).export.models;
            let mut names: Vec<String> = Vec::new();
            let mut jobs = Vec::new();
            for r in reqs {
                // Repeats of a seed beyond the first: `-s<i>` after the typed / default name.
                let seed_suffix = match r.train_seed {
                    Some(i) if i > 0 => format!("-s{i}"),
                    _ => String::new(),
                };
                let typed = name.as_ref().map(|n| format!("{n}{seed_suffix}"));
                let name = typed.unwrap_or_else(|| {
                    let base = format!(
                        "{}{seed_suffix}",
                        r.recipe.default_name(r.scene.label(), &stamp)
                    );
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
            crate::training::check_jobs(&config, &jobs)?;
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
            // The ticked benchmark devices become the setting (the next form starts from them).
            if bench_devices != config.training.benchmark_devices
                && let Err(e) = save_config(&state, "training page", |c| {
                    c.training.benchmark_devices.clone_from(&bench_devices);
                    Ok(())
                })
            {
                tracing::warn!("benchmark devices not saved: {e:#}");
            }
            Ok(format!("Queued: {}.", queued.join(", ")))
        })();
        match result {
            Ok(m) => render(&view(&state, TRAIN, Some(m), None)),
            Err(e) => render(&view(
                &state,
                TRAIN,
                None,
                Some(format!("Not queued: {e:#}")),
            )),
        }
    })
    .await
}

/// `POST /training/cancel` (`id`).
pub(in crate::server) async fn cancel(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let f = crate::server::form::parse(&body);
        let tab = jobs_tab(field(&f, "tab"), TRAIN);
        let id = field(&f, "id").unwrap_or_default().to_string();
        let ok = started_queue().is_some_and(|s| s.cancel(&id));
        if ok {
            render(&view(
                &state,
                tab,
                Some(format!("Cancelling run {id}.")),
                None,
            ))
        } else {
            render(&view(
                &state,
                tab,
                None,
                Some(format!("No queued or running run {id}.")),
            ))
        }
    })
    .await
}

/// `POST /training/finish` (`id`): stop the running run after its current epoch and export its
/// best epoch ([`TrainingService::finish`]).
pub(in crate::server) async fn finish(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let f = crate::server::form::parse(&body);
        let tab = jobs_tab(field(&f, "tab"), TRAIN);
        let id = field(&f, "id").unwrap_or_default().to_string();
        let r = started_queue()
            .ok_or_else(|| format!("no run {id}"))
            .and_then(|s| s.finish(&id));
        match r {
            Ok(m) => render(&view(&state, tab, Some(m), None)),
            Err(e) => render(&view(
                &state,
                tab,
                None,
                Some(format!("Not finished: {e}.")),
            )),
        }
    })
    .await
}

/// `POST /training/pause` (`id`): suspend the running run or job's helper processes where they
/// are (models stay loaded, GPU memory stays held); the queue keeps its place.
pub(in crate::server) async fn pause(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    pause_or_resume(state, body, true).await
}

/// `POST /training/resume` (`id`): continue a paused run or job.
pub(in crate::server) async fn resume(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    pause_or_resume(state, body, false).await
}

async fn pause_or_resume(state: Arc<AppState>, body: axum::body::Bytes, pause: bool) -> Response {
    blocking(move || {
        let f = crate::server::form::parse(&body);
        let tab = jobs_tab(field(&f, "tab"), TRAIN);
        let id = field(&f, "id").unwrap_or_default().to_string();
        let ok = started_queue().is_some_and(|s| if pause { s.pause(&id) } else { s.resume(&id) });
        match (ok, pause) {
            (true, true) => render(&view(
                &state,
                tab,
                Some(format!(
                    "Paused {id}: its processes are suspended (models stay loaded and GPU memory \
                     stays in use); nothing else starts until you resume or cancel it."
                )),
                None,
            )),
            (true, false) => render(&view(&state, tab, Some(format!("Resumed {id}.")), None)),
            (false, true) => render(&view(
                &state,
                tab,
                None,
                Some(format!("No running job {id}.")),
            )),
            (false, false) => render(&view(
                &state,
                tab,
                None,
                Some(format!("No paused job {id}.")),
            )),
        }
    })
    .await
}

/// `POST /training/retry-export` (`id`): queue the export of a failed run's kept checkpoint
/// ([`retry::plan_retry`]).
pub(in crate::server) async fn retry_export(
    State(shared): State<Shared>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let state = shared.current();
        let f = crate::server::form::parse(&body);
        let tab = jobs_tab(field(&f, "tab"), TRAIN);
        let id = field(&f, "id").unwrap_or_default().to_string();
        let result = (|| -> anyhow::Result<String> {
            let svc = service(&shared);
            let rec = svc.get(&id).ok_or_else(|| anyhow::anyhow!("no run {id}"))?;
            let config = state.config_read().clone();
            let cap = crate::training::capability_with_runs(&runs(&state));
            let job = retry::plan_retry(&config, &rec, &cap)?;
            svc.enqueue_retry(job).map_err(anyhow::Error::msg)?;
            Ok(format!(
                "Exporting {} from its best checkpoint. It is added to the config and benchmarked \
                 when that works.",
                rec.name
            ))
        })();
        match result {
            Ok(m) => render(&view(&state, tab, Some(m), None)),
            Err(e) => render(&view(
                &state,
                tab,
                None,
                Some(format!("Not queued: {e:#}.")),
            )),
        }
    })
    .await
}

/// `POST /training/remove` (`id`, optional `delete_files`): forget a finished run; with
/// `delete_files` its folder goes too ([`retry::delete_run_files`]), else its files stay.
pub(in crate::server) async fn remove(
    State(shared): State<Shared>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let state = shared.current();
        let f = crate::server::form::parse(&body);
        let tab = jobs_tab(field(&f, "tab"), TRAIN);
        let id = field(&f, "id").unwrap_or_default().to_string();
        if checked(&f, "delete_files") {
            let config = state.config_read().clone();
            let size = service(&shared)
                .record(&id)
                .map_or(0, |r| retry::disk_use(&r.run_dir));
            let done =
                service(&shared).remove_with_files(&id, |r| retry::delete_files_of(&config, r));
            return match done {
                Ok(r) => render(&view(
                    &state,
                    tab,
                    Some(format!(
                        "Removed run {} and deleted its files ({}).",
                        r.name,
                        catalog::format_size(size)
                    )),
                    None,
                )),
                Err(e) => render(&view(&state, tab, None, Some(e))),
            };
        }
        match service(&shared).remove(&id) {
            Ok(()) => render(&view(
                &state,
                tab,
                Some(format!("Removed run {id} from the list.")),
                None,
            )),
            Err(e) => render(&view(&state, tab, None, Some(e))),
        }
    })
    .await
}

/// `POST /training/use` (`id`, `role` = `default` | `night` | `verify`, `target`): use a
/// trained model as the default model, or as the night model / verifier of `target`. Default:
/// `copy` copies the settings of the current default, `use_threshold` takes the suggested
/// threshold (the Tune view's cross-validated suggestion for the saved goal, recomputed here,
/// [`crate::server::training_switch::suggestion_for`]; refused when it must not be saved, and a
/// low-confidence one only with `confirm=1`, as the Tune step asks), `expect_current` the
/// default the page's preview compared with. Verifier: `replace_verifier` names the existing
/// verifier it may replace.
pub(in crate::server) async fn use_model(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let f = crate::server::form::parse(&body);
        let mut message = String::new();
        let result = (|| -> anyhow::Result<String> {
            let id = field(&f, "id").ok_or_else(|| anyhow::anyhow!("no run id"))?;
            let rec = records(&state)
                .into_iter()
                .find(|r| r.id == id)
                .ok_or_else(|| anyhow::anyhow!("no run {id}"))?;
            let out = rec
                .output
                .ok_or_else(|| anyhow::anyhow!("run {id} has no installed model"))?;
            let role = field(&f, "role").unwrap_or("default").to_string();
            let target = field(&f, "target").map(str::to_string);
            let threshold = if checked(&f, "use_threshold") {
                let config = state.config_read().clone();
                let m = crate::benchmark::find_model(&config, &out.name).ok_or_else(|| {
                    anyhow::anyhow!("{} is not in the config, so it has no benchmark", out.name)
                })?;
                let s = crate::server::training_switch::suggestion_for(&state, &config, m)
                    .map_err(anyhow::Error::msg)?;
                if s.confirm && !checked(&f, "confirm") {
                    anyhow::bail!(
                        "the suggested threshold {:.2} rests on little data ({}); confirm to \
                         save it anyway (send confirm=1), or keep the threshold",
                        s.threshold,
                        s.low_confidence.as_deref().unwrap_or("low confidence")
                    );
                }
                Some(crate::training::switch::Accepted::from(&s))
            } else {
                None
            };
            let opts = UseOptions {
                copy: checked(&f, "copy"),
                threshold,
                expect_current: field(&f, "expect_current").map(str::to_string),
                replace_verifier: field(&f, "replace_verifier").map(str::to_string),
                at: chrono::Utc::now().to_rfc3339(),
            };
            save_config(&state, "training page", |c| {
                message = apply_use(c, &out, &role, target.as_deref(), &opts)?;
                Ok(())
            })
        })();
        match result {
            Ok(m) => render(&view(&state, MODELS, Some(format!("{message} {m}")), None)),
            Err(e) => render(&view(
                &state,
                MODELS,
                None,
                Some(format!("Not saved: {e:#}")),
            )),
        }
    })
    .await
}

/// The choices of a "Use as" action ([`use_model`]).
#[derive(Debug, Clone, Default)]
pub(crate) struct UseOptions {
    /// Default: copy the current default's settings.
    pub copy: bool,
    /// Default: the suggested threshold, accepted.
    pub threshold: Option<crate::training::switch::Accepted>,
    /// Default: refuse when the default is no longer this one (the page's preview went stale).
    pub expect_current: Option<String>,
    /// Verifier: the existing verifier the owner agreed to replace.
    pub replace_verifier: Option<String>,
    /// RFC 3339 time of the action.
    pub at: String,
}

/// Apply a "Use as" action to `c` (the trained model is added first when missing); returns what
/// happened, for the page.
pub(crate) fn apply_use(
    c: &mut crate::config::Config,
    out: &TrainOutput,
    role: &str,
    target: Option<&str>,
    opts: &UseOptions,
) -> anyhow::Result<String> {
    use crate::training::switch;
    let idx = |c: &crate::config::Config, n: &str| {
        c.models
            .iter()
            .position(|m| m.effective_name().eq_ignore_ascii_case(n))
    };
    let from = switch::current_default(c);
    if role == "default"
        && let (Some(expected), Some(now)) = (&opts.expect_current, &from)
        && crate::registry::normalize_name(expected) != crate::registry::normalize_name(now)
    {
        anyhow::bail!(
            "the default model changed to {now} while this page was open, so what it showed is \
             out of date; open Use as default again and check it"
        );
    }
    let before = idx(c, &out.name).and_then(|i| c.models.get(i)).cloned();
    if idx(c, &out.name).is_none() {
        crate::training::cli::add_trained_model(c, out)?;
    }
    if let Some(i) = idx(c, &out.name)
        && let Some(m) = c.models.get_mut(i)
    {
        m.enabled = true;
    }
    match role {
        "default" => {
            // Added by this switch: a Revert leaves the entry, not loaded.
            let before = before.or_else(|| {
                let m = idx(c, &out.name).and_then(|i| c.models.get(i))?;
                Some(crate::config::ModelConfig {
                    enabled: false,
                    ..m.clone()
                })
            });
            let outcome = switch::apply_switch(
                c,
                &out.name,
                switch::SwitchOptions {
                    from,
                    before,
                    copy: opts.copy,
                    classes: out.classes.clone(),
                    threshold: opts.threshold.clone(),
                    at: opts.at.clone(),
                },
            )?;
            Ok(outcome.message(&out.name))
        }
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
            let entry = c
                .models
                .get(i)
                .ok_or_else(|| anyhow::anyhow!("no configured model named {t}"))?;
            if role == "verify" {
                switch::check_verifier(c, entry, &out.name, opts.replace_verifier.as_deref())?;
            }
            let m = c
                .models
                .get_mut(i)
                .ok_or_else(|| anyhow::anyhow!("no configured model named {t}"))?;
            if role == "night" {
                m.night_model = Some(out.name.clone());
                Ok(format!("{} is now the night model of {t}.", out.name))
            } else {
                m.verify = Some(crate::verify::VerifyConfig::with_model(out.name.clone()));
                Ok(format!("{} now verifies {t}.", out.name))
            }
        }
        other => anyhow::bail!("role: '{other}' is not default, night or verify"),
    }
}

/// `POST /training/datasets` (see [`dataset_edit`]): the export folders offered.
pub(in crate::server) async fn datasets(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let f = crate::server::form::parse(&body);
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
            Ok(m) => render(&view(&state, TRAIN, Some(m), None)),
            Err(e) => render(&view(
                &state,
                TRAIN,
                None,
                Some(format!("Not saved: {e:#}")),
            )),
        }
    })
    .await
}

/// `POST /training/dir` (`training_dir`; empty = the default `<data root>/training`): the
/// training folder (`training.dir`), saved with [`save_config`]. The folder must exist and pass
/// [`checked_folder`]. Refused (409) while a run or task is queued or running; otherwise the
/// save and the queue's move to the new folder ([`sync_store`]) run with every enqueue held off
/// ([`while_queue_idle`]), so no job is queued into the old folder in between: runs, imports
/// and exports in the old folder stay there.
pub(in crate::server) async fn set_dir(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> Response {
    blocking(move || {
        let refused = |message: String| {
            (
                StatusCode::CONFLICT,
                render(&view(&state, TRAIN, None, Some(message))),
            )
                .into_response()
        };
        if started_queue().is_some_and(|s| s.busy()) {
            return refused(
                "Not saved: a training run, import or auto-label job is queued or running; wait \
                 for it or cancel it, then change the training folder."
                    .into(),
            );
        }
        let f = crate::server::form::parse(&body);
        let config = state.config_read().clone();
        let root = config.data_root();
        let new = match checked_folder(
            "training folder",
            field(&f, "training_dir"),
            &root,
            &allow_in(&config),
        ) {
            // The default folder is kept as the default (it follows the data root).
            Ok(new) => {
                let default = crate::training::config::TrainingConfig::default().dir_in(&root);
                new.filter(|p| default.canonicalize().ok().as_ref() != Some(p))
            }
            Err(e) => {
                return render(&view(
                    &state,
                    TRAIN,
                    None,
                    Some(format!("Not saved: {e:#}")),
                ));
            }
        };
        let save = || -> anyhow::Result<String> {
            let old = crate::training::training_dir(&state.config_read());
            let saved = save_config(&state, "training page", |c| {
                c.training.dir.clone_from(&new);
                Ok(())
            })?;
            let now = crate::training::training_dir(&state.config_read());
            if let Some(svc) = started_queue() {
                sync_store(&svc, &now.join(crate::training::service::RUNS_FILE));
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
        };
        match while_queue_idle(started_queue().as_deref(), save) {
            Err(why) => refused(format!("Not saved: {why}.")),
            Ok(Ok(m)) => render(&view(&state, TRAIN, Some(m), None)),
            Ok(Err(e)) => render(&view(
                &state,
                TRAIN,
                None,
                Some(format!("Not saved: {e:#}")),
            )),
        }
    })
    .await
}

/// `POST /training/remove-env`: delete the training environment ([`remove_env_in`]).
pub(in crate::server) async fn remove_env(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        let paths = TrainPaths::for_config(&state.config_read());
        match remove_env_in(started_queue().as_deref(), &paths) {
            Ok(message) => render(&view(&state, TRAIN, Some(message), None)),
            Err((code, message)) => {
                (code, render(&view(&state, TRAIN, None, Some(message)))).into_response()
            }
        }
    })
    .await
}

/// Run `act` with every enqueue of `queue` held off ([`TrainingService::while_idle`]); refused
/// with its reason while the queue is busy or stopped. Without a started queue nothing of this
/// process can run, so `act` runs at once (the tool locks cover a job that starts meanwhile).
pub(super) fn while_queue_idle<T>(
    queue: Option<&TrainingService>,
    act: impl FnOnce() -> T,
) -> Result<T, String> {
    match queue {
        Some(q) => q.while_idle(&|_, _| None, act),
        None => Ok(act()),
    }
}

/// Delete the training environment under `paths`: refused (409) while `queue` has a job queued
/// or running, and while a training, import, auto-label, dataset check, environment setup or
/// export of any process holds a tool lock ([`env::remove_train_env`], [`ToolsBusy`]). The
/// removal runs with enqueues held off ([`while_queue_idle`]). Another failure is shown with
/// 200, as the page always did.
///
/// [`ToolsBusy`]: crate::resources::export::ToolsBusy
pub(super) fn remove_env_in(
    queue: Option<&TrainingService>,
    paths: &TrainPaths,
) -> Result<String, (StatusCode, String)> {
    if queue.is_some_and(TrainingService::busy) {
        return Err((
            StatusCode::CONFLICT,
            "A run is queued or running; cancel it first.".into(),
        ));
    }
    match while_queue_idle(queue, || env::remove_train_env(paths)) {
        Err(why) => Err((StatusCode::CONFLICT, format!("Not removed: {why}."))),
        Ok(Ok(())) => Ok("Removed the training environment.".into()),
        Ok(Err(e)) if e.is::<crate::resources::export::ToolsBusy>() => {
            Err((StatusCode::CONFLICT, format!("Not removed: {e:#}.")))
        }
        Ok(Err(e)) => Err((StatusCode::OK, format!("Not removed: {e:#}"))),
    }
}
