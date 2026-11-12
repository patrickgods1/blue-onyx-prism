//! Benchmark run endpoints: start (`POST /v1/benchmark`, refused while a job of the training
//! queue runs, and for several GPUs until confirmed), cancel, save the form as the
//! `benchmark` config defaults, drop a saved folder dataset (`POST /v1/benchmark/datasets/remove`),
//! and the per-image drill-down (`/v1/benchmark/images`) with the image file itself
//! (`/v1/benchmark/image`, only files listed in the saved results, through a cached index keyed
//! by the results file stamp). File I/O runs in `server::blocking`.

use super::{file_stamp, json_error, loaded_devices, truthy};
use crate::benchmark::images;
use crate::benchmark::service::{RunContext, RunRequest, SEVERAL_GPUS_WARNING, parse_devices};
use crate::benchmark::{BenchmarkResults, results_path};
use crate::config::DatasetRef;
use crate::server::{AppState, save_config};
use crate::sync::LockExt;
use crate::training::service::RunRecord;
use crate::training::task::{TaskKind, TaskRecord};
use axum::Json;
use axum::extract::rejection::{FormRejection, QueryRejection};
use axum::extract::{Form, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

/// The folder set `id` names in `results`: a folder dataset with a name instead of a `dir:` id.
/// The results record the images folder; `load_dir` wants the folder holding it.
fn named_folder(results: &BenchmarkResults, id: &str) -> Option<DatasetRef> {
    if id.starts_with(images::DIR_PREFIX) {
        return None;
    }
    let set = results
        .image_sets
        .iter()
        .find(|s| s.id == id && s.kind == images::SetKind::Dir)?;
    let dir = set.dir.as_ref()?;
    let root = match (dir.file_name(), dir.parent()) {
        (Some(n), Some(p)) if n == "images" => p.to_path_buf(),
        _ => dir.clone(),
    };
    Some(DatasetRef::Dir {
        dir: root,
        gt: None,
        name: Some(id.to_string()),
    })
}

/// Parse a start/settings form over the configured defaults and check the datasets: a folder
/// set (`dir:<path>`) the config does not already list must be a browsable folder or one
/// picked with Browse... ([`crate::server::browse::allowed_path`]), like `add_folder`.
fn parse_request(state: &AppState, form: &[(String, String)]) -> anyhow::Result<RunRequest> {
    let cfg = state.config_read().clone();
    let mut req = RunRequest::from_form(form, &cfg.benchmark)?;
    let mut configured: Vec<std::path::PathBuf> = cfg
        .benchmark
        .datasets
        .iter()
        .filter_map(|d| d.dir().map(|(dir, _, _)| cfg.data_path(&dir)))
        .collect();
    // A named folder set of the saved results (a training run's held-out frames, offered by
    // Compare's "Run on these images"): the form can only name it, the results know where it is.
    let saved = BenchmarkResults::load_or_warn(&results_path(&state.config_path));
    for d in &mut req.datasets {
        if let Some(named) = saved.as_ref().and_then(|r| named_folder(r, &d.label())) {
            if let Some((dir, _, _)) = named.dir() {
                configured.push(cfg.data_path(&dir));
            }
            *d = named;
        }
    }
    for d in &req.datasets {
        let Some((dir, _, _)) = d.dir() else {
            continue;
        };
        if configured.contains(&cfg.data_path(&dir)) {
            continue;
        }
        crate::server::browse::allowed_path(&cfg, &dir.to_string_lossy()).map_err(|e| {
            anyhow::anyhow!(
                "dataset {}: {}; choose the folder with Browse... (or one inside the browsable \
                 folders)",
                d.label(),
                e.into_parts().1
            )
        })?;
    }
    images::validate(&req.datasets, &cfg.data_root())?;
    Ok(req)
}

/// Form field that accepts a run of several GPUs ([`gpu_devices`]).
pub(super) const CONFIRM_GPUS: &str = "confirm_gpus";

/// The GPUs among `devices`: more than one needs [`CONFIRM_GPUS`]. A device of each runtime counts
/// on its own: the spec does not tell which of them share a card.
pub(super) fn gpu_devices(devices: &[String]) -> Vec<String> {
    parse_devices(devices)
        .ok()
        .flatten()
        .unwrap_or_default()
        .iter()
        .filter(|d| d.is_gpu_like())
        .map(ToString::to_string)
        .collect()
}

/// The job of the training queue that holds the queue's thread now, in words: a GPU and the
/// machine's memory are its own while it runs. Only this process's queue: a `train` command in
/// another process is not seen here.
fn running_job(runs: &[RunRecord], tasks: &[TaskRecord]) -> Option<String> {
    let run = runs
        .iter()
        .find(|r| r.state.is_active())
        .map(|r| format!("Training run '{}'", r.name));
    run.or_else(|| {
        tasks.iter().find(|t| t.state.is_active()).map(|t| {
            let kind = match t.kind {
                TaskKind::Import => "Import",
                TaskKind::Autolabel => "Auto-label run",
                TaskKind::Overlaps => "Overlap check",
                TaskKind::Refine => "Dataset check",
            };
            format!("{kind} '{}'", t.title)
        })
    })
}

/// `POST /v1/benchmark` (urlencoded; every field optional, defaults from the `benchmark`
/// config): `model`, `device`, `dataset` (repeated), `max_images`, `repeat`, `warmup`,
/// `reference_model`, `accuracy_weight`, and [`CONFIRM_GPUS`] for more than one GPU. No devices
/// anywhere = each model on the device it runs on. Starts a run in the background: 202, 400 (bad
/// input) or 409 (a run or a job of the training queue is in progress, or several GPUs without
/// the confirmation).
pub(in crate::server) async fn start(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    crate::server::blocking(move || start_blocking(&state, &form)).await
}

/// [`start`] off the runtime threads: checking the datasets stats directories and building the
/// run context clones the config (SV-06).
fn start_blocking(state: &AppState, form: &[(String, String)]) -> Response {
    let req = match parse_request(state, form) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    if state.benchmark.is_running() {
        return json_error(
            StatusCode::CONFLICT,
            "a benchmark is already running; cancel it or wait for it to finish".into(),
        );
    }
    let job = running_job(
        &crate::server::training::runs(state),
        &crate::server::training::tasks(state),
    );
    if let Some(job) = job {
        return json_error(
            StatusCode::CONFLICT,
            format!(
                "{job} is running, and a benchmark would compete with it for the same CPU and \
                 GPU. Wait for it to finish or cancel it on the Jobs tab of Train, then start \
                 the benchmark."
            ),
        );
    }
    let gpus = gpu_devices(&req.devices);
    if gpus.len() > 1 && !form.iter().any(|(k, v)| k == CONFIRM_GPUS && truthy(v)) {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "success": false,
                "message": format!(
                    "{} GPU devices were chosen ({}). {SEVERAL_GPUS_WARNING} To run them anyway, \
                     send {CONFIRM_GPUS}=1.",
                    gpus.len(),
                    gpus.join(", ")
                ),
                "confirm": CONFIRM_GPUS,
            })),
        )
            .into_response();
    }
    let ctx = RunContext {
        config: state.config_read().clone(),
        results_path: results_path(&state.config_path),
        runtimes: state.registry.runtimes().cloned(),
        loaded: loaded_devices(state),
        resources: state.resources.clone(),
    };
    info!(
        models = ?req.models,
        devices = ?req.devices,
        datasets = ?req.datasets.iter().map(DatasetRef::label).collect::<Vec<_>>(),
        repeat = req.repeat,
        warmup = req.warmup,
        low_priority = req.low_priority,
        "benchmark started from the web UI"
    );
    match state.benchmark.start(ctx, req) {
        Ok(()) => (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({
                "success": true,
                "message": "benchmark started; poll GET /v1/benchmark for progress",
            })),
        )
            .into_response(),
        Err(e) => json_error(StatusCode::CONFLICT, e.to_string()),
    }
}

/// `POST /v1/benchmark/settings`: the same fields as a start, saved as the `benchmark` config
/// defaults (plus `add_folder` = a directory to add as `dir:<path>`). The saved results are
/// then re-graded for the accuracy metric and weight and their threshold advice recomputed for
/// the objective ([`crate::benchmark::rescore`], no inference); `regraded` says whether they
/// changed.
pub(in crate::server) async fn settings(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let mut form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    crate::server::blocking(move || {
        if let Err(r) = add_folder(&state, &mut form) {
            return *r;
        }
        settings_blocking(&state, &form)
    })
    .await
}

/// The form's `add_folder` (a typed or picked folder, with or without `dir:`) as a `dataset`
/// field of its canonical path: only a directory inside the browsable folders or picked with
/// Browse... in the last hour ([`crate::server::browse::allowed_path`]); otherwise the error
/// answer.
fn add_folder(state: &AppState, form: &mut Vec<(String, String)>) -> Result<(), Box<Response>> {
    let Some(raw) = form
        .iter()
        .find(|(k, _)| k == "add_folder")
        .map(|(_, v)| v.trim().to_string())
        .filter(|v| !v.is_empty())
    else {
        return Ok(());
    };
    let shown = raw.strip_prefix(images::DIR_PREFIX).unwrap_or(&raw).trim();
    let cfg = state.config_read().clone();
    let path = crate::server::browse::allowed_path(&cfg, shown).map_err(|e| {
        let (status, message) = e.into_parts();
        Box::new(json_error(
            status,
            format!(
                "The folder was not added: {message}. Choose it with Browse... (or type a folder \
                 inside the browsable folders)."
            ),
        ))
    })?;
    if !path.is_dir() {
        return Err(Box::new(json_error(
            StatusCode::BAD_REQUEST,
            format!(
                "The folder was not added: {shown} is a file, not a folder. Use Browse... to \
                 choose the folder that holds the images."
            ),
        )));
    }
    let v = format!("{}{}", images::DIR_PREFIX, path.display());
    form.retain(|(k, _)| k != "add_folder");
    if !form.iter().any(|(k, x)| k == "dataset" && *x == v) {
        form.push(("dataset".into(), v));
    }
    Ok(())
}

/// [`settings`] off the runtime threads: dataset checks and the config write (SV-06).
fn settings_blocking(state: &Arc<AppState>, form: &[(String, String)]) -> Response {
    let req = match parse_request(state, form) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    match save_config(state, "benchmark settings", |c| {
        req.to_config(&mut c.benchmark);
        Ok(())
    }) {
        Ok(_) => {
            let (regraded, note) = rescore_saved(state, &req);
            let mut message = format!(
                "Benchmark defaults saved to {} (datasets: {}).",
                state.config_path.display(),
                req.datasets
                    .iter()
                    .map(DatasetRef::label)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            if regraded {
                message.push_str(" The saved results were re-ranked for these settings.");
            }
            if let Some(n) = note {
                message.push(' ');
                message.push_str(&n);
            }
            Json(serde_json::json!({
                "success": true,
                "message": message,
                "regraded": regraded,
            }))
            .into_response()
        }
        Err(e) => json_error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

/// `POST /v1/benchmark/datasets/remove` (form field `dataset` = the row's value, a saved
/// folder dataset's label such as `dir:/srv/alerts`): drop that folder from the saved
/// `benchmark.datasets` and nothing else. It does not go through [`settings`]: that handler
/// keeps the saved list when the form has no `dataset` field (so removing the last one was a
/// no-op) and checks every remaining dataset, so a folder that is gone from the disk could never
/// be removed. The list may end up empty; a run then asks for a dataset. Built-in set ids are
/// not removed here (untick them); a value that is not a saved folder answers 404 and writes
/// nothing.
pub(in crate::server) async fn remove_dataset(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    crate::server::blocking(move || remove_dataset_blocking(&state, &form)).await
}

/// [`remove_dataset`] off the runtime threads: the config write (SV-06).
fn remove_dataset_blocking(state: &AppState, form: &[(String, String)]) -> Response {
    let Some(value) = form
        .iter()
        .find(|(k, _)| k == "dataset")
        .map(|(_, v)| v.trim().to_string())
        .filter(|v| !v.is_empty())
    else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "give `dataset`, the folder dataset to remove".into(),
        );
    };
    let is_it = |d: &DatasetRef| d.dir().is_some() && d.label() == value;
    if !state.config_read().benchmark.datasets.iter().any(is_it) {
        return json_error(
            StatusCode::NOT_FOUND,
            format!(
                "'{value}' is not a saved folder dataset; it may have been removed already \
                 (reload the page). Built-in datasets are unticked, not removed."
            ),
        );
    }
    let mut left = 0;
    match save_config(state, "benchmark dataset removed", |c| {
        c.benchmark.datasets.retain(|d| !is_it(d));
        left = c.benchmark.datasets.len();
        Ok(())
    }) {
        Ok(_) => {
            info!(dataset = %value, left, "benchmark folder dataset removed from the web UI");
            let mut message = format!("Removed {value} from the benchmark datasets.");
            if left == 0 {
                message.push_str(" No dataset is selected; tick one before starting a run.");
            }
            Json(serde_json::json!({ "success": true, "message": message })).into_response()
        }
        Err(e) => json_error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

/// Re-grade the saved results for `req`'s metric, weight and objective and save them when
/// they changed: (changed, a note when it was not done).
fn rescore_saved(state: &AppState, req: &RunRequest) -> (bool, Option<String>) {
    use crate::benchmark::search::{StoredPreds, preds_path};
    if state.benchmark.is_running() {
        return (
            false,
            Some(
                "A benchmark is running; the saved results are re-ranked when it saves them."
                    .into(),
            ),
        );
    }
    let path = results_path(&state.config_path);
    let mut results = match BenchmarkResults::load_with_warnings(&path) {
        Ok((Some(r), w)) if w.is_empty() => r,
        Ok((None, _)) => return (false, None),
        Ok((Some(_), w)) => {
            return (
                false,
                Some(format!(
                    "The saved results were not re-ranked: {} (run the benchmark again to                      replace the entries that do not load).",
                    w.join("; ")
                )),
            );
        }
        Err(e) => {
            return (
                false,
                Some(format!("The saved results were not re-ranked: {e:#}.")),
            );
        }
    };
    let preds = StoredPreds::load_cached(&preds_path(&state.config_path))
        .map_err(|e| tracing::warn!("{e:#}"))
        .ok()
        .flatten();
    let r = crate::benchmark::rescore::rescore(
        &mut results,
        preds.as_deref(),
        req.accuracy_metric,
        req.weights,
        req.threshold_objective,
    );
    if !r.changed {
        return (false, None);
    }
    if let Err(e) = results.save(&path) {
        tracing::warn!("re-ranking the saved benchmark results: {e:#}");
        return (
            false,
            Some(format!(
                "The saved results were not re-ranked: {e:#}. Check that the folder is writable."
            )),
        );
    }
    info!(
        exact = r.exact,
        grid = r.grid,
        "benchmark results re-ranked for new settings"
    );
    (true, None)
}

/// `POST /v1/benchmark/cancel`: 200 when a run was asked to stop, 409 when none runs.
pub(in crate::server) async fn cancel(State(state): State<Arc<AppState>>) -> Response {
    if state.benchmark.cancel() {
        info!("benchmark cancel requested from the web UI");
        Json(serde_json::json!({ "success": true, "message": "cancelling the benchmark" }))
            .into_response()
    } else {
        json_error(StatusCode::CONFLICT, "no benchmark is running".into())
    }
}

/// `GET /v1/benchmark/images?model=&device=`: the images of a model's run on a device with
/// their ground truth and predictions (TP/FP flags, matched/missed objects).
pub(in crate::server) async fn images_detail(
    State(state): State<Arc<AppState>>,
    query: Result<Query<HashMap<String, String>>, QueryRejection>,
) -> Response {
    let q = query.map(|Query(q)| q).unwrap_or_default();
    crate::server::blocking(move || images_detail_blocking(&state, &q)).await
}

/// [`images_detail`] off the runtime threads: it parses `benchmark.json` (SV-06).
fn images_detail_blocking(state: &AppState, q: &HashMap<String, String>) -> Response {
    let (Some(model), Some(device)) = (q.get("model"), q.get("device")) else {
        return json_error(StatusCode::BAD_REQUEST, "give `model` and `device`".into());
    };
    let Some(results) = BenchmarkResults::load_or_warn(&results_path(&state.config_path)) else {
        return json_error(StatusCode::NOT_FOUND, "no benchmark results yet".into());
    };
    let Some(m) = results.find(model) else {
        return json_error(StatusCode::NOT_FOUND, format!("no results for '{model}'"));
    };
    let Some(run) = m
        .devices
        .iter()
        .find(|d| d.device == *device)
        .and_then(|d| d.run.as_ref())
    else {
        return json_error(
            StatusCode::NOT_FOUND,
            format!("'{model}' did not run on {device}"),
        );
    };
    let images: Vec<serde_json::Value> = run
        .per_image
        .iter()
        .map(|r| {
            let info = results
                .image_set(&r.set)
                .and_then(|s| s.images.iter().find(|i| i.file == r.file));
            serde_json::json!({
                "set": r.set,
                "file": r.file,
                "width": info.map(|i| i.width),
                "height": info.map(|i| i.height),
                "tags": info.map(|i| i.tags.clone()).unwrap_or_default(),
                "objects": info.and_then(|i| i.objects.clone()),
                "gtMatched": r.gt_matched,
                "preds": r.preds,
                "counts": r.counts,
                "totalMs": r.total_ms,
                "inferMs": r.infer_ms,
            })
        })
        .collect();
    Json(serde_json::json!({
        "success": true,
        "model": m.model,
        "device": device,
        "groundTruth": run.accuracy.as_ref().map(|a| a.ground_truth.clone()),
        "images": images,
    }))
    .into_response()
}

/// `GET /v1/benchmark/image?set=&file=`: an image of the saved results (the embedded sample, or
/// a file listed in the results under its dataset's directory; nothing else is served).
pub(in crate::server) async fn image_file(
    State(state): State<Arc<AppState>>,
    query: Result<Query<HashMap<String, String>>, QueryRejection>,
) -> Response {
    let q = query.map(|Query(q)| q).unwrap_or_default();
    let (Some(set), Some(file)) = (q.get("set"), q.get("file")) else {
        return json_error(StatusCode::BAD_REQUEST, "give `set` and `file`".into());
    };
    let ctype = |f: &str| {
        if f.to_ascii_lowercase().ends_with(".png") {
            "image/png"
        } else {
            "image/jpeg"
        }
    };
    if set == images::SAMPLE_ID {
        return (
            [(header::CONTENT_TYPE, "image/jpeg")],
            crate::benchmark::DEFAULT_IMAGE,
        )
            .into_response();
    }
    let (set, file) = (set.clone(), file.clone());
    crate::server::blocking(move || {
        let paths = image_paths(&results_path(&state.config_path));
        let Some(path) = paths.get(&(set, file.clone())) else {
            return json_error(
                StatusCode::NOT_FOUND,
                "not an image of the saved results".into(),
            );
        };
        match std::fs::read(path) {
            Ok(bytes) => (
                [
                    (header::CONTENT_TYPE, ctype(&file)),
                    (header::CACHE_CONTROL, "private, max-age=600"),
                ],
                bytes,
            )
                .into_response(),
            Err(e) => json_error(StatusCode::NOT_FOUND, format!("{}: {e}", path.display())),
        }
    })
    .await
}

/// (dataset id, file) -> path of every image of the saved results that may be served
/// ([`crate::benchmark::report::ImageSetInfo::image_path`]).
type ImagePaths = HashMap<(String, String), std::path::PathBuf>;

/// The last [`ImagePaths`] and the results file stamp it was built from: the drill-down asks
/// for every image of a run, which must not parse `benchmark.json` (tens of MB) per image
/// (SV-06).
type ImagePathsCache = Option<(String, Arc<ImagePaths>)>;
static IMAGE_PATHS: std::sync::Mutex<ImagePathsCache> = std::sync::Mutex::new(None);

/// The servable images of the results at `path`, cached by the file's stamp.
fn image_paths(path: &std::path::Path) -> Arc<ImagePaths> {
    image_paths_in(&IMAGE_PATHS, path)
}

/// [`image_paths`] with the cache given (tests).
fn image_paths_in(
    cache: &std::sync::Mutex<ImagePathsCache>,
    path: &std::path::Path,
) -> Arc<ImagePaths> {
    let key = format!("{}|{:?}", path.display(), file_stamp(path));
    if let Some((k, v)) = cache.lock_recover().as_ref()
        && *k == key
    {
        return v.clone();
    }
    let mut map = ImagePaths::new();
    if let Some(r) = BenchmarkResults::load_or_warn(path) {
        for s in &r.image_sets {
            for i in &s.images {
                if let Some(p) = s.image_path(&i.file) {
                    map.entry((s.id.clone(), i.file.clone())).or_insert(p);
                }
            }
        }
    }
    let map = Arc::new(map);
    *cache.lock_recover() = Some((key, map.clone()));
    map
}

#[cfg(test)]
mod tests {
    use super::named_folder;
    use crate::benchmark::BenchmarkResults;
    use crate::benchmark::images::{GroundTruth, SetKind};
    use crate::benchmark::report::ImageSetInfo;
    use crate::config::DatasetRef;
    use std::path::PathBuf;

    #[test]
    fn a_held_out_set_is_found_by_its_name_only() {
        let mut r: BenchmarkResults = serde_json::from_str("{}").unwrap();
        let set = |id: &str, kind, dir: &str| ImageSetInfo {
            id: id.into(),
            title: String::new(),
            kind,
            dir: Some(PathBuf::from(dir)),
            license: String::new(),
            attribution: String::new(),
            source: String::new(),
            ground_truth: GroundTruth::Yolo,
            scored_labels: None,
            images: Vec::new(),
        };
        r.image_sets = vec![
            set("heldout-a", SetKind::Dir, "/t/run/heldout/images"),
            set("plain", SetKind::Dir, "/t/flat"),
            set("coco-cctv", SetKind::Builtin, "/t/bench/coco"),
            set("dir:/x", SetKind::Dir, "/x"),
        ];
        assert_eq!(
            named_folder(&r, "heldout-a"),
            Some(DatasetRef::Dir {
                dir: PathBuf::from("/t/run/heldout"),
                gt: None,
                name: Some("heldout-a".into()),
            }),
            "the folder that holds images/"
        );
        assert!(matches!(
            named_folder(&r, "plain"),
            Some(DatasetRef::Dir { dir, .. }) if dir.as_path() == std::path::Path::new("/t/flat")
        ));
        assert_eq!(
            named_folder(&r, "coco-cctv"),
            None,
            "built-in: an id stays an id"
        );
        assert_eq!(named_folder(&r, "dir:/x"), None, "already a path");
        assert_eq!(named_folder(&r, "unknown"), None);
    }

    use super::super::tests::{call, get, post, state};
    use super::*;
    use crate::server::router;
    use std::time::Duration;
    use tower::ServiceExt;
    #[tokio::test]
    async fn settings_save_accuracy_metric_and_objective() {
        let st = state();
        let (s, v) = call(
            &st,
            post(
                "/v1/benchmark/settings",
                "dataset=sample&accuracy_metric=roc_auc&threshold_objective=youden                 &low_priority=0&low_priority=1",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let last = st.config.changes(1).pop().expect("logged");
        assert_eq!(
            (last.source.as_str(), last.revision),
            ("benchmark settings", 2)
        );
        assert!(
            last.summary
                .contains("Benchmark accuracy metric \u{2192} roc_auc"),
            "{}",
            last.summary
        );
        let c = crate::config::Config::load(&st.config_path).unwrap();
        assert_eq!(
            c.benchmark.accuracy_metric,
            crate::benchmark::grade::AccuracyMetric::RocAuc
        );
        assert_eq!(
            c.benchmark.threshold_objective,
            crate::benchmark::threshold::Objective::Youden
        );
        assert!(
            c.benchmark.low_priority,
            "checked checkbox after the hidden 0"
        );
        let resp = router(st.clone()).oneshot(get("/benchmark")).await.unwrap();
        let html = String::from_utf8(
            axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(
            html.contains("<option value=\"roc_auc\" selected>"),
            "{html}"
        );
        assert!(
            html.contains("<option value=\"youden\" selected>"),
            "{html}"
        );
        assert!(
            html.contains(r#"name="low_priority" value="1" checked>"#),
            "{html}"
        );
        // Unchecked: only the hidden 0 is sent.
        let (s, v) = call(
            &st,
            post("/v1/benchmark/settings", "dataset=sample&low_priority=0"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let c = crate::config::Config::load(&st.config_path).unwrap();
        assert!(!c.benchmark.low_priority);
        let resp = router(st.clone()).oneshot(get("/benchmark")).await.unwrap();
        let html = String::from_utf8(
            axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(html.contains(r#"name="low_priority" value="1">"#), "{html}");
        let (s, _) = call(
            &st,
            post(
                "/v1/benchmark/settings",
                "dataset=sample&accuracy_metric=map",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn settings_add_folder_only_inside_the_browsable_folders() {
        let st = state();
        let root =
            crate::server::browse::plain(&st.config_path.parent().unwrap().canonicalize().unwrap());
        let r = root.clone();
        crate::server::save_config(&st, "test", move |c| {
            c.download_dir = Some(r);
            Ok(())
        })
        .unwrap();
        let enc =
            |p: &std::path::Path| crate::server::encode_uri_component(&p.display().to_string());
        let dirs = |st: &AppState| -> Vec<String> {
            crate::config::Config::load(&st.config_path)
                .unwrap()
                .benchmark
                .datasets
                .iter()
                .map(DatasetRef::label)
                .collect()
        };

        // Inside the data root: added under its canonical path (a relative path resolves
        // against the data root).
        let imgs = root.join("imgs");
        std::fs::create_dir_all(&imgs).unwrap();
        let (s, v) = call(
            &st,
            post("/v1/benchmark/settings", "dataset=sample&add_folder=imgs"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let want = format!("{}{}", images::DIR_PREFIX, imgs.display());
        assert!(dirs(&st).contains(&want), "{:?} lacks {want}", dirs(&st));

        // A file is not a folder.
        std::fs::write(root.join("f.jpg"), b"x").unwrap();
        let (s, v) = call(
            &st,
            post(
                "/v1/benchmark/settings",
                &format!("dataset=sample&add_folder={}", enc(&root.join("f.jpg"))),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        assert!(v["message"].as_str().unwrap().contains("Browse..."), "{v}");

        // Outside every root (Unix only: on Windows every drive is a root): refused, nothing
        // saved.
        #[cfg(unix)]
        {
            let outside = std::env::temp_dir()
                .join(format!("bo_bench_out_{}", uuid::Uuid::new_v4().simple()));
            std::fs::create_dir_all(&outside).unwrap();
            let before = dirs(&st);
            let (s, v) = call(
                &st,
                post(
                    "/v1/benchmark/settings",
                    &format!("dataset=sample&add_folder=dir%3A{}", enc(&outside)),
                ),
            )
            .await;
            assert_eq!(s, StatusCode::FORBIDDEN, "{v}");
            let m = v["message"].as_str().unwrap();
            assert!(
                m.contains("outside the browsable folders") && m.contains("Browse..."),
                "{m}"
            );
            assert_eq!(dirs(&st), before);
            // Start sends the typed folder as a `dir:` dataset: the same check.
            let (s, v) = call(
                &st,
                post("/v1/benchmark", &format!("dataset=dir%3A{}", enc(&outside))),
            )
            .await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
            let m = v["message"].as_str().unwrap();
            assert!(m.contains("outside the browsable folders"), "{m}");
            assert!(!st.benchmark.is_running());
            let _ = std::fs::remove_dir_all(&outside);
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Save `datasets` as the config's benchmark datasets (a test seed, not under test).
    fn seed_datasets(st: &Arc<AppState>, datasets: Vec<DatasetRef>) {
        crate::server::save_config(st, "test", move |c| {
            c.benchmark.datasets = datasets;
            Ok(())
        })
        .unwrap();
    }

    /// The saved dataset labels, read back from the config file.
    fn saved_labels(st: &AppState) -> Vec<String> {
        crate::config::Config::load(&st.config_path)
            .unwrap()
            .benchmark
            .datasets
            .iter()
            .map(DatasetRef::label)
            .collect()
    }

    /// `POST /v1/benchmark/datasets/remove` with `dataset` = `value`.
    fn remove_req(value: &str) -> axum::http::Request<axum::body::Body> {
        post(
            "/v1/benchmark/datasets/remove",
            &format!("dataset={}", crate::server::encode_uri_component(value)),
        )
    }

    /// The `/benchmark` page body (status checked).
    async fn page_html(st: &Arc<AppState>) -> String {
        let resp = router(st.clone()).oneshot(get("/benchmark")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        String::from_utf8(
            axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap()
    }

    /// A folder dataset ref for `dir`, as the page saves it.
    fn folder_ref(dir: &std::path::Path) -> DatasetRef {
        DatasetRef::Id(format!("{}{}", images::DIR_PREFIX, dir.display()))
    }

    #[tokio::test]
    async fn remove_dataset_drops_the_only_saved_folder() {
        let st = state();
        let dir = st
            .config_path
            .parent()
            .unwrap()
            .join("bo_only_folder_marker");
        std::fs::create_dir_all(&dir).unwrap();
        let folder = folder_ref(&dir);
        seed_datasets(&st, vec![folder.clone()]);
        let html = page_html(&st).await;
        assert!(html.contains("bo_only_folder_marker"), "{html}");
        assert!(html.contains("data-remove-folder"), "{html}");

        let (s, v) = call(&st, remove_req(&folder.label())).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["success"], true, "{v}");
        let m = v["message"].as_str().unwrap();
        assert!(
            m.contains("bo_only_folder_marker") && m.contains("tick one before starting a run"),
            "{m}"
        );
        let last = st.config.changes(1).pop().expect("logged");
        assert_eq!(last.source, "benchmark dataset removed");
        // Explicit empty list on disk: it must not fall back to the built-in sets.
        assert_eq!(saved_labels(&st), Vec::<String>::new());
        assert_eq!(saved_labels(&st), Vec::<String>::new(), "after a reload");
        assert!(st.config_read().benchmark.datasets.is_empty());
        // The page still renders and has no row for the folder.
        let html = page_html(&st).await;
        assert!(!html.contains("bo_only_folder_marker"), "{html}");
        assert!(!html.contains("data-remove-folder"), "{html}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn remove_dataset_drops_a_folder_that_is_gone_from_the_disk() {
        let st = state();
        let gone = st.config_path.parent().unwrap().join("bo_gone_folder");
        assert!(!gone.exists());
        let folder = folder_ref(&gone);
        seed_datasets(&st, vec![folder.clone()]);
        let html = page_html(&st).await;
        assert!(html.contains("directory not found"), "{html}");
        // The settings form cannot do this (it validates the folder); the dedicated route can.
        let (s, v) = call(&st, remove_req(&folder.label())).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert!(saved_labels(&st).is_empty());
        assert!(!page_html(&st).await.contains("bo_gone_folder"));
    }

    #[tokio::test]
    async fn remove_dataset_keeps_the_other_datasets() {
        let st = state();
        let base = st.config_path.parent().unwrap();
        let (a, b) = (base.join("bo_keep_a"), base.join("bo_keep_b"));
        seed_datasets(
            &st,
            vec![
                DatasetRef::Id("coco-cctv".into()),
                folder_ref(&a),
                DatasetRef::Dir {
                    dir: b.clone(),
                    gt: None,
                    name: Some("porch".into()),
                },
            ],
        );
        let (s, v) = call(&st, remove_req(&folder_ref(&a).label())).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert!(!v["message"].as_str().unwrap().contains("tick one"), "{v}");
        assert_eq!(saved_labels(&st), ["coco-cctv", "porch"]);
        // A named folder is removed by its label.
        let (s, v) = call(&st, remove_req("porch")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(saved_labels(&st), ["coco-cctv"]);
    }

    #[tokio::test]
    async fn remove_dataset_refuses_built_in_and_unknown_values_without_writing() {
        let st = state();
        let dir = st.config_path.parent().unwrap().join("bo_refuse_folder");
        seed_datasets(
            &st,
            vec![DatasetRef::Id("coco-cctv".into()), folder_ref(&dir)],
        );
        let revision = || st.config.changes(1).pop().map(|c| c.revision);
        let (before, rev) = (saved_labels(&st), revision());
        for value in [
            "coco-cctv",
            "sample",
            "nope",
            "dir:/not/a/saved/folder",
            "bo_refuse_folder",
        ] {
            let (s, v) = call(&st, remove_req(value)).await;
            assert_eq!(s, StatusCode::NOT_FOUND, "{value}: {v}");
            assert_eq!(v["success"], false, "{value}: {v}");
            assert_eq!(saved_labels(&st), before, "{value}");
            assert_eq!(revision(), rev, "{value} must not write the config");
        }
        // No `dataset` field at all.
        let (s, v) = call(&st, post("/v1/benchmark/datasets/remove", "")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        assert_eq!(saved_labels(&st), before);
        assert_eq!(revision(), rev);
    }

    #[tokio::test]
    async fn start_with_no_saved_dataset_and_no_field_is_refused() {
        let st = state();
        seed_datasets(&st, Vec::new());
        let (s, v) = call(&st, post("/v1/benchmark", "model=ipcam-general")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .contains("choose at least one dataset"),
            "{v}"
        );
        assert!(!st.benchmark.is_running());
    }

    #[tokio::test]
    async fn start_status_busy_and_cancel() {
        let _queue = crate::server::training::QUEUE_TEST_LOCK.lock().await;
        let st = state();
        let (s, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["state"], "idle");
        assert_eq!(v["running"], false);
        assert!(v["results"].is_null());
        assert_eq!(v["config"]["models"][0]["name"], "ipcam-general");
        assert_eq!(v["config"]["models"][1]["enabled"], false);

        // Bad input.
        let (s, v) = call(&st, post("/v1/benchmark", "repeat=0")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert!(v["message"].as_str().unwrap().contains("repeat"));
        let (s, v) = call(&st, post("/v1/benchmark", "dataset=nope")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert!(v["message"].as_str().unwrap().contains("valid ids"), "{v}");
        let (s, _) = call(
            &st,
            post("/v1/benchmark", "model=ipcam-general&device=vulkan"),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _) = call(&st, post("/v1/benchmark/cancel", "")).await;
        assert_eq!(s, StatusCode::CONFLICT);

        // A held run: a second start is refused with 409, cancel ends it.
        let mut req = RunRequest::defaults(&crate::config::BenchmarkConfig::default());
        req.models = vec!["ipcam-general".into()];
        req.low_priority = false;
        st.benchmark
            .start_job(req, |h| {
                while !h.is_cancelled() {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(crate::benchmark::Cancelled.into())
            })
            .unwrap();
        let (s, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(
            (s, v["running"].clone()),
            (StatusCode::OK, serde_json::json!(true))
        );
        let (s, v) = call(&st, post("/v1/benchmark", "model=ipcam-general")).await;
        assert_eq!(s, StatusCode::CONFLICT, "{v}");
        let (s, _) = call(&st, post("/v1/benchmark/cancel", "")).await;
        assert_eq!(s, StatusCode::OK);
        assert!(st.benchmark.wait_idle(Duration::from_secs(5)));
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(v["state"], "cancelled");

        // A real start: the model file is missing, so the run ends quickly without inference
        // and records why (or fails when no runtime can be initialized in this environment).
        let (s, v) = call(
            &st,
            post(
                "/v1/benchmark",
                "model=ipcam-general&device=openvino:cpu&repeat=1&warmup=0&low_priority=0",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        assert!(st.benchmark.wait_idle(Duration::from_secs(60)));
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        match v["state"].as_str().unwrap() {
            "done" => {
                let err = v["run"]["models"][0]["error"].as_str().unwrap();
                assert!(err.contains("not found"), "{v}");
                // Saved, so the page shows it after a reload.
                assert_eq!(v["results"]["models"][0]["model"], "ipcam-general");
            }
            "failed" => assert!(v["run"]["error"].is_string(), "{v}"),
            other => panic!("unexpected state {other}: {v}"),
        }
    }

    /// G1: more than one GPU needs the confirmation, and the answer says what it risks; a device
    /// of each runtime on the same card counts too. The refused run never starts.
    #[tokio::test]
    async fn several_gpus_need_a_confirmation() {
        let _queue = crate::server::training::QUEUE_TEST_LOCK.lock().await;
        assert_eq!(
            gpu_devices(&[
                "openvino:cpu".into(),
                "ort:cuda:0".into(),
                "openvino:gpu.0".into()
            ]),
            ["ort:cuda:0", "openvino:gpu.0"]
        );
        assert!(gpu_devices(&["ort:cpu".into(), "openvino:cpu".into()]).is_empty());
        assert_eq!(gpu_devices(&["ort:coreml".into()]), ["ort:coreml"]);

        let st = state();
        let two = "model=ipcam-general&device=ort:directml:0&device=ort:directml:1\
                   &repeat=1&warmup=0&low_priority=0";
        let (s, v) = call(&st, post("/v1/benchmark", two)).await;
        assert_eq!(s, StatusCode::CONFLICT, "{v}");
        assert_eq!(v["success"], false);
        assert_eq!(v["confirm"], CONFIRM_GPUS);
        let m = v["message"].as_str().unwrap();
        assert!(m.contains("ort:directml:0, ort:directml:1"), "{m}");
        assert!(
            m.contains("has hung a machine") && m.contains("never found"),
            "{m}"
        );
        assert!(!st.benchmark.is_running());
        let (s, v) = call(
            &st,
            post("/v1/benchmark", &format!("{two}&{CONFIRM_GPUS}=0")),
        )
        .await;
        assert_eq!(s, StatusCode::CONFLICT, "{v}");

        // Neither device exists on the test machines' selection or loads: failed rows, no GPU
        // is touched.
        let (s, v) = call(
            &st,
            post("/v1/benchmark", &format!("{two}&{CONFIRM_GPUS}=1")),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        assert!(st.benchmark.wait_idle(Duration::from_secs(60)));
    }

    /// The running job of the queue is named; a queued one holds nothing yet.
    #[test]
    fn a_running_job_is_named() {
        use crate::training::dataset::SceneFilter;
        use crate::training::service::RunState;
        let run = |state| {
            let mut r = crate::server::training::tests::record(
                "r",
                SceneFilter::default(),
                None,
                state,
                None,
            );
            r.name = "day-yolo26s".into();
            r
        };
        let task = |kind, state| TaskRecord {
            id: "t".into(),
            kind,
            title: "900 frames".into(),
            created: String::new(),
            state,
            stage: None,
            step: 0,
            steps: 1,
            percent: 0,
            detail: String::new(),
            log: vec![],
            summary: None,
            result: None,
            error: None,
            finished: None,
            after: None,
            uses: Vec::new(),
        };
        assert_eq!(running_job(&[], &[]), None);
        for state in [RunState::Queued, RunState::Done, RunState::Failed] {
            assert_eq!(
                running_job(&[run(state)], &[task(TaskKind::Import, state)]),
                None,
                "{state:?}"
            );
        }
        for state in [RunState::Running, RunState::Paused] {
            assert_eq!(
                running_job(&[run(state)], &[]).as_deref(),
                Some("Training run 'day-yolo26s'")
            );
        }
        for (kind, text) in [
            (TaskKind::Import, "Import '900 frames'"),
            (TaskKind::Autolabel, "Auto-label run '900 frames'"),
            (TaskKind::Overlaps, "Overlap check '900 frames'"),
            (TaskKind::Refine, "Dataset check '900 frames'"),
        ] {
            assert_eq!(
                running_job(&[], &[task(kind, RunState::Running)]).as_deref(),
                Some(text)
            );
        }
    }

    /// A job on the real queue, through the real route: 409 with its name while it runs, and the
    /// benchmark is free again when it ended.
    #[tokio::test]
    async fn a_running_queue_job_refuses_a_benchmark() {
        use crate::training::task::{Task, TaskContext, TaskOutcome, TaskReport};
        use std::sync::atomic::{AtomicBool, Ordering};
        /// Runs until its cancel flag is set; tells when it started.
        struct Waits(Arc<AtomicBool>, String);
        impl Task for Waits {
            fn id(&self) -> &str {
                &self.1
            }
            fn kind(&self) -> TaskKind {
                TaskKind::Overlaps
            }
            fn title(&self) -> String {
                "overlaps of the test".into()
            }
            fn steps(&self) -> usize {
                1
            }
            fn run(
                &self,
                _: TaskContext<'_>,
                _: &mut dyn FnMut(TaskReport<'_>),
                cancel: &AtomicBool,
            ) -> anyhow::Result<TaskOutcome> {
                self.0.store(true, Ordering::SeqCst);
                while !cancel.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(crate::resources::export::Cancelled.into())
            }
        }
        let _queue = crate::server::training::QUEUE_TEST_LOCK.lock().await;
        // Its own data root: the queue keeps its records in the config's training folder.
        let st = crate::server::test_util::test_state();
        let shared: crate::server::Shared = st.clone().into();
        let svc = crate::server::training::service(&shared);
        let started = Arc::new(AtomicBool::new(false));
        let id = uuid::Uuid::new_v4().simple().to_string();
        svc.enqueue_task(Box::new(Waits(started.clone(), id.clone())))
            .unwrap();
        for _ in 0..2000 {
            if started.load(Ordering::SeqCst) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(started.load(Ordering::SeqCst), "the job did not start");

        let (s, v) = call(&st, post("/v1/benchmark", "model=ipcam-general")).await;
        assert_eq!(s, StatusCode::CONFLICT, "{v}");
        let m = v["message"].as_str().unwrap();
        assert!(
            m.contains("Overlap check 'overlaps of the test' is running"),
            "{m}"
        );
        assert!(m.contains("Jobs tab"), "{m}");
        assert!(!st.benchmark.is_running());

        svc.cancel(&id);
        let done = svc.wait_task(&id, Duration::from_secs(30)).unwrap();
        assert!(!done.state.is_busy(), "{done:?}");
        let (s, v) = call(
            &st,
            post(
                "/v1/benchmark",
                "model=ipcam-general&device=openvino:cpu&repeat=1&warmup=0&low_priority=0",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        assert!(st.benchmark.wait_idle(Duration::from_secs(60)));
    }

    #[tokio::test]
    async fn image_file_serves_only_listed_images_from_a_cached_index() {
        let st = state();
        let cfg_path = st.config_path.clone();
        let dir = cfg_path.parent().unwrap().join("imgs");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.jpg"), b"jpeg bytes").unwrap();
        std::fs::write(dir.join("secret.jpg"), b"not listed").unwrap();
        let (mut results, _) = crate::benchmark::search::tests::fixture();
        results.image_sets[0].dir = Some(dir.clone());
        let path = results_path(&cfg_path);
        results.save(&path).unwrap();

        let (s, v) = call(&st, get("/v1/benchmark/image?set=night-set&file=a.jpg")).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v, serde_json::Value::String("jpeg bytes".into()));
        for q in [
            "set=night-set&file=secret.jpg",
            "set=night-set&file=..%2Fimgs%2Fa.jpg",
            "set=pseudo-set&file=a.jpg",
        ] {
            let (s, _) = call(&st, get(&format!("/v1/benchmark/image?{q}"))).await;
            assert_eq!(s, StatusCode::NOT_FOUND, "{q}");
        }

        // The index is built once per results file version.
        let cache = std::sync::Mutex::new(None);
        let a = image_paths_in(&cache, &path);
        assert_eq!(
            a.get(&("night-set".to_string(), "a.jpg".to_string())),
            Some(&dir.join("a.jpg"))
        );
        // No directory (pseudo-set): nothing servable.
        assert!(!a.keys().any(|(set, _)| set == "pseudo-set"));
        assert!(Arc::ptr_eq(&a, &image_paths_in(&cache, &path)));
        results.image_sets[0].images.truncate(1);
        results.save(&path).unwrap();
        let b = image_paths_in(&cache, &path);
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(b.len(), 1);
    }
}
