//! Benchmark run endpoints: start (`POST /v1/benchmark`), cancel, save the form as the
//! `benchmark` config defaults, and the per-image drill-down (`/v1/benchmark/images`) with the
//! image file itself (`/v1/benchmark/image`, only files listed in the saved results, through a
//! cached index keyed by the results file stamp). File I/O runs in `server::blocking`.

use super::{file_stamp, json_error, loaded_devices};
use crate::benchmark::images;
use crate::benchmark::service::{RunContext, RunRequest};
use crate::benchmark::{BenchmarkResults, results_path};
use crate::config::DatasetRef;
use crate::server::{AppState, save_config};
use crate::sync::LockExt;
use axum::Json;
use axum::extract::rejection::{FormRejection, QueryRejection};
use axum::extract::{Form, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

/// Parse a start/settings form over the configured defaults and check the datasets.
fn parse_request(state: &AppState, form: &[(String, String)]) -> anyhow::Result<RunRequest> {
    let cfg = state.config_read().clone();
    let req = RunRequest::from_form(form, &cfg.benchmark)?;
    images::validate(&req.datasets, &cfg.data_root())?;
    Ok(req)
}

/// `POST /v1/benchmark` (urlencoded; every field optional, defaults from the `benchmark`
/// config): `model`, `device`, `dataset` (repeated), `max_images`, `repeat`, `warmup`,
/// `reference_model`, `accuracy_weight`. Starts a run in the background: 202, 400 (bad input)
/// or 409 (a run is in progress).
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
/// defaults (plus `add_folder` = a directory to add as `dir:<path>`).
pub(in crate::server) async fn settings(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let mut form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    let folder: Option<String> = form
        .iter()
        .find(|(k, _)| k == "add_folder")
        .map(|(_, v)| v.trim().to_string())
        .filter(|v| !v.is_empty());
    if let Some(f) = &folder {
        let v = if f.starts_with(images::DIR_PREFIX) {
            f.clone()
        } else {
            format!("{}{f}", images::DIR_PREFIX)
        };
        if !form.iter().any(|(k, x)| k == "dataset" && *x == v) {
            form.push(("dataset".into(), v));
        }
    }
    crate::server::blocking(move || settings_blocking(&state, &form)).await
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
        Ok(_) => Json(serde_json::json!({
            "success": true,
            "message": format!(
                "Benchmark defaults saved to {} (datasets: {}).",
                state.config_path.display(),
                req.datasets.iter().map(DatasetRef::label).collect::<Vec<_>>().join(", ")
            ),
        }))
        .into_response(),
        Err(e) => json_error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
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
    async fn start_status_busy_and_cancel() {
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
