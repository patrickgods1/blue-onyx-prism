//! Benchmark apply endpoints: write measured per-model devices (`/v1/benchmark/apply`) and
//! searched confidence thresholds (`/v1/benchmark/apply-threshold`) into the config, and the
//! exact threshold search over the stored predictions (`/v1/benchmark/threshold-search`).
//! Every config write goes through `save_config`; file I/O and search run in `server::blocking`.

use super::{ApiError, form_field, form_models, json_error, truthy};
use crate::benchmark::{BenchmarkResults, results_path};
use crate::config::Config;
use crate::config_form::{
    DeviceChange, ThresholdChange, apply_model_devices, apply_model_thresholds,
};
use crate::server::{AppState, save_config, schedule_restart};
use axum::Json;
use axum::extract::rejection::FormRejection;
use axum::extract::{Form, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

/// `POST /v1/benchmark/apply`: set per-model devices in the config file.
/// - `model=<name>&device=<spec>`: one model ("" or `global` = follow the global device);
/// - `all=1` (optionally limited by repeated `model`): every recommendation of the saved results.
///
/// The changed models reload on their new devices at once (per-model reload); `restart=1`
/// restarts the server when a change could not be applied that way. Answers JSON with the
/// changes and `previous` (model -> its device before, a spec or `global`: what Undo posts
/// back as `device`); 400 on bad input, 404 when `all` finds no recommendation.
pub(in crate::server) async fn apply(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    crate::server::blocking(move || apply_blocking(&state, &form)).await
}

/// [`apply`] off the runtime threads: it parses the results, writes the config and reloads
/// the changed models (SV-06).
fn apply_blocking(state: &Arc<AppState>, form: &[(String, String)]) -> Response {
    let restart = form_field(form, "restart").is_some_and(truthy);
    let picks = match device_picks(state, form) {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };
    let mut changes: Vec<DeviceChange> = Vec::new();
    let saved = save_config(state, "benchmark apply", |c| {
        changes = apply_model_devices(c, &picks)?;
        Ok(())
    });
    if let Err(e) = saved {
        warn!("benchmark apply refused: {e:#}");
        return json_error(StatusCode::BAD_REQUEST, format!("{e:#}"));
    }
    let (message, restarting) = device_apply_outcome(state, &changes, restart);
    info!("benchmark apply: {message}");
    // For Undo: POST `model=<m>&device=<previous>` (`global` = follow the global device).
    let previous: serde_json::Map<String, serde_json::Value> = changes
        .iter()
        .map(|c| {
            (
                c.model.clone(),
                c.from.clone().unwrap_or_else(|| "global".into()).into(),
            )
        })
        .collect();
    Json(serde_json::json!({
        "success": true,
        "message": message,
        "changes": changes,
        "previous": previous,
        "restarting": restarting,
    }))
    .into_response()
}

/// The saved results of the apply actions: 404 when there are none yet, 500 when they cannot
/// be read.
fn results_for_apply(state: &AppState) -> Result<BenchmarkResults, ApiError> {
    match BenchmarkResults::load(&results_path(&state.config_path)) {
        Ok(Some(r)) => Ok(r),
        Ok(None) => Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "no benchmark results yet; run a benchmark first",
        )),
        Err(e) => Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{e:#}"),
        )),
    }
}

/// The (model, device) pairs of an apply form: `all=1` takes the saved recommendations of
/// the configured models (limited to the form's models, if any), else one `model` + `device`.
fn device_picks(
    state: &AppState,
    form: &[(String, String)],
) -> Result<Vec<(String, Option<String>)>, ApiError> {
    let models = form_models(form);
    if !form_field(form, "all").is_some_and(truthy) {
        return match (models.as_slice(), form_field(form, "device")) {
            ([model], Some(device)) => Ok(vec![(model.clone(), Some(device.to_string()))]),
            _ => Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "give `model` and `device`, or `all=1` to apply every recommendation",
            )),
        };
    }
    let results = results_for_apply(state)?;
    let only = (!models.is_empty()).then_some(models.as_slice());
    let recs = results.recommendations(only);
    if recs.is_empty() {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "the saved results recommend no device for these models",
        ));
    }
    // Results may name models that were removed from the config since: skip those.
    let cfg = state.config_read().clone();
    Ok(recs
        .into_iter()
        .filter(|(m, _)| crate::benchmark::find_model(&cfg, m).is_some())
        .map(|(m, d)| (m, Some(d)))
        .collect())
}

/// The answer to saved device `changes` and whether the server restarts (scheduled here when
/// `restart` is asked and a change could not be applied by a per-model reload).
fn device_apply_outcome(
    state: &Arc<AppState>,
    changes: &[DeviceChange],
    restart: bool,
) -> (String, bool) {
    let force_cpu = state.config_read().force_cpu;
    let mut message = if changes.is_empty() {
        "No change: the models already use these devices.".to_string()
    } else {
        format!(
            "Saved {}: {}.",
            state.config_path.display(),
            changes
                .iter()
                .map(DeviceChange::describe)
                .collect::<Vec<_>>()
                .join("; ")
        )
    };
    if force_cpu {
        message.push_str(" Note: Force CPU is on and overrides per-model devices.");
    }
    // `save_config` reloaded the changed models (per-model reload); a change that could not
    // be applied that way stays pending and needs a restart (`restart=1` does it now).
    let pending = crate::config_merge::pending(&state.running(), &state.config_read());
    let unapplied = changes
        .iter()
        .any(|c| !pending.for_model(&c.model).is_empty());
    let restarting = restart && unapplied;
    if restarting {
        message.push_str(" Restarting the server to load the models on their new devices.");
        schedule_restart(state);
    } else if unapplied {
        message.push_str(" Restart the server to apply.");
    } else if changes
        .iter()
        .any(|c| state.registry.by_name(&c.model).is_some())
    {
        message.push_str(" Reloading the models on their new devices (no restart needed).");
    } else if !changes.is_empty() {
        message.push_str(" Used when the model is enabled (no restart needed).");
    }
    (message, restarting)
}

/// The saved results and stored predictions, or a 400 telling to run a benchmark first.
fn search_data(
    state: &AppState,
) -> Result<(BenchmarkResults, Arc<crate::benchmark::search::StoredPreds>), ApiError> {
    use crate::benchmark::search::{StoredPreds, preds_path};
    let none = || {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "no stored predictions: run a benchmark first (it stores the predictions the \
             threshold search needs)",
        )
    };
    let results = match BenchmarkResults::load(&results_path(&state.config_path)) {
        Ok(Some(r)) => r,
        Ok(None) => return Err(none()),
        Err(e) => {
            return Err(ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("{e:#}"),
            ));
        }
    };
    match StoredPreds::load_cached(&preds_path(&state.config_path)) {
        Ok(Some(p)) if !p.runs.is_empty() => Ok((results, p)),
        Ok(_) => Err(none()),
        Err(e) => Err(ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{e:#}"),
        )),
    }
}

/// Normalized model name -> its effective confidence threshold in the config (2 decimals).
fn current_thresholds(cfg: &Config) -> HashMap<String, f64> {
    cfg.models
        .iter()
        .map(|m| {
            (
                crate::registry::normalize_name(&m.effective_name()),
                crate::config::round_threshold_f64(
                    m.confidence_threshold.unwrap_or(cfg.confidence_threshold),
                ),
            )
        })
        .collect()
}

/// `GET /v1/benchmark/threshold-search`: the searchable models (with their stored devices and
/// recommended device), datasets, tags and classes, the configured objective and thresholds.
/// With nothing to search yet (no results or no stored predictions) it answers 200 with
/// `available: false` and the message, so the page's empty state is not a console error.
pub(in crate::server) async fn search_options(State(state): State<Arc<AppState>>) -> Response {
    crate::server::blocking(move || {
        let (results, preds) = match search_data(&state) {
            Ok(d) => d,
            Err(e) if e.status == StatusCode::BAD_REQUEST => {
                return Json(serde_json::json!({
                    "success": true,
                    "available": false,
                    "message": e.message,
                }))
                .into_response();
            }
            Err(e) => return e.into_response(),
        };
        let cfg = state.config_read().clone();
        Json(serde_json::json!({
            "success": true,
            "available": true,
            "options": crate::benchmark::search::options(&results, &preds),
            "objective": cfg.benchmark.threshold_objective,
            "thresholds": current_thresholds(&cfg),
        }))
        .into_response()
    })
    .await
}

/// `POST /v1/benchmark/threshold-search` (urlencoded): `model` (repeated or comma separated;
/// `all` or none = every model), `device` (none = recommended), `dataset`, `tag`, `class`
/// (repeated), `objective` (`f1`, `f2`, `precision:<p>`, `recall:<r>`; default the configured
/// one), `iou` (0.5). Answers per model the best threshold, P/R/F1/F2/FP per image at the best
/// and the configured thresholds, the exact curve and the 0.05 grid, per-dataset and per-class
/// bests. 400 on bad input, when nothing is stored ("run a benchmark first") or nothing
/// matches.
pub(in crate::server) async fn threshold_search(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    crate::server::blocking(move || threshold_search_blocking(&state, &form)).await
}

/// [`threshold_search`] off the runtime threads: it parses the results and the stored
/// predictions and runs the exhaustive search (SV-06, BQ-04).
fn threshold_search_blocking(state: &AppState, form: &[(String, String)]) -> Response {
    let cfg = state.config_read().clone();
    let req = match crate::benchmark::search::SearchRequest::from_form(
        form,
        cfg.benchmark.threshold_objective,
    ) {
        Ok(r) => r,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    let (results, preds) = match search_data(state) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };
    let current = current_thresholds(&cfg);
    let lookup = |m: &str| {
        current.get(&crate::registry::normalize_name(m)).map(|t| {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "a confidence threshold (0..1); f32 is the precision the models use"
            )]
            let t = *t as f32;
            t
        })
    };
    match crate::benchmark::search::search(&results, &preds, &req, &lookup) {
        Ok(out) if out.results.is_empty() => json_error(
            StatusCode::BAD_REQUEST,
            format!("nothing to search: {}", out.errors.join("; ")),
        ),
        Ok(out) => {
            let csv = crate::benchmark::search::to_csv(&out);
            Json(serde_json::json!({
                "success": true,
                "request": req,
                "results": out.results,
                "errors": out.errors,
                "elapsedMs": out.elapsed_ms,
                "csv": csv,
                "note": BLUE_IRIS_THRESHOLD_NOTE,
            }))
            .into_response()
        }
        Err(e) => json_error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

/// Shown with threshold changes: how the server threshold relates to Blue Iris's settings.
pub(super) const BLUE_IRIS_THRESHOLD_NOTE: &str = "Blue Iris may send its own min_confidence \
     with each request (it then replaces this threshold for that request), and each camera's \
     minimum confidence filters the returned objects again.";

/// `POST /v1/benchmark/apply-threshold`: set per-model `confidence_threshold` in the config.
/// - `model=<name>&threshold=<0..1>`: one model (repeat the pair for several);
///   `threshold=global` clears the model's own threshold (the Undo of a first apply);
/// - `all=1` (optionally limited by repeated `model`): every best threshold of the saved results.
///
/// Thresholds apply live (no restart; `restart=1` is accepted and ignored, `restarting` is
/// always false). Answers JSON with the changes and `previous` (model -> its own threshold
/// before, `0.40` or `global`: what Undo posts back as `threshold`); 400 on bad input, 404
/// when `all` finds no best threshold.
pub(in crate::server) async fn apply_threshold(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    crate::server::blocking(move || apply_threshold_blocking(&state, &form)).await
}

/// [`apply_threshold`] off the runtime threads: it parses the results and writes the config
/// (SV-06).
fn apply_threshold_blocking(state: &Arc<AppState>, form: &[(String, String)]) -> Response {
    let picks = match threshold_picks(state, form) {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };
    let mut changes: Vec<ThresholdChange> = Vec::new();
    if let Err(e) = save_config(state, "benchmark apply-threshold", |c| {
        let set: Vec<(String, f32)> = picks
            .iter()
            .filter_map(|(m, t)| Some((m.clone(), (*t)?)))
            .collect();
        let clear: Vec<String> = picks
            .iter()
            .filter(|(_, t)| t.is_none())
            .map(|(m, _)| m.clone())
            .collect();
        changes = apply_model_thresholds(c, &set)?;
        changes.extend(clear_model_thresholds(c, &clear)?);
        Ok(())
    }) {
        warn!("benchmark threshold apply refused: {e:#}");
        return json_error(StatusCode::BAD_REQUEST, format!("{e:#}"));
    }
    let mut message = if changes.is_empty() {
        "No change: the models already use these thresholds.".to_string()
    } else {
        format!(
            "Saved {}: {}.",
            state.config_path.display(),
            changes
                .iter()
                .map(ThresholdChange::describe)
                .collect::<Vec<_>>()
                .join("; ")
        )
    };
    // Thresholds apply live (`save_config` swapped the workers' settings): no restart.
    let restarting = false;
    if !changes.is_empty() {
        message.push_str(" Applied now: the next detections use it.");
    }
    message.push(' ');
    message.push_str(BLUE_IRIS_THRESHOLD_NOTE);
    info!("benchmark threshold apply: {message}");
    // For Undo: POST `model=<m>&threshold=<previous>` (`global` = use the global threshold).
    let previous: serde_json::Map<String, serde_json::Value> = changes
        .iter()
        .map(|c| {
            (
                c.model.clone(),
                c.from
                    .map_or_else(|| "global".to_string(), |t| format!("{t:.2}"))
                    .into(),
            )
        })
        .collect();
    Json(serde_json::json!({
        "success": true,
        "message": message,
        "changes": changes,
        "previous": previous,
        "restarting": restarting,
    }))
    .into_response()
}

/// Clear the per-model `confidence_threshold` of the named models (they use the global one
/// again): the Undo of a threshold apply on a model that had none. Every name is checked
/// first; on error `config` is unchanged. Returns the entries that changed.
fn clear_model_thresholds(
    config: &mut Config,
    names: &[String],
) -> anyhow::Result<Vec<ThresholdChange>> {
    use crate::registry::normalize_name;
    let global = config.confidence_threshold;
    for n in names {
        if !config
            .models
            .iter()
            .any(|m| normalize_name(&m.effective_name()) == normalize_name(n))
        {
            anyhow::bail!("'{n}' is not one of the configured models");
        }
    }
    let mut changes = Vec::new();
    for n in names {
        let key = normalize_name(n);
        for m in config
            .models
            .iter_mut()
            .filter(|m| normalize_name(&m.effective_name()) == key)
        {
            if let Some(old) = m.confidence_threshold.take() {
                changes.push(ThresholdChange {
                    model: m.effective_name(),
                    from: Some(old),
                    to: global,
                    global,
                });
            }
        }
    }
    Ok(changes)
}

/// The (model, threshold) pairs of an apply-threshold form: `all=1` takes the saved best
/// thresholds of the configured models (limited to the form's models, if any), else the
/// `model` / `threshold` pairs (`threshold=global`: None, the model's own threshold is
/// cleared).
fn threshold_picks(
    state: &AppState,
    form: &[(String, String)],
) -> Result<Vec<(String, Option<f32>)>, ApiError> {
    let models = form_models(form);
    if form_field(form, "all").is_some_and(truthy) {
        let results = results_for_apply(state)?;
        let only = (!models.is_empty()).then_some(models.as_slice());
        let cfg = state.config_read().clone();
        let picks: Vec<(String, Option<f32>)> = results
            .thresholds(only)
            .into_iter()
            .filter(|(m, _)| crate::benchmark::find_model(&cfg, m).is_some())
            .map(|(m, t)| (m, Some(t)))
            .collect();
        if picks.is_empty() {
            return Err(ApiError::new(
                StatusCode::NOT_FOUND,
                "the saved results have no best threshold for these models",
            ));
        }
        return Ok(picks);
    }
    let values: Vec<&str> = form
        .iter()
        .filter(|(k, _)| k == "threshold")
        .map(|(_, v)| v.trim())
        .collect();
    if models.is_empty() || models.len() != values.len() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "give `model` and `threshold` (repeated in pairs), or `all=1` to apply every \
             best threshold",
        ));
    }
    let mut picks = Vec::new();
    for (m, t) in models.iter().zip(values) {
        if t.eq_ignore_ascii_case("global") {
            picks.push((m.clone(), None));
            continue;
        }
        match t.parse::<f32>() {
            Ok(v) => picks.push((m.clone(), Some(v))),
            Err(_) => {
                return Err(ApiError::new(
                    StatusCode::BAD_REQUEST,
                    format!("threshold: '{t}' is not a number"),
                ));
            }
        }
    }
    Ok(picks)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{call, get, post, state};
    use super::*;
    use crate::server::router;
    use std::time::Duration;
    use tower::ServiceExt;
    #[tokio::test]
    async fn apply_one_and_all() {
        let st = state();
        let cfg_path = st.config_path.clone();

        // `all` without results: 404.
        let (s, _) = call(&st, post("/v1/benchmark/apply", "all=1")).await;
        assert_eq!(s, StatusCode::NOT_FOUND);

        // One model.
        let (s, v) = call(
            &st,
            post("/v1/benchmark/apply", "model=DFINE-S&device=ort%3Acoreml"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["changes"][0]["model"], "dfine-s");
        assert_eq!(v["changes"][0]["to"], "ort:coreml");
        assert_eq!(v["restarting"], false);
        // dfine-s is disabled: nothing to reload or restart.
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .contains("Used when the model is enabled"),
            "{v}"
        );
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(saved.models[1].device.as_deref(), Some("ort:coreml"));
        assert_eq!(*st.config_read(), saved);

        // Invalid input changes nothing.
        for body in [
            "model=dfine-s&device=vulkan",
            "model=zzz&device=ort%3Acpu",
            "model=dfine-s",
            "",
        ] {
            let (s, _) = call(&st, post("/v1/benchmark/apply", body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(Config::load(&cfg_path).unwrap(), saved, "{body}");
        }

        // Every recommendation of the saved results; models no longer configured are skipped.
        let mk = |name: &str, rec: Option<&str>| {
            let mut m = crate::benchmark::ModelResult::failed(name, "", String::new());
            m.error = None;
            m.recommended = rec.map(str::to_string);
            m
        };
        let results = BenchmarkResults::new(
            Default::default(),
            Default::default(),
            vec![
                mk("ipcam-general", Some("openvino:cpu")),
                mk("dfine-s", Some("ort:coreml")),
                mk("removed", Some("ort:cpu")),
                mk("none", None),
            ],
        );
        results.save(&results_path(&cfg_path)).unwrap();
        let (s, v) = call(&st, post("/v1/benchmark/apply", "all=1&restart=1")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        // dfine-s already had ort:coreml: one change.
        assert_eq!(v["changes"].as_array().unwrap().len(), 1, "{v}");
        assert_eq!(v["changes"][0]["model"], "ipcam-general");
        assert_eq!(v["restarting"], true);
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(saved.models[0].device.as_deref(), Some("openvino:cpu"));
        tokio::time::timeout(Duration::from_secs(5), st.restart.cancelled())
            .await
            .expect("restart requested");

        // The config page shows the per-model Device select and the recommendation.
        let resp = router(st.clone()).oneshot(get("/config")).await.unwrap();
        let html = String::from_utf8(
            axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap();
        assert!(html.contains("name=\"device.ipcam-general\""), "{html}");
        assert!(html.contains("<option value=\"\">Global (auto"), "{html}");
        // Configured but not loaded on it (the test workers never load): restart to use.
        assert!(
            html.contains("benchmark: openvino:cpu (configured; restart to use)"),
            "{html}"
        );
        assert!(
            html.contains("benchmark: ort:coreml (configured; restart to use)"),
            "{html}"
        );
    }

    #[tokio::test]
    async fn apply_threshold_one_and_all() {
        let st = state();
        let cfg_path = st.config_path.clone();
        let (s, _) = call(&st, post("/v1/benchmark/apply-threshold", "all=1")).await;
        assert_eq!(s, StatusCode::NOT_FOUND);

        // One model.
        let (s, v) = call(
            &st,
            post(
                "/v1/benchmark/apply-threshold",
                "model=DFINE-S&threshold=0.35",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["changes"][0]["model"], "dfine-s");
        assert_eq!(v["changes"][0]["to"].as_f64().unwrap() as f32, 0.35);
        let msg = v["message"].as_str().unwrap();
        assert!(msg.contains("0.50 (global) -> 0.35"), "{msg}");
        assert!(msg.contains("min_confidence"), "{msg}");
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(saved.models[1].confidence_threshold, Some(0.35));
        assert_eq!(saved.models[0].confidence_threshold, None);
        assert_eq!(*st.config_read(), saved);
        // The status view shows the effective thresholds.
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(
            v["config"]["models"][1]["confidenceThreshold"]
                .as_f64()
                .unwrap() as f32,
            0.35
        );
        assert_eq!(v["config"]["models"][0]["ownThreshold"], false);

        // Invalid input changes nothing.
        for body in [
            "model=dfine-s&threshold=0",
            "model=dfine-s&threshold=1.5",
            "model=dfine-s&threshold=abc",
            "model=zzz&threshold=0.3",
            "model=dfine-s",
            "",
        ] {
            let (s, _) = call(&st, post("/v1/benchmark/apply-threshold", body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(Config::load(&cfg_path).unwrap(), saved, "{body}");
        }

        // Every best threshold of the saved results; unconfigured models are skipped, a model
        // without advice is left alone.
        let mk = |name: &str, t: Option<f32>| {
            let mut m = crate::benchmark::ModelResult::failed(name, "", String::new());
            m.error = None;
            m.threshold = t.map(crate::benchmark::threshold::test_advice);
            m
        };
        let results = BenchmarkResults::new(
            Default::default(),
            Default::default(),
            vec![
                mk("ipcam-general", Some(0.42)),
                mk("dfine-s", Some(0.35)),
                mk("removed", Some(0.3)),
                mk("none", None),
            ],
        );
        results.save(&results_path(&cfg_path)).unwrap();
        let (s, v) = call(
            &st,
            post("/v1/benchmark/apply-threshold", "all=1&restart=1"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["changes"].as_array().unwrap().len(), 1, "{v}");
        assert_eq!(v["changes"][0]["model"], "ipcam-general");
        // Thresholds apply live: no restart even when asked for.
        assert_eq!(v["restarting"], false);
        assert!(
            v["message"].as_str().unwrap().contains("Applied now"),
            "{v}"
        );
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(saved.models[0].confidence_threshold, Some(0.42));
        // Devices are untouched by the threshold apply.
        assert_eq!(saved.models[0].device, None);
        assert!(!st.restart.is_cancelled());
        // Several pairs at once (the search page's Apply all).
        let (s, v) = call(
            &st,
            post(
                "/v1/benchmark/apply-threshold",
                "model=ipcam-general&threshold=0.3&model=dfine-s&threshold=0.25",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["changes"].as_array().unwrap().len(), 2);
        let (s, _) = call(
            &st,
            post(
                "/v1/benchmark/apply-threshold",
                "model=ipcam-general&threshold=0.3&model=dfine-s",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let saved = Config::load(&cfg_path).unwrap();
        assert_eq!(saved.models[1].confidence_threshold, Some(0.25));
        // Back to the results' values for the check below.
        let (s, _) = call(&st, post("/v1/benchmark/apply-threshold", "all=1")).await;
        assert_eq!(s, StatusCode::OK);
        // Limited to one model: nothing left to change.
        let (s, v) = call(
            &st,
            post("/v1/benchmark/apply-threshold", "all=1&model=dfine-s"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert!(v["changes"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn run_tab_browses_for_a_folder() {
        let st = state();
        let (s, page) = call(&st, get("/benchmark")).await;
        assert_eq!(s, StatusCode::OK);
        let page = page.as_str().unwrap();
        assert!(page.contains("data-browse=\"folder\" data-target=\"add-folder\""));
        assert!(page.contains("<dialog id=\"browse-dialog\">"));
        assert!(page.contains("<script src=\"/static/browse.js?v="));
        // The low-priority box is a check label; device ids keep their line.
        assert!(
            page.contains("<label class=\"check\"><input type=\"checkbox\" name=\"low_priority\"")
        );
        assert!(page.contains("class=\"bench-settings\""));
    }

    #[tokio::test]
    async fn threshold_search_endpoint() {
        let st = state();
        let cfg_path = st.config_path.clone();
        // Nothing stored yet: the options GET answers 200 `available: false`, the search POST
        // 400; both tell to run a benchmark.
        for (req, status) in [
            (get("/v1/benchmark/threshold-search"), StatusCode::OK),
            (
                post("/v1/benchmark/threshold-search", "model=all"),
                StatusCode::BAD_REQUEST,
            ),
        ] {
            let (s, v) = call(&st, req).await;
            assert_eq!(s, status, "{v}");
            assert!(
                v["message"]
                    .as_str()
                    .unwrap()
                    .contains("run a benchmark first"),
                "{v}"
            );
            if status == StatusCode::OK {
                assert_eq!(v["available"], false, "{v}");
                assert!(v.get("options").is_none(), "{v}");
            }
        }
        let (results, preds) = crate::benchmark::search::tests::fixture();
        results.save(&results_path(&cfg_path)).unwrap();
        // Results but no stored predictions (an older version's results): still 400 / unavailable.
        let (s, _) = call(&st, post("/v1/benchmark/threshold-search", "")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, v) = call(&st, get("/v1/benchmark/threshold-search")).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["available"], false, "{v}");
        preds
            .save(&crate::benchmark::search::preds_path(&cfg_path))
            .unwrap();

        let (s, v) = call(&st, get("/v1/benchmark/threshold-search")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["available"], true, "{v}");
        assert_eq!(v["options"]["models"][0]["model"], "m");
        assert_eq!(
            v["options"]["models"][0]["devices"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(v["options"]["models"][0]["recommended"], "ort:coreml");
        assert_eq!(v["options"]["tags"], serde_json::json!(["day", "night"]));
        assert_eq!(v["objective"], "f1");

        // A valid search (default: every model, recommended device, f1).
        let (s, v) = call(&st, post("/v1/benchmark/threshold-search", "")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let r = &v["results"][0];
        assert_eq!(r["device"], "ort:coreml");
        assert_eq!(r["best"]["threshold"].as_f64().unwrap() as f32, 0.35);
        assert_eq!(r["grid"].as_array().unwrap().len(), 19);
        assert!(v["csv"].as_str().unwrap().starts_with("model,device"));
        assert!(v["elapsedMs"].as_f64().unwrap() >= 0.0);

        // Filters.
        let (s, v) = call(
            &st,
            post(
                "/v1/benchmark/threshold-search",
                "model=m&device=openvino%3Acpu&tag=night&class=person&objective=recall%3A0.5&iou=0.5",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let r = &v["results"][0];
        assert_eq!(r["device"], "openvino:cpu");
        assert_eq!((r["images"].as_u64(), r["gt"].as_u64()), (Some(1), Some(1)));
        assert_eq!(r["objective"], "recall:0.5");
        assert_eq!(r["classes"], serde_json::json!(["person"]));

        // Frame-level objective from stored data: Youden's J. Frames (real set): a person+ .85,
        // a car+ .4, b person+ .6, b car- (no car prediction: 0). J = 1 for thresholds up to
        // 0.4 -> plateau [0.05, 0.4], midpoint 0.22.
        let (s, v) = call(
            &st,
            post("/v1/benchmark/threshold-search", "objective=youden"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let r = &v["results"][0];
        assert_eq!(r["objective"], "youden");
        assert_eq!(r["best"]["threshold"].as_f64().unwrap() as f32, 0.22);
        assert_eq!(r["best"]["exact_threshold"].as_f64().unwrap() as f32, 0.4);
        assert_eq!(r["best"]["point"]["frame"]["tpr"], 1.0);
        assert_eq!(r["best"]["point"]["frame"]["fpr"], 0.0);
        assert_eq!(r["roc"]["auc"], 1.0);
        assert!(r["roc"]["curve"].as_array().unwrap().len() >= 2);
        assert!(v["csv"].as_str().unwrap().contains("roc_auc"));
        let (s, v) = call(
            &st,
            post("/v1/benchmark/threshold-search", "objective=fpr%3A0.05"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["results"][0]["objective"], "fpr:0.05");

        // Bad input / nothing matching: 400 with a message.
        for body in [
            "objective=f9",
            "iou=2",
            "model=zzz",
            "tag=fog",
            "device=ort%3Acpu",
        ] {
            let (s, v) = call(&st, post("/v1/benchmark/threshold-search", body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}: {v}");
            assert!(v["message"].is_string());
        }
    }
}
