//! Benchmark run history endpoints over `benchmark-history.json`
//! ([`crate::benchmark::history`]): `GET /v1/benchmark/history[?model=]` (entries newest
//! first, with their comparability to the newest), `GET /v1/benchmark/history/compare?model=&a=
//! &b=` (what changed from run `a` to run `b`), and the edits `POST /v1/benchmark/history/pin`
//! and `POST /v1/benchmark/history/delete`. Without a history file the history is seeded from
//! `benchmark.json` (in memory for reads, written by the first edit). Admin routes; file I/O
//! runs in `server::blocking`.

use crate::benchmark::history::{self, History, HistoryEntry, comparable, history_path};
use crate::benchmark::{BenchmarkResults, results_path};
use crate::server::AppState;
use axum::Json;
use axum::extract::rejection::{FormRejection, QueryRejection};
use axum::extract::{Form, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

/// `{"success": false, "error": ..., "message": ...}` with `status` (the history routes answer
/// `error`; `message` keeps the shape of the other benchmark endpoints).
fn fail(status: StatusCode, msg: impl Into<String>) -> Response {
    let msg = msg.into();
    (
        status,
        Json(serde_json::json!({ "success": false, "error": msg, "message": msg })),
    )
        .into_response()
}

/// The saved results (for seeding) and the history beside them.
fn load(state: &AppState) -> (Option<BenchmarkResults>, History, std::path::PathBuf) {
    let path = results_path(&state.config_path);
    let results = BenchmarkResults::load_or_warn(&path);
    let hist = history_path(&path);
    let h = History::load_or_seed(&hist, results.as_ref());
    (results, h, hist)
}

/// `{run_id, comparable, reasons}` of every entry (newest first) against the newest.
fn comparability(newest_first: &[&HistoryEntry]) -> Vec<serde_json::Value> {
    let Some(newest) = newest_first.first() else {
        return Vec::new();
    };
    newest_first
        .iter()
        .map(|e| {
            let reasons = if e.run_id == newest.run_id {
                Vec::new()
            } else {
                comparable(e, newest).err().unwrap_or_default()
            };
            serde_json::json!({
                "run_id": e.run_id,
                "comparable": reasons.is_empty(),
                "reasons": reasons,
            })
        })
        .collect()
}

/// `GET /v1/benchmark/history[?model=NAME]`: `models` = model -> entries newest first (only
/// NAME when given), `comparability` = model -> per entry whether it is comparable with the
/// model's newest entry, and why not.
pub(in crate::server) async fn history_get(
    State(state): State<Arc<AppState>>,
    query: Result<Query<HashMap<String, String>>, QueryRejection>,
) -> Response {
    let q = query.map(|Query(q)| q).unwrap_or_default();
    crate::server::blocking(move || {
        let (_, h, _) = load(&state);
        let only = q.get("model").map(|m| m.trim()).filter(|m| !m.is_empty());
        let key = crate::registry::normalize_name;
        let mut models = serde_json::Map::new();
        let mut comp = serde_json::Map::new();
        for (name, list) in &h.models {
            if only.is_some_and(|o| key(o) != key(name)) {
                continue;
            }
            let newest: Vec<&HistoryEntry> = list.iter().rev().collect();
            comp.insert(name.clone(), comparability(&newest).into());
            models.insert(
                name.clone(),
                serde_json::to_value(&newest).unwrap_or_default(),
            );
        }
        Json(serde_json::json!({
            "success": true,
            "models": models,
            "comparability": comp,
            "max_entries_per_model": history::MAX_ENTRIES_PER_MODEL,
        }))
        .into_response()
    })
    .await
}

/// `GET /v1/benchmark/history/compare?model=NAME&a=RUN_ID&b=RUN_ID`: what changed from run `a`
/// (before) to run `b` (after). 400 without the three, 404 for an unknown model or run.
pub(in crate::server) async fn history_compare(
    State(state): State<Arc<AppState>>,
    query: Result<Query<HashMap<String, String>>, QueryRejection>,
) -> Response {
    let q = query.map(|Query(q)| q).unwrap_or_default();
    crate::server::blocking(move || {
        let field = |k: &str| q.get(k).map(|v| v.trim()).filter(|v| !v.is_empty());
        let (Some(model), Some(a), Some(b)) = (field("model"), field("a"), field("b")) else {
            return fail(
                StatusCode::BAD_REQUEST,
                "give `model`, `a` (the earlier run id) and `b` (the later run id)",
            );
        };
        let (_, h, _) = load(&state);
        let missing = |id: &str| {
            fail(
                StatusCode::NOT_FOUND,
                format!(
                    "'{model}' has no run {id} in the benchmark history; reload the page (the \
                     run may have been deleted or dropped past the last {} runs)",
                    history::MAX_ENTRIES_PER_MODEL
                ),
            )
        };
        let (ea, eb) = match (h.find(model, a), h.find(model, b)) {
            (Some(x), Some(y)) => (x, y),
            (None, _) => return missing(a),
            (_, None) => return missing(b),
        };
        Json(serde_json::json!({
            "success": true,
            "changes": history::compare(ea, eb),
        }))
        .into_response()
    })
    .await
}

/// The form fields, or why the form does not parse.
fn form_of(
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Result<Vec<(String, String)>, String> {
    form.map(|Form(f)| f)
        .map_err(|e| format!("invalid form: {e}"))
}

fn field<'f>(form: &'f [(String, String)], key: &str) -> Option<&'f str> {
    form.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.trim())
        .filter(|v| !v.is_empty())
}

/// Save `edit` of the history (seeded from the saved results when there is no file yet).
fn edit_history(
    state: &AppState,
    what: &str,
    edit: impl FnOnce(&mut History) -> anyhow::Result<String>,
) -> Response {
    let path = results_path(&state.config_path);
    let results = BenchmarkResults::load_or_warn(&path);
    let hist = history_path(&path);
    match History::update(&hist, results.as_ref(), edit) {
        Ok((message, warnings)) => {
            info!("benchmark history {what}: {message}");
            Json(serde_json::json!({
                "success": true,
                "message": message,
                "warnings": warnings,
            }))
            .into_response()
        }
        Err(e) => {
            warn!("benchmark history {what} failed: {e:#}");
            fail(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!(
                    "The benchmark history was not changed: {e:#}. Check that {} is writable.",
                    hist.display()
                ),
            )
        }
    }
}

/// `POST /v1/benchmark/history/pin` (form `model`, `run_id`, `pinned=1|0`, default 1): pin a
/// run as the model's baseline (unpinning its other runs), or unpin it. 400 without `model` /
/// `run_id`, 404 for an unknown run.
pub(in crate::server) async fn history_pin(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form_of(form) {
        Ok(f) => f,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    crate::server::blocking(move || {
        let (Some(model), Some(run_id)) = (field(&form, "model"), field(&form, "run_id")) else {
            return fail(StatusCode::BAD_REQUEST, "give `model` and `run_id`");
        };
        let pinned = field(&form, "pinned").is_none_or(super::truthy);
        let (_, h, _) = load(&state);
        if h.find(model, run_id).is_none() {
            return fail(
                StatusCode::NOT_FOUND,
                format!("'{model}' has no run {run_id} in the benchmark history; reload the page"),
            );
        }
        let (model, run_id) = (model.to_string(), run_id.to_string());
        edit_history(&state, "pin", move |h| {
            h.pin(&model, &run_id, pinned)?;
            Ok(if pinned {
                format!(
                    "Run {run_id} of {model} is now its baseline: changes are shown against it."
                )
            } else {
                format!(
                    "Run {run_id} of {model} is no longer pinned: changes are shown against the \
                     previous comparable run."
                )
            })
        })
    })
    .await
}

/// `POST /v1/benchmark/history/delete` (form `model` + `run_id`: one run; `model` + `all=1`:
/// every run of the model; `all=1` alone: the whole history). 400 for another combination,
/// 404 for an unknown run.
pub(in crate::server) async fn history_delete(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form_of(form) {
        Ok(f) => f,
        Err(e) => return fail(StatusCode::BAD_REQUEST, e),
    };
    crate::server::blocking(move || {
        let model = field(&form, "model").map(str::to_string);
        let run_id = field(&form, "run_id").map(str::to_string);
        let all = field(&form, "all").is_some_and(super::truthy);
        match (model, run_id, all) {
            (Some(model), Some(run_id), false) => {
                let (_, h, _) = load(&state);
                if h.find(&model, &run_id).is_none() {
                    return fail(
                        StatusCode::NOT_FOUND,
                        format!(
                            "'{model}' has no run {run_id} in the benchmark history; reload the \
                             page"
                        ),
                    );
                }
                edit_history(&state, "delete", move |h| {
                    h.delete(&model, &run_id)?;
                    Ok(format!(
                        "Deleted run {run_id} of {model} from the benchmark history."
                    ))
                })
            }
            (Some(model), None, true) => edit_history(&state, "delete", move |h| {
                let n = h.delete_model(&model);
                Ok(format!(
                    "Deleted {n} run(s) of {model} from the benchmark history."
                ))
            }),
            (None, None, true) => edit_history(&state, "delete", |h| {
                let n = h.clear();
                Ok(format!(
                    "Deleted the whole benchmark history ({n} run(s)). The saved results are \
                     kept; the next run starts a new history."
                ))
            }),
            _ => fail(
                StatusCode::BAD_REQUEST,
                "give `model` and `run_id` (one run), `model` and `all=1` (every run of the \
                 model), or `all=1` (the whole history)",
            ),
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::super::tests::{call, get, post, state};
    use super::*;
    use crate::benchmark::report::test_support::{dev, model, scene};
    use axum::body::Body;
    use axum::http::{Request, header};
    use tower::ServiceExt;

    /// Saved results with model `ipcam-general` (OpenVINO CPU at `p50` ms, ONNX Runtime CPU at
    /// 80 ms) of run `run_id`.
    fn results_with(run_id: &str, p50: f64) -> BenchmarkResults {
        let mut m = model(
            vec![
                dev("openvino:cpu", p50, scene()),
                dev("ort:cpu", 80.0, scene()),
            ],
            None,
        );
        m.model = "ipcam-general".into();
        m.run_id = Some(run_id.into());
        BenchmarkResults::new(Default::default(), Default::default(), vec![m])
    }

    const T1: &str = "2026-10-16T14:00:00.000Z";
    const T2: &str = "2026-10-16T15:00:00.000Z";
    const T1_FORM: &str = "2026-10-16T14%3A00%3A00.000Z";

    #[tokio::test]
    async fn history_get_compare_pin_delete() {
        let st = state();
        let path = results_path(&st.config_path);
        let hist = history_path(&path);
        // Nothing saved: an empty history.
        let (s, v) = call(&st, get("/v1/benchmark/history")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["models"], serde_json::json!({}));

        // Saved results without a history file: seeded in memory, nothing written.
        let first = results_with(T1, 40.0);
        first.save(&path).unwrap();
        let (_, v) = call(&st, get("/v1/benchmark/history?model=IPcam-General")).await;
        let list = v["models"]["ipcam-general"].as_array().unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0]["run_id"], T1);
        assert!(!hist.exists());

        // A second run, recorded like a benchmark save does.
        let second = results_with(T2, 20.0);
        second.save(&path).unwrap();
        let w = crate::benchmark::history::record(
            &path,
            &crate::benchmark::history::seed_entries(&first),
            &second,
            &["ipcam-general".into()],
        );
        assert!(w.is_empty(), "{w:?}");
        let (_, v) = call(&st, get("/v1/benchmark/history")).await;
        let list = v["models"]["ipcam-general"].as_array().unwrap();
        assert_eq!(list.len(), 2);
        // Newest first; both comparable with the newest.
        assert_eq!(list[0]["run_id"], T2);
        assert_eq!(list[0]["devices"][0]["runtime"], "openvino");
        let comp = v["comparability"]["ipcam-general"].as_array().unwrap();
        assert!(comp.iter().all(|c| c["comparable"] == true), "{v}");
        let (_, v) = call(&st, get("/v1/benchmark/history?model=other")).await;
        assert_eq!(v["models"], serde_json::json!({}));

        // Compare: 50% faster on the CPU.
        let uri = format!("/v1/benchmark/history/compare?model=ipcam-general&a={T1}&b={T2}");
        let (s, v) = call(&st, get(&uri)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["changes"]["comparable"], true);
        assert_eq!(v["changes"]["metrics"][0]["metric"], "p50_ms");
        assert_eq!(v["changes"]["metrics"][0]["verdict"], "better");
        assert!(
            v["changes"]["summary"]
                .as_str()
                .unwrap()
                .starts_with("50% faster"),
            "{v}"
        );
        let uri = format!("/v1/benchmark/history/compare?model=ipcam-general&a=nope&b={T2}");
        let (s, v) = call(&st, get(&uri)).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert_eq!(v["success"], false);
        assert!(v["error"].as_str().unwrap().contains("nope"), "{v}");
        let (s, _) = call(
            &st,
            get("/v1/benchmark/history/compare?model=ipcam-general"),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);

        // Pin the first run: the status compares against it.
        let body = format!("model=ipcam-general&run_id={T1_FORM}&pinned=1");
        let (s, v) = call(&st, post("/v1/benchmark/history/pin", &body)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert!(v["message"].as_str().unwrap().contains("baseline"), "{v}");
        let (_, v) = call(&st, get("/v1/benchmark/history")).await;
        assert_eq!(v["models"]["ipcam-general"][1]["pinned"], true);
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        let changes = &v["results"]["models"][0]["changes"];
        assert_eq!(changes["baseline"]["pinned"], true, "{v}");
        assert_eq!(changes["baseline"]["run_id"], T1);
        let (s, _) = call(
            &st,
            post(
                "/v1/benchmark/history/pin",
                "model=ipcam-general&run_id=nope",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        let (s, _) = call(
            &st,
            post("/v1/benchmark/history/pin", "model=ipcam-general"),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);

        // Delete one run, then everything.
        let body = format!("model=ipcam-general&run_id={T1_FORM}");
        let (s, v) = call(&st, post("/v1/benchmark/history/delete", &body)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let (_, v) = call(&st, get("/v1/benchmark/history")).await;
        assert_eq!(v["models"]["ipcam-general"].as_array().unwrap().len(), 1);
        let (s, _) = call(
            &st,
            post(
                "/v1/benchmark/history/delete",
                "model=ipcam-general&run_id=nope",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        for bad in ["", "run_id=x", "model=ipcam-general"] {
            let (s, _) = call(&st, post("/v1/benchmark/history/delete", bad)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{bad}");
        }
        let (s, v) = call(&st, post("/v1/benchmark/history/delete", "all=1")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert!(v["message"].as_str().unwrap().contains("1 run(s)"), "{v}");
        let (_, v) = call(&st, get("/v1/benchmark/history")).await;
        assert_eq!(v["models"], serde_json::json!({}));
        // The empty file stays: the history is not seeded again.
        assert!(hist.exists());
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(
            v["results"]["models"][0]["changes"]["baseline"],
            serde_json::Value::Null
        );
    }

    #[tokio::test]
    async fn history_routes_refuse_remote_peers() {
        let st = state();
        let form = |uri: &str, body: &'static str| {
            Request::post(uri)
                .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
                .body(Body::from(body))
                .unwrap()
        };
        for req in [
            Request::get("/v1/benchmark/history")
                .body(Body::empty())
                .unwrap(),
            Request::get("/v1/benchmark/history/compare?model=a&a=1&b=2")
                .body(Body::empty())
                .unwrap(),
            form("/v1/benchmark/history/pin", "model=a&run_id=1"),
            form("/v1/benchmark/history/delete", "all=1"),
        ] {
            let uri = req.uri().to_string();
            let req = crate::server::test_util::from_peer(req, "10.0.0.5:40000");
            let resp = crate::server::router(st.clone())
                .oneshot(req)
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::FORBIDDEN, "{uri}");
        }
    }

    #[tokio::test]
    async fn status_carries_history_fields() {
        let st = state();
        let path = results_path(&st.config_path);
        results_with(T2, 20.0).save(&path).unwrap();
        let (s, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["latest_run_id"], T2);
        assert_eq!(v["noise"]["speed_abs_ms"], 0.5);
        assert_eq!(v["noise"]["se_factor"], 2.0);
        let m = &v["results"]["models"][0];
        assert_eq!(m["run_id"], T2);
        assert_eq!(m["this_run"], true);
        assert_eq!(m["trust"]["level"], "medium", "{m}");
        // Seeded from these results: the first recorded run.
        assert_eq!(m["changes"]["baseline"], serde_json::Value::Null);
        assert_eq!(m["changes"]["comparable"], false);
        assert!(m["changes"]["summary"].is_string());
        assert_eq!(m["devices"][0]["runtime"], "openvino");
        assert_eq!(m["devices"][1]["runtime"], "onnxruntime");
        assert!(m["devices"][1]["trust"]["reasons"].is_array());
        let r = &v["ranking"][0];
        assert_eq!(r["device"], "openvino:cpu");
        assert_eq!(r["runtime"], "openvino");
        assert_eq!(r["best_tradeoff"], true);
        assert_eq!(r["this_run"], true);
        assert_eq!(r["trust"]["level"], "medium");
        assert!(r.get("recall").is_some() && r.get("threshold").is_some());
        let points = v["points"].as_array().unwrap();
        assert_eq!(points.len(), 2);
        assert_eq!(points[1]["best_tradeoff"], false);
        // A new history file changes the status (the cache key has its stamp).
        let older = results_with(T1, 40.0);
        let mut h = History::default();
        h.append(HistoryEntry::of(&older.models[0], &older));
        h.save(&history_path(&path)).unwrap();
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        let c = &v["results"]["models"][0]["changes"];
        assert_eq!(c["baseline"]["run_id"], T1, "{c}");
        assert_eq!(c["comparable"], true);
    }

    #[tokio::test]
    async fn settings_rescore_the_saved_results() {
        use crate::benchmark::report::test_support::with_sweep;
        let st = state();
        let path = results_path(&st.config_path);
        let mut r = results_with(T2, 20.0);
        let curve = [(0.9, true), (0.6, true), (0.4, true), (0.3, false)];
        let m = &mut r.models[0];
        m.devices = vec![with_sweep(dev("openvino:cpu", 20.0, scene()), 3, &curve)];
        m.finish();
        r.save(&path).unwrap();
        // The objective of the results: nothing to re-rank.
        let (s, v) = call(
            &st,
            post(
                "/v1/benchmark/settings",
                "dataset=sample&threshold_objective=f1",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["regraded"], false);
        // Another objective: the saved advice follows it (no stored predictions: the grid).
        let (s, v) = call(
            &st,
            post(
                "/v1/benchmark/settings",
                "dataset=sample&threshold_objective=f2",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["regraded"], true, "{v}");
        assert!(v["message"].as_str().unwrap().contains("re-ranked"), "{v}");
        let saved = BenchmarkResults::load(&path).unwrap().unwrap();
        assert_eq!(saved.models[0].threshold.as_ref().unwrap().objective, "f2");
        let (_, v) = call(&st, get("/v1/benchmark")).await;
        assert_eq!(v["results"]["models"][0]["threshold"]["objective"], "f2");
    }

    #[tokio::test]
    async fn apply_answers_previous_values_for_undo() {
        let st = state();
        let cfg = || crate::config::Config::load(&st.config_path).unwrap();
        let apply = |body: &str| post("/v1/benchmark/apply", body);
        let thr = |body: &str| post("/v1/benchmark/apply-threshold", body);
        // Device: from the global device; Undo posts `global`.
        let (s, v) = call(&st, apply("model=dfine-s&device=ort%3Acpu")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["previous"]["dfine-s"], "global");
        let (s, v) = call(&st, apply("model=dfine-s&device=global")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["previous"]["dfine-s"], "ort:cpu");
        assert_eq!(cfg().models[1].device, None);
        // Threshold: from the global one; Undo posts `global` and clears the model's own.
        let (s, v) = call(&st, thr("model=dfine-s&threshold=0.4")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["previous"]["dfine-s"], "global");
        let (s, v) = call(&st, thr("model=dfine-s&threshold=0.3")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["previous"]["dfine-s"], "0.40");
        let (s, v) = call(&st, thr("model=dfine-s&threshold=0.40")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["previous"]["dfine-s"], "0.30");
        let (s, v) = call(&st, thr("model=dfine-s&threshold=global")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["previous"]["dfine-s"], "0.40");
        assert_eq!(v["changes"][0]["to"].as_f64().unwrap() as f32, 0.5);
        assert_eq!(cfg().models[1].confidence_threshold, None);
        // Nothing to clear: no change; an unknown model: 400.
        let (s, v) = call(&st, thr("model=dfine-s&threshold=global")).await;
        assert_eq!(s, StatusCode::OK);
        assert!(v["changes"].as_array().unwrap().is_empty());
        let (s, _) = call(&st, thr("model=zzz&threshold=global")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
    }
}
