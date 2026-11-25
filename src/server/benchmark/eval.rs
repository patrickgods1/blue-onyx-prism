//! The evaluation core's endpoints ([`crate::benchmark::eval`]), JSON in camelCase, signed in
//! (admin routes), computed off the runtime threads ([`crate::server::blocking`]):
//!
//! - `GET /v1/benchmark/compare?set=&scene=&goal=&tune_for=&classes=&budget=`: every combo on
//!   one image set judged for the goal ([`crate::benchmark::eval::compare`]); `goal=` etc. try a
//!   goal without saving it;
//! - `GET /v1/benchmark/tune?model=&device=&set=&goal=&tune_for=&classes=`: one combo's 0.01
//!   grid, suggested pick, interval, targets and the saved value
//!   ([`crate::benchmark::eval::tune`]);
//! - `GET /v1/benchmark/tune/images?model=&device=&set=&threshold=&kind=miss|fa|all&camera=
//!   &scene=&offset=&limit=`: the frames for the Tune strips and the Images page, worst first;
//! - `POST /v1/benchmark/goal` (form: `optimize_for` (or `goal`), `tune_for`, `classes`,
//!   `speed_budget_ms` (or `budget`); absent = unchanged, `classes=all` / `budget=none` clear):
//!   validates and saves `benchmark.goal` through `save_config`, answers the new ranking.
//!
//! Answers are cached in memory ([`VIEWS`], at most [`CACHE_ENTRIES`], each dropped after
//! [`crate::benchmark::search::CACHE_IDLE`] unread) per (results, stored predictions and labels
//! snapshot file stamps, config revision, goal, set, scene, ...), so a page switching tabs does
//! not recompute.

use super::{ApiError, json_fail};
use crate::benchmark::combos::{Inputs, Parts};
use crate::benchmark::current::ModelNow;
use crate::benchmark::eval::{Goal, compare, tune};
use crate::benchmark::search::file_stamp;
use crate::benchmark::search::{CACHE_IDLE, IdleCache, StoredPreds, preds_path};
use crate::benchmark::{BenchmarkResults, results_path};
use crate::config::Config;
use crate::server::{AppState, save_config};
use axum::Json;
use axum::extract::rejection::{FormRejection, QueryRejection};
use axum::extract::{Form, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use std::sync::Arc;

/// Answers kept at most (the least recently read goes first).
pub(super) const CACHE_ENTRIES: usize = 16;
/// Example frames per answer by default, and at most.
const IMAGES_DEFAULT: usize = 24;
const IMAGES_MAX: usize = 200;

/// The computed answers (see the module docs): at most [`CACHE_ENTRIES`], each dropped by the
/// cache's own thread after [`CACHE_IDLE`] unread, so a closed page leaves nothing behind
/// (RA-098).
type ViewCache = IdleCache<String, Arc<serde_json::Value>>;
static VIEWS: ViewCache = IdleCache::with_capacity(CACHE_IDLE, CACHE_ENTRIES);

/// The saved results seen through the config now, the stored predictions and what the combos
/// reads of the config.
pub(super) struct Loaded {
    pub(super) results: BenchmarkResults,
    pub(super) preds: Arc<StoredPreds>,
    pub(super) cfg: Config,
    pub(super) now: HashMap<String, ModelNow>,
    pub(super) parts: Parts,
}

/// What the config and the registry say about each model now ([`super::models_now`]: the
/// threshold each loaded worker runs with, the device the config resolves to).
fn now_of(state: &AppState, cfg: &Config) -> HashMap<String, ModelNow> {
    let snap = crate::server::devices_snapshot(state);
    super::models_now(state, cfg, &snap)
}

/// The file stamps, the config revision and the models' state now (sorted: a reload, a
/// fallback to another device or a new running threshold changes it, as for `saved_view`)
/// the answers depend on.
fn stamp_of(state: &AppState, revision: u64, now: &HashMap<String, ModelNow>) -> String {
    let path = results_path(&state.config_path);
    let preds = preds_path(&state.config_path);
    let gt = crate::benchmark::cameras::snapshot::path_beside(&path);
    let mut sorted: Vec<_> = now.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    format!(
        "{:?}|{:?}|{:?}|{revision}|{sorted:?}",
        file_stamp(&path),
        file_stamp(&preds),
        file_stamp(&gt)
    )
}

/// Load what the views read, seen through `cfg` and the models' state `now` (the snapshot the
/// caller keyed its answer by, [`stamp_of`]: the key and the value come from one state); 404
/// without results or stored predictions.
pub(super) fn load(
    state: &AppState,
    cfg: Config,
    now: HashMap<String, ModelNow>,
) -> Result<Loaded, ApiError> {
    let mut results = BenchmarkResults::load_or_warn(&results_path(&state.config_path))
        .ok_or_else(|| ApiError::new(StatusCode::NOT_FOUND, "no benchmark results yet"))?;
    let preds = StoredPreds::load_cached(&preds_path(&state.config_path))
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?
        .filter(|p| !p.runs.is_empty())
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::NOT_FOUND,
                "no stored predictions: run a benchmark first",
            )
        })?;
    crate::benchmark::current::rebase(&mut results, Some(&preds), &now);
    crate::benchmark::trained_on::backfill(&mut results, &cfg);
    let parts = Parts::of(&results, &cfg);
    Ok(Loaded {
        results,
        preds,
        cfg,
        now,
        parts,
    })
}

impl Loaded {
    /// The compare view of `goal` on `set` / `scene`.
    pub(super) fn compare(
        &self,
        goal: &Goal,
        set: Option<&str>,
        scene: Option<&str>,
    ) -> compare::CompareView {
        let default_model = crate::training::switch::current_default(&self.cfg);
        let inputs = Inputs {
            results: &self.results,
            preds: Some(&self.preds),
            now: &self.now,
            default_model: default_model.as_deref(),
            configured: &self.parts.configured,
            trained: &self.parts.trained,
            seen: &self.parts.seen,
            reachable: &|s| s.dir.as_deref().is_some_and(std::path::Path::is_dir),
        };
        compare::build(
            &inputs,
            compare::Request {
                goal,
                weights: self.cfg.benchmark.weights,
                set,
                scene,
            },
        )
    }
}

/// A trimmed, non-empty query value.
fn q<'a>(m: &'a HashMap<String, String>, k: &str) -> Option<&'a str> {
    m.get(k).map(|s| s.trim()).filter(|s| !s.is_empty())
}

/// The saved goal with what the query tries.
fn goal_of(cfg: &Config, m: &HashMap<String, String>) -> Result<Goal, ApiError> {
    cfg.benchmark
        .goal
        .try_with(
            q(m, "goal").or_else(|| q(m, "optimize_for")),
            q(m, "tune_for"),
            q(m, "classes"),
            q(m, "budget").or_else(|| q(m, "speed_budget_ms")),
        )
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, e))
}

/// The cached answer under `key`, else `make()`'s (cached when it succeeded).
fn cached(
    key: String,
    make: impl FnOnce() -> Result<serde_json::Value, ApiError>,
) -> Result<Arc<serde_json::Value>, ApiError> {
    cached_in(&VIEWS, key, make)
}

/// [`cached`] in `cache` (tests).
fn cached_in(
    cache: &'static ViewCache,
    key: String,
    make: impl FnOnce() -> Result<serde_json::Value, ApiError>,
) -> Result<Arc<serde_json::Value>, ApiError> {
    if let Some(v) = cache.get(&key) {
        return Ok(v);
    }
    let v = Arc::new(make()?);
    cache.put(key, v.clone());
    Ok(v)
}

fn answer(r: Result<Arc<serde_json::Value>, ApiError>) -> Response {
    match r {
        Ok(v) => Json((*v).clone()).into_response(),
        Err(e) => e.into_response(),
    }
}

type QueryMap = Result<Query<HashMap<String, String>>, QueryRejection>;

fn query_map(query: QueryMap) -> Result<HashMap<String, String>, Box<Response>> {
    query.map(|Query(q)| q).map_err(|e| {
        Box::new(json_fail(
            StatusCode::BAD_REQUEST,
            format!("invalid query: {e}"),
        ))
    })
}

/// `GET /v1/benchmark/compare` (see the module docs).
pub(in crate::server) async fn compare_route(
    State(state): State<Arc<AppState>>,
    query: QueryMap,
) -> Response {
    let m = match query_map(query) {
        Ok(m) => m,
        Err(r) => return *r,
    };
    crate::server::blocking(move || answer(compare_blocking(&state, &m))).await
}

fn compare_blocking(
    state: &AppState,
    m: &HashMap<String, String>,
) -> Result<Arc<serde_json::Value>, ApiError> {
    let (revision, cfg) = state.config.snapshot();
    let goal = goal_of(&cfg, m)?;
    let set = q(m, "set").map(str::to_string);
    let scene = q(m, "scene").filter(|s| *s != "all").map(str::to_string);
    if let Some(s) = &scene
        && !compare::SCENES.contains(&s.as_str())
    {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("scene '{s}': use all, day, night or ir"),
        ));
    }
    let now = now_of(state, &cfg);
    let key = format!(
        "compare|{}|{}|{set:?}|{scene:?}",
        stamp_of(state, revision, &now),
        goal.key()
    );
    cached(key, || {
        let l = load(state, cfg, now)?;
        let view = l.compare(&goal, set.as_deref(), scene.as_deref());
        Ok(serde_json::json!({ "success": true, "compare": view }))
    })
}

/// The saved confidence of `model` on `device` and its source.
fn saved_of(cfg: &Config, model: &str, device: &str) -> Option<(f32, String)> {
    let m = crate::benchmark::variant::owner(cfg, model)?;
    let (t, src) = m.threshold_on(Some(device), cfg.confidence_threshold);
    Some((t, src.label()))
}

/// `GET /v1/benchmark/tune` (see the module docs).
pub(in crate::server) async fn tune_route(
    State(state): State<Arc<AppState>>,
    query: QueryMap,
) -> Response {
    let m = match query_map(query) {
        Ok(m) => m,
        Err(r) => return *r,
    };
    crate::server::blocking(move || answer(tune_blocking(&state, &m))).await
}

fn tune_blocking(
    state: &AppState,
    m: &HashMap<String, String>,
) -> Result<Arc<serde_json::Value>, ApiError> {
    let (revision, cfg) = state.config.snapshot();
    let model = q(m, "model")
        .ok_or_else(|| ApiError::new(StatusCode::BAD_REQUEST, "give `model`"))?
        .to_string();
    let device = q(m, "device").map(str::to_string);
    let set = q(m, "set").map(str::to_string);
    let goal = goal_of(&cfg, m)?;
    let now = now_of(state, &cfg);
    let key = format!(
        "tune|{}|{}|{}|{device:?}|{set:?}",
        stamp_of(state, revision, &now),
        goal.key(),
        tune::key_of(&model)
    );
    cached(key, || {
        let l = load(state, cfg, now)?;
        let probe = tune::build(
            &l.results,
            &l.preds,
            tune::Request {
                model: &model,
                device: device.as_deref(),
                set: set.as_deref(),
                goal: &goal,
                weights: l.cfg.benchmark.weights,
                saved: None,
            },
        )
        .map_err(|e| ApiError::new(StatusCode::NOT_FOUND, format!("{e:#}")))?;
        // The saved value of the device the view chose.
        let mut view = probe;
        view.saved = saved_of(&l.cfg, &view.model, &view.device).map(|(t, s)| tune::Saved {
            threshold: t,
            source: s,
        });
        Ok(serde_json::json!({ "success": true, "tune": view }))
    })
}

/// `GET /v1/benchmark/tune/images` (see the module docs).
pub(in crate::server) async fn tune_images_route(
    State(state): State<Arc<AppState>>,
    query: QueryMap,
) -> Response {
    let m = match query_map(query) {
        Ok(m) => m,
        Err(r) => return *r,
    };
    crate::server::blocking(move || answer(tune_images_blocking(&state, &m))).await
}

fn tune_images_blocking(
    state: &AppState,
    m: &HashMap<String, String>,
) -> Result<Arc<serde_json::Value>, ApiError> {
    let bad = |s: String| ApiError::new(StatusCode::BAD_REQUEST, s);
    let (revision, cfg) = state.config.snapshot();
    let model = q(m, "model")
        .ok_or_else(|| bad("give `model`".into()))?
        .to_string();
    let device = q(m, "device").map(str::to_string);
    let set = q(m, "set").map(str::to_string);
    let threshold: f32 = q(m, "threshold")
        .ok_or_else(|| bad("give `threshold` (0..1)".into()))?
        .parse()
        .map_err(|_| bad("threshold: not a number".into()))?;
    if !(threshold.is_finite() && (0.0..=1.0).contains(&threshold)) {
        return Err(bad("threshold: must be in 0..1".into()));
    }
    let kind: tune::Kind = q(m, "kind").unwrap_or("miss").parse().map_err(bad)?;
    let limit = match q(m, "limit") {
        Some(v) => v
            .parse::<usize>()
            .map_err(|_| bad("limit: not a whole number".into()))?
            .clamp(1, IMAGES_MAX),
        None => IMAGES_DEFAULT,
    };
    let offset = match q(m, "offset") {
        Some(v) => v
            .parse::<usize>()
            .map_err(|_| bad("offset: not a whole number".into()))?,
        None => 0,
    };
    let camera = q(m, "camera").filter(|c| *c != "all").map(str::to_string);
    let scene = q(m, "scene").filter(|s| *s != "all").map(str::to_string);
    if let Some(s) = &scene
        && !compare::SCENES.contains(&s.as_str())
    {
        return Err(bad(format!("scene '{s}': use all, day, night or ir")));
    }
    let goal = goal_of(&cfg, m)?;
    let now = now_of(state, &cfg);
    let key = format!(
        "images|{}|{}|{}|{device:?}|{set:?}|{threshold}|{kind:?}|{limit}|{offset}|{camera:?}|{scene:?}",
        stamp_of(state, revision, &now),
        goal.key(),
        tune::key_of(&model)
    );
    cached(key, || {
        let l = load(state, cfg, now)?;
        let ex = tune::images(
            &l.results,
            &l.preds,
            tune::ImagesRequest {
                model: &model,
                device: device.as_deref(),
                set: set.as_deref(),
                goal: &goal,
                threshold,
                kind,
                camera: camera.as_deref(),
                scene: scene.as_deref(),
                offset,
                limit,
            },
        )
        .map_err(|e| ApiError::new(StatusCode::NOT_FOUND, format!("{e:#}")))?;
        Ok(serde_json::json!({ "success": true, "images": ex }))
    })
}

/// A row of the ranking summary of `POST /v1/benchmark/goal`.
fn ranking_summary(view: &compare::CompareView) -> serde_json::Value {
    let rows: Vec<serde_json::Value> = view
        .rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "model": r.model,
                "device": r.device,
                "rank": r.rank,
                "position": r.position,
                "block": r.block,
                "metric": r.metric,
                "confidence": r.confidence,
            })
        })
        .collect();
    serde_json::json!({
        "set": view.set.as_ref().map(|s| s.id.clone()),
        "headline": view.headline,
        "rows": rows,
    })
}

/// `POST /v1/benchmark/goal` (see the module docs).
pub(in crate::server) async fn goal_route(
    State(state): State<Arc<AppState>>,
    form: Result<Form<Vec<(String, String)>>, FormRejection>,
) -> Response {
    let form = match form {
        Ok(Form(f)) => f,
        Err(e) => return json_fail(StatusCode::BAD_REQUEST, format!("invalid form: {e}")),
    };
    crate::server::blocking(move || goal_blocking(&state, &form)).await
}

fn goal_blocking(state: &AppState, form: &[(String, String)]) -> Response {
    let m: HashMap<String, String> = form.iter().cloned().collect();
    let cfg = state.config_read().clone();
    let goal = match goal_of(&cfg, &m) {
        Ok(g) => g,
        Err(e) => return e.into_response(),
    };
    let saved = goal.clone();
    let message = match save_config(state, "benchmark goal", move |c| {
        c.benchmark.set_goal(saved);
        Ok(())
    }) {
        Ok(msg) => msg,
        Err(e) => return json_fail(StatusCode::BAD_REQUEST, format!("{e:#}")),
    };
    // The config the save just published.
    let (_, cfg) = state.config.snapshot();
    let now = now_of(state, &cfg);
    let ranking = load(state, cfg, now)
        .ok()
        .map(|l| ranking_summary(&l.compare(&goal, None, None)));
    Json(serde_json::json!({
        "success": true,
        "message": message,
        "goal": compare::GoalView::of(&goal),
        "ranking": ranking,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::super::tests::{call, get, post, state};
    use super::*;

    /// The synthetic results of the evaluation tests beside the state's config.
    fn with_results(st: &AppState) {
        let (r, p) = crate::benchmark::eval::tests::fixture(10, 10);
        r.save(&results_path(&st.config_path)).unwrap();
        p.save(&preds_path(&st.config_path)).unwrap();
    }

    /// R4: a model reload on another device, or a new running threshold, is another cache key
    /// for the Compare and Tune views (files and config unchanged).
    #[test]
    fn the_view_stamp_follows_the_models_now() {
        let st = state();
        let one = |running: Option<f32>, device: &str| {
            let mut now = HashMap::new();
            now.insert(
                "ipcam-general".to_string(),
                ModelNow::new(0.5, running, Some(device.into())),
            );
            now.insert(
                "dfine-s".to_string(),
                ModelNow::new(0.4, None, Some("ort:cpu".into())),
            );
            now
        };
        let base = stamp_of(&st, 7, &one(Some(0.5), "openvino:gpu"));
        assert_eq!(base, stamp_of(&st, 7, &one(Some(0.5), "openvino:gpu")));
        assert_ne!(
            base,
            stamp_of(&st, 7, &one(Some(0.5), "openvino:cpu")),
            "fallback"
        );
        assert_ne!(
            base,
            stamp_of(&st, 7, &one(Some(0.35), "openvino:gpu")),
            "threshold"
        );
        assert_ne!(
            base,
            stamp_of(&st, 7, &one(None, "openvino:gpu")),
            "unloaded"
        );
    }

    #[tokio::test]
    async fn compare_tune_images_and_goal_answer_their_shapes() {
        let st = state();
        // Nothing yet: 404 with a message.
        let (s, v) = call(&st, get("/v1/benchmark/compare")).await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{v}");
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .contains("no benchmark results")
        );
        with_results(&st);
        let (s, v) = call(&st, get("/v1/benchmark/compare")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let rows = v["compare"]["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0]["model"], "ipcam-general");
        assert_eq!(rows[0]["rank"], 1);
        assert!(rows[0]["metric"]["low"].is_number());
        assert_eq!(v["compare"]["set"]["id"], "own");
        // Trying a goal does not save it.
        let (s, v) = call(
            &st,
            get("/v1/benchmark/compare?goal=ap50&budget=100&scene=day"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["compare"]["goal"]["metric"], "AP50");
        assert_eq!(v["compare"]["set"]["scene"], "day");
        assert_eq!(v["compare"]["rows"][2]["flags"]["tooSlow"], true);
        assert_eq!(
            st.config_read().benchmark.goal,
            crate::benchmark::eval::Goal::default()
        );
        for bad in ["goal=f3", "scene=dusk", "budget=-2", "tune_for=ap50"] {
            let (s, v) = call(&st, get(&format!("/v1/benchmark/compare?{bad}"))).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{bad}: {v}");
        }
        // Tune.
        let (s, v) = call(&st, get("/v1/benchmark/tune?model=dfine-s")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let t = &v["tune"];
        assert_eq!(t["grid"].as_array().unwrap().len(), 96);
        assert_eq!(t["device"], "ort:coreml");
        assert_eq!(t["saved"]["source"], "global");
        assert!(t["suggested"]["threshold"].is_number());
        assert_eq!(t["savable"], true);
        let (s, _) = call(&st, get("/v1/benchmark/tune")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _) = call(&st, get("/v1/benchmark/tune?model=nobody")).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        // Example frames.
        let (s, v) = call(
            &st,
            get("/v1/benchmark/tune/images?model=dfine-s&threshold=0.5&kind=fa&limit=3"),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let frames = v["images"]["frames"].as_array().unwrap();
        assert_eq!(frames.len(), 3);
        assert!(frames[0]["preds"][0]["bbox"].is_array());
        for bad in [
            "threshold=2",
            "threshold=x",
            "kind=both&threshold=0.5",
            "limit=x&threshold=0.5",
        ] {
            let (s, _) = call(
                &st,
                get(&format!("/v1/benchmark/tune/images?model=dfine-s&{bad}")),
            )
            .await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{bad}");
        }
        // Saving a goal: validated, written with the legacy keys, the new ranking answered.
        let (s, v) = call(
            &st,
            post("/v1/benchmark/goal", "optimize_for=recall&tune_for=f1"),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        let (s, v) = call(
            &st,
            post(
                "/v1/benchmark/goal",
                "optimize_for=f2&classes=person&speed_budget_ms=250",
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["goal"]["metric"], "F2");
        assert_eq!(v["ranking"]["rows"][0]["model"], "ipcam-general");
        let b = st.config_read().benchmark.clone();
        assert_eq!(b.goal.classes, ["person"]);
        assert_eq!(b.goal.speed_budget_ms, Some(250.0));
        assert_eq!(
            b.threshold_objective,
            crate::benchmark::threshold::Objective::F2
        );
        let text = std::fs::read_to_string(&st.config_path).unwrap();
        assert!(
            text.contains(r#""optimize_for": "f2""#) || text.contains(r#""optimize_for":"f2""#),
            "{text}"
        );
        // The saved goal is the compare default now; `none` clears the budget.
        let (_, v) = call(&st, get("/v1/benchmark/compare")).await;
        assert_eq!(v["compare"]["goal"]["metric"], "F2");
        let (s, _) = call(&st, post("/v1/benchmark/goal", "budget=none")).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(st.config_read().benchmark.goal.speed_budget_ms, None);
        assert_eq!(st.config_read().benchmark.goal.classes, ["person"]);
    }

    /// The answers are bounded, keyed, and dropped once unread for the idle time (RA-098).
    #[test]
    fn the_cache_is_bounded_keyed_and_idle_dropped() {
        let c: &'static ViewCache = Box::leak(Box::new(IdleCache::with_capacity(
            CACHE_IDLE,
            CACHE_ENTRIES,
        )));
        for i in 0..(CACHE_ENTRIES + 5) {
            c.put(format!("k{i}"), Arc::new(serde_json::json!(i)));
        }
        assert_eq!(c.len(), CACHE_ENTRIES);
        assert!(c.get(&"k0".to_string()).is_none(), "the oldest went first");
        assert_eq!(
            c.get(&format!("k{}", CACHE_ENTRIES + 4)).as_deref(),
            Some(&serde_json::json!(CACHE_ENTRIES + 4))
        );
        c.put("k20".into(), Arc::new(serde_json::json!("new")));
        assert_eq!(
            c.get(&"k20".to_string()).as_deref(),
            Some(&serde_json::json!("new"))
        );
        assert_eq!(c.len(), CACHE_ENTRIES);
        // Computed once while it is read, then gone once nobody asked for the idle time.
        let made = std::cell::Cell::new(0);
        let make = || {
            made.set(made.get() + 1);
            Ok(serde_json::json!("answer"))
        };
        let a = cached_in(c, "view".into(), make).unwrap();
        let b = cached_in(c, "view".into(), make).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(made.get(), 1);
        let later = std::time::Instant::now() + CACHE_IDLE;
        assert!(!c.sweep_at(later), "every entry idle: the sweeper ends");
        assert!(c.is_empty());
        assert_eq!(Arc::strong_count(&a), 2, "only the two handles here");
        cached_in(c, "view".into(), make).unwrap();
        assert_eq!(made.get(), 2, "computed again");
    }
}
