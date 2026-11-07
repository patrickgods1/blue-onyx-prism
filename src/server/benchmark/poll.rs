//! Polling `GET /v1/benchmark` cheaply ([`answer`]).
//!
//! The answer is two parts: the run (state, progress, this run's models), which changes every
//! second during a run, and the rest (saved results, lineup, profiles, config view), which
//! changes when a model's results are saved or the config, a worker or the saved files change.
//! The rest has a tag (`viewTag`): the [`super::SAVED_CACHE`] key (the files' stamps, the config
//! revision and what it says about each model), the registry generation and the config epoch,
//! and the bytes of the small parts built per request (config view, profiles, goal). So a tag
//! that matches means the same bytes, never a stale view the cache would not also serve.
//!
//! - `GET /v1/benchmark?only=run`: `{success, state, running, run, viewTag}`, what the page
//!   polls during a run; it fetches the full answer when `viewTag` changes or the run ends.
//! - `GET /v1/benchmark`: the full answer, unchanged, with `ETag: W/"<viewTag>-<run hash>"`
//!   (`If-None-Match` that matches = 304, no body) and `X-Benchmark-View: <viewTag>`.
//!
//! The saved trees are cached serialized ([`super::SavedView`]), so neither answer copies or
//! re-serializes them; the body is serialized here, inside `blocking`.

use super::{
    AppState, LIVE_WARNING, config_view, configured_device, devices_snapshot, loaded_devices,
    models_now, profiles, raw, results_path, saved_view,
};
use crate::benchmark::service::{RunState, Status};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::value::RawValue;
use std::hash::{Hash, Hasher};

/// The `X-Benchmark-View` header of the full answer: its `viewTag`.
pub(super) const VIEW_HEADER: &str = "x-benchmark-view";

/// What a `GET /v1/benchmark` query asks for.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Ask {
    /// `only=run`: the run alone (any other query: the full answer, as before).
    pub only_run: bool,
    /// `models=<modelsTag>` (with `only=run`): the run's models the page has; when they are
    /// still these, the answer leaves `run.models` out.
    pub models: Option<String>,
}

pub(super) fn ask(query: Option<&str>) -> Ask {
    let mut a = Ask::default();
    for pair in query.unwrap_or_default().split('&') {
        match pair.split_once('=') {
            Some(("only", "run")) => a.only_run = true,
            Some(("models", t)) if !t.is_empty() => a.models = Some(t.to_string()),
            _ => {}
        }
    }
    a
}

/// The run's models as last serialized: the models as the service had them (before the
/// rebase), the view tag (which covers what the rebase reads, the config now), their JSON and
/// its tag. Their progress ticks every second, they change once per device run, so most polls
/// of a run reuse the JSON (equal models, compared, never guessed). Kept only while a run goes.
struct RunModels {
    view_tag: String,
    models: Vec<crate::benchmark::ModelResult>,
    json: std::sync::Arc<RawValue>,
    tag: String,
}

type RunModelsSlot = std::sync::Mutex<Option<RunModels>>;

/// The service's slot (one benchmark per process).
static RUN_MODELS: RunModelsSlot = std::sync::Mutex::new(None);

/// `models` shown against the config now (as the saved results are), serialized, and the tag
/// of that JSON; `slot` keeps the last one ([`RunModels`]).
fn models_json(
    slot: &RunModelsSlot,
    view_tag: &str,
    models: Vec<crate::benchmark::ModelResult>,
    now: &std::collections::HashMap<String, crate::benchmark::current::ModelNow>,
    running: bool,
) -> (std::sync::Arc<RawValue>, String) {
    use crate::sync::LockExt;
    let mut cache = slot.lock_recover();
    if let Some(c) = cache
        .as_ref()
        .filter(|c| running && c.view_tag == view_tag && c.models == models)
    {
        return (c.json.clone(), c.tag.clone());
    }
    let mut rebased = models.clone();
    for m in &mut rebased {
        if let Some(n) = now.get(&crate::registry::normalize_name(&m.model)) {
            crate::benchmark::current::rebase_model(m, &[], None, n);
        }
    }
    let json: std::sync::Arc<RawValue> = raw(&rebased).into();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    json.get().hash(&mut h);
    let tag = format!("{:016x}", h.finish());
    *cache = running.then(|| RunModels {
        view_tag: view_tag.to_string(),
        models,
        json: json.clone(),
        tag: tag.clone(),
    });
    (json, tag)
}

/// `run` of the answer: [`Status`] as its `serde_json::Value` serializes (sorted keys), the
/// models spliced in from [`models_json`], or left out (`None`).
#[derive(Serialize)]
struct RunView<'a> {
    error: &'a Option<String>,
    #[serde(rename = "finishedAt")]
    finished_at: &'a Option<String>,
    message: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    models: Option<&'a RawValue>,
    progress: Box<RawValue>,
    request: Box<RawValue>,
    running: bool,
    #[serde(rename = "startedAt")]
    started_at: &'a Option<String>,
    state: RunState,
    warnings: &'a [String],
}

fn run_view<'a>(run: &'a Status, models: Option<&'a RawValue>) -> RunView<'a> {
    RunView {
        error: &run.error,
        finished_at: &run.finished_at,
        message: &run.message,
        models,
        progress: raw(&run.progress),
        request: raw(&run.request),
        running: run.running,
        started_at: &run.started_at,
        state: run.state,
        warnings: &run.warnings,
    }
}

/// Everything of the answer but the run, with its tag.
struct Views {
    tag: String,
    saved: std::sync::Arc<super::SavedView>,
    config: Box<RawValue>,
    profiles: Box<RawValue>,
    goal: Box<RawValue>,
    other_hardware: bool,
    results_file: String,
    /// What the config says now about each model (the run's models are shown against it).
    now: std::collections::HashMap<String, crate::benchmark::current::ModelNow>,
}

fn views(state: &AppState) -> Views {
    let snap = devices_snapshot(state);
    let config = raw(&config_view(state, &snap));
    let (revision, cfg) = state.config.snapshot();
    let now = models_now(state, &cfg, &snap);
    let saved = saved_view(state, revision, &now);
    let loaded = loaded_devices(state);
    let profile_rows = raw(&profiles::profiles(
        &saved.facts,
        &cfg,
        &|name| loaded.get(&crate::registry::normalize_name(name)).cloned(),
        &|m| configured_device(&cfg, m, &snap.model_selection(state, &cfg, m)),
    ));
    let other_hardware = saved
        .hardware
        .as_ref()
        .is_some_and(|h| !h.same_gpus(&snap.hardware));
    let goal = raw(&serde_json::json!({
        "label": profiles::goal_label(&cfg.benchmark),
        "objective": cfg.benchmark.threshold_objective.to_string(),
    }));
    let results_file = results_path(&state.config_path).display().to_string();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (
        &saved.key,
        state.config.generation(),
        state.config.epoch(),
        config.get(),
        profile_rows.get(),
        goal.get(),
        other_hardware,
        &results_file,
    )
        .hash(&mut h);
    Views {
        tag: format!("{revision}.{:016x}", h.finish()),
        saved,
        config,
        profiles: profile_rows,
        goal,
        other_hardware,
        results_file,
        now,
    }
}

/// `GET /v1/benchmark?only=run`.
#[derive(Serialize)]
struct RunAnswer<'a> {
    success: bool,
    state: RunState,
    running: bool,
    run: RunView<'a>,
    #[serde(rename = "viewTag")]
    view_tag: &'a str,
    /// The tag of `run.models` (sent back as `models=`).
    #[serde(rename = "modelsTag")]
    models_tag: &'a str,
}

/// The full `GET /v1/benchmark`, fields in the order `json!` gave them (sorted), so the bytes
/// are those of the answer built as a `serde_json::Value` before.
#[derive(Serialize)]
struct FullAnswer<'a> {
    config: &'a RawValue,
    goal: &'a RawValue,
    latest_run_id: &'a Option<String>,
    lineup: &'a RawValue,
    noise: serde_json::Value,
    #[serde(rename = "otherHardware")]
    other_hardware: bool,
    points: &'a RawValue,
    profiles: &'a RawValue,
    ranking: &'a RawValue,
    results: &'a RawValue,
    #[serde(rename = "resultsFile")]
    results_file: &'a str,
    run: RunView<'a>,
    running: bool,
    state: RunState,
    success: bool,
    warning: &'static str,
}

fn json_response(body: Vec<u8>) -> Response {
    (
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            ),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
        ],
        body,
    )
        .into_response()
}

fn serialize<T: Serialize>(value: &T) -> Vec<u8> {
    serde_json::to_vec(value).unwrap_or_else(|e| {
        tracing::warn!("could not serialize the benchmark status: {e}");
        b"{\"success\":false,\"message\":\"could not serialize the benchmark status (see the log)\"}"
            .to_vec()
    })
}

/// `GET /v1/benchmark` (`query`: [`ask`]), built and serialized on the calling (blocking)
/// thread.
pub(super) fn answer(
    state: &AppState,
    query: Option<&str>,
    if_none_match: Option<&str>,
) -> Response {
    let ask = ask(query);
    let v = views(state);
    let mut run = state.benchmark.status();
    let (models, models_tag) = models_json(
        &RUN_MODELS,
        &v.tag,
        std::mem::take(&mut run.models),
        &v.now,
        run.running,
    );
    if ask.only_run {
        let same = ask.models.as_deref() == Some(models_tag.as_str());
        return json_response(serialize(&RunAnswer {
            success: true,
            state: run.state,
            running: run.running,
            run: run_view(&run, (!same).then_some(&*models)),
            view_tag: &v.tag,
            models_tag: &models_tag,
        }));
    }
    let run_view = run_view(&run, Some(&models));
    let mut h = std::collections::hash_map::DefaultHasher::new();
    serialize(&run_view).hash(&mut h);
    let etag = format!("W/\"{}-{:016x}\"", v.tag, h.finish());
    let etag_value = HeaderValue::from_str(&etag).unwrap_or(HeaderValue::from_static("W/\"0\""));
    let view_value = HeaderValue::from_str(&v.tag).unwrap_or(HeaderValue::from_static("0"));
    let matches = if_none_match.is_some_and(|m| m.split(',').any(|t| t.trim() == etag));
    let mut resp = if matches {
        (
            StatusCode::NOT_MODIFIED,
            [(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"))],
        )
            .into_response()
    } else {
        let saved = &v.saved;
        json_response(serialize(&FullAnswer {
            config: &v.config,
            goal: &v.goal,
            latest_run_id: &saved.latest_run_id,
            lineup: &saved.lineup,
            noise: crate::benchmark::history::view::noise_json(),
            other_hardware: v.other_hardware,
            points: &saved.points,
            profiles: &v.profiles,
            ranking: &saved.ranking,
            results: &saved.results,
            results_file: &v.results_file,
            run: run_view,
            running: run.running,
            state: run.state,
            success: true,
            warning: LIVE_WARNING,
        }))
    };
    resp.headers_mut().insert(header::ETAG, etag_value);
    resp.headers_mut().insert(VIEW_HEADER, view_value);
    resp
}

#[cfg(test)]
mod tests {
    use super::super::tests::{get, profile_results, state};
    use super::super::*;
    use super::{Ask, VIEW_HEADER, ask};
    use crate::benchmark::service::{ProgressView, RunRequest, RunState, Status};
    use crate::config::ModelConfig;
    use axum::body::Body;
    use axum::http::{HeaderMap, Request, header};
    use tower::ServiceExt;

    /// Removes a test state's folder.
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            if std::env::var_os("BOP_KEEP_TEST_DIRS").is_none() {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    /// [`state`] and the guard that removes its folder.
    fn guarded() -> (Arc<AppState>, Cleanup) {
        let st = state();
        let dir = st.config_path.parent().unwrap().to_path_buf();
        (st, Cleanup(dir))
    }

    async fn fetch(st: &Arc<AppState>, req: Request<Body>) -> (StatusCode, HeaderMap, Vec<u8>) {
        let resp = super::super::super::router(st.clone())
            .oneshot(req)
            .await
            .unwrap();
        let status = resp.status();
        let headers = resp.headers().clone();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec();
        (status, headers, body)
    }

    fn get_if(uri: &str, etag: &str) -> Request<Body> {
        let mut r = get(uri);
        r.headers_mut()
            .insert(header::IF_NONE_MATCH, etag.parse().unwrap());
        r
    }

    fn json(body: &[u8]) -> serde_json::Value {
        serde_json::from_slice(body).unwrap()
    }

    fn header_of<'h>(h: &'h HeaderMap, name: &str) -> &'h str {
        h.get(name).unwrap().to_str().unwrap()
    }

    /// A run that waits for its sender to drop; `set` (progress) and `current` (the model in
    /// progress) move it on.
    struct Run {
        progress: std::sync::mpsc::Sender<Option<Step>>,
        st: Arc<AppState>,
    }

    enum Step {
        /// Progress at this percent.
        Progress(f64),
        Current(Box<crate::benchmark::ModelResult>),
    }
    impl Run {
        fn start(st: &Arc<AppState>, done: Vec<crate::benchmark::ModelResult>) -> Self {
            let (tx, rx) = std::sync::mpsc::channel::<Option<Step>>();
            let (ack_tx, ack_rx) = std::sync::mpsc::channel::<()>();
            let ack = ack_tx.clone();
            st.benchmark
                .start_job(
                    RunRequest::defaults(&Config::default().benchmark),
                    move |h| {
                        for m in done {
                            h.push_done(m);
                        }
                        let _ = ack.send(());
                        while let Ok(Some(step)) = rx.recv() {
                            match step {
                                Step::Progress(percent) => h.set_progress(ProgressView {
                                    stage: "benchmark".into(),
                                    model: "m".into(),
                                    model_index: 1,
                                    model_count: 2,
                                    percent,
                                    ..Default::default()
                                }),
                                Step::Current(m) => h.set_current(*m),
                            }
                        }
                        Ok("finished".into())
                    },
                )
                .unwrap();
            ack_rx.recv().unwrap();
            // Later steps: `set` / `current` wait for theirs to show in the status.
            drop((ack_rx, ack_tx));
            Self {
                progress: tx,
                st: st.clone(),
            }
        }

        fn set(&self, percent: f64) {
            self.progress.send(Some(Step::Progress(percent))).unwrap();
            let t = std::time::Instant::now();
            while self
                .st
                .benchmark
                .status()
                .progress
                .is_none_or(|p| p.percent != percent)
            {
                assert!(t.elapsed() < std::time::Duration::from_secs(5));
                std::thread::yield_now();
            }
        }

        fn current(&self, m: crate::benchmark::ModelResult) {
            let n = self.st.benchmark.status().models.len();
            self.progress
                .send(Some(Step::Current(Box::new(m))))
                .unwrap();
            let t = std::time::Instant::now();
            while self.st.benchmark.status().models.len() == n {
                assert!(t.elapsed() < std::time::Duration::from_secs(5));
                std::thread::yield_now();
            }
        }

        fn finish(self) {
            let _ = self.progress.send(None);
            assert!(
                self.st
                    .benchmark
                    .wait_idle(std::time::Duration::from_secs(5))
            );
        }
    }

    #[test]
    fn the_query_asks_for_exact_pairs() {
        let run = |models: Option<&str>| Ask {
            only_run: true,
            models: models.map(String::from),
        };
        for (q, want) in [
            (None, Ask::default()),
            (Some(""), Ask::default()),
            (Some("only=run"), run(None)),
            (Some("x=1&only=run"), run(None)),
            (Some("only=run&models=ab12"), run(Some("ab12"))),
            (Some("only=run&models="), run(None)),
            (Some("only=runs"), Ask::default()),
            (Some("only=all"), Ask::default()),
            (Some("ONLY=run"), Ask::default()),
        ] {
            assert_eq!(ask(q), want, "{q:?}");
        }
    }

    /// `run` is [`Status`] as `json!` serialized it: the same bytes, with or without models.
    #[test]
    fn the_run_keeps_its_bytes() {
        use crate::benchmark::report::test_support::{dev, model, scene};
        let mut m = model(vec![dev("ort:cpu", 12.5, scene())], None);
        m.model = "x".into();
        let status = Status {
            state: RunState::Running,
            running: true,
            request: Some(RunRequest::defaults(&Config::default().benchmark)),
            progress: Some(ProgressView {
                stage: "benchmark".into(),
                percent: 12.5,
                eta_secs: Some(3.25),
                ..Default::default()
            }),
            models: vec![m.clone()],
            message: None,
            error: Some("e".into()),
            warnings: vec!["w".into()],
            started_at: Some("t0".into()),
            finished_at: None,
        };
        let old = serde_json::to_value(&status).unwrap().to_string();
        let models = raw(&status.models);
        let new = serde_json::to_string(&super::run_view(&status, Some(&models))).unwrap();
        assert_eq!(new, old);
        let mut bare = serde_json::to_value(&status).unwrap();
        bare.as_object_mut().unwrap().remove("models");
        let new = serde_json::to_string(&super::run_view(&status, None)).unwrap();
        assert_eq!(new, bare.to_string());
    }

    /// The full answer is what `json!` built before: the same keys, serialized as a
    /// `serde_json::Value` serializes (sorted keys, compact), so its bytes are unchanged.
    #[tokio::test]
    async fn the_full_answer_keeps_its_shape_and_bytes() {
        let (st, _g) = guarded();
        profile_results(Default::default())
            .save(&results_path(&st.config_path))
            .unwrap();
        for _ in 0..2 {
            let (s, h, body) = fetch(&st, get("/v1/benchmark")).await;
            assert_eq!(s, StatusCode::OK);
            assert_eq!(header_of(&h, "content-type"), "application/json");
            let v = json(&body);
            assert_eq!(
                v.to_string(),
                String::from_utf8(body).unwrap(),
                "Value's own bytes"
            );
            let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(|k| k.as_str()).collect();
            keys.sort_unstable();
            assert_eq!(
                keys,
                [
                    "config",
                    "goal",
                    "latest_run_id",
                    "lineup",
                    "noise",
                    "otherHardware",
                    "points",
                    "profiles",
                    "ranking",
                    "results",
                    "resultsFile",
                    "run",
                    "running",
                    "state",
                    "success",
                    "warning"
                ]
            );
            assert_eq!(v["results"]["models"][0]["model"], "ipcam-general");
            assert!(v["ranking"].is_array() && v["points"].is_array(), "{v}");
            assert_eq!(v["state"], "idle");
            assert_eq!(v["running"], false);
            assert_eq!(v["success"], true);
            assert_eq!(v["warning"], LIVE_WARNING);
        }
        // No results: the empty answer as before.
        let (st, _g) = guarded();
        let (_, _, body) = fetch(&st, get("/v1/benchmark")).await;
        let v = json(&body);
        assert_eq!(v.to_string(), String::from_utf8(body).unwrap());
        for k in ["results", "ranking", "points", "latest_run_id"] {
            assert_eq!(v[k], serde_json::Value::Null, "{k}");
        }
        assert_eq!(v["lineup"]["sets"], serde_json::json!([]));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_run_poll_is_small_and_tags_the_rest() {
        let (st, _g) = guarded();
        let results = profile_results(Default::default());
        results.save(&results_path(&st.config_path)).unwrap();
        let run = Run::start(&st, vec![results.models[0].summary()]);
        run.set(10.0);

        let (s, h, full) = fetch(&st, get("/v1/benchmark")).await;
        assert_eq!(s, StatusCode::OK);
        let tag = header_of(&h, VIEW_HEADER).to_string();
        let etag = header_of(&h, "etag").to_string();
        assert!(etag.starts_with(&format!("W/\"{tag}-")), "{etag} {tag}");
        let full = json(&full);
        assert_eq!(full["running"], true);

        let (s, h, small) = fetch(&st, get("/v1/benchmark?only=run")).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(header_of(&h, "content-type"), "application/json");
        assert!(
            small.len() < full.to_string().len(),
            "smaller than the full answer"
        );
        let small = json(&small);
        let mut keys: Vec<&str> = small
            .as_object()
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            ["modelsTag", "run", "running", "state", "success", "viewTag"]
        );
        assert_eq!(small["viewTag"], tag.as_str());
        assert_eq!(small["state"], "running");
        assert_eq!(small["running"], true);
        // The same run as in the full answer, its models shown against the config now.
        assert_eq!(small["run"], full["run"]);
        assert_eq!(small["run"]["models"][0]["model"], "ipcam-general");
        assert_eq!(small["run"]["progress"]["percent"], 10.0);
        // The page has these models: they are left out, the rest of the run is there.
        let mt = small["modelsTag"].as_str().unwrap().to_string();
        let uri = format!("/v1/benchmark?only=run&models={mt}");
        let (_, _, same) = fetch(&st, get(&uri)).await;
        let same = json(&same);
        assert_eq!(same["modelsTag"], mt.as_str());
        assert!(same["run"].get("models").is_none(), "{same}");
        let mut want = small["run"].clone();
        want.as_object_mut().unwrap().remove("models");
        assert_eq!(same["run"], want);
        // Another tag (another run, the model's results moved on): sent in full.
        let (_, _, other) = fetch(&st, get("/v1/benchmark?only=run&models=0")).await;
        assert_eq!(json(&other)["run"], small["run"]);

        // Progress moves: the run changes, the rest does not.
        run.set(20.0);
        let (_, _, small) = fetch(&st, get("/v1/benchmark?only=run")).await;
        let small = json(&small);
        assert_eq!(small["run"]["progress"]["percent"], 20.0);
        assert_eq!(small["viewTag"], tag.as_str());
        // ... so the full answer does (a new ETag, never a 304 with an old run).
        let (s, h, _) = fetch(&st, get_if("/v1/benchmark", &etag)).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(header_of(&h, VIEW_HEADER), tag);
        let etag = header_of(&h, "etag").to_string();
        // Nothing changed: 304, no body, the same tags.
        let (s, h, body) = fetch(&st, get_if("/v1/benchmark", &etag)).await;
        assert_eq!(s, StatusCode::NOT_MODIFIED);
        assert!(body.is_empty());
        assert_eq!(header_of(&h, "etag"), etag);
        assert_eq!(header_of(&h, VIEW_HEADER), tag);
        // A list of tags, one of them ours (weak comparison as sent).
        let (s, _, _) = fetch(&st, get_if("/v1/benchmark", &format!("W/\"x\", {etag}"))).await;
        assert_eq!(s, StatusCode::NOT_MODIFIED);

        // A model's results saved (the file changes): a new view tag.
        let mut more = results.clone();
        more.models.truncate(2);
        more.save(&results_path(&st.config_path)).unwrap();
        let (_, _, small) = fetch(&st, get("/v1/benchmark?only=run")).await;
        let tag2 = json(&small)["viewTag"].as_str().unwrap().to_string();
        assert_ne!(tag2, tag, "saved results changed");
        let (s, h, _) = fetch(&st, get_if("/v1/benchmark", &etag)).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(header_of(&h, VIEW_HEADER), tag2);

        // A config change: a new view tag too.
        st.config
            .update("test", |c| {
                c.confidence_threshold = 0.42;
                Ok(())
            })
            .unwrap();
        let (_, _, small) = fetch(&st, get("/v1/benchmark?only=run")).await;
        let tag3 = json(&small)["viewTag"].as_str().unwrap().to_string();
        assert_ne!(tag3, tag2, "config changed");

        // The models change (a device run finished): sent again under a new tag, with the
        // page's old tag.
        let (_, _, small) = fetch(&st, get("/v1/benchmark?only=run")).await;
        let mt = json(&small)["modelsTag"].as_str().unwrap().to_string();
        let mut cur = results.models[1].summary();
        cur.devices.truncate(1);
        run.current(cur);
        let (_, _, small) = fetch(&st, get(&format!("/v1/benchmark?only=run&models={mt}"))).await;
        let small = json(&small);
        assert_ne!(small["modelsTag"], mt.as_str());
        assert_eq!(small["run"]["models"][1]["model"], "dfine-s");

        // The run ends: the poll says so.
        run.finish();
        let (_, _, small) = fetch(&st, get("/v1/benchmark?only=run")).await;
        let small = json(&small);
        assert_eq!(small["running"], false);
        assert_eq!(small["state"], "done");
        assert_eq!(small["run"]["message"], "finished");
    }

    /// The view tag follows the saved view alone: results saved again with only their time
    /// changed (profiles, config and goal the same) give a new tag and the new time; nothing
    /// changed, the same tag.
    #[tokio::test]
    async fn the_view_tag_follows_the_saved_results_alone() {
        let (st, _g) = guarded();
        let path = results_path(&st.config_path);
        let mut results = profile_results(Default::default());
        results.timestamp = "2026-11-01T10:00:00Z".into();
        results.save(&path).unwrap();
        let full = |st: Arc<AppState>| async move {
            let (s, h, body) = fetch(&st, get("/v1/benchmark")).await;
            assert_eq!(s, StatusCode::OK);
            (header_of(&h, VIEW_HEADER).to_string(), json(&body))
        };
        let (tag, v) = full(st.clone()).await;
        let (again, v_again) = full(st.clone()).await;
        assert_eq!(again, tag, "nothing changed");
        assert_eq!(v_again, v);

        // A longer time, so the file's length changes too (coarse mtimes cannot hide it).
        results.timestamp = "2026-11-02T10:00:00.123456Z".into();
        results.save(&path).unwrap();
        let (tag2, v2) = full(st.clone()).await;
        assert_eq!(v2["results"]["timestamp"], "2026-11-02T10:00:00.123456Z");
        for k in ["config", "profiles", "goal", "otherHardware"] {
            assert_eq!(v2[k], v[k], "{k} is the same");
        }
        assert_ne!(tag2, tag, "the saved view changed");
        let (_, _, small) = fetch(&st, get("/v1/benchmark?only=run")).await;
        assert_eq!(json(&small)["viewTag"], tag2.as_str());
    }

    /// The run's models are reused only while they are the service's models: progress inside
    /// a device run (the same models) reuses the JSON, a model with one more device, a changed
    /// device or the next model (same view tag) is serialized again.
    #[test]
    fn the_run_models_follow_the_service() {
        use crate::benchmark::report::test_support::{dev, model, scene};
        let slot: super::RunModelsSlot = std::sync::Mutex::new(None);
        let now = std::collections::HashMap::new();
        let mk = |devices: &[(&str, f64)]| {
            let mut m = model(
                devices.iter().map(|(d, ms)| dev(d, *ms, scene())).collect(),
                None,
            );
            m.model = "x".into();
            m
        };
        let one = vec![mk(&[("openvino:cpu", 10.0)])];
        let models_of =
            |j: &RawValue| -> serde_json::Value { serde_json::from_str(j.get()).unwrap() };

        let (a, ta) = super::models_json(&slot, "v1", one.clone(), &now, true);
        let (b, tb) = super::models_json(&slot, "v1", one.clone(), &now, true);
        assert!(Arc::ptr_eq(&a, &b), "the same models: reused");
        assert_eq!(ta, tb);

        // The model moves to its next device.
        let two = vec![mk(&[("openvino:cpu", 10.0), ("ort:cpu", 20.0)])];
        let (c, tc) = super::models_json(&slot, "v1", two.clone(), &now, true);
        assert_ne!(tc, ta);
        assert_eq!(models_of(&c)[0]["devices"].as_array().unwrap().len(), 2);
        // A device's numbers change with the device count the same.
        let changed = vec![mk(&[("openvino:cpu", 10.0), ("ort:cpu", 25.0)])];
        let (d, td) = super::models_json(&slot, "v1", changed.clone(), &now, true);
        assert_ne!(td, tc);
        assert_eq!(models_of(&d), serde_json::to_value(&changed).unwrap());
        // The next model.
        let mut next = changed.clone();
        let mut y = mk(&[("openvino:cpu", 5.0)]);
        y.model = "y".into();
        next.push(y);
        let (e, _) = super::models_json(&slot, "v1", next.clone(), &now, true);
        assert_eq!(models_of(&e)[1]["model"], "y");
        // Another view tag (the config now changed): serialized again; not running: not kept.
        let (f, _) = super::models_json(&slot, "v2", next.clone(), &now, true);
        assert!(!Arc::ptr_eq(&e, &f));
        let _ = super::models_json(&slot, "v2", next, &now, false);
        assert!(slot.lock().unwrap().is_none());
    }

    /// The run poll never rebuilds the saved view while nothing it is made of changes.
    #[tokio::test]
    async fn the_run_poll_reuses_the_saved_view() {
        use std::sync::atomic::Ordering;
        let (st, _g) = guarded();
        profile_results(Default::default())
            .save(&results_path(&st.config_path))
            .unwrap();
        let (_, _, first) = fetch(&st, get("/v1/benchmark?only=run")).await;
        let tag = json(&first)["viewTag"].clone();
        // Other tests compute views in parallel: count only while this one's tag holds.
        for _ in 0..20 {
            let before = SAVED_COMPUTED.load(Ordering::Relaxed);
            let (_, _, b) = fetch(&st, get("/v1/benchmark?only=run")).await;
            assert_eq!(json(&b)["viewTag"], tag);
            if SAVED_COMPUTED.load(Ordering::Relaxed) == before {
                return;
            }
        }
        panic!("every poll recomputed the saved view");
    }

    /// Whether `node` runs here; missing, a skip, or with `BOP_REQUIRE_NODE=1` (CI) a failure.
    fn have_node() -> bool {
        let ok = std::process::Command::new("node")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success());
        if !ok {
            assert!(
                std::env::var("BOP_REQUIRE_NODE").as_deref() != Ok("1"),
                "BOP_REQUIRE_NODE=1 but `node` does not run: install Node or unset the variable"
            );
            println!("skipped: no node");
        }
        ok
    }

    /// `BenchPoll` of `assets/benchmark.js` (loaded without a DOM): each case's `step(last, s)`
    /// and `runUri(tag)`.
    const STEP_DRIVER: &str = r#"
const fs = require('fs'), vm = require('vm');
const ctx = vm.createContext({});
vm.runInContext(fs.readFileSync(process.env.BOP_BENCHMARK_JS, 'utf8'), ctx);
const P = ctx.BenchPoll;
const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify({
  steps: cases.steps.map((c) => P.step(c.last, c.answer)),
  uris: cases.uris.map((t) => P.runUri(t)),
}));
"#;

    fn node_steps(cases: &serde_json::Value) -> serde_json::Value {
        use std::io::Write;
        let js = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/benchmark.js");
        let mut child = std::process::Command::new("node")
            .args(["-e", STEP_DRIVER])
            .env("BOP_BENCHMARK_JS", js)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(cases.to_string().as_bytes())
            .unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "node: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    }

    /// The page's decision per run poll: merge the run into what it shows, or fetch the full
    /// answer (another view, the run ended, models it cannot fill in), or show an old server's
    /// full answer as it is.
    #[test]
    fn the_page_fetches_the_full_answer_only_when_it_must() {
        use serde_json::json;
        if !have_node() {
            return;
        }
        let shown = json!({
            "state": "running", "running": true, "results": {"models": [{"model": "a"}]},
            "run": {"progress": {"percent": 10}, "models": [{"model": "a", "x": 1}]},
        });
        let last = |tag: Option<&str>, models: Option<&str>| json!({"status": shown, "viewTag": tag, "modelsTag": models});
        let answer = |view: &str, running: bool, models: Option<serde_json::Value>| {
            let mut run = json!({"progress": {"percent": 20}});
            if let Some(m) = models {
                run["models"] = m;
            }
            json!({"success": true, "state": if running { "running" } else { "done" },
                   "running": running, "run": run, "viewTag": view, "modelsTag": "m2"})
        };
        let merged = |models: serde_json::Value, tag: &str| {
            let mut s = shown.clone();
            s["run"] = json!({"progress": {"percent": 20}, "models": models});
            json!({"status": s, "modelsTag": tag})
        };
        let full = json!({"full": true});
        let new_models = json!([{"model": "a", "x": 2}]);
        let cases: Vec<(
            &str,
            serde_json::Value,
            serde_json::Value,
            serde_json::Value,
        )> = vec![
            (
                "same view, models sent: merged",
                last(Some("v1"), None),
                answer("v1", true, Some(new_models.clone())),
                merged(new_models.clone(), "m2"),
            ),
            (
                "same view, models left out: the page's own",
                last(Some("v1"), Some("m2")),
                answer("v1", true, None),
                merged(shown["run"]["models"].clone(), "m2"),
            ),
            (
                "models left out under another tag: full",
                last(Some("v1"), Some("m1")),
                answer("v1", true, None),
                full.clone(),
            ),
            (
                "models left out, the page has no tag: full",
                last(Some("v1"), None),
                answer("v1", true, None),
                full.clone(),
            ),
            (
                "another view: full",
                last(Some("v1"), Some("m2")),
                answer("v2", true, None),
                full.clone(),
            ),
            (
                "the run ended: full",
                last(Some("v1"), None),
                answer("v1", false, Some(json!([]))),
                full.clone(),
            ),
            (
                "no view tag yet: full",
                last(None, None),
                answer("v1", true, Some(json!([]))),
                full.clone(),
            ),
            (
                "nothing shown yet: full",
                json!({"status": null, "viewTag": "v1"}),
                answer("v1", true, Some(json!([]))),
                full.clone(),
            ),
            (
                "an old server answers the full status",
                last(Some("v1"), None),
                json!({"success": true, "running": true, "run": {}, "results": null}),
                json!({"old": true}),
            ),
            (
                "not JSON: full",
                last(Some("v1"), None),
                json!(null),
                full.clone(),
            ),
        ];
        // A run that starts while the page shows an idle answer merges (no full answer).
        let mut idle = last(Some("v1"), None);
        idle["status"]["running"] = json!(false);
        idle["status"]["state"] = json!("idle");
        let started = answer("v1", true, Some(json!([])));
        let out = node_steps(&json!({
            "steps": cases.iter().map(|c| json!({"last": c.1, "answer": c.2}))
                .chain([json!({"last": idle, "answer": started})]).collect::<Vec<_>>(),
            "uris": [null, "", "ab12", "a&b"],
        }));
        for (i, (name, _, _, want)) in cases.iter().enumerate() {
            assert_eq!(&out["steps"][i], want, "{name}");
        }
        let s = &out["steps"][cases.len()]["status"];
        assert_eq!(
            (&s["running"], &s["state"]),
            (&json!(true), &json!("running"))
        );
        assert_eq!(s["results"], shown["results"], "the rest kept");
        assert_eq!(
            out["uris"],
            json!([
                "/v1/benchmark?only=run",
                "/v1/benchmark?only=run",
                "/v1/benchmark?only=run&models=ab12",
                "/v1/benchmark?only=run&models=a%26b"
            ])
        );
        // The page polls through these, PrismPoll's loop, and the full answer with its ETag.
        let js = crate::server::pages::static_file("benchmark.js").expect("benchmark.js");
        for needle in [
            "BP.runUri(polled.modelsTag)",
            "BP.step(polled, s)",
            "PrismPoll.create({run: pollRun",
            "\"If-None-Match\": etag",
            "X-Benchmark-View",
        ] {
            assert!(js.contains(needle), "{needle}");
        }
        assert!(!js.contains("setInterval"), "chained timeouts only");
    }

    /// Bytes and build time per poll on the saved results in `BOP_BENCH_FIXTURE` (a folder
    /// with `benchmark.json`, `benchmark-preds.json`, `benchmark-history.json`), during a run.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "measurement; needs BOP_BENCH_FIXTURE"]
    async fn poll_cost() {
        let Some(src) = std::env::var_os("BOP_BENCH_FIXTURE") else {
            eprintln!("BOP_BENCH_FIXTURE not set");
            return;
        };
        let src = std::path::PathBuf::from(src);
        let (st, _g) = guarded();
        let dir = st.config_path.parent().unwrap().to_path_buf();
        for f in [
            "benchmark.json",
            "benchmark-preds.json",
            "benchmark-history.json",
        ] {
            if src.join(f).is_file() {
                std::fs::copy(src.join(f), dir.join(f)).unwrap();
            }
        }
        let saved = BenchmarkResults::load_or_warn(&dir.join("benchmark.json")).unwrap();
        st.config
            .update("test", |c| {
                for m in &saved.models {
                    let key = crate::registry::normalize_name(&m.model);
                    if c.models
                        .iter()
                        .any(|x| crate::registry::normalize_name(&x.effective_name()) == key)
                    {
                        continue;
                    }
                    c.models.push(ModelConfig {
                        name: Some(m.model.clone()),
                        path: dir.join(format!("{}.onnx", m.model)),
                        enabled: false,
                        ..Default::default()
                    });
                }
                Ok(())
            })
            .unwrap();
        let run = Run::start(&st, vec![saved.models[0].summary()]);
        run.set(40.0);
        let (_, h, _) = fetch(&st, get("/v1/benchmark")).await;
        let etag = header_of(&h, "etag").to_string();
        let measure = |req: fn(&str) -> Request<Body>| {
            let st = st.clone();
            async move {
                fetch(&st, req("")).await;
                let n = 20;
                let t = std::time::Instant::now();
                let mut bytes = 0;
                let mut status = StatusCode::OK;
                for _ in 0..n {
                    let (s, _, b) = fetch(&st, req("")).await;
                    (status, bytes) = (s, b.len());
                }
                (
                    status,
                    bytes,
                    t.elapsed().as_secs_f64() * 1000.0 / f64::from(n),
                )
            }
        };
        let (s, b, ms) = measure(|_| get("/v1/benchmark/no-such-route")).await;
        eprintln!("baseline (test router + a 404): {s} {b} bytes, {ms:.2} ms per request");
        let (s, b, ms) = measure(|_| get("/v1/benchmark")).await;
        eprintln!("full /v1/benchmark: {s} {b} bytes, {ms:.2} ms per poll");
        let (s, b, ms) = measure(|_| get("/v1/benchmark?only=run")).await;
        eprintln!("/v1/benchmark?only=run: {s} {b} bytes, {ms:.2} ms per poll");
        let leaked: &'static str = Box::leak(etag.into_boxed_str());
        thread_local!(static TAG: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") });
        TAG.with(|t| t.set(leaked));
        let (s, b, ms) = measure(|_| get_if("/v1/benchmark", TAG.with(|t| t.get()))).await;
        eprintln!("full with a matching If-None-Match: {s} {b} bytes, {ms:.2} ms per poll");
        // The page's poll once it has the run's models.
        let (_, _, b) = fetch(&st, get("/v1/benchmark?only=run")).await;
        let mt = json(&b)["modelsTag"].as_str().unwrap().to_string();
        let leaked: &'static str =
            Box::leak(format!("/v1/benchmark?only=run&models={mt}").into_boxed_str());
        thread_local!(static URI: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") });
        URI.with(|t| t.set(leaked));
        let (s, b, ms) = measure(|_| get(URI.with(|t| t.get()))).await;
        eprintln!("/v1/benchmark?only=run&models=<same>: {s} {b} bytes, {ms:.2} ms per poll");
        let n = 20u32;
        let mut t = [0f64; 3];
        for _ in 0..n {
            let mut lap = std::time::Instant::now();
            let mut tick = |i: usize| {
                t[i] += lap.elapsed().as_secs_f64() * 1000.0 / f64::from(n);
                lap = std::time::Instant::now();
            };
            let v = super::views(&st);
            tick(0);
            let mut run = st.benchmark.status();
            let models = std::mem::take(&mut run.models);
            tick(1);
            let _ = super::models_json(&super::RUN_MODELS, &v.tag, models, &v.now, run.running);
            tick(2);
        }
        eprintln!(
            "  views {:.2} ms, run status {:.2} ms, run models JSON (cached) {:.2} ms",
            t[0], t[1], t[2]
        );
        run.finish();
    }
}
