//! `GET /v1/status`: machine-readable readiness for container orchestrators, systemd wrappers
//! and Blue Iris watchdogs. Cheap (no file I/O) and open to every client; `curl -f` works
//! because the HTTP status is 503 while the service cannot serve detections: no model is usable
//! (ready or lazy). A usable model next to one still downloading or compiling is `degraded` with
//! 200 (audit 2026-11, QA-069: a model added on the Config page, or a download that keeps
//! failing, must not mark a container that serves Blue Iris unhealthy); `loading` names the
//! models still initializing, so that case is told apart from a failed one. `fileLogCapped`
//! says today's log file reached its cap ([`crate::cli::file_log_capped`]): only errors still
//! reach it until tomorrow, so something logs far more than usual.

use super::{AppState, state_view};
use crate::startup::ModelState;
use crate::tray::probe::label;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;
use std::sync::Arc;

/// Overall readiness derived from the per-model states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Overall {
    Ready,
    Initializing,
    Degraded,
    Failed,
}

impl Overall {
    fn label(self) -> &'static str {
        match self {
            Self::Ready => label::READY,
            Self::Initializing => label::INITIALIZING,
            Self::Degraded => label::DEGRADED,
            Self::Failed => label::FAILED,
        }
    }

    /// HTTP status: 200 when the service can serve (ready or degraded), 503 otherwise.
    pub(super) fn http(self) -> StatusCode {
        match self {
            Self::Ready | Self::Degraded => StatusCode::OK,
            Self::Initializing | Self::Failed => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

/// Per-model state as reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Ready,
    Initializing,
    Lazy,
    Failed,
}

impl Kind {
    /// A model's `state` in the answer ([`label`], shared with the tray).
    fn label(self) -> &'static str {
        match self {
            Self::Ready => label::READY,
            Self::Initializing => label::INITIALIZING,
            Self::Lazy => label::LAZY,
            Self::Failed => label::FAILED,
        }
    }
}

/// Combine model states. A lazy model (loads on its first request) can serve, so it counts as
/// usable like a Ready one. Usable: ready when every other model is usable too, else degraded
/// (another one failed or is still initializing; both answer 200). Nothing usable but something
/// initializing = initializing. Otherwise failed (also with no models).
pub(super) fn overall(kinds: &[Kind]) -> Overall {
    let has = |k| kinds.contains(&k);
    let usable = has(Kind::Ready) || has(Kind::Lazy);
    if usable {
        if has(Kind::Failed) || has(Kind::Initializing) {
            Overall::Degraded
        } else {
            Overall::Ready
        }
    } else if has(Kind::Initializing) {
        Overall::Initializing
    } else {
        Overall::Failed
    }
}

#[allow(clippy::unused_async, reason = "axum handlers are async")]
pub(super) async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    let workers = state.registry.workers();
    let mut kinds = Vec::with_capacity(workers.len());
    let mut models = Vec::with_capacity(workers.len());
    let mut loading = Vec::new();
    for w in &workers {
        let (kind, detail) = match w.state.get() {
            ModelState::Ready => (Kind::Ready, None),
            ModelState::Failed(m) => (Kind::Failed, Some(m)),
            ModelState::Initializing if w.accepts_while_initializing() => (Kind::Lazy, None),
            ModelState::Initializing => (Kind::Initializing, state_view(w).2),
        };
        kinds.push(kind);
        if kind == Kind::Initializing {
            loading.push(w.name.clone());
        }
        // The threshold that runs: the profile for the device it loaded on, else the model's /
        // global value (additive fields; the Blue Iris response shape is untouched).
        let (threshold, source) = w.effective_threshold();
        models.push(json!({
            "name": w.name,
            "state": kind.label(),
            "device": w.metrics.device(),
            "detail": detail,
            "threshold": crate::config::round_threshold_f64(threshold),
            "thresholdSource": source.label(),
        }));
    }
    let overall = overall(&kinds);
    let body = json!({
        "success": true,
        "ready": matches!(overall, Overall::Ready | Overall::Degraded),
        "status": overall.label(),
        "uptimeSecs": state.metrics.uptime().as_secs(),
        "generation": state.config.generation(),
        "models": models,
        // Additive: non-lazy models still downloading or compiling (QA-069).
        "loading": loading,
        "version": crate::VERSION,
        "fileLogCapped": crate::cli::file_log_capped(),
    });
    (overall.http(), Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::metrics::{Metrics, ModelMetrics};
    use crate::registry::{ModelRegistry, RuntimeInfo};
    use crate::server::test_util::*;
    use crate::worker::WorkerHandle;

    fn state_with(models: &[(&str, ModelState)]) -> Arc<AppState> {
        let metrics = Metrics::new("0.0.0");
        let workers = models
            .iter()
            .map(|(n, s)| {
                let mm = Arc::new(ModelMetrics::new(*n, "CPU", ""));
                metrics.models.write().unwrap().push(mm.clone());
                let h = WorkerHandle::failed(*n, "x", mm);
                h.state.set(s.clone());
                h
            })
            .collect();
        let info = RuntimeInfo {
            openvino_version: "t".into(),
            available_devices: vec![],
            has_gpu: false,
        };
        let registry = ModelRegistry::from_handles(workers, Some(0), info);
        let dir = std::env::temp_dir().join(format!("bo_st_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        Arc::new(AppState::new(
            Arc::new(registry),
            Arc::new(metrics),
            Config::default(),
            dir.join("cfg.json"),
        ))
    }

    async fn get(state: &Arc<AppState>) -> (StatusCode, serde_json::Value) {
        let resp = readiness(State(state.clone())).await;
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[test]
    fn overall_table() {
        use Kind::*;
        assert_eq!(overall(&[]), Overall::Failed);
        assert_eq!(overall(&[Failed]), Overall::Failed);
        assert_eq!(overall(&[Ready]), Overall::Ready);
        assert_eq!(overall(&[Ready, Lazy]), Overall::Ready);
        assert_eq!(overall(&[Lazy]), Overall::Ready);
        assert_eq!(overall(&[Ready, Failed]), Overall::Degraded);
        // QA-069: a usable model next to one still loading serves: degraded, 200.
        assert_eq!(overall(&[Ready, Initializing]), Overall::Degraded);
        assert_eq!(overall(&[Lazy, Initializing]), Overall::Degraded);
        assert_eq!(overall(&[Initializing]), Overall::Initializing);
        assert_eq!(overall(&[Failed, Initializing]), Overall::Initializing);
        assert_eq!(overall(&[Lazy, Failed, Initializing]), Overall::Degraded);
        assert_eq!(Overall::Degraded.http(), StatusCode::OK);
        assert_eq!(
            Overall::Initializing.http(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    /// What the tray makes of an answer: its health and status line, through the tray's own
    /// parser and display model.
    fn tray_view(code: StatusCode, body: &[u8]) -> (crate::tray::model::Health, String) {
        use crate::tray::model::{Inputs, display};
        use crate::tray::probe::{HttpStatus, parse_body};
        let report = parse_body(code.as_u16(), body).expect("the tray reads the answer");
        let http = HttpStatus::Answer(report);
        let d = display(&Inputs {
            service: Some(crate::service_ctl::ServiceState::Running),
            http: Some(&http),
            ..Inputs::default()
        });
        (d.health, d.status_line)
    }

    /// QA-206: the tray reads the labels this route writes. Real answers for ready, degraded,
    /// initializing and failed models go through the tray's parser and display model; a lazy
    /// model (no fake worker accepts while initializing) through an answer built from the same
    /// labels. A label renamed on one side only fails here.
    #[tokio::test]
    async fn the_tray_reads_every_label() {
        use crate::tray::model::Health;
        let cases: [(&[(&str, ModelState)], Health); 4] = [
            (&[("a", ModelState::Ready)], Health::Ready),
            (
                &[
                    ("a", ModelState::Ready),
                    ("b", ModelState::Failed("x".into())),
                ],
                Health::Degraded,
            ),
            (&[("a", ModelState::Initializing)], Health::Starting),
            (&[("a", ModelState::Failed("boom".into()))], Health::Failed),
        ];
        for (models, want) in cases {
            let resp = readiness(State(state_with(models))).await;
            let code = resp.status();
            let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            let (health, line) = tray_view(code, &bytes);
            assert_eq!(health, want, "{models:?}: {line}");
        }
        let (_, line) = tray_view(
            StatusCode::OK,
            &serde_json::to_vec(&json!({
                "status": overall(&[Kind::Ready, Kind::Lazy]).label(),
                "models": [
                    {"name": "a", "state": Kind::Ready.label()},
                    {"name": "b", "state": Kind::Lazy.label()},
                ],
            }))
            .unwrap(),
        );
        assert!(line.ends_with("2 models ready"), "{line}");
        let (health, line) = tray_view(
            StatusCode::OK,
            &serde_json::to_vec(&json!({
                "status": overall(&[Kind::Ready, Kind::Initializing]).label(),
                "models": [
                    {"name": "a", "state": Kind::Ready.label()},
                    {"name": "b", "state": Kind::Initializing.label()},
                ],
            }))
            .unwrap(),
        );
        assert_eq!(health, Health::Degraded);
        assert!(line.ends_with("1 of 2 models loading"), "{line}");
    }

    #[tokio::test]
    async fn all_failed_is_503() {
        let (code, v) = get(&test_state()).await;
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(v["status"], "failed");
        assert_eq!(v["ready"], false);
        assert_eq!(v["success"], true);
        assert_eq!(v["models"].as_array().unwrap().len(), 2);
        assert_eq!(v["models"][0]["state"], "failed");
    }

    #[tokio::test]
    async fn no_models_is_failed() {
        let (code, v) = get(&state_with(&[])).await;
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(v["status"], "failed");
    }

    #[tokio::test]
    async fn ready_degraded_initializing() {
        let (code, v) = get(&state_with(&[("a", ModelState::Ready)])).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["status"], "ready");
        assert_eq!(v["ready"], true);
        assert_eq!(v["generation"], 1);
        // Tests log to no file, so the file log is never capped here.
        assert_eq!(v["fileLogCapped"], false);
        assert_eq!(v["loading"], json!([]));

        let (code, v) = get(&state_with(&[
            ("a", ModelState::Ready),
            ("b", ModelState::Failed("boom".into())),
        ]))
        .await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["status"], "degraded");
        assert_eq!(v["models"][1]["detail"], "boom");

        let (code, v) = get(&state_with(&[
            ("a", ModelState::Ready),
            ("b", ModelState::Initializing),
        ]))
        .await;
        // QA-069: "a" serves Blue Iris while "b" downloads or compiles.
        assert_eq!(code, StatusCode::OK);
        assert_eq!(v["status"], "degraded");
        assert_eq!(v["ready"], true);
        assert_eq!(v["models"][1]["state"], "initializing");
        assert_eq!(v["loading"], json!(["b"]));

        let (code, v) = get(&state_with(&[
            ("a", ModelState::Failed("boom".into())),
            ("b", ModelState::Initializing),
        ]))
        .await;
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(v["status"], "initializing");
        assert_eq!(v["ready"], false);
        assert_eq!(v["loading"], json!(["b"]));
    }

    #[tokio::test]
    async fn reports_the_threshold_of_the_loaded_device() {
        let state = state_with(&[("a", ModelState::Ready), ("b", ModelState::Ready)]);
        let workers = state.registry.workers();
        let mut m = crate::config::ModelConfig::default();
        m.device_thresholds.insert("openvino:cpu".into(), 0.3);
        workers[0].set_live(crate::worker::LiveSettings::for_model(
            &Config::default(),
            &m,
        ));
        *workers[0].device.write().unwrap() = Some(crate::backend::DeviceInfo {
            spec: "openvino:cpu".into(),
            ..crate::backend::DeviceInfo::default()
        });
        let (_, v) = get(&state).await;
        assert_eq!(v["models"][0]["threshold"], 0.3);
        assert_eq!(
            v["models"][0]["thresholdSource"],
            "profile for openvino:cpu"
        );
        // Not loaded: the fallback.
        assert_eq!(v["models"][1]["threshold"], 0.5);
        assert_eq!(v["models"][1]["thresholdSource"], "global");
    }
}
