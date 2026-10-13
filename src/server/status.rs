//! `GET /v1/status`: machine-readable readiness for container orchestrators, systemd wrappers
//! and Blue Iris watchdogs. Cheap (no file I/O) and open to every client; `curl -f` works
//! because the HTTP status is 503 while the service cannot serve detections.

use super::{AppState, state_view};
use crate::startup::ModelState;
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
            Self::Ready => "ready",
            Self::Initializing => "initializing",
            Self::Degraded => "degraded",
            Self::Failed => "failed",
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

/// Combine model states. A lazy model (loads on its first request) can serve, so it counts as
/// usable like a Ready one but never blocks readiness. Usable and no non-lazy model still
/// initializing = ready (or degraded when another failed). Nothing usable but something
/// initializing = initializing. Otherwise failed (also with no models).
pub(super) fn overall(kinds: &[Kind]) -> Overall {
    let has = |k| kinds.contains(&k);
    let usable = has(Kind::Ready) || has(Kind::Lazy);
    if has(Kind::Initializing) {
        Overall::Initializing
    } else if !usable {
        Overall::Failed
    } else if has(Kind::Failed) {
        Overall::Degraded
    } else {
        Overall::Ready
    }
}

#[allow(clippy::unused_async, reason = "axum handlers are async")]
pub(super) async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    let workers = state.registry.workers();
    let mut kinds = Vec::with_capacity(workers.len());
    let mut models = Vec::with_capacity(workers.len());
    for w in &workers {
        let (kind, label, detail) = match w.state.get() {
            ModelState::Ready => (Kind::Ready, "ready", None),
            ModelState::Failed(m) => (Kind::Failed, "failed", Some(m)),
            ModelState::Initializing if w.accepts_while_initializing() => {
                (Kind::Lazy, "lazy", None)
            }
            ModelState::Initializing => (Kind::Initializing, "initializing", state_view(w).2),
        };
        kinds.push(kind);
        models.push(json!({
            "name": w.name,
            "state": label,
            "device": w.metrics.device(),
            "detail": detail,
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
        "version": crate::VERSION,
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
        assert_eq!(overall(&[Ready, Initializing]), Overall::Initializing);
        assert_eq!(overall(&[Failed, Initializing]), Overall::Initializing);
        assert_eq!(Overall::Degraded.http(), StatusCode::OK);
        assert_eq!(
            Overall::Initializing.http(),
            StatusCode::SERVICE_UNAVAILABLE
        );
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
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(v["status"], "initializing");
        assert_eq!(v["ready"], false);
    }
}
