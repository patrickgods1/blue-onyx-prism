//! `GET /v1/diagnostics` (admin): the diagnostics zip of the running service
//! ([`crate::diagnostics`]), with the live parts only the service has: the `/v1/devices` and
//! `/v1/resources` answers, the metrics and the in-memory log.

use super::{AppState, blocking};
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use std::sync::Arc;

/// Largest JSON answer of a sibling handler read back into the bundle.
const PART_LIMIT: usize = 16 * 1024 * 1024;

/// The JSON body of a sibling handler's answer (None when it is not JSON).
async fn json_of(resp: Response) -> Option<serde_json::Value> {
    let bytes = axum::body::to_bytes(resp.into_body(), PART_LIMIT)
        .await
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// The in-memory log as text, one event per line, oldest first.
fn recent_log(state: &AppState) -> Option<String> {
    let page = state.log_buffer()?.since(0, None, usize::MAX);
    Some(
        page.entries
            .iter()
            .map(|e| format!("{} {:5} {}: {}\n", e.ts, e.level, e.target, e.message))
            .collect(),
    )
}

pub(super) async fn download(State(state): State<Arc<AppState>>) -> Response {
    let devices = super::devices_json(State(state.clone())).await.0;
    let resources = json_of(super::resources_json(State(state.clone())).await).await;
    let metrics = crate::metrics::render_prometheus(&state.metrics, &state.registry.gauges());
    let live = crate::diagnostics::Live {
        devices: Some(devices),
        resources,
        metrics: Some(metrics),
        recent_logs: recent_log(&state),
        notes: Vec::new(),
    };
    let built = blocking(move || {
        let config = state.config_read().clone();
        crate::diagnostics::build(&config, &state.config_path, live)
    })
    .await;
    match built {
        Ok(bytes) => (
            [
                (header::CONTENT_TYPE, "application/zip".to_string()),
                (
                    header::CONTENT_DISPOSITION,
                    format!(
                        "attachment; filename=\"{}\"",
                        crate::diagnostics::file_name()
                    ),
                ),
                (header::CACHE_CONTROL, "no-store".to_string()),
            ],
            bytes,
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({
                "success": false,
                "message": format!(
                    "could not build the diagnostics zip: {e:#}. Try `blue-onyx-prism \
                     diagnostics --out <dir>` on this machine, or check the log"
                ),
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use crate::diagnostics::tests::{ADMIN, HF, entries, planted};
    use crate::server::test_util::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode, header};

    /// The admin route answers a zip with the live parts, and never the planted tokens; a remote
    /// peer without the token is refused like every admin route.
    #[tokio::test]
    async fn diagnostics_route_answers_a_zip_without_secrets() {
        let state = test_state();
        let root = state.config_read().data_root();
        planted(&root);
        // `planted` saved its own config over the state's file; the state's live config
        // points its log dir at the planted logs too.
        state
            .config
            .update("test", |c| {
                c.log_path = Some(root.join("logs"));
                Ok(())
            })
            .unwrap();
        let (s, h, body) = call(
            &state,
            Request::get("/v1/diagnostics").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(h[header::CONTENT_TYPE], "application/zip");
        assert!(
            h[header::CONTENT_DISPOSITION]
                .to_str()
                .unwrap()
                .starts_with("attachment; filename=\"blue-onyx-prism-diagnostics-")
        );
        let all = entries(&body);
        let names: Vec<&str> = all.iter().map(|(n, _)| n.as_str()).collect();
        for want in [
            "devices.json",
            "resources.json",
            "metrics.txt",
            "config.json",
            "logs/blue_onyx_prism.log.2026-11-07",
        ] {
            assert!(names.contains(&want), "{want} missing from {names:?}");
        }
        let devices: serde_json::Value =
            serde_json::from_str(&all.iter().find(|(n, _)| n == "devices.json").unwrap().1)
                .unwrap();
        assert_eq!(devices["runtimes"]["openvino"]["version"], "2026.0.0-test");
        for (name, text) in &all {
            assert!(!text.contains(ADMIN) && !text.contains(HF), "{name}");
        }
        let _ = std::fs::remove_dir_all(&root);

        let remote = test_state();
        let resp = tower::ServiceExt::oneshot(
            crate::server::router(remote.clone()),
            from_peer(
                Request::get("/v1/diagnostics").body(Body::empty()).unwrap(),
                REMOTE,
            ),
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let _ = std::fs::remove_dir_all(remote.config_read().data_root());
    }
}
