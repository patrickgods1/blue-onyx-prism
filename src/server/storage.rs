//! The Storage card of the Config page's Resources tab. Admin only.
//!
//! - `GET /v1/storage`: the card ([`crate::storage::usage::Report`]): "safe to clear" and "your
//!   data", with sizes. Measured on its own thread and kept a minute; while a measurement runs the
//!   answer has `computing: true` (and the previous report, if any), and the page asks again.
//! - `POST /v1/storage/clear` (`id`: `cache:<name>`, `staging`, `work`, `logs`): one Clear. Only
//!   rebuildable files go; a crashed auto-label's finished frames are written to their label files
//!   before its folder goes.
//! - `POST /v1/storage/exports/remove` (`name`): delete one review export; refused while a queued
//!   or running training run reads it.

use super::{AppState, blocking};
use crate::resources::catalog::format_size;
use crate::storage::{self, sweep, usage};
use axum::Json;
use axum::extract::rejection::FormRejection;
use axum::extract::{Form, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// How long `GET /v1/storage` waits for a measurement before answering `computing`.
const WAIT: Duration = Duration::from_millis(1500);

fn reply(code: StatusCode, v: serde_json::Value) -> Response {
    (code, Json(v)).into_response()
}

/// `GET /v1/storage`.
pub(super) async fn status(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        let s = sweep::Settings::for_config(&state.config_read());
        let runs = super::training::records(&state);
        let (report, computing) = usage::current(s, runs, WAIT);
        reply(
            StatusCode::OK,
            serde_json::json!({
                "success": true,
                "computing": computing,
                "report": report.as_deref(),
            }),
        )
    })
    .await
}

fn field(form: Result<Form<HashMap<String, String>>, FormRejection>, key: &str) -> String {
    form.map(|Form(f)| f)
        .unwrap_or_default()
        .get(key)
        .map(|v| v.trim().to_string())
        .unwrap_or_default()
}

/// `POST /v1/storage/clear` (`id`).
pub(super) async fn clear(
    State(state): State<Arc<AppState>>,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    let id = field(form, "id");
    blocking(move || {
        let Some(what) = sweep::Clear::parse(&id) else {
            return reply(
                StatusCode::BAD_REQUEST,
                serde_json::json!({
                    "success": false,
                    "message": format!(
                        "'{id}' is not something the Storage card can clear; reload the page"
                    ),
                }),
            );
        };
        let s = sweep::Settings::for_config(&state.config_read());
        let result = sweep::clear(&s, &what);
        usage::invalidate();
        match result {
            Ok(rep) => {
                let freed = storage::total(&rep.removed);
                let mut message = if rep.removed.is_empty() {
                    "nothing to clear".to_string()
                } else {
                    format!(
                        "removed {} item(s), {} freed",
                        rep.removed.len(),
                        format_size(freed)
                    )
                };
                if !rep.kept.is_empty() {
                    message.push_str(&format!("; kept: {}", rep.kept.join("; ")));
                }
                reply(
                    StatusCode::OK,
                    serde_json::json!({
                        "success": rep.kept.is_empty(),
                        "message": message,
                        "freed": freed,
                        "removed": rep.removed,
                    }),
                )
            }
            Err(e) => reply(
                StatusCode::CONFLICT,
                serde_json::json!({"success": false, "message": format!("{e:#}")}),
            ),
        }
    })
    .await
}

/// `POST /v1/storage/exports/remove` (`name`).
pub(super) async fn remove_export(
    State(state): State<Arc<AppState>>,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    let name = field(form, "name");
    blocking(move || {
        let config = state.config_read().clone();
        let runs = super::training::records(&state);
        let training = crate::training::training_dir(&config);
        let configured: Vec<_> = config
            .training
            .datasets
            .iter()
            .map(|d| config.data_path(d))
            .collect();
        let result = sweep::remove_export(&training, &name, &runs, &configured);
        usage::invalidate();
        match result {
            Ok(r) => reply(
                StatusCode::OK,
                serde_json::json!({
                    "success": true,
                    "message": format!("removed the export {name} ({} freed)", format_size(r.bytes)),
                    "freed": r.bytes,
                }),
            ),
            Err(e) => reply(
                StatusCode::CONFLICT,
                serde_json::json!({"success": false, "message": format!("{e:#}")}),
            ),
        }
    })
    .await
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::test_util::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};

    fn post(uri: &str, body: &str) -> Request<Body> {
        Request::post(uri)
            .header("content-type", "application/x-www-form-urlencoded")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn the_card_lists_clears_and_removes() {
        let state = test_state();
        let (root, training) = {
            let c = state.config_read();
            (c.data_root(), crate::training::training_dir(&c))
        };
        let staging = root.join("models/.m.train-staging/x.onnx");
        std::fs::create_dir_all(staging.parent().unwrap()).unwrap();
        std::fs::write(&staging, b"12345").unwrap();
        let export = training.join("exports/e1/classes.txt");
        std::fs::create_dir_all(export.parent().unwrap()).unwrap();
        std::fs::write(&export, b"person\n").unwrap();

        let (code, _, body) = call(
            &state,
            Request::get("/v1/storage").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let report = &v["report"];
        assert!(
            report["safe"]
                .as_array()
                .unwrap()
                .iter()
                .any(|i| i["id"] == "staging" && i["action"] == "clear"),
            "{v}"
        );
        assert!(
            report["data"]
                .as_array()
                .unwrap()
                .iter()
                .any(|i| i["id"] == "export:e1" && i["action"] == "remove"),
            "{v}"
        );

        let (code, _, body) = call(&state, post("/v1/storage/clear", "id=staging")).await;
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(code, StatusCode::OK, "{v}");
        assert_eq!(v["freed"], 5);
        assert!(!staging.exists());

        let (code, _, _) = call(&state, post("/v1/storage/clear", "id=cache%3A..%2Fmodels")).await;
        assert_eq!(code, StatusCode::BAD_REQUEST);

        let (code, _, _) = call(
            &state,
            post("/v1/storage/exports/remove", "name=..%2Fexports"),
        )
        .await;
        assert_eq!(code, StatusCode::CONFLICT);
        assert!(export.is_file());
        let (code, _, body) = call(&state, post("/v1/storage/exports/remove", "name=e1")).await;
        assert_eq!(code, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
        assert!(!export.exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
