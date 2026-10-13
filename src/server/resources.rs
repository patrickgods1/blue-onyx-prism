//! Resource, model-add and model-export endpoints of the admin API.

use super::apply::apply_config;
use super::config_page::{config_template, save_config};
use super::{AppState, blocking, render, wants_html};
use axum::Json;
use axum::extract::rejection::FormRejection;
use axum::extract::{Form, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

// ---------------------------------------------------------------------------------------------
// Resources (phase 7.5)

/// `GET /v1/resources`: every downloadable resource for this platform with its state, size,
/// what it provides and which models wait for it.
pub(super) async fn resources_json(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    blocking(move || {
        let cfg = state.config_read().clone();
        let rows = crate::resources::status::rows(state.resources.as_ref(), &cfg);
        let hw = crate::backend::detect::hardware();
        let manual = state
            .resources
            .as_ref()
            .and_then(|r| r.provision.resolution.manual_message());
        Json(serde_json::json!({
            "success": true,
            "platform": format!("{}-{}", hw.os, hw.arch),
            "autoDownload": cfg.auto_download,
            "allowLargeDownloads": cfg.allow_large_downloads,
            "downloadRoot": cfg.data_root().display().to_string(),
            "manualMessage": manual,
            "resources": rows,
            "localModels": crate::resources::status::local_models(state.resources.as_ref(), &cfg),
        }))
    })
    .await
}

/// Form (or urlencoded API) fields of the resource actions: `id`, optional `confirm_large`.
fn action_fields(form: &HashMap<String, String>) -> (String, bool) {
    let id = form
        .get("id")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let confirm = form.get("confirm_large").is_some_and(|v| {
        !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off"
        )
    });
    (id, confirm)
}

/// Answer a resource action: the config page (browsers) or `{success, message}` JSON with
/// 200 / 404 / 409 / 500 / 503.
fn action_response(
    state: &AppState,
    headers: &HeaderMap,
    id: &str,
    result: Result<String, crate::resources::actions::ActionError>,
) -> Response {
    action_outcome(
        state,
        headers,
        id,
        result.map_err(|e| {
            (
                StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                e.to_string(),
            )
        }),
    )
}

/// [`action_response`] with the refusal as a status and message.
fn action_outcome(
    state: &AppState,
    headers: &HeaderMap,
    id: &str,
    result: Result<String, (StatusCode, String)>,
) -> Response {
    if let Err((status, e)) = &result {
        warn!(%status, "resource action refused: {e}");
    }
    if wants_html(headers) {
        let (msg, err) = match result {
            Ok(m) => (Some(m), None),
            Err((_, e)) => (None, Some(e)),
        };
        // Shown in the resource's row (the form posts to `#resource-<id>`), else at the top of
        // the Resources section (with that anchor) when the row is gone (added, removed).
        let mut t = config_template(state, msg, err, None);
        let has_row = t
            .resource_groups
            .iter()
            .flat_map(|g| g.rows.iter())
            .any(|r| r.id == id)
            || t.local_models.iter().any(|l| l.id == id)
            || id == crate::resources::actions::CUSTOM_MODEL_ROW;
        t.at = if has_row {
            "resources-row"
        } else {
            "resources"
        };
        t.at_row = id.to_string();
        return render(&t);
    }
    match result {
        Ok(m) => Json(serde_json::json!({"success": true, "message": m})).into_response(),
        Err((status, e)) => (
            status,
            Json(serde_json::json!({"success": false, "message": e})),
        )
            .into_response(),
    }
}

fn no_manager() -> crate::resources::actions::ActionError {
    crate::resources::actions::ActionError::Failed(
        "this server has no download manager (run the service, or use `blue-onyx-prism fetch`)"
            .into(),
    )
}

/// `POST /v1/resources/download` (`id`, `confirm_large` for large resources).
pub(super) async fn resources_download(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    blocking(move || {
        let form = form.map(|Form(f)| f).unwrap_or_default();
        let (id, confirm) = action_fields(&form);
        let cfg = state.config_read().clone();
        let result = match &state.resources {
            Some(ctx) => crate::resources::actions::download(ctx, &cfg, &id, confirm),
            None => Err(no_manager()),
        };
        action_response(&state, &headers, &id, result)
    })
    .await
}

/// `POST /v1/resources/remove` (`id`).
pub(super) async fn resources_remove(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    blocking(move || {
        let form = form.map(|Form(f)| f).unwrap_or_default();
        let (id, _) = action_fields(&form);
        let cfg = state.config_read().clone();
        let result = crate::resources::actions::remove(state.resources.as_ref(), &cfg, &id);
        action_response(&state, &headers, &id, result)
    })
    .await
}

/// `POST /v1/resources/add-to-config` (`id` of an installed model): appends it to `models`,
/// saves the config file and starts its worker (no restart).
pub(super) async fn resources_add_to_config(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    blocking(move || {
        let form = form.map(|Form(f)| f).unwrap_or_default();
        let (id, _) = action_fields(&form);
        let mut outcome = String::new();
        let saved = save_config(&state, "resources add-to-config", |c| {
            outcome = crate::resources::actions::add_to_config(state.resources.as_ref(), c, &id)
                .map_err(|e| anyhow::anyhow!(e))?;
            Ok(())
        });
        let result = match saved {
            Ok(msg) => Ok(format!("{outcome}. {msg}")),
            Err(e) => match e.downcast::<crate::resources::actions::ActionError>() {
                Ok(a) => Err(a),
                Err(e) => Err(crate::resources::actions::ActionError::Failed(format!(
                    "{e:#}"
                ))),
            },
        };
        action_response(&state, &headers, &id, result)
    })
    .await
}

/// `POST /v1/models/add` (`path`, optional `name`, `family`, `classes`): appends a custom model
/// (a file, or the directory holding one) to `models`, saves the config file and starts its
/// worker (no restart).
pub(super) async fn models_add(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    blocking(move || {
        use crate::resources::actions::{ActionError, CUSTOM_MODEL_ROW, CustomModel};
        let form = form.map(|Form(f)| f).unwrap_or_default();
        let field = |k: &str| {
            form.get(k)
                .map(|s| s.trim().to_string())
                .unwrap_or_default()
        };
        let some = |s: String| (!s.is_empty()).then_some(s);
        let family = match field("family").to_ascii_lowercase().as_str() {
            "" => Ok(crate::model::ModelFamilyKind::Auto),
            f => f.parse::<crate::model::ModelFamilyKind>().map_err(|_| {
                ActionError::BadRequest(format!(
                    "unknown family '{f}'; valid: {}",
                    family_names().join(", ")
                ))
            }),
        };
        let req = family.map(|family| CustomModel {
            path: field("path"),
            name: some(field("name")),
            family,
            classes: some(field("classes")),
        });
        let refused = |e: ActionError| {
            (
                StatusCode::from_u16(e.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                e.to_string(),
            )
        };
        // The same root check as the file browser (SV-10): the model and classes files must
        // lie inside a browsable root; the message names only what the client sent.
        let req = req.map_err(refused).and_then(|req| {
            let cfg = state.config_read().clone();
            for raw in std::iter::once(&req.path).chain(req.classes.iter()) {
                if raw.trim().is_empty() {
                    continue;
                }
                super::browse::allowed_path(&cfg, raw).map_err(|e| e.into_parts())?;
            }
            Ok(req)
        });
        let result = match req {
            Err(e) => Err(e),
            Ok(req) => {
                let mut outcome = String::new();
                let saved = save_config(&state, "config page (add model by path)", |c| {
                    outcome = crate::resources::actions::add_custom_model(c, &req)
                        .map_err(|e| anyhow::anyhow!(e))?;
                    Ok(())
                });
                match saved {
                    Ok(msg) => Ok(format!("{outcome}. {msg}")),
                    Err(e) => match e.downcast::<ActionError>() {
                        Ok(a) => Err(refused(a)),
                        Err(e) => Err(refused(ActionError::Failed(format!("{e:#}")))),
                    },
                }
            }
        };
        action_outcome(&state, &headers, CUSTOM_MODEL_ROW, result)
    })
    .await
}

fn family_names() -> Vec<String> {
    crate::model::ModelFamilyKind::ALL
        .iter()
        .map(ToString::to_string)
        .collect()
}

/// Options of the "Add a model by path" family select (auto is the empty default option).
pub(super) fn family_choices() -> Vec<FamilyChoice> {
    use crate::model::ModelFamilyKind as K;
    K::ALL
        .iter()
        .filter(|k| **k != K::Auto)
        .map(|k| FamilyChoice {
            value: k.to_string(),
            label: match k {
                K::Auto => "auto",
                K::Yolo26 => "yolo26 - YOLO26 / YOLOv10 end-to-end [1,300,6]",
                K::Yolo5 => "yolo5 - YOLOv5 [1,N,5+C]",
                K::Yolo8 => "yolo8 - YOLOv8/9/11/12 [1,4+C,8400]",
                K::RtDetr => "rtdetr - RT-DETRv2, DEIMv2",
                K::Detr => "detr - D-FINE (Hugging Face DETR ports)",
                K::RfDetr => "rfdetr - RF-DETR incl. fine-tuned",
            }
            .to_string(),
        })
        .collect()
}

/// One `<option>` of the family select.
pub(super) struct FamilyChoice {
    pub(super) value: String,
    pub(super) label: String,
}

/// `POST /v1/resources/export` (`id` of an exportable model, `confirm_large` when the export
/// toolchain is not installed yet, `add_to_config` to append it to `models` when done): queues the
/// export (one at a time). It runs the model's exporter locally; see `resources::export`.
pub(super) async fn resources_export(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    blocking(move || {
        let form = form.map(|Form(f)| f).unwrap_or_default();
        let (id, confirm) = action_fields(&form);
        let add = form.get("add_to_config").is_some_and(|v| {
            !matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "off"
            )
        });
        let cfg = state.config_read().clone();
        let on_done: Option<crate::resources::export::OnDone> = add.then(|| {
            let config = state.config.clone();
            // Weak: an export can outlive this generation (a restart must not wait for it).
            let generation = Arc::downgrade(&state);
            let id = id.clone();
            Box::new(move |_: &crate::resources::export::ExportOutput| {
                add_export_to_config(&config, &id);
                // Start the model's worker now (a later generation reads the file anyway, and
                // `sync_config` catches up with a generation that started meanwhile).
                if let Some(state) = generation.upgrade() {
                    apply_config(&state);
                }
            }) as crate::resources::export::OnDone
        });
        let result = match &state.resources {
            Some(ctx) => {
                crate::resources::actions::export(ctx, &cfg, &id, confirm, on_done).map(|m| {
                    if add {
                        format!("{m}; it is added to the config when done")
                    } else {
                        m
                    }
                })
            }
            None => Err(no_manager()),
        };
        action_response(&state, &headers, &id, result)
    })
    .await
}

/// After an export: append the model to the config (through the config store, which first
/// picks up changes made to the file meanwhile, writes the file and logs the change).
fn add_export_to_config(config: &crate::config_store::ConfigStore, id: &str) {
    let mut msg = String::new();
    match config.update("resources export", |c| {
        msg = crate::resources::actions::add_to_config(None, c, id)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(())
    }) {
        Ok(_) => info!("{msg} ({})", config.path().display()),
        Err(e) => warn!("not adding {id} to the config: {e:#}"),
    }
}

/// `POST /v1/resources/export/cancel` (`id`): cancels a queued or running export (the exporter
/// process is stopped).
pub(super) async fn resources_export_cancel(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    form: Result<Form<HashMap<String, String>>, FormRejection>,
) -> Response {
    blocking(move || {
        let form = form.map(|Form(f)| f).unwrap_or_default();
        let (id, _) = action_fields(&form);
        let result = match &state.resources {
            Some(ctx) => crate::resources::actions::cancel_export(ctx, &id),
            None => Err(no_manager()),
        };
        action_response(&state, &headers, &id, result)
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    use crate::registry::RuntimeInfo;
    use crate::server::test_util::*;

    use crate::config::Config;
    use crate::metrics::Metrics;
    use crate::registry::ModelRegistry;

    use crate::resources::provision::{Provision, ProvisionOptions, Provisioner};
    use crate::resources::status::ResourcesCtx;

    use axum::http::StatusCode;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[tokio::test]
    async fn resources_download_add_and_remove() {
        let (state, root) = resources_state();
        let j = resources(&state).await;
        assert_eq!(j["success"], true);
        assert_eq!(j["autoDownload"], true);
        let fx = row(&j, "model:fixture-model");
        assert_eq!(fx["state"], "available");
        assert_eq!(fx["kind"], "model");
        assert_eq!(fx["provides"][0], "model:fixture-model");
        assert_eq!(fx["large"], false);
        // The platform's catalog is listed too.
        assert_eq!(fx["group"], "models-yolov5");
        let ov = row(&j, "openvino-runtime");
        assert_eq!(ov["group"], "runtimes");
        assert!(ov["size"].as_u64().unwrap() > 0);
        assert!(ov["provides"][0].as_str().unwrap().starts_with("openvino:"));

        // Unknown ids and large downloads without confirmation are refused.
        let (s, b) = post_action(&state, "/v1/resources/download", "id=nope").await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{b}");
        let (s, b) = post_action(&state, "/v1/resources/download", "id=model%3Abig-model").await;
        assert_eq!(s, StatusCode::CONFLICT, "{b}");
        assert!(b["message"].as_str().unwrap().contains("confirm"), "{b}");
        assert_eq!(
            row(&resources(&state).await, "model:big-model")["large"],
            true
        );

        // Download the fixture and wait until it is installed.
        let (s, b) =
            post_action(&state, "/v1/resources/download", "id=model%3Afixture-model").await;
        assert_eq!(s, StatusCode::OK, "{b}");
        let started = Instant::now();
        loop {
            let j = resources(&state).await;
            let st = row(&j, "model:fixture-model")["state"]
                .as_str()
                .unwrap()
                .to_string();
            if st == "installed" {
                break;
            }
            assert!(st != "failed", "{j}");
            assert!(started.elapsed() < Duration::from_secs(20), "stuck in {st}");
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(root.join("models/fixture-model.onnx").is_file());
        let fx = row(&resources(&state).await, "model:fixture-model").clone();
        assert_eq!(fx["removable"], true);
        assert_eq!(fx["can_add_to_config"], true);

        // Add to config: saved, and the row no longer offers it.
        let (s, b) = post_action(
            &state,
            "/v1/resources/add-to-config",
            "id=model%3Afixture-model",
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert!(
            b["message"]
                .as_str()
                .unwrap()
                .contains("added 'fixture-model'"),
            "{b}"
        );
        assert!(
            state
                .config_read()
                .models
                .iter()
                .any(|m| m.effective_name() == "fixture-model")
        );
        let saved = Config::load(&state.config_path).unwrap();
        assert_eq!(saved.models.len(), 2);
        assert_eq!(
            row(&resources(&state).await, "model:fixture-model")["can_add_to_config"],
            false
        );

        // The new entry is disabled (another model was enabled), so it can be removed.
        let (s, b) = post_action(&state, "/v1/resources/remove", "id=model%3Afixture-model").await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert!(!root.join("models/fixture-model.onnx").exists());
        assert_eq!(
            row(&resources(&state).await, "model:fixture-model")["state"],
            "available"
        );

        // A loaded runtime is not removed: restart required.
        std::fs::create_dir_all(root.join("openvino")).unwrap();
        std::fs::write(
            root.join("openvino/.installed.json"),
            r#"{"id":"openvino-runtime","version":"x","sha256":[],"files":[]}"#,
        )
        .unwrap();
        let (s, b) = post_action(&state, "/v1/resources/remove", "id=openvino-runtime").await;
        assert_eq!(s, StatusCode::CONFLICT, "{b}");
        assert!(
            b["message"].as_str().unwrap().contains("restart required"),
            "{b}"
        );
        assert!(root.join("openvino/.installed.json").exists());
        let ov = row(&resources(&state).await, "openvino-runtime").clone();
        assert_eq!(ov["state"], "installed");
        assert_eq!(ov["loaded"], true);
        assert_eq!(ov["removable"], false);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn local_models_listed_and_added() {
        let (state, root) = resources_state();
        let models = root.join("models");
        std::fs::create_dir_all(&models).unwrap();
        for f in [
            "mynet.xml",
            "mynet.bin",
            "mynet.onnx",
            "custom.xml",
            "custom.bin",
            "other.onnx",
            "IPcam-general.onnx",
            "yolo26s.onnx",
            "notes.txt",
        ] {
            std::fs::write(models.join(f), b"x").unwrap();
        }
        let j = resources(&state).await;
        let local = j["localModels"].as_array().unwrap();
        let ids: Vec<&str> = local.iter().map(|l| l["id"].as_str().unwrap()).collect();
        // `other.onnx` is configured, IPcam-general is a catalog download, yolo26s.onnx is an
        // export (its own row); .onnx wins over .xml.
        assert_eq!(ids, ["local:custom.xml", "local:mynet.onnx"], "{j}");
        assert_eq!(local[1]["also"], "mynet.xml");
        assert_eq!(local[1]["family"], "auto");
        assert_eq!(local[1]["group"], "local-models");
        assert_eq!(local[0]["format"], "openvino-ir");

        let (_, html) = get_text(&state, "/config").await;
        assert!(html.contains("value=\"local:mynet.onnx\""), "{html}");
        assert!(html.contains("<h3 id=\"res-runtimes\">Runtimes</h3>"));

        let (s, b) = post_action(
            &state,
            "/v1/resources/add-to-config",
            "id=local%3Amynet.onnx",
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert!(b["message"].as_str().unwrap().contains("added 'mynet'"));
        let last = state.config.changes(1).pop().expect("logged");
        assert_eq!(last.source, "resources add-to-config");
        assert!(last.summary.contains("mynet added"), "{}", last.summary);
        let saved = Config::load(&state.config_path).unwrap();
        let m = saved
            .models
            .iter()
            .find(|m| m.effective_name() == "mynet")
            .unwrap();
        assert_eq!(m.path, PathBuf::from("models/mynet.onnx"));
        assert_eq!(m.family, crate::model::ModelFamilyKind::Auto);
        let j = resources(&state).await;
        let ids: Vec<&str> = j["localModels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["local:custom.xml"]);

        for bad in [
            "id=local%3A..%2Fx.onnx",
            "id=local%3Anotes.txt",
            "id=local%3Amissing.onnx",
        ] {
            let (s, b) = post_action(&state, "/v1/resources/add-to-config", bad).await;
            assert_eq!(s, StatusCode::NOT_FOUND, "{bad}: {b}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn add_model_by_path() {
        let (state, root) = resources_state();
        let dir = root.join("rf");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("inference_model.onnx"), b"x").unwrap();
        std::fs::write(dir.join("classes.yaml"), b"names:\n- a\n").unwrap();
        let enc =
            |p: &std::path::Path| p.to_string_lossy().replace('\\', "%5C").replace('/', "%2F");

        let (_, html) = get_text(&state, "/config").await;
        assert!(html.contains("action=\"/v1/models/add"), "{html}");
        assert!(html.contains("<option value=\"rfdetr\">"));

        let body = format!("path={}&name=mine&family=rfdetr&classes=", enc(&dir));
        let (s, b) = post_action(&state, "/v1/models/add", &body).await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert!(
            b["message"].as_str().unwrap().contains("added 'mine'"),
            "{b}"
        );
        let last = state.config.changes(1).pop().expect("logged");
        assert_eq!(last.source, "config page (add model by path)");
        let saved = Config::load(&state.config_path).unwrap();
        let m = saved
            .models
            .iter()
            .find(|m| m.effective_name() == "mine")
            .unwrap();
        assert_eq!(m.family, crate::model::ModelFamilyKind::RfDetr);
        assert_eq!(m.path, PathBuf::from("rf/inference_model.onnx"));
        assert_eq!(m.classes, Some(PathBuf::from("rf/classes.yaml")));

        let (s, _) = post_action(&state, "/v1/models/add", &body).await;
        assert_eq!(s, StatusCode::CONFLICT);
        // Outside every browsable root (SV-10): refused before the file is looked at.
        let (s, b) = post_action(&state, "/v1/models/add", "path=%2Fno%2Fsuch.onnx").await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{b}");
        assert!(
            b["message"]
                .as_str()
                .unwrap()
                .contains("outside the browsable folders"),
            "{b}"
        );
        let (s, _) = post_action(&state, "/v1/models/add", "path=rf%2Fmissing.onnx").await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        let (s, b) = post_action(
            &state,
            "/v1/models/add",
            &format!("path={}&classes=%2Fetc%2Fpasswd", enc(&dir)),
        )
        .await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{b}");
        let (s, b) = post_action(&state, "/v1/models/add", "path=..%2Fx.onnx").await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{b}");
        let (s, b) = post_action(
            &state,
            "/v1/models/add",
            &format!("path={}&name=zz&family=nope", enc(&dir)),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{b}");
        assert!(b["message"].as_str().unwrap().contains("rfdetr"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A server whose exporter uses the fake toolchain (no network, no Python). The config has an
    /// enabled `other.onnx` and a `yolo26s` entry whose file is not exported yet.
    fn export_state(fake: Arc<crate::resources::export::fake::Fake>) -> (Arc<AppState>, PathBuf) {
        let root = std::env::temp_dir().join(format!("bop-srv-export-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("models")).unwrap();
        let config = Config {
            download_dir: Some(root.clone()),
            models: vec![
                ModelConfig {
                    path: "models/other.onnx".into(),
                    ..Default::default()
                },
                ModelConfig {
                    path: "models/yolo26s.onnx".into(),
                    family: crate::model::ModelFamilyKind::Yolo26,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let provisioner = Arc::new(Provisioner::with_options(ProvisionOptions {
            export_hooks: Some((fake.clone(), fake)),
            ..ProvisionOptions::default()
        }));
        let registry = ModelRegistry::from_handles(
            vec![],
            None,
            RuntimeInfo {
                openvino_version: "2026.0.0-test".into(),
                available_devices: vec!["CPU".into()],
                has_gpu: false,
            },
        );
        let mut state = AppState::new(
            Arc::new(registry),
            Arc::new(Metrics::new("0.0.0")),
            config.clone(),
            root.join("cfg.json"),
        );
        // What the runner plans for this config (the resolver's "needs export").
        let hw = crate::backend::detect::hardware();
        let installed = crate::resources::detect_installed(&config, &root, None);
        let resolution = crate::resources::needed(&config, hw, &installed);
        let mut provision = Provision::none(&config);
        provision.resolution = resolution;
        state.resources = Some(ResourcesCtx {
            provisioner,
            provision: Arc::new(provision),
            openvino_loaded: true,
            ort_loaded: None,
        });
        (Arc::new(state), root)
    }

    fn export_supported() -> bool {
        let p = crate::resources::catalog::Platform::current();
        crate::resources::export::unsupported_reason(p.os, p.arch).is_none()
    }

    fn exporter(state: &AppState) -> Arc<crate::resources::export::Exporter> {
        state
            .resources
            .as_ref()
            .unwrap()
            .provisioner
            .exporter()
            .expect("exporter started")
    }

    #[tokio::test]
    async fn yolo26_export_endpoints() {
        if !export_supported() {
            return;
        }
        let fake = Arc::new(crate::resources::export::fake::Fake::new());
        let (state, root) = export_state(fake.clone());
        let j = resources(&state).await;
        let n = row(&j, "model:yolo26n");
        assert_eq!(n["group"], "models-export");
        assert_eq!(n["action"], "export");
        // Every exportable model has an Export row; the DEIMv2 source is part of the toolchain.
        for r in crate::resources::catalog::EXPORT_MODELS {
            let e = row(&j, r.id);
            assert_eq!(e["action"], "export", "{}", r.id);
            assert_eq!(e["group"], "models-export", "{}", r.id);
            assert!(e["confirm"].as_str().unwrap().contains(r.title), "{e}");
        }
        assert!(
            !j["resources"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r["id"] == crate::resources::catalog::DEIMV2_SRC_ID)
        );
        let d = row(&j, "model:deimv2-n");
        assert!(
            d["confirm"].as_str().unwrap().contains("DEIMv2 code"),
            "{d}"
        );
        assert_eq!(row(&j, "model:rfdetr-large")["license"], "Apache-2.0");
        assert_eq!(n["state"], "available", "{n}");
        assert_eq!(n["large"], true, "toolchain not installed yet");
        assert!(n["confirm"].as_str().unwrap().contains("AGPL-3.0"));
        assert!(
            n["confirm"]
                .as_str()
                .unwrap()
                .contains("runs Ultralytics' exporter locally")
        );
        // The configured, not yet exported yolo26s is "needs export", never auto-exported.
        let s_row = row(&j, "model:yolo26s");
        assert_eq!(s_row["state"], "needs-export", "{s_row}");
        assert_eq!(s_row["blocking"], serde_json::json!(["yolo26s"]));
        let uv = row(&j, "tool:uv");
        assert_eq!(uv["action"], "toolchain");
        assert_eq!(uv["state"], "available");
        assert_eq!(uv["removable"], false);

        // The Download endpoint refuses exports; Export without confirmation is refused too.
        let (s, b) = post_action(&state, "/v1/resources/download", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::CONFLICT, "{b}");
        let (s, b) = post_action(&state, "/v1/resources/export", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::CONFLICT, "{b}");
        assert!(b["message"].as_str().unwrap().contains("confirm"), "{b}");
        assert!(fake.cmds().is_empty());
        let (s, _) = post_action(&state, "/v1/resources/export", "id=model%3Anope").await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        let (s, _) = post_action(&state, "/v1/resources/export", "id=openvino-runtime").await;
        assert_eq!(s, StatusCode::CONFLICT);

        // Confirmed export, added to the config when done.
        let (s, b) = post_action(
            &state,
            "/v1/resources/export",
            "id=model%3Ayolo26n&confirm_large=1&add_to_config=1",
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{b}");
        let st = exporter(&state)
            .wait("model:yolo26n", Some(Duration::from_secs(20)))
            .unwrap();
        assert_eq!(
            st.state,
            crate::resources::export::ExportState::Installed,
            "{st:?}"
        );
        assert!(root.join("models/yolo26n.onnx").is_file());
        assert!(root.join("models/yolo26n.yaml").is_file());
        let last = state.config.changes(1).pop().expect("logged");
        assert_eq!(last.source, "resources export");
        // Memory and file agree (the in-memory config got the model too).
        assert_eq!(
            *state.config_read(),
            Config::load(&state.config_path).unwrap()
        );
        let saved = Config::load(&state.config_path).unwrap();
        let m = saved
            .models
            .iter()
            .find(|m| m.effective_name() == "yolo26n")
            .expect("added to the config file");
        assert_eq!(m.path, PathBuf::from("models/yolo26n.onnx"));
        assert_eq!(m.family, crate::model::ModelFamilyKind::Yolo26);
        assert_eq!(m.classes, Some(PathBuf::from("models/yolo26n.yaml")));
        assert!(!m.enabled, "another model is enabled");

        let j = resources(&state).await;
        let n = row(&j, "model:yolo26n");
        assert_eq!(n["state"], "installed", "{n}");
        assert_eq!(n["state_text"], "exported");
        assert_eq!(n["can_add_to_config"], false, "already in the config");
        assert_eq!(n["removable"], true);
        assert_eq!(n["large"], false, "toolchain installed now");
        let uv = row(&j, "tool:uv");
        assert_eq!(uv["state"], "installed", "{uv}");
        assert_eq!(uv["removable"], true);
        // Not listed again as a local model.
        assert!(j["localModels"].as_array().unwrap().is_empty(), "{j}");

        // Re-export of an exported model is a no-op; the second size needs no confirmation now.
        let (s, b) = post_action(&state, "/v1/resources/export", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::OK);
        assert!(b["message"].as_str().unwrap().contains("already exported"));
        let (s, b) = post_action(&state, "/v1/resources/export", "id=model%3Ayolo26s").await;
        assert_eq!(s, StatusCode::OK, "{b}");
        exporter(&state).wait("model:yolo26s", Some(Duration::from_secs(20)));
        assert!(root.join("models/yolo26s.onnx").is_file());

        // yolo26s is used by an enabled model: Remove is refused. yolo26n can go.
        let (s, b) = post_action(&state, "/v1/resources/remove", "id=model%3Ayolo26s").await;
        assert_eq!(s, StatusCode::CONFLICT, "{b}");
        let (s, b) = post_action(&state, "/v1/resources/remove", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert!(!root.join("models/yolo26n.onnx").exists());
        assert!(!root.join("models/.yolo26n.installed.json").exists());
        // Remove export toolchain.
        let (s, b) = post_action(&state, "/v1/resources/remove", "id=tool%3Auv").await;
        assert_eq!(s, StatusCode::OK, "{b}");
        assert!(!root.join("tools/export-env").exists());
        assert!(!root.join("tools/uv").exists());
        let j = resources(&state).await;
        assert_eq!(row(&j, "tool:uv")["state"], "available");
        assert_eq!(row(&j, "model:yolo26n")["state"], "available");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn yolo26_export_cancel_and_page() {
        if !export_supported() {
            return;
        }
        let fake = Arc::new(crate::resources::export::fake::Fake {
            hang_on: Some("export_model.py"),
            ..crate::resources::export::fake::Fake::new()
        });
        let (state, root) = export_state(fake);
        {
            // allow_large_downloads: no confirmation needed.
            state
                .config
                .update("test", |c| {
                    c.allow_large_downloads = true;
                    Ok(())
                })
                .unwrap();
        }
        let (s, _) = post_action(&state, "/v1/resources/cancel", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::NOT_FOUND, "no such endpoint");
        let (s, b) = post_action(&state, "/v1/resources/export/cancel", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::CONFLICT, "nothing to cancel: {b}");
        let (s, b) = post_action(&state, "/v1/resources/export", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::OK, "{b}");
        // Wait until it hangs in the exporter: the row is busy with stage text and progress.
        let deadline = Instant::now() + Duration::from_secs(20);
        let n = loop {
            let j = resources(&state).await;
            let n = row(&j, "model:yolo26n").clone();
            if n["state_text"]
                .as_str()
                .unwrap()
                .starts_with("exporting: exporting to ONNX (5/6)")
            {
                break n;
            }
            assert!(Instant::now() < deadline, "{n}");
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        assert_eq!(n["state"], "exporting");
        assert_eq!(n["busy"], true);
        assert_eq!(n["cancellable"], true);
        assert!(n["progress"].as_u64().unwrap() >= 78, "{n}");
        assert!(!n["log"].as_array().unwrap().is_empty());
        // Busy: the toolchain cannot be removed, the page shows Cancel and the group.
        let j = resources(&state).await;
        assert_eq!(row(&j, "tool:uv")["removable"], false);
        let (_, html) = get_text(&state, "/config").await;
        assert!(
            html.contains("exported on this machine (Ultralytics YOLO, RF-DETR L, DEIMv2)"),
            "{html}"
        );
        assert!(html.contains("action=\"/v1/resources/export/cancel#resource-"));
        assert!(html.contains("needs export"), "Models card marks yolo26s");
        assert!(html.contains("href=\"#res-models-export\""));
        assert!(html.contains("Remove export toolchain") || html.contains("an export is running"));
        let (s, b) = post_action(&state, "/v1/resources/remove", "id=tool%3Auv").await;
        assert_eq!(s, StatusCode::CONFLICT, "{b}");

        let (s, b) = post_action(&state, "/v1/resources/export/cancel", "id=model%3Ayolo26n").await;
        assert_eq!(s, StatusCode::OK, "{b}");
        let st = exporter(&state)
            .wait("model:yolo26n", Some(Duration::from_secs(20)))
            .unwrap();
        assert_eq!(st.state, crate::resources::export::ExportState::Cancelled);
        let j = resources(&state).await;
        let n = row(&j, "model:yolo26n");
        assert_eq!(n["state"], "available");
        assert_eq!(n["state_text"], "export cancelled");
        assert!(!root.join("models/yolo26n.onnx").exists());
        // The page renders the Export button with the AGPL confirm.
        let (_, html) = get_text(&state, "/config").await;
        assert!(html.contains("action=\"/v1/resources/export#resource-model:yolo26n\""));
        assert!(
            html.contains("data-confirm=\"Export YOLO26 nano?"),
            "{html}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
