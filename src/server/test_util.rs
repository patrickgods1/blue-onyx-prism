//! Shared helpers of the `server` tests.

use super::*;
use crate::config::ModelConfig;
use crate::metrics::ModelMetrics;
use crate::registry::RuntimeInfo;
use axum::body::Body;
use axum::http::Request;
use tower::ServiceExt;

/// State with two models that failed to load (no OpenVINO needed); the second name contains
/// markup to check escaping.
pub(super) fn test_state() -> Arc<AppState> {
    test_state_with(|_| {})
}

/// [`test_state`] with `edit` applied to its config.
pub(super) fn test_state_with(edit: impl FnOnce(&mut Config)) -> Arc<AppState> {
    let dir = std::env::temp_dir().join(format!("bo_srv_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let names = ["ipcam-general", "<b>evil</b>"];
    let metrics = Metrics::new("0.0.0");
    let workers = names
        .iter()
        .map(|n| {
            let mm = Arc::new(ModelMetrics::new(*n, "CPU", ""));
            metrics.models.write().unwrap().push(mm.clone());
            WorkerHandle::failed(*n, "model file not found: x.onnx", mm)
        })
        .collect();
    let info = RuntimeInfo {
        openvino_version: "2026.0.0-test".into(),
        available_devices: vec!["CPU".into()],
        has_gpu: false,
    };
    let registry = ModelRegistry::from_handles(workers, Some(0), info);
    let mut config = Config {
        models: names
            .iter()
            .map(|n| ModelConfig {
                name: Some(n.to_string()),
                path: format!("models/{n}.onnx").into(),
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    edit(&mut config);
    Arc::new(AppState::new(
        Arc::new(registry),
        Arc::new(metrics),
        config,
        dir.join("cfg.json"),
    ))
}

pub(super) async fn call(
    state: &Arc<AppState>,
    req: Request<Body>,
) -> (StatusCode, HeaderMap, Vec<u8>) {
    let resp = router(state.clone())
        .oneshot(from_peer(req, LOCAL))
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

pub(super) const LOCAL: &str = "127.0.0.1:50000";
pub(super) const REMOTE: &str = "10.0.0.5:50000";

/// `req` as if it came from `peer` (what `serve_on` adds), unless it already names one.
pub(crate) fn from_peer(mut req: Request<Body>, peer: &str) -> Request<Body> {
    let ext = req.extensions_mut();
    if ext
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .is_none()
    {
        ext.insert(axum::extract::ConnectInfo::<std::net::SocketAddr>(
            peer.parse().unwrap(),
        ));
    }
    req
}

/// `/v1/vision/custom/list` names through `app`.
pub(super) async fn listed(app: &Router) -> Vec<String> {
    let req = Request::get("/v1/vision/custom/list")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let b = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
    serde_json::from_value(v["models"].clone()).unwrap()
}

pub(super) async fn get_text(state: &Arc<AppState>, uri: &str) -> (StatusCode, String) {
    let (s, _, b) = call(state, Request::get(uri).body(Body::empty()).unwrap()).await;
    (s, String::from_utf8(b).unwrap())
}

pub(super) fn post_form(uri: &str, body: &str, accept_html: bool) -> Request<Body> {
    let mut b =
        Request::post(uri).header(header::CONTENT_TYPE, "application/x-www-form-urlencoded");
    if accept_html {
        b = b.header(header::ACCEPT, "text/html");
    }
    b.body(Body::from(body.to_string())).unwrap()
}

pub(super) fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

pub(super) fn multipart(fields: &[(&str, &[u8])]) -> Request<Body> {
    let boundary = "XBOUNDARYX";
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        if *name == "image" {
            body.extend_from_slice(
                b"Content-Disposition: form-data; name=\"image\"; filename=\"t.jpg\"\r\n\
                  Content-Type: image/jpeg\r\n\r\n",
            );
        } else {
            body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            );
        }
        body.extend_from_slice(value);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    Request::post("/test")
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap()
}

use crate::resources::catalog::{Layout, Part, Provides, Resource, ResourceKind};
use crate::resources::provision::{Provision, ProvisionOptions, Provisioner};
use crate::resources::status::ResourcesCtx;

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// Serve `files` (name -> bytes) on 127.0.0.1; returns the base URL.
fn serve_files(files: Vec<(&'static str, Vec<u8>)>) -> String {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let mut head = Vec::new();
            let mut b = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                if s.read(&mut b).map_or(true, |n| n == 0) {
                    break;
                }
                head.push(b[0]);
            }
            let head = String::from_utf8_lossy(&head).to_string();
            let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
            match files.iter().find(|(n, _)| path == format!("/{n}")) {
                Some((_, body)) => {
                    let _ = write!(
                        s,
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = s.write_all(body);
                }
                None => {
                    let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
                }
            }
        }
    });
    base
}

fn model_res(
    id: &'static str,
    name: &'static str,
    base: &str,
    files: &[(&str, &[u8])],
    size: Option<u64>,
) -> &'static Resource {
    let parts: Vec<Part> = files
        .iter()
        .map(|(f, data)| Part {
            url: leak(format!("{base}/{f}")),
            sha256: leak(crate::resources::manager::tests_sha256(data)),
            size: size.unwrap_or(data.len() as u64),
            file_name: leak(f.to_string()),
            archive: None,
            layout: Layout::File,
        })
        .collect();
    Box::leak(Box::new(Resource {
        id,
        kind: ResourceKind::Model,
        version: "1",
        platform: None,
        parts: Box::leak(parts.into_boxed_slice()),
        dest: "models",
        provides: Provides::Model {
            name,
            family: crate::model::ModelFamilyKind::Yolo5,
        },
        title: leak(format!("Model {name}")),
        description: "fixture",
        license: "MIT",
    }))
}

/// State with a download manager over a temp data root, a fixture model served locally and
/// a fake large model that is never fetched; OpenVINO counts as loaded.
pub(super) fn resources_state() -> (Arc<AppState>, PathBuf) {
    let root = std::env::temp_dir().join(format!("bo_res_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let base = serve_files(vec![
        ("fixture-model.onnx", b"onnx bytes".to_vec()),
        ("fixture-model.yaml", b"NAMES:\n- person\n".to_vec()),
    ]);
    let fixture = model_res(
        "model:fixture-model",
        "fixture-model",
        &base,
        &[
            ("fixture-model.onnx", b"onnx bytes"),
            ("fixture-model.yaml", b"NAMES:\n- person\n"),
        ],
        None,
    );
    let big = model_res(
        "model:big-model",
        "big-model",
        "http://127.0.0.1:9",
        &[("big-model.onnx", b"x")],
        Some(600 * 1024 * 1024),
    );
    let config = Config {
        download_dir: Some(root.clone()),
        models: vec![ModelConfig {
            path: "models/other.onnx".into(),
            ..Default::default()
        }],
        ..Default::default()
    };
    let provisioner = Arc::new(Provisioner::with_options(ProvisionOptions {
        extra_models: vec![fixture, big],
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
    state.resources = Some(ResourcesCtx {
        provisioner,
        provision: Arc::new(Provision::none(&config)),
        openvino_loaded: true,
        ort_loaded: None,
    });
    (Arc::new(state), root)
}

pub(super) async fn resources(state: &Arc<AppState>) -> serde_json::Value {
    let (s, body) = get_text(state, "/v1/resources").await;
    assert_eq!(s, StatusCode::OK);
    serde_json::from_str(&body).unwrap()
}

pub(super) fn row<'a>(j: &'a serde_json::Value, id: &str) -> &'a serde_json::Value {
    j["resources"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["id"] == id)
        .unwrap_or_else(|| panic!("{id} not listed"))
}

pub(super) async fn post_action(
    state: &Arc<AppState>,
    uri: &str,
    body: &str,
) -> (StatusCode, serde_json::Value) {
    let (s, _, b) = call(state, post_form(uri, body, false)).await;
    (s, serde_json::from_slice(&b).unwrap())
}
