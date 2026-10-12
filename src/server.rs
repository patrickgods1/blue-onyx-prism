//! axum HTTP server: CodeProject.AI compatible detection endpoints plus the web UI
//! (welcome, stats, test, config and logs pages rendered with askama templates from `templates/`).
//! Handler flow and page set modeled on blue-onyx `server.rs` and its templates (MIT).

use crate::cli::LogReloadHandle;
use crate::config::Config;
use crate::metrics::Metrics;
use crate::registry::ModelRegistry;
use crate::startup::ModelState;
use crate::sync::RwLockExt;
use crate::worker::WorkerHandle;
use askama::Template;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

mod apply;
mod benchmark;
mod browse;
mod config_page;
mod detect;
mod devices;
mod live;
mod pages;
mod resources;
#[cfg(test)]
mod test_util;

pub(crate) use apply::reload_models;
use apply::sync_config;
use config_page::{
    ConfigView, MessageTemplate, config_loglevel, config_models, config_page, config_restart,
    config_submit, encode_uri_component, global_device_label, save_config, schedule_restart,
};
pub use detect::{BODY_LIMIT, RESPONSE_GRACE};
use detect::{custom_list, detection_custom, detection_default};
use devices::model_selection;
use devices::{DevicesSnapshot, devices_json, devices_snapshot};
use pages::{
    fallback, favicon, live_js, logs_json, logs_page, prometheus, stats_json, stats_page,
    style_css, test_page, test_submit, update_available, welcome,
};
use resources::{
    models_add, resources_add_to_config, resources_download, resources_export,
    resources_export_cancel, resources_json, resources_remove,
};

pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    pub metrics: Arc<Metrics>,
    /// The live config (revision, change log, file sync). Every change goes through
    /// [`save_config`] / [`crate::config_store::ConfigStore::update`]. Shared by every
    /// generation of the process.
    pub config: Arc<crate::config_store::ConfigStore>,
    /// What this generation runs: the config its registry was started with, updated as
    /// changes apply without a restart (live settings, per-model reloads; see
    /// [`apply_config`]). The pages compare it with `config` to show what a restart would
    /// apply. Read it with [`AppState::running`].
    pub running: arc_swap::ArcSwap<Config>,
    /// One [`apply_config`] at a time.
    pub apply_lock: std::sync::Mutex<()>,
    /// Config revision [`apply_config`] last applied (0: none yet).
    pub applied_revision: std::sync::atomic::AtomicU64,
    /// Process start (survives in-process restarts).
    pub started: Instant,
    pub config_path: PathBuf,
    /// Runtime log level control; `None` when logging was not initialized by this process.
    pub log_reload: Option<LogReloadHandle>,
    /// This generation's token, cancelled by `POST /config/restart`: the run loop
    /// (`runner`) then builds the next generation and swaps it in ([`Shared::swap`]).
    pub restart: CancellationToken,
    /// Download manager and this generation's provisioning plan (None in tests that do not
    /// exercise resources).
    pub resources: Option<crate::resources::status::ResourcesCtx>,
    /// Benchmark runner (one per process, shared by every generation).
    pub benchmark: Arc<crate::benchmark::service::BenchmarkService>,
}

impl AppState {
    /// State without a log reload handle and with a fresh restart token.
    pub fn new(
        registry: Arc<ModelRegistry>,
        metrics: Arc<Metrics>,
        config: Config,
        config_path: PathBuf,
    ) -> Self {
        let mut running = config.clone();
        running.normalize();
        Self {
            registry,
            metrics,
            config: Arc::new(crate::config_store::ConfigStore::new(
                config,
                config_path.clone(),
            )),
            running: arc_swap::ArcSwap::from_pointee(running),
            apply_lock: std::sync::Mutex::new(()),
            applied_revision: std::sync::atomic::AtomicU64::new(0),
            started: Instant::now(),
            config_path,
            log_reload: None,
            restart: CancellationToken::new(),
            resources: None,
            benchmark: crate::benchmark::service::BenchmarkService::new(),
        }
    }

    /// What this generation runs now (see [`AppState::running`](field@AppState::running)).
    pub fn running(&self) -> Arc<Config> {
        self.running.load_full()
    }

    fn config_read(&self) -> crate::config_store::ConfigRef<'_> {
        self.config.read()
    }

    /// Recent log events for the Logs page (kept by the logging setup's reload handle).
    fn log_buffer(&self) -> Option<&crate::logbuf::LogBuffer> {
        self.log_reload.as_ref().map(LogReloadHandle::buffer)
    }
}

/// The router. Every handler reads the generation that is current when its request arrives
/// (see [`Shared`]); a plain `Arc<AppState>` works for tests and makes a fixed one.
pub fn router(state: impl Into<Shared>) -> Router {
    let shared: Shared = state.into();
    // Open to every client: Blue Iris (often on another machine), monitoring and read-only JSON.
    let open = Router::new()
        .route("/", get(welcome))
        .route("/v1/vision/detection", post(detection_default))
        .route("/v1/vision/custom/list", get(custom_list).post(custom_list))
        .route("/v1/vision/custom/{model}", post(detection_custom))
        .route("/v1/status/updateavailable", get(update_available))
        .route("/v1/devices", get(devices_json))
        .route("/v1/config", get(live::config_json))
        .route("/stats.json", get(stats_json))
        .route("/prometheus", get(prometheus))
        .route("/static/style.css", get(style_css))
        .route("/static/live.js", get(live_js))
        .route("/favicon.ico", get(favicon));
    // Administration: loopback clients only unless `allow_remote_admin` ([`admin_only`]).
    let admin = Router::new()
        .route("/v1/resources", get(resources_json))
        .route("/v1/resources/download", post(resources_download))
        .route("/v1/resources/remove", post(resources_remove))
        .route("/v1/resources/add-to-config", post(resources_add_to_config))
        .route("/v1/models/add", post(models_add))
        .route("/v1/fs/browse", get(browse::browse))
        .route("/v1/resources/export", post(resources_export))
        .route("/v1/resources/export/cancel", post(resources_export_cancel))
        .route("/stats", get(stats_page))
        .route("/test", get(test_page).post(test_submit))
        .route("/config", get(config_page).post(config_submit))
        .route("/config/models", post(config_models))
        .route("/config/restart", post(config_restart))
        .route("/config/loglevel", post(config_loglevel))
        .route("/logs", get(logs_page))
        .route("/logs.json", get(logs_json))
        .route("/benchmark", get(benchmark::page))
        .route(
            "/v1/benchmark",
            get(benchmark::status).post(benchmark::start),
        )
        .route("/v1/benchmark/cancel", post(benchmark::cancel))
        .route("/v1/benchmark/apply", post(benchmark::apply))
        .route(
            "/v1/benchmark/apply-threshold",
            post(benchmark::apply_threshold),
        )
        .route(
            "/v1/benchmark/threshold-search",
            get(benchmark::search_options).post(benchmark::threshold_search),
        )
        .route("/v1/benchmark/settings", post(benchmark::settings))
        .route("/v1/benchmark/images", get(benchmark::images_detail))
        .route("/v1/benchmark/image", get(benchmark::image_file))
        .route_layer(axum::middleware::from_fn_with_state(
            shared.clone(),
            admin_only,
        ));
    open.merge(admin)
        .fallback(fallback)
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(shared)
}

/// Whether a client may use the admin routes: a loopback peer, or anyone with
/// `allow_remote_admin`. A request without a peer address (not from [`serve_on`]) is refused.
pub(super) fn admin_allowed(peer: Option<std::net::SocketAddr>, allow_remote_admin: bool) -> bool {
    allow_remote_admin
        || peer.is_some_and(|a| match a.ip() {
            std::net::IpAddr::V4(ip) => ip.is_loopback(),
            // `::1`, or an IPv4 loopback client on a dual-stack socket (`::ffff:127.0.0.1`).
            std::net::IpAddr::V6(ip) => {
                ip.is_loopback() || ip.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
            }
        })
}

/// Middleware of the admin routes: 403 (an HTML page for browsers, else JSON) for a client
/// that is not on this machine, unless `allow_remote_admin` is set (read per request, so it
/// applies at once).
pub(super) async fn admin_only(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let peer = req
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|c| c.0);
    let allow_remote = state.config_read().allow_remote_admin;
    if admin_allowed(peer, allow_remote) {
        return next.run(req).await;
    }
    let who = peer.map_or_else(|| "unknown".to_string(), |a| a.ip().to_string());
    warn!(peer = %who, path = %req.uri().path(), "admin route refused for a remote client");
    let msg = format!(
        "Administration is only available from this machine (your address: {who}). Set \
         \"allow_remote_admin\": true in the config (or start with --allow-remote-admin, or \
         check \"Allow remote administration\" on the Config page from this machine) to allow it."
    );
    if wants_html(req.headers()) {
        let page = MessageTemplate {
            nav: "",
            version: crate::VERSION,
            heading: "Forbidden".into(),
            message: msg,
            refresh_secs: 0,
            refresh_url: String::new(),
            restarting: false,
            generation: 0,
        };
        (StatusCode::FORBIDDEN, render(&page)).into_response()
    } else {
        (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({ "success": false, "message": msg, "error": msg })),
        )
            .into_response()
    }
}

/// The process-wide router state: the generation (`AppState`) that is current. A generation
/// restart swaps it ([`Shared::swap`]); the listener stays up. Handlers extract
/// `State<Arc<AppState>>` as before ([`axum::extract::FromRef`]), one `Arc` clone per request,
/// so an in-flight request keeps its generation (and its workers) alive until it finishes.
#[derive(Clone)]
pub struct Shared {
    current: Arc<std::sync::RwLock<Arc<AppState>>>,
}

impl Shared {
    pub fn new(state: Arc<AppState>) -> Self {
        Self {
            current: Arc::new(std::sync::RwLock::new(state)),
        }
    }

    /// The generation new requests are served by.
    pub fn current(&self) -> Arc<AppState> {
        self.current.read_recover().clone()
    }

    /// Make `state` current; returns the previous generation.
    pub fn swap(&self, state: Arc<AppState>) -> Arc<AppState> {
        let mut g = self.current.write_recover();
        std::mem::replace(&mut *g, state)
    }
}

impl From<Arc<AppState>> for Shared {
    fn from(state: Arc<AppState>) -> Self {
        Self::new(state)
    }
}

impl axum::extract::FromRef<Shared> for Arc<AppState> {
    fn from_ref(shared: &Shared) -> Self {
        shared.current()
    }
}

/// The process's HTTP listeners (bound once, see [`bind`]).
pub struct Listeners {
    v4: tokio::net::TcpListener,
    v6: Option<tokio::net::TcpListener>,
    port: u16,
}

impl Listeners {
    /// The bound port (useful after binding port 0).
    pub fn port(&self) -> u16 {
        self.port
    }
}

/// Bind `address:port`; for `0.0.0.0` and `127.0.0.1` also `[::1]:port` (best effort,
/// `localhost` may resolve to it). Called once per process run: a port in use is a startup
/// error.
pub async fn bind(address: std::net::IpAddr, port: u16) -> anyhow::Result<Listeners> {
    let addr = std::net::SocketAddr::new(address, port);
    let v4 = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        anyhow::anyhow!("binding {addr}: {e} (is another instance running on port {port}?)")
    })?;
    let port = v4.local_addr().map(|a| a.port()).unwrap_or(port);
    // `localhost` may resolve to ::1 first, and some clients do not fall back to 127.0.0.1.
    let v6 = match address {
        std::net::IpAddr::V4(a) if a.is_unspecified() || a.is_loopback() => {
            let loopback6 = std::net::SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, port));
            match tokio::net::TcpListener::bind(loopback6).await {
                Ok(l) => Some(l),
                Err(e) => {
                    warn!("not listening on {loopback6}: {e}");
                    None
                }
            }
        }
        _ => None,
    };
    info!("listening on {addr}; open http://localhost:{port}/");
    Ok(Listeners { v4, v6, port })
}

/// Serve `shared` on `listeners` until `shutdown` is cancelled (in-flight requests finish).
/// Runs for the whole process: generation restarts swap `shared` instead of stopping it.
pub async fn serve_on(
    listeners: Listeners,
    shared: Shared,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let app = router(shared);
    let stopped = |t: CancellationToken| async move { t.cancelled().await };
    // The peer address feeds the admin-route check ([`admin_only`]).
    let service = || {
        app.clone()
            .into_make_service_with_connect_info::<std::net::SocketAddr>()
    };
    let v4 = axum::serve(listeners.v4, service()).with_graceful_shutdown(stopped(shutdown.clone()));
    match listeners.v6 {
        Some(l6) => {
            let v6 = axum::serve(l6, service()).with_graceful_shutdown(stopped(shutdown));
            tokio::try_join!(v4.into_future(), v6.into_future())?;
        }
        None => v4.await?,
    }
    info!("HTTP server stopped");
    Ok(())
}

/// Run blocking handler work (file reads and walks, runtime library probes, page rendering) on
/// tokio's blocking pool so a slow page never holds up the detection routes. A panic in `f` is
/// resumed in the handler (axum answers 500 as for any handler panic).
pub(crate) async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    match tokio::task::spawn_blocking(f).await {
        Ok(v) => v,
        Err(e) => match e.try_into_panic() {
            Ok(p) => std::panic::resume_unwind(p),
            Err(e) => panic!("blocking handler task did not finish: {e}"),
        },
    }
}

pub(super) fn wants_html(headers: &HeaderMap) -> bool {
    headers
        .get(header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.contains("text/html"))
}

pub(super) fn render<T: Template>(t: &T) -> Response {
    match t.render() {
        Ok(html) => Html(html).into_response(),
        Err(e) => {
            error!("template rendering failed: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("template error: {e}"),
            )
                .into_response()
        }
    }
}

/// Short state, badge class and detail of a worker for the pages. Ready stays exactly
/// "Ready" and a failure starts with "Failed:" (the HTTP integration tests poll for both).
pub(super) fn state_view(w: &WorkerHandle) -> (String, &'static str, Option<String>) {
    match w.state.get() {
        ModelState::Ready => ("Ready".into(), "ok", None),
        ModelState::Failed(m) => ("Failed:".into(), "fail", Some(m)),
        ModelState::Initializing if w.accepts_while_initializing() => {
            ("Lazy".into(), "lazy", Some("loads on first request".into()))
        }
        ModelState::Initializing => match w.state.detail() {
            Some(d) if d.starts_with(crate::worker::COMPILING_ON) => {
                ("Loading".into(), "loading", Some(d))
            }
            Some(d) => ("Waiting for download".into(), "wait", Some(d)),
            None => ("Loading".into(), "loading", None),
        },
    }
}

pub(super) fn default_name(reg: &ModelRegistry) -> String {
    reg.default_model()
        .map(|w| w.name.clone())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::metrics::ModelMetrics;
    use crate::registry::RuntimeInfo;
    use crate::server::test_util::*;

    use crate::api::VisionDetectionResponse;
    use crate::config::Config;
    use crate::metrics::Metrics;
    use crate::registry::ModelRegistry;

    use axum::http::{StatusCode, header};

    use std::sync::Arc;

    use axum::body::Body;
    use axum::http::Request;

    /// Answers given before the image is used (unknown model, no usable default model) still
    /// read the upload: otherwise hyper closes the connection, and a client that pooled it
    /// (Blue Iris, reqwest) fails its next request (WSAECONNABORTED on Windows, a reset or
    /// broken pipe elsewhere).
    #[tokio::test]
    async fn early_answers_keep_the_connection_usable() {
        let listeners = bind([127, 0, 0, 1].into(), 0).await.unwrap();
        let port = listeners.v4.local_addr().unwrap().port();
        let token = CancellationToken::new();
        let server = tokio::spawn(serve_on(listeners, test_state().into(), token.clone()));
        // One pooled connection: every request reuses the one the previous answer left behind.
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(1)
            .build()
            .unwrap();
        for i in 0..6 {
            let path = if i % 2 == 0 {
                "v1/vision/custom/no-such-model"
            } else {
                "v1/vision/detection"
            };
            let part = reqwest::multipart::Part::bytes(vec![0u8; 8 << 20])
                .file_name("big.jpg")
                .mime_str("image/jpeg")
                .unwrap();
            let form = reqwest::multipart::Form::new().part("image", part);
            let resp = client
                .post(format!("http://127.0.0.1:{port}/{path}"))
                .multipart(form)
                .send()
                .await
                .unwrap_or_else(|e| panic!("request {i} ({path}): {e:?}"));
            let r: VisionDetectionResponse = resp.json().await.unwrap();
            assert!(!r.success, "{r:?}");
        }
        token.cancel();
        server.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn shared_swap_serves_the_new_generation() {
        let a = test_state();
        let mm = Arc::new(ModelMetrics::new("next", "CPU", ""));
        let registry = ModelRegistry::from_handles(
            vec![WorkerHandle::failed("next", "x", mm)],
            Some(0),
            RuntimeInfo::default(),
        );
        let b = Arc::new(AppState::new(
            Arc::new(registry),
            Arc::new(Metrics::new("0.0.0")),
            Config::default(),
            a.config_path.clone(),
        ));
        let shared = Shared::new(a.clone());
        let app = router(shared.clone());
        assert_eq!(listed(&app).await, ["ipcam-general", "<b>evil</b>"]);

        // Same router (= same listener): the next request sees the new generation.
        let prev = shared.swap(b.clone());
        assert!(Arc::ptr_eq(&prev, &a));
        assert!(Arc::ptr_eq(&shared.current(), &b));
        assert_eq!(listed(&app).await, ["next"]);
        // Nothing but the caller holds the old generation once its requests are done.
        drop(prev);
        assert_eq!(Arc::strong_count(&a), 1);
    }

    /// `method uri` from `peer` (an empty urlencoded body for POST), optionally as a browser.
    fn from(peer: &str, method: &str, uri: &str, html: bool) -> Request<Body> {
        let mut b = Request::builder().method(method).uri(uri);
        if method == "POST" {
            b = b.header(header::CONTENT_TYPE, "application/x-www-form-urlencoded");
        }
        if html {
            b = b.header(header::ACCEPT, "text/html");
        }
        from_peer(b.body(Body::empty()).unwrap(), peer)
    }

    #[test]
    fn loopback_peers_are_local() {
        let ok = |a: &str| admin_allowed(Some(a.parse().unwrap()), false);
        assert!(ok("127.0.0.1:1"));
        assert!(ok("127.8.9.10:1"));
        assert!(ok("[::1]:1"));
        assert!(ok("[::ffff:127.0.0.1]:1"));
        assert!(!ok("10.0.0.5:1"));
        assert!(!ok("192.168.1.20:1"));
        assert!(!ok("[fe80::1]:1"));
        assert!(!ok("[::ffff:10.0.0.5]:1"));
        assert!(!admin_allowed(None, false));
        assert!(admin_allowed(Some("10.0.0.5:1".parse().unwrap()), true));
    }

    #[tokio::test]
    async fn admin_routes_answer_loopback_only() {
        let state = test_state();
        let admin = [
            ("GET", "/config"),
            ("POST", "/config"),
            ("GET", "/v1/fs/browse"),
            ("POST", "/v1/resources/export"),
            ("POST", "/config/restart"),
        ];
        for (method, uri) in admin {
            // A remote client gets 403 with the setting that enables remote administration.
            let (s, h, b) = call(&state, from(REMOTE, method, uri, false)).await;
            let body = String::from_utf8_lossy(&b).to_string();
            assert_eq!(s, StatusCode::FORBIDDEN, "{method} {uri}: {body}");
            assert!(
                h[header::CONTENT_TYPE]
                    .to_str()
                    .unwrap()
                    .starts_with("application/json"),
                "{method} {uri}"
            );
            let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
            assert_eq!(j["success"], false);
            assert!(
                j["message"]
                    .as_str()
                    .unwrap()
                    .contains("allow_remote_admin"),
                "{j}"
            );
            assert!(j["message"].as_str().unwrap().contains("10.0.0.5"), "{j}");
            // Browsers get a page.
            let (s, h, b) = call(&state, from(REMOTE, method, uri, true)).await;
            assert_eq!(s, StatusCode::FORBIDDEN, "{method} {uri}");
            assert!(
                h[header::CONTENT_TYPE]
                    .to_str()
                    .unwrap()
                    .starts_with("text/html")
            );
            let html = String::from_utf8_lossy(&b);
            assert!(html.contains("Forbidden") && html.contains("allow_remote_admin"));
            // This machine gets the route.
            let (s, _, b) = call(&state, from(LOCAL, method, uri, true)).await;
            assert_ne!(
                s,
                StatusCode::FORBIDDEN,
                "{method} {uri}: {}",
                String::from_utf8_lossy(&b)
            );
            let (s, _, _) = call(&state, from("[::1]:50000", method, uri, true)).await;
            assert_ne!(s, StatusCode::FORBIDDEN, "{method} {uri} from ::1");
        }

        // Read-only and Blue Iris routes stay open to remote clients.
        for uri in [
            "/",
            "/stats.json",
            "/prometheus",
            "/v1/devices",
            "/v1/config",
            "/static/style.css",
            "/favicon.ico",
            "/v1/vision/custom/list",
        ] {
            let (s, _, _) = call(&state, from(REMOTE, "GET", uri, false)).await;
            assert_eq!(s, StatusCode::OK, "{uri}");
        }
        let (s, _, b) = call(
            &state,
            from_peer(
                Request::post("/v1/vision/detection")
                    .header(
                        header::CONTENT_TYPE,
                        "multipart/form-data; boundary=XBOUNDARYX",
                    )
                    .body(Body::from(
                        "--XBOUNDARYX\r\nContent-Disposition: form-data; name=\"image\"; \
                         filename=\"t.jpg\"\r\nContent-Type: image/jpeg\r\n\r\nxx\r\n--XBOUNDARYX--\r\n",
                    ))
                    .unwrap(),
                REMOTE,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let r: VisionDetectionResponse = serde_json::from_slice(&b).unwrap();
        assert!(!r.success, "the test models failed to load: {r:?}");
        assert!(r.error.unwrap().contains("ipcam-general"));

        // `allow_remote_admin` applies at once.
        state
            .config
            .update("test", |c| {
                c.allow_remote_admin = true;
                Ok(())
            })
            .unwrap();
        let (s, _, _) = call(&state, from(REMOTE, "GET", "/config", true)).await;
        assert_eq!(s, StatusCode::OK);
    }
}
