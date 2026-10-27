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
use axum::routing::{MethodRouter, get, post};
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
mod dataset;
mod detect;
mod devices;
mod guard;
mod live;
mod pages;
mod pick;
mod resources;
mod review;
mod status;
#[cfg(test)]
mod test_util;
mod training;
mod training_tasks;

pub(crate) use apply::{apply_config, reload_models, sync_config};
use config_page::{
    ConfigView, MessageTemplate, config_loglevel, config_models, config_models_remove, config_page,
    config_restart, config_submit, encode_uri_component, global_device_label, save_config,
    schedule_restart,
};
pub(crate) use detect::BODY_LIMIT;
use detect::{custom_list, detection_custom, detection_default};
use devices::model_selection;
use devices::{DevicesSnapshot, devices_json, devices_snapshot};
use pages::{
    favicon, live_js, logs_json, logs_page, prometheus, static_asset, stats_json, stats_page,
    style_css, test_page, test_submit, update_available, welcome,
};
use resources::{
    models_add, resources_add_to_config, resources_download, resources_export,
    resources_export_cancel, resources_json, resources_remove,
};

pub struct AppState {
    pub(crate) registry: Arc<ModelRegistry>,
    pub(crate) metrics: Arc<Metrics>,
    /// The live config (revision, change log, file sync). Every change goes through
    /// [`save_config`] / [`crate::config_store::ConfigStore::update`]. Shared by every
    /// generation of the process.
    pub(crate) config: Arc<crate::config_store::ConfigStore>,
    /// What this generation runs: the config its registry was started with, updated as
    /// changes apply without a restart (live settings, per-model reloads; see
    /// [`apply_config`]). The pages compare it with `config` to show what a restart would
    /// apply. Read it with [`AppState::running`].
    pub(crate) running: arc_swap::ArcSwap<Config>,
    /// One [`apply_config`] at a time.
    pub(crate) apply_lock: std::sync::Mutex<()>,
    /// Config revision [`apply_config`] last applied (0: none yet).
    pub(crate) applied_revision: std::sync::atomic::AtomicU64,
    /// Process start (survives in-process restarts).
    pub(crate) started: Instant,
    pub(crate) config_path: PathBuf,
    /// Runtime log level control; `None` when logging was not initialized by this process.
    pub(crate) log_reload: Option<LogReloadHandle>,
    /// This generation's token, cancelled by `POST /config/restart`: the run loop
    /// (`runner`) then builds the next generation and swaps it in ([`Shared::swap`]).
    pub(crate) restart: CancellationToken,
    /// Download manager and this generation's provisioning plan (None in tests that do not
    /// exercise resources).
    pub(crate) resources: Option<crate::resources::status::ResourcesCtx>,
    /// Benchmark runner (one per process, shared by every generation).
    pub(crate) benchmark: Arc<crate::benchmark::service::BenchmarkService>,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("registry", &self.registry)
            .field("config_path", &self.config_path)
            .field("started", &self.started)
            .field("restart", &self.restart)
            .finish_non_exhaustive()
    }
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
    pub(crate) fn running(&self) -> Arc<Config> {
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

/// The open routes: Blue Iris (often on another machine), monitoring, static files and the
/// minimal `/v1/config` the pages' restart poller uses. Every path here answers remote peers;
/// `router_classification` in the tests checks the list against the expected set.
fn open_routes() -> Vec<(&'static str, MethodRouter<Shared>)> {
    vec![
        ("/", get(welcome)),
        ("/v1/vision/detection", post(detection_default)),
        ("/v1/vision/custom/list", get(custom_list).post(custom_list)),
        ("/v1/vision/custom/{model}", post(detection_custom)),
        ("/v1/status", get(status::readiness)),
        ("/v1/status/updateavailable", get(update_available)),
        ("/v1/devices", get(devices_json)),
        ("/v1/config", get(live::config_json)),
        ("/stats.json", get(stats_json)),
        ("/prometheus", get(prometheus)),
        ("/static/style.css", get(style_css)),
        ("/static/live.js", get(live_js)),
        ("/static/{file}", get(static_asset)),
        ("/favicon.ico", get(favicon)),
    ]
}

/// The administration routes: loopback clients only unless `allow_remote_admin`
/// ([`admin_only`]), and no cross-site state changes ([`guard::csrf`]).
fn admin_routes() -> Vec<(&'static str, MethodRouter<Shared>)> {
    vec![
        ("/v1/resources", get(resources_json)),
        ("/v1/resources/download", post(resources_download)),
        ("/v1/resources/remove", post(resources_remove)),
        ("/v1/resources/add-to-config", post(resources_add_to_config)),
        ("/v1/models/add", post(models_add)),
        ("/v1/fs/browse", get(browse::browse)),
        ("/v1/fs/pick", get(pick::status).post(pick::pick)),
        ("/v1/resources/export", post(resources_export)),
        ("/v1/resources/export/cancel", post(resources_export_cancel)),
        ("/stats", get(stats_page)),
        ("/test", get(test_page).post(test_submit)),
        ("/config", get(config_page).post(config_submit)),
        ("/config/models", post(config_models)),
        ("/config/models/remove", post(config_models_remove)),
        ("/config/restart", post(config_restart)),
        ("/config/loglevel", post(config_loglevel)),
        ("/dataset", get(dataset::page)),
        ("/dataset/settings", post(dataset::settings)),
        ("/dataset/review", post(dataset::review)),
        ("/dataset/resume", post(dataset::resume)),
        ("/dataset/image", get(dataset::image)),
        ("/dataset/sidecar", get(dataset::sidecar)),
        ("/v1/dataset", get(dataset::json)),
        ("/review", get(review::page)),
        ("/v1/review/next", get(review::next)),
        (
            "/v1/review/labels",
            get(review::labels_get)
                .post(review::labels_post)
                .layer(DefaultBodyLimit::max(review::LABELS_BODY_LIMIT)),
        ),
        (
            "/v1/review/accept-auto",
            post(review::accept_auto).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        (
            "/v1/review/reject-low",
            post(review::reject_low).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        (
            "/v1/review/export",
            post(review::export).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        (
            "/v1/review/import",
            post(review::import).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        (
            "/v1/review/classes",
            post(review::classes).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        ("/v1/review/class-catalog", get(review::class_catalog)),
        (
            "/v1/review/folder",
            post(review::folder).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        ("/training", get(training::page)),
        ("/v1/training", get(training::json)),
        (
            "/v1/training/metric-columns",
            post(training::metric_columns).layer(DefaultBodyLimit::max(16 * 1024)),
        ),
        ("/training/queue", post(training::queue)),
        ("/training/cancel", post(training::cancel)),
        ("/training/pause", post(training::pause)),
        ("/training/resume", post(training::resume)),
        ("/training/remove", post(training::remove)),
        ("/training/use", post(training::use_model)),
        ("/training/datasets", post(training::datasets)),
        ("/training/dir", post(training::set_dir)),
        ("/training/remove-env", post(training::remove_env)),
        ("/training/import", post(training_tasks::import)),
        (
            "/v1/training/import/preview",
            get(training_tasks::import_preview),
        ),
        ("/training/autolabel", post(training_tasks::autolabel)),
        (
            "/v1/autolabel/models",
            get(training_tasks::autolabel_models),
        ),
        ("/logs", get(logs_page)),
        ("/logs.json", get(logs_json)),
        ("/benchmark", get(benchmark::page)),
        (
            "/v1/benchmark",
            get(benchmark::status).post(benchmark::start),
        ),
        ("/v1/benchmark/cancel", post(benchmark::cancel)),
        ("/v1/benchmark/apply", post(benchmark::apply)),
        (
            "/v1/benchmark/apply-threshold",
            post(benchmark::apply_threshold),
        ),
        (
            "/v1/benchmark/apply-profiles",
            post(benchmark::apply_profiles),
        ),
        (
            "/v1/benchmark/threshold-search",
            get(benchmark::search_options).post(benchmark::threshold_search),
        ),
        ("/v1/benchmark/settings", post(benchmark::settings)),
        ("/v1/benchmark/report", post(benchmark::save_report)),
        ("/v1/benchmark/images", get(benchmark::images_detail)),
        ("/v1/benchmark/image", get(benchmark::image_file)),
        ("/v1/benchmark/history", get(benchmark::history_get)),
        (
            "/v1/benchmark/history/compare",
            get(benchmark::history_compare),
        ),
        ("/v1/benchmark/history/pin", post(benchmark::history_pin)),
        (
            "/v1/benchmark/history/delete",
            post(benchmark::history_delete),
        ),
    ]
}

/// The router. Every handler reads the generation that is current when its request arrives
/// (see [`Shared`]); a plain `Arc<AppState>` works for tests and makes a fixed one.
pub(crate) fn router(state: impl Into<Shared>) -> Router {
    let shared: Shared = state.into();
    let mut open = Router::new();
    for (path, method) in open_routes() {
        open = open.route(path, method);
    }
    let mut admin = Router::new();
    for (path, method) in admin_routes() {
        admin = admin.route(path, method);
    }
    let admin = admin
        .route_layer(axum::middleware::from_fn(guard::csrf))
        .route_layer(axum::middleware::from_fn_with_state(
            shared.clone(),
            admin_only,
        ));
    #[cfg(test)]
    let open = open
        .route("/__test/panic", get(test_panic))
        .route("/__test/blocking-panic", get(test_blocking_panic));
    open.merge(admin)
        .fallback(fallback)
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .layer(axum::middleware::from_fn(catch_panic))
        .layer(axum::middleware::map_response(guard::security_headers))
        .with_state(shared)
}

/// 404 for an unknown path: JSON for API clients, a short page for browsers. The path is
/// never echoed (not even escaped), so nothing a client sends appears in the answer.
async fn fallback(headers: HeaderMap) -> Response {
    if wants_html(&headers) {
        (
            StatusCode::NOT_FOUND,
            Html(
                "<!doctype html><title>Not found</title><h1>Not found</h1>\
                 <p>There is no page at this address. <a href=\"/\">Home</a></p>",
            ),
        )
            .into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "success": false, "error": "Not found" })),
        )
            .into_response()
    }
}

/// Test-only route: a handler that panics.
#[cfg(test)]
#[allow(clippy::unused_async, reason = "axum handlers are async")]
async fn test_panic() -> &'static str {
    std::panic::panic_any("test handler panic")
}

/// Test-only route: a handler whose [`blocking`] closure panics.
#[cfg(test)]
async fn test_blocking_panic() -> &'static str {
    blocking(|| std::panic::panic_any("test blocking panic")).await
}

/// The panic payload's message (`panic!` with a literal or a format string), else a placeholder.
fn panic_message(p: &(dyn std::any::Any + Send)) -> &str {
    p.downcast_ref::<&str>()
        .copied()
        .or_else(|| p.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic payload")
}

/// Outermost middleware: a panic in a handler (or in its [`blocking`] work) becomes a 500
/// answer instead of a dropped connection. The panic message is logged with the method and
/// path (never the request body). API clients get `{"success":false,"error":"internal error"}`,
/// browsers a short HTML page.
async fn catch_panic(req: axum::extract::Request, next: axum::middleware::Next) -> Response {
    use futures_util::FutureExt;
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let html = wants_html(req.headers());
    match std::panic::AssertUnwindSafe(next.run(req))
        .catch_unwind()
        .await
    {
        Ok(resp) => resp,
        Err(p) => {
            error!(
                %method,
                %path,
                panic = panic_message(p.as_ref()),
                "request handler panicked; answering 500"
            );
            if html {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Html(
                        "<!doctype html><title>Internal error</title>\
                         <h1>Internal error</h1><p>The server hit an internal error while \
                         handling this page. Details are in the log.</p>",
                    ),
                )
                    .into_response()
            } else {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "success": false, "error": "internal error" })),
                )
                    .into_response()
            }
        }
    }
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

/// Set once the "local reverse proxy" warning was logged (SV-12).
static LOCAL_PROXY_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether a request carries a proxy's forwarding header (`X-Forwarded-For` or `Forwarded`).
fn is_forwarded(headers: &HeaderMap) -> bool {
    headers.contains_key("x-forwarded-for") || headers.contains_key(header::FORWARDED)
}

/// Log, once per `flag` (once per process for [`LOCAL_PROXY_WARNED`]), that admin requests
/// arrive through a reverse proxy on this machine: the admin guard trusts the peer address
/// only, so every proxied client counts as local. True when this call logged it.
fn warn_local_proxy_once(flag: &std::sync::atomic::AtomicBool) -> bool {
    if flag.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    warn!(
        "requests arrive through a local reverse proxy: every proxied client is treated as \
         local admin; set allow_remote_admin or restrict the proxy"
    );
    true
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
        if !allow_remote && is_forwarded(req.headers()) {
            warn_local_proxy_once(&LOCAL_PROXY_WARNED);
        }
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
/// restart swaps it (`Shared::swap`); the listener stays up. Handlers extract
/// `State<Arc<AppState>>` as before ([`axum::extract::FromRef`]), one `Arc` clone per request,
/// so an in-flight request keeps its generation (and its workers) alive until it finishes.
#[derive(Clone, Debug)]
pub struct Shared {
    current: Arc<std::sync::RwLock<Arc<AppState>>>,
}

impl Shared {
    pub(crate) fn new(state: Arc<AppState>) -> Self {
        Self {
            current: Arc::new(std::sync::RwLock::new(state)),
        }
    }

    /// The generation new requests are served by.
    pub(crate) fn current(&self) -> Arc<AppState> {
        self.current.read_recover().clone()
    }

    /// Make `state` current; returns the previous generation.
    pub(crate) fn swap(&self, state: Arc<AppState>) -> Arc<AppState> {
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
#[derive(Debug)]
pub struct Listeners {
    v4: tokio::net::TcpListener,
    v6: Option<tokio::net::TcpListener>,
    port: u16,
}

impl Listeners {
    /// The bound port (useful after binding port 0).
    pub(crate) fn port(&self) -> u16 {
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

/// Longest a client may take to send a request head. With hyper 1.x HTTP/1 this timer also
/// runs while a keep-alive connection waits for its next request (hyper restarts it as soon as
/// a response is done: `proto/h1/conn.rs`, `poll_read_head`), so it is the idle timeout too: a
/// connection silent for this long is closed (slowloris and idle keep-alive alike).
pub(crate) const HEADER_READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// How long a shutdown waits for in-flight requests to finish before closing their
/// connections anyway.
pub(crate) const GRACEFUL_SHUTDOWN_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(10);

/// Pause after an accept error that is not about one connection (e.g. out of file
/// descriptors), so the loop does not spin.
const ACCEPT_ERROR_PAUSE: std::time::Duration = std::time::Duration::from_secs(1);

/// Serve `shared` on `listeners` until `shutdown` is cancelled (in-flight requests finish,
/// for at most `GRACEFUL_SHUTDOWN_TIMEOUT`). Runs for the whole process: generation restarts
/// swap `shared` instead of stopping it. Connections are HTTP/1.1 served by hyper with a
/// timer, so `HEADER_READ_TIMEOUT` applies to request heads and idle keep-alive waits.
pub async fn serve_on(
    listeners: Listeners,
    shared: Shared,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let app = router(shared);
    match listeners.v6 {
        Some(l6) => {
            tokio::join!(
                accept_loop(listeners.v4, app.clone(), shutdown.clone()),
                accept_loop(l6, app, shutdown)
            );
        }
        None => accept_loop(listeners.v4, app, shutdown).await,
    }
    info!("HTTP server stopped");
    Ok(())
}

/// The hyper connection builder: HTTP/1.1 only (as before: axum's `http1` feature), keep-alive,
/// [`HEADER_READ_TIMEOUT`] driven by tokio's timer.
fn connection_builder() -> hyper_util::server::conn::auto::Builder<hyper_util::rt::TokioExecutor> {
    let mut builder =
        hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new())
            .http1_only();
    builder
        .http1()
        .timer(hyper_util::rt::TokioTimer::new())
        .header_read_timeout(HEADER_READ_TIMEOUT)
        .keep_alive(true);
    builder
}

/// Accept connections on `listener` until `shutdown`, then let the open ones finish their
/// requests (graceful shutdown, at most [`GRACEFUL_SHUTDOWN_TIMEOUT`]).
async fn accept_loop(listener: tokio::net::TcpListener, app: Router, shutdown: CancellationToken) {
    let builder = connection_builder();
    let graceful = hyper_util::server::graceful::GracefulShutdown::new();
    loop {
        let (stream, peer) = tokio::select! {
            () = shutdown.cancelled() => break,
            accepted = listener.accept() => match accepted {
                Ok(c) => c,
                Err(e) if is_connection_error(&e) => continue,
                Err(e) => {
                    error!("accepting an HTTP connection: {e}");
                    tokio::time::sleep(ACCEPT_ERROR_PAUSE).await;
                    continue;
                }
            },
        };
        let router = app.clone();
        // The peer address feeds the admin-route check ([`admin_only`]).
        let service =
            hyper::service::service_fn(move |mut req: hyper::Request<hyper::body::Incoming>| {
                req.extensions_mut()
                    .insert(axum::extract::ConnectInfo(peer));
                // `Router` is always ready (its `poll_ready` returns `Ready` at once).
                tower::Service::call(&mut router.clone(), req)
            });
        let conn = builder
            .serve_connection(hyper_util::rt::TokioIo::new(stream), service)
            .into_owned();
        let conn = graceful.watch(conn);
        tokio::spawn(async move {
            if let Err(e) = conn.await {
                tracing::debug!(%peer, "HTTP connection ended: {e}");
            }
        });
    }
    drop(listener);
    if tokio::time::timeout(GRACEFUL_SHUTDOWN_TIMEOUT, graceful.shutdown())
        .await
        .is_err()
    {
        warn!(
            "HTTP connections still busy {} s after shutdown; closing them",
            GRACEFUL_SHUTDOWN_TIMEOUT.as_secs()
        );
    }
}

/// Accept errors about one connection (the next accept may well succeed), as axum's `serve`.
fn is_connection_error(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::ConnectionRefused
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::ConnectionReset
    )
}

/// Run blocking handler work (file reads and walks, runtime library probes, page rendering) on
/// tokio's blocking pool so a slow page never holds up the detection routes. A panic in `f`
/// (or a task cancelled by runtime shutdown) unwinds into the handler, and the router's
/// [`catch_panic`] middleware turns it into a 500 answer; the connection stays usable.
pub(crate) async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    match tokio::task::spawn_blocking(f).await {
        Ok(v) => v,
        Err(e) => match e.try_into_panic() {
            Ok(p) => std::panic::resume_unwind(p),
            Err(e) => std::panic::resume_unwind(Box::new(format!(
                "blocking handler task did not finish: {e}"
            ))),
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

    /// SV-04: a client that never finishes its request head, and a keep-alive connection
    /// left idle after its answer, are both closed after [`HEADER_READ_TIMEOUT`]; shutdown
    /// still completes.
    #[tokio::test]
    async fn slow_heads_and_idle_connections_are_closed() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listeners = bind([127, 0, 0, 1].into(), 0).await.unwrap();
        let port = listeners.v4.local_addr().unwrap().port();
        let token = CancellationToken::new();
        let server = tokio::spawn(serve_on(listeners, test_state().into(), token.clone()));
        /// Seconds until the server closes a connection that sent `send`.
        async fn closed_after(port: u16, send: &'static [u8]) -> (f64, Vec<u8>) {
            let mut s = tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .unwrap();
            s.write_all(send).await.unwrap();
            let started = std::time::Instant::now();
            let mut got = Vec::new();
            let limit = HEADER_READ_TIMEOUT + std::time::Duration::from_secs(10);
            tokio::time::timeout(limit, s.read_to_end(&mut got))
                .await
                .expect("the server should close the connection")
                .ok();
            (started.elapsed().as_secs_f64(), got)
        }
        let (slow, idle) = tokio::join!(
            closed_after(port, b"GET / HTTP/1.1\r\nHost: localhost\r\n"),
            closed_after(
                port,
                b"GET /favicon.ico HTTP/1.1\r\nHost: localhost\r\n\r\n"
            ),
        );
        let t = HEADER_READ_TIMEOUT.as_secs_f64();
        assert!(
            slow.0 >= t - 0.5 && slow.0 < t + 5.0,
            "slow head closed after {}",
            slow.0
        );
        assert!(
            idle.0 >= t - 0.5 && idle.0 < t + 5.0,
            "idle closed after {}",
            idle.0
        );
        assert!(
            idle.1.starts_with(b"HTTP/1.1 200"),
            "the full request was answered"
        );
        token.cancel();
        server.await.unwrap().unwrap();
    }

    /// SV-15: a panicking handler (directly or in its `blocking` work) answers 500 and the
    /// server, and the connection, keep serving.
    #[tokio::test]
    async fn handler_panic_answers_500_and_server_keeps_serving() {
        let listeners = bind([127, 0, 0, 1].into(), 0).await.unwrap();
        let port = listeners.v4.local_addr().unwrap().port();
        let token = CancellationToken::new();
        let server = tokio::spawn(serve_on(listeners, test_state().into(), token.clone()));
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(1)
            .build()
            .unwrap();
        for path in ["__test/panic", "__test/blocking-panic"] {
            let resp = client
                .get(format!("http://127.0.0.1:{port}/{path}"))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR, "{path}");
            let body: serde_json::Value = resp.json().await.unwrap();
            assert_eq!(
                body,
                serde_json::json!({ "success": false, "error": "internal error" })
            );
            let html = client
                .get(format!("http://127.0.0.1:{port}/{path}"))
                .header(header::ACCEPT, "text/html")
                .send()
                .await
                .unwrap();
            assert_eq!(html.status(), StatusCode::INTERNAL_SERVER_ERROR);
            assert!(html.text().await.unwrap().contains("Internal error"));
            let ok = client
                .get(format!("http://127.0.0.1:{port}/v1/vision/custom/list"))
                .send()
                .await
                .unwrap();
            assert_eq!(ok.status(), StatusCode::OK, "after {path}");
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

    /// SV-12: a loopback admin request carrying a proxy's forwarding header logs the
    /// "local reverse proxy" warning, once per process.
    #[tokio::test]
    async fn local_proxy_is_warned_once() {
        let mut h = HeaderMap::new();
        assert!(!is_forwarded(&h));
        h.insert("x-forwarded-for", "203.0.113.9".parse().unwrap());
        assert!(is_forwarded(&h));
        let mut h = HeaderMap::new();
        h.insert(header::FORWARDED, "for=203.0.113.9".parse().unwrap());
        assert!(is_forwarded(&h));

        let flag = std::sync::atomic::AtomicBool::new(false);
        assert!(warn_local_proxy_once(&flag));
        assert!(!warn_local_proxy_once(&flag));

        // Through the admin guard: the proxied loopback request is served and sets the
        // process-wide flag.
        let state = test_state();
        let mut req = from("127.0.0.1:40000", "GET", "/config", true);
        req.headers_mut()
            .insert("x-forwarded-for", "203.0.113.9".parse().unwrap());
        let (s, _, _) = call(&state, req).await;
        assert_eq!(s, StatusCode::OK);
        assert!(LOCAL_PROXY_WARNED.load(std::sync::atomic::Ordering::Relaxed));
    }

    /// Every route of the router, classified. A new route must be added here: the test fails
    /// when the router has a path this list does not (and the other way round), and probes
    /// each path from a remote peer (admin routes answer 403, open ones never do).
    #[tokio::test]
    async fn router_classification() {
        const OPEN: &[&str] = &[
            "/",
            "/v1/vision/detection",
            "/v1/vision/custom/list",
            "/v1/vision/custom/{model}",
            "/v1/status",
            "/v1/status/updateavailable",
            "/v1/devices",
            "/v1/config",
            "/stats.json",
            "/prometheus",
            "/favicon.ico",
        ];
        const ADMIN: &[&str] = &[
            "/v1/resources",
            "/v1/resources/download",
            "/v1/resources/remove",
            "/v1/resources/add-to-config",
            "/v1/models/add",
            "/v1/fs/browse",
            "/v1/fs/pick",
            "/v1/resources/export",
            "/v1/resources/export/cancel",
            "/stats",
            "/test",
            "/config",
            "/config/models",
            "/config/models/remove",
            "/config/restart",
            "/config/loglevel",
            "/dataset",
            "/dataset/settings",
            "/dataset/review",
            "/dataset/resume",
            "/dataset/image",
            "/dataset/sidecar",
            "/v1/dataset",
            "/review",
            "/v1/review/next",
            "/v1/review/labels",
            "/v1/review/accept-auto",
            "/v1/review/reject-low",
            "/v1/review/export",
            "/v1/review/import",
            "/v1/review/classes",
            "/v1/review/class-catalog",
            "/v1/review/folder",
            "/training",
            "/v1/training",
            "/v1/training/metric-columns",
            "/training/queue",
            "/training/cancel",
            "/training/pause",
            "/training/resume",
            "/training/remove",
            "/training/use",
            "/training/datasets",
            "/training/dir",
            "/training/remove-env",
            "/training/import",
            "/v1/training/import/preview",
            "/training/autolabel",
            "/v1/autolabel/models",
            "/logs",
            "/logs.json",
            "/benchmark",
            "/v1/benchmark",
            "/v1/benchmark/cancel",
            "/v1/benchmark/apply",
            "/v1/benchmark/apply-threshold",
            "/v1/benchmark/apply-profiles",
            "/v1/benchmark/threshold-search",
            "/v1/benchmark/settings",
            "/v1/benchmark/report",
            "/v1/benchmark/images",
            "/v1/benchmark/image",
            "/v1/benchmark/history",
            "/v1/benchmark/history/compare",
            "/v1/benchmark/history/pin",
            "/v1/benchmark/history/delete",
        ];
        // Static files are open as a group (`/static/*`).
        let is_open = |p: &str| OPEN.contains(&p) || p.starts_with("/static/");
        let open: Vec<&str> = open_routes().into_iter().map(|(p, _)| p).collect();
        let admin: Vec<&str> = admin_routes().into_iter().map(|(p, _)| p).collect();
        for p in &open {
            assert!(is_open(p), "open route {p} is not in the expected open set");
            assert!(!ADMIN.contains(p), "{p} is both open and admin");
        }
        for p in OPEN {
            assert!(open.contains(p), "expected open route {p} is not routed");
        }
        for p in &admin {
            assert!(
                ADMIN.contains(p),
                "admin route {p} is not in the expected admin set"
            );
        }
        for p in ADMIN {
            assert!(admin.contains(p), "expected admin route {p} is not routed");
        }
        let mut all: Vec<&str> = open.iter().chain(admin.iter()).copied().collect();
        let n = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), n, "a path is routed twice");

        let state = test_state();
        let probe = |p: &str| p.replace("{model}", "x").replace("{file}", "x.js");
        for p in &admin {
            for method in ["GET", "POST"] {
                let (s, _, _) = call(&state, from(REMOTE, method, &probe(p), false)).await;
                assert_eq!(s, StatusCode::FORBIDDEN, "{method} {p} from a remote peer");
            }
        }
        for p in &open {
            for method in ["GET", "POST"] {
                let (s, _, _) = call(&state, from(REMOTE, method, &probe(p), false)).await;
                assert_ne!(s, StatusCode::FORBIDDEN, "{method} {p} from a remote peer");
            }
        }
    }

    /// SV-01: a state-changing admin request from another site is refused, same-origin and
    /// header-less ones reach the handler.
    #[tokio::test]
    async fn cross_site_admin_posts_are_refused() {
        let state = test_state();
        let post = |headers: &[(&str, &str)], html: bool| {
            let mut b = Request::post("/config")
                .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded");
            for (k, v) in headers {
                b = b.header(*k, *v);
            }
            if html {
                b = b.header(header::ACCEPT, "text/html");
            }
            from_peer(b.body(Body::from("port=4000")).unwrap(), LOCAL)
        };
        let host = ("host", "localhost:32168");
        // Cross-site: 403 as JSON, and as a page for browsers; the config is untouched.
        for headers in [
            vec![host, ("sec-fetch-site", "cross-site")],
            vec![host, ("origin", "http://evil.example")],
            vec![host, ("origin", "null")],
            vec![host, ("referer", "http://evil.example/page")],
            vec![host, ("origin", "http://localhost:8080")],
        ] {
            let (s, h, b) = call(&state, post(&headers, false)).await;
            let body = String::from_utf8_lossy(&b).to_string();
            assert_eq!(s, StatusCode::FORBIDDEN, "{headers:?}: {body}");
            assert!(
                h[header::CONTENT_TYPE]
                    .to_str()
                    .unwrap()
                    .starts_with("application/json")
            );
            let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
            assert_eq!(j["success"], false);
            assert!(j["message"].as_str().unwrap().contains("Cross-site"), "{j}");
            let (s, h, b) = call(&state, post(&headers, true)).await;
            assert_eq!(s, StatusCode::FORBIDDEN);
            assert!(
                h[header::CONTENT_TYPE]
                    .to_str()
                    .unwrap()
                    .starts_with("text/html")
            );
            assert!(String::from_utf8_lossy(&b).contains("Forbidden"));
            assert_eq!(state.config_read().port, Config::default().port);
        }
        // Same-origin (Origin, Referer, Sec-Fetch-Site) and header-less: the handler runs.
        for (i, headers) in [
            vec![host, ("origin", "http://localhost:32168")],
            vec![host, ("referer", "http://localhost:32168/config")],
            vec![host, ("sec-fetch-site", "same-origin")],
            vec![host],
            vec![
                host,
                ("origin", "http://LOCALHOST:32168"),
                ("sec-fetch-site", "none"),
            ],
        ]
        .into_iter()
        .enumerate()
        {
            let port = format!("port={}", 4001 + i);
            let req = {
                let mut b = Request::post("/config")
                    .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
                    .header(header::ACCEPT, "application/json");
                for (k, v) in &headers {
                    b = b.header(*k, *v);
                }
                from_peer(b.body(Body::from(port)).unwrap(), LOCAL)
            };
            let (s, _, b) = call(&state, req).await;
            assert_eq!(
                s,
                StatusCode::OK,
                "{headers:?}: {}",
                String::from_utf8_lossy(&b)
            );
            assert_eq!(state.config_read().port, 4001 + i as u16, "{headers:?}");
        }
        // GET is never checked (nothing changes, nothing is readable cross-site).
        let req = from_peer(
            Request::get("/config")
                .header("host", "localhost:32168")
                .header("origin", "http://evil.example")
                .body(Body::empty())
                .unwrap(),
            LOCAL,
        );
        let (s, _, _) = call(&state, req).await;
        assert_eq!(s, StatusCode::OK);
        // Detection and custom/list are never affected.
        let req = from_peer(
            Request::post("/v1/vision/custom/list")
                .header("sec-fetch-site", "cross-site")
                .header("origin", "http://evil.example")
                .body(Body::empty())
                .unwrap(),
            REMOTE,
        );
        let (s, _, _) = call(&state, req).await;
        assert_eq!(s, StatusCode::OK);
    }

    /// SV-05: `GET /v1/config` gives a non-admin peer only the restart poller's counters.
    #[tokio::test]
    async fn config_json_is_minimal_for_non_admin_peers() {
        let state = test_state();
        let (s, h, b) = call(&state, from(REMOTE, "GET", "/v1/config", false)).await;
        assert_eq!(s, StatusCode::OK);
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        let mut keys: Vec<&str> = j.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "epoch",
                "generation",
                "pendingCount",
                "restartNeeded",
                "revision",
                "success"
            ]
        );
        assert!(j["revision"].is_u64() && j["generation"].is_u64() && j["epoch"].is_u64());
        assert!(h.contains_key(header::ETAG));
        // The ETag round-trips to 304.
        let etag = h[header::ETAG].to_str().unwrap().to_string();
        let req = from_peer(
            Request::get("/v1/config")
                .header(header::IF_NONE_MATCH, etag)
                .body(Body::empty())
                .unwrap(),
            REMOTE,
        );
        let (s, _, _) = call(&state, req).await;
        assert_eq!(s, StatusCode::NOT_MODIFIED);
        // Nothing of the config was read or written from disk for it.
        assert!(!state.config_path.exists());
        // This machine (and remote peers with `allow_remote_admin`) get the full payload.
        let (_, _, b) = call(&state, from(LOCAL, "GET", "/v1/config", false)).await;
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert!(j["config"].is_object() && j["models"].is_array() && j["path"].is_string());
        state
            .config
            .update("test", |c| {
                c.allow_remote_admin = true;
                Ok(())
            })
            .unwrap();
        let (_, _, b) = call(&state, from(REMOTE, "GET", "/v1/config", false)).await;
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert!(j["config"].is_object());
    }

    /// SV-09: security headers on every answer, CSP on pages, no-store on JSON, a fallback
    /// that never echoes the path, 405 for a method a path does not take.
    #[tokio::test]
    async fn security_headers_fallback_and_405() {
        let state = test_state();
        let (s, h, _) = call(&state, from(LOCAL, "GET", "/config", true)).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(h["x-content-type-options"], "nosniff");
        assert_eq!(h["referrer-policy"], "same-origin");
        assert_eq!(h["x-frame-options"], "DENY");
        assert_eq!(h["content-security-policy"], guard::CSP);
        assert_eq!(h[header::CONTENT_TYPE], "text/html; charset=utf-8");
        let (s, h, _) = call(&state, from(REMOTE, "GET", "/stats.json", false)).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(h["x-content-type-options"], "nosniff");
        assert_eq!(h[header::CACHE_CONTROL], "no-store");
        assert!(!h.contains_key("content-security-policy"));
        // `/v1/config` keeps its own cache policy (ETag revalidation).
        let (_, h, _) = call(&state, from(REMOTE, "GET", "/v1/config", false)).await;
        assert_eq!(h[header::CACHE_CONTROL], "no-cache");
        // The 403 page and the panic page carry them too.
        let (s, h, _) = call(&state, from(REMOTE, "GET", "/config", true)).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        assert_eq!(h["content-security-policy"], guard::CSP);
        assert_eq!(h["x-frame-options"], "DENY");

        // Fallback: the path (with markup) is not in the answer, as JSON or HTML.
        let evil = "/no/such/%3Cscript%3Ealert(1)%3C/script%3E";
        let (s, h, b) = call(&state, from(REMOTE, "GET", evil, false)).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert!(
            h[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("application/json")
        );
        let body = String::from_utf8_lossy(&b);
        assert!(
            !body.contains("script") && !body.contains("no/such"),
            "{body}"
        );
        let (s, h, b) = call(&state, from(REMOTE, "GET", evil, true)).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert!(
            h[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("text/html")
        );
        let body = String::from_utf8_lossy(&b);
        assert!(
            !body.contains("script") && !body.contains("no/such"),
            "{body}"
        );
        assert_eq!(h["content-security-policy"], guard::CSP);
        let (s, _, _) = call(&state, from(REMOTE, "POST", "/nope", false)).await;
        assert_eq!(s, StatusCode::NOT_FOUND);

        // A method the path does not take: 405 (open route), 403 first on admin routes.
        for (method, uri) in [
            ("DELETE", "/v1/devices"),
            ("PUT", "/"),
            ("PATCH", "/stats.json"),
        ] {
            let (s, _, _) = call(&state, from(REMOTE, method, uri, false)).await;
            assert_eq!(s, StatusCode::METHOD_NOT_ALLOWED, "{method} {uri}");
        }
        let (s, _, _) = call(&state, from(LOCAL, "DELETE", "/logs", false)).await;
        assert_eq!(s, StatusCode::METHOD_NOT_ALLOWED);
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
