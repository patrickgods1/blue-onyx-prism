//! axum HTTP server: CodeProject.AI compatible detection endpoints plus the web UI
//! (welcome, stats, test, config and logs pages rendered with askama templates from `templates/`).
//! Handler flow and page set modeled on blue-onyx `server.rs` and its templates (MIT).
//!
//! Only the local network is answered ([`network`]): the accept loop closes a connection from
//! outside it before reading anything, and the router's outermost middleware
//! ([`guard::network_rule`]) refuses such a request, or one whose forwarding headers name such a
//! client, with 403 on every route.
//!
//! The listeners ([`listen`]): plain HTTP on `port` and HTTPS on `https_port` serve this same
//! router; HTTP never depends on HTTPS. The plain port answers the open routes only (Blue Iris,
//! status, static files; [`plain_gate`] sends browsers to HTTPS); everything else is HTTPS with a
//! signed-in session or the API token ([`admin_only`], [`auth`]), from every address alike.

use crate::cli::LogReloadHandle;
use crate::config::Config;
use crate::metrics::Metrics;
use crate::registry::ModelRegistry;
use crate::startup::ModelState;
use crate::sync::RwLockExt;
use crate::worker::WorkerHandle;
use askama::Template;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{MethodRouter, get, post};
use axum::{Json, Router};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::error;

mod accounts;
mod apply;
mod auth;
mod benchmark;
mod body_budget;
mod browse;
mod config_page;
mod dataset;
mod detect;
mod devices;
mod diagnostics;
#[cfg(test)]
mod gate_tests;
mod guard;
mod listen;
mod live;
mod network;
mod pages;
mod pick;
mod refine;
mod resources;
mod review;
mod sam3_access;
mod status;
mod storage;
#[cfg(test)]
mod test_util;
mod train_area;
mod train_jobs;
mod training;
mod training_switch;
mod training_tasks;

pub(crate) use apply::{apply_config, reload_models, sync_config};
pub(crate) use auth::{FailureLimit, Logins};
use config_page::{
    ConfigView, MessageTemplate, config_loglevel, config_models, config_models_remove, config_page,
    config_restart, config_submit, encode_uri_component, global_device_label, save_config,
    schedule_restart,
};
pub(crate) use detect::BODY_LIMIT;
use detect::{custom_list, detection_custom, detection_default};
use devices::model_selection;
use devices::{DevicesSnapshot, devices_json, devices_snapshot};
#[cfg(test)]
use listen::HEADER_READ_TIMEOUT;
pub use listen::{HttpsPlan, HttpsPort, Listeners, bind, serve_on};
use pages::{
    apple_touch_icon, favicon, live_js, logs_json, logs_page, prometheus, static_asset, stats_json,
    stats_page, style_css, test_page, test_submit, update_available, welcome,
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
    /// Wrong API tokens per peer (one list for the process, like the config).
    pub(crate) admin_failures: Arc<std::sync::Mutex<FailureLimit>>,
    /// Wrong passwords and setup codes per peer, the hashing permits (one for the process).
    pub(crate) login: Arc<Logins>,
    /// Accounts, sessions and device cookies (one for the process, like `admin_failures`, so a
    /// restart signs nobody out).
    pub(crate) auth: Arc<crate::accounts::Auth>,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("registry", &self.registry)
            .field("config_path", &self.config_path)
            .field("started", &self.started)
            .field("restart", &self.restart)
            .field("sessions", &self.auth.sessions.len())
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
        let auth = crate::accounts::Auth::load(&running.data_root(), crate::accounts::now());
        // Unit tests: an account and a session every test request carries unless it says
        // otherwise (`test_util::from_peer`).
        #[cfg(test)]
        test_util::seed_tester(&running.data_root(), &auth);
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
            admin_failures: Arc::default(),
            login: Arc::default(),
            auth: Arc::new(auth),
        }
    }

    /// What this generation runs now (see [`AppState::running`](field@AppState::running)).
    pub(crate) fn running(&self) -> Arc<Config> {
        self.running.load_full()
    }

    /// The current config: one atomic load that never waits, also not while a save writes the
    /// file (see [`crate::config_store`]).
    fn config_read(&self) -> crate::config_store::ConfigRef {
        self.config.read()
    }

    /// Recent log events for the Logs page (kept by the logging setup's reload handle).
    fn log_buffer(&self) -> Option<&crate::logbuf::LogBuffer> {
        self.log_reload.as_ref().map(LogReloadHandle::buffer)
    }
}

/// The open routes: Blue Iris (often on another machine), the readiness probe, static files and
/// the minimal `/v1/config` the pages' restart poller uses. No sign-in, on both ports, for every
/// peer of the local network (the network rule, [`network`], comes first on every route);
/// `router_classification` in the tests checks the list against the expected set, and
/// [`plain_gate`] serves exactly these on the plain port.
fn open_routes() -> Vec<(&'static str, MethodRouter<Shared>)> {
    vec![
        ("/v1/vision/detection", post(detection_default)),
        ("/v1/vision/custom/list", get(custom_list).post(custom_list)),
        ("/v1/vision/custom/{model}", post(detection_custom)),
        ("/v1/status", get(status::readiness)),
        ("/v1/status/updateavailable", get(update_available)),
        ("/v1/config", get(live::config_json)),
        ("/static/style.css", get(style_css)),
        ("/static/live.js", get(live_js)),
        ("/static/{file}", get(static_asset)),
        ("/favicon.ico", get(favicon)),
        ("/apple-touch-icon.png", get(apple_touch_icon)),
        ("/apple-touch-icon-precomposed.png", get(apple_touch_icon)),
    ]
}

/// Signing in, out and the first account: nobody is signed in yet, so they take no credential,
/// but they are HTTPS only ([`plain_gate`]) and pass the admin routes' `Host` check
/// ([`login_only`]) and the CSRF guard.
fn login_routes() -> Vec<(&'static str, MethodRouter<Shared>)> {
    vec![
        (
            "/login",
            get(auth::login_page)
                .post(auth::login_submit)
                .layer(DefaultBodyLimit::max(auth::LOGIN_BODY_LIMIT)),
        ),
        ("/logout", post(auth::logout)),
        (
            "/setup",
            get(auth::setup_page)
                .post(auth::setup_submit)
                .layer(DefaultBodyLimit::max(auth::LOGIN_BODY_LIMIT)),
        ),
    ]
}

/// The routes behind sign-in: HTTPS only ([`plain_gate`]), a session or the API token
/// ([`admin_only`]), under an IP address, `localhost` or an `allowed_hosts` name, and no
/// cross-site state changes ([`guard::csrf`]).
fn admin_routes() -> Vec<(&'static str, MethodRouter<Shared>)> {
    vec![
        ("/", get(welcome)),
        ("/stats.json", get(stats_json)),
        ("/prometheus", get(prometheus)),
        ("/v1/devices", get(devices_json)),
        ("/v1/resources", get(resources_json)),
        ("/v1/resources/download", post(resources_download)),
        ("/v1/resources/remove", post(resources_remove)),
        ("/v1/resources/add-to-config", post(resources_add_to_config)),
        ("/v1/models/add", post(models_add)),
        ("/v1/fs/browse", get(browse::browse)),
        ("/v1/fs/pick", get(pick::status).post(pick::pick)),
        ("/v1/resources/export", post(resources_export)),
        ("/v1/resources/export/cancel", post(resources_export_cancel)),
        ("/v1/storage", get(storage::status)),
        ("/v1/storage/clear", post(storage::clear)),
        ("/v1/storage/exports/remove", post(storage::remove_export)),
        ("/v1/diagnostics", get(diagnostics::download)),
        ("/stats", get(stats_page)),
        ("/test", get(test_page).post(test_submit)),
        ("/config", get(config_page).post(config_submit)),
        ("/config/models", post(config_models)),
        ("/config/models/remove", post(config_models_remove)),
        ("/config/restart", post(config_restart)),
        ("/config/loglevel", post(config_loglevel)),
        ("/config/admin-token/rotate", post(auth::rotate)),
        (
            "/config/accounts/add",
            post(accounts::add).layer(DefaultBodyLimit::max(auth::LOGIN_BODY_LIMIT)),
        ),
        (
            "/config/accounts/remove",
            post(accounts::remove).layer(DefaultBodyLimit::max(auth::LOGIN_BODY_LIMIT)),
        ),
        (
            "/config/accounts/unlock",
            post(accounts::unlock).layer(DefaultBodyLimit::max(auth::LOGIN_BODY_LIMIT)),
        ),
        (
            "/config/accounts/password",
            post(accounts::change_password).layer(DefaultBodyLimit::max(auth::LOGIN_BODY_LIMIT)),
        ),
        (
            "/config/accounts/sign-out-everywhere",
            post(accounts::sign_out_everywhere)
                .layer(DefaultBodyLimit::max(auth::LOGIN_BODY_LIMIT)),
        ),
        ("/train", get(train_area::entry)),
        ("/train/collect", get(dataset::page)),
        ("/train/import", get(training::import_tab)),
        ("/train/autolabel", get(training::autolabel_tab)),
        ("/train/review", get(review::page)),
        ("/train/refine", get(refine::page)),
        ("/v1/refine", get(refine::status)),
        ("/v1/refine/preview", get(refine::preview)),
        (
            "/v1/refine/check",
            post(refine::start).layer(DefaultBodyLimit::max(refine::BODY_LIMIT)),
        ),
        ("/train/dataset", get(train_area::dataset_page)),
        ("/train/train", get(training::train_tab)),
        ("/train/models", get(training::models_tab)),
        ("/train/jobs", get(train_jobs::page)),
        ("/v1/train/summary", get(train_area::summary)),
        ("/v1/train/jobs", get(train_jobs::jobs)),
        ("/dataset", get(train_area::old_dataset)),
        ("/dataset/settings", post(dataset::settings)),
        ("/dataset/review", post(dataset::review)),
        ("/dataset/resume", post(dataset::resume)),
        ("/dataset/image", get(dataset::image)),
        ("/dataset/sidecar", get(dataset::sidecar)),
        ("/v1/dataset", get(dataset::json)),
        ("/review", get(train_area::old_review)),
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
            "/v1/review/reject-other-class",
            post(review::reject_other_class).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        (
            "/v1/review/reject-unwanted-reviewed",
            post(review::reject_unwanted_reviewed)
                .layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        (
            "/v1/review/reject-stacked",
            post(review::reject_stacked).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        (
            "/v1/review/restore-reopened",
            post(review::restore_reopened).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        (
            "/v1/review/suggest-overlaps",
            post(review::suggest_overlaps).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
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
        ("/v1/review/history", get(review::history)),
        (
            "/v1/review/reject-spot",
            post(review::reject_spot).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        (
            "/v1/review/accept-spot",
            post(review::accept_spot).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        (
            "/v1/review/accept-repeats",
            post(review::accept_repeats).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        ("/v1/review/spots", get(review::spots)),
        (
            "/v1/review/spots/parked",
            post(review::spots_parked).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        (
            "/v1/review/spots/retire",
            post(review::spots_retire).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        (
            "/v1/review/spots/moved",
            post(review::spots_moved).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        (
            "/v1/review/folder",
            post(review::folder).layer(DefaultBodyLimit::max(review::SMALL_BODY_LIMIT)),
        ),
        ("/training", get(train_area::old_training)),
        ("/v1/training", get(training::json)),
        (
            "/v1/training/metric-columns",
            post(training::metric_columns).layer(DefaultBodyLimit::max(16 * 1024)),
        ),
        (
            "/v1/training/dataset-classes",
            get(training::dataset_classes),
        ),
        (
            "/v1/training/balance-preview",
            post(training::balance_preview).layer(DefaultBodyLimit::max(256 * 1024)),
        ),
        ("/v1/training/compare", get(training::compare_json)),
        ("/training/queue", post(training::queue)),
        ("/training/cancel", post(training::cancel)),
        ("/training/finish", post(training::finish)),
        ("/training/pause", post(training::pause)),
        ("/training/resume", post(training::resume)),
        ("/training/remove", post(training::remove)),
        ("/training/retry-export", post(training::retry_export)),
        ("/training/use", post(training::use_model)),
        ("/training/revert", post(training_switch::revert)),
        ("/v1/training/switch-preview", get(training_switch::preview)),
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
            "/training/autolabel/preview",
            post(training_tasks::autolabel_preview),
        ),
        (
            "/v1/autolabel/models",
            get(training_tasks::autolabel_models),
        ),
        ("/v1/autolabel/sam3", get(sam3_access::status)),
        ("/v1/autolabel/sam3/download", post(sam3_access::download)),
        ("/v1/autolabel/sam3/remove", post(sam3_access::remove)),
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
        (
            "/v1/benchmark/datasets/remove",
            post(benchmark::remove_dataset),
        ),
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

/// How a connection reached the server, a request extension the accept loop adds next to
/// `ConnectInfo`: the HTTP port (`Plain`) or the HTTPS port (`Tls`, [`listen`]). The plain port
/// serves the open routes only ([`plain_gate`]); a request without the marker (none outside
/// tests) counts as plain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Transport {
    Plain,
    Tls,
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
    let mut login = Router::new();
    for (path, method) in login_routes() {
        login = login.route(path, method);
    }
    let login = login
        .route_layer(axum::middleware::from_fn(guard::csrf))
        .route_layer(axum::middleware::from_fn_with_state(
            shared.clone(),
            login_only,
        ));
    #[cfg(test)]
    let open = open
        .route("/__test/panic", get(test_panic))
        .route("/__test/blocking-panic", get(test_blocking_panic))
        .route("/__test/transport", get(test_transport));
    open.merge(admin)
        .merge(login)
        .fallback(fallback)
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .layer(axum::middleware::from_fn_with_state(
            shared.clone(),
            plain_gate,
        ))
        .layer(axum::middleware::from_fn_with_state(
            shared.clone(),
            guard::network_rule,
        ))
        .layer(axum::middleware::from_fn(guard::legacy_cookie))
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

/// Test-only route: the request's [`Transport`] marker, as `Debug` (`None` without one).
#[cfg(test)]
#[allow(clippy::unused_async, reason = "axum handlers are async")]
async fn test_transport(t: Option<axum::Extension<Transport>>) -> String {
    format!("{:?}", t.map(|t| t.0))
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

/// Whether the peer is this machine. A request without a peer address (not from [`serve_on`])
/// is not.
pub(super) fn loopback_peer(peer: Option<std::net::SocketAddr>) -> bool {
    peer.is_some_and(|a| match a.ip() {
        std::net::IpAddr::V4(ip) => ip.is_loopback(),
        // `::1`, or an IPv4 loopback client on a dual-stack socket (`::ffff:127.0.0.1`).
        std::net::IpAddr::V6(ip) => {
            ip.is_loopback() || ip.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
        }
    })
}

/// Whether the client is a person on this machine: a loopback peer with no forwarding header.
/// Never a way in (every admin route needs a sign-in); an extra condition of what only makes
/// sense on the server's own screen or must not be done remotely: the native file dialog
/// (`pick`) and the API token's rotation (`auth::rotate`). A proxy that adds no forwarding
/// header cannot be told apart.
pub(super) fn direct_local(peer: Option<std::net::SocketAddr>, headers: &HeaderMap) -> bool {
    loopback_peer(peer) && guard::proxy_header(headers).is_none()
}

/// The peer of a request (what [`serve_on`] adds).
fn peer_of(req: &axum::extract::Request) -> Option<std::net::SocketAddr> {
    req.extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|c| c.0)
}

/// The request's [`Transport`]; none counts as plain.
fn transport_of(req: &axum::extract::Request) -> Transport {
    req.extensions()
        .get::<Transport>()
        .copied()
        .unwrap_or(Transport::Plain)
}

/// The `Host` check of the admin and login routes: 403 (an HTML page for browsers, else JSON)
/// under a host name that could be a DNS rebinding ([`guard::foreign_host`]). `None` passes.
/// `allowed_hosts` is read per request, so a change applies at once.
fn host_refusal(state: &AppState, req: &axum::extract::Request) -> Option<Response> {
    let foreign = {
        let c = state.config_read();
        guard::foreign_host(req.headers(), req.uri(), &c.allowed_hosts)
    };
    let host = foreign?;
    Some(auth::no_store(guard::host_refused(
        req.headers(),
        req.uri().path(),
        &host,
    )))
}

/// Middleware of the admin routes, every method (inside [`plain_gate`], so HTTPS only):
/// [`host_refusal`], then a session cookie or the API token ([`auth::authenticate`]); a browser
/// without either goes to `/login` (or `/setup` before the first account), other clients get a
/// JSON 401, too many wrong tokens 429. No address passes without one. A signed-in request
/// carries [`auth::SignedIn`], and its HTML pages show the account's name.
pub(super) async fn admin_only(
    State(state): State<Arc<AppState>>,
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if let Some(refusal) = host_refusal(&state, &req) {
        return refusal;
    }
    let peer = peer_of(&req);
    match auth::authenticate(&state, peer, req.headers()).await {
        auth::Outcome::SignedIn(signed) => {
            let user = (signed.via == auth::Via::Session).then(|| signed.user.clone());
            let poll_load = auth::cookie(req.headers(), auth::POLL_COOKIE).is_some();
            req.extensions_mut().insert(signed);
            let resp = next.run(req).await;
            let mut resp = match user {
                Some(user) => auth::show_user(resp, &user).await,
                None => resp,
            };
            if poll_load {
                auth::clear_poll_cookie(&mut resp);
            }
            resp
        }
        other => auth::refused(&other, req.method(), req.uri(), req.headers(), peer),
    }
}

/// Middleware of `/login`, `/logout` and `/setup`: [`host_refusal`], with no credential.
pub(super) async fn login_only(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    match host_refusal(&state, &req) {
        Some(refusal) => refusal,
        None => next.run(req).await,
    }
}

/// The paths of the open routes, `{param}` segments as wildcards (built once).
fn open_paths() -> &'static [&'static str] {
    static PATHS: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    PATHS.get_or_init(|| {
        let mut paths: Vec<&'static str> = open_routes().into_iter().map(|(p, _)| p).collect();
        if cfg!(test) {
            paths.extend([
                "/__test/panic",
                "/__test/blocking-panic",
                "/__test/transport",
            ]);
        }
        paths
    })
}

/// Whether `path` is one of the open routes ([`open_paths`]; a `{param}` segment matches any one
/// non-empty segment).
fn is_open_path(path: &str) -> bool {
    open_paths().iter().any(|pattern| {
        let (mut want, mut got) = (pattern.split('/'), path.split('/'));
        loop {
            match (want.next(), got.next()) {
                (None, None) => return true,
                (Some(w), Some(g)) if w.starts_with('{') && w.ends_with('}') => {
                    if g.is_empty() {
                        return false;
                    }
                }
                (Some(w), Some(g)) if w == g => {}
                _ => return false,
            }
        }
    })
}

/// The HTTPS address of what a plain-port request asked for: `https://<its host>:<port><path>`
/// under the request's own `Host` (when it may be answered under it, [`guard::host_allowed`]),
/// without the client's port, the path and query through [`auth::safe_next`] (else `/`). None:
/// no usable host.
fn https_location(req: &axum::extract::Request, port: u16, allowed: &[String]) -> Option<String> {
    let host = guard::request_host(req.headers(), req.uri())?;
    if !guard::host_allowed(&host, allowed) {
        return None;
    }
    let name = guard::host_part(&host)?;
    let name = if name.contains(':') {
        format!("[{name}]")
    } else {
        name
    };
    let path = req
        .uri()
        .path_and_query()
        .map(|p| p.as_str())
        .and_then(auth::safe_next)
        .unwrap_or("/");
    Some(format!("https://{name}:{port}{path}"))
}

/// The transport gate (inside the network rule, outside everything else): the plain HTTP port
/// serves the open routes only. Anything else there: a browser's GET or HEAD is redirected (308)
/// to the same page over HTTPS ([`https_location`]; a foreign host name gets 403, a down HTTPS
/// listener 503); any other method or client gets a 403 JSON answer naming the HTTPS address. A
/// request with `Authorization` is refused (403) before anything is compared or counted: the
/// API token is for HTTPS only. No HSTS: with a self-signed certificate it would make the
/// browser's warning impossible to click through.
pub(super) async fn plain_gate(
    State(state): State<Arc<AppState>>,
    State(https): State<HttpsPort>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if transport_of(&req) == Transport::Tls || is_open_path(req.uri().path()) {
        return next.run(req).await;
    }
    let (allowed, configured) = {
        let c = state.config_read();
        (c.allowed_hosts.clone(), c.https_port)
    };
    let port = https.get();
    let shown_port = port.unwrap_or(configured);
    let location = https_location(&req, shown_port, &allowed);
    let url = location
        .clone()
        .unwrap_or_else(|| format!("https://<this server's address>:{shown_port}/"));
    let headers = req.headers();
    if headers.contains_key(header::AUTHORIZATION) {
        return auth::no_store(guard::refused(
            StatusCode::FORBIDDEN,
            &HeaderMap::new(),
            "Forbidden",
            format!(
                "This port is plain HTTP, where no credential is accepted (it would travel \
                 unencrypted). Send the API token to {url} instead; it may have been seen on \
                 the network, so consider rotating it (`blue-onyx-prism admin-token --rotate` \
                 on the server)."
            ),
        ));
    }
    let browser_get = matches!(*req.method(), Method::GET | Method::HEAD) && wants_html(headers);
    if !browser_get {
        return auth::no_store(guard::refused(
            StatusCode::FORBIDDEN,
            &HeaderMap::new(),
            "Forbidden",
            format!(
                "This port (plain HTTP) serves Blue Iris and the detection API only. The web UI \
                 and the admin API are at {url}, with a sign-in or `Authorization: Bearer <API \
                 token>`."
            ),
        ));
    }
    let Some(location) = location else {
        let host = guard::foreign_host(headers, req.uri(), &allowed).unwrap_or_default();
        return auth::no_store(guard::host_refused(headers, req.uri().path(), &host));
    };
    if port.is_none() {
        return auth::no_store(guard::refused(
            StatusCode::SERVICE_UNAVAILABLE,
            headers,
            "Web UI not available",
            format!(
                "The web UI is served over HTTPS on port {configured}, and that listener is not \
                 running right now: its port may be taken by another program or its \
                 certificate may not load. The server's log says which; it keeps trying. Blue \
                 Iris on this port is not affected."
            ),
        ));
    }
    let value = HeaderValue::from_str(&location).unwrap_or(HeaderValue::from_static("/"));
    auth::no_store(
        (
            StatusCode::PERMANENT_REDIRECT,
            [(header::LOCATION, value)],
            Html(
                "<!doctype html><title>Moved to HTTPS</title><h1>Moved to HTTPS</h1>\
                 <p>The web UI is served over HTTPS. Your browser warns once about the \
                 server's self-signed certificate.</p>",
            ),
        )
            .into_response(),
    )
}

/// See [`training::queue_busy`]: for the supervisor, which holds back GPU retries meanwhile.
pub(crate) fn training_queue_busy() -> bool {
    training::queue_busy()
}

/// The process-wide router state: the generation (`AppState`) that is current. A generation
/// restart swaps it (`Shared::swap`); the listener stays up. Handlers extract
/// `State<Arc<AppState>>` as before ([`axum::extract::FromRef`]), one `Arc` clone per request,
/// so an in-flight request keeps its generation (and its workers) alive until it finishes.
#[derive(Clone, Debug)]
pub struct Shared {
    current: Arc<std::sync::RwLock<Arc<AppState>>>,
    /// The bound HTTPS port of the process ([`listen`]): one for every generation.
    https: HttpsPort,
}

impl Shared {
    pub(crate) fn new(state: Arc<AppState>) -> Self {
        Self {
            current: Arc::new(std::sync::RwLock::new(state)),
            https: HttpsPort::default(),
        }
    }

    /// The bound HTTPS port of this process (`None` while the HTTPS listener is down).
    pub fn https_port(&self) -> HttpsPort {
        self.https.clone()
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

/// A handler can take `State<HttpsPort>` next to its `State<Arc<AppState>>`.
impl axum::extract::FromRef<Shared> for HttpsPort {
    fn from_ref(shared: &Shared) -> Self {
        shared.https_port()
    }
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

    /// Sessions belong to the process: the next generation's `AppState` gets the same
    /// [`crate::accounts::Auth`] (as `runner`'s `state_for` hands it over), so a session made
    /// before a restart still checks after the swap. `AppState::new` makes its own.
    #[test]
    fn sessions_survive_a_generation_swap() {
        use crate::accounts::sessions::Activity;
        let now = crate::accounts::now();
        let users = crate::accounts::Users {
            format: crate::accounts::FORMAT,
            users: vec![crate::accounts::User {
                name: "alice".into(),
                hash: "$argon2id$x".into(),
                created: now,
                password_changed: now,
                generation: 1,
                failures: 0,
                last_failure: None,
                disabled: false,
            }],
        };
        let first = test_state();
        let shared = Shared::new(first.clone());
        let id = shared.current().auth.sessions.create("alice", 1, now);
        let mut next = Arc::try_unwrap(test_state()).unwrap();
        assert!(
            next.auth.sessions.record(&id).is_none(),
            "a fresh state has its own (only the seeded test session)"
        );
        assert_eq!(next.auth.sessions.len(), 1);
        next.auth = first.auth.clone();
        drop(first);
        let old = shared.swap(Arc::new(next));
        drop(old);
        let current = shared.current();
        let valid = current
            .auth
            .sessions
            .check(id.reveal().as_bytes(), &users, now, Activity::User)
            .unwrap();
        assert_eq!(valid.user, "alice");
        assert!(
            format!("{current:?}").contains("sessions: 2"),
            "{current:?}"
        );
    }

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

    /// A token in the right shape, for the tests that are about something else.
    const TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    /// `req` with the API token instead of the seeded session.
    fn signed_in(req: Request<Body>) -> Request<Body> {
        bearer(req, TOKEN)
    }

    /// [`test_state_with`] with [`TOKEN`] in its API token file.
    fn remote_state_with_token(edit: impl FnOnce(&mut Config)) -> Arc<AppState> {
        let state = test_state_with(edit);
        std::fs::write(
            crate::admin_token::prepare(&state.config_read()).unwrap(),
            TOKEN,
        )
        .unwrap();
        state
    }

    /// What counts as this machine for the file dialog and the token's rotation (never a way in).
    #[test]
    fn loopback_peers_are_local() {
        let none = HeaderMap::new();
        let ok = |a: &str| direct_local(Some(a.parse().unwrap()), &none);
        assert!(ok("127.0.0.1:1"));
        assert!(ok("127.8.9.10:1"));
        assert!(ok("[::1]:1"));
        assert!(ok("[::ffff:127.0.0.1]:1"));
        assert!(!ok("10.0.0.5:1"));
        assert!(!ok("192.168.1.20:1"));
        assert!(!ok("[fe80::1]:1"));
        assert!(!ok("[::ffff:10.0.0.5]:1"));
        assert!(!direct_local(None, &none));
        assert!(loopback_peer(Some("127.0.0.1:1".parse().unwrap())));
        assert!(!loopback_peer(None));
    }

    /// SV-12, rewritten for sign-in: a loopback request carrying a proxy's forwarding header is no
    /// request of this machine (for the dialog and the rotation), and like every other request it
    /// needs a sign-in; signed in, it is served.
    #[tokio::test]
    async fn a_local_proxy_is_no_local_request() {
        let local: std::net::SocketAddr = "127.0.0.1:1".parse().unwrap();
        assert!(direct_local(Some(local), &HeaderMap::new()));
        for (name, value) in [
            ("x-forwarded-for", "192.168.1.9"),
            ("forwarded", "for=192.168.1.9"),
            ("x-real-ip", "192.168.1.9"),
        ] {
            let mut h = HeaderMap::new();
            h.insert(name, value.parse().unwrap());
            assert!(!direct_local(Some(local), &h), "{name}");
        }
        let state = test_state();
        let proxied = |r: Request<Body>| {
            let mut r = r;
            r.headers_mut()
                .insert("x-forwarded-for", "192.168.1.9".parse().unwrap());
            r
        };
        let req = proxied(anonymous(from("127.0.0.1:40000", "GET", "/config", true)));
        let (s, h, _) = call(&state, req).await;
        assert_eq!(s, StatusCode::SEE_OTHER);
        assert_eq!(h[header::LOCATION], "/login?next=%2Fconfig");
        let req = proxied(from("127.0.0.1:40000", "GET", "/config", true));
        assert_eq!(call(&state, req).await.0, StatusCode::OK);
    }

    /// Every route of the router, classified. A new route must be added here: the test fails
    /// when the router has a path this list does not (and the other way round), and probes each
    /// path: anonymous on HTTPS, admin routes want a sign-in (401), login routes take none, open
    /// routes never answer 401 or 403; on the plain port only the open routes are served.
    #[tokio::test]
    async fn router_classification() {
        const OPEN: &[&str] = &[
            "/v1/vision/detection",
            "/v1/vision/custom/list",
            "/v1/vision/custom/{model}",
            "/v1/status",
            "/v1/status/updateavailable",
            "/v1/config",
            "/favicon.ico",
            "/apple-touch-icon.png",
            "/apple-touch-icon-precomposed.png",
        ];
        const LOGIN: &[&str] = &["/login", "/logout", "/setup"];
        const ADMIN: &[&str] = &[
            "/",
            "/stats.json",
            "/prometheus",
            "/v1/devices",
            "/v1/resources",
            "/v1/resources/download",
            "/v1/resources/remove",
            "/v1/resources/add-to-config",
            "/v1/models/add",
            "/v1/fs/browse",
            "/v1/fs/pick",
            "/v1/resources/export",
            "/v1/resources/export/cancel",
            "/v1/storage",
            "/v1/storage/clear",
            "/v1/storage/exports/remove",
            "/v1/diagnostics",
            "/stats",
            "/test",
            "/config",
            "/config/models",
            "/config/models/remove",
            "/config/restart",
            "/config/loglevel",
            "/config/admin-token/rotate",
            "/config/accounts/add",
            "/config/accounts/remove",
            "/config/accounts/unlock",
            "/config/accounts/password",
            "/config/accounts/sign-out-everywhere",
            "/train",
            "/train/collect",
            "/train/import",
            "/train/autolabel",
            "/train/review",
            "/train/refine",
            "/v1/refine",
            "/v1/refine/preview",
            "/v1/refine/check",
            "/train/dataset",
            "/train/train",
            "/train/models",
            "/train/jobs",
            "/v1/train/summary",
            "/v1/train/jobs",
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
            "/v1/review/reject-other-class",
            "/v1/review/reject-unwanted-reviewed",
            "/v1/review/reject-stacked",
            "/v1/review/restore-reopened",
            "/v1/review/suggest-overlaps",
            "/v1/review/export",
            "/v1/review/import",
            "/v1/review/classes",
            "/v1/review/class-catalog",
            "/v1/review/history",
            "/v1/review/reject-spot",
            "/v1/review/accept-spot",
            "/v1/review/accept-repeats",
            "/v1/review/spots",
            "/v1/review/spots/parked",
            "/v1/review/spots/retire",
            "/v1/review/spots/moved",
            "/v1/review/folder",
            "/training",
            "/v1/training",
            "/v1/training/metric-columns",
            "/v1/training/dataset-classes",
            "/v1/training/balance-preview",
            "/v1/training/compare",
            "/training/queue",
            "/training/cancel",
            "/training/finish",
            "/training/pause",
            "/training/resume",
            "/training/remove",
            "/training/retry-export",
            "/training/use",
            "/training/revert",
            "/v1/training/switch-preview",
            "/training/datasets",
            "/training/dir",
            "/training/remove-env",
            "/training/import",
            "/v1/training/import/preview",
            "/training/autolabel",
            "/training/autolabel/preview",
            "/v1/autolabel/models",
            "/v1/autolabel/sam3",
            "/v1/autolabel/sam3/download",
            "/v1/autolabel/sam3/remove",
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
            "/v1/benchmark/datasets/remove",
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
        let login: Vec<&str> = login_routes().into_iter().map(|(p, _)| p).collect();
        assert_eq!(login, LOGIN, "login routes");
        for p in &admin {
            assert!(
                ADMIN.contains(p),
                "admin route {p} is not in the expected admin set"
            );
        }
        for p in ADMIN {
            assert!(admin.contains(p), "expected admin route {p} is not routed");
        }
        let mut all: Vec<&str> = open
            .iter()
            .chain(admin.iter())
            .chain(login.iter())
            .copied()
            .collect();
        let n = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), n, "a path is routed twice");

        let state = test_state();
        let probe = |p: &str| p.replace("{model}", "x").replace("{file}", "x.js");
        // The plain port's gate knows exactly the open routes.
        for p in &open {
            assert!(is_open_path(&probe(p)), "{p} is open but gated");
        }
        for p in admin.iter().chain(login.iter()) {
            assert!(!is_open_path(p), "{p} is not open but passes the gate");
        }
        for peer in [REMOTE, LOCAL] {
            for p in &admin {
                for method in ["GET", "POST"] {
                    let r = anonymous(from(peer, method, &probe(p), false));
                    let (s, h, _) = call(&state, r).await;
                    assert_eq!(
                        s,
                        StatusCode::UNAUTHORIZED,
                        "{method} {p} anonymous from {peer}"
                    );
                    assert_eq!(h[header::WWW_AUTHENTICATE], "Bearer", "{method} {p}");
                }
            }
        }
        for (method, p) in [
            ("GET", "/login"),
            ("POST", "/login"),
            ("POST", "/logout"),
            ("GET", "/setup"),
            ("POST", "/setup"),
        ] {
            let (s, _, b) = call(&state, anonymous(from(REMOTE, method, p, false))).await;
            let answer = String::from_utf8_lossy(&b);
            assert_ne!(s, StatusCode::FORBIDDEN, "{method} {p}: {answer}");
            // POST /login with no form is no sign-in; the route itself let the peer in.
            if p != "/login" || method == "GET" {
                assert_ne!(s, StatusCode::UNAUTHORIZED, "{method} {p}: {answer}");
            }
        }
        for p in &open {
            for method in ["GET", "POST"] {
                let (s, _, _) =
                    call(&state, anonymous(from(REMOTE, method, &probe(p), false))).await;
                assert_ne!(s, StatusCode::FORBIDDEN, "{method} {p} from a remote peer");
                assert_ne!(
                    s,
                    StatusCode::UNAUTHORIZED,
                    "{method} {p} from a remote peer"
                );
            }
        }
        // Signed in, every admin route is reached (whatever it answers then).
        for p in &admin {
            let (s, _, _) = call(&state, from(REMOTE, "GET", &probe(p), false)).await;
            assert!(![401, 403].contains(&s.as_u16()), "GET {p} signed in: {s}");
        }
        // The plain port: open routes as before, everything else refused (non-browser: 403).
        for p in &open {
            let (s, _, _) = call(&state, plain(from(REMOTE, "GET", &probe(p), false))).await;
            assert_ne!(s, StatusCode::FORBIDDEN, "GET {p} on the plain port");
        }
        for p in admin.iter().chain(login.iter()) {
            for method in ["GET", "POST"] {
                let (s, _, _) = call(&state, plain(from(REMOTE, method, &probe(p), false))).await;
                assert_eq!(s, StatusCode::FORBIDDEN, "{method} {p} on the plain port");
            }
        }
    }

    /// S1: a request as a DNS-rebound page makes it. The peer is this machine and `Origin`
    /// matches `Host`, so only the `Host` itself gives it away.
    fn with_host(mut req: Request<Body>, host: &str) -> Request<Body> {
        req.headers_mut()
            .insert(header::HOST, host.parse().unwrap());
        let origin = format!("https://{host}");
        req.headers_mut()
            .insert(header::ORIGIN, origin.parse().unwrap());
        req
    }

    /// S1: the admin routes refuse a DNS name they do not know, on every method, whatever the
    /// peer, and the `Origin == Host` check alone would have let these through.
    #[tokio::test]
    async fn dns_rebound_admin_requests_are_refused() {
        let state = test_state();
        let rebound = "rebind.attacker.test:32168";
        for (method, uri) in [
            ("GET", "/config"),
            ("POST", "/config"),
            ("GET", "/logs"),
            ("GET", "/dataset/image?path=x.jpg"),
            ("POST", "/config/restart"),
            ("GET", "/v1/fs/browse"),
        ] {
            for host in [rebound, "localhost.evil.test", "localhost.evil.test:32168"] {
                let (s, h, b) =
                    call(&state, with_host(from(LOCAL, method, uri, false), host)).await;
                let body = String::from_utf8_lossy(&b).to_string();
                assert_eq!(s, StatusCode::FORBIDDEN, "{method} {uri} {host}: {body}");
                assert!(
                    h[header::CONTENT_TYPE]
                        .to_str()
                        .unwrap()
                        .starts_with("application/json"),
                    "{method} {uri}"
                );
                let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
                assert_eq!(j["success"], false);
                let msg = j["message"].as_str().unwrap();
                assert_eq!(j["error"], msg);
                assert!(msg.contains(host), "{msg}");
                assert!(msg.contains("allowed_hosts"), "{msg}");
            }
            // Browsers get a page.
            let (s, h, b) = call(&state, with_host(from(LOCAL, method, uri, true), rebound)).await;
            assert_eq!(s, StatusCode::FORBIDDEN, "{method} {uri}");
            assert!(
                h[header::CONTENT_TYPE]
                    .to_str()
                    .unwrap()
                    .starts_with("text/html")
            );
            assert!(String::from_utf8_lossy(&b).contains("allowed_hosts"));
        }
        assert_eq!(state.config_read().port, Config::default().port);
        // Neither a session (above) nor the API token lifts it: the rebound browser may be on the
        // LAN.
        let open = remote_state_with_token(|_| {});
        let req = with_host(signed_in(from(REMOTE, "GET", "/config", false)), rebound);
        assert_eq!(call(&open, req).await.0, StatusCode::FORBIDDEN);
        // A remote peer, signed in or not, gets the same refusal: the name is the problem.
        for req in [
            from(REMOTE, "GET", "/config", false),
            anonymous(from(REMOTE, "GET", "/config", false)),
        ] {
            let (s, _, b) = call(&state, with_host(req, rebound)).await;
            assert_eq!(s, StatusCode::FORBIDDEN);
            assert!(String::from_utf8_lossy(&b).contains("allowed_hosts"));
        }
    }

    /// S1: IP literals, `localhost` and `allowed_hosts` names reach the handler.
    #[tokio::test]
    async fn admin_hosts_that_cannot_be_rebound_pass() {
        let state = test_state();
        for host in [
            "localhost",
            "localhost:32168",
            "LOCALHOST.:32168",
            "LocalHost.",
            "127.0.0.1:32168",
            "[::1]:32168",
            "[::1]",
        ] {
            let req = with_host(from(LOCAL, "GET", "/config", false), host);
            assert_eq!(call(&state, req).await.0, StatusCode::OK, "{host}");
        }
        // LAN administration by address, with a browser's matching Origin on a POST.
        let lan = remote_state_with_token(|_| {});
        for host in ["192.168.1.10:32168", "[fe80::1]:32168"] {
            let req = with_host(signed_in(from(REMOTE, "GET", "/config", false)), host);
            assert_eq!(call(&lan, req).await.0, StatusCode::OK, "{host}");
        }
        // A listed name, in any case, with or without a port or a trailing dot; its
        // subdomains and other names stay out.
        let named = remote_state_with_token(|c| c.allowed_hosts = vec!["nas.home.arpa".into()]);
        for host in [
            "nas.home.arpa",
            "NAS.Home.Arpa:32168",
            "nas.home.arpa.:32168",
        ] {
            let req = with_host(signed_in(from(REMOTE, "GET", "/config", false)), host);
            assert_eq!(call(&named, req).await.0, StatusCode::OK, "{host}");
        }
        for host in ["x.nas.home.arpa", "nas.home.arpa.evil.test", "home.arpa"] {
            let req = with_host(signed_in(from(REMOTE, "GET", "/config", false)), host);
            assert_eq!(call(&named, req).await.0, StatusCode::FORBIDDEN, "{host}");
        }
    }

    /// S1: a saved `allowed_hosts` is read per request, and the refusal never echoes more than
    /// 100 characters of what the client sent.
    #[tokio::test]
    async fn allowed_hosts_apply_at_once_and_the_refusal_stays_short() {
        let state = test_state();
        let req = with_host(from(LOCAL, "GET", "/config", false), "nas.home.arpa");
        assert_eq!(call(&state, req).await.0, StatusCode::FORBIDDEN);
        let body = "allowed_hosts=nas.home.arpa";
        let (s, _, b) = call(&state, post_form("/config", body, false)).await;
        assert_eq!(s, StatusCode::OK, "{}", String::from_utf8_lossy(&b));
        let req = with_host(from(LOCAL, "GET", "/config", false), "nas.home.arpa");
        assert_eq!(call(&state, req).await.0, StatusCode::OK);

        let long = format!("{}.test", "a".repeat(300));
        let req = with_host(from(LOCAL, "GET", "/config", false), &long);
        let (s, _, b) = call(&state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        let text = String::from_utf8_lossy(&b);
        assert!(text.contains(&"a".repeat(100)), "{text}");
        assert!(!text.contains(&"a".repeat(101)), "{text}");
    }

    /// S1: the open routes do not look at `Host`; Blue Iris may use any name it likes.
    #[tokio::test]
    async fn open_routes_ignore_the_host() {
        let state = test_state();
        for host in [
            "rebind.attacker.test:32168",
            "blue-iris-target",
            "localhost",
        ] {
            for (method, uri) in [
                ("POST", "/v1/vision/detection"),
                ("POST", "/v1/vision/custom/list"),
                ("GET", "/v1/status"),
                ("GET", "/favicon.ico"),
            ] {
                let (s, _, _) =
                    call(&state, with_host(from(REMOTE, method, uri, false), host)).await;
                assert_ne!(s, StatusCode::FORBIDDEN, "{method} {uri} {host}");
            }
        }
    }

    /// Whether `h` sets the session cookie.
    fn set_cookie_has_session(h: &HeaderMap) -> bool {
        h.get_all(header::SET_COOKIE).iter().any(|v| {
            v.to_str()
                .unwrap_or_default()
                .starts_with("__Host-bop_session=")
                && !v.to_str().unwrap_or_default().contains("Max-Age=0")
        })
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
        // Cross-site: 403 as JSON, and as a page for browsers; the config is untouched. Over
        // HTTPS (the test default) a plain-HTTP page of this very host and a same-site page are
        // foreign too.
        for headers in [
            vec![host, ("sec-fetch-site", "cross-site")],
            vec![host, ("origin", "http://evil.example")],
            vec![host, ("origin", "null")],
            vec![host, ("referer", "http://evil.example/page")],
            vec![host, ("origin", "https://localhost:8080")],
            vec![host, ("origin", "http://localhost:32168")],
            vec![host, ("referer", "http://localhost:32168/config")],
            vec![host, ("sec-fetch-site", "same-site")],
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
            assert!(!set_cookie_has_session(&h), "nothing signs in on a refusal");
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
            vec![host, ("origin", "https://localhost:32168")],
            vec![host, ("referer", "https://localhost:32168/config")],
            vec![host, ("sec-fetch-site", "same-origin")],
            vec![host],
            vec![
                host,
                ("origin", "https://LOCALHOST:32168"),
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

    /// SV-05: `GET /v1/config` gives a request that is not signed in (on any address, loopback
    /// included) only the restart poller's counters.
    #[tokio::test]
    async fn config_json_is_minimal_for_non_admin_peers() {
        let state = test_state();
        for peer in [REMOTE, LOCAL] {
            let (s, h, b) = call(&state, anonymous(from(peer, "GET", "/v1/config", false))).await;
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
                ],
                "{peer}"
            );
            assert!(j["revision"].is_u64() && j["generation"].is_u64() && j["epoch"].is_u64());
            assert!(h.contains_key(header::ETAG));
        }
        // The ETag round-trips to 304.
        let (_, h, _) = call(&state, anonymous(from(REMOTE, "GET", "/v1/config", false))).await;
        let etag = h[header::ETAG].to_str().unwrap().to_string();
        let req = anonymous(from_peer(
            Request::get("/v1/config")
                .header(header::IF_NONE_MATCH, etag)
                .body(Body::empty())
                .unwrap(),
            REMOTE,
        ));
        let (s, _, _) = call(&state, req).await;
        assert_eq!(s, StatusCode::NOT_MODIFIED);
        // Nothing of the config was read or written from disk for it.
        assert!(!state.config_path.exists());
        // A signed-in browser gets the full payload, from any address, and so does the API token
        // (more cases in `auth::tests`).
        let (_, _, b) = call(&state, from(REMOTE, "GET", "/v1/config", false)).await;
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert!(j["config"].is_object() && j["models"].is_array() && j["path"].is_string());
        std::fs::write(
            crate::admin_token::prepare(&state.config_read()).unwrap(),
            TOKEN,
        )
        .unwrap();
        let req = signed_in(from(REMOTE, "GET", "/v1/config", false));
        let (_, _, b) = call(&state, req).await;
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert!(j["config"].is_object());
        // The plain port never gives it, whatever the request shows.
        let mut req = plain(from(LOCAL, "GET", "/v1/config", false));
        req.headers_mut().insert(
            header::COOKIE,
            format!("__Host-bop_session={TEST_SESSION}")
                .parse()
                .unwrap(),
        );
        let (s, _, b) = call(&state, req).await;
        assert_eq!(s, StatusCode::OK);
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert!(j.get("config").is_none(), "{j}");
    }

    /// `GET /v1/config` is an open route with its own gate, so [`admin_only`] never sees it: a
    /// rebound name gets the minimal answer of a stranger, even from this machine.
    #[tokio::test]
    async fn the_config_payload_is_not_served_to_a_rebound_name() {
        let state = test_state();
        let rebound = "rebind.attacker.test:32168";
        let req = with_host(from(LOCAL, "GET", "/v1/config", false), rebound);
        let (s, _, b) = call(&state, req).await;
        assert_eq!(s, StatusCode::OK);
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert!(j.get("config").is_none() && j.get("path").is_none(), "{j}");
        // The names an admin page is opened under keep the full payload.
        for host in ["127.0.0.1:32168", "localhost:32168", "[::1]:32168"] {
            let req = with_host(from(LOCAL, "GET", "/v1/config", false), host);
            let (_, _, b) = call(&state, req).await;
            let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
            assert!(j["config"].is_object(), "{host}");
        }
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
        // The 403 page carries them too (and no-store: nobody is signed in to it).
        let req = with_host(from(REMOTE, "GET", "/config", true), "rebind.attacker.test");
        let (s, h, _) = call(&state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        assert_eq!(h[header::CACHE_CONTROL], "no-store");
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

        // A method the path does not take: 405 (open route, or an admin route signed in).
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

    /// Every admin route needs a sign-in, from every address: loopback is no exception any more.
    /// A browser is sent to the login page, others get a JSON 401; signed in, a remote peer of
    /// the local network gets the route. Blue Iris's routes stay open to everyone on the LAN.
    #[tokio::test]
    async fn admin_routes_need_a_sign_in_from_every_address() {
        let state = test_state();
        let admin = [
            ("GET", "/config"),
            ("POST", "/config"),
            ("GET", "/v1/fs/browse"),
            ("POST", "/v1/resources/export"),
            ("POST", "/config/restart"),
            ("GET", "/"),
            ("GET", "/stats.json"),
            ("GET", "/prometheus"),
            ("GET", "/v1/devices"),
        ];
        for (method, uri) in admin {
            for peer in [REMOTE, LOCAL, "[::1]:50000"] {
                let (s, h, b) = call(&state, anonymous(from(peer, method, uri, false))).await;
                let body = String::from_utf8_lossy(&b).to_string();
                assert_eq!(s, StatusCode::UNAUTHORIZED, "{method} {uri} {peer}: {body}");
                assert_eq!(h[header::WWW_AUTHENTICATE], "Bearer");
                assert!(
                    h[header::CONTENT_TYPE]
                        .to_str()
                        .unwrap()
                        .starts_with("application/json"),
                    "{method} {uri}"
                );
                assert_eq!(h[header::CACHE_CONTROL], "no-store");
                let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
                assert_eq!(j["success"], false);
                assert!(j["message"].as_str().unwrap().contains("/login"), "{j}");
                // Browsers go to the login page (back here only for a GET).
                let (s, h, _) = call(&state, anonymous(from(peer, method, uri, true))).await;
                assert_eq!(s, StatusCode::SEE_OTHER, "{method} {uri}");
                let to = h[header::LOCATION].to_str().unwrap();
                if method == "GET" {
                    assert_eq!(to, format!("/login?next={}", encode_uri_component(uri)));
                } else {
                    assert_eq!(to, "/login");
                }
                assert_eq!(h[header::CACHE_CONTROL], "no-store");
            }
            // Signed in: the route, from another machine too.
            let (s, _, b) = call(&state, from(REMOTE, method, uri, true)).await;
            assert!(
                ![401, 403].contains(&s.as_u16()),
                "{method} {uri}: {}",
                String::from_utf8_lossy(&b)
            );
        }
        assert!(!state.restart.is_cancelled());

        // Blue Iris's routes, the readiness probe and static files stay open to remote clients.
        for uri in [
            "/v1/config",
            "/static/style.css",
            "/favicon.ico",
            "/apple-touch-icon.png",
            "/static/manifest.webmanifest",
            "/v1/vision/custom/list",
        ] {
            let (s, _, _) = call(&state, anonymous(from(REMOTE, "GET", uri, false))).await;
            assert_eq!(s, StatusCode::OK, "{uri}");
        }
        let (s, _, b) = call(
            &state,
            anonymous(from_peer(
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
            )),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let r: VisionDetectionResponse = serde_json::from_slice(&b).unwrap();
        assert!(!r.success, "the test models failed to load: {r:?}");
        assert!(r.error.unwrap().contains("ipcam-general"));
    }
}
