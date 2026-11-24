//! The listeners: plain HTTP on `port` (Blue Iris and the detection API) and HTTPS on
//! `https_port` (the web UI; certificate from [`crate::tls`]), bound once per process and serving
//! the same router.
//!
//! - **HTTP does not depend on HTTPS.** [`bind`] binds HTTP only, and its failure is the
//!   process's start-up error; `/v1/status`, systemd's `READY=1` and the Windows service's
//!   Running all follow it. The HTTPS side ([`Listeners::with_https`]) starts inside
//!   [`serve_on`]: when its certificate does not load or its port is taken, that is logged once
//!   (with what holds the port, [`crate::port_holder`]) and tried again every [`HTTPS_RETRY`]
//!   until it works or the server stops.
//! - **The bound HTTPS port** is process-wide in [`HttpsPort`] (`Shared::https_port`, also an
//!   axum `State` extractor through `FromRef<Shared>`): `None` while the listener is down. A
//!   later handler that sends a browser to HTTPS reads it there.
//! - **Every accepted connection** passes the network rule first ([`guard::admit`]: a refused
//!   peer is closed before a byte is read or written, before any TLS handshake), then the
//!   connection caps ([`ConnCaps`]: [`MAX_CONNECTIONS`] open per listener, IPv4 and IPv6
//!   together, and [`MAX_CONNECTIONS_PER_PEER`] per address, an IPv6 one by its /64; over a cap
//!   the connection is closed at once).
//! - **TLS**: the handshake runs in the connection's own task under [`HANDSHAKE_TIMEOUT`], so a
//!   client that connects and never handshakes holds only its own slot. Then the same hyper
//!   builder as HTTP takes over ([`HEADER_READ_TIMEOUT`], keep-alive, graceful shutdown).
//! - **A peer that stops reading** (audit 2026-11, RA-032): every accepted socket is wrapped,
//!   below TLS, in [`super::deadline::WriteStall`], so a write that waits on the peer for
//!   [`super::deadline::WRITE_STALL_TIMEOUT`] closes the connection and gives its slot back. A
//!   request body that stops arriving is ended by the router ([`super::deadline::IdleBody`]).
//! - **Repeated failures stay cheap** (audit 2026-11, phase 6): while the HTTPS port stays taken,
//!   what holds it is looked up ([`crate::port_holder`], which runs `netstat` + `tasklist` on
//!   Windows) only when the bind error changes or the last lookup is [`HOLDER_REFRESH`] old
//!   ([`HolderCache`]); the attempt itself stays every [`HTTPS_RETRY`]. A lasting accept failure
//!   logs at most one ERROR per [`ACCEPT_ERROR_LOG_EVERY`], with the number held back.
//! - **Stop** (audit 2026-11, RA-023): every connection task is tracked per accept loop
//!   ([`TaskTracker`]). On shutdown the loop stops accepting, lets hyper finish in-flight requests
//!   and close idle keep-alive connections for at most [`GRACEFUL_SHUTDOWN_TIMEOUT`], then cancels
//!   a hard-stop token every connection task selects on, so the connections still open are dropped
//!   with their router clones (and through them the last `AppState`), and waits for the tasks for
//!   at most [`HARD_CLOSE_WAIT`]. [`serve_on`] therefore returns within about 11 s of the stop
//!   request. A request whose handler runs on the blocking pool loses its answer, but the blocking
//!   closure itself runs on (it cannot be interrupted from here).
//!
//! # The per-address cap and Blue Iris
//!
//! Blue Iris is one address that sends every camera's frames, possibly many at once: the
//! detection path allows [`super::detect::BODY_READS_PER_PEER`] (32) uploads being read per
//! address, and does not limit the frames past that stage (read, queued or being inferred), up to
//! the server-wide in-flight cap (the models' queue room plus a margin, up to 64 + 16 for one
//! model). Each of those requests holds a connection. A fixed 32 per address on the HTTP port
//! would close connections the detection path would serve, so there the per-address cap is the
//! larger of 32 and the current in-flight cap ([`plain_per_peer`]); a request beyond it gets the
//! detection path's own 503 answer rather than a closed socket. The HTTPS port carries browsers
//! (a handful of connections each), so 32 applies as is.

use super::{Shared, Transport, guard};
use crate::logthrottle::Throttle;
use crate::sync::LockExt;
use axum::Router;
use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;
use tracing::{debug, error, info, warn};

/// Longest a client may take to send a request head. With hyper 1.x HTTP/1 this timer also
/// runs while a keep-alive connection waits for its next request (hyper restarts it as soon as
/// a response is done: `proto/h1/conn.rs`, `poll_read_head`), so it is the idle timeout too: a
/// connection silent for this long is closed (slowloris and idle keep-alive alike).
pub(crate) const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(10);

/// How long a shutdown waits for in-flight requests to finish before closing their
/// connections anyway.
pub(crate) const GRACEFUL_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(10);

/// After [`GRACEFUL_SHUTDOWN_TIMEOUT`], how long the accept loop waits for the connection tasks
/// it just told to close (dropping a connection's future takes microseconds; the bound only
/// keeps a task stuck in a synchronous call from holding the stop).
pub(crate) const HARD_CLOSE_WAIT: Duration = Duration::from_secs(1);

/// The longest [`serve_on`] takes to return after its stop token is cancelled (the runner's
/// stop timeline budgets the HTTP step with it).
pub const SERVER_STOP_BOUND: Duration = GRACEFUL_SHUTDOWN_TIMEOUT.saturating_add(HARD_CLOSE_WAIT);

/// Pause after an accept error that is not about one connection (e.g. out of file
/// descriptors), so the loop does not spin.
const ACCEPT_ERROR_PAUSE: Duration = Duration::from_secs(1);

/// At most one ERROR line per this long about failed accepts (audit 2026-11, RA-119). A lasting
/// failure (out of file descriptors or handles) fails every [`ACCEPT_ERROR_PAUSE`] on up to four
/// sockets (IPv4 and IPv6, HTTP and HTTPS); unthrottled that is some 345 000 lines a day, and on
/// Windows each is an Error record in the Event Log that rolls the Application log over and buries
/// the first line, the one that shows the cause. A minute keeps the fault visible.
const ACCEPT_ERROR_LOG_EVERY: Duration = Duration::from_secs(60);

/// Failed accepts logged by this process (every listener: one kind of line).
static ACCEPT_ERRORS: Throttle = Throttle::new(ACCEPT_ERROR_LOG_EVERY);

/// How long a port-holder lookup is reused while the HTTPS bind keeps failing the same way
/// (audit 2026-11, RA-033). The lookup spawns `netstat -ano` and `tasklist` on Windows (`lsof` /
/// `ss` elsewhere) and its result only feeds a message that repeats at debug level, so doing it
/// on every [`HTTPS_RETRY`] attempt cost some 2 900 lookups a day on an idle service. Ten minutes
/// still notices a different program taking the port the same day.
pub(crate) const HOLDER_REFRESH: Duration = Duration::from_secs(10 * 60);

/// Longest a TLS handshake may take before its connection is closed.
pub(crate) const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// Pause between attempts to start the HTTPS listener.
pub(crate) const HTTPS_RETRY: Duration = Duration::from_secs(30);

/// Open connections per listener (its IPv4 and IPv6 sockets together).
pub(crate) const MAX_CONNECTIONS: usize = 256;

/// Open connections per address on the HTTPS listener, and the least on the HTTP one (see the
/// module docs).
pub(crate) const MAX_CONNECTIONS_PER_PEER: usize = 32;

/// At most one log line per this long about connections closed by a cap.
const CAP_WARN_EVERY: Duration = Duration::from_secs(60);

/// Connections closed by a cap, logged by this process.
static CAP_WARNS: Throttle = Throttle::new(CAP_WARN_EVERY);

/// The timings and caps of the listeners (shorter ones in tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Limits {
    pub(crate) total: usize,
    pub(crate) per_peer: usize,
    pub(crate) handshake_timeout: Duration,
    pub(crate) https_retry: Duration,
    /// [`GRACEFUL_SHUTDOWN_TIMEOUT`].
    pub(crate) graceful: Duration,
    /// [`HARD_CLOSE_WAIT`].
    pub(crate) hard_close: Duration,
    /// [`super::deadline::WRITE_STALL_TIMEOUT`].
    pub(crate) write_stall: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            total: MAX_CONNECTIONS,
            per_peer: MAX_CONNECTIONS_PER_PEER,
            handshake_timeout: HANDSHAKE_TIMEOUT,
            https_retry: HTTPS_RETRY,
            graceful: GRACEFUL_SHUTDOWN_TIMEOUT,
            hard_close: HARD_CLOSE_WAIT,
            write_stall: super::deadline::WRITE_STALL_TIMEOUT,
        }
    }
}

/// What the HTTPS listener needs: where to bind (the same address rules as HTTP) and its
/// certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpsPlan {
    pub address: IpAddr,
    /// The configured `https_port` (0: any free port, for tests).
    pub port: u16,
    pub source: crate::tls::Source,
}

impl HttpsPlan {
    /// The plan of `config`: its `bind_address`, `https_port` and certificate source.
    pub fn of(config: &crate::config::Config) -> Self {
        Self {
            address: config.bind_address,
            port: config.https_port,
            source: crate::tls::Source::of(config),
        }
    }
}

/// The process's listeners: HTTP bound once ([`bind`]), HTTPS started by [`serve_on`].
#[derive(Debug)]
pub struct Listeners {
    pub(super) v4: tokio::net::TcpListener,
    pub(super) v6: Option<tokio::net::TcpListener>,
    port: u16,
    https: Option<HttpsPlan>,
    limits: Limits,
}

impl Listeners {
    /// The bound HTTP port (useful after binding port 0).
    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    /// Serve HTTPS as `plan` says too (see the module docs).
    #[must_use]
    pub fn with_https(mut self, plan: HttpsPlan) -> Self {
        self.https = Some(plan);
        self
    }

    /// Other timings and caps (tests).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }
}

/// The bound HTTPS port of this process; `None` while the HTTPS listener is down (not started
/// yet, its port taken, its certificate not loading). One per [`Shared`], so it survives
/// generation restarts.
#[derive(Debug, Clone, Default)]
pub struct HttpsPort(Arc<AtomicU16>);

impl HttpsPort {
    /// The port HTTPS answers on now.
    pub fn get(&self) -> Option<u16> {
        match self.0.load(Ordering::Acquire) {
            0 => None,
            p => Some(p),
        }
    }

    pub(crate) fn set(&self, port: Option<u16>) {
        self.0.store(port.unwrap_or(0), Ordering::Release);
    }
}

/// Bind `address:port`, and for `0.0.0.0` and `127.0.0.1` also `[::1]:port` (best effort,
/// `localhost` may resolve to it, and some clients do not fall back to 127.0.0.1).
async fn bind_pair(
    address: IpAddr,
    port: u16,
) -> std::io::Result<(
    tokio::net::TcpListener,
    Option<tokio::net::TcpListener>,
    u16,
)> {
    let addr = SocketAddr::new(address, port);
    let v4 = tokio::net::TcpListener::bind(addr).await?;
    let port = v4.local_addr().map(|a| a.port()).unwrap_or(port);
    let v6 = match address {
        IpAddr::V4(a) if a.is_unspecified() || a.is_loopback() => {
            let loopback6 = SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, port));
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
    Ok((v4, v6, port))
}

/// Bind the HTTP listener on `address:port` (see [`bind_pair`]). Called once per process run: a
/// port in use is a startup error.
pub async fn bind(address: IpAddr, port: u16) -> anyhow::Result<Listeners> {
    let addr = SocketAddr::new(address, port);
    let (v4, v6, port) = bind_pair(address, port).await.map_err(|e| {
        anyhow::anyhow!("binding {addr}: {e} (is another instance running on port {port}?)")
    })?;
    info!("listening on {addr} (HTTP); Blue Iris uses http://<this machine>:{port}/");
    Ok(Listeners {
        v4,
        v6,
        port,
        https: None,
        limits: Limits::default(),
    })
}

/// Serve `shared` on `listeners` until `shutdown` is cancelled (in-flight requests finish,
/// for at most [`GRACEFUL_SHUTDOWN_TIMEOUT`]; then the connections still open are closed, so it
/// returns within [`SERVER_STOP_BOUND`]). Runs for the whole process: generation restarts
/// swap `shared` instead of stopping it. Connections are HTTP/1.1 served by hyper with a
/// timer, so [`HEADER_READ_TIMEOUT`] applies to request heads and idle keep-alive waits.
pub async fn serve_on(
    listeners: Listeners,
    shared: Shared,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let Listeners {
        v4,
        v6,
        port,
        https,
        limits,
    } = listeners;
    let app = super::router(shared.clone());
    let base = Loop {
        app,
        shared,
        shutdown,
        caps: Arc::new(ConnCaps::default()),
        limits,
        accept: Accept::Plain,
    };
    let http = run_pair(v4, v6, base.clone());
    match https {
        Some(plan) => {
            let tls = Loop {
                caps: Arc::new(ConnCaps::default()),
                ..base
            };
            tokio::join!(http, serve_https(plan, tls, port));
        }
        None => http.await,
    }
    info!("HTTP server stopped");
    Ok(())
}

/// The accept loops of one listener's IPv4 and IPv6 sockets.
async fn run_pair(v4: tokio::net::TcpListener, v6: Option<tokio::net::TcpListener>, l: Loop) {
    match v6 {
        Some(l6) => {
            tokio::join!(accept_loop(v4, l.clone()), accept_loop(l6, l));
        }
        None => accept_loop(v4, l).await,
    }
}

/// Start the HTTPS listener of `plan` and serve on it until `l.shutdown`; attempts that fail
/// are logged (a repeated failure once, at debug afterwards) and tried again every
/// `l.limits.https_retry`. `http_port`: for the message (HTTP keeps serving).
async fn serve_https(plan: HttpsPlan, l: Loop, http_port: u16) {
    let mut resolver: Option<crate::tls::Resolver> = None;
    let mut last: Option<String> = None;
    let mut holder = HolderCache::default();
    loop {
        if l.shutdown.is_cancelled() {
            return;
        }
        let attempt = start_https(
            &plan,
            &mut resolver,
            &mut holder,
            http_port,
            l.limits.https_retry,
        );
        match attempt.await {
            Ok((config, v4, v6, port, fingerprint)) => {
                l.shared.https_port().set(Some(port));
                info!("HTTPS on https://localhost:{port}/ (certificate SHA-256 {fingerprint})");
                let tls = Loop {
                    accept: Accept::Tls(tokio_rustls::TlsAcceptor::from(config)),
                    ..l.clone()
                };
                run_pair(v4, v6, tls).await;
                l.shared.https_port().set(None);
                return;
            }
            Err(message) => {
                if last.as_deref() == Some(message.as_str()) {
                    debug!("{message}");
                } else {
                    warn!("{message}");
                }
                last = Some(message);
                tokio::select! {
                    () = l.shutdown.cancelled() => return,
                    () = tokio::time::sleep(l.limits.https_retry) => {}
                }
            }
        }
    }
}

/// What a started HTTPS listener runs with.
type Started = (
    Arc<rustls::ServerConfig>,
    tokio::net::TcpListener,
    Option<tokio::net::TcpListener>,
    u16,
    String,
);

/// One attempt: the certificate (loaded once; own files that fail are read again next time),
/// then the bind (what holds a taken port through `holder`). `Err`: the message to log.
async fn start_https(
    plan: &HttpsPlan,
    resolver: &mut Option<crate::tls::Resolver>,
    holder: &mut HolderCache,
    http_port: u16,
    retry: Duration,
) -> Result<Started, String> {
    let after = format!(
        "plain HTTP keeps serving on port {http_port} (Blue Iris is not affected); trying again \
         every {} s",
        retry.as_secs().max(1)
    );
    let r = match resolver {
        Some(r) => r.clone(),
        None => {
            let source = plan.source.clone();
            let loaded = tokio::task::spawn_blocking(move || crate::tls::Resolver::new(source))
                .await
                .map_err(|e| {
                    format!("the HTTPS certificate check did not finish ({e}); {after}")
                })?;
            let r = loaded
                .map_err(|e| format!("HTTPS listener (web UI) not started: {e:#}; {after}"))?;
            *resolver = Some(r.clone());
            r
        }
    };
    let config = crate::tls::server_config(&r)
        .map_err(|e| format!("HTTPS listener (web UI) not started: {e:#}; {after}"))?;
    let addr = SocketAddr::new(plan.address, plan.port);
    match bind_pair(plan.address, plan.port).await {
        Ok((v4, v6, port)) => Ok((config, v4, v6, port, r.fingerprint())),
        Err(e) => {
            let port = plan.port;
            let held = if port == 0 {
                String::new()
            } else {
                holder
                    .text(port, &e, Instant::now(), crate::port_holder::lookup)
                    .await
            };
            Err(format!(
                "HTTPS listener (web UI) not started: binding {addr}: {e}{held}. Stop the program \
                 that holds the port, or set another `https_port` in the config file and restart \
                 the service; {after}"
            ))
        }
    }
}

/// How a bind failed, as far as the cached port-holder text depends on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BindFailure {
    kind: std::io::ErrorKind,
    os: Option<i32>,
}

impl BindFailure {
    fn of(e: &std::io::Error) -> Self {
        Self {
            kind: e.kind(),
            os: e.raw_os_error(),
        }
    }
}

/// The last port-holder lookup of the HTTPS retry loop (see the module docs).
#[derive(Debug, Default)]
struct HolderCache {
    last: Option<(BindFailure, Instant, String)>,
}

impl HolderCache {
    /// The `; port N is held by ...` part of the message for a bind of `port` that failed with
    /// `error` at `now`: the cached text while the failure is the same and the lookup younger
    /// than [`HOLDER_REFRESH`], else a new `lookup` (on the blocking pool).
    async fn text<F>(
        &mut self,
        port: u16,
        error: &std::io::Error,
        now: Instant,
        lookup: F,
    ) -> String
    where
        F: FnOnce(u16) -> crate::port_holder::Lookup + Send + 'static,
    {
        let failure = BindFailure::of(error);
        if let Some((seen, at, held)) = &self.last
            && *seen == failure
            && now.saturating_duration_since(*at) < HOLDER_REFRESH
        {
            return held.clone();
        }
        let held = tokio::task::spawn_blocking(move || lookup(port))
            .await
            .map(|h| format!("; {}", h.describe(port)))
            .unwrap_or_default();
        self.last = Some((failure, now, held.clone()));
        held
    }
}

/// How a listener's connections start.
#[derive(Clone)]
enum Accept {
    Plain,
    Tls(tokio_rustls::TlsAcceptor),
}

/// What an accept loop serves with.
#[derive(Clone)]
struct Loop {
    app: Router,
    shared: Shared,
    shutdown: CancellationToken,
    /// Shared by the listener's IPv4 and IPv6 loops.
    caps: Arc<ConnCaps>,
    limits: Limits,
    accept: Accept,
}

/// The per-address cap of the HTTP listener: at least `floor`, else the detection path's
/// current in-flight cap (see the module docs).
pub(crate) fn plain_per_peer(floor: usize, state: &super::AppState) -> usize {
    floor.max(super::detect::max_inflight(state))
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

/// Accept connections on `listener` until `l.shutdown`, then let the open ones finish their
/// requests (graceful shutdown, at most `l.limits.graceful`), close the ones still open and wait
/// for their tasks (at most `l.limits.hard_close`; see the module docs). A peer outside the
/// network rule (read from the current config at each accept, [`guard::admit`]) or over a
/// connection cap is closed at once: nothing is read from it or written to it.
async fn accept_loop(listener: tokio::net::TcpListener, l: Loop) {
    let builder = connection_builder();
    let graceful = hyper_util::server::graceful::GracefulShutdown::new();
    // Every connection task, and the token that drops the ones still open after the grace.
    let tracker = TaskTracker::new();
    let hard_stop = CancellationToken::new();
    let transport = match l.accept {
        Accept::Plain => Transport::Plain,
        Accept::Tls(_) => Transport::Tls,
    };
    loop {
        let (stream, peer) = tokio::select! {
            () = l.shutdown.cancelled() => break,
            accepted = listener.accept() => match accepted {
                Ok(c) => c,
                Err(e) if is_connection_error(&e) => continue,
                Err(e) => {
                    note_accept_error(&e, transport, &ACCEPT_ERRORS, crate::logthrottle::process_ms());
                    tokio::time::sleep(ACCEPT_ERROR_PAUSE).await;
                    continue;
                }
            },
        };
        if !guard::admit(peer, &l.shared) {
            drop(stream);
            continue;
        }
        let per_peer = match l.accept {
            Accept::Plain => plain_per_peer(l.limits.per_peer, &l.shared.current()),
            Accept::Tls(_) => l.limits.per_peer,
        };
        let slot = match l.caps.take(peer.ip(), l.limits.total, per_peer) {
            Ok(slot) => slot,
            Err(hit) => {
                note_cap(hit, peer, transport, l.limits.total, per_peer);
                drop(stream);
                continue;
            }
        };
        // Below TLS, so the handshake and every TLS write are covered too.
        let stream = super::deadline::WriteStall::new(stream, l.limits.write_stall);
        let router = l.app.clone();
        // The peer address feeds the network rule and the admin-route check (`admin_only`).
        let service =
            hyper::service::service_fn(move |mut req: hyper::Request<hyper::body::Incoming>| {
                let ext = req.extensions_mut();
                ext.insert(axum::extract::ConnectInfo(peer));
                ext.insert(transport);
                // `Router` is always ready (its `poll_ready` returns `Ready` at once).
                tower::Service::call(&mut router.clone(), req)
            });
        match &l.accept {
            Accept::Plain => {
                let conn = builder
                    .serve_connection(hyper_util::rt::TokioIo::new(stream), service)
                    .into_owned();
                let conn = graceful.watch(conn);
                let hard_stop = hard_stop.clone();
                tracker.spawn(async move {
                    let _slot = slot;
                    tokio::select! {
                        ended = conn => if let Err(e) = ended {
                            debug!(%peer, "HTTP connection ended: {e}");
                        },
                        () = hard_stop.cancelled() => {
                            debug!(%peer, "HTTP connection closed by the shutdown");
                        }
                    }
                });
            }
            Accept::Tls(acceptor) => {
                let acceptor = acceptor.clone();
                let builder = builder.clone();
                let watcher = graceful.watcher();
                let shutdown = l.shutdown.clone();
                let hard_stop = hard_stop.clone();
                let timeout = l.limits.handshake_timeout;
                tracker.spawn(async move {
                    let _slot = slot;
                    let handshake = tokio::time::timeout(timeout, acceptor.accept(stream));
                    let tls = tokio::select! {
                        () = shutdown.cancelled() => return,
                        done = handshake => match done {
                            Ok(Ok(tls)) => tls,
                            Ok(Err(e)) => {
                                debug!(%peer, "TLS handshake failed: {e}");
                                return;
                            }
                            Err(_) => {
                                debug!(%peer, "TLS handshake not finished in {} s; closed", timeout.as_secs_f32());
                                return;
                            }
                        },
                    };
                    let conn = builder
                        .serve_connection(hyper_util::rt::TokioIo::new(tls), service)
                        .into_owned();
                    tokio::select! {
                        ended = watcher.watch(conn) => if let Err(e) = ended {
                            debug!(%peer, "HTTPS connection ended: {e}");
                        },
                        () = hard_stop.cancelled() => {
                            debug!(%peer, "HTTPS connection closed by the shutdown");
                        }
                    }
                });
            }
        }
    }
    drop(listener);
    tracker.close();
    let label = match transport {
        Transport::Plain => "HTTP",
        Transport::Tls => "HTTPS",
    };
    let grace = l.limits.graceful;
    if tokio::time::timeout(grace, graceful.shutdown())
        .await
        .is_err()
    {
        warn!(
            open = tracker.len(),
            "{label} connections still busy {:.1} s after the shutdown request (a slow \
             client or a long request); closing them, their requests get no answer",
            grace.as_secs_f32()
        );
    }
    // Drops every connection still open (idle ones hyper already closed) and, with it, its
    // router clone.
    hard_stop.cancel();
    if tokio::time::timeout(l.limits.hard_close, tracker.wait())
        .await
        .is_err()
    {
        warn!(
            open = tracker.len(),
            "{label} connection tasks did not end {:.1} s after they were closed (one is \
             inside a call that does not yield); the server stops without them",
            l.limits.hard_close.as_secs_f32()
        );
    }
}

/// Log an accept that failed for a reason not about one connection, through `throttle` (the
/// process's [`ACCEPT_ERRORS`]) at `now_ms` on its clock; whether the line was written.
fn note_accept_error(
    e: &std::io::Error,
    transport: Transport,
    throttle: &Throttle,
    now_ms: u64,
) -> bool {
    let Some(dropped) = throttle.allow(now_ms) else {
        return false;
    };
    let listener = match transport {
        Transport::Plain => "HTTP",
        Transport::Tls => "HTTPS",
    };
    error!(
        dropped,
        "accepting a {listener} connection failed: {e}. The likely cause is that the service or \
         the machine ran out of file handles or sockets (too many open files or connections, or \
         a leak); the listener tries again every {} s. If this repeats, check the service's open \
         handles and restart it",
        ACCEPT_ERROR_PAUSE.as_secs()
    );
    true
}

/// Log a connection closed by a cap (throttled).
fn note_cap(hit: CapHit, peer: SocketAddr, transport: Transport, total: usize, per_peer: usize) {
    let Some(dropped) = CAP_WARNS.allow_now() else {
        return;
    };
    let listener = match transport {
        Transport::Plain => "HTTP",
        Transport::Tls => "HTTPS",
    };
    match hit {
        CapHit::Total => warn!(
            peer = %peer.ip().to_canonical(),
            dropped,
            "closed a new {listener} connection: {total} connections are open already, the most \
             this listener keeps; if these are real clients, they hold connections open for \
             longer than they need (a client that never closes its sockets)"
        ),
        CapHit::Peer => warn!(
            peer = %peer.ip().to_canonical(),
            dropped,
            "closed a new {listener} connection: this address has {per_peer} open already, the \
             most one address may keep; check that client for connections it never closes"
        ),
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

/// Which cap a connection hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CapHit {
    Total,
    Peer,
}

/// Open connections of one listener: the total and per address ([`super::auth::peer_key`]: an
/// IPv6 address by its /64, an IPv4-mapped one as IPv4).
#[derive(Debug, Default)]
pub(crate) struct ConnCaps {
    open: Mutex<Open>,
}

#[derive(Debug, Default)]
struct Open {
    total: usize,
    by_peer: HashMap<IpAddr, usize>,
}

impl ConnCaps {
    /// A slot for a connection from `ip`, or the cap it would pass.
    pub(crate) fn take(
        self: &Arc<Self>,
        ip: IpAddr,
        total: usize,
        per_peer: usize,
    ) -> Result<ConnSlot, CapHit> {
        let key = super::auth::peer_key(ip);
        let mut open = self.open.lock_recover();
        if open.total >= total {
            return Err(CapHit::Total);
        }
        let n = open.by_peer.entry(key).or_insert(0);
        if *n >= per_peer {
            if *n == 0 {
                open.by_peer.remove(&key);
            }
            return Err(CapHit::Peer);
        }
        *n += 1;
        open.total += 1;
        Ok(ConnSlot {
            caps: self.clone(),
            key,
        })
    }

    /// Open connections: (total, from the address `ip` counts under).
    #[cfg(test)]
    pub(crate) fn open(&self, ip: IpAddr) -> (usize, usize) {
        let open = self.open.lock_recover();
        let key = super::auth::peer_key(ip);
        (open.total, open.by_peer.get(&key).copied().unwrap_or(0))
    }
}

/// One open connection's place under the caps; given back when dropped.
#[derive(Debug)]
pub(crate) struct ConnSlot {
    caps: Arc<ConnCaps>,
    key: IpAddr,
}

impl Drop for ConnSlot {
    fn drop(&mut self) {
        let mut open = self.caps.open.lock_recover();
        open.total = open.total.saturating_sub(1);
        if let Some(n) = open.by_peer.get_mut(&self.key) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                open.by_peer.remove(&self.key);
            }
        }
    }
}

#[cfg(test)]
mod tests;
