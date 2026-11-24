//! Real-socket tests of the listeners: HTTPS serves the router with the `Tls` marker, a stalled
//! handshake holds no one else up, HTTP serves while the HTTPS port is taken and HTTPS comes up
//! once it is free, the connection caps, a peer that stops reading; and the port-holder cache and
//! the accept-error throttle.

use super::*;
use crate::server::test_util::*;
use rustls::pki_types::pem::PemObject;
use std::path::PathBuf;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch() -> Scratch {
    let d = std::env::temp_dir().join(format!("bo_listen_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&d).unwrap();
    Scratch(d)
}

fn plan(root: &Scratch, port: u16) -> HttpsPlan {
    HttpsPlan {
        address: [127, 0, 0, 1].into(),
        port,
        source: crate::tls::Source::Generated {
            data_root: root.0.clone(),
        },
    }
}

/// A running server: HTTP on 127.0.0.1 (any port) and HTTPS as `plan` says.
struct Running {
    http: u16,
    https: HttpsPort,
    token: CancellationToken,
    task: tokio::task::JoinHandle<anyhow::Result<()>>,
}

impl Running {
    async fn start(plan: HttpsPlan, limits: Limits) -> Self {
        let listeners = bind([127, 0, 0, 1].into(), 0)
            .await
            .unwrap()
            .with_https(plan)
            .with_limits(limits);
        let http = listeners.port();
        let shared = Shared::from(test_state());
        let https = shared.https_port();
        let token = CancellationToken::new();
        let task = tokio::spawn(serve_on(listeners, shared, token.clone()));
        Self {
            http,
            https,
            token,
            task,
        }
    }

    /// The HTTPS port once it is up (within `wait`).
    async fn https_up(&self, wait: Duration) -> u16 {
        let deadline = Instant::now() + wait;
        loop {
            if let Some(p) = self.https.get() {
                return p;
            }
            assert!(Instant::now() < deadline, "HTTPS did not come up");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    async fn stop(self) {
        self.token.cancel();
        tokio::time::timeout(Duration::from_secs(30), self.task)
            .await
            .expect("the server stops")
            .unwrap()
            .unwrap();
    }
}

/// The generated certificate of `root`, the client's only root.
fn trust(root: &Scratch) -> rustls::pki_types::CertificateDer<'static> {
    let pem = std::fs::read(root.0.join("auth").join("tls").join("cert.pem")).unwrap();
    rustls::pki_types::CertificateDer::from_pem_slice(&pem).unwrap()
}

/// Connect over TLS to 127.0.0.1:`port` as `name`, offering `http/1.1`.
async fn tls_connect(
    port: u16,
    root: &rustls::pki_types::CertificateDer<'static>,
    name: &str,
) -> std::io::Result<tokio_rustls::client::TlsStream<tokio::net::TcpStream>> {
    let tcp = tokio::net::TcpStream::connect(("127.0.0.1", port)).await?;
    let connector =
        tokio_rustls::TlsConnector::from(crate::tls::tests::client_config(root, &[b"http/1.1"]));
    let sn = rustls::pki_types::ServerName::try_from(name.to_string()).unwrap();
    connector.connect(sn, tcp).await
}

/// `GET path` over `stream` (closing it): the status line and the body.
async fn get<S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin>(
    mut stream: S,
    path: &str,
) -> (String, String) {
    let req = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut got = Vec::new();
    tokio::time::timeout(Duration::from_secs(10), stream.read_to_end(&mut got))
        .await
        .expect("an answer")
        .ok();
    let text = String::from_utf8_lossy(&got).into_owned();
    let status = text.lines().next().unwrap_or_default().to_string();
    let body = text
        .split("\r\n\r\n")
        .nth(1)
        .unwrap_or_default()
        .to_string();
    (status, body)
}

/// Whether the server closed `s` within `wait` (true) or it is still open (false).
async fn closed_within(s: &mut tokio::net::TcpStream, wait: Duration) -> bool {
    let mut buf = [0u8; 16];
    match tokio::time::timeout(wait, s.read(&mut buf)).await {
        Ok(Ok(0) | Err(_)) => true,
        Ok(Ok(_)) => false,
        Err(_) => false,
    }
}

#[tokio::test]
async fn https_serves_the_router_with_the_tls_marker() {
    let root = scratch();
    let running = Running::start(plan(&root, 0), Limits::default()).await;
    let port = running.https_up(Duration::from_secs(30)).await;
    assert_ne!(port, running.http);
    let cert = trust(&root);
    for name in ["localhost", "127.0.0.1"] {
        let s = tls_connect(port, &cert, name).await.unwrap();
        assert_eq!(s.get_ref().1.alpn_protocol(), Some(&b"http/1.1"[..]));
        let (status, body) = get(s, "/__test/transport").await;
        assert!(status.starts_with("HTTP/1.1 200"), "{status}");
        assert_eq!(body, "Some(Tls)");
    }
    let (status, _) = get(
        tls_connect(port, &cert, "localhost").await.unwrap(),
        "/v1/status",
    )
    .await;
    assert!(
        status.starts_with("HTTP/1.1 200") || status.starts_with("HTTP/1.1 503"),
        "{status}"
    );
    // The HTTP port serves the same router, marked plain.
    let tcp = tokio::net::TcpStream::connect(("127.0.0.1", running.http))
        .await
        .unwrap();
    let (_, body) = get(tcp, "/__test/transport").await;
    assert_eq!(body, "Some(Plain)");
    running.stop().await;
}

/// A client that connects and never sends its ClientHello is closed after the handshake
/// timeout, and another client's handshake meanwhile goes through at once.
#[tokio::test]
async fn a_stalled_handshake_is_closed_and_blocks_no_one() {
    let root = scratch();
    let limits = Limits {
        handshake_timeout: Duration::from_secs(1),
        ..Limits::default()
    };
    let running = Running::start(plan(&root, 0), limits).await;
    let port = running.https_up(Duration::from_secs(30)).await;
    let cert = trust(&root);
    let mut stalled = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    let started = Instant::now();
    let s = tokio::time::timeout(
        Duration::from_millis(900),
        tls_connect(port, &cert, "localhost"),
    )
    .await
    .expect("a second handshake is not held up")
    .unwrap();
    let (status, _) = get(s, "/v1/status").await;
    assert!(status.starts_with("HTTP/1.1"), "{status}");
    assert!(closed_within(&mut stalled, Duration::from_secs(5)).await);
    let waited = started.elapsed();
    assert!(
        waited >= Duration::from_millis(900) && waited < Duration::from_secs(5),
        "closed after {waited:?}"
    );
    running.stop().await;
}

/// HTTPS's port is taken: HTTP serves anyway, and HTTPS comes up once the port is free.
#[tokio::test]
async fn http_serves_while_the_https_port_is_taken_and_https_follows() {
    let root = scratch();
    let blocker = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let taken = blocker.local_addr().unwrap().port();
    let limits = Limits {
        https_retry: Duration::from_millis(200),
        ..Limits::default()
    };
    let running = Running::start(plan(&root, taken), limits).await;
    let tcp = tokio::net::TcpStream::connect(("127.0.0.1", running.http))
        .await
        .unwrap();
    let (status, _) = get(tcp, "/v1/status").await;
    assert!(
        status.starts_with("HTTP/1.1 200") || status.starts_with("HTTP/1.1 503"),
        "{status}"
    );
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert_eq!(running.https.get(), None, "down while the port is held");
    drop(blocker);
    let port = running.https_up(Duration::from_secs(10)).await;
    assert_eq!(port, taken);
    let s = tls_connect(port, &trust(&root), "localhost").await.unwrap();
    let (_, body) = get(s, "/__test/transport").await;
    assert_eq!(body, "Some(Tls)");
    running.stop().await;
}

/// Own certificate files that do not load: HTTPS stays down, HTTP serves.
#[tokio::test]
async fn https_with_broken_own_files_stays_down_and_http_serves() {
    let root = scratch();
    std::fs::write(root.0.join("c.pem"), "nope").unwrap();
    std::fs::write(root.0.join("k.pem"), "nope").unwrap();
    let plan = HttpsPlan {
        source: crate::tls::Source::Files {
            cert: root.0.join("c.pem"),
            key: root.0.join("k.pem"),
        },
        ..plan(&root, 0)
    };
    let running = Running::start(
        plan,
        Limits {
            https_retry: Duration::from_millis(100),
            ..Limits::default()
        },
    )
    .await;
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(running.https.get(), None);
    let tcp = tokio::net::TcpStream::connect(("127.0.0.1", running.http))
        .await
        .unwrap();
    let (_, body) = get(tcp, "/__test/transport").await;
    assert_eq!(body, "Some(Plain)");
    assert_eq!(
        std::fs::read_to_string(root.0.join("c.pem")).unwrap(),
        "nope"
    );
    running.stop().await;
}

/// 32 connections from one address are open; the 33rd is closed at once; when one goes, a new
/// one gets its place.
#[tokio::test]
async fn the_33rd_connection_from_one_address_is_closed() {
    let root = scratch();
    let limits = Limits {
        handshake_timeout: Duration::from_secs(20),
        ..Limits::default()
    };
    let running = Running::start(plan(&root, 0), limits).await;
    let port = running.https_up(Duration::from_secs(30)).await;
    let mut open = Vec::new();
    for _ in 0..MAX_CONNECTIONS_PER_PEER {
        open.push(
            tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .unwrap(),
        );
    }
    // Accepting is asynchronous: wait until all of them hold a slot.
    let mut extra = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    assert!(
        closed_within(&mut extra, Duration::from_secs(5)).await,
        "the 33rd is closed"
    );
    assert!(!closed_within(&mut open[0], Duration::from_millis(300)).await);
    assert!(!closed_within(&mut open[31], Duration::from_millis(300)).await);
    drop(open.pop());
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let mut next = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .unwrap();
        if !closed_within(&mut next, Duration::from_millis(300)).await {
            break;
        }
        assert!(Instant::now() < deadline, "the freed slot never came back");
    }
    drop(open);
    running.stop().await;
}

#[test]
fn caps_count_per_address_and_in_total_and_give_slots_back() {
    let caps = Arc::new(ConnCaps::default());
    let a: IpAddr = "192.168.1.5".parse().unwrap();
    let b: IpAddr = "192.168.1.6".parse().unwrap();
    let held: Vec<_> = (0..3).map(|_| caps.take(a, 5, 3).unwrap()).collect();
    assert_eq!(caps.take(a, 5, 3).unwrap_err(), CapHit::Peer);
    // The same address through a dual-stack socket counts as the same peer.
    assert_eq!(
        caps.take("::ffff:192.168.1.5".parse().unwrap(), 5, 3)
            .unwrap_err(),
        CapHit::Peer
    );
    let b1 = caps.take(b, 5, 3).unwrap();
    let b2 = caps.take(b, 5, 3).unwrap();
    assert_eq!(caps.take(b, 5, 3).unwrap_err(), CapHit::Total);
    assert_eq!(caps.open(a), (5, 3));
    drop(held);
    assert_eq!(caps.open(a), (2, 0));
    drop((b1, b2));
    assert_eq!(caps.open(b), (0, 0));
    // An IPv6 host is one peer across its /64 (privacy addresses).
    let v6 = |s: &str| s.parse::<IpAddr>().unwrap();
    let _x = caps.take(v6("fd00:1:2:3::1"), 9, 2).unwrap();
    let _y = caps.take(v6("fd00:1:2:3:ffff::9"), 9, 2).unwrap();
    assert_eq!(
        caps.take(v6("fd00:1:2:3::77"), 9, 2).unwrap_err(),
        CapHit::Peer
    );
    assert!(caps.take(v6("fd00:1:2:4::1"), 9, 2).is_ok());
}

/// The HTTP port's per-address cap is never below the detection path's in-flight cap, so it
/// never closes a connection the detection path would serve (Blue Iris is one address).
#[test]
fn the_http_cap_follows_the_inflight_cap() {
    let state = test_state();
    let inflight = crate::server::detect::max_inflight(&state);
    assert_eq!(plain_per_peer(32, &state), inflight.max(32));
    state.metrics.inflight_limit.store(500, Ordering::Relaxed);
    assert_eq!(plain_per_peer(32, &state), 500);
    state.metrics.inflight_limit.store(4, Ordering::Relaxed);
    assert_eq!(plain_per_peer(32, &state), 32);
    const { assert!(crate::server::detect::BODY_READS_PER_PEER <= MAX_CONNECTIONS_PER_PEER) };
}

/// The pinned client (`tls::pinned_client`, what `diagnostics` and the load-test client use)
/// completes a real handshake with the service's own certificate, and refuses another
/// self-signed certificate.
#[tokio::test]
async fn the_pinned_client_trusts_the_pinned_certificate_only() {
    let root = scratch();
    let running = Running::start(plan(&root, 0), Limits::default()).await;
    let port = running.https_up(Duration::from_secs(30)).await;
    let cert = root.0.join("auth").join("tls").join("cert.pem");
    let client = crate::tls::pinned_client(&cert).unwrap().build().unwrap();
    for host in ["127.0.0.1", "localhost"] {
        let r = client
            .get(format!("https://{host}:{port}/v1/status"))
            .send()
            .await
            .unwrap_or_else(|e| panic!("{host}: {e:?}"));
        assert!(
            r.status() == reqwest::StatusCode::OK
                || r.status() == reqwest::StatusCode::SERVICE_UNAVAILABLE,
            "{host}: {}",
            r.status()
        );
    }
    // An admin route over HTTPS without a credential: refused, as JSON.
    let r = client
        .get(format!("https://127.0.0.1:{port}/prometheus"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::UNAUTHORIZED);
    // Pinned to another self-signed certificate: the handshake fails.
    let other = scratch();
    let made = crate::tls::generate(
        &crate::tls::Identity::default(),
        std::time::SystemTime::now(),
    )
    .unwrap();
    let other_cert = other.0.join("cert.pem");
    std::fs::write(&other_cert, &made.cert_pem).unwrap();
    let wrong = crate::tls::pinned_client(&other_cert)
        .unwrap()
        .build()
        .unwrap();
    let e = wrong
        .get(format!("https://127.0.0.1:{port}/v1/status"))
        .send()
        .await
        .unwrap_err();
    assert!(e.is_connect() || e.is_request(), "{e:?}");
    // A file with no certificate is an error before any connection.
    let empty = other.0.join("empty.pem");
    std::fs::write(&empty, "").unwrap();
    assert!(crate::tls::pinned_client(&empty).is_err());
    running.stop().await;
}

/// A browser's way in, over real sockets: the plain port sends it to the same page on HTTPS, the
/// HTTPS port wants a sign-in, `POST /login` sets the session cookie, and that cookie opens the
/// page; the API token (Bearer) is refused on the plain port.
#[tokio::test]
async fn a_browser_signs_in_by_hand_over_real_sockets() {
    let root = scratch();
    let running = Running::start(plan(&root, 0), Limits::default()).await;
    let port = running.https_up(Duration::from_secs(30)).await;
    let cert = root.0.join("auth").join("tls").join("cert.pem");
    let https = crate::tls::pinned_client(&cert)
        .unwrap()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let plain = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .unwrap();
    let r = plain
        .get(format!("http://127.0.0.1:{}/config?x=1", running.http))
        .header("accept", "text/html")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::PERMANENT_REDIRECT);
    assert_eq!(
        r.headers()["location"],
        format!("https://127.0.0.1:{port}/config?x=1").as_str()
    );
    let r = plain
        .get(format!("http://127.0.0.1:{}/prometheus", running.http))
        .bearer_auth("ab".repeat(32))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::FORBIDDEN);
    let base = format!("https://127.0.0.1:{port}");
    let r = https
        .get(format!("{base}/config"))
        .header("accept", "text/html")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::SEE_OTHER);
    assert_eq!(r.headers()["location"], "/login?next=%2Fconfig");
    let r = https
        .post(format!("{base}/login"))
        .header("accept", "text/html")
        .header("origin", base.as_str())
        .header("content-type", "application/x-www-form-urlencoded")
        .body(format!(
            "username={TEST_USER}&password={}&next=%2Fconfig",
            urlencode(TEST_PASSWORD)
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::SEE_OTHER);
    assert_eq!(r.headers()["location"], "/config");
    let session = r
        .headers()
        .get_all("set-cookie")
        .iter()
        .filter_map(|v| v.to_str().ok())
        .find_map(|c| c.strip_prefix("__Host-bop_session="))
        .and_then(|c| c.split(';').next())
        .unwrap()
        .to_string();
    let r = https
        .get(format!("{base}/config"))
        .header("accept", "text/html")
        .header("cookie", format!("__Host-bop_session={session}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), reqwest::StatusCode::OK);
    assert!(r.text().await.unwrap().contains(TEST_USER));
    running.stop().await;
}

/// One plain accept loop on 127.0.0.1 serving `app` (not the real router), with `limits`:
/// (port, stop token, the loop's task).
async fn plain_loop(
    app: Router,
    limits: Limits,
) -> (u16, CancellationToken, tokio::task::JoinHandle<()>) {
    let (port, shutdown, task, _) = plain_loop_with_caps(app, limits).await;
    (port, shutdown, task)
}

/// [`plain_loop`] and the loop's connection caps.
async fn plain_loop_with_caps(
    app: Router,
    limits: Limits,
) -> (
    u16,
    CancellationToken,
    tokio::task::JoinHandle<()>,
    Arc<ConnCaps>,
) {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let port = listener.local_addr().unwrap().port();
    let shutdown = CancellationToken::new();
    let caps = Arc::new(ConnCaps::default());
    let l = Loop {
        app,
        shared: Shared::from(test_state()),
        shutdown: shutdown.clone(),
        caps: caps.clone(),
        limits,
        accept: Accept::Plain,
    };
    (port, shutdown, tokio::spawn(accept_loop(listener, l)), caps)
}

/// RA-023: a request that never finishes holds its connection only for the grace. After it the
/// connection is closed, the loop returns, and no connection task keeps a copy of the router
/// (which holds the server's state) alive.
#[tokio::test]
async fn a_stuck_request_is_closed_after_the_grace_and_drops_the_router() {
    let probe = Arc::new(());
    let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let app = {
        let (probe, entered) = (probe.clone(), entered.clone());
        Router::new().route(
            "/stuck",
            axum::routing::get(move || {
                let (probe, entered) = (probe.clone(), entered.clone());
                async move {
                    let _held = probe;
                    entered.store(true, Ordering::SeqCst);
                    std::future::pending::<&'static str>().await
                }
            }),
        )
    };
    let limits = Limits {
        graceful: Duration::from_millis(300),
        hard_close: Duration::from_secs(1),
        ..Limits::default()
    };
    let (port, stop, task) = plain_loop(app, limits).await;
    let mut client = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    client
        .write_all(b"GET /stuck HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !entered.load(Ordering::SeqCst) {
        assert!(Instant::now() < deadline, "the handler never ran");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let started = Instant::now();
    stop.cancel();
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("the accept loop returns")
        .unwrap();
    let took = started.elapsed();
    assert!(
        took >= Duration::from_millis(300) && took < Duration::from_secs(3),
        "returned after {took:?}"
    );
    assert!(
        closed_within(&mut client, Duration::from_secs(1)).await,
        "the stuck connection is closed"
    );
    assert_eq!(
        Arc::strong_count(&probe),
        1,
        "a connection task still holds the router"
    );
}

/// An idle keep-alive connection does not hold the stop: it is closed at once, well within the
/// grace.
#[tokio::test]
async fn an_idle_keep_alive_connection_is_closed_at_once() {
    let app = Router::new().route("/ok", axum::routing::get(|| async { "ok" }));
    let limits = Limits {
        graceful: Duration::from_secs(5),
        ..Limits::default()
    };
    let (port, stop, task) = plain_loop(app, limits).await;
    let mut client = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .unwrap();
    client
        .write_all(b"GET /ok HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();
    // The whole answer (keep-alive: the server leaves the connection open after it).
    let mut got = Vec::new();
    let mut buf = [0u8; 512];
    while !String::from_utf8_lossy(&got).ends_with("ok") {
        let n = tokio::time::timeout(Duration::from_secs(5), client.read(&mut buf))
            .await
            .expect("an answer")
            .unwrap();
        assert!(n > 0, "closed before the answer");
        got.extend_from_slice(buf.get(..n).unwrap());
    }
    let started = Instant::now();
    stop.cancel();
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("the accept loop returns")
        .unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "an idle connection held the stop for {:?}",
        started.elapsed()
    );
    assert!(closed_within(&mut client, Duration::from_secs(1)).await);
}

/// A route whose answer never ends (64 KiB chunks for ever), so only the client's reading paces
/// the server's writes.
fn endless_app() -> Router {
    Router::new().route(
        "/endless",
        axum::routing::get(|| async {
            let chunk = axum::body::Bytes::from(vec![b'x'; 64 * 1024]);
            axum::body::Body::from_stream(futures_util::stream::repeat_with(move || {
                Ok::<_, std::io::Error>(chunk.clone())
            }))
        }),
    )
}

/// A client that has asked for `/endless`, with a receive buffer of `recv_buffer` bytes when
/// given (a small one fills at once; below the loopback MSS it also delays the window updates
/// a reader's reads send, so a reading client must keep the default).
async fn endless_client(port: u16, recv_buffer: Option<u32>) -> tokio::net::TcpStream {
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    if let Some(size) = recv_buffer {
        socket.set_recv_buffer_size(size).unwrap();
    }
    let mut s = socket
        .connect(SocketAddr::from(([127, 0, 0, 1], port)))
        .await
        .unwrap();
    s.write_all(b"GET /endless HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();
    s
}

const LOCAL: IpAddr = IpAddr::V4(std::net::Ipv4Addr::LOCALHOST);

/// RA-032: a client that asks for an answer and never reads it holds its connection only for
/// the write-stall timeout: then the server closes it and its slot under the caps goes back.
#[tokio::test]
async fn a_client_that_never_reads_is_closed_and_frees_its_slot() {
    let limits = Limits {
        write_stall: Duration::from_millis(400),
        ..Limits::default()
    };
    let (port, stop, task, caps) = plain_loop_with_caps(endless_app(), limits).await;
    let mut client = endless_client(port, Some(4096)).await;
    let deadline = Instant::now() + Duration::from_secs(5);
    while caps.open(LOCAL).0 == 0 {
        assert!(
            Instant::now() < deadline,
            "the connection was never accepted"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    // Once the buffers are full the writes wait on the client; within the timeout (plus the
    // time to fill them) the connection is gone.
    let deadline = Instant::now() + Duration::from_secs(10);
    while caps.open(LOCAL).0 != 0 {
        assert!(
            Instant::now() < deadline,
            "a client that never reads still holds its connection"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // What was buffered can still be read, then the stream ends (or is reset).
    let mut buf = vec![0u8; 64 * 1024];
    let ended = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match client.read(&mut buf).await {
                Ok(0) | Err(_) => return,
                Ok(_) => {}
            }
        }
    })
    .await;
    assert!(ended.is_ok(), "the stream did not end");
    stop.cancel();
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap();
}

/// The control of the test above, what the server did before the write-stall deadline (an
/// hour here stands for none): the client that never reads keeps its connection and its slot,
/// so the test above fails without the deadline.
#[tokio::test]
async fn without_the_deadline_a_client_that_never_reads_holds_its_slot() {
    let limits = Limits {
        write_stall: Duration::from_secs(3600),
        graceful: Duration::from_millis(200),
        ..Limits::default()
    };
    let (port, stop, task, caps) = plain_loop_with_caps(endless_app(), limits).await;
    let _client = endless_client(port, Some(4096)).await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    assert_eq!(caps.open(LOCAL).0, 1, "the connection is still held");
    stop.cancel();
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(caps.open(LOCAL).0, 0, "the shutdown closes it");
}

/// A client that reads slower than the server writes but keeps reading is never cut, though the
/// transfer lasts four write-stall timeouts and the server's writes wait on it all along. (The
/// kernel reports the socket writable again only after a large part of its send buffer drained,
/// some 340 KB on macOS loopback, so the gaps the server sees are that over the read rate: about
/// 0.1 s here.)
#[tokio::test]
async fn a_slow_reader_keeps_its_connection() {
    let stall = Duration::from_secs(1);
    let limits = Limits {
        write_stall: stall,
        ..Limits::default()
    };
    let (port, stop, task, caps) = plain_loop_with_caps(endless_app(), limits).await;
    let mut client = endless_client(port, None).await;
    let mut buf = vec![0u8; 64 * 1024];
    let started = Instant::now();
    while started.elapsed() < stall * 4 {
        tokio::time::sleep(Duration::from_millis(20)).await;
        let n = tokio::time::timeout(Duration::from_secs(5), client.read(&mut buf))
            .await
            .expect("data keeps coming")
            .unwrap();
        assert!(n > 0, "the server closed a client that kept reading");
    }
    assert_eq!(caps.open(LOCAL).0, 1);
    drop(client);
    stop.cancel();
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap();
}

/// RA-033: the port holder is looked up again only when the bind error changes or the last
/// lookup is [`HOLDER_REFRESH`] old; otherwise the 30 s retry reuses the text.
#[tokio::test]
async fn the_port_holder_is_looked_up_only_when_needed() {
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let lookup = |calls: &Arc<std::sync::atomic::AtomicUsize>| {
        let calls = calls.clone();
        move |_port: u16| {
            let n = calls.fetch_add(1, Ordering::SeqCst) + 1;
            crate::port_holder::Lookup::Held(crate::port_holder::Holder {
                pid: u32::try_from(n).unwrap(),
                name: Some("other.exe".into()),
            })
        }
    };
    let in_use = std::io::Error::from(std::io::ErrorKind::AddrInUse);
    let denied = std::io::Error::from(std::io::ErrorKind::PermissionDenied);
    let mut cache = HolderCache::default();
    let t0 = Instant::now();
    let first = cache.text(9, &in_use, t0, lookup(&calls)).await;
    assert_eq!(first, "; port 9 is held by other.exe (PID 1)");
    // The same failure on the next attempts (every 30 s) for almost ten minutes: no lookup.
    for step in 1..20u32 {
        let now = t0 + HTTPS_RETRY * step;
        assert_eq!(cache.text(9, &in_use, now, lookup(&calls)).await, first);
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    // Another kind of failure: looked up at once.
    let other = cache
        .text(9, &denied, t0 + HTTPS_RETRY * 20, lookup(&calls))
        .await;
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(other.contains("PID 2"), "{other}");
    // The same one past the refresh age: looked up again.
    let later = t0 + HTTPS_RETRY * 20 + HOLDER_REFRESH;
    cache.text(9, &denied, later, lookup(&calls)).await;
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    // Back to "in use": a change again.
    cache
        .text(9, &in_use, later + Duration::from_secs(1), lookup(&calls))
        .await;
    assert_eq!(calls.load(Ordering::SeqCst), 4);
}

/// RA-119: a lasting accept failure writes one ERROR line per minute, not one per second.
#[test]
fn a_lasting_accept_failure_logs_once_a_minute() {
    let throttle = Throttle::new(ACCEPT_ERROR_LOG_EVERY);
    let e = std::io::Error::other("Too many open files (os error 24)");
    let pause = u64::try_from(ACCEPT_ERROR_PAUSE.as_millis()).unwrap();
    let every = u64::try_from(ACCEPT_ERROR_LOG_EVERY.as_millis()).unwrap();
    // One failure a second for a minute on one listener.
    let logged = (0..60u64)
        .filter(|s| note_accept_error(&e, Transport::Plain, &throttle, 5_000 + s * pause))
        .count();
    assert_eq!(logged, 1);
    assert_eq!(throttle.held_back(), 59);
    assert!(note_accept_error(
        &e,
        Transport::Tls,
        &throttle,
        5_000 + every
    ));
    assert_eq!(throttle.held_back(), 0);
}

/// QA-089: `bind_address` `::` serves IPv4 clients too. Windows binds `::` IPv6-only, so the
/// companion `0.0.0.0` socket carries them there; a dual-stack system serves them on `::` itself
/// (the companion then fails with `AddrInUse`, and this still passes).
#[tokio::test]
async fn an_ipv6_wildcard_bind_also_serves_ipv4() {
    let any6: IpAddr = std::net::Ipv6Addr::UNSPECIFIED.into();
    let (main, companion, port) = match bind_pair(any6, 0).await {
        Ok(bound) => bound,
        Err(e) => {
            eprintln!("skipped: this host cannot bind [::] ({e})");
            return;
        }
    };
    let accept = async {
        match &companion {
            Some(c) => tokio::select! {
                a = main.accept() => a,
                a = c.accept() => a,
            },
            None => main.accept().await,
        }
    };
    let connect = tokio::net::TcpStream::connect(SocketAddr::from(([127, 0, 0, 1], port)));
    let (accepted, connected) = tokio::time::timeout(Duration::from_secs(30), async {
        tokio::join!(accept, connect)
    })
    .await
    .expect("the IPv4 connection was accepted");
    let connected = connected.expect("127.0.0.1 reaches a `::` bind");
    let (_, peer) = accepted.expect("accepted");
    assert_eq!(peer.port(), connected.local_addr().unwrap().port());
}
