//! Real-socket tests of the listeners: HTTPS serves the router with the `Tls` marker, a stalled
//! handshake holds no one else up, HTTP serves while the HTTPS port is taken and HTTPS comes up
//! once it is free, and the connection caps.

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
