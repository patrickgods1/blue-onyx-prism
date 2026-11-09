//! The network rule in the router and the accept loop: who is refused, what the answer says,
//! that nothing of a refused request is read, and that `allowed_networks` applies at once.

use super::super::test_util::*;
use super::super::{Shared, bind, guard, router, serve_on};
use crate::server::AppState;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode, header};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

fn get(uri: &str, peer: &str) -> Request<Body> {
    from_peer(Request::get(uri).body(Body::empty()).unwrap(), peer)
}

fn json(b: &[u8]) -> serde_json::Value {
    serde_json::from_slice(b).unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(b)))
}

/// `(status, headers, message)` of `req` (the message of a JSON refusal, else empty).
async fn answer(state: &Arc<AppState>, req: Request<Body>) -> (StatusCode, HeaderMap, String) {
    let (s, h, b) = call(state, req).await;
    let msg = serde_json::from_slice::<serde_json::Value>(&b)
        .ok()
        .and_then(|v| v["message"].as_str().map(str::to_string))
        .unwrap_or_default();
    (s, h, msg)
}

/// A peer on the internet gets 403 on every kind of route, open (detection, status), admin and
/// login alike, and on unknown paths: always short JSON, also for a browser, naming its address
/// and `allowed_networks`, with the security headers.
#[tokio::test]
async fn a_public_peer_is_refused_on_every_route() {
    let state = test_state();
    for (method, uri) in [
        ("GET", "/v1/status"),
        ("GET", "/v1/vision/custom/list"),
        ("POST", "/v1/vision/detection"),
        ("POST", "/v1/vision/custom/ipcam-general"),
        ("GET", "/v1/config"),
        ("GET", "/"),
        ("GET", "/static/style.css"),
        ("GET", "/config"),
        ("POST", "/config/restart"),
        ("GET", "/login"),
        ("GET", "/no/such/page"),
    ] {
        for html in [false, true] {
            let mut b = Request::builder().method(method).uri(uri);
            if html {
                b = b.header(header::ACCEPT, "text/html");
            }
            let req = from_peer(b.body(Body::empty()).unwrap(), PUBLIC);
            let (s, h, b) = call(&state, req).await;
            assert_eq!(s, StatusCode::FORBIDDEN, "{method} {uri}");
            assert!(
                h[header::CONTENT_TYPE]
                    .to_str()
                    .unwrap()
                    .starts_with("application/json"),
                "{method} {uri}"
            );
            assert_eq!(h["x-content-type-options"], "nosniff");
            assert_eq!(h[header::CACHE_CONTROL], "no-store");
            let j = json(&b);
            assert_eq!(j["success"], false);
            let msg = j["message"].as_str().unwrap();
            assert_eq!(j["error"].as_str(), Some(msg));
            assert!(msg.contains("203.0.113.5"), "{msg}");
            assert!(msg.contains("allowed_networks"), "{msg}");
            assert!(msg.contains("local network"), "{msg}");
        }
    }
    assert!(!state.restart.is_cancelled());
    // Mapped into IPv6, the same address is refused the same way.
    let (s, _, msg) = answer(&state, get("/v1/status", "[::ffff:203.0.113.5]:1")).await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert!(msg.contains("(203.0.113.5)"), "{msg}");
    for peer in ["[2001:db8::1]:1", "[64:ff9b::808:808]:1", "100.64.0.1:1"] {
        let (s, _, _) = answer(&state, get("/v1/vision/custom/list", peer)).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{peer}");
    }
}

/// Peers of the local network reach the routes as before; the admin routes then want a sign-in,
/// from every address alike (not the network rule's refusal).
#[tokio::test]
async fn local_network_peers_pass() {
    let state = test_state();
    for peer in [
        LOCAL,
        REMOTE,
        "172.16.0.9:1",
        "192.168.1.20:1",
        "169.254.3.4:1",
        "[::1]:1",
        "[fd00::5]:1",
        "[fe80::1]:1",
        "[::ffff:192.168.1.2]:1",
    ] {
        let (s, _, _) = answer(&state, get("/v1/vision/custom/list", peer)).await;
        assert_eq!(s, StatusCode::OK, "{peer}");
    }
    // The status route answers 503 for these failed models, not 403.
    let (s, _, _) = answer(&state, get("/v1/status", REMOTE)).await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
    for peer in [REMOTE, LOCAL] {
        let (s, _, msg) = answer(&state, anonymous(get("/config", peer))).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "{peer}");
        assert!(msg.contains("/login"), "{msg}");
        assert!(!msg.contains("allowed_networks"), "{msg}");
        assert_eq!(answer(&state, get("/config", peer)).await.0, StatusCode::OK);
    }
}

/// Forwarding headers can only refuse: a public address in one refuses a loopback request on an
/// open route, a private one passes the rule, and the admin routes then want a sign-in as from
/// any other address.
#[tokio::test]
async fn forwarding_headers_only_refuse() {
    let state = test_state();
    let with = |uri: &str, name: &str, value: &str| {
        let mut r = get(uri, LOCAL);
        r.headers_mut().append(
            header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
            value.parse().unwrap(),
        );
        r
    };
    for (name, value) in [
        ("x-forwarded-for", "8.8.8.8"),
        ("x-forwarded-for", "192.168.1.9, 8.8.8.8"),
        ("x-real-ip", "8.8.8.8"),
        ("forwarded", "for=\"[2001:db8::1]:80\""),
        ("forwarded", "for=unknown"),
        ("forwarded", "for=_hidden"),
        ("x-forwarded-for", "not-an-address"),
    ] {
        for uri in ["/v1/vision/custom/list", "/config"] {
            let (s, _, msg) = answer(&state, with(uri, name, value)).await;
            assert_eq!(s, StatusCode::FORBIDDEN, "{name}: {value} {uri}");
            assert!(msg.contains("allowed_networks"), "{msg}");
            assert!(msg.contains(&format!("{name} header")), "{msg}");
            assert!(msg.contains("127.0.0.1"), "{msg}");
            // The header's value is never repeated.
            for part in ["8.8.8.8", "2001:db8", "_hidden", "not-an-address"] {
                assert!(!msg.contains(part), "{msg}");
            }
        }
    }
    let (s, _, _) = answer(
        &state,
        with("/v1/vision/custom/list", "x-forwarded-for", "192.168.1.9"),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let (s, _, msg) = answer(
        &state,
        anonymous(with("/config", "x-forwarded-for", "192.168.1.9")),
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert!(msg.contains("/login"), "{msg}");
    assert!(!msg.contains("allowed_networks"), "{msg}");
    let (s, _, _) = answer(&state, with("/config", "x-forwarded-for", "192.168.1.9")).await;
    assert_eq!(s, StatusCode::OK);
    // A private header never lets a public peer in.
    let mut r = get("/v1/vision/custom/list", PUBLIC);
    r.headers_mut()
        .insert("x-forwarded-for", "192.168.1.9".parse().unwrap());
    assert_eq!(answer(&state, r).await.0, StatusCode::FORBIDDEN);
}

/// A request that did not come through the listener has no peer and is refused.
#[tokio::test]
async fn a_request_without_a_peer_is_refused() {
    let state = test_state();
    let r = Request::get("/v1/vision/custom/list")
        .header(header::HOST, TEST_HOST)
        .body(Body::empty())
        .unwrap();
    let resp = router(state).oneshot(r).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    let b = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let msg = json(&b)["message"].as_str().unwrap().to_string();
    assert!(msg.contains("no client address"), "{msg}");
}

/// The refusal comes before the body: an upload from outside is never read.
#[tokio::test]
async fn a_refused_upload_is_not_read() {
    let state = test_state();
    let polled = Arc::new(AtomicBool::new(false));
    let flag = polled.clone();
    let body = Body::from_stream(futures_util::stream::once(async move {
        flag.store(true, Ordering::SeqCst);
        Ok::<_, std::io::Error>(bytes::Bytes::from_static(b"--x--\r\n"))
    }));
    let req = Request::post("/v1/vision/detection")
        .header(header::CONTENT_TYPE, "multipart/form-data; boundary=x")
        .body(body)
        .unwrap();
    let (s, _, _) = call(&state, from_peer(req, PUBLIC)).await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert!(!polled.load(Ordering::SeqCst));
}

/// `allowed_networks` is read per request and per connection: a change through the config store
/// applies to the next one, in both directions.
#[tokio::test]
async fn allowed_networks_apply_at_once() {
    let state = test_state();
    let shared = Shared::from(state.clone());
    let public: std::net::SocketAddr = PUBLIC.parse().unwrap();
    assert_eq!(
        answer(&state, get("/v1/vision/custom/list", PUBLIC))
            .await
            .0,
        StatusCode::FORBIDDEN
    );
    assert!(!guard::admit(public, &shared));
    state
        .config
        .update("test", |c| {
            c.allowed_networks = vec!["203.0.113.0/24".into()];
            Ok(())
        })
        .unwrap();
    assert_eq!(
        answer(&state, get("/v1/vision/custom/list", PUBLIC))
            .await
            .0,
        StatusCode::OK
    );
    assert!(guard::admit(public, &shared));
    // A listed network passes in a forwarding header too.
    let mut r = get("/v1/vision/custom/list", LOCAL);
    r.headers_mut()
        .insert("x-forwarded-for", "203.0.113.77".parse().unwrap());
    assert_eq!(answer(&state, r).await.0, StatusCode::OK);
    // The admin routes want a sign-in there like everywhere.
    assert_eq!(
        answer(&state, as_peer(get("/config", LOCAL), PUBLIC))
            .await
            .0,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        answer(&state, get("/config", PUBLIC)).await.0,
        StatusCode::OK
    );
    state
        .config
        .update("test", |c| {
            c.allowed_networks.clear();
            Ok(())
        })
        .unwrap();
    assert_eq!(
        answer(&state, get("/v1/vision/custom/list", PUBLIC))
            .await
            .0,
        StatusCode::FORBIDDEN
    );
    assert!(!guard::admit(public, &shared));
}

/// The accept-time decision: loopback and the local network are served, others are not, and a
/// swapped-in generation's config is the one read.
#[test]
fn the_accept_loop_admits_the_local_network_only() {
    let shared = Shared::from(test_state());
    for peer in [
        LOCAL,
        REMOTE,
        "[::1]:1",
        "[fe80::1]:1",
        "[::ffff:127.0.0.1]:1",
    ] {
        assert!(guard::admit(peer.parse().unwrap(), &shared), "{peer}");
    }
    for peer in [PUBLIC, "8.8.8.8:1", "[2001:db8::1]:1", "100.100.1.2:1"] {
        assert!(!guard::admit(peer.parse().unwrap(), &shared), "{peer}");
    }
    shared.swap(test_state_with(|c| {
        c.allowed_networks = vec!["100.64.0.0/10".into()];
    }));
    assert!(guard::admit("100.100.1.2:1".parse().unwrap(), &shared));
    assert!(!guard::admit(PUBLIC.parse().unwrap(), &shared));
}

/// Through a real socket: a loopback client is still served (IPv4 and the `[::1]` listener), and
/// every request carries the `Plain` transport marker.
#[tokio::test]
async fn a_loopback_client_is_served_over_a_real_socket() {
    let listeners = bind([127, 0, 0, 1].into(), 0).await.unwrap();
    let port = listeners.port();
    let has_v6 = listeners.v6.is_some();
    let token = CancellationToken::new();
    let server = tokio::spawn(serve_on(listeners, test_state().into(), token.clone()));
    let client = reqwest::Client::new();
    let mut bases = vec![format!("http://127.0.0.1:{port}")];
    if has_v6 {
        bases.push(format!("http://[::1]:{port}"));
    }
    for base in bases {
        let r = client
            .get(format!("{base}/v1/vision/custom/list"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), reqwest::StatusCode::OK, "{base}");
        let r = client
            .get(format!("{base}/__test/transport"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.text().await.unwrap(), "Some(Plain)", "{base}");
        // Through the socket, a public address in a forwarding header is still refused.
        let r = client
            .get(format!("{base}/v1/vision/custom/list"))
            .header("x-forwarded-for", "8.8.8.8")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), reqwest::StatusCode::FORBIDDEN, "{base}");
    }
    token.cancel();
    server.await.unwrap().unwrap();
}

/// The tests' own requests carry the `Tls` marker unless they set one.
#[tokio::test]
async fn test_requests_are_marked_tls_unless_marked() {
    let state = test_state();
    let (_, _, b) = call(&state, get("/__test/transport", LOCAL)).await;
    assert_eq!(String::from_utf8(b).unwrap(), "Some(Tls)");
    let mut r = Request::get("/__test/transport")
        .body(Body::empty())
        .unwrap();
    r.extensions_mut().insert(super::super::Transport::Plain);
    let (_, _, b) = call(&state, r).await;
    assert_eq!(String::from_utf8(b).unwrap(), "Some(Plain)");
}
