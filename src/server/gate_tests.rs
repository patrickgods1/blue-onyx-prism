//! The plain HTTP port's gate (`plain_gate`) and the legacy cookie: what the plain port serves,
//! where it sends a browser, what it refuses, and that the `bop_admin` cookie of earlier versions
//! is cleared on both ports and never read.

use super::test_util::*;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode, header};

/// The HTTPS port the gate redirects to in these tests.
const HTTPS: u16 = 32169;

fn get(uri: &str, host: &str, html: bool) -> Request<Body> {
    let mut b = Request::get(uri).header(header::HOST, host);
    if html {
        b = b.header(header::ACCEPT, "text/html,application/xhtml+xml");
    }
    plain(from_peer(b.body(Body::empty()).unwrap(), REMOTE))
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn cookies(h: &HeaderMap) -> Vec<String> {
    h.get_all(header::SET_COOKIE)
        .iter()
        .map(|v| v.to_str().unwrap().to_string())
        .collect()
}

const CLEARED: &str = "bop_admin=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0";

/// A browser's GET of a UI page on the plain port is sent to the same page over HTTPS, under the
/// host it used (its port dropped, IPv6 bracketed), with the query; no HSTS.
#[tokio::test]
async fn browsers_are_redirected_to_https_with_the_exact_location() {
    let state = test_state();
    for (host, uri, want) in [
        (
            "localhost:32168",
            "/config",
            "https://localhost:32169/config",
        ),
        (
            "192.168.1.5:32168",
            "/train/review?a=1&b=2",
            "https://192.168.1.5:32169/train/review?a=1&b=2",
        ),
        ("[fe80::1]:32168", "/", "https://[fe80::1]:32169/"),
        ("[::1]", "/logs", "https://[::1]:32169/logs"),
        ("LOCALHOST", "/stats", "https://localhost:32169/stats"),
        (
            "localhost:32168",
            "/login?next=%2Fconfig",
            "https://localhost:32169/login?next=%2Fconfig",
        ),
        // Paths a browser would read as another host never get into Location.
        (
            "localhost:32168",
            "//evil.test/x",
            "https://localhost:32169/",
        ),
        (
            "localhost:32168",
            "/%5Cevil.test",
            "https://localhost:32169/%5Cevil.test",
        ),
    ] {
        let (s, h, b) = call_with_https(&state, Some(HTTPS), get(uri, host, true)).await;
        assert_eq!(
            s,
            StatusCode::PERMANENT_REDIRECT,
            "{host} {uri}: {}",
            text(&b)
        );
        assert_eq!(h[header::LOCATION], want, "{host} {uri}");
        assert!(!h.contains_key("strict-transport-security"));
        assert_eq!(h[header::CACHE_CONTROL], "no-store");
    }
    // A raw backslash path (an HTTP client may send one) is refused by `safe_next`: home.
    let req = Request::get("/x")
        .header(header::HOST, "localhost:32168")
        .header(header::ACCEPT, "text/html")
        .body(Body::empty())
        .unwrap();
    let mut req = plain(from_peer(req, REMOTE));
    *req.uri_mut() = "/\\evil.test"
        .parse()
        .unwrap_or_else(|_| "/x".parse().unwrap());
    let (s, h, _) = call_with_https(&state, Some(HTTPS), req).await;
    assert_eq!(s, StatusCode::PERMANENT_REDIRECT);
    let to = h[header::LOCATION].to_str().unwrap();
    assert!(
        to == "https://localhost:32169/" || to == "https://localhost:32169/x",
        "{to}"
    );
    // HEAD is a browser's GET too.
    let req = plain(from_peer(
        Request::head("/config")
            .header(header::HOST, "localhost:32168")
            .header(header::ACCEPT, "text/html")
            .body(Body::empty())
            .unwrap(),
        REMOTE,
    ));
    let (s, _, _) = call_with_https(&state, Some(HTTPS), req).await;
    assert_eq!(s, StatusCode::PERMANENT_REDIRECT);
}

/// A host the admin routes would not answer under gets 403, not a redirect to that name.
#[tokio::test]
async fn a_foreign_host_is_refused_not_redirected() {
    let state = test_state();
    for host in [
        "rebind.attacker.test:32168",
        "nas.home.arpa",
        "localhost.evil.test",
    ] {
        let (s, h, b) = call_with_https(&state, Some(HTTPS), get("/config", host, true)).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{host}");
        assert!(!h.contains_key(header::LOCATION));
        assert!(text(&b).contains("allowed_hosts"), "{}", text(&b));
    }
    // Listed: redirected under that name.
    let named = test_state_with(|c| c.allowed_hosts = vec!["nas.home.arpa".into()]);
    let (s, h, _) = call_with_https(
        &named,
        Some(HTTPS),
        get("/config", "NAS.home.arpa:32168", true),
    )
    .await;
    assert_eq!(s, StatusCode::PERMANENT_REDIRECT);
    assert_eq!(h[header::LOCATION], "https://nas.home.arpa:32169/config");
}

/// Anything but a browser's GET: 403 JSON naming the HTTPS address. A request with
/// `Authorization` is refused before anything is compared or counted.
#[tokio::test]
async fn other_requests_get_403_json_and_credentials_are_never_looked_at() {
    let state = test_state();
    std::fs::write(
        crate::admin_token::prepare(&state.config_read()).unwrap(),
        "ab".repeat(32),
    )
    .unwrap();
    let post = |uri: &str, html: bool| {
        let mut b = Request::post(uri)
            .header(header::HOST, "192.168.1.5:32168")
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded");
        if html {
            b = b.header(header::ACCEPT, "text/html");
        }
        plain(from_peer(b.body(Body::from("port=4000")).unwrap(), REMOTE))
    };
    for req in [
        post("/config", true),
        post("/config/restart", false),
        post("/login", true),
        post("/setup", false),
        get("/config", "192.168.1.5:32168", false),
        get("/prometheus", "192.168.1.5:32168", false),
    ] {
        let what = format!("{} {}", req.method(), req.uri());
        let (s, h, b) = call_with_https(&state, Some(HTTPS), req).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{what}");
        assert!(
            h[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("application/json"),
            "{what}"
        );
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert!(
            j["message"]
                .as_str()
                .unwrap()
                .contains("https://192.168.1.5:32169/"),
            "{what}: {j}"
        );
    }
    assert_eq!(
        state.config_read().port,
        crate::config::Config::default().port
    );
    assert!(!state.restart.is_cancelled());
    // Authorization: refused for a browser too, uncounted, even with the right token.
    for (i, token) in ["ab".repeat(32), "cd".repeat(32)]
        .into_iter()
        .cycle()
        .take(12)
        .enumerate()
    {
        let mut req = get("/prometheus", "localhost:32168", i % 2 == 0);
        req.headers_mut().insert(
            header::AUTHORIZATION,
            format!("Bearer {token}").parse().unwrap(),
        );
        let (s, _, b) = call_with_https(&state, Some(HTTPS), req).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{}", text(&b));
        assert!(text(&b).contains("plain HTTP"), "{}", text(&b));
    }
    let remote = "10.0.0.5".parse().unwrap();
    assert_eq!(state.admin_failures.lock().unwrap().count(remote), 0);
    assert_eq!(state.login.failures.lock().unwrap().count(remote), 0);
}

/// The HTTPS listener down: a browser learns that, rather than being sent nowhere.
#[tokio::test]
async fn https_down_is_503() {
    let state = test_state();
    let (s, h, b) = call_with_https(&state, None, get("/config", "localhost:32168", true)).await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
    assert!(!h.contains_key(header::LOCATION));
    assert!(text(&b).contains("HTTPS"), "{}", text(&b));
    // Blue Iris is not affected.
    let req = get("/v1/vision/custom/list", "localhost:32168", false);
    assert_eq!(call_with_https(&state, None, req).await.0, StatusCode::OK);
}

/// The open routes are served on the plain port as on HTTPS, for anyone on the LAN.
#[tokio::test]
async fn open_routes_are_served_on_the_plain_port() {
    let state = test_state();
    for uri in [
        "/v1/vision/custom/list",
        "/v1/config",
        "/static/style.css",
        "/static/live.js",
        "/favicon.ico",
    ] {
        let (s, _, _) = call_with_https(&state, Some(HTTPS), get(uri, "blue-iris-pc", true)).await;
        assert_eq!(s, StatusCode::OK, "{uri}");
    }
    let (s, _, _) =
        call_with_https(&state, Some(HTTPS), get("/v1/status", "localhost", false)).await;
    assert_eq!(
        s,
        StatusCode::SERVICE_UNAVAILABLE,
        "the test models failed: 503, not 403"
    );
    let req = plain(from_peer(
        Request::post("/v1/vision/custom/ipcam-general")
            .header(header::CONTENT_TYPE, "multipart/form-data; boundary=X")
            .body(Body::from("--X--\r\n"))
            .unwrap(),
        REMOTE,
    ));
    let (s, _, _) = call_with_https(&state, Some(HTTPS), req).await;
    assert_ne!(s, StatusCode::FORBIDDEN);
    // An unknown path is no open route: gated like the UI.
    let (s, _, _) = call_with_https(&state, Some(HTTPS), get("/no/such", "localhost", false)).await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    // A request without the transport marker (not from the accept loop) counts as plain, also
    // with the test session's cookie.
    let mut req = from_peer(
        Request::get("/config")
            .header(header::ACCEPT, "text/html")
            .body(Body::empty())
            .unwrap(),
        LOCAL,
    );
    req.extensions_mut().remove::<super::Transport>();
    let resp = tower::ServiceExt::oneshot(router_with_https(&state, Some(HTTPS)), req)
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::PERMANENT_REDIRECT);
}

/// `bop_admin` is cleared on every plain-port answer and on HTTPS answers to requests that send
/// it; it never signs anyone in.
#[tokio::test]
async fn the_legacy_cookie_is_cleared_on_both_ports_and_ignored() {
    let state = test_state();
    std::fs::write(
        crate::admin_token::prepare(&state.config_read()).unwrap(),
        "ab".repeat(32),
    )
    .unwrap();
    for req in [
        get("/v1/vision/custom/list", "localhost", false),
        get("/config", "localhost:32168", true),
        get("/config", "localhost:32168", false),
    ] {
        let (_, h, _) = call_with_https(&state, Some(HTTPS), req).await;
        assert!(
            cookies(&h).contains(&CLEARED.to_string()),
            "{:?}",
            cookies(&h)
        );
    }
    // HTTPS: only when the request sends it.
    let (_, h, _) = call(
        &state,
        from_peer(
            Request::get("/v1/status").body(Body::empty()).unwrap(),
            LOCAL,
        ),
    )
    .await;
    assert!(cookies(&h).is_empty(), "{:?}", cookies(&h));
    let legacy = |uri: &str, html: bool| {
        let mut b =
            Request::get(uri).header(header::COOKIE, format!("bop_admin={}", "ab".repeat(32)));
        if html {
            b = b.header(header::ACCEPT, "text/html");
        }
        from_peer(b.body(Body::empty()).unwrap(), REMOTE)
    };
    let (s, h, _) = call(&state, legacy("/config", false)).await;
    assert_eq!(
        s,
        StatusCode::UNAUTHORIZED,
        "the old cookie with the right token is no sign-in"
    );
    assert!(
        cookies(&h).contains(&CLEARED.to_string()),
        "{:?}",
        cookies(&h)
    );
    let (s, h, _) = call(&state, legacy("/config", true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert!(cookies(&h).contains(&CLEARED.to_string()));
    let (_, h, b) = call(&state, legacy("/v1/config", false)).await;
    assert!(cookies(&h).contains(&CLEARED.to_string()));
    let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
    assert!(j.get("config").is_none(), "{j}");
    // Uncounted: it is not compared at all.
    assert_eq!(
        state
            .admin_failures
            .lock()
            .unwrap()
            .count("10.0.0.5".parse().unwrap()),
        0
    );
}
