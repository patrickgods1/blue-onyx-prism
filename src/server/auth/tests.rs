//! The admin token at the router: who needs it, what it opens and what a refusal says.

use super::super::test_util::*;
use super::*;
use crate::server::{AppState, router};
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode, header};
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower::ServiceExt;

fn token_a() -> String {
    "ab".repeat(32)
}

fn token_b() -> String {
    "cd".repeat(32)
}

fn token_path(state: &AppState) -> PathBuf {
    state.config_read().data_root().join("admin-token")
}

/// Remote administration on, the token file holding [`token_a`].
fn remote_state() -> Arc<AppState> {
    let state = test_state_with(|c| c.allow_remote_admin = true);
    std::fs::write(token_path(&state), format!("{}\n", token_a())).unwrap();
    state
}

/// `method uri` from `peer`, as a browser or not; POSTs have an empty form body.
fn req(peer: &str, method: &str, uri: &str, html: bool) -> Request<Body> {
    let mut b = Request::builder().method(method).uri(uri);
    if method == "POST" {
        b = b.header(header::CONTENT_TYPE, "application/x-www-form-urlencoded");
    }
    if html {
        b = b.header(header::ACCEPT, "text/html");
    }
    from_peer(b.body(Body::empty()).unwrap(), peer)
}

fn bearer(mut r: Request<Body>, token: &str) -> Request<Body> {
    r.headers_mut().insert(
        header::AUTHORIZATION,
        format!("Bearer {token}").parse().unwrap(),
    );
    r
}

fn cookie(mut r: Request<Body>, token: &str) -> Request<Body> {
    r.headers_mut().insert(
        header::COOKIE,
        format!("other=1; bop_admin={token}; last=2")
            .parse()
            .unwrap(),
    );
    r
}

/// `POST /login` from `peer` with this urlencoded body.
fn login(peer: &str, body: String, html: bool) -> Request<Body> {
    let mut b =
        Request::post("/login").header(header::CONTENT_TYPE, "application/x-www-form-urlencoded");
    if html {
        b = b.header(header::ACCEPT, "text/html");
    }
    from_peer(b.body(Body::from(body)).unwrap(), peer)
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn location(h: &HeaderMap) -> &str {
    h[header::LOCATION].to_str().unwrap()
}

fn set_cookie(h: &HeaderMap) -> String {
    h.get_all(header::SET_COOKIE)
        .iter()
        .map(|v| v.to_str().unwrap())
        .collect::<Vec<_>>()
        .join("\n")
}

fn with_header(mut r: Request<Body>, name: header::HeaderName, value: &str) -> Request<Body> {
    r.headers_mut().insert(name, value.parse().unwrap());
    r
}

fn authorization(r: Request<Body>, value: &str) -> Request<Body> {
    with_header(r, header::AUTHORIZATION, value)
}

fn origin(r: Request<Body>, value: &str) -> Request<Body> {
    with_header(r, header::ORIGIN, value)
}

fn host(r: Request<Body>, value: &str) -> Request<Body> {
    with_header(r, header::HOST, value)
}

#[tokio::test]
async fn a_remote_peer_without_the_token_is_refused_on_get_and_post() {
    let state = remote_state();
    for (method, uri) in [
        ("GET", "/config"),
        ("POST", "/config"),
        ("GET", "/logs.json"),
        ("POST", "/config/restart"),
        ("GET", "/dataset/image?path=x.jpg"),
    ] {
        let (s, h, b) = call(&state, req(REMOTE, method, uri, false)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "{method} {uri}: {}", text(&b));
        assert_eq!(h[header::WWW_AUTHENTICATE], "Bearer");
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(j["success"], false);
        assert_eq!(j["error"], j["message"]);
        assert!(j["message"].as_str().unwrap().contains("/login"), "{j}");
    }
    assert!(!state.restart.is_cancelled());
    // A browser is sent to the login page, and back to where it was going.
    let (s, h, _) = call(&state, req(REMOTE, "GET", "/config?x=1", true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/login?next=%2Fconfig%3Fx%3D1");
    // A POST cannot be repeated by a redirect, and its path may not take GET at all.
    let (s, h, _) = call(&state, req(REMOTE, "POST", "/config/restart", true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/login");
}

#[tokio::test]
async fn bearer_and_cookie_open_the_admin_routes() {
    let state = remote_state();
    for (method, uri) in [
        ("GET", "/config"),
        ("GET", "/logs.json"),
        ("POST", "/v1/benchmark/cancel"),
    ] {
        let (s, _, b) = call(&state, bearer(req(REMOTE, method, uri, false), &token_a())).await;
        assert!(
            ![401, 403, 429].contains(&s.as_u16()),
            "{method} {uri}: {}",
            text(&b)
        );
        let (s, _, b) = call(&state, cookie(req(REMOTE, method, uri, false), &token_a())).await;
        assert!(
            ![401, 403, 429].contains(&s.as_u16()),
            "{method} {uri}: {}",
            text(&b)
        );
    }
    // The scheme is case-insensitive (RFC 9110).
    let r = authorization(
        req(REMOTE, "GET", "/config", false),
        &format!("bearer {}", token_a()),
    );
    assert_eq!(call(&state, r).await.0, StatusCode::OK);
    // Valid traffic never fills the failure list, however much of it there is.
    for _ in 0..(MAX_FAILURES * 3) {
        let r = cookie(req(REMOTE, "GET", "/logs.json", false), &token_a());
        assert_eq!(call(&state, r).await.0, StatusCode::OK);
    }
}

#[tokio::test]
async fn a_wrong_token_is_refused_and_never_repeated() {
    let state = remote_state();
    let wrong = token_b();
    let (s, h, b) = call(&state, bearer(req(REMOTE, "GET", "/config", false), &wrong)).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert!(!text(&b).contains(&wrong) && !text(&b).contains(&token_a()));
    assert!(!format!("{h:?}").contains(&token_a()));
    // A stale cookie is dropped, so one rotated token costs a browser one failure, not five.
    let (s, h, _) = call(&state, cookie(req(REMOTE, "GET", "/config", true), &wrong)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    let cleared = set_cookie(&h);
    assert!(
        cleared.contains("bop_admin=;") && cleared.contains("Max-Age=0"),
        "{cleared}"
    );
    // Other kinds of credential are no token, and bytes that are not text are a wrong one.
    let r = authorization(
        req(REMOTE, "GET", "/config", false),
        &format!("Basic {}", token_a()),
    );
    assert_eq!(call(&state, r).await.0, StatusCode::UNAUTHORIZED);
    let mut r = req(REMOTE, "GET", "/config", false);
    r.headers_mut().insert(
        header::AUTHORIZATION,
        header::HeaderValue::from_bytes(b"Bearer \xff\xfe").unwrap(),
    );
    assert_eq!(call(&state, r).await.0, StatusCode::UNAUTHORIZED);
}

#[test]
fn the_limit_allows_five_failures_a_minute_per_peer() {
    let mut l = FailureLimit::default();
    let ip: IpAddr = "10.0.0.5".parse().unwrap();
    let other: IpAddr = "10.0.0.6".parse().unwrap();
    let t0 = Instant::now();
    for i in 0..5 {
        assert_eq!(
            l.blocked(ip, t0 + Duration::from_secs(i)),
            None,
            "failure {i}"
        );
        l.fail(ip, t0 + Duration::from_secs(i));
    }
    // Blocked until the oldest of the five is a minute old, and the wait is rounded up.
    assert_eq!(l.blocked(ip, t0 + Duration::from_secs(5)), Some(55));
    assert_eq!(l.blocked(ip, t0 + Duration::from_millis(59_500)), Some(1));
    assert_eq!(l.blocked(other, t0 + Duration::from_secs(5)), None);
    assert_eq!(l.blocked(ip, t0 + Duration::from_secs(60)), None);
    // One failure left the window: one more may happen, then the next oldest decides.
    l.fail(ip, t0 + Duration::from_secs(60));
    assert_eq!(l.blocked(ip, t0 + Duration::from_secs(60)), Some(1));
    assert_eq!(l.blocked(ip, t0 + Duration::from_secs(120)), None);
}

#[test]
fn the_limit_treats_a_mapped_address_as_the_same_peer() {
    let mut l = FailureLimit::default();
    let t0 = Instant::now();
    for _ in 0..5 {
        l.fail("::ffff:10.0.0.5".parse().unwrap(), t0);
    }
    assert!(l.blocked("10.0.0.5".parse().unwrap(), t0).is_some());
}

#[test]
fn the_limit_keeps_at_most_1024_peers_and_drops_the_oldest() {
    let mut l = FailureLimit::default();
    let t0 = Instant::now();
    let peer = |i: u32| IpAddr::from(std::net::Ipv4Addr::from(0x0a00_0000 + i));
    for _ in 0..5 {
        l.fail(peer(0), t0);
    }
    for i in 1..=(MAX_PEERS as u32) {
        l.fail(peer(i), t0 + Duration::from_millis(u64::from(i)));
        assert!(l.peers() <= MAX_PEERS);
    }
    assert_eq!(l.peers(), MAX_PEERS);
    // Peer 0 was the oldest; the newest are all there.
    let t = t0 + Duration::from_secs(2);
    assert_eq!(l.blocked(peer(0), t), None);
    for _ in 0..4 {
        l.fail(peer(MAX_PEERS as u32), t);
    }
    assert!(l.blocked(peer(MAX_PEERS as u32), t).is_some());
    // Peers whose failures are over a minute old go first.
    l.fail(peer(5000), t0 + Duration::from_secs(120));
    assert_eq!(l.peers(), 1);
}

/// Six wrong tokens: the sixth is 429 without being compared, so the right token is refused as
/// well, and a minute later (the failures aged by the test, no sleep) it works again.
#[tokio::test]
async fn wrong_tokens_are_limited_per_peer_and_recover_after_a_minute() {
    let state = remote_state();
    let wrong = token_b();
    for i in 0..MAX_FAILURES {
        let r = bearer(req(REMOTE, "GET", "/config", false), &wrong);
        assert_eq!(
            call(&state, r).await.0,
            StatusCode::UNAUTHORIZED,
            "failure {i}"
        );
    }
    let r = bearer(req(REMOTE, "GET", "/config", false), &wrong);
    let (s, h, b) = call(&state, r).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    let wait: u64 = h[header::RETRY_AFTER].to_str().unwrap().parse().unwrap();
    assert!((1..=60).contains(&wait), "{wait}");
    let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
    assert_eq!(j["success"], false);
    assert_eq!(j["error"], j["message"]);
    let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::TOO_MANY_REQUESTS);
    // A valid cookie is not thrown away by what another client at this address did.
    let r = cookie(req(REMOTE, "GET", "/config", false), &token_a());
    let (s, h, _) = call(&state, r).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(set_cookie(&h), "");
    // A browser with nothing to try still reaches the login page.
    let (s, _, _) = call(&state, req(REMOTE, "GET", "/config", true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    // Another peer is not affected, nor is this machine.
    let r = bearer(req("10.0.0.6:50000", "GET", "/config", false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::OK);
    assert_eq!(
        call(&state, req(LOCAL, "GET", "/config", false)).await.0,
        StatusCode::OK
    );

    state.admin_failures.lock().unwrap().age(WINDOW);
    let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::OK);
}

#[tokio::test]
async fn wrong_logins_count_towards_the_same_limit() {
    let state = remote_state();
    let wrong = format!("token={}", token_b());
    for i in 0..MAX_FAILURES {
        let (s, _, b) = call(&state, login(REMOTE, wrong.clone(), true)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "failure {i}");
        assert!(text(&b).contains("name=\"token\""), "the form again");
        assert!(!text(&b).contains(&token_b()), "the page repeats the try");
    }
    let (s, h, b) = call(&state, login(REMOTE, format!("token={}", token_a()), true)).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert!(h.contains_key(header::RETRY_AFTER));
    assert_eq!(set_cookie(&h), "");
    assert!(text(&b).contains("Too many"), "{}", text(&b));
    // One peer, one budget: a Bearer from it is limited as well.
    let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::TOO_MANY_REQUESTS);
    // A non-browser client gets JSON; an empty form is no attempt.
    state.admin_failures.lock().unwrap().age(WINDOW);
    let (s, h, b) = call(&state, login(REMOTE, wrong.clone(), false)).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert!(
        h[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("application/json")
    );
    let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
    assert_eq!(j["success"], false);
    for _ in 0..(MAX_FAILURES * 2) {
        let (s, _, _) = call(&state, login(REMOTE, "token=".into(), true)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
    }
    let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::OK);
}

#[tokio::test]
async fn the_login_form_sets_the_cookie_and_logout_clears_it() {
    let state = remote_state();
    let uri = "/login?next=%2Fconfig%3Fa%3D%22x";
    let (s, h, b) = call(&state, req(REMOTE, "GET", uri, true)).await;
    assert_eq!(s, StatusCode::OK);
    assert!(
        h[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/html")
    );
    assert!(h.contains_key("content-security-policy"));
    let html = text(&b);
    assert!(html.contains("type=\"password\""), "{html}");
    assert!(html.contains("action=\"/login\""), "{html}");
    assert!(
        !html.contains(" style=") && !html.contains("<script>"),
        "CSP: {html}"
    );
    // Escaped: the quote in `next` cannot leave the attribute.
    assert!(
        html.contains("&#34;x") || html.contains("&quot;x"),
        "{html}"
    );
    assert!(!html.contains(&token_a()));

    let body = format!("token={}&next=%2Ftrain%2Freview%3Fa%3D1", token_a());
    let (s, h, _) = call(&state, login(REMOTE, body, true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/train/review?a=1");
    assert_eq!(
        set_cookie(&h),
        format!("bop_admin={}; HttpOnly; SameSite=Strict; Path=/", token_a())
    );
    // Pasted with the newline `cat` prints.
    let (s, h, _) = call(
        &state,
        login(REMOTE, format!("token={}%0A", token_a()), true),
    )
    .await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/");

    let r = cookie(req(REMOTE, "POST", "/logout", true), &token_a());
    let (s, h, _) = call(&state, r).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/login");
    let cleared = set_cookie(&h);
    assert!(
        cleared.starts_with("bop_admin=;")
            && cleared.contains("Max-Age=0")
            && cleared.contains("Path=/"),
        "{cleared}"
    );
    assert!(!cleared.contains(&token_a()));
}

/// `/login` answers remote peers with no token, and only then: the same `Host` and CSRF
/// checks as an admin route, and the same 403 while remote administration is off.
#[tokio::test]
async fn login_goes_through_the_host_and_csrf_checks() {
    let state = remote_state();
    let rebound = "rebind.attacker.test:32168";
    for (method, uri) in [("GET", "/login"), ("POST", "/login"), ("POST", "/logout")] {
        let (s, _, b) = call(&state, host(req(REMOTE, method, uri, false), rebound)).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{method} {uri}");
        assert!(text(&b).contains("allowed_hosts"), "{}", text(&b));
    }
    for uri in ["/login", "/logout"] {
        let mut r = req(REMOTE, "POST", uri, false);
        r.headers_mut()
            .insert("sec-fetch-site", "cross-site".parse().unwrap());
        assert_eq!(call(&state, r).await.0, StatusCode::FORBIDDEN, "{uri}");
        let r = origin(req(REMOTE, "POST", uri, false), "http://evil.test");
        assert_eq!(call(&state, r).await.0, StatusCode::FORBIDDEN, "{uri}");
    }
    // A cross-site login with the right token sets nothing.
    let r = origin(
        login(REMOTE, format!("token={}", token_a()), true),
        "http://evil.test",
    );
    let (s, h, _) = call(&state, r).await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert_eq!(set_cookie(&h), "");
    // The same name passes once it is listed, with the browser's own Origin.
    let named = test_state_with(|c| {
        c.allow_remote_admin = true;
        c.allowed_hosts = vec!["nas.home.arpa".into()];
    });
    std::fs::write(token_path(&named), token_a()).unwrap();
    let r = login(REMOTE, format!("token={}", token_a()), true);
    let r = origin(host(r, "nas.home.arpa:32168"), "http://nas.home.arpa:32168");
    assert_eq!(call(&named, r).await.0, StatusCode::SEE_OTHER);
    // Remote administration off: nothing changes, not even for the new paths.
    let off = test_state();
    let (s, _, b) = call(&off, req(REMOTE, "GET", "/login", false)).await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert!(text(&b).contains("allow_remote_admin"));
    let (s, _, _) = call(&off, login(REMOTE, format!("token={}", token_a()), false)).await;
    assert_eq!(s, StatusCode::FORBIDDEN);
}

#[test]
fn next_is_a_local_path_or_nothing() {
    for ok in [
        "/",
        "/config",
        "/train/review?a=1&b=/x",
        "/config#x",
        "/a//b",
        "/a?u=http://x",
    ] {
        assert_eq!(safe_next(ok), Some(ok), "{ok}");
    }
    for bad in [
        "",
        "//evil",
        "//evil.test/x",
        "/\\evil",
        "\\\\evil",
        "\\/evil",
        "http://evil",
        "https://evil/x",
        "HTTP://evil",
        "javascript:alert(1)",
        "evil",
        "config",
        " /config",
        "/\t/evil",
        "/\n/evil",
        "/\r/evil",
        "/ /evil",
        "/a\\b",
        "/caf\u{e9}",
    ] {
        assert_eq!(safe_next(bad), None, "{bad:?}");
    }
    assert_eq!(safe_next(&format!("/{}", "a".repeat(5000))), None);
}

#[tokio::test]
async fn login_never_redirects_off_this_server() {
    let state = remote_state();
    for evil in [
        "//evil.test",
        "/%5Cevil.test",
        "%2F%5Cevil.test",
        "http://evil.test",
        "https%3A%2F%2Fevil.test",
        "%2F%2Fevil.test",
        "/%09/evil.test",
        "javascript:alert(1)",
    ] {
        let body = format!("token={}&next={evil}", token_a());
        let (s, h, _) = call(&state, login(REMOTE, body, true)).await;
        assert_eq!(s, StatusCode::SEE_OTHER, "{evil}");
        assert_eq!(location(&h), "/", "{evil}");
        // The form does not carry it on either.
        let uri = format!("/login?next={evil}");
        let (s, _, b) = call(&state, req(REMOTE, "GET", &uri, true)).await;
        assert_eq!(s, StatusCode::OK, "{evil}");
        assert!(!text(&b).contains("evil.test"), "{}", text(&b));
    }
}

#[tokio::test]
async fn loopback_needs_no_token() {
    let state = remote_state();
    std::fs::remove_file(token_path(&state)).unwrap();
    assert_eq!(
        call(&state, req(LOCAL, "GET", "/config", true)).await.0,
        StatusCode::OK
    );
    assert_eq!(
        call(&state, req("[::1]:1", "GET", "/config", false))
            .await
            .0,
        StatusCode::OK
    );
    // A wrong credential from this machine is ignored, and never counted.
    for _ in 0..(MAX_FAILURES * 2) {
        let r = bearer(req(LOCAL, "GET", "/config", false), &token_b());
        assert_eq!(call(&state, r).await.0, StatusCode::OK);
    }
    let (_, _, b) = call(&state, req(LOCAL, "GET", "/v1/config", false)).await;
    let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
    assert!(j["config"].is_object());
    // The login page has nothing to ask this machine for.
    let (s, h, _) = call(&state, req(LOCAL, "GET", "/login?next=/config", true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/config");
    // Remote administration off changes nothing for this machine either.
    let off = test_state();
    assert_eq!(
        call(&off, req(LOCAL, "GET", "/config", true)).await.0,
        StatusCode::OK
    );
    assert!(!token_path(&off).exists());
}

#[tokio::test]
async fn open_routes_never_need_the_token() {
    let state = remote_state();
    for uri in [
        "/",
        "/stats.json",
        "/prometheus",
        "/v1/devices",
        "/v1/status",
        "/v1/status/updateavailable",
        "/static/style.css",
        "/favicon.ico",
        "/v1/vision/custom/list",
    ] {
        let (s, _, _) = call(&state, req(REMOTE, "GET", uri, false)).await;
        assert!(
            s != StatusCode::UNAUTHORIZED && s != StatusCode::FORBIDDEN,
            "{uri}: {s}"
        );
        // And a wrong credential is not looked at, nor counted.
        for _ in 0..MAX_FAILURES {
            let r = bearer(req(REMOTE, "GET", uri, false), &token_b());
            assert!(call(&state, r).await.0 != StatusCode::TOO_MANY_REQUESTS);
        }
    }
    let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::OK);
}

fn has_config(b: &[u8]) -> bool {
    let j: serde_json::Value = serde_json::from_slice(b).unwrap();
    j.get("config").is_some() || j.get("path").is_some()
}

#[tokio::test]
async fn the_full_config_payload_needs_the_token() {
    let state = remote_state();
    let (s, _, b) = call(&state, req(REMOTE, "GET", "/v1/config", false)).await;
    assert_eq!(s, StatusCode::OK);
    assert!(!has_config(&b), "{}", text(&b));
    let r = bearer(req(REMOTE, "GET", "/v1/config", false), &token_a());
    let (_, _, b) = call(&state, r).await;
    assert!(has_config(&b));
    let r = cookie(req(REMOTE, "GET", "/v1/config", false), &token_a());
    let (_, _, b) = call(&state, r).await;
    assert!(has_config(&b));

    // A wrong token gets the minimal answer, counts, and a stale cookie goes.
    let r = cookie(req(REMOTE, "GET", "/v1/config", false), &token_b());
    let (s, h, b) = call(&state, r).await;
    assert_eq!(s, StatusCode::OK);
    assert!(!has_config(&b));
    assert!(set_cookie(&h).contains("Max-Age=0"));
    for _ in 1..MAX_FAILURES {
        let r = bearer(req(REMOTE, "GET", "/v1/config", false), &token_b());
        let (s, _, b) = call(&state, r).await;
        assert_eq!(s, StatusCode::OK);
        assert!(!has_config(&b));
    }
    // Limited: still the minimal answer (the pages' poller reads this route), and the right
    // token no longer tells the two answers apart.
    let r = bearer(req(REMOTE, "GET", "/v1/config", false), &token_a());
    let (s, _, b) = call(&state, r).await;
    assert_eq!(s, StatusCode::OK);
    assert!(!has_config(&b));
    let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::TOO_MANY_REQUESTS);

    // A token does not lift the `Host` check.
    state.admin_failures.lock().unwrap().age(WINDOW);
    let rebound = "rebind.attacker.test";
    let r = bearer(req(REMOTE, "GET", "/v1/config", false), &token_a());
    let (_, _, b) = call(&state, host(r, rebound)).await;
    assert!(!has_config(&b));
    let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
    assert_eq!(
        call(&state, host(r, rebound)).await.0,
        StatusCode::FORBIDDEN
    );

    // With remote administration off a token means nothing.
    let off = test_state();
    std::fs::write(token_path(&off), token_a()).unwrap();
    let r = bearer(req(REMOTE, "GET", "/v1/config", false), &token_a());
    let (_, _, b) = call(&off, r).await;
    assert!(!has_config(&b));
    let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
    assert_eq!(call(&off, r).await.0, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn rotating_invalidates_the_old_token_at_once() {
    let state = remote_state();
    let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::OK);
    // As `admin-token --rotate` from another process does: only the file changes.
    crate::admin_token::rotate(&token_path(&state)).unwrap();
    let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::UNAUTHORIZED);
    let new = std::fs::read_to_string(token_path(&state)).unwrap();
    let r = bearer(req(REMOTE, "GET", "/config", false), new.trim());
    assert_eq!(call(&state, r).await.0, StatusCode::OK);
}

#[tokio::test]
async fn the_rotate_button_is_for_this_machine_and_never_shows_the_token() {
    const ROTATE: &str = "/config/admin-token/rotate";
    let state = remote_state();
    let on_disk = |s: &AppState| {
        std::fs::read_to_string(token_path(s))
            .unwrap()
            .trim()
            .to_string()
    };
    // A remote peer without the token, then with it: not for it, nothing changes.
    assert_eq!(
        call(&state, req(REMOTE, "POST", ROTATE, false)).await.0,
        StatusCode::UNAUTHORIZED
    );
    let r = bearer(req(REMOTE, "POST", ROTATE, false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::FORBIDDEN);
    assert_eq!(on_disk(&state), token_a());
    // A local proxy's client is not this machine: it needs the token, then is refused.
    let mut r = req(LOCAL, "POST", ROTATE, false);
    r.headers_mut()
        .insert("x-forwarded-for", "203.0.113.9".parse().unwrap());
    assert_eq!(call(&state, r).await.0, StatusCode::UNAUTHORIZED);
    let mut r = bearer(req(LOCAL, "POST", ROTATE, false), &token_a());
    r.headers_mut()
        .insert("x-forwarded-for", "203.0.113.9".parse().unwrap());
    assert_eq!(call(&state, r).await.0, StatusCode::FORBIDDEN);
    // CSRF: the guard runs before the handler.
    let r = origin(req(LOCAL, "POST", ROTATE, true), "http://evil.test");
    assert_eq!(call(&state, r).await.0, StatusCode::FORBIDDEN);
    assert_eq!(on_disk(&state), token_a());

    let (s, h, b) = call(&state, req(LOCAL, "POST", ROTATE, true)).await;
    assert_eq!(s, StatusCode::OK);
    let new = on_disk(&state);
    assert_ne!(new, token_a());
    assert!(!text(&b).contains(&new) && !text(&b).contains(&token_a()));
    assert!(!format!("{h:?}").contains(&new));
    let (s, _, b) = call(&state, req(LOCAL, "POST", ROTATE, false)).await;
    assert_eq!(s, StatusCode::OK);
    let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
    assert_eq!(j["success"], true);
    assert!(!text(&b).contains(&on_disk(&state)));
}

#[tokio::test]
async fn the_config_page_shows_the_file_to_this_machine_only() {
    let state = remote_state();
    let path = token_path(&state).display().to_string();
    let (s, _, b) = call(&state, req(LOCAL, "GET", "/config", true)).await;
    assert_eq!(s, StatusCode::OK);
    let html = text(&b);
    assert!(html.contains(&path), "the file's path");
    assert!(
        html.contains("action=\"/config/admin-token/rotate#admin-token\""),
        "the Rotate button"
    );
    assert!(!html.contains(&token_a()));
    let r = cookie(req(REMOTE, "GET", "/config", true), &token_a());
    let (s, _, b) = call(&state, r).await;
    assert_eq!(s, StatusCode::OK);
    let html = text(&b);
    assert!(!html.contains(&path), "the file's path, for a remote peer");
    assert!(!html.contains("/config/admin-token/rotate"));
    assert!(
        html.contains("action=\"/logout\""),
        "a remote browser can sign out"
    );
    assert!(!html.contains(&token_a()));
    // Remote administration off: the page does not change.
    let off = test_state();
    let (_, _, b) = call(&off, req(LOCAL, "GET", "/config", true)).await;
    assert!(!text(&b).contains("/config/admin-token/rotate"));
    // Saving the checkbox on from this machine shows it right away, and makes the token.
    let r = from_peer(
        post_form("/config", "access_settings=1&allow_remote_admin=on", true),
        LOCAL,
    );
    let (s, _, b) = call(&off, r).await;
    assert_eq!(s, StatusCode::OK);
    assert!(off.config_read().allow_remote_admin);
    assert!(text(&b).contains("/config/admin-token/rotate"));
    assert!(token_path(&off).is_file());
}

/// The file missing, empty, malformed or unreadable: no remote peer gets in, whatever it
/// sends, the error is what the operator needs, and this machine is not affected.
#[tokio::test]
async fn a_missing_or_unreadable_token_file_fails_closed() {
    type Damage = fn(&PathBuf);
    let cases: [(&str, Damage); 5] = [
        ("missing", |p| std::fs::remove_file(p).unwrap()),
        ("empty", |p| std::fs::write(p, "").unwrap()),
        ("blank", |p| std::fs::write(p, " \n").unwrap()),
        ("short", |p| std::fs::write(p, "ab").unwrap()),
        ("a folder", |p| {
            std::fs::remove_file(p).unwrap();
            std::fs::create_dir(p).unwrap();
        }),
    ];
    for (name, damage) in cases {
        let state = remote_state();
        let path = token_path(&state);
        damage(&path);
        for attempt in [token_a(), "ab".to_string()] {
            let r = bearer(req(REMOTE, "GET", "/config", false), &attempt);
            let (s, _, b) = call(&state, r).await;
            assert_eq!(s, StatusCode::UNAUTHORIZED, "{name} with {attempt:?}");
            assert!(
                !text(&b).contains(&path.display().to_string()),
                "{}",
                text(&b)
            );
            let r = cookie(req(REMOTE, "GET", "/config", false), &attempt);
            let (s, _, _) = call(&state, r).await;
            assert_eq!(s, StatusCode::UNAUTHORIZED, "{name} with a cookie");
            let r = bearer(req(REMOTE, "GET", "/v1/config", false), &attempt);
            let (_, _, b) = call(&state, r).await;
            assert!(!has_config(&b), "{name}");
        }
        // Nothing at all after `Bearer` is no empty token that an empty file could match.
        let r = authorization(req(REMOTE, "GET", "/config", false), "Bearer ");
        assert_eq!(call(&state, r).await.0, StatusCode::UNAUTHORIZED, "{name}");
        let r = cookie(req(REMOTE, "GET", "/config", false), "");
        assert_eq!(call(&state, r).await.0, StatusCode::UNAUTHORIZED, "{name}");

        let (s, h, b) = call(&state, login(REMOTE, format!("token={}", token_a()), true)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "{name}");
        assert_eq!(set_cookie(&h), "");
        assert!(
            text(&b).contains("admin-token"),
            "the next action: {}",
            text(&b)
        );
        // The file's own problem is no attempt: the token works the moment it is back.
        if !path.is_dir() {
            std::fs::write(&path, token_a()).unwrap();
            let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
            assert_eq!(call(&state, r).await.0, StatusCode::OK, "{name}");
        }
        assert_eq!(
            call(&state, req(LOCAL, "GET", "/config", false)).await.0,
            StatusCode::OK
        );
    }
}

/// A live change that turns remote administration on makes the token; nothing else does.
#[tokio::test]
async fn turning_remote_admin_on_creates_the_token() {
    let state = test_state();
    let path = token_path(&state);
    crate::server::save_config(&state, "test", |c| {
        c.port = 4000;
        Ok(())
    })
    .unwrap();
    assert!(!path.exists());
    crate::server::save_config(&state, "test", |c| {
        c.allow_remote_admin = true;
        Ok(())
    })
    .unwrap();
    let token = crate::admin_token::read(&path).unwrap();
    // Later saves keep it.
    crate::server::save_config(&state, "test", |c| {
        c.port = 4001;
        Ok(())
    })
    .unwrap();
    assert_eq!(
        crate::admin_token::read(&path).unwrap().reveal(),
        token.reveal()
    );
}

/// A request with no peer address is not trusted as this machine's.
#[tokio::test]
async fn a_request_without_a_peer_address_needs_the_token() {
    let state = remote_state();
    let r = Request::get("/config")
        .header(header::HOST, TEST_HOST)
        .body(Body::empty())
        .unwrap();
    let resp = router(state.clone()).oneshot(r).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

/// Every header a reverse proxy adds, as a client sends it.
const PROXY_HEADERS: [(&str, &str); 4] = [
    ("x-forwarded-for", "203.0.113.9"),
    ("X-Forwarded-For", "203.0.113.9, 198.51.100.7"),
    ("forwarded", "for=203.0.113.9"),
    ("x-real-ip", "203.0.113.9"),
];

fn forwarded(mut r: Request<Body>, name: &str, value: &str) -> Request<Body> {
    r.headers_mut().insert(
        header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
        value.parse().unwrap(),
    );
    r
}

/// A forwarded request is a proxied client, not a person on this machine: with remote
/// administration off it gets the usual 403, and it is told why it does not count as local.
#[tokio::test]
async fn a_forwarded_loopback_request_is_refused_with_remote_admin_off() {
    let state = test_state();
    for (name, value) in PROXY_HEADERS {
        for (method, uri) in [
            ("GET", "/config"),
            ("POST", "/config/restart"),
            ("GET", "/logs.json"),
            ("GET", "/login"),
            ("POST", "/logout"),
        ] {
            let r = forwarded(req(LOCAL, method, uri, false), name, value);
            let (s, _, b) = call(&state, r).await;
            assert_eq!(s, StatusCode::FORBIDDEN, "{name} {method} {uri}");
            let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
            let msg = j["message"].as_str().unwrap();
            assert!(msg.contains("allow_remote_admin"), "{msg}");
            assert!(msg.contains("proxy"), "{msg}");
        }
        let r = forwarded(req(LOCAL, "GET", "/config", true), name, value);
        let (s, h, b) = call(&state, r).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{name}");
        assert!(
            h[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("text/html")
        );
        assert!(text(&b).contains("Forbidden"));
        let r = forwarded(req(LOCAL, "GET", "/v1/config", false), name, value);
        let (s, _, b) = call(&state, r).await;
        assert_eq!(s, StatusCode::OK, "{name}");
        assert!(!has_config(&b), "{name}");
    }
    assert!(!state.restart.is_cancelled());
    // Open routes do not care.
    let r = forwarded(
        req(LOCAL, "GET", "/v1/vision/custom/list", false),
        "x-forwarded-for",
        "203.0.113.9",
    );
    assert_eq!(call(&state, r).await.0, StatusCode::OK);
    // Without the header this machine is unchanged.
    assert_eq!(
        call(&state, req(LOCAL, "GET", "/config", false)).await.0,
        StatusCode::OK
    );
}

/// With remote administration on, a forwarded request needs the token like any remote peer.
#[tokio::test]
async fn a_forwarded_loopback_request_needs_the_token_with_remote_admin_on() {
    for (name, value) in PROXY_HEADERS {
        let state = remote_state();
        let plain = |method, uri, html| forwarded(req(LOCAL, method, uri, html), name, value);
        assert_eq!(
            call(&state, plain("GET", "/config", false)).await.0,
            StatusCode::UNAUTHORIZED,
            "{name}"
        );
        let (s, h, _) = call(&state, plain("GET", "/config", true)).await;
        assert_eq!(s, StatusCode::SEE_OTHER, "{name}");
        assert_eq!(location(&h), "/login?next=%2Fconfig");
        // The login page is for it: it does not count as this machine.
        let (s, _, b) = call(&state, plain("GET", "/login?next=/config", true)).await;
        assert_eq!(s, StatusCode::OK, "{name}");
        assert!(text(&b).contains("type=\"password\""));
        let r = bearer(plain("GET", "/config", false), &token_a());
        assert_eq!(call(&state, r).await.0, StatusCode::OK, "{name}");
        let r = cookie(plain("GET", "/config", false), &token_a());
        assert_eq!(call(&state, r).await.0, StatusCode::OK, "{name}");
        let (_, _, b) = call(&state, plain("GET", "/v1/config", false)).await;
        assert!(!has_config(&b), "{name}");
        let r = bearer(plain("GET", "/v1/config", false), &token_a());
        let (_, _, b) = call(&state, r).await;
        assert!(has_config(&b), "{name}");
        // The card is the remote one: no file, no Rotate.
        let path = token_path(&state).display().to_string();
        let r = bearer(plain("GET", "/config", true), &token_a());
        let (_, _, b) = call(&state, r).await;
        assert!(!text(&b).contains(&path) && text(&b).contains("action=\"/logout\""));
        // Rotation stays refused, with the token too.
        let r = bearer(
            plain("POST", "/config/admin-token/rotate", false),
            &token_a(),
        );
        assert_eq!(call(&state, r).await.0, StatusCode::FORBIDDEN, "{name}");
        // Wrong tokens count, under the proxy's address: that is all there is to go by.
        for _ in 0..MAX_FAILURES {
            let r = bearer(plain("GET", "/config", false), &token_b());
            assert_eq!(call(&state, r).await.0, StatusCode::UNAUTHORIZED);
        }
        let r = bearer(plain("GET", "/config", false), &token_a());
        assert_eq!(call(&state, r).await.0, StatusCode::TOO_MANY_REQUESTS);
        // Nothing of this touches a person on this machine.
        assert_eq!(
            call(&state, req(LOCAL, "GET", "/config", false)).await.0,
            StatusCode::OK,
            "{name}"
        );
    }
}

/// The header only ever takes trust away: its value is not an address anyone is believed about.
#[tokio::test]
async fn a_forwarding_header_never_makes_a_remote_peer_local() {
    let state = remote_state();
    let r = forwarded(
        req(REMOTE, "GET", "/config", false),
        "x-forwarded-for",
        "127.0.0.1",
    );
    assert_eq!(call(&state, r).await.0, StatusCode::UNAUTHORIZED);
    let off = test_state();
    let r = forwarded(
        req(REMOTE, "GET", "/config", false),
        "x-real-ip",
        "127.0.0.1",
    );
    assert_eq!(call(&off, r).await.0, StatusCode::FORBIDDEN);
}
