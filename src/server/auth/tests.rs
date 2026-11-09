//! Sign-in at the router: sessions and the API token, the login, logout and setup pages, the
//! limits (per address, per account, the hashing permits) and what every refusal says.

use super::super::test_util::*;
use super::*;
use crate::accounts::{Store, User};
use crate::server::{AppState, router};
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode, header};
use rustls::pki_types::pem::PemObject;
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
    crate::admin_token::prepare(&state.config_read()).unwrap()
}

fn root(state: &AppState) -> PathBuf {
    state.running().data_root()
}

/// A test state whose API token file holds [`token_a`].
fn token_state() -> Arc<AppState> {
    let state = test_state();
    std::fs::write(token_path(&state), format!("{}\n", token_a())).unwrap();
    state
}

/// `method uri` from `peer`, signed in as the test user, as a browser or not; POSTs have an
/// empty form body.
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

/// [`req`] with no credential.
fn anon(peer: &str, method: &str, uri: &str, html: bool) -> Request<Body> {
    anonymous(req(peer, method, uri, html))
}

/// `POST uri` from `peer` with this urlencoded body and no session.
fn form_post(peer: &str, uri: &str, body: String, html: bool) -> Request<Body> {
    let mut b =
        Request::post(uri).header(header::CONTENT_TYPE, "application/x-www-form-urlencoded");
    if html {
        b = b.header(header::ACCEPT, "text/html");
    }
    anonymous(from_peer(b.body(Body::from(body)).unwrap(), peer))
}

/// `POST /login` from `peer` with this urlencoded body.
fn login(peer: &str, body: String, html: bool) -> Request<Body> {
    form_post(peer, "/login", body, html)
}

/// The login body for `user` / `password`.
fn creds(user: &str, password: &str) -> String {
    format!(
        "username={}&password={}",
        urlencode(user),
        urlencode(password)
    )
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn location(h: &HeaderMap) -> &str {
    h[header::LOCATION].to_str().unwrap()
}

fn set_cookies(h: &HeaderMap) -> Vec<String> {
    h.get_all(header::SET_COOKIE)
        .iter()
        .map(|v| v.to_str().unwrap().to_string())
        .collect()
}

/// The value a `Set-Cookie` of `name` sets (not a clearing one).
fn cookie_set(h: &HeaderMap, name: &str) -> Option<String> {
    set_cookies(h).into_iter().find_map(|c| {
        let v = c
            .strip_prefix(&format!("{name}="))?
            .split(';')
            .next()?
            .to_string();
        (!v.is_empty()).then_some(v)
    })
}

fn with_header(mut r: Request<Body>, name: &str, value: &str) -> Request<Body> {
    r.headers_mut().insert(
        header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
        value.parse().unwrap(),
    );
    r
}

/// Change the test user in `users.json` (through the store, under its lock).
fn edit_user(state: &AppState, f: impl FnOnce(&mut User)) {
    Store::open(&root(state))
        .locked(|_, cur| {
            let mut users = cur.unwrap();
            f(&mut users.users[0]);
            Ok((Some(users), ()))
        })
        .unwrap();
    state.auth.users.invalidate();
}

/// Wait (10 s at most) until every sign-in result of `state` is written (`record_later`).
async fn recorded(state: &AppState) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while state.auth.attempts.total() > 0 {
        assert!(Instant::now() < deadline, "sign-in results not written");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// The recorded failures of the test user.
fn failures(state: &AppState) -> u32 {
    Store::open(&root(state)).load().unwrap().unwrap().users[0].failures
}

/// The hashes run so far for `state`.
fn hashed(state: &AppState) -> u64 {
    state
        .login
        .hashed
        .load(std::sync::atomic::Ordering::Relaxed)
}

// ---------------------------------------------------------------------------------------------
// The per-address limit
// ---------------------------------------------------------------------------------------------

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

/// A reservation counts at once (so concurrent attempts see it) and a refund takes exactly that
/// one back.
#[test]
fn reserve_counts_before_the_work_and_refund_gives_it_back() {
    let mut l = FailureLimit::default();
    let ip: IpAddr = "10.0.0.5".parse().unwrap();
    let t0 = Instant::now();
    let mut held = Vec::new();
    for i in 0..MAX_FAILURES {
        held.push(l.reserve(ip, t0 + Duration::from_millis(i as u64)).unwrap());
    }
    assert_eq!(l.reserve(ip, t0), Err(60));
    assert_eq!(l.count(ip), MAX_FAILURES);
    // One succeeded: its slot comes back, the others still count.
    l.refund(held.pop().unwrap());
    assert_eq!(l.count(ip), MAX_FAILURES - 1);
    let again = l.reserve(ip, t0).unwrap();
    assert!(l.reserve(ip, t0).is_err());
    for r in held.drain(..).chain([again]) {
        l.refund(r);
    }
    assert_eq!(l.count(ip), 0);
    assert_eq!(l.peers(), 0);
    // A refund after the entry aged out is harmless.
    let r = l.reserve(ip, t0).unwrap();
    assert!(l.blocked(ip, t0 + WINDOW).is_none());
    l.refund(r);
    assert_eq!(l.count(ip), 0);
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

/// One IPv6 host can use any address of its /64, so the limit counts the /64; IPv4 counts the
/// address, also when it arrives mapped into IPv6.
#[test]
fn the_limit_keys_ipv6_peers_by_their_64() {
    let key = |s: &str| peer_key(s.parse().unwrap());
    assert_eq!(
        key("2001:db8:1:2:aaaa::1"),
        key("2001:db8:1:2:ffff:ffff:ffff:ffff")
    );
    assert_eq!(
        key("2001:db8:1:2:aaaa::1"),
        "2001:db8:1:2::".parse::<IpAddr>().unwrap()
    );
    assert_ne!(key("2001:db8:1:2::1"), key("2001:db8:1:3::1"));
    assert_eq!(key("fe80::1"), key("fe80::2"));
    assert_eq!(key("10.0.0.5"), "10.0.0.5".parse::<IpAddr>().unwrap());
    assert_eq!(key("::ffff:10.0.0.5"), key("10.0.0.5"));
    assert_ne!(key("10.0.0.5"), key("10.0.0.6"));

    // Two tiers (rewritten: one IPv6 host no longer blocks its whole /64). One address at its
    // limit does not block another address of the same /64 ...
    let mut l = FailureLimit::default();
    let t0 = Instant::now();
    let one: IpAddr = "fd00:0:0:7::1".parse().unwrap();
    for _ in 0..MAX_FAILURES {
        assert!(l.reserve(one, t0).is_ok());
    }
    assert!(l.reserve(one, t0).is_err(), "the address is at its limit");
    assert_eq!(l.blocked("fd00:0:0:7:beef::9".parse().unwrap(), t0), None);
    let mut l = FailureLimit::default();
    for _ in 0..MAX_FAILURES {
        assert!(l.reserve("fe80::1".parse().unwrap(), t0).is_ok());
    }
    assert_eq!(
        l.blocked("fe80::2".parse().unwrap(), t0),
        None,
        "link-local neighbours"
    );
    // ... but rotating through addresses of one /64 meets the /64 cap.
    let mut l = FailureLimit::default();
    for i in 0..NET_MAX_FAILURES as u16 {
        let ip = IpAddr::V6(std::net::Ipv6Addr::new(0xfd00, 0, 0, 7, i, 0, 0, 1));
        assert!(l.reserve(ip, t0).is_ok(), "address {i}");
    }
    assert_eq!(l.count(fresh_in(7)), 0, "never seen by itself");
    assert!(l.reserve(fresh_in(7), t0).is_err(), "the /64 cap");
    assert_eq!(l.peers(), NET_MAX_FAILURES);
    // The next /64 is another network; a refund gives back both tiers.
    assert!(l.reserve("fd00:0:0:8::1".parse().unwrap(), t0).is_ok());
    let mut l = FailureLimit::default();
    let mut held = Vec::new();
    for i in 0..NET_MAX_FAILURES as u16 {
        let ip = IpAddr::V6(std::net::Ipv6Addr::new(0xfd00, 0, 0, 9, i, 0, 0, 1));
        held.push(l.reserve(ip, t0).unwrap());
    }
    assert!(l.reserve(fresh_in(9), t0).is_err());
    l.refund(held.pop().unwrap());
    assert!(
        l.reserve(fresh_in(9), t0).is_ok(),
        "the refund freed the /64 slot"
    );
    // IPv4 neighbours are not lumped together.
    let mut l = FailureLimit::default();
    for _ in 0..5 {
        l.fail("10.0.0.5".parse().unwrap(), t0);
    }
    assert!(l.blocked("10.0.0.5".parse().unwrap(), t0).is_some());
    assert_eq!(l.blocked("10.0.0.6".parse().unwrap(), t0), None);
}

fn fresh_in(net: u16) -> IpAddr {
    IpAddr::V6(std::net::Ipv6Addr::new(0xfd00, 0, 0, net, 0xbeef, 0, 0, 9))
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
    let t = t0 + Duration::from_secs(2);
    assert_eq!(l.blocked(peer(0), t), None);
    for _ in 0..4 {
        l.fail(peer(MAX_PEERS as u32), t);
    }
    assert!(l.blocked(peer(MAX_PEERS as u32), t).is_some());
    l.fail(peer(5000), t0 + Duration::from_secs(120));
    assert_eq!(l.peers(), 1);
}

// ---------------------------------------------------------------------------------------------
// Who is signed in
// ---------------------------------------------------------------------------------------------

/// Rewritten from "a remote peer without the token is refused": now nobody without a session or
/// the token gets in, this machine included. A browser goes to the login page (and back for a
/// GET), anything else gets a JSON 401 with `WWW-Authenticate: Bearer`.
#[tokio::test]
async fn anonymous_requests_go_to_login_or_get_401() {
    let state = test_state();
    for peer in [REMOTE, LOCAL] {
        for (method, uri) in [
            ("GET", "/config"),
            ("POST", "/config"),
            ("GET", "/logs.json"),
            ("POST", "/config/restart"),
            ("GET", "/dataset/image?path=x.jpg"),
        ] {
            let (s, h, b) = call(&state, anon(peer, method, uri, false)).await;
            assert_eq!(s, StatusCode::UNAUTHORIZED, "{method} {uri}: {}", text(&b));
            assert_eq!(h[header::WWW_AUTHENTICATE], "Bearer");
            let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
            assert_eq!(j["success"], false);
            assert_eq!(j["error"], j["message"]);
            assert!(j["message"].as_str().unwrap().contains("/login"), "{j}");
        }
    }
    assert!(!state.restart.is_cancelled());
    let (s, h, _) = call(&state, anon(REMOTE, "GET", "/config?x=1", true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/login?next=%2Fconfig%3Fx%3D1");
    assert_eq!(h[header::CACHE_CONTROL], "no-store");
    // A POST cannot be repeated by a redirect, and its path may not take GET at all.
    let (s, h, _) = call(&state, anon(REMOTE, "POST", "/config/restart", true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/login");
    // Signed in, the same requests pass.
    let (s, _, _) = call(&state, req(REMOTE, "GET", "/config", true)).await;
    assert_eq!(s, StatusCode::OK);
}

/// No `users.json`: browsers go to `/setup`, others get a 401 naming it; the login page too.
#[tokio::test]
async fn with_no_accounts_everyone_is_sent_to_setup() {
    let state = test_state();
    std::fs::remove_file(Store::open(&root(&state)).path()).unwrap();
    let (s, h, _) = call(&state, req(REMOTE, "GET", "/config", true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER, "the old session is no account");
    assert_eq!(location(&h), "/setup");
    let (s, h, b) = call(&state, anon(LOCAL, "GET", "/v1/training", false)).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert_eq!(h[header::WWW_AUTHENTICATE], "Bearer");
    assert!(text(&b).contains("setup-code"), "{}", text(&b));
    let (s, h, _) = call(&state, anon(REMOTE, "GET", "/login", true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/setup");
    let (s, h, _) = call(&state, login(REMOTE, creds("tester", TEST_PASSWORD), true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/setup");
    // The API token still works meanwhile.
    std::fs::write(token_path(&state), token_a()).unwrap();
    let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::OK);
}

/// A damaged, unreadable or newer `users.json` closes sign-in with 503 and never opens setup.
#[tokio::test]
async fn a_broken_accounts_file_closes_sign_in_and_never_opens_setup() {
    for damage in [
        "{ not json",
        r#"{"format": 99, "users": []}"#,
        r#"{"format": 1, "users": [{"name": "BAD NAME"}]}"#,
    ] {
        let state = test_state();
        let path = Store::open(&root(&state)).path();
        std::fs::write(&path, damage).unwrap();
        for r in [
            req(REMOTE, "GET", "/config", true),
            anon(REMOTE, "GET", "/config", false),
            anon(REMOTE, "GET", "/login", true),
            anon(REMOTE, "GET", "/setup", true),
            login(REMOTE, creds("tester", TEST_PASSWORD), true),
            form_post(
                REMOTE,
                "/setup",
                "code=x&username=a&password=b&password2=b".into(),
                true,
            ),
        ] {
            let what = format!("{} {}", r.method(), r.uri());
            let (s, h, b) = call(&state, r).await;
            assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{damage}: {what}");
            assert!(!h.contains_key(header::LOCATION), "{what}");
            assert!(text(&b).contains("Nobody can sign in"), "{}", text(&b));
            assert!(
                !text(&b).contains(&path.display().to_string()),
                "the path stays in the log"
            );
        }
        assert!(
            crate::accounts::setup::read_code(&root(&state))
                .unwrap()
                .is_none(),
            "no setup code is made"
        );
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            damage,
            "left as it is"
        );
    }
}

/// The seeded session passes; a session past its idle or absolute limit, of a removed account or
/// of an older generation (a password reset) sends the browser to the login page and is cleared.
#[tokio::test]
async fn sessions_pass_until_they_expire_or_their_account_changes() {
    let state = test_state();
    let now = crate::accounts::now();
    let idle = "11".repeat(32);
    let old = "22".repeat(32);
    let bumped = "33".repeat(32);
    state.auth.sessions.insert_known(
        &idle,
        TEST_USER,
        1,
        now - crate::accounts::sessions::IDLE_SECS - 5,
    );
    state.auth.sessions.insert_known(
        &old,
        TEST_USER,
        1,
        now - crate::accounts::sessions::ABSOLUTE_SECS - 5,
    );
    state.auth.sessions.insert_known(&bumped, TEST_USER, 1, now);
    assert_eq!(
        call(&state, req(REMOTE, "GET", "/config", true)).await.0,
        StatusCode::OK
    );
    for id in [&idle, &old] {
        let (s, h, _) = call(
            &state,
            with_session(req(REMOTE, "GET", "/config", true), id),
        )
        .await;
        assert_eq!(s, StatusCode::SEE_OTHER);
        assert_eq!(location(&h), "/login?next=%2Fconfig");
        assert!(
            set_cookies(&h)
                .iter()
                .any(|c| c.starts_with("__Host-bop_session=;") && c.contains("Max-Age=0")),
            "{:?}",
            set_cookies(&h)
        );
    }
    // A password reset by the CLI bumps the generation: every session of the account ends.
    edit_user(&state, |u| u.generation += 1);
    for id in [&bumped, &TEST_SESSION.to_string()] {
        let (s, _, _) = call(
            &state,
            with_session(req(REMOTE, "GET", "/config", false), id),
        )
        .await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
    }
    // A removed account's: a fresh state, the user renamed away.
    let state = test_state();
    edit_user(&state, |u| u.name = "someone-else".into());
    let (s, _, _) = call(&state, req(REMOTE, "GET", "/config", false)).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    // Ids of another shape, and planted cookies of similar names, are nobody.
    let state = test_state();
    for cookie in [
        "__Host-bop_session=abc".to_string(),
        format!("bop_session={TEST_SESSION}"),
        format!("__host-bop_session={TEST_SESSION}"),
    ] {
        let mut r = anon(REMOTE, "GET", "/config", false);
        r.headers_mut()
            .insert(header::COOKIE, cookie.parse().unwrap());
        assert_eq!(
            call(&state, r).await.0,
            StatusCode::UNAUTHORIZED,
            "{cookie}"
        );
    }
}

/// `X-Prism-Poll: 1` (a page polling on its own) does not move the idle clock; anything else does.
#[tokio::test]
async fn polls_do_not_extend_the_idle_clock() {
    let state = test_state();
    let now = crate::accounts::now();
    let id = "44".repeat(32);
    let at = now - 1000;
    state.auth.sessions.insert_known(&id, TEST_USER, 1, at);
    let sid = crate::accounts::sessions::SessionId::from_presented(id.as_bytes()).unwrap();
    for uri in ["/v1/train/summary", "/v1/config", "/logs.json"] {
        let r = with_header(
            with_session(req(REMOTE, "GET", uri, false), &id),
            "x-prism-poll",
            "1",
        );
        let (s, _, b) = call(&state, r).await;
        assert!(s.is_success(), "{uri}: {s} {}", text(&b));
        assert_eq!(
            state.auth.sessions.record(&sid).unwrap().last_active,
            at,
            "{uri}"
        );
    }
    // The full config answer was given to the poll: it is signed in.
    let r = with_header(
        with_session(req(REMOTE, "GET", "/v1/config", false), &id),
        "x-prism-poll",
        "1",
    );
    let (_, _, b) = call(&state, r).await;
    assert!(serde_json::from_slice::<serde_json::Value>(&b).unwrap()["config"].is_object());
    let (s, _, _) = call(
        &state,
        with_session(req(REMOTE, "GET", "/logs.json", false), &id),
    )
    .await;
    assert!(s.is_success());
    assert!(state.auth.sessions.record(&sid).unwrap().last_active >= now);
}

// ---------------------------------------------------------------------------------------------
// The API token
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn the_api_token_opens_the_admin_routes_over_https() {
    let state = token_state();
    for (method, uri) in [
        ("GET", "/config"),
        ("GET", "/logs.json"),
        ("GET", "/prometheus"),
        ("POST", "/v1/benchmark/cancel"),
    ] {
        let (s, _, b) = call(&state, bearer(req(REMOTE, method, uri, false), &token_a())).await;
        assert!(
            ![401, 403, 429].contains(&s.as_u16()),
            "{method} {uri}: {}",
            text(&b)
        );
    }
    // The scheme is case-insensitive (RFC 9110).
    let mut r = anon(REMOTE, "GET", "/config", false);
    r.headers_mut().insert(
        header::AUTHORIZATION,
        format!("bearer {}", token_a()).parse().unwrap(),
    );
    assert_eq!(call(&state, r).await.0, StatusCode::OK);
    // Valid traffic never fills the failure list, however much of it there is.
    for _ in 0..(MAX_FAILURES * 3) {
        let r = bearer(req(REMOTE, "GET", "/logs.json", false), &token_a());
        assert_eq!(call(&state, r).await.0, StatusCode::OK);
    }
    // A wrong token is refused, never repeated; other schemes and non-text are no token.
    let (s, h, b) = call(
        &state,
        bearer(req(REMOTE, "GET", "/config", false), &token_b()),
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert!(!text(&b).contains(&token_b()) && !text(&b).contains(&token_a()));
    assert!(!format!("{h:?}").contains(&token_a()));
    let mut r = anon(REMOTE, "GET", "/config", false);
    r.headers_mut().insert(
        header::AUTHORIZATION,
        format!("Basic {}", token_a()).parse().unwrap(),
    );
    assert_eq!(call(&state, r).await.0, StatusCode::UNAUTHORIZED);
    let mut r = anon(REMOTE, "GET", "/config", false);
    r.headers_mut().insert(
        header::AUTHORIZATION,
        header::HeaderValue::from_bytes(b"Bearer \xff\xfe").unwrap(),
    );
    assert_eq!(call(&state, r).await.0, StatusCode::UNAUTHORIZED);
}

/// Six wrong tokens: the sixth is 429 without being compared, and a minute later (aged by the
/// test) the right one works again. The token's bucket is its own: a person's password sign-in
/// from the same address is not blocked by it.
#[tokio::test]
async fn wrong_tokens_count_in_their_own_bucket() {
    let state = token_state();
    for i in 0..MAX_FAILURES {
        let r = bearer(req(REMOTE, "GET", "/config", false), &token_b());
        assert_eq!(
            call(&state, r).await.0,
            StatusCode::UNAUTHORIZED,
            "failure {i}"
        );
    }
    let (s, h, b) = call(
        &state,
        bearer(req(REMOTE, "GET", "/config", false), &token_b()),
    )
    .await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    let wait: u64 = h[header::RETRY_AFTER].to_str().unwrap().parse().unwrap();
    assert!((1..=60).contains(&wait), "{wait}");
    let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
    assert_eq!(j["error"], j["message"]);
    let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::TOO_MANY_REQUESTS);
    // The session and a password sign-in from the same address are not affected.
    assert_eq!(
        call(&state, req(REMOTE, "GET", "/config", false)).await.0,
        StatusCode::OK
    );
    let (s, h, _) = call(&state, login(REMOTE, creds(TEST_USER, TEST_PASSWORD), true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert!(cookie_set(&h, crate::accounts::cookies::SESSION).is_some());
    // Another peer is not affected.
    let r = bearer(req("10.0.0.6:50000", "GET", "/config", false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::OK);
    state.admin_failures.lock().unwrap().age(WINDOW);
    let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::OK);
}

#[tokio::test]
async fn rotating_invalidates_the_old_token_at_once() {
    let state = token_state();
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

/// The token file missing, empty, malformed or unreadable: nobody passes with a token, the
/// answer names no path, and accounts still sign in.
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
        let state = token_state();
        let path = token_path(&state);
        damage(&path);
        for attempt in [token_a(), "ab".to_string()] {
            let (s, _, b) = call(
                &state,
                bearer(req(REMOTE, "GET", "/config", false), &attempt),
            )
            .await;
            assert_eq!(s, StatusCode::UNAUTHORIZED, "{name} with {attempt:?}");
            assert!(
                !text(&b).contains(&path.display().to_string()),
                "{}",
                text(&b)
            );
            let (_, _, b) = call(
                &state,
                bearer(req(REMOTE, "GET", "/v1/config", false), &attempt),
            )
            .await;
            assert!(!has_config(&b), "{name}");
        }
        // Nothing at all after `Bearer` is no empty token an empty file could match.
        let mut r = anon(REMOTE, "GET", "/config", false);
        r.headers_mut()
            .insert(header::AUTHORIZATION, "Bearer ".parse().unwrap());
        assert_eq!(call(&state, r).await.0, StatusCode::UNAUTHORIZED, "{name}");
        if !path.is_dir() {
            std::fs::write(&path, token_a()).unwrap();
            let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
            assert_eq!(call(&state, r).await.0, StatusCode::OK, "{name}");
        }
        assert_eq!(
            call(&state, req(REMOTE, "GET", "/config", false)).await.0,
            StatusCode::OK
        );
    }
}

fn has_config(b: &[u8]) -> bool {
    let j: serde_json::Value = serde_json::from_slice(b).unwrap();
    j.get("config").is_some() || j.get("path").is_some()
}

/// Rewritten from "the full config payload needs the token": it needs a sign-in now (a session
/// or the token); a wrong token gets the minimal answer and counts.
#[tokio::test]
async fn the_full_config_payload_needs_a_sign_in() {
    let state = token_state();
    for peer in [REMOTE, LOCAL] {
        let (s, _, b) = call(&state, anon(peer, "GET", "/v1/config", false)).await;
        assert_eq!(s, StatusCode::OK);
        assert!(!has_config(&b), "{peer}: {}", text(&b));
    }
    let (_, _, b) = call(&state, req(REMOTE, "GET", "/v1/config", false)).await;
    assert!(has_config(&b));
    let (_, _, b) = call(
        &state,
        bearer(req(REMOTE, "GET", "/v1/config", false), &token_a()),
    )
    .await;
    assert!(has_config(&b));
    for _ in 0..MAX_FAILURES {
        let (s, _, b) = call(
            &state,
            bearer(req(REMOTE, "GET", "/v1/config", false), &token_b()),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert!(!has_config(&b));
    }
    // Limited: still the minimal answer, and the right token no longer tells them apart.
    let (s, _, b) = call(
        &state,
        bearer(req(REMOTE, "GET", "/v1/config", false), &token_a()),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert!(!has_config(&b));
    // Neither credential lifts the `Host` check.
    state.admin_failures.lock().unwrap().age(WINDOW);
    let r = with_header(
        bearer(req(REMOTE, "GET", "/v1/config", false), &token_a()),
        "host",
        "rebind.attacker.test",
    );
    assert!(!has_config(&call(&state, r).await.2));
    let r = with_header(
        req(REMOTE, "GET", "/v1/config", false),
        "host",
        "rebind.attacker.test",
    );
    assert!(!has_config(&call(&state, r).await.2));
}

/// Open routes never look at a credential: no login needed, a wrong token is not counted.
#[tokio::test]
async fn open_routes_never_need_a_sign_in() {
    let state = token_state();
    for uri in [
        "/v1/status",
        "/v1/status/updateavailable",
        "/static/style.css",
        "/favicon.ico",
        "/v1/vision/custom/list",
    ] {
        let (s, _, _) = call(&state, anon(REMOTE, "GET", uri, false)).await;
        assert!(
            s != StatusCode::UNAUTHORIZED && s != StatusCode::FORBIDDEN,
            "{uri}: {s}"
        );
        for _ in 0..MAX_FAILURES {
            let r = bearer(req(REMOTE, "GET", uri, false), &token_b());
            assert!(call(&state, r).await.0 != StatusCode::TOO_MANY_REQUESTS);
        }
    }
    let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::OK);
}

// ---------------------------------------------------------------------------------------------
// /login and /logout
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn the_login_page_is_a_plain_form_with_no_inline_code() {
    let state = test_state();
    let (s, h, b) = call(
        &state,
        anon(REMOTE, "GET", "/login?next=%2Fconfig%3Fa%3D%22x", true),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert!(
        h[header::CONTENT_TYPE]
            .to_str()
            .unwrap()
            .starts_with("text/html")
    );
    assert!(h.contains_key("content-security-policy"));
    assert_eq!(h[header::CACHE_CONTROL], "no-store");
    let html = text(&b);
    for want in [
        "action=\"/login\"",
        "name=\"username\"",
        "autocomplete=\"username\"",
        "type=\"password\"",
        "autocomplete=\"current-password\"",
    ] {
        assert!(html.contains(want), "{want}: {html}");
    }
    assert!(
        !html.contains(" style=") && !html.contains("<script>"),
        "CSP: {html}"
    );
    assert!(!html.contains("onpaste"), "paste allowed");
    assert!(
        html.contains("&#34;x") || html.contains("&quot;x"),
        "escaped: {html}"
    );
    assert!(
        !html.contains("action=\"/logout\""),
        "no sign-out form for nobody"
    );
    // A browser that is signed in already goes on.
    let (s, h, _) = call(&state, req(REMOTE, "GET", "/login?next=/config", true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/config");
}

/// A good password: a new session id every time (never the one the browser sent, which ends),
/// both cookies with their exact attributes, and the way back.
#[tokio::test]
async fn signing_in_sets_both_cookies_and_a_new_session_every_time() {
    let state = test_state();
    let mut seen = std::collections::HashSet::new();
    for i in 0..3 {
        let body = format!(
            "{}&next=%2Ftrain%2Freview%3Fa%3D1",
            creds("TESTER", TEST_PASSWORD)
        );
        let mut r = login(REMOTE, body, true);
        if i > 0 {
            // A session cookie the browser already has (planted or old) is never reused.
            r.headers_mut().insert(
                header::COOKIE,
                format!("__Host-bop_session={TEST_SESSION}")
                    .parse()
                    .unwrap(),
            );
        }
        let (s, h, _) = call(&state, r).await;
        assert_eq!(s, StatusCode::SEE_OTHER);
        assert_eq!(location(&h), "/train/review?a=1");
        let cookies = set_cookies(&h);
        let session = cookie_set(&h, "__Host-bop_session").unwrap();
        let device = cookie_set(&h, "__Host-bop_device").unwrap();
        assert!(cookies.contains(&format!(
            "__Host-bop_session={session}; Secure; HttpOnly; SameSite=Strict; Path=/; Max-Age=43200"
        )), "{cookies:?}");
        assert!(cookies.contains(&format!(
            "__Host-bop_device={device}; Secure; HttpOnly; SameSite=Strict; Path=/; Max-Age=15552000"
        )), "{cookies:?}");
        assert_ne!(session, TEST_SESSION);
        assert!(seen.insert(session.clone()), "a new id each time");
        // It works.
        let (s, _, _) = call(
            &state,
            with_session(req(REMOTE, "GET", "/config", false), &session),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
    }
    // The pre-set session was ended by the sign-in that came with it.
    let (s, _, _) = call(&state, req(REMOTE, "GET", "/config", false)).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    // A JSON client gets JSON (and the cookies).
    let (s, h, b) = call(
        &state,
        login(REMOTE, creds(TEST_USER, TEST_PASSWORD), false),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&b).unwrap()["success"],
        true
    );
    assert!(cookie_set(&h, "__Host-bop_session").is_some());
    // Without a next: home.
    let (_, h, _) = call(&state, login(REMOTE, creds(TEST_USER, TEST_PASSWORD), true)).await;
    assert_eq!(location(&h), "/");
}

/// A wrong password and an unknown name get the same status and the same words; what was typed
/// is never repeated, a typed name is kept only when it has the shape of one.
#[tokio::test]
async fn wrong_passwords_and_unknown_names_answer_alike() {
    let state = test_state();
    let (s1, h1, b1) = call(
        &state,
        login(REMOTE, creds(TEST_USER, "not the password at all"), true),
    )
    .await;
    let (s2, h2, b2) = call(
        &state,
        login(REMOTE, creds("nobody", "not the password at all"), true),
    )
    .await;
    assert_eq!(s1, StatusCode::UNAUTHORIZED);
    assert_eq!(s1, s2);
    assert!(set_cookies(&h1).iter().all(|c| !c.starts_with("__Host-")));
    assert!(set_cookies(&h2).iter().all(|c| !c.starts_with("__Host-")));
    for (b, typed) in [(&b1, TEST_USER), (&b2, "nobody")] {
        let html = text(b);
        assert!(html.contains("Wrong username or password."), "{html}");
        assert!(!html.contains("not the password at all"));
        assert!(
            html.contains(&format!("value=\"{typed}\"")),
            "the name is kept: {html}"
        );
    }
    let (_, _, b) = call(
        &state,
        login(REMOTE, creds("<b>x</b> y", "pw pw pw pw pw pw"), true),
    )
    .await;
    assert!(
        !text(&b).contains("<b>x</b>") && !text(&b).contains("&lt;b&gt;"),
        "{}",
        text(&b)
    );
    // JSON clients: the same message.
    let (s, _, b) = call(&state, login(REMOTE, creds("nobody", "x"), false)).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
    assert_eq!(j["message"], "Wrong username or password.");
}

/// 20 wrong sign-ins at once from one address: five are checked (each reserved its slot before
/// hashing), the others answer 429 without hashing; the right password is refused as well.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_wrong_logins_cannot_pass_the_address_limit() {
    let state = test_state();
    let attempts = (0..20).map(|i| {
        let state = state.clone();
        async move {
            let r = login(
                REMOTE,
                creds(TEST_USER, &format!("wrong password number {i}")),
                false,
            );
            call(&state, r).await.0
        }
    });
    let statuses = futures_util::future::join_all(attempts).await;
    let wrong = statuses
        .iter()
        .filter(|s| **s == StatusCode::UNAUTHORIZED)
        .count();
    let limited = statuses
        .iter()
        .filter(|s| **s == StatusCode::TOO_MANY_REQUESTS)
        .count();
    assert_eq!(
        (wrong, limited),
        (MAX_FAILURES, 20 - MAX_FAILURES),
        "{statuses:?}"
    );
    assert_eq!(hashed(&state), MAX_FAILURES as u64);
    let (s, h, _) = call(&state, login(REMOTE, creds(TEST_USER, TEST_PASSWORD), true)).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert!(h.contains_key(header::RETRY_AFTER));
    assert_eq!(hashed(&state), MAX_FAILURES as u64, "not hashed");
    // Another address may; a minute later this one may too.
    let (s, _, _) = call(
        &state,
        login("10.0.0.6:1", creds(TEST_USER, TEST_PASSWORD), true),
    )
    .await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    state.login.failures.lock().unwrap().age(WINDOW);
    let (s, _, _) = call(&state, login(REMOTE, creds(TEST_USER, TEST_PASSWORD), true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
}

/// After 10 failures in a row the account waits (429 without hashing, for unknown names too);
/// a browser with the account's device cookie skips that wait, but neither the per-address limit
/// nor a disabled account; at 100 the account is disabled and answers like a wrong password.
#[tokio::test]
async fn account_lockout_device_cookies_and_disabling() {
    let state = test_state();
    // A device cookie from a good sign-in first.
    let (_, h, _) = call(
        &state,
        login("10.0.1.1:1", creds(TEST_USER, TEST_PASSWORD), true),
    )
    .await;
    let device = cookie_set(&h, "__Host-bop_device").unwrap();
    let now = crate::accounts::now();
    edit_user(&state, |u| {
        u.failures = crate::accounts::lockout::DELAY_AFTER + 5;
        u.last_failure = Some(now);
    });
    let before = hashed(&state);
    let (s, h, b) = call(
        &state,
        login("10.0.2.1:1", creds(TEST_USER, TEST_PASSWORD), true),
    )
    .await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS, "{}", text(&b));
    let wait: u64 = h[header::RETRY_AFTER].to_str().unwrap().parse().unwrap();
    assert_eq!(
        wait,
        crate::accounts::lockout::delay(crate::accounts::lockout::DELAY_AFTER + 5)
    );
    assert!(text(&b).contains("Too many attempts"));
    assert_eq!(hashed(&state), before, "refused without hashing");
    // The device cookie skips the account's delay.
    let r = with_header(
        login("10.0.3.1:1", creds(TEST_USER, TEST_PASSWORD), true),
        "cookie",
        &format!("__Host-bop_device={device}"),
    );
    let (s, h, _) = call(&state, r).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(
        cookie_set(&h, "__Host-bop_device").as_deref(),
        Some(device.as_str()),
        "renewed"
    );
    // The reset is written after the answer (`record_later`): wait for it.
    recorded(&state).await;
    let failures = Store::open(&root(&state)).load().unwrap().unwrap().users[0].failures;
    assert_eq!(failures, 0, "a success resets the count");
    // But not the per-address limit.
    for i in 0..MAX_FAILURES {
        let (s, _, _) = call(
            &state,
            login("10.0.4.1:1", creds("nobody", &format!("x{i}")), true),
        )
        .await;
        assert_eq!(s, StatusCode::UNAUTHORIZED);
    }
    let r = with_header(
        login("10.0.4.1:1", creds(TEST_USER, TEST_PASSWORD), true),
        "cookie",
        &format!("__Host-bop_device={device}"),
    );
    assert_eq!(call(&state, r).await.0, StatusCode::TOO_MANY_REQUESTS);
    // A wrong password counts on the account (and on the device's own budget).
    let (s, _, _) = call(
        &state,
        login("10.0.5.1:1", creds(TEST_USER, "nope nope nope"), true),
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    recorded(&state).await;
    let u = Store::open(&root(&state)).load().unwrap().unwrap().users[0].clone();
    assert_eq!(u.failures, 1);
    // Disabled at 100: the right password, even with the device cookie, reads like a wrong one.
    edit_user(&state, |u| {
        u.failures = crate::accounts::lockout::DISABLE_AT;
        u.disabled = true;
        u.last_failure = Some(now - 3600);
    });
    let before = hashed(&state);
    for cookie in [None, Some(format!("__Host-bop_device={device}"))] {
        let mut r = login("10.0.6.1:1", creds(TEST_USER, TEST_PASSWORD), true);
        if let Some(c) = &cookie {
            r = with_header(r, "cookie", c);
        }
        let (s, h, b) = call(&state, r).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "{cookie:?}");
        assert!(text(&b).contains("Wrong username or password."));
        assert!(cookie_set(&h, "__Host-bop_session").is_none());
    }
    assert_eq!(
        hashed(&state),
        before,
        "a disabled account is never checked"
    );
    // Unknown names wait the same way after 10 (five more counted here directly, so the delay
    // is 32 s and the next second cannot end it).
    let state = test_state();
    for i in 0..crate::accounts::lockout::DELAY_AFTER {
        let peer = format!("10.1.{i}.1:1");
        let (s, _, _) = call(
            &state,
            login(&peer, creds("ghost", "wrong wrong wrong"), true),
        )
        .await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "{i}");
    }
    let now = crate::accounts::now();
    {
        let mut unknown = state.auth.unknown.lock().unwrap();
        assert_eq!(
            unknown.len(),
            1,
            "the ten failures were counted under one name"
        );
        for _ in 0..5 {
            unknown.record_failure("ghost", now);
        }
    }
    let (s, h, _) = call(
        &state,
        login("10.2.0.1:1", creds("GHOST", "wrong wrong wrong"), true),
    )
    .await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert!(h.contains_key(header::RETRY_AFTER));
}

/// Every hashing permit busy for longer than the wait: 503 with `Retry-After`, nothing counted.
#[tokio::test]
async fn busy_hashing_answers_503() {
    let mut state = test_state();
    let sem: &'static crate::accounts::password::Semaphore =
        Box::leak(Box::new(crate::accounts::password::Semaphore::new(1)));
    Arc::get_mut(&mut state).unwrap().login = Arc::new(Logins {
        hashing: sem,
        hash_wait: Duration::from_millis(50),
        ..Logins::default()
    });
    let held = sem.acquire(Duration::from_secs(1)).unwrap();
    let (s, h, b) = call(&state, login(REMOTE, creds(TEST_USER, TEST_PASSWORD), true)).await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{}", text(&b));
    assert_eq!(h[header::RETRY_AFTER], "2");
    assert!(cookie_set(&h, "__Host-bop_session").is_none());
    assert_eq!(
        state
            .login
            .failures
            .lock()
            .unwrap()
            .count("10.0.0.5".parse().unwrap()),
        0
    );
    drop(held);
    let (s, _, _) = call(&state, login(REMOTE, creds(TEST_USER, TEST_PASSWORD), true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
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
    let state = test_state();
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
        let body = format!("{}&next={evil}", creds(TEST_USER, TEST_PASSWORD));
        let (s, h, _) = call(&state, login(REMOTE, body, true)).await;
        assert_eq!(s, StatusCode::SEE_OTHER, "{evil}");
        assert_eq!(location(&h), "/", "{evil}");
        let uri = format!("/login?next={evil}");
        let (s, _, b) = call(&state, anon(REMOTE, "GET", &uri, true)).await;
        assert_eq!(s, StatusCode::OK, "{evil}");
        assert!(!text(&b).contains("evil.test"), "{}", text(&b));
    }
}

/// Sign-out ends the session on the server: the old cookie stops working, also if it was kept.
#[tokio::test]
async fn logout_ends_the_session_on_the_server() {
    let state = test_state();
    let (_, h, _) = call(&state, login(REMOTE, creds(TEST_USER, TEST_PASSWORD), true)).await;
    let session = cookie_set(&h, "__Host-bop_session").unwrap();
    let (s, _, _) = call(
        &state,
        with_session(req(REMOTE, "GET", "/config", false), &session),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let (s, h, _) = call(
        &state,
        with_session(req(REMOTE, "POST", "/logout", true), &session),
    )
    .await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/login");
    let cookies = set_cookies(&h);
    assert!(
        cookies.contains(
            &"__Host-bop_session=; Secure; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"
                .to_string()
        ),
        "{cookies:?}"
    );
    assert!(
        cookies.iter().all(|c| !c.starts_with("__Host-bop_device")),
        "the device cookie stays"
    );
    let (s, _, _) = call(
        &state,
        with_session(req(REMOTE, "GET", "/config", false), &session),
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    // A sign-out with no session, or an unknown one, still answers.
    let (s, _, _) = call(&state, anon(REMOTE, "POST", "/logout", true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
}

/// `/login`, `/logout` and `/setup` pass the `Host` check and the CSRF guard: on HTTPS an
/// `http` origin and a same-site request are refused, and nothing is set.
#[tokio::test]
async fn login_logout_and_setup_go_through_the_host_and_csrf_checks() {
    let state = test_state();
    let rebound = "rebind.attacker.test:32169";
    for (method, uri) in [
        ("GET", "/login"),
        ("POST", "/login"),
        ("POST", "/logout"),
        ("GET", "/setup"),
        ("POST", "/setup"),
    ] {
        let (s, _, b) = call(
            &state,
            with_header(anon(REMOTE, method, uri, false), "host", rebound),
        )
        .await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{method} {uri}");
        assert!(text(&b).contains("allowed_hosts"), "{}", text(&b));
    }
    for uri in ["/login", "/logout", "/setup"] {
        for (name, value) in [
            ("sec-fetch-site", "cross-site"),
            ("sec-fetch-site", "same-site"),
            ("origin", "http://evil.test"),
            ("origin", "http://localhost:32168"),
            ("origin", "https://evil.test"),
        ] {
            let r = with_header(anon(REMOTE, "POST", uri, false), name, value);
            assert_eq!(
                call(&state, r).await.0,
                StatusCode::FORBIDDEN,
                "{uri} {name}: {value}"
            );
        }
    }
    let r = with_header(
        login(REMOTE, creds(TEST_USER, TEST_PASSWORD), true),
        "origin",
        "http://localhost:32168",
    );
    let (s, h, _) = call(&state, r).await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert!(cookie_set(&h, "__Host-bop_session").is_none());
    // The browser's own HTTPS origin passes.
    let r = with_header(
        login(REMOTE, creds(TEST_USER, TEST_PASSWORD), true),
        "origin",
        "https://localhost:32168",
    );
    assert_eq!(call(&state, r).await.0, StatusCode::SEE_OTHER);
    // A logout from another site does not sign anyone out.
    let r = with_header(
        req(REMOTE, "POST", "/logout", true),
        "sec-fetch-site",
        "cross-site",
    );
    assert_eq!(call(&state, r).await.0, StatusCode::FORBIDDEN);
    assert_eq!(
        call(&state, req(REMOTE, "GET", "/config", false)).await.0,
        StatusCode::OK
    );
}

// ---------------------------------------------------------------------------------------------
// /setup
// ---------------------------------------------------------------------------------------------

/// A state with no accounts and its setup code.
fn setup_state() -> (Arc<AppState>, String) {
    let state = test_state();
    std::fs::remove_file(Store::open(&root(&state)).path()).unwrap();
    let code = match crate::accounts::setup::ensure_code(&root(&state)).unwrap() {
        crate::accounts::setup::Ensured::Created(c)
        | crate::accounts::setup::Ensured::Existing(c) => c.reveal().to_string(),
        crate::accounts::setup::Ensured::AccountsExist => unreachable!(),
    };
    (state, code)
}

fn setup_body(code: &str, user: &str, pw: &str, pw2: &str) -> String {
    format!(
        "code={}&username={}&password={}&password2={}",
        urlencode(code),
        urlencode(user),
        urlencode(pw),
        urlencode(pw2)
    )
}

const NEW_PW: &str = "a fresh long passphrase here";

/// The setup page explains the code and the certificate check, and never shows a fingerprint
/// itself (one served by the page would prove nothing: a page in the middle shows its own).
#[tokio::test]
async fn the_setup_page_explains_and_shows_no_fingerprint() {
    let (state, code) = setup_state();
    let cert = crate::tls::cert_path(&state.running());
    let made = crate::tls::generate(
        &crate::tls::Identity::default(),
        std::time::SystemTime::now(),
    )
    .unwrap();
    std::fs::create_dir_all(cert.parent().unwrap()).unwrap();
    std::fs::write(&cert, &made.cert_pem).unwrap();
    let (s, h, b) = call(&state, anon(REMOTE, "GET", "/setup", true)).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(h[header::CACHE_CONTROL], "no-store");
    let html = text(&b);
    for want in [
        "action=\"/setup\"",
        "name=\"code\"",
        "name=\"password2\"",
        "autocomplete=\"new-password\"",
        "setup-code",
        "tls-fingerprint",
        "only works until",
    ] {
        assert!(html.contains(want), "{want}: {html}");
    }
    assert!(!html.contains(&code), "the code is never on the page");
    assert!(
        !html.contains(" style=") && !html.contains("<script>"),
        "CSP"
    );
    let der = rustls::pki_types::CertificateDer::from_pem_slice(made.cert_pem.as_bytes()).unwrap();
    let fp = crate::tls::fingerprint_sha256(&der);
    assert!(!html.contains(&fp[..20]), "no fingerprint on the page");
}

/// Setup creates the first account and signs the browser in; afterwards it is closed.
#[tokio::test]
async fn setup_creates_the_first_account_and_signs_in() {
    let (state, code) = setup_state();
    // Mismatched passwords: refused before anything, not counted.
    let (s, _, b) = call(
        &state,
        form_post(
            REMOTE,
            "/setup",
            setup_body(&code, "alice", NEW_PW, "something else entirely"),
            true,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(text(&b).contains("differ"));
    // A password the policy refuses: its message (the code was right).
    let (s, _, b) = call(
        &state,
        form_post(
            REMOTE,
            "/setup",
            setup_body(&code, "alice", "short", "short"),
            true,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(text(&b).contains("Not created"), "{}", text(&b));
    assert!(!text(&b).contains(&code));
    assert_eq!(
        state
            .login
            .failures
            .lock()
            .unwrap()
            .count("10.0.0.5".parse().unwrap()),
        0
    );
    let (s, h, _) = call(
        &state,
        form_post(
            REMOTE,
            "/setup",
            setup_body(&code.to_lowercase(), "Alice", NEW_PW, NEW_PW),
            true,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/");
    let session = cookie_set(&h, "__Host-bop_session").unwrap();
    assert!(cookie_set(&h, "__Host-bop_device").is_some());
    let users = Store::open(&root(&state)).load().unwrap().unwrap();
    assert_eq!(users.users.len(), 1);
    assert_eq!(users.users[0].name, "alice");
    assert!(
        crate::accounts::setup::read_code(&root(&state))
            .unwrap()
            .is_none(),
        "the code is gone"
    );
    let (s, _, _) = call(
        &state,
        with_session(req(REMOTE, "GET", "/config", false), &session),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    // Closed now: to the login page, also with the old code.
    let (s, h, _) = call(&state, anon(REMOTE, "GET", "/setup", true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/login");
    let (s, h, _) = call(
        &state,
        form_post(
            REMOTE,
            "/setup",
            setup_body(&code, "mallory", NEW_PW, NEW_PW),
            true,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/login");
    assert_eq!(
        Store::open(&root(&state))
            .load()
            .unwrap()
            .unwrap()
            .users
            .len(),
        1
    );
}

/// A wrong setup code counts against the address like a wrong password.
#[tokio::test]
async fn wrong_setup_codes_are_limited() {
    let (state, code) = setup_state();
    for i in 0..MAX_FAILURES {
        let (s, _, b) = call(
            &state,
            form_post(
                REMOTE,
                "/setup",
                setup_body(&format!("WRONG{i}"), "alice", NEW_PW, NEW_PW),
                true,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "{i}");
        assert!(text(&b).contains("not the setup code"));
    }
    let (s, h, _) = call(
        &state,
        form_post(
            REMOTE,
            "/setup",
            setup_body(&code, "alice", NEW_PW, NEW_PW),
            true,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert!(h.contains_key(header::RETRY_AFTER));
    assert!(Store::open(&root(&state)).load().unwrap().is_none());
    // The same failures block a password sign-in from there too: one bucket.
    let (s, _, _) = call(&state, login(REMOTE, creds("alice", NEW_PW), true)).await;
    assert_eq!(
        s,
        StatusCode::SEE_OTHER,
        "no accounts yet: to setup, nothing checked"
    );
}

/// Of concurrent setups with the right code, one creates an account.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_setups_create_one_account() {
    let (state, code) = setup_state();
    let tries = (0..5).map(|i| {
        let (state, code) = (state.clone(), code.clone());
        async move {
            let peer = format!("10.3.{i}.1:1");
            let r = form_post(
                &peer,
                "/setup",
                setup_body(&code, &format!("user{i}"), NEW_PW, NEW_PW),
                true,
            );
            let (_, h, _) = call(&state, r).await;
            cookie_set(&h, "__Host-bop_session").is_some()
        }
    });
    let signed_in = futures_util::future::join_all(tries).await;
    assert_eq!(signed_in.iter().filter(|b| **b).count(), 1, "{signed_in:?}");
    assert_eq!(
        Store::open(&root(&state))
            .load()
            .unwrap()
            .unwrap()
            .users
            .len(),
        1
    );
}

// ---------------------------------------------------------------------------------------------
// The API token's rotation and card
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn the_rotate_button_is_for_this_machine_and_never_shows_the_token() {
    const ROTATE: &str = "/config/admin-token/rotate";
    let state = token_state();
    let on_disk = |s: &AppState| {
        std::fs::read_to_string(token_path(s))
            .unwrap()
            .trim()
            .to_string()
    };
    // Not signed in: nobody, this machine included.
    assert_eq!(
        call(&state, anon(LOCAL, "POST", ROTATE, false)).await.0,
        StatusCode::UNAUTHORIZED
    );
    // Signed in from another machine, with the session or the token: refused.
    assert_eq!(
        call(&state, req(REMOTE, "POST", ROTATE, false)).await.0,
        StatusCode::FORBIDDEN
    );
    let r = bearer(req(REMOTE, "POST", ROTATE, false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::FORBIDDEN);
    // A local proxy's client is not this machine.
    let r = with_header(
        req(LOCAL, "POST", ROTATE, false),
        "x-forwarded-for",
        "192.168.1.9",
    );
    assert_eq!(call(&state, r).await.0, StatusCode::FORBIDDEN);
    // CSRF: the guard runs before the handler.
    let r = with_header(
        req(LOCAL, "POST", ROTATE, true),
        "origin",
        "http://evil.test",
    );
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

/// The API token card: the file and the command for everyone signed in, never the token; the
/// Rotate button for a browser on the server only; the header names the account and signs out.
#[tokio::test]
async fn the_config_page_shows_the_api_token_card_and_the_signed_in_user() {
    let state = token_state();
    let path = token_path(&state).display().to_string();
    for (peer, rotate) in [(LOCAL, true), (REMOTE, false)] {
        let (s, _, b) = call(&state, req(peer, "GET", "/config", true)).await;
        assert_eq!(s, StatusCode::OK);
        let html = text(&b);
        assert!(html.contains("API token"), "{peer}");
        assert!(html.contains(&path), "{peer}: the file's path");
        assert_eq!(
            html.contains("action=\"/config/admin-token/rotate#admin-token\""),
            rotate,
            "{peer}"
        );
        assert!(!html.contains(&token_a()));
        assert!(!html.contains("allow_remote_admin"), "the checkbox is gone");
        assert!(html.contains("action=\"/logout\""), "Sign out");
        assert!(
            html.contains(&format!("<span class=\"signed-in\">{TEST_USER}</span>")),
            "the name"
        );
    }
    // Over the token, no account name is shown.
    let (_, _, b) = call(
        &state,
        bearer(req(REMOTE, "GET", "/config", true), &token_a()),
    )
    .await;
    assert!(text(&b).contains("<span class=\"signed-in\"><!--bop:user--></span>"));
}

// ---------------------------------------------------------------------------------------------
// Peers
// ---------------------------------------------------------------------------------------------

/// A request with no peer address is refused by the network rule before any credential is
/// looked at, also with the right token.
#[tokio::test]
async fn a_request_without_a_peer_address_is_refused() {
    let state = token_state();
    for with_token in [false, true] {
        let mut r = Request::get("/config")
            .header(header::HOST, TEST_HOST)
            .body(Body::empty())
            .unwrap();
        if with_token {
            r.headers_mut().insert(
                header::AUTHORIZATION,
                format!("Bearer {}", token_a()).parse().unwrap(),
            );
        }
        let resp = router(state.clone()).oneshot(r).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let b = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(text(&b).contains("allowed_networks"), "{}", text(&b));
    }
}

/// Rewritten from the proxy tests: a forwarding header never signs anyone in or makes them this
/// machine; a proxied client signs in like any other.
#[tokio::test]
async fn forwarding_headers_never_stand_in_for_a_sign_in() {
    let state = token_state();
    for (name, value) in [
        ("x-forwarded-for", "127.0.0.1"),
        ("X-Forwarded-For", "192.168.1.9, 10.1.2.3"),
        ("forwarded", "for=192.168.1.9"),
        ("x-real-ip", "127.0.0.1"),
    ] {
        for peer in [LOCAL, REMOTE] {
            let r = with_header(anon(peer, "GET", "/config", false), name, value);
            assert_eq!(
                call(&state, r).await.0,
                StatusCode::UNAUTHORIZED,
                "{peer} {name}"
            );
            let r = with_header(anon(peer, "GET", "/config", true), name, value);
            assert_eq!(
                call(&state, r).await.0,
                StatusCode::SEE_OTHER,
                "{peer} {name}"
            );
            let r = with_header(req(peer, "GET", "/config", false), name, value);
            assert_eq!(call(&state, r).await.0, StatusCode::OK, "{peer} {name}");
            let r = with_header(req(peer, "GET", "/login", true), name, value);
            assert_eq!(
                call(&state, r).await.0,
                StatusCode::SEE_OTHER,
                "signed in already"
            );
        }
        // Wrong tokens count under the proxy's address: that is all there is to go by.
        let state = token_state();
        for _ in 0..MAX_FAILURES {
            let r = with_header(
                bearer(req(LOCAL, "GET", "/config", false), &token_b()),
                name,
                value,
            );
            assert_eq!(call(&state, r).await.0, StatusCode::UNAUTHORIZED);
        }
        let r = with_header(
            bearer(req(LOCAL, "GET", "/config", false), &token_a()),
            name,
            value,
        );
        assert_eq!(call(&state, r).await.0, StatusCode::TOO_MANY_REQUESTS);
    }
}

// ---------------------------------------------------------------------------------------------
// Review findings: the per-account race, the data root, timing, polls
// ---------------------------------------------------------------------------------------------

/// 50 wrong sign-ins for one account at once, each from its own address: decided as if one at a
/// time. Below the delay they run side by side; once failures plus attempts in flight reach 10,
/// at most one more runs per delay window, so a burst cannot reach `disabled`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_wrong_logins_for_one_account_are_judged_one_at_a_time() {
    let state = test_state();
    let t0 = Instant::now();
    let attempts = (0..50).map(|i| {
        let state = state.clone();
        async move {
            let peer = format!("10.6.{i}.1:1");
            let r = login(
                &peer,
                creds(TEST_USER, &format!("wrong password number {i}")),
                false,
            );
            call(&state, r).await.0
        }
    });
    let statuses = futures_util::future::join_all(attempts).await;
    recorded(&state).await;
    let elapsed = t0.elapsed().as_secs();
    let wrong = statuses
        .iter()
        .filter(|s| **s == StatusCode::UNAUTHORIZED)
        .count();
    let waited = statuses
        .iter()
        .filter(|s| **s == StatusCode::TOO_MANY_REQUESTS)
        .count();
    assert_eq!(wrong + waited, 50, "{statuses:?}");
    // Every checked attempt was hashed and recorded, nothing else.
    assert_eq!(hashed(&state), wrong as u64);
    assert_eq!(failures(&state) as usize, wrong);
    // Ten may run side by side; past them one per delay window (1 s, 2 s, 4 s, ...): with the
    // whole burst done in `elapsed` seconds, only a handful more.
    let extra_bound = 2 + elapsed as usize;
    assert!(
        (crate::accounts::lockout::DELAY_AFTER as usize..=10 + extra_bound).contains(&wrong),
        "{wrong} checked in {elapsed} s: {statuses:?}"
    );
    let u = Store::open(&root(&state)).load().unwrap().unwrap().users[0].clone();
    assert!(!u.disabled);
    assert!(u.failures < crate::accounts::lockout::DISABLE_AT);
}

/// Near the disabling count a burst cannot overshoot it: 99 recorded and 30 at once leave exactly
/// one checked, 100 recorded, and the account disabled by that hundredth decided attempt.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_burst_cannot_pass_the_disabling_count() {
    let state = test_state();
    let now = crate::accounts::now();
    edit_user(&state, |u| {
        u.failures = crate::accounts::lockout::DISABLE_AT - 1;
        u.last_failure = Some(now - 3600);
    });
    let attempts = (0..30).map(|i| {
        let state = state.clone();
        async move {
            let r = login(
                &format!("10.7.{i}.1:1"),
                creds(TEST_USER, "wrong wrong wrong"),
                false,
            );
            call(&state, r).await.0
        }
    });
    futures_util::future::join_all(attempts).await;
    recorded(&state).await;
    assert_eq!(hashed(&state), 1, "one attempt checked");
    let u = Store::open(&root(&state)).load().unwrap().unwrap().users[0].clone();
    assert_eq!(u.failures, crate::accounts::lockout::DISABLE_AT);
    assert!(u.disabled);
}

/// The accounts live in the data root the process started with (`Auth::data_root`): a running
/// config that names another one (a restart with a new `download_dir`) changes nothing for them.
#[tokio::test]
async fn sign_in_writes_to_the_data_root_it_started_with() {
    let state = test_state();
    let elsewhere = std::env::temp_dir().join(format!("bo_srv_other_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&elsewhere).unwrap();
    let mut moved = (*state.running()).clone();
    moved.download_dir = Some(elsewhere.clone());
    state.running.store(Arc::new(moved));
    assert_ne!(state.running().data_root(), state.auth.data_root());
    let (s, _, _) = call(
        &state,
        login(REMOTE, creds(TEST_USER, "not the password at all"), true),
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    recorded(&state).await;
    assert_eq!(
        Store::open(state.auth.data_root())
            .load()
            .unwrap()
            .unwrap()
            .users[0]
            .failures,
        1
    );
    assert!(
        !Store::open(&elsewhere).path().exists(),
        "nothing written elsewhere"
    );
    let (s, _, _) = call(&state, login(REMOTE, creds(TEST_USER, TEST_PASSWORD), true)).await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    recorded(&state).await;
    assert_eq!(
        Store::open(state.auth.data_root())
            .load()
            .unwrap()
            .unwrap()
            .users[0]
            .failures,
        0
    );
    // The API token too is the start root's.
    std::fs::write(
        crate::admin_token::path_in(state.auth.data_root()),
        token_a(),
    )
    .unwrap();
    let r = bearer(req(REMOTE, "GET", "/config", false), &token_a());
    assert_eq!(call(&state, r).await.0, StatusCode::OK);
    let _ = std::fs::remove_dir_all(&elsewhere);
}

/// A wrong password for an account is answered before its failure is written (the write is
/// slow here), as an unknown name is; the failure lands afterwards, and the attempt's slot is
/// held until it has.
#[tokio::test]
async fn a_wrong_password_is_answered_before_the_failure_is_written() {
    let mut state = test_state();
    Arc::get_mut(&mut state).unwrap().login = Arc::new(Logins {
        record_delay: Duration::from_millis(600),
        ..Logins::default()
    });
    let t = Instant::now();
    let (s, _, _) = call(
        &state,
        login(REMOTE, creds(TEST_USER, "not the password at all"), true),
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert!(
        t.elapsed() < Duration::from_millis(500),
        "waited {:?}",
        t.elapsed()
    );
    assert_eq!(failures(&state), 0, "not written yet");
    assert_eq!(
        state.auth.attempts.in_flight(TEST_USER),
        1,
        "its slot is held meanwhile"
    );
    recorded(&state).await;
    assert_eq!(failures(&state), 1);
}

/// A reload a poll makes carries `__Host-bop_poll`: it does not move the idle clock, and its
/// answer clears the cookie, so the next load counts again.
#[tokio::test]
async fn a_reload_made_by_a_poll_does_not_count_as_activity() {
    let state = test_state();
    let now = crate::accounts::now();
    let id = "55".repeat(32);
    let at = now - 1000;
    state.auth.sessions.insert_known(&id, TEST_USER, 1, at);
    let sid = crate::accounts::sessions::SessionId::from_presented(id.as_bytes()).unwrap();
    let r = with_header(
        anon(REMOTE, "GET", "/", true),
        "cookie",
        &format!("__Host-bop_session={id}; __Host-bop_poll=1"),
    );
    let (s, h, _) = call(&state, r).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(state.auth.sessions.record(&sid).unwrap().last_active, at);
    assert!(
        set_cookies(&h)
            .contains(&"__Host-bop_poll=; Secure; SameSite=Strict; Path=/; Max-Age=0".to_string()),
        "{:?}",
        set_cookies(&h)
    );
    let r = with_session(req(REMOTE, "GET", "/", true), &id);
    let (_, h, _) = call(&state, r).await;
    assert!(
        set_cookies(&h)
            .iter()
            .all(|c| !c.starts_with("__Host-bop_poll"))
    );
    assert!(state.auth.sessions.record(&sid).unwrap().last_active >= now);
}
