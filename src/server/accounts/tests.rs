//! The Accounts card's routes at the router: each change with the right and a wrong current
//! password, the policy, the last account, what ends which sessions and devices, the guards
//! (CSRF, no sign-in, the API token) and the card on the Config page.

use super::super::test_util::*;
use crate::accounts::{Store, User, password};
use crate::server::AppState;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode, header};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// The second account's name and password.
const BOB: &str = "bob";
const BOB_PASSWORD: &str = "another long passphrase here";
/// A new password that passes the policy for every account here.
const NEW_PASSWORD: &str = "a brand new pass phrase now";
/// A current password that is wrong.
const WRONG: &str = "not the password at all";

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

fn json(b: &[u8]) -> serde_json::Value {
    serde_json::from_slice(b).unwrap_or_else(|e| panic!("{e}: {}", text(b)))
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

/// Whether `h` clears the cookie `name`.
fn clears(h: &HeaderMap, name: &str) -> bool {
    set_cookies(h)
        .iter()
        .any(|c| c.starts_with(&format!("{name}=;")) && c.contains("Max-Age=0"))
}

fn with_header(mut r: Request<Body>, name: &str, value: &str) -> Request<Body> {
    r.headers_mut().insert(
        header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
        value.parse().unwrap(),
    );
    r
}

/// The urlencoded body of `pairs`.
fn body(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={}", urlencode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

fn root(state: &AppState) -> PathBuf {
    state.auth.data_root().to_path_buf()
}

fn store(state: &AppState) -> Store {
    Store::open(&root(state))
}

/// The account `name` as on disk.
fn account(state: &AppState, name: &str) -> Option<User> {
    store(state).load().unwrap().unwrap().get(name).cloned()
}

fn failures(state: &AppState, name: &str) -> u32 {
    account(state, name).unwrap().failures
}

fn hashed(state: &AppState) -> u64 {
    state.login.hashed.load(Ordering::Relaxed)
}

/// Add `name` with `pw` through the store (as the CLI would).
fn add_user(state: &AppState, name: &str, pw: &str) {
    let permit = password::acquire(Duration::from_secs(60)).unwrap();
    store(state)
        .add(name, pw, &permit, crate::accounts::now())
        .unwrap();
    state.auth.users.invalidate();
}

/// A new live session of `user` (at the account's generation): its id.
fn session_of(state: &AppState, user: &str) -> String {
    let generation = account(state, user).unwrap().generation;
    state
        .auth
        .sessions
        .create(user, generation, crate::accounts::now())
        .reveal()
        .to_string()
}

/// `GET /config` (JSON client) with session `id`: whether it is signed in.
async fn signed_in(state: &Arc<AppState>, id: &str) -> bool {
    let r = with_session(Request::get("/config").body(Body::empty()).unwrap(), id);
    let s = call(state, r).await.0;
    assert!(
        s == StatusCode::OK || s == StatusCode::UNAUTHORIZED,
        "unexpected {s}"
    );
    s == StatusCode::OK
}

/// `POST uri` as the test user (its seeded session), a browser or not.
fn post(uri: &str, pairs: &[(&str, &str)], html: bool) -> Request<Body> {
    post_form(uri, &body(pairs), html)
}

/// Every route with a body that would succeed for the test user, given a second account.
fn every_route() -> Vec<(&'static str, Vec<(&'static str, &'static str)>)> {
    vec![
        (
            "/config/accounts/add",
            vec![
                ("name", "carol"),
                ("password", NEW_PASSWORD),
                ("password2", NEW_PASSWORD),
                ("current", TEST_PASSWORD),
            ],
        ),
        (
            "/config/accounts/remove",
            vec![("name", BOB), ("current", TEST_PASSWORD)],
        ),
        (
            "/config/accounts/unlock",
            vec![("name", BOB), ("current", TEST_PASSWORD)],
        ),
        (
            "/config/accounts/password",
            vec![
                ("current", TEST_PASSWORD),
                ("password", NEW_PASSWORD),
                ("password2", NEW_PASSWORD),
            ],
        ),
        (
            "/config/accounts/sign-out-everywhere",
            vec![("current", TEST_PASSWORD)],
        ),
    ]
}

/// A test state with [`BOB`] too.
fn two_accounts() -> Arc<AppState> {
    let state = test_state();
    add_user(&state, BOB, BOB_PASSWORD);
    state
}

// ---------------------------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn adding_an_account_needs_the_current_password_and_the_policy() {
    let state = test_state();
    let add = |name: &'static str, pw: &'static str, pw2: &'static str, current: &'static str| {
        vec![
            ("name", name),
            ("password", pw),
            ("password2", pw2),
            ("current", current),
        ]
    };
    // Policy, mismatch and name problems: 400 / 409, before the current password is looked at
    // (no hash, nothing counted).
    let before = hashed(&state);
    for (pairs, status, words) in [
        (
            add(BOB, "short", "short", TEST_PASSWORD),
            StatusCode::BAD_REQUEST,
            "at least 15",
        ),
        (
            add(
                BOB,
                "bob is my long passphrase",
                "bob is my long passphrase",
                TEST_PASSWORD,
            ),
            StatusCode::BAD_REQUEST,
            "account's name",
        ),
        (
            add(BOB, NEW_PASSWORD, "something else entirely", TEST_PASSWORD),
            StatusCode::BAD_REQUEST,
            "differ",
        ),
        (
            add("Bad Name!", NEW_PASSWORD, NEW_PASSWORD, TEST_PASSWORD),
            StatusCode::BAD_REQUEST,
            "1 to 32 characters",
        ),
        (
            add(TEST_USER, NEW_PASSWORD, NEW_PASSWORD, TEST_PASSWORD),
            StatusCode::CONFLICT,
            "exists already",
        ),
    ] {
        let (s, _, b) = call(&state, post("/config/accounts/add", &pairs, false)).await;
        assert_eq!(s, status, "{pairs:?}: {}", text(&b));
        let j = json(&b);
        assert_eq!(j["success"], false);
        assert!(j["message"].as_str().unwrap().contains(words), "{j}");
        assert!(!text(&b).contains(NEW_PASSWORD), "never echoed");
    }
    assert_eq!(hashed(&state), before, "no current password was checked");
    assert_eq!(failures(&state, TEST_USER), 0);
    assert!(account(&state, BOB).is_none());

    // A wrong current password: 403 (not 401: the browser is signed in), counted against the
    // signed-in account, nothing added; the page again for browsers.
    let (s, h, b) = call(
        &state,
        post(
            "/config/accounts/add",
            &add(BOB, BOB_PASSWORD, BOB_PASSWORD, WRONG),
            true,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert!(!h.contains_key(header::WWW_AUTHENTICATE));
    let html = text(&b);
    assert!(html.contains("id=\"accounts\""), "the Config page");
    assert!(html.contains("Wrong password"), "{html}");
    assert!(!html.contains(WRONG) && !html.contains(BOB_PASSWORD));
    assert_eq!(failures(&state, TEST_USER), 1);
    assert!(account(&state, BOB).is_none());
    // JSON clients get JSON.
    let (s, _, b) = call(
        &state,
        post(
            "/config/accounts/add",
            &add(BOB, BOB_PASSWORD, BOB_PASSWORD, WRONG),
            false,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert_eq!(json(&b)["success"], false);
    assert_eq!(failures(&state, TEST_USER), 2);
    // An empty current password is no attempt.
    let (s, _, _) = call(
        &state,
        post(
            "/config/accounts/add",
            &add(BOB, BOB_PASSWORD, BOB_PASSWORD, ""),
            false,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(failures(&state, TEST_USER), 2);

    // The right one: added (the failures of the signed-in account are forgotten), and it signs in.
    let (s, _, b) = call(
        &state,
        post(
            "/config/accounts/add",
            &add("Bob", BOB_PASSWORD, BOB_PASSWORD, TEST_PASSWORD),
            true,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{}", text(&b));
    assert!(text(&b).contains("Added the account bob"));
    let bob = account(&state, BOB).unwrap();
    assert_ne!(bob.hash, "");
    assert_eq!(failures(&state, TEST_USER), 0);
    let r = anonymous(post_form(
        "/login",
        &body(&[("username", BOB), ("password", BOB_PASSWORD)]),
        true,
    ));
    let (s, h, _) = call(&state, r).await;
    assert_eq!(s, StatusCode::SEE_OTHER, "bob signs in");
    assert!(cookie_set(&h, "__Host-bop_session").is_some());
}

/// The per-address limit applies to the current password as to a sign-in: the sixth wrong one
/// in a minute is refused without a hash, with Retry-After.
#[tokio::test]
async fn wrong_current_passwords_meet_the_address_limit() {
    let state = two_accounts();
    let pairs = vec![("name", BOB), ("current", WRONG)];
    for _ in 0..crate::server::auth::MAX_FAILURES {
        let (s, _, _) = call(&state, post("/config/accounts/unlock", &pairs, false)).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
    }
    let before = hashed(&state);
    let (s, h, b) = call(&state, post("/config/accounts/unlock", &pairs, false)).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS, "{}", text(&b));
    assert!(h.contains_key(header::RETRY_AFTER));
    assert_eq!(hashed(&state), before, "no hash once limited");
    // Even the right password waits now.
    let ok = vec![("name", BOB), ("current", TEST_PASSWORD)];
    let (s, _, _) = call(&state, post("/config/accounts/unlock", &ok, false)).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
}

// ---------------------------------------------------------------------------------------------
// Remove
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn removing_an_account_ends_its_sessions_and_devices() {
    let state = two_accounts();
    let bob_session = session_of(&state, BOB);
    let device = state.auth.devices.issue(BOB, crate::accounts::now());
    assert!(signed_in(&state, &bob_session).await);

    // Wrong current password: 403, counted against the test user (never bob), bob stays.
    let (s, _, _) = call(
        &state,
        post(
            "/config/accounts/remove",
            &[("name", BOB), ("current", WRONG)],
            false,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert_eq!(failures(&state, TEST_USER), 1);
    assert_eq!(failures(&state, BOB), 0);
    assert!(account(&state, BOB).is_some());
    // An account that does not exist.
    let (s, _, _) = call(
        &state,
        post(
            "/config/accounts/remove",
            &[("name", "nobody"), ("current", TEST_PASSWORD)],
            false,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);

    let (s, _, b) = call(
        &state,
        post(
            "/config/accounts/remove",
            &[("name", BOB), ("current", TEST_PASSWORD)],
            false,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{}", text(&b));
    assert_eq!(json(&b)["success"], true);
    assert!(account(&state, BOB).is_none());
    assert!(
        !signed_in(&state, &bob_session).await,
        "bob's session ended"
    );
    assert!(
        state.auth.devices.device(&device).is_none(),
        "bob's device forgotten"
    );
    // A new account of the same name does not bring the old session back.
    add_user(&state, BOB, BOB_PASSWORD);
    assert!(!signed_in(&state, &bob_session).await);
    // The test user is still signed in.
    assert!(signed_in(&state, TEST_SESSION).await);
}

#[tokio::test]
async fn the_last_account_cannot_be_removed() {
    let state = test_state();
    let (s, _, b) = call(
        &state,
        post(
            "/config/accounts/remove",
            &[("name", TEST_USER), ("current", TEST_PASSWORD)],
            false,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::CONFLICT, "{}", text(&b));
    assert!(
        json(&b)["message"]
            .as_str()
            .unwrap()
            .contains("last account")
    );
    assert!(account(&state, TEST_USER).is_some());
    assert!(signed_in(&state, TEST_SESSION).await);
}

#[tokio::test]
async fn removing_your_own_account_signs_you_out() {
    let state = two_accounts();
    let (s, h, _) = call(
        &state,
        post(
            "/config/accounts/remove",
            &[("name", TEST_USER), ("current", TEST_PASSWORD)],
            true,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/login");
    assert!(clears(&h, "__Host-bop_session"));
    assert!(account(&state, TEST_USER).is_none());
    assert!(!signed_in(&state, TEST_SESSION).await);
    assert!(account(&state, BOB).is_some());
}

// ---------------------------------------------------------------------------------------------
// Unlock
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn unlocking_clears_failures_and_disabled() {
    let state = two_accounts();
    store(&state)
        .locked(|_, cur| {
            let mut users = cur.unwrap();
            let bob = users.users.iter_mut().find(|u| u.name == BOB).unwrap();
            bob.failures = 100;
            bob.last_failure = Some(crate::accounts::now());
            bob.disabled = true;
            Ok((Some(users), ()))
        })
        .unwrap();
    state.auth.users.invalidate();
    // The card shows it.
    let (_, html) = get_text(&state, "/config").await;
    assert!(html.contains("disabled after 100 in a row"), "{html}");
    let (s, _, _) = call(
        &state,
        post(
            "/config/accounts/unlock",
            &[("name", BOB), ("current", WRONG)],
            false,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert!(account(&state, BOB).unwrap().disabled);
    let (s, _, b) = call(
        &state,
        post(
            "/config/accounts/unlock",
            &[("name", BOB), ("current", TEST_PASSWORD)],
            true,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert!(text(&b).contains("Unlocked the account bob"));
    let bob = account(&state, BOB).unwrap();
    assert!(!bob.disabled);
    assert_eq!(bob.failures, 0);
    assert_eq!(bob.last_failure, None);
}

// ---------------------------------------------------------------------------------------------
// Change my password
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn changing_my_password_keeps_this_browser_with_a_new_session_id() {
    let state = two_accounts();
    // This browser: a sign-in with its device cookie; another browser of the same account; and
    // bob, who is not touched.
    let (s, h, _) = call(
        &state,
        anonymous(post_form(
            "/login",
            &body(&[("username", TEST_USER), ("password", TEST_PASSWORD)]),
            true,
        )),
    )
    .await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    let mine = cookie_set(&h, "__Host-bop_session").unwrap();
    let device = cookie_set(&h, "__Host-bop_device").unwrap();
    let other = session_of(&state, TEST_USER);
    let bob = session_of(&state, BOB);
    let generation = account(&state, TEST_USER).unwrap().generation;
    let browser = |r: Request<Body>| {
        with_header(
            r,
            "cookie",
            &format!("__Host-bop_session={mine}; __Host-bop_device={device}"),
        )
    };

    // Policy and mismatch: 400, nothing hashed or counted.
    let before = hashed(&state);
    for (pw, pw2, words) in [
        ("too short", "too short", "at least 15"),
        (NEW_PASSWORD, "a different long passphrase", "differ"),
        (
            "tester likes long passphrases",
            "tester likes long passphrases",
            "account's name",
        ),
    ] {
        let r = browser(post(
            "/config/accounts/password",
            &[
                ("current", TEST_PASSWORD),
                ("password", pw),
                ("password2", pw2),
            ],
            false,
        ));
        let (s, _, b) = call(&state, r).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{pw}");
        assert!(
            json(&b)["message"].as_str().unwrap().contains(words),
            "{}",
            text(&b)
        );
    }
    assert_eq!(hashed(&state), before);
    // Wrong current: 403, counted, the password stays.
    let r = browser(post(
        "/config/accounts/password",
        &[
            ("current", WRONG),
            ("password", NEW_PASSWORD),
            ("password2", NEW_PASSWORD),
        ],
        true,
    ));
    let (s, _, b) = call(&state, r).await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert!(text(&b).contains("Wrong password"));
    assert_eq!(failures(&state, TEST_USER), 1);
    assert_eq!(account(&state, TEST_USER).unwrap().generation, generation);

    // The right one.
    let r = browser(post(
        "/config/accounts/password",
        &[
            ("current", TEST_PASSWORD),
            ("password", NEW_PASSWORD),
            ("password2", NEW_PASSWORD),
        ],
        true,
    ));
    let (s, h, b) = call(&state, r).await;
    assert_eq!(s, StatusCode::OK, "{}", text(&b));
    assert!(text(&b).contains("Your password was changed"));
    assert_eq!(h[header::CACHE_CONTROL], "no-store");
    let renewed = cookie_set(&h, "__Host-bop_session").expect("a new session cookie");
    assert_ne!(renewed, mine, "a new session id");
    let cookie = set_cookies(&h)
        .into_iter()
        .find(|c| c.starts_with(&format!("__Host-bop_session={renewed}")))
        .unwrap();
    assert!(
        cookie.contains("Secure; HttpOnly; SameSite=Strict; Path=/; Max-Age="),
        "{cookie}"
    );
    let user = account(&state, TEST_USER).unwrap();
    assert_eq!(user.generation, generation + 1);
    assert_eq!(user.failures, 0);
    assert!(
        signed_in(&state, &renewed).await,
        "this browser stays signed in"
    );
    assert!(!signed_in(&state, &mine).await, "under a new id only");
    assert!(!signed_in(&state, &other).await, "the other browser is out");
    assert!(!signed_in(&state, TEST_SESSION).await);
    assert!(
        signed_in(&state, &bob).await,
        "another account is untouched"
    );
    assert!(
        state
            .auth
            .devices
            .trusted(device.as_bytes(), TEST_USER, crate::accounts::now()),
        "the device cookie stays"
    );
    // The new password signs in, the old one no longer.
    for (pw, want) in [
        (NEW_PASSWORD, StatusCode::SEE_OTHER),
        (TEST_PASSWORD, StatusCode::UNAUTHORIZED),
    ] {
        let r = anonymous(post_form(
            "/login",
            &body(&[("username", TEST_USER), ("password", pw)]),
            true,
        ));
        assert_eq!(call(&state, r).await.0, want);
    }
}

// ---------------------------------------------------------------------------------------------
// Sign out everywhere
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn signing_out_everywhere_ends_this_session_too() {
    let state = two_accounts();
    let other = session_of(&state, TEST_USER);
    let bob = session_of(&state, BOB);
    let generation = account(&state, TEST_USER).unwrap().generation;
    let (s, _, _) = call(
        &state,
        post(
            "/config/accounts/sign-out-everywhere",
            &[("current", WRONG)],
            false,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    assert!(signed_in(&state, TEST_SESSION).await);
    assert_eq!(failures(&state, TEST_USER), 1);
    let (s, h, _) = call(
        &state,
        post(
            "/config/accounts/sign-out-everywhere",
            &[("current", TEST_PASSWORD)],
            true,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::SEE_OTHER);
    assert_eq!(location(&h), "/login");
    assert!(clears(&h, "__Host-bop_session"));
    assert_eq!(
        account(&state, TEST_USER).unwrap().generation,
        generation + 1
    );
    assert!(!signed_in(&state, TEST_SESSION).await, "this one");
    assert!(!signed_in(&state, &other).await, "and the others");
    assert!(signed_in(&state, &bob).await, "not another account's");
    // JSON clients hear it too.
    let mine = session_of(&state, TEST_USER);
    let r = with_session(
        post(
            "/config/accounts/sign-out-everywhere",
            &[("current", TEST_PASSWORD)],
            false,
        ),
        &mine,
    );
    let (s, h, b) = call(&state, r).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(json(&b)["signed_out"], true);
    assert!(clears(&h, "__Host-bop_session"));
    assert!(!signed_in(&state, &mine).await);
}

// ---------------------------------------------------------------------------------------------
// Guards
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn cross_site_anonymous_and_token_callers_change_nothing() {
    let state = two_accounts();
    let token = "ab".repeat(32);
    std::fs::write(
        crate::admin_token::path_in(state.auth.data_root()),
        format!("{token}\n"),
    )
    .unwrap();
    for (uri, pairs) in every_route() {
        // Another site's HTTPS origin, a plain-HTTP page of this host, a same-site page.
        for (name, value) in [
            ("origin", "https://evil.test"),
            ("origin", "http://localhost:32168"),
            ("sec-fetch-site", "same-site"),
        ] {
            let r = with_header(post(uri, &pairs, false), name, value);
            let (s, _, b) = call(&state, r).await;
            assert_eq!(s, StatusCode::FORBIDDEN, "{uri} {name}: {}", text(&b));
            assert!(text(&b).contains("Cross-site"), "{uri}: {}", text(&b));
        }
        // No sign-in: 401 for scripts, the login page for browsers.
        let (s, h, _) = call(&state, anonymous(post(uri, &pairs, false))).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "{uri}");
        assert_eq!(h[header::WWW_AUTHENTICATE], "Bearer");
        let (s, h, _) = call(&state, anonymous(post(uri, &pairs, true))).await;
        assert_eq!(s, StatusCode::SEE_OTHER, "{uri}");
        assert_eq!(location(&h), "/login");
        // The API token has no password to confirm with: refused, whatever the form holds.
        let (s, _, b) = call(&state, bearer(post(uri, &pairs, false), &token)).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "{uri}: {}", text(&b));
        assert!(
            text(&b).contains("API token cannot change accounts"),
            "{uri}"
        );
        // Over 4 KiB: refused before anything is read.
        let big = format!("{}&pad={}", body(&pairs), "x".repeat(5000));
        let (s, _, _) = call(&state, post_form(uri, &big, false)).await;
        assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE, "{uri}");
    }
    // Nothing changed, nothing was hashed, nobody's failures counted.
    assert_eq!(hashed(&state), 0);
    assert!(account(&state, "carol").is_none());
    assert!(account(&state, BOB).is_some());
    let me = account(&state, TEST_USER).unwrap();
    assert_eq!((me.generation, me.failures), (1, 0));
    assert!(signed_in(&state, TEST_SESSION).await);
}

/// Changes go to the data root the process started with, never the running config's.
#[tokio::test]
async fn changes_go_to_the_start_data_root() {
    let state = test_state();
    let elsewhere = std::env::temp_dir().join(format!("bo_acct_other_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&elsewhere).unwrap();
    let mut moved = (*state.running()).clone();
    moved.download_dir = Some(elsewhere.clone());
    state.running.store(Arc::new(moved));
    assert_ne!(state.running().data_root(), state.auth.data_root());
    let (s, _, b) = call(
        &state,
        post(
            "/config/accounts/add",
            &[
                ("name", BOB),
                ("password", BOB_PASSWORD),
                ("password2", BOB_PASSWORD),
                ("current", WRONG),
            ],
            false,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::FORBIDDEN, "{}", text(&b));
    assert_eq!(failures(&state, TEST_USER), 1, "counted in the start root");
    let (s, _, b) = call(
        &state,
        post(
            "/config/accounts/add",
            &[
                ("name", BOB),
                ("password", BOB_PASSWORD),
                ("password2", BOB_PASSWORD),
                ("current", TEST_PASSWORD),
            ],
            false,
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{}", text(&b));
    assert!(account(&state, BOB).is_some());
    assert!(
        !Store::open(&elsewhere).path().exists(),
        "nothing elsewhere"
    );
    let _ = std::fs::remove_dir_all(&elsewhere);
}

// ---------------------------------------------------------------------------------------------
// The card
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn the_card_lists_accounts_marks_you_and_has_no_inline_code() {
    let state = two_accounts();
    let (s, html) = get_text(&state, "/config").await;
    assert_eq!(s, StatusCode::OK);
    let card = {
        let at = html.find("id=\"accounts\"").expect("the card");
        let end = html[at..].find("</section>").unwrap() + at;
        html[at..end].to_string()
    };
    assert!(
        card.contains(&format!(
            "<span class=\"acct-name\">{TEST_USER}</span> <span class=\"tag now\">you</span>"
        )),
        "{card}"
    );
    assert!(card.contains(&format!("<span class=\"acct-name\">{BOB}</span></td>")));
    for want in [
        "action=\"/config/accounts/add#accounts\"",
        "formaction=\"/config/accounts/remove#accounts\"",
        "action=\"/config/accounts/unlock#accounts\"",
        "action=\"/config/accounts/password#accounts\"",
        "action=\"/config/accounts/sign-out-everywhere#accounts\"",
        "autocomplete=\"current-password\"",
        "autocomplete=\"new-password\"",
        "data-label=\"Created\"",
    ] {
        assert!(card.contains(want), "{want}");
    }
    assert!(!card.contains("$argon2"), "never a hash");
    assert!(
        !html.contains(" style=") && !html.contains("<script>") && !html.contains(" onclick="),
        "CSP: no inline code"
    );
    assert!(!card.contains("onpaste"), "paste allowed");
    // Over the API token: the list, no forms.
    let token = "ab".repeat(32);
    std::fs::write(crate::admin_token::path_in(state.auth.data_root()), &token).unwrap();
    let (_, _, b) = call(
        &state,
        bearer(
            Request::get("/config")
                .header(header::ACCEPT, "text/html")
                .body(Body::empty())
                .unwrap(),
            &token,
        ),
    )
    .await;
    let html = text(&b);
    assert!(html.contains("<span class=\"acct-name\">bob</span>"));
    assert!(!html.contains("action=\"/config/accounts/add#accounts\""));
    assert!(!html.contains(">you</span>"));
    // Whatever a name holds, the template escapes it.
    let page = super::super::config_page::page_with_accounts(&state, |a| {
        a.rows[0].name = "<b>x</b>\"".to_string();
    });
    assert!(
        page.contains("&lt;b&gt;x&lt;/b&gt;") || page.contains("&#60;b&#62;x&#60;/b&#62;"),
        "escaped"
    );
    assert!(!page.contains("<b>x</b>"));
}
