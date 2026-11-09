//! Signing in at the HTTP layer: who is signed in ([`authenticate`]), what a refusal says
//! ([`refused`]), and the sign-in, sign-out and first-account pages (`/login`, `/logout`,
//! `/setup`). The accounts, sessions and device cookies themselves are [`crate::accounts`]; the
//! API token's file is [`crate::admin_token`].
//!
//! - **Credentials** (HTTPS only: the plain port never reaches an admin route, see
//!   `server::plain_gate`): a session cookie (`__Host-bop_session`, [`crate::accounts::cookies`])
//!   checked against the accounts as last read, or `Authorization: Bearer <API token>` for scripts
//!   and Prometheus. Nothing else: the `bop_admin` cookie of earlier versions is ignored (and
//!   cleared), and no address, loopback included, passes without one. A request with
//!   `X-Prism-Poll: 1` (a page polling on its own) does not move the session's idle clock.
//! - **No accounts yet** (`users.json` missing) sends browsers to `/setup`; **a damaged accounts
//!   file** closes sign-in (503) and never reopens setup (fail closed, [`crate::accounts`]).
//! - **Per address** ([`FailureLimit`], keyed by [`peer_key`]): at most [`MAX_FAILURES`] wrong
//!   attempts a [`WINDOW`], then 429 with `Retry-After`. A password or setup-code attempt reserves
//!   its slot under the lock *before* hashing ([`FailureLimit::reserve`]) and gets it back on
//!   success, so concurrent guesses cannot pass the limit. Bearer failures count in a bucket of
//!   their own (`AppState::admin_failures`), so a misconfigured scraper never blocks a person's
//!   sign-in from the same address.
//! - **Per account** ([`crate::accounts::lockout`]): after 10 failures in a row a doubling delay,
//!   disabled at 100; names without an account get the same answers ([`crate::accounts::Auth`]
//!   `unknown`). A browser with a device cookie of that account skips the delay (never the
//!   per-address limit, never `disabled`). Every refusal reads like a wrong password (the 429s
//!   say how long to wait); the log says which it was.
//! - **Hashing** takes a permit of the process's semaphore ([`Logins::hashing`]) within
//!   [`HASH_WAIT`], else 503 with `Retry-After`, and runs on the blocking pool.
//!
//! No password, hash, token, session or device id or setup code is in a log line, an answer or an
//! error; an unknown name that was typed is never logged or repeated.

use super::guard::{self, WarnThrottle};
use super::{AppState, blocking, encode_uri_component, render, wants_html};
use crate::accounts::lockout::Decision;
use crate::accounts::sessions::{Activity, SessionId};
use crate::accounts::{self, Snapshot, cookies, lockout, password};
use crate::admin_token::{self, AdminToken};
use crate::sync::LockExt;
use askama::Template;
use axum::Json;
use axum::extract::rejection::{ExtensionRejection, FormRejection, QueryRejection};
use axum::extract::{ConnectInfo, Form, Query, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use std::collections::{HashMap, VecDeque};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Wrong attempts one address may make in [`WINDOW`] (per bucket).
pub(crate) const MAX_FAILURES: usize = 5;
/// Wrong attempts all addresses of one IPv6 /64 together may make in [`WINDOW`] (per bucket): a
/// host can pick any address of its prefix, but a whole SLAAC LAN (and all of `fe80::/64`) shares
/// one, so this is looser than [`MAX_FAILURES`].
pub(crate) const NET_MAX_FAILURES: usize = 30;
/// How long a wrong attempt counts.
pub(crate) const WINDOW: Duration = Duration::from_secs(60);
/// Peers [`FailureLimit`] remembers.
pub(crate) const MAX_PEERS: usize = 1024;
/// Longest `next` kept: a real admin path with its query is far shorter.
const MAX_NEXT: usize = 2048;
/// `POST /login` and `POST /setup` bodies: a few short fields, urlencoded.
pub(super) const LOGIN_BODY_LIMIT: usize = 4 * 1024;
/// Longest wait for a hashing permit before answering 503.
pub(crate) const HASH_WAIT: Duration = Duration::from_secs(2);
/// `Retry-After` of a 503 for busy hashing.
pub(super) const BUSY_RETRY_SECS: u64 = 2;
/// The request header of a page polling on its own (`PrismPoll` in `assets/live.js`).
pub(crate) const POLL_HEADER: &str = "x-prism-poll";
/// The cookie a page sets before a reload or navigation a poll makes (`PrismPoll.navigate`): a
/// page load carries no header of ours. `__Host-`, so plain HTTP cannot plant it; cleared in the
/// answer it arrives with ([`clear_poll_cookie`]).
pub(crate) const POLL_COOKIE: &str = "__Host-bop_poll";
/// What every refused sign-in reads.
const WRONG: &str = "Wrong username or password.";
/// What [`SignedIn::user`] holds for the API token.
pub(crate) const TOKEN_USER: &str = "api-token";
/// At most one log line per this long for refused tokens: a guessing client would write one per
/// request.
const WARN_EVERY: Duration = Duration::from_secs(60);
/// The place in `templates/base.html` where the signed-in name goes ([`show_user`]).
pub(super) const USER_MARKER: &str = "<!--bop:user-->";
/// Largest page [`show_user`] rewrites (every page of the UI is far smaller).
const SHOW_USER_LIMIT: usize = 8 * 1024 * 1024;

/// The refused-token warnings of this process.
static WARNS: Mutex<WarnThrottle> = Mutex::new(WarnThrottle::new());
/// The "sign-in is closed" warnings of this process.
static CLOSED_WARNS: Mutex<WarnThrottle> = Mutex::new(WarnThrottle::new());

/// The prefix key of `ip`: an IPv4 address (also one mapped into IPv6) as it is, an IPv6 address
/// by its /64. One host can pick any of the 2^64 addresses of its prefix (privacy addresses), so
/// [`FailureLimit`] caps the /64 as well as the exact address, and the connection caps count by it.
pub(crate) fn peer_key(ip: IpAddr) -> IpAddr {
    match ip.to_canonical() {
        IpAddr::V6(v6) => {
            let prefix = u128::from(v6) & !(u128::from(u64::MAX));
            IpAddr::V6(std::net::Ipv6Addr::from(prefix))
        }
        v4 => v4,
    }
}

/// Recent wrong attempts, oldest first, in two tiers: per exact address (IPv4, IPv6 /128;
/// [`MAX_FAILURES`]) and, for IPv6, per /64 ([`peer_key`], [`NET_MAX_FAILURES`]), so one IPv6
/// host at its limit does not block its neighbours, and one that rotates through addresses of its
/// prefix still meets the /64 cap. Both are checked and counted under one lock. Bounded: at most
/// [`MAX_PEERS`] keys per tier (the one whose last failure is oldest goes). Every method takes
/// the time, so the tests need no sleep.
#[derive(Debug, Default)]
pub(crate) struct FailureLimit {
    peers: HashMap<IpAddr, VecDeque<Instant>>,
    nets: HashMap<IpAddr, VecDeque<Instant>>,
}

/// An attempt [`FailureLimit::reserve`] counted in advance: give it back with
/// [`FailureLimit::refund`] when the attempt succeeded (or never happened).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub(crate) struct Reserved {
    ip: IpAddr,
    at: Instant,
}

/// The exact address `ip` counts under (an IPv4 address mapped into IPv6 is IPv4).
fn exact_key(ip: IpAddr) -> IpAddr {
    ip.to_canonical()
}

/// The /64 `ip` also counts under, for IPv6 only.
fn net_key(ip: IpAddr) -> Option<IpAddr> {
    let key = peer_key(ip);
    key.is_ipv6().then_some(key)
}

/// One tier: seconds until `key` may try again under `max` failures a [`WINDOW`].
fn tier_blocked(
    map: &mut HashMap<IpAddr, VecDeque<Instant>>,
    key: IpAddr,
    max: usize,
    now: Instant,
) -> Option<u64> {
    let times = map.get_mut(&key)?;
    while times
        .front()
        .is_some_and(|t| now.saturating_duration_since(*t) >= WINDOW)
    {
        times.pop_front();
    }
    let Some(oldest) = times.front().copied() else {
        map.remove(&key);
        return None;
    };
    if times.len() < max {
        return None;
    }
    // The wait ends when enough of the oldest have left the window.
    let decisive = times.get(times.len() - max).copied().unwrap_or(oldest);
    let left = WINDOW.saturating_sub(now.saturating_duration_since(decisive));
    Some(left.as_secs() + u64::from(left.subsec_nanos() > 0))
}

/// One tier: one more failure of `key`.
fn tier_fail(map: &mut HashMap<IpAddr, VecDeque<Instant>>, key: IpAddr, max: usize, now: Instant) {
    if !map.contains_key(&key) && map.len() >= MAX_PEERS {
        map.retain(|_, times| {
            times
                .back()
                .is_some_and(|t| now.saturating_duration_since(*t) < WINDOW)
        });
        if map.len() >= MAX_PEERS {
            let oldest = map
                .iter()
                .min_by_key(|(_, times)| times.back().copied())
                .map(|(ip, _)| *ip);
            if let Some(ip) = oldest {
                map.remove(&ip);
            }
        }
    }
    let times = map.entry(key).or_default();
    times.push_back(now);
    while times.len() > max {
        times.pop_front();
    }
}

/// One tier: take back the failure of `key` made at `at`.
fn tier_refund(map: &mut HashMap<IpAddr, VecDeque<Instant>>, key: IpAddr, at: Instant) {
    if let Some(times) = map.get_mut(&key) {
        if let Some(i) = times.iter().rposition(|t| *t == at) {
            times.remove(i);
        }
        if times.is_empty() {
            map.remove(&key);
        }
    }
}

impl FailureLimit {
    /// Seconds (rounded up, at least 1) until `ip` may try again; `None` when it may now.
    fn blocked(&mut self, ip: IpAddr, now: Instant) -> Option<u64> {
        let exact = tier_blocked(&mut self.peers, exact_key(ip), MAX_FAILURES, now);
        let net = net_key(ip).and_then(|k| tier_blocked(&mut self.nets, k, NET_MAX_FAILURES, now));
        exact.max(net)
    }

    /// One more wrong attempt from `ip`.
    fn fail(&mut self, ip: IpAddr, now: Instant) {
        tier_fail(&mut self.peers, exact_key(ip), MAX_FAILURES, now);
        if let Some(k) = net_key(ip) {
            tier_fail(&mut self.nets, k, NET_MAX_FAILURES, now);
        }
    }

    /// Check and count in one step: `Err(seconds to wait)` when `ip` is over either limit, else
    /// the attempt counts as a failure in both tiers from now on until it is
    /// [refunded](Self::refund). Callers do this under the lock before any slow work, so
    /// concurrent attempts cannot pass the limits.
    pub(crate) fn reserve(&mut self, ip: IpAddr, now: Instant) -> Result<Reserved, u64> {
        if let Some(wait) = self.blocked(ip, now) {
            return Err(wait);
        }
        self.fail(ip, now);
        Ok(Reserved { ip, at: now })
    }

    /// Give back a [`Self::reserve`]d attempt (it succeeded): one entry with its time goes, in
    /// both tiers.
    pub(crate) fn refund(&mut self, r: Reserved) {
        tier_refund(&mut self.peers, exact_key(r.ip), r.at);
        if let Some(k) = net_key(r.ip) {
            tier_refund(&mut self.nets, k, r.at);
        }
    }

    #[cfg(test)]
    fn peers(&self) -> usize {
        self.peers.len()
    }

    /// Attempts counted for the exact address `ip` now (tests).
    #[cfg(test)]
    pub(crate) fn count(&self, ip: IpAddr) -> usize {
        self.peers.get(&exact_key(ip)).map_or(0, VecDeque::len)
    }

    /// Make every failure `by` older.
    #[cfg(test)]
    pub(crate) fn age(&mut self, by: Duration) {
        for t in self
            .peers
            .values_mut()
            .chain(self.nets.values_mut())
            .flatten()
        {
            *t = t.checked_sub(by).unwrap();
        }
    }
}

/// The sign-in guards of the process (one for every generation, like `admin_failures`): the
/// per-address limit of passwords and setup codes, the hashing semaphore and how long a request
/// waits for it, how many hashes ran (a counter for tests and diagnostics), and a pause before a
/// sign-in's result is written (zero; tests make the write slow to show nobody waits for it).
#[derive(Debug)]
pub(crate) struct Logins {
    pub(crate) failures: Mutex<FailureLimit>,
    pub(crate) hashing: &'static password::Semaphore,
    pub(crate) hash_wait: Duration,
    pub(crate) hashed: AtomicU64,
    pub(crate) record_delay: Duration,
}

impl Default for Logins {
    fn default() -> Self {
        Self {
            failures: Mutex::default(),
            hashing: password::semaphore(),
            hash_wait: HASH_WAIT,
            hashed: AtomicU64::new(0),
            record_delay: Duration::ZERO,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Who is signed in
// ---------------------------------------------------------------------------------------------

/// How a request is signed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Via {
    /// A web UI session (an account).
    Session,
    /// The API token (`Authorization: Bearer`).
    Token,
}

/// A request extension of every admin request that got through ([`super::admin_only`]): for
/// templates and logs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SignedIn {
    /// The account's name ([`TOKEN_USER`] for the API token).
    pub(crate) user: String,
    pub(crate) via: Via,
}

/// What came of an API token check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Verdict {
    Valid,
    /// It shows a wrong one.
    Wrong,
    /// Too many wrong ones: wait this many seconds. Not compared.
    Limited(u64),
    /// The token file cannot be used, so nobody passes with a token.
    Unavailable,
}

/// What [`authenticate`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Outcome {
    SignedIn(SignedIn),
    /// No credential, or a session that is not (any more) valid (`stale`: one was shown).
    Anonymous {
        stale: bool,
    },
    /// No accounts exist yet: setup.
    NoAccounts,
    /// The accounts file cannot be used (why, for the log only): nobody signs in.
    Closed(String),
    /// A Bearer token was shown and refused.
    Token(Verdict),
}

/// The Bearer token a request shows. Bytes, so a value that is not text is a wrong token rather
/// than none; an empty value is none.
fn bearer_token(headers: &HeaderMap) -> Option<&[u8]> {
    headers.get_all(header::AUTHORIZATION).iter().find_map(|v| {
        let (scheme, rest) = v.as_bytes().split_at_checked(7)?;
        scheme
            .eq_ignore_ascii_case(b"bearer ")
            .then(|| rest.trim_ascii())
            .filter(|t| !t.is_empty())
    })
}

/// The value of the cookie named exactly `name` over every `Cookie` header.
pub(super) fn cookie<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a [u8]> {
    headers
        .get_all(header::COOKIE)
        .iter()
        .find_map(|v| cookies::value(v.as_bytes(), name))
}

/// Whether the request is a page polling on its own (`X-Prism-Poll: 1`), or the page load a
/// poll made ([`POLL_COOKIE`]).
fn activity(headers: &HeaderMap) -> Activity {
    let poll = headers
        .get(POLL_HEADER)
        .is_some_and(|v| v.as_bytes().trim_ascii() == b"1")
        || cookie(headers, POLL_COOKIE) == Some(b"1");
    if poll { Activity::Poll } else { Activity::User }
}

/// Clear [`POLL_COOKIE`] in `resp` (its request sent it), so only that one load is a poll.
pub(super) fn clear_poll_cookie(resp: &mut Response) {
    if let Some(v) = set_cookie(&format!(
        "{POLL_COOKIE}=; Secure; SameSite=Strict; Path=/; Max-Age=0"
    )) {
        resp.headers_mut().append(header::SET_COOKIE, v);
    }
}

/// `presented` against `stored` for `ip`: limit first, then the comparison and the count under
/// one lock.
fn judge(
    limit: &Mutex<FailureLimit>,
    ip: IpAddr,
    now: Instant,
    stored: &AdminToken,
    presented: &[u8],
) -> Verdict {
    let mut limit = limit.lock_recover();
    if let Some(wait) = limit.blocked(ip, now) {
        return Verdict::Limited(wait);
    }
    if stored.matches(presented) {
        return Verdict::Valid;
    }
    limit.fail(ip, now);
    Verdict::Wrong
}

pub(super) fn who(peer: Option<SocketAddr>) -> String {
    peer.map_or_else(
        || "unknown".to_string(),
        |a| a.ip().to_canonical().to_string(),
    )
}

/// `presented` against the token file, for `peer`.
async fn token_attempt(
    state: &Arc<AppState>,
    peer: Option<SocketAddr>,
    presented: &[u8],
) -> Verdict {
    let path = admin_token::path_in(state.auth.data_root());
    let stored = blocking(move || admin_token::read(&path)).await;
    let stored = match stored {
        Ok(t) => t,
        Err(e) => {
            let dropped = WARNS.lock_recover().allow_at(Instant::now(), WARN_EVERY);
            if let Some(dropped) = dropped {
                warn!(
                    dropped,
                    "the API token cannot be checked, so nobody passes with it: {e:#}"
                );
            }
            return Verdict::Unavailable;
        }
    };
    let ip = peer.map_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED), |a| a.ip());
    let verdict = judge(
        &state.admin_failures,
        ip,
        Instant::now(),
        &stored,
        presented,
    );
    if verdict != Verdict::Valid {
        let dropped = WARNS.lock_recover().allow_at(Instant::now(), WARN_EVERY);
        if let Some(dropped) = dropped {
            warn!(peer = %who(peer), ?verdict, dropped, "wrong API token");
        }
    }
    verdict
}

/// Who `headers` sign in as (see the module docs). The caller has checked the transport (HTTPS)
/// and the `Host`.
pub(super) async fn authenticate(
    state: &Arc<AppState>,
    peer: Option<SocketAddr>,
    headers: &HeaderMap,
) -> Outcome {
    if let Some(token) = bearer_token(headers) {
        return match token_attempt(state, peer, token).await {
            Verdict::Valid => Outcome::SignedIn(SignedIn {
                user: TOKEN_USER.to_string(),
                via: Via::Token,
            }),
            other => Outcome::Token(other),
        };
    }
    let presented = cookie(headers, cookies::SESSION);
    let snapshot = state.auth.users.get();
    let users = match &*snapshot {
        Snapshot::Missing => return Outcome::NoAccounts,
        Snapshot::Broken(why) => return Outcome::Closed(why.clone()),
        Snapshot::Users(u) => u,
    };
    let Some(presented) = presented else {
        return Outcome::Anonymous { stale: false };
    };
    match state
        .auth
        .sessions
        .check(presented, users, accounts::now(), activity(headers))
    {
        Some(valid) => Outcome::SignedIn(SignedIn {
            user: valid.user,
            via: Via::Session,
        }),
        None => Outcome::Anonymous { stale: true },
    }
}

// ---------------------------------------------------------------------------------------------
// Answers
// ---------------------------------------------------------------------------------------------

/// `next` when it is a path on this server: one leading `/`, then printable ASCII with no
/// backslash. Browsers read `\` as `/`, and drop tabs and newlines before they parse a URL, so
/// `/\evil` and `/<tab>/evil` are `//evil`, another host.
pub(super) fn safe_next(next: &str) -> Option<&str> {
    let rest = next.strip_prefix('/')?;
    let local = next.len() <= MAX_NEXT
        && !rest.starts_with('/')
        && next.bytes().all(|b| b.is_ascii_graphic() && b != b'\\');
    local.then_some(next)
}

/// `Cache-Control: no-store` on an answer to someone not signed in (login, setup, refusals and
/// redirects), so no shared cache or back button keeps it.
pub(super) fn no_store(mut resp: Response) -> Response {
    resp.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    resp
}

/// A `Set-Cookie` value (`sensitive`, so it is never logged by a layer that prints headers).
pub(super) fn set_cookie(value: &str) -> Option<HeaderValue> {
    let mut v = HeaderValue::from_str(value).ok()?;
    v.set_sensitive(true);
    Some(v)
}

/// 303 to `location` (a path of ours: printable ASCII, else the home page).
pub(super) fn see_other(location: &str) -> Response {
    let value = HeaderValue::from_str(location).unwrap_or(HeaderValue::from_static("/"));
    no_store((StatusCode::SEE_OTHER, [(header::LOCATION, value)]).into_response())
}

/// Where a refused browser goes: the login page, with the way back. For a GET that is the
/// request itself. Another method cannot be repeated by a redirect (and its path may not take a
/// GET at all), so the way back is the page that sent it: the `Referer`'s path and query, when
/// that page is on this server (its authority is the request's `Host`); else none.
fn login_url(method: &Method, uri: &Uri, headers: &HeaderMap) -> String {
    let back = if matches!(*method, Method::GET | Method::HEAD) {
        uri.path_and_query().map(|p| p.as_str().to_string())
    } else {
        guard::request_host(headers, uri).and_then(|host| guard::referer_on_host(headers, &host))
    };
    match back.as_deref().and_then(safe_next) {
        Some(path) => format!("/login?next={}", encode_uri_component(path)),
        None => "/login".to_string(),
    }
}

/// `status` with the JSON shape of the other refusals.
pub(super) fn json_refusal(status: StatusCode, message: &str) -> Response {
    no_store(
        (
            status,
            Json(serde_json::json!({ "success": false, "message": message, "error": message })),
        )
            .into_response(),
    )
}

pub(super) fn with_retry_after(mut resp: Response, secs: u64) -> Response {
    resp.headers_mut()
        .insert(header::RETRY_AFTER, HeaderValue::from(secs));
    resp
}

fn with_bearer_challenge(mut resp: Response) -> Response {
    resp.headers_mut()
        .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    resp
}

/// Log that sign-in is closed (throttled), with the reason the operator needs.
pub(super) fn note_closed(why: &str) {
    let dropped = CLOSED_WARNS
        .lock_recover()
        .allow_at(Instant::now(), WARN_EVERY);
    if let Some(dropped) = dropped {
        warn!(dropped, "web UI sign-in is closed: {why}");
    }
}

/// The 503 of a damaged accounts file: sign-in is closed (the reason is in the log, not here:
/// it names the file).
fn closed(why: &str, headers: &HeaderMap) -> Response {
    note_closed(why);
    no_store(guard::refused(
        StatusCode::SERVICE_UNAVAILABLE,
        headers,
        "Sign-in is closed",
        "Nobody can sign in right now: the server cannot use its accounts file (it is \
         unreadable, damaged or from a newer version). The server's log says which and what to \
         do; on the server, `blue-onyx-prism user list` shows the same error. Setup does not \
         reopen by itself."
            .to_string(),
    ))
}

/// Browsers to `/setup`, others a JSON 401 that names it.
fn to_setup(headers: &HeaderMap) -> Response {
    if wants_html(headers) {
        return see_other("/setup");
    }
    with_bearer_challenge(json_refusal(
        StatusCode::UNAUTHORIZED,
        "No web UI account exists yet. Open /setup over HTTPS in a browser and create the first \
         account with the setup code that `blue-onyx-prism setup-code` prints on the server (or \
         run `blue-onyx-prism user add <name>` there); scripts can use `Authorization: Bearer \
         <API token>` meanwhile.",
    ))
}

/// The answer to a request [`authenticate`] did not sign in: a browser goes to the login (or
/// setup) page, anything else gets a JSON 401 with `WWW-Authenticate: Bearer`, too many wrong
/// tokens 429, a closed accounts file 503.
pub(super) fn refused(
    outcome: &Outcome,
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
    peer: Option<SocketAddr>,
) -> Response {
    let peer = who(peer);
    debug!(%peer, path = %uri.path(), ?outcome, "admin route refused");
    match outcome {
        Outcome::SignedIn(_) => see_other("/"),
        Outcome::NoAccounts => to_setup(headers),
        Outcome::Closed(why) => closed(why, headers),
        Outcome::Anonymous { stale } => {
            let mut resp = if wants_html(headers) {
                see_other(&login_url(method, uri, headers))
            } else {
                with_bearer_challenge(json_refusal(
                    StatusCode::UNAUTHORIZED,
                    "Sign in first: open this server's /login page over HTTPS in a browser, or \
                     send `Authorization: Bearer <API token>` (`blue-onyx-prism admin-token` \
                     prints it on the server).",
                ))
            };
            if *stale && let Some(v) = set_cookie(&cookies::cleared(cookies::SESSION)) {
                resp.headers_mut().append(header::SET_COOKIE, v);
            }
            resp
        }
        Outcome::Token(Verdict::Limited(wait)) => with_retry_after(
            no_store(guard::refused(
                StatusCode::TOO_MANY_REQUESTS,
                headers,
                "Too many attempts",
                format!(
                    "Too many wrong API tokens from your address ({peer}). Wait {wait} s, then \
                     try again with the token that `blue-onyx-prism admin-token` prints on the \
                     server."
                ),
            )),
            *wait,
        ),
        Outcome::Token(Verdict::Unavailable) => with_bearer_challenge(json_refusal(
            StatusCode::UNAUTHORIZED,
            "The server cannot read its API token, so nobody passes with one. On the server, \
             check the service's log and run `blue-onyx-prism admin-token`; people can still \
             sign in with an account.",
        )),
        Outcome::Token(_) => with_bearer_challenge(json_refusal(
            StatusCode::UNAUTHORIZED,
            &format!(
                "That API token is wrong (your address: {peer}); it may have been rotated. Run \
                 `blue-onyx-prism admin-token` on the server to read the current one."
            ),
        )),
    }
}

/// Put the signed-in `user` into an HTML page at [`USER_MARKER`] (`templates/base.html`'s
/// header). Other answers, and pages over [`SHOW_USER_LIMIT`], pass as they are.
pub(super) async fn show_user(resp: Response, user: &str) -> Response {
    use axum::body::HttpBody as _;
    let html = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/html"));
    let small = resp
        .body()
        .size_hint()
        .exact()
        .is_some_and(|n| n <= SHOW_USER_LIMIT as u64);
    if !html || !small {
        return resp;
    }
    let (mut parts, body) = resp.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, SHOW_USER_LIMIT).await else {
        return (StatusCode::INTERNAL_SERVER_ERROR, "page could not be read").into_response();
    };
    let text = String::from_utf8_lossy(&bytes);
    if !text.contains(USER_MARKER) {
        return Response::from_parts(parts, axum::body::Body::from(bytes));
    }
    // Account names are [a-z0-9._-] only; escaped anyway.
    let name: String = user
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        .collect();
    let page = text.replacen(USER_MARKER, &name, 1);
    parts.headers.remove(header::CONTENT_LENGTH);
    Response::from_parts(parts, axum::body::Body::from(page))
}

// ---------------------------------------------------------------------------------------------
// /login
// ---------------------------------------------------------------------------------------------

/// The login page.
#[derive(Template)]
#[template(path = "login.html")]
struct LoginTemplate {
    nav: &'static str,
    version: &'static str,
    /// Where to go after signing in: checked by [`safe_next`], escaped by the template.
    next: String,
    /// The name typed last time, when it has the shape of one.
    username: String,
    error: Option<String>,
}

fn login_form(status: StatusCode, next: &str, username: &str, error: Option<String>) -> Response {
    let page = LoginTemplate {
        nav: "",
        version: crate::VERSION,
        next: next.to_string(),
        username: username.to_string(),
        error,
    };
    no_store((status, render(&page)).into_response())
}

/// `?next=` of the login page.
#[derive(serde::Deserialize)]
pub(super) struct LoginQuery {
    #[serde(default)]
    next: String,
}

/// `GET /login`: the form. With no accounts yet: to `/setup`; a damaged accounts file: 503; a
/// browser that is signed in already goes on.
pub(super) async fn login_page(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    query: Result<Query<LoginQuery>, QueryRejection>,
) -> Response {
    let next = query.map(|Query(q)| q.next).unwrap_or_default();
    let next = safe_next(&next).unwrap_or_default();
    let snapshot = state.auth.users.get();
    match &*snapshot {
        Snapshot::Missing => return see_other("/setup"),
        Snapshot::Broken(why) => return closed(why, &headers),
        Snapshot::Users(users) => {
            let signed_in = cookie(&headers, cookies::SESSION).is_some_and(|p| {
                state
                    .auth
                    .sessions
                    .check(p, users, accounts::now(), Activity::User)
                    .is_some()
            });
            if signed_in {
                return see_other(if next.is_empty() { "/" } else { next });
            }
        }
    }
    login_form(StatusCode::OK, next, "", None)
}

/// The body of `POST /login`. No `Debug`: it holds the attempt.
#[derive(serde::Deserialize)]
pub(super) struct LoginForm {
    #[serde(default)]
    username: String,
    #[serde(default)]
    password: String,
    #[serde(default)]
    next: String,
}

/// How a refused sign-in is answered: the form again for browsers (the typed name kept only when
/// it has the shape of a name), JSON for other clients.
struct Answerer<'a> {
    html: bool,
    next: &'a str,
    shown: &'a str,
}

impl Answerer<'_> {
    fn fail(&self, status: StatusCode, message: &str) -> Response {
        if self.html {
            login_form(status, self.next, self.shown, Some(message.to_string()))
        } else {
            json_refusal(status, message)
        }
    }

    fn wait(&self, secs: u64) -> Response {
        with_retry_after(
            self.fail(
                StatusCode::TOO_MANY_REQUESTS,
                &format!("Too many attempts. Wait {secs} s, then try again."),
            ),
            secs,
        )
    }

    fn busy(&self) -> Response {
        with_retry_after(
            self.fail(
                StatusCode::SERVICE_UNAVAILABLE,
                &format!(
                    "The server is busy checking other sign-ins. Try again in {BUSY_RETRY_SECS} s."
                ),
            ),
            BUSY_RETRY_SECS,
        )
    }
}

/// The key of a name without an account in [`crate::accounts::lockout::UnknownNames`]: lowercase,
/// at most 64 characters (the map is bounded; a longer name is no account anyway).
fn unknown_key(typed: &str) -> String {
    typed.trim().to_lowercase().chars().take(64).collect()
}

/// Reserve an attempt for `peer` in the password bucket.
pub(super) fn reserve(state: &AppState, peer: Option<SocketAddr>) -> Result<Reserved, u64> {
    let ip = peer.map_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED), |a| a.ip());
    state
        .login
        .failures
        .lock_recover()
        .reserve(ip, Instant::now())
}

pub(super) fn refund(state: &AppState, r: Reserved) {
    state.login.failures.lock_recover().refund(r);
}

/// The cookies and redirect of a browser that just signed in as `user` (generation `generation`):
/// a new session (never the one the browser may have sent), the device cookie renewed or issued,
/// 303 to `next` or home. Written before it returns.
fn signed_in_answer(
    state: &AppState,
    headers: &HeaderMap,
    user: &str,
    generation: u64,
    next: &str,
    html: bool,
) -> Response {
    let now = accounts::now();
    if let Some(old) = cookie(headers, cookies::SESSION).and_then(SessionId::from_presented) {
        state.auth.sessions.end(&old, now);
    }
    let device = cookie(headers, cookies::DEVICE)
        .filter(|d| state.auth.devices.record_success(d, user, now))
        .and_then(accounts::devices::DeviceId::from_presented)
        .unwrap_or_else(|| state.auth.devices.issue(user, now));
    let session = state.auth.sessions.create(user, generation, now);
    let target = if next.is_empty() { "/" } else { next };
    let mut resp = if html {
        see_other(target)
    } else {
        no_store(Json(serde_json::json!({ "success": true, "next": target })).into_response())
    };
    let h = resp.headers_mut();
    let max_age = accounts::sessions::ABSOLUTE_SECS;
    for v in [
        cookies::session_set_cookie(&session, max_age),
        cookies::device_set_cookie(&device),
    ] {
        if let Some(v) = set_cookie(&v) {
            h.append(header::SET_COOKIE, v);
        }
    }
    resp
}

/// `POST /login`: the per-address limit (reserved before anything slow), the account's lockout
/// (or the device cookie that skips its delay), one hash on the blocking pool, then the session
/// and device cookies, or the form again with one generic message. A body that cannot be read,
/// or an empty field, is no attempt.
pub(super) async fn login_submit(
    State(state): State<Arc<AppState>>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    headers: HeaderMap,
    form: Result<Form<LoginForm>, FormRejection>,
) -> Response {
    let peer = peer.ok().map(|c| c.0);
    let (typed, password, next) = form.map_or_else(
        |_| (String::new(), String::new(), String::new()),
        |Form(f)| (f.username, f.password, f.next),
    );
    let next = safe_next(&next).unwrap_or_default().to_string();
    let name = accounts::normalize_name(typed.trim()).ok();
    let answer = Answerer {
        html: wants_html(&headers),
        next: &next,
        shown: name.as_deref().unwrap_or_default(),
    };
    if typed.trim().is_empty() || password.is_empty() {
        return answer.fail(StatusCode::UNAUTHORIZED, WRONG);
    }
    match &*state.auth.users.get() {
        Snapshot::Missing => return to_setup(&headers),
        Snapshot::Broken(why) => return closed(why, &headers),
        Snapshot::Users(_) => {}
    }
    let reserved = match reserve(&state, peer) {
        Ok(r) => r,
        Err(wait) => {
            warn!(peer = %who(peer), wait, "sign-in refused: too many wrong attempts from this address");
            return answer.wait(wait);
        }
    };
    let now = accounts::now();
    // One key per typed name, the same for known and unknown names.
    let key = unknown_key(&typed);
    let device = cookie(&headers, cookies::DEVICE).map(<[u8]>::to_vec);
    // Decided and counted in flight under one lock, with the accounts read inside it: concurrent
    // attempts are judged as if one at a time (`accounts::process::Attempts`).
    let mut account: Option<accounts::User> = None;
    let slot = state.auth.attempts.reserve(&key, now, || {
        account = match &*state.auth.users.get() {
            Snapshot::Users(users) => name.as_deref().and_then(|n| users.get(n)).cloned(),
            _ => None,
        };
        match &account {
            Some(u) => {
                let trusted = device
                    .as_deref()
                    .is_some_and(|d| state.auth.devices.trusted(d, &u.name, now));
                (u.lockout(), trusted)
            }
            None => (state.auth.unknown.lock_recover().state(&key), false),
        }
    });
    let slot = match (slot, &account) {
        (Ok(slot), _) => slot,
        (Err(Decision::Wait(secs)), Some(u)) => {
            warn!(peer = %who(peer), user = %u.name, failures = u.failures, wait = secs, "sign-in refused: the account waits after failures in a row");
            return answer.wait(secs);
        }
        (Err(Decision::Wait(secs)), None) => {
            warn!(peer = %who(peer), wait = secs, "sign-in refused: an unknown username waits after failures in a row");
            return answer.wait(secs);
        }
        (Err(_), Some(u)) => {
            warn!(peer = %who(peer), user = %u.name, "sign-in refused: the account is disabled after too many wrong passwords; run `blue-onyx-prism user unlock <name>` on the server");
            return answer.fail(StatusCode::UNAUTHORIZED, WRONG);
        }
        (Err(_), None) => {
            warn!(peer = %who(peer), "sign-in refused: an unknown username, disabled after too many attempts");
            return answer.fail(StatusCode::UNAUTHORIZED, WRONG);
        }
    };
    let logins = Arc::clone(&state.login);
    let hash = account.as_ref().map(|u| u.hash.clone());
    let verified = blocking(move || {
        let permit = logins.hashing.acquire(logins.hash_wait)?;
        logins.hashed.fetch_add(1, Ordering::Relaxed);
        Some(match hash {
            Some(h) => password::verify(&h, &password, &permit),
            None => Ok(password::dummy_verify(&password, &permit)),
        })
    })
    .await;
    let Some(verified) = verified else {
        refund(&state, reserved);
        drop(slot);
        warn!(peer = %who(peer), "sign-in not checked: every password-hashing slot stayed busy; answered 503");
        return answer.busy();
    };
    let good = match verified {
        Ok(good) => good,
        Err(e) => {
            if let Some(u) = &account {
                error!(user = %u.name, "the stored password of this account cannot be checked: {e:#}; reset it with `blue-onyx-prism user reset-password <name>`");
            }
            false
        }
    };
    match account {
        Some(u) if good => {
            refund(&state, reserved);
            info!(peer = %who(peer), user = %u.name, "signed in to the web UI");
            let (user, generation) = (u.name.clone(), u.generation);
            if u.failures > 0 || u.last_failure.is_some() {
                record_later(&state, slot, move |st| {
                    if let Err(e) =
                        accounts::Store::open(st.auth.data_root()).record_success(&u.name)
                    {
                        warn!(user = %u.name, "could not reset the account's failure count: {e:#}");
                    }
                });
            } else {
                drop(slot);
            }
            let html = answer.html;
            let st = Arc::clone(&state);
            blocking(move || signed_in_answer(&st, &headers, &user, generation, &next, html)).await
        }
        Some(u) => {
            // Answered at once; the write follows on its own task, which holds the attempt's slot
            // until it has landed (so the next decision sees it). An unknown name's failure is a
            // memory write, so both answers take the same work before they go out.
            let who = who(peer);
            record_later(&state, slot, move |st| {
                if let Some(d) = &device {
                    st.auth.devices.record_failure(d, &u.name, now);
                }
                match accounts::Store::open(st.auth.data_root()).record_failure(&u.name, now) {
                    Ok(Some(s)) if s.disabled => {
                        warn!(peer = %who, user = %u.name, failures = s.failures, "wrong password; the account is now disabled until `blue-onyx-prism user unlock <name>` on the server")
                    }
                    Ok(Some(s)) if s.failures >= lockout::DELAY_AFTER => {
                        warn!(peer = %who, user = %u.name, failures = s.failures, "wrong password; the account now waits before each attempt")
                    }
                    Ok(_) => {
                        warn!(peer = %who, user = %u.name, "wrong password for a web UI account")
                    }
                    Err(e) => {
                        warn!(peer = %who, user = %u.name, "wrong password; the failure could not be saved: {e:#}")
                    }
                }
            });
            answer.fail(StatusCode::UNAUTHORIZED, WRONG)
        }
        None => {
            state.auth.unknown.lock_recover().record_failure(&key, now);
            drop(slot);
            warn!(peer = %who(peer), "sign-in with a username that has no account");
            answer.fail(StatusCode::UNAUTHORIZED, WRONG)
        }
    }
}

/// Write a sign-in's result after answering: `write` on a blocking task of its own, holding
/// `slot` until it is done and the accounts cache is dropped, so the next decision on the name
/// sees the result or still counts the attempt (`Logins::record_delay` first).
fn record_later(
    state: &Arc<AppState>,
    slot: accounts::process::AttemptSlot,
    write: impl FnOnce(&AppState) + Send + 'static,
) {
    let st = Arc::clone(state);
    let delay = state.login.record_delay;
    tokio::task::spawn_blocking(move || {
        if !delay.is_zero() {
            std::thread::sleep(delay);
        }
        write(&st);
        st.auth.users.invalidate();
        drop(slot);
    });
}

/// `POST /logout`: the session ends on the server (its cookie stops working everywhere), the
/// session cookie is cleared (the device cookie stays), back to the login page.
pub(super) async fn logout(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let id = cookie(&headers, cookies::SESSION).and_then(SessionId::from_presented);
    if let Some(id) = id {
        let user = state.auth.sessions.record(&id).map(|r| r.user);
        let ended = blocking(move || state.auth.sessions.end(&id, accounts::now())).await;
        if ended && let Some(user) = user {
            info!(%user, "signed out of the web UI");
        }
    }
    let mut resp = see_other("/login");
    if let Some(v) = set_cookie(&cookies::cleared(cookies::SESSION)) {
        resp.headers_mut().append(header::SET_COOKIE, v);
    }
    resp
}

// ---------------------------------------------------------------------------------------------
// /setup
// ---------------------------------------------------------------------------------------------

/// The first-account page.
#[derive(Template)]
#[template(path = "setup.html")]
struct SetupTemplate {
    nav: &'static str,
    version: &'static str,
    /// The command that prints the code on the server.
    command: String,
    /// The command that prints the certificate's fingerprint on the server. The page never shows
    /// a fingerprint itself: one shown by the page proves nothing (a man in the middle would
    /// show its own).
    fingerprint_command: String,
    username: String,
    error: Option<String>,
}

fn setup_form(
    state: &AppState,
    status: StatusCode,
    username: &str,
    error: Option<String>,
) -> Response {
    let page = SetupTemplate {
        nav: "",
        version: crate::VERSION,
        command: crate::resources::hf_token::cli_command_for(&state.config_path, "setup-code"),
        fingerprint_command: crate::resources::hf_token::cli_command_for(
            &state.config_path,
            "tls-fingerprint",
        ),
        username: username.to_string(),
        error,
    };
    no_store((status, render(&page)).into_response())
}

/// `GET /setup`: the form while no accounts exist (a setup code is made if there is none);
/// otherwise to `/login`, or 503 when the accounts file cannot be used.
pub(super) async fn setup_page(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let snapshot = state.auth.users.get();
    match &*snapshot {
        Snapshot::Users(_) => return see_other("/login"),
        Snapshot::Broken(why) => return closed(why, &headers),
        Snapshot::Missing => {}
    }
    blocking(move || {
        if let Err(e) = accounts::setup::ensure_code(state.auth.data_root()) {
            warn!("could not make the setup code: {e:#}");
        }
        setup_form(&state, StatusCode::OK, "", None)
    })
    .await
}

/// The body of `POST /setup`. No `Debug`: it holds the code and the password.
#[derive(serde::Deserialize)]
pub(super) struct SetupForm {
    #[serde(default)]
    code: String,
    #[serde(default)]
    username: String,
    #[serde(default)]
    password: String,
    #[serde(default)]
    password2: String,
}

/// `POST /setup`: the first account, with the setup code. The per-address limit is reserved
/// first (a wrong code keeps it, like a wrong password), mismatched passwords are refused before
/// anything else, then [`accounts::setup::create_first`] with a hashing permit (it checks the
/// code before the policy and the hash, and again under the lock, so of concurrent setups one
/// wins). Success signs the browser in.
pub(super) async fn setup_submit(
    State(state): State<Arc<AppState>>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    headers: HeaderMap,
    form: Result<Form<SetupForm>, FormRejection>,
) -> Response {
    let peer = peer.ok().map(|c| c.0);
    let html = wants_html(&headers);
    let Ok(Form(f)) = form else {
        return setup_refusal(
            &state,
            html,
            StatusCode::BAD_REQUEST,
            "",
            "The form could not be read; reload the page and try again.",
        );
    };
    let name = accounts::normalize_name(f.username.trim()).ok();
    let shown = name.clone().unwrap_or_default();
    let snapshot = state.auth.users.get();
    match &*snapshot {
        Snapshot::Users(_) => return see_other("/login"),
        Snapshot::Broken(why) => return closed(why, &headers),
        Snapshot::Missing => {}
    }
    let reserved = match reserve(&state, peer) {
        Ok(r) => r,
        Err(wait) => {
            warn!(peer = %who(peer), wait, "setup refused: too many wrong attempts from this address");
            let resp = setup_refusal(
                &state,
                html,
                StatusCode::TOO_MANY_REQUESTS,
                &shown,
                &format!("Too many attempts. Wait {wait} s, then try again."),
            );
            return with_retry_after(resp, wait);
        }
    };
    if f.password != f.password2 {
        refund(&state, reserved);
        return setup_refusal(
            &state,
            html,
            StatusCode::BAD_REQUEST,
            &shown,
            "The two passwords differ; type the same password twice.",
        );
    }
    let logins = Arc::clone(&state.login);
    let st = Arc::clone(&state);
    let root = state.auth.data_root().to_path_buf();
    let typed_name = f.username.trim().to_string();
    let outcome = blocking(move || {
        let permit = logins.hashing.acquire(logins.hash_wait)?;
        let out = accounts::setup::create_first(
            &root,
            &f.code,
            &typed_name,
            &f.password,
            &permit,
            accounts::now(),
        );
        if !matches!(
            out,
            Err(accounts::setup::SetupError::WrongCode | accounts::setup::SetupError::NoCode)
        ) {
            logins.hashed.fetch_add(1, Ordering::Relaxed);
        }
        st.auth.users.invalidate();
        Some(out)
    })
    .await;
    use accounts::setup::SetupError;
    let Some(outcome) = outcome else {
        refund(&state, reserved);
        let resp = setup_refusal(
            &state,
            html,
            StatusCode::SERVICE_UNAVAILABLE,
            &shown,
            &format!(
                "The server is busy checking other sign-ins. Try again in {BUSY_RETRY_SECS} s."
            ),
        );
        return with_retry_after(resp, BUSY_RETRY_SECS);
    };
    match outcome {
        Ok(user) => {
            refund(&state, reserved);
            info!(peer = %who(peer), user = %user.name, "created the first web UI account with the setup code");
            blocking(move || {
                signed_in_answer(&state, &headers, &user.name, user.generation, "", html)
            })
            .await
        }
        Err(SetupError::WrongCode | SetupError::NoCode) => {
            warn!(peer = %who(peer), "setup refused: wrong setup code");
            setup_refusal(
                &state,
                html,
                StatusCode::UNAUTHORIZED,
                &shown,
                "That is not the setup code. Run the command below on the server and type what it prints.",
            )
        }
        Err(SetupError::AccountsExist) => {
            refund(&state, reserved);
            see_other("/login")
        }
        Err(SetupError::Refused(e)) => {
            refund(&state, reserved);
            setup_refusal(
                &state,
                html,
                StatusCode::BAD_REQUEST,
                &shown,
                &format!("Not created: {e}."),
            )
        }
        Err(SetupError::Other(e)) => {
            refund(&state, reserved);
            error!("the first account could not be created: {e:#}");
            setup_refusal(
                &state,
                html,
                StatusCode::INTERNAL_SERVER_ERROR,
                &shown,
                "The account could not be saved; the server's log says why. Fix that and try again.",
            )
        }
    }
}

/// A refused setup: the form again (browsers) or JSON.
fn setup_refusal(
    state: &AppState,
    html: bool,
    status: StatusCode,
    shown: &str,
    message: &str,
) -> Response {
    if html {
        setup_form(state, status, shown, Some(message.to_string()))
    } else {
        json_refusal(status, message)
    }
}

// ---------------------------------------------------------------------------------------------
// The API token's rotation
// ---------------------------------------------------------------------------------------------

/// `POST /config/admin-token/rotate` (the Config page's button): replace the API token. For a
/// browser on the server itself only ([`super::direct_local`], on top of the sign-in every admin
/// route needs), so a token that leaked cannot be used to lock everyone else out. Neither the page
/// nor the JSON shows the new token.
pub(super) async fn rotate(
    State(state): State<Arc<AppState>>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    headers: HeaderMap,
) -> Response {
    if !super::direct_local(peer.ok().map(|c| c.0), &headers) {
        return guard::forbidden(
            &headers,
            "The API token can only be rotated from the server itself: open the Config page \
             there (https://localhost:<https port>/config), or run `blue-onyx-prism admin-token \
             --rotate`."
                .to_string(),
        );
    }
    blocking(move || {
        let path = admin_token::path_in(state.auth.data_root());
        let command = admin_token::read_command(&state.config_path);
        let html = wants_html(&headers);
        let rotated = crate::private_dir::ensure(state.auth.data_root())
            .and_then(|_| admin_token::rotate(&path));
        match rotated {
            Ok(_) => {
                info!(path = %path.display(), "the API token was rotated");
                let msg = format!(
                    "The API token was replaced; the old one stopped working, so scripts and \
                     Prometheus need the new one. Read it on this machine with: {command}"
                );
                if html {
                    return super::config_page::token_result(&state, Some(msg), None);
                }
                Json(serde_json::json!({ "success": true, "message": msg })).into_response()
            }
            Err(e) => {
                warn!("the API token was not rotated: {e:#}");
                let msg = format!(
                    "Not rotated, the old token still works: {e:#}. Fix that and try again."
                );
                if html {
                    return super::config_page::token_result(&state, None, Some(msg));
                }
                json_refusal(StatusCode::INTERNAL_SERVER_ERROR, &msg)
            }
        }
    })
    .await
}

#[cfg(test)]
mod tests;
