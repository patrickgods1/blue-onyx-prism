//! The admin token at the HTTP layer (RR-S4): who has to show it, how and what a refusal says.
//! The file itself is [`crate::admin_token`].
//!
//! - A person on this machine never needs it, and with `allow_remote_admin` off a remote peer
//!   never gets this far: it stays refused with 403 whatever it sends (`server::admin_only`). A
//!   loopback peer with a forwarding header is a reverse proxy's client, so it is remote
//!   (`server::direct_local`). Its failures all count under the proxy's address, the only one
//!   there is to go by: the header's value could be anything.
//! - With `allow_remote_admin` on, a remote peer shows the token as `Authorization: Bearer
//!   <token>` or as the cookie `POST /login` sets, on every admin route ([`verify`]) and for the
//!   full `GET /v1/config`. Open routes never look at it.
//! - `/login` and `/logout` are their own class of route (`server::login_routes`): a remote peer
//!   without a token must reach them, so they take no token, but they pass the same `Host` check
//!   and CSRF guard as an admin route, and the same 403 while remote administration is off.
//! - [`FailureLimit`]: at most [`MAX_FAILURES`] wrong tokens a [`WINDOW`] per peer address (an
//!   IPv6 peer by its /64, which one host or LAN usually has to itself: [`peer_key`]), then
//!   429 with `Retry-After`, without comparing, so the right token is refused as well and nothing
//!   tells it apart. The check and the count happen under one lock, so concurrent guesses cannot
//!   pass it, and valid traffic never counts: a page loading many frames at once is safe.
//! - A wrong cookie is cleared in the answer: after a rotation one browser costs one failure, not
//!   the five its polling pages would otherwise reach in seconds. A cookie that was not compared
//!   stays, so nobody else at the same address can sign an admin out.
//! - The token file is read on every check, which is what makes a rotation apply at once. When
//!   it cannot be used nobody passes ([`Verdict::Unavailable`]); that is no attempt, so it does not
//!   count, and the remote peer learns that the server has a problem, not where its file is.
//!
//! The token is compared, and set back in the cookie it arrived in, only here. It is in no error,
//! no log line (requests are logged by path, without the query) and no URL.

use super::guard::{self, WarnThrottle};
use super::{AppState, blocking, encode_uri_component, render, wants_html};
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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Name of the cookie `POST /login` sets.
const COOKIE: &str = "bop_admin";
/// Wrong tokens a peer may send in [`WINDOW`].
pub(crate) const MAX_FAILURES: usize = 5;
/// How long a wrong token counts.
pub(crate) const WINDOW: Duration = Duration::from_secs(60);
/// Peers [`FailureLimit`] remembers.
pub(crate) const MAX_PEERS: usize = 1024;
/// Longest `next` kept: a real admin path with its query is far shorter.
const MAX_NEXT: usize = 2048;
/// `POST /login` body: a token and a `next`, urlencoded.
pub(super) const LOGIN_BODY_LIMIT: usize = 3 * MAX_NEXT;
/// At most one log line per this long for refused tokens: a guessing client would write one per
/// request.
const WARN_EVERY: Duration = Duration::from_secs(60);

/// The refused-token warnings of this process.
static WARNS: Mutex<WarnThrottle> = Mutex::new(WarnThrottle::new());

/// The address [`FailureLimit`] counts `ip` under: an IPv4 address (also one mapped into IPv6)
/// as it is, an IPv6 address by its /64. One host can pick any of the 2^64 addresses of its
/// prefix (privacy addresses), so counting single IPv6 addresses would not limit it at all.
pub(crate) fn peer_key(ip: IpAddr) -> IpAddr {
    match ip.to_canonical() {
        IpAddr::V6(v6) => {
            let prefix = u128::from(v6) & !(u128::from(u64::MAX));
            IpAddr::V6(std::net::Ipv6Addr::from(prefix))
        }
        v4 => v4,
    }
}

/// Recent wrong tokens per peer address ([`peer_key`]), oldest first. Bounded: at most [`MAX_PEERS`] peers
/// ([`Self::fail`] drops the one whose last failure is oldest) and [`MAX_FAILURES`] times each.
/// Every method takes the time, so the tests need no sleep.
#[derive(Debug, Default)]
pub(crate) struct FailureLimit {
    peers: HashMap<IpAddr, VecDeque<Instant>>,
}

impl FailureLimit {
    /// Seconds (rounded up, at least 1) until `ip` may try again; `None` when it may now.
    fn blocked(&mut self, ip: IpAddr, now: Instant) -> Option<u64> {
        let ip = peer_key(ip);
        let times = self.peers.get_mut(&ip)?;
        while times
            .front()
            .is_some_and(|t| now.saturating_duration_since(*t) >= WINDOW)
        {
            times.pop_front();
        }
        let oldest = times.front().copied();
        let full = times.len() >= MAX_FAILURES;
        let Some(oldest) = oldest else {
            self.peers.remove(&ip);
            return None;
        };
        if !full {
            return None;
        }
        let left = WINDOW.saturating_sub(now.saturating_duration_since(oldest));
        Some(left.as_secs() + u64::from(left.subsec_nanos() > 0))
    }

    /// One more wrong token from `ip`.
    fn fail(&mut self, ip: IpAddr, now: Instant) {
        let ip = peer_key(ip);
        if !self.peers.contains_key(&ip) && self.peers.len() >= MAX_PEERS {
            self.make_room(now);
        }
        let times = self.peers.entry(ip).or_default();
        times.push_back(now);
        while times.len() > MAX_FAILURES {
            times.pop_front();
        }
    }

    /// Forget peers with nothing recent, else the one whose last failure is oldest.
    fn make_room(&mut self, now: Instant) {
        self.peers.retain(|_, times| {
            times
                .back()
                .is_some_and(|t| now.saturating_duration_since(*t) < WINDOW)
        });
        if self.peers.len() >= MAX_PEERS {
            let oldest = self
                .peers
                .iter()
                .min_by_key(|(_, times)| times.back().copied())
                .map(|(ip, _)| *ip);
            if let Some(ip) = oldest {
                self.peers.remove(&ip);
            }
        }
    }

    #[cfg(test)]
    fn peers(&self) -> usize {
        self.peers.len()
    }

    /// Make every failure `by` older.
    #[cfg(test)]
    pub(crate) fn age(&mut self, by: Duration) {
        for t in self.peers.values_mut().flatten() {
            *t = t.checked_sub(by).unwrap();
        }
    }
}

/// What came of a token check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Verdict {
    Valid,
    /// The request shows no token.
    Missing,
    /// It shows a wrong one.
    Wrong,
    /// Too many wrong ones: wait this many seconds. Not compared.
    Limited(u64),
    /// The token file cannot be used, so nobody passes.
    Unavailable,
}

/// A [`Verdict`] and where the token came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Checked {
    pub(super) verdict: Verdict,
    from_cookie: bool,
}

impl Checked {
    /// The answer should drop the cookie: it was compared and is wrong.
    pub(super) fn stale_cookie(self) -> bool {
        self.from_cookie && self.verdict == Verdict::Wrong
    }
}

/// The value of a `name=value` pair in a `Cookie` header.
fn cookie_value<'a>(header: &'a [u8], name: &str) -> Option<&'a [u8]> {
    header.split(|b| *b == b';').find_map(|pair| {
        let pair = pair.trim_ascii();
        let (key, value) = pair.split_at_checked(name.len())?;
        (key == name.as_bytes())
            .then(|| value.strip_prefix(b"="))
            .flatten()
    })
}

/// What the request shows as a token: `Authorization: Bearer`, else the cookie (true). Bytes, so
/// a value that is not text is a wrong token rather than none; an empty value is none.
fn presented(headers: &HeaderMap) -> Option<(&[u8], bool)> {
    let bearer = headers.get_all(header::AUTHORIZATION).iter().find_map(|v| {
        let (scheme, rest) = v.as_bytes().split_at_checked(7)?;
        scheme
            .eq_ignore_ascii_case(b"bearer ")
            .then(|| rest.trim_ascii())
            .filter(|t| !t.is_empty())
    });
    if let Some(token) = bearer {
        return Some((token, false));
    }
    headers
        .get_all(header::COOKIE)
        .iter()
        .find_map(|v| cookie_value(v.as_bytes(), COOKIE).filter(|t| !t.is_empty()))
        .map(|token| (token, true))
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

/// Log a refused token, at most once per [`WARN_EVERY`]. The address and the verdict only.
fn warn_refused(peer: Option<SocketAddr>, verdict: Verdict) {
    let dropped = WARNS.lock_recover().allow_at(Instant::now(), WARN_EVERY);
    if let Some(dropped) = dropped {
        let who = peer.map_or_else(|| "unknown".to_string(), |a| a.ip().to_string());
        warn!(peer = %who, ?verdict, dropped, "wrong admin token");
    }
}

/// `presented` against the token file, for `peer`.
async fn attempt(state: &Arc<AppState>, peer: Option<SocketAddr>, presented: &[u8]) -> Verdict {
    let path = admin_token::path(&state.config_read());
    let stored = blocking(move || admin_token::read(&path)).await;
    let stored = match stored {
        Ok(t) => t,
        Err(e) => {
            let dropped = WARNS.lock_recover().allow_at(Instant::now(), WARN_EVERY);
            if let Some(dropped) = dropped {
                warn!(dropped, "remote administration is closed: {e:#}");
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
        warn_refused(peer, verdict);
    }
    verdict
}

/// Check the token a request shows. Only for a remote peer, with `allow_remote_admin` on.
pub(super) async fn verify(
    state: &Arc<AppState>,
    peer: Option<SocketAddr>,
    headers: &HeaderMap,
) -> Checked {
    match presented(headers) {
        None => Checked {
            verdict: Verdict::Missing,
            from_cookie: false,
        },
        Some((token, from_cookie)) => Checked {
            verdict: attempt(state, peer, token).await,
            from_cookie,
        },
    }
}

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

/// `Set-Cookie` of a signed-in browser. No `Secure`: the service speaks plain HTTP.
fn session_cookie(token: &str) -> Option<HeaderValue> {
    let cookie = format!("{COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/");
    let mut value = HeaderValue::from_str(&cookie).ok()?;
    value.set_sensitive(true);
    Some(value)
}

/// `Set-Cookie` that removes the cookie.
pub(super) fn cleared_cookie() -> HeaderValue {
    HeaderValue::from_static("bop_admin=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0")
}

/// 303 to `location` (a path of ours: printable ASCII, else the home page).
fn see_other(location: &str) -> Response {
    let value = HeaderValue::from_str(location).unwrap_or(HeaderValue::from_static("/"));
    (StatusCode::SEE_OTHER, [(header::LOCATION, value)]).into_response()
}

/// Where a refused browser goes: the login page, with the way back. A request that is not a GET
/// has none: a redirect cannot repeat it, and its path may not take a GET at all.
fn login_url(method: &Method, uri: &Uri) -> String {
    let back = uri
        .path_and_query()
        .map(|p| p.as_str())
        .filter(|_| matches!(*method, Method::GET | Method::HEAD))
        .and_then(safe_next);
    match back {
        Some(path) => format!("/login?next={}", encode_uri_component(path)),
        None => "/login".to_string(),
    }
}

/// What a refused client reads. Never the token or the path of its file.
fn refusal_message(verdict: Verdict, who: &str) -> String {
    match verdict {
        Verdict::Limited(wait) => format!(
            "Too many wrong admin tokens from your address ({who}). Wait {wait} s, then try \
             again with the token that `blue-onyx-prism admin-token` prints on the server."
        ),
        Verdict::Unavailable => "The server cannot read its admin token, so nobody can sign in \
             from another machine. On the server, check the service's log and run \
             `blue-onyx-prism admin-token`, or use the admin pages on that machine."
            .to_string(),
        Verdict::Wrong => format!(
            "That admin token is wrong (your address: {who}); it may have been rotated. Run \
             `blue-onyx-prism admin-token` on the server to read the current one."
        ),
        Verdict::Valid | Verdict::Missing => format!(
            "Administration from another machine needs the admin token (your address: {who}). \
             Send `Authorization: Bearer <token>`, or sign in at /login. Run `blue-onyx-prism \
             admin-token` on the server to read it."
        ),
    }
}

/// `status` with the JSON shape of the other refusals.
fn json_refusal(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({ "success": false, "message": message, "error": message })),
    )
        .into_response()
}

/// `Retry-After` when `verdict` is a limit.
fn with_retry_after(mut resp: Response, verdict: Verdict) -> Response {
    if let Verdict::Limited(wait) = verdict {
        resp.headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from(wait));
    }
    resp
}

/// The answer to a request [`verify`] did not pass: a browser goes to the login page, anything
/// else gets a JSON 401, a client over the limit 429.
pub(super) fn refused(
    checked: Checked,
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
    peer: Option<SocketAddr>,
) -> Response {
    let who = peer.map_or_else(|| "unknown".to_string(), |a| a.ip().to_string());
    debug!(peer = %who, path = %uri.path(), verdict = ?checked.verdict, "admin route refused");
    let message = refusal_message(checked.verdict, &who);
    let mut resp = match checked.verdict {
        Verdict::Limited(_) => guard::refused(
            StatusCode::TOO_MANY_REQUESTS,
            headers,
            "Too many attempts",
            message,
        ),
        _ if wants_html(headers) => see_other(&login_url(method, uri)),
        _ => {
            let mut resp = json_refusal(StatusCode::UNAUTHORIZED, &message);
            resp.headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
            resp
        }
    };
    if checked.stale_cookie() {
        resp.headers_mut()
            .append(header::SET_COOKIE, cleared_cookie());
    }
    with_retry_after(resp, checked.verdict)
}

/// The login page.
#[derive(Template)]
#[template(path = "login.html")]
struct LoginTemplate {
    nav: &'static str,
    version: &'static str,
    /// Where to go after signing in: checked by [`safe_next`], escaped by the template.
    next: String,
    error: Option<String>,
}

fn login_form(status: StatusCode, next: &str, error: Option<String>) -> Response {
    let page = LoginTemplate {
        nav: "",
        version: crate::VERSION,
        next: next.to_string(),
        error,
    };
    (status, render(&page)).into_response()
}

/// `?next=` of the login page.
#[derive(serde::Deserialize)]
pub(super) struct LoginQuery {
    #[serde(default)]
    next: String,
}

/// `GET /login`: the form. This machine has nothing to sign in to, so it goes straight on.
pub(super) async fn login_page(
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    headers: HeaderMap,
    query: Result<Query<LoginQuery>, QueryRejection>,
) -> Response {
    let next = query.map(|Query(q)| q.next).unwrap_or_default();
    let next = safe_next(&next).unwrap_or_default();
    if super::direct_local(peer.ok().map(|c| c.0), &headers) {
        return see_other(if next.is_empty() { "/" } else { next });
    }
    login_form(StatusCode::OK, next, None)
}

/// The body of `POST /login`. No `Debug`: it holds the attempt.
#[derive(serde::Deserialize)]
pub(super) struct LoginForm {
    #[serde(default)]
    token: String,
    #[serde(default)]
    next: String,
}

/// `POST /login`: compare, set the cookie and go on. The answer to a failure is the form again
/// (JSON for other clients); what was typed is not repeated. A body that cannot be read is an
/// empty form, which is no attempt: its error could quote the body.
pub(super) async fn login_submit(
    State(state): State<Arc<AppState>>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    headers: HeaderMap,
    form: Result<Form<LoginForm>, FormRejection>,
) -> Response {
    let peer = peer.ok().map(|c| c.0);
    let (token, next) = form.map_or_else(
        |_| (String::new(), String::new()),
        |Form(f)| (f.token, f.next),
    );
    let token = token.trim();
    let next = safe_next(&next).unwrap_or_default();
    let verdict = if token.is_empty() {
        Verdict::Missing
    } else {
        attempt(&state, peer, token.as_bytes()).await
    };
    if verdict == Verdict::Valid
        && let Some(cookie) = session_cookie(token)
    {
        let who = peer.map_or_else(|| "unknown".to_string(), |a| a.ip().to_string());
        info!(peer = %who, "signed in with the admin token");
        let mut resp = see_other(if next.is_empty() { "/" } else { next });
        resp.headers_mut().insert(header::SET_COOKIE, cookie);
        return resp;
    }
    let who = peer.map_or_else(|| "unknown".to_string(), |a| a.ip().to_string());
    let status = match verdict {
        Verdict::Limited(_) => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::UNAUTHORIZED,
    };
    let message = refusal_message(verdict, &who);
    let resp = if wants_html(&headers) {
        login_form(status, next, Some(message))
    } else {
        json_refusal(status, &message)
    };
    with_retry_after(resp, verdict)
}

/// `POST /logout`: drop the cookie. It takes no token: a stale cookie can leave too.
#[allow(clippy::unused_async, reason = "axum handlers are async")]
pub(super) async fn logout() -> Response {
    let mut resp = see_other("/login");
    resp.headers_mut()
        .insert(header::SET_COOKIE, cleared_cookie());
    resp
}

/// `POST /config/admin-token/rotate` (the Config page's button): replace the token. For this
/// machine only: an admin route behind the token, and a remote holder of the token cannot use it
/// to lock everyone else out. Neither the page nor the JSON shows the new token; every signed-in
/// browser has to sign in again.
pub(super) async fn rotate(
    State(state): State<Arc<AppState>>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    headers: HeaderMap,
) -> Response {
    if !super::direct_local(peer.ok().map(|c| c.0), &headers) {
        return guard::forbidden(
            &headers,
            "The admin token can only be rotated from the server itself: open the Config page \
             there, or run `blue-onyx-prism admin-token --rotate`."
                .to_string(),
        );
    }
    blocking(move || {
        let path = admin_token::path(&state.config_read());
        let command = admin_token::read_command(&state.config_path);
        let html = wants_html(&headers);
        match admin_token::rotate(&path) {
            Ok(_) => {
                info!(path = %path.display(), "the admin token was rotated");
                let msg = format!(
                    "The admin token was replaced; the old one stopped working and signed-in \
                     browsers must sign in again. Read the new one on this machine with: \
                     {command}"
                );
                if html {
                    return super::config_page::token_result(&state, Some(msg), None);
                }
                Json(serde_json::json!({ "success": true, "message": msg })).into_response()
            }
            Err(e) => {
                warn!("the admin token was not rotated: {e:#}");
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
