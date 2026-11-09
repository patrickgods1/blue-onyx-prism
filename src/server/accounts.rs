//! The Config page's Accounts card and its routes: list the web UI accounts, add one, remove
//! one, unlock one, change your own password, sign yourself out everywhere.
//!
//! - **Routes** (admin routes, so HTTPS, a session and the CSRF guard come first; bodies of at
//!   most [`super::auth::LOGIN_BODY_LIMIT`]): `POST /config/accounts/add`, `.../remove`,
//!   `.../unlock`, `.../password`, `.../sign-out-everywhere`.
//! - **Your current password** confirms every one of them ([`reauth`]), checked by the same
//!   machinery as `/login`: the per-address reservation before hashing, the account's attempt
//!   slot and lockout (a device cookie of the account skips only its delay), a hashing permit,
//!   one answer for a wrong password and a disabled account, and a failure counted against the
//!   signed-in account. A wrong one answers **403**, never 401: a 401 sends the page to `/login`
//!   (`assets/live.js`), and the person is signed in. The API token has no password, so it
//!   cannot use these routes at all (manage accounts with `blue-onyx-prism user ...` on the
//!   server, or sign in).
//! - **Answers:** browsers get the Config page again at the card (`#accounts`) with a notice or
//!   an error and the route's status; other clients JSON (`success`, `message`). Removing your
//!   own account and signing out everywhere end this browser's session too: 303 to `/login`.
//! - **What ends what:** a removed account's sessions and device cookies go at once; a password
//!   change ends every other session of the account (its generation goes up) and gives this
//!   browser a new session id, keeping its device cookie; signing out everywhere bumps the
//!   generation and ends every session of the account, this one included.
//!
//! Every change is logged with who made it and from where; never a password, hash, session or
//! device id. Everything is written to the data root the process started with
//! ([`crate::accounts::Auth::data_root`]).

use super::auth::{self, SignedIn, Via};
use super::{AppState, blocking, wants_html};
use crate::accounts::lockout::{self, Decision};
use crate::accounts::sessions::{ABSOLUTE_SECS, SessionId};
use crate::accounts::{self, Refused, Snapshot, Store, cookies, password};
use axum::Json;
use axum::extract::rejection::{ExtensionRejection, FormRejection};
use axum::extract::{ConnectInfo, Extension, Form, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tracing::{error, info, warn};

/// What a wrong current password (or a disabled account) reads.
const WRONG_CURRENT: &str = "Wrong password: type the password you sign in with as your current \
                             password. Nothing was changed.";
/// What the API token reads on these routes.
const TOKEN_REFUSED: &str = "The API token cannot change accounts: every change is confirmed \
                             with the signed-in account's current password, and the token has \
                             none. Sign in with an account, or use `blue-onyx-prism user ...` \
                             on the server.";

// ---------------------------------------------------------------------------------------------
// The card
// ---------------------------------------------------------------------------------------------

/// One account in the card. Never the hash.
pub(super) struct AccountRow {
    pub(super) name: String,
    pub(super) created: String,
    pub(super) password_changed: String,
    /// "none", "3 in a row", "disabled after 100 in a row", ...
    pub(super) failures: String,
    /// Whether `failures` is worth a colour (some failures, or disabled).
    pub(super) warn: bool,
    /// The signed-in account.
    pub(super) you: bool,
}

/// The Accounts card of the Config page.
pub(super) struct AccountsPanel {
    pub(super) rows: Vec<AccountRow>,
    /// The accounts file cannot be used (or is missing): why, instead of the list.
    pub(super) closed: Option<String>,
    /// The signed-in account, when the page knows it.
    pub(super) me: Option<String>,
    /// Signed in with the API token: the forms are left out.
    pub(super) token: bool,
    /// The command that resets a forgotten password on the server.
    pub(super) reset_command: String,
}

/// `secs` (unix) as `YYYY-MM-DD HH:MM UTC`.
fn when(secs: u64) -> String {
    i64::try_from(secs)
        .ok()
        .and_then(|s| chrono::DateTime::<chrono::Utc>::from_timestamp(s, 0))
        .map_or_else(
            || secs.to_string(),
            |t| t.format("%Y-%m-%d %H:%M UTC").to_string(),
        )
}

/// The failures column of `u`.
fn failures_text(u: &accounts::User) -> String {
    if u.disabled {
        format!(
            "disabled after {} in a row (unlock it below)",
            u.failures.max(1)
        )
    } else if u.failures >= lockout::DELAY_AFTER {
        format!("{} in a row (each attempt waits)", u.failures)
    } else if u.failures > 0 {
        format!("{} in a row", u.failures)
    } else {
        "none".to_string()
    }
}

impl AccountsPanel {
    /// The card from the accounts as last read, nobody marked "you" yet ([`Self::signed_in`]).
    pub(super) fn of(state: &AppState) -> Self {
        let snapshot = state.auth.users.get();
        let (rows, closed) = match &*snapshot {
            Snapshot::Users(users) => {
                let mut rows: Vec<AccountRow> = users
                    .users
                    .iter()
                    .map(|u| AccountRow {
                        name: u.name.clone(),
                        created: when(u.created),
                        password_changed: when(u.password_changed),
                        failures: failures_text(u),
                        warn: u.disabled || u.failures > 0,
                        you: false,
                    })
                    .collect();
                rows.sort_by(|a, b| a.name.cmp(&b.name));
                (rows, None)
            }
            Snapshot::Missing => (
                Vec::new(),
                Some(
                    "No accounts exist yet: create the first one on the /setup page with the \
                     setup code."
                        .to_string(),
                ),
            ),
            Snapshot::Broken(_) => (
                Vec::new(),
                Some(
                    "The server cannot use its accounts file, so nobody can sign in or change \
                     accounts; the server's log says why and what to do."
                        .to_string(),
                ),
            ),
        };
        Self {
            rows,
            closed,
            me: None,
            token: false,
            reset_command: crate::resources::hf_token::cli_command_for(
                &state.config_path,
                "user reset-password <name>",
            ),
        }
    }

    /// Mark who is signed in (`signed`): their row says "you", the token gets no forms.
    pub(super) fn signed_in(&mut self, signed: Option<&SignedIn>) {
        let Some(s) = signed else { return };
        match s.via {
            Via::Token => self.token = true,
            Via::Session => {
                for r in &mut self.rows {
                    r.you = r.name == s.user;
                }
                self.me = Some(s.user.clone());
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Answers
// ---------------------------------------------------------------------------------------------

/// What every account route knows about its request.
struct Ctx {
    state: Arc<AppState>,
    signed: SignedIn,
    peer: Option<SocketAddr>,
    headers: HeaderMap,
    html: bool,
}

impl Ctx {
    /// The request's context, or the refusal of a caller that is not an account (the API
    /// token, or no sign-in at all, which `admin_only` already refuses).
    fn new(
        state: Arc<AppState>,
        signed: Option<Extension<SignedIn>>,
        peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
        headers: HeaderMap,
    ) -> Result<Self, Box<Response>> {
        let html = wants_html(&headers);
        let peer = peer.ok().map(|c| c.0);
        let Some(Extension(signed)) = signed else {
            return Err(Box::new(super::guard::forbidden(
                &headers,
                "Sign in with an account to change accounts.".to_string(),
            )));
        };
        let ctx = Self {
            state,
            signed,
            peer,
            headers,
            html,
        };
        if ctx.signed.via == Via::Token {
            warn!(peer = %ctx.who(), "an account change with the API token was refused");
            return Err(Box::new(super::guard::forbidden(
                &ctx.headers,
                TOKEN_REFUSED.to_string(),
            )));
        }
        Ok(ctx)
    }

    fn me(&self) -> &str {
        &self.signed.user
    }

    fn who(&self) -> String {
        auth::who(self.peer)
    }

    /// The Config page at the card (browsers) or JSON, with `status`.
    async fn page(
        &self,
        status: StatusCode,
        message: Option<String>,
        error: Option<String>,
    ) -> Response {
        if !self.html {
            return match (message, error) {
                (_, Some(e)) => auth::json_refusal(status, &e),
                (m, None) => (
                    status,
                    Json(serde_json::json!({ "success": true, "message": m.unwrap_or_default() })),
                )
                    .into_response(),
            };
        }
        let state = Arc::clone(&self.state);
        let signed = self.signed.clone();
        let peer = self.peer;
        let headers = self.headers.clone();
        let mut resp = blocking(move || {
            super::config_page::accounts_result(&state, &signed, peer, &headers, message, error)
        })
        .await;
        *resp.status_mut() = status;
        resp
    }

    async fn ok(&self, message: String) -> Response {
        self.page(StatusCode::OK, Some(message), None).await
    }

    async fn fail(&self, status: StatusCode, message: String) -> Response {
        self.page(status, None, Some(message)).await
    }

    /// The answer to a form that could not be read (too large: 413).
    async fn unreadable(&self, rejection: &FormRejection) -> Response {
        self.fail(
            rejection.status(),
            "The form could not be read; reload the page and try again.".to_string(),
        )
        .await
    }

    /// This browser is signed out now (its own account removed, or signed out everywhere):
    /// the session cookie cleared, browsers to `/login`.
    fn signed_out(&self, message: String) -> Response {
        let mut resp = if self.html {
            auth::see_other("/login")
        } else {
            auth::no_store(
                Json(serde_json::json!({
                    "success": true,
                    "message": message,
                    "signed_out": true,
                }))
                .into_response(),
            )
        };
        if let Some(v) = auth::set_cookie(&cookies::cleared(cookies::SESSION)) {
            resp.headers_mut().append(header::SET_COOKIE, v);
        }
        resp
    }

    /// The answer to a [`reauth`] that did not pass.
    async fn denied(&self, d: Denied) -> Response {
        match d {
            Denied::Empty => {
                self.fail(
                    StatusCode::BAD_REQUEST,
                    "Type your current password: every account change asks for it.".to_string(),
                )
                .await
            }
            Denied::Wrong => {
                self.fail(StatusCode::FORBIDDEN, WRONG_CURRENT.to_string())
                    .await
            }
            Denied::Wait(secs) => auth::with_retry_after(
                self.fail(
                    StatusCode::TOO_MANY_REQUESTS,
                    format!("Too many attempts. Wait {secs} s, then try again."),
                )
                .await,
                secs,
            ),
            Denied::Busy => auth::with_retry_after(
                self.fail(
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "The server is busy checking other passwords. Try again in {} s.",
                        auth::BUSY_RETRY_SECS
                    ),
                )
                .await,
                auth::BUSY_RETRY_SECS,
            ),
            Denied::Closed(why) => {
                auth::note_closed(&why);
                self.fail(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "The server cannot use its accounts file, so nothing was changed; the \
                     server's log says why and what to do."
                        .to_string(),
                )
                .await
            }
        }
    }

    /// The answer to an account change that failed after the password passed: the store's
    /// refusals as 400 / 409 with their message, anything else 500 (logged).
    async fn change_failed(&self, what: &str, e: &anyhow::Error) -> Response {
        if let Some(r) = e.downcast_ref::<Refused>() {
            let status = match r {
                Refused::NameTaken | Refused::LastAccount => StatusCode::CONFLICT,
                _ => StatusCode::BAD_REQUEST,
            };
            return self.fail(status, format!("{what}: {r}.")).await;
        }
        error!(by = %self.me(), peer = %self.who(), "{what}: {e:#}");
        self.fail(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{what}: the change could not be saved ({e:#}). Fix that and try again."),
        )
        .await
    }
}

// ---------------------------------------------------------------------------------------------
// Your current password
// ---------------------------------------------------------------------------------------------

/// Why [`reauth`] did not pass.
#[derive(Debug)]
enum Denied {
    /// No current password typed: nothing was counted.
    Empty,
    /// Wrong, or the account is disabled (the same answer; the log says which): counted.
    Wrong,
    /// Too many attempts (the address's or the account's): wait this many seconds.
    Wait(u64),
    /// No hashing permit came in time.
    Busy,
    /// The accounts file cannot be used (why, for the log).
    Closed(String),
}

/// Check `current` as the password of the signed-in account `me`, the way `POST /login` checks
/// one (see the module docs), and run `action` on the blocking pool when it passes, with the
/// hashing permit still held (for a new password's hash) and the attempt's slot taken, so it
/// runs before the account's next attempt is decided.
async fn reauth<T: Send + 'static>(
    ctx: &Ctx,
    current: String,
    action: impl FnOnce(&AppState, &password::Permit<'_>) -> T + Send + 'static,
) -> Result<T, Denied> {
    let state = &ctx.state;
    if current.is_empty() {
        return Err(Denied::Empty);
    }
    match &*state.auth.users.get() {
        Snapshot::Users(_) => {}
        Snapshot::Missing => {
            return Err(Denied::Closed(
                "the accounts file is gone while someone is signed in".to_string(),
            ));
        }
        Snapshot::Broken(why) => return Err(Denied::Closed(why.clone())),
    }
    let me = ctx.me().to_string();
    let who = ctx.who();
    let reserved = match auth::reserve(state, ctx.peer) {
        Ok(r) => r,
        Err(wait) => {
            warn!(peer = %who, user = %me, wait, "current password not checked: too many wrong attempts from this address");
            return Err(Denied::Wait(wait));
        }
    };
    let now = accounts::now();
    let device = auth::cookie(&ctx.headers, cookies::DEVICE).map(<[u8]>::to_vec);
    let mut account: Option<accounts::User> = None;
    let slot = state.auth.attempts.reserve(&me, now, || {
        account = match &*state.auth.users.get() {
            Snapshot::Users(users) => users.get(&me).cloned(),
            _ => None,
        };
        match &account {
            Some(u) => {
                let trusted = device
                    .as_deref()
                    .is_some_and(|d| state.auth.devices.trusted(d, &u.name, now));
                (u.lockout(), trusted)
            }
            None => (lockout::State::default(), false),
        }
    });
    let slot = match slot {
        Ok(slot) => slot,
        Err(Decision::Wait(secs)) => {
            warn!(peer = %who, user = %me, wait = secs, "current password not checked: the account waits after failures in a row");
            return Err(Denied::Wait(secs));
        }
        Err(_) => {
            warn!(peer = %who, user = %me, "current password not checked: the account is disabled after too many wrong passwords");
            return Err(Denied::Wrong);
        }
    };
    let Some(user) = account else {
        // Removed meanwhile: its session ends with the next request.
        auth::refund(state, reserved);
        drop(slot);
        return Err(Denied::Wrong);
    };
    let st = Arc::clone(state);
    blocking(move || {
        let Some(permit) = st.login.hashing.acquire(st.login.hash_wait) else {
            auth::refund(&st, reserved);
            drop(slot);
            warn!(peer = %who, user = %user.name, "current password not checked: every password-hashing slot stayed busy; answered 503");
            return Err(Denied::Busy);
        };
        st.login.hashed.fetch_add(1, Ordering::Relaxed);
        let good = match password::verify(&user.hash, &current, &permit) {
            Ok(good) => good,
            Err(e) => {
                error!(user = %user.name, "the stored password of this account cannot be checked: {e:#}; reset it with `blue-onyx-prism user reset-password <name>`");
                false
            }
        };
        let store = Store::open(st.auth.data_root());
        if good {
            auth::refund(&st, reserved);
            if (user.failures > 0 || user.last_failure.is_some())
                && let Err(e) = store.record_success(&user.name)
            {
                warn!(user = %user.name, "could not reset the account's failure count: {e:#}");
            }
            let out = action(&st, &permit);
            st.auth.users.invalidate();
            drop(slot);
            return Ok(out);
        }
        if let Some(d) = &device {
            st.auth.devices.record_failure(d, &user.name, now);
        }
        match store.record_failure(&user.name, now) {
            Ok(Some(s)) if s.disabled => {
                warn!(peer = %who, user = %user.name, failures = s.failures, "wrong current password for an account change; the account is now disabled until `blue-onyx-prism user unlock <name>` on the server")
            }
            Ok(_) => {
                warn!(peer = %who, user = %user.name, "wrong current password for an account change")
            }
            Err(e) => {
                warn!(peer = %who, user = %user.name, "wrong current password for an account change; the failure could not be saved: {e:#}")
            }
        }
        st.auth.users.invalidate();
        drop(slot);
        Err(Denied::Wrong)
    })
    .await
}

/// The accounts as last read, when they can be used.
fn users(state: &AppState) -> Option<accounts::Users> {
    match &*state.auth.users.get() {
        Snapshot::Users(u) => Some(u.clone()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------------------------

/// The body of `POST /config/accounts/add`. No `Debug`: it holds passwords.
#[derive(serde::Deserialize)]
pub(super) struct AddForm {
    #[serde(default)]
    name: String,
    #[serde(default)]
    password: String,
    #[serde(default)]
    password2: String,
    #[serde(default)]
    current: String,
}

/// `POST /config/accounts/add`: a new account `name` with `password` (twice), confirmed with
/// the signed-in account's `current` password. The name and the policy are checked before the
/// current password, so a typo costs no attempt.
pub(super) async fn add(
    State(state): State<Arc<AppState>>,
    signed: Option<Extension<SignedIn>>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    headers: HeaderMap,
    form: Result<Form<AddForm>, FormRejection>,
) -> Response {
    let ctx = match Ctx::new(state, signed, peer, headers) {
        Ok(c) => c,
        Err(resp) => return *resp,
    };
    let f = match form {
        Ok(Form(f)) => f,
        Err(e) => return ctx.unreadable(&e).await,
    };
    let name = match accounts::normalize_name(f.name.trim()) {
        Ok(n) => n,
        Err(e) => {
            return ctx
                .fail(StatusCode::BAD_REQUEST, format!("Not added: {e}."))
                .await;
        }
    };
    if users(&ctx.state).is_some_and(|u| u.get(&name).is_some()) {
        return ctx
            .fail(
                StatusCode::CONFLICT,
                format!("Not added: {}.", Refused::NameTaken),
            )
            .await;
    }
    if f.password != f.password2 {
        return ctx
            .fail(
                StatusCode::BAD_REQUEST,
                "Not added: the two new passwords differ; type the same password twice."
                    .to_string(),
            )
            .await;
    }
    if let Err(e) = password::check(&f.password, &name) {
        return ctx
            .fail(StatusCode::BAD_REQUEST, format!("Not added: {e}."))
            .await;
    }
    let new = f.password;
    let n = name.clone();
    let done = reauth(&ctx, f.current, move |st, permit| {
        Store::open(st.auth.data_root()).add(&n, &new, permit, accounts::now())
    })
    .await;
    match done {
        Err(d) => ctx.denied(d).await,
        Ok(Err(e)) => ctx.change_failed("Not added", &e).await,
        Ok(Ok(user)) => {
            info!(by = %ctx.me(), peer = %ctx.who(), account = %user.name, "web UI account added");
            ctx.ok(format!(
                "Added the account {}. It can sign in now and do everything you can.",
                user.name
            ))
            .await
        }
    }
}

/// The body of `POST /config/accounts/remove` and `.../unlock`. No `Debug`.
#[derive(serde::Deserialize)]
pub(super) struct TargetForm {
    #[serde(default)]
    name: String,
    #[serde(default)]
    current: String,
}

/// The account `typed` names among the accounts, or the answer that it does not.
async fn target(ctx: &Ctx, typed: &str, what: &str) -> Result<String, Box<Response>> {
    let known = accounts::normalize_name(typed.trim())
        .ok()
        .filter(|n| users(&ctx.state).is_some_and(|u| u.get(n).is_some()));
    match known {
        Some(n) => Ok(n),
        None => Err(Box::new(
            ctx.fail(
                StatusCode::BAD_REQUEST,
                format!("{what}: choose one of the accounts in the list."),
            )
            .await,
        )),
    }
}

/// `POST /config/accounts/remove`: remove the account `name` (never the last one), confirmed
/// with the signed-in account's `current` password; its sessions and device cookies end at once.
/// Removing your own account signs this browser out.
pub(super) async fn remove(
    State(state): State<Arc<AppState>>,
    signed: Option<Extension<SignedIn>>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    headers: HeaderMap,
    form: Result<Form<TargetForm>, FormRejection>,
) -> Response {
    let ctx = match Ctx::new(state, signed, peer, headers) {
        Ok(c) => c,
        Err(resp) => return *resp,
    };
    let f = match form {
        Ok(Form(f)) => f,
        Err(e) => return ctx.unreadable(&e).await,
    };
    let name = match target(&ctx, &f.name, "Not removed").await {
        Ok(n) => n,
        Err(resp) => return *resp,
    };
    if users(&ctx.state).is_some_and(|u| u.users.len() <= 1) {
        return ctx
            .fail(
                StatusCode::CONFLICT,
                format!("Not removed: {}.", Refused::LastAccount),
            )
            .await;
    }
    let n = name.clone();
    let done = reauth(&ctx, f.current, move |st, _permit| {
        Store::open(st.auth.data_root()).remove(&n)?;
        let now = accounts::now();
        let sessions = st.auth.sessions.end_user(&n, now);
        let devices = st.auth.devices.forget_user(&n, now);
        Ok::<_, anyhow::Error>((sessions, devices))
    })
    .await;
    match done {
        Err(d) => ctx.denied(d).await,
        Ok(Err(e)) => ctx.change_failed("Not removed", &e).await,
        Ok(Ok((sessions, devices))) => {
            info!(by = %ctx.me(), peer = %ctx.who(), account = %name, sessions, devices, "web UI account removed");
            if name == ctx.me() {
                return ctx.signed_out(format!(
                    "Removed your own account {name}; this browser is signed out."
                ));
            }
            ctx.ok(format!(
                "Removed the account {name}; its sessions ended ({sessions} signed out)."
            ))
            .await
        }
    }
}

/// `POST /config/accounts/unlock`: clear the wrong-password count and the disabled flag of the
/// account `name`, confirmed with the signed-in account's `current` password.
pub(super) async fn unlock(
    State(state): State<Arc<AppState>>,
    signed: Option<Extension<SignedIn>>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    headers: HeaderMap,
    form: Result<Form<TargetForm>, FormRejection>,
) -> Response {
    let ctx = match Ctx::new(state, signed, peer, headers) {
        Ok(c) => c,
        Err(resp) => return *resp,
    };
    let f = match form {
        Ok(Form(f)) => f,
        Err(e) => return ctx.unreadable(&e).await,
    };
    let name = match target(&ctx, &f.name, "Not unlocked").await {
        Ok(n) => n,
        Err(resp) => return *resp,
    };
    let n = name.clone();
    let done = reauth(&ctx, f.current, move |st, _permit| {
        Store::open(st.auth.data_root()).unlock(&n)
    })
    .await;
    match done {
        Err(d) => ctx.denied(d).await,
        Ok(Err(e)) => ctx.change_failed("Not unlocked", &e).await,
        Ok(Ok(_)) => {
            info!(by = %ctx.me(), peer = %ctx.who(), account = %name, "web UI account unlocked");
            ctx.ok(format!(
                "Unlocked the account {name}: its wrong passwords are forgotten and it can sign \
                 in again."
            ))
            .await
        }
    }
}

/// The body of `POST /config/accounts/password`. No `Debug`.
#[derive(serde::Deserialize)]
pub(super) struct PasswordForm {
    #[serde(default)]
    current: String,
    #[serde(default)]
    password: String,
    #[serde(default)]
    password2: String,
}

/// `POST /config/accounts/password`: the signed-in account's new `password` (twice), confirmed
/// with its `current` one. Every other session of the account ends; this browser stays signed
/// in under a new session id (its device cookie stays).
pub(super) async fn change_password(
    State(state): State<Arc<AppState>>,
    signed: Option<Extension<SignedIn>>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    headers: HeaderMap,
    form: Result<Form<PasswordForm>, FormRejection>,
) -> Response {
    let ctx = match Ctx::new(state, signed, peer, headers) {
        Ok(c) => c,
        Err(resp) => return *resp,
    };
    let f = match form {
        Ok(Form(f)) => f,
        Err(e) => return ctx.unreadable(&e).await,
    };
    if f.password != f.password2 {
        return ctx
            .fail(
                StatusCode::BAD_REQUEST,
                "Not changed: the two new passwords differ; type the same password twice."
                    .to_string(),
            )
            .await;
    }
    if let Err(e) = password::check(&f.password, ctx.me()) {
        return ctx
            .fail(StatusCode::BAD_REQUEST, format!("Not changed: {e}."))
            .await;
    }
    let me = ctx.me().to_string();
    let current_session =
        auth::cookie(&ctx.headers, cookies::SESSION).and_then(SessionId::from_presented);
    let new = f.password;
    let done = reauth(&ctx, f.current, move |st, permit| {
        let now = accounts::now();
        let user = Store::open(st.auth.data_root()).set_password(&me, &new, permit, now)?;
        let sessions = &st.auth.sessions;
        // This session moves to the new generation and gets a new id; the others end. A request
        // of this browser that saw the new file first may have ended it already: then it starts
        // afresh (the password was just shown, as at a sign-in).
        let kept = current_session.and_then(|cur| {
            sessions
                .end_all_except(&me, &cur, user.generation, now)
                .then(|| sessions.rotate(&cur, now))
                .flatten()
        });
        let session = kept.unwrap_or_else(|| {
            sessions.end_user(&me, now);
            sessions.create(&me, user.generation, now)
        });
        let max_age = sessions.record(&session).map_or(ABSOLUTE_SECS, |r| {
            r.created
                .saturating_add(ABSOLUTE_SECS)
                .saturating_sub(now)
                .max(1)
        });
        Ok::<_, anyhow::Error>(cookies::session_set_cookie(&session, max_age))
    })
    .await;
    match done {
        Err(d) => ctx.denied(d).await,
        Ok(Err(e)) => ctx.change_failed("Not changed", &e).await,
        Ok(Ok(set_cookie)) => {
            info!(by = %ctx.me(), peer = %ctx.who(), "web UI password changed; the account's other sessions ended");
            let mut resp = ctx
                .ok(
                    "Your password was changed. Every other browser signed in to your account \
                     was signed out; this one stays signed in."
                        .to_string(),
                )
                .await;
            if let Some(v) = auth::set_cookie(&set_cookie) {
                resp.headers_mut().append(header::SET_COOKIE, v);
            }
            auth::no_store(resp)
        }
    }
}

/// The body of `POST /config/accounts/sign-out-everywhere`. No `Debug`.
#[derive(serde::Deserialize)]
pub(super) struct CurrentForm {
    #[serde(default)]
    current: String,
}

/// `POST /config/accounts/sign-out-everywhere`: end every session of the signed-in account,
/// this one included (its generation goes up, so a session another process or a copy of the
/// file holds ends too), confirmed with its `current` password.
pub(super) async fn sign_out_everywhere(
    State(state): State<Arc<AppState>>,
    signed: Option<Extension<SignedIn>>,
    peer: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    headers: HeaderMap,
    form: Result<Form<CurrentForm>, FormRejection>,
) -> Response {
    let ctx = match Ctx::new(state, signed, peer, headers) {
        Ok(c) => c,
        Err(resp) => return *resp,
    };
    let f = match form {
        Ok(Form(f)) => f,
        Err(e) => return ctx.unreadable(&e).await,
    };
    let me = ctx.me().to_string();
    let done = reauth(&ctx, f.current, move |st, _permit| {
        Store::open(st.auth.data_root()).bump_generation(&me)?;
        Ok::<_, anyhow::Error>(st.auth.sessions.end_user(&me, accounts::now()))
    })
    .await;
    match done {
        Err(d) => ctx.denied(d).await,
        Ok(Err(e)) => ctx.change_failed("Not signed out", &e).await,
        Ok(Ok(sessions)) => {
            info!(by = %ctx.me(), peer = %ctx.who(), sessions, "web UI account signed out everywhere");
            ctx.signed_out(format!(
                "Signed out everywhere: {sessions} session(s) ended, this one included."
            ))
        }
    }
}

#[cfg(test)]
mod tests;
