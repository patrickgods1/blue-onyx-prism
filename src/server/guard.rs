//! Request guards of the router: the admin routes' cross-site check (SV-01), their `Host` check
//! against DNS rebinding and the security response headers (SV-09).
//!
//! # Cross-site requests (CSRF)
//!
//! A loopback peer needs no login (with `allow_remote_admin`, a remote peer needs the admin
//! token, which a browser then sends by itself: [`super::auth`]), and may change the config, start
//! downloads or restart the server. A web page on any site can make
//! the admin's browser send such a request (an auto-submitted form needs no CORS preflight),
//! so [`csrf`] refuses a state-changing request (POST/PUT/PATCH/DELETE) that a browser marks as
//! coming from another site:
//!
//! - `Sec-Fetch-Site: cross-site` (every current browser sends the header);
//! - an `Origin` (else `Referer`) header whose scheme authority is not the request's `Host`
//!   (`Origin: null` counts as foreign unless the browser marks the request
//!   `Sec-Fetch-Site: same-origin`; `Host` must be present).
//!
//! A plain HTML form POST from a page served with `Referrer-Policy: no-referrer` carries
//! `Origin: null` even to its own origin (the Fetch spec ties the serialized origin to the
//! referrer policy), so pages are served with `same-origin` instead, and `null` with a
//! same-origin `Sec-Fetch-Site` (a header pages cannot set) passes.
//!
//! A request with none of those headers passes. Non-browser clients (curl, scripts and tools on
//! the machine, the CLI) send none of them, while a browser sends `Origin` on every cross-origin
//! POST and `Sec-Fetch-Site` on every request, so the absence of all three means the request did
//! not come from a browser page. GET requests are not checked here: they change nothing, and a
//! cross-site page cannot read their answer (no CORS headers are sent).
//!
//! # DNS rebinding
//!
//! The cross-site check compares `Origin` with `Host`, and that says nothing about a rebound
//! page. A hostile site serves a page from `http://rebind.attacker.test:32168`, then changes
//! that name's DNS answer to `127.0.0.1` (or a LAN address). The browser still thinks it talks
//! to the same site, so it lets the page's scripts read the answers, and every request carries
//! `Host: rebind.attacker.test:32168` with `Origin: http://rebind.attacker.test:32168`, or
//! `Sec-Fetch-Site: same-origin`: `Origin` equals `Host` because the two come from the same
//! lie. The TCP peer is the victim's own machine, so the loopback rule passes too.
//!
//! Only the name itself gives the attack away: it needs a DNS name the attacker controls. So
//! [`foreign_host`], applied by `admin_only` to every admin route and every method (and to
//! `/login` and `/logout`), passes an
//! IP literal (nothing to rebind, so LAN administration by address keeps working), `localhost`,
//! and the names listed in `allowed_hosts`, and refuses any other name. It reads the `Host`
//! header, else the URI authority (HTTP/2). A request with neither passes: a browser always
//! sends one, so this is not a page, and [`cross_site`] refuses a state change with no `Host`.
//!
//! # Response headers
//!
//! [`security_headers`] adds `X-Content-Type-Options: nosniff`, `Referrer-Policy: same-origin`
//! (no referrer to other sites) and `X-Frame-Options: DENY` to every answer, [`CSP`] to HTML pages (every script and style
//! must come from `/static/`), `charset=utf-8` to HTML without one, and `Cache-Control:
//! no-store` to JSON that sets no cache policy of its own.

use super::{MessageTemplate, render, wants_html};
use crate::sync::LockExt;
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::{Json, middleware::Next};
use std::net::{Ipv4Addr, Ipv6Addr};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tracing::warn;

/// Content-Security-Policy of the HTML pages: everything from this origin, inline scripts
/// and styles refused, no framing, forms post here only.
pub(crate) const CSP: &str = "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; \
                              script-src 'self'; frame-ancestors 'none'; form-action 'self'";

/// Why [`csrf`] refused a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CrossSite {
    /// `Sec-Fetch-Site: cross-site`.
    FetchSite,
    /// `Host` missing.
    NoHost,
    /// `Origin`/`Referer` (first field) names another authority than `Host` (second field).
    Mismatch(String, String),
}

impl std::fmt::Display for CrossSite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CrossSite::FetchSite => write!(f, "the browser marked the request as cross-site"),
            CrossSite::NoHost => write!(f, "the request has no Host header"),
            CrossSite::Mismatch(origin, host) => write!(
                f,
                "the request's origin ({origin}) is not this server's address ({host})"
            ),
        }
    }
}

/// `scheme` and lowercased `authority` of `http(s)://authority[/...]`; None for anything else
/// (`null`, a bare word, another scheme).
fn authority(url: &str) -> Option<(&str, String)> {
    let (scheme, rest) = url.trim().split_once("://")?;
    let scheme = scheme.trim().to_ascii_lowercase();
    if scheme != "http" && scheme != "https" {
        return None;
    }
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let auth = rest.get(..end)?.trim();
    if auth.is_empty() {
        return None;
    }
    // User info never appears in Origin; a Referer may carry it (`user@host`): drop it.
    let auth = auth.rsplit_once('@').map_or(auth, |(_, h)| h);
    let scheme = if scheme == "https" { "https" } else { "http" };
    Some((scheme, auth.to_ascii_lowercase()))
}

/// `host[:port]` without the scheme's default port, lowercased.
fn normalize_authority(auth: &str, scheme: &str) -> String {
    let auth = auth.trim().to_ascii_lowercase();
    let default = if scheme == "https" { ":443" } else { ":80" };
    match auth.strip_suffix(default) {
        // `[::1]:80` -> `[::1]`; `host:80` -> `host`.
        Some(h) if !h.is_empty() => h.to_string(),
        _ => auth,
    }
}

/// Whether the browser marked the request `Sec-Fetch-Site: cross-site` (also used by GET
/// routes whose answer leaks something or whose work is costly, such as the import preview).
pub(crate) fn fetch_site_cross(headers: &HeaderMap) -> bool {
    headers
        .get("sec-fetch-site")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("cross-site"))
}

/// The cross-site decision for the request headers of a state-changing request (see the
/// module doc). `Ok` passes.
pub(crate) fn cross_site(headers: &HeaderMap) -> Result<(), CrossSite> {
    let text = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
    if fetch_site_cross(headers) {
        return Err(CrossSite::FetchSite);
    }
    let Some(host) = text("host").map(str::trim).filter(|h| !h.is_empty()) else {
        return Err(CrossSite::NoHost);
    };
    let source = text("origin").or_else(|| text("referer"));
    let Some(source) = source.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(());
    };
    let same_origin =
        text("sec-fetch-site").is_some_and(|v| v.trim().eq_ignore_ascii_case("same-origin"));
    if source.eq_ignore_ascii_case("null") && same_origin {
        return Ok(());
    }
    let Some((scheme, auth)) = authority(source) else {
        return Err(CrossSite::Mismatch(source.to_string(), host.to_string()));
    };
    if normalize_authority(&auth, scheme) == normalize_authority(host, scheme) {
        Ok(())
    } else {
        Err(CrossSite::Mismatch(auth, host.to_string()))
    }
}

/// Headers a reverse proxy adds: a request carrying one did not come straight from a browser on
/// this machine, whatever the TCP peer is. Read in that direction only: the header takes the
/// trust of a loopback peer away and its value is never believed, since any client can send it.
const PROXY_HEADERS: [&str; 3] = ["x-forwarded-for", "forwarded", "x-real-ip"];

/// The first [`PROXY_HEADERS`] entry the request carries.
pub(super) fn proxy_header(headers: &HeaderMap) -> Option<&'static str> {
    PROXY_HEADERS.into_iter().find(|h| headers.contains_key(*h))
}

/// Most characters of a client's `Host` repeated in a refusal or in the log.
const MAX_HOST_SHOWN: usize = 100;

/// At most one log line per this long for refused `Host` names: every request of a hostile
/// page would otherwise write one.
const HOST_WARN_EVERY: Duration = Duration::from_secs(60);

/// The request's host: the `Host` header, else the URI's authority (HTTP/2). A header that is
/// not text counts as present, so it cannot pass for a missing one.
pub(super) fn request_host(headers: &HeaderMap, uri: &Uri) -> Option<String> {
    headers
        .get(header::HOST)
        .map(|v| String::from_utf8_lossy(v.as_bytes()).into_owned())
        .or_else(|| uri.authority().map(|a| a.as_str().to_string()))
}

/// The host part of a `Host` value (`localhost:32168` -> `localhost`, `[::1]:80` -> `::1`,
/// lowercased); None when it is not well formed: no name, or a port that is not a number.
pub(super) fn host_part(host: &str) -> Option<String> {
    let h = host.trim();
    let (name, port) = if let Some(rest) = h.strip_prefix('[') {
        let (v6, after) = rest.split_once(']')?;
        match after.strip_prefix(':') {
            Some(port) => (v6, port),
            None if after.is_empty() => (v6, ""),
            None => return None,
        }
    } else if h.matches(':').count() > 1 {
        // A bare IPv6 literal (`::1`), no port.
        (h, "")
    } else {
        h.split_once(':').unwrap_or((h, ""))
    };
    let good = !name.is_empty() && port.bytes().all(|b| b.is_ascii_digit());
    good.then(|| name.to_ascii_lowercase())
}

/// Whether the admin routes answer under this `Host` value (see the module doc): an IPv4
/// literal, a bracketed IPv6 literal, `localhost` or a name of `allowed`, the last two in any
/// case and with one trailing dot. Exact names only: `localhost.evil.test` is a foreign name,
/// and so is a subdomain of a listed name.
pub(super) fn host_allowed(host: &str, allowed: &[String]) -> bool {
    let Some(name) = host_part(host) else {
        return false;
    };
    if host.trim_start().starts_with('[') {
        return name.parse::<Ipv6Addr>().is_ok();
    }
    if name.parse::<Ipv4Addr>().is_ok() {
        return true;
    }
    let name = name.strip_suffix('.').unwrap_or(&name);
    name == "localhost" || allowed.iter().any(|a| a.eq_ignore_ascii_case(name))
}

/// The request's host as it may be shown, when the admin routes must not answer under it
/// ([`host_allowed`]); None lets the request through. No host at all passes (module doc).
pub(super) fn foreign_host(headers: &HeaderMap, uri: &Uri, allowed: &[String]) -> Option<String> {
    let host = request_host(headers, uri)?;
    (!host_allowed(&host, allowed)).then(|| {
        host.chars()
            .filter(|c| !c.is_control())
            .take(MAX_HOST_SHOWN)
            .collect()
    })
}

/// What decides whether a repeated warning is logged now.
#[derive(Debug)]
pub(super) struct WarnThrottle {
    last: Option<Instant>,
    dropped: u64,
}

impl WarnThrottle {
    pub(super) const fn new() -> Self {
        Self {
            last: None,
            dropped: 0,
        }
    }

    /// `Some(lines dropped since the last one)` when a line may be logged at `now`.
    pub(super) fn allow_at(&mut self, now: Instant, every: Duration) -> Option<u64> {
        if self.last.is_some_and(|t| now.duration_since(t) < every) {
            self.dropped = self.dropped.saturating_add(1);
            return None;
        }
        self.last = Some(now);
        Some(std::mem::take(&mut self.dropped))
    }
}

/// The refused-`Host` warnings of this process.
static HOST_WARNS: Mutex<WarnThrottle> = Mutex::new(WarnThrottle::new());

/// 403 for a `host` the admin routes do not answer under: an HTML page for browsers, else
/// JSON. `host` comes from [`foreign_host`].
pub(super) fn host_refused(headers: &HeaderMap, path: &str, host: &str) -> Response {
    let dropped = HOST_WARNS
        .lock_recover()
        .allow_at(Instant::now(), HOST_WARN_EVERY);
    if let Some(dropped) = dropped {
        warn!(%host, path, dropped, "admin route refused for an unknown host name (DNS rebinding?)");
    }
    forbidden(
        headers,
        format!(
            "This server does not answer administration requests under the name {host}: a web \
             page can point a name of its own at this machine (DNS rebinding), so only IP \
             addresses, localhost and the names in allowed_hosts are served. If this is your \
             name for the server, add it to \"allowed_hosts\" in the config (on the Config page, \
             opened by IP address or as localhost, or in the config file); otherwise open the \
             page by IP address."
        ),
    )
}

/// A 403 with `msg`: an HTML page for browsers, else JSON.
pub(super) fn forbidden(headers: &HeaderMap, msg: String) -> Response {
    refused(StatusCode::FORBIDDEN, headers, "Forbidden", msg)
}

/// `status` with `msg`: an HTML page under `heading` for browsers, else JSON.
pub(super) fn refused(
    status: StatusCode,
    headers: &HeaderMap,
    heading: &str,
    msg: String,
) -> Response {
    if wants_html(headers) {
        let page = MessageTemplate {
            nav: "",
            version: crate::VERSION,
            heading: heading.into(),
            message: msg,
            refresh_secs: 0,
            refresh_url: String::new(),
            restarting: false,
            generation: 0,
        };
        (status, render(&page)).into_response()
    } else {
        (
            status,
            Json(serde_json::json!({ "success": false, "message": msg, "error": msg })),
        )
            .into_response()
    }
}

/// Middleware of the admin routes: 403 (an HTML page for browsers, else JSON) for a
/// state-changing request that comes from another site (see the module doc).
pub(super) async fn csrf(req: axum::extract::Request, next: Next) -> Response {
    if !matches!(
        *req.method(),
        Method::POST | Method::PUT | Method::PATCH | Method::DELETE
    ) {
        return next.run(req).await;
    }
    let why = match cross_site(req.headers()) {
        Ok(()) => return next.run(req).await,
        Err(why) => why,
    };
    warn!(path = %req.uri().path(), %why, "cross-site request to an admin route refused");
    let msg = format!(
        "Cross-site request refused: {why}. Changes through the admin API must come from this \
         server's own pages (same scheme, host and port as the address in the browser) or from \
         a non-browser client."
    );
    forbidden(req.headers(), msg)
}

/// `map_response` middleware of the whole router (see the module doc).
pub(super) async fn security_headers(mut resp: Response) -> Response {
    let h = resp.headers_mut();
    h.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    h.insert("referrer-policy", HeaderValue::from_static("same-origin"));
    h.insert("x-frame-options", HeaderValue::from_static("DENY"));
    let ct = h
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.trim().to_ascii_lowercase())
        .unwrap_or_default();
    if ct.starts_with("text/html") {
        h.insert("content-security-policy", HeaderValue::from_static(CSP));
        if !ct.contains("charset=") {
            h.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/html; charset=utf-8"),
            );
        }
    } else if ct.starts_with("application/json") && !h.contains_key(header::CACHE_CONTROL) {
        h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    }
    resp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.append(
                header::HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    #[test]
    fn same_origin_and_headerless_requests_pass() {
        let host = ("host", "localhost:32168");
        assert_eq!(cross_site(&headers(&[host])), Ok(()));
        assert_eq!(
            cross_site(&headers(&[host, ("origin", "http://localhost:32168")])),
            Ok(())
        );
        assert_eq!(
            cross_site(&headers(&[host, ("origin", "HTTP://LocalHost:32168")])),
            Ok(())
        );
        assert_eq!(
            cross_site(&headers(&[
                host,
                ("referer", "http://localhost:32168/config#models")
            ])),
            Ok(())
        );
        assert_eq!(
            cross_site(&headers(&[host, ("sec-fetch-site", "same-origin")])),
            Ok(())
        );
        // A form POST from a page served with `Referrer-Policy: no-referrer` (an old page):
        // `Origin: null`, but the browser vouches that it is same-origin.
        assert_eq!(
            cross_site(&headers(&[
                host,
                ("origin", "null"),
                ("sec-fetch-site", "same-origin")
            ])),
            Ok(())
        );
        // Default ports: `http://host` is `host:80`; TLS behind a proxy that keeps Host.
        assert_eq!(
            cross_site(&headers(&[
                ("host", "cam.local"),
                ("origin", "http://cam.local:80")
            ])),
            Ok(())
        );
        assert_eq!(
            cross_site(&headers(&[
                ("host", "cam.local"),
                ("origin", "https://cam.local:443")
            ])),
            Ok(())
        );
        assert_eq!(
            cross_site(&headers(&[
                ("host", "[::1]:32168"),
                ("origin", "http://[::1]:32168")
            ])),
            Ok(())
        );
        // Origin wins over Referer.
        assert_eq!(
            cross_site(&headers(&[
                host,
                ("origin", "http://localhost:32168"),
                ("referer", "http://evil.example/")
            ])),
            Ok(())
        );
    }

    #[test]
    fn foreign_requests_are_refused() {
        let host = ("host", "localhost:32168");
        assert_eq!(
            cross_site(&headers(&[host, ("sec-fetch-site", "cross-site")])),
            Err(CrossSite::FetchSite)
        );
        assert_eq!(
            cross_site(&headers(&[
                host,
                ("origin", "http://localhost:32168"),
                ("sec-fetch-site", "Cross-Site")
            ])),
            Err(CrossSite::FetchSite)
        );
        assert_eq!(
            cross_site(&headers(&[host, ("origin", "http://evil.example")])),
            Err(CrossSite::Mismatch(
                "evil.example".into(),
                "localhost:32168".into()
            ))
        );
        // Another port of the same host is another origin; so is `null`; so is another name
        // than the Host. (A DNS-rebound page is not caught here: its Origin equals its Host,
        // see the module doc. `foreign_host` stops it.)
        assert!(cross_site(&headers(&[host, ("origin", "http://localhost:8080")])).is_err());
        assert!(cross_site(&headers(&[host, ("origin", "null")])).is_err());
        assert!(
            cross_site(&headers(&[
                host,
                ("origin", "null"),
                ("sec-fetch-site", "same-site")
            ]))
            .is_err()
        );
        assert!(cross_site(&headers(&[host, ("referer", "http://attacker.test/x")])).is_err());
        assert!(cross_site(&headers(&[host, ("origin", "file://x")])).is_err());
        assert_eq!(
            cross_site(&headers(&[("origin", "http://localhost:32168")])),
            Err(CrossSite::NoHost)
        );
        assert_eq!(cross_site(&headers(&[])), Err(CrossSite::NoHost));
    }

    #[test]
    fn host_parts() {
        for (h, want) in [
            ("localhost:32168", Some("localhost")),
            ("LocalHost", Some("localhost")),
            ("LOCALHOST.:32168", Some("localhost.")),
            ("127.0.0.1:80", Some("127.0.0.1")),
            ("[::1]:32168", Some("::1")),
            ("[::1]", Some("::1")),
            ("::1", Some("::1")),
            ("evil.example:32168", Some("evil.example")),
            ("localhost.evil.example", Some("localhost.evil.example")),
            ("[::1]x", None),
            ("[::1]:x", None),
            ("localhost:x", None),
            ("localhost:80@evil.test", None),
            ("", None),
            (":80", None),
        ] {
            assert_eq!(host_part(h).as_deref(), want, "{h}");
        }
    }

    #[test]
    fn hosts_the_admin_routes_answer_under() {
        let allowed = ["nas.home.arpa".to_string(), "Prism_1".to_string()];
        for h in [
            "localhost",
            "localhost:32168",
            "LOCALHOST.:32168",
            "127.0.0.1",
            "127.0.0.1:32168",
            "192.168.1.10:32168",
            "[::1]",
            "[::1]:32168",
            "[fe80::1]:32168",
            "[::ffff:127.0.0.1]:32168",
            "nas.home.arpa",
            "NAS.HOME.ARPA.:80",
            "prism_1:32168",
        ] {
            assert!(host_allowed(h, &allowed), "{h}");
        }
        for h in [
            "",
            ":32168",
            "rebind.attacker.test:32168",
            "localhost.evil.test",
            "localhost.evil.test:32168",
            "evil-localhost",
            "x.nas.home.arpa",
            "nas.home.arpa.evil.test",
            // Not IP literals: a name made of digits resolves through DNS like any other. A
            // browser normalizes the numeric forms before it sends them.
            "127.0.0.1.evil.test",
            "127.0.0.1.",
            "127.1",
            "2130706433",
            "0x7f.0.0.1",
            // Brackets hold an IPv6 address and nothing else; an address needs its brackets.
            "[localhost]:32168",
            "[nas.home.arpa]",
            "[evil.test]",
            "::1",
            "localhost:80@evil.test",
            "localhost:evil.test",
            "localhost..",
        ] {
            assert!(!host_allowed(h, &allowed), "{h}");
        }
        assert!(!host_allowed("nas.home.arpa", &[]));
    }

    #[test]
    fn foreign_host_reads_the_header_then_the_authority() {
        let uri = |s: &str| s.parse::<Uri>().unwrap();
        let allowed: &[String] = &[];
        let host = |v: &str| headers(&[("host", v)]);
        assert_eq!(foreign_host(&host("localhost"), &uri("/x"), allowed), None);
        assert_eq!(
            foreign_host(&host("evil.test:1"), &uri("/x"), allowed).as_deref(),
            Some("evil.test:1")
        );
        // The header wins over an absolute request target.
        assert!(foreign_host(&host("evil.test"), &uri("http://localhost/x"), allowed).is_some());
        // No header: the authority, else nothing to judge.
        let none = HeaderMap::new();
        assert_eq!(
            foreign_host(&none, &uri("http://localhost/x"), allowed),
            None
        );
        assert!(foreign_host(&none, &uri("http://evil.test/x"), allowed).is_some());
        assert_eq!(foreign_host(&none, &uri("/x"), allowed), None);
        // A header that is present but not text is refused, not taken for a missing one.
        let mut bytes = HeaderMap::new();
        bytes.insert(
            header::HOST,
            HeaderValue::from_bytes(b"evil\xff.test").unwrap(),
        );
        assert!(foreign_host(&bytes, &uri("/x"), allowed).is_some());
        // Empty is present, and no name.
        assert!(foreign_host(&host(""), &uri("/x"), allowed).is_some());
        // Short and without control characters, whatever the client sent.
        let shown = foreign_host(&host(&"a".repeat(300)), &uri("/x"), allowed).unwrap();
        assert_eq!(shown.chars().count(), MAX_HOST_SHOWN);
    }

    #[test]
    fn repeated_warnings_are_throttled() {
        let t0 = Instant::now();
        let every = Duration::from_secs(60);
        let mut w = WarnThrottle {
            last: None,
            dropped: 0,
        };
        assert_eq!(w.allow_at(t0, every), Some(0));
        for s in 0..59 {
            assert_eq!(w.allow_at(t0 + Duration::from_secs(s), every), None);
        }
        assert_eq!(w.allow_at(t0 + every, every), Some(59));
        assert_eq!(w.allow_at(t0 + every, every), None);
        assert_eq!(w.allow_at(t0 + every * 2, every), Some(1));
    }

    #[test]
    fn authority_parsing() {
        assert_eq!(
            authority("http://A.b:1/x?y"),
            Some(("http", "a.b:1".to_string()))
        );
        assert_eq!(
            authority("https://u:p@h.i/"),
            Some(("https", "h.i".to_string()))
        );
        assert_eq!(authority("null"), None);
        assert_eq!(authority("ftp://x"), None);
        assert_eq!(authority("http:///x"), None);
        assert_eq!(normalize_authority("H:80", "http"), "h");
        assert_eq!(normalize_authority("h:80", "https"), "h:80");
        assert_eq!(normalize_authority(":80", "http"), ":80");
    }
}
