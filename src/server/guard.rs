//! Request guards of the router: the admin routes' cross-site check (SV-01) and the security
//! response headers (SV-09).
//!
//! # Cross-site requests (CSRF)
//!
//! The admin routes have no login: a loopback peer (or anyone, with `allow_remote_admin`) may
//! change the config, start downloads or restart the server. A web page on any site can make
//! the admin's browser send such a request (an auto-submitted form needs no CORS preflight),
//! so [`csrf`] refuses a state-changing request (POST/PUT/PATCH/DELETE) that a browser marks as
//! coming from another site:
//!
//! - `Sec-Fetch-Site: cross-site` (every current browser sends the header);
//! - an `Origin` (else `Referer`) header whose scheme authority is not the request's `Host`
//!   (`Origin: null` counts as foreign; `Host` must be present).
//!
//! A request with none of those headers passes. Non-browser clients (curl, scripts and tools on
//! the machine, the CLI) send none of them, while a browser sends `Origin` on every cross-origin
//! POST and `Sec-Fetch-Site` on every request, so the absence of all three means the request did
//! not come from a browser page. GET requests are not checked here: they change nothing, and a
//! cross-site page cannot read their answer (no CORS headers are sent).
//!
//! # Response headers
//!
//! [`security_headers`] adds `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`
//! and `X-Frame-Options: DENY` to every answer, [`CSP`] to HTML pages (every script and style
//! must come from `/static/`), `charset=utf-8` to HTML without one, and `Cache-Control:
//! no-store` to JSON that sets no cache policy of its own.

use super::{MessageTemplate, render, wants_html};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::{Json, middleware::Next};
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

/// The cross-site decision for the request headers of a state-changing request (see the
/// module doc). `Ok` passes.
pub(crate) fn cross_site(headers: &HeaderMap) -> Result<(), CrossSite> {
    let text = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
    if text("sec-fetch-site").is_some_and(|v| v.trim().eq_ignore_ascii_case("cross-site")) {
        return Err(CrossSite::FetchSite);
    }
    let Some(host) = text("host").map(str::trim).filter(|h| !h.is_empty()) else {
        return Err(CrossSite::NoHost);
    };
    let source = text("origin").or_else(|| text("referer"));
    let Some(source) = source.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(());
    };
    let Some((scheme, auth)) = authority(source) else {
        return Err(CrossSite::Mismatch(source.to_string(), host.to_string()));
    };
    if normalize_authority(&auth, scheme) == normalize_authority(host, scheme) {
        Ok(())
    } else {
        Err(CrossSite::Mismatch(auth, host.to_string()))
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
    if wants_html(req.headers()) {
        let page = MessageTemplate {
            nav: "",
            version: crate::VERSION,
            heading: "Forbidden".into(),
            message: msg,
            refresh_secs: 0,
            refresh_url: String::new(),
            restarting: false,
            generation: 0,
        };
        (StatusCode::FORBIDDEN, render(&page)).into_response()
    } else {
        (
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({ "success": false, "message": msg, "error": msg })),
        )
            .into_response()
    }
}

/// `map_response` middleware of the whole router (see the module doc).
pub(super) async fn security_headers(mut resp: Response) -> Response {
    let h = resp.headers_mut();
    h.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    h.insert("referrer-policy", HeaderValue::from_static("no-referrer"));
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
        // Another port of the same host is another origin; so is `null`; so is a DNS-rebound
        // name that does not match the Host the page was served under.
        assert!(cross_site(&headers(&[host, ("origin", "http://localhost:8080")])).is_err());
        assert!(cross_site(&headers(&[host, ("origin", "null")])).is_err());
        assert!(cross_site(&headers(&[host, ("referer", "http://attacker.test/x")])).is_err());
        assert!(cross_site(&headers(&[host, ("origin", "file://x")])).is_err());
        assert_eq!(
            cross_site(&headers(&[("origin", "http://localhost:32168")])),
            Err(CrossSite::NoHost)
        );
        assert_eq!(cross_site(&headers(&[])), Err(CrossSite::NoHost));
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
