//! The JSON error answers and the JSON body reader every route shares (QA-131, QA-221).
//!
//! Two shapes, on purpose: [`json_fail`] (`{success: false, message}`) is what the page routes
//! (review, refine, training, benchmark, ...) answer and what their scripts read; [`json_error`]
//! (`{success: false, message, error}`) is the server core's refusals (sign-in, network and host
//! rules, fallback, panics, static files), the Config page's live errors, `/logs.json` and the
//! benchmark history routes, whose callers read `error`. Both carry `message`, so a client that
//! reads `message` reads either.
//!
//! [`Refused`] is a refusal not built yet: a route returning [`Answer`] checks with `?`.
//! [`json_body`] reads a POST's JSON body: 413 over the route's cap, 415 without a JSON content
//! type, 400 when it does not parse.

#![allow(
    dead_code,
    reason = "phase 9: the copies move over to these; the lead removes this allow"
)]

use axum::Json;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

/// `{"success": false, "message": ...}` with `status`.
pub(super) fn json_fail(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(serde_json::json!({ "success": false, "message": message.into() })),
    )
        .into_response()
}

/// `{"success": false, "message": ..., "error": ...}` with `status` (both keys hold the same
/// text).
pub(super) fn json_error(status: StatusCode, message: impl Into<String>) -> Response {
    let message = message.into();
    (
        status,
        Json(serde_json::json!({ "success": false, "message": message, "error": message })),
    )
        .into_response()
}

/// An error answer not built yet (status, message): see [`json_fail`].
pub(super) type Problem = (StatusCode, String);

/// A refused request, answered as [`json_fail`].
#[derive(Debug)]
pub(super) struct Refused(pub(super) Problem);

impl Refused {
    pub(super) fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self((status, message.into()))
    }
}

impl From<Problem> for Refused {
    fn from(p: Problem) -> Self {
        Self(p)
    }
}

impl IntoResponse for Refused {
    fn into_response(self) -> Response {
        let (status, message) = self.0;
        json_fail(status, message)
    }
}

/// What a route answers: its response, or the [`json_fail`] answer of a check that refused it.
pub(super) type Answer = Result<Response, Refused>;

/// Whether `headers` name a JSON body (`application/json`, parameters ignored).
pub(super) fn is_json(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("application/json"))
}

/// The JSON body of a POST: 413 over the route's cap, 415 without a JSON content type, 400
/// (`invalid JSON body: ...`) when it does not parse.
pub(super) fn json_body<T: for<'de> Deserialize<'de>>(
    headers: &HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Result<T, Problem> {
    let body = body.map_err(|e| {
        let s = e.status();
        if s == StatusCode::PAYLOAD_TOO_LARGE {
            (s, "request body too large".to_string())
        } else {
            (s, format!("could not read the request body: {e}"))
        }
    })?;
    if !is_json(headers) {
        return Err((
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "send the body as JSON (Content-Type: application/json)".into(),
        ));
    }
    serde_json::from_slice(&body)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid JSON body: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    async fn body_of(r: Response) -> (StatusCode, serde_json::Value) {
        let status = r.status();
        let bytes = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn the_two_shapes() {
        let (s, v) = body_of(json_fail(StatusCode::CONFLICT, "busy")).await;
        assert_eq!(s, StatusCode::CONFLICT);
        assert_eq!(v, serde_json::json!({"success": false, "message": "busy"}));
        let (s, v) = body_of(json_error(StatusCode::FORBIDDEN, "no")).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        assert_eq!(
            v,
            serde_json::json!({"success": false, "message": "no", "error": "no"})
        );
        let (s, v) = body_of(Refused::new(StatusCode::NOT_FOUND, "gone").into_response()).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert_eq!(v, serde_json::json!({"success": false, "message": "gone"}));
    }

    #[test]
    fn json_body_checks_type_then_parses() {
        #[derive(Deserialize, Debug, PartialEq)]
        struct B {
            n: u32,
        }
        let mut h = HeaderMap::new();
        let err = json_body::<B>(&h, Ok(Bytes::from_static(b"{\"n\":1}"))).unwrap_err();
        assert_eq!(err.0, StatusCode::UNSUPPORTED_MEDIA_TYPE);
        h.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("Application/JSON; charset=utf-8"),
        );
        assert_eq!(
            json_body::<B>(&h, Ok(Bytes::from_static(b"{\"n\":1}"))).unwrap(),
            B { n: 1 }
        );
        let err = json_body::<B>(&h, Ok(Bytes::from_static(b"{\"n\":\"x\"}"))).unwrap_err();
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(err.1.starts_with("invalid JSON body"), "{}", err.1);
    }
}
