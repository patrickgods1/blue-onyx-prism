//! Minimal synchronous `GET /v1/status` probe of the local server over a plain
//! `std::net::TcpStream` (HTTP/1.1, `Connection: close`, short timeouts), and the pure parser of
//! the response (status line, `Content-Length` / chunked body, JSON). Keeps the tray free of an
//! async HTTP stack.

use serde_json::Value;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

/// Connect and read timeout of one probe.
pub const PROBE_TIMEOUT: Duration = Duration::from_millis(1500);
/// Largest response read (the status JSON is a few KB).
const MAX_RESPONSE: u64 = 1024 * 1024;

/// One model in `/v1/status`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelStatus {
    pub name: String,
    /// "ready", "lazy", "initializing" or "failed".
    pub state: String,
    pub device: Option<String>,
    pub detail: Option<String>,
}

/// A parsed `/v1/status` answer (HTTP 200 or 503).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatusReport {
    pub http_code: u16,
    pub ready: bool,
    /// "ready", "initializing", "degraded" or "failed".
    pub status: String,
    pub models: Vec<ModelStatus>,
    pub version: Option<String>,
}

/// Result of one probe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HttpStatus {
    /// Nothing listens / no answer in time.
    Unreachable,
    /// Something answered but not with a status report (message says what).
    Invalid(String),
    Answer(StatusReport),
}

/// Probe `http://<addr>/v1/status`.
pub fn probe(addr: SocketAddr, timeout: Duration) -> HttpStatus {
    match fetch(addr, timeout) {
        Ok(raw) => parse_response(&raw),
        Err(e) => {
            tracing::debug!("status probe of {addr}: {e}");
            HttpStatus::Unreachable
        }
    }
}

fn fetch(addr: SocketAddr, timeout: Duration) -> std::io::Result<Vec<u8>> {
    let mut stream = TcpStream::connect_timeout(&addr, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    stream.write_all(request(addr).as_bytes())?;
    let mut raw = Vec::new();
    stream.take(MAX_RESPONSE).read_to_end(&mut raw)?;
    Ok(raw)
}

/// The request bytes.
pub fn request(addr: SocketAddr) -> String {
    format!(
        "GET /v1/status HTTP/1.1\r\nHost: {addr}\r\nUser-Agent: blue-onyx-prism-tray\r\n\
         Accept: application/json\r\nConnection: close\r\n\r\n"
    )
}

/// Parse a raw HTTP/1.x response to `GET /v1/status`.
pub fn parse_response(raw: &[u8]) -> HttpStatus {
    let Some(split) = find(raw, b"\r\n\r\n") else {
        return HttpStatus::Invalid("incomplete HTTP response".into());
    };
    let head = String::from_utf8_lossy(raw.get(..split).unwrap_or_default());
    let body = raw.get(split + 4..).unwrap_or_default();
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or_default();
    let mut parts = status_line.split_whitespace();
    let code = match (
        parts.next(),
        parts.next().and_then(|c| c.parse::<u16>().ok()),
    ) {
        (Some(v), Some(c)) if v.starts_with("HTTP/1.") => c,
        _ => return HttpStatus::Invalid(format!("not an HTTP response: {status_line:?}")),
    };
    let mut chunked = false;
    let mut length = None;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            let (k, v) = (k.trim(), v.trim());
            if k.eq_ignore_ascii_case("transfer-encoding") && v.eq_ignore_ascii_case("chunked") {
                chunked = true;
            } else if k.eq_ignore_ascii_case("content-length") {
                length = v.parse::<usize>().ok();
            }
        }
    }
    let body = if chunked {
        match dechunk(body) {
            Some(b) => b,
            None => return HttpStatus::Invalid("bad chunked body".into()),
        }
    } else {
        match length {
            Some(n) => match body.get(..n) {
                Some(b) => b.to_vec(),
                None => return HttpStatus::Invalid("truncated body".into()),
            },
            None => body.to_vec(),
        }
    };
    if code != 200 && code != 503 {
        return HttpStatus::Invalid(format!("HTTP {code}"));
    }
    match parse_body(code, &body) {
        Some(r) => HttpStatus::Answer(r),
        None => HttpStatus::Invalid(format!("HTTP {code} without a status report")),
    }
}

/// Parse the JSON body of `/v1/status`.
pub fn parse_body(http_code: u16, body: &[u8]) -> Option<StatusReport> {
    let v: Value = serde_json::from_slice(body).ok()?;
    let status = v.get("status")?.as_str()?.to_string();
    let text = |m: &Value, k: &str| m.get(k).and_then(Value::as_str).map(str::to_string);
    let models = v
        .get("models")
        .and_then(Value::as_array)
        .map(|ms| {
            ms.iter()
                .map(|m| ModelStatus {
                    name: text(m, "name").unwrap_or_default(),
                    state: text(m, "state").unwrap_or_default(),
                    device: text(m, "device"),
                    detail: text(m, "detail"),
                })
                .collect()
        })
        .unwrap_or_default();
    Some(StatusReport {
        http_code,
        ready: v
            .get("ready")
            .and_then(Value::as_bool)
            .unwrap_or(http_code == 200),
        status,
        models,
        version: text(&v, "version"),
    })
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Decode a chunked transfer-encoded body (extensions and trailers ignored).
fn dechunk(mut b: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let eol = find(b, b"\r\n")?;
        let line = std::str::from_utf8(b.get(..eol)?).ok()?;
        let size_hex = line.split(';').next()?.trim();
        let size = usize::from_str_radix(size_hex, 16).ok()?;
        b = b.get(eol + 2..)?;
        if size == 0 {
            return Some(out);
        }
        out.extend_from_slice(b.get(..size)?);
        b = b.get(size..)?;
        b = b.strip_prefix(b"\r\n")?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const READY: &str = r#"{"success":true,"ready":true,"status":"ready","uptimeSecs":12,"generation":1,"models":[{"name":"IPcam-general","state":"ready","device":"GPU","detail":null}],"version":"1.2.3"}"#;

    fn response(code: &str, body: &str) -> Vec<u8> {
        format!(
            "HTTP/1.1 {code}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes()
    }

    #[test]
    fn parses_a_ready_answer() {
        let HttpStatus::Answer(r) = parse_response(&response("200 OK", READY)) else {
            panic!()
        };
        assert_eq!(r.http_code, 200);
        assert!(r.ready);
        assert_eq!(r.status, "ready");
        assert_eq!(r.version.as_deref(), Some("1.2.3"));
        assert_eq!(
            r.models,
            vec![ModelStatus {
                name: "IPcam-general".into(),
                state: "ready".into(),
                device: Some("GPU".into()),
                detail: None,
            }]
        );
    }

    #[test]
    fn parses_an_initializing_503() {
        let body = r#"{"ready":false,"status":"initializing","models":[{"name":"m","state":"initializing","device":"","detail":"downloading OpenVINO runtime 42%"}]}"#;
        let HttpStatus::Answer(r) = parse_response(&response("503 Service Unavailable", body))
        else {
            panic!()
        };
        assert_eq!(r.http_code, 503);
        assert!(!r.ready);
        assert_eq!(r.status, "initializing");
        assert_eq!(
            r.models.first().and_then(|m| m.detail.as_deref()),
            Some("downloading OpenVINO runtime 42%")
        );
    }

    #[test]
    fn parses_a_chunked_body() {
        let (a, b) = READY.split_at(20);
        let raw = format!(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{a}\r\n{:x};ext=1\r\n{b}\r\n0\r\n\r\n",
            a.len(),
            b.len()
        );
        let HttpStatus::Answer(r) = parse_response(raw.as_bytes()) else {
            panic!()
        };
        assert_eq!(r.status, "ready");
    }

    #[test]
    fn rejects_other_answers() {
        assert!(matches!(
            parse_response(&response("404 Not Found", "nope")),
            HttpStatus::Invalid(m) if m == "HTTP 404"
        ));
        assert!(matches!(
            parse_response(&response("200 OK", "<html></html>")),
            HttpStatus::Invalid(_)
        ));
        assert!(matches!(
            parse_response(b"SSH-2.0-OpenSSH\r\n\r\n"),
            HttpStatus::Invalid(_)
        ));
        assert!(matches!(
            parse_response(b"HTTP/1.1 200 OK\r\n"),
            HttpStatus::Invalid(_)
        ));
        let mut truncated = response("200 OK", READY);
        truncated.truncate(truncated.len() - 5);
        assert!(matches!(parse_response(&truncated), HttpStatus::Invalid(_)));
    }

    #[test]
    fn request_is_http11_with_close() {
        let r = request("127.0.0.1:32168".parse().unwrap());
        assert!(r.starts_with("GET /v1/status HTTP/1.1\r\nHost: 127.0.0.1:32168\r\n"));
        assert!(r.contains("Connection: close\r\n"));
        assert!(r.ends_with("\r\n\r\n"));
    }

    #[test]
    fn probes_a_local_listener() {
        use std::net::TcpListener;
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let t = std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut buf = [0u8; 1024];
            let n = s.read(&mut buf).unwrap();
            assert!(buf[..n].starts_with(b"GET /v1/status "));
            s.write_all(&response("200 OK", READY)).unwrap();
        });
        let r = probe(addr, PROBE_TIMEOUT);
        t.join().unwrap();
        assert!(matches!(r, HttpStatus::Answer(ref a) if a.ready), "{r:?}");
    }

    #[test]
    fn nothing_listening_is_unreachable() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        drop(l);
        assert_eq!(probe(addr, PROBE_TIMEOUT), HttpStatus::Unreachable);
    }
}
