//! Release update check against the project's GitHub releases.
//!
//! The answer is untrusted data: the body is read up to [`MAX_BODY_BYTES`] and then parsed
//! (a longer one is an error), the tag is parsed as `major.minor.patch` only, the release
//! notes are cut to [`MAX_NOTES_CHARS`], and nothing is ever downloaded or applied; the result
//! is only shown on the status page.

use crate::api::{StatusUpdateResponse, VersionInfo};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Most bytes of a release answer that are read (a real one is about 2 KB).
pub const MAX_BODY_BYTES: usize = 1 << 20;
/// Most characters of the release notes that are kept and shown.
pub const MAX_NOTES_CHARS: usize = 8 * 1024;
/// Most characters of a tag name that are looked at.
const MAX_TAG_CHARS: usize = 64;

const OK_TTL: Duration = Duration::from_secs(3600);
const ERR_TTL: Duration = Duration::from_secs(300);

/// The last answer: when it was fetched, the `current` version it was computed for, and the
/// response. An async mutex held across the fetch, so concurrent cold calls make one request.
static CACHE: Mutex<Option<(Instant, String, StatusUpdateResponse)>> = Mutex::const_new(None);

/// Whether a result cached at `cached_at` is still usable at `now`: an hour for a successful
/// check, five minutes for a failed one.
fn fresh(cached_at: Instant, ok: bool, now: Instant) -> bool {
    let ttl = if ok { OK_TTL } else { ERR_TTL };
    now.saturating_duration_since(cached_at) < ttl
}

const LATEST_URL: &str =
    "https://api.github.com/repos/patrickgods1/blue-onyx-prism/releases/latest";

#[derive(Debug, Deserialize)]
struct Release {
    tag_name: String,
    #[serde(default)]
    body: Option<String>,
}

/// Parse a release answer: `tag_name` (a string, cut to [`MAX_TAG_CHARS`]) and `body`
/// (a string or null, cut to [`MAX_NOTES_CHARS`]). Any other shape is an error, never a
/// panic; unknown fields are ignored.
fn parse_release(bytes: &[u8]) -> Result<Release> {
    let mut rel: Release = serde_json::from_slice(bytes).context("parsing release JSON")?;
    truncate_chars(&mut rel.tag_name, MAX_TAG_CHARS);
    if let Some(b) = rel.body.as_mut() {
        truncate_chars(b, MAX_NOTES_CHARS);
    }
    Ok(rel)
}

/// Keep the first `max` characters of `s` (on a char boundary).
fn truncate_chars(s: &mut String, max: usize) {
    if let Some((idx, _)) = s.char_indices().nth(max) {
        s.truncate(idx);
    }
}

/// Collect a body stream, failing once it goes past `max` bytes.
async fn read_capped<S, E>(mut stream: S, max: usize) -> Result<Vec<u8>>
where
    S: futures_util::Stream<Item = std::result::Result<bytes::Bytes, E>> + Unpin,
    E: std::error::Error + Send + Sync + 'static,
{
    use futures_util::StreamExt;
    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading the release answer")?;
        if out.len().saturating_add(chunk.len()) > max {
            bail!("release answer is larger than {max} bytes");
        }
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

async fn fetch_latest() -> Result<Release> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .https_only(true)
        .redirect(reqwest::redirect::Policy::limited(3))
        .build()?;
    let resp = client
        .get(LATEST_URL)
        .header(
            reqwest::header::USER_AGENT,
            format!("blue-onyx-prism/{}", crate::VERSION),
        )
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .await
        .context("requesting latest release")?
        .error_for_status()
        .context("release server returned an error")?;
    if let Some(len) = resp.content_length()
        && len > MAX_BODY_BYTES as u64
    {
        bail!("release answer is larger than {MAX_BODY_BYTES} bytes");
    }
    let bytes = read_capped(resp.bytes_stream(), MAX_BODY_BYTES).await?;
    parse_release(&bytes)
}

/// Compare `current` against the latest release. Never fails: errors are reported through
/// `success: false` and `message`.
/// Results are cached in-process (1 hour; 5 minutes after a failure) to stay within GitHub's
/// unauthenticated rate limit.
pub async fn check_update(current: &str) -> StatusUpdateResponse {
    let mut cache = CACHE.lock().await;
    if let Some((at, cur, resp)) = cache.as_ref()
        && cur == current
        && fresh(*at, resp.success, Instant::now())
    {
        return resp.clone();
    }
    let resp = check_uncached(current).await;
    *cache = Some((Instant::now(), current.to_string(), resp.clone()));
    resp
}

async fn check_uncached(current: &str) -> StatusUpdateResponse {
    let run = async {
        let current_v = VersionInfo::parse(current, None)?;
        let rel = fetch_latest().await?;
        let latest = VersionInfo::parse(&rel.tag_name, rel.body)?;
        Ok::<_, anyhow::Error>((current_v, latest))
    };
    match run.await {
        Ok((current_v, latest)) => StatusUpdateResponse {
            success: true,
            message: "Update check completed".to_string(),
            updateAvailable: latest > current_v,
            version: Some(latest.clone()),
            current: current_v,
            latest,
        },
        Err(e) => StatusUpdateResponse {
            success: false,
            message: format!("{e:#}"),
            ..Default::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hostile_release_bodies_never_panic_and_never_update() {
        // Wrong types, missing fields, not JSON, huge strings: all errors, no panic.
        for body in [
            &b"not json"[..],
            b"",
            b"null",
            b"[]",
            b"{}",
            b"{\"tag_name\": 1}",
            b"{\"tag_name\": null}",
            b"{\"tag_name\": [\"v9.9.9\"]}",
            b"{\"tag_name\": \"v1.2.3\", \"body\": 7}",
            b"{\"tag_name\": \"v1.2.3\", \"body\": {\"x\": 1}}",
        ] {
            let r = parse_release(body);
            assert!(r.is_err(), "{}", String::from_utf8_lossy(body));
        }
        // Odd tags parse but do not become a version; the message stays short.
        let huge = format!("{{\"tag_name\": \"{}\"}}", "9".repeat(100_000));
        let rel = parse_release(huge.as_bytes()).unwrap();
        assert_eq!(rel.tag_name.len(), MAX_TAG_CHARS);
        let err = VersionInfo::parse(&rel.tag_name, rel.body).unwrap_err();
        assert!(format!("{err:#}").len() < 400);
        for tag in [
            "",
            "v",
            "latest",
            "1.2",
            "1.2.3.4.5",
            "999.0.0",
            "v1.2.3-rc1",
            "1.-2.3",
        ] {
            let rel = parse_release(format!("{{\"tag_name\": {tag:?}}}").as_bytes()).unwrap();
            let v = VersionInfo::parse(&rel.tag_name, rel.body);
            if tag == "1.2.3.4.5" {
                assert!(v.is_ok(), "extra segments are ignored");
            } else {
                assert!(v.is_err(), "{tag}");
            }
        }
        // Release notes are cut; multi-byte characters keep their boundaries.
        let notes = "é".repeat(MAX_NOTES_CHARS + 500);
        let body = serde_json::json!({ "tag_name": "v0.1.0", "body": notes, "extra": [1, 2] });
        let rel = parse_release(body.to_string().as_bytes()).unwrap();
        assert_eq!(rel.body.as_ref().unwrap().chars().count(), MAX_NOTES_CHARS);
        assert_eq!(rel.tag_name, "v0.1.0");
        let rel = parse_release(b"{\"tag_name\": \"v0.1.0\", \"body\": null}").unwrap();
        assert_eq!(rel.body, None);
    }

    #[test]
    fn body_stream_is_capped() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let chunks = |n: usize, size: usize| {
            futures_util::stream::iter(
                (0..n).map(move |_| Ok::<_, std::io::Error>(bytes::Bytes::from(vec![b'x'; size]))),
            )
        };
        let ok = rt.block_on(read_capped(chunks(4, 100), 400)).unwrap();
        assert_eq!(ok.len(), 400);
        let err = rt.block_on(read_capped(chunks(5, 100), 400)).unwrap_err();
        assert!(format!("{err}").contains("larger than 400"));
        let err = rt
            .block_on(read_capped(
                futures_util::stream::iter([Err::<bytes::Bytes, _>(std::io::Error::other("cut"))]),
                400,
            ))
            .unwrap_err();
        assert!(format!("{err:#}").contains("cut"));
    }

    #[test]
    fn freshness_windows() {
        let t0 = Instant::now();
        let m = |s| t0 + Duration::from_secs(s);
        assert!(fresh(t0, true, m(3599)));
        assert!(!fresh(t0, true, m(3600)));
        assert!(fresh(t0, false, m(299)));
        assert!(!fresh(t0, false, m(300)));
    }
}
