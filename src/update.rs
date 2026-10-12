//! Release update check against the project's GitHub releases.

use crate::api::{StatusUpdateResponse, VersionInfo};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

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

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    #[serde(default)]
    body: Option<String>,
}

async fn fetch_latest() -> Result<Release> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
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
    resp.json::<Release>().await.context("parsing release JSON")
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
    fn freshness_windows() {
        let t0 = Instant::now();
        let m = |s| t0 + Duration::from_secs(s);
        assert!(fresh(t0, true, m(3599)));
        assert!(!fresh(t0, true, m(3600)));
        assert!(fresh(t0, false, m(299)));
        assert!(!fresh(t0, false, m(300)));
    }
}
