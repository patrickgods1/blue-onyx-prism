//! Test client for a running Blue Onyx Prism server (CodeProject.AI compatible API).
//!
//! Sends an image to `/v1/vision/detection` or `/v1/vision/custom/{model}` (optionally many
//! times, optionally in parallel), prints the response or a latency summary and exits non-zero
//! if any request failed.
//!
//! Two load modes:
//! - closed loop (`--repeat N --parallel P`): at most P requests in flight, each next one sent
//!   when one finishes. Its latency is measured from the actual send, so a slow server also
//!   slows the client down and hides the queueing a camera system would see (coordinated
//!   omission). Good for "how fast is one request".
//! - open loop (`--rate R --duration S`): request i is due at `i / R` seconds and starts then,
//!   whatever is still in flight, as cameras do. Latency is measured from the scheduled start, so
//!   a server that falls behind shows it. Reports p50/p95/p99/max, errors, 503s (overload), and
//!   the server's resident memory read from its metrics every second (start, peak, end). The
//!   metrics need the API token over HTTPS (`--https-port`, `--token-file` or `--config`; never
//!   on the command line, where `ps` would show it): the client
//!   pins the server's certificate from its file (`--cert`, or the one `--config` names), or with
//!   `--insecure` (tests only) accepts any certificate.

#![cfg_attr(
    not(test),
    warn(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::large_futures,
        clippy::unused_async,
        clippy::dbg_macro,
        missing_debug_implementations
    )
)]
#![deny(unsafe_code)]

use anyhow::{Context, Result, bail};
use blue_onyx_prism::api::{Prediction, VisionDetectionResponse};
use clap::Parser;
use futures_util::StreamExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

static DEFAULT_IMAGE: &[u8] = include_bytes!("../../tests/data/dog_bike_car.jpg");

#[derive(Debug, Parser)]
#[command(
    name = "test-blue-onyx-prism",
    version = blue_onyx_prism::cli::long_version_str(),
    about = "Send test images to a running Blue Onyx Prism server"
)]
struct Args {
    /// Server base URL.
    #[arg(long, default_value = "http://127.0.0.1:32168")]
    url: String,
    /// Named model: POST /v1/vision/custom/{model}. Default: /v1/vision/detection.
    #[arg(long)]
    model: Option<String>,
    /// Image to send (default: embedded dog_bike_car.jpg).
    #[arg(long)]
    image: Option<PathBuf>,
    /// min_confidence form field (0 = server default).
    #[arg(long, default_value_t = 0.0)]
    min_confidence: f32,
    /// Number of requests to send.
    #[arg(long, default_value_t = 1)]
    repeat: usize,
    /// Delay between requests in milliseconds.
    #[arg(long, default_value_t = 0)]
    interval_ms: u64,
    /// Number of concurrent requests.
    #[arg(long, default_value_t = 1)]
    parallel: usize,
    /// List the configured models (POST /v1/vision/custom/list) and exit.
    #[arg(long)]
    list: bool,
    /// Save an annotated copy of the image (from the last response) to this path.
    #[arg(long)]
    save: Option<PathBuf>,
    /// Open loop: start this many requests per second on schedule, whatever is in flight
    /// (latency from the scheduled start). Needs --duration.
    #[arg(
        long,
        requires = "duration",
        conflicts_with_all = ["repeat", "parallel", "interval_ms", "list", "save"]
    )]
    rate: Option<f64>,
    /// Open loop: seconds to keep sending (with --rate).
    #[arg(long, requires = "rate")]
    duration: Option<f64>,
    /// Open loop: metrics path sampled every second for the server's resident memory, over
    /// HTTPS on --https-port with the API token.
    #[arg(long, default_value = "/prometheus")]
    metrics_path: String,
    /// The server's HTTPS port (the metrics need it: the plain port serves detection only).
    #[arg(long, default_value_t = blue_onyx_prism::DEFAULT_HTTPS_PORT)]
    https_port: u16,
    /// A file holding the API token for the metrics (`-`: read one line from stdin, e.g.
    /// `blue-onyx-prism admin-token | test-blue-onyx-prism --token-file - ...`); default: the
    /// token file of --config. The token is never taken from the command line itself.
    #[arg(long)]
    token_file: Option<PathBuf>,
    /// The server's config file: where its API token and HTTPS certificate are (run as an
    /// account that may read them).
    #[arg(long)]
    config: Option<PathBuf>,
    /// The HTTPS certificate (PEM) to pin for the metrics; default: the one --config names.
    #[arg(long)]
    cert: Option<PathBuf>,
    /// Accept any HTTPS certificate for the metrics (for tests on throwaway servers only; it
    /// gives up the protection the pin gives the token).
    #[arg(long)]
    insecure: bool,
}

/// The client and URL that sample the server's metrics over HTTPS, with the API token as a
/// default header; `Err` says what is missing.
fn metrics_client(args: &Args) -> Result<(reqwest::Client, String)> {
    let base = reqwest::Url::parse(&args.url).with_context(|| format!("--url {}", args.url))?;
    // `host_str` keeps an IPv6 address's brackets.
    let Some(host) = base.host_str().map(str::to_string) else {
        bail!("--url {} has no host", args.url);
    };
    let config = args
        .config
        .as_deref()
        .map(blue_onyx_prism::config::Config::load)
        .transpose()?;
    let token = match (&args.token_file, &config) {
        (Some(p), _) if p.as_os_str() == "-" => {
            let mut line = String::new();
            std::io::stdin()
                .read_line(&mut line)
                .context("reading the API token from stdin")?;
            line.trim().to_string()
        }
        (Some(p), _) => blue_onyx_prism::admin_token::read(p)?.reveal().to_string(),
        (None, Some(c)) => {
            let path = blue_onyx_prism::admin_token::path(c);
            blue_onyx_prism::admin_token::read(&path)?
                .reveal()
                .to_string()
        }
        (None, None) => bail!(
            "the metrics need the API token: pass --token-file <file> (or `-` for stdin) or \
             --config <the server's config file>"
        ),
    };
    if token.is_empty() {
        bail!("the API token read for the metrics is empty");
    }
    let builder = if args.insecure {
        reqwest::Client::builder()
            .danger_accept_invalid_certs(true)
            .no_proxy()
    } else {
        let cert = match (&args.cert, &config) {
            (Some(p), _) => p.clone(),
            (None, Some(c)) => blue_onyx_prism::tls::cert_path(c),
            (None, None) => bail!(
                "the metrics are HTTPS with a self-signed certificate: pass --cert <its PEM \
                 file> or --config <the server's config file>, or --insecure on a test server"
            ),
        };
        blue_onyx_prism::tls::pinned_client(&cert)?
    };
    let mut headers = reqwest::header::HeaderMap::new();
    let mut auth = reqwest::header::HeaderValue::from_str(&format!("Bearer {token}"))
        .context("the API token is not a header value")?;
    auth.set_sensitive(true);
    headers.insert(reqwest::header::AUTHORIZATION, auth);
    let client = builder
        .default_headers(headers)
        .timeout(Duration::from_secs(10))
        .build()?;
    Ok((
        client,
        format!("https://{host}:{}{}", args.https_port, args.metrics_path),
    ))
}

/// When each open-loop request is due: `i / rate` seconds for every `i` with `i / rate <
/// duration`. Errors for a rate or duration that is not a positive finite number, or a run of
/// more than [`MAX_OPEN_LOOP_REQUESTS`] requests.
fn schedule(rate: f64, duration: f64) -> Result<Vec<Duration>> {
    if !(rate.is_finite() && rate > 0.0) {
        bail!("--rate must be a positive number of requests per second, not {rate}");
    }
    if !(duration.is_finite() && duration > 0.0) {
        bail!("--duration must be a positive number of seconds, not {duration}");
    }
    let n = (rate * duration).ceil();
    if n > MAX_OPEN_LOOP_REQUESTS as f64 {
        bail!(
            "--rate {rate} x --duration {duration} is {n} requests, more than the \
             {MAX_OPEN_LOOP_REQUESTS} this client keeps in memory; shorten the run"
        );
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "n is a positive whole number at most MAX_OPEN_LOOP_REQUESTS"
    )]
    let n = n as usize;
    Ok((0..n)
        .map(|i| Duration::from_secs_f64(i as f64 / rate))
        .filter(|at| at.as_secs_f64() < duration)
        .collect())
}

/// Largest open-loop run (requests).
const MAX_OPEN_LOOP_REQUESTS: usize = 1_000_000;

/// Nearest-rank percentile `p` (0..=100) of ascending `sorted`; NaN when empty.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let n = sorted.len();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to 1..=n before the cast"
    )]
    let rank = ((p / 100.0) * n as f64).ceil().clamp(1.0, n as f64) as usize;
    sorted.get(rank - 1).copied().unwrap_or(f64::NAN)
}

/// `<prefix>process_resident_memory_bytes <n>` of a Prometheus text.
fn parse_rss(metrics: &str) -> Option<u64> {
    metrics.lines().find_map(|l| {
        let (name, value) = l.split_once(' ')?;
        name.ends_with("process_resident_memory_bytes")
            .then(|| value.trim().parse::<f64>().ok())
            .flatten()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .map(|v| {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a non-negative byte count"
                )]
                let b = v as u64;
                b
            })
    })
}

/// How one open-loop request ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ended {
    Ok,
    /// HTTP 503: the server's in-flight cap or a model queue (overload).
    Overloaded,
    /// Any other HTTP error, `success: false`, or bad JSON.
    Failed,
    /// No HTTP answer (connect, timeout, reset).
    Transport,
}

struct OpenOutcome {
    ended: Ended,
    /// From the scheduled start to the end of the answer.
    from_schedule_ms: f64,
    /// From the actual send to the end of the answer.
    from_send_ms: f64,
    /// How late the request was sent (client lag).
    send_lag_ms: f64,
    note: String,
}

async fn send_scheduled(
    client: reqwest::Client,
    url: String,
    image: Arc<Vec<u8>>,
    min_confidence: f32,
    due: tokio::time::Instant,
) -> OpenOutcome {
    tokio::time::sleep_until(due).await;
    let sent = tokio::time::Instant::now();
    let part = reqwest::multipart::Part::bytes(image.as_ref().clone()).file_name("image.jpg");
    let mut form = reqwest::multipart::Form::new().part("image", part);
    if min_confidence > 0.0 {
        form = form.text("min_confidence", min_confidence.to_string());
    }
    let result = async {
        let resp = client.post(&url).multipart(form).send().await?;
        let status = resp.status();
        let body = resp.text().await?;
        Ok::<_, reqwest::Error>((status, body))
    }
    .await;
    let done = tokio::time::Instant::now();
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let (ended, note) = match result {
        Err(e) => (Ended::Transport, format!("request failed: {e}")),
        Ok((status, _)) if status == reqwest::StatusCode::SERVICE_UNAVAILABLE => {
            (Ended::Overloaded, "HTTP 503".to_string())
        }
        Ok((status, body)) if !status.is_success() => {
            (Ended::Failed, format!("HTTP {status}: {body}"))
        }
        Ok((_, body)) => match serde_json::from_str::<VisionDetectionResponse>(&body) {
            Ok(r) if r.success => (Ended::Ok, String::new()),
            Ok(r) => (
                Ended::Failed,
                format!("success=false: {}", r.error.unwrap_or(r.message)),
            ),
            Err(e) => (Ended::Failed, format!("invalid JSON ({e}): {body}")),
        },
    };
    OpenOutcome {
        ended,
        from_schedule_ms: ms(done.saturating_duration_since(due)),
        from_send_ms: ms(done.saturating_duration_since(sent)),
        send_lag_ms: ms(sent.saturating_duration_since(due)),
        note,
    }
}

/// Server RSS samples (seconds since start, bytes) until `stop`, one a second.
async fn sample_rss(
    client: reqwest::Client,
    url: String,
    start: tokio::time::Instant,
    stop: tokio_util::sync::CancellationToken,
) -> Vec<(f64, u64)> {
    let mut out = Vec::new();
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    loop {
        tokio::select! {
            _ = stop.cancelled() => break,
            _ = tick.tick() => {}
        }
        if url.is_empty() {
            continue;
        }
        if let Ok(r) = client.get(&url).send().await
            && r.status().is_success()
            && let Ok(text) = r.text().await
            && let Some(b) = parse_rss(&text)
        {
            out.push((start.elapsed().as_secs_f64(), b));
        }
    }
    out
}

fn latency_line(label: &str, mut v: Vec<f64>) {
    if v.is_empty() {
        println!("  {label:<26} (none)");
        return;
    }
    v.sort_by(f64::total_cmp);
    println!(
        "  {label:<26} p50 {:>8.1}  p95 {:>8.1}  p99 {:>8.1}  max {:>8.1}",
        percentile(&v, 50.0),
        percentile(&v, 95.0),
        percentile(&v, 99.0),
        v.last().copied().unwrap_or(f64::NAN)
    );
}

fn mb(b: u64) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

/// `--rate R --duration S`: the open-loop run. Ok(true) when every request succeeded.
async fn run_open_loop(
    args: &Args,
    client: &reqwest::Client,
    base: &str,
    url: &str,
    image: Vec<u8>,
    rate: f64,
    duration: f64,
) -> Result<bool> {
    let due = schedule(rate, duration)?;
    let image = Arc::new(image);
    // A little ahead, so the first request is not late because tasks were still being made.
    let start = tokio::time::Instant::now() + Duration::from_millis(200);
    let stop = tokio_util::sync::CancellationToken::new();
    let (metrics, metrics_url) = match metrics_client(args) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("server RSS will not be sampled: {e:#}");
            (client.clone(), String::new())
        }
    };
    let sampler = tokio::spawn(sample_rss(
        metrics,
        metrics_url.clone(),
        start,
        stop.clone(),
    ));
    let tasks: Vec<_> = due
        .iter()
        .map(|at| {
            tokio::spawn(send_scheduled(
                client.clone(),
                url.to_string(),
                image.clone(),
                args.min_confidence,
                start + *at,
            ))
        })
        .collect();
    let mut outcomes = Vec::with_capacity(tasks.len());
    for t in tasks {
        outcomes.push(t.await.context("a request task failed")?);
    }
    let wall_s = start.elapsed().as_secs_f64();
    // One sample after the last answer, so "end" is the memory once the load stopped.
    tokio::time::sleep(Duration::from_millis(1100)).await;
    stop.cancel();
    let rss = sampler.await.unwrap_or_default();

    let count = |e: Ended| outcomes.iter().filter(|o| o.ended == e).count();
    let (ok, overloaded, failed, transport) = (
        count(Ended::Ok),
        count(Ended::Overloaded),
        count(Ended::Failed),
        count(Ended::Transport),
    );
    println!("POST {url}");
    println!(
        "open loop: {rate} req/s for {duration} s = {} requests scheduled, {wall_s:.1} s until \
         the last answer",
        outcomes.len()
    );
    println!(
        "  {ok} ok, {overloaded} answered 503 (overload), {failed} failed, {transport} without \
         an answer; {:.2} ok/s",
        ok as f64 / wall_s.max(f64::EPSILON)
    );
    println!("latency (ms):");
    latency_line(
        "from scheduled start",
        outcomes
            .iter()
            .filter(|o| o.ended == Ended::Ok)
            .map(|o| o.from_schedule_ms)
            .collect(),
    );
    latency_line(
        "from actual send",
        outcomes
            .iter()
            .filter(|o| o.ended == Ended::Ok)
            .map(|o| o.from_send_ms)
            .collect(),
    );
    latency_line(
        "all answers (sched.)",
        outcomes
            .iter()
            .filter(|o| o.ended != Ended::Transport)
            .map(|o| o.from_schedule_ms)
            .collect(),
    );
    latency_line(
        "client send lag",
        outcomes.iter().map(|o| o.send_lag_ms).collect(),
    );
    match (rss.first(), rss.iter().max_by_key(|s| s.1), rss.last()) {
        (Some(first), Some(peak), Some(last)) => println!(
            "server RSS ({} samples): start {:.1} MB, peak {:.1} MB at {:.0} s, end {:.1} MB",
            rss.len(),
            mb(first.1),
            mb(peak.1),
            peak.0,
            mb(last.1)
        ),
        _ => println!(
            "server RSS: no samples (GET {metrics_url} with the API token gave no \
             process_resident_memory_bytes; for {base}'s server pass --https-port and --token-file or \
             --config)"
        ),
    }
    for o in outcomes.iter().filter(|o| o.ended != Ended::Ok).take(5) {
        eprintln!("failure: {}", o.note);
    }
    Ok(ok == outcomes.len())
}

struct Outcome {
    ok: bool,
    round_trip_ms: f64,
    inference_ms: i32,
    process_ms: i32,
    response: Option<VisionDetectionResponse>,
    note: String,
}

async fn send_one(
    client: &reqwest::Client,
    url: &str,
    image: &[u8],
    min_confidence: f32,
    start_at: tokio::time::Instant,
) -> Outcome {
    tokio::time::sleep_until(start_at).await;
    let start = Instant::now();
    let part = reqwest::multipart::Part::bytes(image.to_vec()).file_name("image.jpg");
    let mut form = reqwest::multipart::Form::new().part("image", part);
    if min_confidence > 0.0 {
        form = form.text("min_confidence", min_confidence.to_string());
    }
    let result = async {
        let resp = client.post(url).multipart(form).send().await?;
        let status = resp.status();
        let body = resp.text().await?;
        Ok::<_, reqwest::Error>((status, body))
    }
    .await;
    let round_trip_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut out = Outcome {
        ok: false,
        round_trip_ms,
        inference_ms: 0,
        process_ms: 0,
        response: None,
        note: String::new(),
    };
    match result {
        Err(e) => out.note = format!("request failed: {e}"),
        Ok((status, body)) => {
            if !status.is_success() {
                out.note = format!("HTTP {status}: {body}");
            } else {
                match serde_json::from_str::<VisionDetectionResponse>(&body) {
                    Ok(r) => {
                        out.ok = r.success;
                        out.inference_ms = r.inferenceMs;
                        out.process_ms = r.processMs;
                        if !r.success {
                            out.note = format!(
                                "success=false: {}",
                                r.error.clone().unwrap_or(r.message.clone())
                            );
                        }
                        out.response = Some(r);
                    }
                    Err(e) => out.note = format!("invalid JSON ({e}): {body}"),
                }
            }
        }
    }
    out
}

fn stats(label: &str, mut v: Vec<f64>) {
    if v.is_empty() {
        return;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n as f64;
    // Nearest-rank p95: ceil(0.95 * n) in integers, 1-based.
    let rank = n.saturating_mul(95).div_ceil(100).clamp(1, n);
    let at = |i: usize| v.get(i).copied().unwrap_or(f64::NAN);
    println!(
        "  {label:<22} min {:>8.1}  mean {:>8.1}  p95 {:>8.1}  max {:>8.1}",
        at(0),
        mean,
        at(rank - 1),
        at(n - 1)
    );
}

async fn run(args: Args) -> Result<bool> {
    let base = args.url.trim_end_matches('/').to_string();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;

    if args.list {
        let resp = client
            .post(format!("{base}/v1/vision/custom/list"))
            .send()
            .await
            .context("POST /v1/vision/custom/list")?;
        let status = resp.status();
        let body = resp.text().await?;
        match serde_json::from_str::<serde_json::Value>(&body) {
            Ok(v) => println!("{}", serde_json::to_string_pretty(&v)?),
            Err(_) => println!("{body}"),
        }
        return Ok(status.is_success());
    }

    let image: Vec<u8> = match &args.image {
        Some(p) => std::fs::read(p).with_context(|| format!("reading {}", p.display()))?,
        None => DEFAULT_IMAGE.to_vec(),
    };
    let url = match &args.model {
        Some(m) => format!("{base}/v1/vision/custom/{m}"),
        None => format!("{base}/v1/vision/detection"),
    };
    if let (Some(rate), Some(duration)) = (args.rate, args.duration) {
        return run_open_loop(&args, &client, &base, &url, image, rate, duration).await;
    }
    let repeat = args.repeat.max(1);
    let parallel = args.parallel.max(1);
    let interval = Duration::from_millis(args.interval_ms);

    // Request i starts at `wall + interval * (i / parallel)` (absolute, so waiting for earlier
    // requests does not add to the delay). Checked up front: a huge interval is an error.
    let start = tokio::time::Instant::now();
    let start_of = |i: usize| -> Option<tokio::time::Instant> {
        let k = u32::try_from(i / parallel).ok()?;
        start.checked_add(interval.checked_mul(k)?)
    };
    if start_of(repeat - 1).is_none() {
        bail!(
            "--interval-ms {} x --repeat {repeat} is too large",
            args.interval_ms
        );
    }
    let wall = Instant::now();
    let outcomes: Vec<Outcome> = futures_util::stream::iter(0..repeat)
        .map(|i| {
            let at = start_of(i).unwrap_or(start);
            let (client, url, image) = (&client, &url, &image);
            async move { send_one(client, url, image, args.min_confidence, at).await }
        })
        .buffered(parallel)
        .collect()
        .await;
    let wall_s = wall.elapsed().as_secs_f64();

    let failures = outcomes.iter().filter(|o| !o.ok).count();
    if let (1, Some(o)) = (repeat, outcomes.first()) {
        match &o.response {
            Some(r) => println!("{}", serde_json::to_string_pretty(r)?),
            None => println!("{}", o.note),
        }
        println!("round trip: {:.1} ms", o.round_trip_ms);
    } else {
        println!("POST {url}");
        println!(
            "{repeat} requests, parallel {parallel}: {} ok, {failures} failed, {wall_s:.2} s total",
            repeat - failures
        );
        println!("latency (ms):");
        stats(
            "client round trip",
            outcomes.iter().map(|o| o.round_trip_ms).collect(),
        );
        let ok: Vec<&Outcome> = outcomes.iter().filter(|o| o.response.is_some()).collect();
        stats(
            "server inferenceMs",
            ok.iter().map(|o| o.inference_ms as f64).collect(),
        );
        stats(
            "server processMs",
            ok.iter().map(|o| o.process_ms as f64).collect(),
        );
        for o in outcomes.iter().filter(|o| !o.ok).take(5) {
            eprintln!("failure: {}", o.note);
        }
    }

    if let Some(path) = &args.save {
        let preds: Vec<Prediction> = outcomes
            .iter()
            .rev()
            .find_map(|o| o.response.as_ref())
            .map(|r| r.predictions.clone())
            .unwrap_or_default();
        let img = blue_onyx_prism::image::decode(&image)?;
        let annotated = blue_onyx_prism::image::draw_predictions(&img, &preds);
        std::fs::write(path, blue_onyx_prism::image::encode_jpeg(&annotated, 95)?)
            .with_context(|| format!("writing {}", path.display()))?;
        println!("saved annotated image to {}", path.display());
    }
    Ok(failures == 0)
}

fn main() -> Result<()> {
    blue_onyx_prism::panic::install();
    let args = Args::parse();
    // Two threads: the open loop's timers and answers must not wait behind each other.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    if !rt.block_on(run(args))? {
        bail!("one or more requests failed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_is_evenly_spaced_and_bounded() {
        let s = schedule(5.0, 2.0).unwrap();
        assert_eq!(s.len(), 10);
        assert_eq!(s[0], Duration::ZERO);
        assert_eq!(s[1], Duration::from_millis(200));
        assert_eq!(s[9], Duration::from_millis(1800));
        // A fraction of a request still counts while it is due before the end.
        assert_eq!(schedule(3.0, 1.0).unwrap().len(), 3);
        assert_eq!(schedule(2.0, 1.25).unwrap().len(), 3);
        assert!(schedule(0.0, 1.0).is_err());
        assert!(schedule(1.0, f64::NAN).is_err());
        assert!(schedule(1e9, 1e9).is_err());
    }

    #[test]
    fn percentiles_are_nearest_rank() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile(&v, 50.0), 50.0);
        assert_eq!(percentile(&v, 95.0), 95.0);
        assert_eq!(percentile(&v, 99.0), 99.0);
        assert_eq!(percentile(&v, 100.0), 100.0);
        assert_eq!(percentile(&[7.0], 99.0), 7.0);
        assert_eq!(percentile(&[1.0, 2.0, 3.0], 50.0), 2.0);
        assert!(percentile(&[], 50.0).is_nan());
    }

    #[test]
    fn rss_is_read_from_the_metrics_text() {
        let text = "# HELP blue_onyx_prism_process_resident_memory_bytes Resident memory\n\
                    # TYPE blue_onyx_prism_process_resident_memory_bytes gauge\n\
                    blue_onyx_prism_process_resident_memory_bytes 123456789\n\
                    blue_onyx_prism_up 1\n";
        assert_eq!(parse_rss(text), Some(123_456_789));
        assert_eq!(parse_rss("blue_onyx_prism_up 1\n"), None);
    }

    #[test]
    fn open_loop_flags_need_each_other_and_exclude_closed_loop_ones() {
        assert!(Args::try_parse_from(["t", "--rate", "5", "--duration", "30"]).is_ok());
        assert!(Args::try_parse_from(["t", "--rate", "5"]).is_err());
        assert!(Args::try_parse_from(["t", "--duration", "5"]).is_err());
        assert!(
            Args::try_parse_from(["t", "--rate", "5", "--duration", "3", "--parallel", "2"])
                .is_err()
        );
    }
}
