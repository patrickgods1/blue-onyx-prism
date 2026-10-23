//! Request/latency counters and Prometheus text rendering.

use crate::model::filters::{SuppressCounts, SuppressReason};
use crate::sync::RwLockExt;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Upper bounds (ms, inclusive) of the histogram buckets; the last bucket is `+Inf`.
pub const BUCKET_BOUNDS_MS: [u64; 12] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
const BUCKETS: usize = BUCKET_BOUNDS_MS.len() + 1;

/// Lock-free latency histogram in milliseconds. Buckets are stored non-cumulatively (one
/// `fetch_add` per sample); the Prometheus rendering accumulates them. `min`/`max` feed the
/// Stats page.
#[derive(Debug)]
pub struct Histogram {
    buckets: [AtomicU64; BUCKETS],
    count: AtomicU64,
    sum: AtomicU64,
    min: AtomicU64,
    max: AtomicU64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    pub fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
            min: AtomicU64::new(u64::MAX),
            max: AtomicU64::new(0),
        }
    }

    /// Index of the first bucket whose bound is `>= ms` (the last one is `+Inf`).
    fn bucket_index(ms: u64) -> usize {
        BUCKET_BOUNDS_MS
            .iter()
            .position(|&b| ms <= b)
            .unwrap_or(BUCKETS - 1)
    }

    pub fn record(&self, ms: u64) {
        if let Some(b) = self.buckets.get(Self::bucket_index(ms)) {
            b.fetch_add(1, Ordering::Relaxed);
        }
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(ms, Ordering::Relaxed);
        self.min.fetch_min(ms, Ordering::Relaxed);
        self.max.fetch_max(ms, Ordering::Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    pub fn sum(&self) -> u64 {
        self.sum.load(Ordering::Relaxed)
    }

    /// Samples in each bucket (not cumulative), `BUCKET_BOUNDS_MS` order then `+Inf`.
    pub fn bucket_counts(&self) -> [u64; BUCKETS] {
        let mut out = [0; BUCKETS];
        for (o, b) in out.iter_mut().zip(&self.buckets) {
            *o = b.load(Ordering::Relaxed);
        }
        out
    }

    /// Average in ms (0.0 when nothing was recorded).
    pub fn avg(&self) -> f64 {
        let c = self.count();
        if c == 0 {
            0.0
        } else {
            self.sum() as f64 / c as f64
        }
    }

    /// Minimum in ms (0 when nothing was recorded).
    pub fn min(&self) -> u64 {
        if self.count() == 0 {
            0
        } else {
            self.min.load(Ordering::Relaxed)
        }
    }

    pub fn max(&self) -> u64 {
        self.max.load(Ordering::Relaxed)
    }
}

/// Process-wide download counters (bytes received and failed jobs), incremented by the
/// resources manager and rendered by [`render_prometheus`].
#[derive(Debug, Default)]
pub struct DownloadCounters {
    pub bytes: AtomicU64,
    pub failures: AtomicU64,
}

pub static DOWNLOADS: DownloadCounters = DownloadCounters {
    bytes: AtomicU64::new(0),
    failures: AtomicU64::new(0),
};

/// Why a detection request was rejected before inference (`requests_rejected_total{reason}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// The server-wide in-flight cap was reached (503).
    Overloaded,
    /// The upload exceeded the body limit.
    BodyTooLarge,
    /// The image could not be decoded or was rejected by its size (counted per model by the
    /// worker, see [`ModelMetrics::record_decode_error`]).
    DecodeError,
    /// Not a multipart body, no `image` field, a malformed or stalled upload.
    BadRequest,
}

impl RejectReason {
    pub const ALL: [RejectReason; 4] = [
        RejectReason::Overloaded,
        RejectReason::BodyTooLarge,
        RejectReason::DecodeError,
        RejectReason::BadRequest,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            RejectReason::Overloaded => "overloaded",
            RejectReason::BodyTooLarge => "body_too_large",
            RejectReason::DecodeError => "decode_error",
            RejectReason::BadRequest => "bad_request",
        }
    }
}

/// Why a request was not served by inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropReason {
    QueueFull,
    Timeout,
    /// Answered at once because the model was still compiling.
    Initializing,
    /// The model failed to load.
    Failed,
    /// The worker thread is gone or dropped the request without replying.
    WorkerGone,
}

impl DropReason {
    pub const ALL: [DropReason; 5] = [
        DropReason::QueueFull,
        DropReason::Timeout,
        DropReason::Initializing,
        DropReason::Failed,
        DropReason::WorkerGone,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            DropReason::QueueFull => "queue_full",
            DropReason::Timeout => "timeout",
            DropReason::Initializing => "initializing",
            DropReason::Failed => "failed",
            DropReason::WorkerGone => "worker_gone",
        }
    }
}

/// Why a night / IR frame was not routed to the model's `night_model` and ran on the model
/// itself (`night_model_fallback_total{reason}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteFallback {
    /// The night model is not served (unknown name, disabled, removed, or the model itself).
    Unknown,
    /// It is still compiling, lazy and not loaded yet, or failed.
    NotReady,
    /// Its queue is full.
    QueueFull,
    /// Its worker stopped (a reload replaced it between lookup and send).
    Stopped,
}

impl RouteFallback {
    pub const ALL: [RouteFallback; 4] = [
        RouteFallback::Unknown,
        RouteFallback::NotReady,
        RouteFallback::QueueFull,
        RouteFallback::Stopped,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            RouteFallback::Unknown => "unknown",
            RouteFallback::NotReady => "not_ready",
            RouteFallback::QueueFull => "queue_full",
            RouteFallback::Stopped => "stopped",
        }
    }
}

/// Per-model counters plus the device/provider strings, which start as the configured device
/// and an empty provider and are updated by the worker once the model has been compiled.
#[derive(Debug)]
pub struct ModelMetrics {
    pub name: String,
    /// Device requested in the config (`GPU`, `GPU.1`, `CPU`); refreshed on every (re)load.
    requested_device: RwLock<String>,
    device: RwLock<String>,
    execution_provider: RwLock<String>,
    pub requests: AtomicU64,
    dropped: [AtomicU64; DropReason::ALL.len()],
    /// Requests whose image failed to decode (or was rejected by its size) before inference.
    decode_errors: AtomicU64,
    /// Detections removed by the night / IR rules and the static-box memory, per
    /// [`SuppressReason`] (`detections_suppressed_total{reason}`).
    suppressed: [AtomicU64; SuppressReason::ALL.len()],
    /// Night / IR frames forwarded to this model's `night_model` (`night_model_routed_total`).
    night_routed: AtomicU64,
    /// Night / IR frames that ran here instead, per [`RouteFallback`].
    night_fallback: [AtomicU64; RouteFallback::ALL.len()],
    /// Throttle of the "night model unavailable" warning.
    pub night_fallback_log: LogThrottle,
    /// Verifier-cascade candidates per [`crate::verify::Outcome`] (`verify_total{outcome}`).
    verify: [AtomicU64; crate::verify::Outcome::ALL.len()],
    /// Throttle of the "verifier unavailable / late" warning.
    pub verify_log: LogThrottle,
    pub inference_ms: Histogram,
    pub process_ms: Histogram,
    pub round_trip_ms: Histogram,
    /// Throttle of the "worker queue is full" warning (one line per
    /// [`LOG_THROTTLE_INTERVAL`] instead of one per rejected request; the service writes
    /// warnings to the Windows event log).
    pub queue_full_log: LogThrottle,
    /// Clock of [`Self::now_ms`].
    started: Instant,
}

impl ModelMetrics {
    pub fn new(
        name: impl Into<String>,
        device: impl Into<String>,
        execution_provider: impl Into<String>,
    ) -> Self {
        let device = device.into();
        Self {
            name: name.into(),
            requested_device: RwLock::new(device.clone()),
            device: RwLock::new(device),
            execution_provider: RwLock::new(execution_provider.into()),
            requests: AtomicU64::new(0),
            dropped: std::array::from_fn(|_| AtomicU64::new(0)),
            decode_errors: AtomicU64::new(0),
            suppressed: std::array::from_fn(|_| AtomicU64::new(0)),
            night_routed: AtomicU64::new(0),
            night_fallback: std::array::from_fn(|_| AtomicU64::new(0)),
            night_fallback_log: LogThrottle::default(),
            verify: std::array::from_fn(|_| AtomicU64::new(0)),
            verify_log: LogThrottle::default(),
            inference_ms: Histogram::new(),
            process_ms: Histogram::new(),
            round_trip_ms: Histogram::new(),
            queue_full_log: LogThrottle::default(),
            started: Instant::now(),
        }
    }

    /// Milliseconds since these metrics were created (the clock of [`LogThrottle::allow`]).
    pub fn now_ms(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// Count one request that was not served by inference.
    pub fn record_drop(&self, reason: DropReason) {
        if let Some(d) = self.dropped.get(reason as usize) {
            d.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn dropped(&self, reason: DropReason) -> u64 {
        self.dropped
            .get(reason as usize)
            .map_or(0, |d| d.load(Ordering::Relaxed))
    }

    /// Count one request whose image could not be decoded (called by the worker's prep
    /// stage); rendered as `requests_rejected_total{reason="decode_error"}`, summed over models.
    pub fn record_decode_error(&self) {
        self.decode_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Add one request's suppressed detections.
    pub fn record_suppressed(&self, counts: &SuppressCounts) {
        for (r, n) in counts.nonzero() {
            if let Some(c) = self.suppressed.get(r as usize) {
                c.fetch_add(u64::from(n), Ordering::Relaxed);
            }
        }
    }

    /// Detections removed for `reason` so far.
    pub fn suppressed(&self, reason: SuppressReason) -> u64 {
        self.suppressed
            .get(reason as usize)
            .map_or(0, |c| c.load(Ordering::Relaxed))
    }

    /// Add one request's verifier-cascade outcomes.
    pub fn record_verify(&self, counts: &crate::verify::Counts) {
        for (i, o) in crate::verify::Outcome::ALL.into_iter().enumerate() {
            if let Some(c) = self.verify.get(i) {
                c.fetch_add(u64::from(counts.get(o)), Ordering::Relaxed);
            }
        }
    }

    /// Verifier-cascade candidates with outcome `o` so far.
    pub fn verified(&self, o: crate::verify::Outcome) -> u64 {
        crate::verify::Outcome::ALL
            .iter()
            .position(|x| *x == o)
            .and_then(|i| self.verify.get(i))
            .map_or(0, |c| c.load(Ordering::Relaxed))
    }

    /// Count one night / IR frame forwarded to the night model.
    pub fn record_night_routed(&self) {
        self.night_routed.fetch_add(1, Ordering::Relaxed);
    }

    /// Night / IR frames forwarded to the night model so far.
    pub fn night_routed(&self) -> u64 {
        self.night_routed.load(Ordering::Relaxed)
    }

    /// Count one night / IR frame that ran on this model instead of its night model.
    pub fn record_night_fallback(&self, reason: RouteFallback) {
        if let Some(c) = self.night_fallback.get(reason as usize) {
            c.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Night-model fallbacks for `reason` so far.
    pub fn night_fallback(&self, reason: RouteFallback) -> u64 {
        self.night_fallback
            .get(reason as usize)
            .map_or(0, |c| c.load(Ordering::Relaxed))
    }

    pub fn decode_errors(&self) -> u64 {
        self.decode_errors.load(Ordering::Relaxed)
    }

    /// Drops that count as failures (everything except answers while initializing).
    pub fn dropped_failures(&self) -> u64 {
        DropReason::ALL
            .iter()
            .filter(|r| **r != DropReason::Initializing)
            .map(|r| self.dropped(*r))
            .sum()
    }

    /// Device requested in the config.
    pub fn requested_device(&self) -> String {
        read_string(&self.requested_device)
    }

    /// A (re)load starts: the counters stay, the device/provider restart from the request.
    pub fn begin_load(&self, requested_device: impl Into<String>) {
        let d = requested_device.into();
        write_string(&self.requested_device, d.clone());
        write_string(&self.device, d);
        write_string(&self.execution_provider, String::new());
    }

    /// Device the model actually runs on (the requested device until it is loaded).
    pub fn device(&self) -> String {
        read_string(&self.device)
    }

    /// `executionProvider` string of the loaded model (empty until it is loaded).
    pub fn execution_provider(&self) -> String {
        read_string(&self.execution_provider)
    }

    /// Record where the model ended up after compilation (called by the worker).
    pub fn set_loaded(&self, device: impl Into<String>, execution_provider: impl Into<String>) {
        write_string(&self.device, device.into());
        write_string(&self.execution_provider, execution_provider.into());
    }
}

fn read_string(l: &RwLock<String>) -> String {
    l.read_recover().clone()
}

fn write_string(l: &RwLock<String>, v: String) {
    *l.write_recover() = v;
}

/// Lifecycle state of a model as exported to Prometheus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStateLabel {
    /// Compiling (or waiting for its turn to compile).
    Initializing,
    Ready,
    Failed,
    /// `lazy: true` and not requested yet.
    Lazy,
}

impl ModelStateLabel {
    pub fn as_str(self) -> &'static str {
        match self {
            ModelStateLabel::Initializing => "initializing",
            ModelStateLabel::Ready => "ready",
            ModelStateLabel::Failed => "failed",
            ModelStateLabel::Lazy => "lazy",
        }
    }
}

/// Point-in-time per-model values that live outside [`ModelMetrics`] (worker state, queue).
/// Built by the registry for each `/prometheus` scrape.
#[derive(Debug, Clone)]
pub struct ModelGauges {
    pub name: String,
    pub device: String,
    pub provider: String,
    pub state: ModelStateLabel,
    pub queue_length: usize,
    pub queue_capacity: usize,
}

#[derive(Debug)]
pub struct Metrics {
    pub started: Instant,
    pub version: String,
    /// Every model seen since the process started (removed models keep their series).
    pub models: RwLock<Vec<Arc<ModelMetrics>>>,
    /// Registry generations started after the first one.
    pub restarts: AtomicU64,
    /// Detection requests answered 503 because the server-wide in-flight cap was reached
    /// (see [`Metrics::try_begin_detection`]).
    pub rejected_overloaded: AtomicU64,
    /// Detection uploads larger than the body limit.
    pub rejected_body_too_large: AtomicU64,
    /// Detection requests without a usable multipart body (see [`RejectReason::BadRequest`]).
    pub rejected_bad_request: AtomicU64,
    /// Fixed server-wide in-flight cap for detection requests; 0 (the default) derives it from
    /// the models' queue capacities on every request (see `server::detect::max_inflight`).
    pub inflight_limit: AtomicUsize,
    /// Rate limit of the "server is overloaded" warning.
    pub overload_log: LogThrottle,
    /// Detection requests being handled now, process-wide (every generation counts: an
    /// in-flight request of an old generation still holds its upload in memory).
    inflight: AtomicUsize,
}

/// A detection request admitted under the server-wide in-flight cap; releases its slot when
/// dropped (the handler holds it from before the body read until the reply is built).
#[derive(Debug)]
pub struct InflightPermit(Arc<Metrics>);

impl Drop for InflightPermit {
    fn drop(&mut self) {
        self.0.inflight.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Smallest gap between two warnings of the same rate-limited kind.
pub const LOG_THROTTLE_INTERVAL: Duration = Duration::from_secs(10);

/// Lets a repeated warning through at most once per [`LOG_THROTTLE_INTERVAL`] and counts the
/// ones it held back, so a flood (e.g. every rejected request under overload) logs one line per
/// interval instead of one per event.
#[derive(Debug)]
pub struct LogThrottle {
    /// Time of the last warning let through, in ms since `Metrics::started` (`u64::MAX`: never).
    last_ms: AtomicU64,
    suppressed: AtomicU64,
}

impl Default for LogThrottle {
    fn default() -> Self {
        Self {
            last_ms: AtomicU64::new(u64::MAX),
            suppressed: AtomicU64::new(0),
        }
    }
}

impl LogThrottle {
    /// At `now_ms` (any monotonic millisecond clock): `Some(n)` when the warning should be
    /// logged now, `n` being how many were held back since the last one; `None` to skip it.
    pub fn allow(&self, now_ms: u64) -> Option<u64> {
        let interval = u64::try_from(LOG_THROTTLE_INTERVAL.as_millis()).unwrap_or(u64::MAX);
        let last = self.last_ms.load(Ordering::Acquire);
        let due = last == u64::MAX || now_ms.saturating_sub(last) >= interval;
        if due
            && self
                .last_ms
                .compare_exchange(last, now_ms, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        {
            Some(self.suppressed.swap(0, Ordering::AcqRel))
        } else {
            self.suppressed.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
}

impl Metrics {
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            started: Instant::now(),
            version: version.into(),
            models: RwLock::new(Vec::new()),
            restarts: AtomicU64::new(0),
            rejected_overloaded: AtomicU64::new(0),
            rejected_body_too_large: AtomicU64::new(0),
            rejected_bad_request: AtomicU64::new(0),
            inflight_limit: AtomicUsize::new(0),
            overload_log: LogThrottle::default(),
            inflight: AtomicUsize::new(0),
        }
    }

    /// Admit one more detection request when fewer than `limit` are in flight; `None` when the
    /// server is at its cap (the caller answers 503 and counts it in `rejected_overloaded`).
    pub fn try_begin_detection(self: &Arc<Self>, limit: usize) -> Option<InflightPermit> {
        self.inflight
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < limit).then_some(n + 1)
            })
            .ok()
            .map(|_| InflightPermit(self.clone()))
    }

    /// Count one rejected detection request ([`RejectReason::DecodeError`] is counted per
    /// model instead, see [`ModelMetrics::record_decode_error`]; recording it here is a no-op).
    pub fn record_rejected(&self, reason: RejectReason) {
        let c = match reason {
            RejectReason::Overloaded => &self.rejected_overloaded,
            RejectReason::BodyTooLarge => &self.rejected_body_too_large,
            RejectReason::BadRequest => &self.rejected_bad_request,
            RejectReason::DecodeError => return,
        };
        c.fetch_add(1, Ordering::Relaxed);
    }

    /// Rejected detection requests for `reason` (decode errors summed over every model).
    pub fn rejected(&self, reason: RejectReason) -> u64 {
        match reason {
            RejectReason::Overloaded => self.rejected_overloaded.load(Ordering::Relaxed),
            RejectReason::BodyTooLarge => self.rejected_body_too_large.load(Ordering::Relaxed),
            RejectReason::BadRequest => self.rejected_bad_request.load(Ordering::Relaxed),
            RejectReason::DecodeError => self
                .models
                .read_recover()
                .iter()
                .map(|m| m.decode_errors())
                .sum(),
        }
    }

    /// Detection requests being handled now.
    pub fn inflight(&self) -> usize {
        self.inflight.load(Ordering::Acquire)
    }

    /// Milliseconds since the process started (the clock of [`LogThrottle::allow`]).
    pub fn now_ms(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// The entry for `name` (keyed by [`crate::registry::normalize_name`]), created on first use
    /// so counters survive registry restarts.
    pub fn model(&self, name: &str) -> Arc<ModelMetrics> {
        let key = crate::registry::normalize_name(name);
        let mut g = self.models.write_recover();
        if let Some(m) = g
            .iter()
            .find(|m| crate::registry::normalize_name(&m.name) == key)
        {
            return m.clone();
        }
        let m = Arc::new(ModelMetrics::new(name, "", ""));
        g.push(m.clone());
        m
    }

    pub fn uptime(&self) -> Duration {
        self.started.elapsed()
    }
}

/// Process resource gauges that are cheap to read on this platform (`None`: not available
/// here, so the series is left out). Linux reads `/proc/self/status` and `/proc/self/fd`;
/// macOS and Windows expose neither through `std`, so both series are omitted there.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProcessStats {
    pub resident_memory_bytes: Option<u64>,
    pub open_fds: Option<u64>,
}

impl ProcessStats {
    #[cfg(target_os = "linux")]
    pub fn read() -> Self {
        let resident_memory_bytes = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| parse_vm_rss(&s));
        // The count includes the descriptor `read_dir` itself holds open.
        let open_fds = std::fs::read_dir("/proc/self/fd")
            .ok()
            .map(|d| d.count() as u64)
            .map(|n| n.saturating_sub(1));
        Self {
            resident_memory_bytes,
            open_fds,
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn read() -> Self {
        Self::default()
    }
}

/// `VmRSS` of a `/proc/<pid>/status` text, in bytes (the kernel prints it in kB).
#[cfg_attr(
    not(target_os = "linux"),
    allow(dead_code, reason = "used on Linux and in tests")
)]
fn parse_vm_rss(status: &str) -> Option<u64> {
    let line = status.lines().find(|l| l.starts_with("VmRSS:"))?;
    let mut it = line.trim_start_matches("VmRSS:").split_whitespace();
    let kb: u64 = it.next()?.parse().ok()?;
    match it.next() {
        Some("kB") | None => kb.checked_mul(1024),
        Some(_) => None,
    }
}

fn escape_label(v: &str) -> String {
    v.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

const PREFIX: &str = "blue_onyx_prism_";

/// Render all metrics in the Prometheus text exposition format. `gauges` adds the per-model
/// `model_info`, `model_ready`, `queue_length` and `queue_capacity` series.
pub fn render_prometheus(m: &Metrics, gauges: &[ModelGauges]) -> String {
    let models: Vec<Arc<ModelMetrics>> = m.models.read_recover().clone();
    let mut out = String::new();

    let _ = writeln!(out, "# HELP {PREFIX}info Build information.");
    let _ = writeln!(out, "# TYPE {PREFIX}info gauge");
    let _ = writeln!(
        out,
        "{PREFIX}info{{version=\"{}\"}} 1",
        escape_label(&m.version)
    );
    let _ = writeln!(
        out,
        "# HELP {PREFIX}build_info Build information: version, git commit and build date; value is always 1."
    );
    let _ = writeln!(out, "# TYPE {PREFIX}build_info gauge");
    let _ = writeln!(
        out,
        "{PREFIX}build_info{{version=\"{}\",git_sha=\"{}\",build_date=\"{}\"}} 1",
        escape_label(&m.version),
        escape_label(crate::GIT_SHA),
        escape_label(crate::BUILD_DATE)
    );
    let _ = writeln!(
        out,
        "# HELP {PREFIX}uptime_seconds Seconds since the service started."
    );
    let _ = writeln!(out, "# TYPE {PREFIX}uptime_seconds gauge");
    let _ = writeln!(out, "{PREFIX}uptime_seconds {}", m.uptime().as_secs());

    let _ = writeln!(
        out,
        "# HELP {PREFIX}restarts_total Registry restarts since the process started."
    );
    let _ = writeln!(out, "# TYPE {PREFIX}restarts_total counter");
    let _ = writeln!(
        out,
        "{PREFIX}restarts_total {}",
        m.restarts.load(Ordering::Relaxed)
    );

    let _ = writeln!(
        out,
        "# HELP {PREFIX}requests_rejected_total Detection requests rejected before inference, by reason (overloaded: the server-wide in-flight cap was reached; body_too_large: the upload exceeded the body limit; decode_error: the image could not be decoded; bad_request: no usable multipart image upload)."
    );
    let _ = writeln!(out, "# TYPE {PREFIX}requests_rejected_total counter");
    for reason in RejectReason::ALL {
        let _ = writeln!(
            out,
            "{PREFIX}requests_rejected_total{{reason=\"{}\"}} {}",
            reason.as_str(),
            m.rejected(reason)
        );
    }
    let downloads = [
        (
            "download_bytes_total",
            "Bytes received by on-demand resource downloads.",
            &DOWNLOADS.bytes,
        ),
        (
            "download_failures_total",
            "On-demand resource download jobs that failed.",
            &DOWNLOADS.failures,
        ),
    ];
    for (name, help, counter) in downloads {
        let _ = writeln!(out, "# HELP {PREFIX}{name} {help}");
        let _ = writeln!(out, "# TYPE {PREFIX}{name} counter");
        let _ = writeln!(out, "{PREFIX}{name} {}", counter.load(Ordering::Relaxed));
    }
    let p = ProcessStats::read();
    if let Some(rss) = p.resident_memory_bytes {
        let _ = writeln!(
            out,
            "# HELP {PREFIX}process_resident_memory_bytes Resident memory of the process in bytes."
        );
        let _ = writeln!(out, "# TYPE {PREFIX}process_resident_memory_bytes gauge");
        let _ = writeln!(out, "{PREFIX}process_resident_memory_bytes {rss}");
    }
    if let Some(fds) = p.open_fds {
        let _ = writeln!(
            out,
            "# HELP {PREFIX}process_open_fds Open file descriptors of the process."
        );
        let _ = writeln!(out, "# TYPE {PREFIX}process_open_fds gauge");
        let _ = writeln!(out, "{PREFIX}process_open_fds {fds}");
    }
    let _ = writeln!(
        out,
        "# HELP {PREFIX}inflight_requests Detection requests being handled now (body read, queue wait, inference)."
    );
    let _ = writeln!(out, "# TYPE {PREFIX}inflight_requests gauge");
    let _ = writeln!(out, "{PREFIX}inflight_requests {}", m.inflight());

    let _ = writeln!(
        out,
        "# HELP {PREFIX}requests_total Detection requests received."
    );
    let _ = writeln!(out, "# TYPE {PREFIX}requests_total counter");
    for mm in &models {
        let _ = writeln!(
            out,
            "{PREFIX}requests_total{{model=\"{}\"}} {}",
            escape_label(&mm.name),
            mm.requests.load(Ordering::Relaxed)
        );
    }
    let _ = writeln!(
        out,
        "# HELP {PREFIX}dropped_total Requests not served by inference, by reason (queue_full|timeout|initializing|failed|worker_gone)."
    );
    let _ = writeln!(out, "# TYPE {PREFIX}dropped_total counter");
    for mm in &models {
        for reason in DropReason::ALL {
            let _ = writeln!(
                out,
                "{PREFIX}dropped_total{{model=\"{}\",reason=\"{}\"}} {}",
                escape_label(&mm.name),
                reason.as_str(),
                mm.dropped(reason)
            );
        }
    }

    let _ = writeln!(
        out,
        "# HELP {PREFIX}detections_suppressed_total Detections removed by the night / IR rules and the verifier, by reason (night_threshold|label_threshold|aspect|min_area|overlap|static|verify)."
    );
    let _ = writeln!(out, "# TYPE {PREFIX}detections_suppressed_total counter");
    for mm in &models {
        for reason in SuppressReason::ALL {
            let _ = writeln!(
                out,
                "{PREFIX}detections_suppressed_total{{model=\"{}\",reason=\"{}\"}} {}",
                escape_label(&mm.name),
                reason.as_str(),
                mm.suppressed(reason)
            );
        }
    }

    let _ = writeln!(
        out,
        "# HELP {PREFIX}night_model_routed_total Night / IR frames forwarded to the model's night_model."
    );
    let _ = writeln!(out, "# TYPE {PREFIX}night_model_routed_total counter");
    for mm in &models {
        let _ = writeln!(
            out,
            "{PREFIX}night_model_routed_total{{model=\"{}\"}} {}",
            escape_label(&mm.name),
            mm.night_routed()
        );
    }
    let _ = writeln!(
        out,
        "# HELP {PREFIX}night_model_fallback_total Night / IR frames that ran on the model itself because its night_model was unavailable, by reason (unknown|not_ready|queue_full|stopped)."
    );
    let _ = writeln!(out, "# TYPE {PREFIX}night_model_fallback_total counter");
    for mm in &models {
        for reason in RouteFallback::ALL {
            let _ = writeln!(
                out,
                "{PREFIX}night_model_fallback_total{{model=\"{}\",reason=\"{}\"}} {}",
                escape_label(&mm.name),
                reason.as_str(),
                mm.night_fallback(reason)
            );
        }
    }

    let _ = writeln!(
        out,
        "# HELP {PREFIX}verify_total Candidate boxes checked by the model's verifier (verify), by outcome (kept|rejected|unverified; unverified = the verifier was unavailable, busy, late or failed, or over max_crops, and the box was kept)."
    );
    let _ = writeln!(out, "# TYPE {PREFIX}verify_total counter");
    for mm in &models {
        for o in crate::verify::Outcome::ALL {
            let _ = writeln!(
                out,
                "{PREFIX}verify_total{{model=\"{}\",outcome=\"{}\"}} {}",
                escape_label(&mm.name),
                o.as_str(),
                mm.verified(o)
            );
        }
    }

    type Getter = fn(&ModelMetrics) -> &Histogram;
    let stats: [(&str, &str, Getter); 3] = [
        ("inference_ms", "Model inference time in ms", |mm| {
            &mm.inference_ms
        }),
        ("process_ms", "Total processing time in ms", |mm| {
            &mm.process_ms
        }),
        ("round_trip_ms", "Request round trip time in ms", |mm| {
            &mm.round_trip_ms
        }),
    ];
    for (base, help, get) in stats {
        // The histogram keeps the `_ms` unit in its name (blue-onyx dashboard compatibility).
        let _ = writeln!(out, "# HELP {PREFIX}{base} {help} (histogram).");
        let _ = writeln!(out, "# TYPE {PREFIX}{base} histogram");
        for mm in &models {
            let h = get(mm);
            let label = escape_label(&mm.name);
            let counts = h.bucket_counts();
            let mut cum = 0u64;
            for (i, c) in counts.iter().enumerate() {
                cum += c;
                let le = BUCKET_BOUNDS_MS
                    .get(i)
                    .map_or_else(|| "+Inf".to_string(), |b| b.to_string());
                let _ = writeln!(
                    out,
                    "{PREFIX}{base}_bucket{{model=\"{label}\",le=\"{le}\"}} {cum}"
                );
            }
            let _ = writeln!(out, "{PREFIX}{base}_sum{{model=\"{label}\"}} {}", h.sum());
            let _ = writeln!(out, "{PREFIX}{base}_count{{model=\"{label}\"}} {cum}");
        }
        for suffix in ["avg", "min", "max"] {
            let name = format!("{base}_{suffix}");
            let _ = writeln!(out, "# HELP {PREFIX}{name} {help} ({suffix}).");
            let _ = writeln!(out, "# TYPE {PREFIX}{name} gauge");
            for mm in &models {
                let s = get(mm);
                let label = escape_label(&mm.name);
                let value = match suffix {
                    "avg" => format!("{:.3}", s.avg()),
                    "min" => s.min().to_string(),
                    _ => s.max().to_string(),
                };
                let _ = writeln!(out, "{PREFIX}{name}{{model=\"{label}\"}} {value}");
            }
        }
    }

    let gauge_header = |out: &mut String, name: &str, help: &str| {
        let _ = writeln!(out, "# HELP {PREFIX}{name} {help}");
        let _ = writeln!(out, "# TYPE {PREFIX}{name} gauge");
    };
    gauge_header(
        &mut out,
        "model_info",
        "Model device, execution provider and state (initializing|ready|failed|lazy); value is always 1.",
    );
    for g in gauges {
        let _ = writeln!(
            out,
            "{PREFIX}model_info{{model=\"{}\",device=\"{}\",provider=\"{}\",state=\"{}\"}} 1",
            escape_label(&g.name),
            escape_label(&g.device),
            escape_label(&g.provider),
            g.state.as_str()
        );
    }
    gauge_header(
        &mut out,
        "model_ready",
        "1 when the model is compiled and serving requests, else 0.",
    );
    for g in gauges {
        let ready = u8::from(g.state == ModelStateLabel::Ready);
        let _ = writeln!(
            out,
            "{PREFIX}model_ready{{model=\"{}\"}} {ready}",
            escape_label(&g.name)
        );
    }
    gauge_header(
        &mut out,
        "queue_length",
        "Requests waiting in the worker queue.",
    );
    for g in gauges {
        let _ = writeln!(
            out,
            "{PREFIX}queue_length{{model=\"{}\"}} {}",
            escape_label(&g.name),
            g.queue_length
        );
    }
    gauge_header(
        &mut out,
        "queue_capacity",
        "Effective worker queue depth limit (sized after warm-up).",
    );
    for g in gauges {
        let _ = writeln!(
            out,
            "{PREFIX}queue_capacity{{model=\"{}\"}} {}",
            escape_label(&g.name),
            g.queue_capacity
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inflight_permits_respect_the_limit() {
        let m = Arc::new(Metrics::new("0"));
        let a = m.try_begin_detection(2).expect("first");
        let b = m.try_begin_detection(2).expect("second");
        assert!(m.try_begin_detection(2).is_none());
        assert_eq!(m.inflight(), 2);
        drop(a);
        assert_eq!(m.inflight(), 1);
        let c = m.try_begin_detection(2).expect("slot freed");
        drop((b, c));
        assert_eq!(m.inflight(), 0);
        assert!(m.try_begin_detection(0).is_none());
    }

    #[test]
    fn log_throttle_once_per_interval() {
        let t = LogThrottle::default();
        let iv = u64::try_from(LOG_THROTTLE_INTERVAL.as_millis()).unwrap();
        assert_eq!(t.allow(5), Some(0));
        assert_eq!(t.allow(6), None);
        assert_eq!(t.allow(5 + iv - 1), None);
        assert_eq!(t.allow(5 + iv), Some(2));
        assert_eq!(t.allow(5 + iv + 1), None);
        assert_eq!(t.allow(5 + 3 * iv), Some(1));
    }

    #[test]
    fn prometheus_verify_outcomes() {
        use crate::verify::{Counts, Outcome};
        let m = Metrics::new("0");
        let mm = m.model("a");
        mm.record_verify(&Counts {
            kept: 2,
            rejected: 1,
            unverified: 0,
        });
        mm.record_verify(&Counts {
            kept: 0,
            rejected: 1,
            unverified: 3,
        });
        assert_eq!(mm.verified(Outcome::Kept), 2);
        assert_eq!(mm.verified(Outcome::Rejected), 2);
        assert_eq!(mm.verified(Outcome::Unverified), 3);
        let text = render_prometheus(&m, &[]);
        assert!(text.contains("# TYPE blue_onyx_prism_verify_total counter"));
        assert!(text.contains("blue_onyx_prism_verify_total{model=\"a\",outcome=\"rejected\"} 2"));
        assert!(
            text.contains("blue_onyx_prism_verify_total{model=\"a\",outcome=\"unverified\"} 3")
        );
        assert!(text.contains(
            "blue_onyx_prism_detections_suppressed_total{model=\"a\",reason=\"verify\"} 0"
        ));
    }

    #[test]
    fn prometheus_suppressed_by_reason() {
        let m = Metrics::new("0");
        let mm = m.model("a");
        let mut c = SuppressCounts::default();
        c.add(SuppressReason::Static, 2);
        c.add(SuppressReason::NightThreshold, 1);
        mm.record_suppressed(&c);
        mm.record_suppressed(&c);
        assert_eq!(mm.suppressed(SuppressReason::Static), 4);
        let text = render_prometheus(&m, &[]);
        assert!(text.contains("# TYPE blue_onyx_prism_detections_suppressed_total counter"));
        assert!(text.contains(
            "blue_onyx_prism_detections_suppressed_total{model=\"a\",reason=\"static\"} 4"
        ));
        assert!(text.contains(
            "blue_onyx_prism_detections_suppressed_total{model=\"a\",reason=\"aspect\"} 0"
        ));
    }

    #[test]
    fn prometheus_night_model_routing() {
        let m = Metrics::new("0");
        let mm = m.model("a");
        mm.record_night_routed();
        mm.record_night_routed();
        mm.record_night_fallback(RouteFallback::NotReady);
        assert_eq!(mm.night_routed(), 2);
        assert_eq!(mm.night_fallback(RouteFallback::NotReady), 1);
        assert_eq!(mm.night_fallback(RouteFallback::QueueFull), 0);
        let text = render_prometheus(&m, &[]);
        assert!(text.contains("# TYPE blue_onyx_prism_night_model_routed_total counter"));
        assert!(text.contains("blue_onyx_prism_night_model_routed_total{model=\"a\"} 2"));
        assert!(text.contains(
            "blue_onyx_prism_night_model_fallback_total{model=\"a\",reason=\"not_ready\"} 1"
        ));
        assert!(text.contains(
            "blue_onyx_prism_night_model_fallback_total{model=\"a\",reason=\"unknown\"} 0"
        ));
    }

    #[test]
    fn prometheus_rejected_and_inflight() {
        let m = Arc::new(Metrics::new("0"));
        m.rejected_overloaded.store(3, Ordering::Relaxed);
        let _p = m.try_begin_detection(5).unwrap();
        let text = render_prometheus(&m, &[]);
        assert!(text.contains("blue_onyx_prism_requests_rejected_total{reason=\"overloaded\"} 3"));
        assert!(text.contains("# TYPE blue_onyx_prism_requests_rejected_total counter"));
        assert!(text.contains("blue_onyx_prism_inflight_requests 1"));
    }

    #[test]
    fn prometheus_reject_reasons_build_info_and_downloads() {
        let m = Arc::new(Metrics::new("9.8.7"));
        m.record_rejected(RejectReason::BodyTooLarge);
        m.record_rejected(RejectReason::BadRequest);
        m.record_rejected(RejectReason::BadRequest);
        m.record_rejected(RejectReason::DecodeError); // per model: no-op here
        m.model("a").record_decode_error();
        m.model("b").record_decode_error();
        m.model("b").record_decode_error();
        assert_eq!(m.rejected(RejectReason::DecodeError), 3);
        let text = render_prometheus(&m, &[]);
        let has = |line: &str| {
            assert!(
                text.lines().any(|l| l == line),
                "missing line {line:?} in:\n{text}"
            )
        };
        has("blue_onyx_prism_requests_rejected_total{reason=\"overloaded\"} 0");
        has("blue_onyx_prism_requests_rejected_total{reason=\"body_too_large\"} 1");
        has("blue_onyx_prism_requests_rejected_total{reason=\"decode_error\"} 3");
        has("blue_onyx_prism_requests_rejected_total{reason=\"bad_request\"} 2");
        has("blue_onyx_prism_info{version=\"9.8.7\"} 1");
        has("# TYPE blue_onyx_prism_build_info gauge");
        has(&format!(
            "blue_onyx_prism_build_info{{version=\"9.8.7\",git_sha=\"{}\",build_date=\"{}\"}} 1",
            crate::GIT_SHA,
            crate::BUILD_DATE
        ));
        // Process-wide counters (other tests may add to them): present and numeric.
        has("# TYPE blue_onyx_prism_download_bytes_total counter");
        has("# TYPE blue_onyx_prism_download_failures_total counter");
        for name in ["download_bytes_total", "download_failures_total"] {
            let prefix = format!("blue_onyx_prism_{name} ");
            let line = text.lines().find(|l| l.starts_with(&prefix)).unwrap();
            assert!(line[prefix.len()..].parse::<u64>().is_ok(), "{line}");
        }
        let linux = cfg!(target_os = "linux");
        assert_eq!(
            text.contains("blue_onyx_prism_process_resident_memory_bytes "),
            linux
        );
        assert_eq!(text.contains("blue_onyx_prism_process_open_fds "), linux);
        for l in text.lines().filter(|l| !l.starts_with('#')) {
            let v = l.rsplit(' ').next().unwrap();
            assert!(v.parse::<f64>().is_ok(), "bad sample line {l:?}");
        }
    }

    #[test]
    fn vm_rss_parsing() {
        let status = "Name:\tblue\nVmPeak:\t  9999 kB\nVmRSS:\t   12345 kB\nThreads:\t4\n";
        assert_eq!(parse_vm_rss(status), Some(12345 * 1024));
        assert_eq!(parse_vm_rss("Name: x\n"), None);
        assert_eq!(parse_vm_rss("VmRSS: lots kB\n"), None);
        assert_eq!(parse_vm_rss("VmRSS: 5 MB\n"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn process_stats_on_linux() {
        let p = ProcessStats::read();
        assert!(p.resident_memory_bytes.is_some_and(|b| b > 0));
        assert!(p.open_fds.is_some_and(|n| n >= 3));
    }

    #[test]
    fn histogram_basics() {
        let s = Histogram::new();
        assert_eq!((s.count(), s.min(), s.max()), (0, 0, 0));
        assert_eq!(s.avg(), 0.0);
        s.record(10);
        s.record(30);
        s.record(20);
        assert_eq!(s.count(), 3);
        assert_eq!(s.min(), 10);
        assert_eq!(s.max(), 30);
        assert!((s.avg() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn prometheus_rendering() {
        let m = Metrics::new("1.2.3");
        let mm = Arc::new(ModelMetrics::new("ipcam\"x", "CPU", "OpenVINO CPU"));
        mm.requests.fetch_add(5, Ordering::Relaxed);
        mm.record_drop(DropReason::QueueFull);
        mm.record_drop(DropReason::Initializing);
        mm.record_drop(DropReason::Initializing);
        m.restarts.fetch_add(1, Ordering::Relaxed);
        mm.inference_ms.record(12);
        mm.inference_ms.record(8);
        m.models.write().unwrap().push(mm);
        let text = render_prometheus(&m, &[]);
        assert!(text.contains("blue_onyx_prism_requests_total{model=\"ipcam\\\"x\"} 5"));
        assert!(text.contains(
            "blue_onyx_prism_dropped_total{model=\"ipcam\\\"x\",reason=\"queue_full\"} 1"
        ));
        assert!(text.contains(
            "blue_onyx_prism_dropped_total{model=\"ipcam\\\"x\",reason=\"initializing\"} 2"
        ));
        assert!(text.contains("blue_onyx_prism_restarts_total 1"));
        assert!(text.contains("# TYPE blue_onyx_prism_inference_ms histogram"));
        assert!(
            text.contains("blue_onyx_prism_inference_ms_bucket{model=\"ipcam\\\"x\",le=\"10\"} 1")
        );
        assert!(
            text.contains("blue_onyx_prism_inference_ms_bucket{model=\"ipcam\\\"x\",le=\"20\"} 2")
        );
        assert!(
            text.contains(
                "blue_onyx_prism_inference_ms_bucket{model=\"ipcam\\\"x\",le=\"+Inf\"} 2"
            )
        );
        assert!(text.contains("blue_onyx_prism_inference_ms_sum{model=\"ipcam\\\"x\"} 20"));
        assert!(text.contains("blue_onyx_prism_inference_ms_count{model=\"ipcam\\\"x\"} 2"));
        assert!(text.contains("blue_onyx_prism_inference_ms_avg{model=\"ipcam\\\"x\"} 10.000"));
        assert!(text.contains("blue_onyx_prism_inference_ms_min{model=\"ipcam\\\"x\"} 8"));
        assert!(text.contains("blue_onyx_prism_inference_ms_max{model=\"ipcam\\\"x\"} 12"));
        assert!(text.contains("blue_onyx_prism_uptime_seconds "));
        assert!(text.contains("# TYPE blue_onyx_prism_round_trip_ms_avg gauge"));
        assert!(text.contains("# TYPE blue_onyx_prism_model_info gauge"));
        assert!(!text.contains("blue_onyx_prism_model_info{"));
    }

    #[test]
    fn device_and_provider_update_after_load() {
        let mm = ModelMetrics::new("m", "GPU", "");
        assert_eq!(mm.requested_device(), "GPU");
        assert_eq!(mm.device(), "GPU");
        assert_eq!(mm.execution_provider(), "");
        mm.set_loaded("CPU", "OpenVINO CPU (Apple M2, fallback)");
        assert_eq!(mm.requested_device(), "GPU");
        assert_eq!(mm.device(), "CPU");
        assert_eq!(mm.execution_provider(), "OpenVINO CPU (Apple M2, fallback)");
    }

    #[test]
    fn prometheus_model_gauges() {
        let m = Metrics::new("0.1.0");
        let gauges = vec![
            ModelGauges {
                name: "yolo26s".into(),
                device: "GPU".into(),
                provider: "OpenVINO GPU (Intel(R) UHD Graphics 630 (iGPU))".into(),
                state: ModelStateLabel::Ready,
                queue_length: 2,
                queue_capacity: 30,
            },
            ModelGauges {
                name: "ipcam-general".into(),
                device: "CPU".into(),
                provider: String::new(),
                state: ModelStateLabel::Initializing,
                queue_length: 0,
                queue_capacity: 64,
            },
            ModelGauges {
                name: "rt\"detr".into(),
                device: "CPU".into(),
                provider: String::new(),
                state: ModelStateLabel::Lazy,
                queue_length: 0,
                queue_capacity: 64,
            },
            ModelGauges {
                name: "broken".into(),
                device: "CPU".into(),
                provider: String::new(),
                state: ModelStateLabel::Failed,
                queue_length: 0,
                queue_capacity: 0,
            },
        ];
        let text = render_prometheus(&m, &gauges);
        let has = |line: &str| {
            assert!(
                text.lines().any(|l| l == line),
                "missing line {line:?} in:\n{text}"
            )
        };
        has("# TYPE blue_onyx_prism_model_info gauge");
        has(
            "blue_onyx_prism_model_info{model=\"yolo26s\",device=\"GPU\",provider=\"OpenVINO GPU (Intel(R) UHD Graphics 630 (iGPU))\",state=\"ready\"} 1",
        );
        has(
            "blue_onyx_prism_model_info{model=\"ipcam-general\",device=\"CPU\",provider=\"\",state=\"initializing\"} 1",
        );
        has(
            "blue_onyx_prism_model_info{model=\"rt\\\"detr\",device=\"CPU\",provider=\"\",state=\"lazy\"} 1",
        );
        has(
            "blue_onyx_prism_model_info{model=\"broken\",device=\"CPU\",provider=\"\",state=\"failed\"} 1",
        );
        has("blue_onyx_prism_model_ready{model=\"yolo26s\"} 1");
        has("blue_onyx_prism_model_ready{model=\"ipcam-general\"} 0");
        has("blue_onyx_prism_model_ready{model=\"broken\"} 0");
        has("blue_onyx_prism_queue_length{model=\"yolo26s\"} 2");
        has("blue_onyx_prism_queue_capacity{model=\"yolo26s\"} 30");
        has("blue_onyx_prism_queue_capacity{model=\"broken\"} 0");
        has("# TYPE blue_onyx_prism_queue_length gauge");
        has("# TYPE blue_onyx_prism_queue_capacity gauge");
        has("# TYPE blue_onyx_prism_model_ready gauge");
        // Every sample line is `name{labels} value` with a numeric value.
        for l in text.lines().filter(|l| !l.starts_with('#')) {
            let v = l.rsplit(' ').next().unwrap();
            assert!(v.parse::<f64>().is_ok(), "bad sample line {l:?}");
        }
    }

    #[test]
    fn histogram_bucket_boundaries() {
        let h = Histogram::new();
        for ms in [0, 1, 2, 3, 5000, 5001] {
            h.record(ms);
        }
        let c = h.bucket_counts();
        assert_eq!(c[0], 2, "0 and 1 ms land in le=1");
        assert_eq!(c[1], 1, "2 ms lands in le=2");
        assert_eq!(c[2], 1, "3 ms lands in le=5");
        assert_eq!(c[11], 1, "5000 ms lands in le=5000");
        assert_eq!(c[12], 1, "5001 ms lands in +Inf");
        assert_eq!(c.iter().sum::<u64>(), 6);
        assert_eq!((h.count(), h.sum()), (6, 10_007));
    }

    #[test]
    fn model_entry_is_shared_by_normalized_name() {
        let m = Metrics::new("0");
        let a = m.model("IPcam-General");
        let b = m.model("ipcam-general.onnx");
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(m.models.read().unwrap().len(), 1);
        assert!(!Arc::ptr_eq(&a, &m.model("other")));
    }
}
