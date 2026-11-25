//! Request/latency counters and Prometheus text rendering.

use crate::model::filters::{SuppressCounts, SuppressReason};
use crate::sync::{LockExt, RwLockExt};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::net::IpAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
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
    /// Not a multipart body, no `image` field, a malformed upload.
    BadRequest,
    /// One peer already had its limit of uploads still being read (503). Counted apart from
    /// [`RejectReason::Overloaded`]: this one points at a stalled or very slow sender, that one
    /// at more frames than the models can serve.
    PeerBodyReads,
    /// The upload was not received within the read deadline.
    BodyTimeout,
    /// The uploads in flight across all requests would pass their byte budget (503,
    /// `server::body_budget`): many large frames at once, not one slow sender.
    UploadBudget,
    /// `/v1/vision/custom/<model>` named no enabled model (a camera still calling a model that
    /// was removed, renamed or disabled).
    UnknownModel,
}

impl RejectReason {
    pub const ALL: [RejectReason; 8] = [
        RejectReason::Overloaded,
        RejectReason::BodyTooLarge,
        RejectReason::DecodeError,
        RejectReason::BadRequest,
        RejectReason::PeerBodyReads,
        RejectReason::BodyTimeout,
        RejectReason::UploadBudget,
        RejectReason::UnknownModel,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            RejectReason::Overloaded => "overloaded",
            RejectReason::BodyTooLarge => "body_too_large",
            RejectReason::DecodeError => "decode_error",
            RejectReason::BadRequest => "bad_request",
            RejectReason::PeerBodyReads => "peer_body_reads",
            RejectReason::BodyTimeout => "body_timeout",
            RejectReason::UploadBudget => "upload_budget",
            RejectReason::UnknownModel => "unknown_model",
        }
    }

    /// What the reason means, in the `requests_rejected_total` HELP text.
    pub fn help(self) -> &'static str {
        match self {
            RejectReason::Overloaded => "the server-wide in-flight cap was reached",
            RejectReason::BodyTooLarge => "the upload exceeded the body limit",
            RejectReason::DecodeError => "the image could not be decoded",
            RejectReason::BadRequest => "no usable multipart image upload",
            RejectReason::PeerBodyReads => {
                "one client had too many uploads still being read, a stalled or very slow sender"
            }
            RejectReason::BodyTimeout => "the upload was not received in time",
            RejectReason::UploadBudget => "the uploads in flight held the most memory allowed",
            RejectReason::UnknownModel => "the request named no enabled model",
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
    /// Throttle of the "dropping request that waited longer than the request timeout" warning
    /// on the worker thread (a slowed device drops at the camera rate; `dropped_total{reason=
    /// "timeout"}` keeps the exact count).
    pub timeout_log: LogThrottle,
    /// Throttle of the "worker dropped the request" / "timed out waiting for the worker"
    /// warnings of the HTTP side (`server::detect`).
    pub worker_gone_log: LogThrottle,
    /// Throttle of the bad or oversized image warning (one camera sending broken frames would
    /// otherwise log one line per frame; `decode_error` rejections keep the exact count).
    pub decode_error_log: LogThrottle,
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
            timeout_log: LogThrottle::default(),
            worker_gone_log: LogThrottle::default(),
            decode_error_log: LogThrottle::default(),
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

    /// The device asked for, when a replacement takes over: it set the rest as it loaded.
    pub fn set_requested_device(&self, requested_device: impl Into<String>) {
        write_string(&self.requested_device, requested_device.into());
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
    /// Rejected detection requests per [`RejectReason`] (`reason as usize`; read with
    /// [`Metrics::rejected`], counted with [`Metrics::record_rejected`]). The
    /// [`RejectReason::DecodeError`] slot stays 0: decode errors are counted per model.
    rejected: [AtomicU64; RejectReason::ALL.len()],
    /// Fixed server-wide in-flight cap for detection requests; 0 (the default) derives it from
    /// the models' queue capacities on every request (see `server::detect::max_inflight`).
    pub inflight_limit: AtomicUsize,
    /// Rate limit of the "server is overloaded" warning.
    pub overload_log: LogThrottle,
    /// Rate limit of the "one client holds too many uploads" warning.
    pub peer_reads_log: LogThrottle,
    /// Rate limit of the "request for unknown model" warning: one for every name, since the
    /// names are the client's text (a map keyed by them would grow without bound).
    pub unknown_model_log: LogThrottle,
    /// Detection requests being handled now, process-wide (every generation counts: an
    /// in-flight request of an old generation still holds its upload in memory).
    inflight: AtomicUsize,
    /// Uploads still being read, per peer. An entry exists only while its count is above zero
    /// and a [`BodyReadPermit`] exists only under an [`InflightPermit`], so the map never holds
    /// more peers than the in-flight cap allows requests.
    body_reads: Mutex<HashMap<IpAddr, usize>>,
}

/// One upload being read from a peer; gives the peer's slot back when dropped, which removes the
/// peer's entry when it was the last one. Held only for the body read, unlike
/// [`InflightPermit`], so a client's total requests are never capped by it: frames waiting for a
/// model hold none.
#[derive(Debug)]
pub struct BodyReadPermit {
    metrics: Arc<Metrics>,
    peer: IpAddr,
}

impl Drop for BodyReadPermit {
    fn drop(&mut self) {
        let mut reads = self.metrics.body_reads.lock_recover();
        if let Some(n) = reads.get_mut(&self.peer) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                reads.remove(&self.peer);
            }
        }
    }
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
pub const LOG_THROTTLE_INTERVAL: Duration = crate::logthrottle::DEFAULT_EVERY;

/// Lets a repeated warning through at most once per [`LOG_THROTTLE_INTERVAL`] (its `Default`)
/// and counts the ones it held back, so a flood (e.g. every rejected request under overload)
/// logs one line per interval instead of one per event. The crate's one throttle,
/// [`crate::logthrottle::Throttle`].
pub type LogThrottle = crate::logthrottle::Throttle;

impl Metrics {
    pub fn new(version: impl Into<String>) -> Self {
        Self {
            started: Instant::now(),
            version: version.into(),
            models: RwLock::new(Vec::new()),
            restarts: AtomicU64::new(0),
            rejected: std::array::from_fn(|_| AtomicU64::new(0)),
            inflight_limit: AtomicUsize::new(0),
            overload_log: LogThrottle::default(),
            peer_reads_log: LogThrottle::default(),
            unknown_model_log: LogThrottle::default(),
            inflight: AtomicUsize::new(0),
            body_reads: Mutex::new(HashMap::new()),
        }
    }

    /// Admit one more detection request when fewer than `limit` are in flight; `None` when the
    /// server is at its cap (the caller answers 503 and counts it as [`RejectReason::Overloaded`]).
    pub fn try_begin_detection(self: &Arc<Self>, limit: usize) -> Option<InflightPermit> {
        self.inflight
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < limit).then_some(n + 1)
            })
            .ok()
            .map(|_| InflightPermit(self.clone()))
    }

    /// Start reading one more upload from `peer` when it has fewer than `limit` being read;
    /// `None` when it has them all (the caller answers 503 and counts it as
    /// [`RejectReason::PeerBodyReads`]). An IPv4 client on a dual-stack socket is one peer with its
    /// plain IPv4 address. The address is the whole key on purpose: a shared IPv6 /64 would tie
    /// a Blue Iris machine to whatever else sits on its LAN.
    pub fn try_begin_body_read(
        self: &Arc<Self>,
        peer: IpAddr,
        limit: usize,
    ) -> Option<BodyReadPermit> {
        let peer = peer.to_canonical();
        let mut reads = self.body_reads.lock_recover();
        let held = reads.get(&peer).copied().unwrap_or(0);
        if held >= limit {
            return None;
        }
        reads.insert(peer, held.saturating_add(1));
        drop(reads);
        Some(BodyReadPermit {
            metrics: self.clone(),
            peer,
        })
    }

    /// Uploads being read now, over every peer.
    pub fn body_reads(&self) -> usize {
        self.body_reads.lock_recover().values().sum()
    }

    /// Peers with an upload being read now (the size of the per-peer map).
    pub fn body_read_peers(&self) -> usize {
        self.body_reads.lock_recover().len()
    }

    /// Count one rejected detection request ([`RejectReason::DecodeError`] is counted per
    /// model instead, see [`ModelMetrics::record_decode_error`]; recording it here is a no-op).
    pub fn record_rejected(&self, reason: RejectReason) {
        if reason == RejectReason::DecodeError {
            return;
        }
        if let Some(c) = self.rejected.get(reason as usize) {
            c.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Rejected detection requests for `reason` (decode errors summed over every model).
    pub fn rejected(&self, reason: RejectReason) -> u64 {
        if reason == RejectReason::DecodeError {
            return self
                .models
                .read_recover()
                .iter()
                .map(|m| m.decode_errors())
                .sum();
        }
        self.rejected
            .get(reason as usize)
            .map_or(0, |c| c.load(Ordering::Relaxed))
    }

    /// Detection requests received so far over every model (`requests_total` summed): what a
    /// benchmark records as the live load it ran beside.
    pub fn detection_requests(&self) -> u64 {
        self.models
            .read_recover()
            .iter()
            .map(|m| m.requests.load(Ordering::Relaxed))
            .sum()
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
/// macOS and Windows ask the OS for the memory ([`crate::backend::memory::process_memory`]) and
/// have no open-descriptor count.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProcessStats {
    pub resident_memory_bytes: Option<u64>,
    /// macOS only: [`crate::backend::memory::ProcessMemory::footprint`], for a soak test on a Mac:
    /// the resident size misses GPU-backed memory there.
    pub memory_footprint_bytes: Option<u64>,
    /// Memory the process owns alone (audit 2026-11, memgap-2): private bytes on Windows (the
    /// commit, which a working-set trim does not lower), the footprint on macOS, `RssAnon` on
    /// Linux ([`crate::backend::memory::ProcessMemory::private`]).
    pub private_memory_bytes: Option<u64>,
    pub open_fds: Option<u64>,
}

impl ProcessStats {
    #[cfg(target_os = "linux")]
    pub fn read() -> Self {
        let status = std::fs::read_to_string("/proc/self/status").ok();
        let resident_memory_bytes = status.as_deref().and_then(parse_vm_rss);
        let private_memory_bytes = status.as_deref().and_then(parse_rss_anon);
        // The count includes the descriptor `read_dir` itself holds open.
        let open_fds = std::fs::read_dir("/proc/self/fd")
            .ok()
            .map(|d| d.count() as u64)
            .map(|n| n.saturating_sub(1));
        Self {
            resident_memory_bytes,
            memory_footprint_bytes: None,
            private_memory_bytes,
            open_fds,
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn read() -> Self {
        let memory = crate::backend::memory::process_memory();
        Self {
            resident_memory_bytes: memory.map(|m| m.resident),
            memory_footprint_bytes: memory.and_then(|m| m.footprint),
            private_memory_bytes: memory.and_then(|m| m.private),
            open_fds: None,
        }
    }
}

/// `VmRSS` of a `/proc/<pid>/status` text, in bytes (the kernel prints it in kB).
#[cfg_attr(
    not(target_os = "linux"),
    allow(dead_code, reason = "used on Linux and in tests")
)]
fn parse_vm_rss(status: &str) -> Option<u64> {
    parse_status_kb(status, "VmRSS:")
}

/// `RssAnon` of a `/proc/<pid>/status` text, in bytes: resident anonymous memory (heap, arenas,
/// runtime pools), what the process owns alone, without its mapped files and shared memory.
#[cfg_attr(
    not(target_os = "linux"),
    allow(dead_code, reason = "used on Linux and in tests")
)]
fn parse_rss_anon(status: &str) -> Option<u64> {
    parse_status_kb(status, "RssAnon:")
}

/// The `<key>   <n> kB` line of a `/proc/<pid>/status` text, in bytes.
#[cfg_attr(
    not(target_os = "linux"),
    allow(dead_code, reason = "used on Linux and in tests")
)]
fn parse_status_kb(status: &str, key: &str) -> Option<u64> {
    let line = status.lines().find(|l| l.starts_with(key))?;
    let mut it = line.trim_start_matches(key).split_whitespace();
    let kb: u64 = it.next()?.parse().ok()?;
    match it.next() {
        Some("kB") | None => kb.checked_mul(1024),
        Some(_) => None,
    }
}

/// `gpu_memory_total_bytes` and `gpu_memory_free_bytes` per adapter with a live reading
/// ([`crate::backend::vram`]): `adapter` (the platform index), `name`, `kind` (dedicated, or shared
/// for an integrated GPU's shared-memory budget) and `source` (nvml, dxgi, sysfs). Nothing at all
/// when no adapter can be read (macOS, a machine without a GPU).
fn render_gpu_memory(out: &mut String, readings: &[crate::backend::vram::VramReading]) {
    if readings.is_empty() {
        return;
    }
    type Get = fn(&crate::backend::vram::VramReading) -> u64;
    const MIB: u64 = 1024 * 1024;
    let series: [(&str, &str, Get); 2] = [
        (
            "gpu_memory_total_bytes",
            "GPU memory of the adapter in bytes (kind=shared: the shared-memory budget of an integrated GPU, not VRAM).",
            |r| r.total_mb,
        ),
        (
            "gpu_memory_free_bytes",
            "Free GPU memory of the adapter in bytes, read live (NVML: the whole card; DXGI: this process's budget minus its usage).",
            |r| r.free_mb,
        ),
    ];
    for (name, help, get) in series {
        family(out, name, Kind::Gauge, help);
        for r in readings {
            let kind = match r.kind {
                crate::backend::vram::MemoryKind::Dedicated => "dedicated",
                crate::backend::vram::MemoryKind::Shared => "shared",
            };
            let _ = writeln!(
                out,
                "{PREFIX}{name}{{adapter=\"{}\",name=\"{}\",kind=\"{kind}\",source=\"{}\"}} {}",
                r.index,
                escape_label(&r.name),
                r.source.as_str(),
                get(r).saturating_mul(MIB)
            );
        }
    }
}

fn escape_label(v: &str) -> String {
    v.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

const PREFIX: &str = "blue_onyx_prism_";

/// Exposition type of a metric family (`# TYPE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Counter,
    Gauge,
    Histogram,
}

impl Kind {
    fn as_str(self) -> &'static str {
        match self {
            Kind::Counter => "counter",
            Kind::Gauge => "gauge",
            Kind::Histogram => "histogram",
        }
    }
}

/// The `# HELP` and `# TYPE` lines that open family `name` (without [`PREFIX`]).
fn family(out: &mut String, name: &str, kind: Kind, help: &str) {
    let _ = writeln!(out, "# HELP {PREFIX}{name} {help}");
    let _ = writeln!(out, "# TYPE {PREFIX}{name} {}", kind.as_str());
}

/// One unlabelled sample of family `name`.
fn sample(out: &mut String, name: &str, value: impl std::fmt::Display) {
    let _ = writeln!(out, "{PREFIX}{name} {value}");
}

/// One sample of family `name` per model: `name{model="..."} value`.
fn per_model<T: std::fmt::Display>(
    out: &mut String,
    name: &str,
    models: &[Arc<ModelMetrics>],
    get: impl Fn(&ModelMetrics) -> T,
) {
    for mm in models {
        let _ = writeln!(
            out,
            "{PREFIX}{name}{{model=\"{}\"}} {}",
            escape_label(&mm.name),
            get(mm)
        );
    }
}

/// One sample of family `name` per model and value of `all`, model by model:
/// `name{model="...",<label>="<as_str(r)>"} get(model, r)`.
fn per_model_labeled<R: Copy>(
    out: &mut String,
    name: &str,
    label: &str,
    models: &[Arc<ModelMetrics>],
    all: &[R],
    as_str: fn(R) -> &'static str,
    get: impl Fn(&ModelMetrics, R) -> u64,
) {
    for mm in models {
        for &r in all {
            let _ = writeln!(
                out,
                "{PREFIX}{name}{{model=\"{}\",{label}=\"{}\"}} {}",
                escape_label(&mm.name),
                as_str(r),
                get(mm, r)
            );
        }
    }
}

/// The `requests_rejected_total` HELP text, built from [`RejectReason::ALL`] so it names every
/// reason the family is exported with.
fn rejected_help() -> String {
    let reasons: Vec<String> = RejectReason::ALL
        .iter()
        .map(|r| format!("{}: {}", r.as_str(), r.help()))
        .collect();
    format!(
        "Detection requests rejected before inference, by reason ({}).",
        reasons.join("; ")
    )
}

/// The build and process-wide families: build info, uptime, restarts, rejections, downloads, the
/// dataset collector, process resources, GPU memory and the in-flight gauge.
fn render_process(out: &mut String, m: &Metrics) {
    family(out, "info", Kind::Gauge, "Build information.");
    let _ = writeln!(
        out,
        "{PREFIX}info{{version=\"{}\"}} 1",
        escape_label(&m.version)
    );
    family(
        out,
        "build_info",
        Kind::Gauge,
        "Build information: version, git commit and build date; value is always 1.",
    );
    let _ = writeln!(
        out,
        "{PREFIX}build_info{{version=\"{}\",git_sha=\"{}\",build_date=\"{}\"}} 1",
        escape_label(&m.version),
        escape_label(crate::GIT_SHA),
        escape_label(crate::BUILD_DATE)
    );
    family(
        out,
        "uptime_seconds",
        Kind::Gauge,
        "Seconds since the service started.",
    );
    sample(out, "uptime_seconds", m.uptime().as_secs());

    family(
        out,
        "restarts_total",
        Kind::Counter,
        "Registry restarts since the process started.",
    );
    sample(out, "restarts_total", m.restarts.load(Ordering::Relaxed));

    family(
        out,
        "requests_rejected_total",
        Kind::Counter,
        &rejected_help(),
    );
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
        family(out, name, Kind::Counter, help);
        sample(out, name, counter.load(Ordering::Relaxed));
    }
    crate::collect::metrics::render_prometheus(out, PREFIX);
    let p = ProcessStats::read();
    let process: [(&str, &str, Option<u64>); 4] = [
        (
            "process_resident_memory_bytes",
            "Resident memory of the process in bytes.",
            p.resident_memory_bytes,
        ),
        (
            "process_private_bytes",
            "Memory the process owns alone in bytes (Windows: private bytes, the commit; macOS: the footprint; Linux: RssAnon).",
            p.private_memory_bytes,
        ),
        (
            "process_memory_footprint_bytes",
            "Physical footprint of the process in bytes (macOS: counts compressed and GPU-backed memory).",
            p.memory_footprint_bytes,
        ),
        (
            "process_open_fds",
            "Open file descriptors of the process.",
            p.open_fds,
        ),
    ];
    for (name, help, value) in process {
        if let Some(v) = value {
            family(out, name, Kind::Gauge, help);
            sample(out, name, v);
        }
    }
    render_gpu_memory(out, &crate::backend::vram::readings());
    family(
        out,
        "inflight_requests",
        Kind::Gauge,
        "Detection requests being handled now (body read, queue wait, inference).",
    );
    sample(out, "inflight_requests", m.inflight());
}

/// The per-model counters: requests, drops, suppressed detections, night-model routing and the
/// verifier's outcomes.
fn render_model_counters(out: &mut String, models: &[Arc<ModelMetrics>]) {
    family(
        out,
        "requests_total",
        Kind::Counter,
        "Detection requests received.",
    );
    per_model(out, "requests_total", models, |mm| {
        mm.requests.load(Ordering::Relaxed)
    });
    family(
        out,
        "dropped_total",
        Kind::Counter,
        "Requests not served by inference, by reason (queue_full|timeout|initializing|failed|worker_gone).",
    );
    per_model_labeled(
        out,
        "dropped_total",
        "reason",
        models,
        &DropReason::ALL,
        DropReason::as_str,
        ModelMetrics::dropped,
    );

    family(
        out,
        "detections_suppressed_total",
        Kind::Counter,
        "Detections removed by the night / IR rules and the verifier, by reason (night_threshold|label_threshold|aspect|min_area|overlap|static|verify).",
    );
    per_model_labeled(
        out,
        "detections_suppressed_total",
        "reason",
        models,
        &SuppressReason::ALL,
        SuppressReason::as_str,
        ModelMetrics::suppressed,
    );

    family(
        out,
        "night_model_routed_total",
        Kind::Counter,
        "Night / IR frames forwarded to the model's night_model.",
    );
    per_model(out, "night_model_routed_total", models, |mm| {
        mm.night_routed()
    });
    family(
        out,
        "night_model_fallback_total",
        Kind::Counter,
        "Night / IR frames that ran on the model itself because its night_model was unavailable, by reason (unknown|not_ready|queue_full|stopped).",
    );
    per_model_labeled(
        out,
        "night_model_fallback_total",
        "reason",
        models,
        &RouteFallback::ALL,
        RouteFallback::as_str,
        ModelMetrics::night_fallback,
    );

    family(
        out,
        "verify_total",
        Kind::Counter,
        "Candidate boxes checked by the model's verifier (verify), by outcome (kept|rejected|unverified; unverified = the verifier was unavailable, busy, late or failed, or over max_crops, and the box was kept).",
    );
    per_model_labeled(
        out,
        "verify_total",
        "outcome",
        models,
        &crate::verify::Outcome::ALL,
        crate::verify::Outcome::as_str,
        ModelMetrics::verified,
    );
}

/// The latency histograms per model, each followed by its `_avg`, `_min` and `_max` gauges.
fn render_latency(out: &mut String, models: &[Arc<ModelMetrics>]) {
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
        family(out, base, Kind::Histogram, &format!("{help} (histogram)."));
        for mm in models {
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
        family(
            out,
            &format!("{base}_avg"),
            Kind::Gauge,
            &format!("{help} (avg)."),
        );
        per_model(out, &format!("{base}_avg"), models, |mm| {
            format!("{:.3}", get(mm).avg())
        });
        family(
            out,
            &format!("{base}_min"),
            Kind::Gauge,
            &format!("{help} (min)."),
        );
        per_model(out, &format!("{base}_min"), models, |mm| get(mm).min());
        family(
            out,
            &format!("{base}_max"),
            Kind::Gauge,
            &format!("{help} (max)."),
        );
        per_model(out, &format!("{base}_max"), models, |mm| get(mm).max());
    }
}

/// The per-model gauges the registry reads at the scrape ([`ModelGauges`]).
fn render_gauges(out: &mut String, gauges: &[ModelGauges]) {
    family(
        out,
        "model_info",
        Kind::Gauge,
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
    type Value = fn(&ModelGauges) -> usize;
    let series: [(&str, &str, Value); 3] = [
        (
            "model_ready",
            "1 when the model is compiled and serving requests, else 0.",
            |g| usize::from(g.state == ModelStateLabel::Ready),
        ),
        (
            "queue_length",
            "Requests waiting in the worker queue.",
            |g| g.queue_length,
        ),
        (
            "queue_capacity",
            "Effective worker queue depth limit (sized after warm-up).",
            |g| g.queue_capacity,
        ),
    ];
    for (name, help, value) in series {
        family(out, name, Kind::Gauge, help);
        for g in gauges {
            let _ = writeln!(
                out,
                "{PREFIX}{name}{{model=\"{}\"}} {}",
                escape_label(&g.name),
                value(g)
            );
        }
    }
}

/// Render all metrics in the Prometheus text exposition format. `gauges` adds the per-model
/// `model_info`, `model_ready`, `queue_length` and `queue_capacity` series.
pub fn render_prometheus(m: &Metrics, gauges: &[ModelGauges]) -> String {
    let models: Vec<Arc<ModelMetrics>> = m.models.read_recover().clone();
    let mut out = String::new();
    render_process(&mut out, m);
    render_model_counters(&mut out, &models);
    render_latency(&mut out, &models);
    render_gauges(&mut out, gauges);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One peer's uploads being read: capped per address, not across peers, and the map holds
    /// nothing once the last permit is gone.
    #[test]
    fn body_read_permits_are_per_peer_and_leave_no_entry() {
        let m = Arc::new(Metrics::new("x"));
        let a: IpAddr = "10.0.0.7".parse().unwrap();
        let b: IpAddr = "10.0.0.8".parse().unwrap();
        let held: Vec<_> = (0..2)
            .map(|_| m.try_begin_body_read(a, 2).unwrap())
            .collect();
        assert!(m.try_begin_body_read(a, 2).is_none(), "a is at its limit");
        let other = m.try_begin_body_read(b, 2).unwrap();
        assert_eq!((m.body_reads(), m.body_read_peers()), (3, 2));
        drop(other);
        assert_eq!((m.body_reads(), m.body_read_peers()), (2, 1));
        drop(held);
        assert_eq!((m.body_reads(), m.body_read_peers()), (0, 0));
        // A refused peer gets no entry.
        assert!(m.try_begin_body_read(a, 0).is_none());
        assert_eq!(m.body_read_peers(), 0);
    }

    /// An IPv4 client seen through a dual-stack socket is the same peer.
    #[test]
    fn body_reads_key_an_ipv4_mapped_peer_as_ipv4() {
        let m = Arc::new(Metrics::new("x"));
        let plain: IpAddr = "192.168.1.5".parse().unwrap();
        let mapped: IpAddr = "::ffff:192.168.1.5".parse().unwrap();
        let held = m.try_begin_body_read(plain, 1).unwrap();
        assert!(m.try_begin_body_read(mapped, 1).is_none());
        drop(held);
        let held = m.try_begin_body_read(mapped, 1).unwrap();
        assert!(m.try_begin_body_read(plain, 1).is_none());
        assert_eq!(m.body_read_peers(), 1);
        drop(held);
        assert_eq!(m.body_read_peers(), 0);
    }

    /// The per-peer map is one mutex on the detection path: many threads hammering it end
    /// with nothing left behind.
    #[test]
    fn body_read_permits_survive_concurrent_use() {
        let m = Arc::new(Metrics::new("x"));
        std::thread::scope(|s| {
            for t in 0..8u8 {
                let m = m.clone();
                s.spawn(move || {
                    let peer = IpAddr::from([10, 0, 0, t % 3]);
                    for _ in 0..2000 {
                        drop(m.try_begin_body_read(peer, 4));
                    }
                });
            }
        });
        assert_eq!((m.body_reads(), m.body_read_peers()), (0, 0));
    }

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
        for _ in 0..3 {
            m.record_rejected(RejectReason::Overloaded);
        }
        let _p = m.try_begin_detection(5).unwrap();
        let text = render_prometheus(&m, &[]);
        assert!(text.contains("blue_onyx_prism_requests_rejected_total{reason=\"overloaded\"} 3"));
        assert!(text.contains("# TYPE blue_onyx_prism_requests_rejected_total counter"));
        assert!(text.contains("blue_onyx_prism_inflight_requests 1"));
    }

    /// The rejection counter's HELP text explains every reason it is exported with (extra [9]:
    /// it once missed `upload_budget`).
    #[test]
    fn the_rejected_help_names_every_reason() {
        let text = render_prometheus(&Arc::new(Metrics::new("0")), &[]);
        let help = text
            .lines()
            .find(|l| l.starts_with("# HELP blue_onyx_prism_requests_rejected_total "))
            .unwrap();
        for r in RejectReason::ALL {
            assert!(help.contains(&format!("{}: ", r.as_str())), "{help}");
        }
    }

    #[test]
    fn prometheus_reject_reasons_build_info_and_downloads() {
        let m = Arc::new(Metrics::new("9.8.7"));
        m.record_rejected(RejectReason::BodyTooLarge);
        m.record_rejected(RejectReason::BadRequest);
        m.record_rejected(RejectReason::BadRequest);
        m.record_rejected(RejectReason::DecodeError); // per model: no-op here
        m.record_rejected(RejectReason::PeerBodyReads);
        m.record_rejected(RejectReason::BodyTimeout);
        m.record_rejected(RejectReason::BodyTimeout);
        m.record_rejected(RejectReason::UnknownModel);
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
        has("blue_onyx_prism_requests_rejected_total{reason=\"peer_body_reads\"} 1");
        has("blue_onyx_prism_requests_rejected_total{reason=\"body_timeout\"} 2");
        has("blue_onyx_prism_requests_rejected_total{reason=\"bad_request\"} 2");
        has("blue_onyx_prism_requests_rejected_total{reason=\"unknown_model\"} 1");
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
        assert!(text.contains("blue_onyx_prism_process_resident_memory_bytes "));
        assert!(text.contains("# TYPE blue_onyx_prism_process_private_bytes gauge"));
        assert!(text.contains("blue_onyx_prism_process_private_bytes "));
        assert_eq!(
            text.contains("blue_onyx_prism_process_memory_footprint_bytes "),
            cfg!(target_os = "macos")
        );
        assert_eq!(
            text.contains("blue_onyx_prism_process_open_fds "),
            cfg!(target_os = "linux")
        );
        for l in text.lines().filter(|l| !l.starts_with('#')) {
            let v = l.rsplit(' ').next().unwrap();
            assert!(v.parse::<f64>().is_ok(), "bad sample line {l:?}");
        }
    }

    /// A [`Metrics`] with every series it renders holding a distinct, non-zero value where it
    /// can (two models, one with a label to escape), and gauges in every state.
    fn populated() -> (Arc<Metrics>, Vec<ModelGauges>, InflightPermit) {
        let m = Arc::new(Metrics::new("1.2.\"3\""));
        m.restarts.store(2, Ordering::Relaxed);
        for (i, r) in RejectReason::ALL.into_iter().enumerate() {
            for _ in 0..=i {
                m.record_rejected(r);
            }
        }
        let permit = m.try_begin_detection(4).unwrap();
        for (ku, name) in ["a", "b\"q"].into_iter().enumerate() {
            let k = u32::try_from(ku).unwrap();
            let mm = m.model(name);
            mm.requests.store(10 + u64::from(k), Ordering::Relaxed);
            for (i, r) in DropReason::ALL.into_iter().enumerate() {
                for _ in 0..=(i + ku) {
                    mm.record_drop(r);
                }
            }
            for _ in 0..=k {
                mm.record_decode_error();
            }
            let mut c = SuppressCounts::default();
            for (i, r) in SuppressReason::ALL.into_iter().enumerate() {
                c.add(r, u32::try_from(i).unwrap() + k + 1);
            }
            mm.record_suppressed(&c);
            for _ in 0..=k {
                mm.record_night_routed();
            }
            for (i, r) in RouteFallback::ALL.into_iter().enumerate() {
                for _ in 0..=(i + ku) {
                    mm.record_night_fallback(r);
                }
            }
            mm.record_verify(&crate::verify::Counts {
                kept: 1 + k,
                rejected: 2 + k,
                unverified: 3 + k,
            });
            for ms in [0, 3, 15, 700, 9000] {
                mm.inference_ms.record(ms + u64::from(k));
                mm.process_ms.record(ms * 2 + 1);
                mm.round_trip_ms.record(ms * 3 + 2);
            }
        }
        let g = |name: &str, state, queue_length, queue_capacity| ModelGauges {
            name: name.into(),
            device: "GPU.\"1\"".into(),
            provider: "OpenVINO GPU (x)".into(),
            state,
            queue_length,
            queue_capacity,
        };
        let gauges = vec![
            g("a", ModelStateLabel::Ready, 2, 30),
            g("b\"q", ModelStateLabel::Initializing, 0, 64),
            g("c", ModelStateLabel::Failed, 0, 0),
            g("d", ModelStateLabel::Lazy, 1, 8),
        ];
        (m, gauges, permit)
    }

    /// The family a line of the exposition belongs to (`# HELP`/`# TYPE` name, else the sample
    /// name before its labels).
    fn family_of(line: &str) -> &str {
        if let Some(rest) = line
            .strip_prefix("# HELP ")
            .or_else(|| line.strip_prefix("# TYPE "))
        {
            return rest.split(' ').next().unwrap();
        }
        line.split(['{', ' ']).next().unwrap()
    }

    /// The exposition without what depends on the host or on other tests: the process and GPU
    /// families are dropped (they vary by platform and machine), and the values of the uptime,
    /// download and dataset samples (process-wide counters) are masked.
    fn stable(text: &str) -> String {
        let mut out = String::new();
        for line in text.lines() {
            let fam = family_of(line);
            if fam.starts_with("blue_onyx_prism_process_")
                || fam.starts_with("blue_onyx_prism_gpu_memory_")
            {
                continue;
            }
            let masked = !line.starts_with('#')
                && (fam == "blue_onyx_prism_uptime_seconds"
                    || fam.starts_with("blue_onyx_prism_download_")
                    || fam.starts_with("blue_onyx_prism_dataset_"));
            if masked {
                let (sample, _) = line.rsplit_once(' ').unwrap();
                out.push_str(sample);
                out.push_str(" <v>\n");
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        out
    }

    /// Exposition-format structure: every family has one `# HELP` then one `# TYPE` before its
    /// samples, and a family's lines are contiguous (a histogram's `_bucket`/`_sum`/`_count`
    /// belong to it).
    fn assert_well_formed(text: &str) {
        let mut seen = std::collections::HashSet::new();
        let mut current: Option<(String, String)> = None; // (family, type)
        let mut lines = text.lines().peekable();
        while let Some(line) = lines.next() {
            if let Some(rest) = line.strip_prefix("# HELP ") {
                let name = rest.split(' ').next().unwrap().to_string();
                assert!(seen.insert(name.clone()), "family {name} twice");
                let ty = lines.next().unwrap();
                let kind = ty
                    .strip_prefix(&format!("# TYPE {name} "))
                    .unwrap_or_else(|| panic!("HELP of {name} not followed by its TYPE: {ty}"));
                assert!(["counter", "gauge", "histogram"].contains(&kind), "{kind}");
                current = Some((name, kind.to_string()));
                continue;
            }
            assert!(!line.starts_with('#'), "TYPE without HELP: {line}");
            let (fam, kind) = current.as_ref().expect("sample before any HELP");
            let name = family_of(line);
            let ok = name == fam
                || (kind == "histogram"
                    && ["_bucket", "_sum", "_count"]
                        .iter()
                        .any(|s| name.strip_suffix(s) == Some(fam.as_str())));
            assert!(ok, "sample {line:?} outside its family {fam}");
        }
    }

    /// Golden: the whole exposition of a fully populated [`Metrics`] plus gauges (audit 2026-11,
    /// QA-217: pinned before the renderer was split into helpers).
    #[test]
    fn prometheus_golden_text() {
        let (m, gauges, _permit) = populated();
        let text = render_prometheus(&m, &gauges);
        assert_well_formed(&text);
        let got = stable(&text);
        let want = GOLDEN
            .replace("@SHA@", crate::GIT_SHA)
            .replace("@DATE@", crate::BUILD_DATE);
        assert_eq!(got, want);
    }

    /// [`prometheus_golden_text`]'s expected text (`@SHA@` / `@DATE@`: the build's
    /// [`crate::GIT_SHA`] / [`crate::BUILD_DATE`]).
    const GOLDEN: &str = r#"# HELP blue_onyx_prism_info Build information.
# TYPE blue_onyx_prism_info gauge
blue_onyx_prism_info{version="1.2.\"3\""} 1
# HELP blue_onyx_prism_build_info Build information: version, git commit and build date; value is always 1.
# TYPE blue_onyx_prism_build_info gauge
blue_onyx_prism_build_info{version="1.2.\"3\"",git_sha="@SHA@",build_date="@DATE@"} 1
# HELP blue_onyx_prism_uptime_seconds Seconds since the service started.
# TYPE blue_onyx_prism_uptime_seconds gauge
blue_onyx_prism_uptime_seconds <v>
# HELP blue_onyx_prism_restarts_total Registry restarts since the process started.
# TYPE blue_onyx_prism_restarts_total counter
blue_onyx_prism_restarts_total 2
# HELP blue_onyx_prism_requests_rejected_total Detection requests rejected before inference, by reason (overloaded: the server-wide in-flight cap was reached; body_too_large: the upload exceeded the body limit; decode_error: the image could not be decoded; bad_request: no usable multipart image upload; peer_body_reads: one client had too many uploads still being read, a stalled or very slow sender; body_timeout: the upload was not received in time; upload_budget: the uploads in flight held the most memory allowed; unknown_model: the request named no enabled model).
# TYPE blue_onyx_prism_requests_rejected_total counter
blue_onyx_prism_requests_rejected_total{reason="overloaded"} 1
blue_onyx_prism_requests_rejected_total{reason="body_too_large"} 2
blue_onyx_prism_requests_rejected_total{reason="decode_error"} 3
blue_onyx_prism_requests_rejected_total{reason="bad_request"} 4
blue_onyx_prism_requests_rejected_total{reason="peer_body_reads"} 5
blue_onyx_prism_requests_rejected_total{reason="body_timeout"} 6
blue_onyx_prism_requests_rejected_total{reason="upload_budget"} 7
blue_onyx_prism_requests_rejected_total{reason="unknown_model"} 8
# HELP blue_onyx_prism_download_bytes_total Bytes received by on-demand resource downloads.
# TYPE blue_onyx_prism_download_bytes_total counter
blue_onyx_prism_download_bytes_total <v>
# HELP blue_onyx_prism_download_failures_total On-demand resource download jobs that failed.
# TYPE blue_onyx_prism_download_failures_total counter
blue_onyx_prism_download_failures_total <v>
# HELP blue_onyx_prism_dataset_requests_total Detection requests received since the service started (test page uploads excluded).
# TYPE blue_onyx_prism_dataset_requests_total counter
blue_onyx_prism_dataset_requests_total <v>
# HELP blue_onyx_prism_dataset_offered_total Answered requests handed to the dataset collector.
# TYPE blue_onyx_prism_dataset_offered_total counter
blue_onyx_prism_dataset_offered_total <v>
# HELP blue_onyx_prism_dataset_saved_total Frames the dataset collector saved.
# TYPE blue_onyx_prism_dataset_saved_total counter
blue_onyx_prism_dataset_saved_total <v>
# HELP blue_onyx_prism_dataset_no_bucket_total Frames the dataset collector skipped because they matched no enabled bucket.
# TYPE blue_onyx_prism_dataset_no_bucket_total counter
blue_onyx_prism_dataset_no_bucket_total <v>
# HELP blue_onyx_prism_dataset_frames_total Frames saved by the dataset collector, by bucket (a frame counts once in each bucket it was kept for).
# TYPE blue_onyx_prism_dataset_frames_total counter
blue_onyx_prism_dataset_frames_total{bucket="positive"} <v>
blue_onyx_prism_dataset_frames_total{bucket="empty"} <v>
blue_onyx_prism_dataset_frames_total{bucket="suppressed"} <v>
blue_onyx_prism_dataset_frames_total{bucket="verify_rejected"} <v>
blue_onyx_prism_dataset_frames_total{bucket="verify_kept"} <v>
blue_onyx_prism_dataset_frames_total{bucket="near_threshold"} <v>
blue_onyx_prism_dataset_frames_total{bucket="disagreement"} <v>
# HELP blue_onyx_prism_dataset_dropped_total Frames the dataset collector did not save, by reason (queue_full|rate|dup|disk|error).
# TYPE blue_onyx_prism_dataset_dropped_total counter
blue_onyx_prism_dataset_dropped_total{reason="queue_full"} <v>
blue_onyx_prism_dataset_dropped_total{reason="rate"} <v>
blue_onyx_prism_dataset_dropped_total{reason="dup"} <v>
blue_onyx_prism_dataset_dropped_total{reason="disk"} <v>
blue_onyx_prism_dataset_dropped_total{reason="error"} <v>
# HELP blue_onyx_prism_dataset_bytes Bytes of the frames and sidecars in the dataset folder.
# TYPE blue_onyx_prism_dataset_bytes gauge
blue_onyx_prism_dataset_bytes <v>
# HELP blue_onyx_prism_dataset_collect_disabled 1 while the collector is disabled for low disk space or a full disk.
# TYPE blue_onyx_prism_dataset_collect_disabled gauge
blue_onyx_prism_dataset_collect_disabled <v>
# HELP blue_onyx_prism_inflight_requests Detection requests being handled now (body read, queue wait, inference).
# TYPE blue_onyx_prism_inflight_requests gauge
blue_onyx_prism_inflight_requests 1
# HELP blue_onyx_prism_requests_total Detection requests received.
# TYPE blue_onyx_prism_requests_total counter
blue_onyx_prism_requests_total{model="a"} 10
blue_onyx_prism_requests_total{model="b\"q"} 11
# HELP blue_onyx_prism_dropped_total Requests not served by inference, by reason (queue_full|timeout|initializing|failed|worker_gone).
# TYPE blue_onyx_prism_dropped_total counter
blue_onyx_prism_dropped_total{model="a",reason="queue_full"} 1
blue_onyx_prism_dropped_total{model="a",reason="timeout"} 2
blue_onyx_prism_dropped_total{model="a",reason="initializing"} 3
blue_onyx_prism_dropped_total{model="a",reason="failed"} 4
blue_onyx_prism_dropped_total{model="a",reason="worker_gone"} 5
blue_onyx_prism_dropped_total{model="b\"q",reason="queue_full"} 2
blue_onyx_prism_dropped_total{model="b\"q",reason="timeout"} 3
blue_onyx_prism_dropped_total{model="b\"q",reason="initializing"} 4
blue_onyx_prism_dropped_total{model="b\"q",reason="failed"} 5
blue_onyx_prism_dropped_total{model="b\"q",reason="worker_gone"} 6
# HELP blue_onyx_prism_detections_suppressed_total Detections removed by the night / IR rules and the verifier, by reason (night_threshold|label_threshold|aspect|min_area|overlap|static|verify).
# TYPE blue_onyx_prism_detections_suppressed_total counter
blue_onyx_prism_detections_suppressed_total{model="a",reason="night_threshold"} 1
blue_onyx_prism_detections_suppressed_total{model="a",reason="label_threshold"} 2
blue_onyx_prism_detections_suppressed_total{model="a",reason="aspect"} 3
blue_onyx_prism_detections_suppressed_total{model="a",reason="min_area"} 4
blue_onyx_prism_detections_suppressed_total{model="a",reason="overlap"} 5
blue_onyx_prism_detections_suppressed_total{model="a",reason="static"} 6
blue_onyx_prism_detections_suppressed_total{model="a",reason="verify"} 7
blue_onyx_prism_detections_suppressed_total{model="b\"q",reason="night_threshold"} 2
blue_onyx_prism_detections_suppressed_total{model="b\"q",reason="label_threshold"} 3
blue_onyx_prism_detections_suppressed_total{model="b\"q",reason="aspect"} 4
blue_onyx_prism_detections_suppressed_total{model="b\"q",reason="min_area"} 5
blue_onyx_prism_detections_suppressed_total{model="b\"q",reason="overlap"} 6
blue_onyx_prism_detections_suppressed_total{model="b\"q",reason="static"} 7
blue_onyx_prism_detections_suppressed_total{model="b\"q",reason="verify"} 8
# HELP blue_onyx_prism_night_model_routed_total Night / IR frames forwarded to the model's night_model.
# TYPE blue_onyx_prism_night_model_routed_total counter
blue_onyx_prism_night_model_routed_total{model="a"} 1
blue_onyx_prism_night_model_routed_total{model="b\"q"} 2
# HELP blue_onyx_prism_night_model_fallback_total Night / IR frames that ran on the model itself because its night_model was unavailable, by reason (unknown|not_ready|queue_full|stopped).
# TYPE blue_onyx_prism_night_model_fallback_total counter
blue_onyx_prism_night_model_fallback_total{model="a",reason="unknown"} 1
blue_onyx_prism_night_model_fallback_total{model="a",reason="not_ready"} 2
blue_onyx_prism_night_model_fallback_total{model="a",reason="queue_full"} 3
blue_onyx_prism_night_model_fallback_total{model="a",reason="stopped"} 4
blue_onyx_prism_night_model_fallback_total{model="b\"q",reason="unknown"} 2
blue_onyx_prism_night_model_fallback_total{model="b\"q",reason="not_ready"} 3
blue_onyx_prism_night_model_fallback_total{model="b\"q",reason="queue_full"} 4
blue_onyx_prism_night_model_fallback_total{model="b\"q",reason="stopped"} 5
# HELP blue_onyx_prism_verify_total Candidate boxes checked by the model's verifier (verify), by outcome (kept|rejected|unverified; unverified = the verifier was unavailable, busy, late or failed, or over max_crops, and the box was kept).
# TYPE blue_onyx_prism_verify_total counter
blue_onyx_prism_verify_total{model="a",outcome="kept"} 1
blue_onyx_prism_verify_total{model="a",outcome="rejected"} 2
blue_onyx_prism_verify_total{model="a",outcome="unverified"} 3
blue_onyx_prism_verify_total{model="b\"q",outcome="kept"} 2
blue_onyx_prism_verify_total{model="b\"q",outcome="rejected"} 3
blue_onyx_prism_verify_total{model="b\"q",outcome="unverified"} 4
# HELP blue_onyx_prism_inference_ms Model inference time in ms (histogram).
# TYPE blue_onyx_prism_inference_ms histogram
blue_onyx_prism_inference_ms_bucket{model="a",le="1"} 1
blue_onyx_prism_inference_ms_bucket{model="a",le="2"} 1
blue_onyx_prism_inference_ms_bucket{model="a",le="5"} 2
blue_onyx_prism_inference_ms_bucket{model="a",le="10"} 2
blue_onyx_prism_inference_ms_bucket{model="a",le="20"} 3
blue_onyx_prism_inference_ms_bucket{model="a",le="50"} 3
blue_onyx_prism_inference_ms_bucket{model="a",le="100"} 3
blue_onyx_prism_inference_ms_bucket{model="a",le="200"} 3
blue_onyx_prism_inference_ms_bucket{model="a",le="500"} 3
blue_onyx_prism_inference_ms_bucket{model="a",le="1000"} 4
blue_onyx_prism_inference_ms_bucket{model="a",le="2000"} 4
blue_onyx_prism_inference_ms_bucket{model="a",le="5000"} 4
blue_onyx_prism_inference_ms_bucket{model="a",le="+Inf"} 5
blue_onyx_prism_inference_ms_sum{model="a"} 9718
blue_onyx_prism_inference_ms_count{model="a"} 5
blue_onyx_prism_inference_ms_bucket{model="b\"q",le="1"} 1
blue_onyx_prism_inference_ms_bucket{model="b\"q",le="2"} 1
blue_onyx_prism_inference_ms_bucket{model="b\"q",le="5"} 2
blue_onyx_prism_inference_ms_bucket{model="b\"q",le="10"} 2
blue_onyx_prism_inference_ms_bucket{model="b\"q",le="20"} 3
blue_onyx_prism_inference_ms_bucket{model="b\"q",le="50"} 3
blue_onyx_prism_inference_ms_bucket{model="b\"q",le="100"} 3
blue_onyx_prism_inference_ms_bucket{model="b\"q",le="200"} 3
blue_onyx_prism_inference_ms_bucket{model="b\"q",le="500"} 3
blue_onyx_prism_inference_ms_bucket{model="b\"q",le="1000"} 4
blue_onyx_prism_inference_ms_bucket{model="b\"q",le="2000"} 4
blue_onyx_prism_inference_ms_bucket{model="b\"q",le="5000"} 4
blue_onyx_prism_inference_ms_bucket{model="b\"q",le="+Inf"} 5
blue_onyx_prism_inference_ms_sum{model="b\"q"} 9723
blue_onyx_prism_inference_ms_count{model="b\"q"} 5
# HELP blue_onyx_prism_inference_ms_avg Model inference time in ms (avg).
# TYPE blue_onyx_prism_inference_ms_avg gauge
blue_onyx_prism_inference_ms_avg{model="a"} 1943.600
blue_onyx_prism_inference_ms_avg{model="b\"q"} 1944.600
# HELP blue_onyx_prism_inference_ms_min Model inference time in ms (min).
# TYPE blue_onyx_prism_inference_ms_min gauge
blue_onyx_prism_inference_ms_min{model="a"} 0
blue_onyx_prism_inference_ms_min{model="b\"q"} 1
# HELP blue_onyx_prism_inference_ms_max Model inference time in ms (max).
# TYPE blue_onyx_prism_inference_ms_max gauge
blue_onyx_prism_inference_ms_max{model="a"} 9000
blue_onyx_prism_inference_ms_max{model="b\"q"} 9001
# HELP blue_onyx_prism_round_trip_ms Request round trip time in ms (histogram).
# TYPE blue_onyx_prism_round_trip_ms histogram
blue_onyx_prism_round_trip_ms_bucket{model="a",le="1"} 0
blue_onyx_prism_round_trip_ms_bucket{model="a",le="2"} 1
blue_onyx_prism_round_trip_ms_bucket{model="a",le="5"} 1
blue_onyx_prism_round_trip_ms_bucket{model="a",le="10"} 1
blue_onyx_prism_round_trip_ms_bucket{model="a",le="20"} 2
blue_onyx_prism_round_trip_ms_bucket{model="a",le="50"} 3
blue_onyx_prism_round_trip_ms_bucket{model="a",le="100"} 3
blue_onyx_prism_round_trip_ms_bucket{model="a",le="200"} 3
blue_onyx_prism_round_trip_ms_bucket{model="a",le="500"} 3
blue_onyx_prism_round_trip_ms_bucket{model="a",le="1000"} 3
blue_onyx_prism_round_trip_ms_bucket{model="a",le="2000"} 3
blue_onyx_prism_round_trip_ms_bucket{model="a",le="5000"} 4
blue_onyx_prism_round_trip_ms_bucket{model="a",le="+Inf"} 5
blue_onyx_prism_round_trip_ms_sum{model="a"} 29164
blue_onyx_prism_round_trip_ms_count{model="a"} 5
blue_onyx_prism_round_trip_ms_bucket{model="b\"q",le="1"} 0
blue_onyx_prism_round_trip_ms_bucket{model="b\"q",le="2"} 1
blue_onyx_prism_round_trip_ms_bucket{model="b\"q",le="5"} 1
blue_onyx_prism_round_trip_ms_bucket{model="b\"q",le="10"} 1
blue_onyx_prism_round_trip_ms_bucket{model="b\"q",le="20"} 2
blue_onyx_prism_round_trip_ms_bucket{model="b\"q",le="50"} 3
blue_onyx_prism_round_trip_ms_bucket{model="b\"q",le="100"} 3
blue_onyx_prism_round_trip_ms_bucket{model="b\"q",le="200"} 3
blue_onyx_prism_round_trip_ms_bucket{model="b\"q",le="500"} 3
blue_onyx_prism_round_trip_ms_bucket{model="b\"q",le="1000"} 3
blue_onyx_prism_round_trip_ms_bucket{model="b\"q",le="2000"} 3
blue_onyx_prism_round_trip_ms_bucket{model="b\"q",le="5000"} 4
blue_onyx_prism_round_trip_ms_bucket{model="b\"q",le="+Inf"} 5
blue_onyx_prism_round_trip_ms_sum{model="b\"q"} 29164
blue_onyx_prism_round_trip_ms_count{model="b\"q"} 5
# HELP blue_onyx_prism_round_trip_ms_avg Request round trip time in ms (avg).
# TYPE blue_onyx_prism_round_trip_ms_avg gauge
blue_onyx_prism_round_trip_ms_avg{model="a"} 5832.800
blue_onyx_prism_round_trip_ms_avg{model="b\"q"} 5832.800
# HELP blue_onyx_prism_round_trip_ms_min Request round trip time in ms (min).
# TYPE blue_onyx_prism_round_trip_ms_min gauge
blue_onyx_prism_round_trip_ms_min{model="a"} 2
blue_onyx_prism_round_trip_ms_min{model="b\"q"} 2
# HELP blue_onyx_prism_round_trip_ms_max Request round trip time in ms (max).
# TYPE blue_onyx_prism_round_trip_ms_max gauge
blue_onyx_prism_round_trip_ms_max{model="a"} 27002
blue_onyx_prism_round_trip_ms_max{model="b\"q"} 27002
# HELP blue_onyx_prism_model_info Model device, execution provider and state (initializing|ready|failed|lazy); value is always 1.
# TYPE blue_onyx_prism_model_info gauge
blue_onyx_prism_model_info{model="a",device="GPU.\"1\"",provider="OpenVINO GPU (x)",state="ready"} 1
blue_onyx_prism_model_info{model="b\"q",device="GPU.\"1\"",provider="OpenVINO GPU (x)",state="initializing"} 1
blue_onyx_prism_model_info{model="c",device="GPU.\"1\"",provider="OpenVINO GPU (x)",state="failed"} 1
blue_onyx_prism_model_info{model="d",device="GPU.\"1\"",provider="OpenVINO GPU (x)",state="lazy"} 1
# HELP blue_onyx_prism_model_ready 1 when the model is compiled and serving requests, else 0.
# TYPE blue_onyx_prism_model_ready gauge
blue_onyx_prism_model_ready{model="a"} 1
blue_onyx_prism_model_ready{model="b\"q"} 0
blue_onyx_prism_model_ready{model="c"} 0
blue_onyx_prism_model_ready{model="d"} 0
# HELP blue_onyx_prism_queue_length Requests waiting in the worker queue.
# TYPE blue_onyx_prism_queue_length gauge
blue_onyx_prism_queue_length{model="a"} 2
blue_onyx_prism_queue_length{model="b\"q"} 0
blue_onyx_prism_queue_length{model="c"} 0
blue_onyx_prism_queue_length{model="d"} 1
# HELP blue_onyx_prism_queue_capacity Effective worker queue depth limit (sized after warm-up).
# TYPE blue_onyx_prism_queue_capacity gauge
blue_onyx_prism_queue_capacity{model="a"} 30
blue_onyx_prism_queue_capacity{model="b\"q"} 64
blue_onyx_prism_queue_capacity{model="c"} 0
blue_onyx_prism_queue_capacity{model="d"} 8
"#;

    /// The counters are indexed by `reason as usize`: [`RejectReason::ALL`] lists the reasons in
    /// declaration order (likewise the per-model label enums).
    #[test]
    fn reason_lists_follow_declaration_order() {
        for (i, r) in RejectReason::ALL.into_iter().enumerate() {
            assert_eq!(r as usize, i, "{r:?}");
        }
        for (i, r) in DropReason::ALL.into_iter().enumerate() {
            assert_eq!(r as usize, i, "{r:?}");
        }
        for (i, r) in RouteFallback::ALL.into_iter().enumerate() {
            assert_eq!(r as usize, i, "{r:?}");
        }
    }

    #[test]
    fn rejected_counts_each_reason_apart() {
        let m = Metrics::new("0");
        for (i, r) in RejectReason::ALL.into_iter().enumerate() {
            for _ in 0..=i {
                m.record_rejected(r);
            }
        }
        for (i, r) in RejectReason::ALL.into_iter().enumerate() {
            let want = if r == RejectReason::DecodeError {
                0
            } else {
                u64::try_from(i).unwrap() + 1
            };
            assert_eq!(m.rejected(r), want, "{r:?}");
        }
    }

    #[test]
    fn family_and_per_model_helpers() {
        let mut out = String::new();
        family(&mut out, "x_total", Kind::Counter, "Some help.");
        sample(&mut out, "x_total", 7);
        let models = vec![
            Arc::new(ModelMetrics::new("a", "", "")),
            Arc::new(ModelMetrics::new("b\"c", "", "")),
        ];
        models[1].record_drop(DropReason::Timeout);
        per_model(&mut out, "y", &models, |mm| mm.name.len());
        per_model_labeled(
            &mut out,
            "z",
            "reason",
            &models,
            &[DropReason::QueueFull, DropReason::Timeout],
            DropReason::as_str,
            ModelMetrics::dropped,
        );
        assert_eq!(
            out,
            r#"# HELP blue_onyx_prism_x_total Some help.
# TYPE blue_onyx_prism_x_total counter
blue_onyx_prism_x_total 7
blue_onyx_prism_y{model="a"} 1
blue_onyx_prism_y{model="b\"c"} 3
blue_onyx_prism_z{model="a",reason="queue_full"} 0
blue_onyx_prism_z{model="a",reason="timeout"} 0
blue_onyx_prism_z{model="b\"c",reason="queue_full"} 0
blue_onyx_prism_z{model="b\"c",reason="timeout"} 1
"#
        );
    }

    #[test]
    fn vm_rss_parsing() {
        let status = "Name:\tblue\nVmPeak:\t  9999 kB\nVmRSS:\t   12345 kB\nThreads:\t4\n";
        assert_eq!(parse_vm_rss(status), Some(12345 * 1024));
        assert_eq!(parse_vm_rss("Name: x\n"), None);
        assert_eq!(parse_vm_rss("VmRSS: lots kB\n"), None);
        assert_eq!(parse_vm_rss("VmRSS: 5 MB\n"), None);
        let status = "VmRSS:\t   12345 kB\nRssAnon:\t    4321 kB\nRssFile:\t 8000 kB\n";
        assert_eq!(parse_rss_anon(status), Some(4321 * 1024));
        assert_eq!(parse_rss_anon("VmRSS:\t1 kB\n"), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn process_stats_on_linux() {
        let p = ProcessStats::read();
        assert!(p.resident_memory_bytes.is_some_and(|b| b > 0));
        assert!(p.open_fds.is_some_and(|n| n >= 3));
        assert_eq!(p.memory_footprint_bytes, None);
        // Anonymous memory is part of the resident set.
        let private = p.private_memory_bytes.unwrap();
        assert!(private > 0, "{p:?}");
        assert!(private <= p.resident_memory_bytes.unwrap(), "{p:?}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn process_stats_on_macos() {
        let p = ProcessStats::read();
        assert!(p.resident_memory_bytes.is_some_and(|b| b > 0));
        assert!(p.memory_footprint_bytes.is_some_and(|b| b > 0));
        assert!(p.private_memory_bytes.is_some_and(|b| b > 0));
        assert_eq!(p.open_fds, None);
    }

    #[cfg(windows)]
    #[test]
    fn process_stats_on_windows() {
        let p = ProcessStats::read();
        assert!(p.resident_memory_bytes.is_some_and(|b| b > 0));
        assert!(p.private_memory_bytes.is_some_and(|b| b > 0));
        assert_eq!((p.memory_footprint_bytes, p.open_fds), (None, None));
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

    #[test]
    fn gpu_memory_gauges_per_adapter() {
        use crate::backend::vram::{MemoryKind, Source, VramReading};
        let mut out = String::new();
        render_gpu_memory(&mut out, &[]);
        assert!(out.is_empty(), "no reading, no series");
        let r = |index, name: &str, kind, total_mb, free_mb, source| VramReading {
            index,
            name: name.into(),
            vendor: crate::backend::detect::GpuVendor::Nvidia,
            kind,
            total_mb,
            free_mb,
            source,
        };
        render_gpu_memory(
            &mut out,
            &[
                r(
                    0,
                    "Intel(R) UHD Graphics 630",
                    MemoryKind::Shared,
                    8000,
                    7000,
                    Source::Dxgi,
                ),
                r(
                    1,
                    "GTX \"1070\" Ti",
                    MemoryKind::Dedicated,
                    8192,
                    2048,
                    Source::Nvml,
                ),
            ],
        );
        assert!(
            out.contains("# TYPE blue_onyx_prism_gpu_memory_total_bytes gauge"),
            "{out}"
        );
        assert!(
            out.contains(
                "blue_onyx_prism_gpu_memory_total_bytes{adapter=\"0\",name=\"Intel(R) UHD Graphics 630\",kind=\"shared\",source=\"dxgi\"} 8388608000"
            ),
            "{out}"
        );
        assert!(
            out.contains(
                "blue_onyx_prism_gpu_memory_free_bytes{adapter=\"1\",name=\"GTX \\\"1070\\\" Ti\",kind=\"dedicated\",source=\"nvml\"} 2147483648"
            ),
            "{out}"
        );
    }
}
