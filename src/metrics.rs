//! Request/latency counters and Prometheus text rendering.

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
    pub inference_ms: Histogram,
    pub process_ms: Histogram,
    pub round_trip_ms: Histogram,
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
            inference_ms: Histogram::new(),
            process_ms: Histogram::new(),
            round_trip_ms: Histogram::new(),
        }
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
        "# HELP {PREFIX}requests_rejected_total Detection requests rejected before reaching a model, by reason (overloaded: the server-wide in-flight cap was reached)."
    );
    let _ = writeln!(out, "# TYPE {PREFIX}requests_rejected_total counter");
    let _ = writeln!(
        out,
        "{PREFIX}requests_rejected_total{{reason=\"overloaded\"}} {}",
        m.rejected_overloaded.load(Ordering::Relaxed)
    );
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
