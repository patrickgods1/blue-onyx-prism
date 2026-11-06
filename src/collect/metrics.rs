//! Process-wide dataset collector counters (`dataset_requests_total`, `dataset_offered_total`,
//! `dataset_saved_total`, `dataset_no_bucket_total`, `dataset_frames_total{bucket}`,
//! `dataset_dropped_total{reason}`, `dataset_bytes`, `dataset_collect_disabled`) and their
//! Prometheus rendering, called from [`crate::metrics::render_prometheus`]. They count since this
//! process started; the Collect tab reads them through [`super::health`].

use super::Bucket;
use std::fmt::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Why a frame was not saved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dropped {
    /// The collector's queue was full (the reply was never delayed).
    QueueFull,
    /// Every bucket it matched was rate limited or not sampled.
    Rate,
    /// A near duplicate of a frame of the same camera saved shortly before.
    Dup,
    /// Collection is disabled for low disk space or a full disk.
    Disk,
    /// The frame could not be decoded or written (other I/O error).
    Error,
}

impl Dropped {
    pub const ALL: [Dropped; 5] = [
        Dropped::QueueFull,
        Dropped::Rate,
        Dropped::Dup,
        Dropped::Disk,
        Dropped::Error,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Dropped::QueueFull => "queue_full",
            Dropped::Rate => "rate",
            Dropped::Dup => "dup",
            Dropped::Disk => "disk",
            Dropped::Error => "error",
        }
    }

    /// The reason in plain words, for the Collect tab and `GET /v1/dataset`.
    pub fn plain(self) -> &'static str {
        match self {
            Dropped::QueueFull => "Images came in faster than the disk could save them",
            Dropped::Rate => "Held back by a bucket's rate or minimum interval",
            Dropped::Dup => "Looked like a frame of the same camera saved shortly before",
            Dropped::Disk => "Collection was stopped for low disk space",
            Dropped::Error => "Could not be read or written",
        }
    }
}

/// The counters.
#[derive(Debug)]
pub struct DatasetMetrics {
    /// Detection requests the HTTP layer took in, collection on or off (no Test page uploads).
    requests: AtomicU64,
    /// Finished frames handed to the collector.
    offered: AtomicU64,
    /// Frames saved (once each, unlike `frames`, which counts per bucket).
    saved: AtomicU64,
    /// Frames that matched no enabled bucket (no drop: nothing wanted them).
    no_bucket: AtomicU64,
    frames: [AtomicU64; Bucket::ALL.len()],
    dropped: [AtomicU64; Dropped::ALL.len()],
    bytes: AtomicU64,
    disabled: AtomicBool,
}

impl DatasetMetrics {
    const fn new() -> Self {
        Self {
            requests: AtomicU64::new(0),
            offered: AtomicU64::new(0),
            saved: AtomicU64::new(0),
            no_bucket: AtomicU64::new(0),
            frames: [const { AtomicU64::new(0) }; Bucket::ALL.len()],
            dropped: [const { AtomicU64::new(0) }; Dropped::ALL.len()],
            bytes: AtomicU64::new(0),
            disabled: AtomicBool::new(false),
        }
    }

    pub fn record_request(&self) {
        self.requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn requests(&self) -> u64 {
        self.requests.load(Ordering::Relaxed)
    }

    pub fn record_offered(&self) {
        self.offered.fetch_add(1, Ordering::Relaxed);
    }

    pub fn offered(&self) -> u64 {
        self.offered.load(Ordering::Relaxed)
    }

    pub fn record_no_bucket(&self) {
        self.no_bucket.fetch_add(1, Ordering::Relaxed);
    }

    pub fn no_bucket(&self) -> u64 {
        self.no_bucket.load(Ordering::Relaxed)
    }

    pub fn saved(&self) -> u64 {
        self.saved.load(Ordering::Relaxed)
    }

    /// One saved frame, kept for `buckets`.
    pub fn record_frame(&self, buckets: &[Bucket]) {
        self.saved.fetch_add(1, Ordering::Relaxed);
        for b in buckets {
            if let Some(c) = self.frames.get(*b as usize) {
                c.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn frames(&self, b: Bucket) -> u64 {
        self.frames
            .get(b as usize)
            .map_or(0, |c| c.load(Ordering::Relaxed))
    }

    pub fn record_drop(&self, r: Dropped) {
        if let Some(c) = self.dropped.get(r as usize) {
            c.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn dropped(&self, r: Dropped) -> u64 {
        self.dropped
            .get(r as usize)
            .map_or(0, |c| c.load(Ordering::Relaxed))
    }

    pub fn set_bytes(&self, b: u64) {
        self.bytes.store(b, Ordering::Relaxed);
    }

    pub fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    pub fn set_disabled(&self, on: bool) {
        self.disabled.store(on, Ordering::Relaxed);
    }

    pub fn disabled(&self) -> bool {
        self.disabled.load(Ordering::Relaxed)
    }
}

/// The process's counters.
pub static METRICS: DatasetMetrics = DatasetMetrics::new();

/// Append the dataset series to a Prometheus text exposition (`prefix` = the metric prefix).
pub fn render_prometheus(out: &mut String, prefix: &str) {
    let m = &METRICS;
    let counters = [
        (
            "requests",
            "Detection requests received since the service started (test page uploads excluded).",
            m.requests(),
        ),
        (
            "offered",
            "Answered requests handed to the dataset collector.",
            m.offered(),
        ),
        ("saved", "Frames the dataset collector saved.", m.saved()),
        (
            "no_bucket",
            "Frames the dataset collector skipped because they matched no enabled bucket.",
            m.no_bucket(),
        ),
    ];
    for (name, help, value) in counters {
        let _ = writeln!(out, "# HELP {prefix}dataset_{name}_total {help}");
        let _ = writeln!(out, "# TYPE {prefix}dataset_{name}_total counter");
        let _ = writeln!(out, "{prefix}dataset_{name}_total {value}");
    }
    let _ = writeln!(
        out,
        "# HELP {prefix}dataset_frames_total Frames saved by the dataset collector, by bucket (a frame counts once in each bucket it was kept for)."
    );
    let _ = writeln!(out, "# TYPE {prefix}dataset_frames_total counter");
    for b in Bucket::ALL {
        let _ = writeln!(
            out,
            "{prefix}dataset_frames_total{{bucket=\"{}\"}} {}",
            b.as_str(),
            m.frames(b)
        );
    }
    let _ = writeln!(
        out,
        "# HELP {prefix}dataset_dropped_total Frames the dataset collector did not save, by reason (queue_full|rate|dup|disk|error)."
    );
    let _ = writeln!(out, "# TYPE {prefix}dataset_dropped_total counter");
    for r in Dropped::ALL {
        let _ = writeln!(
            out,
            "{prefix}dataset_dropped_total{{reason=\"{}\"}} {}",
            r.as_str(),
            m.dropped(r)
        );
    }
    let _ = writeln!(
        out,
        "# HELP {prefix}dataset_bytes Bytes of the frames and sidecars in the dataset folder."
    );
    let _ = writeln!(out, "# TYPE {prefix}dataset_bytes gauge");
    let _ = writeln!(out, "{prefix}dataset_bytes {}", m.bytes());
    let _ = writeln!(
        out,
        "# HELP {prefix}dataset_collect_disabled 1 while the collector is disabled for low disk space or a full disk."
    );
    let _ = writeln!(out, "# TYPE {prefix}dataset_collect_disabled gauge");
    let _ = writeln!(
        out,
        "{prefix}dataset_collect_disabled {}",
        u8::from(m.disabled())
    );
}
