//! Process-wide dataset collector counters (`dataset_frames_total{bucket}`,
//! `dataset_dropped_total{reason}`, `dataset_bytes`, `dataset_collect_disabled`) and their
//! Prometheus rendering, called from [`crate::metrics::render_prometheus`].

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
}

/// The counters.
#[derive(Debug)]
pub struct DatasetMetrics {
    frames: [AtomicU64; Bucket::ALL.len()],
    dropped: [AtomicU64; Dropped::ALL.len()],
    bytes: AtomicU64,
    disabled: AtomicBool,
}

impl DatasetMetrics {
    const fn new() -> Self {
        Self {
            frames: [const { AtomicU64::new(0) }; Bucket::ALL.len()],
            dropped: [const { AtomicU64::new(0) }; Dropped::ALL.len()],
            bytes: AtomicU64::new(0),
            disabled: AtomicBool::new(false),
        }
    }

    pub fn record_frame(&self, buckets: &[Bucket]) {
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
