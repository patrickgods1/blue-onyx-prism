//! In-memory log ring for the web UI's Logs page (`GET /logs`, `GET /logs.json`).
//!
//! [`LogBuffer`] is a `tracing` [`Layer`] that keeps the last [`DEFAULT_CAPACITY`] events
//! (timestamp, level, target, message plus fields) in a bounded ring. Every event gets a
//! monotonically increasing sequence number, so a client polls with `after=<last seq>` and only
//! receives what is new. `cli::init_logging_with` always installs it under the reloadable level
//! filter, so it holds exactly what the configured level emits.

use crate::sync::LockExt;
use serde::Serialize;
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex};
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;

/// Events kept in memory.
pub const DEFAULT_CAPACITY: usize = 2000;
/// Most entries one `/logs.json` reply returns.
pub const MAX_PAGE: usize = 2000;
/// Longest message stored per entry; the rest is cut on a char boundary and replaced by a
/// `… [truncated N bytes]` marker (a request-controlled string can be hundreds of KiB).
pub const MAX_MESSAGE_BYTES: usize = 4096;
/// Longest `target` stored per entry.
pub const MAX_TARGET_BYTES: usize = 256;
/// Byte budget of the whole ring (messages + targets); the oldest entries are evicted beyond it,
/// in addition to the entry cap.
pub const MAX_RING_BYTES: usize = 8 * 1024 * 1024;

/// `s` cut to at most `max` bytes on a char boundary, with a truncation marker when cut.
fn truncate_text(mut s: String, max: usize) -> String {
    if s.len() <= max {
        return s;
    }
    let mut cut = max;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    let dropped = s.len() - cut;
    s.truncate(cut);
    let _ = write!(s, "… [truncated {dropped} bytes]");
    s
}

fn entry_bytes(e: &LogEntry) -> usize {
    e.message.len() + e.target.len()
}

/// One captured event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LogEntry {
    /// 1-based, increasing by one per event (also across evictions).
    pub seq: u64,
    /// RFC 3339 UTC with milliseconds.
    pub ts: String,
    /// `ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE`.
    pub level: &'static str,
    pub target: String,
    /// The message followed by the other fields as `key=value`.
    pub message: String,
}

#[derive(Debug)]
struct Ring {
    entries: VecDeque<LogEntry>,
    capacity: usize,
    /// Sequence number of the last pushed entry (0 = none yet).
    last_seq: u64,
    /// Sum of `entry_bytes` over `entries`.
    bytes: usize,
}

/// Bounded log ring; cheap to clone (shared).
#[derive(Debug, Clone)]
pub struct LogBuffer {
    inner: Arc<Mutex<Ring>>,
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

/// A page of entries for `/logs.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LogPage {
    pub entries: Vec<LogEntry>,
    /// The newest sequence number in the buffer: poll again with `after=last`.
    pub last: u64,
    /// Oldest sequence number still held (entries before it were evicted).
    pub first: u64,
    /// Entries between `after` and `first` were evicted before the client saw them.
    pub truncated: bool,
}

/// `error` < `warn` < `info` < `debug` < `trace` in verbosity; parse a level name (any case).
pub fn parse_level(s: &str) -> Option<Level> {
    match s.trim().to_ascii_lowercase().as_str() {
        "error" => Some(Level::ERROR),
        "warn" | "warning" => Some(Level::WARN),
        "info" => Some(Level::INFO),
        "debug" => Some(Level::DEBUG),
        "trace" => Some(Level::TRACE),
        _ => None,
    }
}

fn level_from_str(s: &str) -> Level {
    parse_level(s).unwrap_or(Level::TRACE)
}

impl LogBuffer {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            inner: Arc::new(Mutex::new(Ring {
                entries: VecDeque::with_capacity(capacity.min(4096)),
                capacity,
                last_seq: 0,
                bytes: 0,
            })),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Ring> {
        self.inner.lock_recover()
    }

    /// Append an entry (evicting the oldest when full); returns its sequence number.
    pub fn push(&self, level: Level, target: &str, message: String) -> u64 {
        let ts = chrono::Utc::now()
            .format("%Y-%m-%dT%H:%M:%S%.3fZ")
            .to_string();
        let entry = LogEntry {
            seq: 0,
            ts,
            level: level.as_str(),
            target: truncate_text(target.to_string(), MAX_TARGET_BYTES),
            message: truncate_text(message, MAX_MESSAGE_BYTES),
        };
        let mut r = self.lock();
        r.last_seq += 1;
        let seq = r.last_seq;
        if r.entries.len() >= r.capacity
            && let Some(old) = r.entries.pop_front()
        {
            r.bytes -= entry_bytes(&old);
        }
        r.bytes += entry_bytes(&entry);
        r.entries.push_back(LogEntry { seq, ..entry });
        while r.bytes > MAX_RING_BYTES && r.entries.len() > 1 {
            if let Some(old) = r.entries.pop_front() {
                r.bytes -= entry_bytes(&old);
            }
        }
        seq
    }

    /// Number of entries held.
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Entries with `seq > after` at `min` severity or more severe (None = all), oldest first,
    /// at most `limit` (the newest ones when there are more).
    pub fn since(&self, after: u64, min: Option<Level>, limit: usize) -> LogPage {
        let r = self.lock();
        let first = r.entries.front().map_or(r.last_seq + 1, |e| e.seq);
        let mut entries: Vec<LogEntry> = r
            .entries
            .iter()
            .filter(|e| e.seq > after)
            // tracing orders levels by verbosity: ERROR < WARN < ... < TRACE.
            .filter(|e| min.is_none_or(|m| level_from_str(e.level) <= m))
            .cloned()
            .collect();
        if entries.len() > limit {
            entries.drain(..entries.len() - limit);
        }
        LogPage {
            entries,
            last: r.last_seq,
            first,
            truncated: after + 1 < first && after < r.last_seq,
        }
    }
}

/// Collects the `message` field and the other fields of an event.
#[derive(Default)]
struct FieldText {
    message: String,
    fields: String,
}

impl Visit for FieldText {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        } else {
            let _ = write!(self.fields, " {}={}", field.name(), value);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
        } else if !field.name().starts_with("log.") {
            let _ = write!(self.fields, " {}={:?}", field.name(), value);
        }
    }
}

impl<S: Subscriber> Layer<S> for LogBuffer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut v = FieldText::default();
        event.record(&mut v);
        let mut text = v.message;
        if !v.fields.is_empty() {
            if text.is_empty() {
                text.push_str(v.fields.trim_start());
            } else {
                text.push_str(&v.fields);
            }
        }
        let meta = event.metadata();
        self.push(*meta.level(), meta.target(), text);
    }
}

// ---------------------------------------------------------------------------------------------
// Messages from before logging is set up (CO-11)

/// Most messages kept until logging starts (the rest are dropped; a run produces a handful).
const MAX_EARLY: usize = 64;

/// Messages logged before the global subscriber exists (config resolution runs before
/// `cli::init_logging` because the config names the log level and directory).
static EARLY: std::sync::Mutex<Vec<(Level, String)>> = std::sync::Mutex::new(Vec::new());

fn emit(level: Level, msg: &str) {
    match level {
        Level::ERROR => tracing::error!("{msg}"),
        Level::WARN => tracing::warn!("{msg}"),
        Level::INFO => tracing::info!("{msg}"),
        Level::DEBUG => tracing::debug!("{msg}"),
        _ => tracing::trace!("{msg}"),
    }
}

/// Log `msg` at `level` now when logging is set up, else keep it until [`replay_early`]
/// (called by `cli::init_logging_with` once the subscriber is installed).
pub fn early(level: Level, msg: impl Into<String>) {
    let msg = msg.into();
    if tracing::dispatcher::has_been_set() {
        emit(level, &msg);
        return;
    }
    let mut q = EARLY.lock_recover();
    if q.len() < MAX_EARLY {
        q.push((level, msg));
    }
}

/// Log (and forget) the messages [`early`] kept.
pub fn replay_early() {
    let pending = std::mem::take(&mut *EARLY.lock_recover());
    for (level, msg) in pending {
        emit(level, &msg);
    }
}

/// Take the messages [`early`] kept without logging them (a command that never sets up
/// logging prints them instead).
pub fn take_early() -> Vec<(Level, String)> {
    std::mem::take(&mut *EARLY.lock_recover())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_subscriber::layer::SubscriberExt;

    #[test]
    fn ring_keeps_the_newest_entries() {
        let b = LogBuffer::new(3);
        for i in 1..=5 {
            assert_eq!(b.push(Level::INFO, "t", format!("m{i}")), i);
        }
        assert_eq!(b.len(), 3);
        let p = b.since(0, None, 100);
        let msgs: Vec<_> = p.entries.iter().map(|e| e.message.as_str()).collect();
        assert_eq!(msgs, ["m3", "m4", "m5"]);
        assert_eq!((p.first, p.last), (3, 5));
        assert!(p.truncated, "entries 1..2 were evicted unseen");
    }

    #[test]
    fn cursor_returns_only_new_entries() {
        let b = LogBuffer::new(10);
        b.push(Level::INFO, "a", "one".into());
        b.push(Level::WARN, "a", "two".into());
        let p = b.since(0, None, 100);
        assert_eq!(p.entries.len(), 2);
        assert_eq!(p.last, 2);
        assert!(!p.truncated);
        let p = b.since(p.last, None, 100);
        assert!(p.entries.is_empty());
        assert_eq!(p.last, 2);
        b.push(Level::ERROR, "a", "three".into());
        let p = b.since(2, None, 100);
        assert_eq!(p.entries.len(), 1);
        assert_eq!(p.entries[0].seq, 3);
        assert_eq!(p.entries[0].level, "ERROR");
        // A cursor ahead of the buffer (server restarted) returns nothing.
        assert!(b.since(99, None, 100).entries.is_empty());
    }

    #[test]
    fn level_filter_and_limit() {
        let b = LogBuffer::new(10);
        for (l, m) in [
            (Level::TRACE, "t"),
            (Level::DEBUG, "d"),
            (Level::INFO, "i"),
            (Level::WARN, "w"),
            (Level::ERROR, "e"),
        ] {
            b.push(l, "x", m.into());
        }
        let msgs =
            |p: LogPage| -> Vec<String> { p.entries.into_iter().map(|e| e.message).collect() };
        assert_eq!(msgs(b.since(0, Some(Level::WARN), 100)), ["w", "e"]);
        assert_eq!(msgs(b.since(0, Some(Level::INFO), 100)), ["i", "w", "e"]);
        assert_eq!(msgs(b.since(0, None, 2)), ["w", "e"]);
        assert_eq!(parse_level("Warning"), Some(Level::WARN));
        assert_eq!(parse_level("loud"), None);
    }

    #[test]
    fn layer_captures_message_and_fields() {
        let b = LogBuffer::new(10);
        let sub = tracing_subscriber::registry().with(b.clone());
        tracing::subscriber::with_default(sub, || {
            tracing::info!(model = "m1", count = 3, "loaded");
            tracing::warn!(target: "custom", "plain");
        });
        let p = b.since(0, None, 10);
        assert_eq!(p.entries.len(), 2);
        assert_eq!(p.entries[0].message, "loaded model=m1 count=3");
        assert_eq!(p.entries[0].level, "INFO");
        assert_eq!(p.entries[1].target, "custom");
        assert_eq!(p.entries[1].message, "plain");
        assert!(p.entries[0].ts.ends_with('Z'));
    }

    #[test]
    fn huge_message_is_truncated_on_a_char_boundary() {
        let b = LogBuffer::new(10);
        b.push(Level::INFO, &"t".repeat(1000), "é".repeat(200 * 1024));
        let e = &b.since(0, None, 10).entries[0];
        assert_eq!(
            e.target.len(),
            MAX_TARGET_BYTES + "… [truncated 744 bytes]".len()
        );
        assert!(
            e.message.len() < MAX_MESSAGE_BYTES + 64,
            "{}",
            e.message.len()
        );
        assert!(e.message.contains("… [truncated "));
        assert!(e.message.starts_with("éé"));
        // Short messages are untouched.
        b.push(Level::INFO, "t", "ok".into());
        assert_eq!(b.since(1, None, 10).entries[0].message, "ok");
    }

    #[test]
    fn ring_stays_under_the_byte_budget() {
        // Entry cap above the byte cap: 3000 x 4 KiB would be 12 MiB.
        let b = LogBuffer::new(4000);
        for _ in 0..3000 {
            b.push(Level::INFO, "t", "x".repeat(4096));
        }
        let bytes = b.lock().bytes;
        assert!(bytes <= MAX_RING_BYTES, "{bytes}");
        assert!(b.len() < 3000, "oldest entries were evicted by bytes");
        let p = b.since(0, None, MAX_PAGE);
        assert_eq!(p.last, 3000);
        assert!(p.truncated);
        // Paging still works from the cursor.
        b.push(Level::INFO, "t", "new".into());
        let p = b.since(3000, None, 10);
        assert_eq!(p.entries.len(), 1);
        assert_eq!(p.entries[0].seq, 3001);
    }
}
