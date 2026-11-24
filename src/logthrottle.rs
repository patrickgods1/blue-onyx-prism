//! One rate limiter for log lines that a lasting fault would repeat per request, frame or
//! connection.
//!
//! A [`Throttle`] lets a line through at most once per its interval and counts the ones it held
//! back, so the line that gets through can say how many were dropped (`similar_since_last_warning`
//! or `dropped`). It is lock-free and `const`-constructible, so a throttle can be a `static`
//! (one per kind of line in a module) or a field (one per model, in `metrics::ModelMetrics`).
//!
//! Why it matters: the Windows service sends every INFO+ line to the Event Log
//! (`ReportEventW` per line) and the file log caps a day at 200 MB, so an unthrottled warning
//! that fires per request can wipe the Application log and fill the day's file within minutes.
//! Use one throttle per kind of line, never one shared by unrelated lines (a flood of one would
//! hide the other), and never a map keyed by client-chosen text (it would grow without bound).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Interval of [`Throttle::default`]: per-model and per-request lines (overload, queue full,
/// night fallback, verifier) log at most once per this long.
pub const DEFAULT_EVERY: Duration = Duration::from_secs(10);

/// `last_ms` before the first line was let through.
const NEVER: u64 = u64::MAX;

/// Lets a repeated log line through at most once per interval and counts the ones it held back.
#[derive(Debug)]
pub struct Throttle {
    every_ms: u64,
    /// When the last line was let through, in the caller's millisecond clock ([`NEVER`]: never).
    last_ms: AtomicU64,
    held: AtomicU64,
}

impl Default for Throttle {
    fn default() -> Self {
        Self::new(DEFAULT_EVERY)
    }
}

impl Throttle {
    /// A throttle that lets one line through per `every` (whole milliseconds; a sub-millisecond
    /// interval lets every line through).
    pub const fn new(every: Duration) -> Self {
        let ms = every.as_millis();
        let every_ms = if ms > u64::MAX as u128 {
            u64::MAX
        } else {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "checked against u64::MAX on the line above"
            )]
            let ms = ms as u64;
            ms
        };
        Self {
            every_ms,
            last_ms: AtomicU64::new(NEVER),
            held: AtomicU64::new(0),
        }
    }

    /// The interval this throttle was made with.
    pub const fn every(&self) -> Duration {
        Duration::from_millis(self.every_ms)
    }

    /// At `now_ms` (any monotonic millisecond clock, the same one on every call): `Some(n)` when
    /// the line should be logged now, `n` being how many were held back since the last one;
    /// `None` to skip it (and count it).
    pub fn allow(&self, now_ms: u64) -> Option<u64> {
        let last = self.last_ms.load(Ordering::Acquire);
        let due = last == NEVER || now_ms.saturating_sub(last) >= self.every_ms;
        if due
            && self
                .last_ms
                .compare_exchange(last, now_ms, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        {
            Some(self.held.swap(0, Ordering::AcqRel))
        } else {
            self.held.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// [`Self::allow`] on the process clock ([`process_ms`]), for `static` throttles.
    pub fn allow_now(&self) -> Option<u64> {
        self.allow(process_ms())
    }

    /// [`Self::allow`] at `now` on the process clock (tests step `now` instead of sleeping).
    pub fn allow_at(&self, now: Instant) -> Option<u64> {
        self.allow(ms_since_start(now))
    }

    /// Lines held back since the last one let through.
    pub fn held_back(&self) -> u64 {
        self.held.load(Ordering::Relaxed)
    }
}

/// How far before its first use the process clock starts, so an `Instant` a caller took just
/// before that first use (a test's `t0`, a line timed before the first throttle call) still maps
/// to its own millisecond instead of all collapsing to 0.
const CLOCK_LEAD: [Duration; 3] = [
    Duration::from_secs(3600),
    Duration::from_secs(60),
    Duration::from_secs(1),
];

/// Start of the process clock: [`CLOCK_LEAD`] before the first use of this module (the largest
/// lead the platform's `Instant` can represent; just after boot it may be none).
fn start() -> Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    *START.get_or_init(|| {
        let now = Instant::now();
        CLOCK_LEAD
            .iter()
            .find_map(|lead| now.checked_sub(*lead))
            .unwrap_or(now)
    })
}

fn ms_since_start(now: Instant) -> u64 {
    u64::try_from(now.saturating_duration_since(start()).as_millis()).unwrap_or(u64::MAX - 1)
}

/// Milliseconds on the process clock (monotonic, starts at the first use of this module).
pub fn process_ms() -> u64 {
    ms_since_start(Instant::now())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lets_one_through_per_interval_and_counts_the_rest() {
        let t = Throttle::new(Duration::from_secs(10));
        assert_eq!(t.allow(1_000), Some(0));
        assert_eq!(t.allow(1_001), None);
        assert_eq!(t.allow(10_999), None);
        assert_eq!(t.held_back(), 2);
        assert_eq!(t.allow(11_000), Some(2));
        assert_eq!(t.held_back(), 0);
        assert_eq!(t.allow(11_001), None);
    }

    #[test]
    fn first_line_passes_at_time_zero() {
        let t = Throttle::default();
        assert_eq!(t.every(), DEFAULT_EVERY);
        assert_eq!(t.allow(0), Some(0));
        assert_eq!(t.allow(0), None);
    }

    #[test]
    fn a_static_throttle_works_on_the_process_clock() {
        static T: Throttle = Throttle::new(Duration::from_secs(60));
        let now = Instant::now();
        assert_eq!(T.allow_at(now), Some(0));
        assert_eq!(T.allow_at(now + Duration::from_secs(59)), None);
        assert_eq!(T.allow_at(now + Duration::from_secs(61)), Some(1));
        assert!(T.allow_now().is_none());
    }

    #[test]
    fn an_instant_taken_before_the_clock_started_keeps_its_place() {
        let t0 = Instant::now();
        let t = Throttle::new(Duration::from_secs(10));
        assert_eq!(t.allow_at(t0), Some(0));
        assert_eq!(t.allow_at(t0 + Duration::from_secs(9)), None);
        assert_eq!(t.allow_at(t0 + Duration::from_secs(10)), Some(1));
    }

    #[test]
    fn huge_and_tiny_intervals() {
        let t = Throttle::new(Duration::MAX);
        assert_eq!(t.every(), Duration::from_millis(u64::MAX));
        assert_eq!(t.allow(5), Some(0));
        assert_eq!(t.allow(u64::MAX - 1), None);
        let z = Throttle::new(Duration::from_micros(10));
        assert_eq!(z.allow(5), Some(0));
        assert_eq!(z.allow(5), Some(0));
    }
}
