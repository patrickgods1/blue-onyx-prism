//! The memory sampler of a training attempt: a thread that reads the system commit, the available
//! RAM and the page-in rate every [`SAMPLE_EVERY`] (`system_info`, `backend::memory`) from the
//! start of the training stage until the trainer exports (or exits), and keeps the start, the
//! peaks and the minimum ([`Usage`], pure) for the run's [`super::RunResources`].
//!
//! The first reading is taken before [`Sampler::start`] returns, so it is the baseline from before
//! the trainer spawned. A reading every 5 s can miss a peak shorter than that; the memory a
//! training run holds grows over minutes (workers spawn, batches prefetch), so the peaks are close.

use crate::backend::memory::{Commit, PageIns};
use crate::sync::LockExt;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// Time between readings.
pub const SAMPLE_EVERY: Duration = Duration::from_secs(5);
/// Longest wait in [`Sampler::start`] for the first reading.
const FIRST_READING_WAIT: Duration = Duration::from_secs(2);

/// One reading.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Sample {
    /// System commit (Windows).
    pub commit: Option<Commit>,
    /// Available RAM, bytes (Windows, Linux).
    pub ram_available: Option<u64>,
    /// Pages read in per second since the reading before (Windows, Linux).
    pub page_ins: Option<f64>,
}

/// What the readings of one attempt add up to.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Usage {
    pub samples: u32,
    /// Commit in use at the first reading with one.
    pub commit_start: Option<u64>,
    /// The most commit in use, and the commit limit at that reading.
    pub commit_peak: Option<u64>,
    pub commit_limit_at_peak: Option<u64>,
    /// Available RAM at the first reading with one, and the least.
    pub ram_start: Option<u64>,
    pub ram_min: Option<u64>,
    /// The most pages read in per second.
    pub page_ins_peak: Option<f64>,
}

impl Usage {
    /// Count `s` in.
    pub fn add(&mut self, s: &Sample) {
        self.samples = self.samples.saturating_add(1);
        if let Some(c) = s.commit {
            let used = c.in_use();
            self.commit_start.get_or_insert(used);
            if self.commit_peak.is_none_or(|p| used > p) {
                self.commit_peak = Some(used);
                self.commit_limit_at_peak = Some(c.limit);
            }
        }
        if let Some(r) = s.ram_available {
            self.ram_start.get_or_insert(r);
            self.ram_min = Some(self.ram_min.map_or(r, |m| m.min(r)));
        }
        if let Some(p) = s.page_ins.filter(|p| p.is_finite() && *p >= 0.0) {
            self.page_ins_peak = Some(self.page_ins_peak.map_or(p, |m| m.max(p)));
        }
    }
}

#[derive(Default)]
struct State {
    stop: bool,
    usage: Usage,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
}

/// The sampling thread; stopped by [`Sampler::stop`], [`Sampler::finish`] or drop.
pub struct Sampler {
    shared: Arc<Shared>,
    join: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for Sampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sampler")
            .field("usage", &self.usage())
            .finish_non_exhaustive()
    }
}

impl Sampler {
    /// Sample this machine every [`SAMPLE_EVERY`].
    pub fn start() -> Self {
        Self::start_with(SAMPLE_EVERY, || {
            // Opened on the sampling thread (the counter is not `Send`); the first reading has no
            // rate (nothing to compare with yet).
            let mut pages = PageIns::open();
            let mut first = true;
            move || Sample {
                commit: crate::system_info::commit(),
                ram_available: crate::system_info::ram_available(),
                page_ins: if std::mem::take(&mut first) {
                    None
                } else {
                    pages.as_mut().and_then(PageIns::rate)
                },
            }
        })
    }

    /// Sample with `probe` (made on the thread by `make`) every `every`. Returns once the first
    /// reading is in (or after [`FIRST_READING_WAIT`]); a thread that cannot start leaves the
    /// usage empty.
    pub fn start_with<F, P>(every: Duration, make: F) -> Self
    where
        F: FnOnce() -> P + Send + 'static,
        P: FnMut() -> Sample,
    {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
        });
        let s = Arc::clone(&shared);
        let join = std::thread::Builder::new()
            .name("train-sampler".into())
            .spawn(move || {
                let mut probe = make();
                loop {
                    let sample = probe();
                    let mut st = s.state.lock_recover();
                    if st.stop {
                        return;
                    }
                    st.usage.add(&sample);
                    s.wake.notify_all();
                    let (st, _) = s
                        .wake
                        .wait_timeout_while(st, every, |st| !st.stop)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if st.stop {
                        return;
                    }
                }
            });
        let join = match join {
            Ok(j) => Some(j),
            Err(e) => {
                tracing::warn!("cannot start the training memory sampler: {e}");
                None
            }
        };
        if join.is_some() {
            let st = shared.state.lock_recover();
            let _ = shared
                .wake
                .wait_timeout_while(st, FIRST_READING_WAIT, |st| {
                    st.usage.samples == 0 && !st.stop
                });
        }
        Self { shared, join }
    }

    /// No more readings (the trainer exports now); what was read is kept.
    pub fn stop(&self) {
        self.shared.state.lock_recover().stop = true;
        self.shared.wake.notify_all();
    }

    /// The readings so far.
    pub fn usage(&self) -> Usage {
        self.shared.state.lock_recover().usage
    }

    /// Stop, wait for the thread and return the readings.
    pub fn finish(mut self) -> Usage {
        self.stop_and_join();
        self.usage()
    }

    fn stop_and_join(&mut self) {
        self.stop();
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    fn commit(limit_gb: f64, used_gb: f64) -> Option<Commit> {
        let b = |g: f64| (g * GB as f64) as u64;
        Some(Commit {
            limit: b(limit_gb),
            available: b(limit_gb) - b(used_gb),
        })
    }

    #[test]
    fn usage_keeps_the_start_the_peaks_and_the_minimum() {
        let mut u = Usage::default();
        u.add(&Sample::default());
        assert_eq!(u.samples, 1);
        assert_eq!(u.commit_start, None);
        let s = |limit, used, ram: u64, pages| Sample {
            commit: commit(limit, used),
            ram_available: Some(ram * GB / 10),
            page_ins: pages,
        };
        u.add(&s(32.4, 17.9, 76, None));
        u.add(&s(32.4, 31.4, 21, Some(2800.0)));
        // A larger limit later (the paging file grew) with less in use: the peak keeps its limit.
        u.add(&s(37.8, 30.0, 30, Some(f64::NAN)));
        u.add(&s(37.8, 29.0, 40, Some(500.0)));
        assert_eq!(u.samples, 5);
        assert_eq!(u.commit_start, commit(32.4, 17.9).map(|c| c.in_use()));
        assert_eq!(u.commit_peak, commit(32.4, 31.4).map(|c| c.in_use()));
        assert_eq!(u.commit_limit_at_peak, commit(32.4, 31.4).map(|c| c.limit));
        assert_eq!(u.ram_start, Some(76 * GB / 10));
        assert_eq!(u.ram_min, Some(21 * GB / 10));
        assert_eq!(u.page_ins_peak, Some(2800.0));
    }

    /// A fake probe: the first reading is in before `start_with` returns, readings follow every
    /// `every`, and none after `stop`.
    #[test]
    fn sampler_reads_until_stopped() {
        let sampler = Sampler::start_with(Duration::from_millis(10), || {
            let mut n = 0u64;
            move || {
                n += 1;
                Sample {
                    commit: None,
                    ram_available: Some(1000 - n),
                    page_ins: None,
                }
            }
        });
        assert!(sampler.usage().samples >= 1);
        assert_eq!(sampler.usage().ram_start, Some(999));
        let t = std::time::Instant::now();
        while sampler.usage().samples < 3 && t.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(5));
        }
        sampler.stop();
        let at_stop = sampler.usage().samples;
        assert!(at_stop >= 3, "{at_stop}");
        std::thread::sleep(Duration::from_millis(50));
        let u = sampler.finish();
        assert_eq!(u.samples, at_stop, "no reading counts after stop");
        assert_eq!(u.ram_min, Some(1000 - u64::from(u.samples)));
    }

    /// The real probe reads something on Windows and Linux; nothing panics elsewhere.
    #[test]
    fn live_sampler_reads_this_machine() {
        let u = Sampler::start().finish();
        assert!(u.samples >= 1);
        if cfg!(windows) {
            assert!(u.commit_start.is_some() && u.ram_start.is_some(), "{u:?}");
        }
        if cfg!(target_os = "linux") {
            assert!(u.ram_start.is_some(), "{u:?}");
        }
    }
}
