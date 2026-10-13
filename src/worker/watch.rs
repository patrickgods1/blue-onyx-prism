//! Watchdogs that run beside a blocking runtime call: [`collect_watched`] marks a model
//! `Failed` with [`stuck_message`] when the collect of an overdue inference outlasts a second
//! request timeout (and `Ready` again if it returns successfully later), and
//! [`watch_slow_compile`] warns about a compile that takes minutes. Each watcher is a scoped
//! thread that ends with the call it watches.

use super::{COMPILING_ON, millis_u64};
use crate::startup::{ModelState, StateHandle};
use anyhow::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tracing::{error, info, warn};

/// The `Failed` text of a model whose inference is stuck (PL-06).
pub(crate) fn stuck_message(device: &str, waited: Duration) -> String {
    format!(
        "inference stuck on {device}: no result {} s after it started; the device driver may \
         have hung or the GPU reset; reload the model (save it again on the Config page) or \
         restart the service, and try another `device` if it happens again",
        waited.as_secs()
    )
}

/// Prefix of [`stuck_message`] (what a caller can match on).
pub(crate) const STUCK_PREFIX: &str = "inference stuck on";

/// Who to mark `Failed` when an inference is stuck (see [`collect_watched`]).
#[derive(Debug, Clone)]
pub(super) struct StuckWatch {
    pub(super) name: String,
    /// Canonical spec of the device ("openvino:gpu").
    pub(super) device: String,
    pub(super) state: StateHandle,
    pub(super) stuck: Arc<AtomicBool>,
}

impl StuckWatch {
    /// Mark the model `Failed` with [`stuck_message`] (`waited`: since the inference started;
    /// `cause`: the backend error, when the session reported its death).
    pub(super) fn mark(&self, waited: Duration, cause: Option<&anyhow::Error>) {
        let mut msg = stuck_message(&self.device, waited);
        if let Some(e) = cause {
            msg = format!("{msg} (runtime error: {e:#})");
        }
        error!(model = %self.name, device = %self.device, "{msg}");
        self.stuck.store(true, Ordering::Release);
        self.state.set(ModelState::Failed(msg));
    }

    /// The stuck inference finished after all: serve again.
    fn clear(&self, waited: Duration) {
        self.stuck.store(false, Ordering::Release);
        if matches!(self.state.get(), ModelState::Failed(m) if m.starts_with(STUCK_PREFIX)) {
            self.state.set(ModelState::Ready);
        }
        info!(
            model = %self.name,
            device = %self.device,
            waited_ms = millis_u64(waited),
            "the stuck inference finished after all; the model is ready again"
        );
    }
}

/// Run `collect` (the blocking collect of an inference that is already a request timeout
/// late) under a watchdog: when it has not returned after `limit`, the model is marked `Failed`
/// (the worker thread stays blocked in `collect`; the server rejects requests at once from
/// then on). If `collect` does return later, a success makes the model `Ready` again and an
/// error leaves it `Failed` (the worker then stops using the engine).
pub(super) fn collect_watched<T>(
    watch: &StuckWatch,
    limit: Duration,
    collect: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let started = Instant::now();
    let (tx, rx) = crossbeam_channel::bounded::<()>(0);
    std::thread::scope(|s| {
        let watcher = std::thread::Builder::new()
            .name(format!("stuck-{}", watch.name))
            .spawn_scoped(s, || {
                if rx.recv_timeout(limit) == Err(crossbeam_channel::RecvTimeoutError::Timeout) {
                    // The inference started a request timeout before this wait.
                    watch.mark(started.elapsed() + limit, None);
                }
            });
        if let Err(e) = &watcher {
            warn!(model = %watch.name, "could not start the stuck-inference watchdog: {e}");
        }
        let r = collect();
        drop(tx);
        if let Ok(w) = watcher {
            let _ = w.join();
        }
        if watch.stuck.load(Ordering::Acquire) && r.is_ok() {
            watch.clear(started.elapsed() + limit);
        }
        r
    })
}

/// When a compile counts as slow and how often the watcher looks.
#[derive(Debug, Clone, Copy)]
pub(super) struct SlowCompile {
    /// Warn (and say "compiling for N min" in the state) after this long, and again every
    /// time this much has passed since the last warning.
    after: Duration,
    tick: Duration,
}

impl SlowCompile {
    pub(super) const DEFAULT: Self = Self {
        after: Duration::from_secs(5 * 60),
        tick: Duration::from_secs(30),
    };
}

/// "compiling on openvino:gpu for 7 min" (keeps the [`COMPILING_ON`] prefix the pages use).
fn compiling_for(device: &str, elapsed: Duration) -> String {
    format!("{COMPILING_ON} {device} for {} min", elapsed.as_secs() / 60)
}

/// Run `compile` (a model compile on `device`) and, from another thread, warn when it takes
/// longer than `slow.after` (again at most every `slow.after`) and show "compiling for N min"
/// in the model state. Returns the result and how many warnings were logged.
pub(super) fn watch_slow_compile<T>(
    name: &str,
    device: &str,
    state: &StateHandle,
    slow: SlowCompile,
    compile: impl FnOnce() -> T,
) -> (T, u32) {
    let started = Instant::now();
    let (tx, rx) = crossbeam_channel::bounded::<()>(0);
    std::thread::scope(|s| {
        let watcher = std::thread::Builder::new()
            .name(format!("compile-{name}"))
            .spawn_scoped(s, || {
                let mut warned = 0u32;
                let mut last_warn: Option<Duration> = None;
                while rx.recv_timeout(slow.tick)
                    == Err(crossbeam_channel::RecvTimeoutError::Timeout)
                {
                    let elapsed = started.elapsed();
                    if elapsed < slow.after {
                        continue;
                    }
                    state.set_initializing(compiling_for(device, elapsed));
                    if last_warn.is_none_or(|w| elapsed.saturating_sub(w) >= slow.after) {
                        warn!(
                            model = %name,
                            device = %device,
                            minutes = elapsed.as_secs() / 60,
                            "still compiling on {device} after {} min; a first GPU compile can \
                             take minutes, check the device and cache_dir if it never finishes",
                            elapsed.as_secs() / 60
                        );
                        last_warn = Some(elapsed);
                        warned += 1;
                    }
                }
                warned
            });
        let r = compile();
        drop(tx);
        let warned = watcher.ok().and_then(|w| w.join().ok()).unwrap_or(0);
        (r, warned)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::run::session_dead;
    use std::sync::Mutex;

    fn watch() -> StuckWatch {
        let state = StateHandle::new();
        state.set(ModelState::Ready);
        StuckWatch {
            name: "m".into(),
            device: "openvino:gpu".into(),
            state,
            stuck: Arc::new(AtomicBool::new(false)),
        }
    }

    #[test]
    fn stuck_watchdog_marks_failed_and_recovers() {
        // Returns in time: nothing changes.
        let w = watch();
        let r = collect_watched(&w, Duration::from_secs(5), || Ok(7));
        assert_eq!(r.unwrap(), 7);
        assert_eq!(w.state.get(), ModelState::Ready);
        assert!(!w.stuck.load(Ordering::Acquire));

        // Outlasts the limit, then fails: the model stays Failed and is flagged stuck.
        let w = watch();
        let seen = Arc::new(Mutex::new(None));
        let r: Result<()> = collect_watched(&w, Duration::from_millis(20), || {
            let t = Instant::now();
            while !w.stuck.load(Ordering::Acquire) && t.elapsed() < Duration::from_secs(5) {
                std::thread::sleep(Duration::from_millis(5));
            }
            *seen.lock().unwrap() = Some(w.state.get());
            anyhow::bail!("device lost")
        });
        assert!(r.is_err());
        let ModelState::Failed(msg) = w.state.get() else {
            panic!("expected Failed");
        };
        assert!(msg.starts_with(STUCK_PREFIX), "{msg}");
        assert!(msg.contains("openvino:gpu"), "{msg}");
        assert!(msg.contains("reload the model"), "{msg}");
        assert!(matches!(*seen.lock().unwrap(), Some(ModelState::Failed(_))));
        assert!(w.stuck.load(Ordering::Acquire));

        // Outlasts the limit, then succeeds: Ready again.
        let w = watch();
        let r = collect_watched(&w, Duration::from_millis(20), || {
            let t = Instant::now();
            while !w.stuck.load(Ordering::Acquire) && t.elapsed() < Duration::from_secs(5) {
                std::thread::sleep(Duration::from_millis(5));
            }
            Ok(1)
        });
        assert_eq!(r.unwrap(), 1);
        assert_eq!(w.state.get(), ModelState::Ready);
        assert!(!w.stuck.load(Ordering::Acquire));

        // A dead session (marked directly) carries the runtime error.
        let w = watch();
        w.mark(
            Duration::from_secs(3),
            Some(&anyhow::anyhow!("session thread exited")),
        );
        let ModelState::Failed(msg) = w.state.get() else {
            panic!("expected Failed");
        };
        assert!(msg.contains("session thread exited"), "{msg}");
        assert!(!session_dead(&anyhow::anyhow!("x")));
    }

    #[test]
    fn slow_compile_warns_and_shows_minutes() {
        let state = StateHandle::new();
        state.set_initializing(format!("{COMPILING_ON} openvino:gpu"));
        let slow = SlowCompile {
            after: Duration::from_millis(40),
            tick: Duration::from_millis(5),
        };
        // Fast compile: no warning, detail untouched.
        let (v, warned) = watch_slow_compile("m", "openvino:gpu", &state, slow, || 3);
        assert_eq!((v, warned), (3, 0));
        assert_eq!(state.detail().as_deref(), Some("compiling on openvino:gpu"));
        // Slow compile: the first warning after `after`, repeated at most every `after`.
        let t = Instant::now();
        let (_, warned) = watch_slow_compile("m", "openvino:gpu", &state, slow, || {
            std::thread::sleep(Duration::from_millis(150));
        });
        // The sleep can overrun on a loaded CI runner, so bound the count by the time that
        // actually passed: one warning per `after`, plus one for the first.
        let max = u32::try_from(t.elapsed().as_millis() / 40).unwrap_or(u32::MAX) + 1;
        assert!(
            (1..=max).contains(&warned),
            "{warned} warnings in {:?}",
            t.elapsed()
        );
        let d = state.detail().unwrap();
        assert!(d.starts_with(COMPILING_ON), "{d}");
        assert_eq!(d, "compiling on openvino:gpu for 0 min");
        assert_eq!(
            compiling_for("ort:cuda", Duration::from_secs(7 * 60 + 5)),
            "compiling on ort:cuda for 7 min"
        );
    }
}
