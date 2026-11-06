//! Dropping a backend that cannot be dropped here: a warm-up that never answered, or the engine
//! of a stuck model. Freeing either waits for the device (OpenVINO cancels and waits, ONNX
//! Runtime waits for its session threads), which may take as long as the device takes, and a
//! worker thread must never wait for that. A detached reaper thread does the waiting, so the
//! memory and, with ONNX Runtime, the session threads go as soon as the hang clears instead of
//! never. Reapers are bounded ([`MAX_REAPERS`]): a device that never answers holds its reapers
//! for good, and past the bound a backend is leaked with a warning.

use crate::sync::LockExt;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Most reaper threads alive at once.
pub(crate) const MAX_REAPERS: usize = 4;

/// The reapers of the process.
static REAPERS: Reapers = Reapers::new(MAX_REAPERS);

/// Free `what` on a reaper thread; never blocks the caller. See the module docs.
pub(super) fn reap<T: Send + 'static>(model: &str, what: T) {
    REAPERS.hand_off(model, what);
}

/// Wait until no reaper is dropping anything, at most `wait` (a device that never answers keeps
/// one busy for good). True when they all finished.
pub(crate) fn wait_for_reapers(wait: Duration) -> bool {
    REAPERS.wait_idle(wait)
}

/// What happened to a value handed to [`Reapers::hand_off`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Handed {
    /// A reaper thread drops it.
    Reaper,
    /// Every reaper is busy or none could start: leaked, never dropped.
    Leaked,
}

/// A bounded set of detached threads that drop values.
#[derive(Debug)]
pub(super) struct Reapers {
    running: Mutex<usize>,
    idle: Condvar,
    limit: usize,
}

/// Takes one reaper off the count when its thread ends, a panicking drop included.
struct Finished(&'static Reapers);

impl Drop for Finished {
    fn drop(&mut self) {
        let mut n = self.0.running.lock_recover();
        *n = n.saturating_sub(1);
        self.0.idle.notify_all();
    }
}

impl Reapers {
    pub(super) const fn new(limit: usize) -> Self {
        Self {
            running: Mutex::new(0),
            idle: Condvar::new(),
            limit,
        }
    }

    fn take_place(&self) -> bool {
        let mut n = self.running.lock_recover();
        if *n >= self.limit {
            return false;
        }
        *n += 1;
        true
    }

    fn give_back(&self) {
        let mut n = self.running.lock_recover();
        *n = n.saturating_sub(1);
        self.idle.notify_all();
    }

    pub(super) fn hand_off<T: Send + 'static>(&'static self, model: &str, what: T) -> Handed {
        if !self.take_place() {
            warn!(
                model = %model,
                reapers = self.limit,
                "a hung device already holds every thread that frees backends; leaking this \
                 one, so its memory stays in use; restart the service if GPU memory runs short"
            );
            std::mem::forget(what);
            return Handed::Leaked;
        }
        // Behind a shared cell: a thread that cannot start drops its closure right here, and
        // that drop is the wait this module exists to avoid.
        let cell = Arc::new(Mutex::new(Some(what)));
        let for_thread = cell.clone();
        let label = model.to_string();
        let spawned = std::thread::Builder::new()
            .name(format!("reap-{model}"))
            .spawn(move || {
                let _done = Finished(self);
                let taken = for_thread.lock_recover().take();
                drop(taken);
                debug!(model = %label, "abandoned backend freed");
            });
        match spawned {
            Ok(_detached) => Handed::Reaper,
            Err(e) => {
                self.give_back();
                warn!(
                    model = %model,
                    "could not start a thread to free a backend ({e}); leaking it, so its \
                     memory stays in use; restart the service if GPU memory runs short"
                );
                std::mem::forget(cell.lock_recover().take());
                Handed::Leaked
            }
        }
    }

    /// See [`wait_for_reapers`].
    pub(super) fn wait_idle(&self, wait: Duration) -> bool {
        let deadline = Instant::now() + wait;
        let mut n = self.running.lock_recover();
        while *n > 0 {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            n = self
                .idle
                .wait_timeout(n, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_channel::{Receiver, Sender, bounded};

    const SOON: Duration = Duration::from_secs(10);

    /// A value whose drop blocks until the test says so (a backend on a hung device).
    struct Blocking {
        release: Receiver<()>,
        dropped: Sender<()>,
    }

    impl Drop for Blocking {
        fn drop(&mut self) {
            let _ = self.release.recv_timeout(SOON * 3);
            let _ = self.dropped.send(());
        }
    }

    fn blocking() -> (Blocking, Sender<()>, Receiver<()>) {
        let (release_tx, release) = bounded(1);
        let (dropped, dropped_rx) = bounded(1);
        (Blocking { release, dropped }, release_tx, dropped_rx)
    }

    #[test]
    fn the_caller_does_not_wait_and_the_drop_happens_once_the_hang_clears() {
        static R: Reapers = Reapers::new(2);
        let (value, release, dropped) = blocking();
        // Returning here at all shows the caller did not wait: the drop blocks until the
        // release below.
        assert_eq!(R.hand_off("m", value), Handed::Reaper);
        assert!(dropped.try_recv().is_err());
        assert!(!R.wait_idle(Duration::from_millis(1)));
        release.send(()).unwrap();
        dropped.recv_timeout(SOON).expect("dropped by the reaper");
        assert!(R.wait_idle(SOON));
    }

    #[test]
    fn past_the_bound_a_value_is_leaked_and_never_dropped() {
        static R: Reapers = Reapers::new(2);
        let held: Vec<_> = (0..2).map(|_| blocking()).collect();
        let mut releases = Vec::new();
        let mut drops = Vec::new();
        for (value, release, dropped) in held {
            assert_eq!(R.hand_off("m", value), Handed::Reaper);
            releases.push(release);
            drops.push(dropped);
        }
        let (extra, extra_release, extra_dropped) = blocking();
        assert_eq!(R.hand_off("m", extra), Handed::Leaked);
        // Never dropped, not even once it could finish.
        extra_release.send(()).unwrap();
        for r in &releases {
            r.send(()).unwrap();
        }
        for d in &drops {
            d.recv_timeout(SOON).expect("dropped by a reaper");
        }
        assert!(R.wait_idle(SOON));
        assert!(extra_dropped.try_recv().is_err());
        // The places came back.
        let (again, release, dropped) = blocking();
        assert_eq!(R.hand_off("m", again), Handed::Reaper);
        release.send(()).unwrap();
        dropped.recv_timeout(SOON).unwrap();
        assert!(R.wait_idle(SOON));
    }

    #[test]
    fn a_panicking_drop_gives_its_place_back() {
        struct Panics;
        impl Drop for Panics {
            fn drop(&mut self) {
                panic!("expected: a drop panics");
            }
        }
        static R: Reapers = Reapers::new(1);
        assert_eq!(R.hand_off("m", Panics), Handed::Reaper);
        assert!(R.wait_idle(SOON));
        assert_eq!(R.hand_off("m", 1u8), Handed::Reaper);
        assert!(R.wait_idle(SOON));
    }
}
