//! Dropping a backend that cannot be dropped here: a warm-up that never answered, or the engine
//! of a stuck model. Freeing either waits for the device (OpenVINO cancels and waits, ONNX
//! Runtime waits for its session threads), which may take as long as the device takes, and a
//! worker thread must never wait for that. A detached reaper thread does the waiting, so the
//! memory and, with ONNX Runtime, the session threads go as soon as the hang clears instead of
//! never. Reapers are bounded ([`MAX_REAPERS`]): a device that never answers holds its reapers
//! for good, and past the bound a backend is leaked with a warning.
//!
//! Each reaper records the device it waits on and since when. A reaper still waiting after
//! [`BLOCKED_AFTER`] proves that device's hang has not cleared, so the device is not tried again
//! ([`hung_device`]) until that reaper finishes: the worker's candidate walk skips it (to the
//! next candidate, CPU last, recording it as a better device that may come back) and the
//! supervisor holds its upgrade tries (audit 2026-11, RA-082). Without this a driver that never
//! answers got a new backend per upgrade try (hourly, for ever), every one past the bound leaked.
//! While every reaper is busy no device but a CPU is tried (a new hang there would leak). The
//! device text is the canonical spec (`DeviceInfo::spec`, `Device`'s `Display`).

use crate::backend::spec::{Device, Target};
use crate::sync::LockExt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Most reaper threads alive at once.
pub(crate) const MAX_REAPERS: usize = 4;

/// How long a reaper must have waited before its device counts as hung. Longer than a normal
/// free: ONNX Runtime's abort waits at most `ABORT_TIMEOUT` (10 s), and an OpenVINO cancel of a
/// device that answers returns within a second; so a reaper still waiting after this waits for
/// a device that does not answer. Short enough that the next try (an upgrade comes at least
/// `UPGRADE_FIRST_DELAY`, 5 min, after the warm-up timeout that made the reaper) sees it.
pub(crate) const BLOCKED_AFTER: Duration = Duration::from_secs(30);

/// The reapers of the process.
static REAPERS: Reapers = Reapers::new(MAX_REAPERS, BLOCKED_AFTER);

/// Free `what`, a backend on `device` (canonical spec), on a reaper thread; never blocks the
/// caller. See the module docs.
pub(super) fn reap<T: Send + 'static>(model: &str, device: &str, what: T) {
    REAPERS.hand_off(model, device, what);
}

/// The process's reapers (tests of the worker pass their own, see `TestFaults::reapers`).
pub(super) fn reapers() -> &'static Reapers {
    &REAPERS
}

/// Wait until no reaper is dropping anything, at most `wait` (a device that never answers keeps
/// one busy for good). True when they all finished.
pub(crate) fn wait_for_reapers(wait: Duration) -> bool {
    REAPERS.wait_idle(wait)
}

/// Why `device` must not be tried now (None: it may be): see the module docs.
pub(crate) fn hung_device(device: &Device) -> Option<Hung> {
    REAPERS.hung(device, Instant::now())
}

/// A device not tried again while reapers wait on it ([`hung_device`]). Its text says what
/// happened, the likely cause and what to do (it is the skipped candidate's load error, and
/// contains the phrase `LoadFailure::of` reads as a lost device, so the device stays a better
/// one to try again once the reapers are done).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Hung {
    pub(crate) device: String,
    /// Reapers waiting on `device` past [`BLOCKED_AFTER`] (0: refused only because every
    /// reaper is busy).
    pub(crate) waiting: usize,
    /// How long the oldest of them has waited.
    pub(crate) longest: Duration,
    /// Every reaper is busy.
    pub(crate) saturated: bool,
}

impl std::fmt::Display for Hung {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.waiting > 0 {
            write!(
                f,
                "{} is still hung: {} backend(s) on it have waited {} s to be freed (the device \
                 driver may have hung); not trying it again until they are freed, so no more \
                 memory is lost to it; restart the service, or the machine if that does not \
                 help, to recover the device",
                self.device,
                self.waiting,
                self.longest.as_secs()
            )
        } else {
            write!(
                f,
                "every thread that frees backends ({MAX_REAPERS}) waits for a device that does \
                 not answer (the device driver may have hung); not trying {} until one is free, \
                 so a new hang cannot leak a backend; restart the service, or the machine if \
                 that does not help, to recover the devices",
                self.device
            )
        }
    }
}

impl std::error::Error for Hung {}

/// What happened to a value handed to [`Reapers::hand_off`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Handed {
    /// A reaper thread drops it.
    Reaper,
    /// Every reaper is busy or none could start: leaked, never dropped.
    Leaked,
}

/// What one running reaper waits on.
#[derive(Debug)]
struct Job {
    id: u64,
    device: String,
    since: Instant,
}

/// A bounded set of detached threads that drop values.
#[derive(Debug)]
pub(crate) struct Reapers {
    running: Mutex<Vec<Job>>,
    idle: Condvar,
    limit: usize,
    /// See [`BLOCKED_AFTER`].
    blocked_after: Duration,
    next_id: AtomicU64,
}

/// Takes one reaper off the list when its thread ends, a panicking drop included: from then on
/// its device may be tried again.
struct Finished(&'static Reapers, u64);

impl Drop for Finished {
    fn drop(&mut self) {
        self.0.give_back(self.1);
    }
}

impl Reapers {
    pub(super) const fn new(limit: usize, blocked_after: Duration) -> Self {
        Self {
            running: Mutex::new(Vec::new()),
            idle: Condvar::new(),
            limit,
            blocked_after,
            next_id: AtomicU64::new(0),
        }
    }

    /// A place for a reaper of `device`, by id; None when every place is taken.
    fn take_place(&self, device: &str) -> Option<u64> {
        let mut jobs = self.running.lock_recover();
        if jobs.len() >= self.limit {
            return None;
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        jobs.push(Job {
            id,
            device: device.to_string(),
            since: Instant::now(),
        });
        Some(id)
    }

    fn give_back(&self, id: u64) {
        let mut jobs = self.running.lock_recover();
        jobs.retain(|j| j.id != id);
        self.idle.notify_all();
    }

    /// See [`hung_device`]; `now` is the clock.
    pub(super) fn hung(&self, device: &Device, now: Instant) -> Option<Hung> {
        let spec = device.to_string();
        let jobs = self.running.lock_recover();
        let saturated = jobs.len() >= self.limit;
        let waits: Vec<Duration> = jobs
            .iter()
            .filter(|j| j.device == spec)
            .map(|j| now.saturating_duration_since(j.since))
            .filter(|w| *w >= self.blocked_after)
            .collect();
        drop(jobs);
        // A CPU is the last fallback and does not hang the way a GPU driver does: refusing it
        // would take a model down that can serve.
        if device.target == Target::Cpu || (waits.is_empty() && !saturated) {
            return None;
        }
        Some(Hung {
            device: spec,
            waiting: waits.len(),
            longest: waits.iter().copied().max().unwrap_or_default(),
            saturated,
        })
    }

    pub(super) fn hand_off<T: Send + 'static>(
        &'static self,
        model: &str,
        device: &str,
        what: T,
    ) -> Handed {
        let Some(id) = self.take_place(device) else {
            warn!(
                model = %model,
                device = %device,
                reapers = self.limit,
                "a hung device already holds every thread that frees backends; leaking this \
                 one, so its memory stays in use; restart the service if GPU memory runs short"
            );
            std::mem::forget(what);
            return Handed::Leaked;
        };
        // Behind a shared cell: a thread that cannot start drops its closure right here, and
        // that drop is the wait this module exists to avoid.
        let cell = Arc::new(Mutex::new(Some(what)));
        let for_thread = cell.clone();
        let label = model.to_string();
        let spawned = std::thread::Builder::new()
            .name(format!("reap-{model}"))
            .spawn(move || {
                let done = Finished(self, id);
                let taken = for_thread.lock_recover().take();
                drop(taken);
                drop(done);
                debug!(model = %label, "abandoned backend freed");
                crate::backend::memory::trim_soon("abandoned backend freed");
            });
        match spawned {
            Ok(_detached) => Handed::Reaper,
            Err(e) => {
                self.give_back(id);
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
        let mut jobs = self.running.lock_recover();
        while !jobs.is_empty() {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return false;
            }
            jobs = self
                .idle
                .wait_timeout(jobs, left)
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
        static R: Reapers = Reapers::new(2, Duration::ZERO);
        let (value, release, dropped) = blocking();
        // Returning here at all shows the caller did not wait: the drop blocks until the
        // release below.
        assert_eq!(R.hand_off("m", "openvino:gpu", value), Handed::Reaper);
        assert!(dropped.try_recv().is_err());
        assert!(!R.wait_idle(Duration::from_millis(1)));
        release.send(()).unwrap();
        dropped.recv_timeout(SOON).expect("dropped by the reaper");
        assert!(R.wait_idle(SOON));
    }

    #[test]
    fn past_the_bound_a_value_is_leaked_and_never_dropped() {
        static R: Reapers = Reapers::new(2, Duration::ZERO);
        let held: Vec<_> = (0..2).map(|_| blocking()).collect();
        let mut releases = Vec::new();
        let mut drops = Vec::new();
        for (value, release, dropped) in held {
            assert_eq!(R.hand_off("m", "openvino:gpu", value), Handed::Reaper);
            releases.push(release);
            drops.push(dropped);
        }
        let (extra, extra_release, extra_dropped) = blocking();
        assert_eq!(R.hand_off("m", "openvino:gpu", extra), Handed::Leaked);
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
        assert_eq!(R.hand_off("m", "openvino:gpu", again), Handed::Reaper);
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
        static R: Reapers = Reapers::new(1, Duration::ZERO);
        assert_eq!(R.hand_off("m", "openvino:gpu", Panics), Handed::Reaper);
        assert!(R.wait_idle(SOON));
        assert_eq!(R.hand_off("m", "openvino:gpu", 1u8), Handed::Reaper);
        assert!(R.wait_idle(SOON));
    }

    fn dev(spec: &str) -> Device {
        *crate::backend::spec::parse(spec).unwrap().device().unwrap()
    }

    /// RA-082: a reaper still waiting on a device keeps that device from being tried again,
    /// until it finishes; another device and the CPU stay retryable.
    #[test]
    fn a_blocked_reaper_holds_its_device_until_it_finishes() {
        static R: Reapers = Reapers::new(4, Duration::ZERO);
        let gpu = dev("openvino:gpu");
        let now = Instant::now;
        assert_eq!(R.hung(&gpu, now()), None);
        let (value, release, dropped) = blocking();
        assert_eq!(R.hand_off("m", "openvino:gpu", value), Handed::Reaper);
        let hung = R.hung(&gpu, now()).expect("held while its reaper waits");
        assert_eq!(hung.device, "openvino:gpu");
        assert_eq!(hung.waiting, 1);
        assert!(!hung.saturated);
        let text = hung.to_string();
        assert!(text.contains("still hung"), "{text}");
        assert!(text.contains("restart the service"), "{text}");
        // Read as a lost device: it stays a better device to try once the reaper is done.
        assert_eq!(
            crate::backend::plan::LoadFailure::of(&anyhow::Error::new(hung)),
            crate::backend::plan::LoadFailure::DeviceLost
        );
        assert_eq!(R.hung(&dev("openvino:gpu.1"), now()), None);
        assert_eq!(R.hung(&dev("ort:cuda"), now()), None);
        assert_eq!(R.hung(&dev("openvino:cpu"), now()), None);
        release.send(()).unwrap();
        dropped.recv_timeout(SOON).expect("dropped by the reaper");
        assert!(R.wait_idle(SOON));
        assert_eq!(R.hung(&gpu, now()), None, "retryable again once freed");
    }

    /// Only a reaper that waited past `blocked_after` counts: a normal free (a stopped warm-up,
    /// an abort that returns) never keeps the device from the next load.
    #[test]
    fn a_reaper_counts_as_blocked_only_after_its_grace() {
        static R: Reapers = Reapers::new(4, Duration::from_secs(30));
        let gpu = dev("openvino:gpu");
        let (value, release, dropped) = blocking();
        let t0 = Instant::now();
        assert_eq!(R.hand_off("m", "openvino:gpu", value), Handed::Reaper);
        assert_eq!(R.hung(&gpu, t0), None);
        let later = t0 + Duration::from_secs(60);
        let hung = R.hung(&gpu, later).expect("waited past the grace");
        assert_eq!(hung.waiting, 1);
        assert!(hung.longest >= Duration::from_secs(29), "{hung:?}");
        release.send(()).unwrap();
        dropped.recv_timeout(SOON).unwrap();
        assert!(R.wait_idle(SOON));
        assert_eq!(R.hung(&gpu, later), None);
    }

    /// Every reaper busy: no device but a CPU is tried, since a new hang there would leak.
    #[test]
    fn with_every_reaper_busy_only_a_cpu_is_tried() {
        static R: Reapers = Reapers::new(1, Duration::from_secs(3600));
        let (value, release, dropped) = blocking();
        assert_eq!(R.hand_off("m", "openvino:gpu", value), Handed::Reaper);
        let other = R
            .hung(&dev("ort:cuda:1"), Instant::now())
            .expect("saturated");
        assert!(other.saturated);
        assert_eq!(other.waiting, 0);
        assert!(other.to_string().contains("every thread"), "{other}");
        assert_eq!(R.hung(&dev("openvino:cpu"), Instant::now()), None);
        assert_eq!(R.hung(&dev("ort:cpu"), Instant::now()), None);
        release.send(()).unwrap();
        dropped.recv_timeout(SOON).unwrap();
        assert!(R.wait_idle(SOON));
        assert_eq!(R.hung(&dev("ort:cuda:1"), Instant::now()), None);
    }
}
