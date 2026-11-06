//! Automatic recovery of models that failed while serving.
//!
//! A model whose inference got stuck or whose runtime session died is marked `Failed` by its
//! worker ([`crate::worker`], `collect_watched` / [`crate::backend::SessionDead`]). Without
//! this module it stayed that way until an operator saved the config or restarted the service,
//! which is how the upstream blue-onyx service "stopped detecting" after days of uptime. Each
//! generation runs one supervisor thread ([`spawn`]) that polls the workers and reloads such a
//! model through [`ModelRegistry::recover_model`] (a fresh worker on the model's planned
//! devices; the stuck thread is left behind, see `registry::retire_worker`).
//!
//! The reload waits for a backoff that doubles with every consecutive failure, from
//! [`FIRST_DELAY`] to at most [`MAX_DELAY`] (a hung GPU driver must not pile up a new stuck
//! thread every few seconds), and the count resets once the model has served for [`STABLE`].
//! An automatic reload that loads on no device goes on the same schedule: the device may come
//! back, and the model has shown it can load. Other models that failed to *load* are not retried
//! here: their file, family or runtime needs fixing, and the resource watcher reloads them when
//! a download lands. A config change wins over all of this: it uses the current config, it is
//! not an automatic reload, and a disabled model leaves the list.
//!
//! The second duty: a model serving on a fallback because a better candidate had no memory or
//! lost its device ([`crate::backend::plan::Better`]) gets that candidate tried again with a long
//! backoff ([`UPGRADE_FIRST_DELAY`] to [`UPGRADE_MAX_DELAY`]; each try costs a compile), beside
//! the worker serving, which it replaces only if it loads. No try starts during a benchmark.
//! The training queue is not checked: nothing outside `server::training` can see it.
//!
//! The decision logic is the pure [`Recovery`] and [`Upgrades`]; the thread only feeds them the
//! workers' states.
//!
//! [`ModelRegistry::recover_model`]: crate::registry::ModelRegistry::recover_model

use crate::backend::plan::Better;
use crate::server::AppState;
use crate::startup::ModelState;
use crate::sync::LockExt;
use crate::worker::WorkerHandle;
use std::collections::HashMap;
use std::sync::Weak;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

/// Delay before the first automatic reload of a model that failed while serving.
pub const FIRST_DELAY: Duration = Duration::from_secs(10);
/// Longest delay between two automatic reloads of the same model.
pub const MAX_DELAY: Duration = Duration::from_secs(60 * 60);
/// A model that has served this long since its last automatic reload starts over at
/// [`FIRST_DELAY`] when it fails again.
pub const STABLE: Duration = Duration::from_secs(10 * 60);
/// Delay before the first new try of a better device that had no memory or was lost. Long:
/// memory held by something else rarely frees within minutes, and each try costs a compile.
pub const UPGRADE_FIRST_DELAY: Duration = Duration::from_secs(5 * 60);
/// Longest delay between two tries of a better device.
pub const UPGRADE_MAX_DELAY: Duration = Duration::from_secs(60 * 60);
/// How often the supervisor looks at the workers.
const POLL: Duration = Duration::from_millis(500);

/// What the supervisor needs to know about a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Serving.
    Ready,
    /// Marked `Failed` after it had loaded: a stuck inference or a dead session.
    Stuck,
    /// An automatic reload that loaded on no device: the model is still down.
    ReloadFailed,
    /// Loading, or failed to load (not the supervisor's business).
    Other,
}

impl Status {
    /// The status of `w` now.
    pub fn of(w: &WorkerHandle) -> Self {
        match w.state.get() {
            ModelState::Ready => Status::Ready,
            ModelState::Failed(_) if w.is_stuck() => Status::Stuck,
            ModelState::Failed(_) if w.is_recovery() => Status::ReloadFailed,
            _ => Status::Other,
        }
    }
}

/// One model's reload history.
#[derive(Debug, Clone, Copy)]
struct Attempts {
    /// Automatic reloads done so far in this failure streak.
    count: u32,
    /// Earliest time of the next reload.
    next: Instant,
    /// When the last reload was done (`None`: the model is waiting for its first one).
    last_reload: Option<Instant>,
}

/// A reload the supervisor decided on (see [`Recovery::due`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reload {
    pub name: String,
    /// 1 for the first automatic reload of this failure streak.
    pub attempt: u32,
    /// Delay before the next reload, should the model fail again.
    pub next_delay: Duration,
}

/// The pure reload schedule: feed it every worker's [`Status`] and it says which models to
/// reload now.
#[derive(Debug, Default)]
pub struct Recovery {
    models: HashMap<String, Attempts>,
}

impl Recovery {
    /// The delay before automatic reload number `attempt` (1-based): [`FIRST_DELAY`] doubled
    /// per earlier attempt, at most [`MAX_DELAY`].
    pub fn delay(attempt: u32) -> Duration {
        let doublings = attempt.saturating_sub(1).min(16);
        FIRST_DELAY
            .checked_mul(1u32 << doublings)
            .unwrap_or(MAX_DELAY)
            .min(MAX_DELAY)
    }

    /// Update the schedule with the workers' states at `now` and return the models to reload
    /// now, in the order given. Models no longer listed are forgotten.
    pub fn due<'a>(
        &mut self,
        now: Instant,
        workers: impl IntoIterator<Item = (&'a str, Status)>,
    ) -> Vec<Reload> {
        let mut seen: Vec<&str> = Vec::new();
        let mut out = Vec::new();
        for (name, status) in workers {
            seen.push(name);
            match status {
                Status::Stuck | Status::ReloadFailed => {
                    let a = self.models.entry(name.to_string()).or_insert(Attempts {
                        count: 0,
                        next: now + Self::delay(1),
                        last_reload: None,
                    });
                    if now < a.next {
                        continue;
                    }
                    a.count = a.count.saturating_add(1);
                    a.last_reload = Some(now);
                    let next_delay = Self::delay(a.count.saturating_add(1));
                    a.next = now + next_delay;
                    out.push(Reload {
                        name: name.to_string(),
                        attempt: a.count,
                        next_delay,
                    });
                }
                Status::Ready => {
                    let stable = self.models.get(name).is_some_and(|a| {
                        a.last_reload
                            .is_some_and(|t| now.saturating_duration_since(t) >= STABLE)
                    });
                    if stable {
                        self.models.remove(name);
                    }
                }
                Status::Other => {}
            }
        }
        self.models.retain(|n, _| seen.contains(&n.as_str()));
        out
    }

    /// Models with a reload history (for tests and diagnostics).
    pub fn tracked(&self) -> usize {
        self.models.len()
    }
}

/// One model's tries of its better device.
#[derive(Debug, Clone, Copy)]
struct Tries {
    /// The device being tried (another one starts over).
    device: crate::backend::spec::Device,
    /// Tries done so far.
    count: u32,
    /// Earliest time of the next try.
    next: Instant,
}

/// A new try of a better device the supervisor decided on (see [`Upgrades::due`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Upgrade {
    pub name: String,
    pub better: Better,
    /// 1 for the first try.
    pub attempt: u32,
    /// Delay before the next try, should this one fail.
    pub next_delay: Duration,
}

/// The pure schedule of [`Upgrade`]s: feed it the serving models that have a better device, and
/// it says which to try now.
#[derive(Debug, Default)]
pub struct Upgrades {
    models: HashMap<String, Tries>,
}

impl Upgrades {
    /// The delay before try number `attempt` (1-based): [`UPGRADE_FIRST_DELAY`] doubled per
    /// earlier try, at most [`UPGRADE_MAX_DELAY`].
    pub fn delay(attempt: u32) -> Duration {
        let doublings = attempt.saturating_sub(1).min(16);
        UPGRADE_FIRST_DELAY
            .checked_mul(1u32 << doublings)
            .unwrap_or(UPGRADE_MAX_DELAY)
            .min(UPGRADE_MAX_DELAY)
    }

    /// Update the schedule at `now` with the models listed and return those to try now, in the
    /// order given. A model no longer listed (it moved, stopped, or tries ended) is forgotten;
    /// one listed with its `trying` flag set has a try under way, which keeps its place in the
    /// schedule. While `busy` (a GPU job runs) nothing starts and the time still passes.
    pub fn due<'a>(
        &mut self,
        now: Instant,
        models: impl IntoIterator<Item = (&'a str, &'a Better, bool)>,
        busy: bool,
    ) -> Vec<Upgrade> {
        let mut seen: Vec<&str> = Vec::new();
        let mut out = Vec::new();
        for (name, better, trying) in models {
            seen.push(name);
            let device = better.candidate.device;
            let a = self.models.entry(name.to_string()).or_insert(Tries {
                device,
                count: 0,
                next: now + Self::delay(1),
            });
            if a.device != device {
                *a = Tries {
                    device,
                    count: 0,
                    next: now + Self::delay(1),
                };
            }
            if busy || trying || now < a.next {
                continue;
            }
            a.count = a.count.saturating_add(1);
            let next_delay = Self::delay(a.count.saturating_add(1));
            a.next = now + next_delay;
            out.push(Upgrade {
                name: name.to_string(),
                better: better.clone(),
                attempt: a.count,
                next_delay,
            });
        }
        self.models.retain(|n, _| seen.contains(&n.as_str()));
        out
    }

    /// Models on the schedule (for tests and diagnostics).
    pub fn tracked(&self) -> usize {
        self.models.len()
    }
}

/// Start the supervisor thread of a generation: it polls the generation's workers until
/// `generation` is cancelled (or the state is gone), reloads models that failed while serving on
/// the schedule of [`Recovery`], and tries better devices on the schedule of [`Upgrades`].
pub fn spawn(state: Weak<AppState>, generation: CancellationToken) -> Option<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("supervisor".into())
        .spawn(move || {
            let mut recovery = Recovery::default();
            let mut upgrades = Upgrades::default();
            loop {
                if generation.is_cancelled() {
                    return;
                }
                std::thread::sleep(POLL);
                if generation.is_cancelled() {
                    return;
                }
                let Some(state) = state.upgrade() else {
                    return;
                };
                let workers = state.registry.workers();
                let mut statuses: Vec<(String, Status)> = Vec::new();
                let mut better: Vec<(String, Better, bool)> = Vec::new();
                for w in &workers {
                    let status = Status::of(w);
                    let replacing = state.registry.is_replacing(&w.name);
                    if status == Status::Ready
                        && let Some(b) = w.better()
                    {
                        better.push((w.name.clone(), b, replacing));
                    }
                    // A model with a replacement on the way waits for its outcome.
                    let status = if replacing { Status::Other } else { status };
                    statuses.push((w.name.clone(), status));
                }
                drop(workers);
                let now = Instant::now();
                let due = recovery.due(now, statuses.iter().map(|(n, s)| (n.as_str(), *s)));
                for r in due {
                    recover(&state, &r);
                }
                let tries = upgrades.due(
                    now,
                    better.iter().map(|(n, b, t)| (n.as_str(), b, *t)),
                    state.benchmark.is_running(),
                );
                for u in tries {
                    try_better(&state, &u);
                }
            }
        })
        .map_err(|e| warn!("could not spawn the model supervisor: {e}"))
        .ok()
}

/// Do the automatic reload `r`.
fn recover(state: &AppState, r: &Reload) {
    let why = state
        .registry
        .by_name(&r.name)
        .and_then(|w| match w.state.get() {
            ModelState::Failed(m) => Some(m),
            _ => None,
        })
        .unwrap_or_default();
    error!(
        model = %r.name,
        attempt = r.attempt,
        next_retry_secs = r.next_delay.as_secs(),
        "model failed while serving ({why}); reloading it automatically"
    );
    // Under the lock of config changes, so this cannot act on a config one is replacing.
    let done = {
        let _one = state.apply_lock.lock_recover();
        state.registry.recover_model(&r.name, &state.running())
    };
    match done {
        Ok(_) => info!(model = %r.name, "automatic reload started"),
        Err(e) => warn!(
            model = %r.name,
            "automatic reload did not start: {e:#}; next try in {} s",
            r.next_delay.as_secs()
        ),
    }
}

/// Start the try `u` of a better device.
fn try_better(state: &AppState, u: &Upgrade) {
    let device = u.better.candidate.device;
    info!(
        model = %u.name,
        device = %device,
        attempt = u.attempt,
        next_retry_secs = u.next_delay.as_secs(),
        "trying the better device again ({}), beside the worker serving",
        u.better.reason
    );
    let started = {
        let _one = state.apply_lock.lock_recover();
        state
            .registry
            .retry_better(&u.name, &state.running(), &u.better)
    };
    if let Err(e) = started {
        warn!(model = %u.name, device = %device, "could not try the better device: {e:#}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delays_double_up_to_the_cap() {
        assert_eq!(Recovery::delay(0), FIRST_DELAY);
        assert_eq!(Recovery::delay(1), FIRST_DELAY);
        assert_eq!(Recovery::delay(2), FIRST_DELAY * 2);
        assert_eq!(Recovery::delay(3), FIRST_DELAY * 4);
        assert_eq!(Recovery::delay(9), FIRST_DELAY * 256);
        assert_eq!(Recovery::delay(10), MAX_DELAY);
        assert_eq!(Recovery::delay(40), MAX_DELAY);
        assert_eq!(Recovery::delay(u32::MAX), MAX_DELAY);
    }

    #[test]
    fn stuck_model_reloads_after_the_first_delay_then_backs_off() {
        let t0 = Instant::now();
        let mut r = Recovery::default();
        let stuck = [("a", Status::Stuck), ("b", Status::Ready)];
        // Seen stuck: not yet.
        assert!(r.due(t0, stuck).is_empty());
        assert!(r.due(t0 + FIRST_DELAY / 2, stuck).is_empty());
        // First reload after FIRST_DELAY.
        let d = r.due(t0 + FIRST_DELAY, stuck);
        assert_eq!(
            d,
            vec![Reload {
                name: "a".into(),
                attempt: 1,
                next_delay: FIRST_DELAY * 2
            }]
        );
        // Reloading: nothing to do while it compiles.
        assert!(
            r.due(
                t0 + FIRST_DELAY + Duration::from_secs(1),
                [("a", Status::Other)]
            )
            .is_empty()
        );
        // Stuck again right away: waits for the doubled delay.
        let t1 = t0 + FIRST_DELAY + Duration::from_secs(2);
        assert!(r.due(t1, stuck).is_empty());
        let d = r.due(t0 + FIRST_DELAY * 3, stuck);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].attempt, 2);
        assert_eq!(d[0].next_delay, FIRST_DELAY * 4);
    }

    #[test]
    fn a_stable_model_starts_over() {
        let t0 = Instant::now();
        let mut r = Recovery::default();
        let stuck = [("a", Status::Stuck)];
        r.due(t0, stuck);
        assert_eq!(r.due(t0 + FIRST_DELAY, stuck).len(), 1);
        // Ready, but not for long enough: the streak continues (the next reload still waits
        // for the doubled delay).
        let ready = [("a", Status::Ready)];
        let soon = t0 + FIRST_DELAY + Duration::from_secs(5);
        r.due(soon, ready);
        assert_eq!(r.tracked(), 1);
        let d = r.due(soon, stuck);
        assert!(d.is_empty(), "{d:?}");
        // Served for STABLE after the reload: forgotten, the next failure is attempt 1 again.
        r.due(t0 + FIRST_DELAY + STABLE, ready);
        assert_eq!(r.tracked(), 0);
        r.due(t0 + FIRST_DELAY + STABLE, stuck);
        let d = r.due(t0 + FIRST_DELAY * 2 + STABLE, stuck);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].attempt, 1);
    }

    #[test]
    fn removed_models_are_forgotten_and_load_failures_ignored() {
        let t0 = Instant::now();
        let mut r = Recovery::default();
        r.due(t0, [("a", Status::Stuck), ("b", Status::Other)]);
        assert_eq!(r.tracked(), 1);
        r.due(t0, [("b", Status::Other)]);
        assert_eq!(r.tracked(), 0);
        assert!(r.due(t0 + MAX_DELAY, [("b", Status::Other)]).is_empty());
    }

    #[test]
    fn a_reload_that_loads_nowhere_stays_on_the_backoff() {
        let t0 = Instant::now();
        let mut r = Recovery::default();
        r.due(t0, [("a", Status::Stuck)]);
        assert_eq!(r.due(t0 + FIRST_DELAY, [("a", Status::Stuck)]).len(), 1);
        // The reload ran on every candidate and failed: the delays go on doubling.
        let failed = [("a", Status::ReloadFailed)];
        let t1 = t0 + FIRST_DELAY;
        assert!(r.due(t1 + FIRST_DELAY, failed).is_empty());
        let d = r.due(t1 + FIRST_DELAY * 2, failed);
        assert_eq!(d.len(), 1);
        assert_eq!((d[0].attempt, d[0].next_delay), (2, FIRST_DELAY * 4));
        let t2 = t1 + FIRST_DELAY * 2;
        assert!(r.due(t2 + FIRST_DELAY * 3, failed).is_empty());
        let d = r.due(t2 + FIRST_DELAY * 4, failed);
        assert_eq!((d[0].attempt, d[0].next_delay), (3, FIRST_DELAY * 8));
        // The delay stops growing at the cap, and retries never end.
        let mut t = t2 + FIRST_DELAY * 4;
        for attempt in 4..20 {
            t += MAX_DELAY;
            let d = r.due(t, failed);
            assert_eq!(d.len(), 1, "attempt {attempt}");
            assert_eq!(d[0].attempt, attempt);
            assert!(d[0].next_delay <= MAX_DELAY);
        }
        // Back up: stable service starts over.
        r.due(t + STABLE, [("a", Status::Ready)]);
        assert_eq!(r.tracked(), 0);
    }

    #[test]
    fn a_saved_config_or_a_disable_ends_the_retries() {
        let t0 = Instant::now();
        let mut r = Recovery::default();
        r.due(t0, [("a", Status::Stuck)]);
        assert_eq!(r.due(t0 + FIRST_DELAY, [("a", Status::Stuck)]).len(), 1);
        // The operator saved the config: that worker is no automatic reload, so its failure to
        // load shows as `Other` and nothing retries it.
        assert!(r.due(t0 + MAX_DELAY, [("a", Status::Other)]).is_empty());
        assert!(r.due(t0 + MAX_DELAY * 2, [("a", Status::Other)]).is_empty());
        // Disabled: gone from the list, forgotten.
        r.due(t0 + MAX_DELAY * 3, []);
        assert_eq!(r.tracked(), 0);
    }

    fn better(device: crate::backend::spec::Device) -> Better {
        Better {
            candidate: crate::backend::Candidate {
                device,
                fell_back: false,
                note: None,
            },
            failure: crate::backend::plan::LoadFailure::Memory,
            reason: "CL_OUT_OF_RESOURCES".into(),
        }
    }

    #[test]
    fn upgrade_delays_double_up_to_the_cap() {
        assert_eq!(Upgrades::delay(1), Duration::from_secs(5 * 60));
        assert_eq!(Upgrades::delay(2), Duration::from_secs(10 * 60));
        assert_eq!(Upgrades::delay(3), Duration::from_secs(20 * 60));
        assert_eq!(Upgrades::delay(4), Duration::from_secs(40 * 60));
        assert_eq!(Upgrades::delay(5), UPGRADE_MAX_DELAY);
        assert_eq!(Upgrades::delay(u32::MAX), UPGRADE_MAX_DELAY);
    }

    #[test]
    fn a_better_device_is_tried_on_a_long_backoff() {
        use crate::backend::spec::{Device, Runtime, Target};
        let gpu = Device::new(Runtime::OpenVino, Target::Gpu, None).unwrap();
        let t0 = Instant::now();
        let mut u = Upgrades::default();
        let b = better(gpu);
        let list = [("a", &b, false)];
        assert!(u.due(t0, list, false).is_empty());
        assert!(u.due(t0 + UPGRADE_FIRST_DELAY / 2, list, false).is_empty());
        let t1 = t0 + UPGRADE_FIRST_DELAY;
        let d = u.due(t1, list, false);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].attempt, 1);
        assert_eq!(d[0].better, b);
        assert_eq!(d[0].next_delay, UPGRADE_FIRST_DELAY * 2);
        // Under way: the schedule stays. Failed: the current worker stays and the next try
        // waits for the doubled delay.
        assert!(u.due(t1, [("a", &b, true)], false).is_empty());
        assert_eq!(u.tracked(), 1);
        assert!(u.due(t1 + UPGRADE_FIRST_DELAY, list, false).is_empty());
        let t2 = t1 + UPGRADE_FIRST_DELAY * 2;
        assert_eq!(u.due(t2, list, false)[0].attempt, 2);
        // A GPU job holds the device: nothing starts, and the try comes when it ends.
        let t3 = t2 + UPGRADE_FIRST_DELAY * 4;
        assert!(u.due(t3, list, true).is_empty());
        assert!(u.due(t3 + UPGRADE_MAX_DELAY, list, true).is_empty());
        assert_eq!(u.due(t3 + UPGRADE_MAX_DELAY, list, false)[0].attempt, 3);
        // Moved, or the tries ended: forgotten.
        u.due(t3 + UPGRADE_MAX_DELAY, [], false);
        assert_eq!(u.tracked(), 0);
    }

    #[test]
    fn another_better_device_starts_over() {
        use crate::backend::spec::{Device, Runtime, Target};
        let gpu = better(Device::new(Runtime::OpenVino, Target::Gpu, None).unwrap());
        let other = better(Device::new(Runtime::Ort, Target::Cpu, None).unwrap());
        let t0 = Instant::now();
        let mut u = Upgrades::default();
        let list = [("a", &gpu, false)];
        u.due(t0, list, false);
        assert_eq!(u.due(t0 + UPGRADE_FIRST_DELAY, list, false).len(), 1);
        let t1 = t0 + UPGRADE_FIRST_DELAY;
        let list = [("a", &other, false)];
        assert!(u.due(t1, list, false).is_empty());
        assert!(u.due(t1 + UPGRADE_FIRST_DELAY / 2, list, false).is_empty());
        let d = u.due(t1 + UPGRADE_FIRST_DELAY, list, false);
        assert_eq!(d[0].attempt, 1);
    }

    #[test]
    fn status_of_handles() {
        use crate::metrics::ModelMetrics;
        use std::sync::Arc;
        let m = Arc::new(ModelMetrics::new("x", "CPU", ""));
        let h = WorkerHandle::failed("x", "could not load", m);
        assert_eq!(Status::of(&h), Status::Other);
        h.stuck_flag()
            .store(true, std::sync::atomic::Ordering::Release);
        assert_eq!(Status::of(&h), Status::Stuck);
        h.state.set(ModelState::Ready);
        assert_eq!(Status::of(&h), Status::Ready);
        h.join();
        // A worker of an automatic reload that failed to load is the supervisor's business.
        let mut h = WorkerHandle::failed("y", "every device failed", Arc::new(metric()));
        assert_eq!(Status::of(&h), Status::Other);
        h.set_history(true, None);
        assert_eq!(Status::of(&h), Status::ReloadFailed);
        h.state.set(ModelState::Initializing);
        assert_eq!(Status::of(&h), Status::Other);
    }

    fn metric() -> crate::metrics::ModelMetrics {
        crate::metrics::ModelMetrics::new("y", "CPU", "")
    }
}
