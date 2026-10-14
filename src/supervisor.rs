//! Automatic recovery of models that failed while serving.
//!
//! A model whose inference got stuck or whose runtime session died is marked `Failed` by its
//! worker ([`crate::worker`], `collect_watched` / [`crate::backend::SessionDead`]). Without
//! this module it stayed that way until an operator saved the config or restarted the service,
//! which is how the upstream blue-onyx service "stopped detecting" after days of uptime. Each
//! generation runs one supervisor thread ([`spawn`]) that polls the workers and reloads such a
//! model through [`crate::server::reload_models`] (a fresh worker on the model's planned
//! devices; the stuck thread is left behind, see `registry::retire_worker`).
//!
//! The reload waits for a backoff that doubles with every consecutive failure, from
//! [`FIRST_DELAY`] to at most [`MAX_DELAY`] (a hung GPU driver must not pile up a new stuck
//! thread every few seconds), and the count resets once the model has served for [`STABLE`].
//! Models that failed to *load* are not retried here: their file, family or runtime needs
//! fixing, and the resource watcher reloads them when a download lands. The decision logic is
//! the pure [`Recovery`]; the thread only feeds it the workers' states.

use crate::server::AppState;
use crate::startup::ModelState;
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
/// How often the supervisor looks at the workers.
const POLL: Duration = Duration::from_millis(500);

/// What the supervisor needs to know about a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Serving.
    Ready,
    /// Marked `Failed` after it had loaded: a stuck inference or a dead session.
    Stuck,
    /// Loading, or failed to load (not the supervisor's business).
    Other,
}

impl Status {
    /// The status of `w` now.
    pub fn of(w: &WorkerHandle) -> Self {
        match w.state.get() {
            ModelState::Ready => Status::Ready,
            ModelState::Failed(_) if w.is_stuck() => Status::Stuck,
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
                Status::Stuck => {
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

/// Start the supervisor thread of a generation: it polls the generation's workers until
/// `generation` is cancelled (or the state is gone) and reloads models that failed while
/// serving on the schedule of [`Recovery`].
pub fn spawn(state: Weak<AppState>, generation: CancellationToken) -> Option<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("supervisor".into())
        .spawn(move || {
            let mut recovery = Recovery::default();
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
                let statuses: Vec<(String, Status)> = workers
                    .iter()
                    .map(|w| (w.name.clone(), Status::of(w)))
                    .collect();
                drop(workers);
                let due = recovery.due(
                    Instant::now(),
                    statuses.iter().map(|(n, s)| (n.as_str(), *s)),
                );
                for r in due {
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
                    let done = crate::server::reload_models(&state, std::slice::from_ref(&r.name));
                    if done.is_empty() {
                        warn!(model = %r.name, "automatic reload did not start; next try in {} s", r.next_delay.as_secs());
                    } else {
                        info!(model = %r.name, "automatic reload started");
                    }
                }
            }
        })
        .map_err(|e| warn!("could not spawn the model supervisor: {e}"))
        .ok()
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
    }
}
