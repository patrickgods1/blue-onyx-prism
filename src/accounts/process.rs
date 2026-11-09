//! [`Auth`]: the sign-in state of one process, made once at start (`runner`) and shared by every
//! generation's `AppState`, so a restart from the Config page signs nobody out and forgives no
//! failure. It holds the accounts as last read ([`Cached`]), the sessions, the device cookies and
//! the failures of names that have no account, the sign-in attempts still being checked
//! ([`Attempts`]) and the data root all of it lives in (fixed at start: a restart that changes
//! `download_dir` does not move the accounts; a process restart does).
//!
//! [`Auth::spawn_flusher`] writes what waits (moved `last_active`, expired entries) every
//! [`FLUSH_EVERY_SECS`] and once more, forced, when it is stopped at shutdown.

use super::devices::Devices;
use super::lockout::UnknownNames;
use super::lockout::{self, Decision};
use super::sessions::{FLUSH_EVERY_SECS, Sessions};
use super::{Cached, Snapshot};
use crate::sync::LockExt;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

/// The sign-in state of the process (see the module docs).
pub struct Auth {
    /// `users.json` as last read.
    pub users: Cached,
    /// Web UI sessions.
    pub sessions: Arc<Sessions>,
    /// Device cookies.
    pub devices: Arc<Devices>,
    /// Failures of names with no account.
    pub unknown: Mutex<UnknownNames>,
    /// Sign-in attempts decided and not yet recorded, per name.
    pub attempts: Attempts,
    /// The data root everything above was loaded from and is written to.
    data_root: PathBuf,
}

impl std::fmt::Debug for Auth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Unknown names are what someone typed: left out.
        f.debug_struct("Auth")
            .field("users", &self.users)
            .field("sessions", &self.sessions)
            .field("devices", &self.devices)
            .finish_non_exhaustive()
    }
}

impl Auth {
    /// The state saved under `data_root` at `now` (reads only).
    pub fn load(data_root: &Path, now: u64) -> Self {
        Self {
            users: Cached::new(data_root),
            sessions: Arc::new(Sessions::load(data_root, now)),
            devices: Arc::new(Devices::load(data_root, now)),
            unknown: Mutex::new(UnknownNames::new()),
            attempts: Attempts::default(),
            data_root: data_root.to_path_buf(),
        }
    }

    /// The data root of the accounts, sessions, devices, setup code and API token this process
    /// uses: the one at start (see the module docs).
    pub fn data_root(&self) -> &Path {
        &self.data_root
    }

    /// Drop the sessions the accounts no longer allow (when they read), then flush sessions and
    /// devices (see [`Sessions::flush`]).
    pub fn flush(&self, now: u64, force: bool) {
        if let Snapshot::Users(users) = &*self.users.get() {
            self.sessions.prune_users(users, now);
        }
        self.sessions.flush(now, force);
        self.devices.flush(now, force);
    }

    /// A thread that calls [`Self::flush`] every [`FLUSH_EVERY_SECS`] until `stop`, then once
    /// more, forced.
    pub fn spawn_flusher(
        self: &Arc<Self>,
        stop: CancellationToken,
    ) -> std::io::Result<std::thread::JoinHandle<()>> {
        self.spawn_flusher_every(stop, Duration::from_secs(FLUSH_EVERY_SECS))
    }

    fn spawn_flusher_every(
        self: &Arc<Self>,
        stop: CancellationToken,
        every: Duration,
    ) -> std::io::Result<std::thread::JoinHandle<()>> {
        let me = Arc::clone(self);
        std::thread::Builder::new()
            .name("auth-flush".into())
            .spawn(move || {
                while nap(every, &stop) {
                    me.flush(super::now(), false);
                }
                me.flush(super::now(), true);
            })
    }
}

/// Sign-in attempts in flight per name (the account's normalized name, or the key of a name
/// with no account): decided as allowed, being checked, and not yet recorded. The lockout decision
/// counts them as failures already, so a burst of concurrent guesses (from many addresses) is
/// judged as if they came one at a time: below [`lockout::DELAY_AFTER`] they may run side by side;
/// from there on, while one is in flight, the next waits as if it had failed just now, so at most
/// one guess runs per delay window; and the account can never pass [`lockout::DISABLE_AT`].
#[derive(Debug, Clone, Default)]
pub struct Attempts {
    map: Arc<Mutex<HashMap<String, u32>>>,
}

/// One attempt's place in [`Attempts`]; dropping it (once the result is recorded) frees it.
#[derive(Debug)]
#[must_use]
pub struct AttemptSlot {
    map: Arc<Mutex<HashMap<String, u32>>>,
    key: String,
}

impl Drop for AttemptSlot {
    fn drop(&mut self) {
        let mut map = self.map.lock_recover();
        if let Some(n) = map.get_mut(&self.key) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                map.remove(&self.key);
            }
        }
    }
}

impl Attempts {
    /// Decide an attempt on `key` at `now` and, when it may run, take its slot. `base` runs under
    /// the same lock (so a result recorded meanwhile is either in what it reads or still counted
    /// in flight) and returns the recorded state and whether the account's wait is skipped (a
    /// trusted device; never a disabled account).
    pub fn reserve(
        &self,
        key: &str,
        now: u64,
        base: impl FnOnce() -> (lockout::State, bool),
    ) -> Result<AttemptSlot, Decision> {
        let mut map = self.map.lock_recover();
        let (state, skip_wait) = base();
        let in_flight = map.get(key).copied().unwrap_or(0);
        let decision = lockout::decide(&effective(&state, in_flight, now), now);
        match decision {
            Decision::Allow => {}
            Decision::Wait(_) if skip_wait => {}
            refused => return Err(refused),
        }
        *map.entry(key.to_string()).or_insert(0) += 1;
        Ok(AttemptSlot {
            map: Arc::clone(&self.map),
            key: key.to_string(),
        })
    }

    /// Attempts in flight on `key`.
    pub fn in_flight(&self, key: &str) -> u32 {
        self.map.lock_recover().get(key).copied().unwrap_or(0)
    }

    /// Attempts in flight on every name.
    pub fn total(&self) -> u32 {
        self.map.lock_recover().values().sum()
    }
}

/// `state` with `in_flight` attempts counted as failures made at `now`.
fn effective(state: &lockout::State, in_flight: u32, now: u64) -> lockout::State {
    if in_flight == 0 {
        return *state;
    }
    lockout::State {
        failures: state.failures.saturating_add(in_flight),
        last_failure: Some(now),
        disabled: state.disabled,
    }
}

/// Sleep `total` in short steps; false when `stop` was triggered meanwhile.
fn nap(total: Duration, stop: &CancellationToken) -> bool {
    let step = Duration::from_millis(200);
    let end = Instant::now() + total;
    while Instant::now() < end {
        if stop.is_cancelled() {
            return false;
        }
        std::thread::sleep(step.min(end.saturating_duration_since(Instant::now())));
    }
    !stop.is_cancelled()
}

#[cfg(test)]
mod tests {
    use super::super::sessions::Activity;
    use super::*;

    fn state(failures: u32, last_failure: Option<u64>) -> lockout::State {
        lockout::State {
            failures,
            last_failure,
            disabled: false,
        }
    }

    /// In-flight attempts count as failures made now: side by side below the delay, one at a
    /// time from there, never past the disabling count, and a trusted device skips only the wait.
    #[test]
    fn attempts_in_flight_count_as_failures() {
        let a = Attempts::default();
        let now = 1_000_000;
        // Nine recorded: one more may run, then the tenth in flight makes the next wait.
        let first = a
            .reserve("alice", now, || (state(9, Some(now - 5000)), false))
            .unwrap();
        assert_eq!(a.in_flight("alice"), 1);
        assert_eq!(
            a.reserve("alice", now, || (state(9, Some(now - 5000)), false))
                .unwrap_err(),
            Decision::Wait(lockout::delay(10))
        );
        // Below the delay several run at once.
        let few: Vec<_> = (0..5)
            .map(|_| a.reserve("bob", now, || (state(0, None), false)).unwrap())
            .collect();
        assert_eq!(a.in_flight("bob"), 5);
        drop(few);
        assert_eq!(a.in_flight("bob"), 0);
        // A trusted device skips the wait, and still takes a slot.
        let dev = a
            .reserve("alice", now, || (state(9, Some(now - 5000)), true))
            .unwrap();
        assert_eq!(a.in_flight("alice"), 2);
        drop((first, dev));
        // Freed and recorded: the normal rule decides again.
        assert!(
            a.reserve("alice", now, || (state(10, Some(now - 5000)), false))
                .is_ok()
        );
        // 99 recorded and one in flight: the next is refused as disabled, even trusted.
        let last = a.reserve("carol", now, || (state(99, Some(now - 10_000)), false));
        let last = last.unwrap_or_else(|d| panic!("{d:?}"));
        assert_eq!(
            a.reserve("carol", now, || (state(99, Some(now - 10_000)), true))
                .unwrap_err(),
            Decision::Disabled
        );
        drop(last);
        assert_eq!(a.total(), 0, "every slot was given back");
    }

    #[test]
    fn the_flusher_writes_what_waits_and_flushes_at_stop() {
        let d = std::env::temp_dir().join(format!("bo_authproc_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        let now = super::super::now();
        let auth = Arc::new(Auth::load(&d, now));
        let id = auth.sessions.create("alice", 1, now);
        let users = super::super::Users {
            format: super::super::FORMAT,
            users: vec![],
        };
        // No accounts file: the flusher leaves the session to `check`, which ends it.
        let stop = CancellationToken::new();
        let h = auth
            .spawn_flusher_every(stop.clone(), Duration::from_millis(20))
            .unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(auth.sessions.len(), 1);
        assert!(
            auth.sessions
                .check(id.reveal().as_bytes(), &users, now, Activity::User)
                .is_none()
        );
        stop.cancel();
        h.join().unwrap();
        // The forced flush at stop wrote the removal.
        assert!(Sessions::load(&d, now).is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }
}
