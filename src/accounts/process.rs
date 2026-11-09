//! [`Auth`]: the sign-in state of one process, made once at start (`runner`) and shared by every
//! generation's `AppState`, so a restart from the Config page signs nobody out and forgives no
//! failure. It holds the accounts as last read ([`Cached`]), the sessions, the device cookies and
//! the failures of names that have no account. Nothing reads it over HTTP yet.
//!
//! [`Auth::spawn_flusher`] writes what waits (moved `last_active`, expired entries) every
//! [`FLUSH_EVERY_SECS`] and once more, forced, when it is stopped at shutdown.

use super::devices::Devices;
use super::lockout::UnknownNames;
use super::sessions::{FLUSH_EVERY_SECS, Sessions};
use super::{Cached, Snapshot};
use std::path::Path;
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
        }
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
