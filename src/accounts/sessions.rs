//! Web UI sessions: the store behind the `__Host-bop_session` cookie ([`super::cookies`]). Pure
//! apart from its file; every method takes `now` in unix seconds, so tests drive the clock.
//!
//! - **Ids** ([`SessionId`]): 32 random bytes as 64 lowercase hex characters in the cookie; only
//!   their SHA-256 is kept, in memory and in the file, and looked up in a map (why that is
//!   constant-time enough: [`super::opaque`]). A presented value of any other shape is refused
//!   without a lookup.
//! - **Expiry:** a session is valid while `now - last_active <= IDLE_SECS` and
//!   `now - created <= ABSOLUTE_SECS`: at exactly [`IDLE_SECS`] idle it still passes, one second
//!   later it is gone (the same for [`ABSOLUTE_SECS`]). [`Activity::User`] moves `last_active`
//!   to now, [`Activity::Poll`] (a page polling on its own) does not, so an open tab nobody
//!   looks at signs out after an hour, and nothing extends a session past 12 hours.
//! - **Accounts:** [`Sessions::check`] takes the accounts as the caller read them
//!   ([`super::Cached`]). A session ends when its account is gone or disabled, or when the
//!   account's `generation` is no longer the one the session was made with: a password change,
//!   a reset or "sign out everywhere" by the CLI ends the live sessions without ever touching
//!   this file. A session that fails a check is removed.
//! - **Limits:** at most [`MAX_PER_USER`] sessions per account; a new one evicts the oldest.
//! - **The file** `auth/sessions.json` (format [`FORMAT`]): `{"format": 1, "sessions": [{"hash",
//!   "user", "generation", "created", "last_active"}]}`, hashes only, so the file signs nobody
//!   in. Missing = no sessions. Damaged (or of a newer format) = no sessions, a warning that
//!   quotes nothing, and the file left as it is until the next write replaces it: sessions are
//!   not precious, everyone just signs in again. Creating, ending and rotating a session write
//!   it before the call returns (a sign-in must survive a restart right after it); a moved
//!   `last_active` waits for a flush at most every [`FLUSH_EVERY_SECS`] ([`Sessions::flush`], the
//!   process's flusher, see [`super::process`]). A session can therefore outlive its idle limit
//!   across a restart by that much at most. Expired entries are dropped at every flush.
//!   A failed write is logged (throttled); memory stays authoritative and checks go on.

use super::opaque::{self, Digest, Persist, Secret, Versions};
use super::{Users, lockout};
use crate::sync::LockExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

/// The file's name in `auth/`.
pub const FILE_NAME: &str = "sessions.json";
/// The format this version writes and reads.
pub const FORMAT: u32 = 1;
/// A session unused this long (by [`Activity::User`] requests) ends.
pub const IDLE_SECS: u64 = 3600;
/// A session ends this long after it was made, however active.
pub const ABSOLUTE_SECS: u64 = 12 * 3600;
/// Sessions an account may have at once; a new one evicts the oldest.
pub const MAX_PER_USER: usize = 20;
/// A moved `last_active` is written at most this often.
pub const FLUSH_EVERY_SECS: u64 = 120;

/// A session's id as the cookie holds it. No `Display` or `Serialize`; `Debug` hides it.
#[derive(Clone, PartialEq, Eq)]
pub struct SessionId(Secret);

impl std::fmt::Debug for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SessionId(..)")
    }
}

impl SessionId {
    /// A cookie's value as an id, when it has the shape of one (64 lowercase hex characters).
    /// Whether it is a live session is [`Sessions::check`]'s question.
    pub fn from_presented(presented: &[u8]) -> Option<Self> {
        Secret::parse(presented).map(Self)
    }

    /// The value, for the `Set-Cookie` writer ([`super::cookies`]) only.
    pub fn reveal(&self) -> &str {
        self.0.reveal()
    }

    fn digest(&self) -> Digest {
        self.0.digest()
    }
}

/// What is kept of a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// The account's name.
    pub user: String,
    /// The account's `generation` when the session was made.
    pub generation: u64,
    /// Unix seconds.
    pub created: u64,
    /// Unix seconds of the last [`Activity::User`] request.
    pub last_active: u64,
}

impl Record {
    /// Why the clock ended it at `now`, if it did (see the module docs for the boundaries).
    fn timed_out(&self, now: u64) -> Option<&'static str> {
        if now.saturating_sub(self.created) > ABSOLUTE_SECS {
            Some("12 hours since sign-in")
        } else if now.saturating_sub(self.last_active) > IDLE_SECS {
            Some("idle for an hour")
        } else {
            None
        }
    }

    /// Why it is no longer valid at `now` for `users`, if it is not.
    fn ended(&self, users: &Users, now: u64) -> Option<&'static str> {
        if let Some(why) = self.timed_out(now) {
            return Some(why);
        }
        let Some(u) = users.get(&self.user) else {
            return Some("account removed");
        };
        if matches!(
            lockout::decide(&u.lockout(), now),
            lockout::Decision::Disabled
        ) {
            return Some("account disabled");
        }
        (u.generation != self.generation).then_some("password changed or signed out everywhere")
    }
}

/// What kind of request presents the session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activity {
    /// Someone did something: `last_active` moves to now.
    User,
    /// A page polling on its own (`X-Prism-Poll: 1`): `last_active` stays.
    Poll,
}

/// A valid session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Valid {
    /// The account's name.
    pub user: String,
    /// Unix seconds.
    pub created: u64,
    /// Seconds left before [`ABSOLUTE_SECS`] ends it (the cookie's `Max-Age`; 0 in its last
    /// second, so a caller writing a cookie uses at least 1).
    pub expires_in: u64,
}

/// The sessions of the process (see the module docs). Share it in an `Arc`.
pub struct Sessions {
    persist: Persist,
    state: Mutex<State>,
}

struct State {
    map: HashMap<Digest, Record>,
    v: Versions,
}

impl std::fmt::Debug for Sessions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sessions")
            .field("path", &self.persist.path())
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

#[derive(Serialize, Deserialize)]
struct FileRepr {
    format: u32,
    sessions: Vec<Entry>,
}

#[derive(Serialize, Deserialize)]
struct Entry {
    hash: String,
    user: String,
    generation: u64,
    created: u64,
    last_active: u64,
}

#[derive(Deserialize)]
struct Head {
    format: u32,
}

/// The sessions in `bytes`, or why there are none (without quoting it).
fn parse(bytes: &[u8]) -> Result<HashMap<Digest, Record>, String> {
    let head: Head = serde_json::from_slice(bytes).map_err(|e| opaque::json_problem(&e))?;
    if head.format != FORMAT {
        return Err(format!(
            "format {} (this version reads {FORMAT}{})",
            head.format,
            if head.format > FORMAT {
                "; written by a newer version"
            } else {
                ""
            }
        ));
    }
    let file: FileRepr = serde_json::from_slice(bytes).map_err(|e| opaque::json_problem(&e))?;
    let mut map = HashMap::with_capacity(file.sessions.len());
    for (i, e) in file.sessions.into_iter().enumerate() {
        let Some(hash) = opaque::parse_digest_hex(&e.hash) else {
            return Err(format!("session {} has an invalid hash", i + 1));
        };
        if super::normalize_name(&e.user).ok().as_deref() != Some(e.user.as_str()) {
            return Err(format!("session {} has an invalid account name", i + 1));
        }
        map.insert(
            hash,
            Record {
                user: e.user,
                generation: e.generation,
                created: e.created,
                last_active: e.last_active,
            },
        );
    }
    Ok(map)
}

impl Sessions {
    /// The sessions saved under `data_root` (see the module docs for a missing or damaged
    /// file), those expired at `now` left out. Reads only.
    pub fn load(data_root: &Path, now: u64) -> Self {
        let persist = Persist::new(data_root, FILE_NAME, now);
        let path = persist.path();
        let mut map = match opaque::read(&path) {
            opaque::Loaded::Missing => HashMap::new(),
            opaque::Loaded::Failed(why) => {
                tracing::warn!(
                    "{}: {why}; starting with no web UI sessions (everyone signs in again). \
                     Check that the service's account may read it",
                    path.display()
                );
                HashMap::new()
            }
            opaque::Loaded::Bytes(bytes) => parse(&bytes).unwrap_or_else(|why| {
                tracing::warn!(
                    "{} is damaged ({why}); starting with no web UI sessions (everyone signs \
                     in again). The file is replaced at the next sign-in",
                    path.display()
                );
                HashMap::new()
            }),
        };
        map.retain(|_, r| r.timed_out(now).is_none());
        Self {
            persist,
            state: Mutex::new(State {
                map,
                v: Versions::default(),
            }),
        }
    }

    /// The file's path.
    pub fn path(&self) -> std::path::PathBuf {
        self.persist.path()
    }

    /// Start a session for `user` at `generation` (the account's current one): its id for the
    /// cookie. The account's oldest sessions beyond [`MAX_PER_USER`] end. Written before it
    /// returns.
    pub fn create(&self, user: &str, generation: u64, now: u64) -> SessionId {
        let id = SessionId(Secret::generate());
        let key = id.digest();
        {
            let mut s = self.state.lock_recover();
            s.map.insert(
                key,
                Record {
                    user: user.to_string(),
                    generation,
                    created: now,
                    last_active: now,
                },
            );
            let mut mine: Vec<(u64, u64, Digest)> = s
                .map
                .iter()
                .filter(|(k, r)| r.user == user && **k != key)
                .map(|(k, r)| (r.created, r.last_active, *k))
                .collect();
            let over = (mine.len() + 1).saturating_sub(MAX_PER_USER);
            if over > 0 {
                mine.sort_unstable();
                for (_, _, k) in mine.into_iter().take(over) {
                    s.map.remove(&k);
                }
            }
            s.v.force();
        }
        self.flush(now, false);
        id
    }

    /// The session `presented` (a cookie's value) at `now`, judged against `users` (see the
    /// module docs); `None` when it is not one. One that is no longer valid is removed.
    pub fn check(
        &self,
        presented: &[u8],
        users: &Users,
        now: u64,
        activity: Activity,
    ) -> Option<Valid> {
        // Hashed before the lock.
        let key = opaque::digest_of(presented)?;
        let mut s = self.state.lock_recover();
        let record = s.map.get(&key)?;
        if let Some(why) = record.ended(users, now) {
            tracing::debug!(user = %record.user, why, "web UI session ended");
            s.map.remove(&key);
            s.v.lazy();
            return None;
        }
        let valid = Valid {
            user: record.user.clone(),
            created: record.created,
            expires_in: (record.created.saturating_add(ABSOLUTE_SECS)).saturating_sub(now),
        };
        let moves = activity == Activity::User && record.last_active < now;
        if moves {
            if let Some(r) = s.map.get_mut(&key) {
                r.last_active = now;
            }
            s.v.lazy();
        }
        Some(valid)
    }

    /// What is kept of the session `id` (expired or not), for tests and listings.
    pub fn record(&self, id: &SessionId) -> Option<Record> {
        self.state.lock_recover().map.get(&id.digest()).cloned()
    }

    /// End the session `id` (signing out): whether there was one. Written before it returns.
    pub fn end(&self, id: &SessionId, now: u64) -> bool {
        let key = id.digest();
        let ended = {
            let mut s = self.state.lock_recover();
            let ended = s.map.remove(&key).is_some();
            if ended {
                s.v.force();
            }
            ended
        };
        if ended {
            self.flush(now, false);
        }
        ended
    }

    /// End every session of `user` (removed, or signed out everywhere): how many. Written
    /// before it returns.
    pub fn end_user(&self, user: &str, now: u64) -> usize {
        self.end_where(now, |r, _| r.user == user)
    }

    /// End every session of `user` but `keep`, which moves to `generation` (the account's new
    /// one), so it survives the bump of a password change made through it: whether `keep` is
    /// still a session of `user`. Pair it with [`Self::rotate`] to give that browser a new id.
    /// Written before it returns.
    pub fn end_all_except(&self, user: &str, keep: &SessionId, generation: u64, now: u64) -> bool {
        let keep = keep.digest();
        self.end_where(now, |r, k| r.user == user && *k != keep);
        let (kept, moved) = {
            let mut s = self.state.lock_recover();
            let (kept, moved) = match s.map.get_mut(&keep) {
                Some(r) if r.user == user => {
                    let moved = r.generation != generation;
                    r.generation = generation;
                    (true, moved)
                }
                _ => (false, false),
            };
            if moved {
                s.v.force();
            }
            (kept, moved)
        };
        if moved {
            self.flush(now, false);
        }
        kept
    }

    /// Remove the sessions `f` picks; written at once when there were any.
    fn end_where(&self, now: u64, f: impl Fn(&Record, &Digest) -> bool) -> usize {
        let n = {
            let mut s = self.state.lock_recover();
            let before = s.map.len();
            s.map.retain(|k, r| !f(r, k));
            let n = before - s.map.len();
            if n > 0 {
                s.v.force();
            }
            n
        };
        if n > 0 {
            self.flush(now, false);
        }
        n
    }

    /// Give the session `old` a new id (a change of privilege): the same account, generation and
    /// `created`, `last_active` now; `old` no longer works. `None` when `old` is no session or
    /// has timed out. Written before it returns.
    pub fn rotate(&self, old: &SessionId, now: u64) -> Option<SessionId> {
        let id = SessionId(Secret::generate());
        let alive = {
            let mut s = self.state.lock_recover();
            let mut record = s.map.remove(&old.digest())?;
            s.v.force();
            let alive = record.timed_out(now).is_none();
            if alive {
                record.last_active = record.last_active.max(now);
                s.map.insert(id.digest(), record);
            }
            alive
        };
        self.flush(now, false);
        alive.then_some(id)
    }

    /// Remove the sessions timed out at `now`: how many.
    pub fn prune(&self, now: u64) -> usize {
        let mut s = self.state.lock_recover();
        let before = s.map.len();
        s.map.retain(|_, r| r.timed_out(now).is_none());
        let n = before - s.map.len();
        if n > 0 {
            s.v.lazy();
        }
        n
    }

    /// Remove the sessions `users` no longer allows (gone, disabled, another generation) or the
    /// clock ended at `now`: how many. For the flusher, so a removed account's sessions leave
    /// the file without waiting for a request.
    pub fn prune_users(&self, users: &Users, now: u64) -> usize {
        let mut s = self.state.lock_recover();
        let before = s.map.len();
        s.map.retain(|_, r| r.ended(users, now).is_none());
        let n = before - s.map.len();
        if n > 0 {
            s.v.lazy();
        }
        n
    }

    /// Sessions held.
    pub fn len(&self) -> usize {
        self.state.lock_recover().map.len()
    }

    /// Whether none are held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Write the file when something changed and either `force`, a change that must be written
    /// (creating, ending, rotating), or [`FLUSH_EVERY_SECS`] since the last write; expired
    /// sessions are dropped first. Whether it wrote. A failure is logged (throttled).
    pub fn flush(&self, now: u64, force: bool) -> bool {
        let mut w = self.persist.writer();
        let (text, version) = {
            let mut s = self.state.lock_recover();
            let before = s.map.len();
            s.map.retain(|_, r| r.timed_out(now).is_none());
            if s.map.len() != before {
                s.v.lazy();
            }
            if !s.v.due(force, now, w.last, FLUSH_EVERY_SECS) {
                return false;
            }
            let mut sessions: Vec<Entry> = s
                .map
                .iter()
                .map(|(k, r)| Entry {
                    hash: opaque::digest_hex(k),
                    user: r.user.clone(),
                    generation: r.generation,
                    created: r.created,
                    last_active: r.last_active,
                })
                .collect();
            sessions.sort_unstable_by(|a, b| a.hash.cmp(&b.hash));
            let file = FileRepr {
                format: FORMAT,
                sessions,
            };
            match opaque::to_text(&file, "the web UI sessions") {
                Ok(t) => (t, s.v.changed),
                Err(e) => {
                    tracing::warn!("{e:#}");
                    return false;
                }
            }
        };
        if !self.persist.write(&mut w, &text, now) {
            return false;
        }
        let mut s = self.state.lock_recover();
        s.v.written = s.v.written.max(version);
        true
    }

    /// Tests only: a session with the known id `id` (64 lowercase hex characters) for `user` at
    /// `generation`, made and last active at `at` (in memory; the next flush writes it). The
    /// server tests' signed-in requests carry such an id (`server::test_util`).
    #[cfg(test)]
    pub(crate) fn insert_known(&self, id: &str, user: &str, generation: u64, at: u64) {
        let id = SessionId::from_presented(id.as_bytes()).expect("a 64-hex test session id");
        let mut s = self.state.lock_recover();
        s.map.insert(
            id.digest(),
            Record {
                user: user.to_string(),
                generation,
                created: at,
                last_active: at,
            },
        );
        s.v.lazy();
    }
}

#[cfg(test)]
mod tests {
    use super::super::User;
    use super::*;
    use std::path::PathBuf;
    use std::sync::Arc;

    pub(crate) struct Scratch(pub(crate) PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    pub(crate) fn scratch() -> Scratch {
        let d = std::env::temp_dir().join(format!("bo_sessions_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        Scratch(d)
    }

    fn user(name: &str, generation: u64) -> User {
        User {
            name: name.to_string(),
            hash: "$argon2id$x".to_string(),
            created: 1,
            password_changed: 1,
            generation,
            failures: 0,
            last_failure: None,
            disabled: false,
        }
    }

    fn users(list: &[(&str, u64)]) -> Users {
        Users {
            format: super::super::FORMAT,
            users: list.iter().map(|(n, g)| user(n, *g)).collect(),
        }
    }

    const T: u64 = 1_700_000_000;

    fn file_text(s: &Sessions) -> String {
        std::fs::read_to_string(s.path()).unwrap()
    }

    fn file_last_active(s: &Sessions) -> Vec<u64> {
        let f: serde_json::Value = serde_json::from_str(&file_text(s)).unwrap();
        f["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["last_active"].as_u64().unwrap())
            .collect()
    }

    #[test]
    fn a_new_session_checks_and_hides_its_id() {
        let d = scratch();
        let s = Sessions::load(&d.0, T);
        assert!(s.is_empty());
        let u = users(&[("alice", 1)]);
        let id = s.create("alice", 1, T);
        assert_eq!(format!("{id:?}"), "SessionId(..)");
        assert!(!format!("{s:?}").contains(id.reveal()));
        let v = s
            .check(id.reveal().as_bytes(), &u, T + 5, Activity::User)
            .unwrap();
        assert_eq!(
            v,
            Valid {
                user: "alice".into(),
                created: T,
                expires_in: ABSOLUTE_SECS - 5,
            }
        );
        assert_eq!(s.len(), 1);
        assert_eq!(SessionId::from_presented(id.reveal().as_bytes()), Some(id));
    }

    #[test]
    fn idle_expiry_passes_at_the_limit_and_ends_one_second_later() {
        let d = scratch();
        let s = Sessions::load(&d.0, T);
        let u = users(&[("alice", 1)]);
        let a = s.create("alice", 1, T);
        let b = s.create("alice", 1, T);
        // Exactly IDLE_SECS idle: still valid (and the check moves last_active).
        assert!(
            s.check(a.reveal().as_bytes(), &u, T + IDLE_SECS, Activity::User)
                .is_some()
        );
        // One second more: gone, and removed.
        assert!(
            s.check(b.reveal().as_bytes(), &u, T + IDLE_SECS + 1, Activity::User)
                .is_none()
        );
        assert!(s.record(&b).is_none());
        assert!(s.record(&a).is_some());
    }

    #[test]
    fn absolute_expiry_ends_even_a_busy_session() {
        let d = scratch();
        let s = Sessions::load(&d.0, T);
        let u = users(&[("alice", 1)]);
        let id = s.create("alice", 1, T);
        let mut t = T;
        while t + 600 <= T + ABSOLUTE_SECS {
            t += 600;
            let v = s
                .check(id.reveal().as_bytes(), &u, t, Activity::User)
                .unwrap();
            assert_eq!(v.expires_in, T + ABSOLUTE_SECS - t);
        }
        assert_eq!(t, T + ABSOLUTE_SECS);
        assert!(
            s.check(id.reveal().as_bytes(), &u, t + 1, Activity::User)
                .is_none()
        );
        assert!(s.is_empty());
    }

    #[test]
    fn polls_do_not_extend_the_idle_limit() {
        let d = scratch();
        let s = Sessions::load(&d.0, T);
        let u = users(&[("alice", 1)]);
        let id = s.create("alice", 1, T);
        for k in 1..=6 {
            assert!(
                s.check(id.reveal().as_bytes(), &u, T + k * 600, Activity::Poll)
                    .is_some()
            );
        }
        assert_eq!(s.record(&id).unwrap().last_active, T);
        assert!(
            s.check(
                id.reveal().as_bytes(),
                &u,
                T + IDLE_SECS + 1,
                Activity::Poll
            )
            .is_none()
        );
    }

    #[test]
    fn the_account_decides_too() {
        let d = scratch();
        let s = Sessions::load(&d.0, T);
        let a = s.create("alice", 1, T);
        let b = s.create("bob", 3, T);
        let c = s.create("carol", 1, T);
        let mut u = users(&[("alice", 2), ("bob", 3), ("carol", 1)]);
        // A generation bump (password change, reset, sign out everywhere) ends it.
        assert!(
            s.check(a.reveal().as_bytes(), &u, T, Activity::User)
                .is_none()
        );
        assert!(s.record(&a).is_none());
        // Disabled.
        u.users[1].disabled = true;
        assert!(
            s.check(b.reveal().as_bytes(), &u, T, Activity::User)
                .is_none()
        );
        // Removed.
        u.users.retain(|x| x.name != "carol");
        assert!(
            s.check(c.reveal().as_bytes(), &u, T, Activity::User)
                .is_none()
        );
        assert!(s.is_empty());
    }

    #[test]
    fn ending_and_rotating() {
        let d = scratch();
        let s = Sessions::load(&d.0, T);
        let u = users(&[("alice", 1), ("bob", 1)]);
        let a1 = s.create("alice", 1, T);
        let a2 = s.create("alice", 1, T + 1);
        let a3 = s.create("alice", 1, T + 2);
        let b1 = s.create("bob", 1, T);
        let ok = |id: &SessionId, users: &Users| {
            s.check(id.reveal().as_bytes(), users, T + 10, Activity::Poll)
                .is_some()
        };
        assert!(s.end(&a1, T + 3));
        assert!(!s.end(&a1, T + 3));
        assert!(!ok(&a1, &u));

        // A password change through a2: the others end, a2 moves to the new generation.
        let u2 = users(&[("alice", 2), ("bob", 1)]);
        assert!(s.end_all_except("alice", &a2, 2, T + 4));
        assert!(ok(&a2, &u2));
        assert!(!ok(&a3, &u2));
        assert!(ok(&b1, &u2));

        // Rotation: the old id dies, the new one keeps `created`.
        let created = s.record(&a2).unwrap().created;
        let new = s.rotate(&a2, T + 5).unwrap();
        assert_ne!(new, a2);
        assert!(!ok(&a2, &u2));
        assert!(ok(&new, &u2));
        assert_eq!(s.record(&new).unwrap().created, created);
        assert_eq!(s.record(&new).unwrap().last_active, T + 5);
        assert!(s.rotate(&a2, T + 5).is_none());

        // Another account's session is no keeper (and stays).
        let a4 = s.create("alice", 2, T + 6);
        assert!(!s.end_all_except("alice", &b1, 2, T + 6));
        assert!(s.record(&a4).is_none());
        assert!(ok(&b1, &u2));

        assert_eq!(s.end_user("bob", T + 6), 1);
        assert!(!ok(&b1, &u2));
        assert!(s.is_empty());
        // A timed-out one is not rotated, and goes.
        let late = s.create("alice", 2, T + 7);
        assert!(s.rotate(&late, T + 7 + ABSOLUTE_SECS + 1).is_none());
        assert!(s.record(&late).is_none());
        assert!(s.is_empty());
    }

    #[test]
    fn the_oldest_goes_beyond_the_per_account_limit() {
        let d = scratch();
        let s = Sessions::load(&d.0, T);
        let u = users(&[("alice", 1), ("bob", 1)]);
        let b = s.create("bob", 1, T);
        let ids: Vec<SessionId> = (0..MAX_PER_USER as u64 + 2)
            .map(|i| s.create("alice", 1, T + i))
            .collect();
        assert_eq!(s.len(), MAX_PER_USER + 1);
        let alive = |id: &SessionId| {
            s.check(id.reveal().as_bytes(), &u, T + 100, Activity::Poll)
                .is_some()
        };
        assert!(!alive(&ids[0]));
        assert!(!alive(&ids[1]));
        assert!(ids[2..].iter().all(alive));
        assert!(alive(&b), "another account's sessions stay");
    }

    #[test]
    fn malformed_ids_are_refused_without_a_lookup() {
        let d = scratch();
        let s = Sessions::load(&d.0, T);
        let u = users(&[("alice", 1)]);
        let id = s.create("alice", 1, T);
        let good = id.reveal().as_bytes();
        let upper = id.reveal().to_ascii_uppercase();
        let long = format!("{}a", id.reveal());
        let spaced = format!(" {}", &id.reveal()[1..]);
        for bad in [
            &b""[..],
            &good[..63],
            long.as_bytes(),
            upper.as_bytes(),
            spaced.as_bytes(),
            &[0xc3u8; 64][..],
        ] {
            assert!(s.check(bad, &u, T, Activity::User).is_none());
            assert!(SessionId::from_presented(bad).is_none());
        }
        // A well-formed id that is no session.
        let other = crate::secret::random_hex(32);
        assert!(s.check(other.as_bytes(), &u, T, Activity::User).is_none());
        assert!(s.check(good, &u, T, Activity::User).is_some());
    }

    #[test]
    fn sessions_survive_a_reload_and_the_file_holds_no_id() {
        let d = scratch();
        let u = users(&[("alice", 1), ("bob", 1)]);
        let s = Sessions::load(&d.0, T);
        let a = s.create("alice", 1, T);
        let b = s.create("bob", 1, T + 1);
        let text = file_text(&s);
        assert!(!text.contains(a.reveal()) && !text.contains(b.reveal()));
        assert!(text.contains(&opaque::digest_hex(&a.digest())));
        let again = Sessions::load(&d.0, T + 2);
        assert_eq!(again.len(), 2);
        assert_eq!(again.record(&a), s.record(&a));
        assert!(
            again
                .check(a.reveal().as_bytes(), &u, T + 2, Activity::User)
                .is_some()
        );
        // Expired ones are not loaded.
        let late = Sessions::load(&d.0, T + ABSOLUTE_SECS + 2);
        assert!(late.is_empty());
    }

    #[test]
    fn a_damaged_file_means_no_sessions_and_stays_until_a_write() {
        let d = scratch();
        let auth = crate::private_dir::ensure(&d.0).unwrap();
        let path = auth.join(FILE_NAME);
        for junk in [
            &b"{\"format\": 1, \"sessions\": [{\"hash\": 5"[..],
            b"{\"format\": 9, \"sessions\": []}",
            b"{\"format\": 1, \"sessions\": [{\"hash\": \"zz\", \"user\": \"a\", \
              \"generation\": 1, \"created\": 1, \"last_active\": 1}]}",
            b"\xff\xfe",
        ] {
            std::fs::write(&path, junk).unwrap();
            let s = Sessions::load(&d.0, T);
            assert!(s.is_empty());
            assert!(!s.flush(T + 10_000, true), "nothing to write");
            assert_eq!(std::fs::read(&path).unwrap(), junk);
            s.create("alice", 1, T);
            assert_eq!(Sessions::load(&d.0, T).len(), 1);
        }
    }

    #[test]
    fn a_moved_last_active_waits_for_the_flush_interval() {
        let d = scratch();
        let u = users(&[("alice", 1)]);
        let s = Sessions::load(&d.0, T);
        let id = s.create("alice", 1, T);
        // Written at once by `create`.
        assert_eq!(file_last_active(&s), vec![T]);
        assert!(
            s.check(id.reveal().as_bytes(), &u, T + 10, Activity::User)
                .is_some()
        );
        assert!(!s.flush(T + 10, false));
        assert!(!s.flush(T + FLUSH_EVERY_SECS - 1, false));
        assert_eq!(file_last_active(&s), vec![T]);
        assert!(s.flush(T + FLUSH_EVERY_SECS, false));
        assert_eq!(file_last_active(&s), vec![T + 10]);
        // Nothing changed: no write, forced or not.
        assert!(!s.flush(T + 10 * FLUSH_EVERY_SECS, true));
        // A forced flush (shutdown) writes a pending move at once.
        let t = T + FLUSH_EVERY_SECS + 1;
        s.check(id.reveal().as_bytes(), &u, t, Activity::User);
        assert!(s.flush(t, true));
        assert_eq!(file_last_active(&s), vec![t]);
        // A second session is written at once, whatever the interval.
        let other = s.create("alice", 1, t + 1);
        assert_eq!(file_last_active(&s).len(), 2);
        assert!(s.end(&other, t + 2));
        assert_eq!(file_last_active(&s).len(), 1);
    }

    #[test]
    fn a_flush_drops_expired_entries() {
        let d = scratch();
        let s = Sessions::load(&d.0, T);
        s.create("alice", 1, T);
        s.create("alice", 1, T + IDLE_SECS);
        // The expired one goes at once; the file waits (written by `create` a second ago).
        assert!(!s.flush(T + IDLE_SECS + 1, false));
        assert_eq!(s.len(), 1);
        assert!(s.flush(T + IDLE_SECS + 1, true));
        assert_eq!(file_last_active(&s), vec![T + IDLE_SECS]);
        assert_eq!(s.prune(T + 3 * IDLE_SECS), 1);
        assert!(s.is_empty());
    }

    #[test]
    fn concurrent_checks_agree() {
        let d = scratch();
        let s = Arc::new(Sessions::load(&d.0, T));
        let u = Arc::new(users(&[("alice", 1)]));
        let ids: Vec<SessionId> = (0..4).map(|_| s.create("alice", 1, T)).collect();
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let (s, u, id) = (s.clone(), u.clone(), ids[t % 4].clone());
                std::thread::spawn(move || {
                    for i in 0..500u64 {
                        let act = if i % 2 == 0 {
                            Activity::User
                        } else {
                            Activity::Poll
                        };
                        assert!(s.check(id.reveal().as_bytes(), &u, T + i, act).is_some());
                        if i % 100 == 0 {
                            s.flush(T + i, false);
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(s.len(), 4);
        for id in &ids {
            assert_eq!(s.record(id).unwrap().last_active, T + 498);
        }
        // Whether this forced flush still has anything to write depends on the interleaving:
        // a thread's lazy flush may already have written the last move (a check that finds
        // `last_active` moved past it by another thread marks nothing). Either way the file
        // ends up holding the final state.
        s.flush(T + 1000, true);
        assert_eq!(file_last_active(&s), vec![T + 498; 4]);
        assert_eq!(Sessions::load(&d.0, T + 1000).len(), 4);
    }

    #[cfg(unix)]
    #[test]
    fn the_file_is_private() {
        use std::os::unix::fs::PermissionsExt;
        let d = scratch();
        let s = Sessions::load(&d.0, T);
        s.create("alice", 1, T);
        let mode = std::fs::metadata(s.path()).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
        let dir = std::fs::metadata(s.path().parent().unwrap())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(dir & 0o777, 0o700);
    }
}
