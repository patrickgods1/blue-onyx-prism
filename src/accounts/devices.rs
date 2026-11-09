//! Device cookies (OWASP "Slow Down Online Guessing Attacks with Device Cookies"): a browser that
//! signed in to an account before keeps `__Host-bop_device` ([`super::cookies`]), so an attacker
//! guessing that account's password elsewhere cannot lock its owner out.
//!
//! **How sign-in uses it** (the HTTP layer): an attempt for account `name` that shows a device
//! cookie [`Devices::trusted`] for `name` skips the account-wide lockout delay
//! ([`super::lockout`]) and counts its failures against the device's own budget instead
//! ([`Devices::record_failure`]; at [`DEVICE_FAILURE_BUDGET`] the device is forgotten and the
//! browser is an unknown one again). It never skips the per-address limit, and never a
//! `disabled` account. A successful sign-in calls [`Devices::record_success`] (budget reset,
//! expiry extended) or, for a browser without a known device, [`Devices::issue`].
//!
//! Ids, hashing and the file work as for sessions ([`super::opaque`]): 64 lowercase hex
//! characters in the cookie, only the SHA-256 in memory and in `auth/devices.json` (format
//! [`FORMAT`]: `{"format": 1, "devices": [{"hash", "user", "failures", "expires"}]}`). A device
//! is valid while `now < expires`. Every change is written before the call returns, except
//! dropping an expired device, which waits for a flush. Missing or damaged = no devices (a
//! warning that quotes nothing; the file stays until the next write). At most [`MAX_PER_USER`]
//! devices per account and [`MAX_DEVICES`] in all; past that the one expiring soonest goes.

use super::opaque::{self, Digest, Persist, Secret, Versions};
use crate::sync::LockExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

/// The file's name in `auth/`.
pub const FILE_NAME: &str = "devices.json";
/// The format this version writes and reads.
pub const FORMAT: u32 = 1;
/// A device is trusted this long after its last successful sign-in.
pub const MAX_AGE_SECS: u64 = 180 * 24 * 3600;
/// Wrong passwords a trusted device may send before it is forgotten.
pub const DEVICE_FAILURE_BUDGET: u32 = 10;
/// Devices kept in all.
pub const MAX_DEVICES: usize = 4096;
/// Devices kept per account.
pub const MAX_PER_USER: usize = 32;
/// An expired device's removal is written at most this often.
pub const FLUSH_EVERY_SECS: u64 = super::sessions::FLUSH_EVERY_SECS;

/// A device cookie's value. No `Display` or `Serialize`; `Debug` hides it.
#[derive(Clone, PartialEq, Eq)]
pub struct DeviceId(Secret);

impl std::fmt::Debug for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DeviceId(..)")
    }
}

impl DeviceId {
    /// A cookie's value as an id, when it has the shape of one.
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

/// What is kept of a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    /// The account it signed in to.
    pub user: String,
    /// Wrong passwords since its last success.
    pub failures: u32,
    /// Unix seconds; valid while now is before it.
    pub expires: u64,
}

/// The device cookies of the process (see the module docs). Share it in an `Arc`.
pub struct Devices {
    persist: Persist,
    state: Mutex<State>,
}

struct State {
    map: HashMap<Digest, Device>,
    v: Versions,
}

impl std::fmt::Debug for Devices {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Devices")
            .field("path", &self.persist.path())
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

#[derive(Serialize, Deserialize)]
struct FileRepr {
    format: u32,
    devices: Vec<Entry>,
}

#[derive(Serialize, Deserialize)]
struct Entry {
    hash: String,
    user: String,
    failures: u32,
    expires: u64,
}

#[derive(Deserialize)]
struct Head {
    format: u32,
}

fn parse(bytes: &[u8]) -> Result<HashMap<Digest, Device>, String> {
    let head: Head = serde_json::from_slice(bytes).map_err(|e| opaque::json_problem(&e))?;
    if head.format != FORMAT {
        return Err(format!(
            "format {} (this version reads {FORMAT})",
            head.format
        ));
    }
    let file: FileRepr = serde_json::from_slice(bytes).map_err(|e| opaque::json_problem(&e))?;
    let mut map = HashMap::with_capacity(file.devices.len());
    for (i, e) in file.devices.into_iter().enumerate() {
        let Some(hash) = opaque::parse_digest_hex(&e.hash) else {
            return Err(format!("device {} has an invalid hash", i + 1));
        };
        if super::normalize_name(&e.user).ok().as_deref() != Some(e.user.as_str()) {
            return Err(format!("device {} has an invalid account name", i + 1));
        }
        map.insert(
            hash,
            Device {
                user: e.user,
                failures: e.failures,
                expires: e.expires,
            },
        );
    }
    Ok(map)
}

/// Remove the soonest-expiring entries `pick` selects until at most `keep` of them are left.
fn evict(map: &mut HashMap<Digest, Device>, keep: usize, pick: impl Fn(&Device) -> bool) {
    let mut chosen: Vec<(u64, Digest)> = map
        .iter()
        .filter(|(_, d)| pick(d))
        .map(|(k, d)| (d.expires, *k))
        .collect();
    let over = chosen.len().saturating_sub(keep);
    if over == 0 {
        return;
    }
    chosen.sort_unstable();
    for (_, k) in chosen.into_iter().take(over) {
        map.remove(&k);
    }
}

impl Devices {
    /// The devices saved under `data_root`, those expired at `now` left out. Reads only.
    pub fn load(data_root: &Path, now: u64) -> Self {
        let persist = Persist::new(data_root, FILE_NAME, now);
        let path = persist.path();
        let mut map = match opaque::read(&path) {
            opaque::Loaded::Missing => HashMap::new(),
            opaque::Loaded::Failed(why) => {
                tracing::warn!(
                    "{}: {why}; no browser counts as a known device until it signs in again. \
                     Check that the service's account may read it",
                    path.display()
                );
                HashMap::new()
            }
            opaque::Loaded::Bytes(bytes) => parse(&bytes).unwrap_or_else(|why| {
                tracing::warn!(
                    "{} is damaged ({why}); no browser counts as a known device until it signs \
                     in again. The file is replaced at the next sign-in",
                    path.display()
                );
                HashMap::new()
            }),
        };
        map.retain(|_, d| now < d.expires && d.failures < DEVICE_FAILURE_BUDGET);
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

    /// A new device for `user` after a successful sign-in at `now`. Written before it returns.
    pub fn issue(&self, user: &str, now: u64) -> DeviceId {
        let id = DeviceId(Secret::generate());
        {
            let mut s = self.state.lock_recover();
            s.map.retain(|_, d| now < d.expires);
            evict(&mut s.map, MAX_PER_USER - 1, |d| d.user == user);
            evict(&mut s.map, MAX_DEVICES - 1, |_| true);
            s.map.insert(
                id.digest(),
                Device {
                    user: user.to_string(),
                    failures: 0,
                    expires: now.saturating_add(MAX_AGE_SECS),
                },
            );
            s.v.force();
        }
        self.flush(now, false);
        id
    }

    /// Whether `presented` (a cookie's value) is a device of `user` at `now` with budget left.
    pub fn trusted(&self, presented: &[u8], user: &str, now: u64) -> bool {
        let Some(key) = opaque::digest_of(presented) else {
            return false;
        };
        let mut s = self.state.lock_recover();
        let Some(d) = s.map.get(&key) else {
            return false;
        };
        if now >= d.expires {
            s.map.remove(&key);
            s.v.lazy();
            return false;
        }
        d.user == user && d.failures < DEVICE_FAILURE_BUDGET
    }

    /// What is kept of the device `id`, for tests and listings.
    pub fn device(&self, id: &DeviceId) -> Option<Device> {
        self.state.lock_recover().map.get(&id.digest()).cloned()
    }

    /// A wrong password for `user` from the trusted device `presented`: counted, and at
    /// [`DEVICE_FAILURE_BUDGET`] the device is forgotten. Whether it was counted (false when it
    /// is no trusted device of `user`; the account's own lockout counts it then). Written
    /// before it returns.
    pub fn record_failure(&self, presented: &[u8], user: &str, now: u64) -> bool {
        if !self.trusted(presented, user, now) {
            return false;
        }
        let Some(key) = opaque::digest_of(presented) else {
            return false;
        };
        let counted = {
            let mut s = self.state.lock_recover();
            let counted = match s.map.get_mut(&key) {
                Some(d) if d.user == user => {
                    d.failures = d.failures.saturating_add(1);
                    if d.failures >= DEVICE_FAILURE_BUDGET {
                        s.map.remove(&key);
                    }
                    true
                }
                _ => false,
            };
            if counted {
                s.v.force();
            }
            counted
        };
        if counted {
            self.flush(now, false);
        }
        counted
    }

    /// A good password for `user` from the device `presented`: its failures reset and its expiry
    /// moves to `now` + [`MAX_AGE_SECS`]. Whether it was a trusted device of `user` (if not, the
    /// caller issues a new one). Written before it returns.
    pub fn record_success(&self, presented: &[u8], user: &str, now: u64) -> bool {
        if !self.trusted(presented, user, now) {
            return false;
        }
        let Some(key) = opaque::digest_of(presented) else {
            return false;
        };
        let known = {
            let mut s = self.state.lock_recover();
            let known = match s.map.get_mut(&key) {
                Some(d) if d.user == user => {
                    d.failures = 0;
                    d.expires = now.saturating_add(MAX_AGE_SECS);
                    true
                }
                _ => false,
            };
            if known {
                s.v.force();
            }
            known
        };
        if known {
            self.flush(now, false);
        }
        known
    }

    /// Forget every device of `user` (an account removed): how many. Written before it returns.
    pub fn forget_user(&self, user: &str, now: u64) -> usize {
        let n = {
            let mut s = self.state.lock_recover();
            let before = s.map.len();
            s.map.retain(|_, d| d.user != user);
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

    /// Devices held.
    pub fn len(&self) -> usize {
        self.state.lock_recover().map.len()
    }

    /// Whether none are held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Write the file when something changed and either `force`, a change that must be written,
    /// or [`FLUSH_EVERY_SECS`] since the last write; expired devices are dropped first. Whether
    /// it wrote.
    pub fn flush(&self, now: u64, force: bool) -> bool {
        let mut w = self.persist.writer();
        let (text, version) = {
            let mut s = self.state.lock_recover();
            let before = s.map.len();
            s.map.retain(|_, d| now < d.expires);
            if s.map.len() != before {
                s.v.lazy();
            }
            if !s.v.due(force, now, w.last, FLUSH_EVERY_SECS) {
                return false;
            }
            let mut devices: Vec<Entry> = s
                .map
                .iter()
                .map(|(k, d)| Entry {
                    hash: opaque::digest_hex(k),
                    user: d.user.clone(),
                    failures: d.failures,
                    expires: d.expires,
                })
                .collect();
            devices.sort_unstable_by(|a, b| a.hash.cmp(&b.hash));
            let file = FileRepr {
                format: FORMAT,
                devices,
            };
            match opaque::to_text(&file, "the known devices") {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct Scratch(PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn scratch() -> Scratch {
        let d = std::env::temp_dir().join(format!("bo_devices_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        Scratch(d)
    }

    const T: u64 = 1_700_000_000;

    #[test]
    fn an_issued_device_is_trusted_for_its_account_only() {
        let d = scratch();
        let s = Devices::load(&d.0, T);
        let id = s.issue("alice", T);
        assert_eq!(format!("{id:?}"), "DeviceId(..)");
        assert!(!format!("{s:?}").contains(id.reveal()));
        let p = id.reveal().as_bytes();
        assert!(s.trusted(p, "alice", T + 1));
        assert!(!s.trusted(p, "bob", T + 1));
        assert!(!s.record_failure(p, "bob", T + 1), "not bob's device");
        assert_eq!(s.device(&id).unwrap().failures, 0);
        assert!(!s.trusted(b"not-an-id", "alice", T));
        assert!(!s.trusted(&p[..63], "alice", T));
        assert!(!s.trusted(crate::secret::random_hex(32).as_bytes(), "alice", T));
        assert_eq!(DeviceId::from_presented(p), Some(id));
    }

    #[test]
    fn devices_expire_and_a_success_extends_them() {
        let d = scratch();
        let s = Devices::load(&d.0, T);
        let id = s.issue("alice", T);
        let p = id.reveal().as_bytes();
        assert!(s.trusted(p, "alice", T + MAX_AGE_SECS - 1));
        assert!(s.record_success(p, "alice", T + 100));
        assert!(s.trusted(p, "alice", T + MAX_AGE_SECS + 99));
        assert!(!s.trusted(p, "alice", T + MAX_AGE_SECS + 100));
        assert!(s.device(&id).is_none(), "an expired device is dropped");
        assert!(!s.record_success(p, "alice", T + MAX_AGE_SECS + 100));
    }

    #[test]
    fn the_failure_budget_forgets_the_device_and_a_success_resets_it() {
        let d = scratch();
        let s = Devices::load(&d.0, T);
        let id = s.issue("alice", T);
        let p = id.reveal().as_bytes();
        for _ in 0..DEVICE_FAILURE_BUDGET - 1 {
            assert!(s.record_failure(p, "alice", T));
        }
        assert_eq!(s.device(&id).unwrap().failures, DEVICE_FAILURE_BUDGET - 1);
        assert!(s.record_success(p, "alice", T + 5));
        assert_eq!(s.device(&id).unwrap().failures, 0);
        for _ in 0..DEVICE_FAILURE_BUDGET - 1 {
            assert!(s.record_failure(p, "alice", T));
        }
        assert!(s.trusted(p, "alice", T));
        assert!(s.record_failure(p, "alice", T));
        assert!(!s.trusted(p, "alice", T));
        assert!(s.device(&id).is_none());
        assert!(!s.record_failure(p, "alice", T));
        // Persisted: a reload does not bring it back.
        assert!(Devices::load(&d.0, T).is_empty());
    }

    #[test]
    fn caps_evict_the_soonest_expiring() {
        let d = scratch();
        let s = Devices::load(&d.0, T);
        let first = s.issue("alice", T);
        let mine: Vec<DeviceId> = (1..MAX_PER_USER as u64 + 1)
            .map(|i| s.issue("alice", T + i))
            .collect();
        assert_eq!(s.len(), MAX_PER_USER);
        assert!(s.device(&first).is_none());
        assert!(mine.iter().all(|id| s.device(id).is_some()));
        assert_eq!(s.forget_user("alice", T + 100), MAX_PER_USER);
        assert!(s.is_empty());

        // The total cap, over many accounts.
        let oldest = s.issue("u0", T);
        for i in 1..MAX_DEVICES as u64 + 10 {
            s.state.lock_recover().map.insert(
                Secret::generate().digest(),
                Device {
                    user: format!("u{}", i % 1000),
                    failures: 0,
                    expires: T + MAX_AGE_SECS + i,
                },
            );
        }
        let newest = s.issue("zed", T + 1000);
        assert_eq!(s.len(), MAX_DEVICES);
        assert!(s.device(&oldest).is_none());
        assert!(s.device(&newest).is_some());
    }

    #[test]
    fn devices_survive_a_reload_without_their_ids() {
        let d = scratch();
        let s = Devices::load(&d.0, T);
        let a = s.issue("alice", T);
        let b = s.issue("bob", T);
        s.record_failure(b.reveal().as_bytes(), "bob", T + 1);
        let text = std::fs::read_to_string(s.path()).unwrap();
        assert!(!text.contains(a.reveal()) && !text.contains(b.reveal()));
        let again = Devices::load(&d.0, T + 2);
        assert_eq!(again.len(), 2);
        assert_eq!(again.device(&b).unwrap().failures, 1);
        assert!(again.trusted(a.reveal().as_bytes(), "alice", T + 2));
        // Damaged: none, and the file stays until a write.
        std::fs::write(s.path(), b"{\"format\": 1, \"devices\": 7}").unwrap();
        let broken = Devices::load(&d.0, T);
        assert!(broken.is_empty());
        assert!(!broken.flush(T + 10_000, true));
        assert_eq!(
            std::fs::read(s.path()).unwrap(),
            b"{\"format\": 1, \"devices\": 7}"
        );
        broken.issue("alice", T);
        assert_eq!(Devices::load(&d.0, T).len(), 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(s.path()).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
    }
}
