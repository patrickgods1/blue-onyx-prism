//! The web UI's accounts: `<data root>/auth/users.json`, its rules and the store that changes
//! it. All accounts are full administrators. Nothing here is a route yet; the CLI
//! (`blue-onyx-prism user ...`, [`commands`]) is the only writer so far.
//!
//! - [`password`]: the password policy (NFC, 15 ..= 128 code points, blocklist, no name inside)
//!   and Argon2id hashing behind a process-wide semaphore. [`blocklist`]: the common passwords.
//! - [`lockout`]: per-account delays and disabling (pure). [`setup`]: the one-time setup code
//!   that creates the first account. [`lock`]: `auth/.lock`.
//! - [`sessions`] and [`devices`]: the web UI's sessions and device cookies (hashes only, in
//!   `auth/sessions.json` / `auth/devices.json`), [`cookies`]: their names and `Set-Cookie`
//!   values, [`Auth`] ([`process`]): all of it for one process, shared by its generations.
//!
//! **The file** (`users.json`, format [`FORMAT`]):
//! `{"format": 1, "users": [{"name", "hash", "created", "password_changed", "generation",
//! "failures", "last_failure", "disabled"}]}`. Times are unix seconds (UTC), so lockout
//! arithmetic needs no parsing; `user list` prints them as RFC 3339. `hash` is the PHC string of
//! [`password::hash`]. `generation` starts at 1 and is bumped by a password change or reset and
//! by "sign out everywhere": sessions (a later change) record it, so a change made by the CLI ends
//! the live sessions of that account without touching them.
//!
//! **Fail closed.** A missing `users.json` is the only state that means "no accounts" (setup
//! mode). One that cannot be read, does not parse, holds an invalid entry or has a newer format
//! is an error ([`Store::load`]): nobody can sign in, setup stays closed, and the file is never
//! moved aside or replaced automatically. A file with an empty list is no setup mode either.
//!
//! **Writes.** Every read-modify-write holds `auth/.lock` ([`lock`]) and re-reads the file under
//! it, so neither threads nor processes (the service and a command) lose a change; a password is
//! hashed before the lock is taken, never under it. Files are written through
//! [`crate::private_dir::write_file`] (a temporary file in `auth/`, then a rename).
//!
//! **Names:** 1 ..= 32 of `[a-z0-9._-]`, not starting with `.` or `-`; input is lowercased
//! (ASCII only, so a look-alike such as the Kelvin sign is refused rather than folded onto `k`).

pub mod blocklist;
pub mod commands;
pub mod cookies;
pub mod devices;
pub mod lock;
pub mod lockout;
mod opaque;
pub mod password;
pub mod process;
pub mod sessions;
pub mod setup;

pub use process::Auth;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// The accounts file's name in `auth/`.
pub const FILE_NAME: &str = "users.json";
/// The format this version writes and reads.
pub const FORMAT: u32 = 1;
/// The longest name.
pub const NAME_MAX: usize = 32;
/// The largest accounts file read (a valid one is a few hundred bytes per account).
const READ_LIMIT: u64 = 4 * 1024 * 1024;

/// Now in unix seconds (0 before 1970).
pub fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// `secs` (unix) as RFC 3339 UTC, for people.
pub fn rfc3339(secs: u64) -> String {
    i64::try_from(secs)
        .ok()
        .and_then(|s| chrono::DateTime::<chrono::Utc>::from_timestamp(s, 0))
        .map_or_else(
            || secs.to_string(),
            |t| t.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        )
}

/// Why an account change was refused. Callers may downcast an `anyhow::Error` to it; the
/// messages never repeat the name or the password that was typed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Refused {
    #[error(
        "a username has 1 to 32 characters of a-z, 0-9, '.', '_' and '-', and does not start \
         with '.' or '-': choose another"
    )]
    BadName,
    #[error("an account with that name exists already: choose another name")]
    NameTaken,
    #[error("no account has that name; `blue-onyx-prism user list` shows them")]
    UnknownName,
    #[error(
        "there are no accounts yet: create the first one with `blue-onyx-prism user add <name>` \
         or with the setup code (`blue-onyx-prism setup-code`)"
    )]
    NoAccounts,
    #[error(
        "that is the last account, and removing it would lock everyone out of the web UI: add \
         another account first"
    )]
    LastAccount,
    #[error(transparent)]
    Password(#[from] password::Refusal),
}

/// `input` as a username: ASCII-lowercased, then checked (see the module docs).
pub fn normalize_name(input: &str) -> Result<String, Refused> {
    let name = input.to_ascii_lowercase();
    let ok = !name.is_empty()
        && name.len() <= NAME_MAX
        && name.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
        })
        && !name.starts_with(['.', '-']);
    if ok { Ok(name) } else { Err(Refused::BadName) }
}

/// The accounts file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Users {
    pub format: u32,
    pub users: Vec<User>,
}

impl Users {
    fn empty() -> Self {
        Self {
            format: FORMAT,
            users: Vec::new(),
        }
    }

    /// The account `name` (normalized by the caller).
    pub fn get(&self, name: &str) -> Option<&User> {
        self.users.iter().find(|u| u.name == name)
    }

    fn get_mut(&mut self, name: &str) -> Result<&mut User> {
        self.users
            .iter_mut()
            .find(|u| u.name == name)
            .ok_or_else(|| Refused::UnknownName.into())
    }
}

/// One account. `Debug` leaves the hash out.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct User {
    pub name: String,
    /// PHC string (Argon2id).
    pub hash: String,
    /// Unix seconds.
    pub created: u64,
    /// Unix seconds.
    pub password_changed: u64,
    pub generation: u64,
    #[serde(default)]
    pub failures: u32,
    /// Unix seconds.
    #[serde(default)]
    pub last_failure: Option<u64>,
    #[serde(default)]
    pub disabled: bool,
}

impl std::fmt::Debug for User {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("User")
            .field("name", &self.name)
            .field("hash", &"..")
            .field("created", &self.created)
            .field("password_changed", &self.password_changed)
            .field("generation", &self.generation)
            .field("failures", &self.failures)
            .field("last_failure", &self.last_failure)
            .field("disabled", &self.disabled)
            .finish()
    }
}

impl User {
    /// The lockout state of this account.
    pub fn lockout(&self) -> lockout::State {
        lockout::State {
            failures: self.failures,
            last_failure: self.last_failure,
            disabled: self.disabled,
        }
    }

    fn set_lockout(&mut self, s: lockout::State) {
        self.failures = s.failures;
        self.last_failure = s.last_failure;
        self.disabled = s.disabled;
    }
}

/// `text` as an accounts file at `path`, checked (see the module docs). The error never quotes
/// the file (it holds hashes): only the kind of problem and where.
fn parse(text: &[u8], path: &Path) -> Result<Users> {
    #[derive(Deserialize)]
    struct Head {
        format: u32,
    }
    let where_ = |e: &serde_json::Error| {
        let kind = match e.classify() {
            serde_json::error::Category::Io => "unreadable",
            serde_json::error::Category::Syntax => "not valid JSON",
            serde_json::error::Category::Data => "not an accounts file",
            serde_json::error::Category::Eof => "cut short",
        };
        format!("{kind} at line {} column {}", e.line(), e.column())
    };
    let broken = |what: String| {
        anyhow::anyhow!(
            "{} is damaged ({what}), so nobody can sign in to the web UI and setup stays \
             closed. Restore it from a backup, or reset an account's password with \
             `blue-onyx-prism user reset-password <name>` once it reads again; to start over, \
             delete the file (then `blue-onyx-prism setup-code`). It is never replaced \
             automatically",
            path.display()
        )
    };
    let head: Head = serde_json::from_slice(text).map_err(|e| broken(where_(&e)))?;
    if head.format > FORMAT {
        bail!(
            "{} was written by a newer version of Blue Onyx Prism (format {}, this one reads \
             {FORMAT}), so nobody can sign in to the web UI; upgrade this installation again, \
             or restore the file from a backup",
            path.display(),
            head.format
        );
    }
    if head.format != FORMAT {
        return Err(broken(format!(
            "format {} is not one this version knows",
            head.format
        )));
    }
    let users: Users = serde_json::from_slice(text).map_err(|e| broken(where_(&e)))?;
    let mut seen = std::collections::BTreeSet::new();
    for (i, u) in users.users.iter().enumerate() {
        let n = i + 1;
        if normalize_name(&u.name).ok().as_deref() != Some(u.name.as_str()) {
            return Err(broken(format!("account {n} has an invalid name")));
        }
        if !seen.insert(u.name.as_str()) {
            return Err(broken(format!("account {n} repeats another's name")));
        }
        if u.hash.is_empty() {
            return Err(broken(format!("account {n} has no password hash")));
        }
    }
    Ok(users)
}

/// Read the accounts file at `path`: `None` only when it does not exist.
fn read_file(path: &Path) -> Result<Option<Users>> {
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => bail!(
            "could not read {} ({e}), so nobody can sign in to the web UI and setup stays \
             closed; run this as the service's account (the folder is readable by it and \
             administrators only)",
            path.display()
        ),
    };
    let mut bytes = Vec::new();
    file.take(READ_LIMIT + 1)
        .read_to_end(&mut bytes)
        .with_context(|| {
            format!(
                "reading {}, so nobody can sign in to the web UI; run this as the service's \
                 account",
                path.display()
            )
        })?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > READ_LIMIT {
        bail!(
            "{} is larger than {} MiB, far more than an accounts file holds, so nobody can sign \
             in to the web UI; restore it from a backup",
            path.display(),
            READ_LIMIT / 1024 / 1024
        );
    }
    parse(&bytes, path).map(Some)
}

/// The accounts store of one data root. Cheap to make; every call reads the file.
#[derive(Debug, Clone)]
pub struct Store {
    data_root: PathBuf,
}

impl Store {
    /// The store of `data_root` (no I/O).
    pub fn open(data_root: &Path) -> Self {
        Self {
            data_root: data_root.to_path_buf(),
        }
    }

    /// The data root.
    pub fn data_root(&self) -> &Path {
        &self.data_root
    }

    /// `<data root>/auth`.
    pub fn dir(&self) -> PathBuf {
        crate::private_dir::path(&self.data_root)
    }

    /// `<data root>/auth/users.json`.
    pub fn path(&self) -> PathBuf {
        self.dir().join(FILE_NAME)
    }

    /// The accounts: `Ok(None)` only when the file does not exist; anything else wrong with it
    /// is an error (see the module docs).
    pub fn load(&self) -> Result<Option<Users>> {
        read_file(&self.path())
    }

    /// Under `auth/.lock`: read the file, hand it to `f`, write what `f` returns (`Some`).
    /// `f` gets the auth folder too, for other files changed under the same lock.
    pub(crate) fn locked<T>(
        &self,
        f: impl FnOnce(&Path, Option<Users>) -> Result<(Option<Users>, T)>,
    ) -> Result<T> {
        let auth = crate::private_dir::ensure(&self.data_root)?;
        let _held = lock::lock(&auth)?;
        let current = read_file(&auth.join(FILE_NAME))?;
        let (write, out) = f(&auth, current)?;
        if let Some(users) = write {
            write_users(&auth, &users)?;
        }
        Ok(out)
    }

    /// Change the existing accounts with `f` (an error when there are none).
    fn change<T>(&self, f: impl FnOnce(&mut Users) -> Result<T>) -> Result<T> {
        self.locked(|_, current| {
            let mut users = current.ok_or(Refused::NoAccounts)?;
            let out = f(&mut users)?;
            Ok((Some(users), out))
        })
    }

    /// Add the account `name` with `password` (checked against the policy, then hashed with
    /// `permit` before the lock is taken). The first account may be made this way too (the
    /// server's own command line is trusted like the setup code); that removes the setup code.
    pub fn add(
        &self,
        name: &str,
        password: &str,
        permit: &password::Permit<'_>,
        now: u64,
    ) -> Result<User> {
        let name = normalize_name(name)?;
        password::check(password, &name).map_err(Refused::from)?;
        let hash = password::hash(password, permit)?;
        self.add_hashed(&name, hash, now)
    }

    /// [`Self::add`] with the hash made already.
    pub fn add_hashed(&self, name: &str, hash: String, now: u64) -> Result<User> {
        let name = normalize_name(name)?;
        self.locked(|auth, current| {
            let first = current.is_none();
            let mut users = current.unwrap_or_else(Users::empty);
            if users.get(&name).is_some() {
                return Err(Refused::NameTaken.into());
            }
            let user = User {
                name,
                hash,
                created: now,
                password_changed: now,
                generation: 1,
                failures: 0,
                last_failure: None,
                disabled: false,
            };
            users.users.push(user.clone());
            if first {
                // Written before the code goes, so a failure here leaves setup possible.
                write_users(auth, &users)?;
                setup::remove_code_file(auth)?;
                return Ok((None, user));
            }
            Ok((Some(users), user))
        })
    }

    /// Remove the account `name`; never the last one.
    pub fn remove(&self, name: &str) -> Result<()> {
        let name = normalize_name(name).map_err(|_| Refused::UnknownName)?;
        self.change(|users| {
            if users.get(&name).is_none() {
                return Err(Refused::UnknownName.into());
            }
            if users.users.len() <= 1 {
                return Err(Refused::LastAccount.into());
            }
            users.users.retain(|u| u.name != name);
            Ok(())
        })
    }

    /// Give `name` a new password (policy checked, hashed before the lock): the generation goes
    /// up (its sessions end), `password_changed` is now, failures and disabled are cleared.
    pub fn set_password(
        &self,
        name: &str,
        password: &str,
        permit: &password::Permit<'_>,
        now: u64,
    ) -> Result<User> {
        let name = normalize_name(name).map_err(|_| Refused::UnknownName)?;
        if self
            .load()?
            .ok_or(Refused::NoAccounts)?
            .get(&name)
            .is_none()
        {
            return Err(Refused::UnknownName.into());
        }
        password::check(password, &name).map_err(Refused::from)?;
        let hash = password::hash(password, permit)?;
        self.change(|users| {
            let u = users.get_mut(&name)?;
            u.hash = hash;
            u.password_changed = now;
            u.generation = u.generation.saturating_add(1);
            u.set_lockout(lockout::State::default());
            Ok(u.clone())
        })
    }

    /// Clear `name`'s failures and disabled flag (its sessions stay).
    pub fn unlock(&self, name: &str) -> Result<User> {
        let name = normalize_name(name).map_err(|_| Refused::UnknownName)?;
        self.change(|users| {
            let u = users.get_mut(&name)?;
            u.set_lockout(lockout::State::default());
            Ok(u.clone())
        })
    }

    /// Sign `name` out everywhere: the generation goes up. The new generation.
    pub fn bump_generation(&self, name: &str) -> Result<u64> {
        let name = normalize_name(name).map_err(|_| Refused::UnknownName)?;
        self.change(|users| {
            let u = users.get_mut(&name)?;
            u.generation = u.generation.saturating_add(1);
            Ok(u.generation)
        })
    }

    /// Count a wrong password for `name` at `now`; the new state, `None` when no such account
    /// (the caller counts it in [`lockout::UnknownNames`]).
    pub fn record_failure(&self, name: &str, now: u64) -> Result<Option<lockout::State>> {
        let Ok(name) = normalize_name(name) else {
            return Ok(None);
        };
        self.locked(|_, current| {
            let Some(mut users) = current else {
                return Ok((None, None));
            };
            let Some(u) = users.users.iter_mut().find(|u| u.name == name) else {
                return Ok((None, None));
            };
            let s = lockout::on_failure(&u.lockout(), now);
            u.set_lockout(s);
            Ok((Some(users), Some(s)))
        })
    }

    /// A good password for `name`: its failures reset (written only when there were some).
    pub fn record_success(&self, name: &str) -> Result<()> {
        let Ok(name) = normalize_name(name) else {
            return Ok(());
        };
        self.locked(|_, current| {
            let Some(mut users) = current else {
                return Ok((None, ()));
            };
            let Some(u) = users.users.iter_mut().find(|u| u.name == name) else {
                return Ok((None, ()));
            };
            // A disabled account is never checked, so it cannot get here; keep it disabled.
            if (u.failures == 0 && u.last_failure.is_none()) || u.disabled {
                return Ok((None, ()));
            }
            u.set_lockout(lockout::State::default());
            Ok((Some(users), ()))
        })
    }
}

/// Write `users` as `<auth>/users.json`.
fn write_users(auth: &Path, users: &Users) -> Result<()> {
    let mut text = serde_json::to_string_pretty(users).context("serializing the accounts")?;
    text.push('\n');
    crate::private_dir::write_file(auth, FILE_NAME, &text)
}

/// What [`Cached::get`] returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Snapshot {
    /// No `users.json`: setup mode.
    Missing,
    /// The accounts.
    Users(Users),
    /// The file cannot be used (the message says why and what to do): nobody signs in.
    Broken(String),
}

/// What identifies one version of the file: modification time, length and (Unix) inode. Every
/// write is a rename of a new file, so the inode alone changes with each one on Unix; NTFS
/// times have 100 ns steps.
type Stamp = Option<(std::time::SystemTime, u64, u64)>;

/// `users.json` for the service, read again only when the file changed (one `stat` per call).
#[derive(Debug)]
pub struct Cached {
    path: PathBuf,
    slot: Mutex<Option<(Stamp, Arc<Snapshot>)>>,
}

impl Cached {
    /// The cache of `data_root`'s accounts (nothing read yet).
    pub fn new(data_root: &Path) -> Self {
        Self {
            path: Store::open(data_root).path(),
            slot: Mutex::new(None),
        }
    }

    fn stamp(&self) -> std::io::Result<Stamp> {
        match std::fs::metadata(&self.path) {
            Ok(m) => {
                #[cfg(unix)]
                let ino = std::os::unix::fs::MetadataExt::ino(&m);
                #[cfg(not(unix))]
                let ino = 0;
                Ok(Some((
                    m.modified().unwrap_or(std::time::UNIX_EPOCH),
                    m.len(),
                    ino,
                )))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// The accounts as on disk now.
    pub fn get(&self) -> Arc<Snapshot> {
        let stamp = match self.stamp() {
            Ok(s) => s,
            Err(e) => {
                return Arc::new(Snapshot::Broken(format!(
                    "could not read {} ({e}), so nobody can sign in to the web UI; check that \
                     the service's account may read it",
                    self.path.display()
                )));
            }
        };
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((s, snap)) = slot.as_ref()
            && *s == stamp
        {
            return Arc::clone(snap);
        }
        let snap = Arc::new(match read_file(&self.path) {
            Ok(None) => Snapshot::Missing,
            Ok(Some(u)) => Snapshot::Users(u),
            Err(e) => Snapshot::Broken(format!("{e:#}")),
        });
        *slot = Some((stamp, Arc::clone(&snap)));
        snap
    }

    /// Forget what was read, so the next [`Self::get`] reads the file.
    pub fn invalidate(&self) {
        *self.slot.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}

#[cfg(test)]
mod tests;
