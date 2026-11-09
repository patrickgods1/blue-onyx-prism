//! What [`super::sessions`] and [`super::devices`] share: the random id a browser keeps in a
//! cookie, stored only as its SHA-256, and the small JSON file of such hashes in `auth/`.
//!
//! **Ids** are 32 random bytes from [`crate::secret`] as 64 lowercase hex characters, which is
//! what the cookie holds. A presented value that is not exactly that is refused before anything
//! is hashed or looked up ([`digest_of`]).
//!
//! **Lookup** is by the SHA-256 of the presented bytes in a `HashMap`. That is constant-time
//! enough: the map compares digests, so any timing it leaks is about how a digest of the
//! attacker's choosing relates to a stored digest, and turning a stored digest into a cookie
//! that matches it means finding a SHA-256 preimage. The ids themselves are never compared.
//!
//! **Files** ([`Persist`]) are written whole through [`crate::private_dir::write_file`] under
//! `auth/.lock` (held for the write only; the text is built before it is taken), serialized by a
//! writer mutex so an older snapshot never lands after a newer one. A write that fails is logged
//! at most once every [`WARN_EVERY_SECS`] and changes nothing in memory, which stays
//! authoritative. Reads never quote the file: a damaged one is described by the kind of problem
//! and its line and column only.

use crate::sync::LockExt;
use anyhow::{Context, Result};
use sha2::{Digest as _, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

/// Random bytes in an id.
pub const ID_BYTES: usize = 32;
/// Hex characters of an id (what the cookie holds).
pub const ID_HEX: usize = 2 * ID_BYTES;
/// The largest file read: far more than 4096 entries of about 200 bytes.
const READ_LIMIT: u64 = 8 * 1024 * 1024;
/// At most one warning about failed writes of one file per this long.
pub const WARN_EVERY_SECS: u64 = 300;
/// The longest wait for `auth/.lock` before a write gives up (sessions and devices are not
/// precious, and a sign-in waits for this write).
const LOCK_WAIT: std::time::Duration = std::time::Duration::from_secs(5);

/// The SHA-256 of an id's 64 hex characters: the key in memory and, as hex, in the file.
pub type Digest = [u8; 32];

/// A well-formed random id. No `Display` or `Serialize`; `Debug` hides it.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Secret(String);

impl Secret {
    /// A new random id.
    pub(crate) fn generate() -> Self {
        Self(crate::secret::random_hex(ID_BYTES))
    }

    /// `presented` as an id when it is exactly 64 lowercase hex characters.
    pub(crate) fn parse(presented: &[u8]) -> Option<Self> {
        well_formed(presented)
            .then(|| std::str::from_utf8(presented).ok().map(str::to_owned))
            .flatten()
            .map(Self)
    }

    /// The value, for the cookie writer only.
    pub(crate) fn reveal(&self) -> &str {
        &self.0
    }

    /// Its SHA-256.
    pub(crate) fn digest(&self) -> Digest {
        Sha256::digest(self.0.as_bytes()).into()
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Secret(..)")
    }
}

/// Whether `presented` is exactly 64 lowercase hex characters (the only form ids are issued in).
pub fn well_formed(presented: &[u8]) -> bool {
    presented.len() == ID_HEX
        && presented
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
}

/// The lookup key of `presented`, `None` (nothing hashed) when it is not well formed.
pub fn digest_of(presented: &[u8]) -> Option<Digest> {
    well_formed(presented).then(|| Sha256::digest(presented).into())
}

/// `d` as 64 lowercase hex characters.
pub fn digest_hex(d: &Digest) -> String {
    crate::secret::hex(d)
}

/// 64 lowercase hex characters as a digest.
pub fn parse_digest_hex(text: &str) -> Option<Digest> {
    if !well_formed(text.as_bytes()) {
        return None;
    }
    let nibble = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        _ => None,
    };
    let mut out = [0u8; 32];
    let (pairs, _) = text.as_bytes().as_chunks::<2>();
    for (slot, [hi, lo]) in out.iter_mut().zip(pairs) {
        *slot = (nibble(*hi)? << 4) | nibble(*lo)?;
    }
    Some(out)
}

/// A serde error as words that never quote the input: its kind, line and column.
pub fn json_problem(e: &serde_json::Error) -> String {
    let kind = match e.classify() {
        serde_json::error::Category::Io => "unreadable",
        serde_json::error::Category::Syntax => "not valid JSON",
        serde_json::error::Category::Data => "unexpected content",
        serde_json::error::Category::Eof => "cut short",
    };
    format!("{kind} at line {} column {}", e.line(), e.column())
}

/// What reading a file found.
#[derive(Debug)]
pub enum Loaded {
    /// No file.
    Missing,
    /// Its bytes.
    Bytes(Vec<u8>),
    /// It could not be read (the message says why, without content).
    Failed(String),
}

/// Read `path`, at most [`READ_LIMIT`] bytes.
pub fn read(path: &Path) -> Loaded {
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Loaded::Missing,
        Err(e) => return Loaded::Failed(format!("could not open it ({e})")),
    };
    let mut bytes = Vec::new();
    if let Err(e) = file.take(READ_LIMIT + 1).read_to_end(&mut bytes) {
        return Loaded::Failed(format!("could not read it ({e})"));
    }
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > READ_LIMIT {
        return Loaded::Failed(format!(
            "it is larger than {} MiB",
            READ_LIMIT / 1024 / 1024
        ));
    }
    Loaded::Bytes(bytes)
}

/// Change counters of a store, kept inside its state mutex.
#[derive(Debug, Default, Clone, Copy)]
pub struct Versions {
    /// Bumped by every change.
    pub changed: u64,
    /// The last change that must be written at the next flush (not only after
    /// `FLUSH_EVERY_SECS`).
    pub forced: u64,
    /// The last change on disk.
    pub written: u64,
}

impl Versions {
    /// A change that may wait for the next timed flush.
    pub fn lazy(&mut self) {
        self.changed = self.changed.saturating_add(1);
    }

    /// A change the next flush writes at once.
    pub fn force(&mut self) {
        self.lazy();
        self.forced = self.changed;
    }

    /// Whether a flush at `now` writes: something changed, and the write is forced (by the call
    /// or by a change), or `every` seconds passed since `last` (or the clock went back).
    pub fn due(&self, force: bool, now: u64, last: u64, every: u64) -> bool {
        if self.changed == self.written {
            return false;
        }
        force || self.forced > self.written || now < last || now - last >= every
    }
}

/// When the file was last written or tried, and the warning throttle.
#[derive(Debug)]
pub struct Writer {
    /// Unix seconds of the last write attempt (the load time before the first).
    pub last: u64,
    warned: Option<u64>,
    dropped: u64,
}

/// One file of hashes in `auth/` (see the module docs).
#[derive(Debug)]
pub struct Persist {
    data_root: PathBuf,
    name: &'static str,
    writer: Mutex<Writer>,
}

impl Persist {
    /// `<data_root>/auth/<name>`, loaded at `now` (no I/O).
    pub fn new(data_root: &Path, name: &'static str, now: u64) -> Self {
        Self {
            data_root: data_root.to_path_buf(),
            name,
            writer: Mutex::new(Writer {
                last: now,
                warned: None,
                dropped: 0,
            }),
        }
    }

    /// The file's path.
    pub fn path(&self) -> PathBuf {
        crate::private_dir::path(&self.data_root).join(self.name)
    }

    /// The writer: hold it from the snapshot to [`Self::write`], so writes land in order.
    pub fn writer(&self) -> MutexGuard<'_, Writer> {
        self.writer.lock_recover()
    }

    /// Write `text` at `now` (the writer held): the folder made private, `auth/.lock` held for
    /// the rename only. A failure is logged (throttled) and returned as `false`.
    pub fn write(&self, w: &mut Writer, text: &str, now: u64) -> bool {
        w.last = now;
        let done = crate::private_dir::ensure(&self.data_root).and_then(|auth| {
            let _held = super::lock::lock_within(&auth, LOCK_WAIT)?;
            crate::private_dir::write_file(&auth, self.name, text)
        });
        match done {
            Ok(()) => true,
            Err(e) => {
                self.warn(w, now, &e);
                false
            }
        }
    }

    fn warn(&self, w: &mut Writer, now: u64, e: &anyhow::Error) {
        if w.warned
            .is_some_and(|t| now >= t && now - t < WARN_EVERY_SECS)
        {
            w.dropped = w.dropped.saturating_add(1);
            return;
        }
        w.warned = Some(now);
        let dropped = std::mem::take(&mut w.dropped);
        tracing::warn!(
            "could not save {} ({e:#}); sign-ins keep working from memory but are lost at the \
             next restart. Check the free space and the permissions of the data folder{}",
            self.path().display(),
            if dropped > 0 {
                format!(" ({dropped} more failures since the last warning)")
            } else {
                String::new()
            }
        );
    }
}

/// `text` with a newline, from a serializable value.
pub fn to_text<T: serde::Serialize>(value: &T, what: &str) -> Result<String> {
    let mut text =
        serde_json::to_string_pretty(value).with_context(|| format!("serializing {what}"))?;
    text.push('\n');
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_64_lowercase_hex_and_hide_in_debug() {
        let s = Secret::generate();
        assert!(well_formed(s.reveal().as_bytes()));
        assert_eq!(format!("{s:?}"), "Secret(..)");
        assert_eq!(Secret::parse(s.reveal().as_bytes()), Some(s.clone()));
        assert_eq!(digest_of(s.reveal().as_bytes()), Some(s.digest()));
        let upper = s.reveal().to_ascii_uppercase();
        for bad in [
            &b""[..],
            b"abc",
            &s.reveal().as_bytes()[..63],
            format!("{}0", s.reveal()).as_bytes(),
            upper.as_bytes(),
            &[0xffu8; 64][..],
            "g".repeat(64).as_bytes(),
        ] {
            assert!(Secret::parse(bad).is_none());
            assert!(digest_of(bad).is_none());
        }
        let d = s.digest();
        assert_eq!(parse_digest_hex(&digest_hex(&d)), Some(d));
        assert_eq!(parse_digest_hex("zz"), None);
    }

    #[test]
    fn versions_decide_when_to_write() {
        let mut v = Versions::default();
        assert!(!v.due(true, 10, 0, 120));
        v.lazy();
        assert!(!v.due(false, 119, 0, 120));
        assert!(v.due(false, 120, 0, 120));
        assert!(v.due(true, 1, 0, 120));
        // A clock that went back writes rather than waiting for it to catch up.
        assert!(v.due(false, 5, 100, 120));
        v.written = v.changed;
        v.force();
        assert!(v.due(false, 1, 0, 120));
    }
}
