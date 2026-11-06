//! The admin token: what a remote peer shows to use the admin routes while `allow_remote_admin`
//! is on (`server::auth` checks it). 256 bits of OS randomness as hex in `<data root>/admin-token`,
//! made when first needed: at startup or on a config change with `allow_remote_admin` on
//! ([`ensure_for`]), or by `blue-onyx-prism admin-token`. Nothing here is a route: the service only
//! compares, and only the CLI prints.
//!
//! The file is the one source. The service reads it on every check ([`read`]), so a rotation by
//! another process applies at once and there is no cache to get stale; a file that is missing,
//! unreadable or not exactly one token passes nobody, which also keeps an empty file from ever
//! matching an empty credential.
//!
//! It is written through a temporary file and a rename, with mode 0600 from its first byte on Unix.
//! On Windows it inherits the data folder's DACL (no ACL of its own): whoever may read that folder
//! may read the token. On a default install that takes in local accounts, which already administer
//! the service over loopback without any token.
//!
//! [`AdminToken`] has no `Display` or `Serialize` and its `Debug` hides the value, so it cannot end
//! up in a log line, an HTTP answer or an error by accident; no error here quotes the file.

use crate::config::Config;
use crate::resources::{hf_token, manager};
use anyhow::{Context, Result, bail};
use std::io::Read;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// File name under the data root.
pub const FILE_NAME: &str = "admin-token";
/// Hex characters of a token (256 bits).
const TOKEN_LEN: usize = 64;
/// The most of the file that is read: a token, its newline and room for the file to be too long.
const READ_LIMIT: u64 = 256;

/// A well-formed token. See the module docs for what it refuses to do.
#[derive(Clone, PartialEq, Eq)]
pub struct AdminToken(String);

impl std::fmt::Debug for AdminToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AdminToken(..)")
    }
}

impl AdminToken {
    /// The value, for `admin-token` to print and for the login cookie. Nothing else should
    /// call this.
    pub fn reveal(&self) -> &str {
        &self.0
    }

    /// Whether `presented` is this token, in time that depends on neither: both are hashed and
    /// the digests compared with no early exit.
    pub fn matches(&self, presented: &[u8]) -> bool {
        use sha2::{Digest, Sha256};
        let ours = Sha256::digest(self.0.as_bytes());
        let theirs = Sha256::digest(presented);
        let diff = ours
            .iter()
            .zip(theirs.iter())
            .fold(0u8, |acc, (a, b)| acc | (a ^ b));
        std::hint::black_box(diff) == 0
    }
}

/// Where the token of `config` lives: the data root, the same for the service and the CLI.
pub fn path(config: &Config) -> PathBuf {
    config.data_root().join(FILE_NAME)
}

/// 256 bits from the OS. `uuid` is this crate's source of randomness (`getrandom`, no userspace
/// generator: the build has neither `fast-rng` nor `rng-rand`); one UUID holds 122 random bits,
/// so three (366 bits) go through SHA-256.
fn generate() -> String {
    let mut seed = Vec::with_capacity(48);
    for _ in 0..3 {
        seed.extend_from_slice(uuid::Uuid::new_v4().as_bytes());
    }
    manager::sha256_hex(&seed)
}

/// The token in `path`; `None` when there is no such file.
fn load(path: &Path) -> Result<Option<AdminToken>> {
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(unreadable(path, &e)),
    };
    let mut bytes = Vec::with_capacity(TOKEN_LEN + 2);
    file.take(READ_LIMIT)
        .read_to_end(&mut bytes)
        .map_err(|e| unreadable(path, &e))?;
    let text = bytes.trim_ascii();
    let is_token = text.len() == TOKEN_LEN
        && text
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b));
    if !is_token {
        bail!(
            "{} does not hold an admin token (64 hexadecimal characters), so no remote peer can \
             sign in; it was probably edited by hand. Replace it with `blue-onyx-prism \
             admin-token --rotate`",
            path.display()
        );
    }
    Ok(Some(AdminToken(String::from_utf8_lossy(text).into_owned())))
}

fn unreadable(path: &Path, e: &std::io::Error) -> anyhow::Error {
    anyhow::anyhow!(
        "could not read the admin token in {} ({e}), so no remote peer can sign in; run this as \
         the service's account, or replace the file with `blue-onyx-prism admin-token --rotate`",
        path.display()
    )
}

/// The token in `path`. Missing, unreadable or malformed is an error: the caller refuses.
pub fn read(path: &Path) -> Result<AdminToken> {
    match load(path)? {
        Some(t) => Ok(t),
        None => bail!(
            "{} does not exist, so no remote peer can sign in; it is made when \
             allow_remote_admin is turned on, or by `blue-onyx-prism admin-token`",
            path.display()
        ),
    }
}

/// Run as root (`sudo ... admin-token --rotate`), the data root's owner has to own the file, or
/// the service could not read it and every remote peer would be refused.
#[cfg(unix)]
fn hand_over(made: &Path, dir: &Path) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, chown};
    let me = std::fs::metadata(made)
        .with_context(|| format!("reading {}", made.display()))?
        .uid();
    let root = std::fs::metadata(dir).with_context(|| format!("reading {}", dir.display()))?;
    if let Some((uid, gid)) = hf_token::chown_to(me, (root.uid(), root.gid())) {
        chown(made, Some(uid), Some(gid))
            .with_context(|| format!("giving {} to uid {uid}", made.display()))?;
    }
    Ok(())
}

/// Replace `path` with a new token (a new file: whatever was there, even unusable, goes). A
/// failure leaves the old token working, and no copy of the new one behind.
pub fn rotate(path: &Path) -> Result<AdminToken> {
    let dir = path
        .parent()
        .context("the admin token file has no parent folder")?;
    std::fs::create_dir_all(dir).with_context(|| {
        format!(
            "creating {}; check that this account may write there",
            dir.display()
        )
    })?;
    let value = generate();
    let tmp = dir.join(format!(
        ".{FILE_NAME}.{}.tmp",
        uuid::Uuid::new_v4().simple()
    ));
    let written = hf_token::write_private(&tmp, &format!("{value}\n")).and_then(|()| {
        #[cfg(unix)]
        hand_over(&tmp, dir)?;
        std::fs::rename(&tmp, path).with_context(|| {
            format!(
                "replacing {}; check that this account may write there",
                path.display()
            )
        })
    });
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(AdminToken(value))
}

/// The token in `path`, made if there is none (true: made now). A file that exists and cannot be
/// used is an error and stays: replacing it silently would lock out whoever holds the token,
/// and it may only be unreadable for this account.
pub fn ensure(path: &Path) -> Result<(AdminToken, bool)> {
    if let Some(t) = load(path)? {
        return Ok((t, false));
    }
    rotate(path)?;
    // Another process may have made one at the same moment: what is on disk counts.
    Ok((read(path)?, true))
}

/// Make the token when `config` needs one. A failure is logged and leaves remote administration
/// closed (the check fails on a missing file); detection and this machine are not affected.
pub fn ensure_for(config: &Config) {
    if !config.allow_remote_admin {
        return;
    }
    let path = path(config);
    match ensure(&path) {
        Ok((_, true)) => info!(path = %path.display(), "created the admin token"),
        Ok((_, false)) => {}
        Err(e) => warn!("remote administration stays closed: {e:#}"),
    }
}

/// The command that prints the token of the service using the config at `config`.
pub fn read_command(config: &Path) -> String {
    hf_token::cli_command_for(config, "admin-token")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir() -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("bo_token_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn hex64(t: &str) -> bool {
        t.len() == 64
            && t.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    }

    #[test]
    fn ensure_creates_once_and_keeps_the_token() {
        let d = dir();
        let p = d.join(FILE_NAME);
        let (a, created) = ensure(&p).unwrap();
        assert!(created);
        assert!(hex64(a.reveal()), "not 256 bits of hex");
        let (b, created) = ensure(&p).unwrap();
        assert!(!created);
        assert_eq!(a.reveal(), b.reveal());
        assert_eq!(read(&p).unwrap().reveal(), a.reveal());
        let left: Vec<_> = std::fs::read_dir(&d).unwrap().collect();
        assert_eq!(left.len(), 1, "a temporary file stayed behind");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn ensure_makes_the_data_root_when_it_is_missing() {
        let d = dir();
        let p = d.join("not").join("yet").join(FILE_NAME);
        assert!(ensure(&p).unwrap().1);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn ensure_never_replaces_a_file_it_cannot_use() {
        let d = dir();
        let p = d.join(FILE_NAME);
        std::fs::write(&p, "not a token").unwrap();
        let e = format!("{:#}", ensure(&p).unwrap_err());
        assert!(e.contains("--rotate"), "{e}");
        assert!(!e.contains("not a token"), "{e}");
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "not a token");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn tokens_differ_and_rotate_replaces_the_file() {
        let d = dir();
        let p = d.join(FILE_NAME);
        let (a, _) = ensure(&p).unwrap();
        let b = rotate(&p).unwrap();
        assert_ne!(a.reveal(), b.reveal());
        assert!(hex64(b.reveal()));
        assert_eq!(read(&p).unwrap().reveal(), b.reveal());
        // An unusable file does not stop a rotation: it is the way out.
        std::fs::write(&p, "").unwrap();
        assert!(hex64(rotate(&p).unwrap().reveal()));
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 1);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn the_file_is_private_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let d = dir();
        let p = d.join(FILE_NAME);
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        ensure(&p).unwrap();
        assert_eq!(mode(&p), 0o600);
        rotate(&p).unwrap();
        assert_eq!(mode(&p), 0o600);
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// Reading fails closed: nothing but one well-formed token counts, so an empty file can
    /// never match an empty credential.
    #[test]
    fn read_refuses_everything_but_one_token() {
        let d = dir();
        let p = d.join(FILE_NAME);
        assert!(read(&p).is_err(), "missing");
        let good = "ab".repeat(32);
        for bad in [
            String::new(),
            "\n".into(),
            "abc".into(),
            format!("{good}a"),
            good.replace('a', "z"),
            format!("{good}\n{good}"),
            "A".repeat(10_000),
        ] {
            std::fs::write(&p, &bad).unwrap();
            let e = format!("{:#}", read(&p).unwrap_err());
            assert!(bad.len() < 8 || !e.contains(&bad), "{e}");
        }
        std::fs::write(&p, format!("{good}\n")).unwrap();
        assert_eq!(read(&p).unwrap().reveal(), good);
        // A directory in its place, what an unreadable file looks like on every OS.
        std::fs::remove_file(&p).unwrap();
        std::fs::create_dir(&p).unwrap();
        assert!(read(&p).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn matches_needs_the_exact_token() {
        let d = dir();
        let p = d.join(FILE_NAME);
        let (t, _) = ensure(&p).unwrap();
        let value = t.reveal().to_string();
        assert!(t.matches(value.as_bytes()));
        assert!(!t.matches(b""));
        assert!(!t.matches(&value.as_bytes()[..63]));
        assert!(!t.matches(format!("{value}0").as_bytes()));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn debug_never_shows_the_value() {
        let d = dir();
        let p = d.join(FILE_NAME);
        let (t, _) = ensure(&p).unwrap();
        assert!(!format!("{t:?}").contains(t.reveal()));
        std::fs::remove_dir_all(&d).unwrap();
    }
}
