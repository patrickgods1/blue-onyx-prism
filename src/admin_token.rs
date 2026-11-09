//! The API token (called the admin token in file and command names): what scripts and Prometheus
//! send as `Authorization: Bearer` to the HTTPS port instead of signing in with an account
//! (`server::auth` checks it). 256 bits of OS randomness ([`crate::secret`]) as hex in
//! `<data root>/auth/admin-token`, made at every start when missing ([`ensure_for`]) or by
//! `blue-onyx-prism admin-token`. Nothing here is a route: the service only compares, and only
//! the CLI prints.
//!
//! Earlier versions also accepted it in a `bop_admin` cookie over plain HTTP when
//! `allow_remote_admin` was on, so it may have been seen on the network: the first start of this
//! version with that key in the config rotates it once ([`rotate_once_after_remote_admin`]).
//!
//! The file is the one source. The service reads it on every check ([`read`]), so a rotation by
//! another process applies at once and there is no cache to get stale; a file that is missing,
//! unreadable or not exactly one token passes nobody, which also keeps an empty file from ever
//! matching an empty credential.
//!
//! It lives in the private `auth` folder ([`crate::private_dir`]: 0700/0600 on Unix, a protected
//! DACL without Users on Windows) and is written through a temporary file there and a rename.
//! Earlier versions kept it in the data root (`<data root>/admin-token`); [`prepare`], which the
//! service (at start and on config changes) and the CLI go through, moves such a file into
//! `auth/` under `auth/.lock` (see [`migrate`]).
//!
//! [`AdminToken`] has no `Display` or `Serialize` and its `Debug` hides the value, so it cannot end
//! up in a log line, an HTTP answer or an error by accident; no error here quotes the file.

use crate::config::Config;
use crate::resources::hf_token;
use anyhow::{Context, Result, bail};
use std::io::Read;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// File name in `<data root>/auth/` (and, from earlier versions, in the data root itself).
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

/// Where the token of `config` lives: `<data root>/auth/admin-token`, the same for the service
/// and the CLI. No I/O; [`prepare`] makes the folder and moves an old file.
pub fn path(config: &Config) -> PathBuf {
    path_in(&config.data_root())
}

/// The token's path under the data root `data_root` (the service checks the one of the data root
/// it started with, `accounts::Auth::data_root`).
pub fn path_in(data_root: &Path) -> PathBuf {
    crate::private_dir::path(data_root).join(FILE_NAME)
}

/// Where earlier versions kept it: `<data root>/admin-token`.
pub fn old_path(config: &Config) -> PathBuf {
    config.data_root().join(FILE_NAME)
}

/// 256 bits from the OS as 64 hex characters.
fn generate() -> String {
    crate::secret::random_hex(TOKEN_LEN / 2)
}

/// What [`migrate`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Migrated {
    /// No old file: nothing to do.
    Nothing,
    /// The old file was moved into `auth/`.
    Moved,
    /// Both exist: the new one counts, the old one was left alone (a warning names it).
    BothKept,
}

/// Move a token file from the data root (`old`) to `new` in the auth folder, under
/// `auth/.lock`: when `new` is missing and `old` exists. On Windows (and wherever a rename
/// fails) the content is written anew through [`crate::private_dir::write_file`] and the old file
/// deleted, because a renamed file keeps its own permissions, which on Windows are the data
/// root's (Users may read) rather than the auth folder's. On Unix a rename keeps the content and
/// mode 0600 is set again. When both exist the new one wins and the old one is left as it is.
pub fn migrate(old: &Path, new: &Path) -> Result<Migrated> {
    let auth = new
        .parent()
        .context("the admin token file has no parent folder")?;
    if !old.exists() {
        return Ok(Migrated::Nothing);
    }
    let _held = crate::accounts::lock::lock(auth)?;
    if !old.exists() {
        return Ok(Migrated::Nothing);
    }
    if new.exists() {
        return Ok(Migrated::BothKept);
    }
    let renamed = if cfg!(windows) {
        false
    } else {
        std::fs::rename(old, new).is_ok()
    };
    if renamed {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(new, std::fs::Permissions::from_mode(0o600)).with_context(
                || format!("restricting {} to its owner (mode 0600)", new.display()),
            )?;
        }
        return Ok(Migrated::Moved);
    }
    let text = std::fs::read_to_string(old).map_err(|e| unreadable(old, &e))?;
    crate::private_dir::write_file(auth, FILE_NAME, &text)?;
    std::fs::remove_file(old).with_context(|| {
        format!(
            "removing the old admin token {} after copying it to {}; delete it by hand",
            old.display(),
            new.display()
        )
    })?;
    Ok(Migrated::Moved)
}

/// Make the auth folder of `config` and move an old token into it ([`migrate`]): the token's
/// path. Both the service and the CLI go through this before they use the token.
pub fn prepare(config: &Config) -> Result<PathBuf> {
    crate::private_dir::ensure(&config.data_root())?;
    let (old, new) = (old_path(config), path(config));
    match migrate(&old, &new)? {
        Migrated::Nothing => {}
        Migrated::Moved => info!(
            from = %old.display(),
            to = %new.display(),
            "moved the admin token into the private auth folder"
        ),
        Migrated::BothKept => warn!(
            "{} is an admin token from an earlier version, but {} exists and is the one in use; \
             delete the old file (it is not read any more)",
            old.display(),
            new.display()
        ),
    }
    Ok(new)
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
            "{} does not hold an admin token (64 hexadecimal characters), so no request passes \
             with the API token; it was probably edited by hand. Replace it with `blue-onyx-prism \
             admin-token --rotate`",
            path.display()
        );
    }
    Ok(Some(AdminToken(String::from_utf8_lossy(text).into_owned())))
}

fn unreadable(path: &Path, e: &std::io::Error) -> anyhow::Error {
    anyhow::anyhow!(
        "could not read the admin token in {} ({e}), so no request passes with the API token; run this as \
         the service's account, or replace the file with `blue-onyx-prism admin-token --rotate`",
        path.display()
    )
}

/// The token in `path`. Missing, unreadable or malformed is an error: the caller refuses.
pub fn read(path: &Path) -> Result<AdminToken> {
    match load(path)? {
        Some(t) => Ok(t),
        None => bail!(
            "{} does not exist, so no request passes with the API token; it is made at the \
             service's start, or by `blue-onyx-prism admin-token`",
            path.display()
        ),
    }
}

/// Replace `path` with a new token (a new file: whatever was there, even unusable, goes). Its
/// folder must exist ([`prepare`] makes the real one). A failure leaves the old token working,
/// and no copy of the new one behind. Run as root, the file goes to the folder's owner.
pub fn rotate(path: &Path) -> Result<AdminToken> {
    let dir = path
        .parent()
        .context("the admin token file has no parent folder")?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .context("the admin token file has no name")?;
    let value = generate();
    crate::private_dir::write_file(dir, name, &format!("{value}\n"))?;
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

/// Make the token when there is none, after moving a token of an earlier version into `auth/`
/// ([`prepare`]). A failure is logged and leaves the token closed (the check fails on a missing
/// file); sign-in with an account and detection are not affected.
pub fn ensure_for(config: &Config) {
    let path = match prepare(config) {
        Ok(p) => p,
        Err(e) => {
            warn!("the API token is not available (Bearer requests are refused): {e:#}");
            return;
        }
    };
    match ensure(&path) {
        Ok((_, true)) => info!(path = %path.display(), "created the API token"),
        Ok((_, false)) => {}
        Err(e) => warn!("the API token is not available (Bearer requests are refused): {e:#}"),
    }
}

/// The marker in `auth/` that the one-time rotation of [`rotate_once_after_remote_admin`] was
/// done.
pub const ROTATED_MARKER: &str = ".token-rotated-v1";

/// When `config` (as read from its file) still has the retired `allow_remote_admin: true` and
/// this data root has no [`ROTATED_MARKER`] yet: rotate the token once (earlier versions sent it
/// in a cookie over plain HTTP, so it may have been seen on the network), then write the marker.
/// True when it rotated now. Without that key, or with the marker, nothing happens.
pub fn rotate_once_after_remote_admin(config: &Config) -> Result<bool> {
    if !config.allow_remote_admin {
        return Ok(false);
    }
    let path = prepare(config)?;
    let auth = path
        .parent()
        .context("the admin token file has no parent folder")?;
    if auth.join(ROTATED_MARKER).exists() {
        return Ok(false);
    }
    rotate(&path)?;
    crate::private_dir::write_file(
        auth,
        ROTATED_MARKER,
        "The API token was rotated once by the upgrade that retired allow_remote_admin.\n",
    )?;
    Ok(true)
}

/// `blue-onyx-prism admin-token [--rotate]` for `config`: the token to print (made when there is
/// none, replaced with `rotate`), after an old file was moved ([`prepare`]).
pub fn for_command(config: &Config, rotate_it: bool) -> Result<AdminToken> {
    let path = prepare(config)?;
    if rotate_it {
        rotate(&path)
    } else {
        Ok(ensure(&path)?.0)
    }
}

/// The command that prints the token of the service using the config at `config`.
pub fn read_command(config: &Path) -> String {
    hf_token::cli_command_for(config, "admin-token")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch folder, removed on drop (also when the test fails). A removal that fails, such
    /// as antivirus holding a file on Windows, never fails a passing test.
    struct Scratch(std::path::PathBuf);

    impl std::ops::Deref for Scratch {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.0
        }
    }

    impl AsRef<Path> for Scratch {
        fn as_ref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn dir() -> Scratch {
        let d = std::env::temp_dir().join(format!("bo_token_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        Scratch(d)
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
    }

    fn config_in(root: &Path) -> Config {
        Config {
            download_dir: Some(root.to_path_buf()),
            ..Default::default()
        }
    }

    #[test]
    fn prepare_makes_the_private_folder_when_it_is_missing() {
        let d = dir();
        let config = config_in(&d.join("not").join("yet"));
        let p = prepare(&config).unwrap();
        assert_eq!(p, path(&config));
        assert_eq!(p.parent().unwrap().file_name().unwrap(), "auth");
        assert!(p.parent().unwrap().is_dir());
        assert!(ensure(&p).unwrap().1);
    }

    #[test]
    fn an_old_token_moves_into_the_auth_folder_unchanged() {
        let d = dir();
        let config = config_in(&d);
        let value = "ab".repeat(32);
        std::fs::write(old_path(&config), format!("{value}\n")).unwrap();
        let p = prepare(&config).unwrap();
        assert!(!old_path(&config).exists(), "the old file is gone");
        assert_eq!(std::fs::read_to_string(&p).unwrap(), format!("{value}\n"));
        assert_eq!(read(&p).unwrap().reveal(), value);
        // The command prints that same token, and keeps doing so.
        assert_eq!(for_command(&config, false).unwrap().reveal(), value);
        assert_eq!(for_command(&config, false).unwrap().reveal(), value);
        // Nothing is left in either folder but the token, the lock and the auth folder.
        let mut top: Vec<String> = std::fs::read_dir(&d)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        top.sort();
        assert_eq!(top, ["auth"]);
        let mut inside: Vec<String> = std::fs::read_dir(p.parent().unwrap())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        inside.sort();
        assert_eq!(inside, [".lock", FILE_NAME]);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
    }

    #[test]
    fn with_both_files_the_new_one_wins_and_the_old_stays() {
        let d = dir();
        let config = config_in(&d);
        let (old, new) = ("ab".repeat(32), "cd".repeat(32));
        let p = prepare(&config).unwrap();
        std::fs::write(&p, &new).unwrap();
        std::fs::write(old_path(&config), &old).unwrap();
        assert_eq!(migrate(&old_path(&config), &p).unwrap(), Migrated::BothKept);
        prepare(&config).unwrap();
        assert_eq!(std::fs::read_to_string(old_path(&config)).unwrap(), old);
        assert_eq!(for_command(&config, false).unwrap().reveal(), new);
        assert_eq!(migrate(&d.join("none"), &p).unwrap(), Migrated::Nothing);
    }

    /// The service makes the token at every start, and moves an old one first.
    #[test]
    fn the_service_ensures_the_token_and_moves_an_old_one() {
        let d = dir();
        let config = config_in(&d);
        ensure_for(&config);
        let made = read(&path(&config)).unwrap();
        ensure_for(&config);
        assert_eq!(
            read(&path(&config)).unwrap().reveal(),
            made.reveal(),
            "kept"
        );
        let d = dir();
        let config = config_in(&d);
        std::fs::write(old_path(&config), "ab".repeat(32)).unwrap();
        ensure_for(&config);
        assert!(!old_path(&config).exists());
        assert_eq!(read(&path(&config)).unwrap().reveal(), "ab".repeat(32));
    }

    /// An old config with `allow_remote_admin: true` rotates the token once (marker), never again,
    /// and a config without the key never does.
    #[test]
    fn remote_admin_in_an_old_config_rotates_the_token_once() {
        let d = dir();
        let mut config = config_in(&d);
        std::fs::write(old_path(&config), "ab".repeat(32)).unwrap();
        ensure_for(&config);
        assert!(
            !rotate_once_after_remote_admin(&config).unwrap(),
            "key absent"
        );
        assert_eq!(read(&path(&config)).unwrap().reveal(), "ab".repeat(32));
        config.allow_remote_admin = true;
        assert!(rotate_once_after_remote_admin(&config).unwrap());
        let new = read(&path(&config)).unwrap();
        assert_ne!(new.reveal(), "ab".repeat(32));
        let marker = path(&config).parent().unwrap().join(ROTATED_MARKER);
        assert!(marker.is_file());
        assert!(
            !std::fs::read_to_string(&marker)
                .unwrap()
                .contains(new.reveal())
        );
        assert!(!rotate_once_after_remote_admin(&config).unwrap(), "once");
        assert_eq!(read(&path(&config)).unwrap().reveal(), new.reveal());
    }

    #[test]
    fn rotating_through_the_command_replaces_the_moved_token() {
        let d = dir();
        let config = config_in(&d);
        std::fs::write(old_path(&config), "ab".repeat(32)).unwrap();
        let t = for_command(&config, true).unwrap();
        assert_ne!(t.reveal(), "ab".repeat(32));
        assert!(!old_path(&config).exists());
        assert_eq!(read(&path(&config)).unwrap().reveal(), t.reveal());
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
    }

    #[test]
    fn debug_never_shows_the_value() {
        let d = dir();
        let p = d.join(FILE_NAME);
        let (t, _) = ensure(&p).unwrap();
        assert!(!format!("{t:?}").contains(t.reveal()));
    }
}
