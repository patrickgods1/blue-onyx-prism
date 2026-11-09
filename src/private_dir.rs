//! `<data root>/auth/`: the folder of the service's secrets, readable by the service's account and
//! administrators only. Today it holds the HTTPS certificate and its key (`auth/tls/`, see
//! [`crate::tls`]).
//!
//! - **Unix:** the folder and its subfolders are 0700, files 0600 from their first byte
//!   ([`crate::resources::hf_token::write_private`]). Run as root (a `sudo` CLI command), what is
//!   made is handed to the owner of the folder above it, so the service account can still read
//!   and replace it.
//! - **Windows:** the data root's DACL lets every local user read (the tray reads the config
//!   there), so `auth` gets a protected DACL of its own when this creates it: Administrators and
//!   SYSTEM full control, the account creating it full control, the installed service's account
//!   modify, nothing inherited, no Users. The MSI creates the folder with the same DACL first
//!   (`ProtectAuthFolder` in `packaging/windows/main.wxs`), and `scripts/install_service.ps1`
//!   for archive installs. A folder that exists is left as it is: the service (LocalService,
//!   modify) could not rewrite the DACL of the folder the installer made, and resetting it on
//!   every start would only log a failure each time. Its subfolders and files inherit it.
//! - **Writes** ([`write_file`]) go through a temporary file made inside the folder, never in the
//!   data root (which Users can read on Windows), then a rename over the old file, retried for a
//!   short while when Windows reports a sharing violation (an antivirus or indexer holding it).

use crate::resources::hf_token;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Folder name under the data root.
pub const DIR_NAME: &str = "auth";

/// How many times a rename that hit a sharing violation is tried, [`RENAME_PAUSE`] apart.
const RENAME_ATTEMPTS: u32 = 20;
/// Pause between rename attempts (20 x 50 ms: one second at most).
const RENAME_PAUSE: std::time::Duration = std::time::Duration::from_millis(50);

/// `<data_root>/auth`.
pub fn path(data_root: &Path) -> PathBuf {
    data_root.join(DIR_NAME)
}

/// Make sure `<data_root>/auth` exists and is private (see the module docs); its path. A Windows
/// DACL that could not be set is logged as a warning and the folder is used anyway (the files
/// then keep the data root's permissions); everything else that fails is an error.
pub fn ensure(data_root: &Path) -> Result<PathBuf> {
    let (dir, warning) = ensure_with(data_root, &protect_new_folder)?;
    if let Some(w) = warning {
        tracing::warn!("{w}");
    }
    Ok(dir)
}

/// [`ensure`] with `protect` as the Windows protection of a folder made now (`Err`: why not),
/// returning the warning instead of logging it.
fn ensure_with(
    data_root: &Path,
    protect: &dyn Fn(&Path) -> Result<(), String>,
) -> Result<(PathBuf, Option<String>)> {
    std::fs::create_dir_all(data_root).with_context(|| {
        format!(
            "creating the data folder {}; check that this account may write there",
            data_root.display()
        )
    })?;
    let dir = path(data_root);
    let created = make_dir(&dir)?;
    let warning = if created {
        protect(&dir).err().map(|why| acl_warning(&dir, &why))
    } else {
        None
    };
    #[cfg(unix)]
    close_unix(&dir, data_root)?;
    Ok((dir, warning))
}

/// Make sure `<auth>/<name>` exists, private like `auth` (0700 on Unix; on Windows it inherits
/// the folder's DACL); its path.
pub fn ensure_child(auth: &Path, name: &str) -> Result<PathBuf> {
    let dir = auth.join(name);
    make_dir(&dir)?;
    #[cfg(unix)]
    close_unix(&dir, auth)?;
    Ok(dir)
}

/// Create `dir` (its parent exists): true when made now. Something that is not a folder in its
/// place is an error.
fn make_dir(dir: &Path) -> Result<bool> {
    match std::fs::create_dir(dir) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && dir.is_dir() => Ok(false),
        Err(e) => Err(anyhow::anyhow!(
            "creating {}: {e}; check that this account may write in {} and that no file has \
             that name",
            dir.display(),
            dir.parent().unwrap_or(dir).display()
        )),
    }
}

/// The warning of a Windows folder that kept its parent's permissions.
fn acl_warning(dir: &Path, why: &str) -> String {
    format!(
        "could not restrict {} to the service's account and Administrators ({why}); the folder \
         keeps the data folder's permissions, so other accounts on this computer may be able to \
         read the HTTPS key in it. Run the service (or this command) once as an administrator, \
         or reinstall with the MSI, which protects the folder; on a drive without NTFS \
         permissions keep the data folder on an NTFS drive instead",
        dir.display()
    )
}

/// Windows: protect a folder made now ([`hf_token::protect_private_folder`], the service gets
/// modify). Elsewhere: nothing to do here (Unix uses modes, see `close_unix`).
fn protect_new_folder(dir: &Path) -> Result<(), String> {
    #[cfg(windows)]
    {
        hf_token::protect_private_folder(dir, hf_token::ServiceAccess::Modify)
    }
    #[cfg(not(windows))]
    {
        let _ = dir;
        Ok(())
    }
}

/// Unix: `dir` is 0700, and owned like `parent` when this runs as root (a root-owned `auth` in
/// a data root of the service's account would lock the service out of its own secrets). Who we
/// are is read from what we made (no libc call): a folder made by root is root's; a folder that
/// existed and is root's in a data root that is not is given back too, which only root can do
/// (any other account gets EPERM, ignored: such a folder is not this account's to fix).
#[cfg(unix)]
fn close_unix(dir: &Path, parent: &Path) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt, chown};
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).with_context(|| {
        format!(
            "restricting {} to its owner (mode 0700); run this as the account that owns it \
             (the service's)",
            dir.display()
        )
    })?;
    let mine = std::fs::metadata(dir).with_context(|| format!("reading {}", dir.display()))?;
    let up = std::fs::metadata(parent).with_context(|| format!("reading {}", parent.display()))?;
    if let Some((uid, gid)) = hf_token::chown_to(mine.uid(), (up.uid(), up.gid())) {
        chown(dir, Some(uid), Some(gid))
            .with_context(|| format!("giving {} to uid {uid}", dir.display()))?;
    }
    Ok(())
}

/// Write `text` as `<dir>/<name>`, replacing what is there: a temporary file inside `dir` (0600
/// on Unix from its first byte, handed to `dir`'s owner when run as root), then a rename, tried
/// again for up to a second while Windows reports the target in use. A failure leaves the old
/// file and no temporary one.
pub fn write_file(dir: &Path, name: &str, text: &str) -> Result<()> {
    let target = dir.join(name);
    let tmp = dir.join(format!(".{name}.{}.tmp", uuid::Uuid::new_v4().simple()));
    let written = hf_token::write_private(&tmp, text).and_then(|()| {
        #[cfg(unix)]
        give_to_folder_owner(&tmp, dir)?;
        rename_retrying(&tmp, &target, RENAME_ATTEMPTS, RENAME_PAUSE).with_context(|| {
            format!(
                "replacing {}; check that this account may write there and that no other \
                 program holds the file open",
                target.display()
            )
        })
    });
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

/// Run as root, `made` goes to the owner of `dir` (see `close_unix`).
#[cfg(unix)]
fn give_to_folder_owner(made: &Path, dir: &Path) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, chown};
    let me = std::fs::metadata(made)
        .with_context(|| format!("reading {}", made.display()))?
        .uid();
    let owner = std::fs::metadata(dir).with_context(|| format!("reading {}", dir.display()))?;
    if let Some((uid, gid)) = hf_token::chown_to(me, (owner.uid(), owner.gid())) {
        chown(made, Some(uid), Some(gid))
            .with_context(|| format!("giving {} to uid {uid}", made.display()))?;
    }
    Ok(())
}

/// `rename(from, to)`, tried `attempts` times `pause` apart while the error is a Windows sharing
/// violation ([`is_sharing_violation`]); any other error at once.
fn rename_retrying(
    from: &Path,
    to: &Path,
    attempts: u32,
    pause: std::time::Duration,
) -> std::io::Result<()> {
    let mut left = attempts.max(1);
    loop {
        match std::fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(e) if left > 1 && is_sharing_violation(&e) => {
                left -= 1;
                std::thread::sleep(pause);
            }
            Err(e) => return Err(e),
        }
    }
}

/// Whether `e` is Windows saying another process has the file open: ERROR_ACCESS_DENIED (5, a
/// file open without delete sharing), ERROR_SHARING_VIOLATION (32), ERROR_LOCK_VIOLATION (33).
/// Never on other systems, where a rename over an open file works.
fn is_sharing_violation(e: &std::io::Error) -> bool {
    cfg!(windows) && matches!(e.raw_os_error(), Some(5 | 32 | 33))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch(PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn scratch() -> Scratch {
        let d = std::env::temp_dir().join(format!("bo_private_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        Scratch(d)
    }

    fn names(dir: &Path) -> Vec<String> {
        let mut n: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        n.sort();
        n
    }

    #[test]
    fn ensure_makes_the_folder_once_and_protects_only_a_new_one() {
        let root = scratch();
        let data_root = root.0.join("data");
        let calls = std::cell::Cell::new(0);
        let protect = |dir: &Path| {
            assert!(dir.is_dir());
            calls.set(calls.get() + 1);
            Ok(())
        };
        let (dir, warning) = ensure_with(&data_root, &protect).unwrap();
        assert_eq!(dir, data_root.join(DIR_NAME));
        assert_eq!(warning, None);
        let (again, _) = ensure_with(&data_root, &protect).unwrap();
        assert_eq!(again, dir);
        assert_eq!(calls.get(), 1, "an existing folder keeps its DACL");
        // A file in its place is an error, never replaced.
        let other = root.0.join("other");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join(DIR_NAME), "x").unwrap();
        assert!(ensure_with(&other, &protect).is_err());
    }

    #[test]
    fn a_failed_acl_is_a_warning_and_the_folder_is_still_usable() {
        let root = scratch();
        let (dir, warning) =
            ensure_with(&root.0, &|_| Err("Access is denied".to_string())).unwrap();
        let w = warning.unwrap();
        assert!(w.contains("Access is denied"), "{w}");
        assert!(w.contains("administrator"), "{w}");
        write_file(&dir, "x", "y").unwrap();
        assert_eq!(std::fs::read_to_string(dir.join("x")).unwrap(), "y");
    }

    #[test]
    fn write_file_replaces_and_leaves_no_temporary_file() {
        let root = scratch();
        let dir = ensure(&root.0).unwrap();
        let tls = ensure_child(&dir, "tls").unwrap();
        write_file(&tls, "key.pem", "one").unwrap();
        write_file(&tls, "key.pem", "two").unwrap();
        assert_eq!(std::fs::read_to_string(tls.join("key.pem")).unwrap(), "two");
        assert_eq!(names(&tls), ["key.pem"]);
        // A rename that cannot happen (a folder with something in it in the way) keeps it so.
        std::fs::create_dir_all(tls.join("busy").join("inside")).unwrap();
        assert!(write_file(&tls, "busy", "x").is_err());
        assert_eq!(names(&tls), ["busy", "key.pem"]);
        assert_eq!(
            names(&root.0),
            [DIR_NAME],
            "nothing written in the data root"
        );
    }

    #[cfg(unix)]
    #[test]
    fn folders_are_0700_and_files_0600_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        let root = scratch();
        let dir = ensure(&root.0).unwrap();
        assert_eq!(mode(&dir), 0o700);
        // A folder made looser by hand is closed again.
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        ensure(&root.0).unwrap();
        assert_eq!(mode(&dir), 0o700);
        let tls = ensure_child(&dir, "tls").unwrap();
        assert_eq!(mode(&tls), 0o700);
        write_file(&tls, "key.pem", "secret").unwrap();
        assert_eq!(mode(&tls.join("key.pem")), 0o600);
    }

    #[test]
    fn only_windows_sharing_violations_are_retried() {
        let err = |code| std::io::Error::from_raw_os_error(code);
        assert_eq!(is_sharing_violation(&err(32)), cfg!(windows));
        assert_eq!(is_sharing_violation(&err(5)), cfg!(windows));
        assert!(!is_sharing_violation(&std::io::Error::other("x")));
        // A missing source fails at once, with its own error.
        let root = scratch();
        let started = std::time::Instant::now();
        let e = rename_retrying(
            &root.0.join("missing"),
            &root.0.join("to"),
            50,
            std::time::Duration::from_millis(100),
        )
        .unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
    }

    /// The real Windows DACL: nothing for Users (S-1-5-32-545), Administrators and SYSTEM in it,
    /// nothing inherited.
    #[cfg(windows)]
    #[test]
    fn the_windows_folder_has_no_users_entry() {
        let root = scratch();
        let (dir, warning) = ensure_with(&root.0, &protect_new_folder).unwrap();
        assert_eq!(warning, None);
        let script = format!(
            "$a = Get-Acl -LiteralPath '{}'; \"protected=$($a.AreAccessRulesProtected)\"; \
             $a.Access | ForEach-Object {{ \
               $_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value }}",
            dir.display().to_string().replace('\'', "''")
        );
        let out = std::process::Command::new("powershell.exe")
            .args(["-NoProfile", "-NonInteractive", "-Command", &script])
            .output()
            .unwrap();
        assert!(out.status.success(), "{out:?}");
        let text = String::from_utf8_lossy(&out.stdout);
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        assert!(lines.contains(&"protected=True"), "{text}");
        assert!(lines.contains(&"S-1-5-32-544"), "{text}");
        assert!(lines.contains(&"S-1-5-18"), "{text}");
        assert!(!lines.contains(&"S-1-5-32-545"), "{text}");
        assert!(!lines.contains(&"S-1-1-0"), "Everyone: {text}");
    }
}
