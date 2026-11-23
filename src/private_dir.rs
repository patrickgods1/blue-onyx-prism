//! `<data root>/auth/`: the folder of the service's secrets, readable by the service's account and
//! administrators only: the HTTPS certificate and its key (`auth/tls/`, see [`crate::tls`]), the
//! web UI's accounts and the setup code ([`crate::accounts`]) and the admin token
//! ([`crate::admin_token`]). Every read-modify-write in it holds `auth/.lock`
//! ([`crate::accounts::lock`]).
//!
//! - **Unix:** the folder and its subfolders are 0700, files 0600 from their first byte
//!   ([`crate::fsx::Options::private`]). Run as root (a `sudo` CLI command), what is
//!   made is handed to the owner of the folder above it, so the service account can still read
//!   and replace it.
//! - **Windows:** the data root's DACL lets every local user read (the tray reads the config
//!   there), so `auth` gets a protected DACL of its own when this creates it: Administrators and
//!   SYSTEM full control, the account creating it full control, the installed service's account
//!   modify, nothing inherited, no Users. The MSI creates the folder with the same DACL first
//!   (`ProtectAuthFolder` in `packaging/windows/main.wxs`), and `scripts/install_service.ps1`
//!   for archive installs. A folder that exists is protected again only when its DACL is not
//!   protected ([`crate::backend::acl::dacl_protected`], checked once per folder and process):
//!   a start that crashed or failed between creating it and restricting it left it with the data
//!   root's permissions, so the next start repairs it and warns that the secrets written while
//!   it was open (the HTTPS key, the admin token) should be replaced. A protected folder (the
//!   installer's) is left as it is: the service (LocalService, modify) could not rewrite its
//!   DACL. Its subfolders and files inherit it.
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
    let (dir, warning) = ensure_with(data_root, &protect_new_folder, &first_check_protected)?;
    if let Some(w) = warning {
        tracing::warn!("{w}");
    }
    Ok(dir)
}

/// Folders whose DACL [`first_check_protected`] already looked at in this process.
static CHECKED: std::sync::Mutex<Vec<PathBuf>> = std::sync::Mutex::new(Vec::new());

/// [`crate::backend::acl::dacl_protected`] the first time this process asks about `dir`; later
/// calls answer `Ok(true)`. `ensure` runs on every write of a secret, so a folder that cannot be
/// protected (an account without the right) is tried, and warned about, once per process.
fn first_check_protected(dir: &Path) -> Result<bool> {
    {
        let mut checked = CHECKED
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if checked.iter().any(|d| d == dir) {
            return Ok(true);
        }
        checked.push(dir.to_path_buf());
    }
    crate::backend::acl::dacl_protected(dir)
}

/// [`ensure`] with `protect` as the Windows protection of a folder (`Err`: why not) and
/// `protected` telling whether an existing folder's DACL is already protected, returning the
/// warning instead of logging it. A folder made now is protected; an existing one only when
/// `protected` says `Ok(false)` (an `Err` leaves it alone: nothing proves it needs it).
fn ensure_with(
    data_root: &Path,
    protect: &dyn Fn(&Path) -> Result<(), String>,
    protected: &dyn Fn(&Path) -> Result<bool>,
) -> Result<(PathBuf, Option<String>)> {
    std::fs::create_dir_all(data_root).with_context(|| {
        format!(
            "creating the data folder {}; check that this account may write there",
            data_root.display()
        )
    })?;
    let dir = path(data_root);
    let created = make_dir(&dir)?;
    let repair = !created
        && match protected(&dir) {
            Ok(p) => !p,
            Err(e) => {
                tracing::debug!("not checking the permissions of {}: {e:#}", dir.display());
                false
            }
        };
    let warning = if created || repair {
        match protect(&dir) {
            Err(why) => Some(acl_warning(&dir, &why)),
            Ok(()) if repair => Some(repaired_warning(&dir)),
            Ok(()) => None,
        }
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
         read the secrets in it (the HTTPS key, the accounts' password hashes, the admin \
         token). Every start tries again while the folder is not restricted: run \
         `blue-onyx-prism admin-token --rotate --config <the service's config>` once as an \
         administrator (it restricts the folder, then replaces the token) and then \
         `blue-onyx-prism tls-regenerate --config <the service's config>`, or reinstall with \
         the MSI, which protects the folder. On a drive without NTFS permissions keep the data \
         folder on an NTFS drive instead",
        dir.display()
    )
}

/// The warning of an existing Windows folder that was protected now: its secrets were readable
/// until this start.
fn repaired_warning(dir: &Path) -> String {
    format!(
        "{} had kept the data folder's permissions (an earlier start stopped, or failed, before \
         restricting it) and is now restricted to the service's account and Administrators. \
         Until now other accounts on this computer could read the secrets in it: replace the \
         HTTPS key with `blue-onyx-prism tls-regenerate` and the admin token with \
         `blue-onyx-prism admin-token --rotate` (both with `--config <the service's config>`, \
         run as the service's account)",
        dir.display()
    )
}

/// Windows: protect the folder ([`hf_token::protect_private_folder`], the service gets modify;
/// `icacls /reset` then `/inheritance:r`, so it is safe to run again on a folder that has files).
/// Elsewhere: nothing to do here (Unix uses modes, see `close_unix`).
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

/// Write `text` as `<dir>/<name>`, replacing what is there: a temporary file inside `dir`
/// (`.<name>.<uuid>.tmp`, 0600 on Unix from its first byte, handed to `dir`'s owner when run as
/// root; [`crate::fsx::write_atomic_with`] with [`crate::fsx::Options::private`]), then a rename,
/// tried again for up to a second while Windows reports the target in use. A failure leaves the
/// old file and no temporary one.
pub fn write_file(dir: &Path, name: &str, text: &str) -> Result<()> {
    let target = dir.join(name);
    #[cfg(unix)]
    let hand_over = |tmp: &Path| {
        give_to_folder_owner(tmp, dir).map_err(|e| std::io::Error::other(format!("{e:#}")))
    };
    let opts = crate::fsx::Options {
        private: true,
        rename_attempts: RENAME_ATTEMPTS,
        rename_pause: RENAME_PAUSE,
        #[cfg(unix)]
        before_rename: Some(&hand_over),
        ..crate::fsx::Options::default()
    };
    crate::fsx::write_atomic_with(&target, text.as_bytes(), &opts).map_err(|e| match e.step {
        crate::fsx::Step::Rename => anyhow::Error::new(e.source).context(format!(
            "replacing {}; check that this account may write there and that no other program \
             holds the file open",
            target.display()
        )),
        crate::fsx::Step::Create | crate::fsx::Step::CreateParent => {
            anyhow::Error::new(e.source).context(format!("creating {}", e.temp.display()))
        }
        crate::fsx::Step::Write => {
            anyhow::Error::new(e.source).context(format!("writing {}", e.temp.display()))
        }
        crate::fsx::Step::Prepare => anyhow::Error::new(e.source),
    })
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

/// `rename(from, to)`, retried while Windows reports a sharing violation
/// ([`crate::fsx::rename_retrying`], [`crate::fsx::is_sharing_violation`]; kept here for its
/// callers). Also used for folders.
pub(crate) fn rename_retrying(
    from: &Path,
    to: &Path,
    attempts: u32,
    pause: std::time::Duration,
) -> std::io::Result<()> {
    crate::fsx::rename_retrying(from, to, attempts, pause)
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

    /// A probe answering "protected" (the installer's folder, or any Unix folder).
    fn protected(_: &Path) -> Result<bool> {
        Ok(true)
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
        let probed = std::cell::Cell::new(0);
        let probe = |_: &Path| {
            probed.set(probed.get() + 1);
            Ok(true)
        };
        let (dir, warning) = ensure_with(&data_root, &protect, &probe).unwrap();
        assert_eq!(dir, data_root.join(DIR_NAME));
        assert_eq!(warning, None);
        assert_eq!(
            probed.get(),
            0,
            "a folder made now is protected without asking"
        );
        let (again, warning) = ensure_with(&data_root, &protect, &probe).unwrap();
        assert_eq!(again, dir);
        assert_eq!(warning, None);
        assert_eq!(
            calls.get(),
            1,
            "an existing protected folder keeps its DACL"
        );
        assert_eq!(probed.get(), 1);
        // A file in its place is an error, never replaced.
        let other = root.0.join("other");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join(DIR_NAME), "x").unwrap();
        assert!(ensure_with(&other, &protect, &probe).is_err());
    }

    /// RA-044: a start that crashed (or failed) between creating the folder and restricting it
    /// left an existing, unprotected folder; the next start protects it and says which secrets to
    /// replace. A probe that cannot read the DACL leaves the folder alone.
    #[test]
    fn an_existing_unprotected_folder_is_protected_again() {
        let root = scratch();
        // The crash: the folder exists, its DACL was never restricted.
        std::fs::create_dir_all(path(&root.0)).unwrap();
        let calls = std::cell::Cell::new(0);
        let protect = |_: &Path| {
            calls.set(calls.get() + 1);
            Ok(())
        };
        let (dir, warning) = ensure_with(&root.0, &protect, &|_| Ok(false)).unwrap();
        assert_eq!(calls.get(), 1, "the unprotected folder is protected");
        let w = warning.unwrap();
        assert!(w.contains("tls-regenerate"), "{w}");
        assert!(w.contains("admin-token --rotate"), "{w}");
        assert!(w.contains(&dir.display().to_string()), "{w}");
        // Unknown (the DACL cannot be read): nothing proves it needs it, nothing is tried.
        let (_, warning) =
            ensure_with(&root.0, &protect, &|_| Err(anyhow::anyhow!("denied"))).unwrap();
        assert_eq!(warning, None);
        assert_eq!(calls.get(), 1);
        // Protecting it fails (LocalService without WRITE_DAC): the advice says the next start
        // tries again.
        let (_, warning) = ensure_with(&root.0, &|_| Err("Access is denied".to_string()), &|_| {
            Ok(false)
        })
        .unwrap();
        let w = warning.unwrap();
        assert!(w.contains("Access is denied"), "{w}");
        assert!(w.contains("Every start tries again"), "{w}");
    }

    /// The real probe runs once per folder and process (every secret write calls `ensure`).
    #[test]
    fn the_real_probe_asks_once_per_folder() {
        let root = scratch();
        let dir = ensure(&root.0).unwrap();
        // Off Windows the probe always says protected; on Windows `ensure` made and protected
        // it. Either way the next ask is answered from memory.
        assert!(first_check_protected(&dir).unwrap());
        let gone = root.0.join("never-made");
        assert!(first_check_protected(&gone).unwrap_or(false) || cfg!(windows));
        assert!(first_check_protected(&gone).unwrap(), "asked once");
    }

    #[test]
    fn a_failed_acl_is_a_warning_and_the_folder_is_still_usable() {
        let root = scratch();
        let (dir, warning) = ensure_with(
            &root.0,
            &|_| Err("Access is denied".to_string()),
            &protected,
        )
        .unwrap();
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
        assert_eq!(crate::fsx::is_sharing_violation(&err(32)), cfg!(windows));
        assert_eq!(crate::fsx::is_sharing_violation(&err(5)), cfg!(windows));
        assert!(!crate::fsx::is_sharing_violation(&std::io::Error::other(
            "x"
        )));
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

    /// RA-044 on the real OS: an `auth` folder left with the inherited DACL (a crash before
    /// icacls) is protected by the next `ensure`, and then reads as protected.
    #[cfg(windows)]
    #[test]
    fn an_inherited_windows_folder_is_repaired() {
        let root = scratch();
        let dir = path(&root.0);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!crate::backend::acl::dacl_protected(&dir).unwrap());
        let (_, warning) = ensure_with(
            &root.0,
            &protect_new_folder,
            &crate::backend::acl::dacl_protected,
        )
        .unwrap();
        let w = warning.unwrap();
        assert!(w.contains("tls-regenerate"), "{w}");
        assert!(crate::backend::acl::dacl_protected(&dir).unwrap());
        let (_, warning) = ensure_with(
            &root.0,
            &protect_new_folder,
            &crate::backend::acl::dacl_protected,
        )
        .unwrap();
        assert_eq!(warning, None, "a protected folder is left alone");
    }

    /// The real Windows DACL: nothing for Users (S-1-5-32-545), Administrators and SYSTEM in it,
    /// nothing inherited.
    #[cfg(windows)]
    #[test]
    fn the_windows_folder_has_no_users_entry() {
        let root = scratch();
        let (dir, warning) = ensure_with(&root.0, &protect_new_folder, &protected).unwrap();
        assert_eq!(warning, None);
        // Passed as -EncodedCommand (base64 of UTF-16LE): quotes inside a -Command argument do
        // not survive Windows' command-line parsing intact. Stop on the first error, so a failed
        // Get-Acl fails the run with its message instead of printing empty values.
        let script = format!(
            "$ErrorActionPreference = 'Stop'\n\
             $a = Get-Acl -LiteralPath '{}'\n\
             \"protected=$($a.AreAccessRulesProtected)\"\n\
             foreach ($r in $a.Access) {{ \
               $r.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value }}\n",
            dir.display().to_string().replace('\'', "''")
        );
        let encoded = {
            use base64::Engine as _;
            let bytes: Vec<u8> = script.encode_utf16().flat_map(u16::to_le_bytes).collect();
            base64::engine::general_purpose::STANDARD.encode(bytes)
        };
        // Without PSModulePath (see the helper): started from PowerShell 7, as CI's steps are,
        // Windows PowerShell would inherit 7's module path and fail to load Get-Acl's module.
        let out = crate::system_info::windows_powershell_command(
            crate::system_info::windows_powershell_exe(),
        )
        .args(["-NoProfile", "-NonInteractive", "-EncodedCommand", &encoded])
        .output()
        .unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{text}\n{err}");
        let lines: Vec<&str> = text.lines().map(str::trim).collect();
        assert!(lines.contains(&"protected=True"), "{text}");
        assert!(lines.contains(&"S-1-5-32-544"), "{text}");
        assert!(lines.contains(&"S-1-5-18"), "{text}");
        assert!(!lines.contains(&"S-1-5-32-545"), "{text}");
        assert!(!lines.contains(&"S-1-1-0"), "Everyone: {text}");
    }
}
