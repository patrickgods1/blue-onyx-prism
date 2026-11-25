//! Windows: give a folder of secrets a protected DACL, inheriting nothing: Administrators and
//! SYSTEM full control, the account running this full control, the installed service's account
//! read or modify ([`ServiceAccess`]). Used for `<data root>/auth` ([`crate::private_dir`],
//! modify: the service writes its certificate, accounts and sessions there) and the Hugging Face
//! token's folder `tools\hf` ([`crate::resources::hf_token`], read: the service never writes the
//! token).
//!
//! The data root's own ACL lets every local user read (the tray reads the config there), so a
//! secret cannot rely on it. `icacls` does it, since `unsafe` stays in `src/backend/`: every
//! argument is one array element with no shell, and no secret is in any of them.
//!
//! Everything here but `service_account` builds and runs on every OS, so the tests reach it
//! through a fake [`Run`].

use std::ffi::OsString;
use std::path::{Path, PathBuf};

// icacls' `*SID` form: account names are translated on a non-English Windows.
const ADMINISTRATORS: &str = "*S-1-5-32-544";
const SYSTEM: &str = "*S-1-5-18";
const LOCAL_SERVICE: &str = "*S-1-5-19";
const NETWORK_SERVICE: &str = "*S-1-5-20";
const FULL: &str = "(OI)(CI)F";
const READ: &str = "(OI)(CI)RX";
const MODIFY: &str = "(OI)(CI)M";

/// What the service's account gets on a protected folder: read for the Hugging Face token
/// (the service never writes it), modify for `<data root>/auth` ([`crate::private_dir`]),
/// where the service writes its certificate and, later, accounts and sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ServiceAccess {
    Read,
    Modify,
}

impl ServiceAccess {
    fn rights(self) -> &'static str {
        match self {
            Self::Read => READ,
            Self::Modify => MODIFY,
        }
    }
}
/// The most of a tool's answer kept in a message.
const MAX_ANSWER_CHARS: usize = 300;

/// Runs one system tool: its standard output, or why it failed.
pub(crate) type Run<'a> = &'a dyn Fn(&str, &[OsString]) -> Result<String, String>;

/// The name `icacls` should get for the Service Control Manager's account `account`. `None`
/// for LocalSystem (an empty name means it too), which [`grants`] always lists. The built-in
/// accounts become SIDs: their names are translated, and only the SCM takes the English ones
/// everywhere. `.\user` is the SCM's own spelling of a local account; `icacls` cannot look up
/// the dot.
fn service_principal(account: &str) -> Option<String> {
    let account = account.trim();
    match account.to_ascii_lowercase().as_str() {
        "" | "localsystem" | ".\\localsystem" | "nt authority\\system" => None,
        "nt authority\\localservice" | "nt authority\\local service" => {
            Some(LOCAL_SERVICE.to_string())
        }
        "nt authority\\networkservice" | "nt authority\\network service" => {
            Some(NETWORK_SERVICE.to_string())
        }
        _ => Some(account.strip_prefix(".\\").unwrap_or(account).to_string()),
    }
}

/// The grants of the folder: `me` is the account running this, `service` the service's,
/// which gets `access`.
fn grants(service: Option<&str>, me: &str, access: ServiceAccess) -> Vec<String> {
    let mut out = vec![
        format!("{ADMINISTRATORS}:{FULL}"),
        format!("{SYSTEM}:{FULL}"),
    ];
    if me != ADMINISTRATORS && me != SYSTEM {
        out.push(format!("{me}:{FULL}"));
    }
    if let Some(service) = service.filter(|s| *s != me && *s != SYSTEM) {
        out.push(format!("{service}:{}", access.rights()));
    }
    out
}

/// The two `icacls` runs. `/reset` first: `/grant:r` replaces only the accounts it names, so
/// a grant of an earlier login would stay. Inherited access is back only until the second
/// run, and the new secret is written after both.
fn icacls_commands(dir: &Path, grants: &[String]) -> [Vec<OsString>; 2] {
    let reset = vec![dir.as_os_str().to_owned(), "/reset".into(), "/Q".into()];
    let mut protect = vec![
        dir.as_os_str().to_owned(),
        "/inheritance:r".into(),
        "/grant:r".into(),
    ];
    protect.extend(grants.iter().map(OsString::from));
    protect.push("/Q".into());
    [reset, protect]
}

/// The SID in the output of `whoami /user /fo csv /nh`, as `*S-1-...`. The account name before
/// it can hold a comma, and is translated.
fn parse_whoami(output: &str) -> Option<String> {
    let sid = output.trim().rsplit(',').next()?.trim_matches('"');
    let rest = sid.strip_prefix("S-1-")?;
    let shaped = !rest.is_empty()
        && rest.bytes().all(|b| b.is_ascii_digit() || b == b'-')
        && !rest.ends_with('-');
    shaped.then(|| format!("*{sid}"))
}

/// Protect `dir`. `service`: the service's account, `None` with no service installed. When
/// either account is unknown the ACL stays as it is: guessing could lock the service, or the
/// user, out of the secret.
pub(crate) fn protect(
    dir: &Path,
    service: Result<Option<String>, String>,
    run: Run<'_>,
    access: ServiceAccess,
) -> Result<(), String> {
    let service = service.map_err(|e| format!("the service's account is unknown: {e}"))?;
    let whoami = ["/user", "/fo", "csv", "/nh"].map(OsString::from);
    let answer = run("whoami.exe", &whoami)?;
    let me = parse_whoami(&answer)
        .ok_or_else(|| "whoami did not name the current account".to_string())?;
    let principal = service.as_deref().and_then(service_principal);
    for args in icacls_commands(dir, &grants(principal.as_deref(), &me, access)) {
        run("icacls.exe", &args)?;
    }
    Ok(())
}

/// Give `dir` a protected DACL: Administrators and SYSTEM full control, the account running this
/// full control, the installed service's account `access`, nothing inherited, no Users. The
/// shared half of the Hugging Face token's folder and [`crate::private_dir`]. `Err`: why not.
#[cfg(windows)]
pub(crate) fn protect_private_folder(dir: &Path, access: ServiceAccess) -> Result<(), String> {
    protect(dir, service_account(), &run_system_tool, access)
}

/// `name` in System32 of the Windows folder `root`: by absolute path, because an elevated
/// login must not run an `icacls.exe` that happens to lie on the search path.
fn tool_path(root: Option<OsString>, name: &str) -> PathBuf {
    Path::new(&root.unwrap_or_else(|| "C:\\Windows".into()))
        .join("System32")
        .join(name)
}

/// Runs a System32 tool by absolute path (see [`tool_path`]).
#[cfg(windows)]
fn run_system_tool(name: &str, args: &[OsString]) -> Result<String, String> {
    run_at(&tool_path(std::env::var_os("SystemRoot"), name), args)
}

/// Run `exe`: its standard output when it exits with 0, otherwise what it said.
fn run_at(exe: &Path, args: &[OsString]) -> Result<String, String> {
    let out = std::process::Command::new(exe)
        .args(args)
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("{} did not run: {e}", exe.display()))?;
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    if out.status.success() {
        return Ok(stdout);
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    let said: String = format!("{} {}", stdout.trim(), stderr.trim())
        .trim()
        .chars()
        .take(MAX_ANSWER_CHARS)
        .collect();
    Err(format!("{} failed, {}: {said}", exe.display(), out.status))
}

/// The account the installed service runs as: `None` with no service installed. Asking needs
/// no administrator rights.
#[cfg(windows)]
fn service_account() -> Result<Option<String>, String> {
    use crate::service_ctl::WINDOWS_SERVICE_NAME;
    use windows_service::service::ServiceAccess;
    use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};
    const ERROR_SERVICE_DOES_NOT_EXIST: i32 = 1060;
    let manager = ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)
        .map_err(|e| e.to_string())?;
    let service = match manager.open_service(WINDOWS_SERVICE_NAME, ServiceAccess::QUERY_CONFIG) {
        Ok(service) => service,
        Err(windows_service::Error::Winapi(e))
            if e.raw_os_error() == Some(ERROR_SERVICE_DOES_NOT_EXIST) =>
        {
            return Ok(None);
        }
        Err(e) => return Err(e.to_string()),
    };
    let config = service.query_config().map_err(|e| e.to_string())?;
    let account = config.account_name.unwrap_or_default();
    Ok(Some(account.to_string_lossy().into_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[OsString]) -> Vec<String> {
        args.iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    /// Every SID form, so a non-English Windows resolves them; one element per argument, so a
    /// path or an account with a space stays one.
    #[test]
    fn icacls_arguments_are_a_protected_dacl_for_three_accounts() {
        let dir = Path::new("D:/Data Root/tools/hf");
        let grants = grants(
            Some("NT SERVICE\\BlueOnyxPrismService"),
            "*S-1-5-21-1-2-3-1001",
            ServiceAccess::Read,
        );
        let [reset, protect] = icacls_commands(dir, &grants);
        assert_eq!(strings(&reset), [dir.to_str().unwrap(), "/reset", "/Q"]);
        assert_eq!(
            strings(&protect),
            [
                dir.to_str().unwrap(),
                "/inheritance:r",
                "/grant:r",
                "*S-1-5-32-544:(OI)(CI)F",
                "*S-1-5-18:(OI)(CI)F",
                "*S-1-5-21-1-2-3-1001:(OI)(CI)F",
                "NT SERVICE\\BlueOnyxPrismService:(OI)(CI)RX",
                "/Q",
            ]
        );
    }

    /// `<data root>/auth` (`private_dir`): the service writes there, so it gets modify.
    #[test]
    fn a_folder_the_service_writes_grants_it_modify() {
        assert_eq!(
            grants(
                Some("*S-1-5-19"),
                "*S-1-5-21-1-2-3-1001",
                ServiceAccess::Modify
            ),
            [
                "*S-1-5-32-544:(OI)(CI)F",
                "*S-1-5-18:(OI)(CI)F",
                "*S-1-5-21-1-2-3-1001:(OI)(CI)F",
                "*S-1-5-19:(OI)(CI)M",
            ]
        );
    }

    #[test]
    fn an_account_that_already_has_full_control_is_not_granted_twice() {
        // The login ran as the service's account, or as SYSTEM.
        assert_eq!(
            grants(Some("*S-1-5-19"), "*S-1-5-19", ServiceAccess::Read),
            [
                "*S-1-5-32-544:(OI)(CI)F",
                "*S-1-5-18:(OI)(CI)F",
                "*S-1-5-19:(OI)(CI)F",
            ]
        );
        assert_eq!(
            grants(None, "*S-1-5-18", ServiceAccess::Modify),
            ["*S-1-5-32-544:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F"]
        );
    }

    #[test]
    fn the_scm_account_maps_to_what_icacls_resolves_in_any_language() {
        let cases = [
            ("NT AUTHORITY\\LocalService", Some("*S-1-5-19")),
            ("nt authority\\local service", Some("*S-1-5-19")),
            ("NT AUTHORITY\\NetworkService", Some("*S-1-5-20")),
            (
                "NT SERVICE\\BlueOnyxPrismService",
                Some("NT SERVICE\\BlueOnyxPrismService"),
            ),
            ("DOMAIN\\John Smith", Some("DOMAIN\\John Smith")),
            // SCM's spelling of a local account: icacls cannot look up the dot.
            (".\\camera", Some("camera")),
            // Always in the list.
            ("LocalSystem", None),
            ("NT AUTHORITY\\SYSTEM", None),
            ("", None),
        ];
        for (account, want) in cases {
            assert_eq!(service_principal(account).as_deref(), want, "{account}");
        }
    }

    #[test]
    fn whoami_output_gives_the_sid() {
        let out = "\"desktop-1\\pat, jr\",\"S-1-5-21-111-222-333-1001\"\r\n";
        assert_eq!(
            parse_whoami(out).as_deref(),
            Some("*S-1-5-21-111-222-333-1001")
        );
        for bad in [
            "",
            "\"a\",\"b\"",
            "\"a\",\"S-1-\"",
            "\"a\",\"S-1-5-x\"",
            "denied",
        ] {
            assert_eq!(parse_whoami(bad), None, "{bad}");
        }
    }

    type Calls = std::cell::RefCell<Vec<(String, Vec<String>)>>;

    /// A fake of the tools: `whoami` answers, the rest succeed; the calls are kept.
    fn recording(calls: &Calls) -> impl Fn(&str, &[OsString]) -> Result<String, String> {
        move |tool, args| {
            calls.borrow_mut().push((tool.to_string(), strings(args)));
            Ok(if tool == "whoami.exe" {
                "\"pc\\me\",\"S-1-5-21-9-1001\"\r\n".to_string()
            } else {
                String::new()
            })
        }
    }

    #[test]
    fn protect_runs_whoami_then_both_icacls_steps() {
        let calls = Calls::default();
        let run = recording(&calls);
        let service = Ok(Some("NT AUTHORITY\\LocalService".to_string()));
        protect(Path::new("auth"), service, &run, ServiceAccess::Modify).unwrap();
        let calls = calls.borrow();
        let tools: Vec<_> = calls.iter().map(|(tool, _)| tool.as_str()).collect();
        assert_eq!(tools, ["whoami.exe", "icacls.exe", "icacls.exe"]);
        assert_eq!(calls[0].1, ["/user", "/fo", "csv", "/nh"]);
        assert_eq!(calls[1].1, ["auth", "/reset", "/Q"]);
        assert!(calls[2].1.contains(&"*S-1-5-19:(OI)(CI)M".to_string()));
        assert!(
            calls[2]
                .1
                .contains(&"*S-1-5-21-9-1001:(OI)(CI)F".to_string())
        );
    }

    /// Protecting a folder without knowing who needs it could lock the service or the user out
    /// of their own secret: nothing changes, and the error says why.
    #[test]
    fn an_unknown_account_leaves_the_acl_alone() {
        let calls = Calls::default();
        let run = recording(&calls);
        let dir = Path::new("hf");
        let why = protect(
            dir,
            Err("the SCM said no".to_string()),
            &run,
            ServiceAccess::Read,
        )
        .unwrap_err();
        assert!(why.contains("the SCM said no"), "{why}");
        assert!(calls.borrow().is_empty());

        let no_sid: Run<'_> = &|tool, args| {
            calls.borrow_mut().push((tool.to_string(), strings(args)));
            Ok("not csv".to_string())
        };
        let why = protect(dir, Ok(None), no_sid, ServiceAccess::Modify).unwrap_err();
        assert!(why.contains("whoami"), "{why}");
        assert!(calls.borrow().iter().all(|(tool, _)| tool == "whoami.exe"));
    }

    /// Tools are found by absolute path, never through the search path.
    #[test]
    fn tools_come_from_system32() {
        assert_eq!(
            tool_path(Some("D:\\Win".into()), "icacls.exe"),
            Path::new("D:\\Win").join("System32").join("icacls.exe")
        );
        assert_eq!(
            tool_path(None, "icacls.exe"),
            Path::new("C:\\Windows").join("System32").join("icacls.exe")
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_tool_that_fails_reports_what_it_said() {
        let sh = Path::new("/bin/sh");
        let args = |script: &str| ["-c".into(), script.into()];
        assert_eq!(run_at(sh, &args("printf ok")).as_deref(), Ok("ok"));
        let why = run_at(sh, &args("echo denied; echo more >&2; exit 3")).unwrap_err();
        assert!(why.contains("denied") && why.contains("more"), "{why}");
        let why = run_at(Path::new("/no/such/tool"), &[]).unwrap_err();
        assert!(why.contains("/no/such/tool"), "{why}");
    }
}
