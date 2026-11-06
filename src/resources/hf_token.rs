//! The Hugging Face token of gated downloads ([`catalog::GATED`], SAM 3): where it comes from
//! ([`lookup`]: `HF_TOKEN`, then the login of `blue-onyx-prism hf-login` in
//! `<data root>/tools/hf/token`, then the `hf` CLI's file), what an answer of Hugging Face means
//! ([`refusal`], [`probe`]), `hf-login` / `hf-logout` ([`login`], [`logout`]) and the command the
//! page shows ([`login_command`]).
//!
//! The value leaves this module only through [`HfToken::secret`], for one `Authorization` header to
//! the Hugging Face host. [`HfToken`] has no `Display` or `Serialize`, and its `Debug` prints the
//! source only, so a token cannot end up in a log line, an HTTP response or an error by accident.
//! It is never an argument of a command (`Cmd::display()` logs argv, shells keep history), and the
//! helpers stay offline: they get no token.
//!
//! The environment is an injected lookup ([`Env`]), so the tests never touch the process's.

use super::catalog::{self, Gated, Resource};
use super::export::train::TrainPaths;
use super::manager;
use crate::config::Config;
use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The only host a token is sent to.
pub const HF_HOST: &str = "huggingface.co";
/// Who a token belongs to.
pub const WHOAMI_URL: &str = "https://huggingface.co/api/whoami-v2";
/// Where an account makes a token.
pub const TOKENS_URL: &str = "https://huggingface.co/settings/tokens";
/// Header Hugging Face adds to a refusal from a gated repo.
pub const ERROR_CODE_HEADER: &str = "x-error-code";
/// Its value.
pub const GATED_REPO: &str = "GatedRepo";

/// How long [`probe`] and [`whoami`] wait for an answer.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Name of the CLI, beside the service exe.
const CLI_NAME: &str = "blue-onyx-prism";

/// Reads one environment variable.
pub type Env<'a> = &'a dyn Fn(&str) -> Option<String>;

/// The process's environment.
fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Where a token was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenSource {
    /// `HF_TOKEN`.
    Env,
    /// `<data root>/tools/hf/token`, written by `hf-login`.
    Saved,
    /// The file of `hf auth login`.
    HfCli,
}

impl TokenSource {
    /// For people: the page and the error messages.
    pub const fn label(self) -> &'static str {
        match self {
            TokenSource::Env => "the HF_TOKEN environment variable",
            TokenSource::Saved => "the saved login (hf-login)",
            TokenSource::HfCli => "the hf CLI's login file",
        }
    }
}

/// A token and where it came from. See the module docs for what it refuses to do.
#[derive(Clone, PartialEq, Eq)]
pub struct HfToken {
    value: String,
    source: TokenSource,
}

impl std::fmt::Debug for HfToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfToken")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl HfToken {
    /// `raw` without the whitespace around it. What cannot go into an HTTP header is refused;
    /// the reason does not quote it.
    pub fn new(raw: &str, source: TokenSource) -> Result<Self, String> {
        let value = raw.trim();
        if value.is_empty() {
            return Err("it is empty".to_string());
        }
        if !value.bytes().all(|b| b.is_ascii_graphic()) {
            return Err("it has characters a token cannot have".to_string());
        }
        Ok(Self {
            value: value.to_string(),
            source,
        })
    }

    /// The value, for the `Authorization` header. Nothing else should call this.
    pub(crate) fn secret(&self) -> &str {
        &self.value
    }

    pub fn source(&self) -> TokenSource {
        self.source
    }

    /// Tells two tokens apart without keeping or showing either.
    pub(crate) fn fingerprint(&self) -> String {
        manager::sha256_hex(self.value.as_bytes())
    }
}

/// What [`lookup`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lookup {
    Found(HfToken),
    /// The first source that exists does not give a token. It wins over the sources below it: a
    /// login the service cannot read should be fixed, not silently replaced by another account.
    Unreadable {
        source: TokenSource,
        /// The file (None: `HF_TOKEN`).
        path: Option<PathBuf>,
        reason: String,
    },
    Missing,
}

impl Lookup {
    pub fn token(&self) -> Option<&HfToken> {
        match self {
            Lookup::Found(t) => Some(t),
            _ => None,
        }
    }

    /// What is wrong with an [`Lookup::Unreadable`] source and what to do; None otherwise.
    pub fn problem(&self) -> Option<String> {
        let Lookup::Unreadable {
            source,
            path,
            reason,
        } = self
        else {
            return None;
        };
        Some(match path {
            Some(p) => format!(
                "{} ({}) cannot be read: {reason}. This account needs to read it: fix its \
                 permissions, or sign in again with the command below",
                source.label(),
                p.display()
            ),
            None => format!(
                "{} is not usable: {reason}. Correct it or unset it",
                source.label()
            ),
        })
    }
}

/// Where `hf-login` keeps the token under `data_root`.
pub fn saved_token_path(data_root: &Path) -> PathBuf {
    data_root.join("tools").join("hf").join("token")
}

/// The files of `hf auth login`, as `huggingface_hub` finds them, first one first.
fn hf_cli_paths(env: Env<'_>) -> Vec<PathBuf> {
    let set = |name: &str| env(name).filter(|v| !v.trim().is_empty());
    let mut out = Vec::new();
    if let Some(p) = set("HF_TOKEN_PATH") {
        out.push(PathBuf::from(p));
    }
    if let Some(p) = set("HF_HOME") {
        out.push(Path::new(&p).join("token"));
    }
    if let Some(p) = set("XDG_CACHE_HOME") {
        out.push(Path::new(&p).join("huggingface").join("token"));
    }
    if let Some(home) = set(if cfg!(windows) { "USERPROFILE" } else { "HOME" }) {
        out.push(
            Path::new(&home)
                .join(".cache")
                .join("huggingface")
                .join("token"),
        );
    }
    out
}

/// The token in `path`. None: no such file, or an empty one (`hf auth logout` may leave that).
fn from_file(path: &Path, source: TokenSource) -> Option<Lookup> {
    let unreadable = |reason: String| {
        Some(Lookup::Unreadable {
            source,
            path: Some(path.to_path_buf()),
            reason,
        })
    };
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => return unreadable(e.to_string()),
    };
    let Ok(text) = std::str::from_utf8(&bytes) else {
        return unreadable("it is not text".to_string());
    };
    if text.trim().is_empty() {
        return None;
    }
    match HfToken::new(text, source) {
        Ok(t) => Some(Lookup::Found(t)),
        Err(reason) => unreadable(reason),
    }
}

/// What the files below `HF_TOKEN` hold.
fn lookup_files(env: Env<'_>, data_root: &Path) -> Lookup {
    from_file(&saved_token_path(data_root), TokenSource::Saved)
        .or_else(|| {
            hf_cli_paths(env)
                .iter()
                .find_map(|p| from_file(p, TokenSource::HfCli))
        })
        .unwrap_or(Lookup::Missing)
}

/// The token to use, with `env` as the environment: `HF_TOKEN`, then `hf-login`'s file, then the
/// `hf` CLI's (`HF_TOKEN_PATH`, `HF_HOME/token`, `XDG_CACHE_HOME/huggingface/token`,
/// `~/.cache/huggingface/token`). An unset or blank variable and a missing or empty file are
/// skipped; see [`Lookup::Unreadable`] for the rest.
pub fn lookup_with(env: Env<'_>, data_root: &Path) -> Lookup {
    match env("HF_TOKEN").filter(|v| !v.trim().is_empty()) {
        Some(raw) => match HfToken::new(&raw, TokenSource::Env) {
            Ok(t) => Lookup::Found(t),
            Err(reason) => Lookup::Unreadable {
                source: TokenSource::Env,
                path: None,
                reason,
            },
        },
        None => lookup_files(env, data_root),
    }
}

/// [`lookup_with`] on the process's environment.
pub fn lookup(data_root: &Path) -> Lookup {
    lookup_with(&process_env, data_root)
}

/// `HF_TOKEN` is in use and hides a login on disk: signing in again will not change the token
/// this process sends.
pub fn env_overrides_login_with(env: Env<'_>, data_root: &Path) -> bool {
    lookup_with(env, data_root)
        .token()
        .is_some_and(|t| t.source() == TokenSource::Env)
        && lookup_files(env, data_root) != Lookup::Missing
}

/// [`env_overrides_login_with`] on the process's environment.
pub fn env_overrides_login(data_root: &Path) -> bool {
    env_overrides_login_with(&process_env, data_root)
}

/// Write `value` to a new file only this account can read: mode 0600 on Unix, on Windows the
/// protected ACL of its folder (`folder_acl`). `path` must not exist yet: nothing is ever
/// written through a file or link already there.
fn write_private(path: &Path, value: &str) -> Result<()> {
    use std::io::Write;
    let mut open = std::fs::OpenOptions::new();
    open.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        open.mode(0o600);
    }
    let mut f = open
        .open(path)
        .with_context(|| format!("creating {}", path.display()))?;
    f.write_all(value.as_bytes())
        .and_then(|()| f.sync_all())
        .with_context(|| format!("writing {}", path.display()))
}

/// Run as root, the data root's owner has to own what was made, or the service cannot read the
/// token (nor add its own folders to a `tools` that root created).
#[cfg(any(unix, test))]
fn chown_to(me: u32, data_root_owner: (u32, u32)) -> Option<(u32, u32)> {
    (me == 0 && data_root_owner.0 != 0).then_some(data_root_owner)
}

#[cfg(unix)]
fn hand_over(data_root: &Path, made: &Path) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt, chown};
    let dir = made.parent().unwrap_or(data_root);
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
        .with_context(|| format!("restricting {}", dir.display()))?;
    // Our own uid, without a libc call: we just created this file.
    let me = std::fs::metadata(made)
        .with_context(|| format!("reading {}", made.display()))?
        .uid();
    let root_meta =
        std::fs::metadata(data_root).with_context(|| format!("reading {}", data_root.display()))?;
    let Some((uid, gid)) = chown_to(me, (root_meta.uid(), root_meta.gid())) else {
        return Ok(());
    };
    let tools = dir.parent().unwrap_or(data_root);
    for p in [tools, dir, made] {
        let owned_by_root = std::fs::metadata(p).is_ok_and(|m| m.uid() == 0);
        if owned_by_root {
            chown(p, Some(uid), Some(gid))
                .with_context(|| format!("giving {} to uid {uid}", p.display()))?;
        }
    }
    Ok(())
}

/// The Windows half of [`save`]: who may reach the token's folder.
///
/// The data root's own ACL lets every local user read (the tray reads the config there), so the
/// token cannot rely on it. `tools\hf` gets a protected DACL, inheriting nothing: Administrators
/// and SYSTEM (full control, the uninstaller and `hf-logout` of another admin), the account running
/// this login (full control), the service's account (read: it never writes the token). `icacls`
/// does it, since `unsafe` stays in `src/backend/`: every argument is one array element with no
/// shell, and the token is in none of them.
///
/// Everything here but `service_account` builds and runs on every OS, so the tests reach it
/// through a fake [`Run`].
#[cfg(any(windows, test))]
mod folder_acl {
    use std::ffi::OsString;
    use std::path::{Path, PathBuf};

    // icacls' `*SID` form: account names are translated on a non-English Windows.
    const ADMINISTRATORS: &str = "*S-1-5-32-544";
    const SYSTEM: &str = "*S-1-5-18";
    const LOCAL_SERVICE: &str = "*S-1-5-19";
    const NETWORK_SERVICE: &str = "*S-1-5-20";
    const FULL: &str = "(OI)(CI)F";
    const READ: &str = "(OI)(CI)RX";
    /// The most of a tool's answer kept in a message.
    const MAX_ANSWER_CHARS: usize = 300;

    /// Runs one system tool: its standard output, or why it failed.
    pub(super) type Run<'a> = &'a dyn Fn(&str, &[OsString]) -> Result<String, String>;

    /// The name `icacls` should get for the Service Control Manager's account `account`. `None`
    /// for LocalSystem (an empty name means it too), which [`grants`] always lists. The built-in
    /// accounts become SIDs: their names are translated, and only the SCM takes the English ones
    /// everywhere. `.\user` is the SCM's own spelling of a local account; `icacls` cannot look up
    /// the dot.
    pub(super) fn service_principal(account: &str) -> Option<String> {
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

    /// The grants of the folder: `me` is the account running this, `service` the service's.
    pub(super) fn grants(service: Option<&str>, me: &str) -> Vec<String> {
        let mut out = vec![
            format!("{ADMINISTRATORS}:{FULL}"),
            format!("{SYSTEM}:{FULL}"),
        ];
        if me != ADMINISTRATORS && me != SYSTEM {
            out.push(format!("{me}:{FULL}"));
        }
        if let Some(service) = service.filter(|s| *s != me && *s != SYSTEM) {
            out.push(format!("{service}:{READ}"));
        }
        out
    }

    /// The two `icacls` runs. `/reset` first: `/grant:r` replaces only the accounts it names, so
    /// a grant of an earlier login would stay. Inherited access is back only until the second
    /// run, and the new token is written after both.
    pub(super) fn icacls_commands(dir: &Path, grants: &[String]) -> [Vec<OsString>; 2] {
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
    pub(super) fn parse_whoami(output: &str) -> Option<String> {
        let sid = output.trim().rsplit(',').next()?.trim_matches('"');
        let rest = sid.strip_prefix("S-1-")?;
        let shaped = !rest.is_empty()
            && rest.bytes().all(|b| b.is_ascii_digit() || b == b'-')
            && !rest.ends_with('-');
        shaped.then(|| format!("*{sid}"))
    }

    /// Protect `dir`. `service`: the service's account, `None` with no service installed. When
    /// either account is unknown the ACL stays as it is: guessing could lock the service, or the
    /// user, out of the token.
    pub(super) fn protect(
        dir: &Path,
        service: Result<Option<String>, String>,
        run: Run<'_>,
    ) -> Result<(), String> {
        let service = service.map_err(|e| format!("the service's account is unknown: {e}"))?;
        let whoami = ["/user", "/fo", "csv", "/nh"].map(OsString::from);
        let answer = run("whoami.exe", &whoami)?;
        let me = parse_whoami(&answer)
            .ok_or_else(|| "whoami did not name the current account".to_string())?;
        let principal = service.as_deref().and_then(service_principal);
        for args in icacls_commands(dir, &grants(principal.as_deref(), &me)) {
            run("icacls.exe", &args)?;
        }
        Ok(())
    }

    /// [`protect`], with its failure as what the user reads.
    pub(super) fn protect_or_warn(
        dir: &Path,
        service: Result<Option<String>, String>,
        run: Run<'_>,
    ) -> Option<String> {
        let why = protect(dir, service, run).err()?;
        Some(format!(
            "could not restrict {} to the service's account, Administrators and you ({why}). The \
             token is saved, but the folder keeps the permissions of its parent, so other \
             accounts on this computer may be able to read it. Run this command again from an \
             elevated terminal (Run as administrator); if that fails too, the drive may not \
             support NTFS permissions: keep the data folder on an NTFS drive.",
            dir.display()
        ))
    }

    /// `name` in System32 of the Windows folder `root`: by absolute path, because an elevated
    /// login must not run an `icacls.exe` that happens to lie on the search path.
    pub(super) fn tool_path(root: Option<OsString>, name: &str) -> PathBuf {
        Path::new(&root.unwrap_or_else(|| "C:\\Windows".into()))
            .join("System32")
            .join(name)
    }

    /// Run `exe`: its standard output when it exits with 0, otherwise what it said.
    pub(super) fn run_at(exe: &Path, args: &[OsString]) -> Result<String, String> {
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
    pub(super) fn service_account() -> Result<Option<String>, String> {
        use crate::service_ctl::WINDOWS_SERVICE_NAME;
        use windows_service::service::ServiceAccess;
        use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};
        const ERROR_SERVICE_DOES_NOT_EXIST: i32 = 1060;
        let manager = ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)
            .map_err(|e| e.to_string())?;
        let service = match manager.open_service(WINDOWS_SERVICE_NAME, ServiceAccess::QUERY_CONFIG)
        {
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
}

/// Protect the token's folder: the warning for the user when that failed.
#[cfg(windows)]
fn protect_token_folder(dir: &Path) -> Option<String> {
    let run = |name: &str, args: &[std::ffi::OsString]| {
        folder_acl::run_at(
            &folder_acl::tool_path(std::env::var_os("SystemRoot"), name),
            args,
        )
    };
    folder_acl::protect_or_warn(dir, folder_acl::service_account(), &run)
}

/// Unix needs nothing first: the file is 0600 from its first byte, and `hand_over` closes the
/// folder.
#[cfg(not(windows))]
fn protect_token_folder(_dir: &Path) -> Option<String> {
    None
}

/// A saved login.
#[derive(Debug)]
pub struct Saved {
    pub path: PathBuf,
    /// The login stands, but its folder could not be protected: tell the user.
    pub warning: Option<String>,
}

/// Save `token` as the login under `data_root`.
pub fn save(data_root: &Path, token: &HfToken) -> Result<Saved> {
    save_with(data_root, token, &protect_token_folder)
}

/// [`save`] with `protect` as the folder's protection, run before the token is on disk. The old
/// login stays until the rename replaces it, so a failure at any point leaves it working, and no
/// copy of the new token behind.
fn save_with(
    data_root: &Path,
    token: &HfToken,
    protect: &dyn Fn(&Path) -> Option<String>,
) -> Result<Saved> {
    let path = saved_token_path(data_root);
    let dir = path
        .parent()
        .context("the token file has no parent directory")?;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let warning = protect(dir);
    if let Some(w) = &warning {
        tracing::warn!("{w}");
    }
    let tmp = dir.join(format!(".token.{}.tmp", uuid::Uuid::new_v4()));
    let replaced = write_private(&tmp, token.secret()).and_then(|()| {
        #[cfg(unix)]
        hand_over(data_root, &tmp)?;
        std::fs::rename(&tmp, &path).with_context(|| format!("writing {}", path.display()))
    });
    if let Err(e) = replaced {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(Saved { path, warning })
}

/// Delete the saved login. Returns whether there was one.
pub fn remove_saved(data_root: &Path) -> Result<bool> {
    let path = saved_token_path(data_root);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e).with_context(|| format!("removing {}", path.display())),
    }
}

// ---------------------------------------------------------------------------------------------
// What Hugging Face answered
// ---------------------------------------------------------------------------------------------

/// Why a gated repo said no. Measured on `facebook/sam3`: no token or a bad one = 401, a valid
/// token of an account not approved = 403, both with `X-Error-Code: GatedRepo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    NeedsLogin,
    NeedsAccess,
}

/// Whether the account may read a gated repo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Granted,
    Refused(Refusal),
}

/// The refusal in a response: a 401 or 403 from the Hugging Face host, or from anywhere with the
/// gated header. A 403 from the CDN without that header (an expired signed URL) is not one.
pub fn refusal(status: u16, gated_header: bool, from_hf: bool) -> Option<Refusal> {
    if !(gated_header || from_hf) {
        return None;
    }
    match status {
        401 => Some(Refusal::NeedsLogin),
        403 => Some(Refusal::NeedsAccess),
        _ => None,
    }
}

/// `headers` carry `X-Error-Code: GatedRepo`.
pub fn has_gated_header(headers: &reqwest::header::HeaderMap) -> bool {
    headers
        .get(ERROR_CODE_HEADER)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.trim() == GATED_REPO)
}

/// `url`'s host is exactly `host`.
pub fn url_on_host(url: &str, host: &str) -> bool {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.eq_ignore_ascii_case(host)))
        .unwrap_or(false)
}

/// `req` with the token when `url` is on `host` and there is a token. A redirect to another host
/// loses the header: reqwest strips it (a test holds it to that).
pub(crate) fn signed(
    req: reqwest::RequestBuilder,
    url: &str,
    token: Option<&HfToken>,
    host: &str,
) -> reqwest::RequestBuilder {
    match token {
        Some(t) if url_on_host(url, host) => req.bearer_auth(t.secret()),
        _ => req,
    }
}

/// What to tell the user about a refusal of `gated`. `token`: the source of the token that was
/// sent, if one was.
pub fn refusal_message(refusal: Refusal, gated: &Gated, token: Option<TokenSource>) -> String {
    match (refusal, token) {
        (Refusal::NeedsLogin, None) => format!(
            "Hugging Face asks for a login for this repo and no token is set up. Accept the {} \
             at {}, then sign in with the hf-login command the Auto-label tab's SAM 3 panel shows \
             with its full path (a read-only token is enough) and try again",
            gated.license, gated.repo_url
        ),
        (Refusal::NeedsLogin, Some(s)) => format!(
            "Hugging Face refused the token from {} (expired, revoked or mistyped). Make a new \
             read-only token at {TOKENS_URL}, sign in with the hf-login command the Auto-label \
             tab's SAM 3 panel shows with its full path and try again",
            s.label()
        ),
        (Refusal::NeedsAccess, source) => format!(
            "the account behind {} has no access to this repo yet. Request it at {} ({}) and try \
             again once it is approved. A fine-grained token also needs read access to public \
             gated repos",
            source.map_or("the token", TokenSource::label),
            gated.repo_url,
            gated.license
        ),
    }
}

/// Ask Hugging Face whether the account behind `token` can read `url` (the pinned `config.json`
/// of a gated repo). Reads one byte. `Err`: no usable answer (network, a server error).
pub async fn probe(
    client: &reqwest::Client,
    url: &str,
    token: Option<&HfToken>,
    host: &str,
) -> Result<Access> {
    let req = client
        .get(url)
        .header(reqwest::header::RANGE, "bytes=0-0")
        .timeout(REQUEST_TIMEOUT);
    let resp = signed(req, url, token, host)
        .send()
        .await
        .map_err(reqwest::Error::without_url)
        .with_context(|| format!("asking {host} about the gated repo"))?;
    let status = resp.status();
    if status.is_success() {
        return Ok(Access::Granted);
    }
    let from_hf = resp.url().host_str() == Some(host);
    match refusal(status.as_u16(), has_gated_header(resp.headers()), from_hf) {
        Some(r) => Ok(Access::Refused(r)),
        None => bail!("{host} answered {status}; try again in a moment"),
    }
}

/// The account name of `token`, from `url` ([`WHOAMI_URL`]). `Err`: the token was refused, or no
/// usable answer.
pub async fn whoami(client: &reqwest::Client, url: &str, token: &HfToken) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct Who {
        name: String,
    }
    let resp = client
        .get(url)
        .bearer_auth(token.secret())
        .timeout(REQUEST_TIMEOUT)
        .send()
        .await
        .map_err(reqwest::Error::without_url)
        .context("asking Hugging Face who this token belongs to")?;
    let status = resp.status();
    if status == reqwest::StatusCode::UNAUTHORIZED {
        bail!(
            "Hugging Face refused this token (expired, revoked or mistyped); nothing was saved. \
             Make a read-only token at {TOKENS_URL} and run this again"
        );
    }
    if !status.is_success() {
        bail!("Hugging Face answered {status}; nothing was saved. Try again in a moment");
    }
    let who: Who = resp
        .json()
        .await
        .map_err(reqwest::Error::without_url)
        .context("reading Hugging Face's answer; nothing was saved")?;
    Ok(who.name)
}

// ---------------------------------------------------------------------------------------------
// SAM 3 on disk
// ---------------------------------------------------------------------------------------------

/// The SAM 3 pin.
pub fn sam3_resource() -> Option<&'static Resource> {
    catalog::autolabel_weights(catalog::SAM3_KEY)
}

/// The pinned file that tells whether an account has access.
pub fn probe_url(res: &Resource) -> Option<&'static str> {
    res.parts
        .iter()
        .find(|p| p.file_name == "config.json")
        .map(|p| p.url)
}

/// Every file of `res` is in `dir` at its pinned size. The hash is the manager's to check, but a
/// folder that passes this needs no download, so no login.
pub fn parts_present(res: &Resource, dir: &Path) -> bool {
    res.parts.iter().all(|p| {
        std::fs::metadata(dir.join(p.file_name)).is_ok_and(|m| m.is_file() && m.len() == p.size)
    })
}

/// SAM 3 is installed: verified by hash, its manifest written.
pub fn sam3_ready(paths: &TrainPaths) -> bool {
    sam3_resource()
        .is_some_and(|r| manager::is_installed(r, &crate::training::pyenv::weights_dir(paths, r)))
}

/// [`parts_present`] for SAM 3: installed, or adoptable with no network.
pub fn sam3_parts_present(paths: &TrainPaths) -> bool {
    sam3_resource()
        .is_some_and(|r| parts_present(r, &crate::training::pyenv::weights_dir(paths, r)))
}

// ---------------------------------------------------------------------------------------------
// The command shown
// ---------------------------------------------------------------------------------------------

/// Where the login has to happen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandPlace<'a> {
    /// `windows`, `linux`, `macos`.
    pub os: &'a str,
    /// Folder of the running exe: the CLI sits beside the service exe, which has no CLI, and it
    /// is not on PATH.
    pub exe_dir: &'a str,
    pub config: &'a str,
    /// Account of the running service (Unix), for `sudo -u`.
    pub account: Option<&'a str>,
}

/// The login command and how to run it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LoginCommand {
    pub command: String,
    pub note: String,
}

/// Quoted for PowerShell, the default Windows terminal: `'` keeps `$` and backticks literal.
fn powershell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

/// Quoted for a POSIX shell.
fn sh_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// An MSI install: the program under Program Files or the data under ProgramData.
fn windows_system_install(place: &CommandPlace<'_>) -> bool {
    let inside = |p: &str, what: &str| p.to_ascii_lowercase().contains(what);
    inside(place.exe_dir, "\\program files") || inside(place.config, "\\programdata\\")
}

/// The command that signs in for the service at `place`. `--config` goes before the subcommand:
/// it is not a global option. Docker has no shape here: `docker exec -it <name> ...` or `HF_TOKEN`,
/// in the docs.
pub fn login_command(place: &CommandPlace<'_>) -> LoginCommand {
    const HIDDEN: &str = "It asks for the token and does not show what you paste; a read-only \
                          token is enough.";
    if place.os == "windows" {
        let exe = format!(
            "{}\\{CLI_NAME}.exe",
            place.exe_dir.trim_end_matches(['\\', '/'])
        );
        let how = if windows_system_install(place) {
            "Run it in PowerShell from an elevated terminal (Run as administrator)."
        } else {
            "Run it in PowerShell."
        };
        return LoginCommand {
            command: format!(
                "& {} --config {} hf-login",
                powershell_quote(&exe),
                powershell_quote(place.config)
            ),
            note: format!("{how} {HIDDEN}"),
        };
    }
    let exe = format!("{}/{CLI_NAME}", place.exe_dir.trim_end_matches('/'));
    let fallback = if place.os == "macos" {
        "_blueonyx"
    } else {
        "blueonyx"
    };
    let account = place
        .account
        .map(str::trim)
        .filter(|a| !a.is_empty())
        .unwrap_or(fallback);
    let sudo = if account == "root" {
        "sudo".to_string()
    } else {
        format!("sudo -u {}", sh_quote(account))
    };
    LoginCommand {
        command: format!(
            "{sudo} {} --config {} hf-login",
            sh_quote(&exe),
            sh_quote(place.config)
        ),
        note: format!(
            "Run it as the service's account, so the service can read the token. {HIDDEN}"
        ),
    }
}

/// [`login_command`] for this process, serving the config at `config`.
pub fn login_command_for(config: &Path) -> LoginCommand {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(crate::exe_dir);
    let config = std::path::absolute(config).unwrap_or_else(|_| config.to_path_buf());
    let account = process_env("USER").or_else(|| process_env("LOGNAME"));
    login_command(&CommandPlace {
        os: std::env::consts::OS,
        exe_dir: &exe_dir.to_string_lossy(),
        config: &config.to_string_lossy(),
        account: account.as_deref(),
    })
}

// ---------------------------------------------------------------------------------------------
// hf-login, hf-logout
// ---------------------------------------------------------------------------------------------

/// What a login talks to.
#[derive(Debug, Clone, Copy)]
struct Endpoints<'a> {
    whoami: &'a str,
    probe: &'a str,
    host: &'a str,
}

/// What a login found.
#[derive(Debug)]
struct LoginReport {
    name: String,
    path: PathBuf,
    /// Err: the access check itself failed; the login still stands.
    access: Result<Access, String>,
    /// The token's folder could not be protected ([`Saved::warning`]).
    warning: Option<String>,
}

impl LoginReport {
    /// The answer of `hf-login`.
    fn text(&self, gated: Option<&Gated>) -> String {
        let access = match &self.access {
            Ok(Access::Granted) => "yes".to_string(),
            Ok(Access::Refused(_)) => match gated {
                Some(g) => format!("not yet (request it at {}, {})", g.repo_url, g.license),
                None => "not yet".to_string(),
            },
            Err(e) => format!("unknown ({e})"),
        };
        let saved = format!(
            "Logged in as {}, SAM 3 access: {access}\nSaved to {}. A read-only token is enough.",
            self.name,
            self.path.display()
        );
        match &self.warning {
            Some(w) => format!("{saved}\nwarning: {w}"),
            None => saved,
        }
    }
}

/// Validate `token`, save it under `data_root`, then see what it can read.
async fn sign_in(
    client: &reqwest::Client,
    at: Endpoints<'_>,
    data_root: &Path,
    token: &HfToken,
) -> Result<LoginReport> {
    let name = whoami(client, at.whoami, token).await?;
    let Saved { path, warning } = save(data_root, token)?;
    let access = probe(client, at.probe, Some(token), at.host)
        .await
        .map_err(|e| format!("{e:#}"));
    Ok(LoginReport {
        name,
        path,
        access,
        warning,
    })
}

/// The token from the terminal with no echo, or from stdin when piped. Never from argv.
fn read_secret() -> Result<String> {
    use std::io::{BufRead, IsTerminal};
    eprintln!("Paste a Hugging Face token ({TOKENS_URL}). A read-only token is enough.");
    if std::io::stdin().is_terminal() {
        rpassword::prompt_password("Token (hidden): ").context("reading the token")
    } else {
        let mut line = String::new();
        std::io::stdin()
            .lock()
            .read_line(&mut line)
            .context("reading the token from stdin")?;
        Ok(line)
    }
}

/// `blue-onyx-prism hf-login`.
pub fn login(config: &Config) -> Result<()> {
    let raw = read_secret()?;
    let token = HfToken::new(&raw, TokenSource::Saved)
        .map_err(|why| anyhow!("that is not a token: {why}"))?;
    let root = config.data_root();
    let res = sam3_resource().context("this build has no SAM 3 pin")?;
    let probe = probe_url(res).context("the SAM 3 pin has no config.json")?;
    let client = manager::http_client()?;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("creating the async runtime")?;
    let report = rt.block_on(sign_in(
        &client,
        Endpoints {
            whoami: WHOAMI_URL,
            probe,
            host: HF_HOST,
        },
        &root,
        &token,
    ))?;
    println!("{}", report.text(catalog::gated(res.id)));
    if env_overrides_login(&root) {
        println!("note: HF_TOKEN is set here, and it wins over this login while it stays set.");
    }
    Ok(())
}

/// `blue-onyx-prism hf-logout`: deletes the saved login. Says what is still in effect, because
/// the `hf` CLI's file and `HF_TOKEN` are not ours to remove.
pub fn logout(config: &Config) -> Result<()> {
    let root = config.data_root();
    let path = saved_token_path(&root);
    if remove_saved(&root)? {
        println!("Removed the saved login {}.", path.display());
    } else {
        println!("No saved login at {}.", path.display());
    }
    if let Some(t) = lookup(&root).token() {
        println!(
            "note: a token is still in use, from {}.",
            t.source().label()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const TOKEN: &str = "hf_NotARealTokenOnlyForTests123";

    fn tmp() -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop-hf-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
        let m: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k: &str| m.get(k).cloned()
    }

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    fn found(l: &Lookup) -> (&str, TokenSource) {
        let t = l.token().unwrap_or_else(|| panic!("{l:?}"));
        (t.secret(), t.source())
    }

    #[test]
    fn lookup_order_and_source_labels() {
        let root = tmp();
        let cli = root.join("hf-home");
        write(&cli.join("token"), "from-hf-home\n");
        let path_file = root.join("elsewhere/my-token");
        write(&path_file, "from-token-path");
        let home = root.join("home");
        write(&home.join(".cache/huggingface/token"), "from-home");
        let xdg = root.join("xdg");
        write(&xdg.join("huggingface/token"), "from-xdg");
        let home_var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
        let home_s = home.to_str().unwrap();
        let xdg_s = xdg.to_str().unwrap();
        let cli_s = cli.to_str().unwrap();
        let path_s = path_file.to_str().unwrap();

        // Every source set: each one wins until it is taken away.
        let all = [
            ("HF_TOKEN", "from-env"),
            ("HF_TOKEN_PATH", path_s),
            ("HF_HOME", cli_s),
            ("XDG_CACHE_HOME", xdg_s),
            (home_var, home_s),
        ];
        write(&saved_token_path(&root), "from-saved");
        let l = lookup_with(&env_of(&all), &root);
        assert_eq!(found(&l), ("from-env", TokenSource::Env));
        let l = lookup_with(&env_of(&all[1..]), &root);
        assert_eq!(found(&l), ("from-saved", TokenSource::Saved));
        remove_saved(&root).unwrap();
        let l = lookup_with(&env_of(&all[1..]), &root);
        assert_eq!(found(&l), ("from-token-path", TokenSource::HfCli));
        let l = lookup_with(&env_of(&all[2..]), &root);
        assert_eq!(found(&l), ("from-hf-home", TokenSource::HfCli));
        let l = lookup_with(&env_of(&all[3..]), &root);
        assert_eq!(found(&l), ("from-xdg", TokenSource::HfCli));
        let l = lookup_with(&env_of(&all[4..]), &root);
        assert_eq!(found(&l), ("from-home", TokenSource::HfCli));
        assert_eq!(lookup_with(&env_of(&[]), &root), Lookup::Missing);

        assert!(TokenSource::Env.label().contains("HF_TOKEN"));
        assert!(TokenSource::Saved.label().contains("hf-login"));
        assert!(TokenSource::HfCli.label().contains("hf CLI"));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn blank_sources_are_skipped_and_whitespace_trimmed() {
        let root = tmp();
        write(&saved_token_path(&root), "  \n");
        let hf = root.join("hf");
        write(&hf.join("token"), "  real-token\r\n");
        let env = env_of(&[("HF_TOKEN", "   "), ("HF_HOME", hf.to_str().unwrap())]);
        let l = lookup_with(&env, &root);
        assert_eq!(found(&l), ("real-token", TokenSource::HfCli));
        std::fs::remove_dir_all(&root).ok();
    }

    /// A file that exists but cannot be read is not "no file", and it stops the lookup. A
    /// directory in its place fails to read on every OS without changing permissions.
    #[test]
    fn unreadable_file_is_told_apart_from_no_file() {
        let root = tmp();
        assert_eq!(lookup_with(&env_of(&[]), &root), Lookup::Missing);
        std::fs::create_dir_all(saved_token_path(&root)).unwrap();
        let hf = root.join("hf");
        write(&hf.join("token"), "would-work");
        let env = env_of(&[("HF_HOME", hf.to_str().unwrap())]);
        let l = lookup_with(&env, &root);
        match &l {
            Lookup::Unreadable { source, path, .. } => {
                assert_eq!(*source, TokenSource::Saved);
                assert_eq!(path.as_deref(), Some(&*saved_token_path(&root)));
            }
            other => panic!("{other:?}"),
        }
        assert!(l.problem().unwrap().contains("cannot be read"));
        assert!(l.token().is_none());
        std::fs::remove_dir_all(&root).ok();

        // Not text, or not a token: unreadable too, and the content is not quoted.
        let root = tmp();
        let p = saved_token_path(&root);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, [0xff, 0xfe, 0x00]).unwrap();
        let l = lookup_with(&env_of(&[]), &root);
        assert!(matches!(l, Lookup::Unreadable { .. }), "{l:?}");
        std::fs::write(&p, "two words secret-bit").unwrap();
        let l = lookup_with(&env_of(&[]), &root);
        assert!(!format!("{l:?}{}", l.problem().unwrap()).contains("secret-bit"));
        std::fs::remove_dir_all(&root).ok();
        let l = lookup_with(&env_of(&[("HF_TOKEN", "bad token")]), &root);
        assert!(matches!(
            l,
            Lookup::Unreadable {
                source: TokenSource::Env,
                path: None,
                ..
            }
        ));
    }

    #[test]
    fn debug_never_shows_the_value() {
        let t = HfToken::new(TOKEN, TokenSource::Saved).unwrap();
        let shown = format!("{t:?} {:?}", Lookup::Found(t.clone()));
        assert!(!shown.contains(TOKEN), "{shown}");
        assert!(shown.contains("Saved"), "{shown}");
        assert_ne!(t.fingerprint(), TOKEN);
        assert!(!t.fingerprint().contains(TOKEN));
    }

    #[test]
    fn save_and_remove_round_trip() {
        let root = tmp();
        let t = HfToken::new(TOKEN, TokenSource::Saved).unwrap();
        let path = save(&root, &t).unwrap().path;
        assert_eq!(path, root.join("tools/hf/token"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), TOKEN);
        save(&root, &HfToken::new("second", TokenSource::Saved).unwrap()).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
        let names: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert_eq!(names, ["token"], "no temp file left");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        assert!(remove_saved(&root).unwrap());
        assert!(!remove_saved(&root).unwrap());
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn only_root_hands_files_over_and_never_to_root() {
        assert_eq!(chown_to(0, (1001, 1002)), Some((1001, 1002)));
        assert_eq!(chown_to(0, (0, 0)), None);
        assert_eq!(chown_to(1001, (1001, 1001)), None);
        assert_eq!(chown_to(1000, (1001, 1001)), None);
    }

    /// A save that cannot finish keeps the old login and leaves no copy of the new token behind.
    #[test]
    fn a_failed_save_keeps_the_old_login_and_leaves_no_temp_file() {
        let root = tmp();
        let target = saved_token_path(&root);
        // A folder with something in it where the file goes: the rename cannot replace it.
        write(&target.join("in-the-way"), "old");
        let t = HfToken::new(TOKEN, TokenSource::Saved).unwrap();
        assert!(save(&root, &t).is_err());
        assert_eq!(
            std::fs::read_to_string(target.join("in-the-way")).unwrap(),
            "old"
        );
        let names: Vec<_> = std::fs::read_dir(target.parent().unwrap())
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert_eq!(names, ["token"], "no temp file left");
        std::fs::remove_dir_all(&root).ok();
    }

    /// The folder is protected before the token exists in it.
    #[test]
    fn the_folder_is_protected_before_the_token_is_written() {
        let root = tmp();
        let t = HfToken::new(TOKEN, TokenSource::Saved).unwrap();
        let seen = std::cell::Cell::new(false);
        let saved = save_with(&root, &t, &|dir| {
            assert_eq!(dir, root.join("tools/hf"));
            assert!(dir.is_dir());
            assert_eq!(std::fs::read_dir(dir).unwrap().count(), 0);
            seen.set(true);
            None
        })
        .unwrap();
        assert!(seen.get());
        assert_eq!(saved.warning, None);
        assert_eq!(std::fs::read_to_string(saved.path).unwrap(), TOKEN);
        std::fs::remove_dir_all(&root).ok();
    }

    /// An ACL that cannot be set does not lose the login, and the user is told.
    #[test]
    fn a_failed_acl_is_a_visible_warning_and_the_token_is_still_saved() {
        let root = tmp();
        let t = HfToken::new(TOKEN, TokenSource::Saved).unwrap();
        let saved = save_with(&root, &t, &|dir| {
            let refused: folder_acl::Run<'_> = &|_, _| Err("Access is denied".to_string());
            folder_acl::protect_or_warn(dir, Ok(None), refused)
        })
        .unwrap();
        let warning = saved.warning.clone().unwrap();
        assert!(warning.contains("Access is denied"), "{warning}");
        assert!(warning.contains("Run as administrator"), "{warning}");
        assert!(!warning.contains(TOKEN), "{warning}");
        assert_eq!(std::fs::read_to_string(&saved.path).unwrap(), TOKEN);
        let report = LoginReport {
            name: "someone".to_string(),
            path: saved.path,
            access: Ok(Access::Granted),
            warning: saved.warning,
        };
        assert!(
            report.text(None).contains(&warning),
            "{}",
            report.text(None)
        );
        std::fs::remove_dir_all(&root).ok();
    }

    fn strings(args: &[std::ffi::OsString]) -> Vec<String> {
        args.iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    /// Every SID form, so a non-English Windows resolves them; one element per argument, so a
    /// path or an account with a space stays one.
    #[test]
    fn icacls_arguments_are_a_protected_dacl_for_three_accounts() {
        let dir = Path::new("D:/Data Root/tools/hf");
        let grants = folder_acl::grants(
            Some("NT SERVICE\\BlueOnyxPrismService"),
            "*S-1-5-21-1-2-3-1001",
        );
        let [reset, protect] = folder_acl::icacls_commands(dir, &grants);
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

    #[test]
    fn an_account_that_already_has_full_control_is_not_granted_twice() {
        // The login ran as the service's account, or as SYSTEM.
        assert_eq!(
            folder_acl::grants(Some("*S-1-5-19"), "*S-1-5-19"),
            [
                "*S-1-5-32-544:(OI)(CI)F",
                "*S-1-5-18:(OI)(CI)F",
                "*S-1-5-19:(OI)(CI)F",
            ]
        );
        assert_eq!(
            folder_acl::grants(None, "*S-1-5-18"),
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
            assert_eq!(
                folder_acl::service_principal(account).as_deref(),
                want,
                "{account}"
            );
        }
    }

    #[test]
    fn whoami_output_gives_the_sid() {
        let out = "\"desktop-1\\pat, jr\",\"S-1-5-21-111-222-333-1001\"\r\n";
        assert_eq!(
            folder_acl::parse_whoami(out).as_deref(),
            Some("*S-1-5-21-111-222-333-1001")
        );
        for bad in [
            "",
            "\"a\",\"b\"",
            "\"a\",\"S-1-\"",
            "\"a\",\"S-1-5-x\"",
            "denied",
        ] {
            assert_eq!(folder_acl::parse_whoami(bad), None, "{bad}");
        }
    }

    type Calls = std::cell::RefCell<Vec<(String, Vec<String>)>>;

    /// A fake of the tools: `whoami` answers, the rest succeed; the calls are kept.
    fn recording(calls: &Calls) -> impl Fn(&str, &[std::ffi::OsString]) -> Result<String, String> {
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
    fn protect_runs_whoami_then_both_icacls_steps_and_no_argument_is_the_token() {
        let root = tmp();
        let calls = Calls::default();
        let run = recording(&calls);
        let t = HfToken::new(TOKEN, TokenSource::Saved).unwrap();
        let saved = save_with(&root, &t, &|dir| {
            let service = Ok(Some("NT AUTHORITY\\LocalService".to_string()));
            folder_acl::protect_or_warn(dir, service, &run)
        })
        .unwrap();
        assert_eq!(saved.warning, None);
        let calls = calls.borrow();
        let tools: Vec<_> = calls.iter().map(|(tool, _)| tool.as_str()).collect();
        assert_eq!(tools, ["whoami.exe", "icacls.exe", "icacls.exe"]);
        assert_eq!(calls[0].1, ["/user", "/fo", "csv", "/nh"]);
        assert!(calls[2].1.contains(&"*S-1-5-19:(OI)(CI)RX".to_string()));
        assert!(
            calls[2]
                .1
                .contains(&"*S-1-5-21-9-1001:(OI)(CI)F".to_string())
        );
        for (_, args) in calls.iter() {
            assert!(!args.iter().any(|a| a.contains(TOKEN)), "{args:?}");
        }
        std::fs::remove_dir_all(&root).ok();
    }

    /// Protecting a folder without knowing who needs it could lock the service or the user out
    /// of their own token: nothing changes, and the warning says why.
    #[test]
    fn an_unknown_account_leaves_the_acl_alone() {
        let calls = Calls::default();
        let run = recording(&calls);
        let dir = Path::new("hf");
        let why = folder_acl::protect(dir, Err("the SCM said no".to_string()), &run).unwrap_err();
        assert!(why.contains("the SCM said no"), "{why}");
        assert!(calls.borrow().is_empty());

        let no_sid: folder_acl::Run<'_> = &|tool, args| {
            calls.borrow_mut().push((tool.to_string(), strings(args)));
            Ok("not csv".to_string())
        };
        let why = folder_acl::protect(dir, Ok(None), no_sid).unwrap_err();
        assert!(why.contains("whoami"), "{why}");
        assert!(calls.borrow().iter().all(|(tool, _)| tool == "whoami.exe"));
    }

    /// Tools are found by absolute path, never through the search path.
    #[test]
    fn tools_come_from_system32() {
        assert_eq!(
            folder_acl::tool_path(Some("D:\\Win".into()), "icacls.exe"),
            Path::new("D:\\Win").join("System32").join("icacls.exe")
        );
        assert_eq!(
            folder_acl::tool_path(None, "icacls.exe"),
            Path::new("C:\\Windows").join("System32").join("icacls.exe")
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_tool_that_fails_reports_what_it_said() {
        let sh = Path::new("/bin/sh");
        let args = |script: &str| ["-c".into(), script.into()];
        assert_eq!(
            folder_acl::run_at(sh, &args("printf ok")).as_deref(),
            Ok("ok")
        );
        let why = folder_acl::run_at(sh, &args("echo denied; echo more >&2; exit 3")).unwrap_err();
        assert!(why.contains("denied") && why.contains("more"), "{why}");
        let why = folder_acl::run_at(Path::new("/no/such/tool"), &[]).unwrap_err();
        assert!(why.contains("/no/such/tool"), "{why}");
    }

    #[test]
    fn env_override_is_reported_only_over_a_login_on_disk() {
        let root = tmp();
        let env = env_of(&[("HF_TOKEN", "from-env")]);
        assert!(!env_overrides_login_with(&env, &root));
        write(&saved_token_path(&root), "from-saved");
        assert!(env_overrides_login_with(&env, &root));
        assert!(!env_overrides_login_with(&env_of(&[]), &root));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn refusals_follow_the_measured_answers() {
        // Host only, gated header only, both.
        assert_eq!(refusal(401, false, true), Some(Refusal::NeedsLogin));
        assert_eq!(refusal(403, false, true), Some(Refusal::NeedsAccess));
        assert_eq!(refusal(401, true, false), Some(Refusal::NeedsLogin));
        assert_eq!(refusal(403, true, false), Some(Refusal::NeedsAccess));
        assert_eq!(refusal(403, true, true), Some(Refusal::NeedsAccess));
        // The CDN's own 403 (expired signature) is an ordinary failure.
        assert_eq!(refusal(403, false, false), None);
        assert_eq!(refusal(401, false, false), None);
        for ok in [200, 206, 404, 429, 500] {
            assert_eq!(refusal(ok, true, true), None, "{ok}");
        }
    }

    #[test]
    fn hosts_match_exactly() {
        assert!(url_on_host("https://huggingface.co/x/y", HF_HOST));
        assert!(url_on_host("https://HuggingFace.co/x", HF_HOST));
        for other in [
            "https://us.aws.cdn.hf.co/x",
            "https://huggingface.co.evil.example/x",
            "https://evil.example/huggingface.co",
            "https://cdn.huggingface.co/x",
            "not a url",
        ] {
            assert!(!url_on_host(other, HF_HOST), "{other}");
        }
    }

    #[test]
    fn messages_say_what_to_do_and_never_hold_the_token() {
        let g = catalog::gated(catalog::SAM3_ID).unwrap();
        let none = refusal_message(Refusal::NeedsLogin, g, None);
        assert!(
            none.contains("hf-login") && none.contains(g.repo_url),
            "{none}"
        );
        assert!(none.contains("SAM License"), "{none}");
        let bad = refusal_message(Refusal::NeedsLogin, g, Some(TokenSource::Env));
        assert!(bad.contains("HF_TOKEN") && bad.contains("refused"), "{bad}");
        for text in [&none, &bad] {
            assert!(!text.contains(&format!("{CLI_NAME} hf-login")), "{text}");
        }
        let wait = refusal_message(Refusal::NeedsAccess, g, Some(TokenSource::Saved));
        assert!(
            wait.contains(g.repo_url) && wait.contains("no access"),
            "{wait}"
        );
        assert!(wait.contains("fine-grained"), "{wait}");
    }

    #[test]
    fn parts_present_needs_every_file_at_its_pinned_size() {
        let res = sam3_resource().unwrap();
        let dir = tmp();
        assert!(!parts_present(res, &dir));
        // Sparse files: the sizes count, not the bytes.
        for p in res.parts {
            let f = std::fs::File::create(dir.join(p.file_name)).unwrap();
            f.set_len(p.size).unwrap();
        }
        assert!(parts_present(res, &dir));
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(dir.join("config.json"))
            .unwrap();
        f.set_len(1).unwrap();
        drop(f);
        assert!(!parts_present(res, &dir));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sam3_is_not_ready_until_its_manifest_says_so() {
        let root = tmp();
        let paths = TrainPaths::new(&root, root.join("models"));
        assert!(!sam3_ready(&paths));
        assert!(!sam3_parts_present(&paths));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn windows_command_shapes() {
        let c = login_command(&CommandPlace {
            os: "windows",
            exe_dir: r"C:\Users\Pat\Blue Onyx",
            config: r"C:\Users\Pat\Blue Onyx\blue_onyx_prism_config.json",
            account: None,
        });
        assert_eq!(
            c.command,
            r"& 'C:\Users\Pat\Blue Onyx\blue-onyx-prism.exe' --config 'C:\Users\Pat\Blue Onyx\blue_onyx_prism_config.json' hf-login"
        );
        assert!(c.note.contains("PowerShell") && !c.note.contains("elevated"));
        assert!(c.note.contains("read-only"));

        // MSI: the service exe's sibling, the service config, elevated.
        let c = login_command(&CommandPlace {
            os: "windows",
            exe_dir: r"C:\Program Files\Blue Onyx Prism\",
            config: r"C:\ProgramData\BlueOnyxPrism\blue_onyx_prism_config_service.json",
            account: Some("LOCAL SERVICE"),
        });
        assert_eq!(
            c.command,
            r"& 'C:\Program Files\Blue Onyx Prism\blue-onyx-prism.exe' --config 'C:\ProgramData\BlueOnyxPrism\blue_onyx_prism_config_service.json' hf-login"
        );
        assert!(c.note.contains("elevated terminal"), "{}", c.note);

        // The data alone can be under ProgramData; a quote in a path is doubled.
        let c = login_command(&CommandPlace {
            os: "windows",
            exe_dir: r"D:\Pat's tools",
            config: r"C:\ProgramData\x\c.json",
            account: None,
        });
        assert!(
            c.command
                .starts_with(r"& 'D:\Pat''s tools\blue-onyx-prism.exe'")
        );
        assert!(c.note.contains("elevated terminal"));
    }

    #[test]
    fn unix_command_shapes() {
        let c = login_command(&CommandPlace {
            os: "linux",
            exe_dir: "/opt/blue-onyx-prism",
            config: "/var/lib/blue-onyx-prism/blue_onyx_prism_config.json",
            account: Some("blueonyx"),
        });
        assert_eq!(
            c.command,
            "sudo -u 'blueonyx' '/opt/blue-onyx-prism/blue-onyx-prism' --config \
             '/var/lib/blue-onyx-prism/blue_onyx_prism_config.json' hf-login"
        );
        assert!(c.note.contains("service's account"));

        // No account in the environment: the one the installers create, per OS.
        let c = login_command(&CommandPlace {
            os: "macos",
            exe_dir: "/usr/local/blue-onyx-prism/",
            config: "/Library/Application Support/c.json",
            account: None,
        });
        assert_eq!(
            c.command,
            "sudo -u '_blueonyx' '/usr/local/blue-onyx-prism/blue-onyx-prism' --config \
             '/Library/Application Support/c.json' hf-login"
        );
        let c = login_command(&CommandPlace {
            os: "linux",
            exe_dir: "/opt/x",
            config: "/opt/x/it's.json",
            account: Some(" "),
        });
        assert!(
            c.command.starts_with("sudo -u 'blueonyx' "),
            "{}",
            c.command
        );
        assert!(
            c.command.contains(r"'/opt/x/it'\''s.json'"),
            "{}",
            c.command
        );

        // A service running as root: no `-u`.
        let c = login_command(&CommandPlace {
            os: "linux",
            exe_dir: "/opt/x",
            config: "/opt/x/c.json",
            account: Some("root"),
        });
        assert!(c.command.starts_with("sudo '/opt/x/"), "{}", c.command);
    }

    #[test]
    fn subcommand_comes_after_config() {
        for os in ["windows", "linux", "macos"] {
            let c = login_command(&CommandPlace {
                os,
                exe_dir: "x",
                config: "y",
                account: None,
            });
            let config = c.command.find("--config").unwrap();
            let sub = c.command.find("hf-login").unwrap();
            assert!(config < sub, "{}", c.command);
        }
    }

    #[test]
    fn login_report_words() {
        let g = catalog::gated(catalog::SAM3_ID);
        let report = |access| LoginReport {
            name: "someone".into(),
            path: PathBuf::from("t"),
            access,
            warning: None,
        };
        let yes = report(Ok(Access::Granted)).text(g);
        assert!(
            yes.contains("Logged in as someone, SAM 3 access: yes"),
            "{yes}"
        );
        assert!(yes.contains("read-only"), "{yes}");
        let no = report(Ok(Access::Refused(Refusal::NeedsAccess))).text(g);
        assert!(no.contains("SAM 3 access: not yet"), "{no}");
        assert!(no.contains("huggingface.co/facebook/sam3"), "{no}");
        let unknown = report(Err("offline".into())).text(g);
        assert!(
            unknown.contains("SAM 3 access: unknown (offline)"),
            "{unknown}"
        );
    }

    /// One-shot answers: `(path prefix, status line + headers, body)`. Returns the base URL and
    /// the request heads seen.
    fn fake(
        answers: Vec<(&'static str, &'static str, &'static str)>,
    ) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let heads = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = heads.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut buf = Vec::new();
                let mut b = [0u8; 1];
                while !buf.ends_with(b"\r\n\r\n") {
                    if s.read(&mut b).map_or(true, |n| n == 0) {
                        break;
                    }
                    buf.push(b[0]);
                }
                let head = String::from_utf8_lossy(&buf).to_string();
                seen.lock().unwrap().push(head.clone());
                let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
                let (status, body) = answers
                    .iter()
                    .find(|(p, ..)| path.starts_with(p))
                    .map_or(("404 Not Found", ""), |(_, st, body)| (*st, *body));
                let _ = write!(
                    s,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        (base, heads)
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    fn token() -> HfToken {
        HfToken::new(TOKEN, TokenSource::Saved).unwrap()
    }

    /// The answers of the measured table, through [`probe`].
    #[test]
    fn probe_reads_each_answer() {
        const GATED_401: &str = "401 Unauthorized\r\nX-Error-Code: GatedRepo";
        const GATED_403: &str = "403 Forbidden\r\nX-Error-Code: GatedRepo";
        let (base, heads) = fake(vec![
            ("/anon", GATED_401, ""),
            ("/refused", GATED_403, ""),
            ("/ok", "206 Partial Content", "{"),
            ("/down", "503 Service Unavailable", ""),
        ]);
        let client = manager::http_client().unwrap();
        let t = token();
        let ask = |path: &str, tok: Option<&HfToken>| {
            rt().block_on(probe(&client, &format!("{base}{path}"), tok, "127.0.0.1"))
        };
        assert_eq!(
            ask("/anon", None).unwrap(),
            Access::Refused(Refusal::NeedsLogin)
        );
        assert_eq!(
            ask("/refused", Some(&t)).unwrap(),
            Access::Refused(Refusal::NeedsAccess)
        );
        assert_eq!(ask("/ok", Some(&t)).unwrap(), Access::Granted);
        let e = ask("/down", Some(&t)).unwrap_err();
        assert!(!format!("{e:#}").contains(TOKEN));
        let h = heads.lock().unwrap();
        assert!(!h[0].to_ascii_lowercase().contains("authorization"));
        assert!(h[1].contains(&format!("Bearer {TOKEN}")) || h[1].contains(TOKEN));
    }

    #[test]
    fn probe_sends_the_token_to_the_hugging_face_host_only() {
        let (base, heads) = fake(vec![("/", "200 OK", "{}")]);
        let client = manager::http_client().unwrap();
        let t = token();
        let got = rt()
            .block_on(probe(&client, &format!("{base}/x"), Some(&t), HF_HOST))
            .unwrap();
        assert_eq!(got, Access::Granted);
        assert!(!heads.lock().unwrap()[0].contains(TOKEN));
    }

    #[test]
    fn login_validates_then_saves_then_reports() {
        let (base, _) = fake(vec![
            ("/whoami", "200 OK", r#"{"name":"someone","type":"user"}"#),
            ("/config", "403 Forbidden\r\nX-Error-Code: GatedRepo", ""),
        ]);
        let root = tmp();
        let client = manager::http_client().unwrap();
        let (whoami, probe) = (format!("{base}/whoami"), format!("{base}/config"));
        let at = Endpoints {
            whoami: &whoami,
            probe: &probe,
            host: "127.0.0.1",
        };
        let r = rt()
            .block_on(sign_in(&client, at, &root, &token()))
            .unwrap();
        assert_eq!(r.name, "someone");
        assert_eq!(r.access, Ok(Access::Refused(Refusal::NeedsAccess)));
        assert_eq!(
            std::fs::read_to_string(saved_token_path(&root)).unwrap(),
            TOKEN
        );
        std::fs::remove_dir_all(&root).ok();
    }

    /// A bad token is refused before anything is written, and the message does not repeat it.
    #[test]
    fn a_refused_login_saves_nothing() {
        let (base, _) = fake(vec![("/whoami", "401 Unauthorized", "{}")]);
        let root = tmp();
        let client = manager::http_client().unwrap();
        let (whoami, probe) = (format!("{base}/whoami"), format!("{base}/config"));
        let at = Endpoints {
            whoami: &whoami,
            probe: &probe,
            host: "127.0.0.1",
        };
        let e = rt()
            .block_on(sign_in(&client, at, &root, &token()))
            .unwrap_err();
        let text = format!("{e:#}");
        assert!(text.contains("nothing was saved"), "{text}");
        assert!(!text.contains(TOKEN), "{text}");
        assert!(!saved_token_path(&root).exists());
        std::fs::remove_dir_all(&root).ok();
    }
}
