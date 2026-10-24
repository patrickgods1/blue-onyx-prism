//! `GET|POST /v1/fs/pick`: the web UI's Browse... buttons open a native file or folder dialog
//! on the screen of the machine running the service (falling back, in the page, to the
//! server-side browser of [`super::browse`] when this answers "unavailable").
//!
//! - `GET` answers `{"available", "reason", "busy"}` without opening anything.
//! - `POST` (JSON `{"kind": "video"|"folder"|"model"|"classes", "start"?: path}`) opens the
//!   dialog and answers `{"path"}` or `{"cancelled": true}` (also after [`DIALOG_TIMEOUT`],
//!   when the child is killed), else 409 `{"available": false, "message"}` (no desktop, a peer
//!   that is not this machine, the dialog failed to start) or 409 `{"busy": true, "message"}`
//!   (one dialog at a time, process-wide). An admin route: loopback only unless
//!   `allow_remote_admin`, and the POST passes the CSRF guard; even with `allow_remote_admin` a
//!   remote peer gets "unavailable", since the dialog would open on a screen it cannot see.
//!
//! Platforms (the command building and the output parsing are pure and tested on every OS):
//! - macOS: `osascript` (AppleScript `choose file of type` / `choose folder`, activated so it
//!   comes to the front). Run as root (a launchd daemon), it runs in the console user's GUI
//!   session (`launchctl asuser <uid> sudo -n -u #<uid> osascript ...`; the console user is the
//!   owner of `/dev/console`, root there = the login window = unavailable).
//! - Windows: PowerShell with System.Windows.Forms (`OpenFileDialog` / `FolderBrowserDialog`
//!   owned by a hidden TopMost form), `CREATE_NO_WINDOW`. Unavailable when `SESSIONNAME` is
//!   not set: a Windows service runs in session 0, where a dialog would wait invisibly (a
//!   heuristic: interactive logons set `SESSIONNAME` to `Console` or `RDP-Tcp#N`, services
//!   and scheduled tasks without a desktop do not).
//! - Other Unix: `zenity` (preferred) or `kdialog` when `DISPLAY` or `WAYLAND_DISPLAY` is set.
//!
//! The start folder, prompt and filters never become script source: they are argv elements
//! (osascript `on run argv`, zenity/kdialog options) or environment variables (PowerShell).
//! Output is UTF-8 with the trailing newline trimmed; empty output or a cancel exit code means
//! cancelled.
//!
//! Picked paths are trusted: each one the dialog returns is recorded (canonical and as the
//! dialog spelled it, process-wide, at most [`PICK_CAP`] entries for [`PICK_TTL`]), and
//! [`super::browse::allowed_path`], the check of Add a model by path and the footage import,
//! accepts it and anything under it even outside the browsable roots (another volume such as
//! `/Volumes/...`, `/mnt/...`, a mapped drive or a `\\server\share`). This is safe because the
//! path was chosen by the person at the server's own screen in an OS dialog, which only a
//! loopback peer can open (see above): a remote client cannot add entries, and it gains only
//! the paths that person chose, for an hour. The in-page listing (`GET /v1/fs/browse`) keeps
//! the roots rule (SV-10 unchanged), so a pick never widens what the page can list.

use super::{AppState, admin_allowed, blocking};
use axum::Json;
use axum::extract::{ConnectInfo, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use std::io::Read;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// A dialog nobody answers is closed (its process killed) after this long.
pub(super) const DIALOG_TIMEOUT: Duration = Duration::from_secs(10 * 60);

/// Largest request body taken (a kind and a path).
const BODY_LIMIT: usize = 64 * 1024;

/// Bytes of the child's stdout/stderr kept (the rest is read and dropped).
const OUTPUT_LIMIT: usize = 64 * 1024;

/// Set while a dialog is open (one at a time, process-wide).
static BUSY: AtomicBool = AtomicBool::new(false);

const BUSY_MESSAGE: &str = "A file dialog is already open on the server's screen.";

const FALLBACK: &str = "browse the server's folders in the page instead.";

/// What the dialog picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum PickKind {
    /// A video file (the footage import; folders have their own button).
    Video,
    /// A folder.
    Folder,
    /// A `.onnx` / `.xml` model file.
    Model,
    /// A `.yaml` / `.yml` classes file.
    Classes,
}

impl PickKind {
    fn prompt(self) -> &'static str {
        match self {
            PickKind::Video => "Choose a video file",
            PickKind::Folder => "Choose a folder",
            PickKind::Model => "Choose a model file (.onnx or .xml)",
            PickKind::Classes => "Choose a classes YAML (.yaml or .yml)",
        }
    }

    /// Lowercase extensions of the files the dialog offers (empty for folders).
    fn exts(self) -> &'static [&'static str] {
        match self {
            PickKind::Video => &crate::training::import::source::VIDEO_EXTS,
            PickKind::Folder => &[],
            PickKind::Model => &["onnx", "xml"],
            PickKind::Classes => &["yaml", "yml"],
        }
    }

    fn filter_label(self) -> &'static str {
        match self {
            PickKind::Video => "Video files",
            PickKind::Folder => "Folders",
            PickKind::Model => "Model files",
            PickKind::Classes => "Classes YAML",
        }
    }

    fn is_folder(self) -> bool {
        self == PickKind::Folder
    }

    fn name(self) -> &'static str {
        match self {
            PickKind::Video => "video",
            PickKind::Folder => "folder",
            PickKind::Model => "model",
            PickKind::Classes => "classes",
        }
    }

    /// `*.ext` patterns, lower and upper case (zenity and kdialog match case-sensitively).
    fn patterns(self) -> Vec<String> {
        self.exts()
            .iter()
            .flat_map(|e| [format!("*.{e}"), format!("*.{}", e.to_ascii_uppercase())])
            .collect()
    }
}

/// The program that shows the dialog (decides how its output reads).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Tool {
    Osascript,
    PowerShell,
    Zenity,
    Kdialog,
}

/// A dialog process to start: program, argv and extra environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PickCommand {
    pub(super) tool: Tool,
    pub(super) program: String,
    pub(super) args: Vec<String>,
    pub(super) env: Vec<(String, String)>,
}

/// A start folder for a dialog: dropped when it could read as an option.
fn safe_start(start: Option<&str>) -> Option<&str> {
    start.filter(|s| !s.is_empty() && !s.starts_with('-'))
}

/// AppleScript of the macOS dialog. argv: kind, prompt, start folder ("" = none), then the
/// accepted extensions. User cancel (-128) prints nothing.
const MAC_SCRIPT: &str = r#"on run argv
	set theKind to item 1 of argv
	set thePrompt to item 2 of argv
	set theStart to item 3 of argv
	set theTypes to {}
	if (count of argv) > 3 then set theTypes to items 4 thru -1 of argv
	activate
	try
		if theKind is "folder" then
			if theStart is "" then
				set picked to choose folder with prompt thePrompt
			else
				set picked to choose folder with prompt thePrompt default location (POSIX file theStart)
			end if
		else
			if theStart is "" then
				set picked to choose file with prompt thePrompt of type theTypes
			else
				set picked to choose file with prompt thePrompt of type theTypes default location (POSIX file theStart)
			end if
		end if
	on error number -128
		return ""
	end try
	return POSIX path of picked
end run"#;

/// macOS: `osascript` directly, or, for a process running as root, in the GUI session of the
/// console user `as_user` (`launchctl asuser` for the session, `sudo -u` for the identity).
pub(super) fn macos_command(
    kind: PickKind,
    start: Option<&str>,
    as_user: Option<u32>,
) -> PickCommand {
    let mut args: Vec<String> = Vec::new();
    let program = match as_user {
        Some(uid) => {
            args.extend([
                "asuser".to_string(),
                uid.to_string(),
                "/usr/bin/sudo".to_string(),
                "-n".to_string(),
                "-u".to_string(),
                format!("#{uid}"),
                "--".to_string(),
                "/usr/bin/osascript".to_string(),
            ]);
            "/bin/launchctl".to_string()
        }
        None => "/usr/bin/osascript".to_string(),
    };
    args.extend(["-e".to_string(), MAC_SCRIPT.to_string()]);
    // The kind comes first: a fixed word, so osascript stops reading options there.
    args.push(kind.name().to_string());
    args.push(kind.prompt().to_string());
    args.push(safe_start(start).unwrap_or("").to_string());
    args.extend(kind.exts().iter().map(|e| e.to_string()));
    PickCommand {
        tool: Tool::Osascript,
        program,
        args,
        env: Vec::new(),
    }
}

/// PowerShell of the Windows dialog, one statement per line, no double quotes (the whole
/// script is one `-Command` argument). Inputs come from `BOP_PICK_*` environment variables.
const WINDOWS_SCRIPT: &[&str] = &[
    "$ErrorActionPreference = 'Stop'",
    "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false",
    "Add-Type -AssemblyName System.Windows.Forms",
    "[System.Windows.Forms.Application]::EnableVisualStyles()",
    "$owner = New-Object System.Windows.Forms.Form",
    "$owner.TopMost = $true",
    "$owner.ShowInTaskbar = $false",
    "$owner.Opacity = 0",
    "$owner.StartPosition = 'CenterScreen'",
    "$owner.Show()",
    "$owner.Activate()",
    "$start = $env:BOP_PICK_START",
    "if ($env:BOP_PICK_KIND -eq 'folder') { $d = New-Object System.Windows.Forms.FolderBrowserDialog; $d.Description = $env:BOP_PICK_TITLE; $d.ShowNewFolderButton = $true; if ($start) { $d.SelectedPath = $start } } else { $d = New-Object System.Windows.Forms.OpenFileDialog; $d.Title = $env:BOP_PICK_TITLE; $d.Filter = $env:BOP_PICK_FILTER; $d.CheckFileExists = $true; $d.Multiselect = $false; if ($start) { $d.InitialDirectory = $start } }",
    "$result = $d.ShowDialog($owner)",
    "$owner.Close()",
    "if ($result -eq [System.Windows.Forms.DialogResult]::OK) { if ($env:BOP_PICK_KIND -eq 'folder') { [Console]::Out.Write($d.SelectedPath) } else { [Console]::Out.Write($d.FileName) } }",
];

/// The `OpenFileDialog.Filter` of a file kind.
fn windows_filter(kind: PickKind) -> String {
    let pats = kind
        .exts()
        .iter()
        .map(|e| format!("*.{e}"))
        .collect::<Vec<_>>()
        .join(";");
    format!(
        "{} ({pats})|{pats}|All files (*.*)|*.*",
        kind.filter_label()
    )
}

/// Windows: `powershell -NoProfile -NonInteractive -STA -Command <WINDOWS_SCRIPT>`.
pub(super) fn windows_command(
    kind: PickKind,
    start: Option<&str>,
    powershell: &str,
) -> PickCommand {
    let mut env = vec![
        ("BOP_PICK_KIND".to_string(), kind.name().to_string()),
        ("BOP_PICK_TITLE".to_string(), kind.prompt().to_string()),
        (
            "BOP_PICK_START".to_string(),
            safe_start(start).unwrap_or("").to_string(),
        ),
    ];
    if !kind.is_folder() {
        env.push(("BOP_PICK_FILTER".to_string(), windows_filter(kind)));
    }
    PickCommand {
        tool: Tool::PowerShell,
        program: powershell.to_string(),
        args: vec![
            "-NoProfile".to_string(),
            "-NonInteractive".to_string(),
            "-STA".to_string(),
            "-Command".to_string(),
            WINDOWS_SCRIPT.join("\n"),
        ],
        env,
    }
}

/// Linux / other Unix with GTK: `zenity --file-selection ...`.
pub(super) fn zenity_command(kind: PickKind, start: Option<&str>, zenity: &str) -> PickCommand {
    let mut args = vec![
        "--file-selection".to_string(),
        "--title".to_string(),
        kind.prompt().to_string(),
    ];
    if kind.is_folder() {
        args.push("--directory".to_string());
    } else {
        args.push("--file-filter".to_string());
        args.push(format!(
            "{} | {}",
            kind.filter_label(),
            kind.patterns().join(" ")
        ));
        args.push("--file-filter".to_string());
        args.push("All files | *".to_string());
    }
    if let Some(s) = safe_start(start) {
        // A trailing separator opens the folder itself rather than selecting it in its parent.
        let s = if s.ends_with('/') {
            s.to_string()
        } else {
            format!("{s}/")
        };
        args.push("--filename".to_string());
        args.push(s);
    }
    PickCommand {
        tool: Tool::Zenity,
        program: zenity.to_string(),
        args,
        env: Vec::new(),
    }
}

/// KDE: `kdialog --getopenfilename <start> <filter>` / `--getexistingdirectory <start>`.
pub(super) fn kdialog_command(kind: PickKind, start: Option<&str>, kdialog: &str) -> PickCommand {
    let start = safe_start(start).unwrap_or(".").to_string();
    let mut args = vec!["--title".to_string(), kind.prompt().to_string()];
    if kind.is_folder() {
        args.push("--getexistingdirectory".to_string());
        args.push(start);
    } else {
        args.push("--getopenfilename".to_string());
        args.push(start);
        args.push(format!(
            "{} ({})",
            kind.filter_label(),
            kind.patterns().join(" ")
        ));
    }
    PickCommand {
        tool: Tool::Kdialog,
        program: kdialog.to_string(),
        args,
        env: Vec::new(),
    }
}

/// What a dialog run produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Outcome {
    Picked(String),
    Cancelled,
    /// The dialog did not run (no window server, a missing assembly, ...): the detail.
    Failed(String),
}

/// The outcome of a finished dialog process from its exit code (None = killed by a signal),
/// stdout and stderr.
pub(super) fn parse_output(tool: Tool, code: Option<i32>, stdout: &[u8], stderr: &[u8]) -> Outcome {
    let err = || {
        let e = String::from_utf8_lossy(stderr).trim().to_string();
        let e = if e.chars().count() > 300 {
            format!("{}...", e.chars().take(300).collect::<String>())
        } else {
            e
        };
        match code {
            Some(c) if e.is_empty() => format!("exit code {c}"),
            Some(c) => format!("exit code {c}: {e}"),
            None => "the dialog process was killed".to_string(),
        }
    };
    match code {
        Some(0) => {
            let text = String::from_utf8_lossy(stdout);
            let text = text.trim_start_matches('\u{feff}');
            let text = text.trim_end_matches(['\r', '\n']);
            if text.trim().is_empty() {
                return Outcome::Cancelled;
            }
            // osascript ends a folder's POSIX path with `/`; keep `/` itself.
            let text = if text.len() > 1 && text.ends_with('/') {
                &text[..text.len() - 1]
            } else {
                text
            };
            Outcome::Picked(text.to_string())
        }
        Some(1) if matches!(tool, Tool::Zenity | Tool::Kdialog) => Outcome::Cancelled,
        Some(_)
            if tool == Tool::Osascript && String::from_utf8_lossy(stderr).contains("(-128)") =>
        {
            Outcome::Cancelled
        }
        _ => Outcome::Failed(err()),
    }
}

/// How this host shows a dialog.
// Every platform's variant and decision is compiled on every OS, so the tests cover them all.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Launcher {
    /// osascript, as the console user `as_user` when the service runs as root.
    Mac {
        as_user: Option<u32>,
    },
    Windows {
        powershell: String,
    },
    Zenity(String),
    Kdialog(String),
}

impl Launcher {
    pub(super) fn command(&self, kind: PickKind, start: Option<&str>) -> PickCommand {
        match self {
            Launcher::Mac { as_user } => macos_command(kind, start, *as_user),
            Launcher::Windows { powershell } => windows_command(kind, start, powershell),
            Launcher::Zenity(p) => zenity_command(kind, start, p),
            Launcher::Kdialog(p) => kdialog_command(kind, start, p),
        }
    }

    fn describe(&self) -> String {
        match self {
            Launcher::Mac { as_user: None } => "macOS file dialog (osascript)".to_string(),
            Launcher::Mac { as_user: Some(u) } => {
                format!("macOS file dialog (osascript, as the console user {u})")
            }
            Launcher::Windows { .. } => "Windows file dialog (PowerShell)".to_string(),
            Launcher::Zenity(_) => "GTK file dialog (zenity)".to_string(),
            Launcher::Kdialog(_) => "KDE file dialog (kdialog)".to_string(),
        }
    }
}

// Every platform's variant and decision is compiled on every OS, so the tests cover them all.
#[cfg_attr(not(test), allow(dead_code))]
/// macOS decision from the process's effective uid, the owner of `/dev/console` and whether
/// osascript exists.
pub(super) fn mac_decision(
    euid: Option<u32>,
    console_uid: Option<u32>,
    osascript: bool,
) -> Result<Launcher, String> {
    if !osascript {
        return Err(format!(
            "/usr/bin/osascript was not found, so no file dialog can open: {FALLBACK}"
        ));
    }
    let Some(console) = console_uid else {
        return Err(format!(
            "The Mac's console user could not be determined (no /dev/console): {FALLBACK}"
        ));
    };
    if console == 0 {
        return Err(format!(
            "No one is logged in on the Mac's screen (it shows the login window): log in on \
             the Mac, or {FALLBACK}"
        ));
    }
    match euid {
        Some(0) => Ok(Launcher::Mac {
            as_user: Some(console),
        }),
        Some(u) if u == console => Ok(Launcher::Mac { as_user: None }),
        Some(u) => Err(format!(
            "The service runs as user {u} but the Mac's screen belongs to user {console}: \
             {FALLBACK}"
        )),
        None => Err(format!(
            "The service's user id could not be determined (`id -u` failed): {FALLBACK}"
        )),
    }
}

// Every platform's variant and decision is compiled on every OS, so the tests cover them all.
#[cfg_attr(not(test), allow(dead_code))]
/// Windows decision from `SESSIONNAME` (see the module doc) and the PowerShell to run.
pub(super) fn windows_decision(
    session_name: Option<&str>,
    powershell: String,
) -> Result<Launcher, String> {
    match session_name.map(str::trim) {
        Some(s) if !s.is_empty() => Ok(Launcher::Windows { powershell }),
        _ => Err(format!(
            "The service runs as a Windows service (no desktop session), so a file dialog \
             would open where no one can see it: {FALLBACK}"
        )),
    }
}

// Every platform's variant and decision is compiled on every OS, so the tests cover them all.
#[cfg_attr(not(test), allow(dead_code))]
/// Other-Unix decision from `DISPLAY` / `WAYLAND_DISPLAY` and the dialog tools found.
pub(super) fn unix_decision(
    display: bool,
    zenity: Option<String>,
    kdialog: Option<String>,
) -> Result<Launcher, String> {
    if !display {
        return Err(format!(
            "The service has no desktop (neither DISPLAY nor WAYLAND_DISPLAY is set): \
             {FALLBACK}"
        ));
    }
    match (zenity, kdialog) {
        (Some(z), _) => Ok(Launcher::Zenity(z)),
        (None, Some(k)) => Ok(Launcher::Kdialog(k)),
        (None, None) => Err(format!(
            "Neither zenity nor kdialog is installed, so no file dialog can open (install \
             zenity for native dialogs), or {FALLBACK}"
        )),
    }
}

/// `name` in the folders of a `PATH`-style list.
#[cfg_attr(any(windows, target_os = "macos"), allow(dead_code))]
fn find_in_path(name: &str, path_var: Option<&std::ffi::OsStr>) -> Option<String> {
    std::env::split_paths(path_var?)
        .map(|d| d.join(name))
        .find(|p| p.is_file())
        .map(|p| p.to_string_lossy().into_owned())
}

/// This process's effective uid, from `/usr/bin/id -u` (no `unsafe` outside `backend/`).
#[cfg(target_os = "macos")]
fn euid() -> Option<u32> {
    static EUID: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *EUID.get_or_init(|| {
        let out = Command::new("/usr/bin/id")
            .arg("-u")
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    })
}

/// How this host shows a dialog, or why it cannot (the message for the page).
fn host_launcher() -> Result<Launcher, String> {
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::fs::MetadataExt;
        let console = std::fs::metadata("/dev/console").ok().map(|m| m.uid());
        mac_decision(euid(), console, Path::new("/usr/bin/osascript").is_file())
    }
    #[cfg(windows)]
    {
        let ps = std::env::var_os("SystemRoot")
            .map(|r| PathBuf::from(r).join(r"System32\WindowsPowerShell\v1.0\powershell.exe"))
            .filter(|p| p.is_file())
            .map_or_else(
                || "powershell.exe".to_string(),
                |p| p.to_string_lossy().into_owned(),
            );
        let session = std::env::var("SESSIONNAME").ok();
        windows_decision(session.as_deref(), ps)
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let set = |v: &str| std::env::var_os(v).is_some_and(|s| !s.is_empty());
        let display = set("DISPLAY") || set("WAYLAND_DISPLAY");
        let path = std::env::var_os("PATH");
        unix_decision(
            display,
            find_in_path("zenity", path.as_deref()),
            find_in_path("kdialog", path.as_deref()),
        )
    }
    #[cfg(not(any(unix, windows)))]
    {
        Err(format!(
            "This operating system has no supported file dialog: {FALLBACK}"
        ))
    }
}

/// Without the Windows `\\?\` verbatim prefix (canonical paths carry it; dialogs do not take it).
fn strip_verbatim(s: String) -> String {
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{rest}");
    }
    match s.strip_prefix(r"\\?\") {
        Some(rest) => rest.to_string(),
        None => s,
    }
}

/// The folder a dialog starts in for the input's current value `raw` (relative = under the
/// data root): the value itself when it is a folder, else its nearest existing ancestor.
fn start_folder(cfg: &crate::config::Config, raw: Option<&str>) -> Option<String> {
    let raw = raw?.trim().trim_matches(|c| c == '"' || c == '\'').trim();
    if raw.is_empty() || crate::config::check_path_syntax("start", raw).is_err() {
        return None;
    }
    let mut p: PathBuf = cfg.data_path(Path::new(raw));
    while !p.is_dir() {
        if !p.pop() || p.as_os_str().is_empty() {
            return None;
        }
    }
    let c = p.canonicalize().ok()?;
    Some(strip_verbatim(c.to_string_lossy().into_owned()))
}

/// How long a picked path stays trusted (see the module doc).
pub(super) const PICK_TTL: Duration = Duration::from_secs(60 * 60);

/// Picked paths remembered at most (the oldest is dropped first).
pub(super) const PICK_CAP: usize = 64;

/// A path a dialog returned.
#[derive(Debug, Clone)]
struct PickedPath {
    /// As the dialog spelled it, without a verbatim prefix (matches a path the page sends back
    /// before it is canonicalized, e.g. `/tmp/x` for `/private/tmp/x`).
    dialog: PathBuf,
    /// Canonical, without a verbatim prefix.
    canonical: PathBuf,
    at: Instant,
}

/// The trusted picks, oldest first; expired entries are pruned on every insert and lookup.
#[derive(Debug, Default)]
pub(super) struct PickedPaths {
    entries: Vec<PickedPath>,
}

impl PickedPaths {
    pub(super) const fn new() -> PickedPaths {
        PickedPaths {
            entries: Vec::new(),
        }
    }

    fn prune(&mut self, now: Instant) {
        self.entries
            .retain(|e| now.saturating_duration_since(e.at) < PICK_TTL);
    }

    /// Remember a pick (`dialog` as returned, `canonical` its canonical form) made at `now`;
    /// a path picked again is refreshed, past [`PICK_CAP`] the oldest pick is dropped.
    pub(super) fn record_at(&mut self, dialog: &Path, canonical: &Path, now: Instant) {
        self.prune(now);
        let canonical = plain(canonical);
        let dialog = plain(dialog);
        self.entries
            .retain(|e| e.canonical != canonical || e.dialog != dialog);
        while self.entries.len() >= PICK_CAP {
            self.entries.remove(0);
        }
        self.entries.push(PickedPath {
            dialog,
            canonical,
            at: now,
        });
    }

    /// Whether `p` is an unexpired pick or under one, at `now`. Compared component-wise
    /// without the Windows verbatim prefix (a canonical path carries `\\?\`, a path the page
    /// sends does not). The dialog's spelling is a safe match for a canonical `p` too: a
    /// prefix of a canonical path is itself canonical.
    pub(super) fn covers_at(&mut self, p: &Path, now: Instant) -> bool {
        self.prune(now);
        let p = plain(p);
        self.entries
            .iter()
            .any(|e| p.starts_with(&e.canonical) || p.starts_with(&e.dialog))
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// The process-wide picks (see the module doc).
static PICKED: Mutex<PickedPaths> = Mutex::new(PickedPaths::new());

fn picked() -> std::sync::MutexGuard<'static, PickedPaths> {
    PICKED.lock().unwrap_or_else(|e| e.into_inner())
}

/// `p` without the Windows `\\?\` verbatim prefix (unchanged elsewhere).
fn plain(p: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        PathBuf::from(strip_verbatim(p.to_string_lossy().into_owned()))
    }
    #[cfg(not(windows))]
    {
        p.to_path_buf()
    }
}

/// Trust a path the dialog returned (ignored unless it is absolute and exists). Touches the
/// file system: call it off the runtime threads.
pub(super) fn record_pick(path: &str) {
    let dialog = Path::new(path);
    if !dialog.is_absolute() {
        return;
    }
    match dialog.canonicalize() {
        Ok(c) => picked().record_at(dialog, &c, Instant::now()),
        Err(e) => warn!("the picked path {path} cannot be resolved ({e}); not trusted"),
    }
}

/// Whether `p` (canonical, or as the page sent it) is a path the native dialog returned in the
/// last [`PICK_TTL`], or under one.
pub(super) fn picked_covers(p: &Path) -> bool {
    picked().covers_at(p, Instant::now())
}

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Ended {
    Exited,
    TimedOut,
    /// The HTTP request went away (tab closed, navigated away).
    Abandoned,
}

#[derive(Debug)]
pub(super) struct RunOutput {
    pub(super) code: Option<i32>,
    pub(super) stdout: Vec<u8>,
    pub(super) stderr: Vec<u8>,
    pub(super) ended: Ended,
}

/// Read `r` to its end, keeping the first [`OUTPUT_LIMIT`] bytes.
fn drain(mut r: impl Read + Send + 'static) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut kept = Vec::new();
        let mut buf = [0u8; 8192];
        while let Ok(n) = r.read(&mut buf) {
            if n == 0 {
                break;
            }
            let room = OUTPUT_LIMIT.saturating_sub(kept.len());
            kept.extend_from_slice(buf.get(..n.min(room)).unwrap_or_default());
        }
        kept
    })
}

/// Stop a dialog process: SIGTERM to its process group first (sudo relays it, osascript
/// closes its dialog), then a kill.
fn stop(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let group = format!("-{}", child.id());
        for kill in ["/bin/kill", "/usr/bin/kill"] {
            if Path::new(kill).is_file() {
                let _ = Command::new(kill)
                    .args(["-s", "TERM", "--", &group])
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
                break;
            }
        }
        let until = Instant::now() + Duration::from_secs(2);
        while Instant::now() < until {
            if matches!(child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Run `cmd` until it exits, `timeout` passes or `abandoned` is set (then it is stopped).
pub(super) fn run(
    cmd: &PickCommand,
    timeout: Duration,
    abandoned: &AtomicBool,
) -> std::io::Result<RunOutput> {
    let mut c = Command::new(&cmd.program);
    c.args(&cmd.args)
        .envs(cmd.env.iter().map(|(k, v)| (k, v)))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        // Its own process group, so [`stop`] reaches sudo's child too.
        use std::os::unix::process::CommandExt;
        c.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        c.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = c.spawn()?;
    let out = child.stdout.take().map(drain);
    let err = child.stderr.take().map(drain);
    let deadline = Instant::now() + timeout;
    let (status, ended) = loop {
        if let Some(s) = child.try_wait()? {
            break (Some(s), Ended::Exited);
        }
        if abandoned.load(Ordering::Relaxed) {
            stop(&mut child);
            break (None, Ended::Abandoned);
        }
        if Instant::now() >= deadline {
            stop(&mut child);
            break (None, Ended::TimedOut);
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let join = |h: Option<std::thread::JoinHandle<Vec<u8>>>| {
        h.and_then(|h| h.join().ok()).unwrap_or_default()
    };
    Ok(RunOutput {
        code: status.and_then(|s| s.code()),
        stdout: join(out),
        stderr: join(err),
        ended,
    })
}

/// Holds [`BUSY`] while a dialog is open.
struct BusyGuard;

impl BusyGuard {
    fn acquire() -> Option<BusyGuard> {
        BUSY.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| BusyGuard)
    }
}

impl Drop for BusyGuard {
    fn drop(&mut self) {
        BUSY.store(false, Ordering::Release);
    }
}

/// Sets its flag when dropped: the handler's future went away with its request.
struct AbandonOnDrop(Arc<AtomicBool>);

impl Drop for AbandonOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

fn remote_message() -> String {
    format!(
        "The file dialog would open on the server's screen, and this browser is on another \
         machine: {FALLBACK}"
    )
}

fn peer_of(ci: Option<&ConnectInfo<SocketAddr>>) -> Option<SocketAddr> {
    ci.map(|c| c.0)
}

/// Whether the peer is this machine (a dialog on the server's screen is useless otherwise).
fn local_peer(peer: Option<SocketAddr>) -> bool {
    admin_allowed(peer, false)
}

fn unavailable(message: String) -> Response {
    (
        StatusCode::CONFLICT,
        Json(serde_json::json!({ "available": false, "message": message })),
    )
        .into_response()
}

/// `GET /v1/fs/pick`: whether a native dialog can open (nothing is opened).
pub(super) async fn status(req: axum::extract::Request) -> Response {
    let peer = peer_of(req.extensions().get::<ConnectInfo<SocketAddr>>());
    let busy = BUSY.load(Ordering::Acquire);
    if !local_peer(peer) {
        return Json(serde_json::json!({
            "available": false, "reason": remote_message(), "busy": busy
        }))
        .into_response();
    }
    let answer = blocking(host_launcher).await;
    let (available, reason) = match answer {
        Ok(l) => (true, l.describe()),
        Err(m) => (false, m),
    };
    Json(serde_json::json!({ "available": available, "reason": reason, "busy": busy }))
        .into_response()
}

#[derive(Debug, Deserialize)]
struct PickRequest {
    kind: PickKind,
    #[serde(default)]
    start: Option<String>,
}

/// `POST /v1/fs/pick`: open the dialog and answer what was picked (see the module doc).
pub(super) async fn pick(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
) -> Response {
    let peer = peer_of(req.extensions().get::<ConnectInfo<SocketAddr>>());
    if !local_peer(peer) {
        return unavailable(remote_message());
    }
    let body = match axum::body::to_bytes(req.into_body(), BODY_LIMIT).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "message": format!("Cannot read the request: {e}") })),
            )
                .into_response();
        }
    };
    let pr: PickRequest = match serde_json::from_slice(&body) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "message": format!(
                        "Expected JSON {{\"kind\": \"video\"|\"folder\"|\"model\"|\"classes\", \
                         \"start\": path}}: {e}"
                    )
                })),
            )
                .into_response();
        }
    };
    let Some(guard) = BusyGuard::acquire() else {
        return (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "busy": true, "message": BUSY_MESSAGE })),
        )
            .into_response();
    };
    let abandoned = Arc::new(AtomicBool::new(false));
    let _on_drop = AbandonOnDrop(abandoned.clone());
    let outcome = blocking(move || {
        let _guard = guard;
        let launcher = host_launcher()?;
        let cfg = state.config_read().clone();
        let start = start_folder(&cfg, pr.start.as_deref());
        let cmd = launcher.command(pr.kind, start.as_deref());
        info!(kind = pr.kind.name(), dialog = %launcher.describe(), "opening a file dialog");
        let out = run(&cmd, DIALOG_TIMEOUT, &abandoned).map_err(|e| {
            format!(
                "The file dialog could not start ({}: {e}): {FALLBACK}",
                cmd.program
            )
        })?;
        let outcome = match out.ended {
            Ended::Exited => parse_output(cmd.tool, out.code, &out.stdout, &out.stderr),
            Ended::TimedOut => {
                warn!(
                    "the file dialog was not answered in {} min; closed it",
                    DIALOG_TIMEOUT.as_secs() / 60
                );
                Outcome::Cancelled
            }
            Ended::Abandoned => Outcome::Cancelled,
        };
        if let Outcome::Picked(p) = &outcome {
            record_pick(p);
        }
        Ok::<_, String>(outcome)
    })
    .await;
    match outcome {
        Err(m) => unavailable(m),
        Ok(Outcome::Picked(path)) => Json(serde_json::json!({ "path": path })).into_response(),
        Ok(Outcome::Cancelled) => Json(serde_json::json!({ "cancelled": true })).into_response(),
        Ok(Outcome::Failed(detail)) => {
            warn!("the native file dialog failed: {detail}");
            unavailable(format!(
                "The file dialog could not open on the server's screen ({detail}): {FALLBACK}"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOSTILE: &str = "\"; rm -rf / #";
    const KINDS: [PickKind; 4] = [
        PickKind::Video,
        PickKind::Folder,
        PickKind::Model,
        PickKind::Classes,
    ];

    fn all_commands(start: Option<&str>) -> Vec<PickCommand> {
        let mut v = Vec::new();
        for k in KINDS {
            v.push(macos_command(k, start, None));
            v.push(macos_command(k, start, Some(501)));
            v.push(windows_command(k, start, "powershell.exe"));
            v.push(zenity_command(k, start, "/usr/bin/zenity"));
            v.push(kdialog_command(k, start, "/usr/bin/kdialog"));
        }
        v
    }

    #[test]
    fn hostile_start_is_only_ever_its_own_argument() {
        let hostile = format!("/tmp/{HOSTILE}");
        for c in all_commands(Some(&hostile)) {
            let script_like: Vec<&String> = c
                .args
                .iter()
                .chain(c.env.iter().map(|(_, v)| v))
                .filter(|a| a.contains(HOSTILE))
                .collect();
            assert_eq!(script_like.len(), 1, "{c:?}");
            let only = script_like.first().unwrap();
            // The path itself (zenity adds the folder's trailing separator).
            assert!(
                **only == hostile || **only == format!("{hostile}/"),
                "{c:?}"
            );
            assert!(!c.program.contains(HOSTILE));
        }
        // Scripts are fixed text.
        assert!(!MAC_SCRIPT.contains(HOSTILE));
        assert!(!WINDOWS_SCRIPT.join("\n").contains('"'));
    }

    #[test]
    fn option_like_start_is_dropped() {
        for c in all_commands(Some("--help")) {
            assert!(!c.args.iter().any(|a| a.contains("--help")), "{c:?}");
            assert!(!c.env.iter().any(|(_, v)| v.contains("--help")), "{c:?}");
        }
    }

    #[test]
    fn macos_argv() {
        let c = macos_command(PickKind::Model, Some("/Users/a/models"), None);
        assert_eq!(c.program, "/usr/bin/osascript");
        assert_eq!(c.args[0], "-e");
        assert_eq!(c.args[1], MAC_SCRIPT);
        assert_eq!(
            &c.args[2..],
            [
                "model",
                PickKind::Model.prompt(),
                "/Users/a/models",
                "onnx",
                "xml"
            ]
        );
        let c = macos_command(PickKind::Folder, None, Some(501));
        assert_eq!(c.program, "/bin/launchctl");
        assert_eq!(
            &c.args[..8],
            [
                "asuser",
                "501",
                "/usr/bin/sudo",
                "-n",
                "-u",
                "#501",
                "--",
                "/usr/bin/osascript"
            ]
        );
        assert_eq!(&c.args[10..], ["folder", PickKind::Folder.prompt(), ""]);
        let c = macos_command(PickKind::Video, None, None);
        assert_eq!(&c.args[5..], crate::training::import::source::VIDEO_EXTS);
    }

    #[test]
    fn windows_argv_and_env() {
        let c = windows_command(PickKind::Classes, Some(r"C:\data"), "ps.exe");
        assert_eq!(c.program, "ps.exe");
        assert_eq!(
            &c.args[..4],
            ["-NoProfile", "-NonInteractive", "-STA", "-Command"]
        );
        assert_eq!(c.args.len(), 5);
        let env = |k: &str| c.env.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
        assert_eq!(env("BOP_PICK_KIND"), Some("classes"));
        assert_eq!(env("BOP_PICK_START"), Some(r"C:\data"));
        assert_eq!(
            env("BOP_PICK_FILTER"),
            Some("Classes YAML (*.yaml;*.yml)|*.yaml;*.yml|All files (*.*)|*.*")
        );
        let c = windows_command(PickKind::Folder, None, "ps.exe");
        assert!(!c.env.iter().any(|(k, _)| k == "BOP_PICK_FILTER"));
        assert!(c.args[4].contains("FolderBrowserDialog"));
        assert!(c.args[4].contains("TopMost"));
    }

    #[test]
    fn zenity_and_kdialog_argv() {
        let c = zenity_command(PickKind::Folder, Some("/srv"), "z");
        assert_eq!(
            c.args,
            [
                "--file-selection",
                "--title",
                "Choose a folder",
                "--directory",
                "--filename",
                "/srv/"
            ]
        );
        let c = zenity_command(PickKind::Model, None, "z");
        assert_eq!(
            c.args[3..5],
            ["--file-filter", "Model files | *.onnx *.ONNX *.xml *.XML"]
        );
        let c = kdialog_command(PickKind::Classes, None, "k");
        assert_eq!(
            c.args,
            [
                "--title",
                PickKind::Classes.prompt(),
                "--getopenfilename",
                ".",
                "Classes YAML (*.yaml *.YAML *.yml *.YML)"
            ]
        );
        let c = kdialog_command(PickKind::Folder, Some("/srv"), "k");
        assert_eq!(&c.args[2..], ["--getexistingdirectory", "/srv"]);
    }

    #[test]
    fn output_parsing() {
        use Outcome::*;
        let p = |t, c, o: &str, e: &str| parse_output(t, c, o.as_bytes(), e.as_bytes());
        assert_eq!(
            p(Tool::Osascript, Some(0), "/Users/a/clip.mp4\n", ""),
            Picked("/Users/a/clip.mp4".into())
        );
        assert_eq!(
            p(Tool::Osascript, Some(0), "/Users/a/Clips/\n", ""),
            Picked("/Users/a/Clips".into())
        );
        assert_eq!(p(Tool::Osascript, Some(0), "/\n", ""), Picked("/".into()));
        assert_eq!(p(Tool::Osascript, Some(0), "\n", ""), Cancelled);
        assert_eq!(
            p(
                Tool::Osascript,
                Some(1),
                "",
                "execution error: User canceled. (-128)"
            ),
            Cancelled
        );
        assert!(matches!(
            p(Tool::Osascript, Some(1), "", "No user interaction allowed. (-1713)"),
            Failed(m) if m.contains("-1713")
        ));
        assert_eq!(
            p(
                Tool::PowerShell,
                Some(0),
                "\u{feff}C:\\Clips\\é.mp4\r\n",
                ""
            ),
            Picked("C:\\Clips\\é.mp4".into())
        );
        assert_eq!(p(Tool::PowerShell, Some(0), "", ""), Cancelled);
        assert!(matches!(p(Tool::PowerShell, Some(1), "", ""), Failed(_)));
        assert_eq!(p(Tool::Zenity, Some(1), "", "Gtk-WARNING"), Cancelled);
        assert_eq!(p(Tool::Kdialog, Some(1), "", ""), Cancelled);
        assert!(matches!(p(Tool::Zenity, Some(255), "", ""), Failed(_)));
        assert!(matches!(p(Tool::Zenity, None, "", ""), Failed(_)));
        // A path with spaces and a trailing space keeps them (only line ends are trimmed).
        assert_eq!(
            p(Tool::Zenity, Some(0), "/a b/c \n", ""),
            Picked("/a b/c ".into())
        );
    }

    #[test]
    fn platform_decisions() {
        assert_eq!(
            mac_decision(Some(501), Some(501), true),
            Ok(Launcher::Mac { as_user: None })
        );
        assert_eq!(
            mac_decision(Some(0), Some(501), true),
            Ok(Launcher::Mac { as_user: Some(501) })
        );
        assert!(
            mac_decision(Some(0), Some(0), true)
                .unwrap_err()
                .contains("login window")
        );
        assert!(mac_decision(Some(502), Some(501), true).is_err());
        assert!(mac_decision(Some(501), Some(501), false).is_err());
        assert!(mac_decision(None, Some(501), true).is_err());
        assert!(
            windows_decision(None, "p".into())
                .unwrap_err()
                .contains("Windows service")
        );
        assert!(windows_decision(Some(""), "p".into()).is_err());
        assert_eq!(
            windows_decision(Some("Console"), "p".into()),
            Ok(Launcher::Windows {
                powershell: "p".into()
            })
        );
        assert!(unix_decision(false, Some("z".into()), None).is_err());
        assert_eq!(
            unix_decision(true, Some("z".into()), Some("k".into())),
            Ok(Launcher::Zenity("z".into()))
        );
        assert_eq!(
            unix_decision(true, None, Some("k".into())),
            Ok(Launcher::Kdialog("k".into()))
        );
        assert!(
            unix_decision(true, None, None)
                .unwrap_err()
                .contains("zenity")
        );
    }

    #[test]
    fn find_in_path_and_start_folder() {
        let dir = std::env::temp_dir().join(format!("bo_pick_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("tool"), b"").unwrap();
        let pv = std::env::join_paths([dir.join("nope"), dir.clone()]).unwrap();
        assert!(find_in_path("tool", Some(&pv)).is_some());
        assert!(find_in_path("other", Some(&pv)).is_none());
        assert!(find_in_path("tool", None).is_none());

        let cfg = crate::config::Config {
            download_dir: Some(dir.clone()),
            ..Default::default()
        };
        let canon = |p: PathBuf| strip_verbatim(p.canonicalize().unwrap().to_string_lossy().into());
        assert_eq!(
            start_folder(&cfg, Some("sub")),
            Some(canon(dir.join("sub")))
        );
        // A file or a missing path starts in the nearest existing folder.
        assert_eq!(start_folder(&cfg, Some("tool")), Some(canon(dir.clone())));
        assert_eq!(
            start_folder(&cfg, Some("\"sub/missing/x.onnx\"")),
            Some(canon(dir.join("sub")))
        );
        assert_eq!(start_folder(&cfg, Some("  ")), None);
        assert_eq!(start_folder(&cfg, None), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn picked_paths_cover_children_until_expiry() {
        let mut picks = PickedPaths::new();
        let t0 = Instant::now();
        let base = std::env::temp_dir().join("bop-pick-cover");
        let other = base.join("Volume");
        assert!(!picks.covers_at(&other, t0), "nothing picked yet");
        picks.record_at(&other, &other, t0);
        assert!(picks.covers_at(&other, t0));
        assert!(picks.covers_at(&other.join("clips/a.mp4"), t0), "a child");
        assert!(!picks.covers_at(&base, t0), "the parent");
        assert!(
            !picks.covers_at(&base.join("Volume2"), t0),
            "a sibling sharing a prefix"
        );
        let late = t0 + PICK_TTL - Duration::from_secs(1);
        assert!(picks.covers_at(&other.join("x"), late));
        assert!(!picks.covers_at(&other, t0 + PICK_TTL), "expired");
        assert_eq!(picks.len(), 0, "pruned on lookup");
    }

    #[test]
    fn picked_paths_match_the_dialog_spelling_and_refresh() {
        let mut picks = PickedPaths::new();
        let t0 = Instant::now();
        let dialog = std::env::temp_dir().join("bop-pick-link");
        let canon = std::env::temp_dir().join("bop-pick-real");
        picks.record_at(&dialog, &canon, t0);
        assert!(picks.covers_at(&dialog.join("m.onnx"), t0));
        assert!(picks.covers_at(&canon.join("m.onnx"), t0));
        // Picked again later: one entry, its clock restarted.
        let t1 = t0 + Duration::from_secs(1800);
        picks.record_at(&dialog, &canon, t1);
        assert_eq!(picks.len(), 1);
        assert!(picks.covers_at(&canon, t0 + PICK_TTL + Duration::from_secs(1)));
    }

    #[test]
    fn picked_paths_evict_the_oldest_past_the_cap() {
        let mut picks = PickedPaths::new();
        let t0 = Instant::now();
        let dir = |i: usize| std::env::temp_dir().join(format!("bop-pick-cap-{i}"));
        for i in 0..=PICK_CAP {
            let d = dir(i);
            picks.record_at(&d, &d, t0 + Duration::from_millis(i as u64));
        }
        assert_eq!(picks.len(), PICK_CAP);
        let now = t0 + Duration::from_secs(1);
        assert!(!picks.covers_at(&dir(0), now), "the oldest was dropped");
        assert!(picks.covers_at(&dir(1), now));
        assert!(picks.covers_at(&dir(PICK_CAP), now));
        // Insert prunes the expired first, so a late pick never evicts a live one.
        let late = t0 + PICK_TTL + Duration::from_secs(5);
        picks.record_at(&dir(999), &dir(999), late);
        assert_eq!(picks.len(), 1);
    }

    #[cfg(windows)]
    #[test]
    fn picked_paths_ignore_the_verbatim_prefix() {
        let mut picks = PickedPaths::new();
        let t0 = Instant::now();
        picks.record_at(
            Path::new(r"\\nas\clips"),
            Path::new(r"\\?\UNC\nas\clips"),
            t0,
        );
        picks.record_at(Path::new(r"E:\footage"), Path::new(r"\\?\E:\footage"), t0);
        assert!(picks.covers_at(Path::new(r"\\nas\clips\a.mp4"), t0));
        assert!(picks.covers_at(Path::new(r"\\?\UNC\nas\clips\a"), t0));
        assert!(picks.covers_at(Path::new(r"\\?\E:\footage\x"), t0));
        assert!(picks.covers_at(Path::new(r"E:\footage\x"), t0));
        assert!(!picks.covers_at(Path::new(r"E:\footage2"), t0));
    }

    #[test]
    fn record_pick_needs_an_existing_absolute_path() {
        let dir = std::env::temp_dir().join(format!("bo_pick_rec_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!picked_covers(&dir));
        record_pick(&dir.join("missing").to_string_lossy());
        record_pick("relative/dir");
        assert!(!picked_covers(&dir.join("missing")));
        record_pick(&dir.to_string_lossy());
        assert!(picked_covers(&dir.join("child")));
        assert!(picked_covers(&dir.canonicalize().unwrap().join("child")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn run_reads_output_and_stops_on_timeout_or_abandon() {
        let sh = |script: &str| PickCommand {
            tool: Tool::Zenity,
            program: "/bin/sh".into(),
            args: vec!["-c".into(), script.into(), "sh".into(), "/x y/é".into()],
            env: vec![("BOP_T".into(), "v".into())],
        };
        let never = AtomicBool::new(false);
        let out = run(
            &sh("printf '%s|%s\\n' \"$1\" \"$BOP_T\""),
            Duration::from_secs(10),
            &never,
        )
        .unwrap();
        assert_eq!(out.ended, Ended::Exited);
        assert_eq!(
            parse_output(Tool::Zenity, out.code, &out.stdout, &out.stderr),
            Outcome::Picked("/x y/é|v".into())
        );
        let t = Instant::now();
        let out = run(&sh("sleep 30"), Duration::from_millis(300), &never).unwrap();
        assert_eq!(out.ended, Ended::TimedOut);
        assert!(t.elapsed() < Duration::from_secs(10));
        let gone = AtomicBool::new(true);
        let out = run(&sh("sleep 30"), Duration::from_secs(60), &gone).unwrap();
        assert_eq!(out.ended, Ended::Abandoned);
    }

    /// Remote peers (even with `allow_remote_admin`) never open a dialog; a held flag answers
    /// busy before anything is started; bad bodies are 400.
    #[tokio::test]
    async fn handler_refusals() {
        use super::super::test_util::*;
        use axum::body::Body;
        use axum::http::{Request, header};
        let state = test_state_with(|c| c.allow_remote_admin = true);
        let post = |peer: &str, body: &str| {
            from_peer(
                Request::post("/v1/fs/pick")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
                peer,
            )
        };
        let (s, _, b) = call(&state, post(REMOTE, r#"{"kind":"folder"}"#)).await;
        assert_eq!(s, StatusCode::CONFLICT);
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(j["available"], false);
        assert!(j["message"].as_str().unwrap().contains("another machine"));

        let get = from_peer(
            Request::get("/v1/fs/pick").body(Body::empty()).unwrap(),
            REMOTE,
        );
        let (s, _, b) = call(&state, get).await;
        assert_eq!(s, StatusCode::OK);
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(j["available"], false);

        let (s, _, _) = call(&state, post(LOCAL, r#"{"kind":"photo"}"#)).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _, _) = call(&state, post(LOCAL, "kind=folder")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);

        // Local GET answers a shape (the value depends on the host's desktop).
        let get = Request::get("/v1/fs/pick").body(Body::empty()).unwrap();
        let (s, _, b) = call(&state, get).await;
        assert_eq!(s, StatusCode::OK);
        let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
        assert!(
            j["available"].is_boolean() && j["reason"].is_string(),
            "{j}"
        );

        // Busy: no dialog is started while the flag is held.
        let g = BusyGuard::acquire();
        if g.is_some() {
            assert!(BusyGuard::acquire().is_none());
            let (s, _, b) = call(&state, post(LOCAL, r#"{"kind":"folder"}"#)).await;
            assert_eq!(s, StatusCode::CONFLICT);
            let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
            assert_eq!(j["busy"], true);
            assert_eq!(j["message"], BUSY_MESSAGE);
        }
        drop(g);
        assert!(BusyGuard::acquire().is_some(), "released by the guard");

        // Cross-site POSTs are refused by the CSRF guard before the handler.
        let mut req = post(LOCAL, r#"{"kind":"folder"}"#);
        req.headers_mut()
            .insert("sec-fetch-site", "cross-site".parse().unwrap());
        let (s, _, _) = call(&state, req).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
    }
}
