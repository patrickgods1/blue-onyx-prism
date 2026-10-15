//! Running `<cli> service start|stop|restart` with administrator rights, showing an error
//! dialog and opening the web UI, per OS:
//!
//! - Windows: a hidden PowerShell runs `Start-Process -Verb RunAs` (the UAC prompt) on a second,
//!   elevated PowerShell that runs the CLI and saves its output to a temp file. Both scripts go
//!   through `-EncodedCommand` (base64 UTF-16LE), so no command-line quoting is involved, and
//!   every path inside them is a PowerShell single-quoted literal. Declining UAC exits 1223
//!   (`ERROR_CANCELLED`) and counts as cancelled.
//! - macOS: `osascript` `do shell script ... with administrator privileges` (the password
//!   prompt), the CLI path POSIX-quoted inside an escaped AppleScript string. Cancel = -128.
//!
//! The script builders are pure and unit-tested; text shown in dialogs is passed in an
//! environment variable, never interpolated into a script.

use crate::service_ctl::Action;
#[cfg(any(windows, target_os = "macos"))]
use std::path::Path;

/// How an action ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Done,
    /// The user declined the UAC / password prompt.
    Cancelled,
    Failed(String),
}

/// Environment variable carrying the text of an error dialog.
pub const MESSAGE_ENV: &str = "BOP_TRAY_MESSAGE";
/// Exit code of the Windows wrapper when UAC was declined (`ERROR_CANCELLED`).
pub const EXIT_CANCELLED: i32 = 1223;

// ---------------------------------------------------------------------------------------------
// Pure builders
// ---------------------------------------------------------------------------------------------

/// PowerShell single-quoted literal. PowerShell also ends single-quoted strings at the
/// typographic quotes U+2018..U+201B; each quote character is doubled.
pub fn ps_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        if matches!(c, '\'' | '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}') {
            out.push(c);
        }
        out.push(c);
    }
    out.push('\'');
    out
}

/// `-EncodedCommand` payload: base64 of the UTF-16LE script.
pub fn ps_encode(script: &str) -> String {
    use base64::Engine as _;
    let bytes: Vec<u8> = script.encode_utf16().flat_map(u16::to_le_bytes).collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Script run elevated: the CLI with its exit code passed on (its output is not captured).
pub fn windows_inner_script(cli: &str, action: Action) -> String {
    format!(
        "& {cli} 'service' {verb}\n\
         exit $(if ($null -eq $LASTEXITCODE) {{ 1 }} else {{ $LASTEXITCODE }})\n",
        cli = ps_quote(cli),
        verb = ps_quote(action.verb()),
    )
}

/// Script run by the tray (not elevated): UAC-elevate `powershell` on the inner script, wait,
/// pass its exit code on; a declined prompt exits [`EXIT_CANCELLED`].
pub fn windows_outer_script(powershell: &str, inner: &str) -> String {
    format!(
        "try {{\n\
         \x20 $p = Start-Process -FilePath {ps} -ArgumentList '-NoProfile','-NonInteractive','-WindowStyle','Hidden','-EncodedCommand','{enc}' -Verb RunAs -WindowStyle Hidden -PassThru -ErrorAction Stop\n\
         }} catch {{\n\
         \x20 $e = $_.Exception\n\
         \x20 while ($e -and -not ($e -is [System.ComponentModel.Win32Exception])) {{ $e = $e.InnerException }}\n\
         \x20 if ($e -and $e.NativeErrorCode -eq 1223) {{ exit {cancel} }}\n\
         \x20 [Console]::Error.WriteLine(\"$_\")\n\
         \x20 exit 1\n\
         }}\n\
         $null = $p.Handle\n\
         $p.WaitForExit()\n\
         exit $p.ExitCode\n",
        ps = ps_quote(powershell),
        enc = ps_encode(inner),
        cancel = EXIT_CANCELLED,
    )
}

/// Script of the Windows error dialog (text from [`MESSAGE_ENV`]).
pub const WINDOWS_ALERT_SCRIPT: &str = "Add-Type -AssemblyName PresentationFramework\n\
     [void][System.Windows.MessageBox]::Show($env:BOP_TRAY_MESSAGE, 'Blue Onyx Prism', 'OK', 'Error')\n";

/// POSIX shell single-quoted word.
pub fn posix_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// AppleScript string literal.
pub fn applescript_literal(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', r"\\").replace('"', "\\\""))
}

/// The `osascript -e` source that runs the CLI as root after a password prompt.
pub fn macos_admin_script(cli: &str, action: Action) -> String {
    let verb = action.verb();
    let command = format!("{} service {verb} 2>&1", posix_quote(cli));
    let prompt = format!(
        "Blue Onyx Prism needs an administrator password to {verb} its background service."
    );
    format!(
        "do shell script {} with prompt {} with administrator privileges",
        applescript_literal(&command),
        applescript_literal(&prompt)
    )
}

/// Script of the macOS error dialog (text from [`MESSAGE_ENV`]).
pub const MACOS_ALERT_SCRIPT: &str =
    "display alert \"Blue Onyx Prism\" message (system attribute \"BOP_TRAY_MESSAGE\") as critical";

/// Whether osascript's stderr means the user cancelled the password prompt.
pub fn macos_cancelled(stderr: &str) -> bool {
    stderr.contains("(-128)")
        || stderr.contains("User canceled")
        || stderr.contains("User cancelled")
}

/// osascript's error text without the `0:123: execution error: ` prefix and ` (N)` suffix.
pub fn clean_osascript_error(stderr: &str) -> String {
    let s = stderr.replace('\r', "\n");
    let s = s.trim();
    let s = match s.find("execution error: ") {
        Some(i) => s.get(i + "execution error: ".len()..).unwrap_or(s),
        None => s,
    };
    let s = match s.rfind(" (") {
        Some(i) if s.ends_with(')') && s.get(i + 2..s.len() - 1).is_some_and(is_int) => {
            s.get(..i).unwrap_or(s)
        }
        _ => s,
    };
    s.trim().to_string()
}

fn is_int(s: &str) -> bool {
    let s = s.strip_prefix('-').unwrap_or(s);
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// Error text for a failed CLI run: its output (last lines) or the exit code.
pub fn failure_text(action: Action, code: Option<i32>, output: &str) -> String {
    let lines: Vec<&str> = output
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let tail = lines
        .get(lines.len().saturating_sub(6)..)
        .unwrap_or_default()
        .join("\n");
    let head = match code {
        Some(c) => format!("Could not {} the service (exit code {c}).", action.verb()),
        None => format!("Could not {} the service.", action.verb()),
    };
    if tail.is_empty() {
        head
    } else {
        format!("{head}\n{tail}")
    }
}

// ---------------------------------------------------------------------------------------------
// Running them
// ---------------------------------------------------------------------------------------------

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[cfg(windows)]
fn powershell_exe() -> std::path::PathBuf {
    let root = std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into());
    std::path::PathBuf::from(root).join(r"System32\WindowsPowerShell\v1.0\powershell.exe")
}

#[cfg(windows)]
fn hidden_powershell(script: &str) -> std::process::Command {
    use std::os::windows::process::CommandExt as _;
    let mut c = std::process::Command::new(powershell_exe());
    c.args([
        "-NoProfile",
        "-NonInteractive",
        "-WindowStyle",
        "Hidden",
        "-EncodedCommand",
        &ps_encode(script),
    ])
    .creation_flags(CREATE_NO_WINDOW);
    c
}

/// Run `<cli> service <action>` with administrator rights (blocks until it ends).
#[cfg(windows)]
pub fn run_elevated(cli: &Path, action: Action) -> Outcome {
    if !cli.is_file() {
        return Outcome::Failed(missing_cli(cli));
    }
    // The elevated process's output is not captured: it would have to be written to a file
    // the (possibly standard) user controls, from a process running as an administrator.
    let inner = windows_inner_script(&cli.to_string_lossy(), action);
    let outer = windows_outer_script(&powershell_exe().to_string_lossy(), &inner);
    let out = hidden_powershell(&outer).output();
    match out {
        Err(e) => Outcome::Failed(format!("Could not run PowerShell to elevate: {e}")),
        Ok(o) if o.status.success() => Outcome::Done,
        Ok(o) if o.status.code() == Some(EXIT_CANCELLED) => Outcome::Cancelled,
        Ok(o) => {
            // stderr carries only the outer script's own errors (e.g. PowerShell missing).
            let mut text =
                failure_text(action, o.status.code(), &String::from_utf8_lossy(&o.stderr));
            text.push_str(&format!(
                "\nTo see why, run `blue-onyx-prism service {}` in an administrator terminal, or \
                 check Event Viewer (Windows Logs > Application, source BlueOnyxPrism).",
                action.verb()
            ));
            Outcome::Failed(text)
        }
    }
}

/// Run `<cli> service <action>` as root after the password prompt (blocks until it ends).
#[cfg(target_os = "macos")]
pub fn run_elevated(cli: &Path, action: Action) -> Outcome {
    if !cli.is_file() {
        return Outcome::Failed(missing_cli(cli));
    }
    let script = macos_admin_script(&cli.to_string_lossy(), action);
    match std::process::Command::new("/usr/bin/osascript")
        .args(["-e", &script])
        .output()
    {
        Err(e) => Outcome::Failed(format!("Could not run osascript: {e}")),
        Ok(o) if o.status.success() => Outcome::Done,
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            if macos_cancelled(&stderr) {
                Outcome::Cancelled
            } else {
                Outcome::Failed(failure_text(action, None, &clean_osascript_error(&stderr)))
            }
        }
    }
}

#[cfg(any(windows, target_os = "macos"))]
fn missing_cli(cli: &Path) -> String {
    format!(
        "The Blue Onyx Prism command-line tool was not found at {}. Reinstall Blue Onyx Prism or \
         start the tray with --cli <path>.",
        cli.display()
    )
}

/// Show an error dialog without blocking the caller (the dialog process is reaped on a thread).
#[cfg(any(windows, target_os = "macos"))]
pub fn show_error(message: &str) {
    #[cfg(windows)]
    let mut cmd = hidden_powershell(WINDOWS_ALERT_SCRIPT);
    #[cfg(target_os = "macos")]
    let mut cmd = {
        let mut c = std::process::Command::new("/usr/bin/osascript");
        c.args(["-e", MACOS_ALERT_SCRIPT]);
        c
    };
    cmd.env(MESSAGE_ENV, message);
    match cmd.spawn() {
        Ok(mut child) => {
            std::thread::spawn(move || child.wait());
        }
        Err(e) => tracing::warn!("could not show the error dialog: {e}"),
    }
}

/// Open `url` in the default browser.
#[cfg(any(windows, target_os = "macos"))]
pub fn open_url(url: &str) {
    #[cfg(windows)]
    let result = {
        use std::os::windows::process::CommandExt as _;
        std::process::Command::new("cmd")
            .args(["/c", "start", "", url])
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
    };
    #[cfg(target_os = "macos")]
    let result = std::process::Command::new("/usr/bin/open").arg(url).spawn();
    match result {
        Ok(mut child) => {
            std::thread::spawn(move || child.wait());
        }
        Err(e) => tracing::warn!("could not open {url}: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;

    fn decode(b64: &str) -> String {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .unwrap();
        let units: Vec<u16> = bytes
            .chunks(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        String::from_utf16(&units).unwrap()
    }

    #[test]
    fn powershell_quoting() {
        assert_eq!(ps_quote("plain"), "'plain'");
        assert_eq!(
            ps_quote(r"C:\Program Files\O'Brien\x.exe"),
            r"'C:\Program Files\O''Brien\x.exe'"
        );
        assert_eq!(ps_quote("a\u{2019}b"), "'a\u{2019}\u{2019}b'");
        assert_eq!(ps_quote("$env:x \"q\""), "'$env:x \"q\"'");
    }

    #[test]
    fn encoded_command_roundtrips() {
        let s = "Write-Host 'h\u{e9}llo' \u{1F600}";
        assert_eq!(decode(&ps_encode(s)), s);
    }

    #[test]
    fn windows_scripts() {
        let inner = windows_inner_script(
            r"C:\Program Files\Blue Onyx 'Prism'\blue-onyx-prism.exe",
            Action::Restart,
        );
        assert!(inner.starts_with(
            r"& 'C:\Program Files\Blue Onyx ''Prism''\blue-onyx-prism.exe' 'service' 'restart'"
        ));
        // Nothing is written anywhere by the elevated process.
        assert!(!inner.contains("Set-Content") && !inner.contains("Out-File"));
        assert!(inner.contains("exit $(if ($null -eq $LASTEXITCODE)"));

        let outer = windows_outer_script(r"C:\Windows\System32\powershell.exe", &inner);
        assert!(outer.contains(r"-FilePath 'C:\Windows\System32\powershell.exe'"));
        assert!(outer.contains("-Verb RunAs"));
        assert!(outer.contains("exit 1223"));
        // The inner script travels base64-encoded, so its quotes cannot break the outer one.
        let enc = outer
            .split("'-EncodedCommand','")
            .nth(1)
            .and_then(|r| r.split('\'').next())
            .unwrap();
        assert_eq!(decode(enc), inner);
    }

    #[test]
    fn posix_and_applescript_quoting() {
        assert_eq!(posix_quote("/a b/c"), "'/a b/c'");
        assert_eq!(posix_quote("/it's"), r"'/it'\''s'");
        assert_eq!(applescript_literal(r#"a "b" \c"#), r#""a \"b\" \\c""#);
    }

    #[test]
    fn macos_script_quotes_a_hostile_path() {
        let s = macos_admin_script(r#"/Apps/My "Prism" it's\x/blue-onyx-prism"#, Action::Stop);
        assert_eq!(
            s,
            r#"do shell script "'/Apps/My \"Prism\" it'\\''s\\x/blue-onyx-prism' service stop 2>&1" with prompt "Blue Onyx Prism needs an administrator password to stop its background service." with administrator privileges"#
        );
    }

    #[test]
    fn macos_cancel_and_error_text() {
        assert!(macos_cancelled(
            "0:120: execution error: User canceled. (-128)\n"
        ));
        assert!(!macos_cancelled("0:120: execution error: boom (1)\n"));
        assert_eq!(
            clean_osascript_error("0:181: execution error: Error: not installed\rmore (1)\n"),
            "Error: not installed\nmore"
        );
        assert_eq!(clean_osascript_error("plain (x)"), "plain (x)");
    }

    #[test]
    fn failure_text_keeps_the_last_lines() {
        let out = (1..=10).map(|i| format!("line {i}\n")).collect::<String>();
        let t = failure_text(Action::Start, Some(3), &out);
        assert!(t.starts_with("Could not start the service (exit code 3).\nline 5\n"));
        assert!(t.ends_with("line 10"));
        assert_eq!(
            failure_text(Action::Stop, None, "  \n"),
            "Could not stop the service."
        );
    }

    #[test]
    fn alert_scripts_read_the_message_from_the_environment() {
        assert!(WINDOWS_ALERT_SCRIPT.contains("$env:BOP_TRAY_MESSAGE"));
        assert!(MACOS_ALERT_SCRIPT.contains("system attribute \"BOP_TRAY_MESSAGE\""));
    }
}
