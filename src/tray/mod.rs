//! The tray (Windows) / menu bar (macOS) app `blue-onyx-prism-tray`: shows whether the
//! background service runs and is ready, and starts, stops or restarts it by running
//! `blue-onyx-prism service <action>` with administrator rights. Quitting the tray leaves the
//! service running.
//!
//! - [`model`]: pure state -> texts, icon color and enabled menu actions.
//! - [`probe`]: `GET /v1/status` over a plain `TcpStream` and its parser.
//! - [`icon`]: the runtime-drawn icon.
//! - [`config`]: port / address / CLI path resolution.
//! - [`elevate`]: UAC / password-prompt elevation, error dialog, opening the browser.
//! - `app` (Windows and macOS only): the tao event loop, the menu and the poll thread.
//!
//! On other OSes [`run`] prints how to control the service there and exits 2.

pub mod config;
pub mod elevate;
pub mod icon;
pub mod model;
pub mod probe;

#[cfg(any(windows, target_os = "macos"))]
mod app;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Command line of `blue-onyx-prism-tray`.
#[derive(Debug, Clone, Default, clap::Parser)]
#[command(
    name = "blue-onyx-prism-tray",
    version,
    about = "Blue Onyx Prism tray / menu bar app: shows the background service's status and \
             starts, stops or restarts it"
)]
pub struct TrayArgs {
    /// HTTP port of the service (default: the installed service config's port, else 32168).
    #[arg(long)]
    pub port: Option<u16>,
    /// The `blue-onyx-prism` CLI used to control the service (default: next to the tray on
    /// Windows, /usr/local/blue-onyx-prism/blue-onyx-prism on macOS).
    #[arg(long)]
    pub cli: Option<PathBuf>,
}

/// Printed on OSes without a tray app.
pub const UNSUPPORTED_MESSAGE: &str = "The tray app is available on Windows and macOS; on Linux \
     use `sudo systemctl start|stop blue-onyx-prism` or `blue-onyx-prism service ...`";

/// Name of the single-instance lock file in the per-user temp dir.
pub const LOCK_FILE: &str = "blue-onyx-prism-tray.lock";

/// Run the tray until the user quits it. A second instance for the same user exits 0 at once.
pub fn run(args: TrayArgs) -> ExitCode {
    #[cfg(any(windows, target_os = "macos"))]
    {
        app::run(args)
    }
    #[cfg(not(any(windows, target_os = "macos")))]
    {
        let _ = args;
        eprintln!("{UNSUPPORTED_MESSAGE}");
        ExitCode::from(2)
    }
}

/// Take the exclusive lock on `path` (created if missing). `Ok(None)`: another process holds it.
/// The lock lasts as long as the returned file is open.
pub fn try_lock_file(path: &Path) -> std::io::Result<Option<File>> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)?;
    match fs4::FileExt::try_lock(&file) {
        Ok(()) => Ok(Some(file)),
        Err(fs4::TryLockError::WouldBlock) => Ok(None),
        Err(fs4::TryLockError::Error(e)) => Err(e),
    }
}

/// The per-user single-instance lock ([`LOCK_FILE`] in `std::env::temp_dir()`, which is
/// per-user on Windows and macOS).
pub fn single_instance() -> std::io::Result<Option<File>> {
    try_lock_file(&std::env::temp_dir().join(LOCK_FILE))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser as _;

    #[test]
    fn second_lock_is_refused_until_the_first_is_dropped() {
        let dir = std::env::temp_dir().join(format!("bop-tray-lock-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(LOCK_FILE);
        let first = try_lock_file(&path).unwrap();
        assert!(first.is_some());
        assert!(try_lock_file(&path).unwrap().is_none());
        drop(first);
        assert!(try_lock_file(&path).unwrap().is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parses_the_command_line() {
        let a = TrayArgs::try_parse_from(["t", "--port", "8080", "--cli", "/x/y"]).unwrap();
        assert_eq!(a.port, Some(8080));
        assert_eq!(a.cli.as_deref(), Some(Path::new("/x/y")));
        let a = TrayArgs::try_parse_from(["t"]).unwrap();
        assert_eq!((a.port, a.cli), (None, None));
        assert!(TrayArgs::try_parse_from(["t", "--port", "70000"]).is_err());
    }
}
