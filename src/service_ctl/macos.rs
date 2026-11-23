//! macOS: the launchd daemon `system/com.blueonyx.prism` through `launchctl`. Stop unloads the
//! job (`bootout`; KeepAlive would respawn a killed process) until the next boot or start; start
//! enables and bootstraps the plist; restart is `kickstart -k`.

use super::{
    Action, CHANGE_TIMEOUT, LAUNCHD_LABEL, LAUNCHD_PLIST, QUERY_TIMEOUT, ServiceState,
    admin_rights_error, is_root, not_installed_error, output_text, parse_launchctl_pid,
    parse_launchctl_print, run,
};
use anyhow::{Result, bail};
use std::path::Path;
use std::time::{Duration, Instant};

const LAUNCHCTL: &str = "/bin/launchctl";
/// How long a start waits for `state = running`.
const START_TIMEOUT: Duration = Duration::from_secs(30);
/// How long a stop waits for the job to be gone: above the plist's ExitTimeOut
/// (`runner::SERVICE_MANAGER_STOP`, 75 s), since `bootout` returns before the process has exited.
const STOP_TIMEOUT: Duration = crate::runner::SERVICE_CTL_STOP_WAIT;
/// How long a restart (`kickstart -k`: the old process stops first, up to ExitTimeOut) waits for
/// the new process to run.
const RESTART_TIMEOUT: Duration = STOP_TIMEOUT.saturating_add(START_TIMEOUT);
const POLL: Duration = Duration::from_millis(250);

fn target() -> String {
    format!("system/{LAUNCHD_LABEL}")
}

/// The loaded job: its state and pid, or None when launchd does not know it.
fn loaded() -> Result<Option<(ServiceState, Option<u32>)>> {
    let out = run(LAUNCHCTL, &["print", &target()], QUERY_TIMEOUT)?;
    if !out.success() {
        return Ok(None);
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Ok(Some((
        parse_launchctl_print(&text),
        parse_launchctl_pid(&text),
    )))
}

pub(super) fn query() -> Result<ServiceState> {
    Ok(match loaded()? {
        Some((state, _)) => state,
        None if Path::new(LAUNCHD_PLIST).exists() => ServiceState::Stopped,
        None => ServiceState::NotInstalled,
    })
}

pub(super) fn control(action: Action) -> Result<ServiceState> {
    if !Path::new(LAUNCHD_PLIST).exists() && loaded()?.is_none() {
        return Err(not_installed_error(action));
    }
    if !is_root() {
        return Err(admin_rights_error(action));
    }
    match action {
        Action::Start => start(),
        Action::Stop => stop(),
        Action::Restart => match loaded()? {
            Some((_, pid)) => {
                launchctl(&["kickstart", "-k", &target()], "restart")?;
                wait_running(pid, RESTART_TIMEOUT)
            }
            None => start(),
        },
    }
}

fn start() -> Result<ServiceState> {
    match loaded()? {
        Some((ServiceState::Running, _)) => return Ok(ServiceState::Running),
        Some(_) => {
            launchctl(&["kickstart", &target()], "start")?;
        }
        None => {
            if !Path::new(LAUNCHD_PLIST).exists() {
                return Err(not_installed_error(Action::Start));
            }
            // A `launchctl disable` (by hand or an older uninstall) would make bootstrap fail.
            launchctl(&["enable", &target()], "start")?;
            let out = run(
                LAUNCHCTL,
                &["bootstrap", "system", LAUNCHD_PLIST],
                CHANGE_TIMEOUT,
            )?;
            // "Bootstrap failed: 5: Input/output error" also means "already loaded" (e.g. a
            // concurrent start); only an unloaded job after the call is a failure.
            if !out.success() && loaded()?.is_none() {
                bail!(
                    "Starting the service failed: `launchctl bootstrap system {LAUNCHD_PLIST}` \
                     said: {}. Check the plist with `plutil -lint {LAUNCHD_PLIST}` and the log \
                     in /var/log/blue-onyx-prism/, or reinstall the service.",
                    output_text(&out)
                );
            }
        }
    }
    wait_running(None, START_TIMEOUT)
}

fn stop() -> Result<ServiceState> {
    if loaded()?.is_none() {
        return Ok(ServiceState::Stopped);
    }
    // bootout may report an error such as "36: Operation now in progress" while the job is
    // still exiting; whether the job is gone in the end is what counts.
    let out = run(LAUNCHCTL, &["bootout", &target()], CHANGE_TIMEOUT)?;
    let deadline = Instant::now() + STOP_TIMEOUT;
    loop {
        if loaded()?.is_none() {
            return Ok(ServiceState::Stopped);
        }
        if Instant::now() >= deadline {
            let said = if out.success() {
                String::new()
            } else {
                format!(" (`launchctl bootout` said: {})", output_text(&out))
            };
            bail!(
                "Stopping the service timed out: launchd still has the job after {} s{said}. \
                 It may still be shutting down; check `blue-onyx-prism service status` again, \
                 or the log in /var/log/blue-onyx-prism/.",
                STOP_TIMEOUT.as_secs()
            );
        }
        std::thread::sleep(POLL);
    }
}

/// Poll until the job runs (with a pid other than `old_pid`, after a restart) or `timeout`;
/// returns the last state seen.
fn wait_running(old_pid: Option<u32>, timeout: Duration) -> Result<ServiceState> {
    let deadline = Instant::now() + timeout;
    loop {
        let state = match loaded()? {
            Some((ServiceState::Running, pid)) if old_pid.is_none() || pid != old_pid => {
                return Ok(ServiceState::Running);
            }
            Some((ServiceState::Running, _)) => ServiceState::Starting,
            Some((state, _)) => state,
            None => ServiceState::Stopped,
        };
        if Instant::now() >= deadline {
            return Ok(state);
        }
        std::thread::sleep(POLL);
    }
}

/// Run `launchctl <args>` (a state change: `kickstart`, `enable`), failing with its message.
fn launchctl(args: &[&str], what: &str) -> Result<()> {
    let out = run(LAUNCHCTL, args, CHANGE_TIMEOUT)?;
    if !out.success() {
        bail!(
            "Could not {what} the service: `launchctl {}` said: {}. Check the log in \
             /var/log/blue-onyx-prism/ or reinstall the service.",
            args.join(" "),
            output_text(&out)
        );
    }
    Ok(())
}
