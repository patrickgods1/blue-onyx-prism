//! Linux: the systemd unit `blue-onyx-prism` through `systemctl` (`show` for the state,
//! `start|stop|restart`, which block until the job is done).

use super::{
    Action, SYSTEMD_UNIT, ServiceState, admin_rights_error, is_root, not_installed_error,
    output_text, parse_systemctl_show, run,
};
use anyhow::{Result, bail};
use std::path::Path;

/// systemd is PID 1 when this directory exists (`sd_booted`).
fn has_systemd() -> bool {
    Path::new("/run/systemd/system").is_dir()
}

pub(super) fn query() -> Result<ServiceState> {
    if !has_systemd() {
        return Ok(ServiceState::Unknown);
    }
    let out = run(
        "systemctl",
        &["show", "-p", "LoadState", "-p", "ActiveState", SYSTEMD_UNIT],
    )?;
    if !out.status.success() {
        bail!(
            "Could not ask systemd for the service state: `systemctl show {SYSTEMD_UNIT}` said: \
             {}",
            output_text(&out)
        );
    }
    Ok(parse_systemctl_show(&String::from_utf8_lossy(&out.stdout)))
}

pub(super) fn control(action: Action) -> Result<ServiceState> {
    if !has_systemd() {
        bail!(
            "Cannot {} the service: this system does not run systemd (no /run/systemd/system). \
             Control `blue-onyx-prism` with your init system, or run it in a terminal.",
            action.verb()
        );
    }
    if query()? == ServiceState::NotInstalled {
        return Err(not_installed_error(action));
    }
    // No polkit password prompt: without root, fail fast with the sudo hint.
    let out = run(
        "systemctl",
        &["--no-ask-password", action.verb(), SYSTEMD_UNIT],
    )?;
    if !out.status.success() {
        let text = output_text(&out);
        let denied = text.contains("Access denied")
            || text.contains("authentication required")
            || text.contains("Interactive authentication")
            || text.contains("Authentication is required")
            || text.contains("Permission denied");
        // "Job for ... failed" is the unit failing, whoever asked.
        if denied || (!is_root() && !text.contains("Job for")) {
            return Err(admin_rights_error(action));
        }
        bail!(
            "Could not {} the service: `systemctl {} {SYSTEMD_UNIT}` said: {text}. See \
             `journalctl -u {SYSTEMD_UNIT} -e` for the service log.",
            action.verb(),
            action.verb()
        );
    }
    query()
}
