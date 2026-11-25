//! Start, stop, restart and query the installed background service: the Windows service
//! `BlueOnyxPrismService`, the launchd daemon `com.blueonyx.prism` or the systemd unit
//! `blue-onyx-prism`. Used by `blue-onyx-prism service ...` and, for the state only, by the tray
//! app (which runs the CLI elevated to control the service). The OS-specific code lives in
//! `windows.rs` / `macos.rs` / `linux.rs`; the parsing of service-manager output is pure and
//! lives here so it is tested on every platform.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(windows)]
mod windows;

/// Windows service name (keep equal to `blue_onyx_prism_service.rs` and the MSI).
pub const WINDOWS_SERVICE_NAME: &str = "BlueOnyxPrismService";
/// launchd label of the daemon (deploy/com.blueonyx.prism.plist).
pub const LAUNCHD_LABEL: &str = "com.blueonyx.prism";
/// The installed daemon plist.
pub const LAUNCHD_PLIST: &str = "/Library/LaunchDaemons/com.blueonyx.prism.plist";
/// systemd unit (deploy/blue-onyx-prism.service).
pub const SYSTEMD_UNIT: &str = "blue-onyx-prism";

/// What to do with the service.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Start,
    Stop,
    Restart,
}

impl Action {
    /// Lowercase verb as typed on the command line ("start", "stop", "restart").
    pub fn verb(self) -> &'static str {
        match self {
            Action::Start => "start",
            Action::Stop => "stop",
            Action::Restart => "restart",
        }
    }
}

/// The service as the OS service manager reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServiceState {
    /// No service / daemon / unit is installed.
    NotInstalled,
    Stopped,
    Starting,
    Running,
    Stopping,
    /// The service manager could not be asked (e.g. no systemd).
    Unknown,
}

impl ServiceState {
    /// Human-readable state: "not installed", "stopped", "starting", "running", "stopping",
    /// "unknown".
    pub fn label(self) -> &'static str {
        match self {
            ServiceState::NotInstalled => "not installed",
            ServiceState::Stopped => "stopped",
            ServiceState::Starting => "starting",
            ServiceState::Running => "running",
            ServiceState::Stopping => "stopping",
            ServiceState::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for ServiceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// The service's state. Needs no administrator rights.
pub fn query() -> anyhow::Result<ServiceState> {
    #[cfg(windows)]
    {
        self::windows::query()
    }
    #[cfg(target_os = "macos")]
    {
        macos::query()
    }
    #[cfg(target_os = "linux")]
    {
        linux::query()
    }
    #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
    {
        Ok(ServiceState::Unknown)
    }
}

/// Start, stop or restart the service and wait (bounded) until it got there. Needs
/// administrator rights (root on Unix); fails with a message saying so otherwise.
pub fn control(action: Action) -> anyhow::Result<ServiceState> {
    #[cfg(windows)]
    {
        self::windows::control(action)
    }
    #[cfg(target_os = "macos")]
    {
        macos::control(action)
    }
    #[cfg(target_os = "linux")]
    {
        linux::control(action)
    }
    #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
    {
        anyhow::bail!(
            "Cannot {} the service: this platform has no supported service manager (Windows \
             services, launchd and systemd are). Start `blue-onyx-prism` by hand or with your \
             init system.",
            action.verb()
        )
    }
}

/// The error for a control action refused for lack of privileges: what happened, the likely
/// cause and what to do.
pub fn admin_rights_error(action: Action) -> anyhow::Error {
    let verb = action.verb();
    let gerund = match action {
        Action::Start => "Starting",
        Action::Stop => "Stopping",
        Action::Restart => "Restarting",
    };
    anyhow::anyhow!(
        "{gerund} the service needs administrator rights: run `sudo blue-onyx-prism service \
         {verb}` (macOS/Linux) or `blue-onyx-prism service {verb}` from an administrator \
         terminal (Windows)"
    )
}

/// The error for a control action on a service that is not installed.
pub fn not_installed_error(action: Action) -> anyhow::Error {
    anyhow::anyhow!(
        "Cannot {} the service: Blue Onyx Prism is not installed as a service. Install it \
         first (the installer, or scripts/install_*.sh / install_service.ps1), or run \
         `blue-onyx-prism` in a terminal.",
        action.verb()
    )
}

/// State from `launchctl print system/<label>` output (exit status 0, i.e. the job is loaded):
/// the top-level `state = ...` line (one tab deep; nested blocks such as the coalitions have
/// their own `state` lines, deeper). `running` is Running; any other state of a loaded job
/// (`spawn scheduled`, `waiting`, `not running` between KeepAlive respawns) is Starting.
pub fn parse_launchctl_print(output: &str) -> ServiceState {
    match launchctl_top_level(output, "state") {
        Some("running") => ServiceState::Running,
        Some(_) => ServiceState::Starting,
        None => ServiceState::Unknown,
    }
}

/// `pid = N` from `launchctl print` output, when the job has a process.
pub fn parse_launchctl_pid(output: &str) -> Option<u32> {
    launchctl_top_level(output, "pid")?.parse().ok()
}

/// The value of a top-level `key = value` line of `launchctl print` output.
fn launchctl_top_level<'a>(output: &'a str, key: &str) -> Option<&'a str> {
    output.lines().find_map(|line| {
        let rest = line.strip_prefix('\t')?;
        if rest.starts_with(char::is_whitespace) {
            return None;
        }
        let (k, v) = rest.split_once(" = ")?;
        (k == key).then(|| v.trim())
    })
}

/// State from `systemctl show -p LoadState -p ActiveState <unit>` output (`Key=value` lines).
pub fn parse_systemctl_show(output: &str) -> ServiceState {
    let mut load = "";
    let mut active = "";
    for line in output.lines() {
        if let Some(v) = line.trim().strip_prefix("LoadState=") {
            load = v;
        } else if let Some(v) = line.trim().strip_prefix("ActiveState=") {
            active = v;
        }
    }
    systemd_state(load, active)
}

/// State from a unit's `LoadState` and `ActiveState`.
pub fn systemd_state(load_state: &str, active_state: &str) -> ServiceState {
    if load_state == "not-found" {
        return ServiceState::NotInstalled;
    }
    match active_state {
        "active" | "reloading" | "refreshing" => ServiceState::Running,
        "activating" => ServiceState::Starting,
        "deactivating" => ServiceState::Stopping,
        "inactive" | "failed" | "maintenance" => ServiceState::Stopped,
        _ => ServiceState::Unknown,
    }
}

/// Whether the effective user is root ([`crate::command::euid`]). False when `id` cannot be run.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn is_root() -> bool {
    crate::command::euid() == Some(0)
}

/// Longest wait for a service-manager query (`launchctl print`, `systemctl show`): it answers in
/// milliseconds, so a longer wait means launchd or systemd is wedged (the tray polls with it).
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) const QUERY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Longest wait for a state change (`launchctl bootstrap|bootout|kickstart|enable`,
/// `systemctl start|stop|restart`, which waits for the job): above systemd's default 90 s start
/// timeout and the unit's stop timeout (`runner::SERVICE_MANAGER_STOP`, 75 s).
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) const CHANGE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Run a command for at most `timeout` ([`QUERY_TIMEOUT`] or [`CHANGE_TIMEOUT`]) through
/// [`crate::command::run_within`] and capture its output. A command that cannot be started or
/// does not answer in time (it is stopped) is an error naming it.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn run(
    program: &str,
    args: &[&str],
    timeout: std::time::Duration,
) -> anyhow::Result<crate::command::RunOutput> {
    use anyhow::Context;
    let mut cmd = std::process::Command::new(program);
    cmd.args(args);
    let out = crate::command::run_within(cmd, timeout, None)
        .with_context(|| format!("could not run `{program} {}`", args.join(" ")))?;
    if out.ended != crate::command::Ended::Exited {
        anyhow::bail!(
            "`{program} {}` did not answer within {} s and was stopped: the service manager is \
             busy or hung (during boot or shutdown, or under heavy load). Try again in a minute.",
            args.join(" "),
            timeout.as_secs()
        );
    }
    Ok(out)
}

/// stderr (else stdout) of a finished command, trimmed, for error messages.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn output_text(out: &crate::command::RunOutput) -> String {
    let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
    if err.is_empty() {
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    } else {
        err
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAUNCHCTL_RUNNING: &str = "system/com.blueonyx.prism = {
\tactive count = 1
\tpath = /Library/LaunchDaemons/com.blueonyx.prism.plist
\ttype = LaunchDaemon
\tstate = running

\tprogram = /usr/local/blue-onyx-prism/blue-onyx-prism
\tdomain = system
\tusername = _blueonyx
\texit timeout = 30
\truns = 1
\tpid = 6364
\timmediate reason = speculative
\tlast exit code = (never exited)

\tresource coalition = {
\t\tID = 2359
\t\ttype = resource
\t\tstate = active
\t\tactive count = 1
\t}

\tjob state = running

\tproperties = keepalive | runatload | inferred program
}
";

    #[test]
    fn launchctl_running() {
        assert_eq!(
            parse_launchctl_print(LAUNCHCTL_RUNNING),
            ServiceState::Running
        );
        assert_eq!(parse_launchctl_pid(LAUNCHCTL_RUNNING), Some(6364));
    }

    #[test]
    fn launchctl_loaded_not_running_is_starting() {
        let out = "system/com.blueonyx.prism = {
\tactive count = 0
\tstate = spawn scheduled
\truns = 3
\tlast exit code = 1

\tjetsam coalition = {
\t\tstate = active
\t}
\tjob state = exited
}
";
        assert_eq!(parse_launchctl_print(out), ServiceState::Starting);
        assert_eq!(parse_launchctl_pid(out), None);
        let waiting = out.replace("spawn scheduled", "not running");
        assert_eq!(parse_launchctl_print(&waiting), ServiceState::Starting);
    }

    #[test]
    fn launchctl_nested_state_is_ignored() {
        // Only a nested `state` line (and `job state`): not the job's own state.
        let out = "x = {\n\tjob state = running\n\tc = {\n\t\tstate = running\n\t}\n}\n";
        assert_eq!(parse_launchctl_print(out), ServiceState::Unknown);
        assert_eq!(parse_launchctl_print(""), ServiceState::Unknown);
    }

    #[test]
    fn systemctl_show_states() {
        let show = |load: &str, active: &str| {
            parse_systemctl_show(&format!("LoadState={load}\nActiveState={active}\n"))
        };
        assert_eq!(show("loaded", "active"), ServiceState::Running);
        assert_eq!(show("loaded", "activating"), ServiceState::Starting);
        assert_eq!(show("loaded", "deactivating"), ServiceState::Stopping);
        assert_eq!(show("loaded", "inactive"), ServiceState::Stopped);
        assert_eq!(show("loaded", "failed"), ServiceState::Stopped);
        assert_eq!(show("not-found", "inactive"), ServiceState::NotInstalled);
        assert_eq!(show("loaded", "weird"), ServiceState::Unknown);
        // Order of the lines does not matter.
        assert_eq!(
            parse_systemctl_show("ActiveState=active\nLoadState=loaded\n"),
            ServiceState::Running
        );
        assert_eq!(systemd_state("masked", "inactive"), ServiceState::Stopped);
    }

    #[test]
    fn labels() {
        assert_eq!(ServiceState::NotInstalled.label(), "not installed");
        assert_eq!(ServiceState::Running.to_string(), "running");
        assert_eq!(ServiceState::Stopping.label(), "stopping");
    }

    #[test]
    fn admin_message_names_the_command() {
        let msg = admin_rights_error(Action::Stop).to_string();
        assert!(msg.starts_with("Stopping the service needs administrator rights"));
        assert!(msg.contains("sudo blue-onyx-prism service stop"));
        assert!(msg.contains("administrator terminal"));
    }

    /// `run` answers a tool's output and exit status, and stops a tool that hangs (the tray's
    /// poll must not freeze on a wedged launchctl) with a message naming it.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn run_is_bounded_and_names_the_command() {
        let out = run("/bin/sh", &["-c", "echo said >&2; exit 2"], QUERY_TIMEOUT).unwrap();
        assert!(!out.success());
        assert_eq!(output_text(&out), "said");
        let out = run("/bin/sh", &["-c", "echo state=ok"], QUERY_TIMEOUT).unwrap();
        assert!(out.success());
        assert_eq!(output_text(&out), "state=ok");
        let t = std::time::Instant::now();
        let e = run(
            "/bin/sh",
            &["-c", "sleep 60"],
            std::time::Duration::from_secs(1),
        )
        .unwrap_err()
        .to_string();
        assert!(t.elapsed() < std::time::Duration::from_secs(8));
        assert!(
            e.starts_with("`/bin/sh -c sleep 60` did not answer within 1 s and was stopped"),
            "{e}"
        );
        let e = run("/nonexistent/launchctl", &["print"], QUERY_TIMEOUT)
            .unwrap_err()
            .to_string();
        assert_eq!(e, "could not run `/nonexistent/launchctl print`");
        assert_eq!(is_root(), crate::command::euid() == Some(0));
    }
}
