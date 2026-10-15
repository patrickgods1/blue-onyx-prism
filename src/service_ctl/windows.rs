//! Windows: the service `BlueOnyxPrismService` through the Service Control Manager
//! (`windows-service`). Query needs only SERVICE_QUERY_STATUS (works unelevated); start / stop
//! need SERVICE_START / SERVICE_STOP (administrators) and poll until the state is reached.

use super::{Action, ServiceState, WINDOWS_SERVICE_NAME, admin_rights_error, not_installed_error};
use anyhow::{Context, Result, bail};
use std::time::{Duration, Instant};
use windows_service::service::{Service, ServiceAccess, ServiceState as WinState};
use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};

const ERROR_ACCESS_DENIED: i32 = 5;
const ERROR_SERVICE_ALREADY_RUNNING: i32 = 1056;
const ERROR_SERVICE_DOES_NOT_EXIST: i32 = 1060;
const ERROR_SERVICE_CANNOT_ACCEPT_CTRL: i32 = 1061;
const ERROR_SERVICE_NOT_ACTIVE: i32 = 1062;

/// Start and stop waits (the service has its own bounded shutdown).
const TIMEOUT: Duration = Duration::from_secs(60);
const POLL: Duration = Duration::from_millis(250);

/// The Win32 error code of a `windows-service` error.
fn os_code(e: &windows_service::Error) -> Option<i32> {
    match e {
        windows_service::Error::Winapi(io) => io.raw_os_error(),
        _ => None,
    }
}

fn map_state(s: WinState) -> ServiceState {
    match s {
        WinState::Stopped => ServiceState::Stopped,
        WinState::StartPending | WinState::ContinuePending => ServiceState::Starting,
        WinState::StopPending | WinState::PausePending => ServiceState::Stopping,
        WinState::Running => ServiceState::Running,
        WinState::Paused => ServiceState::Unknown,
    }
}

fn manager() -> Result<ServiceManager> {
    ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT).context(
        "Could not connect to the Windows Service Control Manager; check that the \
         `services` process is running (services.msc)",
    )
}

fn status(svc: &Service) -> Result<WinState> {
    Ok(svc
        .query_status()
        .context("Could not query the Blue Onyx Prism service status")?
        .current_state)
}

pub(super) fn query() -> Result<ServiceState> {
    match manager()?.open_service(WINDOWS_SERVICE_NAME, ServiceAccess::QUERY_STATUS) {
        Ok(svc) => Ok(map_state(status(&svc)?)),
        Err(e) if os_code(&e) == Some(ERROR_SERVICE_DOES_NOT_EXIST) => {
            Ok(ServiceState::NotInstalled)
        }
        Err(e) => Err(anyhow::Error::new(e).context(format!(
            "Could not open the Windows service {WINDOWS_SERVICE_NAME}"
        ))),
    }
}

pub(super) fn control(action: Action) -> Result<ServiceState> {
    let access = ServiceAccess::QUERY_STATUS | ServiceAccess::START | ServiceAccess::STOP;
    let svc = match manager()?.open_service(WINDOWS_SERVICE_NAME, access) {
        Ok(svc) => svc,
        Err(e) if os_code(&e) == Some(ERROR_SERVICE_DOES_NOT_EXIST) => {
            return Err(not_installed_error(action));
        }
        Err(e) if os_code(&e) == Some(ERROR_ACCESS_DENIED) => {
            return Err(admin_rights_error(action));
        }
        Err(e) => {
            return Err(anyhow::Error::new(e).context(format!(
                "Could not open the Windows service {WINDOWS_SERVICE_NAME}"
            )));
        }
    };
    match action {
        Action::Start => start(&svc, action),
        Action::Stop => stop(&svc, action),
        Action::Restart => {
            stop(&svc, action)?;
            start(&svc, action)
        }
    }
}

/// Poll while `pending(state)` holds, up to [`TIMEOUT`]; returns the last state.
fn wait_while(svc: &Service, pending: impl Fn(WinState) -> bool) -> Result<WinState> {
    let deadline = Instant::now() + TIMEOUT;
    loop {
        let state = status(svc)?;
        if !pending(state) || Instant::now() >= deadline {
            return Ok(state);
        }
        std::thread::sleep(POLL);
    }
}

fn start(svc: &Service, action: Action) -> Result<ServiceState> {
    // A stop in progress has to finish first.
    let state = wait_while(svc, |s| s == WinState::StopPending)?;
    if state == WinState::Running {
        return Ok(ServiceState::Running);
    }
    if state == WinState::Stopped {
        match svc.start::<&str>(&[]) {
            Ok(()) => {}
            Err(e) if os_code(&e) == Some(ERROR_SERVICE_ALREADY_RUNNING) => {}
            Err(e) if os_code(&e) == Some(ERROR_ACCESS_DENIED) => {
                return Err(admin_rights_error(action));
            }
            Err(e) => {
                return Err(anyhow::Error::new(e).context(
                    "Starting the service failed; see Event Viewer > Windows Logs > Application \
                     (source BlueOnyxPrism) and the service log next to the executable",
                ));
            }
        }
    }
    let state = wait_while(svc, |s| s != WinState::Running && s != WinState::Stopped)?;
    if state == WinState::Stopped {
        bail!(
            "The service stopped right after starting. See Event Viewer > Windows Logs > \
             Application (source BlueOnyxPrism) and the service log next to the executable for \
             the reason."
        );
    }
    Ok(map_state(state))
}

fn stop(svc: &Service, action: Action) -> Result<ServiceState> {
    // A start in progress cannot take a stop control yet.
    let state = wait_while(svc, |s| s == WinState::StartPending)?;
    if state == WinState::Stopped {
        return Ok(ServiceState::Stopped);
    }
    if state != WinState::StopPending {
        match svc.stop() {
            Ok(_) => {}
            Err(e)
                if matches!(
                    os_code(&e),
                    Some(ERROR_SERVICE_NOT_ACTIVE | ERROR_SERVICE_CANNOT_ACCEPT_CTRL)
                ) => {}
            Err(e) if os_code(&e) == Some(ERROR_ACCESS_DENIED) => {
                return Err(admin_rights_error(action));
            }
            Err(e) => {
                return Err(anyhow::Error::new(e).context("Stopping the service failed"));
            }
        }
    }
    let state = wait_while(svc, |s| s != WinState::Stopped)?;
    if state != WinState::Stopped {
        bail!(
            "Stopping the service timed out: it is still {} after {} s. It may still be \
             shutting down; check `blue-onyx-prism service status` again, or end the process \
             in Task Manager.",
            map_state(state).label(),
            TIMEOUT.as_secs()
        );
    }
    Ok(ServiceState::Stopped)
}
