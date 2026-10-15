//! Pure display model of the tray app: combines the service manager's state, the `/v1/status`
//! probe and an action in progress into one [`Display`] (menu status line, tooltip, icon color
//! and which menu actions are enabled). No GUI and no I/O, so it is unit-tested everywhere.

use crate::service_ctl::{Action, ServiceState};
use crate::tray::probe::{HttpStatus, StatusReport};

/// Product name used in every text.
pub const PRODUCT: &str = "Blue Onyx Prism";
/// Longest status detail (e.g. a download progress line) shown in the menu.
const MAX_DETAIL_CHARS: usize = 70;
/// Windows truncates tray tooltips at 127 UTF-16 units; keep a margin on every OS.
const MAX_TOOLTIP_CHARS: usize = 120;

/// Overall state as the tray shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Health {
    NotInstalled,
    Stopped,
    /// Service starting, or running but HTTP not answering yet / models initializing.
    Starting,
    Ready,
    /// Serving, but some model failed.
    Degraded,
    Failed,
    Stopping,
    /// A start/stop/restart started from the tray is in progress.
    Busy(Action),
    Unknown,
}

/// Status dot color of the icon.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IconColor {
    Green,
    Amber,
    Red,
    Grey,
}

/// Everything the menu and the icon show for one state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Display {
    pub health: Health,
    /// First (disabled) menu line.
    pub status_line: String,
    pub tooltip: String,
    pub color: IconColor,
    pub can_start: bool,
    pub can_stop: bool,
    pub can_restart: bool,
    pub can_open_ui: bool,
}

/// Inputs of [`display`]: the latest poll plus the tray's own state.
#[derive(Clone, Debug, Default)]
pub struct Inputs<'a> {
    /// `service_ctl::query()`; `None` when it failed (treated as unknown).
    pub service: Option<ServiceState>,
    pub http: Option<&'a HttpStatus>,
    /// Action running right now.
    pub busy: Option<Action>,
    /// Error of the last action, still within its display time.
    pub error: Option<&'a str>,
}

/// Text of a busy state ("Starting…").
pub fn busy_text(action: Action) -> &'static str {
    match action {
        Action::Start => "Starting\u{2026}",
        Action::Stop => "Stopping\u{2026}",
        Action::Restart => "Restarting\u{2026}",
    }
}

/// Combine the inputs into what the tray shows.
pub fn display(inp: &Inputs<'_>) -> Display {
    let report = match inp.http {
        Some(HttpStatus::Answer(r)) => Some(r),
        _ => None,
    };
    let answers = report.is_some();
    let service = inp.service.unwrap_or(ServiceState::Unknown);

    let (health, text) = if let Some(a) = inp.busy {
        (Health::Busy(a), busy_text(a).to_string())
    } else if inp.service.is_none() && inp.http.is_none() {
        // Before the first poll.
        (Health::Unknown, "checking\u{2026}".to_string())
    } else {
        match service {
            ServiceState::NotInstalled => (Health::NotInstalled, String::new()),
            ServiceState::Stopped => (Health::Stopped, "stopped".to_string()),
            ServiceState::Stopping => (Health::Stopping, "stopping".to_string()),
            ServiceState::Starting => match report {
                Some(r) => from_report(r),
                None => (Health::Starting, "starting".to_string()),
            },
            ServiceState::Running => match (report, inp.http) {
                (Some(r), _) => from_report(r),
                (None, Some(HttpStatus::Invalid(e))) => (
                    Health::Unknown,
                    format!("running (status unreadable: {})", clip(e, MAX_DETAIL_CHARS)),
                ),
                _ => (
                    Health::Starting,
                    "starting (waiting for the web server)".to_string(),
                ),
            },
            ServiceState::Unknown => match report {
                Some(r) => from_report(r),
                None => (Health::Unknown, "status unknown".to_string()),
            },
        }
    };

    let status_line = if health == Health::NotInstalled {
        format!("{PRODUCT} is not installed")
    } else {
        format!("{PRODUCT}: {text}")
    };
    let color = match health {
        Health::Ready => IconColor::Green,
        Health::Starting | Health::Degraded | Health::Busy(_) | Health::Stopping => {
            IconColor::Amber
        }
        Health::Failed => IconColor::Red,
        Health::Stopped | Health::Unknown | Health::NotInstalled => IconColor::Grey,
    };

    let idle = inp.busy.is_none();
    // Controls follow the service manager. With an unknown service state the HTTP probe
    // decides: a server that answers can be stopped / restarted, a silent one started.
    let (can_start, can_stop) = match service {
        ServiceState::Stopped => (true, false),
        ServiceState::Running | ServiceState::Starting => (false, true),
        ServiceState::Unknown if inp.http.is_none() => (false, false),
        ServiceState::Unknown => (!answers, answers),
        _ => (false, false),
    };
    let tooltip_base = match (&health, inp.error) {
        (_, Some(e)) => format!("{status_line}\nLast action failed: {e}"),
        (Health::Ready | Health::Degraded | Health::Starting | Health::Failed, None) => {
            format!("{status_line}\nQuitting the tray keeps the service running")
        }
        _ => status_line.clone(),
    };
    Display {
        status_line,
        tooltip: clip(&tooltip_base, MAX_TOOLTIP_CHARS),
        color,
        can_start: idle && can_start,
        can_stop: idle && can_stop,
        can_restart: idle && can_stop,
        can_open_ui: idle && answers,
        health,
    }
}

/// Health and text from a `/v1/status` answer.
fn from_report(r: &StatusReport) -> (Health, String) {
    let usable = r
        .models
        .iter()
        .filter(|m| m.state == "ready" || m.state == "lazy")
        .count();
    let failed: Vec<_> = r.models.iter().filter(|m| m.state == "failed").collect();
    let total = r.models.len();
    match r.status.as_str() {
        "ready" => (
            Health::Ready,
            format!("running \u{2014} {}", models_ready(usable)),
        ),
        "degraded" => (
            Health::Degraded,
            format!(
                "running \u{2014} {} of {total} model{} failed",
                failed.len(),
                if total == 1 { "" } else { "s" }
            ),
        ),
        "initializing" => {
            let detail = r
                .models
                .iter()
                .filter(|m| m.state == "initializing")
                .find_map(|m| m.detail.as_deref().filter(|d| !d.trim().is_empty()));
            let text = match detail {
                Some(d) => format!("starting ({})", clip(d.trim(), MAX_DETAIL_CHARS)),
                None => "starting".to_string(),
            };
            (Health::Starting, text)
        }
        "failed" => {
            let text = match failed.first() {
                Some(m) => match m.detail.as_deref().filter(|d| !d.trim().is_empty()) {
                    Some(d) => format!("failed ({}: {})", m.name, clip(d.trim(), MAX_DETAIL_CHARS)),
                    None => format!("failed ({})", m.name),
                },
                None if total == 0 => "failed (no models configured)".to_string(),
                None => "failed".to_string(),
            };
            (Health::Failed, text)
        }
        // Unknown labels from a newer server: trust the `ready` flag.
        other if r.ready => (
            Health::Ready,
            format!("running ({other}) \u{2014} {}", models_ready(usable)),
        ),
        other => (Health::Starting, format!("starting ({other})")),
    }
}

fn models_ready(n: usize) -> String {
    format!("{n} model{} ready", if n == 1 { "" } else { "s" })
}

/// At most `max` characters of `s` (on a char boundary), with an ellipsis when cut.
pub fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('\u{2026}');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tray::probe::ModelStatus;

    fn model(name: &str, state: &str, detail: Option<&str>) -> ModelStatus {
        ModelStatus {
            name: name.into(),
            state: state.into(),
            device: Some("GPU".into()),
            detail: detail.map(Into::into),
        }
    }

    fn answer(code: u16, status: &str, models: Vec<ModelStatus>) -> HttpStatus {
        HttpStatus::Answer(StatusReport {
            http_code: code,
            ready: code == 200,
            status: status.into(),
            models,
            version: Some("1.0.0".into()),
        })
    }

    fn show(service: Option<ServiceState>, http: &HttpStatus) -> Display {
        display(&Inputs {
            service,
            http: Some(http),
            busy: None,
            error: None,
        })
    }

    #[test]
    fn running_and_ready() {
        let http = answer(
            200,
            "ready",
            vec![model("a", "ready", None), model("b", "lazy", None)],
        );
        let d = show(Some(ServiceState::Running), &http);
        assert_eq!(d.health, Health::Ready);
        assert_eq!(
            d.status_line,
            "Blue Onyx Prism: running \u{2014} 2 models ready"
        );
        assert_eq!(d.color, IconColor::Green);
        assert!(!d.can_start && d.can_stop && d.can_restart && d.can_open_ui);
        assert!(d.tooltip.contains("keeps the service running"));
    }

    #[test]
    fn one_model_ready_is_singular() {
        let http = answer(200, "ready", vec![model("a", "ready", None)]);
        let d = show(Some(ServiceState::Running), &http);
        assert_eq!(
            d.status_line,
            "Blue Onyx Prism: running \u{2014} 1 model ready"
        );
    }

    #[test]
    fn starting_shows_the_download_progress() {
        let http = answer(
            503,
            "initializing",
            vec![model(
                "IPcam-general",
                "initializing",
                Some("downloading OpenVINO runtime 42%"),
            )],
        );
        let d = show(Some(ServiceState::Running), &http);
        assert_eq!(d.health, Health::Starting);
        assert_eq!(
            d.status_line,
            "Blue Onyx Prism: starting (downloading OpenVINO runtime 42%)"
        );
        assert_eq!(d.color, IconColor::Amber);
        assert!(d.can_stop && d.can_restart && !d.can_start && d.can_open_ui);
    }

    #[test]
    fn running_without_http_is_starting() {
        let d = show(Some(ServiceState::Running), &HttpStatus::Unreachable);
        assert_eq!(d.health, Health::Starting);
        assert_eq!(
            d.status_line,
            "Blue Onyx Prism: starting (waiting for the web server)"
        );
        assert!(!d.can_open_ui && d.can_stop);
    }

    #[test]
    fn service_starting_without_http() {
        let d = show(Some(ServiceState::Starting), &HttpStatus::Unreachable);
        assert_eq!(d.health, Health::Starting);
        assert_eq!(d.status_line, "Blue Onyx Prism: starting");
        assert!(d.can_stop && !d.can_start);
    }

    #[test]
    fn degraded_and_failed() {
        let http = answer(
            200,
            "degraded",
            vec![
                model("a", "ready", None),
                model("b", "failed", Some("boom")),
            ],
        );
        let d = show(Some(ServiceState::Running), &http);
        assert_eq!(d.health, Health::Degraded);
        assert_eq!(
            d.status_line,
            "Blue Onyx Prism: running \u{2014} 1 of 2 models failed"
        );
        assert_eq!(d.color, IconColor::Amber);

        let http = answer(
            503,
            "failed",
            vec![model("b", "failed", Some("no GPU found"))],
        );
        let d = show(Some(ServiceState::Running), &http);
        assert_eq!(d.health, Health::Failed);
        assert_eq!(d.status_line, "Blue Onyx Prism: failed (b: no GPU found)");
        assert_eq!(d.color, IconColor::Red);
        assert!(d.can_stop && d.can_restart && d.can_open_ui && !d.can_start);
    }

    #[test]
    fn stopped_and_not_installed() {
        let d = show(Some(ServiceState::Stopped), &HttpStatus::Unreachable);
        assert_eq!(d.health, Health::Stopped);
        assert_eq!(d.status_line, "Blue Onyx Prism: stopped");
        assert_eq!(d.color, IconColor::Grey);
        assert!(d.can_start && !d.can_stop && !d.can_restart && !d.can_open_ui);

        let d = show(Some(ServiceState::NotInstalled), &HttpStatus::Unreachable);
        assert_eq!(d.health, Health::NotInstalled);
        assert_eq!(d.status_line, "Blue Onyx Prism is not installed");
        assert_eq!(d.color, IconColor::Grey);
        assert!(!d.can_start && !d.can_stop && !d.can_restart && !d.can_open_ui);
    }

    #[test]
    fn stopping_allows_nothing_but_the_ui() {
        let http = answer(200, "ready", vec![model("a", "ready", None)]);
        let d = show(Some(ServiceState::Stopping), &http);
        assert_eq!(d.health, Health::Stopping);
        assert_eq!(d.status_line, "Blue Onyx Prism: stopping");
        assert!(!d.can_start && !d.can_stop && !d.can_restart && d.can_open_ui);
    }

    #[test]
    fn busy_disables_everything() {
        let http = answer(200, "ready", vec![model("a", "ready", None)]);
        let d = display(&Inputs {
            service: Some(ServiceState::Running),
            http: Some(&http),
            busy: Some(Action::Restart),
            error: None,
        });
        assert_eq!(d.health, Health::Busy(Action::Restart));
        assert_eq!(d.status_line, "Blue Onyx Prism: Restarting\u{2026}");
        assert_eq!(d.color, IconColor::Amber);
        assert!(!d.can_start && !d.can_stop && !d.can_restart && !d.can_open_ui);
    }

    #[test]
    fn unknown_service_follows_http() {
        let http = answer(200, "ready", vec![model("a", "ready", None)]);
        let d = show(None, &http);
        assert_eq!(d.health, Health::Ready);
        assert!(d.can_stop && d.can_restart && !d.can_start);

        let d = show(None, &HttpStatus::Unreachable);
        assert_eq!(d.health, Health::Unknown);
        assert_eq!(d.status_line, "Blue Onyx Prism: status unknown");
        assert!(d.can_start && !d.can_stop);
    }

    #[test]
    fn before_the_first_poll() {
        let d = display(&Inputs::default());
        assert_eq!(d.health, Health::Unknown);
        assert_eq!(d.status_line, "Blue Onyx Prism: checking\u{2026}");
        assert!(!d.can_open_ui && !d.can_stop && !d.can_start);
    }

    #[test]
    fn error_goes_into_the_tooltip_clipped() {
        let long = "x".repeat(500);
        let d = display(&Inputs {
            service: Some(ServiceState::Stopped),
            http: Some(&HttpStatus::Unreachable),
            busy: None,
            error: Some(&long),
        });
        assert!(d.tooltip.contains("Last action failed: x"));
        assert_eq!(d.tooltip.chars().count(), MAX_TOOLTIP_CHARS);
        assert!(d.tooltip.ends_with('\u{2026}'));
    }

    #[test]
    fn long_detail_is_clipped() {
        let detail = "d".repeat(200);
        let http = answer(
            503,
            "initializing",
            vec![model("a", "initializing", Some(&detail))],
        );
        let d = show(Some(ServiceState::Running), &http);
        assert!(d.status_line.chars().count() < 100);
        assert!(d.status_line.ends_with("\u{2026})"));
    }

    #[test]
    fn clip_respects_char_boundaries() {
        assert_eq!(clip("h\u{e9}llo", 10), "h\u{e9}llo");
        assert_eq!(clip("h\u{e9}llo", 3), "h\u{e9}\u{2026}");
    }
}
