//! When the tray's poll thread asks the service manager and the server (pure, tested on every
//! OS; `app` drives it).
//!
//! Every round probes `GET /v1/status`. The service manager (`launchctl print` on macOS, a
//! fork + exec and an XPC round trip; the SCM on Windows) is asked only when the probe alone
//! cannot tell: right after an action (a poke) or at the first round, when the probe gets no
//! status answer, when the last known service state is not Running, and otherwise every
//! [`SERVICE_REFRESH`] (a CLI server started by hand on the service's port answers the probe
//! too, so this is what still shows "service stopped" then). The wait between rounds is
//! [`POLL_FAST`] for [`FAST_FOR`] after an action or a change of what the tray shows, and while
//! the service is starting or stopping; otherwise [`POLL_IDLE`]. An idle tray thus sends four
//! status requests a minute and asks launchd twice a minute (it was 20 of each).

use super::probe::HttpStatus;
use crate::service_ctl::ServiceState;
use std::time::{Duration, Instant};

/// Wait between rounds after an action or a change (the tray's earlier fixed interval): a
/// start, stop or model load shows within this.
pub const POLL_FAST: Duration = Duration::from_secs(3);

/// Wait between rounds while nothing changes: a change nobody asked for through the tray (the
/// service stopped elsewhere, a model failed) shows within this.
pub const POLL_IDLE: Duration = Duration::from_secs(15);

/// How long the tray keeps polling at [`POLL_FAST`] after the last action or change: covers a
/// service start up to its HTTP listener and the first model loads, whose state changes keep
/// it fast for longer.
pub const FAST_FOR: Duration = Duration::from_secs(30);

/// Longest the tray trusts a Running service state while the probe keeps answering, before it
/// asks the service manager again.
pub const SERVICE_REFRESH: Duration = Duration::from_secs(30);

/// The poll thread's memory of past rounds.
#[derive(Debug, Default)]
pub struct Schedule {
    /// When the service manager was last asked, and what it said (`None`: the query failed).
    queried: Option<(Instant, Option<ServiceState>)>,
    /// What the last round showed (service state, probe result).
    shown: Option<(Option<ServiceState>, HttpStatus)>,
    /// The last action or change: polls are fast until [`FAST_FOR`] after it.
    fast_since: Option<Instant>,
}

impl Schedule {
    /// Whether this round asks the service manager, given this round's probe result and whether
    /// an action just poked the thread.
    pub fn query_service(&self, now: Instant, poked: bool, http: &HttpStatus) -> bool {
        let Some((at, state)) = self.queried else {
            return true;
        };
        poked
            || !matches!(http, HttpStatus::Answer(_))
            || state != Some(ServiceState::Running)
            || now.saturating_duration_since(at) >= SERVICE_REFRESH
    }

    /// Record the service manager's answer of this round.
    pub fn queried(&mut self, now: Instant, state: Option<ServiceState>) {
        self.queried = Some((now, state));
    }

    /// The last known service state (what the tray shows between queries).
    pub fn service(&self) -> Option<ServiceState> {
        self.queried.and_then(|(_, s)| s)
    }

    /// Record what this round shows; returns the wait before the next round.
    pub fn next_wait(&mut self, now: Instant, poked: bool, http: &HttpStatus) -> Duration {
        let service = self.service();
        let changed = self
            .shown
            .as_ref()
            .is_none_or(|(s, h)| *s != service || h != http);
        if changed {
            self.shown = Some((service, http.clone()));
        }
        if changed || poked {
            self.fast_since = Some(now);
        }
        let moving = matches!(
            service,
            Some(ServiceState::Starting | ServiceState::Stopping)
        );
        let recent = self
            .fast_since
            .is_some_and(|t| now.saturating_duration_since(t) < FAST_FOR);
        if moving || recent {
            POLL_FAST
        } else {
            POLL_IDLE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tray::probe::StatusReport;

    fn answer(status: &str) -> HttpStatus {
        HttpStatus::Answer(StatusReport {
            http_code: 200,
            ready: status == "ready",
            status: status.into(),
            models: vec![],
            version: Some("1".into()),
        })
    }

    /// One round as the poll thread runs it; true when it asked the service manager.
    fn round(
        s: &mut Schedule,
        now: Instant,
        poked: bool,
        http: &HttpStatus,
        service: Option<ServiceState>,
    ) -> (bool, Duration) {
        let asked = s.query_service(now, poked, http);
        if asked {
            s.queried(now, service);
        }
        (asked, s.next_wait(now, poked, http))
    }

    #[test]
    fn running_and_answering_asks_launchd_only_at_the_refresh_mark() {
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        let ok = answer("ready");
        let running = Some(ServiceState::Running);
        let mut s = Schedule::default();
        // First round: asked, fast.
        assert_eq!(round(&mut s, t0, true, &ok, running), (true, POLL_FAST));
        // Nothing changes: not asked again until the refresh mark; fast for FAST_FOR, then idle.
        let mut asked_at = vec![];
        let mut t = 0;
        while t <= 125 {
            let (asked, wait) = round(&mut s, at(t), false, &ok, running);
            if asked {
                asked_at.push(t);
            }
            assert_eq!(wait, if t < 30 { POLL_FAST } else { POLL_IDLE }, "at {t} s");
            t += wait.as_secs();
        }
        assert!(!asked_at.is_empty());
        for (i, t) in asked_at.iter().enumerate() {
            let since = if i == 0 { 0 } else { asked_at[i - 1] };
            assert!(t - since >= SERVICE_REFRESH.as_secs(), "{asked_at:?}");
            assert!(t - since <= SERVICE_REFRESH.as_secs() + POLL_IDLE.as_secs());
        }
        assert!(asked_at.len() <= 4, "{asked_at:?}");
        assert_eq!(s.service(), running, "the known state stays shown");
    }

    #[test]
    fn a_poke_or_a_failed_probe_asks_now() {
        let t0 = Instant::now();
        let ok = answer("ready");
        let running = Some(ServiceState::Running);
        let mut s = Schedule::default();
        round(&mut s, t0, true, &ok, running);
        let t1 = t0 + Duration::from_secs(5);
        assert!(!s.query_service(t1, false, &ok));
        assert!(s.query_service(t1, true, &ok), "after an action");
        assert!(s.query_service(t1, false, &HttpStatus::Unreachable));
        assert!(s.query_service(t1, false, &HttpStatus::Invalid("HTTP 404".into())));
        // A state other than Running (or a failed query) is asked again every round.
        for other in [
            None,
            Some(ServiceState::Stopped),
            Some(ServiceState::Starting),
        ] {
            s.queried(t1, other);
            assert!(s.query_service(t1, false, &ok), "{other:?}");
        }
    }

    #[test]
    fn the_status_poll_backs_off_and_resets_on_change_or_action() {
        let t0 = Instant::now();
        let at = |s: u64| t0 + Duration::from_secs(s);
        let running = Some(ServiceState::Running);
        let mut s = Schedule::default();
        s.queried(t0, running);
        assert_eq!(s.next_wait(t0, true, &answer("ready")), POLL_FAST);
        assert_eq!(s.next_wait(at(29), false, &answer("ready")), POLL_FAST);
        assert_eq!(s.next_wait(at(30), false, &answer("ready")), POLL_IDLE);
        assert_eq!(s.next_wait(at(100), false, &answer("ready")), POLL_IDLE);
        // A change of what the probe shows: fast again, for FAST_FOR.
        assert_eq!(s.next_wait(at(115), false, &answer("degraded")), POLL_FAST);
        assert_eq!(s.next_wait(at(144), false, &answer("degraded")), POLL_FAST);
        assert_eq!(s.next_wait(at(145), false, &answer("degraded")), POLL_IDLE);
        // An action: fast again.
        assert_eq!(s.next_wait(at(200), true, &answer("degraded")), POLL_FAST);
        assert_eq!(s.next_wait(at(230), false, &answer("degraded")), POLL_IDLE);
        // A service state change counts as a change; starting / stopping stays fast.
        s.queried(at(300), Some(ServiceState::Stopping));
        assert_eq!(
            s.next_wait(at(300), false, &HttpStatus::Unreachable),
            POLL_FAST
        );
        assert_eq!(
            s.next_wait(at(400), false, &HttpStatus::Unreachable),
            POLL_FAST
        );
        s.queried(at(500), Some(ServiceState::Stopped));
        assert_eq!(
            s.next_wait(at(500), false, &HttpStatus::Unreachable),
            POLL_FAST
        );
        assert_eq!(
            s.next_wait(at(530), false, &HttpStatus::Unreachable),
            POLL_IDLE
        );
    }
}
