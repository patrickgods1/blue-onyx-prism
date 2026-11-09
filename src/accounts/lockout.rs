//! Per-account lockout (pure; the caller passes `now` in unix seconds).
//!
//! NIST SP 800-63B-4 3.2.2 asks for at most 100 consecutive failures per account. Here:
//! - fewer than [`DELAY_AFTER`] (10) failures in a row: every attempt is allowed;
//! - from 10 on: an attempt is refused without hashing until `last_failure + delay`, the delay
//!   1 s at 10 failures and doubling with each further one, capped at [`MAX_DELAY_SECS`]
//!   (15 min; reached at 20 failures);
//! - at [`DISABLE_AT`] (100) in a row the account is disabled until `user unlock` or
//!   `user reset-password` (which clear it), whatever the time;
//! - a success resets the count.
//!
//! [`UnknownNames`] applies the same rules, in memory, to names that have no account, so the
//! answers (and their delays) never tell which names exist. It holds at most
//! [`UNKNOWN_NAMES_CAP`] names and forgets the one with the oldest failure first.

use std::collections::HashMap;

/// Failures in a row after which attempts wait.
pub const DELAY_AFTER: u32 = 10;
/// Failures in a row at which the account is disabled.
pub const DISABLE_AT: u32 = 100;
/// The delay at [`DELAY_AFTER`] failures.
pub const FIRST_DELAY_SECS: u64 = 1;
/// The longest delay.
pub const MAX_DELAY_SECS: u64 = 15 * 60;
/// Names [`UnknownNames`] remembers at most.
pub const UNKNOWN_NAMES_CAP: usize = 1024;

/// An account's failure state (the fields of `users.json` it reads and writes).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct State {
    /// Consecutive failures.
    pub failures: u32,
    /// When the last one happened (unix seconds).
    pub last_failure: Option<u64>,
    /// Disabled (stays until unlocked).
    pub disabled: bool,
}

/// What to do with an attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Check the password.
    Allow,
    /// Refuse without checking; this many seconds are left (at least 1).
    Wait(u64),
    /// Refuse without checking: the account is disabled.
    Disabled,
}

/// The delay after `failures` consecutive failures (0 below [`DELAY_AFTER`]).
pub fn delay(failures: u32) -> u64 {
    if failures < DELAY_AFTER {
        return 0;
    }
    let doublings = failures - DELAY_AFTER;
    if doublings >= 20 {
        return MAX_DELAY_SECS;
    }
    (FIRST_DELAY_SECS << doublings).min(MAX_DELAY_SECS)
}

/// The decision for an attempt at `now` on an account in `state`.
pub fn decide(state: &State, now: u64) -> Decision {
    if state.disabled || state.failures >= DISABLE_AT {
        return Decision::Disabled;
    }
    let wait = delay(state.failures);
    if wait == 0 {
        return Decision::Allow;
    }
    let last = state.last_failure.unwrap_or(0);
    let until = last.saturating_add(wait);
    if now < until {
        Decision::Wait(until - now)
    } else {
        Decision::Allow
    }
}

/// `state` after a failure at `now`.
pub fn on_failure(state: &State, now: u64) -> State {
    let failures = state.failures.saturating_add(1);
    State {
        failures,
        last_failure: Some(now),
        disabled: state.disabled || failures >= DISABLE_AT,
    }
}

/// `state` after a success: nothing counted.
pub fn on_success(_state: &State) -> State {
    State::default()
}

/// The failure state of names that have no account (see the module docs).
#[derive(Debug, Default)]
pub struct UnknownNames {
    names: HashMap<String, (State, u64)>,
    /// Bumped by each failure: the order to forget in.
    seq: u64,
}

impl UnknownNames {
    /// An empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many names it holds.
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether it holds no name.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// The decision for `name` (lowercased by the caller) at `now`.
    pub fn decide(&self, name: &str, now: u64) -> Decision {
        self.names
            .get(name)
            .map_or(Decision::Allow, |(s, _)| decide(s, now))
    }

    /// Count a failure of `name` at `now`, forgetting the oldest name when full.
    pub fn record_failure(&mut self, name: &str, now: u64) {
        self.seq = self.seq.wrapping_add(1);
        let seq = self.seq;
        if let Some((state, at)) = self.names.get_mut(name) {
            *state = on_failure(state, now);
            *at = seq;
            return;
        }
        if self.names.len() >= UNKNOWN_NAMES_CAP {
            let oldest = self
                .names
                .iter()
                .min_by_key(|(_, (_, at))| *at)
                .map(|(n, _)| n.clone());
            if let Some(n) = oldest {
                self.names.remove(&n);
            }
        }
        self.names
            .insert(name.to_string(), (on_failure(&State::default(), now), seq));
    }

    /// A success cannot happen for a name with no account, but a name that gets an account
    /// starts clean: forget it.
    pub fn forget(&mut self, name: &str) {
        self.names.remove(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn after(n: u32, at: u64) -> State {
        (0..n).fold(State::default(), |s, _| on_failure(&s, at))
    }

    #[test]
    fn delays_double_from_ten_and_cap_at_fifteen_minutes() {
        let table: [(u32, u64); 14] = [
            (0, 0),
            (1, 0),
            (9, 0),
            (10, 1),
            (11, 2),
            (12, 4),
            (13, 8),
            (15, 32),
            (19, 512),
            (20, 900),
            (21, 900),
            (50, 900),
            (99, 900),
            (u32::MAX, 900),
        ];
        for (failures, want) in table {
            assert_eq!(delay(failures), want, "{failures}");
        }
    }

    #[test]
    fn decisions_follow_the_time() {
        let t = 1_000_000;
        assert_eq!(decide(&State::default(), t), Decision::Allow);
        assert_eq!(decide(&after(9, t), t), Decision::Allow);
        let ten = after(10, t);
        assert_eq!(ten.failures, 10);
        assert_eq!(decide(&ten, t), Decision::Wait(1));
        assert_eq!(decide(&ten, t + 1), Decision::Allow);
        let twelve = after(12, t);
        assert_eq!(decide(&twelve, t), Decision::Wait(4));
        assert_eq!(decide(&twelve, t + 3), Decision::Wait(1));
        assert_eq!(decide(&twelve, t + 4), Decision::Allow);
        let many = after(60, t);
        assert_eq!(decide(&many, t), Decision::Wait(900));
        assert_eq!(decide(&many, t + 899), Decision::Wait(1));
        assert_eq!(decide(&many, t + 900), Decision::Allow);
        // A clock that went back still waits at most the delay from the failure.
        assert_eq!(decide(&many, t - 5), Decision::Wait(905));
    }

    #[test]
    fn a_hundred_in_a_row_disable_until_unlocked() {
        let t = 5_000;
        let s = after(99, t);
        assert!(!s.disabled);
        let s = on_failure(&s, t);
        assert_eq!(s.failures, 100);
        assert!(s.disabled);
        assert_eq!(decide(&s, t + 10 * 365 * 86_400), Decision::Disabled);
        // Disabled stays through more failures, and a success does not happen (no check runs).
        assert!(on_failure(&s, t).disabled);
        assert_eq!(on_success(&s), State::default());
        let flagged = State {
            disabled: true,
            ..State::default()
        };
        assert_eq!(decide(&flagged, t), Decision::Disabled);
    }

    #[test]
    fn a_success_resets_the_count() {
        let t = 77;
        let s = after(15, t);
        let s = on_success(&s);
        assert_eq!(s, State::default());
        assert_eq!(decide(&s, t), Decision::Allow);
        assert_eq!(decide(&after(9, t), t), Decision::Allow);
    }

    #[test]
    fn unknown_names_get_the_same_delays_and_stay_bounded() {
        let t = 10_000;
        let mut u = UnknownNames::new();
        assert!(u.is_empty());
        for _ in 0..12 {
            u.record_failure("ghost", t);
        }
        assert_eq!(u.decide("ghost", t), decide(&after(12, t), t));
        assert_eq!(u.decide("ghost", t), Decision::Wait(4));
        assert_eq!(u.decide("other", t), Decision::Allow);
        for _ in 0..88 {
            u.record_failure("ghost", t);
        }
        assert_eq!(u.decide("ghost", t), Decision::Disabled);
        // Bounded: the name with the oldest failure goes first.
        for i in 0..UNKNOWN_NAMES_CAP + 50 {
            u.record_failure(&format!("n{i}"), t);
        }
        assert_eq!(u.len(), UNKNOWN_NAMES_CAP);
        assert_eq!(
            u.decide("ghost", t),
            Decision::Allow,
            "the oldest is forgotten"
        );
        assert_eq!(u.decide("n0", t), Decision::Allow);
        let last = format!("n{}", UNKNOWN_NAMES_CAP + 49);
        assert!(u.names.contains_key(&last));
        // A name failing again moves to the back of the line.
        let early = format!("n{}", 60);
        u.record_failure(&early, t);
        u.record_failure("fresh", t);
        assert!(u.names.contains_key(&early));
        u.forget("fresh");
        assert!(!u.names.contains_key("fresh"));
    }
}
