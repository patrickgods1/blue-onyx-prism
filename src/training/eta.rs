//! How long the job on the queue has run and how long its current step should still take, for the
//! Jobs tab of the Train area. The clock lives in the service's memory only: a job that
//! survives no restart needs no record.
//!
//! The rate is measured from the first percent a step gains, not from its start: what comes before
//! it (loading a model, opening a video) does not repeat, and counting it would overstate every
//! early estimate. Time spent paused counts as nothing. A step's percent going back (a stage that
//! starts over) measures again from there.

use std::time::{Duration, Instant};

/// Percent a step must gain from where it is measured before it gives an estimate.
const MIN_GAIN: u8 = 2;
/// Seconds of measured progress before a step gives an estimate.
const MIN_MEASURED_S: f64 = 5.0;

/// What the clock says at one moment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JobTiming {
    /// Time the job ran, pauses left out.
    pub elapsed_s: f64,
    /// Time it spent paused in the current step (a pause before it is no part of that step's
    /// own elapsed time).
    pub paused_s: f64,
    /// Estimated rest of the current step; None until it can be measured.
    pub step_left_s: Option<f64>,
}

/// The clock of the running job.
#[derive(Debug, Clone)]
pub struct JobClock {
    started: Instant,
    paused: Duration,
    paused_since: Option<Instant>,
    /// Paused time when the current step began.
    paused_before_step: Duration,
    step: usize,
    percent: u8,
    /// Running seconds and percent the current step is measured from.
    from: Option<(f64, u8)>,
}

impl JobClock {
    pub fn new(now: Instant) -> Self {
        Self {
            started: now,
            paused: Duration::ZERO,
            paused_since: None,
            paused_before_step: Duration::ZERO,
            step: 0,
            percent: 0,
            from: None,
        }
    }

    fn paused_at(&self, now: Instant) -> Duration {
        self.paused
            + self
                .paused_since
                .map_or(Duration::ZERO, |s| now.saturating_duration_since(s))
    }

    fn running_at(&self, now: Instant) -> f64 {
        now.saturating_duration_since(self.started)
            .saturating_sub(self.paused_at(now))
            .as_secs_f64()
    }

    /// Step `step` stands at `percent` percent of itself.
    pub fn report(&mut self, now: Instant, step: usize, percent: u8) {
        if step != self.step {
            self.paused_before_step = self.paused_at(now);
        }
        if step != self.step || percent < self.percent {
            self.step = step;
            self.from = None;
        } else if percent > self.percent && self.from.is_none() {
            self.from = Some((self.running_at(now), percent));
        }
        self.percent = percent;
    }

    pub fn pause(&mut self, now: Instant) {
        self.paused_since.get_or_insert(now);
    }

    pub fn resume(&mut self, now: Instant) {
        if let Some(s) = self.paused_since.take() {
            self.paused += now.saturating_duration_since(s);
        }
    }

    pub fn timing(&self, now: Instant) -> JobTiming {
        let running = self.running_at(now);
        let step_left_s = self.from.and_then(|(at, from)| {
            let gained = self.percent.saturating_sub(from);
            let measured = running - at;
            (gained >= MIN_GAIN && measured >= MIN_MEASURED_S && self.percent < 100)
                .then(|| f64::from(100 - self.percent) * measured / f64::from(gained))
        });
        JobTiming {
            elapsed_s: running,
            paused_s: self
                .paused_at(now)
                .saturating_sub(self.paused_before_step)
                .as_secs_f64(),
            step_left_s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(t0: Instant, s: u64) -> Instant {
        t0 + Duration::from_secs(s)
    }

    #[test]
    fn a_step_is_measured_from_its_first_gain_so_a_slow_start_does_not_count() {
        let t0 = Instant::now();
        let mut c = JobClock::new(t0);
        c.report(t0, 1, 0);
        // A model takes 100 s to load, then 1% every 2 s.
        c.report(at(t0, 100), 1, 1);
        assert_eq!(c.timing(at(t0, 100)).step_left_s, None);
        c.report(at(t0, 120), 1, 11);
        let left = c.timing(at(t0, 120)).step_left_s.unwrap();
        assert!((left - 178.0).abs() < 1e-6, "{left}");
    }

    #[test]
    fn no_estimate_until_enough_has_been_measured_nor_at_the_end() {
        let t0 = Instant::now();
        let mut c = JobClock::new(t0);
        c.report(t0, 1, 0);
        c.report(at(t0, 1), 1, 10);
        c.report(at(t0, 2), 1, 11);
        assert_eq!(c.timing(at(t0, 2)).step_left_s, None, "one percent gained");
        c.report(at(t0, 3), 1, 50);
        assert_eq!(
            c.timing(at(t0, 3)).step_left_s,
            None,
            "two seconds measured"
        );
        c.report(at(t0, 20), 1, 100);
        assert_eq!(c.timing(at(t0, 20)).step_left_s, None);
    }

    #[test]
    fn a_new_step_or_a_fall_measures_again() {
        let t0 = Instant::now();
        let mut c = JobClock::new(t0);
        c.report(t0, 1, 0);
        c.report(at(t0, 10), 1, 10);
        c.report(at(t0, 20), 1, 20);
        assert!(c.timing(at(t0, 20)).step_left_s.is_some());
        c.report(at(t0, 21), 2, 40);
        assert_eq!(c.timing(at(t0, 40)).step_left_s, None);
        c.report(at(t0, 50), 2, 50);
        c.report(at(t0, 60), 2, 60);
        assert!(c.timing(at(t0, 60)).step_left_s.is_some());
        c.report(at(t0, 61), 2, 10);
        assert_eq!(c.timing(at(t0, 61)).step_left_s, None);
    }

    #[test]
    fn a_stall_lengthens_the_estimate() {
        let t0 = Instant::now();
        let mut c = JobClock::new(t0);
        c.report(t0, 1, 0);
        c.report(at(t0, 10), 1, 10);
        c.report(at(t0, 20), 1, 20);
        let soon = c.timing(at(t0, 20)).step_left_s.unwrap();
        let later = c.timing(at(t0, 120)).step_left_s.unwrap();
        assert!(later > soon, "{later} > {soon}");
    }

    #[test]
    fn only_the_current_steps_pauses_are_reported() {
        let t0 = Instant::now();
        let mut c = JobClock::new(t0);
        c.report(t0, 1, 0);
        c.pause(at(t0, 10));
        c.resume(at(t0, 3610));
        c.report(at(t0, 3620), 2, 0);
        let t = c.timing(at(t0, 3700));
        assert_eq!(t.paused_s, 0.0);
        assert_eq!(t.elapsed_s, 100.0);
        c.pause(at(t0, 3700));
        assert_eq!(c.timing(at(t0, 3730)).paused_s, 30.0);
    }

    #[test]
    fn paused_time_counts_as_nothing() {
        let t0 = Instant::now();
        let mut c = JobClock::new(t0);
        c.report(t0, 1, 0);
        c.report(at(t0, 10), 1, 10);
        c.report(at(t0, 20), 1, 20);
        let before = c.timing(at(t0, 20));
        c.pause(at(t0, 20));
        c.pause(at(t0, 500));
        let during = c.timing(at(t0, 1000));
        assert_eq!(during.step_left_s, before.step_left_s);
        assert_eq!(during.elapsed_s, 20.0);
        assert_eq!(during.paused_s, 980.0);
        c.resume(at(t0, 1000));
        c.resume(at(t0, 1100));
        let after = c.timing(at(t0, 1000));
        assert_eq!(after.step_left_s, before.step_left_s);
        assert_eq!(after.paused_s, 980.0);
    }
}
