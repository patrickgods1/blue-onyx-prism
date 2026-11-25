//! The polling core the CLI's job commands share (`train run`, `train retry-export` and the task
//! commands: import, auto-label, overlaps, refine): poll a queued job until it is no longer busy,
//! pass a Ctrl-C on, print the log lines new since the last poll (`--verbose`, through
//! [`LogFollower`]) and a progress line when it changed, then hand the final record back. What
//! each command prints at the end stays its own.

use crate::training::service::LogFollower;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// A progress line that changed but not in its key is printed at most this often.
pub(super) const QUIET_PRINT_EVERY: Duration = Duration::from_secs(10);

/// What one poll of a followed job shows ([`follow`]).
pub(super) struct Shown<'a, K> {
    /// Still queued, running or paused: the follower polls again.
    pub busy: bool,
    /// Running: progress lines print only then.
    pub running: bool,
    /// The progress line.
    pub line: String,
    /// What prints a changed line at once (the stage and percent of a run, a task's step; the
    /// line itself to print every change); a change outside it prints at most every
    /// [`QUIET_PRINT_EVERY`].
    pub key: K,
    /// The record's log tail ([`crate::training::service::push_log`]).
    pub log: &'a [String],
}

/// Follow a job: `poll` gives its record and the push count of its log (None once the record is
/// gone), `show` how the record reads. Per poll, in order: a set `cancel` calls `on_cancel` (every
/// poll while set, as the commands always did), the new log lines go to `out` indented by four
/// spaces when `verbose`, `each` sees the record (a run's one-time balance note), and a running
/// job's progress line goes to `out` when it differs from the last one printed and its key
/// changed since that one, or [`QUIET_PRINT_EVERY`] passed (`first` is the key before any
/// line). Returns the first record that is no longer busy; None when the record vanished.
#[allow(
    clippy::too_many_arguments,
    reason = "the job's poll and view, its cancel, the output and the one per-poll extra"
)]
pub(super) fn follow<R, K: PartialEq>(
    mut poll: impl FnMut() -> Option<(R, u64)>,
    show: impl for<'a> Fn(&'a R) -> Shown<'a, K>,
    first: K,
    cancel: &AtomicBool,
    on_cancel: &dyn Fn(),
    verbose: bool,
    each: &mut dyn FnMut(&R),
    out: &mut dyn FnMut(&str),
) -> Option<R> {
    let mut last_line = String::new();
    let mut last_key = first;
    let mut last_print = Instant::now();
    // QA-039: followed by the push count, not the tail's length (which stops at its cap).
    let mut follower = LogFollower::new();
    while let Some((r, seq)) = poll() {
        if cancel.load(Ordering::Relaxed) {
            on_cancel();
        }
        let s = show(&r);
        let fresh = follower.new_lines(s.log, seq);
        if verbose {
            for l in fresh {
                out(&format!("    {l}"));
            }
        }
        each(&r);
        if s.running
            && s.line != last_line
            && (s.key != last_key || last_print.elapsed() >= QUIET_PRINT_EVERY)
        {
            out(&s.line);
            last_line = s.line;
            last_key = s.key;
            last_print = Instant::now();
        }
        if !s.busy {
            return Some(r);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::service::push_log;

    /// A followed record: its state, line, key and log tail with the push count.
    #[derive(Clone, Default)]
    struct Rec {
        busy: bool,
        running: bool,
        line: String,
        key: u32,
        log: Vec<String>,
    }

    fn show(r: &Rec) -> Shown<'_, u32> {
        Shown {
            busy: r.busy,
            running: r.running,
            line: r.line.clone(),
            key: r.key,
            log: &r.log,
        }
    }

    /// Follow the scripted polls `polls` (each a record and its push count); what was printed,
    /// how often the cancel was passed on, and the returned record's line.
    fn run(
        polls: Vec<(Rec, u64)>,
        cancel: bool,
        verbose: bool,
    ) -> (Vec<String>, usize, Option<String>) {
        let mut polls = polls.into_iter();
        let cancels = std::cell::Cell::new(0);
        let mut printed = Vec::new();
        let mut seen = 0;
        let end = follow(
            || polls.next(),
            show,
            0,
            &AtomicBool::new(cancel),
            &|| cancels.set(cancels.get() + 1),
            verbose,
            &mut |_| seen += 1,
            &mut |l| printed.push(l.to_string()),
        );
        assert!(seen > 0 || end.is_none());
        (printed, cancels.get(), end.map(|r| r.line))
    }

    /// A running record at `key` with line `line` and the tail `log` / `seq`.
    fn at(line: &str, key: u32, log: &[String], seq: u64) -> (Rec, u64) {
        let r = Rec {
            busy: true,
            running: true,
            line: line.into(),
            key,
            log: log.to_vec(),
        };
        (r, seq)
    }

    /// QA-249 / QA-039: a job printing 200 lines, read 10 per poll: every line once, in order,
    /// past the tail's cap of 60; nothing without `--verbose`.
    #[test]
    fn two_hundred_lines_are_each_printed_once_in_order() {
        let (mut log, mut seq) = (Vec::new(), 0u64);
        let mut polls = Vec::new();
        for chunk in 0..20 {
            for i in 0..10 {
                push_log(&mut log, &mut seq, &format!("line {}", chunk * 10 + i));
            }
            polls.push(at("10% training", 1, &log, seq));
        }
        let mut done = polls.last().unwrap().0.clone();
        done.busy = false;
        done.running = false;
        done.line = "done".into();
        polls.push((done, seq));
        let (printed, cancels, end) = run(polls.clone(), false, true);
        let lines: Vec<&String> = printed.iter().filter(|l| l.starts_with("    ")).collect();
        let want: Vec<String> = (0..200).map(|i| format!("    line {i}")).collect();
        assert_eq!(lines, want.iter().collect::<Vec<_>>());
        assert_eq!(cancels, 0);
        assert_eq!(end.as_deref(), Some("done"));
        // The progress line once (it never changed).
        assert_eq!(printed.iter().filter(|l| *l == "10% training").count(), 1);
        let (quiet, _, _) = run(polls, false, false);
        assert_eq!(quiet, ["10% training"]);
    }

    /// Lines pushed and evicted between two polls are skipped: the tail holds the last 60 of a
    /// burst of 100 (the full text is in the job's log file).
    #[test]
    fn a_burst_past_the_tail_shows_its_last_lines() {
        let (mut log, mut seq) = (Vec::new(), 0u64);
        for i in 0..100 {
            push_log(&mut log, &mut seq, &format!("line {i}"));
        }
        let (printed, _, end) = run(vec![at("x", 1, &log, seq)], false, true);
        let want: Vec<String> = (40..100).map(|i| format!("    line {i}")).collect();
        assert_eq!(printed[..60], want[..]);
        assert_eq!(printed[60], "x");
        assert_eq!(end, None, "the record vanished");
    }

    /// A progress line prints when it changed and its key did (else at most every 10 s), only
    /// while running; the cancel is passed on at every poll while set.
    #[test]
    fn progress_lines_print_on_a_key_change_and_only_while_running() {
        let mut paused = at("paused", 3, &[], 0);
        paused.0.running = false;
        let mut done = at("done", 4, &[], 0);
        done.0.busy = false;
        done.0.running = false;
        let polls = vec![
            at("1% a", 1, &[], 0),
            at("1% a", 1, &[], 0),
            // Changed text, same key: within 10 s, not printed.
            at("1% a: detail", 1, &[], 0),
            at("2% b", 2, &[], 0),
            paused,
            done,
        ];
        let (printed, cancels, end) = run(polls.clone(), false, false);
        assert_eq!(printed, ["1% a", "2% b"]);
        assert_eq!(cancels, 0);
        assert_eq!(end.as_deref(), Some("done"));
        let (_, cancels, _) = run(polls, true, false);
        assert_eq!(cancels, 6);
        // The key before any line: a first line at that key waits like any unchanged key.
        let (printed, _, _) = run(vec![at("0% starting", 0, &[], 0)], false, false);
        assert!(printed.is_empty(), "{printed:?}");
    }
}
