//! Work-stealing maps over scoped threads (`std::thread::scope`, nothing outlives the call),
//! results by index. A leaf: it knows nothing of what the items are. Two shapes:
//!
//! - [`map`]: long CPU-bound work with progress and a cancel flag (an import's analysis); the
//!   calling thread polls a counter and reports.
//! - [`map_all`]: a read walk over the dataset folder (`collect::labels`); the calling thread
//!   works too and nothing polls, so a short walk pays no wake-up.
//!
//! With one item or one thread both run on the calling thread: no spawn, no sleep. [`walk_threads`]
//! is how many threads a walk may take.

use std::num::NonZero;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Most threads of a read walk: the walks are syscalls, which stop scaling well before the core
/// count, and the machine is also serving detections.
pub const WALK_THREADS_MAX: usize = 4;

/// A walk over fewer frames than this stays on the calling thread: spawning costs about what
/// reading a few frames does.
pub const WALK_SERIAL_UNDER: usize = 32;

/// Threads of a read walk with `parallelism` hardware threads: all but one, at most
/// [`WALK_THREADS_MAX`], at least 1. Pure.
pub fn walk_threads_for(parallelism: usize) -> usize {
    parallelism.saturating_sub(1).clamp(1, WALK_THREADS_MAX)
}

/// [`walk_threads_for`] on this machine.
pub fn walk_threads() -> usize {
    walk_threads_for(std::thread::available_parallelism().map_or(1, NonZero::get))
}

/// Threads of a read walk over `frames` frames: [`walk_threads`], or 1 under
/// [`WALK_SERIAL_UNDER`].
pub fn walk_threads_over(frames: usize) -> usize {
    if frames < WALK_SERIAL_UNDER {
        1
    } else {
        walk_threads()
    }
}

/// A worker: takes the next index until none is left or `cancel` is set.
fn steal<T>(
    next: &AtomicUsize,
    done: &AtomicUsize,
    n: usize,
    f: &(dyn Fn(usize) -> T + Sync),
    cancel: &AtomicBool,
) -> Vec<(usize, T)> {
    let mut mine = Vec::new();
    while !cancel.load(Ordering::Relaxed) {
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= n {
            break;
        }
        mine.push((i, f(i)));
        done.fetch_add(1, Ordering::Relaxed);
    }
    mine
}

/// The workers' results by index, `None` when one is missing.
fn by_index<T>(n: usize, parts: Vec<Vec<(usize, T)>>) -> Option<Vec<T>> {
    let mut slots: Vec<Option<T>> = (0..n).map(|_| None).collect();
    for (i, v) in parts.into_iter().flatten() {
        if let Some(s) = slots.get_mut(i) {
            *s = Some(v);
        }
    }
    slots.into_iter().collect()
}

/// `f(i)` for `i` in `0..n` on the calling thread, with [`map`]'s contract.
fn serial<T>(
    n: usize,
    f: &(dyn Fn(usize) -> T + Sync),
    progress: &mut dyn FnMut(usize),
    poll: Duration,
    cancel: &AtomicBool,
) -> Option<Vec<T>> {
    let mut out = Vec::with_capacity(n);
    let mut reported = Instant::now();
    progress(0);
    for i in 0..n {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        out.push(f(i));
        if reported.elapsed() >= poll {
            progress(out.len());
            reported = Instant::now();
        }
    }
    progress(out.len());
    (out.len() == n).then_some(out)
}

/// `f(i)` for `i` in `0..n` on `threads` threads (each takes the next index), results by index.
/// `progress(done)` runs on the calling thread at the start, about every `poll` while the
/// workers run, and once at the end. `None` when `cancel` was set before every item was done.
/// One item or one thread runs on the calling thread, without a sleep.
pub fn map<T: Send>(
    n: usize,
    threads: usize,
    f: &(dyn Fn(usize) -> T + Sync),
    progress: &mut dyn FnMut(usize),
    poll: Duration,
    cancel: &AtomicBool,
) -> Option<Vec<T>> {
    if n <= 1 || threads <= 1 {
        return serial(n, f, progress, poll, cancel);
    }
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, T)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads.min(n))
            .map(|_| {
                let (next, done) = (&next, &done);
                s.spawn(move || steal(next, done, n, f, cancel))
            })
            .collect();
        // Wake often (the end is noticed within 10 ms), report every `poll`.
        let mut reported = Instant::now();
        progress(done.load(Ordering::Relaxed));
        while handles.iter().any(|h| !h.is_finished()) {
            std::thread::sleep(poll.min(Duration::from_millis(10)));
            if reported.elapsed() >= poll {
                progress(done.load(Ordering::Relaxed));
                reported = Instant::now();
            }
        }
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
            .collect()
    });
    let finished = done.load(Ordering::Relaxed);
    progress(finished);
    if finished < n {
        return None;
    }
    by_index(n, parts)
}

/// `f(i)` for `i` in `0..n` on `threads` threads, the calling thread among them, results by
/// index: no progress, no cancel, no polling. A panic in `f` comes out here.
pub fn map_all<T: Send>(n: usize, threads: usize, f: &(dyn Fn(usize) -> T + Sync)) -> Vec<T> {
    if n <= 1 || threads <= 1 {
        return (0..n).map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let cancel = AtomicBool::new(false);
    let parts: Vec<Vec<(usize, T)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (1..threads.min(n))
            .map(|_| s.spawn(|| steal(&next, &done, n, f, &cancel)))
            .collect();
        let mine = steal(&next, &done, n, f, &cancel);
        let mut parts: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
            .collect();
        parts.push(mine);
        parts
    });
    // Every index was taken: nothing cancels, and a panic did not get here.
    by_index(n, parts).unwrap_or_default()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn map_keeps_the_order_and_stops_on_cancel() {
        let cancel = AtomicBool::new(false);
        let mut seen = Vec::new();
        let out = map(
            500,
            4,
            &|i| i * i,
            &mut |d| seen.push(d),
            Duration::from_millis(1),
            &cancel,
        )
        .unwrap();
        assert_eq!(out, (0..500).map(|i| i * i).collect::<Vec<_>>());
        assert_eq!(*seen.last().unwrap(), 500);
        assert!(
            map(0, 4, &|i| i, &mut |_| {}, Duration::from_millis(1), &cancel)
                .unwrap()
                .is_empty()
        );
        let r = map(
            10_000,
            3,
            &|i| {
                if i == 50 {
                    cancel.store(true, Ordering::SeqCst);
                }
                std::thread::sleep(Duration::from_micros(50));
                i
            },
            &mut |_| {},
            Duration::from_millis(1),
            &cancel,
        );
        assert!(r.is_none(), "cancelled");
    }

    #[test]
    fn one_item_or_one_thread_runs_here_with_no_sleep_and_the_same_contract() {
        let me = std::thread::current().id();
        let cancel = AtomicBool::new(false);
        let here = |i: usize| {
            assert_eq!(std::thread::current().id(), me);
            i + 1
        };
        // A poll this long would make any sleep obvious.
        let poll = Duration::from_secs(30);
        for (n, threads) in [(0, 4), (1, 4), (40, 1), (40, 0)] {
            let start = Instant::now();
            let mut seen = Vec::new();
            let out = map(n, threads, &here, &mut |d| seen.push(d), poll, &cancel).unwrap();
            assert_eq!(out, (1..=n).collect::<Vec<_>>());
            assert_eq!((seen[0], *seen.last().unwrap()), (0, n), "start and end");
            assert!(start.elapsed() < Duration::from_secs(5));
            assert_eq!(map_all(n, threads, &here), (1..=n).collect::<Vec<_>>());
        }
        // Progress between items, and a cancel still answers None.
        let mut seen = Vec::new();
        let out = map(
            3,
            1,
            &here,
            &mut |d| seen.push(d),
            Duration::ZERO,
            &AtomicBool::new(false),
        );
        assert_eq!(out, Some(vec![1, 2, 3]));
        assert_eq!(seen, [0, 1, 2, 3, 3]);
        let stop = AtomicBool::new(false);
        let r = map(
            5,
            1,
            &|i| {
                if i == 1 {
                    stop.store(true, Ordering::SeqCst);
                }
                i
            },
            &mut |_| {},
            poll,
            &stop,
        );
        assert!(r.is_none(), "cancelled");
    }

    #[test]
    fn map_all_returns_by_index_on_several_threads() {
        let out = map_all(300, 4, &|i| {
            // Earlier items finish later.
            std::thread::sleep(Duration::from_micros(300 - i as u64));
            i * 2
        });
        assert_eq!(out, (0..300).map(|i| i * 2).collect::<Vec<_>>());
    }

    #[test]
    fn a_panic_in_map_all_reaches_the_caller() {
        let r = std::panic::catch_unwind(|| {
            map_all(50, 4, &|i| {
                assert_ne!(i, 7, "boom");
                i
            })
        });
        assert!(r.is_err());
    }

    #[test]
    fn walk_thread_counts() {
        assert_eq!(walk_threads_for(0), 1);
        assert_eq!(walk_threads_for(1), 1);
        assert_eq!(walk_threads_for(2), 1);
        assert_eq!(walk_threads_for(4), 3);
        assert_eq!(walk_threads_for(5), 4);
        assert_eq!(walk_threads_for(64), 4);
        assert_eq!(walk_threads_over(WALK_SERIAL_UNDER - 1), 1);
        assert_eq!(walk_threads_over(WALK_SERIAL_UNDER), walk_threads());
    }
}
