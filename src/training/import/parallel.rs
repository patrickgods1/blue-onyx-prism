//! The worker pools of an import, on scoped threads (`std::thread::scope`, nothing outlives the
//! call):
//!
//! - [`run_with_lines`]: the extraction, one helper process per video, `workers` at a time.
//!   The task's progress callback is not `Send`, so the workers send owned messages
//!   ([`Event`]: started, an output line, finished) over a channel and only the calling (job)
//!   thread reports them. Results come back by index, so the caller reassembles them in video
//!   order (the dedup that follows depends on the order).
//! - [`map`] ([`crate::parallel::map`]): the analysis (decode, dHash, scene of every candidate),
//!   CPU-bound, work-stealing by index over `threads` threads; the calling thread polls a counter
//!   for the progress.
//!
//! Both stop starting work once `cancel` is set (a running helper is killed by the command
//! runner, which watches the same flag); items never started come back as `None`.

pub use crate::parallel::map;

use anyhow::Result;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Most videos extracted at once by default.
pub const AUTO_EXTRACT_MAX: usize = 4;

/// Videos extracted at the same time: `requested`, or with 0 half the physical cores (each
/// helper decodes with FFmpeg's own threads), 1..=[`AUTO_EXTRACT_MAX`]; never more than
/// `videos`, at least 1. Pure.
pub fn extract_workers_for(requested: usize, physical: usize, videos: usize) -> usize {
    let w = if requested == 0 {
        (physical / 2).clamp(1, AUTO_EXTRACT_MAX)
    } else {
        requested
    };
    w.min(videos).max(1)
}

/// [`extract_workers_for`] on this machine.
pub fn extract_workers(requested: usize, videos: usize) -> usize {
    extract_workers_for(
        requested,
        crate::backend::cpu_threads::physical_cores(),
        videos,
    )
}

/// Threads of the analysis: all physical cores but one (at least 1), never more than one per
/// 8 items. Pure.
pub fn analyse_threads_for(physical: usize, items: usize) -> usize {
    physical
        .saturating_sub(1)
        .max(1)
        .min(items.div_ceil(8))
        .max(1)
}

/// [`analyse_threads_for`] on this machine.
pub fn analyse_threads(items: usize) -> usize {
    analyse_threads_for(crate::backend::cpu_threads::physical_cores(), items)
}

/// What a worker of [`run_with_lines`] tells the calling thread.
#[derive(Debug)]
pub enum Event<'a> {
    /// Item `i` started.
    Started(usize),
    /// A line of item `i`'s output.
    Line(usize, &'a str),
    /// Item `i` finished (its result is in the returned list).
    Finished(usize),
}

/// The work of [`run_with_lines`]: item `i`, a sink for its output lines.
pub type LineWork<'a, T> = dyn Fn(usize, &mut dyn FnMut(&str)) -> Result<T> + Sync + 'a;

enum Msg<T> {
    Started(usize),
    Line(usize, String),
    Finished(usize, Result<T>),
}

/// Run `work(i, on_line)` for `i` in `0..n` on `workers` threads, in index order of starting.
/// `on_event` runs on the calling thread only, in the order the workers sent the events.
/// Returns each item's result by index (`None`: not started, `cancel` was set).
pub fn run_with_lines<T: Send>(
    n: usize,
    workers: usize,
    work: &LineWork<'_, T>,
    on_event: &mut dyn FnMut(Event<'_>),
    cancel: &AtomicBool,
) -> Vec<Option<Result<T>>> {
    let mut out: Vec<Option<Result<T>>> = (0..n).map(|_| None).collect();
    if n == 0 {
        return out;
    }
    let next = AtomicUsize::new(0);
    let (tx, rx) = crossbeam_channel::unbounded::<Msg<T>>();
    std::thread::scope(|s| {
        for _ in 0..workers.clamp(1, n) {
            let tx = tx.clone();
            let next = &next;
            s.spawn(move || {
                while !cancel.load(Ordering::Relaxed) {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let _ = tx.send(Msg::Started(i));
                    let r = work(i, &mut |l| {
                        let _ = tx.send(Msg::Line(i, l.to_string()));
                    });
                    let _ = tx.send(Msg::Finished(i, r));
                }
            });
        }
        drop(tx);
        // Ends when every worker has dropped its sender.
        for m in &rx {
            match m {
                Msg::Started(i) => on_event(Event::Started(i)),
                Msg::Line(i, l) => on_event(Event::Line(i, &l)),
                Msg::Finished(i, r) => {
                    if let Some(slot) = out.get_mut(i) {
                        *slot = Some(r);
                    }
                    on_event(Event::Finished(i));
                }
            }
        }
    });
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;

    #[test]
    fn worker_counts() {
        assert_eq!(extract_workers_for(0, 8, 12), 4);
        assert_eq!(extract_workers_for(0, 6, 12), 3);
        assert_eq!(extract_workers_for(0, 1, 12), 1);
        assert_eq!(
            extract_workers_for(0, 16, 2),
            2,
            "never more than the videos"
        );
        assert_eq!(extract_workers_for(6, 4, 10), 6, "an explicit count wins");
        assert_eq!(extract_workers_for(3, 8, 0), 1);
        assert_eq!(analyse_threads_for(8, 2000), 7);
        assert_eq!(analyse_threads_for(1, 2000), 1);
        assert_eq!(analyse_threads_for(8, 10), 2);
        assert_eq!(analyse_threads_for(8, 0), 1);
    }

    /// Peak number of `work` calls running at once.
    struct Peak {
        now: AtomicUsize,
        peak: AtomicUsize,
    }

    impl Peak {
        fn enter(&self) {
            let n = self.now.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(n, Ordering::SeqCst);
        }
        fn leave(&self) {
            self.now.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn lines_go_through_the_calling_thread_and_results_come_back_by_index() {
        let peak = Peak {
            now: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        };
        let me = std::thread::current().id();
        let cancel = AtomicBool::new(false);
        let mut lines: Vec<(usize, String)> = Vec::new();
        let mut finished = Vec::new();
        let work = |i: usize, line: &mut dyn FnMut(&str)| -> Result<usize> {
            peak.enter();
            // Later items finish first.
            std::thread::sleep(Duration::from_millis(5 * (10 - i as u64)));
            line(&format!("item {i}"));
            peak.leave();
            if i == 3 {
                anyhow::bail!("item 3 failed");
            }
            Ok(i * 10)
        };
        let out = run_with_lines(
            10,
            3,
            &work,
            &mut |e| {
                assert_eq!(std::thread::current().id(), me);
                match e {
                    Event::Line(i, l) => lines.push((i, l.to_string())),
                    Event::Finished(i) => finished.push(i),
                    Event::Started(_) => {}
                }
            },
            &cancel,
        );
        let p = peak.peak.load(Ordering::SeqCst);
        assert!(p > 1 && p <= 3, "peak {p}");
        assert_eq!(out.len(), 10);
        for (i, r) in out.iter().enumerate() {
            match r.as_ref().unwrap() {
                Ok(v) => assert_eq!(*v, i * 10),
                Err(e) => assert!(i == 3 && e.to_string().contains("item 3")),
            }
        }
        assert_eq!(lines.len(), 10);
        assert!(lines.iter().all(|(i, l)| *l == format!("item {i}")));
        assert_ne!(finished, (0..10).collect::<Vec<_>>(), "out of order");
    }

    #[test]
    fn a_cancel_stops_every_worker() {
        let cancel = AtomicBool::new(false);
        let started = Mutex::new(Vec::new());
        let work = |i: usize, _: &mut dyn FnMut(&str)| -> Result<()> {
            started.lock().unwrap().push(i);
            if i == 1 {
                cancel.store(true, Ordering::SeqCst);
            }
            while !cancel.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
            anyhow::bail!("cancelled")
        };
        let out = run_with_lines(50, 2, &work, &mut |_| {}, &cancel);
        let n = started.lock().unwrap().len();
        assert!(n <= 3, "{n} started");
        assert_eq!(out.iter().filter(|r| r.is_some()).count(), n);
        assert!(out[49].is_none());
    }
}
