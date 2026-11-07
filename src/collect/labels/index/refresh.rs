//! The index's thread: the first build, then a look at the folder every minute, a walk of
//! everything every 30 minutes, and the end of it all when nobody has asked for a while.
//!
//! A build publishes each day as it finishes, newest day first, so a query sees rows so far. A
//! later walk works beside the rows, which keep serving, and swaps at the end. Every walk goes
//! `begin_walk`, stats, reads, [`Index::swap`], `end_walk`: what lands after the stats shows at the
//! next scan, and what lands after `begin_walk` is seen again at the swap.
//!
//! Scans re-read a day when its folder's time, or its index file's length or time, moved. A label
//! file replaced by a rename moves the folder's time; one edited in place does not, and neither
//! does anything on a file system with unreliable folder times: the walk of everything is for that.
//!
//! [`crate::backend::priority`] has no way to lower one thread, only a process (and that means
//! `unsafe`, kept to `backend`), so the walks run at normal priority, on at most
//! [`crate::parallel::WALK_THREADS_MAX`] threads.

use super::apply::Swap;
use super::row::Shard;
use super::walk::{self, DayRead, DayStat};
use super::{CHANGED, HUB, Index, queue};
use crate::collect::labels::queue::Candidate;
use crate::collect::layout::Walker;
use crate::parallel;
use crate::sync::{LockExt as _, RwLockExt as _};
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// What a pass does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Pass {
    /// Walk what changed.
    Scan,
    /// Walk everything.
    Full,
}

/// Why the thread woke.
enum Wake {
    Pass,
    Stop,
    Idle,
}

/// Ends the index when the thread does, panic included, so `/next` walks rather than waits.
struct ThreadDone(Arc<Index>);

impl Drop for ThreadDone {
    fn drop(&mut self) {
        let index = &self.0;
        index.building.store(false, Ordering::Release);
        index.catching_up.store(false, Ordering::Release);
        let mut hub = HUB.lock_recover();
        let panicked = std::thread::panicking();
        if hub.of(index).is_some() {
            hub.remove(index);
            if panicked {
                tracing::warn!(
                    "review index of {} stopped on an internal error, so Next walks the folder \
                     for now (seconds on a big folder); the log above says what failed",
                    index.dir.display()
                );
                hub.refuse(&index.dir, Some(Instant::now() + super::AFTER_PANIC));
            }
        }
        index.cancelled.store(true, Ordering::Release);
        CHANGED.notify_all();
    }
}

/// The thread.
pub(super) fn run(index: Arc<Index>) {
    let done = ThreadDone(index);
    let index = &done.0;
    index.first_build();
    let mut last_pass = Instant::now();
    let mut last_full = Instant::now();
    loop {
        match index.sleep(last_pass) {
            Wake::Stop => return,
            Wake::Idle => {
                tracing::debug!("review index of {}: idle, dropped", index.dir.display());
                HUB.lock_recover().remove(index);
                return;
            }
            Wake::Pass => {}
        }
        if last_full.elapsed() >= index.timing.full {
            index.pass(Pass::Full, &|| {});
            last_full = Instant::now();
        } else {
            index.pass(Pass::Scan, &|| {});
        }
        last_pass = Instant::now();
    }
}

/// `candidates` by day folder of their path, newest day first.
fn days_of(candidates: Vec<Candidate>) -> Vec<(String, Vec<Candidate>)> {
    let mut days: Vec<(String, Vec<Candidate>)> = queue::by_day(candidates)
        .into_iter()
        .filter_map(|day| Some((queue::day_of(&day.first()?.rel).to_string(), day)))
        .collect();
    let date = |key: &str| key.rsplit('/').next().unwrap_or_default().to_string();
    days.sort_by_cached_key(|(key, _)| std::cmp::Reverse((date(key), key.clone())));
    days
}

impl Index {
    /// Wait for the next pass.
    fn sleep(&self, last_pass: Instant) -> Wake {
        let t = self.timing;
        let mut hub = HUB.lock_recover();
        loop {
            if self.cancelled() {
                return Wake::Stop;
            }
            let idle_left = t.idle.saturating_sub(self.idle_for());
            if idle_left.is_zero() {
                return Wake::Idle;
            }
            let since = last_pass.elapsed();
            let left = if hub.stale(self) {
                t.pause.saturating_sub(since)
            } else {
                t.refresh.saturating_sub(since)
            };
            if left.is_zero() {
                return Wake::Pass;
            }
            hub = CHANGED
                .wait_timeout(hub, left.min(idle_left))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// The rows of one day folder. Stops early, leaving the shard part-done, if the text alone
    /// goes over the ceiling; the index is dropped then.
    fn walk_shard(self: &Arc<Self>, mut walker: Walker<'_>, day: String, c: &[Candidate]) -> Shard {
        let encoder = self.encoder();
        let mut order: Vec<(String, bool, &Candidate)> = c
            .iter()
            .filter_map(|c| {
                let (stem, json) = super::row::stem_of(&c.rel)?;
                Some((stem, json, c))
            })
            .collect();
        order.sort_unstable_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        let mut shard = Shard::new(day);
        for (stem, json, c) in order {
            #[cfg(test)]
            super::tests::walk_point(&self.dir);
            shard.push(
                &stem,
                json,
                c.queued,
                encoder.cand(c),
                encoder.walked(&mut walker, c),
            );
            if shard.used() > self.timing.ceiling {
                self.over_ceiling(shard.used());
                break;
            }
        }
        shard.shrink();
        shard
    }

    /// The first build: rows go in as each day is finished.
    pub(super) fn first_build(self: &Arc<Self>) {
        let _walk = self.walk.lock_recover();
        self.begin_walk();
        let now = self.now();
        let queue = walk::read_queue(&self.dir, now);
        let days = walk::read_all_days(&self.dir, now);
        let (candidates, stats) = walk::candidates_of(queue.lines, days);
        self.total.store(candidates.len() as u64, Ordering::Release);
        self.swap(Swap {
            foreign: Some(stats.iter().any(|s| s.foreign)),
            days: stats.into_iter().map(|s| (s.key, Some(s.stat))).collect(),
            queue: Some(queue.stat),
            ..Swap::default()
        });
        let threads = parallel::walk_threads_over(candidates.len());
        let days = days_of(candidates);
        let walker = Walker::new(&self.dir);
        parallel::map_all(days.len(), threads, &|i| {
            let Some((key, day)) = days.get(i).filter(|_| !self.cancelled()) else {
                return;
            };
            let shard = self.walk_shard(walker.clone(), key.clone(), day);
            self.swap(Swap {
                shards: vec![shard],
                ..Swap::default()
            });
            self.done.fetch_add(day.len() as u64, Ordering::AcqRel);
            CHANGED.notify_all();
        });
        self.swap(Swap {
            last: true,
            ..Swap::default()
        });
        self.building.store(false, Ordering::Release);
        self.end_walk();
        CHANGED.notify_all();
    }

    /// One pass. `before_swap` runs between the walk and its swap.
    pub(super) fn pass(self: &Arc<Self>, kind: Pass, before_swap: &dyn Fn()) {
        let _walk = self.walk.lock_recover();
        if !self.dir.is_dir() {
            return;
        }
        match kind {
            Pass::Full => self.pass_full(before_swap),
            Pass::Scan => self.pass_scan(before_swap),
        }
        self.catching_up.store(false, Ordering::Release);
        CHANGED.notify_all();
    }

    /// Take the stale days, saying they are being walked.
    fn take_stale(&self) -> BTreeSet<String> {
        let mut hub = HUB.lock_recover();
        let Some(c) = hub.of(self) else {
            return BTreeSet::new();
        };
        let days: BTreeSet<String> = c.log.stale.drain().collect();
        if !days.is_empty() {
            self.catching_up.store(true, Ordering::Release);
        }
        days
    }

    /// The progress of a pass over stale days: `walking` frames of the rows are out of date.
    fn show_progress(&self, walking: u64) {
        let rows = self.state.read_recover().rows().max(walking);
        self.total.store(rows, Ordering::Release);
        self.done.store(rows - walking, Ordering::Release);
    }

    fn pass_full(self: &Arc<Self>, before_swap: &dyn Fn()) {
        self.take_stale();
        self.begin_walk();
        let now = self.now();
        let queue = walk::read_queue(&self.dir, now);
        let days = walk::read_all_days(&self.dir, now);
        let (candidates, stats) = walk::candidates_of(queue.lines, days);
        let threads = parallel::walk_threads_over(candidates.len());
        let days = days_of(candidates);
        let walker = Walker::new(&self.dir);
        let shards = parallel::map_all(days.len(), threads, &|i| {
            let (key, day) = days.get(i).filter(|_| !self.cancelled())?;
            Some(self.walk_shard(walker.clone(), key.clone(), day))
        });
        before_swap();
        self.swap(Swap {
            whole: true,
            shards: shards.into_iter().flatten().collect(),
            foreign: Some(stats.iter().any(|s| s.foreign)),
            days: stats.into_iter().map(|s| (s.key, Some(s.stat))).collect(),
            queue: Some(queue.stat),
            last: true,
            ..Swap::default()
        });
        self.end_walk();
    }

    /// The days whose folder looks different from when they were walked, and those gone.
    fn changed_days(&self, on_disk: &[(String, DayStat)]) -> BTreeSet<String> {
        let state = self.state.read_recover();
        let mut changed: BTreeSet<String> = on_disk
            .iter()
            .filter(|(key, now)| state.days.get(key).is_none_or(|old| now.changed_from(old)))
            .map(|(key, _)| key.clone())
            .collect();
        let there: BTreeSet<&str> = on_disk.iter().map(|(key, _)| key.as_str()).collect();
        changed.extend(
            state
                .days
                .keys()
                .filter(|key| !there.contains(key.as_str()))
                .cloned(),
        );
        changed
    }

    fn pass_scan(self: &Arc<Self>, before_swap: &dyn Fn()) {
        let now = self.now();
        let queue_now = walk::QueueStat::of(&self.dir, now);
        let on_disk = walk::stat_days(&self.dir, now);
        let queue_old = self.state.read_recover().queue;
        if queue_now.changed_from(&queue_old) {
            if let Some(c) = HUB.lock_recover().of(self) {
                c.log.pending.queue = true;
            }
            self.drain();
        }
        let mut walking = self.changed_days(&on_disk);
        walking.extend(self.take_stale());
        if walking.is_empty() {
            return;
        }
        if self.state.read_recover().foreign {
            return self.pass_full(before_swap);
        }
        self.begin_walk();
        let queue = walk::read_queue(&self.dir, now);
        let reads: Vec<DayRead> = walking
            .iter()
            .filter_map(|key| walk::read_day(&self.dir, key, now))
            .collect();
        if reads.iter().any(DayRead::foreign) {
            self.end_walk();
            return self.pass_full(before_swap);
        }
        let (candidates, stats) = walk::candidates_of(queue.lines, reads);
        let mut swap = Swap {
            days: walking
                .iter()
                .map(|key| {
                    let stat = stats.iter().find(|s| s.key == *key).map(|s| s.stat);
                    (key.clone(), stat)
                })
                .collect(),
            queue: None,
            last: true,
            ..Swap::default()
        };
        let days: Vec<(String, Vec<Candidate>)> = days_of(candidates)
            .into_iter()
            .filter(|(key, _)| walking.contains(key))
            .collect();
        swap.emptied = walking
            .iter()
            .filter(|key| !days.iter().any(|(day, _)| day == *key))
            .cloned()
            .collect();
        self.show_progress(days.iter().map(|(_, d)| d.len() as u64).sum());
        let threads = parallel::walk_threads_over(days.iter().map(|(_, d)| d.len()).sum());
        let walker = Walker::new(&self.dir);
        let shards = parallel::map_all(days.len(), threads, &|i| {
            let (key, day) = days.get(i).filter(|_| !self.cancelled())?;
            let shard = self.walk_shard(walker.clone(), key.clone(), day);
            self.done.fetch_add(day.len() as u64, Ordering::AcqRel);
            Some(shard)
        });
        swap.shards = shards.into_iter().flatten().collect();
        before_swap();
        self.swap(swap);
        self.end_walk();
    }
}
