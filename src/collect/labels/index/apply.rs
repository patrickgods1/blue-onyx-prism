//! Putting what was read into the rows. The reads happen first, off the state lock; a drain or a
//! swap then applies them in one step under it.
//!
//! What makes a save visible to the next query, whatever a walk is doing: a query takes
//! `Index::drain` before it looks at what is pending, so it waits for a drain or a swap in
//! progress. A swap holds it from taking the writes its walk may have missed
//! ([`super::Log::since`]) to applying them in the same write lock as the new shards, so no reader
//! sees a shard without them: a walk reads a file before a write and swaps after the query that
//! read that write.

use super::row::{Cand, Encoded, Shard};
use super::walk::{self, DayStat, QueueStat};
use super::{CHANGED, HUB, Index, State, Writes};
use crate::collect::labels::Frame;
use crate::collect::labels::queue::{self, Candidate};
use crate::sync::{LockExt as _, RwLockExt as _};
use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// What a finished walk changes.
#[derive(Debug, Default)]
pub(super) struct Swap {
    /// Everything: nothing of the old rows stays.
    pub whole: bool,
    pub shards: Vec<Shard>,
    /// Days with no candidate left.
    pub emptied: Vec<String>,
    /// Day folders as read: None = no longer there.
    pub days: Vec<(String, Option<DayStat>)>,
    pub queue: Option<QueueStat>,
    pub foreign: Option<bool>,
    /// The last swap of its walk: see again all it may have missed, not just its shards' days.
    pub last: bool,
}

/// The review queue, read and interned.
#[derive(Debug)]
struct QueueUpdate {
    stat: QueueStat,
    cands: Vec<(String, Cand)>,
}

impl State {
    /// Put `fresh` in the rows: the days of frames with no row in `missing`.
    fn apply(&mut self, fresh: Vec<(String, Encoded)>, missing: &mut BTreeSet<String>) {
        for (rel, e) in fresh {
            let found = self.locate(&rel).map(|(s, at)| (s.day.clone(), at));
            match found.and_then(|(day, at)| Some((self.shards.get_mut(&day)?, at))) {
                Some((shard, at)) => shard.update(at, e),
                None => {
                    missing.insert(queue::day_of(&rel).to_string());
                }
            }
        }
    }

    /// Bring the rows in line with the review queue. What a row cannot say by itself goes stale:
    /// a line for a frame with no row, and a row that lost its line (its index line's camera and
    /// buckets are not in memory).
    fn apply_queue(&mut self, q: QueueUpdate, missing: &mut BTreeSet<String>) {
        let mut found = 0usize;
        for (rel, cand) in &q.cands {
            let at = self.locate(rel).map(|(s, at)| (s.day.clone(), at));
            match at.and_then(|(day, at)| self.shards.get_mut(&day)?.rows.get_mut(at)) {
                Some(row) => {
                    row.cand = *cand;
                    row.queued = true;
                    found += 1;
                }
                None => {
                    missing.insert(queue::day_of(rel).to_string());
                }
            }
        }
        let queued = self
            .shards
            .values()
            .flat_map(|s| &s.rows)
            .filter(|r| r.queued)
            .count();
        if queued != found {
            let lines: HashSet<&str> = q.cands.iter().map(|(rel, _)| rel.as_str()).collect();
            for shard in self.shards.values() {
                if shard
                    .rows
                    .iter()
                    .any(|r| r.queued && !lines.contains(shard.rel(r).as_str()))
                {
                    missing.insert(shard.day.clone());
                }
            }
        }
        self.queue = q.stat;
    }
}

impl Index {
    /// What `rels` hold now. `needs_side`: read the sidecar too, the row cannot keep its own.
    fn read_frames(
        &self,
        rels: HashSet<String>,
        needs_side: &dyn Fn(&str) -> bool,
    ) -> Vec<(String, Encoded)> {
        let encoder = self.encoder();
        rels.into_iter()
            .map(|rel| {
                let frame = Frame::resolve(&self.dir, &rel);
                let side = needs_side(&rel);
                let e = encoder.frame(frame, side);
                (rel, e)
            })
            .collect()
    }

    fn read_queue(&self) -> QueueUpdate {
        let read = walk::read_queue(&self.dir, self.now());
        let mut out: Vec<Candidate> = Vec::new();
        queue::review_candidates(read.lines, &mut BTreeSet::new(), &mut out);
        let encoder = self.encoder();
        QueueUpdate {
            stat: read.stat,
            cands: out
                .into_iter()
                .map(|c| {
                    let cand = encoder.cand(&c);
                    (c.rel, cand)
                })
                .collect(),
        }
    }

    /// Days whose rows cannot be brought up to date from what was read: the thread walks them.
    fn mark_stale(&self, days: BTreeSet<String>) {
        if days.is_empty() {
            return;
        }
        let mut hub = HUB.lock_recover();
        if let Some(c) = hub.of(self) {
            c.log.stale.extend(days);
            CHANGED.notify_all();
        }
    }

    /// The writes since the last query, into the rows.
    pub(super) fn drain(&self) {
        let _held = self.drain.lock_recover();
        let taken = {
            let mut hub = HUB.lock_recover();
            hub.of(self)
                .map(|c| std::mem::take(&mut c.log.pending))
                .unwrap_or_default()
        };
        if taken.rels.is_empty() && !taken.queue {
            return;
        }
        let fresh = self.read_frames(taken.rels, &|rel| {
            self.state
                .read_recover()
                .row(rel)
                .is_none_or(super::Row::needs_side)
        });
        let queue = taken.queue.then(|| self.read_queue());
        let mut missing = BTreeSet::new();
        {
            let mut state = self.state.write_recover();
            state.apply(fresh, &mut missing);
            if let Some(q) = queue {
                state.apply_queue(q, &mut missing);
            }
        }
        self.mark_stale(missing);
    }

    /// `rels` turned out to be gone.
    pub(super) fn mark_gone(&self, rels: &[String]) {
        let mut state = self.state.write_recover();
        let gone = |rel: &String| {
            (
                rel.clone(),
                Encoded {
                    gone: true,
                    ..Encoded::default()
                },
            )
        };
        state.apply(rels.iter().map(gone).collect(), &mut BTreeSet::new());
    }

    /// Start noting the writes a walk may miss. Before any of its reads.
    pub(super) fn begin_walk(&self) {
        if let Some(c) = HUB.lock_recover().of(self) {
            c.log.since = Some(Writes::default());
        }
    }

    /// The writes noted for `s`, which leave the log.
    fn take_since(&self, s: &Swap) -> Writes {
        let mut hub = HUB.lock_recover();
        let Some(since) = hub.of(self).and_then(|c| c.log.since.as_mut()) else {
            return Writes::default();
        };
        if s.last {
            return std::mem::take(since);
        }
        let days: HashSet<&str> = s
            .shards
            .iter()
            .map(|shard| shard.day.as_str())
            .chain(s.emptied.iter().map(String::as_str))
            .collect();
        let (mine, others) = std::mem::take(&mut since.rels)
            .into_iter()
            .partition(|rel: &String| days.contains(queue::day_of(rel)));
        since.rels = others;
        Writes {
            rels: mine,
            queue: false,
        }
    }

    /// Stop noting: a walk's last swap is done.
    pub(super) fn end_walk(&self) {
        if let Some(c) = HUB.lock_recover().of(self) {
            c.log.since = None;
        }
    }

    /// Put in what a walk made, and with it what it may have missed.
    pub(super) fn swap(self: &Arc<Self>, s: Swap) {
        {
            let _held = self.drain.lock_recover();
            let taken = self.take_since(&s);
            let fresh = self.read_frames(taken.rels, &|_| true);
            let queue = taken.queue.then(|| self.read_queue());
            let mut missing = BTreeSet::new();
            {
                let mut state = self.state.write_recover();
                if s.whole {
                    state.shards.clear();
                    state.days.clear();
                }
                for shard in s.shards {
                    state.shards.insert(shard.day.clone(), shard);
                }
                for day in s.emptied {
                    state.shards.remove(&day);
                }
                for (day, stat) in s.days {
                    match stat {
                        Some(stat) => state.days.insert(day, stat),
                        None => state.days.remove(&day),
                    };
                }
                if let Some(q) = s.queue {
                    state.queue = q;
                }
                if let Some(foreign) = s.foreign {
                    state.foreign = foreign;
                }
                state.apply(fresh, &mut missing);
                if let Some(q) = queue {
                    state.apply_queue(q, &mut missing);
                }
            }
            self.mark_stale(missing);
        }
        self.check_ceiling();
    }

    /// Over the ceiling, say so once and stop: the route walks from now on.
    pub(super) fn check_ceiling(self: &Arc<Self>) {
        let bytes = self.bytes();
        if bytes > self.timing.ceiling {
            self.over_ceiling(bytes);
        }
    }

    /// Drop the index for good: it needs `bytes`.
    pub(super) fn over_ceiling(&self, bytes: usize) {
        let mut hub = HUB.lock_recover();
        if hub.of(self).is_none() {
            return;
        }
        hub.remove(self);
        hub.refuse(&self.dir, None);
        tracing::warn!(
            "review index of {}: {} MB with {} of {} frames in, over its limit of {} MB, so it \
             was dropped. Next and every save in Review now read the whole folder, seconds on a \
             folder this size, until the service restarts or another folder is chosen",
            self.dir.display(),
            bytes / (1024 * 1024),
            self.done.load(Ordering::Acquire),
            self.total.load(Ordering::Acquire),
            self.timing.ceiling / (1024 * 1024),
        );
    }
}
