//! The answer from the rows: one pass over them, no sort. `next_after` needs a place in the sorted
//! queue and the few frames after it, which counting the rows before `after` and keeping the
//! smallest few gives in one pass; a sort would build the order of every other frame to throw it
//! away. The order is the one of [`super::super::queue::sort_items`]: [`Uncertainty::key`], then
//! the newest, then the path. It has no ties, paths being unique, so what the pass finds is what
//! the sort would put there.
//!
//! The steps of the pass are those of `queue::build`, in its order. Only the found frame and the
//! ones after it become strings.

use super::super::queue::{AutoFilter, Counts, Filters, Item, QueueSource, SortKey, StatusFilter};
use super::State;
use super::answer::{Answer, UPCOMING};
use super::row::{Name, Names, Row, Shard, auto_bit, stem_of};
use std::cmp::Ordering;
use std::collections::BTreeSet;

/// A name filter, over ids.
enum Want {
    Any,
    Is(Name),
    /// A name no row has.
    Never,
}

impl Want {
    fn of(name: Option<&str>, names: &Names) -> Self {
        match name {
            None => Self::Any,
            Some(n) => names.find_name(n).map_or(Self::Never, Self::Is),
        }
    }

    fn keeps(&self, id: Name) -> bool {
        match self {
            Self::Any => true,
            Self::Is(n) => *n == id,
            Self::Never => false,
        }
    }
}

/// [`Filters`] over ids. Everything interned that a row of the state can name is known: rows are
/// published after their names, and this is made under the state's read lock.
struct Compiled {
    queue_only: bool,
    scene: Want,
    camera: Want,
    /// By bucket list.
    bucket: Vec<bool>,
    /// By class set.
    class: Vec<bool>,
    status: StatusFilter,
    auto: Option<AutoFilter>,
}

impl Compiled {
    fn new(f: &Filters, names: &Names) -> Self {
        let wanted: BTreeSet<Name> = f.label.iter().filter_map(|l| names.find_name(l)).collect();
        Self {
            queue_only: f.source == QueueSource::Queue,
            scene: Want::of(f.scene.as_deref(), names),
            camera: Want::of(f.camera.as_deref(), names),
            bucket: names
                .bucket_lists()
                .map(|list| f.bucket.is_none_or(|b| list.contains(&b)))
                .collect(),
            class: names
                .class_sets()
                .map(|set| f.label.is_empty() || set.iter().any(|c| wanted.contains(c)))
                .collect(),
            status: f.status,
            auto: f.auto,
        }
    }

    /// In the candidates of the query's source.
    fn in_source(&self, r: &Row) -> bool {
        !self.queue_only || r.queued
    }

    /// Passes the scene, camera and bucket filters.
    fn by_line(&self, r: &Row) -> bool {
        self.scene.keeps(r.cand.scene)
            && self.camera.keeps(r.cand.camera)
            && self
                .bucket
                .get(r.cand.buckets as usize)
                .copied()
                .unwrap_or(false)
    }

    fn by_class(&self, r: &Row) -> bool {
        self.class
            .get(r.classes() as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Passes the status and auto filters.
    fn by_state(&self, r: &Row) -> bool {
        if let StatusFilter::Is(s) = self.status
            && s != r.part.status
        {
            return false;
        }
        self.auto.is_none_or(|a| r.part.auto & auto_bit(a) != 0)
    }

    /// Whether the queue lists `r`.
    fn lists(&self, r: &Row) -> bool {
        self.in_source(r)
            && self.by_line(r)
            && !r.gone
            && r.readable()
            && self.by_class(r)
            && self.by_state(r)
    }

    /// What sorts `r`. No note text: the sort only asks whether there is one.
    fn key(&self, r: &Row) -> SortKey {
        r.uncertainty(self.auto, (r.part.note != 0).then(String::new))
            .key()
    }
}

/// A listed frame, on its way through the pass.
#[derive(Clone, Copy)]
struct Hit<'a> {
    key: SortKey,
    shard: &'a Shard,
    row: &'a Row,
}

impl Hit<'_> {
    /// [`super::super::queue::sort_items`]' order.
    fn order(&self, other: &Self) -> Ordering {
        self.key
            .cmp(&other.key)
            .then_with(|| {
                other
                    .shard
                    .timestamp(other.row)
                    .cmp(self.shard.timestamp(self.row))
            })
            .then_with(|| {
                self.shard
                    .rel_bytes(self.row)
                    .cmp(other.shard.rel_bytes(other.row))
            })
    }
}

/// The first frames in the order among those offered.
struct Top<'a>(Vec<Hit<'a>>);

impl<'a> Top<'a> {
    const SIZE: usize = UPCOMING + 1;

    fn offer(&mut self, hit: Hit<'a>) {
        if self.0.len() == Self::SIZE
            && self
                .0
                .last()
                .is_some_and(|worst| hit.order(worst) != Ordering::Less)
        {
            return;
        }
        let at = self.0.partition_point(|h| h.order(&hit) == Ordering::Less);
        self.0.insert(at, hit);
        self.0.truncate(Self::SIZE);
    }
}

/// The classes, cameras and counts of a pass, over ids until the end.
#[derive(Default)]
struct Tally {
    counts: Counts,
    /// By class set: a frame the filters above the class filter keep has it.
    class_sets: Vec<bool>,
    /// By name: a candidate has this camera.
    cameras: Vec<bool>,
    /// By training set: reviewed frames that have it.
    training: Vec<u64>,
    unreadable: u64,
}

impl Tally {
    fn new(names: &Names) -> Self {
        Self {
            class_sets: vec![false; names.class_sets().count()],
            cameras: vec![false; names.name_count()],
            training: vec![0; names.training_sets()],
            ..Self::default()
        }
    }

    /// [`Counts`]' `add` of a frame the filters keep.
    fn count(&mut self, r: &Row) {
        use super::super::Status;
        let c = &mut self.counts;
        c.total += 1;
        let has = |a: AutoFilter| u64::from(r.part.auto & auto_bit(a) != 0);
        c.auto_review += has(AutoFilter::Review);
        c.auto_accept += has(AutoFilter::Accept);
        c.repeats_only += has(AutoFilter::Repeats);
        match r.part.status {
            Status::Reviewed => {
                c.reviewed += 1;
                c.empty_confirmed += u64::from(r.part.empty_confirmed);
                c.rejected_boxes += u64::from(r.part.negatives);
                c.audit_accepted += u64::from(r.part.audit_accepted);
                c.audit_rejected += u64::from(r.part.audit_rejected);
                if let Some(n) = self.training.get_mut(r.part.training as usize) {
                    *n += 1;
                }
            }
            Status::Skipped => c.skipped += 1,
            Status::Unreviewed => c.remaining += 1,
        }
    }

    fn mark(list: &mut [bool], id: u32) {
        if let Some(seen) = list.get_mut(id as usize) {
            *seen = true;
        }
    }

    /// The per-class counts, classes and cameras as names.
    fn finish(mut self, names: &Names) -> (Counts, Vec<String>, Vec<String>) {
        for (id, frames) in self.training.iter().enumerate() {
            if *frames == 0 {
                continue;
            }
            for (class, boxes) in names.training(u32::try_from(id).unwrap_or(u32::MAX)) {
                *self
                    .counts
                    .per_class
                    .entry(names.name(*class).to_string())
                    .or_default() += frames * u64::from(*boxes);
            }
        }
        let on = |list: &[bool]| -> Vec<u32> {
            list.iter()
                .enumerate()
                .filter(|(_, seen)| **seen)
                .filter_map(|(id, _)| u32::try_from(id).ok())
                .collect()
        };
        let labels: BTreeSet<&str> = on(&self.class_sets)
            .into_iter()
            .flat_map(|set| names.class_set(set).iter())
            .map(|c| names.name(*c))
            .collect();
        let cameras: BTreeSet<&str> = on(&self.cameras)
            .into_iter()
            .map(|c| names.name(c))
            .collect();
        (
            self.counts,
            cameras.into_iter().map(String::from).collect(),
            labels.into_iter().map(String::from).collect(),
        )
    }
}

/// The row `after` names, when it is listed.
fn pivot<'a>(state: &'a State, q: &Compiled, after: Option<&str>) -> Option<Hit<'a>> {
    let rel = after?;
    let (stem, json) = stem_of(rel)?;
    let (day, _) = rel.rsplit_once('/')?;
    let shard = state.shards.get(day)?;
    let row = shard.rows.get(shard.find(&stem, json)?)?;
    q.lists(row).then(|| Hit {
        key: q.key(row),
        shard,
        row,
    })
}

fn item(hit: &Hit<'_>, auto: Option<AutoFilter>, names: &Names) -> Item {
    let r = hit.row;
    Item {
        frame: hit.shard.rel(r),
        timestamp: hit.shard.timestamp(r).to_string(),
        scene: names.name(r.cand.scene).to_string(),
        camera: names.name(r.cand.camera).to_string(),
        model: names.name(r.side.model).to_string(),
        buckets: names.buckets(r.cand.buckets).to_vec(),
        status: r.part.status,
        queued: r.queued,
        uncertainty: r.uncertainty(auto, hit.shard.note(r).map(str::to_string)),
    }
}

/// The answer of `state` for `f` and `after`, with the frames it shows: [`Answer::indexing`] is the
/// caller's.
pub(super) fn answer(
    state: &State,
    names: &Names,
    f: &Filters,
    after: Option<&str>,
) -> (Answer, Vec<String>) {
    let q = Compiled::new(f, names);
    let pivot = pivot(state, &q, after);
    let mut tally = Tally::new(names);
    let (mut first, mut next) = (Top(Vec::new()), Top(Vec::new()));
    let (mut total, mut before) = (0usize, 0usize);
    for shard in state.shards.values() {
        for row in &shard.rows {
            if !q.in_source(row) {
                continue;
            }
            Tally::mark(&mut tally.cameras, row.cand.camera);
            if row.gone || !q.by_line(row) {
                continue;
            }
            if !row.readable() {
                tally.unreadable += 1;
                continue;
            }
            Tally::mark(&mut tally.class_sets, row.classes());
            if !q.by_class(row) {
                continue;
            }
            tally.count(row);
            if !q.by_state(row) {
                continue;
            }
            total += 1;
            let hit = Hit {
                key: q.key(row),
                shard,
                row,
            };
            first.offer(hit);
            if let Some(p) = &pivot {
                match hit.order(p) {
                    Ordering::Less => before += 1,
                    Ordering::Equal => {}
                    Ordering::Greater => next.offer(hit),
                }
            }
        }
    }
    // As `next_after`: past `after`, past the last back to the first, no `after` from the first.
    let past_pivot = pivot.is_some() && before + 1 < total;
    let (hits, position, wrapped) = if past_pivot {
        (next.0, before + 2, false)
    } else {
        (first.0, 1, pivot.is_some())
    };
    let found = hits.first().map(|h| item(h, f.auto, names));
    let upcoming: Vec<String> = hits
        .iter()
        .skip(1)
        .take(UPCOMING)
        .map(|h| h.shard.rel(h.row))
        .collect();
    let mut shown = upcoming.clone();
    shown.extend(found.iter().map(|i| i.frame.clone()));
    let unreadable = tally.unreadable + state.damaged(f.source == QueueSource::All);
    let (counts, cameras, labels) = tally.finish(names);
    let answer = Answer {
        position: found.as_ref().map(|_| position),
        wrapped: found.is_some() && wrapped,
        item: found,
        total,
        counts,
        cameras,
        labels,
        unreadable,
        upcoming,
        indexing: None,
    };
    (answer, shown)
}
