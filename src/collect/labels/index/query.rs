//! The answer from the rows. In the uncertainty order ([`Order::Uncertain`]): one pass over them,
//! no sort. `next_after` needs a place in the sorted queue and the few frames after it, which
//! counting the rows before `after` (or `then`, the frame it lands on) and keeping the smallest
//! few gives in one pass; a sort would build the order of every other frame to throw it away.
//! The order is the one of [`super::super::queue::sort_items`]: [`Uncertainty::key`], then the
//! newest, then the path. It has no ties, paths being unique, so what the pass finds is what the
//! sort would put there.
//!
//! The Recommended order ([`Order::Recommended`]) is no key of a row alone: a row's event rank
//! depends on the other rows of its event, and its need on the reviewed rows ([`need::CellTally`],
//! counted first by [`need_tally`]) and on its day's events (gaps and visits, [`need::day`] per
//! shard at query time). Its pass gathers the counted rows per day folder and the listed ones'
//! keys and needs ([`need::Scorer::score`], the walk's function on the walk's inputs), and
//! [`recommend::arrange`] sorts them as the walk does.
//!
//! The steps of the pass are those of `queue::build`, in its order. Only the found frame and the
//! ones after it become strings.

use super::super::Status;
use super::super::need::{self, CellBox, CellTally, DayOut, Need, RecallNeed, Scorer};
use super::super::queue::{
    AutoFilter, Balance, BalanceNeed, BalanceTally, ClassNeed, Counts, Filters, Item, Order,
    QueueSource, SortKey, StatusFilter, Uncertainty,
};
use super::super::recommend;
use super::super::statics;
use super::State;
use super::answer::{Answer, UPCOMING};
use super::row::{Name, Names, Row, Shard, auto_bit, stem_of};
use crate::sync::LockExt as _;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

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
    /// By refine summary: the live suggestions of the query's `refine_kind` and their strength.
    refine: Vec<(u32, u32)>,
    /// With `balance=`: what the rows weigh against ([`Weigh::new`], before anything sorts).
    weigh: Option<Weigh>,
}

/// What `balance=` weighs a row by: [`BalanceTally`] of the rows the query counts, and each class
/// set's (per camera), scene's and camera's part of the need, by id. The parts are
/// [`BalanceTally::need`]'s, so a score is the walk's.
struct Weigh {
    by: Balance,
    tally: BalanceTally,
    /// Balancing classes, by (class set id ([`Row::classes`]), camera id) of the listed rows: the
    /// class part depends on the camera through [`BalanceTally::precision`].
    class: HashMap<(u32, Name), ClassNeed>,
    /// By name id: (need, reviewed frames) as a scene, and as a camera.
    scene: Vec<(u32, u64)>,
    camera: Vec<(u32, u64)>,
}

impl Weigh {
    fn new(by: Balance, f: &Filters, state: &State, q: &Compiled, names: &Names) -> Self {
        // Reviewed rows by (training set, proposed labels, scene, camera): a few groups, so few
        // strings. And the (class set, camera) pairs of the listed rows, whose class part is asked.
        let mut groups: HashMap<(u32, u32, Name, Name), u64> = HashMap::new();
        let mut listed: BTreeSet<(u32, Name)> = BTreeSet::new();
        for shard in state.shards.values() {
            for row in &shard.rows {
                if !q.counts(row) {
                    continue;
                }
                if row.part.status == Status::Reviewed {
                    let key = (
                        row.part.training,
                        row.part.proposed,
                        row.cand.scene,
                        row.cand.camera,
                    );
                    *groups.entry(key).or_default() += 1;
                }
                if by.class && q.by_state(row) {
                    listed.insert((row.classes(), row.cand.camera));
                }
            }
        }
        let mut sorted: Vec<_> = groups.into_iter().collect();
        sorted.sort_unstable();
        let mut tally = BalanceTally::default();
        let mut trained: HashMap<u32, BTreeSet<String>> = HashMap::new();
        let mut proposed: HashMap<u32, BTreeSet<String>> = HashMap::new();
        for ((training, offered, scene, camera), n) in sorted {
            let classes = trained.entry(training).or_insert_with(|| {
                f.wanted_of(names.training(training).iter().map(|(c, _)| names.name(*c)))
            });
            let offered = proposed.entry(offered).or_insert_with(|| {
                f.wanted_of(names.proposed(offered).iter().map(|c| names.name(*c)))
            });
            tally.add(n, names.name(scene), names.name(camera), classes, offered);
        }
        let mut sure: HashMap<u32, BTreeMap<String, u8>> = HashMap::new();
        let class = listed
            .into_iter()
            .map(|(set, camera)| {
                let classes = sure.entry(set).or_insert_with(|| {
                    f.wanted_confidence(
                        names
                            .class_set(set)
                            .iter()
                            .map(|(c, s)| (names.name(*c), *s)),
                    )
                });
                let need = tally.class_need(classes, names.name(camera));
                ((set, camera), need)
            })
            .collect();
        let ids = 0..u32::try_from(names.name_count()).unwrap_or(u32::MAX);
        let scene = ids
            .clone()
            .map(|id| tally.scene_need(names.name(id)))
            .collect();
        let camera = ids.map(|id| tally.camera_need(names.name(id))).collect();
        Self {
            by,
            tally,
            class,
            scene,
            camera,
        }
    }

    /// The class part of a listed row (None when not balancing classes).
    fn class_of(&self, r: &Row) -> Option<&ClassNeed> {
        self.class.get(&(r.classes(), r.cand.camera))
    }

    /// [`BalanceNeed::score`] of `r`, without building the rest.
    fn score(&self, r: &Row) -> u32 {
        let part = |on: bool, of: &[(u32, u64)], id: Name| {
            if on {
                of.get(id as usize).map_or(0, |p| p.0)
            } else {
                0
            }
        };
        let class = if self.by.class {
            self.class_of(r).map_or(0, |c| c.need)
        } else {
            0
        };
        class
            + part(self.by.scene, &self.scene, r.cand.scene)
            + part(self.by.camera, &self.camera, r.cand.camera)
    }

    fn need(&self, r: &Row, names: &Names) -> BalanceNeed {
        self.tally.need(
            self.by,
            self.class_of(r).cloned(),
            names.name(r.cand.scene),
            names.name(r.cand.camera),
        )
    }
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
                .map(|set| f.label.is_empty() || set.iter().any(|(c, _)| wanted.contains(c)))
                .collect(),
            status: f.status,
            auto: f.auto,
            refine: if f.auto == Some(AutoFilter::Refine) {
                names.refine_by_summary(f.refine_kind)
            } else {
                Vec::new()
            },
            weigh: None,
        }
    }

    /// The live suggestions of `r` that the query counts, and the strongest one's strength:
    /// [`Filters::refine_open`]. (0, 0) under any filter but `auto=refine`.
    fn refine_of(&self, r: &Row) -> (u32, u32) {
        r.part
            .refine
            .checked_sub(1)
            .and_then(|id| self.refine.get(id as usize))
            .copied()
            .unwrap_or((0, 0))
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
        match self.auto {
            // Of the query's kind: the summary, not a bit.
            Some(AutoFilter::Refine) => self.refine_of(r).0 > 0,
            Some(a) => r.part.auto & auto_bit(a) != 0,
            None => true,
        }
    }

    /// Whether [`Counts`] counts `r`: every filter but status and auto keeps it.
    fn counts(&self, r: &Row) -> bool {
        self.in_source(r) && self.by_line(r) && !r.gone && r.readable() && self.by_class(r)
    }

    /// Whether the queue lists `r`.
    fn lists(&self, r: &Row) -> bool {
        self.counts(r) && self.by_state(r)
    }

    /// What sorts `r`. No note text: the sort only asks whether there is one.
    fn key(&self, r: &Row) -> SortKey {
        self.uncertainty(r).key()
    }

    /// What the orders read of `r`: its uncertainty with the balance score alone and no note text.
    fn uncertainty(&self, r: &Row) -> Uncertainty {
        let mut u = r.uncertainty(
            self.auto,
            self.refine_of(r),
            (r.part.note != 0).then(String::new),
        );
        u.balance = self.weigh.as_ref().map(|w| BalanceNeed {
            score: w.score(r),
            ..BalanceNeed::default()
        });
        u
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
            .map(|(c, _)| names.name(*c))
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

/// Where the placement starts from, as [`super::super::queue::next_after`] reads `after` and
/// `then`: the listed row `after` names, past which it lands, else the listed row `then` names,
/// on which it lands (`on`).
#[derive(Clone, Copy)]
struct Pivot<'a> {
    hit: Hit<'a>,
    on: bool,
}

impl<'a> Pivot<'a> {
    fn of(state: &'a State, q: &Compiled, after: Option<&str>, then: Option<&str>) -> Option<Self> {
        match listed(state, q, after) {
            Some(hit) => Some(Self { hit, on: false }),
            None => listed(state, q, then).map(|hit| Self { hit, on: true }),
        }
    }
}

/// The row `rel` names, when it is listed.
fn listed<'a>(state: &'a State, q: &Compiled, rel: Option<&str>) -> Option<Hit<'a>> {
    let rel = rel?;
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

fn item(shard: &Shard, r: &Row, q: &Compiled, names: &Names) -> Item {
    let mut uncertainty = r.uncertainty(q.auto, q.refine_of(r), shard.note(r).map(str::to_string));
    uncertainty.balance = q.weigh.as_ref().map(|w| w.need(r, names));
    Item {
        frame: shard.rel(r),
        timestamp: shard.timestamp(r).to_string(),
        scene: names.name(r.cand.scene).to_string(),
        camera: names.name(r.cand.camera).to_string(),
        model: names.name(r.side.model).to_string(),
        buckets: names.buckets(r.cand.buckets).to_vec(),
        status: r.part.status,
        queued: r.queued,
        uncertainty,
        event_rank: None,
        random_pick: false,
    }
}

/// What [`pass`] hands on: a row the query counts, with whether it lists it, or the end of a
/// shard (a day folder).
enum Step<'a> {
    Counted(&'a Shard, &'a Row, bool),
    DayDone,
}

/// The steps of `queue::build` over every row, in its order: `tally` takes the cameras, classes,
/// counts and unreadable frames, `each` every counted row and the end of each shard.
fn pass<'a>(state: &'a State, q: &Compiled, tally: &mut Tally, each: &mut dyn FnMut(Step<'a>)) {
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
            each(Step::Counted(shard, row, q.by_state(row)));
        }
        each(Step::DayDone);
    }
}

/// What the Recommended order derives from one shard's rows alone, kept per shard until the shard
/// changes ([`Shard::version`]) or the query counts other rows ([`count_key`]): its counted rows'
/// gaps and visits ([`need::day`]) and its reviewed rows' part of the tally (also kept with the
/// parked spots it was counted with, [`Need::spots`], compared by identity: the entry holds the
/// table, so no other table can take its address). Only what is pure of the rows is kept, so an
/// answer from it is the answer without it.
#[derive(Debug, Default)]
pub(super) struct NeedCache {
    shards: Mutex<HashMap<String, ShardNeed>>,
}

#[derive(Debug, Default)]
struct ShardNeed {
    version: u64,
    filter: String,
    outs: Option<Arc<Vec<DayOut>>>,
    tally: Option<(Option<Arc<statics::Table>>, Arc<CellTally>)>,
}

impl NeedCache {
    /// The entry of `shard` for `filter`, emptied when either changed.
    fn with<T>(&self, shard: &Shard, filter: &str, f: impl FnOnce(&mut ShardNeed) -> T) -> T {
        let mut map = self.shards.lock_recover();
        let e = map.entry(shard.day.clone()).or_default();
        if e.version != shard.version || e.filter != filter {
            *e = ShardNeed {
                version: shard.version,
                filter: filter.to_string(),
                ..ShardNeed::default()
            };
        }
        f(e)
    }

    fn outs(
        &self,
        shard: &Shard,
        filter: &str,
        make: impl FnOnce() -> Vec<DayOut>,
    ) -> Arc<Vec<DayOut>> {
        if let Some(o) = self.with(shard, filter, |e| e.outs.clone()) {
            return o;
        }
        let o = Arc::new(make());
        self.with(shard, filter, |e| e.outs = Some(o.clone()));
        o
    }

    fn tally(
        &self,
        shard: &Shard,
        filter: &str,
        spots: Option<&Arc<statics::Table>>,
        make: impl FnOnce() -> CellTally,
    ) -> Arc<CellTally> {
        let same = |kept: &Option<Arc<statics::Table>>| match (kept, spots) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        let kept = self.with(shard, filter, |e| {
            e.tally
                .as_ref()
                .filter(|(s, _)| same(s))
                .map(|(_, t)| t.clone())
        });
        if let Some(t) = kept {
            return t;
        }
        let t = Arc::new(make());
        self.with(shard, filter, |e| {
            e.tally = Some((spots.cloned(), t.clone()))
        });
        t
    }
}

/// What decides the rows a query counts (every filter but status and auto): the key of the
/// shard parts of [`NeedCache`].
fn count_key(f: &Filters) -> String {
    format!(
        "{:?}\u{1f}{:?}\u{1f}{:?}\u{1f}{:?}\u{1f}{:?}",
        f.source, f.scene, f.camera, f.bucket, f.label
    )
}

/// The reviewed rows `q` counts, as the walk counts them ([`Need::add_reviewed`]), one shard's
/// part at a time ([`NeedCache`]).
fn need_tally(
    state: &State,
    q: &Compiled,
    names: &Names,
    n: &Need<'_>,
    cache: &NeedCache,
    filter: &str,
) -> CellTally {
    let mut tally = CellTally::default();
    for shard in state.shards.values() {
        let part = cache.tally(shard, filter, n.spots(), || {
            let mut t = CellTally::default();
            let date = need::date_of(&shard.day);
            for row in &shard.rows {
                if row.part.status != Status::Reviewed || !q.counts(row) {
                    continue;
                }
                n.add_reviewed(
                    &mut t,
                    &need::Reviewed {
                        camera: names.name(row.cand.camera),
                        scene: names.name(row.cand.scene),
                        day: date,
                        width: row.side.width,
                        height: row.side.height,
                        boxes: shard.boxes_of(row),
                    },
                );
            }
            t
        });
        tally.merge_ref(&part);
    }
    tally
}

/// The answer of `state` for `f`, `after` and `then`, with the frames it shows:
/// [`Answer::indexing`] is the caller's.
pub(super) fn answer(
    state: &State,
    names: &Names,
    f: &Filters,
    after: Option<&str>,
    then: Option<&str>,
    cache: &NeedCache,
) -> (Answer, Vec<String>) {
    let mut q = Compiled::new(f, names);
    // Before anything sorts: the pivot's place needs the scores too.
    if let Some(by) = f.balance_by() {
        q.weigh = Some(Weigh::new(by, f, state, &q, names));
    }
    let pivot = Pivot::of(state, &q, after, then);
    let mut tally = Tally::new(names);
    let n = f.need();
    let filter = count_key(f);
    let cells = n
        .as_ref()
        .map(|n| need_tally(state, &q, names, n, cache, &filter));
    let placed = match (f.order, &n, &cells) {
        (Order::Recommended, Some(n), Some(cells)) => {
            let random = recommend::random_slots(f);
            let scorer = Scorer::new(n, cells);
            let weighs = Weighs {
                scorer: &scorer,
                names,
                cache,
                filter: &filter,
            };
            recommended(state, &q, &mut tally, pivot.as_ref(), random, &weighs)
        }
        _ => uncertain(state, &q, &mut tally, pivot.as_ref()),
    };
    let found = placed.found.map(|(shard, row, rank, random_pick)| {
        let mut it = Item {
            event_rank: rank,
            random_pick,
            ..item(shard, row, &q, names)
        };
        it.uncertainty.need.clone_from(&placed.need);
        it
    });
    let upcoming: Vec<String> = placed
        .upcoming
        .iter()
        .map(|(shard, row)| shard.rel(row))
        .collect();
    let mut shown = upcoming.clone();
    shown.extend(found.iter().map(|i| i.frame.clone()));
    let unreadable = tally.unreadable + state.damaged(f.source == QueueSource::All);
    let (counts, cameras, labels) = tally.finish(names);
    let answer = Answer {
        position: found.as_ref().map(|_| placed.position),
        wrapped: found.is_some() && placed.wrapped,
        item: found,
        total: placed.total,
        counts,
        cameras,
        labels,
        unreadable,
        upcoming,
        indexing: None,
        balance: q.weigh.map(|w| w.tally),
        need: n
            .as_ref()
            .zip(cells.as_ref())
            .map(|(n, c)| c.counts(n.wanted().map_or(&[][..], |w| w.classes))),
    };
    (answer, shown)
}

/// What weighs the listed rows under the Recommended order.
struct Weighs<'a> {
    scorer: &'a Scorer<'a>,
    names: &'a Names,
    cache: &'a NeedCache,
    filter: &'a str,
}

impl Weighs<'_> {
    fn facts<'s>(&'s self, shard: &'s Shard, row: &Row) -> need::Facts<'s> {
        need::Facts {
            camera: self.names.name(row.cand.camera),
            scene: self.names.name(row.cand.scene),
            day: need::date_of(&shard.day),
            width: row.side.width,
            height: row.side.height,
        }
    }

    /// The need of a listed row with gaps `gaps`, and the class that expands its visit of
    /// `visit_classes`.
    fn visit(
        &self,
        shard: &Shard,
        row: &Row,
        gaps: &[CellBox],
        visit_classes: &[u16],
    ) -> (recommend::Visit, Option<u16>) {
        let d = self
            .scorer
            .score(&self.facts(shard, row), shard.boxes_of(row), gaps);
        let class = self
            .scorer
            .expands(self.names.name(row.cand.camera), visit_classes);
        (
            recommend::Visit {
                score: d.score,
                expanded: class.is_some(),
            },
            class,
        )
    }

    /// The page's need of a listed row at event rank `rank`, as the walk makes it.
    fn need(
        &self,
        shard: &Shard,
        row: &Row,
        gaps: &[CellBox],
        visit: Option<u16>,
        rank: u32,
    ) -> RecallNeed {
        let at = self.facts(shard, row);
        let d = self.scorer.score(&at, shard.boxes_of(row), gaps);
        let mut n = self.scorer.explain(&d, at.camera, at.scene, visit);
        n.visit_score = need::visit_score(n.score, rank, visit.is_some());
        n
    }
}

/// Where `next_after` lands: the frame (with its event rank under the Recommended order, and
/// whether it is a random pick), its 1-based position, whether it wrapped, the frames after it
/// and how many are listed.
struct Placed<'a> {
    found: Option<(&'a Shard, &'a Row, Option<u32>, bool)>,
    position: usize,
    wrapped: bool,
    upcoming: Vec<(&'a Shard, &'a Row)>,
    total: usize,
    /// Under the Recommended order: the found frame's need.
    need: Option<RecallNeed>,
}

/// The uncertainty order in one pass, no sort: the rows before the pivot are counted and the
/// first few of the queue and after the pivot (from it, when the answer lands on it) kept.
fn uncertain<'a>(
    state: &'a State,
    q: &Compiled,
    tally: &mut Tally,
    pivot: Option<&Pivot<'a>>,
) -> Placed<'a> {
    let (mut first, mut next) = (Top(Vec::new()), Top(Vec::new()));
    let (mut total, mut before) = (0usize, 0usize);
    pass(state, q, tally, &mut |step| {
        let Step::Counted(shard, row, true) = step else {
            return;
        };
        total += 1;
        let hit = Hit {
            key: q.key(row),
            shard,
            row,
        };
        first.offer(hit);
        if let Some(p) = pivot {
            match hit.order(&p.hit) {
                Ordering::Less => before += 1,
                Ordering::Equal if !p.on => {}
                Ordering::Equal | Ordering::Greater => next.offer(hit),
            }
        }
    });
    // As `next_after`: past `after`, past the last back to the first, on `then`, else from the
    // first.
    let (hits, position, wrapped) = match pivot {
        Some(p) if p.on => (next.0, before + 1, false),
        Some(_) if before + 1 < total => (next.0, before + 2, false),
        Some(_) => (first.0, 1, true),
        None => (first.0, 1, false),
    };
    Placed {
        found: hits.first().map(|h| (h.shard, h.row, None, false)),
        position,
        wrapped,
        upcoming: hits
            .iter()
            .skip(1)
            .take(UPCOMING)
            .map(|h| (h.shard, h.row))
            .collect(),
        total,
        need: None,
    }
}

/// The Recommended order: events need every counted row of a day folder and ranks need their
/// event's other listed rows, so this sorts what it lists ([`recommend::arrange`], as the walk
/// does) and finds the pivot in it. With `random` (the view has random slots,
/// [`recommend::random_slots`]), the pool frames in their slots as the walk places them
/// ([`recommend::with_random_slots`], the hash over [`Shard::rel_bytes`]).
fn recommended<'a>(
    state: &'a State,
    q: &Compiled,
    tally: &mut Tally,
    pivot: Option<&Pivot<'a>>,
    random: bool,
    weighs: &Weighs<'_>,
) -> Placed<'a> {
    let mut members: Vec<recommend::Member> = Vec::new();
    let mut events = recommend::Events::default();
    let mut listed: Vec<(&'a Shard, &'a Row)> = Vec::new();
    let mut keys: Vec<recommend::Key> = Vec::new();
    let mut visits: Vec<recommend::Visit> = Vec::new();
    // Per listed row: the day's outputs and its place in them (its gaps), and the class that
    // expands its visit.
    let mut gaps: Vec<(Arc<Vec<DayOut>>, usize, Option<u16>)> = Vec::new();
    // The day's counted rows, for [`need::day`].
    let mut day: Vec<(&'a Shard, &'a Row, bool)> = Vec::new();
    pass(state, q, tally, &mut |step| match step {
        Step::Counted(shard, row, lists) => {
            members.push(recommend::Member {
                ms: recommend::event_ms(shard.timestamp(row)),
                alone: row.side.alone,
                decided: row.part.status != Status::Unreviewed,
                listed: lists.then_some(listed.len()),
            });
            day.push((shard, row, lists));
            if lists {
                listed.push((shard, row));
            }
        }
        Step::DayDone => {
            events.add_day(&mut members);
            members.clear();
            let Some((this, _, _)) = day.first().copied() else {
                return;
            };
            let outs = weighs.cache.outs(this, weighs.filter, || {
                let frames: Vec<need::DayFrame<'_>> = day
                    .iter()
                    .map(|(shard, row, _)| need::DayFrame {
                        ms: recommend::event_ms(shard.timestamp(row)),
                        alone: row.side.alone,
                        boxes: shard.boxes_of(row),
                    })
                    .collect();
                need::day(&frames)
            });
            for (at, ((shard, row, lists), out)) in day.iter().zip(outs.iter()).enumerate() {
                if !lists {
                    continue;
                }
                let (visit, class) = weighs.visit(shard, row, &out.gaps, &out.visit_classes);
                keys.push(recommend::key(&q.uncertainty(row), visit));
                visits.push(visit);
                gaps.push((outs.clone(), at, class));
            }
            day.clear();
        }
    });
    let tie = |a: usize, b: usize| match (listed.get(a), listed.get(b)) {
        (Some((sa, ra)), Some((sb, rb))) => sb
            .timestamp(rb)
            .cmp(sa.timestamp(ra))
            .then_with(|| sa.rel_bytes(ra).cmp(sb.rel_bytes(rb))),
        _ => a.cmp(&b),
    };
    let order = recommend::arrange(&events, &keys, &visits, &tie);
    let order: Vec<(usize, u32, bool)> = if random {
        // As the walk: one more with each frame reviewed or skipped of those counted.
        let decided = tally.counts.reviewed.saturating_add(tally.counts.skipped);
        let pool_hash_of = |i: usize| {
            let (shard, row) = listed.get(i)?;
            let hash = recommend::pool_hash_bytes(shard.rel_bytes(row));
            recommend::is_pool_frame(hash, row.part.note != 0).then_some(hash)
        };
        let path = |a: usize, b: usize| match (listed.get(a), listed.get(b)) {
            (Some((sa, ra)), Some((sb, rb))) => sa.rel_bytes(ra).cmp(sb.rel_bytes(rb)),
            _ => a.cmp(&b),
        };
        recommend::with_random_slots(&order, &pool_hash_of, &path, decided)
    } else {
        order.into_iter().map(|(i, r)| (i, r, false)).collect()
    };
    let at = |i: usize| listed.get(i).copied();
    // As `next_after`: past `after`, past the last back to the first, on `then`, else from the
    // first.
    let pos = pivot.and_then(|p| {
        order
            .iter()
            .position(|(i, _, _)| at(*i).is_some_and(|(_, r)| std::ptr::eq(r, p.hit.row)))
            .map(|found| (found, p.on))
    });
    let (from, wrapped) = match pos {
        Some((p, true)) => (p, false),
        Some((p, false)) if p + 1 < order.len() => (p + 1, false),
        Some(_) => (0, true),
        None => (0, false),
    };
    let need = order.get(from).and_then(|(i, rank, _)| {
        let (shard, row) = at(*i)?;
        let (outs, place, class) = gaps.get(*i)?;
        let gaps = outs.get(*place).map_or(&[][..], |o| o.gaps.as_slice());
        Some(weighs.need(shard, row, gaps, *class, *rank))
    });
    Placed {
        need,
        found: order
            .get(from)
            .and_then(|(i, rank, random)| at(*i).map(|(s, r)| (s, r, Some(*rank), *random))),
        position: from + 1,
        wrapped,
        upcoming: order
            .iter()
            .skip(from + 1)
            .take(UPCOMING)
            .filter_map(|(i, _, _)| at(*i))
            .collect(),
        total: order.len(),
    }
}
