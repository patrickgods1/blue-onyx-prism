//! The Recommended review order, the default of `GET /v1/review/next` (no `auto=`, no `balance=`;
//! [`super::queue::Order::Recommended`]): the frames of the highest recall need first
//! ([`super::need`]), divided by the event rank (`need / (1 + rank · P)`, [`super::need::visit_score`];
//! P is 1/4 in a visit of a rare or catch-up class, so it serves more of its poses), in bands of
//! [`super::need::NEED_BAND`] so uncertainty decides within a band, then the uncertainty keys of
//! [`super::queue::Uncertainty::key`] from the likely boxes on. Reopened frames (a re-label
//! disagreement) come first and repeat-only frames last, as in the uncertainty order.
//!
//! An event is one camera's activity: the frames of one day folder (`<scene>/<camera>/<date>`, so
//! an event ends at UTC midnight and at a scene change) whose times are at most [`EVENT_GAP_S`]
//! apart, spanning at most [`EVENT_MAX_S`] from its first frame. A frame with no readable time,
//! and an imported frame without a real time within its source ([`own_event`]: an image-folder
//! import carries file times, and its frames were already spread out when chosen), is an event of
//! its own; a frame of an imported video has a real time (the video's start plus its offset) and
//! joins events like a collected one. Events are built over the
//! frames the queue counts (every filter but status and auto), not those it lists, so an event
//! does not split as its frames are reviewed: a frame's event rank is the decided frames of its
//! event that the queue does not list, plus the listed frames of its event ahead of it.
//!
//! Random picks: one served slot in [`RANDOM_EVERY`] is a frame of a fixed pool chosen by a hash
//! of its path alone ([`in_pool`]), whatever it shows, so the held-out frames drawn from reviewed
//! ones include frames nobody chose. Only the Review tab's default view has them
//! ([`random_slots`]); [`interleave`] places them on top of the order [`arrange`] gives the rest.
//!
//! Pure: both the walk ([`super::queue::build`]) and the review index call [`Events::add_day`],
//! [`arrange`] and [`interleave`] on what they read, so their orders are one. [`arrange`] returns
//! the order as indices, for the random slots on top of it.

use super::Status;
use super::need::{NEED_BAND, visit_score};
use super::queue::{Filters, Order, StatusFilter, Uncertainty};
use crate::collect::record::Record;
use std::cmp::{Ordering, Reverse};

/// Most seconds between two frames of one event.
pub const EVENT_GAP_S: i64 = 60;
/// Most seconds from an event's first frame to its last.
pub const EVENT_MAX_S: i64 = 600;
/// The sidecar `model` of an imported frame (`training::import`): an event of its own unless it
/// has a real time within its source ([`own_event`]).
pub const IMPORT_MODEL: &str = "import";
/// The `provenance.kind` of a frame imported from an image folder: its time is the file's.
pub const IMAGE_KIND: &str = "image";

/// The frame of sidecar `rec` is an event of its own: an import ([`IMPORT_MODEL`]) without a real
/// time within its source. An image import's time is the file's modification time, and an import
/// with no provenance or no `time_offset_s` says nothing better; a frame of a video (also a URL or
/// a dataset import) with an offset has the video's start plus the offset, a real time. The walk
/// and the review index both call this, so they agree.
pub fn own_event(rec: &Record) -> bool {
    rec.model == IMPORT_MODEL
        && rec
            .provenance
            .as_ref()
            .is_none_or(|p| p.kind == IMAGE_KIND || p.time_offset_s.is_none())
}

const GAP_MS: i64 = EVENT_GAP_S * 1000;
const MAX_MS: i64 = EVENT_MAX_S * 1000;

/// Milliseconds since the Unix epoch of a sidecar's `timestamp` (`2026-10-22T12:34:56.789Z`,
/// [`crate::collect::utc_rfc3339`], or without the milliseconds), else any RFC 3339 time; None
/// when it is none.
pub fn event_ms(ts: &str) -> Option<i64> {
    fixed_layout(ts.as_bytes()).or_else(|| {
        chrono::DateTime::parse_from_rfc3339(ts)
            .ok()
            .map(|t| t.timestamp_millis())
    })
}

/// `YYYY-MM-DDTHH:MM:SS[.mmm]Z` with valid fields, without a parser: the layout every sidecar
/// has. Anything else (another zone, other digits, a leap second) is chrono's.
fn fixed_layout(b: &[u8]) -> Option<i64> {
    let millis = match b.len() {
        20 => false,
        24 => true,
        _ => return None,
    };
    let num = |from: usize, len: usize| -> Option<i64> {
        let digits = b.get(from..from + len)?;
        digits.iter().try_fold(0i64, |n, d| {
            d.is_ascii_digit().then(|| n * 10 + i64::from(d - b'0'))
        })
    };
    let at = |i: usize, c: u8| b.get(i) == Some(&c);
    if !(at(4, b'-') && at(7, b'-') && at(10, b'T') && at(13, b':') && at(16, b':')) {
        return None;
    }
    let (year, month, day) = (num(0, 4)?, num(5, 2)?, num(8, 2)?);
    let (hour, minute, second) = (num(11, 2)?, num(14, 2)?, num(17, 2)?);
    let ms = if millis {
        if !(at(19, b'.') && at(23, b'Z')) {
            return None;
        }
        num(20, 3)?
    } else {
        if !at(19, b'Z') {
            return None;
        }
        0
    };
    if !(1..=12).contains(&month)
        || day < 1
        || day > days_in_month(year, month)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return None;
    }
    let days = days_from_civil(year, month, day);
    Some(((days * 24 + hour) * 60 + minute) * 60_000 + second * 1000 + ms)
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Days from 1970-01-01 of a proleptic Gregorian date (H. Hinnant's `days_from_civil`).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// A frame of one day folder that the queue counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Member {
    /// [`event_ms`] of its sidecar's time.
    pub ms: Option<i64>,
    /// An event of its own ([`own_event`]).
    pub alone: bool,
    /// Reviewed or skipped.
    pub decided: bool,
    /// Its index among the listed frames, when the queue lists it.
    pub listed: Option<usize>,
}

/// The events of the counted frames, as far as the order needs them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Events {
    /// Per event: its decided frames the queue does not list.
    decided: Vec<u32>,
    /// (event, listed index) of every listed frame.
    listed: Vec<(usize, usize)>,
}

/// The events of one day folder's frames of times `ms` (None: no time, or an event of its own):
/// per frame, in their order, its event's number from 0 (in time order of the events' first
/// frames; frames without a time after them). The order the frames come in changes no event.
pub fn chain(ms: &[Option<i64>]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..ms.len()).collect();
    let at = |i: usize| ms.get(i).copied().flatten();
    // Frames that chain first, by time; the ones alone after them.
    order.sort_by_key(|&i| (at(i).is_none(), at(i), i));
    let mut out = vec![0usize; ms.len()];
    // (first, last) time of the open event.
    let mut open: Option<(i64, i64)> = None;
    let mut events = 0usize;
    for i in order {
        let t = at(i);
        let joins = match (t, open) {
            (Some(t), Some((first, last))) => {
                t.saturating_sub(last) <= GAP_MS && t.saturating_sub(first) <= MAX_MS
            }
            _ => false,
        };
        if joins {
            open = open.map(|(first, last)| (first, t.unwrap_or(last)));
        } else {
            events += 1;
            open = t.map(|t| (t, t));
        }
        if let Some(o) = out.get_mut(i) {
            *o = events - 1;
        }
    }
    out
}

impl Events {
    /// Add the events of one day folder's counted frames ([`chain`]).
    pub fn add_day(&mut self, members: &mut [Member]) {
        let times: Vec<Option<i64>> = members.iter().map(|m| m.ms.filter(|_| !m.alone)).collect();
        let of = chain(&times);
        let base = self.decided.len();
        let count = of.iter().copied().max().map_or(0, |m| m + 1);
        self.decided.extend(std::iter::repeat_n(0, count));
        for (m, e) in members.iter().zip(of) {
            let event = base + e;
            match m.listed {
                Some(i) => self.listed.push((event, i)),
                None if m.decided => {
                    if let Some(n) = self.decided.get_mut(event) {
                        *n = n.saturating_add(1);
                    }
                }
                None => {}
            }
        }
    }

    /// How many events there are (with or without a listed frame).
    pub fn len(&self) -> usize {
        self.decided.len()
    }

    pub fn is_empty(&self) -> bool {
        self.decided.is_empty()
    }
}

/// What the Recommended order sorts by: repeat-only frames last, reopened frames first, then the
/// need band after the visit divisor (highest first), the likely boxes (most first), the flagged
/// boxes (fewest first), the margin in hundredths (no box last), verifier disagreement, suppressed
/// boxes (most first).
pub type Key = (
    bool,
    bool,
    Reverse<u32>,
    Reverse<u32>,
    usize,
    u32,
    bool,
    Reverse<usize>,
);

/// A listed frame's need ([`super::need::RecallNeed::score`]) and whether its visit expands
/// ([`super::need::Scorer::expands`]): what the event rank divides.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Visit {
    pub score: u32,
    pub expanded: bool,
}

impl Visit {
    /// The band at event rank `rank` ([`visit_score`] / [`NEED_BAND`]).
    fn band(self, rank: u32) -> u32 {
        visit_score(self.score, rank, self.expanded) / NEED_BAND
    }
}

/// The key of a frame of uncertainty `u` and need `visit` at event rank 0.
pub fn key(u: &Uncertainty, visit: Visit) -> Key {
    (
        u.repeats_only,
        u.relabel.is_none(),
        Reverse(visit.band(0)),
        Reverse(u.likely_boxes),
        u.auto_review,
        u.margin_hundredths(),
        !u.disagreement,
        Reverse(u.suppressed),
    )
}

/// `k` (of need `visit`) at the event rank `rank`.
fn ranked(mut k: Key, visit: Visit, rank: u32) -> Key {
    k.2 = Reverse(visit.band(rank));
    k
}

/// The Recommended order of the listed frames: `keys` are their [`key`]s at rank 0 of their needs
/// `visits`, `tie` orders two of them that have the same key (newest first, then by path: no two
/// are equal). Each listed frame's rank is its event's decided frames plus the listed frames of
/// its event ahead of it by the key at rank 0, so the rank is well defined; the order is then by
/// the key at that rank. The frames' indices in order, each with its event rank.
pub fn arrange(
    events: &Events,
    keys: &[Key],
    visits: &[Visit],
    tie: &dyn Fn(usize, usize) -> Ordering,
) -> Vec<(usize, u32)> {
    let base = |i: usize| keys.get(i).copied().unwrap_or_default();
    let visit = |i: usize| visits.get(i).copied().unwrap_or_default();
    let mut ranks = vec![0u32; keys.len()];
    let mut by_event = events.listed.clone();
    by_event.sort_unstable_by(|&(ea, a), &(eb, b)| {
        ea.cmp(&eb)
            .then_with(|| base(a).cmp(&base(b)))
            .then_with(|| tie(a, b))
    });
    for run in by_event.chunk_by(|x, y| x.0 == y.0) {
        let decided = run
            .first()
            .and_then(|(e, _)| events.decided.get(*e))
            .copied()
            .unwrap_or(0);
        for (ahead, (_, i)) in run.iter().enumerate() {
            if let Some(r) = ranks.get_mut(*i) {
                *r = decided.saturating_add(u32::try_from(ahead).unwrap_or(u32::MAX));
            }
        }
    }
    let rank = |i: usize| ranks.get(i).copied().unwrap_or(0);
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        ranked(base(a), visit(a), rank(a))
            .cmp(&ranked(base(b), visit(b), rank(b)))
            .then_with(|| tie(a, b))
    });
    order.into_iter().map(|i| (i, rank(i))).collect()
}

/// The Recommended order before the need model (event rank as a strict tier, then the balance of
/// class, scene and camera in bands of 250), kept for tests alone: the real-data comparison of the
/// two orders (`server::review::order_eval`).
#[cfg(test)]
pub(crate) mod legacy {
    use super::{Events, Ordering, Reverse, Uncertainty};

    pub(crate) const BALANCE_BAND: u32 = 250;

    pub(crate) type Key = (
        bool,
        bool,
        u32,
        Reverse<u32>,
        Reverse<u32>,
        usize,
        u32,
        bool,
        Reverse<usize>,
    );

    /// The old key of `u` (its `balance`, every dimension) at event rank `rank`.
    pub(crate) fn key(u: &Uncertainty, rank: u32) -> Key {
        (
            u.repeats_only,
            u.relabel.is_none(),
            rank,
            Reverse(u.balance.as_ref().map_or(0, |b| b.score) / BALANCE_BAND),
            Reverse(u.likely_boxes),
            u.auto_review,
            u.margin_hundredths(),
            !u.disagreement,
            Reverse(u.suppressed),
        )
    }

    /// The old `arrange`.
    pub(crate) fn arrange(
        events: &Events,
        keys: &[Key],
        tie: &dyn Fn(usize, usize) -> Ordering,
    ) -> Vec<(usize, u32)> {
        let base = |i: usize| keys.get(i).copied().unwrap_or_default();
        let mut ranks = vec![0u32; keys.len()];
        let mut by_event = events.listed.clone();
        by_event.sort_unstable_by(|&(ea, a), &(eb, b)| {
            ea.cmp(&eb)
                .then_with(|| base(a).cmp(&base(b)))
                .then_with(|| tie(a, b))
        });
        for run in by_event.chunk_by(|x, y| x.0 == y.0) {
            let decided = run
                .first()
                .and_then(|(e, _)| events.decided.get(*e))
                .copied()
                .unwrap_or(0);
            for (ahead, (_, i)) in run.iter().enumerate() {
                if let Some(r) = ranks.get_mut(*i) {
                    *r = decided.saturating_add(u32::try_from(ahead).unwrap_or(u32::MAX));
                }
            }
        }
        let rank = |i: usize| ranks.get(i).copied().unwrap_or(0);
        let ranked = |i: usize| {
            let mut k = base(i);
            k.2 = rank(i);
            k
        };
        let mut order: Vec<usize> = (0..keys.len()).collect();
        order.sort_unstable_by(|&a, &b| ranked(a).cmp(&ranked(b)).then_with(|| tie(a, b)));
        order.into_iter().map(|i| (i, rank(i))).collect()
    }
}

/// One served slot in this many is a random pick ([`interleave`]), and one frame in this many is
/// in the pool ([`in_pool`]).
pub const RANDOM_EVERY: u64 = 5;
/// Hashed before a frame's path ([`pool_hash`]): another salt is another pool, so it changes only
/// for a reviewed reason (the frames already marked would no longer be pool frames).
pub const RANDOM_SALT: &str = "bop-random-pick-v1/";

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64 of [`RANDOM_SALT`] and then the bytes `rel` of a frame's path
/// (`<scene>/<camera>/<date>/<stem>.jpg`, forward slashes, as `Item.frame`), for a caller that
/// builds no string: [`pool_hash`] of the same path.
pub fn pool_hash_bytes(rel: impl IntoIterator<Item = u8>) -> u64 {
    RANDOM_SALT.bytes().chain(rel).fold(FNV_OFFSET, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(FNV_PRIME)
    })
}

/// [`pool_hash_bytes`] of the frame path `rel`.
pub fn pool_hash(rel: &str) -> u64 {
    pool_hash_bytes(rel.bytes())
}

/// The frame `rel` is in the random-pick pool: its [`pool_hash`] is a multiple of
/// [`RANDOM_EVERY`]. Fixed by the path alone, so nothing a model or a reviewer does moves a frame
/// in or out. A reopened frame of the pool still takes no random slot ([`is_pool_frame`]).
pub fn in_pool(rel: &str) -> bool {
    pool_hash(rel).is_multiple_of(RANDOM_EVERY)
}

/// A listed frame of [`pool_hash`] `hash` takes the random slots: in the pool and not `reopened`
/// (a re-label disagreement keeps its place first in the order).
pub fn is_pool_frame(hash: u64, reopened: bool) -> bool {
    !reopened && hash.is_multiple_of(RANDOM_EVERY)
}

/// The queue `f` has random slots: the Review tab's default view alone, the Recommended order of
/// the unreviewed frames with no scene, camera, class, bucket, auto or balance filter. Any of
/// those would make the picks depend on what the frames show. `source` (the review queue, or
/// every frame) is allowed: it is the population the picks are random among, not a choice by
/// content.
pub fn random_slots(f: &Filters) -> bool {
    f.order == Order::Recommended
        && f.status == StatusFilter::Is(Status::Unreviewed)
        && f.bucket.is_none()
        && f.scene.is_none()
        && f.camera.is_none()
        && f.label.is_empty()
        && f.auto.is_none()
        && f.refine_kind.is_none()
        && f.balance.is_none()
}

/// The served order of `rest` (the listed frames out of the pool, in [`arrange`]'s order) and
/// `pool` (the pool frames, by ascending [`pool_hash`], then path) when `decided` frames of the
/// counted ones are reviewed or skipped: slot `i` (0-based) is a random slot when
/// `(decided + i) % RANDOM_EVERY == RANDOM_EVERY - 1`. A random slot takes the next pool frame,
/// any other slot the next of `rest`; when one side runs out, the other fills its slots. Each
/// index comes with whether it is a pool frame, a random pick whichever slot it fills: its hash
/// chose it either way.
///
/// A save or a skip of the first frame removes it and adds one to `decided`, so every frame after
/// it keeps its slot: the frames the page loaded ahead stay the ones served.
pub fn interleave(rest: &[usize], pool: &[usize], decided: u64) -> Vec<(usize, bool)> {
    let mut out = Vec::with_capacity(rest.len() + pool.len());
    let (mut rest, mut pool) = (rest.iter().copied(), pool.iter().copied());
    let mut phase = decided % RANDOM_EVERY;
    loop {
        let (from_rest, from_pool) = if phase == RANDOM_EVERY - 1 {
            (None, pool.next())
        } else {
            (rest.next(), None)
        };
        let next = from_rest
            .map(|i| (i, false))
            .or_else(|| from_pool.map(|i| (i, true)))
            .or_else(|| rest.next().map(|i| (i, false)))
            .or_else(|| pool.next().map(|i| (i, true)));
        let Some(slot) = next else {
            return out;
        };
        out.push(slot);
        phase = (phase + 1) % RANDOM_EVERY;
    }
}

/// [`arrange`]'s `order` with the random slots ([`interleave`]): `pool_hash_of` is the
/// [`pool_hash`] of a listed frame that takes them ([`is_pool_frame`]), None for any other, and
/// `path` orders two pool frames of the same hash. Each index with its event rank and whether it
/// is a random pick.
pub fn with_random_slots(
    order: &[(usize, u32)],
    pool_hash_of: &dyn Fn(usize) -> Option<u64>,
    path: &dyn Fn(usize, usize) -> Ordering,
    decided: u64,
) -> Vec<(usize, u32, bool)> {
    // `order` holds the indices 0..order.len(), each once.
    let mut rank = vec![0u32; order.len()];
    let (mut rest, mut pool) = (Vec::with_capacity(order.len()), Vec::new());
    for &(i, r) in order {
        if let Some(at) = rank.get_mut(i) {
            *at = r;
        }
        match pool_hash_of(i) {
            Some(h) => pool.push((h, i)),
            None => rest.push(i),
        }
    }
    pool.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| path(a.1, b.1)));
    let pool: Vec<usize> = pool.into_iter().map(|(_, i)| i).collect();
    interleave(&rest, &pool, decided)
        .into_iter()
        .map(|(i, random)| (i, rank.get(i).copied().unwrap_or(0), random))
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn times_parse_exactly() {
        assert_eq!(event_ms("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(event_ms("1970-01-01T00:00:01.250Z"), Some(1250));
        assert_eq!(event_ms("1969-12-31T23:59:59.999Z"), Some(-1));
        // The fixed layout agrees with chrono, leap days and month ends included.
        let mut day = chrono::NaiveDate::from_ymd_opt(1999, 12, 25).unwrap();
        let end = chrono::NaiveDate::from_ymd_opt(2101, 1, 10).unwrap();
        while day < end {
            for (h, m, s, ms) in [(0, 0, 0, 0), (23, 59, 59, 999), (12, 34, 56, 789)] {
                let t = day.and_hms_milli_opt(h, m, s, ms).unwrap().and_utc();
                let text = crate::collect::utc_rfc3339(t.into());
                assert_eq!(
                    fixed_layout(text.as_bytes()),
                    Some(t.timestamp_millis()),
                    "{text}"
                );
            }
            day = day + chrono::Days::new(17);
        }
        for leap in ["2024-02-29T00:00:00Z", "2000-02-29T00:00:00Z"] {
            assert!(fixed_layout(leap.as_bytes()).is_some(), "{leap}");
        }
        // Not the layout: chrono's, or nothing.
        for bad in [
            "2023-02-29T00:00:00Z",
            "1900-02-29T00:00:00Z",
            "2026-13-01T00:00:00Z",
            "2026-04-31T00:00:00Z",
            "2026-10-22T24:00:00Z",
            "2026-10-22T12:00:0xZ",
            "",
            "yesterday",
        ] {
            assert_eq!(fixed_layout(bad.as_bytes()), None, "{bad}");
            assert_eq!(event_ms(bad), None, "{bad}");
        }
        assert_eq!(
            event_ms("2026-10-22T14:00:00+02:00"),
            event_ms("2026-10-22T12:00:00Z")
        );
        assert_eq!(
            event_ms("2026-10-22T12:00:00.5Z"),
            Some(event_ms("2026-10-22T12:00:00Z").unwrap() + 500)
        );
    }

    fn at(s: i64) -> Member {
        Member {
            ms: Some(s * 1000),
            alone: false,
            decided: false,
            listed: None,
        }
    }

    /// The event of each member, numbered from 0 in the order of `members` (each listed as its
    /// index).
    fn events_of(members: &[Member]) -> Vec<usize> {
        let mut ms: Vec<Member> = members
            .iter()
            .enumerate()
            .map(|(i, m)| Member {
                listed: Some(i),
                ..*m
            })
            .collect();
        let mut e = Events::default();
        e.add_day(&mut ms);
        let mut of: Vec<(usize, usize)> = e.listed.iter().map(|&(ev, i)| (i, ev)).collect();
        of.sort_unstable();
        // Renumber by first appearance in `members`' order.
        let mut names = std::collections::BTreeMap::new();
        of.into_iter()
            .map(|(_, ev)| {
                let n = names.len();
                *names.entry(ev).or_insert(n)
            })
            .collect()
    }

    #[test]
    fn a_gap_over_a_minute_starts_an_event() {
        assert_eq!(
            events_of(&[at(0), at(60), at(121), at(150), at(1000)]),
            [0, 0, 1, 1, 2]
        );
        // The input order does not matter.
        assert_eq!(events_of(&[at(121), at(0), at(60)]), [0, 1, 1]);
    }

    #[test]
    fn an_event_spans_at_most_ten_minutes() {
        let steady: Vec<Member> = (0..25).map(|i| at(i * 30)).collect();
        let of = events_of(&steady);
        // 0..=600 s is one event, 630 s starts the next.
        assert!(of[..21].iter().all(|e| *e == 0), "{of:?}");
        assert!(of[21..].iter().all(|e| *e == 1), "{of:?}");
    }

    #[test]
    fn imports_and_unknown_times_are_alone_and_split_nothing() {
        let import = Member {
            alone: true,
            ..at(30)
        };
        let unknown = Member { ms: None, ..at(0) };
        let of = events_of(&[at(0), import, unknown, at(50), import, unknown]);
        assert_eq!(of[0], of[3], "the live frames stay one event: {of:?}");
        let mut alone = vec![of[1], of[2], of[4], of[5]];
        alone.sort_unstable();
        alone.dedup();
        assert_eq!(alone.len(), 4, "{of:?}");
        assert!(!alone.contains(&of[0]));
    }

    /// The sidecar of an imported frame: `kind` with `offset` seconds from the video's start, or
    /// no provenance at all for `None`.
    fn imported(kind: Option<&str>, offset: Option<f64>) -> Record {
        let mut r = crate::collect::record::empty_record();
        r.model = IMPORT_MODEL.into();
        r.provenance = kind.map(|kind| crate::collect::record::Provenance {
            import_id: "20261027-170500-abcd".into(),
            kind: kind.into(),
            path: Some("Garage.20261027_170000_1.mp4".into()),
            time_offset_s: offset,
            ..crate::collect::record::Provenance::default()
        });
        r
    }

    /// The ranks [`arrange`] gives two listed frames of sidecars `a` and `b`, `secs` apart, of one
    /// day folder and equal keys.
    fn ranks_of_two(a: &Record, b: &Record, secs: i64) -> Vec<u32> {
        let mut members = [
            Member {
                alone: own_event(a),
                listed: Some(0),
                ..at(0)
            },
            Member {
                alone: own_event(b),
                listed: Some(1),
                ..at(secs)
            },
        ];
        let mut e = Events::default();
        e.add_day(&mut members);
        let keys = [key(&u(0.2), Visit::default()); 2];
        let mut got: Vec<(usize, u32)> = arrange(&e, &keys, &[], &|a, b| a.cmp(&b));
        got.sort_unstable();
        got.into_iter().map(|(_, r)| r).collect()
    }

    #[test]
    fn frames_of_an_imported_video_chain_and_imported_images_stand_alone() {
        // Two frames of one Blue Iris clip 10 s apart (offsets 0 and 10.034 s): one event.
        let first = imported(Some("video"), Some(0.0));
        let second = imported(Some("video"), Some(10.034));
        assert!(!own_event(&first) && !own_event(&second));
        assert_eq!(ranks_of_two(&first, &second, 10), [0, 1]);
        // A URL or a dataset import is a video too.
        for kind in ["url", "dataset"] {
            let f = imported(Some(kind), Some(3.0));
            assert!(!own_event(&f), "{kind}");
            assert_eq!(ranks_of_two(&first, &f, 10), [0, 1], "{kind}");
        }
        // Two images of a folder 10 s apart: file times, two events.
        let image = imported(Some(IMAGE_KIND), None);
        assert!(own_event(&image));
        assert_eq!(ranks_of_two(&image, &image, 10), [0, 0]);
        // An image with an offset (none is written) is still an image.
        assert!(own_event(&imported(Some(IMAGE_KIND), Some(1.0))));
        // A video frame without an offset, or an import without provenance: alone.
        let no_offset = imported(Some("video"), None);
        assert!(own_event(&no_offset));
        assert_eq!(ranks_of_two(&first, &no_offset, 10), [0, 0]);
        assert!(own_event(&imported(None, None)));
        assert_eq!(ranks_of_two(&first, &imported(None, None), 10), [0, 0]);
        // A collected frame is never alone, provenance or not.
        let mut collected = imported(Some(IMAGE_KIND), None);
        collected.model = "m0".into();
        assert!(!own_event(&collected));
    }

    fn u(margin: f32) -> Uncertainty {
        Uncertainty {
            margin: Some(margin),
            ..Uncertainty::default()
        }
    }

    /// The need every frame of [`order`] has: 5 bands, 2 at rank 1, 1 at rank 2.
    const NEED: Visit = Visit {
        score: 500,
        expanded: false,
    };

    /// `arrange` over one day of listed frames `us` (times `secs`, each of need [`NEED`]) with
    /// `decided` more decided frames at `decided_at`: the listed indices in order, with ranks.
    fn order(us: &[Uncertainty], secs: &[i64], decided_at: &[i64]) -> Vec<(usize, u32)> {
        order_of(us, &vec![NEED; us.len()], secs, decided_at)
    }

    /// [`order`] with each frame's need.
    fn order_of(
        us: &[Uncertainty],
        visits: &[Visit],
        secs: &[i64],
        decided_at: &[i64],
    ) -> Vec<(usize, u32)> {
        let mut members: Vec<Member> = secs
            .iter()
            .enumerate()
            .map(|(i, s)| Member {
                listed: Some(i),
                ..at(*s)
            })
            .collect();
        members.extend(decided_at.iter().map(|s| Member {
            decided: true,
            ..at(*s)
        }));
        let mut e = Events::default();
        e.add_day(&mut members);
        let keys: Vec<Key> = us.iter().zip(visits).map(|(u, v)| key(u, *v)).collect();
        arrange(&e, &keys, visits, &|a, b| a.cmp(&b))
    }

    #[test]
    fn the_first_frame_of_each_event_comes_before_any_second() {
        // Event A at 0..20 s holds the two most uncertain frames; event B at 1000 s one more.
        let us = [u(0.0), u(0.01), u(0.3)];
        let got = order(&us, &[0, 20, 1000], &[]);
        assert_eq!(got, [(0, 0), (2, 0), (1, 1)]);
    }

    #[test]
    fn decided_frames_raise_the_rank() {
        let us = [u(0.0), u(0.3)];
        // A reviewed frame of event A that the queue does not list.
        let got = order(&us, &[0, 1000], &[30]);
        assert_eq!(got, [(1, 0), (0, 1)]);
        // A decided frame far from both is an event of its own: nothing moves.
        assert_eq!(order(&us, &[0, 1000], &[5000]), [(0, 0), (1, 0)]);
    }

    #[test]
    fn reopened_frames_come_first_and_repeat_only_frames_last() {
        let reopened = Uncertainty {
            relabel: Some("model m: 1 new box".into()),
            ..u(0.4)
        };
        let repeats = Uncertainty {
            repeats_only: true,
            relabel: Some("x".into()),
            ..u(0.0)
        };
        // Reopened, second in its event, still before a lone first frame.
        let us = [u(0.0), reopened, u(0.1), repeats];
        let got = order(&us, &[0, 10, 5000, 9000], &[]);
        assert_eq!(
            got.iter().map(|x| x.0).collect::<Vec<_>>(),
            [1, 2, 0, 3],
            "{got:?}"
        );
        // Ahead of its event's other frame, it took rank 0.
        assert_eq!(got[0], (1, 0));
        assert_eq!(got[2], (0, 1));
    }

    fn need(score: u32) -> Visit {
        Visit {
            score,
            expanded: false,
        }
    }

    #[test]
    fn the_need_counts_in_bands_and_uncertainty_decides_within_one() {
        // 0 and 99 share a band: the more uncertain first. 100 is a band above both.
        let us = [u(0.4), u(0.0), u(0.45)];
        let got = order_of(&us, &[need(99), need(0), need(100)], &[0, 1000, 2000], &[]);
        assert_eq!(got.iter().map(|x| x.0).collect::<Vec<_>>(), [2, 1, 0]);
        assert_eq!(NEED_BAND, 100);
    }

    #[test]
    fn a_rare_second_frame_beats_a_common_first() {
        // Event A: a frame of need 3000 and its second frame of the same need; event B a common
        // first frame of need 300, more uncertain. The rank divides: 3000 / 2 = 1500 > 300.
        let us = [u(0.3), u(0.3), u(0.0)];
        let visits = [need(3000), need(3000), need(300)];
        let got = order_of(&us, &visits, &[0, 10, 1000], &[]);
        assert_eq!(got, [(0, 0), (1, 1), (2, 0)]);
        // Equal needs: the first frame of each event before any second.
        let got = order_of(&us, &[need(300); 3], &[0, 10, 1000], &[]);
        assert_eq!(got, [(2, 0), (0, 0), (1, 1)]);
    }

    #[test]
    fn a_rare_visit_expands() {
        // Two events of four frames each, one decided frame ahead in each; one visit expands.
        let us = vec![u(0.2); 6];
        let mut visits = vec![need(1000); 6];
        for v in &mut visits[..3] {
            v.expanded = true;
        }
        let got = order_of(&us, &visits, &[0, 10, 20, 1000, 1010, 1020], &[30, 1030]);
        // Expanded: 1000 * 4 / (4 + r) at r = 1, 2, 3 (800, 666, 571); else 1000 / (1 + r)
        // (500, 333, 250).
        assert_eq!(visit_score(1000, 1, true), 800);
        assert_eq!(visit_score(1000, 3, true), 571);
        assert_eq!(visit_score(1000, 1, false), 500);
        assert_eq!(visit_score(1000, 0, false), 1000);
        let ids: Vec<usize> = got.iter().map(|x| x.0).collect();
        assert_eq!(ids, [0, 1, 2, 3, 4, 5], "{got:?}");
        assert_eq!(got[2].1, 3);
        assert_eq!(got[5].1, 3);
    }

    #[test]
    fn the_order_is_total() {
        // Equal keys: only `tie` tells them apart, and it decides every pair.
        let us = vec![u(0.2); 6];
        let secs = [0, 0, 0, 1000, 1000, 5000];
        let got = order(&us, &secs, &[]);
        let ids: Vec<usize> = got.iter().map(|x| x.0).collect();
        assert_eq!(ids, [0, 3, 5, 1, 4, 2]);
        let ranks: Vec<u32> = got.iter().map(|x| x.1).collect();
        assert_eq!(ranks, [0, 0, 0, 1, 1, 2]);
        // With a tie that never says Equal for two indices, two calls agree whatever the input
        // order.
        let mut e = Events::default();
        let mut members: Vec<Member> = secs
            .iter()
            .enumerate()
            .rev()
            .map(|(i, s)| Member {
                listed: Some(i),
                ..at(*s)
            })
            .collect();
        e.add_day(&mut members);
        let keys: Vec<Key> = us.iter().map(|u| key(u, NEED)).collect();
        let visits = vec![NEED; us.len()];
        assert_eq!(arrange(&e, &keys, &visits, &|a, b| a.cmp(&b)), got);
    }

    #[test]
    fn the_pool_hash_is_pinned() {
        // FNV-1a 64 itself: of nothing, the offset basis; of "a", the published value.
        let fnv = |bytes: &[u8]| {
            bytes.iter().fold(FNV_OFFSET, |h, b| {
                (h ^ u64::from(*b)).wrapping_mul(FNV_PRIME)
            })
        };
        assert_eq!(fnv(b""), 14_695_981_039_346_656_037);
        assert_eq!(fnv(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(RANDOM_SALT, "bop-random-pick-v1/");
        assert_eq!(RANDOM_EVERY, 5);
        // A changed salt or hash would move every frame in or out of the pool: the frames already
        // marked would no longer be pool frames.
        let a = "day/drive/2026-10-21/20261021T010203004Z_a.jpg";
        assert_eq!(pool_hash(a), 3_014_258_616_504_270_761);
        assert_eq!(pool_hash(a), fnv(format!("{RANDOM_SALT}{a}").as_bytes()));
        assert_eq!(pool_hash_bytes(a.bytes()), pool_hash(a));
        assert!(!in_pool(a));
        let b = "day/drive/2026-10-21/20261021T010203004Z_b.jpg";
        assert_eq!(pool_hash(b), 8_372_720_720_226_647_124);
        // About one frame in five, by path alone.
        let n = 10_000;
        let pooled = (0..n)
            .filter(|i| in_pool(&format!("day/cam/2026-10-21/f{i}.jpg")))
            .count();
        assert!((1_800..=2_200).contains(&pooled), "{pooled}");
        // A reopened frame takes no random slot.
        let h = (0..)
            .map(|i| pool_hash(&format!("day/cam/2026-10-21/f{i}.jpg")))
            .find(|h| h.is_multiple_of(RANDOM_EVERY))
            .unwrap();
        assert!(is_pool_frame(h, false));
        assert!(!is_pool_frame(h, true));
        assert!(!is_pool_frame(h + 1, false));
    }

    /// The served list as text: `r<i>` for rest, `P<i>` for pool.
    fn served(rest: usize, pool: usize, decided: u64) -> Vec<String> {
        let r: Vec<usize> = (0..rest).collect();
        let p: Vec<usize> = (100..100 + pool).collect();
        interleave(&r, &p, decided)
            .into_iter()
            .map(|(i, random)| {
                assert_eq!(random, i >= 100);
                if random {
                    format!("P{}", i - 100)
                } else {
                    format!("r{i}")
                }
            })
            .collect()
    }

    #[test]
    fn every_fifth_slot_is_random_from_the_decided_count() {
        let s = |v: &[&str]| v.iter().map(|x| (*x).to_string()).collect::<Vec<_>>();
        assert_eq!(
            served(8, 3, 0),
            s(&[
                "r0", "r1", "r2", "r3", "P0", "r4", "r5", "r6", "r7", "P1", "P2"
            ])
        );
        assert_eq!(served(4, 2, 4), s(&["P0", "r0", "r1", "r2", "r3", "P1"]));
        assert_eq!(served(4, 2, 2), s(&["r0", "r1", "P0", "r2", "r3", "P1"]));
        // Only the phase counts: 7 decided is 2 decided.
        assert_eq!(served(4, 2, 7), served(4, 2, 2));
        assert_eq!(served(4, 2, u64::MAX), served(4, 2, u64::MAX % 5));
        for decided in 0..20u64 {
            let got = served(40, 40, decided);
            assert_eq!(got.len(), 80);
            for (i, x) in got.iter().take(45).enumerate() {
                let random = (decided + i as u64) % RANDOM_EVERY == RANDOM_EVERY - 1;
                assert_eq!(x.starts_with('P'), random, "{decided}: slot {i} of {got:?}");
            }
        }
    }

    #[test]
    fn when_one_side_runs_out_the_other_fills() {
        let s = |v: &[&str]| v.iter().map(|x| (*x).to_string()).collect::<Vec<_>>();
        // No pool frame left: the random slots take the rest.
        assert_eq!(served(3, 0, 4), s(&["r0", "r1", "r2"]));
        assert_eq!(
            served(7, 1, 0),
            s(&["r0", "r1", "r2", "r3", "P0", "r4", "r5", "r6"])
        );
        // Nothing but pool frames: in hash order, all random picks.
        assert_eq!(served(0, 3, 0), s(&["P0", "P1", "P2"]));
        assert_eq!(served(1, 4, 0), s(&["r0", "P0", "P1", "P2", "P3"]));
        assert!(served(0, 0, 3).is_empty());
    }

    #[test]
    fn the_frames_ahead_keep_their_slots_across_a_save_or_a_skip() {
        // A save or a skip of the frame shown (the first) takes it out and adds one decided
        // frame: what the page loaded ahead is still what comes, whichever side it was.
        for decided in 0..10u64 {
            let rest: Vec<usize> = (0..12).collect();
            let pool: Vec<usize> = (100..106).collect();
            let before = interleave(&rest, &pool, decided);
            let (first, _) = before[0];
            let rest2: Vec<usize> = rest.iter().copied().filter(|i| *i != first).collect();
            let pool2: Vec<usize> = pool.iter().copied().filter(|i| *i != first).collect();
            assert_eq!(
                interleave(&rest2, &pool2, decided + 1),
                before[1..],
                "decided {decided}"
            );
        }
        // Next without a save changes nothing: the same list, one place on.
        assert_eq!(served(5, 5, 3), served(5, 5, 3));
    }

    #[test]
    fn random_slots_hold_pool_frames_and_the_rest_keeps_its_order() {
        // Six listed frames in arrange's order 5, 3, 1, 0, 2, 4 (ranks = index); 1 and 4 are pool
        // frames, 4 of the smaller hash.
        let order: Vec<(usize, u32)> = [5, 3, 1, 0, 2, 4]
            .iter()
            .map(|&i| (i, u32::try_from(i).unwrap()))
            .collect();
        let hash = |i: usize| match i {
            1 => Some(10),
            4 => Some(5),
            _ => None,
        };
        let got = with_random_slots(&order, &hash, &|a, b| a.cmp(&b), 3);
        assert_eq!(
            got,
            [
                (5, 5, false),
                (4, 4, true),
                (3, 3, false),
                (0, 0, false),
                (2, 2, false),
                (1, 1, true),
            ]
        );
        // Pool and rest are disjoint, and every frame comes once.
        let mut seen: Vec<usize> = got.iter().map(|x| x.0).collect();
        seen.sort_unstable();
        assert_eq!(seen, [0, 1, 2, 3, 4, 5]);
        // A hash tie: the path decides.
        let tie = |i: usize| (i == 1 || i == 4).then_some(5);
        let got = with_random_slots(&order, &tie, &|a, b| b.cmp(&a), 0);
        let pool: Vec<usize> = got.iter().filter(|x| x.2).map(|x| x.0).collect();
        assert_eq!(pool, [4, 1]);
    }

    #[test]
    fn only_the_default_view_has_random_slots() {
        let default = Filters {
            order: Order::Recommended,
            ..Filters::default()
        };
        assert!(random_slots(&default));
        // The source is the population: allowed.
        assert!(random_slots(&Filters {
            source: super::super::queue::QueueSource::All,
            ..default.clone()
        }));
        let not = [
            Filters {
                order: Order::Uncertain,
                ..default.clone()
            },
            Filters {
                status: StatusFilter::Any,
                ..default.clone()
            },
            Filters {
                status: StatusFilter::Is(Status::Skipped),
                ..default.clone()
            },
            Filters {
                camera: Some("gate".into()),
                ..default.clone()
            },
            Filters {
                scene: Some("night".into()),
                ..default.clone()
            },
            Filters {
                label: ["person".to_string()].into(),
                ..default.clone()
            },
            Filters {
                bucket: Some(crate::collect::Bucket::Positive),
                ..default.clone()
            },
            Filters {
                auto: Some(super::super::queue::AutoFilter::Review),
                ..default.clone()
            },
            Filters {
                balance: super::super::queue::Balance::parse("class").unwrap(),
                ..default.clone()
            },
        ];
        for f in not {
            assert!(!random_slots(&f), "{f:?}");
        }
        // The class list the route adds is no filter.
        assert!(random_slots(&Filters {
            wanted: Some((vec!["person".into()], vec![])),
            ..default
        }));
    }
}
