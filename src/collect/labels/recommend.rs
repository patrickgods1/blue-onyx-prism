//! The Recommended review order, the default of `GET /v1/review/next` (no `auto=`, no `balance=`;
//! [`super::queue::Order::Recommended`]): the first frame of each event comes before any second
//! one, then the frames of the classes, scenes and cameras with the fewest reviewed frames
//! (balance score in bands of [`BALANCE_BAND`], so uncertainty decides within a band), then the
//! uncertainty keys of [`super::queue::Uncertainty::key`]. Reopened frames (a re-label
//! disagreement) come first and repeat-only frames last, as in the uncertainty order.
//!
//! An event is one camera's activity: the frames of one day folder (`<scene>/<camera>/<date>`, so
//! an event ends at UTC midnight and at a scene change) whose times are at most [`EVENT_GAP_S`]
//! apart, spanning at most [`EVENT_MAX_S`] from its first frame. A frame with no readable time,
//! and an imported frame ([`IMPORT_MODEL`]: an image-folder import carries file times, and its
//! frames were already spread out when chosen), is an event of its own. Events are built over the
//! frames the queue counts (every filter but status and auto), not those it lists, so an event
//! does not split as its frames are reviewed: a frame's event rank is the decided frames of its
//! event that the queue does not list, plus the listed frames of its event ahead of it.
//!
//! Pure: both the walk ([`super::queue::build`]) and the review index call [`Events::add_day`]
//! and [`arrange`] on what they read, so their orders are one. [`arrange`] returns the order as
//! indices, for later slots on top of it.

use super::queue::Uncertainty;
use std::cmp::{Ordering, Reverse};

/// Most seconds between two frames of one event.
pub const EVENT_GAP_S: i64 = 60;
/// Most seconds from an event's first frame to its last.
pub const EVENT_MAX_S: i64 = 600;
/// Balance scores ([`super::queue::BalanceNeed::score`]) in one band compare equal: uncertainty
/// orders the frames of a band.
pub const BALANCE_BAND: u32 = 250;
/// The sidecar `model` of an imported frame (`training::import`): an event of its own.
pub const IMPORT_MODEL: &str = "import";

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
    /// Imported ([`IMPORT_MODEL`]): an event of its own.
    pub import: bool,
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

impl Events {
    /// Add the events of one day folder's counted frames (reordered here).
    pub fn add_day(&mut self, members: &mut [Member]) {
        // Frames that chain first, by time; the ones alone after them, in any order.
        members.sort_unstable_by_key(|m| (m.import || m.ms.is_none(), m.ms));
        // (first, last) time of the open event.
        let mut open: Option<(i64, i64)> = None;
        for m in members.iter() {
            let ms = m.ms.filter(|_| !m.import);
            let joins = match (ms, open) {
                (Some(t), Some((first, last))) => {
                    t.saturating_sub(last) <= GAP_MS && t.saturating_sub(first) <= MAX_MS
                }
                _ => false,
            };
            if joins {
                open = open.map(|(first, _)| (first, ms.unwrap_or(first)));
            } else {
                self.decided.push(0);
                open = ms.map(|t| (t, t));
            }
            let event = self.decided.len().saturating_sub(1);
            match m.listed {
                Some(i) => self.listed.push((event, i)),
                None if m.decided => {
                    if let Some(n) = self.decided.last_mut() {
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
/// event rank, the balance band (highest first), the likely boxes (most first), the flagged boxes
/// (fewest first), the margin in hundredths (no box last), verifier disagreement, suppressed boxes
/// (most first).
pub type Key = (
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

/// The key of a frame of uncertainty `u` (its `balance` the Recommended one, class, scene and
/// camera) at event rank `rank`.
pub fn key(u: &Uncertainty, rank: u32) -> Key {
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

/// `k` with the event rank `rank`.
fn ranked(mut k: Key, rank: u32) -> Key {
    k.2 = rank;
    k
}

/// The Recommended order of the listed frames: `keys` are their [`key`]s at rank 0, `tie` orders
/// two of them that have the same key (newest first, then by path: no two are equal). Each
/// listed frame's rank is its event's decided frames plus the listed frames of its event ahead of
/// it by the key without the rank, so the rank is well defined. The frames' indices in order,
/// each with its event rank.
pub fn arrange(
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
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        ranked(base(a), rank(a))
            .cmp(&ranked(base(b), rank(b)))
            .then_with(|| tie(a, b))
    });
    order.into_iter().map(|i| (i, rank(i))).collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::queue::BalanceNeed;
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
            import: false,
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
            import: true,
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

    fn u(margin: f32) -> Uncertainty {
        Uncertainty {
            margin: Some(margin),
            ..Uncertainty::default()
        }
    }

    /// `arrange` over one day of listed frames `us` (times `secs`) with `decided` more decided
    /// frames at `decided_at`: the listed indices in order, with ranks.
    fn order(us: &[Uncertainty], secs: &[i64], decided_at: &[i64]) -> Vec<(usize, u32)> {
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
        let keys: Vec<Key> = us.iter().map(|u| key(u, 0)).collect();
        arrange(&e, &keys, &|a, b| a.cmp(&b))
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

    fn balanced(score: u32, margin: f32) -> Uncertainty {
        Uncertainty {
            balance: Some(BalanceNeed {
                score,
                ..BalanceNeed::default()
            }),
            ..u(margin)
        }
    }

    #[test]
    fn balance_counts_in_bands_and_uncertainty_decides_within_one() {
        // 0 and 249 share a band: the more uncertain first. 250 is a band above both.
        let us = [balanced(249, 0.4), balanced(0, 0.0), balanced(250, 0.45)];
        let got = order(&us, &[0, 1000, 2000], &[]);
        assert_eq!(got.iter().map(|x| x.0).collect::<Vec<_>>(), [2, 1, 0]);
        assert_eq!(BALANCE_BAND, 250);
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
        let keys: Vec<Key> = us.iter().map(|u| key(u, 0)).collect();
        assert_eq!(arrange(&e, &keys, &|a, b| a.cmp(&b)), got);
    }
}
