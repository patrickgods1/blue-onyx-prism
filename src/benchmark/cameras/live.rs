//! The reviewed frames of "Your cameras" as they are now, for the pages: the Run tab's line
//! ([`super::Survey`]) and `GET /v1/benchmark`'s `labelsChanged` ([`super::snapshot::changes`]).
//! Both are polled, and a walk of a large review folder takes seconds, so [`peek`] never walks:
//! it answers the last walk and starts one on its own thread when that one is stale. Stale: of
//! another folder or class list, else (with the review index built, [`index::rows_version`]) when
//! the index's rows changed or [`MAX_AGE`] passed, else (no index) after [`FRESH`]. One walk at a
//! time; the values go after [`crate::benchmark::search::CACHE_IDLE`] without a reader. A few
//! sources are kept at once ([`KEPT`]: a class list edited back and forth, tests side by side).

use super::{Load, Source, Survey};
use crate::benchmark::search::{CACHE_IDLE, IdleCache};
use crate::collect::labels::index;
use crate::sync::LockExt;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// A walk this old is redone (no index to tell what changed). Labels change while someone reviews,
/// and the Review tab builds the index then; without it a walk of a large folder every few
/// minutes is all a polling page costs.
pub const FRESH: Duration = Duration::from_secs(5 * 60);
/// A walk this old is redone even when the index's rows did not change (another process's
/// edits reach the index within its own refresh, so this is only a backstop).
pub const MAX_AGE: Duration = Duration::from_secs(10 * 60);

/// What a walk found.
#[derive(Debug, Clone, PartialEq)]
pub enum Found {
    Ready {
        survey: Survey,
        /// File name -> label revision of every reviewed frame.
        reviewed: BTreeMap<String, String>,
    },
    /// The folder under review cannot be read: why.
    Missing(String),
    /// No reviewed frame.
    Empty,
}

/// One walk's result.
#[derive(Debug, Clone)]
pub struct Current {
    pub found: Found,
    /// When the walk finished.
    pub at: Instant,
    /// The index rows' stamp before the walk (None without a built index).
    pub rows: Option<u64>,
}

/// Most sources kept; the oldest walk goes first.
pub const KEPT: usize = 8;

/// The walks by source key ([`Source::key`]).
type Walks = Arc<Mutex<HashMap<String, Arc<Current>>>>;

static CACHE: IdleCache<(), Walks> = IdleCache::new(CACHE_IDLE);
static WALKING: AtomicBool = AtomicBool::new(false);

/// The walks, now counted as read.
fn walks() -> Walks {
    CACHE.get_or_insert_with((), Walks::default)
}

fn wanted(src: &Source) -> (Vec<String>, Vec<String>) {
    (src.classes.clone(), src.hidden.clone())
}

/// Walk `src` now (on the calling thread) and cache the result.
pub fn walk(src: &Source) -> Arc<Current> {
    let rows = index::rows_version(&src.dir, Some(&wanted(src)));
    let found = match super::load(src) {
        Load::Ready(l) => Found::Ready {
            survey: l.survey,
            reviewed: l.reviewed,
        },
        Load::Missing(why) => Found::Missing(why),
        Load::Empty(_) => Found::Empty,
    };
    let current = Arc::new(Current {
        found,
        at: Instant::now(),
        rows,
    });
    let all = walks();
    let mut map = all.lock_recover();
    map.insert(src.key(), current.clone());
    evict(&mut map);
    drop(map);
    current
}

/// Drop the oldest walks (by [`Current::at`]) until at most [`KEPT`] are left.
fn evict(map: &mut HashMap<String, Arc<Current>>) {
    while map.len() > KEPT {
        let Some(oldest) = map.iter().min_by_key(|(_, c)| c.at).map(|(k, _)| k.clone()) else {
            break;
        };
        map.remove(&oldest);
    }
}

/// Whether `c` (of `src`) should be walked again ([`stale_by`]).
fn stale(src: &Source, c: &Current) -> bool {
    let age = c.at.elapsed();
    stale_by(
        index::rows_version(&src.dir, Some(&wanted(src))),
        c.rows,
        age,
    )
}

/// Whether a walk `age` old should be redone: with the index's row stamp now (`now_rows`) and
/// before that walk (`then_rows`), when they differ or after [`MAX_AGE`]; else after [`FRESH`].
fn stale_by(now_rows: Option<u64>, then_rows: Option<u64>, age: Duration) -> bool {
    match (now_rows, then_rows) {
        (Some(now), Some(then)) => now != then || age >= MAX_AGE,
        _ => age >= FRESH,
    }
}

/// The last walk of `src` (None before the first finished), starting a new one in the
/// background when it is stale. Never waits for a walk. A folder that cannot be read now answers
/// that at once.
pub fn peek(src: &Source) -> Option<Arc<Current>> {
    if let Some(why) = &src.problem {
        return Some(Arc::new(Current {
            found: Found::Missing(why.clone()),
            at: Instant::now(),
            rows: None,
        }));
    }
    let cached = walks().lock_recover().get(&src.key()).cloned();
    if cached.as_ref().is_none_or(|c| stale(src, c)) {
        start(src.clone());
    }
    cached
}

/// Start a background walk of `src` unless one runs.
fn start(src: Source) {
    if WALKING.swap(true, Ordering::AcqRel) {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("bop-bench-cameras".into())
        .spawn(move || {
            // Cleared however the walk ends.
            struct Done;
            impl Drop for Done {
                fn drop(&mut self) {
                    WALKING.store(false, Ordering::Release);
                }
            }
            let _done = Done;
            walk(&src);
        });
    if let Err(e) = spawned {
        WALKING.store(false, Ordering::Release);
        tracing::warn!("could not start the walk of {}: {e}", super::TITLE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN: Duration = Duration::from_secs(60);

    /// QA-190: staleness by the index's row stamp, else by age.
    #[test]
    fn staleness_by_rows_then_age() {
        // Index rows unchanged: fresh until MAX_AGE.
        assert!(!stale_by(Some(1), Some(1), 9 * MIN));
        assert!(stale_by(Some(1), Some(1), 10 * MIN));
        // Rows changed: stale at once.
        assert!(stale_by(Some(2), Some(1), Duration::ZERO));
        // No index now: FRESH.
        assert!(!stale_by(None, Some(1), 4 * MIN));
        assert!(stale_by(None, Some(1), 5 * MIN));
        assert!(!stale_by(None, None, 4 * MIN));
        assert!(stale_by(None, None, 5 * MIN));
        // An index built since the walk: FRESH too (not "changed").
        assert!(!stale_by(Some(1), None, 4 * MIN));
        assert!(stale_by(Some(1), None, 5 * MIN));
    }

    /// QA-190: past KEPT sources the oldest walks go.
    #[test]
    fn eviction_keeps_the_newest() {
        let base = Instant::now();
        let mut map: HashMap<String, Arc<Current>> = HashMap::new();
        for k in 0..=KEPT as u64 {
            let at = base + Duration::from_secs(k);
            map.insert(
                format!("src-{k}"),
                Arc::new(Current {
                    found: Found::Empty,
                    at,
                    rows: None,
                }),
            );
        }
        assert_eq!(map.len(), KEPT + 1);
        evict(&mut map);
        assert_eq!(map.len(), KEPT);
        assert!(!map.contains_key("src-0"), "the oldest goes");
        assert!((1..=KEPT as u64).all(|k| map.contains_key(&format!("src-{k}"))));
        // At or under KEPT nothing goes.
        evict(&mut map);
        assert_eq!(map.len(), KEPT);
    }
}
