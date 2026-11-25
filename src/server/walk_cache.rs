//! The kept answers of the Refine tab's label-file walks (QA-132): one [`WalkCache`] per card
//! (open suggestion counts, static spots, size floor, coverage), each a process-wide `static`.
//!
//! A walk reads every label file of the folder under review, so its answer is kept per key
//! ([`wanted_key`]: the folder and both class lists, plus what else the walk depends on) for
//! `fresh_ttl`, and only while the `stamp` its caller names (the checks on the queue) is the
//! same. One walk runs at a time: a second caller of the same key waits for the first and takes
//! its answer. [`WalkCache::forget`] drops every answer and bumps a generation under the map's
//! lock, so a walk that began before the drop answers its caller and keeps nothing (it may have
//! read a file before the change that caused the drop). At most `keep` answers stay: answers past
//! `evict_ttl` go first, then the oldest.

use crate::sync::LockExt;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// One kept answer.
pub(super) struct Kept<T> {
    /// When the walk that found it ended.
    pub(super) at: Instant,
    /// What the caller named the state outside the files with (the queue's checks).
    pub(super) stamp: String,
    pub(super) value: T,
}

/// The kept answers of one kind of walk (see the module doc). The fields are visible to the
/// caches' own tests, which age and count what is kept.
pub(super) struct WalkCache<T> {
    /// The kept answers, by key.
    pub(super) kept: Mutex<BTreeMap<String, Kept<T>>>,
    /// One walk at a time: a second caller waits and finds the first one's answer.
    walking: Mutex<()>,
    /// Bumped by [`Self::forget`] under `kept`'s lock: a walk that began before a drop answers
    /// its caller and keeps nothing.
    pub(super) generation: AtomicU64,
    /// Most answers kept at once.
    keep: usize,
    /// How long an answer answers [`Self::get_or_walk`].
    fresh_ttl: Duration,
    /// How long an answer stays at all (for [`Self::peek`]); at least `fresh_ttl`.
    evict_ttl: Duration,
}

impl<T: Clone> WalkCache<T> {
    pub(super) const fn new(keep: usize, fresh_ttl: Duration, evict_ttl: Duration) -> Self {
        Self {
            kept: Mutex::new(BTreeMap::new()),
            walking: Mutex::new(()),
            generation: AtomicU64::new(0),
            keep,
            fresh_ttl,
            evict_ttl,
        }
    }

    /// The answer kept under `key` for `stamp` if it is under `fresh_ttl` old, else `walk`'s,
    /// kept unless [`Self::forget`] ran meanwhile.
    pub(super) fn get_or_walk(&self, key: String, stamp: &str, walk: impl FnOnce() -> T) -> T {
        let fresh = |k: &Kept<T>| k.at.elapsed() < self.fresh_ttl && k.stamp == stamp;
        if let Some(k) = self.kept.lock_recover().get(&key).filter(|k| fresh(k)) {
            return k.value.clone();
        }
        let _walking = self.walking.lock_recover();
        if let Some(k) = self.kept.lock_recover().get(&key).filter(|k| fresh(k)) {
            return k.value.clone();
        }
        let generation = self.generation.load(Ordering::SeqCst);
        let value = walk();
        let mut map = self.kept.lock_recover();
        // A drop since the walk began: what it found may be the files before a change.
        if self.generation.load(Ordering::SeqCst) == generation {
            self.remember(
                &mut map,
                key,
                Kept {
                    at: Instant::now(),
                    stamp: stamp.to_string(),
                    value: value.clone(),
                },
            );
        }
        value
    }

    /// The answer kept under `key` for `stamp`, whatever its age, with its age; never waits for
    /// a walk.
    pub(super) fn peek(&self, key: &str, stamp: &str) -> Option<(T, Duration)> {
        self.kept
            .lock_recover()
            .get(key)
            .filter(|k| k.stamp == stamp)
            .map(|k| (k.value.clone(), k.at.elapsed()))
    }

    /// Drop every kept answer.
    pub(super) fn forget(&self) {
        // Under the map's lock: a walk storing its answer checks the generation under it too.
        let mut kept = self.kept.lock_recover();
        self.generation.fetch_add(1, Ordering::SeqCst);
        kept.clear();
    }

    /// Put `kept` in `map` under `key`: answers past `evict_ttl` go first, then the oldest, so at
    /// most `keep` stay (each folder under review and class list ever asked about would otherwise
    /// stay for the life of the process).
    fn remember(&self, map: &mut BTreeMap<String, Kept<T>>, key: String, kept: Kept<T>) {
        map.retain(|k, v| *k != key && v.at.elapsed() < self.evict_ttl);
        while map.len() >= self.keep {
            let Some(oldest) = map.iter().min_by_key(|(_, v)| v.at).map(|(k, _)| k.clone()) else {
                break;
            };
            map.remove(&oldest);
        }
        map.insert(key, kept);
    }
}

/// The key of a walk of the folder `dir` for the class lists `classes` and `hidden`: the folder,
/// then `\u{1f}`-separated lists joined by `\u{1e}` (the folder first, so the keys of one folder
/// share the prefix `"<dir>\u{1f}"`). A walk that depends on more appends `\u{1f}` and its parts.
pub(super) fn wanted_key(dir: &Path, classes: &[String], hidden: &[String]) -> String {
    format!(
        "{}\u{1f}{}\u{1f}{}",
        dir.display(),
        classes.join("\u{1e}"),
        hidden.join("\u{1e}")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    const TTL: Duration = Duration::from_secs(60);

    /// A cache with a walk counter: `walk(n)` answers `n` and counts.
    struct Counted {
        cache: WalkCache<u32>,
        walks: AtomicUsize,
    }

    impl Counted {
        fn new(keep: usize, evict: Duration) -> Self {
            Self {
                cache: WalkCache::new(keep, TTL, evict),
                walks: AtomicUsize::new(0),
            }
        }

        fn get(&self, key: &str, stamp: &str, n: u32) -> u32 {
            self.cache.get_or_walk(key.into(), stamp, || {
                self.walks.fetch_add(1, Ordering::SeqCst);
                n
            })
        }

        fn walks(&self) -> usize {
            self.walks.load(Ordering::SeqCst)
        }

        /// As if every kept answer were `ago` old.
        fn age(&self, ago: Duration) {
            for k in self.cache.kept.lock_recover().values_mut() {
                k.at = Instant::now() - ago;
            }
        }
    }

    #[test]
    fn an_answer_is_kept_per_key_and_stamp_until_forgotten() {
        let c = Counted::new(4, TTL);
        assert_eq!(c.get("a", "", 1), 1);
        assert_eq!(c.get("a", "", 2), 1, "kept");
        assert_eq!(c.walks(), 1);
        assert_eq!(c.get("b", "", 3), 3, "another key");
        assert_eq!(c.get("a", "t1:running", 4), 4, "another stamp");
        assert_eq!(c.get("a", "t1:running", 5), 4);
        assert_eq!(c.walks(), 3);
        c.cache.forget();
        assert!(c.cache.kept.lock_recover().is_empty());
        assert_eq!(c.cache.generation.load(Ordering::SeqCst), 1);
        assert_eq!(c.get("a", "t1:running", 6), 6, "dropped");
        assert_eq!(c.walks(), 4);
    }

    #[test]
    fn past_the_fresh_age_a_walk_runs_and_peek_still_answers() {
        let c = Counted::new(4, TTL * 10);
        assert_eq!(c.cache.peek("a", ""), None, "nothing before the first walk");
        c.get("a", "", 1);
        c.age(TTL);
        let (value, age) = c.cache.peek("a", "").unwrap();
        assert_eq!(value, 1);
        assert!(age >= TTL);
        assert_eq!(c.cache.peek("a", "t1:running"), None, "another stamp");
        assert_eq!(c.get("a", "", 2), 2, "stale: walked again");
        assert_eq!(c.walks(), 2);
        let (value, age) = c.cache.peek("a", "").unwrap();
        assert_eq!(value, 2);
        assert!(age < TTL);
    }

    #[test]
    fn a_walk_from_before_a_forget_is_not_kept() {
        let cache: WalkCache<u32> = WalkCache::new(4, TTL, TTL);
        let got = cache.get_or_walk("a".into(), "", || {
            cache.forget();
            5
        });
        assert_eq!(got, 5);
        assert!(cache.kept.lock_recover().is_empty(), "not kept");
        assert_eq!(cache.get_or_walk("a".into(), "", || 7), 7);
        assert_eq!(cache.get_or_walk("a".into(), "", || 8), 7, "kept");
    }

    /// The eviction rule of [`WalkCache::remember`]: an expired answer goes at the next insert
    /// whatever the room, then the oldest, and the same key replaces its own answer.
    #[test]
    fn at_most_keep_answers_stay_and_the_oldest_go_first() {
        const KEEP: usize = 8;
        let evict = Duration::from_secs(600);
        let cache: WalkCache<u32> = WalkCache::new(KEEP, TTL, evict);
        let mut map = BTreeMap::new();
        let base = Instant::now();
        let kept = |secs: u64| Kept {
            at: base - evict + Duration::from_secs(secs),
            stamp: String::new(),
            value: 0,
        };
        map.insert(
            "expired".to_string(),
            Kept {
                at: base - evict - Duration::from_secs(1),
                ..kept(0)
            },
        );
        for i in 0..(KEEP as u64 + 5) {
            cache.remember(&mut map, format!("k{i:02}"), kept(30 + i));
            assert!(map.len() <= KEEP, "{}", map.len());
        }
        assert!(!map.contains_key("expired"));
        let left: Vec<&String> = map.keys().collect();
        let want: Vec<String> = (5..(KEEP as u64 + 5)).map(|i| format!("k{i:02}")).collect();
        assert_eq!(left, want.iter().collect::<Vec<_>>(), "the newest stay");
        cache.remember(&mut map, "k12".into(), kept(59));
        assert_eq!(map.len(), KEEP);
        assert!(map.contains_key("k05"));
    }

    #[test]
    fn get_or_walk_keeps_at_most_keep() {
        let c = Counted::new(3, TTL);
        for i in 0..10 {
            c.get(&format!("k{i}"), "", i);
        }
        assert_eq!(c.cache.kept.lock_recover().len(), 3);
        assert_eq!(c.get("k9", "", 0), 9, "the newest stays");
    }

    /// A second caller of a key being walked waits and takes the first walk's answer.
    #[test]
    fn one_walk_at_a_time_and_a_waiting_caller_takes_its_answer() {
        let c = Arc::new(Counted::new(4, TTL));
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let first = {
            let c = Arc::clone(&c);
            std::thread::spawn(move || {
                c.cache.get_or_walk("a".into(), "", || {
                    started_tx.send(()).unwrap();
                    go_rx.recv().unwrap();
                    1
                })
            })
        };
        started_rx.recv().unwrap();
        let second = {
            let c = Arc::clone(&c);
            std::thread::spawn(move || c.get("a", "", 2))
        };
        go_tx.send(()).unwrap();
        assert_eq!(first.join().unwrap(), 1);
        assert_eq!(second.join().unwrap(), 1, "the first walk's answer");
        assert_eq!(c.walks(), 0, "the second caller never walked");
    }

    #[test]
    fn the_key_names_the_folder_first_then_both_lists() {
        let dir = Path::new("data").join("review");
        let key = wanted_key(
            &dir,
            &["car".to_string(), "person".to_string()],
            &["cat".to_string()],
        );
        assert_eq!(
            key,
            format!("{}\u{1f}car\u{1e}person\u{1f}cat", dir.display())
        );
        assert!(key.starts_with(&format!("{}\u{1f}", dir.display())));
        assert_eq!(
            wanted_key(&dir, &[], &[]),
            format!("{}\u{1f}\u{1f}", dir.display())
        );
    }
}
