//! What steers the Review queue's Recommended order from outside its rows ([`snapshot`], one
//! [`Steer`] per `GET /v1/review/next`, handed to the review index and the walk alike): the
//! measured recall of the newest trained models ([`recall`]) and the folder's parked spots
//! ([`spots`]). Nothing here runs a walk on the request path, and nothing here writes a file.
//!
//! Recall. A [`Measured`] from `runs.json` through [`crate::training::bench::measured_recall`],
//! one cached entry keyed on `runs.json`'s (path, modification time, length): a request costs one
//! `stat` while nothing changed, and the table is rebuilt only after the training queue saved a run
//! (it rewrites the whole file). Records come from the started queue, else from the file through
//! the pure [`crate::training::task::read_records`] ([`super::super::training::records_read_only`]):
//! a damaged `runs.json` is no measured run here, and is never renamed, moved aside or written (the
//! queue deals with it when it starts). A missing file, or one without measured runs, is None,
//! logged once per change of the key; never an error to the request.
//!
//! Spots. The static spots of the folder's reviewed frames, built on one background thread at a
//! time ([`SpotCache`]): from the review index's reviewed rows when the folder has a built index
//! ([`crate::collect::labels::index::reviewed_obs`], no label file read), rebuilt when those rows
//! changed and the last build is [`SPOTS_INDEXED_EVERY`] old; else from the Refine tab's cached
//! table ([`super::super::refine_spots::cached_table`]: a small folder, walked at most every
//! [`SPOTS_EVERY`], or not at all when the tab walked it lately). A request serves the last table
//! built for the folder and class list, empty (so neutral: no box counts as parked) until the
//! first is in; so the order may shift a little when a new table lands. A build asked for while
//! one runs is remembered (the latest) and run when it ends; a build that panics leaves the cache
//! able to build again (a guard). In tests the table is built on the request's thread, so an answer
//! never depends on a thread's timing.

use crate::collect::labels::need::Steer;
use crate::collect::labels::recall::Measured;
use crate::collect::labels::statics;
use crate::server::AppState;
use crate::sync::LockExt;
use crate::training::service::RunRecord;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

/// Least time between two builds of a folder's spot table without an index.
pub const SPOTS_EVERY: Duration = Duration::from_secs(10 * 60);
/// Least time between two builds from the index's rows, while they change.
pub const SPOTS_INDEXED_EVERY: Duration = Duration::from_secs(60);

/// What steers the Recommended order of the folder under review `dir` for the class list
/// (`classes`, `hidden`) now: one snapshot for the whole request.
pub(super) fn snapshot(
    state: &AppState,
    dir: &Path,
    classes: &[String],
    hidden: &[String],
) -> Arc<Steer> {
    Arc::new(Steer {
        recall: recall(state),
        spots: spots(state, dir, classes, hidden),
        heldout_listed: Arc::default(),
    })
}

/// A spot table to build: for which folder and class list, from which rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Job {
    /// The folder and class list.
    pub key: String,
    /// The index rows' stamp (None: no built index, Refine's table).
    pub stamp: Option<u64>,
    pub dir: PathBuf,
    pub classes: Vec<String>,
    pub hidden: Vec<String>,
    /// The checks on the queue (Refine's cache key).
    pub checks: String,
}

/// The last table built and what was asked meanwhile.
#[derive(Debug, Default)]
struct SpotSlot {
    key: String,
    stamp: Option<u64>,
    /// When the table was built.
    at: Option<Instant>,
    table: Arc<statics::Table>,
    building: bool,
    /// The latest build asked for while one ran.
    queued: Option<Job>,
}

/// The spot tables of the process (see the module docs): one build at a time, the latest asked
/// for run next, a panicking build leaving it able to build again.
#[derive(Debug)]
pub(super) struct SpotCache {
    slot: Mutex<Option<SpotSlot>>,
}

/// Clears `building` unless the build finished: a panic in a build must not stop every later one.
struct Building<'a> {
    cache: &'a SpotCache,
    done: bool,
}

impl Drop for Building<'_> {
    fn drop(&mut self) {
        if !self.done
            && let Some(s) = self.cache.slot.lock_recover().as_mut()
        {
            s.building = false;
        }
    }
}

impl SpotCache {
    pub(super) const fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }

    /// The table to serve for `job` now, and the job to build (the caller runs it with
    /// [`Self::run`], on a thread or inline) when one is due: none built for the key yet, or the
    /// rows changed and the last build is `every` old. While a build runs, `job` is remembered
    /// for after it.
    pub(super) fn get(&self, job: Job, every: Duration) -> (Arc<statics::Table>, Option<Job>) {
        let mut guard = self.slot.lock_recover();
        let slot = guard.get_or_insert_with(SpotSlot::default);
        let same = slot.key == job.key && slot.at.is_some();
        let table = if same {
            slot.table.clone()
        } else {
            Arc::default()
        };
        let due =
            !same || (slot.stamp != job.stamp && slot.at.is_none_or(|t| t.elapsed() >= every));
        if !due {
            return (table, None);
        }
        if slot.building {
            slot.queued = Some(job);
            return (table, None);
        }
        slot.building = true;
        (table, Some(job))
    }

    /// Build `job` with `build`, then the latest job asked for meanwhile, until none is left.
    pub(super) fn run(&self, mut job: Job, build: &dyn Fn(&Job) -> statics::Table) {
        loop {
            let mut guard = Building {
                cache: self,
                done: false,
            };
            let table = Arc::new(build(&job));
            let next = {
                let mut s = self.slot.lock_recover();
                let slot = s.get_or_insert_with(SpotSlot::default);
                slot.key.clone_from(&job.key);
                slot.stamp = job.stamp;
                slot.at = Some(Instant::now());
                slot.table = table;
                let next = slot.queued.take().filter(|q| *q != job);
                if next.is_none() {
                    slot.building = false;
                }
                next
            };
            guard.done = true;
            drop(guard);
            match next {
                Some(n) => job = n,
                None => return,
            }
        }
    }

    /// The table last built for `key`, if any.
    #[cfg(test)]
    fn table_of(&self, key: &str) -> Option<Arc<statics::Table>> {
        self.slot
            .lock_recover()
            .as_ref()
            .filter(|s| s.key == key && s.at.is_some())
            .map(|s| s.table.clone())
    }
}

/// The spot table of `job`: from the index's rows when it has some, else Refine's.
fn build(job: &Job) -> statics::Table {
    let wanted = (job.classes.clone(), job.hidden.clone());
    match crate::collect::labels::index::reviewed_obs(&job.dir, Some(&wanted)) {
        Some((_, obs)) => statics::Table::build(obs),
        None => super::super::refine_spots::cached_table(
            &job.dir,
            &job.classes,
            &job.hidden,
            &job.checks,
        ),
    }
}

/// The parked spots of `dir`'s reviewed frames for the class list, built on the request's thread
/// in tests (see the module docs).
#[cfg(test)]
pub(super) fn spots(
    _state: &AppState,
    dir: &Path,
    classes: &[String],
    hidden: &[String],
) -> Arc<statics::Table> {
    Arc::new(super::super::refine_spots::spot_table(dir, classes, hidden))
}

#[cfg(not(test))]
static SPOTS: SpotCache = SpotCache::new();

/// The parked spots of `dir`'s reviewed frames for the class list (see the module docs).
#[cfg(not(test))]
pub(super) fn spots(
    state: &AppState,
    dir: &Path,
    classes: &[String],
    hidden: &[String],
) -> Arc<statics::Table> {
    let wanted = (classes.to_vec(), hidden.to_vec());
    let stamp = crate::collect::labels::index::rows_version(dir, Some(&wanted));
    let job = Job {
        key: format!(
            "{}\u{1f}{}\u{1f}{}",
            dir.display(),
            classes.join("\u{1e}"),
            hidden.join("\u{1e}")
        ),
        stamp,
        dir: dir.to_path_buf(),
        classes: wanted.0,
        hidden: wanted.1,
        checks: super::super::refine::queue_stamp(&super::super::training::tasks_read_only(state)),
    };
    let every = if stamp.is_some() {
        SPOTS_INDEXED_EVERY
    } else {
        SPOTS_EVERY
    };
    let (table, due) = SPOTS.get(job, every);
    if let Some(job) = due {
        let spawned = std::thread::Builder::new()
            .name("review-spots".into())
            .spawn(move || {
                // A walk for the page: never ahead of detections for the CPU.
                crate::backend::priority::lower_this_thread(
                    crate::backend::priority::ThreadRole::Background,
                );
                SPOTS.run(job, &build);
            });
        if let Err(e) = spawned {
            tracing::warn!(
                "review queue: no thread to find the parked spots ({e}), so the order counts \
                 every box as unparked for now; it tries again on the next Next"
            );
            if let Some(s) = SPOTS.slot.lock_recover().as_mut() {
                s.building = false;
            }
        }
    }
    table
}

/// What the cached table was built from.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Key {
    path: PathBuf,
    modified: Option<SystemTime>,
    len: u64,
}

/// The one cached table (None = the file was missing or measured nothing).
struct Entry {
    key: Option<Key>,
    path: PathBuf,
    value: Option<Arc<Measured>>,
}

static CACHE: Mutex<Option<Entry>> = Mutex::new(None);

/// The measured recall of the training runs of the current config (see the module docs).
pub fn recall(state: &AppState) -> Option<Arc<Measured>> {
    let path = crate::training::runs_file(&state.config_read());
    cached(&CACHE, &path, || {
        super::super::training::records_read_only(state)
    })
}

/// The key of `path` now; None when it cannot be read (missing).
fn key_of(path: &Path) -> Option<Key> {
    let m = std::fs::metadata(path).ok().filter(|m| m.is_file())?;
    Some(Key {
        path: path.to_path_buf(),
        modified: m.modified().ok(),
        len: m.len(),
    })
}

/// [`recall`] over `cache` for the `runs.json` at `path`, reading the records with `read` only
/// when the key changed.
fn cached(
    cache: &Mutex<Option<Entry>>,
    path: &Path,
    read: impl FnOnce() -> Vec<RunRecord>,
) -> Option<Arc<Measured>> {
    let key = key_of(path);
    let mut c = cache.lock_recover();
    if let Some(e) = c.as_ref()
        && e.path == path
        && e.key == key
    {
        return e.value.clone();
    }
    let value = match &key {
        None => {
            tracing::debug!(
                "no {}: the Review queue has no measured recall to steer by",
                path.display()
            );
            None
        }
        Some(_) => {
            let v = crate::training::bench::measured_recall(&read()).map(Arc::new);
            if v.is_none() {
                tracing::debug!(
                    "{} holds no finished run with per-class held-out recall: the Review queue \
                     has no measured recall to steer by",
                    path.display()
                );
            }
            v
        }
    };
    *c = Some(Entry {
        key,
        path: path.to_path_buf(),
        value: value.clone(),
    });
    value
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::benchmark::ClassRecall;
    use crate::resources::catalog::TrainFamily;
    use crate::training::job::{BenchSummary, RecallCells, TrainOutput, TrainRequest};
    use crate::training::recipe::TrainRecipe;
    use crate::training::service::RunState;
    use std::cell::Cell;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("bop_steer_{name}_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A finished run measured over every scene (person: `gt` objects, `missed` missed).
    fn run(id: &str, gt: usize, missed: usize) -> serde_json::Value {
        let mut cells = RecallCells::default();
        cells
            .by_scene
            .insert("all".into(), vec![ClassRecall::new("person", gt, missed)]);
        cells
            .by_scene
            .insert("day".into(), vec![ClassRecall::new("person", gt, missed)]);
        let b = BenchSummary {
            heldout_fingerprint: Some("fp".into()),
            recall_cells: cells,
            ..BenchSummary::default()
        };
        let r = RunRecord {
            id: id.into(),
            name: id.into(),
            created: "2026-11-01T10:00:00Z".into(),
            request: TrainRequest {
                recipe: TrainRecipe::new(TrainFamily::Yolo26, "n").unwrap(),
                datasets: Vec::new(),
                include_public: false,
                scene: Default::default(),
                classes: None,
                rename: Default::default(),
                benchmark_sets: Vec::new(),
                benchmark_devices: Vec::new(),
                allow_large: false,
                install_dir: None,
                gpu: None,
                excluded: Default::default(),
                balance: Default::default(),
                compare_group: None,
                train_seed: None,
                ignore_memory_check: false,
            },
            device: "cpu".into(),
            state: RunState::Done,
            stage: None,
            step: 8,
            steps: 8,
            percent: 100,
            detail: String::new(),
            progress: None,
            log: Vec::new(),
            dataset: None,
            output: Some(TrainOutput {
                name: id.into(),
                onnx: PathBuf::from(format!("/models/{id}.onnx")),
                yaml: PathBuf::from(format!("/models/{id}.yaml")),
                provenance: PathBuf::new(),
                family: crate::model::ModelFamilyKind::Yolo26,
                classes: vec!["person".into()],
                license: String::new(),
                heldout: None,
                device: String::new(),
                train_seconds: 0.0,
            }),
            registered: None,
            benchmark: Some(b),
            benchmark_error: None,
            baseline: None,
            error: None,
            finished: Some("2026-11-01T11:00:00Z".into()),
            run_dir: PathBuf::from(format!("/training/{id}")),
            retrying: false,
            resources: None,
        };
        serde_json::to_value(r).unwrap()
    }

    fn write(path: &Path, runs: &[serde_json::Value]) {
        std::fs::write(path, serde_json::to_vec(runs).unwrap()).unwrap();
    }

    fn read(path: &Path) -> Vec<RunRecord> {
        crate::training::service::load_bare_records(path)
    }

    #[test]
    fn the_table_is_cached_until_runs_json_changes() {
        let dir = temp("cache");
        let path = dir.join("runs.json");
        let cache = Mutex::new(None);
        let reads = Cell::new(0);
        let get = || {
            cached(&cache, &path, || {
                reads.set(reads.get() + 1);
                read(&path)
            })
        };
        // Missing: None, and nothing read.
        assert!(get().is_none());
        assert_eq!(reads.get(), 0);
        write(&path, &[run("a", 40, 4)]);
        // The record format must parse, or this test proves nothing.
        assert_eq!(read(&path).len(), 1);
        let first = get().unwrap();
        assert_eq!(reads.get(), 1);
        assert_eq!(first.overall.as_ref().unwrap().run.id, "a");
        // Unchanged: the same table, not read again.
        let again = get().unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        assert_eq!(reads.get(), 1);
        // Changed (another length): read again.
        write(&path, &[run("a", 40, 4), run("bb", 50, 5)]);
        let next = get().unwrap();
        assert_eq!(reads.get(), 2);
        assert_eq!(next.overall.as_ref().unwrap().class("person").gt, 50);
        // Removed: None.
        std::fs::remove_file(&path).unwrap();
        assert!(get().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_damaged_or_empty_runs_json_is_none() {
        let dir = temp("damaged");
        let path = dir.join("runs.json");
        let cache = Mutex::new(None);
        std::fs::write(&path, b"{ not json").unwrap();
        assert!(cached(&cache, &path, || read(&path)).is_none());
        std::fs::write(&path, b"[]").unwrap();
        assert!(cached(&cache, &path, || read(&path)).is_none());
        // A run without recall cells measures nothing.
        let mut r = run("a", 40, 4);
        r["benchmark"]["recall_cells"] = serde_json::json!({});
        write(&path, &[r]);
        assert!(cached(&cache, &path, || read(&path)).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Through the app state: a config whose training folder has no `runs.json` answers None.
    #[test]
    fn the_state_without_runs_is_none() {
        let state = crate::server::test_util::test_state();
        assert!(recall(&state).is_none());
    }

    /// The snapshot of a folder without reviewed frames steers by nothing: no recall, no spot.
    /// The spot table is rebuilt at most every ten minutes outside tests.
    #[test]
    fn a_snapshot_of_a_new_folder_is_neutral() {
        let state = crate::server::test_util::test_state();
        let dir = temp("snapshot");
        let s = snapshot(&state, &dir, &["car".to_string()], &[]);
        assert!(s.recall.is_none());
        assert!(s.spots.is_empty());
        assert!(s.heldout_listed.is_empty());
        // The checks' stamp the spots key on comes from a reader that never writes.
        assert!(crate::server::training::tasks_read_only(&state).is_empty());
        assert_eq!(SPOTS_EVERY, std::time::Duration::from_secs(600));
        assert_eq!(SPOTS_INDEXED_EVERY, std::time::Duration::from_secs(60));
        // Without an index the builder takes Refine's table (none here).
        let job = job_for(&dir, "k", None);
        assert!(build(&job).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn job_for(dir: &Path, key: &str, stamp: Option<u64>) -> Job {
        Job {
            key: key.into(),
            stamp,
            dir: dir.to_path_buf(),
            classes: vec!["car".into()],
            hidden: Vec::new(),
            checks: String::new(),
        }
    }

    /// A table of `n` made-up spots of camera `cam` (each a car parked on two days).
    fn table(cam: &str, n: usize) -> statics::Table {
        let obs = (0..n).flat_map(|s| {
            (0..12).map(move |i| statics::Obs {
                camera: cam.into(),
                frame: format!("{s}-{i}"),
                day: if i < 6 { "d1" } else { "d2" }.into(),
                scene: "day".into(),
                label: "car".into(),
                bbox: [0.2 * s as f32, 0.0, 0.2 * s as f32 + 0.1, 0.1],
            })
        });
        statics::Table::build(obs)
    }

    #[test]
    fn the_spot_cache_builds_once_serves_the_last_and_follows_the_latest_key() {
        let cache = SpotCache::new();
        let dir = PathBuf::from("/nowhere");
        let every = Duration::from_secs(3600);
        // Nothing built: empty, and a build is due.
        let (t, due) = cache.get(job_for(&dir, "a", Some(1)), every);
        assert!(t.is_empty());
        let due = due.unwrap();
        // While it runs, a second ask for another folder waits; one for the same is no build.
        assert!(cache.get(job_for(&dir, "b", Some(7)), every).1.is_none());
        assert!(cache.get(job_for(&dir, "a", Some(1)), every).1.is_none());
        // The build of `a`, then the latest asked for (`a` again, which is no new work).
        let built = std::cell::Cell::new(0);
        cache.run(due, &|j| {
            built.set(built.get() + 1);
            table("cam", if j.key == "a" { 1 } else { 2 })
        });
        assert_eq!(built.get(), 1, "the queued job was the one just built");
        assert_eq!(cache.table_of("a").unwrap().len(), 1);
        // The same key: served, nothing due while the rows did not change, nor within `every`.
        let (t, due) = cache.get(job_for(&dir, "a", Some(1)), every);
        assert_eq!(t.len(), 1);
        assert!(due.is_none());
        assert!(cache.get(job_for(&dir, "a", Some(2)), every).1.is_none());
        // Past `every` the changed rows rebuild, the old table serving meanwhile.
        let (t, due) = cache.get(job_for(&dir, "a", Some(2)), Duration::ZERO);
        assert_eq!(t.len(), 1);
        assert!(due.is_some());
        // Another key while that build runs: queued, built right after.
        assert!(cache.get(job_for(&dir, "b", Some(9)), every).1.is_none());
        let seen = std::cell::RefCell::new(Vec::new());
        cache.run(due.unwrap(), &|j| {
            seen.borrow_mut().push(j.key.clone());
            table("cam", if j.key == "a" { 1 } else { 2 })
        });
        assert_eq!(*seen.borrow(), ["a", "b"]);
        let (t, due) = cache.get(job_for(&dir, "b", Some(9)), every);
        assert_eq!(t.len(), 2);
        assert!(due.is_none());
        // The old key is no longer served: a switch back builds again, empty meanwhile.
        let (t, due) = cache.get(job_for(&dir, "a", Some(2)), every);
        assert!(t.is_empty() && due.is_some());
    }

    #[test]
    fn a_panicking_spot_build_leaves_the_cache_able_to_build() {
        let cache = SpotCache::new();
        let dir = PathBuf::from("/nowhere");
        let every = Duration::from_secs(3600);
        let due = cache.get(job_for(&dir, "a", None), every).1.unwrap();
        let hit = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cache.run(due, &|_| panic!("a build that fails"));
        }));
        assert!(hit.is_err());
        // Not stuck "building": the next ask builds again.
        let (t, due) = cache.get(job_for(&dir, "a", None), every);
        assert!(t.is_empty());
        cache.run(due.expect("a build is due again"), &|_| table("cam", 1));
        assert_eq!(cache.get(job_for(&dir, "a", None), every).0.len(), 1);
    }
}
