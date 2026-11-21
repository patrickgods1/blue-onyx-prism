//! What steers the Review queue's Recommended order from outside its rows ([`snapshot`], one
//! [`Steer`] per `GET /v1/review/next`, handed to the review index and the walk alike): the
//! measured recall of the newest trained models ([`recall`]) and the folder's parked spots
//! ([`spots`]). Nothing here runs a walk on the request path.
//!
//! Recall. A [`Measured`] from `runs.json` through [`crate::training::bench::measured_recall`],
//! one cached entry keyed on `runs.json`'s (path, modification time, length): a request costs one
//! `stat` while nothing changed, and the table is rebuilt only after the training queue saved a run
//! (it rewrites the whole file). Records are read through [`crate::server::training::records`]
//! (the started queue's, else the file without starting the queue). A missing `runs.json`, or one
//! without measured runs (a damaged file reads as no runs), is None, logged once per change of the
//! key; never an error to the request. A rewrite of the same length within the file system's time
//! resolution is seen at the next change.
//!
//! Spots. The static spots of the folder's reviewed frames
//! ([`super::super::refine_spots::spot_table`], the Refine tab's table), built on a background
//! thread at most every [`SPOTS_EVERY`] (a walk of every reviewed frame), one at a time; a request
//! serves the last table built for the folder and class list, empty (so neutral: no box counts as
//! parked) until the first is in. In tests the table is built on the request's thread, so an
//! answer never depends on a thread's timing.

use crate::collect::labels::need::Steer;
use crate::collect::labels::recall::Measured;
use crate::collect::labels::statics;
use crate::server::AppState;
use crate::sync::LockExt;
use crate::training::service::RunRecord;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Least time between two builds of a folder's spot table.
pub const SPOTS_EVERY: std::time::Duration = std::time::Duration::from_secs(10 * 60);

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
        spots: spots(dir, classes, hidden),
        heldout_listed: Arc::default(),
    })
}

/// The parked spots of `dir`'s reviewed frames for the class list, built on the request's thread
/// in tests (see the module docs).
#[cfg(test)]
pub(super) fn spots(dir: &Path, classes: &[String], hidden: &[String]) -> Arc<statics::Table> {
    Arc::new(super::super::refine_spots::spot_table(dir, classes, hidden))
}

/// The last spot table built (one: the folder under review).
#[cfg(not(test))]
struct Spots {
    key: String,
    /// When the last build started.
    at: std::time::Instant,
    building: bool,
    table: Arc<statics::Table>,
}

#[cfg(not(test))]
static SPOTS: Mutex<Option<Spots>> = Mutex::new(None);

/// The parked spots of `dir`'s reviewed frames for the class list (see the module docs): the last
/// table built, a build started in the background when none is or the last is
/// [`SPOTS_EVERY`] old.
#[cfg(not(test))]
pub(super) fn spots(dir: &Path, classes: &[String], hidden: &[String]) -> Arc<statics::Table> {
    use std::time::Instant;
    let key = format!(
        "{}\u{1f}{}\u{1f}{}",
        dir.display(),
        classes.join("\u{1e}"),
        hidden.join("\u{1e}")
    );
    let mut slot = SPOTS.lock_recover();
    let table = match slot.as_mut() {
        Some(s) if s.key == key => {
            if s.building || s.at.elapsed() < SPOTS_EVERY {
                return s.table.clone();
            }
            s.building = true;
            s.at = Instant::now();
            s.table.clone()
        }
        _ => {
            *slot = Some(Spots {
                key: key.clone(),
                at: Instant::now(),
                building: true,
                table: Arc::default(),
            });
            Arc::default()
        }
    };
    drop(slot);
    let (dir, classes, hidden) = (dir.to_path_buf(), classes.to_vec(), hidden.to_vec());
    let spawned = std::thread::Builder::new()
        .name("review-spots".into())
        .spawn(move || {
            // A walk for the page: never ahead of detections for the CPU.
            crate::backend::priority::lower_this_thread(
                crate::backend::priority::ThreadRole::Background,
            );
            let built = Arc::new(super::super::refine_spots::spot_table(
                &dir, &classes, &hidden,
            ));
            tracing::debug!(
                "review queue: {} parked spot(s) in {}",
                built.len(),
                dir.display()
            );
            if let Some(s) = SPOTS.lock_recover().as_mut().filter(|s| s.key == key) {
                s.table = built;
                s.building = false;
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(
            "review queue: no thread to find the parked spots ({e}), so the order counts every box \
             as unparked; it tries again in {} minutes",
            SPOTS_EVERY.as_secs() / 60
        );
        if let Some(s) = SPOTS.lock_recover().as_mut() {
            s.building = false;
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
    cached(&CACHE, &path, || super::super::training::records(state))
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
        assert_eq!(SPOTS_EVERY, std::time::Duration::from_secs(600));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
