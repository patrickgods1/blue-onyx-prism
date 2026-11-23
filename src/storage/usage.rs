//! The Storage card's numbers (Config page, Resources tab): what is safe to clear and what is
//! the user's data, each with its size.
//!
//! - Safe to clear: each cache folder ([`super::caches`]), leftover staging folders (training
//!   runs' in the models folder and in every install folder a run record names, and the scratch
//!   folders of review exports that crashed), the work folders of crashed jobs, old log files.
//!   Each has a Clear ([`super::sweep::clear`]).
//! - Your data: the dataset and review folders, each review export (with Remove), unfinished
//!   review exports no sweep removes (no busy marker: an older version's, or a finished export
//!   whose final move failed), training runs, the checkpoints failed runs keep for a retry, models,
//!   and work folders kept on purpose. Sizes only: nothing here is deleted automatically, and only
//!   exports have a Remove on this card.
//!
//! Sizes are bytes deleting a thing frees ([`crate::training::retry::disk_use`]: an export's frames
//! are hard links to the dataset's, so they count where they are not shared). Measuring walks
//! every file, so [`measure`] runs off the request thread and its result is kept for [`FRESH`].

use super::sweep::{self, Settings};
use super::{caches, size_of};
use crate::resources::catalog::format_size;
use crate::sync::LockExt;
use crate::training::service::RunRecord;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// How long a measurement is served before it is taken again.
pub const FRESH: Duration = Duration::from_secs(60);

/// One row of the card.
#[derive(Debug, Clone, Serialize)]
pub struct Item {
    /// What Clear / Remove name (`cache:coreml`, `staging`, `work`, `logs`, `export:<name>`), or
    /// a plain key for rows without an action.
    pub id: String,
    pub title: String,
    pub path: String,
    pub bytes: u64,
    pub size: String,
    pub note: String,
    /// `clear`, `remove` or None.
    pub action: Option<&'static str>,
}

impl Item {
    fn new(id: impl Into<String>, title: impl Into<String>, path: &Path, bytes: u64) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            path: path.display().to_string(),
            bytes,
            size: format_size(bytes),
            note: String::new(),
            action: None,
        }
    }

    fn note(mut self, n: impl Into<String>) -> Self {
        self.note = n.into();
        self
    }

    fn action(mut self, a: &'static str) -> Self {
        self.action = Some(a);
        self
    }
}

/// The card.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub safe: Vec<Item>,
    pub data: Vec<Item>,
    pub safe_bytes: u64,
    pub data_bytes: u64,
    /// Free space under the data root.
    pub free_bytes: Option<u64>,
    /// Unix seconds of the measurement.
    pub measured_at: u64,
    /// Seconds the measurement took.
    pub seconds: f64,
}

/// "today", "yesterday", "12 days ago".
fn days_ago(secs: u64, now: SystemTime) -> String {
    let at = UNIX_EPOCH + Duration::from_secs(secs);
    let days = now.duration_since(at).unwrap_or_default().as_secs() / 86_400;
    match days {
        0 => "today".into(),
        1 => "yesterday".into(),
        d => format!("{d} days ago"),
    }
}

/// A cache folder's title: `openvino-gpu-f16` -> "OpenVINO cache (GPU, f16)".
pub fn cache_title(name: &str) -> String {
    match name {
        "coreml" => "CoreML cache (compiled models and graph rewrites)".into(),
        "tensorrt" => "TensorRT engine cache".into(),
        n => match n.strip_prefix("openvino-").and_then(|r| r.split_once('-')) {
            Some((dev, prec)) => format!("OpenVINO cache ({}, {prec})", dev.to_uppercase()),
            None => "OpenVINO cache".into(),
        },
    }
}

/// Training run folders and failed runs' checkpoints, from the run records.
fn run_items(runs: &[RunRecord], out: &mut Vec<Item>, training_dir: &Path) {
    let mut total = 0u64;
    let mut counted = 0usize;
    let mut checkpoints = 0u64;
    let mut failed = 0usize;
    for r in runs.iter().filter(|r| !r.state.is_busy()) {
        let all = crate::training::retry::disk_use(&r.run_dir);
        if all == 0 {
            continue;
        }
        total += all;
        counted += 1;
        if r.state == crate::training::service::RunState::Failed {
            let w = crate::training::retry::disk_use(&crate::training::retry::work_dir(&r.run_dir));
            if w > 0 {
                checkpoints += w;
                failed += 1;
            }
        }
    }
    out.push(
        Item::new("runs", "Training runs", training_dir, total).note(format!(
            "{counted} finished run folder(s) (datasets, logs, checkpoints); remove a run on the \
             Train area's Models tab"
        )),
    );
    if failed > 0 {
        out.push(
            Item::new(
                "checkpoints",
                "Checkpoints of failed runs",
                training_dir,
                checkpoints,
            )
            .note(format!(
                "{failed} failed run(s) keep their work folder so Retry export can make the model \
                 without training again; never deleted automatically (included in Training runs)"
            )),
        );
    }
}

/// Every review export under `<training dir>/exports`: folders with `classes.txt` or `data.yaml`.
pub fn exports(training_dir: &Path) -> Vec<PathBuf> {
    let root = training_dir.join("exports");
    let Ok(rd) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut v: Vec<PathBuf> = rd
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| e.path())
        .filter(|p| p.join("classes.txt").is_file() || p.join("data.yaml").is_file())
        .collect();
    v.sort();
    v
}

/// The "Leftover staging" row (Clear `staging`): training runs' staging folders and the scratch
/// folders of review exports that crashed. None when there is neither.
fn staging_item(
    staging: &[PathBuf],
    crashed_exports: &[sweep::ExportScratch],
    models: &Path,
    training: &Path,
) -> Option<Item> {
    if staging.is_empty() && crashed_exports.is_empty() {
        return None;
    }
    let bytes: u64 = staging
        .iter()
        .map(|p| size_of(p))
        .chain(crashed_exports.iter().map(|x| size_of(&x.path)))
        .sum();
    let mut notes = Vec::new();
    if !staging.is_empty() {
        let elsewhere: Vec<String> = staging
            .iter()
            .filter(|p| p.parent() != Some(models))
            .map(|p| p.display().to_string())
            .collect();
        let mut n = format!(
            "{} folder(s) of training runs that stopped before installing their model",
            staging.len()
        );
        if !elsewhere.is_empty() {
            n.push_str(&format!(
                " (outside the models folder: {})",
                elsewhere.join(", ")
            ));
        }
        notes.push(n);
    }
    if !crashed_exports.is_empty() {
        notes.push(format!(
            "{} unfinished review export(s) a crash or stop cut off: {}",
            crashed_exports.len(),
            crashed_exports
                .iter()
                .take(5)
                .map(|x| x.path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    let (title, path) = if crashed_exports.is_empty() {
        ("Leftover training staging", models)
    } else if staging.is_empty() {
        ("Unfinished review exports", training)
    } else {
        ("Leftover staging (training runs, review exports)", training)
    };
    Some(
        Item::new("staging", title, path, bytes)
            .note(notes.join("; "))
            .action("clear"),
    )
}

/// The report-only row of review exports' scratch folders no sweep removes (no busy marker: an
/// older version's, or a complete export whose final move failed), except those inside a folder
/// `listed` as an export (counted in its row). None when there are none.
fn unfinished_item(
    unfinished: &[sweep::ExportScratch],
    listed: &[PathBuf],
    training: &Path,
) -> Option<Item> {
    let shown: Vec<&PathBuf> = unfinished
        .iter()
        .map(|x| &x.path)
        .filter(|p| !listed.iter().any(|e| p.parent() == Some(e.as_path())))
        .collect();
    if shown.is_empty() {
        return None;
    }
    let bytes: u64 = shown.iter().map(|p| size_of(p)).sum();
    Some(
        Item::new(
            "unfinished-exports",
            "Unfinished review exports (kept)",
            training,
            bytes,
        )
        .note(format!(
            "{} hidden folder(s) of exports that never got into place: left by an older version, \
             or complete but not moved (the export said where); never deleted automatically. Move \
             one into place or delete it by hand: their frames are hard links to the dataset's, so \
             they keep the bytes of frames the dataset has since deleted. {}",
            shown.len(),
            shown
                .iter()
                .take(5)
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    )
}

/// Measure the card for `s`, with the training runs `runs` (their folders are skipped while
/// busy).
pub fn measure(s: &Settings, runs: &[RunRecord]) -> Report {
    let started = Instant::now();
    let now = SystemTime::now();
    let mut safe = Vec::new();
    if let Some(root) = s.cache_root() {
        for c in caches::infos(&root) {
            let mut note = match c.last_used {
                Some(t) => format!("last used {}", days_ago(t, now)),
                None => "not used since this version started recording".into(),
            };
            if c.in_use {
                note.push_str("; in use: loaded models keep running, the next load compiles again");
            } else {
                note.push_str(&format!(
                    "; deleted automatically after {} days unused",
                    caches::UNUSED_DAYS
                ));
            }
            safe.push(
                Item::new(
                    format!("cache:{}", c.name),
                    cache_title(&c.name),
                    &c.path,
                    c.bytes,
                )
                .note(note)
                .action("clear"),
            );
        }
    }
    let models = s.models_dir();
    let training = s.training_dir();
    let staging = sweep::leftover_staging_all(s);
    // Only what lies in the models folder is inside the Models row below.
    let staging_bytes: u64 = staging
        .iter()
        .filter(|p| p.parent() == Some(models.as_path()))
        .map(|p| size_of(p))
        .sum();
    let (crashed_exports, unfinished_exports): (Vec<_>, Vec<_>) = sweep::export_scratch(s)
        .into_iter()
        .partition(|x| x.crashed);
    if let Some(item) = staging_item(&staging, &crashed_exports, &models, &training) {
        safe.push(item);
    }
    let crashed = sweep::crashed_all(s);
    if !crashed.is_empty() {
        let bytes: u64 = crashed.iter().map(|w| size_of(&w.dir)).sum();
        let autolabel = crashed
            .iter()
            .filter(|w| w.kind == sweep::WorkKind::Autolabel)
            .count();
        let mut note = format!("{} folder(s) of jobs that crashed", crashed.len());
        if autolabel > 0 {
            note.push_str(&format!(
                "; {autolabel} auto-label folder(s): their finished frames are written to the \
                 label files first"
            ));
        }
        safe.push(
            Item::new("work", "Work folders of crashed jobs", &training, bytes)
                .note(note)
                .action("clear"),
        );
    }
    if let Some(dir) = s.log_dir() {
        let old = sweep::old_logs(&dir, now);
        if !old.is_empty() {
            let bytes: u64 = old.iter().map(|p| size_of(p)).sum();
            safe.push(
                Item::new("logs", "Old log files", &dir, bytes)
                    .note(format!(
                        "{} file(s) before today; the newest file is kept",
                        old.len()
                    ))
                    .action("clear"),
            );
        }
    }

    let mut data = Vec::new();
    for (i, dir) in s.dataset_dirs().iter().enumerate() {
        let (id, title) = if i == 0 {
            ("dataset", "Dataset (collected and imported frames, labels)")
        } else {
            ("review-folder", "Folder under review")
        };
        data.push(Item::new(
            id,
            title,
            dir,
            crate::training::retry::disk_use(dir),
        ));
    }
    let listed = exports(&training);
    if let Some(item) = unfinished_item(&unfinished_exports, &listed, &training) {
        data.push(item);
    }
    for e in listed {
        let name = e
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        data.push(
            Item::new(
                format!("export:{name}"),
                // A frame-stamp name (a time) is shown as one, in the configured zone.
                format!(
                    "Export {}",
                    crate::clock::parse_frame_stamp(&name)
                        .map_or_else(|| name.clone(), crate::clock::display)
                ),
                &e,
                crate::training::retry::disk_use(&e),
            )
            .note("frames shared with the dataset as hard links count where they are not shared")
            .action("remove"),
        );
    }
    run_items(runs, &mut data, &training);
    data.push(Item::new(
        "models",
        "Models",
        &models,
        crate::training::retry::disk_use(&models).saturating_sub(staging_bytes),
    ));
    let kept = sweep::kept_all(s);
    if !kept.is_empty() {
        let bytes: u64 = kept.iter().map(|w| size_of(&w.dir)).sum();
        data.push(
            Item::new("kept-work", "Kept job work folders", &training, bytes).note(format!(
                "{} folder(s) kept on purpose (an import's keep work, auto-label proposals that \
                 could not be saved) or left by an older version: {}",
                kept.len(),
                kept.iter()
                    .take(5)
                    .map(|w| w.dir.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        );
    }
    Report {
        safe_bytes: safe.iter().map(|i| i.bytes).sum(),
        data_bytes: data
            .iter()
            .filter(|i| i.id != "checkpoints")
            .map(|i| i.bytes)
            .sum(),
        safe,
        data,
        free_bytes: crate::retention::free_bytes(&s.config().data_root()),
        measured_at: now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        seconds: (started.elapsed().as_secs_f64() * 10.0).round() / 10.0,
    }
}

/// One finished measurement: when, of which folders ([`key_of`]), and the report.
struct Measured {
    at: Instant,
    key: String,
    report: Arc<Report>,
}

/// The last measurement of each set of folders (newest first, at most [`KEPT`]) and whether one
/// is running. One slot per set: with a single slot shared by all, two callers of different
/// folders (tests with their own data roots; a generation restart that moved `download_dir`)
/// overwrite each other's report before the other looks, and each measures again, round after
/// round, until its wait runs out (seen on Linux CI, where measuring an empty folder takes less
/// than one 25 ms look).
#[derive(Default)]
struct Cache {
    last: Vec<Measured>,
    running: bool,
}

/// How many sets of folders [`Cache`] keeps a report of (the service has one; tests have more).
const KEPT: usize = 8;

static CACHE: Mutex<Cache> = Mutex::new(Cache {
    last: Vec::new(),
    running: false,
});

/// Clears [`Cache::running`] when the measuring thread ends, also by a panic in [`measure`] (which
/// would otherwise leave every later look waiting on a measurement that never ends).
struct Measuring;

impl Drop for Measuring {
    fn drop(&mut self) {
        if std::thread::panicking() {
            tracing::warn!(
                "storage: measuring the folders panicked; the Storage card measures again on the \
                 next look (please report this with the log)"
            );
        }
        CACHE.lock_recover().running = false;
    }
}

/// Tests: [`measure`] panics for settings of this data root (and says it did).
#[cfg(test)]
static PANIC_FOR: Mutex<Option<PathBuf>> = Mutex::new(None);
#[cfg(test)]
static PANICKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The measurement to show: a fresh one of the same folders ([`key_of`]) is returned at once;
/// otherwise one is started on its own thread (after a measurement of other folders has ended,
/// and again when a Clear forgot it before it was read) and awaited for at most `wait`. Returns the report of these folders (None before the first
/// one ends) and whether a measurement is still running.
pub fn current(s: Settings, runs: Vec<RunRecord>, wait: Duration) -> (Option<Arc<Report>>, bool) {
    let key = key_of(&s);
    let until = Instant::now() + wait;
    loop {
        {
            let mut c = CACHE.lock_recover();
            let last = c
                .last
                .iter()
                .find(|l| l.key == key)
                .map(|l| (l.at, l.report.clone()));
            if let Some((at, r)) = &last
                && at.elapsed() < FRESH
            {
                return (Some(r.clone()), false);
            }
            // Nothing fresh of these folders and nothing running (none yet, another folder's
            // ended, or a Clear forgot ours): measure them.
            if !c.running {
                let (s, runs) = (s.clone(), runs.clone());
                c.running = true;
                let k = key.clone();
                let spawned = std::thread::Builder::new()
                    .name("storage-usage".into())
                    .spawn(move || {
                        let _measuring = Measuring;
                        crate::backend::priority::lower_this_thread(
                            crate::backend::priority::ThreadRole::Background,
                        );
                        #[cfg(test)]
                        if PANIC_FOR.lock_recover().as_deref() == Some(&s.config().data_root()) {
                            PANICKED.store(true, std::sync::atomic::Ordering::SeqCst);
                            panic!("test: measuring panics");
                        }
                        let report = Arc::new(measure(&s, &runs));
                        let mut c = CACHE.lock_recover();
                        c.last.retain(|l| l.key != k);
                        c.last.insert(
                            0,
                            Measured {
                                at: Instant::now(),
                                key: k,
                                report,
                            },
                        );
                        c.last.truncate(KEPT);
                    });
                if let Err(e) = spawned {
                    c.running = false;
                    tracing::warn!("storage: could not start measuring ({e})");
                    return (last.map(|(_, r)| r), false);
                }
            }
            if Instant::now() >= until {
                return (last.map(|(_, r)| r), c.running);
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

/// The folders a measurement covers: a report counts only for the same ones (a generation
/// restart that moves `download_dir`, or tests with their own data roots, measure others).
fn key_of(s: &Settings) -> String {
    format!(
        "{:?}",
        (
            s.config().data_root(),
            s.models_dir(),
            s.training_dir(),
            s.cache_root()
        )
    )
}

/// Forget the last measurement (after a Clear or Remove), so the next look measures again.
pub fn invalidate() {
    CACHE.lock_recover().last.clear();
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn cache_titles() {
        assert_eq!(cache_title("openvino-gpu-f16"), "OpenVINO cache (GPU, f16)");
        assert_eq!(cache_title("openvino-cpu-f32"), "OpenVINO cache (CPU, f32)");
        assert!(cache_title("coreml").starts_with("CoreML"));
        assert!(cache_title("tensorrt").starts_with("TensorRT"));
    }

    #[test]
    fn ages_read_as_days() {
        let now = UNIX_EPOCH + Duration::from_secs(100 * 86_400);
        assert_eq!(days_ago(100 * 86_400 - 5, now), "today");
        assert_eq!(days_ago(99 * 86_400 - 5, now), "yesterday");
        assert_eq!(days_ago(88 * 86_400, now), "12 days ago");
        assert_eq!(days_ago(200 * 86_400, now), "today", "a future time");
    }

    #[test]
    fn a_report_counts_only_for_the_folders_it_measured() {
        let settings = |tag: &str| {
            let root =
                std::env::temp_dir().join(format!("bop-usage-{tag}-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&root).unwrap();
            let mut c = crate::config::Config {
                download_dir: Some(root.clone()),
                cache_dir: root.join("cache").display().to_string(),
                ..Default::default()
            };
            c.models.clear();
            (root, Settings::for_config(&c))
        };
        let (ra, a) = settings("a");
        let (rb, b) = settings("b");
        assert_ne!(key_of(&a), key_of(&b));
        let long = Duration::from_secs(60);
        // Other tests share the process-wide cache: wait for each answer to be complete.
        let (first, running) = current(a.clone(), Vec::new(), long);
        assert!(!running);
        let first = first.unwrap();
        let (other, running) = current(b, Vec::new(), long);
        assert!(!running);
        let other = other.unwrap();
        assert!(!Arc::ptr_eq(&first, &other), "b got a's report");
        // a measures again rather than reusing b's report.
        let (again, _) = current(a, Vec::new(), long);
        assert!(!Arc::ptr_eq(&again.unwrap(), &other));
        for r in [ra, rb] {
            std::fs::remove_dir_all(r).ok();
        }
    }

    /// A caller that keeps measuring other folders never starves ours: each set of folders keeps
    /// its own report (with one shared slot this failed 10 of 10 runs; on Linux CI two tests of
    /// this module starved each other that way until their 60 s waits ran out).
    #[test]
    fn a_look_gets_its_report_while_others_measure() {
        let settings = |tag: &str| {
            let root =
                std::env::temp_dir().join(format!("bop-usage-{tag}-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&root).unwrap();
            let mut c = crate::config::Config {
                download_dir: Some(root.clone()),
                cache_dir: root.join("cache").display().to_string(),
                ..Default::default()
            };
            c.models.clear();
            (root, Settings::for_config(&c))
        };
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (other_root, other) = settings("other");
        let st = stop.clone();
        let t = std::thread::spawn(move || {
            while !st.load(std::sync::atomic::Ordering::SeqCst) {
                let _ = current(other.clone(), Vec::new(), Duration::ZERO);
                std::thread::sleep(Duration::from_millis(1));
            }
        });
        std::thread::sleep(Duration::from_millis(50));
        let (mine_root, mine) = settings("mine");
        let (r, running) = current(mine, Vec::new(), Duration::from_secs(30));
        stop.store(true, std::sync::atomic::Ordering::SeqCst);
        t.join().unwrap();
        assert!(
            r.is_some() && !running,
            "starved (still running: {running})"
        );
        for root in [other_root, mine_root] {
            std::fs::remove_dir_all(root).ok();
        }
    }

    /// Audit 2026-11 phase 1: a panic while measuring must not leave the card "measuring" for
    /// good; the next look measures again.
    #[test]
    fn a_panic_while_measuring_lets_the_next_look_measure_again() {
        let root = std::env::temp_dir().join(format!("bop-usage-panic-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let mut c = crate::config::Config {
            download_dir: Some(root.clone()),
            cache_dir: root.join("cache").display().to_string(),
            ..Default::default()
        };
        c.models.clear();
        let s = Settings::for_config(&c);
        *PANIC_FOR.lock().unwrap() = Some(c.data_root());
        // Other tests share the cache and may be measuring their own folders: wait for ours.
        let deadline = Instant::now() + Duration::from_secs(60);
        while !PANICKED.load(std::sync::atomic::Ordering::SeqCst) {
            let (r, _) = current(s.clone(), Vec::new(), Duration::ZERO);
            assert!(r.is_none(), "nothing measured yet");
            assert!(Instant::now() < deadline, "the measurement never started");
            std::thread::sleep(Duration::from_millis(10));
        }
        while CACHE.lock_recover().running {
            assert!(
                Instant::now() < deadline,
                "still running long after the measuring thread panicked"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        *PANIC_FOR.lock().unwrap() = None;
        let (r, running) = current(s, Vec::new(), Duration::from_secs(60));
        assert!(r.is_some() && !running);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn the_card_splits_safe_from_data() {
        let root = std::env::temp_dir().join(format!("bop-usage-{}", uuid::Uuid::new_v4()));
        let put = |p: &Path, n: usize| {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, vec![0u8; n]).unwrap();
        };
        let mut c = crate::config::Config {
            download_dir: Some(root.clone()),
            cache_dir: root.join("cache").display().to_string(),
            ..Default::default()
        };
        c.models.clear();
        let s = Settings::for_config(&c);
        put(&root.join("cache/openvino-cpu-f16/blob"), 1000);
        put(&s.models_dir().join("m.onnx"), 300);
        put(&s.models_dir().join(".m.train-staging/x.onnx"), 50);
        let t = s.training_dir();
        put(&t.join("exports/e1/classes.txt"), 10);
        put(&t.join("imports/20261107-1/frames/a.jpg"), 70);
        std::fs::write(
            super::super::busy::marker_of(&t.join("imports/20261107-1")),
            b"1",
        )
        .unwrap();
        let r = measure(&s, &[]);
        let ids = |v: &[Item]| v.iter().map(|i| i.id.clone()).collect::<Vec<_>>();
        assert_eq!(
            ids(&r.safe),
            ["cache:openvino-cpu-f16", "staging", "work"],
            "{r:?}"
        );
        assert!(ids(&r.data).contains(&"export:e1".to_string()));
        let models = r.data.iter().find(|i| i.id == "models").unwrap();
        assert_eq!(
            models.bytes, 300,
            "staging is counted under safe, not twice"
        );
        let e = r.data.iter().find(|i| i.id == "export:e1").unwrap();
        assert_eq!((e.bytes, e.action), (10, Some("remove")));
        assert!(r.safe.iter().all(|i| i.action == Some("clear")));
        assert!(
            r.data
                .iter()
                .all(|i| i.action.is_none() || i.id.starts_with("export:"))
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Audit 2026-11 phase 2: a crashed review export's scratch folder is safe to clear (in the
    /// staging row); one without a marker is the user's to look at, listed with its size.
    #[test]
    fn export_scratch_rows_split_crashed_from_kept() {
        let root = std::env::temp_dir().join(format!("bop-usage-scratch-{}", uuid::Uuid::new_v4()));
        let put = |p: &Path, n: usize| {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, vec![0u8; n]).unwrap();
        };
        let mut c = crate::config::Config {
            download_dir: Some(root.clone()),
            cache_dir: String::new(),
            ..Default::default()
        };
        c.models.clear();
        let s = Settings::for_config(&c);
        let ex = s.training_dir().join("exports");
        let id = |i: u128| format!("{i:032x}");
        let crashed = ex.join(format!(".a.building-{}", id(1)));
        let kept = ex.join(format!(".b.building-{}", id(2)));
        put(&crashed.join("export/images/a.jpg"), 40);
        std::fs::write(super::super::busy::marker_of(&crashed), b"1").unwrap();
        put(&kept.join("export/images/a.jpg"), 25);
        put(&ex.join("e1/classes.txt"), 10);
        let r = measure(&s, &[]);
        let staging = r.safe.iter().find(|i| i.id == "staging").unwrap();
        assert_eq!(
            (staging.bytes, staging.title.as_str()),
            (40, "Unfinished review exports")
        );
        assert!(
            staging.note.contains(&*crashed.to_string_lossy()),
            "{}",
            staging.note
        );
        let unfinished = r
            .data
            .iter()
            .find(|i| i.id == "unfinished-exports")
            .unwrap();
        assert_eq!(unfinished.bytes, 25);
        assert_eq!(unfinished.action, None, "never removed from the card");
        assert!(unfinished.note.contains(&*kept.to_string_lossy()));
        assert_eq!(r.data_bytes, r.data.iter().map(|i| i.bytes).sum::<u64>());
        let _ = std::fs::remove_dir_all(&root);
    }
}
