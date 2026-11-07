//! The Storage card's numbers (Config page, Resources tab): what is safe to clear and what is
//! the user's data, each with its size.
//!
//! - Safe to clear: each cache folder ([`super::caches`]), leftover staging folders, the work
//!   folders of crashed jobs, old log files. Each has a Clear ([`super::sweep::clear`]).
//! - Your data: the dataset and review folders, each review export (with Remove), training runs,
//!   the checkpoints failed runs keep for a retry, models, and work folders kept on purpose. Sizes
//!   only: nothing here is deleted automatically, and only exports have a Remove on this card.
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
    let staging = sweep::leftover_staging(&models);
    let staging_bytes: u64 = staging.iter().map(|p| size_of(p)).sum();
    if !staging.is_empty() {
        safe.push(
            Item::new(
                "staging",
                "Leftover training staging",
                &models,
                staging_bytes,
            )
            .note(format!(
                "{} folder(s) of training runs that stopped before installing their model",
                staging.len()
            ))
            .action("clear"),
        );
    }
    let training = s.training_dir();
    let crashed = sweep::crashed_work(&training);
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
    for e in exports(&training) {
        let name = e
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        data.push(
            Item::new(
                format!("export:{name}"),
                format!("Export {name}"),
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
    let kept = sweep::kept_work(&training);
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

/// The last measurement and whether one is running.
#[derive(Default)]
struct Cache {
    last: Option<(Instant, Arc<Report>)>,
    running: bool,
}

static CACHE: Mutex<Cache> = Mutex::new(Cache {
    last: None,
    running: false,
});

/// The measurement to show: a fresh one is returned at once; otherwise one is started on its own
/// thread (unless running) and awaited for at most `wait`. Returns the report (None before the
/// first one ends) and whether a measurement is still running.
pub fn current(s: Settings, runs: Vec<RunRecord>, wait: Duration) -> (Option<Arc<Report>>, bool) {
    {
        let mut c = CACHE.lock_recover();
        if let Some((at, r)) = &c.last
            && at.elapsed() < FRESH
        {
            return (Some(r.clone()), false);
        }
        if !c.running {
            c.running = true;
            let spawned = std::thread::Builder::new()
                .name("storage-usage".into())
                .spawn(move || {
                    let r = Arc::new(measure(&s, &runs));
                    let mut c = CACHE.lock_recover();
                    c.last = Some((Instant::now(), r));
                    c.running = false;
                });
            if let Err(e) = spawned {
                c.running = false;
                tracing::warn!("storage: could not start measuring ({e})");
            }
        }
    }
    let until = Instant::now() + wait;
    loop {
        {
            let c = CACHE.lock_recover();
            if !c.running || Instant::now() >= until {
                return (c.last.as_ref().map(|(_, r)| r.clone()), c.running);
            }
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

/// Forget the last measurement (after a Clear or Remove), so the next look measures again.
pub fn invalidate() {
    CACHE.lock_recover().last = None;
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
}
