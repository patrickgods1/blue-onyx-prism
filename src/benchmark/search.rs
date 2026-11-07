//! Confidence-threshold search over stored predictions: no inference.
//!
//! Every benchmark run keeps, per model and device, the predictions it made at the evaluation
//! threshold ([`super::EVAL_CONFIDENCE`]) in a scored class, per image, in post-processing
//! order. They are persisted in `PREDS_FILE` beside `benchmark.json` (kept out of the main
//! file so polling stays light) as `[class index, confidence, x_min, y_min, x_max, y_max]`
//! (integer pixel boxes, exactly what was scored). The ground truth (or pseudo ground truth) is
//! the one recorded in `benchmark.json`'s `image_sets`.
//!
//! [`search`] re-runs the exact sorted sweep ([`super::metrics::Breakpoints`],
//! [`super::threshold::exact_pick`]) over any subset: datasets, images carrying given tags
//! (e.g. `night`), scored classes, IoU and objective. Filtering the stored predictions at a
//! higher threshold is exact for every family (see [`super::metrics`]), so the search equals a
//! benchmark run at that threshold, without running one.
//!
//! The same stored predictions give the frame-level ROC ([`super::roc`]): a frame's score for a
//! class is its most confident stored prediction of that class, so the search reports frame ROC
//! AUC (overall, per dataset, per class) and can optimize the frame objectives (`youden`,
//! `fpr:<x>`) without inference.

use super::images::GroundTruth;
use super::metrics::{
    Breakpoints, ClassMap, EvalImage, GtBox, PredBox, SWEEP_THRESHOLDS, SweepImage, SweepScope,
    ThresholdPoint, canonical_label,
};
use super::report::{BenchmarkResults, ImageInfo, ImageSetInfo, ModelResult};
use super::roc::{FrameRoc, ROC_POINTS};
use super::threshold::{GroupPick, Objective, Pick, exact_pick};
use crate::sync::LockExt;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// File name of the stored predictions, beside `benchmark.json`.
pub(crate) const PREDS_FILE: &str = "benchmark-preds.json";
/// Points of the exact curve a search returns.
pub(crate) const SEARCH_EXACT_POINTS: usize = 200;
/// Stored predictions per image of one run at most, the most confident ones (BQ-04: high-recall
/// models at the evaluation threshold can make hundreds of boxes per image). A search at a
/// threshold above an image's most confident dropped prediction sees exactly what was scored.
pub(crate) const MAX_STORED_PREDS_PER_IMAGE: usize = 100;

/// `benchmark-preds.json` beside `config_path`.
pub fn preds_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .map(|d| d.join(PREDS_FILE))
        .unwrap_or_else(|| PathBuf::from(PREDS_FILE))
}

/// One stored prediction: `[class index, confidence, x_min, y_min, x_max, y_max]`, the box in
/// integer pixels (the scored predictions are integer pixel boxes, so this is lossless).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(
    from = "(u32, f32, u32, u32, u32, u32)",
    into = "(u32, f32, u32, u32, u32, u32)"
)]
pub struct StoredPred {
    pub class: u32,
    pub confidence: f32,
    pub bbox: [f32; 4],
}

impl From<(u32, f32, u32, u32, u32, u32)> for StoredPred {
    fn from(t: (u32, f32, u32, u32, u32, u32)) -> Self {
        Self {
            class: t.0,
            confidence: t.1,
            bbox: [t.2 as f32, t.3 as f32, t.4 as f32, t.5 as f32],
        }
    }
}

impl From<StoredPred> for (u32, f32, u32, u32, u32, u32) {
    fn from(p: StoredPred) -> Self {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "clamped to >= 0 (NaN -> 0 via f32::max); `as` saturates large values"
        )]
        let c = |v: f32| v.max(0.0).round() as u32;
        (
            p.class,
            p.confidence,
            c(p.bbox[0]),
            c(p.bbox[1]),
            c(p.bbox[2]),
            c(p.bbox[3]),
        )
    }
}

/// The stored predictions of one image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredImage {
    /// Dataset id.
    pub set: String,
    pub file: String,
    #[serde(default)]
    pub preds: Vec<StoredPred>,
}

/// The stored predictions of one model on one device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredRun {
    pub model: String,
    /// Device spec ("ort:coreml").
    pub device: String,
    #[serde(default)]
    pub timestamp: String,
    /// Scored classes in the model's label space; predictions refer to them by index.
    #[serde(default)]
    pub classes: Vec<String>,
    /// The run's configured confidence threshold.
    pub configured: f32,
    /// Lowest stored confidence (the evaluation threshold of the run).
    pub eval_confidence: f32,
    #[serde(default)]
    pub images: Vec<StoredImage>,
}

impl StoredRun {
    /// The stored form of a device run (None without accuracy or stored predictions).
    pub fn of(m: &ModelResult, device: &str, run: &super::RunResult) -> Option<Self> {
        let acc = run.accuracy.as_ref()?;
        if run.eval_preds.len() != run.per_image.len() || run.per_image.is_empty() {
            return None;
        }
        let index: HashMap<&str, u32> = acc
            .classes
            .iter()
            .enumerate()
            .filter_map(|(i, c)| Some((c.as_str(), u32::try_from(i).ok()?)))
            .collect();
        Some(Self {
            model: m.model.clone(),
            device: device.to_string(),
            timestamp: m.timestamp.clone(),
            classes: acc.classes.clone(),
            configured: acc.threshold,
            eval_confidence: super::EVAL_CONFIDENCE.min(acc.threshold),
            images: run
                .per_image
                .iter()
                .zip(&run.eval_preds)
                .map(|(img, preds)| StoredImage {
                    set: img.set.clone(),
                    file: img.file.clone(),
                    preds: most_confident(
                        preds
                            .iter()
                            .filter_map(|p| {
                                Some(StoredPred {
                                    class: *index.get(p.label.as_str())?,
                                    confidence: p.confidence,
                                    bbox: p.bbox,
                                })
                            })
                            .collect(),
                        MAX_STORED_PREDS_PER_IMAGE,
                    ),
                })
                .collect(),
        })
    }

    /// The predictions of image `k` as scored boxes.
    pub(crate) fn preds(&self, k: usize) -> Vec<PredBox> {
        let Some(img) = self.images.get(k) else {
            return Vec::new();
        };
        img.preds
            .iter()
            .filter_map(|p| {
                Some(PredBox {
                    label: self.classes.get(p.class as usize)?.clone(),
                    confidence: p.confidence,
                    bbox: p.bbox,
                })
            })
            .collect()
    }
}

/// The `max` most confident of `preds` (ties: the earlier first), kept in their order
/// (post-processing order, which the scoring follows).
fn most_confident(preds: Vec<StoredPred>, max: usize) -> Vec<StoredPred> {
    if preds.len() <= max {
        return preds;
    }
    let mut order: Vec<usize> = (0..preds.len()).collect();
    // Stable: equal confidences keep their order. NaN sorts last (`total_cmp` on the negation
    // would put it first).
    order.sort_by(|&a, &b| {
        let (ca, cb) = (
            preds.get(a).map_or(f32::NAN, |p| p.confidence),
            preds.get(b).map_or(f32::NAN, |p| p.confidence),
        );
        match (ca.is_nan(), cb.is_nan()) {
            (false, false) => cb.total_cmp(&ca),
            (a_nan, b_nan) => a_nan.cmp(&b_nan),
        }
    });
    order.truncate(max);
    order.sort_unstable();
    order
        .into_iter()
        .filter_map(|i| preds.get(i).copied())
        .collect()
}

/// The stored-predictions file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct StoredPreds {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub runs: Vec<StoredRun>,
}

impl StoredPreds {
    /// The file at `path`; None when it does not exist. Run entries that do not load are
    /// dropped (logged).
    pub fn load(path: &Path) -> Result<Option<Self>> {
        let (r, warnings) = Self::load_with_warnings(path)?;
        for w in warnings {
            tracing::warn!("{w}");
        }
        Ok(r)
    }

    /// [`Self::load`] through a one-entry, process-wide cache keyed by the file's path,
    /// modification time and length (BQ-04): repeated searches and options requests parse the
    /// file (tens of MB) once until it changes. Load errors are not cached. The parsed file goes
    /// after [`CACHE_IDLE`] without a reader (M10).
    pub fn load_cached(path: &Path) -> Result<Option<Arc<Self>>> {
        Self::load_cached_in(&PREDS_CACHE, path)
    }

    /// [`Self::load_cached`] with the cache given (tests).
    fn load_cached_in(cache: &'static PredsCache, path: &Path) -> Result<Option<Arc<Self>>> {
        let stamp = file_stamp(path);
        let key = (path.to_path_buf(), stamp);
        if let Some(v) = cache.get(&key) {
            return Ok(v);
        }
        // Parsed outside the lock (seconds for a large file): a concurrent request parses too.
        let loaded = Self::load(path)?.map(Arc::new);
        // The stamp taken before reading: a file replaced while it was read is parsed again
        // next time instead of being cached under the new stamp.
        cache.put(key, loaded.clone());
        Ok(loaded)
    }

    /// [`Self::load`] returning the warnings: `runs` is deserialized entry by entry, so one bad
    /// entry drops only that run (the rule of [`super::report::parse_lenient`]). Streamed from
    /// the file: one run entry is held as JSON at a time, never the whole text or its tree.
    pub fn load_with_warnings(path: &Path) -> Result<(Option<Self>, Vec<String>)> {
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((None, Vec::new())),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        let mut de = serde_json::Deserializer::from_reader(std::io::BufReader::new(file));
        let parsed = serde::de::Deserializer::deserialize_map(&mut de, lenient::FileVisitor)
            .and_then(|v| de.end().map(|()| v))
            .with_context(|| format!("parsing {}", path.display()))?;
        let lenient::Parsed {
            version,
            runs,
            dropped,
        } = parsed;
        let warnings = dropped
            .into_iter()
            .map(|(name, err)| {
                format!(
                    "{}: dropped `runs` entry {name}, which does not load: {err}",
                    path.display()
                )
            })
            .collect();
        Ok((Some(Self { version, runs }), warnings))
    }

    /// Compact JSON through a temporary file and a rename.
    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string(self)?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
        std::fs::write(&tmp, text).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))
    }

    /// The previous runs of models still in `merged` and not re-run in `new`, plus the runs of
    /// `new` (like [`BenchmarkResults::merge`]).
    pub fn merge(previous: Option<Self>, merged: &BenchmarkResults, new: &[ModelResult]) -> Self {
        let key = crate::registry::normalize_name;
        let mut runs: Vec<StoredRun> = previous
            .map(|p| p.runs)
            .unwrap_or_default()
            .into_iter()
            .filter(|r| {
                !new.iter().any(|m| key(&m.model) == key(&r.model))
                    && merged.find(&r.model).is_some()
            })
            .map(|mut r| {
                // Files written before the cap shrink on the next save.
                for img in &mut r.images {
                    img.preds =
                        most_confident(std::mem::take(&mut img.preds), MAX_STORED_PREDS_PER_IMAGE);
                }
                r
            })
            .collect();
        for m in new {
            for d in &m.devices {
                if let Some(run) = d.run.as_ref().filter(|_| d.ok())
                    && let Some(s) = StoredRun::of(m, &d.device, run)
                {
                    runs.push(s);
                }
            }
        }
        Self {
            version: crate::VERSION.to_string(),
            runs,
        }
    }

    /// The stored runs of `model` (normalized match).
    pub fn runs_of(&self, model: &str) -> Vec<&StoredRun> {
        let key = crate::registry::normalize_name(model);
        self.runs
            .iter()
            .filter(|r| crate::registry::normalize_name(&r.model) == key)
            .collect()
    }
}

/// (modified, length) of a file; None when it does not exist.
type FileStamp = Option<(Option<std::time::SystemTime>, u64)>;

fn file_stamp(p: &Path) -> FileStamp {
    std::fs::metadata(p)
        .ok()
        .map(|m| (m.modified().ok(), m.len()))
}

/// The last parsed stored-predictions file: (path, stamp) -> contents (None = no file).
type PredsCache = IdleCache<(PathBuf, FileStamp), Option<Arc<StoredPreds>>>;
static PREDS_CACHE: PredsCache = IdleCache::new(CACHE_IDLE);

/// A parsed file nobody asked for in this long is dropped from its cache (M10).
pub(crate) const CACHE_IDLE: Duration = Duration::from_secs(10 * 60);

/// A one-entry cache whose entry is dropped once nobody read it for its idle time: a thread
/// started with an entry looks every tenth of that time (at most a minute) and ends with the
/// entry, so an idle service holds neither. Process-wide caches are `static`s of it.
pub(crate) struct IdleCache<K, V> {
    state: Mutex<IdleState<K, V>>,
    idle: Duration,
}

struct IdleState<K, V> {
    /// Key, value and when it was last read or put.
    entry: Option<(K, V, Instant)>,
    /// The thread that drops the entry is running.
    sweeping: bool,
}

impl<K, V> IdleCache<K, V>
where
    K: PartialEq + Send + 'static,
    V: Clone + Send + 'static,
{
    pub(crate) const fn new(idle: Duration) -> Self {
        Self {
            state: Mutex::new(IdleState {
                entry: None,
                sweeping: false,
            }),
            idle,
        }
    }

    /// The value cached under `key`, now counted as read.
    pub(crate) fn get(&self, key: &K) -> Option<V> {
        let mut st = self.state.lock_recover();
        let (k, v, used) = st.entry.as_mut()?;
        if k != key {
            return None;
        }
        *used = Instant::now();
        Some(v.clone())
    }

    /// Cache `value` under `key` (replacing any other entry).
    pub(crate) fn put(&'static self, key: K, value: V) {
        let mut st = self.state.lock_recover();
        st.entry = Some((key, value, Instant::now()));
        self.start_sweeping(&mut st);
    }

    /// The value under `key`, or `make()`'s, cached. `make` runs under the cache's lock, so
    /// callers asking at once compute it once.
    pub(crate) fn get_or_insert_with(&'static self, key: K, make: impl FnOnce() -> V) -> V {
        let mut st = self.state.lock_recover();
        if let Some((k, v, used)) = st.entry.as_mut()
            && *k == key
        {
            *used = Instant::now();
            return v.clone();
        }
        let v = make();
        st.entry = Some((key, v.clone(), Instant::now()));
        self.start_sweeping(&mut st);
        v
    }

    /// Start the thread that drops the entry when none runs (under the lock, so it cannot end
    /// meanwhile). A thread that cannot start leaves the entry until the next put.
    fn start_sweeping(&'static self, st: &mut IdleState<K, V>) {
        if st.sweeping {
            return;
        }
        let every = (self.idle / 10).clamp(Duration::from_millis(1), Duration::from_secs(60));
        let spawned = std::thread::Builder::new()
            .name("bop-cache-idle".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(every);
                    if !self.sweep(Instant::now()) {
                        break;
                    }
                }
            });
        st.sweeping = spawned.is_ok();
    }

    /// Drop the entry when it was not read for the idle time at `now`. False when no entry is
    /// left: the sweeping thread then ends (marked under the lock, so a put starts a new one).
    fn sweep(&self, now: Instant) -> bool {
        let mut st = self.state.lock_recover();
        if st
            .entry
            .as_ref()
            .is_some_and(|(_, _, used)| now.saturating_duration_since(*used) >= self.idle)
        {
            st.entry = None;
        }
        if st.entry.is_none() {
            st.sweeping = false;
            return false;
        }
        true
    }

    /// Whether an entry is cached (tests).
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.state.lock_recover().entry.is_none()
    }
}

/// The streaming, lenient reader of the stored-predictions file: the object's `version` and its
/// `runs`, each run parsed from its own JSON value so a bad one is dropped with its name, as
/// [`super::report::parse_lenient`] does for a text read whole. Other keys are skipped.
mod lenient {
    use super::StoredRun;
    use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};
    use std::fmt;

    pub(super) struct Parsed {
        pub(super) version: String,
        pub(super) runs: Vec<StoredRun>,
        /// (entry name, error) of runs that did not load.
        pub(super) dropped: Vec<(String, String)>,
    }

    pub(super) struct FileVisitor;

    impl<'de> Visitor<'de> for FileVisitor {
        type Value = Parsed;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a JSON object")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Parsed, A::Error> {
            let mut out = Parsed {
                version: String::new(),
                runs: Vec::new(),
                dropped: Vec::new(),
            };
            while let Some(key) = map.next_key::<String>()? {
                match key.as_str() {
                    "version" => {
                        out.version = map.next_value::<Option<String>>()?.unwrap_or_default();
                    }
                    "runs" => {
                        let (runs, dropped) = map.next_value_seed(RunsSeed)?;
                        out.runs = runs;
                        out.dropped = dropped;
                    }
                    _ => {
                        map.next_value::<IgnoredAny>()?;
                    }
                }
            }
            Ok(out)
        }
    }

    type Runs = (Vec<StoredRun>, Vec<(String, String)>);

    /// `runs`: null or a list.
    struct RunsSeed;

    impl<'de> DeserializeSeed<'de> for RunsSeed {
        type Value = Runs;

        fn deserialize<D: de::Deserializer<'de>>(self, d: D) -> Result<Runs, D::Error> {
            d.deserialize_any(RunsVisitor)
        }
    }

    struct RunsVisitor;

    impl<'de> Visitor<'de> for RunsVisitor {
        type Value = Runs;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("`runs` as a list")
        }

        fn visit_unit<E: de::Error>(self) -> Result<Runs, E> {
            Ok((Vec::new(), Vec::new()))
        }

        fn visit_none<E: de::Error>(self) -> Result<Runs, E> {
            Ok((Vec::new(), Vec::new()))
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Runs, A::Error> {
            let mut runs = Vec::new();
            let mut dropped = Vec::new();
            let mut i = 0usize;
            while let Some(e) = seq.next_element::<serde_json::Value>()? {
                i += 1;
                let field = |k: &str| e.get(k).and_then(|x| x.as_str()).map(str::to_string);
                let name = match (field("model"), field("device")) {
                    (Some(m), Some(d)) => format!("#{i} ('{m}' on {d})"),
                    (Some(m), None) => format!("#{i} ('{m}')"),
                    _ => format!("#{i}"),
                };
                match serde_json::from_value::<StoredRun>(e) {
                    Ok(r) => runs.push(r),
                    Err(err) => dropped.push((name, err.to_string())),
                }
            }
            Ok((runs, dropped))
        }
    }
}

/// Merge the predictions of `new` into the file beside `results_file`. Does not fail the
/// benchmark: returns the warnings (also logged). A file that does not load is moved aside
/// (`<name>.bad-<timestamp>`) rather than overwritten; when that fails nothing is saved.
pub(crate) fn save_beside(
    results_file: &Path,
    merged: &BenchmarkResults,
    new: &[ModelResult],
) -> Vec<String> {
    let path = results_file
        .parent()
        .map(|d| d.join(PREDS_FILE))
        .unwrap_or_else(|| PathBuf::from(PREDS_FILE));
    let (prev, mut warnings) =
        match super::report::load_or_move_aside(&path, StoredPreds::load_with_warnings) {
            Ok(x) => x,
            Err(e) => {
                let w = format!("stored predictions not saved: {e:#}");
                tracing::warn!("{w}");
                return vec![w];
            }
        };
    if let Err(e) = StoredPreds::merge(prev, merged, new).save(&path) {
        let w = format!("saving the stored predictions: {e:#}");
        tracing::warn!("{w}");
        warnings.push(w);
    }
    warnings
}

/// What to search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Model names; empty = every model with stored predictions.
    pub models: Vec<String>,
    /// Device spec; None = each model's recommended device.
    pub device: Option<String>,
    /// Dataset ids; empty = all.
    pub datasets: Vec<String>,
    /// Only images carrying every one of these tags (e.g. `night`).
    pub tags: Vec<String>,
    /// Scored classes to count (model label space, e.g. `person`); empty = all.
    pub classes: Vec<String>,
    pub objective: Objective,
    /// Matching IoU.
    pub iou: f32,
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            models: Vec::new(),
            device: None,
            datasets: Vec::new(),
            tags: Vec::new(),
            classes: Vec::new(),
            objective: Objective::F1,
            iou: 0.5,
        }
    }
}

impl SearchRequest {
    /// Form fields: `model` (repeated, or `all`), `device` ("" = recommended), `dataset`, `tag`,
    /// `class` (repeated), `objective` (`f1`, `f2`, `precision:<p>`, `recall:<r>`), `iou`.
    pub fn from_form(form: &[(String, String)], default_objective: Objective) -> Result<Self> {
        let all = |k: &str| -> Vec<String> {
            form.iter()
                .filter(|(key, _)| key == k)
                .flat_map(|(_, v)| v.split(','))
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .collect()
        };
        let one = |k: &str| {
            form.iter()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v.trim())
                .filter(|v| !v.is_empty())
        };
        let mut models = all("model");
        if models.iter().any(|m| m.eq_ignore_ascii_case("all")) {
            models.clear();
        }
        let req = Self {
            models,
            device: one("device")
                .filter(|d| !d.eq_ignore_ascii_case("recommended"))
                .map(str::to_string),
            datasets: all("dataset"),
            tags: all("tag"),
            classes: all("class"),
            objective: match one("objective") {
                Some(o) => o.parse().map_err(|e: String| anyhow::anyhow!(e))?,
                None => default_objective,
            },
            iou: match one("iou") {
                Some(v) => v
                    .parse()
                    .map_err(|_| anyhow::anyhow!("iou: '{v}' is not a number"))?,
                None => 0.5,
            },
        };
        req.validate()?;
        Ok(req)
    }

    pub fn validate(&self) -> Result<()> {
        if !(self.iou.is_finite() && self.iou > 0.0 && self.iou < 1.0) {
            bail!("iou: must be between 0 and 1 (e.g. 0.5)");
        }
        if let Some(d) = &self.device {
            crate::backend::spec::parse(d).map_err(|e| anyhow::anyhow!("device: {e}"))?;
        }
        Ok(())
    }
}

/// The search result of one model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    pub model: String,
    pub device: String,
    pub objective: String,
    pub iou: f32,
    /// "ground truth" or "relative to `<reference>`, not ground truth".
    pub ground_truth: String,
    pub relative: bool,
    pub images: usize,
    /// Scored ground-truth objects.
    pub gt: usize,
    /// Classes counted.
    pub classes: Vec<String>,
    /// At the threshold the run was configured with.
    pub configured: ThresholdPoint,
    /// At the threshold configured now (the model's in the config, else the run's): what the
    /// summary's "configured" columns and the gain compare against.
    pub now: ThresholdPoint,
    /// At the model's threshold in the config now (when it differs from the run's).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current: Option<ThresholdPoint>,
    pub best: Option<Pick>,
    /// The best for each standard objective (comparison).
    pub alternatives: Vec<Pick>,
    /// At [`SWEEP_THRESHOLDS`].
    pub grid: Vec<ThresholdPoint>,
    /// The exact curve, thinned to `SEARCH_EXACT_POINTS`.
    pub exact: Vec<ThresholdPoint>,
    pub by_dataset: Vec<GroupPick>,
    pub per_class: Vec<GroupPick>,
    /// Frame-level ROC of the searched images (macro / micro / per class AUC and the curve).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roc: Option<FrameRoc>,
}

/// A search over several models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchOutput {
    pub results: Vec<SearchResult>,
    /// Models that could not be searched, and why.
    pub errors: Vec<String>,
    pub elapsed_ms: f64,
}

/// What can be searched (the page's controls).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchOptions {
    /// (model, devices with stored predictions, recommended device).
    pub models: Vec<SearchModel>,
    pub datasets: Vec<String>,
    pub tags: Vec<String>,
    pub classes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchModel {
    pub model: String,
    pub devices: Vec<String>,
    pub recommended: Option<String>,
}

/// The controls of the search page from the stored predictions and results.
pub fn options(results: &BenchmarkResults, preds: &StoredPreds) -> SearchOptions {
    let mut models: Vec<SearchModel> = Vec::new();
    let mut datasets: Vec<String> = Vec::new();
    let mut classes: Vec<String> = Vec::new();
    for r in &preds.runs {
        match models.iter_mut().find(|m| m.model == r.model) {
            Some(m) => m.devices.push(r.device.clone()),
            None => models.push(SearchModel {
                model: r.model.clone(),
                devices: vec![r.device.clone()],
                recommended: results.find(&r.model).and_then(|m| m.recommended.clone()),
            }),
        }
        for i in &r.images {
            if !datasets.contains(&i.set) {
                datasets.push(i.set.clone());
            }
        }
        for c in &r.classes {
            if !classes.contains(c) {
                classes.push(c.clone());
            }
        }
    }
    let mut tags: Vec<String> = results
        .image_sets
        .iter()
        .filter(|s| datasets.contains(&s.id))
        .flat_map(|s| s.images.iter().flat_map(|i| i.tags.iter().cloned()))
        .collect();
    tags.sort();
    tags.dedup();
    SearchOptions {
        models,
        datasets,
        tags,
        classes,
    }
}

/// The run of `model` to search: on `device`, else the recommended device, else the CPU
/// reference, else the first stored one.
fn pick_run<'a>(
    results: &BenchmarkResults,
    runs: &[&'a StoredRun],
    model: &str,
    device: Option<&str>,
) -> Result<&'a StoredRun> {
    if let Some(d) = device {
        let want = crate::backend::spec::parse(d)
            .map(|s| s.to_string())
            .unwrap_or_else(|_| d.to_string());
        return runs
            .iter()
            .find(|r| r.device == want)
            .copied()
            .with_context(|| {
                format!(
                    "no stored predictions on {want} (stored: {})",
                    runs.iter()
                        .map(|r| r.device.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            });
    }
    let rec = results.find(model).and_then(|m| m.recommended.clone());
    let reference = super::REFERENCE_DEVICES.map(|d| d.to_string());
    rec.iter()
        .chain(reference.iter())
        .find_map(|d| runs.iter().find(|r| r.device == *d).copied())
        .or_else(|| runs.first().copied())
        .context("no stored predictions")
}

/// Search the stored predictions (see the module docs). `current` gives a model's threshold in
/// the config now. Errors when there are no stored predictions at all.
pub fn search(
    results: &BenchmarkResults,
    preds: &StoredPreds,
    req: &SearchRequest,
    current: &dyn Fn(&str) -> Option<f32>,
) -> Result<SearchOutput> {
    let started = Instant::now();
    if preds.runs.is_empty() {
        bail!("no stored predictions; run a benchmark first");
    }
    req.validate()?;
    let mut names: Vec<String> = if req.models.is_empty() {
        let mut v: Vec<String> = Vec::new();
        for r in &preds.runs {
            if !v.contains(&r.model) {
                v.push(r.model.clone());
            }
        }
        v
    } else {
        req.models.clone()
    };
    // One search per model: `a,b,a` and `YOLO` + `yolo` are the same model (first spelling wins).
    let mut seen = std::collections::HashSet::new();
    names.retain(|n| seen.insert(crate::registry::normalize_name(n)));
    let sets: HashMap<&str, &ImageSetInfo> = results
        .image_sets
        .iter()
        .map(|s| (s.id.as_str(), s))
        .collect();
    let mut out = SearchOutput {
        results: Vec::new(),
        errors: Vec::new(),
        elapsed_ms: 0.0,
    };
    for name in &names {
        let runs = preds.runs_of(name);
        if runs.is_empty() {
            out.errors.push(format!(
                "{name}: no stored predictions; run a benchmark of this model first"
            ));
            continue;
        }
        let run = match pick_run(results, &runs, name, req.device.as_deref()) {
            Ok(r) => r,
            Err(e) => {
                out.errors.push(format!("{name}: {e:#}"));
                continue;
            }
        };
        match search_run(run, &sets, req, current(&run.model)) {
            Ok(r) => out.results.push(r),
            Err(e) => out.errors.push(format!("{name}: {e:#}")),
        }
    }
    out.elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
    Ok(out)
}

/// (set, image info, eval input, scored classes the set annotates) of one stored image.
type Input<'a> = (&'a ImageSetInfo, &'a ImageInfo, EvalImage, Vec<String>);

/// The scored images of a stored run matching the request's datasets and tags, restricted to
/// the `wanted` classes.
fn scored_inputs<'a>(
    run: &StoredRun,
    sets: &HashMap<&str, &'a ImageSetInfo>,
    map: &ClassMap,
    req: &SearchRequest,
    wanted: &[String],
) -> Vec<Input<'a>> {
    let mut inputs: Vec<Input<'a>> = Vec::new();
    for (k, img) in run.images.iter().enumerate() {
        if !req.datasets.is_empty() && !req.datasets.contains(&img.set) {
            continue;
        }
        let Some(set) = sets.get(img.set.as_str()).copied() else {
            continue;
        };
        let Some(info) = set.images.iter().find(|i| i.file == img.file) else {
            continue;
        };
        if !req.tags.iter().all(|t| info.tags.iter().any(|x| x == t)) {
            continue;
        }
        let Some(gt) = &info.objects else { continue };
        let (_, mut eval, allowed) =
            super::eval_input(map, set.scored_labels.as_deref(), gt, &run.preds(k));
        eval.gt.retain(|g| wanted.contains(&g.label));
        eval.preds.retain(|p| wanted.contains(&p.label));
        let mut fc = super::frame_classes(map, allowed.as_deref());
        fc.retain(|c| wanted.contains(c));
        inputs.push((set, info, eval, fc));
    }
    inputs
}

/// A stored run evaluated at one threshold, grouped like the run's own threshold sweep (see
/// [`super::threshold::ThresholdAdvice`]).
#[derive(Debug, Clone, PartialEq)]
pub struct PointsAt {
    /// Over the images of the overall score (real ground truth, else all).
    pub overall: ThresholdPoint,
    /// Overall at the run's configured threshold: equal to the advice's `configured` when the
    /// stored predictions and ground truth are the ones the run scored.
    pub at_run: ThresholdPoint,
    /// Per dataset id, per tag ([`super::threshold::SWEEP_TAGS`] and
    /// [`super::threshold::SMALL_KEY`]) and per class.
    pub by_dataset: HashMap<String, ThresholdPoint>,
    pub by_tag: HashMap<String, ThresholdPoint>,
    pub per_class: HashMap<String, ThresholdPoint>,
}

/// The exact counts of a stored run at `t` (every class, every dataset, IoU 0.5; ground truth
/// from `image_sets` of the results), overall and per group. None when nothing of the run is
/// scored.
pub fn points_at(image_sets: &[ImageSetInfo], run: &StoredRun, t: f32) -> Option<PointsAt> {
    use super::threshold::{SMALL_KEY, SWEEP_TAGS};
    let sets: HashMap<&str, &ImageSetInfo> =
        image_sets.iter().map(|s| (s.id.as_str(), s)).collect();
    let map = ClassMap::for_model(&run.classes);
    let wanted = map.classes.clone();
    let inputs = scored_inputs(run, &sets, &map, &SearchRequest::default(), &wanted);
    if inputs.is_empty() {
        return None;
    }
    let matched: Vec<SweepImage> = inputs
        .iter()
        .map(|(_, _, e, fc)| SweepImage::of(e).with_frames(e, fc))
        .collect();
    let real: Vec<usize> = inputs
        .iter()
        .enumerate()
        .filter(|(_, i)| i.0.ground_truth.is_real())
        .map(|(k, _)| k)
        .collect();
    let chosen: Vec<usize> = if real.is_empty() {
        (0..inputs.len()).collect()
    } else {
        real
    };
    let refs =
        |ks: &[usize]| -> Vec<&SweepImage> { ks.iter().filter_map(|&k| matched.get(k)).collect() };
    let overall = Breakpoints::new(&refs(&chosen), SweepScope::All);
    let mut by_dataset = HashMap::new();
    for (s, _, _, _) in &inputs {
        if by_dataset.contains_key(&s.id) {
            continue;
        }
        let ks: Vec<usize> = inputs
            .iter()
            .enumerate()
            .filter(|(_, i)| i.0.id == s.id)
            .map(|(k, _)| k)
            .collect();
        let bp = Breakpoints::new(&refs(&ks), SweepScope::All);
        by_dataset.insert(s.id.clone(), bp.point_at(t));
    }
    let mut by_tag = HashMap::new();
    for tag in SWEEP_TAGS {
        let ks: Vec<usize> = chosen
            .iter()
            .copied()
            .filter(|&k| {
                inputs
                    .get(k)
                    .is_some_and(|i| i.1.tags.iter().any(|x| x == tag))
            })
            .collect();
        if !ks.is_empty() {
            let bp = Breakpoints::new(&refs(&ks), SweepScope::All);
            by_tag.insert(tag.to_string(), bp.point_at(t));
        }
    }
    let small = Breakpoints::new(&refs(&chosen), SweepScope::Small);
    if small.npos > 0 {
        by_tag.insert(SMALL_KEY.to_string(), small.point_at(t));
    }
    let per_class = wanted
        .iter()
        .map(|c| (c, Breakpoints::new(&refs(&chosen), SweepScope::Class(c))))
        .filter(|(_, bp)| bp.npos > 0)
        .map(|(c, bp)| (c.clone(), bp.point_at(t)))
        .collect();
    Some(PointsAt {
        overall: overall.point_at(t),
        at_run: overall.point_at(run.configured),
        by_dataset,
        by_tag,
        per_class,
    })
}

/// The exact best threshold for `objective` over every scored image of a stored run (the
/// run's own overall curve, IoU 0.5): what a run with that objective would have picked. None
/// when nothing of the run is scored or the objective has no pick.
pub(crate) fn exact_overall_pick(
    image_sets: &[ImageSetInfo],
    run: &StoredRun,
    objective: Objective,
) -> Option<Pick> {
    let sets: HashMap<&str, &ImageSetInfo> =
        image_sets.iter().map(|s| (s.id.as_str(), s)).collect();
    let req = SearchRequest {
        objective,
        ..SearchRequest::default()
    };
    search_run(run, &sets, &req, None).ok()?.best
}

/// Search one stored run.
fn search_run(
    run: &StoredRun,
    sets: &HashMap<&str, &ImageSetInfo>,
    req: &SearchRequest,
    current: Option<f32>,
) -> Result<SearchResult> {
    let map = ClassMap::for_model(&run.classes);
    let wanted = wanted_classes(&map, &req.classes)?;
    let inputs = scored_inputs(run, sets, &map, req, &wanted);
    if inputs.is_empty() {
        bail!("no scored images match these filters");
    }
    let real: Vec<usize> = inputs
        .iter()
        .enumerate()
        .filter(|(_, i)| i.0.ground_truth.is_real())
        .map(|(k, _)| k)
        .collect();
    let (chosen, relative) = if real.is_empty() {
        ((0..inputs.len()).collect::<Vec<_>>(), true)
    } else {
        (real, false)
    };
    let matched: Vec<SweepImage> = inputs
        .iter()
        .map(|(_, _, e, fc)| SweepImage::with_iou(e, req.iou).with_frames(e, fc))
        .collect();
    let roc_of = |ks: &[usize], classes: &[String], points: usize| {
        FrameRoc::of(
            ks.iter()
                .filter_map(|&k| matched.get(k))
                .flat_map(|m| m.frames.iter()),
            classes,
            points,
        )
    };
    let refs =
        |ks: &[usize]| -> Vec<&SweepImage> { ks.iter().filter_map(|&k| matched.get(k)).collect() };
    let overall = Breakpoints::new(&refs(&chosen), SweepScope::All);
    if overall.npos == 0 {
        bail!("no ground-truth objects of these classes in the matching images");
    }
    // The threshold configured now (the run's when the config has none for the model).
    let now = current.unwrap_or(run.configured);
    let group = |key: &str, bp: &Breakpoints, relative: bool, roc_auc: Option<f64>| GroupPick {
        key: key.to_string(),
        images: bp.images,
        gt: bp.npos,
        relative,
        configured: bp.point_at(run.configured),
        best: exact_pick(bp, req.objective),
        roc_auc,
        now: Some(bp.point_at(now)),
    };
    let mut by_dataset = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for (s, _, _, _) in &inputs {
        if seen.contains(&s.id.as_str()) {
            continue;
        }
        seen.push(&s.id);
        let ks: Vec<usize> = inputs
            .iter()
            .enumerate()
            .filter(|(_, i)| i.0.id == s.id)
            .map(|(k, _)| k)
            .collect();
        let bp = Breakpoints::new(&refs(&ks), SweepScope::All);
        if bp.npos > 0 {
            let auc = roc_of(&ks, &wanted, 0).auc;
            by_dataset.push(group(&s.id, &bp, !s.ground_truth.is_real(), auc));
        }
    }
    let roc = roc_of(&chosen, &wanted, ROC_POINTS);
    let per_class = wanted
        .iter()
        .map(|c| (c, Breakpoints::new(&refs(&chosen), SweepScope::Class(c))))
        .filter(|(_, bp)| bp.npos > 0)
        .map(|(c, bp)| group(c, &bp, relative, roc.class_auc(c)))
        .collect();
    let ground_truth = ground_truth_desc(&inputs, &chosen, relative);
    Ok(SearchResult {
        model: run.model.clone(),
        device: run.device.clone(),
        objective: req.objective.to_string(),
        iou: req.iou,
        ground_truth,
        relative,
        images: overall.images,
        gt: overall.npos,
        classes: wanted.clone(),
        configured: overall.point_at(run.configured),
        now: overall.point_at(now),
        current: current
            .filter(|c| (c - run.configured).abs() > 1e-6)
            .map(|c| overall.point_at(c)),
        best: exact_pick(&overall, req.objective),
        alternatives: Objective::STANDARD
            .iter()
            .filter_map(|o| exact_pick(&overall, *o))
            .collect(),
        grid: SWEEP_THRESHOLDS
            .iter()
            .map(|&t| overall.point_at(t))
            .collect(),
        exact: overall.decimated(SEARCH_EXACT_POINTS),
        by_dataset,
        per_class,
        roc: (roc.positives + roc.negatives > 0).then_some(roc),
    })
}

/// The scored classes of `map` a search is limited to: `requested` (any label naming a scored
/// class, deduplicated), else every scored class.
fn wanted_classes(map: &ClassMap, requested: &[String]) -> Result<Vec<String>> {
    if requested.is_empty() {
        return Ok(map.classes.clone());
    }
    let mut v = Vec::new();
    for c in requested {
        let canon = canonical_label(c);
        let class = map
            .classes
            .iter()
            .find(|k| **k == canon)
            .or_else(|| {
                map.gt_class(&canon)
                    .and_then(|g| map.classes.iter().find(|k| *k == g))
            })
            .with_context(|| {
                format!(
                    "class '{c}' is not scored for this model (scored: {})",
                    map.classes.join(", ")
                )
            })?;
        if !v.contains(class) {
            v.push(class.clone());
        }
    }
    Ok(v)
}

/// [`SearchResult::ground_truth`] of the `chosen` inputs.
fn ground_truth_desc(inputs: &[Input], chosen: &[usize], relative: bool) -> String {
    if relative {
        let refs: std::collections::BTreeSet<String> = chosen
            .iter()
            .filter_map(|&k| match &inputs.get(k)?.0.ground_truth {
                GroundTruth::Pseudo(r) => Some(r.clone()),
                _ => None,
            })
            .collect();
        format!(
            "relative to {}, not ground truth",
            refs.into_iter().collect::<Vec<_>>().join(", ")
        )
    } else {
        "ground truth".to_string()
    }
}

/// CSV of the search summary (the `configured` columns at the threshold configured now) and
/// each model's 0.05 grid.
pub fn to_csv(out: &SearchOutput) -> String {
    let f = |v: Option<f64>| v.map_or(String::new(), |v| format!("{v:.4}"));
    let ft = |p: Option<&ThresholdPoint>| p.and_then(|p| p.frame).and_then(|f| f.tpr);
    let ff = |p: Option<&ThresholdPoint>| p.and_then(|p| p.frame).and_then(|f| f.fpr);
    let mut s = String::from(
        "model,device,objective,configured,configured_precision,configured_recall,configured_f1,best,best_precision,best_recall,best_f1,f1_gain,roc_auc,roc_auc_micro,configured_frame_tpr,configured_frame_fpr,best_frame_tpr,best_frame_fpr\n",
    );
    for r in &out.results {
        let b = r.best.as_ref();
        let roc = r.roc.as_ref();
        s.push_str(&format!(
            "{},{},{},{:.2},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            csv_field(&r.model),
            csv_field(&r.device),
            csv_field(&r.objective),
            r.now.threshold,
            f(r.now.precision),
            f(r.now.recall),
            f(r.now.f1),
            b.map_or(String::new(), |b| format!("{:.2}", b.threshold)),
            f(b.and_then(|b| b.point.precision)),
            f(b.and_then(|b| b.point.recall)),
            f(b.and_then(|b| b.point.f1)),
            f(b.and_then(|b| Some(b.point.f1? - r.now.f1?))),
            f(roc.and_then(|x| x.auc)),
            f(roc.and_then(|x| x.micro_auc)),
            f(ft(Some(&r.now))),
            f(ff(Some(&r.now))),
            f(ft(b.map(|b| &b.point))),
            f(ff(b.map(|b| &b.point))),
        ));
    }
    s.push_str(
        "\nmodel,threshold,tp,fp,fn,precision,recall,f1,f2,fp_per_image,frame_tpr,frame_fpr\n",
    );
    for r in &out.results {
        for p in &r.grid {
            s.push_str(&format!(
                "{},{:.2},{},{},{},{},{},{},{},{:.4},{},{}\n",
                csv_field(&r.model),
                p.threshold,
                p.counts.tp,
                p.counts.fp,
                p.counts.fn_,
                f(p.precision),
                f(p.recall),
                f(p.f1),
                f(p.f2),
                p.fp_per_image,
                f(ft(Some(p))),
                f(ff(Some(p))),
            ));
        }
    }
    s.push_str("\nmodel,class,positive_frames,negative_frames,roc_auc\n");
    for r in &out.results {
        for c in r.roc.iter().flat_map(|x| &x.per_class) {
            s.push_str(&format!(
                "{},{},{},{},{}\n",
                csv_field(&r.model),
                csv_field(&c.class),
                c.positives,
                c.negatives,
                f(c.auc)
            ));
        }
    }
    s
}

/// One CSV text field: quoted when it holds a separator or quote, and prefixed with `'` when a
/// spreadsheet would read it as a formula (`=`, `+`, `-`, `@`, tab or CR first).
fn csv_field(s: &str) -> String {
    let guarded;
    let s = if s.starts_with(['=', '+', '-', '@', '\t', '\r']) {
        guarded = format!("'{s}");
        guarded.as_str()
    } else {
        s
    };
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Ground truth of a stored image (tests and tools).
pub fn image_gt<'a>(results: &'a BenchmarkResults, set: &str, file: &str) -> Option<&'a [GtBox]> {
    results
        .image_set(set)?
        .images
        .iter()
        .find(|i| i.file == file)?
        .objects
        .as_deref()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::benchmark::images::SetKind;

    fn gt(label: &str, b: [f32; 4], ignore: bool) -> GtBox {
        GtBox {
            label: label.into(),
            bbox: b,
            ignore,
            crowd: false,
        }
    }

    /// Results with two datasets (one real, night-tagged; one pseudo) and stored predictions of
    /// one model on two devices.
    pub(crate) fn fixture() -> (BenchmarkResults, StoredPreds) {
        let info = |file: &str, tags: &[&str], objects: Vec<GtBox>| ImageInfo {
            file: file.into(),
            width: 640,
            height: 480,
            tags: tags.iter().map(|t| t.to_string()).collect(),
            objects: Some(objects),
        };
        let set = |id: &str, gt: GroundTruth, images: Vec<ImageInfo>| ImageSetInfo {
            id: id.into(),
            title: id.into(),
            kind: SetKind::Builtin,
            dir: None,
            license: String::new(),
            attribution: String::new(),
            source: String::new(),
            ground_truth: gt,
            scored_labels: None,
            images,
        };
        let mut results = BenchmarkResults::new(Default::default(), Default::default(), vec![]);
        results.image_sets = vec![
            set(
                "night-set",
                GroundTruth::Coco,
                vec![
                    info(
                        "a.jpg",
                        &["night"],
                        vec![
                            gt("person", [0.0, 0.0, 100.0, 100.0], false),
                            gt("car", [200.0, 200.0, 300.0, 300.0], false),
                            gt("person", [400.0, 0.0, 500.0, 100.0], true),
                        ],
                    ),
                    info(
                        "b.jpg",
                        &["day"],
                        vec![gt("person", [0.0, 0.0, 50.0, 50.0], false)],
                    ),
                ],
            ),
            set(
                "pseudo-set",
                GroundTruth::Pseudo("ref on openvino:cpu".into()),
                vec![info(
                    "c.jpg",
                    &["day"],
                    vec![gt("person", [0.0, 0.0, 100.0, 100.0], false)],
                )],
            ),
        ];
        let mut m = ModelResult::failed("m", "m.onnx", String::new());
        m.error = None;
        m.recommended = Some("ort:coreml".into());
        results.models.push(m);
        let p = |class: u32, confidence: f32, b: [f32; 4]| StoredPred {
            class,
            confidence,
            bbox: b,
        };
        let run = |device: &str, shift: f32| StoredRun {
            model: "m".into(),
            device: device.into(),
            timestamp: String::new(),
            classes: vec!["person".into(), "car".into()],
            configured: 0.5,
            eval_confidence: 0.05,
            images: vec![
                StoredImage {
                    set: "night-set".into(),
                    file: "a.jpg".into(),
                    preds: vec![
                        p(0, 0.9 - shift, [0.0, 0.0, 100.0, 100.0]), // TP person
                        p(1, 0.4, [200.0, 200.0, 300.0, 300.0]),     // TP car
                        p(0, 0.7, [400.0, 0.0, 500.0, 100.0]),       // ignore region
                        p(0, 0.3, [600.0, 300.0, 640.0, 400.0]),     // FP
                    ],
                },
                StoredImage {
                    set: "night-set".into(),
                    file: "b.jpg".into(),
                    preds: vec![p(0, 0.6, [0.0, 0.0, 50.0, 60.0])], // IoU 0.83: TP at 0.5
                },
                StoredImage {
                    set: "pseudo-set".into(),
                    file: "c.jpg".into(),
                    preds: vec![p(0, 0.8, [0.0, 0.0, 100.0, 100.0])],
                },
            ],
        };
        let preds = StoredPreds {
            version: "test".into(),
            runs: vec![run("openvino:cpu", 0.0), run("ort:coreml", 0.05)],
        };
        (results, preds)
    }

    #[test]
    fn wanted_classes_resolve_dedupe_and_reject() {
        let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let coco = ClassMap::for_model(&names(&["person", "car", "dog"]));
        assert_eq!(wanted_classes(&coco, &[]).unwrap(), coco.classes);
        assert_eq!(
            wanted_classes(&coco, &names(&["car", "person", "car"])).unwrap(),
            ["car", "person"]
        );
        let err = wanted_classes(&coco, &names(&["zebra"])).unwrap_err();
        assert_eq!(
            format!("{err:#}"),
            format!(
                "class 'zebra' is not scored for this model (scored: {})",
                coco.classes.join(", ")
            )
        );
        // A ground-truth label of a merged model class names that class.
        let ipcam = ClassMap::for_model(&names(&["person", "vehicle"]));
        assert_eq!(
            wanted_classes(&ipcam, &names(&["car"])).unwrap(),
            ["vehicle"]
        );
    }

    #[test]
    fn search_hand_computed_and_filters() {
        let (results, preds) = fixture();
        let none = |_: &str| None;
        let out = search(&results, &preds, &SearchRequest::default(), &none).unwrap();
        assert!(out.errors.is_empty(), "{:?}", out.errors);
        let r = &out.results[0];
        // Recommended device; only the real-ground-truth set counts overall.
        assert_eq!(r.device, "ort:coreml");
        assert!(!r.relative);
        assert_eq!((r.images, r.gt), (2, 3));
        // At 0.5: person .85 TP, .7 ignored, b .6 TP; car .4 and FP .3 below -> 2 TP, 0 FP, 1 FN.
        let c = r.configured.counts;
        assert_eq!((c.tp, c.fp, c.fn_), (2, 0, 1));
        // F1 by kept set: down to .6: 2/(2+0+1) -> .8; +car .4: 1.0; +FP .3: 6/7.
        let b = r.best.as_ref().unwrap();
        assert_eq!(b.exact_threshold, Some(0.4));
        assert_eq!(b.plateau, Some([0.3, 0.4]));
        assert_eq!(b.threshold, 0.35);
        assert_eq!(b.point.f1, Some(1.0));
        assert_eq!(r.grid.len(), 19);
        assert!(!r.exact.is_empty());
        assert_eq!(r.by_dataset.len(), 2);
        assert!(r.by_dataset[1].relative);
        let classes: Vec<&str> = r.per_class.iter().map(|g| g.key.as_str()).collect();
        assert_eq!(classes, ["person", "car"]);

        // Tag filter: night only (a.jpg): person TP, car TP, FP.
        let req = SearchRequest {
            tags: vec!["night".into()],
            device: Some("openvino:cpu".into()),
            ..Default::default()
        };
        let r = &search(&results, &preds, &req, &none).unwrap().results[0];
        assert_eq!((r.device.as_str(), r.images, r.gt), ("openvino:cpu", 1, 2));
        // Class subset: cars only.
        let req = SearchRequest {
            classes: vec!["Car".into()],
            ..Default::default()
        };
        let r = &search(&results, &preds, &req, &none).unwrap().results[0];
        assert_eq!((r.gt, r.classes.clone()), (1, vec!["car".to_string()]));
        // Pseudo set only: relative.
        let req = SearchRequest {
            datasets: vec!["pseudo-set".into()],
            ..Default::default()
        };
        let r = &search(&results, &preds, &req, &none).unwrap().results[0];
        assert!(r.relative && r.ground_truth.contains("ref on openvino:cpu"));
        // Stricter IoU: b.jpg (IoU 0.83) no longer matches at 0.9.
        let req = SearchRequest {
            iou: 0.9,
            ..Default::default()
        };
        let r = &search(&results, &preds, &req, &none).unwrap().results[0];
        assert_eq!(r.configured.counts.fp, 1);
        // The config's current threshold is evaluated too.
        let cur = |_: &str| Some(0.3);
        let r = &search(&results, &preds, &SearchRequest::default(), &cur)
            .unwrap()
            .results[0];
        assert_eq!(r.current.unwrap().counts.fp, 1);

        // Errors: unknown model / device / class / nothing left; no stored predictions at all.
        for req in [
            SearchRequest {
                models: vec!["zzz".into()],
                ..Default::default()
            },
            SearchRequest {
                device: Some("ort:cpu".into()),
                ..Default::default()
            },
            SearchRequest {
                classes: vec!["dog".into()],
                ..Default::default()
            },
            SearchRequest {
                tags: vec!["fog".into()],
                ..Default::default()
            },
        ] {
            let out = search(&results, &preds, &req, &none).unwrap();
            assert!(out.results.is_empty() && out.errors.len() == 1, "{req:?}");
        }
        let err = search(
            &results,
            &StoredPreds::default(),
            &SearchRequest::default(),
            &none,
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("run a benchmark first"));
        let csv = to_csv(&out_of(&results, &preds));
        assert!(csv.starts_with("model,device,objective,configured"));
        assert!(csv.contains("\nm,0.05,"), "{csv}");
    }

    /// Frame ROC and the frame objectives from the stored predictions alone.
    #[test]
    fn search_frame_roc_and_youden() {
        let (results, preds) = fixture();
        let none = |_: &str| None;
        let req = SearchRequest {
            objective: Objective::Youden,
            ..Default::default()
        };
        let r = &search(&results, &preds, &req, &none).unwrap().results[0];
        // Frames (real set, coreml): a person+ .85 (max of .85, the .7 in the ignore region and
        // the .3 FP), a car+ .4, b person+ .6, b car- 0. Person has no negative frame:
        // undefined; car 1.0.
        let roc = r.roc.as_ref().unwrap();
        let person = roc.per_class.iter().find(|c| c.class == "person").unwrap();
        assert_eq!(
            (person.positives, person.negatives, person.auc),
            (2, 0, None)
        );
        assert_eq!(roc.class_auc("car"), Some(1.0));
        assert_eq!((roc.auc, roc.micro_auc), (Some(1.0), Some(1.0)));
        assert_eq!((roc.positives, roc.negatives), (3, 1));
        let b = r.best.as_ref().unwrap();
        assert_eq!((b.threshold, b.exact_threshold), (0.22, Some(0.4)));
        assert_eq!(b.plateau, Some([0.05, 0.4]));
        let f = b.point.frame.unwrap();
        assert_eq!((f.tpr, f.fpr), (Some(1.0), Some(0.0)));
        // At the configured 0.5 the car frame (.4) does not alert: TPR 2/3.
        let c = r.configured.frame.unwrap();
        assert_eq!((c.tp, c.positives), (2, 3));
        // Youden is among the alternatives of every search.
        assert!(r.alternatives.iter().any(|p| p.objective == "youden"));
        // Per class / per dataset AUC.
        let car = r.per_class.iter().find(|g| g.key == "car").unwrap();
        assert_eq!(car.roc_auc, Some(1.0));
        assert_eq!(r.by_dataset[0].roc_auc, Some(1.0));
        // Class filter: person only has no negative frames -> no Youden pick, no AUC.
        let req = SearchRequest {
            objective: Objective::Youden,
            classes: vec!["person".into()],
            ..Default::default()
        };
        let r = &search(&results, &preds, &req, &none).unwrap().results[0];
        assert!(r.best.is_none());
        assert_eq!(r.roc.as_ref().unwrap().auc, None);
        // fpr:0 -> the highest TPR without false alerts: everything (the negative scores 0).
        let req = SearchRequest {
            objective: Objective::Fpr(0.0),
            ..Default::default()
        };
        let r = &search(&results, &preds, &req, &none).unwrap().results[0];
        let f = r.best.as_ref().unwrap().point.frame.unwrap();
        assert_eq!((f.tpr, f.fpr), (Some(1.0), Some(0.0)));
        // CSV carries the AUC and the frame rates.
        let csv = to_csv(&out_of(&results, &preds));
        assert!(csv.contains(",roc_auc,roc_auc_micro,"), "{csv}");
        assert!(csv.contains("\nm,car,1,1,1.0000\n"), "{csv}");
    }

    /// BQ-13: repeated or differently-cased model names search once.
    #[test]
    fn duplicate_model_names_search_once() {
        let (results, preds) = fixture();
        for names in [["m", "m", "m"], ["m", "M", "m"]] {
            let req = SearchRequest {
                models: names.iter().map(|n| n.to_string()).collect(),
                ..Default::default()
            };
            let out = search(&results, &preds, &req, &|_| None).unwrap();
            assert_eq!(out.results.len(), 1, "{names:?}");
            assert!(out.errors.is_empty());
        }
    }

    /// BQ-13: CSV text fields are quoted and neutralized against spreadsheet formulas.
    #[test]
    fn csv_fields_are_escaped_and_formula_safe() {
        assert_eq!(csv_field("plain"), "plain");
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("=cmd()"), "'=cmd()");
        assert_eq!(csv_field("+1"), "'+1");
        assert_eq!(csv_field("-1"), "'-1");
        assert_eq!(csv_field("@SUM(A1)"), "'@SUM(A1)");
        assert_eq!(csv_field("\tx"), "'\tx");
        assert_eq!(csv_field("=a,b"), "\"'=a,b\"");
        let (results, preds) = fixture();
        let mut out = out_of(&results, &preds);
        out.results[0].device = "=evil,x".into();
        out.results[0].objective = "@o".into();
        let csv = to_csv(&out);
        assert!(csv.contains(",\"'=evil,x\",'@o,"), "{csv}");
    }

    fn out_of(results: &BenchmarkResults, preds: &StoredPreds) -> SearchOutput {
        search(results, preds, &SearchRequest::default(), &|_| None).unwrap()
    }

    #[test]
    fn form_parsing() {
        let f = |pairs: &[(&str, &str)]| -> Vec<(String, String)> {
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        };
        let r = SearchRequest::from_form(
            &f(&[
                ("model", "a"),
                ("model", "b"),
                ("device", ""),
                ("dataset", "x,y"),
                ("tag", "night"),
                ("objective", "recall:0.9"),
                ("iou", "0.6"),
            ]),
            Objective::F1,
        )
        .unwrap();
        assert_eq!(r.models, ["a", "b"]);
        assert_eq!(r.device, None);
        assert_eq!(r.datasets, ["x", "y"]);
        assert_eq!(r.objective, Objective::Recall(0.9));
        assert_eq!(r.iou, 0.6);
        let r = SearchRequest::from_form(&f(&[("model", "all")]), Objective::F2).unwrap();
        assert!(r.models.is_empty());
        assert_eq!(r.objective, Objective::F2);
        for bad in [
            ("iou", "1.2"),
            ("objective", "f9"),
            ("device", "vulkan"),
            ("iou", "x"),
        ] {
            assert!(
                SearchRequest::from_form(&f(&[bad]), Objective::F1).is_err(),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn stored_predictions_round_trip_and_merge() {
        let (results, preds) = fixture();
        let dir = std::env::temp_dir().join(format!("bop-preds-{}", uuid::Uuid::new_v4()));
        let path = dir.join(PREDS_FILE);
        assert!(StoredPreds::load(&path).unwrap().is_none());
        preds.save(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("[0,0.9,0,0,100,100]"), "{text}");
        let back = StoredPreds::load(&path).unwrap().unwrap();
        assert_eq!(back, preds);
        // The search over the loaded file equals the one over the original.
        assert_eq!(
            out_of(&results, &back).results,
            out_of(&results, &preds).results
        );
        // Merge: a model no longer in the results is dropped; a re-run model is replaced.
        let mut other = preds.runs[0].clone();
        other.model = "gone".into();
        let prev = StoredPreds {
            version: "x".into(),
            runs: vec![preds.runs[0].clone(), other],
        };
        let merged = StoredPreds::merge(Some(prev.clone()), &results, &[]);
        assert_eq!(merged.runs.len(), 1);
        assert_eq!(merged.runs[0].model, "m");
        let rerun = results.models[0].clone();
        let merged = StoredPreds::merge(Some(prev), &results, &[rerun]);
        assert!(
            merged.runs.is_empty(),
            "re-run without stored runs replaces the old ones"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unloadable_preds_are_moved_aside_and_bad_runs_dropped() {
        let (results, preds) = fixture();
        let dir = std::env::temp_dir().join(format!("bop-preds-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(PREDS_FILE);
        preds.save(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();

        // One bad run: dropped with a warning naming it; the others load.
        let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
        v["runs"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"model": "broken", "device": "ort:cpu", "configured": "x"}));
        std::fs::write(&path, v.to_string()).unwrap();
        let (back, warnings) = StoredPreds::load_with_warnings(&path).unwrap();
        assert_eq!(back.unwrap().runs, preds.runs);
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].contains("'broken' on ort:cpu"),
            "{}",
            warnings[0]
        );

        // Truncated: moved aside before the new file is written.
        std::fs::write(&path, &text[..text.len() / 2]).unwrap();
        let warnings = save_beside(&dir.join("benchmark.json"), &results, &[]);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains(".bad-"), "{}", warnings[0]);
        let aside = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with(&format!("{PREDS_FILE}.bad-"))
            })
            .count();
        assert_eq!(aside, 1);
        assert!(
            StoredPreds::load(&path).unwrap().is_some(),
            "a new file was saved"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn stored_preds_keep_the_most_confident_in_order() {
        let p = |c: f32| StoredPred {
            class: 0,
            confidence: c,
            bbox: [c, 0.0, 1.0, 1.0],
        };
        let conf = |v: &[StoredPred]| v.iter().map(|p| p.confidence).collect::<Vec<_>>();
        let few = vec![p(0.1), p(0.9)];
        assert_eq!(most_confident(few.clone(), 2), few);
        // Top 3 of 0.2 .9 .5 .9 .1 .7: .9 .9 .7, in their original order; ties keep the first.
        let v = vec![p(0.2), p(0.9), p(0.5), p(0.9), p(0.1), p(0.7)];
        assert_eq!(conf(&most_confident(v, 3)), [0.9, 0.9, 0.7]);
        let v = vec![p(0.5), p(0.6), p(0.6), p(0.6)];
        let kept = most_confident(v, 2);
        assert_eq!(conf(&kept), [0.6, 0.6]);
        assert_eq!(kept.len(), 2);
        // NaN goes last.
        assert_eq!(conf(&most_confident(vec![p(f32::NAN), p(0.3)], 1)), [0.3]);
        // The per-image cap applies to previously stored runs on merge.
        let many: Vec<StoredPred> = (0..250).map(|i| p(i as f32 / 1000.0)).collect();
        let (results, mut preds) = fixture();
        preds.runs[0].images[0].preds = many;
        let merged = StoredPreds::merge(Some(preds), &results, &[]);
        let img = &merged.runs[0].images[0].preds;
        assert_eq!(img.len(), MAX_STORED_PREDS_PER_IMAGE);
        assert_eq!(img[0].confidence, 0.15);
        assert_eq!(img.last().unwrap().confidence, 0.249);
    }

    #[test]
    fn stored_preds_cache_hits_until_the_file_changes() {
        let dir = std::env::temp_dir().join(format!("bop-preds-cache-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(PREDS_FILE);
        let cache: &'static PredsCache = Box::leak(Box::new(IdleCache::new(CACHE_IDLE)));
        assert!(StoredPreds::load_cached_in(cache, &path).unwrap().is_none());
        let (_, preds) = fixture();
        preds.save(&path).unwrap();
        let a = StoredPreds::load_cached_in(cache, &path).unwrap().unwrap();
        assert_eq!(*a, preds);
        let b = StoredPreds::load_cached_in(cache, &path).unwrap().unwrap();
        assert!(Arc::ptr_eq(&a, &b), "same stamp: no re-parse");
        // A changed file (other length) is parsed again.
        let mut fewer = preds.clone();
        fewer.runs.truncate(1);
        fewer.save(&path).unwrap();
        let c = StoredPreds::load_cached_in(cache, &path).unwrap().unwrap();
        assert!(!Arc::ptr_eq(&a, &c));
        assert_eq!(*c, fewer);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// M10: an entry nobody reads goes after the idle time, a read keeps it, and the thread that
    /// drops it ends with it.
    #[test]
    fn an_idle_cache_entry_is_dropped_and_a_read_keeps_it() {
        let cache: &'static IdleCache<u32, Arc<String>> =
            Box::leak(Box::new(IdleCache::new(Duration::from_secs(600))));
        let v = Arc::new("parsed".to_string());
        cache.put(1, v.clone());
        assert_eq!(cache.get(&2), None, "another key");
        let t0 = Instant::now();
        assert!(cache.get(&1).is_some());
        assert!(cache.sweep(t0 + Duration::from_secs(599)), "read just now");
        assert!(!cache.is_empty());
        assert!(!cache.sweep(t0 + Duration::from_secs(1200)));
        assert!(cache.is_empty());
        assert_eq!(Arc::strong_count(&v), 1, "the cache let go of it");
        assert_eq!(cache.get(&1), None);
        // Under the lock: a compute for the same key happens once.
        let made = std::sync::atomic::AtomicUsize::new(0);
        let make = || {
            made.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Arc::new("x".to_string())
        };
        cache.get_or_insert_with(3, make);
        cache.get_or_insert_with(3, make);
        assert_eq!(made.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    /// The real thread drops an idle entry and ends; a later put starts another.
    #[test]
    fn the_idle_thread_drops_the_entry_on_its_own() {
        let cache: &'static IdleCache<u32, u32> =
            Box::leak(Box::new(IdleCache::new(Duration::from_millis(40))));
        for round in 0..2 {
            cache.put(round, round);
            let until = Instant::now() + Duration::from_secs(30);
            while !cache.is_empty() {
                assert!(Instant::now() < until, "round {round}: never dropped");
                std::thread::sleep(Duration::from_millis(5));
            }
            let until = Instant::now() + Duration::from_secs(30);
            while cache.state.lock_recover().sweeping {
                assert!(
                    Instant::now() < until,
                    "round {round}: the thread never ended"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    /// The streamed read keeps the lenient rules: a bad run is dropped by name, other keys and a
    /// null list are fine, a missing file is None and a broken one an error naming it.
    #[test]
    fn stored_preds_stream_leniently() {
        let dir = std::env::temp_dir().join(format!("bop-preds-stream-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(PREDS_FILE);
        assert!(StoredPreds::load_with_warnings(&path).unwrap().0.is_none());
        let (_, preds) = fixture();
        preds.save(&path).unwrap();
        let (back, w) = StoredPreds::load_with_warnings(&path).unwrap();
        assert_eq!(back.unwrap(), preds);
        assert!(w.is_empty(), "{w:?}");
        let mut v = serde_json::to_value(&preds).unwrap();
        v["extra"] = serde_json::json!({"nested": [1, 2]});
        v["runs"].as_array_mut().unwrap().insert(
            0,
            serde_json::json!({"model": "bad", "device": "cpu", "images": 7}),
        );
        std::fs::write(&path, v.to_string()).unwrap();
        let (back, w) = StoredPreds::load_with_warnings(&path).unwrap();
        assert_eq!(back.as_ref().unwrap(), &preds);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(
            w[0].contains("dropped `runs` entry #1 ('bad' on cpu)"),
            "{}",
            w[0]
        );
        // The same as the whole-text reader it replaced.
        let text = std::fs::read_to_string(&path).unwrap();
        let (head, runs, old) =
            super::super::report::parse_lenient::<StoredPreds, StoredRun>(&text, "runs", &path)
                .unwrap();
        assert_eq!(StoredPreds { runs, ..head }, back.unwrap());
        assert_eq!(old, w);
        std::fs::write(&path, r#"{"version":"v","runs":null}"#).unwrap();
        let back = StoredPreds::load_with_warnings(&path).unwrap().0.unwrap();
        assert_eq!((back.version.as_str(), back.runs.len()), ("v", 0));
        for broken in [r#"{"runs":[{"#, r#"{"runs":5}"#, "[1]", r#"{"runs":[]} x"#] {
            std::fs::write(&path, broken).unwrap();
            let err = format!("{:#}", StoredPreds::load_with_warnings(&path).unwrap_err());
            assert!(
                err.contains("parsing") && err.contains(PREDS_FILE),
                "{broken}: {err}"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
