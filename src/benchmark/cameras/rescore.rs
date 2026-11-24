//! Re-score "Your cameras" against the labels as they are now, without inference
//! (`POST /v1/benchmark/rescore`, `benchmark --rescore`): every stored run of the set is scored
//! again from its stored predictions (`benchmark-preds.json`) against the current reviewed frames,
//! exactly as a run scores ([`crate::benchmark::scoring`]), and the snapshot is replaced.
//!
//! - A scored frame whose labels changed is scored against its new labels.
//! - A frame that left the reviewed set (reopened, skipped, deleted) leaves the set and every run.
//! - A frame the walk could not read this time (changed while read, a label file or sidecar that
//!   does not read, a lost image or sidecar, a folder that could not be listed:
//!   [`crate::collect::labels::export::Selection::unsure`] / `unreadable_dirs`) is unknown, not
//!   gone: it keeps its place, its snapshot entry and its labels, so the runs keep its figures;
//!   it is listed ([`Rescored::unknown`]).
//! - A frame reviewed since the run has no predictions: it is counted as needing a run
//!   ([`Rescored::need_run`]), never guessed, and the new snapshot still counts it as new.
//! - A device run without stored predictions for every one of its images keeps its figures and
//!   is listed ([`Rescored::not_rescored`]).
//! - A snapshot of another folder under review or another class list (`collect.review_dir`,
//!   `collect.classes` changed since the run) is refused ([`SourceChanged`]; the route answers
//!   409): its frames are of another set, so the benchmark has to run again.
//!
//! The other sets of a run keep their labels; each model's recommendation, threshold advice and
//! grades are recomputed. Under the results lock ([`crate::benchmark::results_io::lock`], as a
//! [`crate::benchmark::results_io::Holder::Rescore`]): the snapshot is written first, then
//! `benchmark.json` with the snapshot's identity ([`super::snapshot::stamp`]).

use super::super::images::{ImageSet, SetImage};
use super::super::report::{BenchmarkResults, ImageInfo, ImageSetInfo};
use super::super::results_io::{self, Holder, ResultsLock};
use super::super::search::StoredPreds;
use super::snapshot::{self, GtSnapshot};
use super::{ID, Source, TITLE};
use crate::config::Config;
use anyhow::{Result, bail};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;

/// What a re-score did.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Rescored {
    /// Model results with a device run re-scored.
    pub models: usize,
    /// Device runs re-scored.
    pub runs: usize,
    /// Frames of the set now.
    pub frames: usize,
    /// What had changed since the snapshot.
    pub changes: snapshot::Changes,
    /// Frames reviewed since the run, without predictions: a run scores them.
    pub need_run: usize,
    /// `<model> on <device>` runs that could not be re-scored (no stored predictions).
    pub not_rescored: Vec<String>,
    /// Scored frames the walk could not read this time (see the module docs): kept as they
    /// were, file names.
    pub unknown: Vec<String>,
}

/// The snapshot is of another folder or class list than the one under review now: re-scoring
/// would mix two sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceChanged(pub String);

impl std::fmt::Display for SourceChanged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SourceChanged {}

/// The set `info` describes, as a run holds it (labels and tags; no file is read).
fn image_set_of(info: &ImageSetInfo) -> ImageSet {
    ImageSet {
        id: info.id.clone(),
        title: info.title.clone(),
        kind: info.kind,
        dir: info.dir.clone(),
        license: info.license.clone(),
        attribution: info.attribution.clone(),
        source: info.source.clone(),
        ground_truth: info.ground_truth.clone(),
        scored_labels: info.scored_labels.clone(),
        transform: None,
        images: info
            .images
            .iter()
            .map(|i| SetImage {
                file: i.file.clone(),
                path: None,
                width: i.width,
                height: i.height,
                tags: i.tags.clone(),
                gt: i.objects.clone(),
            })
            .collect(),
    }
}

/// The cameras set of `results` against `now` (the current selection's set): frames still
/// reviewed keep their place, scene and group with the current labels, tags and size; frames
/// `unknown` says the walk could not read stay as they were.
fn rebuilt(old: &ImageSetInfo, now: &ImageSet, unknown: &dyn Fn(&str) -> bool) -> ImageSetInfo {
    let current: HashMap<&str, &SetImage> =
        now.images.iter().map(|i| (i.file.as_str(), i)).collect();
    let mut out = old.clone();
    out.scored_labels.clone_from(&now.scored_labels);
    out.dir.clone_from(&now.dir);
    out.images = old
        .images
        .iter()
        .filter_map(|i| {
            let Some(c) = current.get(i.file.as_str()) else {
                return unknown(&i.file).then(|| i.clone());
            };
            Some(ImageInfo {
                file: i.file.clone(),
                width: c.width,
                height: c.height,
                tags: c.tags.clone(),
                objects: c.gt.clone(),
                scene: i.scene,
                group: super::group_of(&c.tags),
            })
        })
        .collect();
    out
}

/// Re-score the device runs of `results` that ran images of the cameras set, against its new
/// images (already in `results`); `gone`: file names that left the set. Returns (models, runs,
/// runs not re-scored).
fn rescore_runs(
    results: &mut BenchmarkResults,
    preds: Option<&StoredPreds>,
    gone: &std::collections::HashSet<String>,
) -> (usize, usize, Vec<String>) {
    let sets: Vec<ImageSet> = results.image_sets.iter().map(image_set_of).collect();
    let by_key: HashMap<(&str, &str), &SetImage> = sets
        .iter()
        .flat_map(|s| {
            s.images
                .iter()
                .map(move |i| ((s.id.as_str(), i.file.as_str()), i))
        })
        .collect();
    let set_of: HashMap<&str, &ImageSet> = sets.iter().map(|s| (s.id.as_str(), s)).collect();
    let (mut models, mut runs, mut skipped) = (0, 0, Vec::new());
    for m in &mut results.models {
        let stored = preds.map(|p| p.runs_of(&m.model)).unwrap_or_default();
        let mut touched = false;
        for d in &mut m.devices {
            let device = d.device.clone();
            let ok = d.ok();
            let Some(run) = d.run.as_mut().filter(|_| ok) else {
                continue;
            };
            if !run.per_image.iter().any(|i| i.set == ID) {
                continue;
            }
            let Some(acc) = run.accuracy.as_ref() else {
                continue;
            };
            let Some(s) = stored.iter().find(|s| s.device == device) else {
                skipped.push(format!("{} on {device}", m.model));
                continue;
            };
            let index: HashMap<(&str, &str), usize> = s
                .images
                .iter()
                .enumerate()
                .map(|(k, i)| ((i.set.as_str(), i.file.as_str()), k))
                .collect();
            let kept: Vec<_> = run
                .per_image
                .iter()
                .filter(|i| !(i.set == ID && gone.contains(&i.file)))
                .cloned()
                .collect();
            let mut all = Vec::with_capacity(kept.len());
            let mut eval_preds = Vec::with_capacity(kept.len());
            let complete = kept.iter().all(|i| {
                let key = (i.set.as_str(), i.file.as_str());
                match (set_of.get(key.0), by_key.get(&key), index.get(&key)) {
                    (Some(set), Some(img), Some(k)) => {
                        all.push((*set, *img));
                        eval_preds.push(s.preds(*k));
                        true
                    }
                    _ => false,
                }
            });
            if !complete {
                skipped.push(format!("{} on {device}", m.model));
                continue;
            }
            let job = super::super::Job {
                name: m.model.clone(),
                path: std::path::PathBuf::from(&m.path),
                family: Default::default(),
                classes: s.classes.clone(),
                device: device.clone(),
                gpu_precision: None,
                confidence_threshold: None,
                object_filter: None,
                variant: None,
                concurrency: None,
                device_thresholds: Default::default(),
            };
            let threshold = acc.threshold;
            let mut per_image = kept;
            let (accuracy, note) = super::super::scoring::score(
                &job,
                &all,
                &mut per_image,
                &eval_preds,
                threshold,
                m.threshold_objective,
            );
            run.tag_recall = if accuracy.is_none() {
                super::super::scoring::tag_recall_only(&job, &all, &eval_preds, threshold)
            } else {
                Vec::new()
            };
            run.accuracy = accuracy;
            run.accuracy_note = note;
            run.images = per_image.len();
            run.per_image = per_image;
            runs += 1;
            touched = true;
        }
        if touched {
            models += 1;
            m.finish();
        }
    }
    (models, runs, skipped)
}

/// Re-score the saved results beside `results_path` (stored predictions at `preds_path`) against
/// the current labels of `config`'s review folder, and write the new snapshot and results; takes
/// the results lock ([`rescore_locked`] for a caller that holds it).
pub fn rescore(config: &Config, results_path: &Path, preds_path: &Path) -> Result<Rescored> {
    let held = results_io::lock(results_path, Holder::Rescore)?;
    rescore_locked(&held, config, results_path, preds_path)
}

/// [`rescore`] under `_held`, the caller's hold of the results lock (the route checks under it
/// that no run is active).
pub fn rescore_locked(
    _held: &ResultsLock,
    config: &Config,
    results_path: &Path,
    preds_path: &Path,
) -> Result<Rescored> {
    let Some(mut results) = BenchmarkResults::load(results_path)? else {
        bail!("no benchmark results yet; run the benchmark on {TITLE} first");
    };
    let gt_path = snapshot::path_beside(results_path);
    let Some(old_snap) = GtSnapshot::load(&gt_path)? else {
        bail!(
            "{} is missing: there are no stored labels of {TITLE} to compare with; run the \
             benchmark on {TITLE} again",
            gt_path.display()
        );
    };
    let Some(old_set) = results.image_sets.iter().position(|s| s.id == ID) else {
        bail!("the saved results hold no run on {TITLE}; run the benchmark on it first");
    };
    let src = Source::of(config);
    if old_snap.dir != src.dir || old_snap.classes != src.classes {
        let what = if old_snap.dir != src.dir {
            format!(
                "the folder under review is {} now; the run scored {}",
                src.dir.display(),
                old_snap.dir.display()
            )
        } else {
            format!(
                "the class list is {} now; the run scored {}",
                src.classes.join(", "),
                old_snap.classes.join(", ")
            )
        };
        return Err(SourceChanged(format!(
            "{TITLE} cannot be re-scored: {what}. Its stored predictions are of another set: run \
             the benchmark on {TITLE} again"
        ))
        .into());
    }
    let (frames, sel) = super::select(&src).map_err(|why| anyhow::anyhow!("{why}"))?;
    let now = super::set_of(&src, &frames);
    let old_frames = old_snap.by_file();
    // A frame the walk could not read: in its list, or inside a folder it could not list.
    let unknown = |file: &str| {
        !sel.revisions.contains_key(file)
            && (sel.unsure.contains(file)
                || old_frames.get(file).is_some_and(|f| {
                    let at = src.dir.join(&f.frame);
                    sel.unreadable_dirs.iter().any(|d| at.starts_with(d))
                }))
    };
    let changes = snapshot::changes_with(&old_snap, &sel.revisions, &|f| unknown(&f.file));
    let (mut new_info, gone) = {
        let old = results
            .image_sets
            .get(old_set)
            .ok_or_else(|| anyhow::anyhow!("the saved results changed while re-scoring"))?;
        let info = rebuilt(old, &now, &unknown);
        let kept: std::collections::HashSet<&str> =
            info.images.iter().map(|i| i.file.as_str()).collect();
        let gone: std::collections::HashSet<String> = old
            .images
            .iter()
            .filter(|i| !kept.contains(i.file.as_str()))
            .map(|i| i.file.clone())
            .collect();
        (info, gone)
    };
    let kept_unknown: Vec<String> = new_info
        .images
        .iter()
        .filter(|i| !sel.revisions.contains_key(&i.file))
        .map(|i| i.file.clone())
        .collect();
    let new_snap = {
        let scored = image_set_of(&new_info);
        let mut snap = GtSnapshot::of(&src, &scored, &sel.revisions);
        // Frames reviewed since the run stay new: they still need a run.
        snap.reviewed
            .retain(|k, _| old_snap.reviewed.contains_key(k));
        // Unknown frames keep their entries as they were.
        for f in &mut snap.frames {
            if let Some(old) = old_frames.get(f.file.as_str())
                && !sel.revisions.contains_key(&f.file)
            {
                f.revision.clone_from(&old.revision);
                if let Some(r) = old_snap.reviewed.get(&f.file) {
                    snap.reviewed.insert(f.file.clone(), r.clone());
                }
            }
        }
        snap
    };
    // QA-050: every read that can fail comes before the first write, so a damaged preds file
    // leaves the snapshot and the results as they were (not a new snapshot beside old results).
    let preds = StoredPreds::load(preds_path)?;
    let id = new_snap.save(&gt_path)?;
    new_info.snapshot = Some(id);
    new_info.labels_stale = None;
    if let Some(slot) = results.image_sets.get_mut(old_set) {
        *slot = new_info;
    }
    let (models, runs, not_rescored) = rescore_runs(&mut results, preds.as_ref(), &gone);
    results.regrade(config.benchmark.accuracy_metric, config.benchmark.weights);
    let frames_now = results.image_set(ID).map_or(0, |s| s.images.len());
    results.save(results_path)?;
    Ok(Rescored {
        models,
        runs,
        frames: frames_now,
        need_run: changes.joined,
        changes,
        not_rescored,
        unknown: kept_unknown,
    })
}
