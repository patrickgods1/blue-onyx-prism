//! Re-score "Your cameras" against the labels as they are now, without inference
//! (`POST /v1/benchmark/rescore`, `benchmark --rescore`): every stored run of the set is scored
//! again from its stored predictions (`benchmark-preds.json`) against the current reviewed frames,
//! exactly as a run scores ([`crate::benchmark::scoring`]), and the snapshot is replaced.
//!
//! - A scored frame whose labels changed is scored against its new labels.
//! - A frame that left the reviewed set (reopened, skipped, deleted) leaves the set and every run.
//! - A frame reviewed since the run has no predictions: it is counted as needing a run
//!   ([`Rescored::need_run`]), never guessed, and the new snapshot still counts it as new.
//! - A device run without stored predictions for every one of its images keeps its figures and
//!   is listed ([`Rescored::not_rescored`]).
//!
//! The other sets of a run keep their labels; each model's recommendation, threshold advice and
//! grades are recomputed. The snapshot is written first, then `benchmark.json`.

use super::super::images::{ImageSet, SetImage};
use super::super::report::{BenchmarkResults, ImageInfo, ImageSetInfo};
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
}

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
/// reviewed keep their place, scene and group with the current labels, tags and size.
fn rebuilt(old: &ImageSetInfo, now: &ImageSet) -> ImageSetInfo {
    let current: HashMap<&str, &SetImage> =
        now.images.iter().map(|i| (i.file.as_str(), i)).collect();
    let mut out = old.clone();
    out.scored_labels.clone_from(&now.scored_labels);
    out.images = old
        .images
        .iter()
        .filter_map(|i| {
            let c = current.get(i.file.as_str())?;
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
/// the current labels of `config`'s review folder, and write the new snapshot and results.
pub fn rescore(config: &Config, results_path: &Path, preds_path: &Path) -> Result<Rescored> {
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
    let (frames, sel) = super::select(&src).map_err(|why| anyhow::anyhow!("{why}"))?;
    let now = super::set_of(&src, &frames);
    let changes = snapshot::changes(&old_snap, &sel.revisions);
    let (new_info, gone) = {
        let old = results
            .image_sets
            .get(old_set)
            .ok_or_else(|| anyhow::anyhow!("the saved results changed while re-scoring"))?;
        let info = rebuilt(old, &now);
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
    let new_snap = {
        let scored = image_set_of(&new_info);
        let mut snap = GtSnapshot::of(&src, &scored, &sel.revisions);
        // Frames reviewed since the run stay new: they still need a run.
        snap.reviewed
            .retain(|k, _| old_snap.reviewed.contains_key(k));
        snap
    };
    if let Some(slot) = results.image_sets.get_mut(old_set) {
        *slot = new_info;
    }
    let preds = StoredPreds::load(preds_path)?;
    let (models, runs, not_rescored) = rescore_runs(&mut results, preds.as_ref(), &gone);
    results.regrade(config.benchmark.accuracy_metric, config.benchmark.weights);
    let frames_now = results.image_set(ID).map_or(0, |s| s.images.len());
    new_snap.save(&gt_path)?;
    results.save(results_path)?;
    Ok(Rescored {
        models,
        runs,
        frames: frames_now,
        need_run: changes.joined,
        changes,
        not_rescored,
    })
}
