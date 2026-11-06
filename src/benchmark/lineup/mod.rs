//! The model-vs-model view of `GET /v1/benchmark` (`lineup`), behind the page's Compare tab: per
//! image set the models scored on exactly those images, side by side, with what is best past the
//! noise, one plain sentence, and who is not on these images and why.
//!
//! - [`sets`]: the sets to choose from and the one to open with;
//! - [`score`]: the figures, rescored from the stored predictions on one label space;
//! - [`comparable`]: whether two models' results compare at all, accuracy and speed apart;
//! - [`verdict`]: best-of-column marks and the sentence, on the run-vs-run noise rules.
//!
//! A model is on a set only when its stored run holds every image of it. Rescoring the stored
//! predictions against the set's recorded labels makes a slice of a larger run (one dataset of
//! three, an export's held-out frames) as exact as a run on it alone. A device is chosen once per
//! set: the same device for every model when they share one, so speed compares like with like;
//! otherwise each model's own, and the speed column is flagged. Each model runs at its own
//! confidence: the one the config gives it on that device (`now`) or its best for the goal
//! (`best`). Pure except for the inputs handed in.

pub mod comparable;
pub mod score;
pub mod sets;
pub mod verdict;

use super::current::ModelNow;
use super::history::{History, HistoryEntry, run_key};
use super::images::{GroundTruth, SetKind};
use super::metrics::ClassMap;
use super::report::{BenchmarkResults, ImageSetInfo, ModelResult};
use super::search::{StoredPreds, StoredRun};
use super::{RunResult, trained_on};
use crate::registry::normalize_name;
use comparable::{ModelComparison, Side, all_pairs, speed_reasons};
use score::{Cell, Objects, SCENES, Scorable, Scored};
use serde::Serialize;
use sets::{Candidate, CompareSet, SetRole, compare_sets, default_set};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use verdict::{Column, Contender, Headline, headline, mark_best};

/// A configured model, for the "Not on these images" list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Configured {
    pub name: String,
    /// Why it cannot run now (the file is missing); None = it can.
    pub blocked: Option<String>,
}

/// What [`build`] reads.
pub struct Inputs<'a> {
    pub results: &'a BenchmarkResults,
    pub preds: Option<&'a StoredPreds>,
    pub history: &'a History,
    /// What the config says about each model now, by normalized name.
    pub now: &'a HashMap<String, ModelNow>,
    pub default_model: Option<&'a str>,
    pub configured: &'a [Configured],
    /// The frames each trained model trained on, by normalized name. A model missing here is
    /// judged by its stored `trained_on_share`.
    pub trained: &'a HashMap<String, BTreeSet<String>>,
    /// Whether a dataset's folder can be read now (a removed drive cannot).
    pub reachable: &'a dyn Fn(&ImageSetInfo) -> bool,
}

impl std::fmt::Debug for Inputs<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inputs")
            .field("models", &self.results.models.len())
            .finish_non_exhaustive()
    }
}

/// A scene the set has images of, shown as its own column.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Scene {
    pub id: String,
    pub images: usize,
}

/// One model at one confidence on a set.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Row {
    pub model: String,
    pub device: String,
    /// The confidence these figures are at.
    pub confidence: f32,
    pub is_default: bool,
    /// Share of the set's frames the model trained on; None = unknown.
    pub trained_on_share: Option<f64>,
    /// Trained on some of these frames: shown, never best.
    pub excluded: bool,
    pub objects: BTreeMap<String, Objects>,
    /// False alerts per empty frame, 0..=1.
    pub false_alerts: Cell,
    /// Empty frames with a false alert (out of `false_alerts.n`).
    pub alerts: usize,
    /// Time per image, ms.
    pub speed: Cell,
}

/// The table of a set at one kind of confidence.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct View {
    /// `now` or `best`.
    pub confidence: String,
    pub headline: Headline,
    pub rows: Vec<Row>,
}

/// A model not scored on a set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Missing {
    pub model: String,
    pub why: String,
    /// The device to run it on (the set's device, else its own).
    pub device: Option<String>,
    /// Why it cannot be run on these images now; None = the button works.
    pub blocked: Option<String>,
}

/// What a run needs to cover a set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunHint {
    pub datasets: Vec<String>,
    /// Images per dataset the set holds (the form's `max_images`).
    pub max_images: usize,
}

/// One image set in `lineup`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SetView {
    pub id: String,
    pub title: String,
    /// What it is, in a few words.
    pub what: String,
    pub role: SetRole,
    /// `frames` or `images`.
    pub unit: String,
    pub images: usize,
    /// Scored objects (of the shared classes), the same for every model.
    pub objects: usize,
    /// Images without any scored object.
    pub empty_frames: usize,
    /// True: the labels are a reference model's, not a person's.
    pub relative: bool,
    pub reference: Option<String>,
    /// The classes every model here is scored on.
    pub classes: Vec<String>,
    /// Classes only some of the models can output, left out so all answer for the same objects.
    pub left_out: Vec<String>,
    /// Scenes covering part of the set (a set of night frames has no Night column).
    pub scenes: Vec<Scene>,
    /// Hardware, runtime and device differences between the rows. Accuracy stays comparable.
    pub speed: Side,
    pub views: Vec<View>,
    pub missing: Vec<Missing>,
    pub run: RunHint,
}

/// `lineup` of `GET /v1/benchmark`.
#[derive(Debug, Clone, PartialEq, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Lineup {
    /// Best first.
    pub sets: Vec<SetView>,
    pub default_set: Option<String>,
    /// [`comparable::comparable_models`] of every pair of models over their whole runs.
    pub pairs: Vec<ModelComparison>,
}

/// One model's usable runs on a set.
struct Entrant<'a> {
    m: &'a ModelResult,
    /// (device, its run, its stored predictions) of runs that hold every image of the set.
    runs: Vec<(&'a str, &'a RunResult, &'a StoredRun)>,
}

/// How much of the set a model has: for the "why".
enum Reach<'a> {
    Whole(Entrant<'a>),
    Part { scored: usize },
    NoStored,
}

fn reach<'a>(
    set: &CompareSet<'_>,
    m: &'a ModelResult,
    preds: Option<&'a StoredPreds>,
) -> Reach<'a> {
    let stored = preds.map(|p| p.runs_of(&m.model)).unwrap_or_default();
    let mut runs = Vec::new();
    let mut most = 0;
    for d in &m.devices {
        let Some(run) = score::usable(d) else {
            continue;
        };
        let Some(s) = stored.iter().find(|s| s.device == d.device) else {
            continue;
        };
        let got = score::covered(set, &score::index_of(s));
        most = most.max(got);
        if got == set.images.len() {
            runs.push((d.device.as_str(), run, *s));
        }
    }
    if !runs.is_empty() {
        Reach::Whole(Entrant { m, runs })
    } else if stored.is_empty() {
        Reach::NoStored
    } else {
        Reach::Part { scored: most }
    }
}

/// The device of each entrant (see the module docs).
fn devices_for(
    list: &[Entrant<'_>],
    now: &HashMap<String, ModelNow>,
    default_model: Option<&str>,
) -> Vec<String> {
    let now_of = |e: &Entrant<'_>| {
        now.get(&normalize_name(&e.m.model))
            .and_then(|n| n.device.clone())
    };
    let has = |e: &Entrant<'_>, d: &str| e.runs.iter().any(|r| r.0 == d);
    let mut shared: Vec<&str> = list
        .first()
        .map(|e| e.runs.iter().map(|r| r.0).collect())
        .unwrap_or_default();
    shared.retain(|d| list.iter().all(|e| has(e, d)));
    if !shared.is_empty() {
        let key = default_model.map(normalize_name);
        let by_default = list
            .iter()
            .find(|e| key.as_deref() == Some(normalize_name(&e.m.model).as_str()))
            .and_then(now_of)
            .filter(|d| shared.contains(&d.as_str()));
        let mut votes: Vec<(usize, &str)> = shared
            .iter()
            .map(|d| {
                let v = list
                    .iter()
                    .filter(|e| now_of(e).as_deref() == Some(*d))
                    .count();
                (v, *d)
            })
            .collect();
        votes.sort_by_key(|(v, _)| std::cmp::Reverse(*v));
        let pick = by_default
            .or_else(|| votes.first().map(|(_, d)| (*d).to_string()))
            .unwrap_or_default();
        return list.iter().map(|_| pick.clone()).collect();
    }
    list.iter()
        .map(|e| {
            now_of(e)
                .filter(|d| has(e, d))
                .or_else(|| e.m.recommended.clone().filter(|d| has(e, d)))
                .or_else(|| e.runs.first().map(|r| r.0.to_string()))
                .unwrap_or_default()
        })
        .collect()
}

fn share_on(set: &CompareSet<'_>, m: &ModelResult, inputs: &Inputs<'_>) -> Option<f64> {
    match inputs.trained.get(&normalize_name(&m.model)) {
        Some(frames) => trained_on::share(set.images.iter().map(|i| i.info.file.as_str()), frames),
        None => m.trained_on_share,
    }
}

fn labels_of(set: &CompareSet<'_>) -> (bool, Option<String>) {
    let reference = set.datasets.iter().find_map(|s| match &s.ground_truth {
        GroundTruth::Pseudo(r) => Some(r.clone()),
        _ => None,
    });
    (reference.is_some(), reference)
}

/// The reference model's own detections are the labels of a relative set: it would score itself.
fn is_reference(reference: Option<&str>, model: &str) -> bool {
    reference
        .and_then(|r| r.split(" on ").next())
        .is_some_and(|r| normalize_name(r.trim()) == normalize_name(model))
}

/// Why `m` is not on `set`, from what it ran and how far it reached.
fn why_not(
    set: &CompareSet<'_>,
    m: &ModelResult,
    reach: &Reach<'_>,
    all: &[ImageSetInfo],
) -> String {
    let total = set.images.len();
    let unit = set.role.unit();
    match reach {
        Reach::Whole(_) => "these images are its own labels: it would score itself".into(),
        Reach::NoStored => {
            if m.devices.iter().any(|d| d.ok()) {
                "its last run kept no predictions to score on these images".into()
            } else {
                "its last run failed on every device".into()
            }
        }
        Reach::Part { scored: 0 } => {
            let names: Vec<&str> = m
                .datasets
                .iter()
                .map(|id| {
                    all.iter()
                        .find(|s| s.id == *id && !s.title.is_empty())
                        .map_or(id.as_str(), |s| s.title.as_str())
                })
                .collect();
            format!(
                "scored on {} other images ({})",
                m.images,
                if names.is_empty() {
                    "none recorded".to_string()
                } else {
                    names.join(", ")
                }
            )
        }
        Reach::Part { scored } => format!("scored on only {scored} of these {total} {unit}"),
    }
}

/// Which `blocked` reason a run of `model` on `set` has.
fn blocked_for(set: &CompareSet<'_>, model: &str, inputs: &Inputs<'_>) -> Option<String> {
    let key = normalize_name(model);
    match inputs
        .configured
        .iter()
        .find(|c| normalize_name(&c.name) == key)
    {
        None => return Some("it is no longer in the config".into()),
        Some(c) if c.blocked.is_some() => return c.blocked.clone(),
        Some(_) => {}
    }
    set.datasets
        .iter()
        .find(|d| d.kind == SetKind::Dir && !(inputs.reachable)(d))
        .map(|d| {
            format!(
                "its folder cannot be read now ({})",
                d.dir
                    .as_deref()
                    .map_or_else(|| d.id.clone(), |p| p.display().to_string())
            )
        })
}

fn cell_mark<F>(rows: &mut [Row], column: Column, cell: F)
where
    F: for<'r> Fn(&'r mut Row) -> Option<&'r mut Cell>,
{
    let values: Vec<Option<(f64, usize)>> = rows
        .iter_mut()
        .map(|r| {
            let fair = !r.excluded;
            cell(r)
                .filter(|_| fair)
                .and_then(|c| c.value.map(|v| (v, c.n)))
        })
        .collect();
    let marks = mark_best(column, &values);
    for (r, m) in rows.iter_mut().zip(marks) {
        if let Some(c) = cell(r) {
            c.best = m;
        }
    }
}

/// Mark the best of every column.
fn mark_columns(rows: &mut [Row], speed_comparable: bool) {
    let kinds: BTreeSet<String> = rows
        .iter()
        .flat_map(|r| r.objects.keys().cloned())
        .collect();
    for kind in &kinds {
        for scene in ["all"].into_iter().chain(SCENES) {
            cell_mark(rows, Column::Share, |r| {
                r.objects.get_mut(kind)?.finds.get_mut(scene)
            });
            cell_mark(rows, Column::Share, |r| {
                r.objects.get_mut(kind)?.correct.get_mut(scene)
            });
        }
    }
    cell_mark(rows, Column::Fewer, |r| Some(&mut r.false_alerts));
    if speed_comparable {
        cell_mark(rows, Column::Time, |r| Some(&mut r.speed));
    }
}

fn contender(r: &Row) -> Contender {
    let finds = |k: &str| {
        r.objects
            .get(k)
            .and_then(|o| Some((o.finds.all.value?, o.finds.all.n)))
    };
    Contender {
        model: r.model.clone(),
        trained_on: r.trained_on_share,
        people: finds("person"),
        vehicles: finds("vehicle"),
        all: finds("all"),
        correct: r
            .objects
            .get("all")
            .and_then(|o| Some((o.correct.all.value?, o.correct.all.n))),
    }
}

/// The figures every row of a view shares.
struct Prepared<'a> {
    m: &'a ModelResult,
    device: String,
    run: &'a RunResult,
    images: Vec<Scorable<'a>>,
    share: Option<f64>,
}

fn row_of(
    p: &Prepared<'_>,
    set: &CompareSet<'_>,
    classes: &[String],
    confidence: f32,
    default_model: Option<&str>,
) -> Row {
    let Scored {
        objects,
        false_alerts,
        alerts,
    } = score::score(&p.images, classes, confidence);
    Row {
        model: p.m.model.clone(),
        device: p.device.clone(),
        confidence,
        is_default: default_model.is_some_and(|d| normalize_name(d) == normalize_name(&p.m.model)),
        trained_on_share: p.share,
        excluded: p.share.is_some_and(|s| s > 0.0),
        objects,
        false_alerts,
        alerts,
        speed: score::speed(set, p.run),
    }
}

/// The set's view, or None when no model is on it.
fn set_view(set: &CompareSet<'_>, inputs: &Inputs<'_>) -> Option<(SetView, Candidate)> {
    let (relative, reference) = labels_of(set);
    let mut entrants: Vec<Entrant<'_>> = Vec::new();
    let mut missing: Vec<(&ModelResult, Reach<'_>)> = Vec::new();
    for m in &inputs.results.models {
        let r = reach(set, m, inputs.preds);
        match r {
            Reach::Whole(e) if !is_reference(reference.as_deref(), &m.model) => entrants.push(e),
            other => missing.push((m, other)),
        }
    }
    if entrants.is_empty() {
        return None;
    }
    let devices = devices_for(&entrants, inputs.now, inputs.default_model);
    let chosen: Vec<(&Entrant<'_>, &str, &RunResult, &StoredRun)> = entrants
        .iter()
        .zip(&devices)
        .filter_map(|(e, d)| {
            let r = e.runs.iter().find(|r| r.0 == d)?;
            Some((e, r.0, r.1, r.2))
        })
        .collect();
    let lists: Vec<Vec<String>> = chosen.iter().map(|c| score::folded_classes(c.3)).collect();
    let classes = score::common_classes(&lists);
    let map = ClassMap::for_model(&classes);
    let prepared: Vec<Prepared<'_>> = chosen
        .iter()
        .map(|(e, device, run, stored)| Prepared {
            m: e.m,
            device: (*device).to_string(),
            run,
            images: score::scorable(set, &map, stored, &score::index_of(stored)),
            share: share_on(set, e.m, inputs),
        })
        .collect();
    let speed = {
        let entries: Vec<HistoryEntry> = chosen
            .iter()
            .map(|(e, device, _, _)| {
                let mut h = inputs
                    .history
                    .find(&e.m.model, &run_key(e.m))
                    .cloned()
                    .unwrap_or_else(|| HistoryEntry::of(e.m, inputs.results));
                h.devices.retain(|d| d.device == *device);
                h
            })
            .collect();
        let mut reasons: Vec<String> = Vec::new();
        for (i, a) in entries.iter().enumerate() {
            for b in entries.iter().skip(i + 1) {
                for r in speed_reasons(a, b) {
                    if !reasons.contains(&r) {
                        reasons.push(r);
                    }
                }
            }
        }
        Side {
            comparable: reasons.is_empty(),
            reasons,
        }
    };
    let unit = set.role.unit();
    let views: Vec<View> = ["now", "best"]
        .into_iter()
        .map(|kind| {
            let mut rows: Vec<Row> = prepared
                .iter()
                .map(|p| {
                    let t = confidence_of(p, kind, inputs.now);
                    row_of(p, set, &classes, t, inputs.default_model)
                })
                .collect();
            mark_columns(&mut rows, speed.comparable);
            let contenders: Vec<Contender> = rows.iter().map(contender).collect();
            View {
                confidence: kind.to_string(),
                headline: headline(&contenders, set.images.len(), unit),
                rows,
            }
        })
        .collect();
    let first = views.first()?.rows.first()?;
    let objects = first.objects.get("all").map_or(0, |o| o.finds.all.n);
    let empty_frames = first.false_alerts.n;
    let scenes: Vec<Scene> = SCENES
        .iter()
        .filter_map(|id| {
            let images = set
                .images
                .iter()
                .filter(|i| i.info.tags.iter().any(|t| t == id))
                .count();
            (images > 0 && images < set.images.len()).then(|| Scene {
                id: (*id).to_string(),
                images,
            })
        })
        .collect();
    let left_out: Vec<String> = {
        let mut all: Vec<String> = lists.iter().flatten().cloned().collect();
        all.sort();
        all.dedup();
        all.retain(|c| !classes.contains(c));
        all
    };
    let set_device = devices
        .first()
        .filter(|d| devices.iter().all(|x| x == *d))
        .cloned();
    let mut out_of_set: Vec<Missing> = missing
        .iter()
        .map(|(m, r)| Missing {
            model: m.model.clone(),
            why: why_not(set, m, r, &inputs.results.image_sets),
            device: set_device.clone().or_else(|| m.recommended.clone()),
            blocked: if matches!(r, Reach::Whole(_)) {
                Some("its detections are the labels of these images".into())
            } else {
                blocked_for(set, &m.model, inputs)
            },
        })
        .collect();
    for c in inputs.configured {
        let key = normalize_name(&c.name);
        let known = inputs
            .results
            .models
            .iter()
            .any(|m| normalize_name(&m.model) == key);
        if !known {
            let device = inputs.now.get(&key).and_then(|n| n.device.clone());
            out_of_set.push(Missing {
                model: c.name.clone(),
                why: "not benchmarked yet".into(),
                device: set_device.clone().or(device),
                blocked: blocked_for(set, &c.name, inputs),
            });
        }
    }
    let fair_models = views
        .first()
        .map_or(0, |v| v.rows.iter().filter(|r| !r.excluded).count());
    let view = SetView {
        id: set.id.clone(),
        title: set.title.clone(),
        what: set.role.what().to_string(),
        role: set.role,
        unit: unit.to_string(),
        images: set.images.len(),
        objects,
        empty_frames,
        relative,
        reference,
        classes,
        left_out,
        scenes,
        speed,
        views,
        missing: out_of_set,
        run: RunHint {
            datasets: set.datasets.iter().map(|d| d.id.clone()).collect(),
            max_images: set
                .datasets
                .iter()
                .map(|d| d.images.len())
                .max()
                .unwrap_or(0),
        },
    };
    let candidate = Candidate {
        id: set.id.clone(),
        models: entrants.len(),
        role: set.role,
        fair_models,
        images: set.images.len(),
    };
    Some((view, candidate))
}

/// The confidence of a row: the one the config gives the model on the device (else the run's), or
/// its best for the goal on that device (else the same).
fn confidence_of(p: &Prepared<'_>, kind: &str, now: &HashMap<String, ModelNow>) -> f32 {
    let configured = now
        .get(&normalize_name(&p.m.model))
        .map(|n| n.threshold_for(&p.device))
        .or_else(|| {
            p.m.threshold
                .as_ref()
                .map(super::threshold::ThresholdAdvice::now_threshold)
        })
        .or_else(|| p.run.accuracy.as_ref().map(|a| a.threshold))
        .unwrap_or(0.5);
    if kind == "best" {
        p.m.threshold
            .as_ref()
            .and_then(|a| a.apply_value_on(&p.device))
            .unwrap_or(configured)
    } else {
        configured
    }
}

/// The `lineup` of `inputs`.
pub fn build(inputs: &Inputs<'_>) -> Lineup {
    let mut views: Vec<(SetView, Candidate)> = compare_sets(inputs.results)
        .iter()
        .filter_map(|s| set_view(s, inputs))
        .collect();
    let candidates: Vec<Candidate> = views.iter().map(|v| v.1.clone()).collect();
    let default = default_set(&candidates).map(str::to_string);
    views.sort_by_cached_key(|(_, c)| std::cmp::Reverse(sets::rank(c)));
    let entries: Vec<HistoryEntry> = inputs
        .results
        .models
        .iter()
        .filter(|m| m.error.is_none())
        .map(|m| {
            inputs
                .history
                .find(&m.model, &run_key(m))
                .cloned()
                .unwrap_or_else(|| HistoryEntry::of(m, inputs.results))
        })
        .collect();
    Lineup {
        sets: views.into_iter().map(|v| v.0).collect(),
        default_set: default,
        pairs: all_pairs(&entries),
    }
}

#[cfg(test)]
mod tests;
