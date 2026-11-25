//! The combos (a model on a device) on each image set of the saved results, for the Compare view
//! of the evaluation core ([`crate::benchmark::eval::compare`]): which combos hold a set, the
//! frames they are judged on, who is not on it and why.
//!
//! - [`sets`]: the sets to choose from and the one to open with;
//! - [`score`]: the shared label space (car, truck and bus are one `vehicle`) and the lookups that
//!   rescore a stored run on a set.
//!
//! A combo is on a set only when its stored run holds every image of it (on "Your cameras", every
//! frame its model did not train or validate on: its run left those out, and the set is cut to the
//! frames no listed model saw, [`ComboSet::frames_left_out`]; a model whose training lists do not
//! read is shown, never ranked among the fair rows). Rescoring the stored predictions against the
//! set's recorded labels makes a slice of a larger run (one dataset of three, an export's
//! held-out frames) as exact as a run on it alone. On other sets a model that trained on some of
//! the frames (its training record's frame list, else its stored `trained_on_share`) is shown,
//! never ranked fairly. The reference model of a relative set (another model's detections as
//! labels) is never on it, nor are its As served and variant results. A result's training
//! records, what it saw and its threshold now are those of the configured model that owns it
//! ([`owner_key`]: `<model>+served` and `<model>+<variant>` are `<model>`'s). Pure except for the
//! inputs handed in.

pub mod score;
pub mod sets;

use super::cameras::seen::Seen;
use super::current::ModelNow;
use super::images::{GroundTruth, SetKind};
use super::report::{BenchmarkResults, ImageSetInfo, ModelResult};
use super::search::{StoredPreds, StoredRun};
use super::{RunResult, trained_on};
use crate::registry::normalize_name;
use serde::Serialize;
use sets::{Candidate, CompareSet};
use std::collections::{BTreeSet, HashMap};

/// A configured model, for the "Not on these images" list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Configured {
    pub name: String,
    /// Why it cannot run now (the file is missing); None = it can.
    pub blocked: Option<String>,
}

/// What [`combos_on`] reads.
pub struct Inputs<'a> {
    pub results: &'a BenchmarkResults,
    pub preds: Option<&'a StoredPreds>,
    /// What the config says about each model now, by normalized name.
    pub now: &'a HashMap<String, ModelNow>,
    pub default_model: Option<&'a str>,
    pub configured: &'a [Configured],
    /// The frames each trained model trained on, by normalized name. A model missing here is
    /// judged by its stored `trained_on_share`.
    pub trained: &'a HashMap<String, BTreeSet<String>>,
    /// What each trained model saw of "Your cameras" (its run's train and validation lists), by
    /// normalized name; a model missing here saw none of it. On a set with the cameras frames,
    /// the frames any listed model saw are out of the view ([`ComboSet::frames_left_out`]), so
    /// every model is scored on the frames none of them saw.
    pub seen: &'a HashMap<String, Seen>,
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

/// Whether `seen` says its model saw `img`, a frame of "Your cameras".
fn saw_image(seen: Option<&Seen>, img: &sets::SetImage<'_>) -> bool {
    img.set.id == super::cameras::ID && seen.is_some_and(|s| s.saw(&img.info.file).is_some())
}

/// How much of `set` model `m` has; `excused` images (on "Your cameras", the frames it saw and
/// so did not run) need not be in its run.
fn reach<'a>(
    set: &CompareSet<'_>,
    m: &'a ModelResult,
    preds: Option<&'a StoredPreds>,
    excused: &dyn Fn(&sets::SetImage<'_>) -> bool,
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
        let index = score::index_of(s);
        let got = score::covered(set, &index);
        most = most.max(got);
        let missed = set.images.iter().any(|i| {
            !excused(i) && !index.contains_key(&(i.set.id.as_str(), i.info.file.as_str()))
        });
        if !missed {
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

/// Normalized name of the configured model the result `model` belongs to
/// ([`super::variant::owner_candidates`] against [`Inputs::configured`]: `<model>+served` and
/// `<model>+<variant>` results are their model's), the key of [`Inputs::trained`],
/// [`Inputs::seen`] and [`Inputs::now`]; a result no configured model owns keys by its own name.
pub(crate) fn owner_key(inputs: &Inputs<'_>, model: &str) -> String {
    let configured = |n: &str| {
        let k = normalize_name(n);
        inputs
            .configured
            .iter()
            .any(|c| normalize_name(&c.name) == k)
    };
    normalize_name(
        super::variant::owner_candidates(model)
            .find(|n| configured(n))
            .unwrap_or(model),
    )
}

fn share_on(set: &CompareSet<'_>, m: &ModelResult, inputs: &Inputs<'_>) -> Option<f64> {
    match inputs.trained.get(&owner_key(inputs, &m.model)) {
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

/// The reference model's own detections are the labels of a relative set: it would score itself
/// ([`crate::benchmark::is_reference`]).
fn is_reference(reference: Option<&str>, model: &str) -> bool {
    reference.is_some_and(|r| crate::benchmark::is_reference(r, model))
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

fn seen_of<'i>(inputs: &Inputs<'i>, model: &str) -> Option<&'i Seen> {
    inputs.seen.get(&owner_key(inputs, model))
}

/// The set's view, or None when no model is on it. On "Your cameras" each model is required to
/// have the frames it did not see, and the view holds the frames no listed model saw.
/// One model on one device on a set ([`combos_on`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Combo<'a> {
    pub(crate) m: &'a ModelResult,
    pub(crate) device: &'a str,
    pub(crate) run: &'a RunResult,
    pub(crate) stored: &'a StoredRun,
    /// Share of the set's frames the model trained on; None = unknown.
    pub(crate) share: Option<f64>,
    /// On "Your cameras": why what it saw cannot be told.
    pub(crate) unknown_seen: Option<&'a str>,
    /// Shown, never ranked among the fair rows: it trained on some of the frames, or (on "Your
    /// cameras") what it saw cannot be told.
    pub(crate) excluded: bool,
}

/// Every combo (model x device) whose stored run holds a set, on the frames none of them saw
/// (the Compare view of the evaluation core, [`crate::benchmark::eval::compare`]).
pub(crate) struct ComboSet<'a> {
    /// The fair frames: every listed combo ran them and no listed model saw them.
    pub(crate) fair: CompareSet<'a>,
    pub(crate) frames_left_out: usize,
    pub(crate) left_out_by: Vec<String>,
    pub(crate) combos: Vec<Combo<'a>>,
    /// Models not on the set, why, and how to run them.
    pub(crate) missing: Vec<Missing>,
    pub(crate) relative: bool,
    pub(crate) reference: Option<String>,
    /// What [`default_set`] weighs.
    pub(crate) candidate: Candidate,
}

/// The combos on `full` (see [`ComboSet`]); None when no combo holds the set: a run holds the
/// set when it has every frame its model did not see; the reference model of a relative set is
/// never on it (see the module docs).
pub(crate) fn combos_on<'a>(full: &CompareSet<'a>, inputs: &Inputs<'a>) -> Option<ComboSet<'a>> {
    let (relative, reference) = labels_of(full);
    let cams = full.datasets.iter().any(|d| d.id == super::cameras::ID);
    // What each result's model saw, resolved once per result (it is asked per frame below).
    let seen_of_result: HashMap<&str, &Seen> = if cams {
        inputs
            .results
            .models
            .iter()
            .filter_map(|m| Some((m.model.as_str(), seen_of(inputs, &m.model)?)))
            .collect()
    } else {
        HashMap::new()
    };
    let seen_by = |model: &str| seen_of_result.get(model).copied();
    let mut entrants: Vec<Entrant<'a>> = Vec::new();
    let mut missing: Vec<(&ModelResult, Reach<'a>)> = Vec::new();
    for m in &inputs.results.models {
        let seen = seen_by(&m.model);
        match reach(full, m, inputs.preds, &|i| saw_image(seen, i)) {
            Reach::Whole(e) if !is_reference(reference.as_deref(), &m.model) => entrants.push(e),
            other => missing.push((m, other)),
        }
    }
    if entrants.is_empty() {
        return None;
    }
    let left_out_by: Vec<String> = entrants
        .iter()
        .filter(|e| {
            let seen = seen_by(&e.m.model);
            full.images.iter().any(|i| saw_image(seen, i))
        })
        .map(|e| e.m.model.clone())
        .collect();
    let fair = CompareSet {
        images: full
            .images
            .iter()
            .filter(|i| !entrants.iter().any(|e| saw_image(seen_by(&e.m.model), i)))
            .copied()
            .collect(),
        ..full.clone()
    };
    if fair.images.is_empty() {
        return None;
    }
    let frames_left_out = full.images.len() - fair.images.len();
    let mut combos = Vec::new();
    for e in &entrants {
        let share = share_on(&fair, e.m, inputs);
        let unknown_seen = match seen_by(&e.m.model) {
            Some(Seen::Unknown(why)) => Some(why.as_str()),
            _ => None,
        };
        let excluded = if cams {
            unknown_seen.is_some()
        } else {
            share.is_some_and(|s| s > 0.0)
        };
        for &(device, run, stored) in &e.runs {
            combos.push(Combo {
                m: e.m,
                device,
                run,
                stored,
                share,
                unknown_seen,
                excluded,
            });
        }
    }
    let mut out_of_set: Vec<Missing> = missing
        .iter()
        .map(|(m, r)| Missing {
            model: m.model.clone(),
            why: why_not(full, m, r, &inputs.results.image_sets),
            device: inputs
                .now
                .get(&owner_key(inputs, &m.model))
                .and_then(|n| n.device.clone())
                .or_else(|| m.recommended.clone()),
            blocked: if matches!(r, Reach::Whole(_)) {
                Some("its detections are the labels of these images".into())
            } else {
                blocked_for(&fair, &m.model, inputs)
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
            out_of_set.push(Missing {
                model: c.name.clone(),
                why: "not benchmarked yet".into(),
                device: inputs.now.get(&key).and_then(|n| n.device.clone()),
                blocked: blocked_for(&fair, &c.name, inputs),
            });
        }
    }
    let candidate = Candidate {
        id: fair.id.clone(),
        models: entrants.len(),
        role: fair.role,
        fair_models: entrants
            .iter()
            .filter(|e| !combos.iter().any(|c| std::ptr::eq(c.m, e.m) && c.excluded))
            .count(),
        images: fair.images.len(),
        cameras: fair.id == super::cameras::ID,
    };
    Some(ComboSet {
        fair,
        frames_left_out,
        left_out_by,
        combos,
        missing: out_of_set,
        relative,
        reference,
        candidate,
    })
}

/// What [`Inputs`] reads of the config and the training records: the frames each trained model
/// trained on, what each saw of "Your cameras" (when the results hold the set) and which
/// configured models can run.
#[derive(Debug, Clone, Default)]
pub struct Parts {
    pub trained: HashMap<String, BTreeSet<String>>,
    pub seen: HashMap<String, Seen>,
    pub configured: Vec<Configured>,
}

impl Parts {
    /// Read from `cfg` (training records, model files) for the configured models that own at
    /// least one result of `results` (its bare, As served or variant result,
    /// [`super::variant::owner`]).
    pub fn of(results: &BenchmarkResults, cfg: &crate::config::Config) -> Self {
        let owners: std::collections::HashSet<String> = results
            .models
            .iter()
            .filter_map(|r| super::variant::owner(cfg, &r.model))
            .map(|m| m.key())
            .collect();
        let in_results = |m: &crate::config::ModelConfig| owners.contains(&m.key());
        let trained = cfg
            .models
            .iter()
            .filter_map(|m| {
                if !in_results(m) {
                    return None;
                }
                // Its As served and variant results trained on the same frames.
                let frames =
                    crate::training::autolabel::trained_frames(cfg, m, &mut Vec::new()).ok()?;
                (!frames.is_empty()).then(|| (m.key(), frames))
            })
            .collect();
        let seen = results
            .image_set(super::cameras::ID)
            .map(|set| {
                cfg.models
                    .iter()
                    .filter_map(|m| {
                        if !in_results(m) {
                            return None;
                        }
                        let files = set.images.iter().map(|i| i.file.as_str());
                        let s = super::cameras::seen::of_model(cfg, m, files);
                        (s != Seen::NoRecord).then(|| (m.key(), s))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let configured = cfg
            .models
            .iter()
            .map(|m| Configured {
                name: m.effective_name(),
                blocked: (!cfg.data_path(&m.path).is_file())
                    .then(|| "its model file is missing".to_string()),
            })
            .collect();
        Self {
            trained,
            seen,
            configured,
        }
    }
}

#[cfg(test)]
mod tests;
