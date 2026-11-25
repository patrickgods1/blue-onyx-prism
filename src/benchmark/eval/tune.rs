//! The Tune view of one combo (`GET /v1/benchmark/tune`) and its example frames
//! (`GET /v1/benchmark/tune/images`).
//!
//! The combo is scored on the frames of one compare set it ran (its run already left out the
//! frames its model trained or validated on), in its own label space (the classes it is scored
//! on, IPcam's `vehicle` taking car, truck and bus labels; limited to the goal's classes), so
//! the confidence it suggests is the one to save for it. Every 0.01 grid value from 0.05 is
//! listed with its counts and rates; the suggestion is the pick on every frame
//! ([`super::assess`]), with the out-of-fold figure of the pick procedure and its interval.
//!
//! Saving: never for a pick on another model's detections (pseudo labels: relative,
//! [`crate::benchmark::threshold::NOT_ON_LABELS`]) or a target not met on its bound; a
//! low-confidence pick may be saved after a confirmation (`confirm`).

use super::assess::{Assessment, FLAT_NOTE, Input, assess};
use super::compare::GoalView;
use super::goal::{Goal, objective_short};
use super::pick::{bound, met, score, target, value};
use super::stats::Estimate;
use super::tally::sum_all;
use crate::benchmark::combos::sets::{CompareSet, compare_sets};
use crate::benchmark::images::GroundTruth;
use crate::benchmark::metrics::{ClassMap, EvalImage, canonical_label, image_counts};
use crate::benchmark::report::BenchmarkResults;
use crate::benchmark::search::{StoredPreds, StoredRun};
use crate::benchmark::threshold::{NOT_ON_LABELS, grid};
use crate::benchmark::{RunResult, eval_input, frame_classes, is_reference};
use crate::registry::normalize_name;
use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::collections::HashMap;

/// What [`build`] reads.
#[derive(Debug, Clone, Copy)]
pub struct Request<'a> {
    pub model: &'a str,
    /// Device spec; None = the model's recommended device, else its first stored run.
    pub device: Option<&'a str>,
    /// Compare set id; None = the largest set the run covers.
    pub set: Option<&'a str>,
    pub goal: &'a Goal,
    pub weights: crate::benchmark::grade::Weights,
    /// The confidence saved for this combo now and where it comes from ("profile for
    /// openvino:gpu", "model", "global").
    pub saved: Option<(f32, &'a str)>,
}

/// One grid value.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GridRow {
    pub threshold: f32,
    pub tp: usize,
    pub fp: usize,
    #[serde(rename = "fn")]
    pub fn_: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    pub f2: Option<f64>,
    pub f05: Option<f64>,
    pub fa100: Option<f64>,
    pub tpr: Option<f64>,
    pub fpr: Option<f64>,
    /// The tuning objective's score (None where its target is not met on the bound).
    pub score: Option<f64>,
    /// The target's bound here (None without a target).
    pub bound: Option<f64>,
}

/// The suggested confidence.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Suggestion {
    pub threshold: f32,
    pub plateau: [f32; 2],
    /// The best-scoring grid value (before the plateau rule).
    pub best: f32,
    /// The objective's value at the suggestion on every frame (in-sample).
    pub in_sample: Option<f64>,
    /// The goal's tuning objective as the procedure scores on frames it did not see
    /// (out-of-fold; in-sample when low confidence), with its 95% interval.
    pub metric: Option<Estimate>,
}

/// A target's state.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TargetState {
    pub value: f64,
    /// Met on the bound of the reported (out-of-fold) counts.
    pub met: bool,
    pub bound: Option<f64>,
    /// The bound at the suggestion on every frame.
    pub bound_in_sample: Option<f64>,
    /// "met (95% bound 0.91 ≥ 0.90)" / "not met: best bound 0.86".
    pub text: String,
}

/// The saved confidence.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Saved {
    pub threshold: f32,
    pub source: String,
}

/// The view.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TuneView {
    pub model: String,
    pub device: String,
    pub devices: Vec<String>,
    pub set: String,
    pub set_title: String,
    pub sets: Vec<String>,
    pub goal: GoalView,
    /// The objective the confidence is picked for.
    pub objective: String,
    pub objective_name: String,
    pub frames: usize,
    pub objects: usize,
    pub empty_frames: usize,
    pub groups: usize,
    pub folds: usize,
    pub classes: Vec<String>,
    pub grid: Vec<GridRow>,
    pub suggested: Option<Suggestion>,
    pub low_confidence: Option<String>,
    pub pick_informative: bool,
    pub flat_note: Option<String>,
    pub target: Option<TargetState>,
    pub saved: Option<Saved>,
    /// The confidence the run served with on this device, and where it came from.
    pub served_threshold: Option<f32>,
    pub served_source: Option<String>,
    pub relative: bool,
    pub savable: bool,
    /// Why Save is off (None = savable).
    pub not_savable: Option<String>,
    /// Save asks for a confirmation first (low confidence).
    pub confirm: bool,
    pub elapsed_ms: f64,
}

/// The run of `model` to tune ([`StoredPreds::run_for`], the rule Search uses) and its result.
fn pick_run<'a>(
    results: &'a BenchmarkResults,
    preds: &'a StoredPreds,
    model: &str,
    device: Option<&str>,
) -> Result<(&'a StoredRun, Option<&'a RunResult>)> {
    if preds.runs_of(model).is_empty() {
        bail!("{model}: no stored predictions; run a benchmark of this model first");
    }
    let m = results.find(model);
    let run = preds
        .run_for(model, device, m.and_then(|m| m.recommended.as_deref()))
        .map_err(|e| match e {
            crate::benchmark::search::NoRun::Empty => anyhow::anyhow!("{e}"),
            crate::benchmark::search::NoRun::NotOn { .. } => anyhow::anyhow!("{model}: {e}"),
        })?;
    let rr = m
        .and_then(|m| m.devices.iter().find(|d| d.device == run.device))
        .and_then(|d| d.run.as_ref());
    Ok((run, rr))
}

/// The scored classes of `map` the goal counts: its classes (any label naming a scored
/// class), else every scored class.
fn counted(map: &ClassMap, goal: &Goal) -> Vec<String> {
    if goal.classes.is_empty() {
        return map.classes.clone();
    }
    let mut v: Vec<String> = Vec::new();
    for c in &goal.classes {
        let canon = canonical_label(c);
        let class = map.classes.iter().find(|k| **k == canon).or_else(|| {
            map.gt_class(&canon)
                .and_then(|g| map.classes.iter().find(|k| *k == g))
        });
        if let Some(c) = class
            && !v.contains(c)
        {
            v.push(c.clone());
        }
    }
    v
}

/// One frame of a combo's tune set.
#[derive(Debug)]
struct TuneFrame<'a> {
    set: &'a str,
    file: &'a str,
    tags: &'a [String],
    scene: Option<crate::image_stats::Scene>,
    eval: EvalImage,
    frame_classes: Vec<String>,
    group: String,
    width: u32,
    height: u32,
}

/// The frames of `run` on `cs` in the model's own label space, counted classes only.
fn frames_of<'a>(
    cs: &CompareSet<'a>,
    run: &StoredRun,
    map: &ClassMap,
    classes: &[String],
) -> Vec<TuneFrame<'a>> {
    let index: HashMap<(&str, &str), usize> = run
        .images
        .iter()
        .enumerate()
        .map(|(k, i)| ((i.set.as_str(), i.file.as_str()), k))
        .collect();
    cs.images
        .iter()
        .filter_map(|img| {
            let k = *index.get(&(img.set.id.as_str(), img.info.file.as_str()))?;
            let gt = img.info.objects.as_deref()?;
            let (_, mut eval, allowed) =
                eval_input(map, img.set.scored_labels.as_deref(), gt, &run.preds(k));
            eval.gt.retain(|g| classes.contains(&g.label));
            eval.preds.retain(|p| classes.contains(&p.label));
            if let Some(a) = &mut eval.annotated {
                a.retain(|c| classes.contains(c));
            }
            let mut fc = frame_classes(map, allowed.as_deref());
            fc.retain(|c| classes.contains(c));
            Some(TuneFrame {
                width: img.info.width,
                height: img.info.height,
                set: img.set.id.as_str(),
                file: img.info.file.as_str(),
                tags: &img.info.tags,
                scene: img.info.scene,
                eval,
                frame_classes: fc,
                group: img
                    .info
                    .group
                    .clone()
                    .unwrap_or_else(|| format!("{}/{}", img.set.id, img.info.file)),
            })
        })
        .collect()
}

/// The sets `run` covers (its images of each, real labels first): (set, its images in the run).
fn sets_of<'a>(results: &'a BenchmarkResults, run: &StoredRun) -> Vec<(CompareSet<'a>, usize)> {
    let have: std::collections::HashSet<(&str, &str)> = run
        .images
        .iter()
        .map(|i| (i.set.as_str(), i.file.as_str()))
        .collect();
    let mut out: Vec<(CompareSet<'a>, usize)> = compare_sets(results)
        .into_iter()
        .filter(|s| {
            !s.datasets.iter().any(|d| {
                matches!(&d.ground_truth, GroundTruth::Pseudo(r) if is_reference(r, &run.model))
            })
        })
        .map(|s| {
            let n = s
                .images
                .iter()
                .filter(|i| have.contains(&(i.set.id.as_str(), i.info.file.as_str())))
                .count();
            (s, n)
        })
        .filter(|(_, n)| *n > 0)
        .collect();
    // Real labels first, then Your cameras, then the most frames.
    out.sort_by_key(|(s, n)| {
        let relative = s.datasets.iter().any(|d| !d.ground_truth.is_real());
        (
            relative,
            s.id != crate::benchmark::cameras::ID,
            std::cmp::Reverse(*n),
        )
    });
    out
}

/// The combo's frames and their assessment (shared by [`build`] and [`images`]).
#[derive(Debug)]
struct Prepared<'a> {
    run: &'a StoredRun,
    rr: Option<&'a RunResult>,
    set: CompareSet<'a>,
    sets: Vec<String>,
    classes: Vec<String>,
    frames: Vec<TuneFrame<'a>>,
    relative: bool,
}

fn prepare<'a>(
    results: &'a BenchmarkResults,
    preds: &'a StoredPreds,
    model: &str,
    device: Option<&str>,
    set: Option<&str>,
    goal: &Goal,
) -> Result<Prepared<'a>> {
    let (run, rr) = pick_run(results, preds, model, device)?;
    let sets = sets_of(results, run);
    let ids: Vec<String> = sets.iter().map(|(s, _)| s.id.clone()).collect();
    let chosen = match set {
        Some(id) => sets
            .into_iter()
            .find(|(s, _)| s.id == id)
            .map(|(s, _)| s)
            .with_context(|| {
                format!(
                    "{model} on {} has no scored frames of the set {id} (its sets: {})",
                    run.device,
                    ids.join(", ")
                )
            })?,
        None => sets
            .into_iter()
            .next()
            .map(|(s, _)| s)
            .with_context(|| format!("{model} on {}: no scored frames", run.device))?,
    };
    let map = ClassMap::for_model(&run.classes);
    let classes = counted(&map, goal);
    if classes.is_empty() {
        bail!(
            "none of the goal's classes ({}) is scored for {model} (scored: {})",
            goal.classes.join(", "),
            map.classes.join(", ")
        );
    }
    let frames = frames_of(&chosen, run, &map, &classes);
    let relative = chosen.datasets.iter().any(|d| !d.ground_truth.is_real());
    Ok(Prepared {
        run,
        rr,
        set: chosen,
        sets: ids,
        classes,
        frames,
        relative,
    })
}

fn assess_prepared(
    p: &Prepared<'_>,
    goal: &Goal,
    weights: crate::benchmark::grade::Weights,
) -> Assessment {
    let images: Vec<EvalImage> = p.frames.iter().map(|f| f.eval.clone()).collect();
    let fcs: Vec<Vec<String>> = p.frames.iter().map(|f| f.frame_classes.clone()).collect();
    let groups: Vec<String> = p.frames.iter().map(|f| f.group.clone()).collect();
    assess(Input {
        images: &images,
        frame_classes: &fcs,
        groups: &groups,
        classes: &p.classes,
        goal,
        // The set's p50, as Compare grades it.
        p50_ms: p.rr.and_then(|r| {
            super::compare::p50_on(&r.per_image, p.frames.iter().map(|f| (f.set, f.file)))
        }),
        weights,
        relative: p.relative,
    })
}

/// The Tune view (see the module docs).
pub fn build(
    results: &BenchmarkResults,
    preds: &StoredPreds,
    req: Request<'_>,
) -> Result<TuneView> {
    let started = std::time::Instant::now();
    let p = prepare(results, preds, req.model, req.device, req.set, req.goal)?;
    let a = assess_prepared(&p, req.goal, req.weights);
    let g = grid();
    let o = a.objective;
    let points = sum_all(&a.tallies, g.len());
    let grid_rows: Vec<GridRow> = g
        .iter()
        .zip(&points)
        .map(|(&t, x)| GridRow {
            threshold: t,
            tp: x.tp,
            fp: x.fp,
            fn_: x.fn_,
            precision: x.precision(),
            recall: x.recall(),
            f1: x.f_beta(1.0),
            f2: x.f_beta(2.0),
            f05: x.f_beta(0.5),
            fa100: x.fa100(),
            tpr: x.tpr(),
            fpr: x.fpr(),
            score: score(o, x),
            bound: bound(o, x),
        })
        .collect();
    // The tuning objective's out-of-fold value and interval (the goal metric when the goal is
    // a threshold objective; else recomputed for the tuning objective).
    let metric = if req.goal.optimize_for.rank_only() {
        let tuned = Goal {
            optimize_for: super::goal::Optimize::Threshold(o),
            ..req.goal.clone()
        };
        assess_prepared(&p, &tuned, req.weights).metric
    } else {
        a.metric
    };
    let suggested = a.suggested.and_then(|s| {
        Some(Suggestion {
            threshold: *g.get(s.index)?,
            plateau: [*g.get(s.first)?, *g.get(s.last)?],
            best: *g.get(s.best)?,
            in_sample: points.get(s.index).and_then(|x| value(o, x)),
            metric,
        })
    });
    let in_sample_at = a.suggested.and_then(|s| points.get(s.index));
    let target_state = target(o).map(|t| {
        let met_oof = met(o, &a.pooled) == Some(true);
        let b = bound(o, &a.pooled);
        let b_in = in_sample_at.and_then(|x| bound(o, x));
        let fmt = |v: f64| match o {
            crate::benchmark::threshold::Objective::RecallAtFa(_) => format!("{v:.1}"),
            _ => format!("{v:.2}"),
        };
        let (rel, word) = match o {
            crate::benchmark::threshold::Objective::Precision(_)
            | crate::benchmark::threshold::Objective::Recall(_) => ("\u{2265}", "lower"),
            _ => ("\u{2264}", "upper"),
        };
        let text = match (met_oof, b) {
            (true, Some(b)) => format!("met (95% {word} bound {} {rel} {})", fmt(b), fmt(t)),
            (false, Some(b)) => format!(
                "not met: 95% {word} bound {} (needs {rel} {})",
                fmt(b),
                fmt(t)
            ),
            (_, None) => "not met: nothing to judge it on".to_string(),
        };
        TargetState {
            value: t,
            met: met_oof,
            bound: b,
            bound_in_sample: b_in,
            text,
        }
    });
    let not_savable = if p.relative {
        Some(format!("not savable: {NOT_ON_LABELS}"))
    } else if target_state.as_ref().is_some_and(|t| !t.met) {
        Some("not savable: the target is not met on its 95% bound".to_string())
    } else if suggested.is_none() {
        Some("not savable: no confidence could be picked on these frames".to_string())
    } else if !a.pick_informative() {
        Some(format!("not savable: {FLAT_NOTE}"))
    } else {
        None
    };
    let devices: Vec<String> = preds
        .runs_of(req.model)
        .iter()
        .map(|r| r.device.clone())
        .collect();
    let rr = p.rr;
    Ok(TuneView {
        model: p.run.model.clone(),
        device: p.run.device.clone(),
        devices,
        set: p.set.id.clone(),
        set_title: p.set.title.clone(),
        sets: p.sets.clone(),
        goal: GoalView::of(req.goal),
        objective: o.to_string(),
        objective_name: objective_short(o),
        frames: p.frames.len(),
        objects: a.objects,
        empty_frames: points.first().map_or(0, |x| x.empty),
        groups: a.groups.len(),
        folds: if a.in_sample { 0 } else { a.cv.k },
        classes: p.classes.clone(),
        grid: grid_rows,
        suggested,
        low_confidence: a.low_confidence.clone(),
        pick_informative: a.pick_informative(),
        flat_note: (!a.pick_informative()).then(|| FLAT_NOTE.to_string()),
        target: target_state,
        saved: req.saved.map(|(t, s)| Saved {
            threshold: t,
            source: s.to_string(),
        }),
        served_threshold: rr.and_then(|r| r.served_threshold),
        served_source: rr.and_then(|r| r.served_threshold_source.clone()),
        relative: p.relative,
        savable: not_savable.is_none(),
        not_savable,
        confirm: a.low_confidence.is_some(),
        elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
    })
}

/// One combo's suggested confidence as every "apply all" writes it ([`suggestions`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Suggested {
    pub model: String,
    pub device: String,
    /// The suggestion; None when nothing could be picked (or the view failed: `refusal`).
    pub threshold: Option<f32>,
    /// Why it must not be saved (pseudo labels, a target not met on its bound, a flat curve, no
    /// pick); None = savable.
    pub refusal: Option<String>,
    /// Too little data to cross-validate: saved only when asked for (the page confirms first).
    pub low_confidence: Option<String>,
    /// The model's recommended device of the saved results.
    pub recommended: bool,
    /// What the suggestion found on the frames it was picked on (in-sample, at the
    /// suggestion); None without one.
    pub figures: Option<PickFigures>,
}

/// The in-sample figures at a suggestion ([`Suggested::figures`]): what "Use as" puts in words.
#[derive(Debug, Clone, PartialEq)]
pub struct PickFigures {
    /// The compare set it was picked on and its frames.
    pub set: String,
    pub frames: usize,
    pub recall: Option<f64>,
    pub precision: Option<f64>,
}

impl PickFigures {
    fn of(v: &TuneView) -> Option<Self> {
        let t = v.suggested.as_ref()?.threshold;
        let row = v.grid.iter().find(|r| (r.threshold - t).abs() < 1e-6)?;
        Some(Self {
            set: v.set.clone(),
            frames: v.frames,
            recall: row.recall,
            precision: row.precision,
        })
    }
}

/// The suggested confidence of every bare combo (no `+served` variant) with stored
/// predictions, models in result order, devices in stored order, limited to `only` (normalized
/// names): the Tune view's pick ([`build`] on its default set), the one rule behind the page,
/// `apply-threshold` / `apply-profiles` `all=1` and the CLI's `--apply-threshold`.
pub fn suggestions(
    results: &BenchmarkResults,
    preds: &StoredPreds,
    goal: &Goal,
    weights: crate::benchmark::grade::Weights,
    only: Option<&[String]>,
) -> Vec<Suggested> {
    let mut out = Vec::new();
    for m in &results.models {
        let key = normalize_name(&m.model);
        if crate::benchmark::variant::base_name(&m.model) != m.model
            || only.is_some_and(|o| !o.iter().any(|x| normalize_name(x) == key))
        {
            continue;
        }
        for run in preds.runs_of(&m.model) {
            let recommended = m.recommended.as_deref() == Some(run.device.as_str());
            let req = Request {
                model: &m.model,
                device: Some(&run.device),
                set: None,
                goal,
                weights,
                saved: None,
            };
            out.push(match build(results, preds, req) {
                Ok(v) => Suggested {
                    model: m.model.clone(),
                    device: run.device.clone(),
                    threshold: v.suggested.as_ref().map(|s| s.threshold),
                    refusal: v.not_savable.clone(),
                    low_confidence: v.low_confidence.clone(),
                    recommended,
                    figures: PickFigures::of(&v),
                },
                Err(e) => Suggested {
                    model: m.model.clone(),
                    device: run.device.clone(),
                    threshold: None,
                    refusal: Some(format!("not savable: {e:#}")),
                    low_confidence: None,
                    recommended,
                    figures: None,
                },
            });
        }
    }
    out
}

/// Which frames [`images`] lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Frames with missed objects, the most missed first.
    Miss,
    /// Frames with false positives, the most confident one first.
    Fa,
    /// Every frame, the most errors (missed objects and false positives) first.
    All,
}

impl std::str::FromStr for Kind {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "miss" | "misses" | "" => Ok(Kind::Miss),
            "fa" | "false_alerts" | "false-alerts" | "fp" => Ok(Kind::Fa),
            "all" => Ok(Kind::All),
            _ => Err(format!("kind '{s}': use miss, fa or all")),
        }
    }
}

/// A box of an example frame.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FrameBox {
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    pub bbox: [f32; 4],
    /// Ground truth: matched; prediction: true positive. None = ignored.
    pub ok: Option<bool>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub ignore: bool,
}

/// An example frame.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExampleFrame {
    pub set: String,
    pub file: String,
    /// The image's size in pixels (the boxes' space).
    pub width: u32,
    pub height: u32,
    /// Your cameras: the frame's id in the review folder (`<scene>/<camera>/<date>/<stem>.jpg`,
    /// for a link into Review).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub review: Option<String>,
    pub camera: Option<String>,
    pub scene: Option<String>,
    pub date: Option<String>,
    pub missed: usize,
    pub false_positives: usize,
    /// The most confident false positive.
    pub top_false: Option<f32>,
    pub gt: Vec<FrameBox>,
    pub preds: Vec<FrameBox>,
}

/// The example frames.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Examples {
    pub model: String,
    pub device: String,
    pub set: String,
    pub threshold: f32,
    pub kind: String,
    /// Frames of the kind and filters (before `offset` and the limit).
    pub total: usize,
    /// Frames skipped before the first one listed.
    pub offset: usize,
    /// The cameras of the set's frames (Your cameras), for a filter.
    pub cameras: Vec<String>,
    pub frames: Vec<ExampleFrame>,
}

/// What [`images`] lists.
#[derive(Debug, Clone, Copy)]
pub struct ImagesRequest<'a> {
    pub model: &'a str,
    pub device: Option<&'a str>,
    pub set: Option<&'a str>,
    pub goal: &'a Goal,
    pub threshold: f32,
    pub kind: Kind,
    /// Only frames of this camera (`camera:` tag).
    pub camera: Option<&'a str>,
    /// Only frames of this scene (`day`, `night`, `ir`).
    pub scene: Option<&'a str>,
    pub offset: usize,
    pub limit: usize,
}

/// Frames for the Tune strips and the Images page: at `threshold`, the frames with misses or
/// false positives (or every frame), worst first, of one camera and scene when asked, from
/// `offset`, at most `limit`.
pub fn images(
    results: &BenchmarkResults,
    preds: &StoredPreds,
    req: ImagesRequest<'_>,
) -> Result<Examples> {
    let ImagesRequest {
        model,
        device,
        set,
        goal,
        threshold,
        kind,
        camera,
        scene,
        offset,
        limit,
    } = req;
    let p = prepare(results, preds, model, device, set, goal)?;
    let tag = |tags: &[String], prefix: &str| {
        tags.iter()
            .find_map(|t| t.strip_prefix(prefix))
            .map(str::to_string)
    };
    let scene_of = |f: &TuneFrame<'_>| {
        f.scene.map(|s| s.as_str().to_string()).or_else(|| {
            ["day", "night", "ir"]
                .iter()
                .find(|s| f.tags.iter().any(|t| t == *s))
                .map(|s| (*s).to_string())
        })
    };
    let mut cameras: Vec<String> = p
        .frames
        .iter()
        .filter_map(|f| tag(f.tags, "camera:"))
        .collect();
    cameras.sort();
    cameras.dedup();
    let cameras_set = p.set.id == crate::benchmark::cameras::ID;
    let mut frames: Vec<ExampleFrame> = p
        .frames
        .iter()
        .filter(|f| camera.is_none_or(|c| tag(f.tags, "camera:").as_deref() == Some(c)))
        .filter(|f| scene.is_none_or(|s| scene_of(f).as_deref() == Some(s)))
        .filter_map(|f| {
            let (c, kept, m) = image_counts(&f.eval, threshold);
            let top_false = kept
                .iter()
                .zip(&m.pred)
                .filter(|(_, r)| **r == Some(false))
                .map(|(p, _)| p.confidence)
                .fold(None, |a: Option<f32>, x| Some(a.map_or(x, |a| a.max(x))));
            let wanted = match kind {
                Kind::Miss => c.fn_ > 0,
                Kind::Fa => c.fp > 0,
                Kind::All => true,
            };
            wanted.then(|| ExampleFrame {
                set: f.set.to_string(),
                file: f.file.to_string(),
                width: f.width,
                height: f.height,
                review: cameras_set
                    .then(|| crate::collect::labels::export::parse_flat_name(f.file))
                    .flatten()
                    .map(|rel| rel.jpg()),
                camera: tag(f.tags, "camera:"),
                scene: scene_of(f),
                date: tag(f.tags, "date:"),
                missed: c.fn_,
                false_positives: c.fp,
                top_false,
                gt: f
                    .eval
                    .gt
                    .iter()
                    .zip(&m.gt_matched)
                    .map(|(g, &hit)| FrameBox {
                        label: g.label.clone(),
                        confidence: None,
                        bbox: g.bbox,
                        ok: (!g.ignore).then_some(hit),
                        ignore: g.ignore,
                    })
                    .collect(),
                preds: kept
                    .iter()
                    .zip(&m.pred)
                    .map(|(p, r)| FrameBox {
                        label: p.label.clone(),
                        confidence: Some(p.confidence),
                        bbox: p.bbox,
                        ok: *r,
                        ignore: false,
                    })
                    .collect(),
            })
        })
        .collect();
    match kind {
        Kind::Miss => frames.sort_by(|a, b| b.missed.cmp(&a.missed).then(a.file.cmp(&b.file))),
        Kind::Fa => frames.sort_by(|a, b| {
            b.top_false
                .unwrap_or(0.0)
                .total_cmp(&a.top_false.unwrap_or(0.0))
                .then(a.file.cmp(&b.file))
        }),
        Kind::All => frames.sort_by(|a, b| {
            (b.missed + b.false_positives)
                .cmp(&(a.missed + a.false_positives))
                .then(a.file.cmp(&b.file))
        }),
    }
    let total = frames.len();
    let frames: Vec<ExampleFrame> = frames.into_iter().skip(offset).take(limit).collect();
    Ok(Examples {
        model: p.run.model.clone(),
        device: p.run.device.clone(),
        set: p.set.id.clone(),
        threshold,
        kind: match kind {
            Kind::Miss => "miss",
            Kind::Fa => "fa",
            Kind::All => "all",
        }
        .to_string(),
        total,
        offset,
        cameras,
        frames,
    })
}

/// The normalized model name of a view (cache keys).
pub fn key_of(model: &str) -> String {
    normalize_name(model)
}

/// The goal metric of one device run on its own frames, for the run history
/// ([`crate::benchmark::history::DeviceSnap::goal`]).
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
pub struct GoalSnap {
    /// `optimize_for` of the goal it was computed for.
    pub optimize_for: String,
    /// The tuning objective (the confidence pick).
    #[serde(default)]
    pub tuning: String,
    pub value: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high: Option<f64>,
    /// The suggested confidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// In-sample figures (too little data to cross-validate).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub low_confidence: bool,
    /// Frames and objects it rests on.
    #[serde(default)]
    pub frames: usize,
    #[serde(default)]
    pub objects: usize,
    /// The classes counted.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub classes: Vec<String>,
}

/// One frame of [`run_snapshot`]: real labels, its evaluation input, its frame classes, its
/// cross-validation group and its (set id, file).
type SnapFrame<'a> = (bool, EvalImage, Vec<String>, String, (&'a str, &'a str));

/// The [`GoalSnap`] of `run` (the stored form of a device run) over every frame it scored with
/// real labels (else every scored frame, relative); None when nothing is scored. A grade's
/// speed is the p50 of `per_image` (the device run's request times) on those frames, as
/// Compare grades a set.
pub fn run_snapshot(
    image_sets: &[crate::benchmark::report::ImageSetInfo],
    run: &StoredRun,
    per_image: &[crate::benchmark::ImageRun],
    goal: &Goal,
    weights: crate::benchmark::grade::Weights,
) -> Option<GoalSnap> {
    let sets: HashMap<&str, &crate::benchmark::report::ImageSetInfo> =
        image_sets.iter().map(|s| (s.id.as_str(), s)).collect();
    let infos: HashMap<(&str, &str), &crate::benchmark::report::ImageInfo> = image_sets
        .iter()
        .flat_map(|s| {
            s.images
                .iter()
                .map(move |i| ((s.id.as_str(), i.file.as_str()), i))
        })
        .collect();
    let map = ClassMap::for_model(&run.classes);
    let classes = counted(&map, goal);
    if classes.is_empty() {
        return None;
    }
    let mut frames: Vec<SnapFrame<'_>> = Vec::new();
    for (k, img) in run.images.iter().enumerate() {
        let Some(set) = sets.get(img.set.as_str()) else {
            continue;
        };
        if matches!(&set.ground_truth, GroundTruth::Pseudo(r) if is_reference(r, &run.model)) {
            continue;
        }
        let Some(info) = infos.get(&(img.set.as_str(), img.file.as_str())) else {
            continue;
        };
        let Some(gt) = &info.objects else { continue };
        let (_, mut eval, allowed) =
            eval_input(&map, set.scored_labels.as_deref(), gt, &run.preds(k));
        eval.gt.retain(|g| classes.contains(&g.label));
        eval.preds.retain(|p| classes.contains(&p.label));
        if let Some(a) = &mut eval.annotated {
            a.retain(|c| classes.contains(c));
        }
        let mut fc = frame_classes(&map, allowed.as_deref());
        fc.retain(|c| classes.contains(c));
        let group = info
            .group
            .clone()
            .unwrap_or_else(|| format!("{}/{}", set.id, info.file));
        frames.push((
            set.ground_truth.is_real(),
            eval,
            fc,
            group,
            (img.set.as_str(), img.file.as_str()),
        ));
    }
    let real = frames.iter().any(|f| f.0);
    frames.retain(|f| f.0 == real);
    if frames.is_empty() {
        return None;
    }
    let images: Vec<EvalImage> = frames.iter().map(|f| f.1.clone()).collect();
    let fcs: Vec<Vec<String>> = frames.iter().map(|f| f.2.clone()).collect();
    let groups: Vec<String> = frames.iter().map(|f| f.3.clone()).collect();
    let p50_ms = super::compare::p50_on(per_image, frames.iter().map(|f| f.4));
    let a = assess(Input {
        images: &images,
        frame_classes: &fcs,
        groups: &groups,
        classes: &classes,
        goal,
        p50_ms,
        weights,
        relative: !real,
    });
    let m = a.metric?;
    Some(GoalSnap {
        optimize_for: goal.optimize_for.to_string(),
        tuning: a.objective.to_string(),
        value: m.value,
        low: m.low,
        high: m.high,
        confidence: a.suggested_threshold(),
        low_confidence: a.low_confidence.is_some(),
        frames: images.len(),
        objects: a.objects,
        classes,
    })
}
