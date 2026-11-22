//! The Compare view (`GET /v1/benchmark/compare`): every combo (model x device) on one image
//! set, judged for the goal ([`super::assess`]), ranked with rank groups ([`super::rank`]), with
//! one headline sentence and the breakdowns (scenes, classes, cameras, speed).
//!
//! - **Frames**: a set's frames that every listed combo ran and no listed model trained or
//!   validated on ([`crate::benchmark::combos`]'s rules; the left-out count is shown), optionally
//!   one scene (`day`, `night`, `ir`). Every combo is scored on exactly these frames, so their
//!   groups and bootstrap resamples are the same and two combos compare pair by pair.
//! - **Classes**: the classes every combo is scored on (car, truck and bus are one `vehicle`, as
//!   on every Compare set), limited to the goal's classes when it names some (`ignoredClasses`: named
//!   but not scored here).
//! - **Speed**: one-at-a-time p50 / p95 / p99 of the whole request on these frames (each frame's
//!   p50 over its timed runs); the budget applies to the p95, the burst test's request p95 once
//!   one ran on the combo ([`crate::benchmark::burst`]: latency under load, queue wait included).
//!   Stage split, first load (compile, warm-up, cache) and the burst's figures are the run's.
//! - **Rank**: [`super::rank`]; a row over the budget is "too slow", a target not met on its
//!   bound is "unmet", a model that trained on these frames (or whose training lists do not
//!   read) is shown, never ranked among the fair rows.
//! - **Breakdowns**, all at each row's reported (cross-validated) picks: per scene (metric,
//!   recall, FA/100, objects; cells under [`crate::benchmark::cameras::MIN_OBJECTS`] objects
//!   flagged), per class (objects, P, R, F2, AP50, false alerts per 100 frames without the
//!   class), per camera on "Your cameras" (frames, objects, recall, FA/100, worst scene).

use super::assess::{FLAT_NOTE, Input, assess};
use super::cv::Groups;
use super::goal::{Goal, Optimize, objective_short};
use super::pick::{bound, target, value};
use super::rank::{Block, Place, Rankable, rank};
use super::rankfree::{ApData, AucData};
use super::stats::{Difference, Estimate};
use crate::benchmark::cameras::MIN_OBJECTS;
use crate::benchmark::combos::sets::{CompareSet, compare_sets, default_set};
use crate::benchmark::combos::{Combo, ComboSet, Inputs, Missing, RunHint, combos_on, score};
use crate::benchmark::grade::Weights;
use crate::benchmark::images::in_split;
use crate::benchmark::metrics::{ClassMap, EvalImage, SweepScope, sweep_counts};
use crate::benchmark::threshold::grid;
use crate::registry::normalize_name;
use serde::Serialize;
use std::collections::HashMap;

/// The scenes of the breakdown and the `scene=` filter.
pub const SCENES: [&str; 3] = ["day", "night", "ir"];

/// What [`build`] reads besides the combos' inputs ([`crate::benchmark::combos::Inputs`]).
#[derive(Debug, Clone, Copy)]
pub struct Request<'a> {
    pub goal: &'a Goal,
    pub weights: Weights,
    /// A compare set id; None = the default set.
    pub set: Option<&'a str>,
    /// `day`, `night` or `ir`; None = every frame.
    pub scene: Option<&'a str>,
}

/// The goal as the view uses it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoalView {
    pub optimize_for: String,
    pub tune_for: String,
    pub classes: Vec<String>,
    pub speed_budget_ms: Option<f64>,
    /// "F2 · person, car · budget 1 s".
    pub summary: String,
    /// The metric column's name ("F2", "AP50").
    pub metric: String,
    pub rank_only: bool,
    /// The objective each row's confidence is picked for.
    pub tuning: String,
}

impl GoalView {
    pub fn of(g: &Goal) -> Self {
        Self {
            optimize_for: g.optimize_for.to_string(),
            tune_for: g.tune_for.to_string(),
            classes: g.classes.clone(),
            speed_budget_ms: g.speed_budget_ms,
            summary: g.describe(),
            metric: g.optimize_for.short(),
            rank_only: g.optimize_for.rank_only(),
            tuning: objective_short(g.tuning()),
        }
    }
}

/// A set to choose from.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SetOption {
    pub id: String,
    pub title: String,
    /// Fair frames (see the module docs).
    pub frames: usize,
    pub combos: usize,
}

/// The chosen set.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SetInfo {
    pub id: String,
    pub title: String,
    pub what: String,
    /// `frames` or `images`.
    pub unit: String,
    /// Frames compared (after the scene filter).
    pub frames: usize,
    /// Frames out because a listed model trained or validated on them.
    pub frames_left_out: usize,
    pub left_out_by: Vec<String>,
    /// `day`, `night`, `ir` or None (every frame).
    pub scene: Option<String>,
    /// Frames per scene of the fair set.
    pub scenes: Vec<SceneCount>,
    /// Labelled objects of the counted classes.
    pub objects: usize,
    pub empty_frames: usize,
    pub relative: bool,
    pub reference: Option<String>,
    /// The counted classes.
    pub classes: Vec<String>,
    /// Every class all combos here are scored on (what the goal may count; the Goal dialog's
    /// choices).
    pub scored_classes: Vec<String>,
    /// Classes the goal names that no combo here is scored on.
    pub ignored_classes: Vec<String>,
    /// Cross-validation groups (camera days on Your cameras, single images elsewhere).
    pub groups: usize,
    pub run: RunHint,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SceneCount {
    pub scene: String,
    pub frames: usize,
}

/// First load of a combo (the run's).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FirstLoad {
    pub compile_ms: f64,
    pub warmup_ms: f64,
    /// `hit`, `miss`, `unknown`, `none`, `disabled` (empty in older results).
    pub cache: String,
}

/// Speed of a combo on the frames.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Speed {
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    pub p99_ms: Option<f64>,
    /// Frames timed.
    pub frames: usize,
    /// Run-level stage p50s (every frame of the run).
    pub decode_ms: f64,
    pub preprocess_ms: f64,
    pub infer_ms: f64,
    pub postprocess_ms: f64,
    /// The burst test's request p95 (None: no burst ran on this combo, or it measured nothing).
    pub burst_p95_ms: Option<f64>,
    /// The burst test's figures (None: none ran).
    pub burst: Option<BurstSpeed>,
    pub first_load: FirstLoad,
}

/// What a combo's burst test measured ([`crate::benchmark::burst::BurstResult`]), for the page.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BurstSpeed {
    /// "8 x 4 x 500 ms" (cameras x images x interval).
    pub load: String,
    pub cameras: usize,
    pub images: usize,
    pub interval_ms: u64,
    pub bursts: usize,
    pub requests: usize,
    pub answered: usize,
    /// Request latency from the intended send time (queue wait included).
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    pub p99_ms: Option<f64>,
    pub max_ms: Option<f64>,
    /// Per trigger: from its first intended send to its last answer.
    pub last_answer_p50_ms: Option<f64>,
    pub last_answer_p95_ms: Option<f64>,
    pub triggers: usize,
    pub triggers_incomplete: usize,
    pub refused: u64,
    /// "queue_full 3, timeout 1".
    pub refused_by: String,
    pub max_queue_depth: usize,
    pub queue_capacity: usize,
    pub inflight: usize,
    pub served: bool,
    /// Why it measured nothing.
    pub error: Option<String>,
    pub notes: Vec<String>,
    /// Detection requests per minute the service answered meanwhile.
    pub live_load: Option<f64>,
}

impl BurstSpeed {
    pub fn of(b: &crate::benchmark::burst::BurstResult) -> Self {
        let measured = b.error.is_none() && b.answered > 0;
        let some = |v: f64| measured.then_some(v);
        let last = b.error.is_none() && b.triggers > b.triggers_incomplete;
        Self {
            load: b.load_text(),
            cameras: b.cameras,
            images: b.images,
            interval_ms: b.interval_ms,
            bursts: b.bursts,
            requests: b.requests,
            answered: b.answered,
            p50_ms: some(b.latency_ms.p50),
            p95_ms: some(b.latency_ms.p95),
            p99_ms: some(b.latency_ms.p99),
            max_ms: some(b.latency_ms.max),
            last_answer_p50_ms: last.then_some(b.trigger_ms.p50),
            last_answer_p95_ms: last.then_some(b.trigger_ms.p95),
            triggers: b.triggers,
            triggers_incomplete: b.triggers_incomplete,
            refused: b.refused_total(),
            refused_by: b
                .refused
                .iter()
                .map(|(k, v)| format!("{k} {v}"))
                .collect::<Vec<_>>()
                .join(", "),
            max_queue_depth: b.max_queue_depth,
            queue_capacity: b.queue_capacity,
            inflight: b.inflight,
            served: b.served,
            error: b.error.clone(),
            notes: b.notes.clone(),
            live_load: b.live_requests_per_min,
        }
    }
}

/// Flags of a row.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Flags {
    pub too_slow: bool,
    /// The target is not met on its bound.
    pub unmet: bool,
    /// Trained on some of these frames, or what it saw cannot be told: never ranked fairly.
    pub trained_here: bool,
    pub trained_on_share: Option<f64>,
    pub unknown_seen: Option<String>,
    pub relative: bool,
    /// An As served result (`<model>+served`).
    pub served: bool,
    /// Its serving rules changed since the run.
    pub stale_rules: bool,
    pub low_confidence: Option<String>,
    /// Detection requests per minute the service answered while it ran.
    pub live_load: Option<f64>,
    /// The GPU's video-decode load while it ran (Windows), percent.
    pub video_decode: Option<f64>,
    /// The pick is not informative (flat curve).
    pub flat: bool,
}

/// A row's target, when the objective has one.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Target {
    pub value: f64,
    pub met: bool,
    /// The bound the target is judged on, of the reported counts (FA/100 in per 100 frames).
    pub bound: Option<f64>,
}

/// One scene of a row.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SceneRow {
    pub scene: String,
    pub frames: usize,
    pub objects: usize,
    pub metric: Option<f64>,
    pub recall: Option<f64>,
    pub fa100: Option<f64>,
    /// Fewer than [`MIN_OBJECTS`] objects: shown greyed.
    pub few_objects: bool,
}

/// One class of a row.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassRow {
    pub class: String,
    pub objects: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f2: Option<f64>,
    pub ap50: Option<f64>,
    /// Frames whose set labels the class and that have no box of it.
    pub frames_without: usize,
    /// False alerts of the class per 100 of those frames.
    pub fa100: Option<f64>,
}

/// One camera of a row (Your cameras).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CameraRow {
    pub camera: String,
    pub frames: usize,
    pub objects: usize,
    pub recall: Option<f64>,
    pub fa100: Option<f64>,
    /// The scene with the lowest recall (with objects).
    pub worst_scene: Option<String>,
}

/// A combo's in-sample figures at one confidence on every frame of the view (Tune's "now ->
/// new": the saved confidence against the suggested one, on the same frames).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AtPoint {
    /// The grid value the figures are at.
    pub threshold: f32,
    /// The tuning objective's value ([`super::pick::value`]).
    pub value: Option<f64>,
    pub recall: Option<f64>,
    pub precision: Option<f64>,
    pub fa100: Option<f64>,
}

/// [`AtPoint`] of `a` at the grid value nearest `t` (None off the grid's range).
fn at_point(a: &super::assess::Assessment, t: f32) -> Option<AtPoint> {
    let g = grid();
    let k = g
        .iter()
        .enumerate()
        .min_by(|x, y| (x.1 - t).abs().total_cmp(&(y.1 - t).abs()))
        .filter(|(_, v)| (*v - t).abs() <= 0.006)
        .map(|(k, _)| k)?;
    let mut at = super::tally::At::default();
    for img in &a.tallies {
        at.add_image(img, k, 1);
    }
    Some(AtPoint {
        threshold: *g.get(k)?,
        value: value(a.objective, &at),
        recall: at.recall(),
        precision: at.precision(),
        fa100: at.fa100(),
    })
}

/// One combo.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Row {
    pub model: String,
    pub device: String,
    pub device_label: String,
    pub is_default: bool,
    /// The goal metric (out-of-fold for a threshold goal) and its 95% interval.
    pub metric: Option<Estimate>,
    pub position: usize,
    pub rank: usize,
    pub block: Block,
    /// Its rank group's leader (`model on device`), when not itself.
    pub leader: Option<String>,
    pub vs_leader: Option<Difference>,
    pub recall: Option<f64>,
    pub precision: Option<f64>,
    pub f1: Option<f64>,
    pub f2: Option<f64>,
    pub f05: Option<f64>,
    pub fa100: Option<f64>,
    pub tpr: Option<f64>,
    pub fpr: Option<f64>,
    pub ap50: Option<f64>,
    pub ap50_95: Option<f64>,
    pub roc_auc: Option<f64>,
    pub grade: Option<String>,
    pub grade_points: Option<f64>,
    /// The suggested confidence (the pick on every frame).
    pub confidence: Option<f32>,
    pub plateau: Option<[f32; 2]>,
    pub pick_informative: bool,
    pub target: Option<Target>,
    /// The confidence the config gives this combo now.
    pub served_threshold: Option<f32>,
    /// In-sample figures at the confidence saved now and at the suggested one.
    pub at_saved: Option<AtPoint>,
    pub at_suggested: Option<AtPoint>,
    /// Cross-validation folds (0 or 1: none).
    pub folds: usize,
    pub speed: Speed,
    /// The CPU agreement verdict (`match`, `minor`, ..., `not_checked`).
    pub agreement: Option<String>,
    pub flags: Flags,
    pub scenes: Vec<SceneRow>,
    pub classes: Vec<ClassRow>,
    pub cameras: Vec<CameraRow>,
}

/// The view.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompareView {
    pub goal: GoalView,
    pub sets: Vec<SetOption>,
    pub set: Option<SetInfo>,
    pub headline: String,
    /// Best first.
    pub rows: Vec<Row>,
    pub missing: Vec<Missing>,
    /// Shown under the table.
    pub notes: Vec<String>,
    pub elapsed_ms: f64,
}

/// The p-th nearest-rank percentile of `v`.
fn pct_of(v: &[f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    Some(crate::benchmark::percentile(&s, p))
}

/// The frames' counted-class evaluation inputs of one combo.
struct Frames {
    images: Vec<EvalImage>,
    frame_classes: Vec<Vec<String>>,
    groups: Vec<String>,
    tags: Vec<Vec<String>>,
    scenes: Vec<Option<crate::image_stats::Scene>>,
}

fn frames_of(set: &CompareSet<'_>, map: &ClassMap, c: &Combo<'_>, classes: &[String]) -> Frames {
    let index = score::index_of(c.stored);
    let mut f = Frames {
        images: Vec::new(),
        frame_classes: Vec::new(),
        groups: Vec::new(),
        tags: Vec::new(),
        scenes: Vec::new(),
    };
    for img in &set.images {
        let Some(&k) = index.get(&(img.set.id.as_str(), img.info.file.as_str())) else {
            continue;
        };
        let mut e = score::eval_of(img, map, c.stored, k);
        e.gt.retain(|g| classes.contains(&g.label));
        e.preds.retain(|p| classes.contains(&p.label));
        if let Some(a) = &mut e.annotated {
            a.retain(|x| classes.contains(x));
        }
        let fc: Vec<String> = classes.iter().filter(|x| e.annotates(x)).cloned().collect();
        f.images.push(e);
        f.frame_classes.push(fc);
        f.groups.push(
            img.info
                .group
                .clone()
                .unwrap_or_else(|| format!("{}/{}", img.set.id, img.info.file)),
        );
        f.tags.push(img.info.tags.clone());
        f.scenes.push(img.info.scene);
    }
    f
}

fn speed_of(set: &CompareSet<'_>, c: &Combo<'_>) -> Speed {
    let by: HashMap<(&str, &str), f64> = c
        .run
        .per_image
        .iter()
        .map(|i| ((i.set.as_str(), i.file.as_str()), i.total_ms))
        .collect();
    let times: Vec<f64> = set
        .images
        .iter()
        .filter_map(|i| by.get(&(i.set.id.as_str(), i.info.file.as_str())).copied())
        .filter(|t| t.is_finite())
        .collect();
    let st = &c.run.stages_ms;
    let burst = c.run.burst.as_ref().map(BurstSpeed::of);
    Speed {
        p50_ms: pct_of(&times, 50.0),
        p95_ms: pct_of(&times, 95.0),
        p99_ms: pct_of(&times, 99.0),
        frames: times.len(),
        decode_ms: st.decode.p50,
        preprocess_ms: st.preprocess.p50,
        infer_ms: st.infer.p50,
        postprocess_ms: st.postprocess.p50,
        burst_p95_ms: burst.as_ref().and_then(|b| b.p95_ms),
        burst,
        first_load: FirstLoad {
            compile_ms: c.run.compile_ms,
            warmup_ms: c.run.warmup_ms,
            cache: c.run.cache.clone(),
        },
    }
}

/// The scene breakdown of an assessment.
fn scene_rows(a: &super::assess::Assessment, f: &Frames, goal: &Goal) -> Vec<SceneRow> {
    SCENES
        .iter()
        .filter_map(|scene| {
            let keep: Vec<usize> = (0..f.images.len())
                .filter(|&i| {
                    in_split(
                        f.tags.get(i).map_or(&[][..], Vec::as_slice),
                        f.scenes.get(i).copied().flatten(),
                        scene,
                    )
                })
                .collect();
            if keep.is_empty() {
                return None;
            }
            let at = a.pooled_where(|i| keep.binary_search(&i).is_ok());
            let metric = match goal.optimize_for {
                Optimize::Threshold(o) => value(o, &at),
                rank_only => rank_only_of(a, f, &keep, rank_only),
            };
            Some(SceneRow {
                scene: (*scene).to_string(),
                frames: keep.len(),
                objects: at.objects(),
                metric,
                recall: at.recall(),
                fa100: at.fa100(),
                few_objects: at.objects() < MIN_OBJECTS,
            })
        })
        .collect()
}

/// A rank-only metric over the images `keep` of an assessment, without an interval: AP50,
/// AP50-95 or the frame ROC AUC; a grade goal's scene cell shows the scene's AP50 (a grade per
/// scene would mix the set's speed into it).
fn rank_only_of(
    a: &super::assess::Assessment,
    f: &Frames,
    keep: &[usize],
    optimize: Optimize,
) -> Option<f64> {
    let images: Vec<EvalImage> = keep
        .iter()
        .filter_map(|&i| f.images.get(i).cloned())
        .collect();
    let groups = Groups::new(
        keep.iter()
            .map(|&i| f.groups.get(i).map_or("", String::as_str)),
    );
    match optimize {
        Optimize::RocAuc => {
            let sweep: Vec<_> = keep
                .iter()
                .filter_map(|&i| a.sweep.get(i).cloned())
                .collect();
            AucData::new(&sweep, &groups).auc(None)
        }
        Optimize::Ap5095 => ApData::new(&images, &groups, &a.ap.classes, true).ap50_95(None),
        _ => ApData::new(&images, &groups, &a.ap.classes, false).ap50(None),
    }
}

/// The per-class breakdown at the reported picks.
fn class_rows(a: &super::assess::Assessment, f: &Frames, classes: &[String]) -> Vec<ClassRow> {
    classes
        .iter()
        .map(|c| {
            let mut counts = crate::benchmark::metrics::Counts::default();
            let (mut without, mut alerts) = (0usize, 0usize);
            for (i, sw) in a.sweep.iter().enumerate() {
                let Some(t) = a.threshold_of(i) else { continue };
                counts.add(sweep_counts(&[sw], SweepScope::Class(c), t));
                let labels = f.frame_classes.get(i).is_some_and(|fc| fc.contains(c));
                let has_box = sw.gt.iter().any(|(l, _)| l == c) || sw.ignored.contains(c);
                if labels && !has_box {
                    without += 1;
                    alerts += usize::from(
                        sw.preds
                            .iter()
                            .any(|p| p.class == *c && !p.tp && p.after < t && p.confidence >= t),
                    );
                }
            }
            ClassRow {
                class: c.clone(),
                objects: counts.tp + counts.fn_,
                precision: counts.precision(),
                recall: counts.recall(),
                f2: counts.f_beta(2.0),
                ap50: a.ap.class_ap50(c),
                frames_without: without,
                fa100: (without > 0).then(|| 100.0 * alerts as f64 / without as f64),
            }
        })
        .collect()
}

/// The per-camera breakdown (Your cameras: `camera:` tags).
fn camera_rows(a: &super::assess::Assessment, f: &Frames) -> Vec<CameraRow> {
    let camera = |i: usize| {
        f.tags
            .get(i)
            .and_then(|t| t.iter().find_map(|x| x.strip_prefix("camera:")))
    };
    let mut names: Vec<&str> = (0..f.images.len()).filter_map(camera).collect();
    names.sort_unstable();
    names.dedup();
    let mut rows: Vec<CameraRow> = names
        .iter()
        .map(|name| {
            let mine = |i: usize| camera(i) == Some(*name);
            let at = a.pooled_where(mine);
            let worst = SCENES
                .iter()
                .filter_map(|s| {
                    let sa = a.pooled_where(|i| {
                        mine(i)
                            && in_split(
                                f.tags.get(i).map_or(&[][..], Vec::as_slice),
                                f.scenes.get(i).copied().flatten(),
                                s,
                            )
                    });
                    Some((*s, sa.recall()?, sa.objects()))
                })
                .min_by(|x, y| x.1.total_cmp(&y.1).then(y.2.cmp(&x.2)))
                .map(|(s, _, _)| s.to_string());
            CameraRow {
                camera: (*name).to_string(),
                frames: (0..f.images.len()).filter(|&i| mine(i)).count(),
                objects: at.objects(),
                recall: at.recall(),
                fa100: at.fa100(),
                worst_scene: worst,
            }
        })
        .collect();
    // Worst first: the lowest recall, then the most false alerts.
    rows.sort_by(|x, y| {
        x.recall
            .unwrap_or(2.0)
            .total_cmp(&y.recall.unwrap_or(2.0))
            .then(y.fa100.unwrap_or(0.0).total_cmp(&x.fa100.unwrap_or(0.0)))
    });
    rows
}

/// The compare sets that have combos, and the one to show.
pub(crate) fn sets_with_combos<'a>(inputs: &Inputs<'a>) -> Vec<(CompareSet<'a>, ComboSet<'a>)> {
    compare_sets(inputs.results)
        .into_iter()
        .filter_map(|s| {
            let c = combos_on(&s, inputs)?;
            Some((s, c))
        })
        .collect()
}

/// The id of the default set among `sets` ([`default_set`]).
pub(crate) fn default_id(sets: &[(CompareSet<'_>, ComboSet<'_>)]) -> Option<String> {
    let candidates: Vec<_> = sets.iter().map(|(_, c)| c.candidate.clone()).collect();
    default_set(&candidates).map(str::to_string)
}

/// The Compare view (see the module docs).
pub fn build(inputs: &Inputs<'_>, req: Request<'_>) -> CompareView {
    let started = std::time::Instant::now();
    let goal = req.goal;
    let all = sets_with_combos(inputs);
    let options: Vec<SetOption> = all
        .iter()
        .map(|(_, c)| SetOption {
            id: c.fair.id.clone(),
            title: c.fair.title.clone(),
            frames: c.fair.images.len(),
            combos: c.combos.len(),
        })
        .collect();
    let wanted = req.set.map(str::to_string).or_else(|| default_id(&all));
    let mut view = CompareView {
        goal: GoalView::of(goal),
        sets: options,
        set: None,
        headline: String::new(),
        rows: Vec::new(),
        missing: Vec::new(),
        notes: Vec::new(),
        elapsed_ms: 0.0,
    };
    let Some((_, combo_set)) = all
        .iter()
        .find(|(_, c)| Some(c.fair.id.as_str()) == wanted.as_deref())
    else {
        view.headline = match &wanted {
            Some(id) if req.set.is_some() => {
                format!("No combo was scored on the set {id}: run a benchmark on it first.")
            }
            _ => "Nothing to compare yet: run a benchmark on labelled images first.".to_string(),
        };
        view.elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        return view;
    };
    let scene = req.scene.filter(|s| SCENES.contains(s));
    let fair = &combo_set.fair;
    let set = CompareSet {
        images: fair
            .images
            .iter()
            .filter(|i| scene.is_none_or(|s| in_split(&i.info.tags, i.info.scene, s)))
            .copied()
            .collect(),
        ..fair.clone()
    };
    // The classes every combo is scored on, limited to the goal's.
    let lists: Vec<Vec<String>> = combo_set
        .combos
        .iter()
        .map(|c| score::folded_classes(c.stored))
        .collect();
    let common = score::common_classes(&lists);
    let named: Vec<String> = goal
        .classes
        .iter()
        .map(|c| score::fold(&crate::benchmark::metrics::canonical_label(c)).to_string())
        .collect();
    let classes: Vec<String> = if named.is_empty() {
        common.clone()
    } else {
        common
            .iter()
            .filter(|c| named.contains(c))
            .cloned()
            .collect()
    };
    let mut ignored: Vec<String> = goal
        .classes
        .iter()
        .zip(&named)
        .filter(|(_, n)| !common.contains(n))
        .map(|(c, _)| c.clone())
        .collect();
    ignored.dedup();
    let map = ClassMap::for_model(&common);
    let default_key = inputs.default_model.map(normalize_name);
    let mut rows = Vec::new();
    let mut assessed = Vec::new();
    let mut first_frames: Option<(usize, usize, usize)> = None;
    for c in &combo_set.combos {
        let f = frames_of(&set, &map, c, &classes);
        let speed = speed_of(&set, c);
        let a = assess(Input {
            images: &f.images,
            frame_classes: &f.frame_classes,
            groups: &f.groups,
            classes: &classes,
            goal,
            p50_ms: speed.p50_ms,
            weights: req.weights,
            relative: combo_set.relative,
        });
        if first_frames.is_none() {
            first_frames = Some((a.objects, a.pooled.empty, a.groups.len()));
        }
        let now = inputs
            .now
            .get(&normalize_name(crate::benchmark::variant::base_name(
                &c.m.model,
            )));
        let too_slow = goal
            .speed_budget_ms
            .zip(speed.burst_p95_ms.or(speed.p95_ms))
            .is_some_and(|(b, p)| p > b);
        let tgt = target(a.objective).map(|t| Target {
            value: t,
            met: a.met() == Some(true),
            bound: bound(a.objective, &a.pooled),
        });
        let unmet = tgt.as_ref().is_some_and(|t| !t.met);
        let g = grid();
        let plateau = a
            .suggested
            .and_then(|p| Some([*g.get(p.first)?, *g.get(p.last)?]));
        let label =
            c.m.devices
                .iter()
                .find(|d| d.device == c.device)
                .map(|d| d.label.clone())
                .unwrap_or_default();
        let p = &a.pooled;
        let served_threshold = now.map(|n| n.threshold_for(c.device));
        rows.push(Row {
            model: c.m.model.clone(),
            device: c.device.to_string(),
            device_label: label,
            is_default: default_key.as_deref() == Some(normalize_name(&c.m.model).as_str()),
            metric: a.metric,
            position: 0,
            rank: 0,
            block: Block::Ok,
            leader: None,
            vs_leader: None,
            recall: p.recall(),
            precision: p.precision(),
            f1: p.f_beta(1.0),
            f2: p.f_beta(2.0),
            f05: p.f_beta(0.5),
            fa100: p.fa100(),
            tpr: p.tpr(),
            fpr: p.fpr(),
            ap50: a.ap50,
            ap50_95: a.ap50_95,
            roc_auc: a.roc_auc,
            grade: a.grade.map(|g| g.overall.letter().to_string()),
            grade_points: a.grade.map(|g| g.overall_points),
            confidence: a.suggested_threshold(),
            plateau,
            pick_informative: a.pick_informative(),
            target: tgt,
            served_threshold,
            at_saved: served_threshold.and_then(|t| at_point(&a, t)),
            at_suggested: a.suggested_threshold().and_then(|t| at_point(&a, t)),
            folds: if a.low_confidence.is_some() {
                0
            } else {
                a.cv.k
            },
            agreement: c
                .run
                .agreement
                .as_ref()
                .and_then(|x| serde_json::to_value(x.verdict).ok())
                .and_then(|v| v.as_str().map(str::to_string)),
            flags: Flags {
                too_slow,
                unmet,
                trained_here: c.excluded,
                trained_on_share: c.share,
                unknown_seen: c.unknown_seen.map(str::to_string),
                relative: combo_set.relative,
                served: crate::benchmark::variant::is_served(&c.m.model),
                stale_rules: c.m.stale_rules == Some(true),
                low_confidence: a.low_confidence.clone(),
                live_load: c.run.live_requests_per_min,
                video_decode: c.run.video_decode_pct.as_ref().map(|v| v.mean),
                flat: !a.pick_informative(),
            },
            scenes: scene_rows(&a, &f, goal),
            classes: class_rows(&a, &f, &classes),
            cameras: if set.id == crate::benchmark::cameras::ID {
                camera_rows(&a, &f)
            } else {
                Vec::new()
            },
            speed,
        });
        assessed.push(a);
    }
    let rankable: Vec<Rankable<'_>> = rows
        .iter()
        .zip(&assessed)
        .map(|(r, a)| Rankable {
            metric: r.metric.map(|m| m.value),
            resampled: &a.resampled,
            too_slow: r.flags.too_slow,
            unmet: r.flags.unmet,
            unfair: r.flags.trained_here,
            p95_ms: r.speed.p95_ms,
        })
        .collect();
    let (order, places) = rank(&rankable);
    let name = |i: usize| {
        rows.get(i)
            .map(|r| format!("{} on {}", r.model, r.device))
            .unwrap_or_default()
    };
    let leaders: Vec<Option<String>> = places
        .iter()
        .enumerate()
        .map(|(i, p): (usize, &Place)| (p.leader != i).then(|| name(p.leader)))
        .collect();
    for ((r, p), l) in rows.iter_mut().zip(&places).zip(leaders) {
        r.position = p.position;
        r.rank = p.rank;
        r.block = p.block;
        r.vs_leader = p.vs_leader;
        r.leader = l;
    }
    let mut by_index: Vec<Option<Row>> = rows.into_iter().map(Some).collect();
    view.rows = order
        .iter()
        .filter_map(|&i| by_index.get_mut(i).and_then(Option::take))
        .collect();
    view.headline = headline(&view.rows, goal);
    let (objects, empty, groups) = first_frames.unwrap_or_default();
    if !classes.is_empty() && view.rows.iter().all(|r| r.metric.is_none()) {
        view.notes.push(format!(
            "No combo has a {} on these frames (no labelled object of the counted classes?).",
            goal.optimize_for.short()
        ));
    }
    if goal.optimize_for.rank_only() {
        view.notes.push(format!(
            "{} needs no confidence; each row's confidence is picked for {}.",
            goal.optimize_for.short(),
            objective_short(goal.tune_for)
        ));
    }
    view.notes.push(
        "Figures are cross-validated over camera days (single images on public sets): the \
         confidence is picked on the other folds and counted on each fold; \u{b1} is the 95% \
         interval of a cluster bootstrap. Rows sharing a rank are not clearly different."
            .to_string(),
    );
    if view.rows.iter().any(|r| !r.pick_informative) {
        view.notes.push(FLAT_NOTE.to_string());
    }
    view.set = Some(SetInfo {
        id: set.id.clone(),
        title: set.title.clone(),
        what: set.role.what().to_string(),
        unit: set.role.unit().to_string(),
        frames: set.images.len(),
        frames_left_out: combo_set.frames_left_out,
        left_out_by: combo_set.left_out_by.clone(),
        scene: scene.map(str::to_string),
        scenes: SCENES
            .iter()
            .map(|s| SceneCount {
                scene: (*s).to_string(),
                frames: fair
                    .images
                    .iter()
                    .filter(|i| in_split(&i.info.tags, i.info.scene, s))
                    .count(),
            })
            .filter(|c| c.frames > 0)
            .collect(),
        objects,
        empty_frames: empty,
        relative: combo_set.relative,
        reference: combo_set.reference.clone(),
        classes,
        scored_classes: common,
        ignored_classes: ignored,
        groups,
        run: RunHint {
            datasets: fair.datasets.iter().map(|d| d.id.clone()).collect(),
            max_images: fair
                .datasets
                .iter()
                .map(|d| d.images.len())
                .max()
                .unwrap_or(0),
        },
    });
    view.missing = combo_set.missing.clone();
    view.elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
    view
}

/// One sentence: the best combo, those not clearly different, or why nothing meets the goal.
pub fn headline(rows: &[Row], goal: &Goal) -> String {
    let metric = goal.optimize_for.short();
    let within = if goal.speed_budget_ms.is_some() {
        " within budget"
    } else {
        ""
    };
    let Some(best) = rows.first().filter(|r| r.block == Block::Ok) else {
        if rows.is_empty() {
            return "No combo is scored on these frames.".to_string();
        }
        let slow = rows.iter().filter(|r| r.flags.too_slow).count();
        let unmet = rows.iter().filter(|r| r.flags.unmet).count();
        let mut why = Vec::new();
        if unmet > 0 {
            why.push(format!("{unmet} miss the target on its 95% bound"));
        }
        if slow > 0 {
            why.push(format!("{slow} are over the speed budget"));
        }
        if why.is_empty() {
            why.push("none can be judged fairly on these frames".to_string());
        }
        return format!(
            "Nothing meets the goal ({}): {}.",
            goal.describe(),
            why.join(", ")
        );
    };
    let ties: Vec<String> = rows
        .iter()
        .skip(1)
        .filter(|r| r.block == Block::Ok && r.rank == best.rank)
        .map(|r| format!("{} on {}", r.model, r.device))
        .collect();
    let mut s = format!(
        "{} on {} has the best {metric}{within}",
        best.model, best.device
    );
    match ties.len() {
        0 if rows.iter().filter(|r| r.block == Block::Ok).count() > 1 => {
            s.push_str("; every other combo is clearly worse")
        }
        0 => {}
        1 => s.push_str(&format!("; {} is not clearly different", ties.join(""))),
        _ => s.push_str(&format!("; {} are not clearly different", ties.join(", "))),
    }
    if best.flags.low_confidence.is_some() {
        s.push_str(" (low confidence: little data)");
    }
    s.push('.');
    s
}
