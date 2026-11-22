//! One combo (a model on a device) on one set of frames, judged for the goal: the
//! cross-validated pick ([`super::cv`]), the reported (out-of-fold) figures, the goal metric with
//! its bootstrap interval, the rank-only metrics and the low-confidence and flat-curve flags.
//!
//! - **Reported figures**: each image counted at its fold's pick, pooled ([`Assessment::pooled`]).
//!   With too little data ([`MIN_GROUPS`] groups or [`MIN_OBJECTS`] objects of the counted
//!   classes) there is no cross-validated claim: the figures are the in-sample ones at the pick
//!   on every frame, flagged [`Assessment::low_confidence`] with the reason (the interval is
//!   still computed).
//! - **Goal metric**: a threshold objective's value ([`super::pick::value`]) of the reported
//!   counts; a rank-only goal's AP50 / AP50-95 / frame ROC AUC over every frame (no threshold),
//!   or the overall grade points (accuracy score from AP50 and the small-object recall at the
//!   reported picks, speed from the set's p50, `benchmark.weights`).
//! - **Interval**: a cluster bootstrap over the groups ([`super::stats::resamples`]):
//!   [`THRESHOLD_RESAMPLES`] resamples of the groups' reported counts for a threshold objective
//!   (the picks stay as they were: the counts are summed and the metric recomputed),
//!   [`RANK_RESAMPLES`] for the rank-only metrics (recomputed from the resampled images' match
//!   records, [`super::rankfree`]). [`Assessment::resampled`] keeps every resample's value: two
//!   combos on the same frames have the same groups in the same order and so the same
//!   resamples, and their paired difference is taken resample by resample
//!   ([`super::stats::Difference`]).
//! - **Flat curve**: every grid value scores within the plateau of the best
//!   ([`super::pick::GridPick::flat`], e.g. every scored frame is answered by the night model or
//!   held at a fixed night threshold): [`Assessment::pick_informative`] is false.

use super::cv::{Cv, Groups, cross_validate};
use super::goal::{Goal, Optimize};
use super::pick::{GridPick, value};
use super::rankfree::{ApData, AucData};
use super::stats::{Estimate, resamples};
use super::tally::{At, ImageTally, tallies};
use crate::benchmark::cameras::{MIN_CAMERA_DAYS, MIN_OBJECTS};
use crate::benchmark::grade::{AccuracyMetric, Grades, Weights, accuracy_score};
use crate::benchmark::metrics::{EvalImage, SweepImage, SweepScope, sweep_counts};
use crate::benchmark::threshold::{Objective, grid};

/// Bootstrap resamples of a threshold objective's interval.
pub const THRESHOLD_RESAMPLES: usize = 1000;
/// Bootstrap resamples of AP50, AP50-95, ROC AUC and the grade (each recomputes the metric from
/// the match records).
pub const RANK_RESAMPLES: usize = 200;
/// Fewer groups than this: low confidence (the plan's rule: camera days on Your cameras).
pub const MIN_GROUPS: usize = MIN_CAMERA_DAYS;

/// Why a pick on a flat curve says nothing.
pub const FLAT_NOTE: &str = "Every confidence from 0.05 to 1.00 scores within 0.002 of the best \
     on these frames (for example every scored frame is answered by the night model or held at a \
     fixed night threshold): the pick says nothing about the confidence to use.";

/// What [`assess`] reads.
#[derive(Debug, Clone, Copy)]
pub struct Input<'a> {
    /// The frames, counted classes only (labels and predictions in one label space).
    pub images: &'a [EvalImage],
    /// Per image: the counted classes its set labels.
    pub frame_classes: &'a [Vec<String>],
    /// Per image: its cross-validation group.
    pub groups: &'a [String],
    /// The counted classes.
    pub classes: &'a [String],
    pub goal: &'a Goal,
    /// One-at-a-time p50 on these frames (the grade's speed).
    pub p50_ms: Option<f64>,
    pub weights: Weights,
    /// Labels are another model's detections.
    pub relative: bool,
}

/// One combo judged (see the module docs).
#[derive(Debug, Clone)]
pub struct Assessment {
    /// The objective the confidence is picked for ([`Goal::tuning`]).
    pub objective: Objective,
    pub optimize: Optimize,
    pub sweep: Vec<SweepImage>,
    pub tallies: Vec<ImageTally>,
    pub groups: Groups,
    pub cv: Cv,
    /// Why the figures are in-sample (too little data); None = cross-validated.
    pub low_confidence: Option<String>,
    /// Per image: the grid index its reported counts are at.
    pub image_index: Vec<Option<usize>>,
    /// Per group: its reported counts.
    pub reported: Vec<At>,
    pub pooled: At,
    /// The suggested pick: on every frame.
    pub suggested: Option<GridPick>,
    /// The goal metric and its 95% interval.
    pub metric: Option<Estimate>,
    /// The goal metric of every bootstrap resample (aligned across combos on the same frames).
    pub resampled: Vec<Option<f64>>,
    pub ap: ApData,
    pub ap50: Option<f64>,
    pub ap50_95: Option<f64>,
    pub roc_auc: Option<f64>,
    /// Recall of small objects at the reported picks.
    pub small_recall: Option<f64>,
    /// The grade of these frames.
    pub grade: Option<Grades>,
    /// Labelled objects of the counted classes.
    pub objects: usize,
}

impl Assessment {
    /// The suggested pick says something about the confidence (the curve is not flat).
    pub fn pick_informative(&self) -> bool {
        !self.suggested.is_some_and(|p| p.flat)
    }

    /// The suggested confidence.
    pub fn suggested_threshold(&self) -> Option<f32> {
        let g = grid();
        self.suggested.and_then(|p| g.get(p.index).copied())
    }

    /// Whether the objective's target holds on the reported counts' bound (None without a
    /// target).
    pub fn met(&self) -> Option<bool> {
        super::pick::met(self.objective, &self.pooled)
    }

    /// The reported counts of the images `keep(i)` selects (scene, camera breakdowns).
    pub fn pooled_where(&self, keep: impl Fn(usize) -> bool) -> At {
        let mut a = At::default();
        for (i, img) in self.tallies.iter().enumerate() {
            if keep(i)
                && let Some(Some(k)) = self.image_index.get(i)
            {
                a.add_image(img, *k, 1);
            }
        }
        a
    }

    /// The grid threshold image `i` is counted at.
    pub fn threshold_of(&self, i: usize) -> Option<f32> {
        let k = (*self.image_index.get(i)?)?;
        grid().get(k).copied()
    }
}

/// Small objects (tp, objects) of `img` at `t`.
fn small_at(img: &SweepImage, t: f32) -> (usize, usize) {
    let c = sweep_counts(&[img], SweepScope::Small, t);
    (c.tp, c.tp + c.fn_)
}

/// Judge one combo (see the module docs).
pub fn assess(input: Input<'_>) -> Assessment {
    let g = grid();
    let n = g.len();
    let goal = input.goal;
    let objective = goal.tuning();
    let sweep: Vec<SweepImage> = input
        .images
        .iter()
        .zip(input.frame_classes)
        .map(|(e, fc)| SweepImage::of(e).with_frames(e, fc))
        .collect();
    let tallies = tallies(&sweep);
    let groups = Groups::new(input.groups.iter().map(String::as_str));
    let cv = cross_validate(objective, &tallies, &groups, n);
    let objects: usize = tallies.iter().map(|t| t.objects as usize).sum();
    let low_confidence = low_confidence(groups.len(), objects);
    let all_index = cv.all.map(|p| p.index);
    let image_index: Vec<Option<usize>> = if low_confidence.is_some() || !cv.validated() {
        vec![all_index; tallies.len()]
    } else {
        cv.image_index.clone()
    };
    let mut reported = vec![At::default(); groups.len()];
    let mut small = vec![(0usize, 0usize); groups.len()];
    for (i, (img, &grp)) in tallies.iter().zip(&groups.of_image).enumerate() {
        let Some(Some(k)) = image_index.get(i) else {
            continue;
        };
        if let Some(r) = reported.get_mut(grp) {
            r.add_image(img, *k, 1);
        }
        if let (Some(s), Some(sw), Some(&t)) = (small.get_mut(grp), sweep.get(i), g.get(*k)) {
            let (tp, all) = small_at(sw, t);
            s.0 += tp;
            s.1 += all;
        }
    }
    let mut pooled = At::default();
    for r in &reported {
        pooled.add(r);
    }
    let ap = ApData::new(input.images, &groups, input.classes, true);
    let auc = AucData::new(&sweep, &groups);
    let small_of = |w: Option<&[u32]>| {
        let (mut tp, mut all) = (0usize, 0usize);
        for (gi, &(t, a)) in small.iter().enumerate() {
            let k = w.map_or(1, |w| w.get(gi).copied().unwrap_or(0) as usize);
            tp += k * t;
            all += k * a;
        }
        (all > 0).then(|| tp as f64 / all as f64)
    };
    let grade_of = |ap50: Option<f64>, small: Option<f64>| {
        let score = ap50.map(|a| accuracy_score(a, small));
        input.p50_ms.map(|p50| {
            Grades::with_metric(
                AccuracyMetric::Ap50,
                score,
                p50,
                input.weights,
                input.relative,
            )
        })
    };
    let ap50 = ap.ap50(None);
    let ap50_95 = ap.ap50_95(None);
    let roc_auc = auc.auc(None);
    let small_recall = small_of(None);
    let grade = grade_of(ap50, small_recall);
    let (point, resampled): (Option<f64>, Vec<Option<f64>>) = match goal.optimize_for {
        Optimize::Threshold(o) => (
            value(o, &pooled),
            resamples(groups.len(), THRESHOLD_RESAMPLES)
                .map(|w| {
                    let mut a = At::default();
                    for (r, &k) in reported.iter().zip(&w) {
                        a.add_scaled(r, k as usize);
                    }
                    value(o, &a)
                })
                .collect(),
        ),
        Optimize::Ap50 => (
            ap50,
            resamples(groups.len(), RANK_RESAMPLES)
                .map(|w| ap.ap50(Some(&w)))
                .collect(),
        ),
        Optimize::Ap5095 => (
            ap50_95,
            resamples(groups.len(), RANK_RESAMPLES)
                .map(|w| ap.ap50_95(Some(&w)))
                .collect(),
        ),
        Optimize::RocAuc => (
            roc_auc,
            resamples(groups.len(), RANK_RESAMPLES)
                .map(|w| auc.auc(Some(&w)))
                .collect(),
        ),
        Optimize::Grade => (
            grade.map(|g| g.overall_points),
            resamples(groups.len(), RANK_RESAMPLES)
                .map(|w| grade_of(ap.ap50(Some(&w)), small_of(Some(&w))).map(|g| g.overall_points))
                .collect(),
        ),
    };
    Assessment {
        objective,
        optimize: goal.optimize_for,
        suggested: cv.all,
        sweep,
        tallies,
        groups,
        cv,
        low_confidence,
        image_index,
        reported,
        pooled,
        metric: point.map(|v| Estimate::of(v, &resampled)),
        resampled,
        ap,
        ap50,
        ap50_95,
        roc_auc,
        small_recall,
        grade,
        objects,
    }
}

/// Why `groups` groups with `objects` objects are too little to cross-validate (None: enough).
pub fn low_confidence(groups: usize, objects: usize) -> Option<String> {
    let mut why = Vec::new();
    if groups < MIN_GROUPS {
        why.push(format!(
            "{groups} group(s) of frames (camera days on Your cameras, single images elsewhere); \
             {MIN_GROUPS} are needed to cross-validate"
        ));
    }
    if objects < MIN_OBJECTS {
        why.push(format!(
            "{objects} labelled object(s) of the counted classes; {MIN_OBJECTS} are needed"
        ));
    }
    (!why.is_empty()).then(|| {
        format!(
            "low confidence: {}. The figures are in-sample, at the pick on every frame.",
            why.join("; ")
        )
    })
}
