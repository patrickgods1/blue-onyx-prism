//! Confidence-threshold sweep: the best confidence threshold of a model for an objective.
//!
//! Every device run stores, next to its AP, the precision / recall / F1 / F2 / false positives
//! per image curve over [`SWEEP_THRESHOLDS`] ([`ThresholdSweep`]: overall, per dataset, per
//! tag group and per class), computed from the predictions the run already made at the
//! evaluation threshold (no extra inference; exact, see [`super::metrics`]). [`advise`] picks
//! the best threshold of a model from its recommended device's curves for the configured
//! [`Objective`] (config `benchmark.threshold_objective`):
//!
//! - `f1` (default): the highest F1, precision and recall balanced;
//! - `f2`: the highest F2, recall weighted 4x (fewer missed people, more false alerts);
//! - `f05`: the highest F0.5, precision weighted 4x (fewer false alerts, more missed objects);
//! - `precision:<p>`: the highest recall with precision >= p (e.g. `precision:0.9` to limit false
//!   alerts). When no threshold reaches p, the most precise threshold is picked, with a note;
//! - `recall:<r>`: the highest precision with recall >= r (e.g. `recall:0.9`: miss at most 10%).
//!   When no threshold reaches r, the threshold with the highest recall is picked, with a note;
//! - `recall_at_fa:<N>`: the highest box recall with at most N false alerts per 100 empty frames
//!   (N in (0, 100]; empty frames and false alerts as [`super::metrics::AlertStats`] counts
//!   them: frames with no object of a scored class their dataset annotates, alerting when a
//!   detection of such a class outside every ignore region is at or above the threshold). When
//!   no threshold reaches N, the threshold with the fewest false alerts is picked, with a note;
//! - `youden`: the highest frame-level Youden's J, TPR - FPR (alert rate on frames with an
//!   object minus false-alert rate on frames without);
//! - `fpr:<x>`: the highest frame TPR with frame FPR <= x (e.g. `fpr:0.05`: at most 5% of the
//!   frames without an object alert).
//!
//! The frame objectives use one sample per frame ([`super::metrics::SweepImage::alert_sample`],
//! [`ThresholdPoint::per_frame`]): positive when the frame has a non-ignored object of a scored
//! class its dataset annotates, negative when it has none (frames whose only such boxes are
//! ignore regions are left out); its score is the highest confidence of its scored-class
//! detections outside every ignore region (0 without one). The pooled (frame, class) samples of
//! the frame ROC AUC ([`super::roc`]) would let the rare classes' negatives (every frame without
//! a dog) dominate the FPR and push the pick low.
//!
//! The pick is evaluated exactly on the 0.01 grid ([`grid`], [`exact_pick`]): every value t
//! from [`MIN_THRESHOLD`] (the evaluation pass's floor) to [`MAX_THRESHOLD`] is scored with the
//! detections of confidence >= t ([`Breakpoints::point_at`]). The grid values scoring within
//! [`PLATEAU`] of the best grid value form runs of consecutive values; the widest run wins
//! (equally wide runs: the higher one) and its middle value is the pick (the lower middle for
//! an even count): a lone spike between two drops is not chosen over a broad flat top, and the
//! pick is always a configurable value within [`PLATEAU`] of the best one (B3: rounding the
//! midpoint of a range narrower than 0.01 could land outside it). Target objectives
//! (`precision:`, `recall:`, `recall_at_fa:`, `fpr:`) count only the grid values meeting the
//! target on these point estimates; when none does, the fallback above applies (`met: false`,
//! never applied). The 0.05 grid ([`SWEEP_THRESHOLDS`]) is only used for the curve tables and
//! charts.
//!
//! A pick fitted on pseudo ground truth (another model's detections, [`Pick::relative`]) is
//! never written into the config ([`ThresholdAdvice::apply_value`], [`NOT_ON_LABELS`]).
//!
//! The server's threshold is what the model reports; Blue Iris can send its own
//! `min_confidence` per request (it then replaces the server threshold for that request) and
//! filters the returned objects again with each camera's minimum confidence.

use super::metrics::{Breakpoints, ThresholdPoint};
use super::report::DeviceResult;
use super::roc::{FrameRates, FrameRoc};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Scores this close count as equal (the higher threshold wins).
const TIE: f64 = 1e-12;
/// Grid values scoring within this of the best form the plateau whose middle is picked.
pub const PLATEAU: f64 = 0.002;
/// Lowest threshold a pick can have: the floor of the evaluation pass.
pub const MIN_THRESHOLD: f32 = super::EVAL_CONFIDENCE;
/// Highest threshold a pick can have (above every prediction's confidence nothing is kept).
pub const MAX_THRESHOLD: f32 = 1.0;
/// Why a pick fitted on pseudo ground truth is not applied (apply endpoints, CLI).
pub const NOT_ON_LABELS: &str = "fitted on another model's detections, not on labels";

/// The thresholds a pick is chosen from: every 0.01 from [`MIN_THRESHOLD`] to
/// [`MAX_THRESHOLD`], ascending (`k / 100` in f32, so 0.35 is the literal `0.35`).
pub fn grid() -> Vec<f32> {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "both bounds are thresholds in 0..=1, so the cents are in 0..=100"
    )]
    let cents = |t: f32| (f64::from(t) * 100.0).round() as u32;
    (cents(MIN_THRESHOLD)..=cents(MAX_THRESHOLD))
        .map(|k| k as f32 / 100.0)
        .collect()
}

/// What the best threshold optimizes.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Objective {
    /// Highest F1.
    #[default]
    F1,
    /// Highest F2 (recall-weighted).
    F2,
    /// Highest F0.5 (precision-weighted: precision counts twice as much as recall in beta).
    F05,
    /// Highest recall with precision at or above the target (0..1].
    Precision(f64),
    /// Highest precision with recall at or above the target (0..1].
    Recall(f64),
    /// Highest frame-level Youden's J (TPR - FPR over frames, classes pooled).
    Youden,
    /// Highest frame TPR with frame FPR at or below the target [0..1).
    Fpr(f64),
    /// Highest box recall with at most this many false alerts per 100 empty frames (0..100].
    RecallAtFa(f64),
}

impl Objective {
    /// The objectives every report shows for comparison.
    pub const STANDARD: [Objective; 4] = [
        Objective::F1,
        Objective::F2,
        Objective::Precision(0.9),
        Objective::Youden,
    ];

    /// Scored on frame-level rates ([`super::roc`]) rather than box counts.
    pub fn is_frame(&self) -> bool {
        matches!(self, Objective::Youden | Objective::Fpr(_))
    }

    /// Scored on false alerts per 100 empty frames ([`ThresholdPoint::false_alerts`]).
    pub fn is_alerts(&self) -> bool {
        matches!(self, Objective::RecallAtFa(_))
    }

    /// The value maximized at `p`: F1, F2, F0.5, or recall when the precision target is met.
    pub fn score(&self, p: &ThresholdPoint) -> Option<f64> {
        match self {
            Objective::F1 => p.f1,
            Objective::F2 => p.f2,
            Objective::F05 => p.counts.f_beta(0.5),
            Objective::Precision(target) => {
                p.precision.filter(|&v| v >= target - TIE).and(p.recall)
            }
            Objective::Recall(target) => p.recall.filter(|&v| v >= target - TIE).and(p.precision),
            Objective::Youden => p.alert_rates().and_then(FrameRates::youden),
            Objective::Fpr(target) => {
                let f = p.alert_rates()?;
                f.fpr.filter(|&v| v <= target + TIE).and(f.tpr)
            }
            Objective::RecallAtFa(n) => {
                let fa = p.false_alerts?.per_100()?;
                p.recall.filter(|_| fa <= n + TIE)
            }
        }
    }

    /// Whether `p` meets the objective's target (precision, recall, frame FPR, false alerts
    /// per 100 empty frames), None without one.
    fn target_met(&self, p: &ThresholdPoint) -> Option<bool> {
        match self {
            Objective::Precision(t) => Some(p.precision.is_some_and(|v| v >= t - TIE)),
            Objective::Recall(t) => Some(p.recall.is_some_and(|v| v >= t - TIE)),
            Objective::Fpr(t) => Some(
                p.alert_rates()
                    .and_then(|f| f.fpr)
                    .is_some_and(|v| v <= t + TIE),
            ),
            Objective::RecallAtFa(n) => Some(
                p.false_alerts
                    .and_then(|f| f.per_100())
                    .is_some_and(|v| v <= n + TIE),
            ),
            _ => None,
        }
    }

    /// Human-readable: "F1", "F2 (favor recall)", "recall at precision >= 90%".
    pub fn describe(&self) -> String {
        match self {
            Objective::F1 => "F1".into(),
            Objective::F2 => "F2 (favor recall)".into(),
            Objective::F05 => "F0.5 (favor precision)".into(),
            Objective::Precision(p) => format!("recall at precision \u{2265} {}%", pct0(*p)),
            Objective::Recall(r) => format!("precision at recall \u{2265} {}%", pct0(*r)),
            Objective::Youden => "Youden's J (frame TPR \u{2212} FPR)".into(),
            Objective::Fpr(x) => format!("frame TPR at frame FPR \u{2264} {}%", pct0(*x)),
            Objective::RecallAtFa(n) => format!(
                "recall at \u{2264} {} false alerts per 100 empty frames",
                num0(*n)
            ),
        }
    }
}

/// `5`, `2.5`: a count per 100 frames without a trailing `.0`.
fn num0(v: f64) -> String {
    let s = format!("{v:.1}");
    s.strip_suffix(".0").map(str::to_string).unwrap_or(s)
}

fn pct0(v: f64) -> String {
    let s = format!("{:.1}", v * 100.0);
    s.strip_suffix(".0").map(str::to_string).unwrap_or(s)
}

impl fmt::Display for Objective {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Objective::F1 => f.write_str("f1"),
            Objective::F2 => f.write_str("f2"),
            Objective::F05 => f.write_str("f05"),
            Objective::Precision(p) => write!(f, "precision:{p}"),
            Objective::Recall(r) => write!(f, "recall:{r}"),
            Objective::Youden => f.write_str("youden"),
            Objective::Fpr(x) => write!(f, "fpr:{x}"),
            Objective::RecallAtFa(n) => write!(f, "recall_at_fa:{n}"),
        }
    }
}

impl FromStr for Objective {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let t = s.trim().to_ascii_lowercase();
        match t.as_str() {
            "f1" | "" => Ok(Objective::F1),
            "f2" => Ok(Objective::F2),
            "f05" | "f0.5" => Ok(Objective::F05),
            "youden" | "youden_j" | "j" => Ok(Objective::Youden),
            _ => {
                if let Some(x) = t
                    .strip_prefix("recall_at_fa:")
                    .or_else(|| t.strip_prefix("recall_at_fa="))
                {
                    let v: f64 = x.trim().parse().map_err(|_| {
                        format!(
                            "threshold objective '{s}': '{x}' is not a number (e.g. recall_at_fa:5)"
                        )
                    })?;
                    if !(v > 0.0 && v <= 100.0) {
                        return Err(format!(
                            "threshold objective '{s}': the false alerts per 100 empty frames \
                             must be in (0, 100]"
                        ));
                    }
                    return Ok(Objective::RecallAtFa(v));
                }
                if let Some(x) = t.strip_prefix("fpr:").or_else(|| t.strip_prefix("fpr=")) {
                    let v: f64 = x.trim().parse().map_err(|_| {
                        format!("threshold objective '{s}': '{x}' is not a number (e.g. fpr:0.05)")
                    })?;
                    if !(0.0..1.0).contains(&v) {
                        return Err(format!(
                            "threshold objective '{s}': the fpr target must be in [0, 1)"
                        ));
                    }
                    return Ok(Objective::Fpr(v));
                }
                let (kind, p) = if let Some(p) = t
                    .strip_prefix("precision:")
                    .or_else(|| t.strip_prefix("precision="))
                {
                    ("precision", p)
                } else if let Some(p) = t
                    .strip_prefix("recall:")
                    .or_else(|| t.strip_prefix("recall="))
                {
                    ("recall", p)
                } else {
                    return Err(format!(
                        "threshold objective '{s}': use f1, f2, f05, precision:<0..1>, \
                         recall:<0..1>, recall_at_fa:<N per 100 empty frames>, youden or \
                         fpr:<0..1>"
                    ));
                };
                let v: f64 = p.trim().parse().map_err(|_| {
                    format!("threshold objective '{s}': '{p}' is not a number (e.g. {kind}:0.9)")
                })?;
                if !(v > 0.0 && v <= 1.0) {
                    return Err(format!(
                        "threshold objective '{s}': the {kind} target must be in (0, 1]"
                    ));
                }
                Ok(if kind == "precision" {
                    Objective::Precision(v)
                } else {
                    Objective::Recall(v)
                })
            }
        }
    }
}

impl Objective {
    /// Why a target no finite set can show is refused where a user sets one (a goal, a flag, a
    /// form): precision or recall 1, or a false positive rate of 0, is met only when the 95%
    /// Wilson bound reaches it, which it never does on a finite count. Stored objectives (older
    /// results, older configs) still parse ([`FromStr`] accepts them).
    pub fn reachable(&self) -> Result<(), String> {
        let unreachable = match *self {
            Objective::Precision(v) | Objective::Recall(v) => v >= 1.0,
            Objective::Fpr(v) => v <= 0.0,
            _ => false,
        };
        if unreachable {
            return Err(format!(
                "threshold objective '{self}': a 100% target can never be shown on a finite set \
                 (its 95% bound never reaches it); use 0.99 or lower (for fpr, 0.01 or higher)"
            ));
        }
        Ok(())
    }

    /// The nearest objective release 0.2.0 parses (it knows `f1`, `f2`, `precision:<p>`,
    /// `recall:<r>`, `youden` and `fpr:<x>`): `f05` and `recall_at_fa:<N>` become `f1`. What the
    /// config's legacy `threshold_objective` is written as, so an older binary still loads the
    /// file (the goal, which it ignores, stays authoritative for this one).
    pub fn for_previous_release(self) -> Self {
        match self {
            Objective::F05 | Objective::RecallAtFa(_) => Objective::F1,
            o => o,
        }
    }

    /// `s` parsed and [`Self::reachable`]: an objective a user sets.
    pub fn parse_target(s: &str) -> Result<Self, String> {
        let o: Self = s.parse()?;
        o.reachable()?;
        Ok(o)
    }
}

impl TryFrom<String> for Objective {
    type Error = String;
    fn try_from(s: String) -> Result<Self, String> {
        s.parse()
    }
}

impl From<Objective> for String {
    fn from(o: Objective) -> String {
        o.to_string()
    }
}

/// The best threshold of a curve for an objective.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pick {
    /// Objective (`f1`, `f2`, `precision:0.9`).
    pub objective: String,
    pub threshold: f32,
    pub point: ThresholdPoint,
    /// False when a target is reached at no threshold (the closest one was picked instead).
    pub met: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// The best value of the 0.01 grid ([`grid`]; ties: the highest), before the plateau rule.
    /// Results of versions before the grid hold the exact optimum over every distinct
    /// confidence; None for a pick from the 0.05 grid only (results of older versions).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_threshold: Option<f32>,
    /// Objective score at `exact_threshold`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact_score: Option<f64>,
    /// The run of grid values `[first, last]` within [`PLATEAU`] of the best score whose middle
    /// is the pick (`[t, t]` for a fallback).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plateau: Option<[f32; 2]>,
    /// Fitted on pseudo ground truth (another model's detections): shown, never applied
    /// ([`NOT_ON_LABELS`]). Set from the curve it was fitted on ([`Curve::best`]).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub relative: bool,
}

/// Round down to a multiple of 0.01 (a hair of tolerance so 0.35 stays 0.35 in f32): the tests'
/// check that every grid value is a whole cent.
#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    reason = "a threshold in 0..=1: the f64 -> f32 narrowing only drops precision"
)]
pub fn floor_cent(t: f64) -> f32 {
    ((t * 100.0 + 1e-4).floor() / 100.0) as f32
}

/// The run of grid values a pick is taken from: indices into the grid scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridRun {
    /// The best-scoring value (ties: the highest).
    pub best: usize,
    /// First and last value of the widest run within the tolerance of the best (ties: the
    /// higher run).
    pub first: usize,
    pub last: usize,
}

impl GridRun {
    /// The picked value: the middle of the run, the lower one for an even count.
    pub fn middle(&self) -> usize {
        self.first + (self.last - self.first) / 2
    }
}

/// The plateau rule over `scores` (one per grid value, ascending thresholds; None = undefined
/// or the target missed): the best value, then the widest run of consecutive values scoring at
/// least `best - tol` (ties: the higher run). None when no value has a score.
pub fn grid_run(scores: &[Option<f64>], tol: f64) -> Option<GridRun> {
    let mut best: Option<(usize, f64)> = None;
    for (i, v) in scores.iter().enumerate() {
        if let Some(v) = *v
            && best.is_none_or(|(_, b)| v >= b - TIE)
        {
            best = Some((i, v));
        }
    }
    let (best, top) = best?;
    let ok = |i: usize| {
        scores
            .get(i)
            .copied()
            .flatten()
            .is_some_and(|v| v >= top - tol - TIE)
    };
    let mut out: Option<(usize, usize)> = None;
    let mut i = 0;
    while i < scores.len() {
        if !ok(i) {
            i += 1;
            continue;
        }
        let first = i;
        while ok(i + 1) {
            i += 1;
        }
        if out.is_none_or(|(a, b)| i - first >= b - a) {
            out = Some((first, i));
        }
        i += 1;
    }
    let (first, last) = out?;
    Some(GridRun { best, first, last })
}

/// The best threshold of `bp` for `objective`, every value of the 0.01 grid scored exactly (see
/// the module docs): None without ground truth (frame objectives: without frames with and
/// without objects; `recall_at_fa`: without empty frames).
pub fn exact_pick(bp: &Breakpoints, objective: Objective) -> Option<Pick> {
    if bp.npos == 0 && !objective.is_frame() {
        return None;
    }
    if objective.is_frame()
        && !bp
            .per_frame
            .as_ref()
            .is_some_and(|f| f.positives > 0 && f.negatives > 0)
    {
        return None;
    }
    if objective.is_alerts() && !bp.empty.as_ref().is_some_and(|e| e.negatives > 0) {
        return None;
    }
    let thresholds = grid();
    let points: Vec<ThresholdPoint> = thresholds.iter().map(|&t| bp.point_at(t)).collect();
    let scores: Vec<Option<f64>> = points.iter().map(|p| objective.score(p)).collect();
    let at = |i: usize| thresholds.get(i).copied().unwrap_or(MIN_THRESHOLD);
    if let Some(run) = grid_run(&scores, PLATEAU) {
        let k = run.middle();
        let p = points.get(k).copied()?;
        let note = (p.counts.tp == 0).then(|| "no threshold finds any object".to_string());
        // Every value of the run meets the target (a missed target scores None).
        let met = objective.target_met(&p) != Some(false);
        return Some(Pick {
            objective: objective.to_string(),
            threshold: p.threshold,
            point: p,
            met,
            note,
            exact_threshold: Some(at(run.best)),
            exact_score: scores.get(run.best).copied().flatten(),
            plateau: Some([at(run.first), at(run.last)]),
            relative: false,
        });
    }
    // Target not reached: the best grid value on the constrained metric (then the other
    // metric, then the higher threshold; the grid is ascending).
    let p = *fallback_point(&points, objective)?;
    Some(Pick {
        objective: objective.to_string(),
        threshold: p.threshold,
        point: p,
        met: false,
        note: Some(unreached_note(objective, &p)),
        exact_threshold: Some(p.threshold),
        exact_score: None,
        plateau: Some([p.threshold, p.threshold]),
        relative: false,
    })
}

/// The best point of `points` for `objective`; ties go to the higher threshold. None without
/// ground truth (or, for a precision target, without any detection).
pub fn pick(points: &[ThresholdPoint], objective: Objective) -> Option<Pick> {
    let mut sorted: Vec<&ThresholdPoint> = points.iter().filter(|p| p.recall.is_some()).collect();
    sorted.sort_by(|a, b| a.threshold.total_cmp(&b.threshold));
    let best_by = |score: &dyn Fn(&ThresholdPoint) -> Option<f64>| {
        let mut best: Option<(&ThresholdPoint, f64)> = None;
        for p in &sorted {
            if let Some(s) = score(p)
                && best.is_none_or(|(_, b)| s >= b - TIE)
            {
                best = Some((p, s));
            }
        }
        best.map(|(p, _)| *p)
    };
    let make = |p: &ThresholdPoint, met: bool, note: Option<String>| Pick {
        objective: objective.to_string(),
        threshold: p.threshold,
        point: *p,
        met,
        note,
        exact_threshold: None,
        exact_score: None,
        plateau: None,
        relative: false,
    };
    if let Some(p) = best_by(&|p| objective.score(p)) {
        let note = (p.counts.tp == 0).then(|| "no threshold finds any object".to_string());
        return Some(make(&p, true, note));
    }
    // Target not reached: the best on the constrained metric (then the other one).
    let p = fallback_point(sorted.iter().copied(), objective)?;
    Some(make(p, false, Some(unreached_note(objective, p))))
}

type Metric = fn(&ThresholdPoint) -> Option<f64>;

/// The point closest to an unmet target of `objective` (None for an objective without one, or
/// when no point has the constrained metric): the best on the constrained metric, a tie within
/// [`TIE`] going to the better (or equal) other metric. `points` must be in ascending
/// threshold order: a full tie goes to the later point, the higher threshold.
fn fallback_point<'a>(
    points: impl IntoIterator<Item = &'a ThresholdPoint>,
    objective: Objective,
) -> Option<&'a ThresholdPoint> {
    let (_, primary, secondary) = fallback_keys(objective)?;
    let mut best: Option<&ThresholdPoint> = None;
    for p in points {
        let Some(v) = primary(p) else { continue };
        let better = match best {
            None => true,
            Some(b) => {
                let bv = primary(b).unwrap_or(0.0);
                v > bv + TIE
                    || ((v - bv).abs() <= TIE
                        && secondary(p).unwrap_or(0.0) >= secondary(b).unwrap_or(0.0) - TIE)
            }
        };
        if better {
            best = Some(p);
        }
    }
    best
}

/// For a target objective: (target, the constrained metric, the maximized one).
fn fallback_keys(objective: Objective) -> Option<(f64, Metric, Metric)> {
    match objective {
        Objective::Precision(t) => Some((t, |p| p.precision, |p| p.recall)),
        Objective::Recall(t) => Some((t, |p| p.recall, |p| p.precision)),
        // Maximize 1 - FPR (the lowest false-alert rate), then TPR.
        Objective::Fpr(t) => Some((
            t,
            |p| p.alert_rates()?.fpr.map(|v| 1.0 - v),
            |p| p.alert_rates()?.tpr,
        )),
        // Maximize 1 - the share of empty frames alerting (the fewest false alerts), then
        // recall.
        Objective::RecallAtFa(n) => Some((
            n,
            |p| p.false_alerts?.per_100().map(|v| 1.0 - v / 100.0),
            |p| p.recall,
        )),
        _ => None,
    }
}

/// "precision 95% is not reached at any threshold; the most precise is 80% at 0.30".
fn unreached_note(objective: Objective, p: &ThresholdPoint) -> String {
    let Some((target, metric, _)) = fallback_keys(objective) else {
        return String::new();
    };
    let (name, most) = match objective {
        Objective::Recall(_) => ("recall", "the highest recall"),
        Objective::Fpr(_) => {
            return format!(
                "frame FPR {}% is not reached at any threshold; the lowest is {}% at {:.2}",
                pct0(target),
                pct0(1.0 - metric(p).unwrap_or(0.0)),
                p.threshold
            );
        }
        Objective::RecallAtFa(_) => {
            return format!(
                "{} false alerts per 100 empty frames is not reached at any threshold; the \
                 fewest is {} at {:.2}",
                num0(target),
                num0(100.0 * (1.0 - metric(p).unwrap_or(0.0))),
                p.threshold
            );
        }
        _ => ("precision", "the most precise"),
    };
    format!(
        "{name} {}% is not reached at any threshold; {most} is {}% at {:.2}",
        pct0(target),
        pct0(metric(p).unwrap_or(0.0)),
        p.threshold
    )
}

/// A curve of one group of images or objects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Curve {
    /// "overall", dataset id, tag, `size:small` or class.
    pub key: String,
    pub images: usize,
    /// Scored ground-truth objects.
    pub gt: usize,
    /// Against pseudo ground truth.
    #[serde(default)]
    pub relative: bool,
    /// At [`SWEEP_THRESHOLDS`].
    pub points: Vec<ThresholdPoint>,
    /// At the run's configured threshold (also when it is not on the grid).
    pub configured: ThresholdPoint,
    /// Grid picks ([`exact_pick`]) for the run's objective and [`Objective::STANDARD`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub picks: Vec<Pick>,
    /// The exact curve, thinned to at most [`EXACT_POINTS`] points (overall curve only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exact: Vec<ThresholdPoint>,
    /// Frame ROC AUC of the group (macro over classes; the class's own for a class curve).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roc_auc: Option<f64>,
}

/// Points of the thinned exact curve.
pub const EXACT_POINTS: usize = 150;

impl Curve {
    /// The best threshold for `objective`: the grid pick when the run computed it, else from
    /// the 0.05 grid; marked [`Pick::relative`] on a curve of pseudo ground truth.
    pub fn best(&self, objective: Objective) -> Option<Pick> {
        let key = objective.to_string();
        self.picks
            .iter()
            .find(|p| p.objective == key)
            .cloned()
            .or_else(|| pick(&self.points, objective))
            .map(|mut p| {
                p.relative |= self.relative;
                p
            })
    }
}

/// The objectives a run computes exact picks for: `objective` and [`Objective::STANDARD`].
pub fn objectives(objective: Objective) -> Vec<Objective> {
    let mut v = vec![objective];
    for o in Objective::STANDARD {
        if !v.contains(&o) {
            v.push(o);
        }
    }
    v
}

/// Threshold curves of one device run (in its [`super::AccuracyReport`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThresholdSweep {
    /// The run's configured confidence threshold.
    pub configured: f32,
    pub overall: Curve,
    #[serde(default)]
    pub by_dataset: Vec<Curve>,
    /// Day / night / small-objects tags and `size:small` (objects under 32x32 px).
    #[serde(default)]
    pub by_tag: Vec<Curve>,
    #[serde(default)]
    pub per_class: Vec<Curve>,
    /// Frame-level ROC of the overall images, with its thinned curve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roc: Option<FrameRoc>,
}

/// Tags that get their own curve (when present).
pub const SWEEP_TAGS: [&str; 5] = ["day", "night", "ir", "hard-negative", "small-objects"];
/// Key of the small-object curve.
pub const SMALL_KEY: &str = "size:small";

/// The best threshold of one group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupPick {
    pub key: String,
    pub images: usize,
    pub gt: usize,
    #[serde(default)]
    pub relative: bool,
    pub configured: ThresholdPoint,
    pub best: Option<Pick>,
    /// Frame ROC AUC of the group.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roc_auc: Option<f64>,
    /// At the threshold in the config now ([`ThresholdAdvice::now`]); filled when results are
    /// served or exported ([`super::current`]), never stored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub now: Option<ThresholdPoint>,
}

impl GroupPick {
    fn of(c: &Curve, objective: Objective) -> Self {
        Self {
            key: c.key.clone(),
            images: c.images,
            gt: c.gt,
            relative: c.relative,
            configured: c.configured,
            best: c.best(objective),
            roc_auc: c.roc_auc,
            now: None,
        }
    }
}

/// The best threshold on one device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DevicePick {
    pub device: String,
    pub best: Option<Pick>,
}

/// A model's threshold advice (on [`super::ModelResult`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThresholdAdvice {
    /// Objective of [`ThresholdAdvice::best`].
    pub objective: String,
    /// Device whose curves were used (the recommended one when it has them).
    pub device: String,
    /// Scored against pseudo ground truth.
    #[serde(default)]
    pub relative: bool,
    /// The configured threshold of the run, and its overall counts.
    pub configured: ThresholdPoint,
    pub best: Option<Pick>,
    /// The best threshold for each of [`Objective::STANDARD`] (comparison).
    #[serde(default)]
    pub alternatives: Vec<Pick>,
    /// Overall curve on [`ThresholdAdvice::device`] at [`SWEEP_THRESHOLDS`] (tables, chart).
    pub curve: Vec<ThresholdPoint>,
    /// The exact overall curve, thinned (chart).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exact: Vec<ThresholdPoint>,
    #[serde(default)]
    pub by_dataset: Vec<GroupPick>,
    #[serde(default)]
    pub by_tag: Vec<GroupPick>,
    #[serde(default)]
    pub per_class: Vec<GroupPick>,
    /// Best threshold per device that ran.
    #[serde(default)]
    pub by_device: Vec<DevicePick>,
    /// Frame-level ROC on [`ThresholdAdvice::device`] (AUCs and the curve).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roc: Option<FrameRoc>,
    /// The threshold in the config now and the counts there; filled when results are served or
    /// exported ([`super::current`]), never stored. [`ThresholdAdvice::configured`] stays the
    /// threshold of the run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub now: Option<NowThreshold>,
}

/// How [`NowThreshold::point`] was evaluated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NowSource {
    /// The run's own configured threshold (unchanged since the run).
    Run,
    /// The stored predictions (`benchmark-preds.json`): exact at any threshold.
    Preds,
    /// A point the results store at exactly this threshold (0.05 grid, best, alternatives).
    Curve,
    /// Not evaluated (no stored predictions; P/R unknown).
    None,
}

/// A model's threshold advice at the confidence threshold in the config now.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NowThreshold {
    /// Effective confidence threshold in the config (the model's own, else the global one).
    pub threshold: f32,
    /// What the loaded model runs with, when that differs (a restart is pending).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub running: Option<f32>,
    /// Overall counts at [`NowThreshold::threshold`]; None when they could not be evaluated
    /// exactly.
    pub point: Option<ThresholdPoint>,
    pub source: NowSource,
}

fn pct(v: Option<f64>) -> String {
    v.map_or("-".into(), |v| format!("{:.1}%", v * 100.0))
}

/// "P 82.1% R 74.0% F1 78.0%".
pub fn prf(p: &ThresholdPoint) -> String {
    format!(
        "P {} R {} F1 {}",
        pct(p.precision),
        pct(p.recall),
        pct(p.f1)
    )
}

/// "frame TPR 91.0% FPR 6.2%" (empty without frame rates), one sample per frame
/// ([`ThresholdPoint::alert_rates`]).
pub fn frame_rates(p: &ThresholdPoint) -> String {
    p.alert_rates().map_or(String::new(), |f| {
        format!("frame TPR {} FPR {}", pct(f.tpr), pct(f.fpr))
    })
}

/// [`prf`] followed by [`frame_rates`] when there are frame rates.
fn prf_frame(p: &ThresholdPoint) -> String {
    match frame_rates(p) {
        f if f.is_empty() => prf(p),
        f => format!("{}, {f}", prf(p)),
    }
}

impl ThresholdAdvice {
    /// The objective, parsed.
    pub fn objective(&self) -> Objective {
        self.objective.parse().unwrap_or_default()
    }

    /// The threshold configured now: [`ThresholdAdvice::now`] when filled, else the run's.
    pub fn now_threshold(&self) -> f32 {
        self.now
            .as_ref()
            .map_or(self.configured.threshold, |n| n.threshold)
    }

    /// The overall counts at [`ThresholdAdvice::now_threshold`] (None: not evaluated).
    pub fn now_point(&self) -> Option<&ThresholdPoint> {
        match &self.now {
            Some(n) => n.point.as_ref(),
            None => Some(&self.configured),
        }
    }

    /// A group's counts at [`ThresholdAdvice::now_threshold`] (None: not evaluated).
    pub fn group_now<'a>(&self, g: &'a GroupPick) -> Option<&'a ThresholdPoint> {
        match &self.now {
            Some(_) => g.now.as_ref(),
            None => Some(&g.configured),
        }
    }

    /// A stored overall point at exactly `t` (the run's configured threshold, the best and the
    /// alternative picks, the 0.05 grid); exact, as every stored point is.
    pub fn known_point_at(&self, t: f32) -> Option<ThresholdPoint> {
        let near = |p: &ThresholdPoint| (p.threshold - t).abs() < 1e-6;
        std::iter::once(&self.configured)
            .chain(self.best.iter().map(|b| &b.point))
            .chain(self.alternatives.iter().map(|b| &b.point))
            .chain(&self.curve)
            .find(|p| near(p))
            .map(|p| ThresholdPoint { threshold: t, ..*p })
    }

    /// "best confidence threshold 0.35 (F1): P 82.1% R 74.0% F1 78.0%; configured 0.50: P ..",
    /// compared with the threshold configured now ([`ThresholdAdvice::now`]) when filled.
    pub fn summary(&self) -> Option<String> {
        let b = self.best.as_ref()?;
        let mut s = format!(
            "P/R at the best confidence threshold {:.2} ({}): {}",
            b.threshold,
            self.objective().describe(),
            prf_frame(&b.point)
        );
        let now = self.now_threshold();
        let running = self
            .now
            .as_ref()
            .and_then(|n| n.running)
            .filter(|r| (r - now).abs() > 1e-6);
        if (b.threshold - now).abs() > 1e-6 {
            match self.now_point() {
                Some(p) => s.push_str(&format!(" vs {} at the configured {now:.2}", prf_frame(p))),
                None => s.push_str(&format!(" (configured {now:.2}, not evaluated)")),
            }
        } else {
            s.push_str(" (the configured threshold)");
        }
        if let Some(r) = running {
            s.push_str(&format!(" (running {r:.2} until a restart)"));
        }
        if let Some(n) = &b.note {
            s.push_str(&format!(" ({n})"));
        }
        Some(s)
    }

    /// The threshold to write into the config: the best one, when the objective was met, it
    /// finds objects and it was fitted on real labels (an unreachable precision target, a
    /// model that finds nothing or a pick on pseudo ground truth changes nothing).
    pub fn apply_value(&self) -> Option<f32> {
        if self.relative {
            return None;
        }
        applicable(self.best.as_ref())
    }

    /// Why the best threshold is not applied when it was fitted on pseudo ground truth
    /// ([`NOT_ON_LABELS`]); None otherwise.
    pub fn not_on_labels(&self) -> Option<&'static str> {
        (self.relative || self.best.as_ref().is_some_and(|b| b.relative)).then_some(NOT_ON_LABELS)
    }

    /// The threshold to write into `device`'s confidence profile
    /// ([`crate::config::ModelConfig::device_thresholds`]): that device's best
    /// ([`ThresholdAdvice::by_device`]) under the rule of [`ThresholdAdvice::apply_value`].
    /// `device` matches a row exactly, else as the same device ([`crate::config::same_device`]). None when the device did not run or its pick does not
    /// apply.
    pub fn apply_value_on(&self, device: &str) -> Option<f32> {
        let pick = self
            .by_device
            .iter()
            .find(|p| p.device == device)
            .or_else(|| {
                self.by_device
                    .iter()
                    .find(|p| crate::config::same_device(&p.device, device))
            })?;
        applicable(pick.best.as_ref())
    }
}

/// A pick's threshold when it may be written into the config: the objective was met, the
/// model finds objects there and the pick was fitted on real labels (not [`Pick::relative`]).
fn applicable(pick: Option<&Pick>) -> Option<f32> {
    pick.filter(|b| b.met && !b.relative && b.point.counts.tp > 0)
        .map(|b| b.threshold)
}

/// The sweep of a device row, when it ran and was scored.
fn sweep_of(d: &DeviceResult) -> Option<&ThresholdSweep> {
    d.run
        .as_ref()
        .filter(|_| d.ok())
        .and_then(|r| r.accuracy.as_ref())
        .and_then(|a| a.sweep.as_ref())
}

/// The threshold advice of a model from its device rows: curves of `recommended` when it has
/// them, else of the CPU reference device, else of the first scored device.
pub fn advise(
    devices: &[DeviceResult],
    recommended: Option<&str>,
    objective: Objective,
) -> Option<ThresholdAdvice> {
    let reference = super::REFERENCE_DEVICES.map(|d| d.to_string());
    let chosen = recommended
        .and_then(|r| {
            devices
                .iter()
                .find(|d| d.device == r && sweep_of(d).is_some())
        })
        .or_else(|| {
            reference.iter().find_map(|r| {
                devices
                    .iter()
                    .find(|d| d.device == *r && sweep_of(d).is_some())
            })
        })
        .or_else(|| devices.iter().find(|d| sweep_of(d).is_some()))?;
    let sweep = sweep_of(chosen)?;
    let groups = |cs: &[Curve]| -> Vec<GroupPick> {
        cs.iter().map(|c| GroupPick::of(c, objective)).collect()
    };
    Some(ThresholdAdvice {
        objective: objective.to_string(),
        device: chosen.device.clone(),
        relative: sweep.overall.relative,
        configured: sweep.overall.configured,
        best: sweep.overall.best(objective),
        alternatives: Objective::STANDARD
            .iter()
            .filter_map(|o| sweep.overall.best(*o))
            .collect(),
        curve: sweep.overall.points.clone(),
        exact: sweep.overall.exact.clone(),
        by_dataset: groups(&sweep.by_dataset),
        by_tag: groups(&sweep.by_tag),
        per_class: groups(&sweep.per_class),
        by_device: devices
            .iter()
            .filter_map(|d| {
                let s = sweep_of(d)?;
                Some(DevicePick {
                    device: d.device.clone(),
                    best: s.overall.best(objective),
                })
            })
            .collect(),
        roc: sweep.roc.clone(),
        now: None,
    })
}

/// Rows of the compact threshold table: 0.20 .. 0.80, plus the configured and best thresholds
/// when outside that range or off the grid.
pub fn table_points(curve: &[ThresholdPoint], extra: &[ThresholdPoint]) -> Vec<ThresholdPoint> {
    let mut v: Vec<ThresholdPoint> = curve
        .iter()
        .filter(|p| p.threshold >= 0.2 - 1e-6 && p.threshold <= 0.8 + 1e-6)
        .copied()
        .collect();
    for e in extra {
        if !v.iter().any(|p| (p.threshold - e.threshold).abs() < 1e-6) {
            v.push(*e);
        }
    }
    v.sort_by(|a, b| a.threshold.total_cmp(&b.threshold));
    v
}

/// Advice whose best threshold is `t` (tests of the apply paths).
#[cfg(test)]
pub(crate) fn test_advice(t: f32) -> ThresholdAdvice {
    use super::metrics::Counts;
    let p = |t: f32| {
        ThresholdPoint::new(
            t,
            Counts {
                tp: 3,
                fp: 1,
                fn_: 1,
            },
            2,
        )
    };
    ThresholdAdvice {
        objective: "f1".into(),
        device: "openvino:cpu".into(),
        relative: false,
        configured: p(0.5),
        best: Some(Pick {
            objective: "f1".into(),
            threshold: t,
            point: p(t),
            met: true,
            note: None,
            exact_threshold: Some(t),
            exact_score: p(t).f1,
            plateau: Some([t, t]),
            relative: false,
        }),
        alternatives: vec![],
        curve: super::metrics::SWEEP_THRESHOLDS
            .iter()
            .map(|&t| p(t))
            .collect(),
        exact: vec![],
        by_dataset: vec![],
        by_tag: vec![],
        per_class: vec![],
        by_device: vec![],
        roc: None,
        now: None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::metrics::Counts;
    use super::*;
    use crate::benchmark::metrics::SWEEP_THRESHOLDS;

    fn pt(t: f32, tp: usize, fp: usize, fn_: usize) -> ThresholdPoint {
        ThresholdPoint::new(t, Counts { tp, fp, fn_ }, 10)
    }

    /// A typical curve over 10 objects: recall falls and precision rises with the threshold.
    fn curve() -> Vec<ThresholdPoint> {
        vec![
            pt(0.1, 10, 30, 0), // P .25  R 1   F1 .4    F2 .625
            pt(0.2, 9, 6, 1),   // P .6   R .9  F1 .72   F2 .818
            pt(0.3, 8, 2, 2),   // P .8   R .8  F1 .8    F2 .8
            pt(0.4, 7, 1, 3),   // P .875 R .7  F1 .778  F2 .729
            pt(0.5, 6, 0, 4),   // P 1    R .6  F1 .75   F2 .652
            pt(0.6, 0, 0, 10),  // nothing detected
        ]
    }

    #[test]
    fn objectives_parse_and_print() {
        for (s, o) in [
            ("f1", Objective::F1),
            (" F2 ", Objective::F2),
            ("precision:0.9", Objective::Precision(0.9)),
            ("Precision:1", Objective::Precision(1.0)),
            ("", Objective::F1),
        ] {
            assert_eq!(s.parse::<Objective>().unwrap(), o, "{s}");
        }
        for bad in [
            "f3",
            "precision:",
            "precision:abc",
            "precision:0",
            "precision:1.5",
        ] {
            assert!(bad.parse::<Objective>().is_err(), "{bad}");
        }
        assert_eq!(Objective::Precision(0.9).to_string(), "precision:0.9");
        assert_eq!(
            "recall:0.8".parse::<Objective>().unwrap(),
            Objective::Recall(0.8)
        );
        assert_eq!(Objective::Recall(0.8).to_string(), "recall:0.8");
        assert!("recall:2".parse::<Objective>().is_err());
        assert_eq!(
            serde_json::to_string(&Objective::Precision(0.85)).unwrap(),
            "\"precision:0.85\""
        );
        let o: Objective = serde_json::from_str("\"f2\"").unwrap();
        assert_eq!(o, Objective::F2);
        assert!(serde_json::from_str::<Objective>("\"recall\"").is_err());
        assert_eq!(
            Objective::Precision(0.9).describe(),
            "recall at precision \u{2265} 90%"
        );
        // Frame objectives.
        assert_eq!("youden".parse::<Objective>().unwrap(), Objective::Youden);
        assert_eq!(" Youden ".parse::<Objective>().unwrap(), Objective::Youden);
        assert_eq!(
            "fpr:0.05".parse::<Objective>().unwrap(),
            Objective::Fpr(0.05)
        );
        assert_eq!("fpr:0".parse::<Objective>().unwrap(), Objective::Fpr(0.0));
        for bad in ["fpr:", "fpr:x", "fpr:1", "fpr:-0.1"] {
            assert!(bad.parse::<Objective>().is_err(), "{bad}");
        }
        assert_eq!(Objective::Youden.to_string(), "youden");
        assert_eq!(Objective::Fpr(0.05).to_string(), "fpr:0.05");
        assert_eq!(
            serde_json::to_string(&Objective::Fpr(0.1)).unwrap(),
            "\"fpr:0.1\""
        );
        assert!(Objective::Youden.is_frame() && Objective::Fpr(0.1).is_frame());
        assert!(!Objective::F1.is_frame());
        assert!(Objective::STANDARD.contains(&Objective::Youden));
    }

    /// Images with person/car frames (scores tend higher on positive frames) and matching box
    /// predictions; `coarse` rounds scores so frames tie.
    fn random_frames(seed: u64, n: usize, coarse: bool) -> Vec<super::super::metrics::SweepImage> {
        use super::super::metrics::{SweepImage, SweepPred};
        use super::super::roc::FrameSample;
        let mut seed = seed;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as f32 / (1u64 << 31) as f32
        };
        (0..n)
            .map(|_| {
                let mut img = SweepImage {
                    frame_classes: vec!["person".into(), "car".into()],
                    ..SweepImage::default()
                };
                for class in ["person", "car"] {
                    let positive = rnd() < 0.4;
                    let mut score = if rnd() < 0.25 {
                        0.0
                    } else {
                        (0.05 + rnd() * 0.6 + if positive { 0.3 } else { 0.0 }).min(1.0)
                    };
                    if coarse && score > 0.0 {
                        score = ((score * 20.0).round() / 20.0).max(0.05);
                    }
                    if positive {
                        img.gt.push((class.into(), false));
                    }
                    if score > 0.0 {
                        img.preds.push(SweepPred {
                            confidence: score,
                            after: f32::NEG_INFINITY,
                            class: class.into(),
                            tp: positive,
                            small: false,
                        });
                    }
                    img.frames.push(FrameSample {
                        class: class.into(),
                        positive,
                        score,
                        spans: None,
                    });
                }
                img
            })
            .collect()
    }

    /// Youden and fpr:x picks equal brute force over the 0.01 grid of the per-frame rates; the
    /// reported threshold keeps the FPR target and sits in the middle of its plateau.
    #[test]
    fn frame_objectives_match_brute_force() {
        use super::super::metrics::SweepScope;
        for seed in 1..40u64 {
            let imgs = random_frames(seed, 20 + seed as usize * 2, seed % 2 == 0);
            let refs: Vec<&super::super::metrics::SweepImage> = imgs.iter().collect();
            let bp = Breakpoints::new(&refs, SweepScope::All);
            let fs = bp.per_frame.as_ref().unwrap();
            // One sample per frame: as many as frames (none is ignore-only here).
            assert_eq!(fs.positives + fs.negatives, imgs.len());
            for o in [
                Objective::Youden,
                Objective::Fpr(0.0),
                Objective::Fpr(0.05),
                Objective::Fpr(0.2),
            ] {
                let score = |t: f32| {
                    let r = fs.rates_at(t);
                    match o {
                        Objective::Youden => r.youden(),
                        Objective::Fpr(x) => r.fpr.filter(|&v| v <= x + 1e-12).and(r.tpr),
                        _ => unreachable!(),
                    }
                };
                let brute = grid()
                    .into_iter()
                    .filter_map(score)
                    .fold(f64::NEG_INFINITY, f64::max);
                let p = exact_pick(&bp, o).unwrap();
                assert!(p.met, "seed {seed} {o}");
                assert!(
                    (p.exact_score.unwrap() - brute).abs() < 1e-12,
                    "seed {seed} {o}: {:?} vs {brute}",
                    p.exact_score
                );
                assert_eq!(p.point, bp.point_at(p.threshold));
                assert_eq!(p.point.per_frame, Some(fs.rates_at(p.threshold)));
                assert!(p.threshold >= MIN_THRESHOLD);
                let f = p.point.per_frame.unwrap();
                if let Objective::Fpr(x) = o {
                    assert!(f.fpr.unwrap() <= x + 1e-12, "seed {seed} {o}: {f:?}");
                }
                // Every grid value of the plateau scores within PLATEAU of the optimum, the
                // pick is its middle.
                let [lo, hi] = p.plateau.unwrap();
                let run: Vec<f32> = grid()
                    .into_iter()
                    .filter(|&t| t >= lo - 1e-6 && t <= hi + 1e-6)
                    .collect();
                for &t in &run {
                    let v = score(t).unwrap();
                    assert!(v >= brute - PLATEAU - 1e-9, "seed {seed} t {t}");
                }
                assert_eq!(run.get((run.len() - 1) / 2).copied(), Some(p.threshold));
                // Ties go to the higher threshold: no higher grid value scores as well.
                let et = p.exact_threshold.unwrap();
                for t in grid().into_iter().filter(|&t| t > et) {
                    assert!(
                        score(t).is_none_or(|v| v < brute - 1e-12),
                        "seed {seed} {o}: {t} ties {et}"
                    );
                }
            }
        }
    }

    #[test]
    fn frame_objectives_hand_computed_and_undefined() {
        use super::super::metrics::{SweepImage, SweepScope};
        // Positive frames .9 .6 .3; negative frames .7 .2 0.
        let mut imgs = Vec::new();
        for (score, positive) in [
            (0.9, true),
            (0.6, true),
            (0.3, true),
            (0.7, false),
            (0.2, false),
            (0.0, false),
        ] {
            let mut i = SweepImage {
                frame_classes: vec!["person".into()],
                ..SweepImage::default()
            };
            if positive {
                i.gt.push(("person".into(), false));
            }
            if score > 0.0 {
                i.preds.push(sp("person", score, positive));
            }
            imgs.push(i);
        }
        let refs: Vec<&SweepImage> = imgs.iter().collect();
        let bp = Breakpoints::new(&refs, SweepScope::All);
        // J = TPR - FPR by alerting set: {.9} 1/3, {+.7} 0, {+.6} 1/3, {+.3} 2/3, {+.2} 1/3.
        // Best: thresholds (0.2, 0.3], grid 0.21 ..= 0.30, middle 0.25.
        let p = exact_pick(&bp, Objective::Youden).unwrap();
        assert_eq!(p.exact_threshold, Some(0.3));
        assert!((p.exact_score.unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(p.plateau, Some([0.21, 0.3]));
        assert_eq!(p.threshold, 0.25);
        let f = p.point.per_frame.unwrap();
        assert_eq!((f.tp, f.fp), (3, 1));
        // FPR <= 0.4 (1 of 3 negatives may alert): highest TPR keeps down to .3 (FPR 1/3).
        let q = exact_pick(&bp, Objective::Fpr(0.4)).unwrap();
        assert_eq!(q.exact_threshold, Some(0.3));
        // FPR 0: only the .9 frame may alert -> 0.71 ..= 0.90; middle 0.80.
        let z = exact_pick(&bp, Objective::Fpr(0.0)).unwrap();
        assert_eq!((z.exact_threshold, z.threshold), (Some(0.9), 0.8));
        assert_eq!(z.point.per_frame.unwrap().fp, 0);
        // No negative frames: frame objectives are undefined; box objectives still work.
        let only_pos: Vec<&SweepImage> = refs.iter().take(3).copied().collect();
        let bp = Breakpoints::new(&only_pos, SweepScope::All);
        assert!(exact_pick(&bp, Objective::Youden).is_none());
        assert!(exact_pick(&bp, Objective::Fpr(0.1)).is_none());
        assert!(exact_pick(&bp, Objective::F1).is_some());
        // No frame samples at all (small-object scope).
        let bp = Breakpoints::new(&refs, SweepScope::Small);
        assert!(bp.frames.is_none() && bp.per_frame.is_none() && bp.empty.is_none());
        assert!(exact_pick(&bp, Objective::Youden).is_none());
        assert!(exact_pick(&bp, Objective::RecallAtFa(5.0)).is_none());
    }

    #[test]
    fn picks_by_objective() {
        let c = curve();
        let f1 = pick(&c, Objective::F1).unwrap();
        assert_eq!((f1.threshold, f1.met, f1.note.is_none()), (0.3, true, true));
        let f2 = pick(&c, Objective::F2).unwrap();
        assert_eq!(f2.threshold, 0.2);
        // Highest recall with precision >= 0.85: 0.4 (P .875 R .7).
        let p = pick(&c, Objective::Precision(0.85)).unwrap();
        assert_eq!((p.threshold, p.met), (0.4, true));
        assert_eq!(p.objective, "precision:0.85");
        // Highest precision with recall >= 0.8: 0.3 (P .8 R .8); recall 1 only at 0.1.
        assert_eq!(pick(&c, Objective::Recall(0.8)).unwrap().threshold, 0.3);
        assert_eq!(pick(&c, Objective::Recall(1.0)).unwrap().threshold, 0.1);
        // Precision 1 is reached at 0.5 only.
        assert_eq!(pick(&c, Objective::Precision(1.0)).unwrap().threshold, 0.5);
    }

    #[test]
    fn ties_go_to_the_higher_threshold() {
        // Same counts at 0.3 and 0.35: same F1, the higher threshold wins.
        let c = vec![
            pt(0.25, 5, 5, 5),
            pt(0.3, 8, 2, 2),
            pt(0.35, 8, 2, 2),
            pt(0.4, 1, 0, 9),
        ];
        assert_eq!(pick(&c, Objective::F1).unwrap().threshold, 0.35);
        assert_eq!(pick(&c, Objective::Precision(0.8)).unwrap().threshold, 0.35);
        // Order of the input does not matter.
        let mut r = c.clone();
        r.reverse();
        assert_eq!(pick(&r, Objective::F1).unwrap().threshold, 0.35);
    }

    #[test]
    fn unreachable_precision_falls_back_with_a_note() {
        let c = vec![pt(0.1, 10, 30, 0), pt(0.2, 9, 6, 1), pt(0.3, 8, 2, 2)];
        let p = pick(&c, Objective::Precision(0.95)).unwrap();
        assert!(!p.met);
        assert_eq!(p.threshold, 0.3, "the most precise threshold");
        let note = p.note.unwrap();
        assert!(note.contains("95%") && note.contains("80%"), "{note}");
        // Equal precision: the higher recall.
        let c = vec![pt(0.1, 4, 4, 6), pt(0.2, 2, 2, 8)];
        let p = pick(&c, Objective::Precision(0.9)).unwrap();
        assert_eq!(p.threshold, 0.1);
        // No ground truth, or no detections at all.
        let none = vec![ThresholdPoint::new(
            0.5,
            Counts {
                tp: 0,
                fp: 3,
                fn_: 0,
            },
            2,
        )];
        assert!(pick(&none, Objective::F1).is_none());
        let silent = vec![pt(0.5, 0, 0, 3), pt(0.6, 0, 0, 3)];
        assert!(pick(&silent, Objective::Precision(0.9)).is_none());
        // F1 0 everywhere: still a pick (highest threshold), with a note.
        let f = pick(&silent, Objective::F1).unwrap();
        assert_eq!(f.threshold, 0.6);
        assert!(f.note.is_some());
    }

    #[test]
    fn table_rows_cover_the_middle_and_the_marks() {
        let full: Vec<ThresholdPoint> = SWEEP_THRESHOLDS.iter().map(|&t| pt(t, 1, 1, 1)).collect();
        let rows = table_points(&full, &[]);
        assert_eq!(rows.len(), 13);
        assert_eq!((rows[0].threshold, rows[12].threshold), (0.2, 0.8));
        let rows = table_points(
            &full,
            &[pt(0.1, 1, 1, 1), pt(0.42, 1, 1, 1), pt(0.5, 1, 1, 1)],
        );
        assert_eq!(rows.len(), 15);
        assert_eq!(rows[0].threshold, 0.1);
        assert!(rows.iter().any(|p| p.threshold == 0.42));
    }

    /// NMS families post-processed at 0.05 and filtered at `t` give exactly the detections of
    /// post-processing at `t` (greedy NMS keeps a box based on more confident boxes only).
    #[test]
    fn nms_after_low_threshold_then_filter_is_exact() {
        use crate::model::yolo5::Yolo5;
        use crate::model::{Family, NamedOutput, OutputBuf, PostParams, PreprocessCtx, ResizeMode};
        let mut seed: u64 = 42;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as f32 / (1u64 << 31) as f32
        };
        let classes = 3;
        let rows = 400;
        let mut data = Vec::with_capacity(rows * (5 + classes));
        for _ in 0..rows {
            // Many boxes on a small canvas: plenty overlap.
            let (cx, cy) = (20.0 + rnd() * 600.0, 20.0 + rnd() * 600.0);
            data.extend([cx, cy, 30.0 + rnd() * 40.0, 30.0 + rnd() * 40.0, rnd()]);
            for _ in 0..classes {
                data.push(rnd());
            }
        }
        let out = vec![NamedOutput {
            name: "output0".into(),
            shape: vec![1, rows, 5 + classes],
            data: OutputBuf::F32(data),
        }];
        let ctx = PreprocessCtx {
            orig_w: 640,
            orig_h: 640,
            input_w: 640,
            input_h: 640,
            mode: ResizeMode::Letterbox,
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        };
        let fam = Yolo5::new(classes);
        let at = |t: f32| {
            fam.postprocess(
                &out,
                &ctx,
                &PostParams {
                    confidence_threshold: t,
                    nms_iou: 0.45,
                },
            )
            .unwrap()
        };
        let low = at(0.05);
        assert!(low.len() > 20);
        for t in SWEEP_THRESHOLDS {
            let filtered: Vec<_> = low.iter().filter(|d| d.score >= t).cloned().collect();
            assert_eq!(filtered, at(t), "t = {t}");
        }
    }

    #[test]
    fn plateau_prefers_the_widest_near_best_run() {
        let v = |x: &[Option<f64>]| x.to_vec();
        // A lone spike (0.800 at index 1) vs a broad top within 0.002 (indices 3..=5).
        let s = v(&[
            Some(0.70),
            Some(0.800),
            Some(0.70),
            Some(0.7985),
            Some(0.7990),
            Some(0.7981),
            Some(0.0),
        ]);
        let r = grid_run(&s, PLATEAU).unwrap();
        assert_eq!((r.best, r.first, r.last, r.middle()), (1, 3, 5, 4));
        // Without tolerance: the spike only.
        let r = grid_run(&s, 0.0).unwrap();
        assert_eq!((r.first, r.last), (1, 1));
        // Equally wide runs: the higher one; equal scores: the higher value is the best. An
        // even count takes the lower middle.
        let t = v(&[Some(0.5), Some(0.5), Some(0.1), Some(0.5), Some(0.5), None]);
        let r = grid_run(&t, PLATEAU).unwrap();
        assert_eq!((r.best, r.first, r.last, r.middle()), (4, 3, 4, 3));
        // A missed target (None) splits runs.
        let u = v(&[Some(0.9), None, Some(0.9), Some(0.9)]);
        assert_eq!(
            grid_run(&u, PLATEAU).map(|r| (r.first, r.last)),
            Some((2, 3))
        );
        assert!(grid_run(&[None, None], PLATEAU).is_none());
        assert!(grid_run(&[], PLATEAU).is_none());
        // The grid: 0.05 ..= 1.00 by 0.01, exact literals.
        let g = grid();
        assert_eq!(g.len(), 96);
        assert_eq!((g[0], g[30], g[95]), (0.05, 0.35, 1.0));
        assert!(g.iter().all(|t| *t == floor_cent(f64::from(*t))));
        assert_eq!(floor_cent(0.35f32 as f64), 0.35);
        assert_eq!(floor_cent(0.3549), 0.35);
        assert_eq!(floor_cent(0.0599), 0.05);
    }

    fn random_breakpoints(seed: u64, n: usize) -> Breakpoints {
        use super::super::metrics::{SweepImage, SweepPred, SweepScope};
        let mut seed = seed;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) as f32 / (1u64 << 31) as f32
        };
        let mut img = SweepImage::default();
        let objects = n / 2;
        for _ in 0..objects {
            img.gt.push(("person".into(), false));
        }
        let mut tps = 0;
        for _ in 0..n {
            let c = 0.05 + rnd() * 0.95;
            // More confident detections are more often right.
            let tp = tps < objects && rnd() < c;
            tps += usize::from(tp);
            img.preds.push(SweepPred {
                confidence: c,
                after: f32::NEG_INFINITY,
                class: "person".into(),
                tp,
                small: false,
            });
        }
        Breakpoints::new(&[&img], SweepScope::All)
    }

    /// The pick equals brute force over the 0.01 grid (so it is at least the 0.05 grid's best),
    /// is the middle of its plateau, at least 0.05, and its numbers are its own.
    #[test]
    fn exact_pick_matches_brute_force_on_the_grid() {
        for seed in 1..40u64 {
            let bp = random_breakpoints(seed, 30 + seed as usize * 3);
            for o in [
                Objective::F1,
                Objective::F2,
                Objective::F05,
                Objective::Precision(0.8),
                Objective::Precision(0.9),
                Objective::Recall(0.7),
                Objective::Recall(0.95),
            ] {
                let score = |t: f32| o.score(&bp.point_at(t));
                let brute = grid()
                    .into_iter()
                    .filter_map(score)
                    .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))));
                let coarse: Vec<ThresholdPoint> =
                    SWEEP_THRESHOLDS.iter().map(|&t| bp.point_at(t)).collect();
                let g = pick(&coarse, o);
                let Some(p) = exact_pick(&bp, o) else {
                    assert!(brute.is_none() && g.is_none(), "seed {seed} {o}");
                    continue;
                };
                if !p.met {
                    // No grid value meets the target.
                    assert!(brute.is_none(), "seed {seed} {o}");
                    assert!(p.note.as_deref().unwrap().contains("not reached"));
                    continue;
                }
                let exact = p.exact_score.unwrap();
                assert!((exact - brute.unwrap()).abs() < 1e-12, "seed {seed} {o}");
                if let Some(g) = g.filter(|g| g.met) {
                    assert!(
                        exact >= o.score(&g.point).unwrap() - 1e-12,
                        "seed {seed} {o}"
                    );
                }
                assert!(p.threshold >= MIN_THRESHOLD);
                assert!(((p.threshold * 100.0).round() / 100.0 - p.threshold).abs() < 1e-6);
                assert_eq!(p.point, bp.point_at(p.threshold));
                let [lo, hi] = p.plateau.unwrap();
                assert!(lo <= hi && lo >= MIN_THRESHOLD && hi <= MAX_THRESHOLD);
                let run: Vec<f32> = grid()
                    .into_iter()
                    .filter(|&t| t >= lo - 1e-6 && t <= hi + 1e-6)
                    .collect();
                for &t in &run {
                    let v = score(t).unwrap();
                    assert!(v >= exact - PLATEAU - 1e-9, "seed {seed} {o} t {t}");
                }
                assert_eq!(run.get((run.len() - 1) / 2).copied(), Some(p.threshold));
                // The pick itself is within PLATEAU of the best grid value and meets the target.
                assert!(score(p.threshold).unwrap() >= exact - PLATEAU - 1e-9);
                assert!(o.target_met(&p.point).unwrap_or(true));
            }
        }
    }

    /// A precision target met only between two 0.01 steps is unmet on the grid, so it is never
    /// applied (BQ-07). Detections: FP .519, TP .515, FP .512: precision 0.5 holds only for
    /// thresholds in (0.512, 0.515]; 0.51 keeps 1/3, 0.52 and above keep nothing.
    #[test]
    fn target_met_only_between_steps_is_not_applied() {
        use super::super::metrics::{SweepImage, SweepScope};
        let mut img = SweepImage::default();
        img.gt.push(("person".into(), false));
        for (c, tp) in [(0.519, false), (0.515, true), (0.512, false)] {
            img.preds.push(sp("person", c, tp));
        }
        let bp = Breakpoints::new(&[&img], SweepScope::All);
        let o = Objective::Precision(0.5);
        let p = exact_pick(&bp, o).unwrap();
        assert!(!p.met, "{p:?}");
        assert!(p.note.as_deref().unwrap().contains("not reached"), "{p:?}");
        // The most precise grid value: 0.05 ..= 0.51 all keep 1 of 3.
        assert_eq!(p.threshold, 0.51);
        assert_eq!(o.target_met(&bp.point_at(p.threshold)), Some(false));
        let mut advice = test_advice(0.5);
        advice.best = Some(p.clone());
        assert_eq!(advice.apply_value(), None);
        // The same pick when the target holds is applied.
        advice.best = Some(Pick { met: true, ..p });
        assert!(advice.apply_value().is_some());
    }

    #[test]
    fn exact_pick_hand_computed() {
        use super::super::metrics::{SweepImage, SweepScope};
        // 4 objects; detections by confidence: TP .92, TP .81, FP .63, TP .47, FP .12.
        let mut img = SweepImage::default();
        for _ in 0..4 {
            img.gt.push(("person".into(), false));
        }
        for (c, tp) in [
            (0.92, true),
            (0.81, true),
            (0.63, false),
            (0.47, true),
            (0.12, false),
        ] {
            img.preds.push(sp("person", c, tp));
        }
        let bp = Breakpoints::new(&[&img], SweepScope::All);
        // F1 by kept set: {.92} 2/5=.4, {..81} 4/6=.667, {..63} 4/7=.571, {..47} 6/8=.75,
        // {..12} 6/9=.667. Best: keep down to .47, i.e. grid 0.13 ..= 0.47 (35 values).
        let p = exact_pick(&bp, Objective::F1).unwrap();
        assert_eq!(p.exact_threshold, Some(0.47));
        assert!((p.exact_score.unwrap() - 0.75).abs() < 1e-12);
        assert_eq!(p.plateau, Some([0.13, 0.47]));
        // The 18th of 35 values.
        assert_eq!(p.threshold, 0.3);
        assert!((p.point.f1.unwrap() - 0.75).abs() < 1e-12);
        // The 0.05 grid would say 0.15..0.45 (equal F1), tie -> 0.45.
        let coarse: Vec<ThresholdPoint> =
            SWEEP_THRESHOLDS.iter().map(|&t| bp.point_at(t)).collect();
        assert_eq!(pick(&coarse, Objective::F1).unwrap().threshold, 0.45);
        // Precision 1: highest recall is keeping {.92, .81} (R .5): 0.64 ..= 0.81 -> 0.72.
        let p = exact_pick(&bp, Objective::Precision(1.0)).unwrap();
        assert_eq!((p.threshold, p.point.recall), (0.72, Some(0.5)));
        // F0.5 (precision counts more): {.92, .81} 1.25*2/(1.25*2+.25*2) = .833 beats
        // {..47} 1.25*3/(3.75+.25+1)=.75.
        let p = exact_pick(&bp, Objective::F05).unwrap();
        assert_eq!(p.plateau, Some([0.64, 0.81]));
    }

    #[test]
    fn device_apply_values_keep_the_apply_rule() {
        let base = test_advice(0.3);
        let pick = |t: f32, met: bool, tp: usize| {
            let mut p = base.best.clone().unwrap();
            p.threshold = t;
            p.met = met;
            p.point.counts.tp = tp;
            Some(p)
        };
        let mut a = base.clone();
        a.by_device = vec![
            DevicePick {
                device: "ort:coreml".into(),
                best: pick(0.3, true, 3),
            },
            DevicePick {
                device: "openvino:cpu".into(),
                best: pick(0.34, true, 3),
            },
            // Target not met: not applied.
            DevicePick {
                device: "openvino:gpu".into(),
                best: pick(0.5, false, 3),
            },
            // Finds nothing: not applied.
            DevicePick {
                device: "ort:cuda:0".into(),
                best: pick(0.6, true, 0),
            },
            DevicePick {
                device: "ort:cpu".into(),
                best: None,
            },
        ];
        assert_eq!(a.apply_value_on("ort:coreml"), Some(0.3));
        assert_eq!(a.apply_value_on("openvino:cpu"), Some(0.34));
        assert_eq!(
            a.apply_value_on("CPU"),
            Some(0.34),
            "same device, other spelling"
        );
        assert_eq!(a.apply_value_on("openvino:gpu"), None);
        assert_eq!(a.apply_value_on("openvino:gpu.0"), None);
        assert_eq!(a.apply_value_on("ort:cuda"), None);
        assert_eq!(a.apply_value_on("ort:cpu"), None);
        assert_eq!(a.apply_value_on("ort:directml"), None, "did not run");
        // The overall pick is independent of the device rows.
        assert_eq!(a.apply_value(), Some(0.3));
        a.by_device[1].best = pick(0.34, true, 3);
        a.by_device.push(DevicePick {
            device: "openvino:gpu.1".into(),
            best: pick(0.41, true, 2),
        });
        assert_eq!(a.apply_value_on("openvino:gpu.1"), Some(0.41));
    }

    /// One detection of the sweep tests.
    /// QA-167: the unmet-target fallback is one rule for the exact grid and the stored grid:
    /// equal precision goes to the higher recall, not the higher threshold, and a full tie to
    /// the higher threshold.
    #[test]
    fn unmet_target_fallback_is_shared_by_both_picks() {
        use super::super::metrics::{SweepImage, SweepScope};
        // Four persons; detections fp .9, tp .8, fp .5, tp .4. Precision .5 at 0.51 ..= 0.80
        // (recall .25) and at or below 0.40 (recall .5); never 0.9.
        let img = SweepImage {
            gt: vec![("person".into(), false); 4],
            preds: vec![
                sp("person", 0.9, false),
                sp("person", 0.8, true),
                sp("person", 0.5, false),
                sp("person", 0.4, true),
            ],
            ..SweepImage::default()
        };
        let bp = Breakpoints::new(&[&img], SweepScope::All);
        let o = Objective::Precision(0.9);
        let exact = exact_pick(&bp, o).unwrap();
        assert!(!exact.met);
        assert_eq!(exact.threshold, 0.4);
        assert_eq!(exact.plateau, Some([0.4, 0.4]));
        let points: Vec<ThresholdPoint> = grid().into_iter().map(|t| bp.point_at(t)).collect();
        let stored = pick(&points, o).unwrap();
        assert_eq!((stored.threshold, stored.met), (0.4, false));
        assert_eq!(stored.note, exact.note);
        // Unsorted input: `pick` sorts before the fallback.
        let mut shuffled = points.clone();
        shuffled.reverse();
        assert_eq!(pick(&shuffled, o).unwrap().threshold, 0.4);
        // A full tie goes to the higher threshold.
        let c = vec![pt(0.3, 2, 2, 8), pt(0.5, 2, 2, 8), pt(0.7, 1, 2, 9)];
        assert_eq!(pick(&c, o).unwrap().threshold, 0.5);
        assert_eq!(fallback_point(&c, o).map(|p| p.threshold), Some(0.5));
        // An objective without a target has no fallback.
        assert!(fallback_point(&c, Objective::F1).is_none());
    }

    fn sp(class: &str, confidence: f32, tp: bool) -> super::super::metrics::SweepPred {
        super::super::metrics::SweepPred {
            confidence,
            after: f32::NEG_INFINITY,
            class: class.into(),
            tp,
            small: false,
        }
    }

    /// B3: the best kept set lies between two 0.01 steps ((0.302, 0.308]: no multiple of 0.01
    /// inside). The pick is the best value of the 0.01 grid, never a rounded value outside the
    /// optimum that scores worse.
    #[test]
    fn grid_pick_is_the_best_grid_value_when_the_optimum_is_between_steps() {
        use super::super::metrics::{SweepImage, SweepScope};
        // 2 objects; TP .9, TP .308, 5 FP at .302.
        let mut img = SweepImage {
            gt: vec![("person".into(), false), ("person".into(), false)],
            preds: vec![sp("person", 0.9, true), sp("person", 0.308, true)],
            ..SweepImage::default()
        };
        img.preds.extend((0..5).map(|_| sp("person", 0.302, false)));
        let bp = Breakpoints::new(&[&img], SweepScope::All);
        let p = exact_pick(&bp, Objective::F1).unwrap();
        let grid_best = (5..=100)
            .filter_map(|k| bp.point_at(k as f32 / 100.0).f1)
            .fold(f64::NEG_INFINITY, f64::max);
        // F1: 2/3 at 0.31 ..= 0.90 (TP .9 only), 4/9 at 0.30 and below.
        assert!((grid_best - 2.0 / 3.0).abs() < 1e-12);
        assert!(
            (p.point.f1.unwrap() - grid_best).abs() < 1e-12,
            "{} at {}",
            p.point.f1.unwrap(),
            p.threshold
        );
        // The 60 values 0.31 ..= 0.90 tie: the lower middle one.
        assert_eq!(p.threshold, 0.60);
        assert_eq!(p.plateau, Some([0.31, 0.90]));
    }

    /// F0.5 weighs precision four times as much as recall.
    #[test]
    fn f05_counts_precision_twice() {
        let o: Objective = "f05".parse().unwrap();
        assert_eq!("F0.5".parse::<Objective>().unwrap(), o);
        assert_eq!(o.to_string(), "f05");
        assert_eq!(serde_json::to_string(&o).unwrap(), "\"f05\"");
        assert_eq!(serde_json::from_str::<Objective>("\"f0.5\"").unwrap(), o);
        assert!(o.describe().contains("F0.5"), "{}", o.describe());
        let c = curve();
        for p in &c {
            assert_eq!(o.score(p), p.counts.f_beta(0.5), "{}", p.threshold);
        }
        // F0.5 by threshold: .3 -> 10/12.5 = .8, .4 -> 8.75/10.5 = .833, .5 -> 7.5/8.5 = .882.
        assert_eq!(pick(&c, o).unwrap().threshold, 0.5);
    }

    /// One image of the frame tests: its ground-truth labels and its (label, confidence)
    /// predictions.
    type FrameSpec<'a> = (&'a [&'a str], &'a [(&'a str, f32)]);

    /// Images of the frame tests, classes person, car, dog, cat, every one annotated.
    fn frame_images(spec: &[FrameSpec<'_>]) -> Vec<super::super::metrics::SweepImage> {
        use super::super::metrics::{EvalImage, GtBox, PredBox, SweepImage};
        let classes: Vec<String> = ["person", "car", "dog", "cat"].map(String::from).to_vec();
        spec.iter()
            .map(|(gt, preds)| {
                let e = EvalImage {
                    gt: gt
                        .iter()
                        .enumerate()
                        .map(|(i, l)| GtBox {
                            label: l.to_string(),
                            bbox: [i as f32 * 200.0, 0.0, i as f32 * 200.0 + 100.0, 100.0],
                            ignore: false,
                            crowd: false,
                        })
                        .collect(),
                    preds: preds
                        .iter()
                        .enumerate()
                        .map(|(i, (l, c))| PredBox {
                            label: l.to_string(),
                            confidence: *c,
                            bbox: [i as f32 * 200.0, 0.0, i as f32 * 200.0 + 100.0, 100.0],
                            window: None,
                        })
                        .collect(),
                    ..Default::default()
                };
                SweepImage::of(&e).with_frames(&e, &classes)
            })
            .collect()
    }

    /// Youden and fpr:x use one sample per frame: frames with an object (.8, .5) and five
    /// without, three of them with a stray car at .6. Pooled (frame, class) samples would count
    /// 26 negatives (each frame once per class it lacks) and pick a low threshold that alerts
    /// on 3 of the 5 empty frames.
    #[test]
    fn frame_objectives_use_one_sample_per_frame() {
        use super::super::metrics::{SweepImage, SweepScope};
        let imgs = frame_images(&[
            (&["person"], &[("person", 0.8)]),
            (&["person"], &[("person", 0.5)]),
            (&[], &[("car", 0.6)]),
            (&[], &[("car", 0.6)]),
            (&[], &[("car", 0.6)]),
            (&[], &[]),
            (&[], &[]),
        ]);
        let refs: Vec<&SweepImage> = imgs.iter().collect();
        let bp = Breakpoints::new(&refs, SweepScope::All);
        // J per frame: t <= .5: 1 - 3/5 = .4; (.6, .8]: .5 - 0 = .5 at 0.61 ..= 0.80.
        let y = exact_pick(&bp, Objective::Youden).unwrap();
        assert_eq!(y.threshold, 0.70, "{y:?}");
        // FPR <= .2: no stray car may alert: 0.61 ..= 0.80 (TPR .5).
        let f = exact_pick(&bp, Objective::Fpr(0.2)).unwrap();
        assert_eq!(f.threshold, 0.70, "{f:?}");
        // The rates beside the pick are per frame: 1 of 2 frames with an object, none of 5
        // without; the pooled (frame, class) rates the ROC chart uses have 26 negatives.
        let f = y.point.per_frame.unwrap();
        assert_eq!((f.tp, f.fp, f.positives, f.negatives), (1, 0, 2, 5));
        assert_eq!(y.point.frame.unwrap().negatives, 26);
        assert_eq!(y.point.alert_rates(), Some(&f));
        assert!(frame_rates(&y.point).contains("TPR 50.0% FPR 0.0%"));
        // Empty frames: 5, three alert at 0.60 and below.
        let fa = bp.point_at(0.6).false_alerts.unwrap();
        assert_eq!((fa.empty_frames, fa.alerts), (5, 3));
        assert_eq!(fa.per_100(), Some(60.0));
        // An old point (pooled `frame` only) still scores the frame objectives.
        let old = ThresholdPoint {
            per_frame: None,
            ..y.point
        };
        assert_eq!(
            Objective::Youden.score(&old),
            old.frame.as_ref().and_then(FrameRates::youden)
        );
    }

    /// recall_at_fa:N: the highest box recall with at most N false alerts per 100 empty
    /// frames. Empty frames: no object of a scored class (an ignore region alone keeps a
    /// frame empty; a detection inside it is no alert).
    #[test]
    fn recall_at_false_alerts_per_100_empty_frames() {
        use super::super::metrics::{EvalImage, GtBox, PredBox, SweepImage, SweepScope};
        let mut imgs = frame_images(&[
            (&["person"], &[("person", 0.9)]),
            (&["person"], &[("person", 0.4)]),
            (&[], &[("car", 0.6)]),
            (&[], &[("person", 0.3)]),
            (&[], &[]),
        ]);
        let region = [0.0, 0.0, 100.0, 100.0];
        let e = EvalImage {
            gt: vec![GtBox {
                label: "person".into(),
                bbox: region,
                ignore: true,
                crowd: false,
            }],
            preds: vec![PredBox {
                label: "person".into(),
                confidence: 0.8,
                bbox: [1.0, 1.0, 100.0, 100.0],
                window: None,
            }],
            ..Default::default()
        };
        let classes: Vec<String> = ["person", "car", "dog", "cat"].map(String::from).to_vec();
        imgs.push(SweepImage::of(&e).with_frames(&e, &classes));
        let refs: Vec<&SweepImage> = imgs.iter().collect();
        let bp = Breakpoints::new(&refs, SweepScope::All);
        // 4 empty frames. <= 25 per 100: one may alert; recall 1 needs t <= .4, where only
        // the car (.6) alerts above .3: 0.31 ..= 0.40.
        let o: Objective = "recall_at_fa:25".parse().unwrap();
        assert_eq!(o.to_string(), "recall_at_fa:25");
        let p = exact_pick(&bp, o).unwrap();
        assert!(p.met);
        assert_eq!(p.threshold, 0.35, "{p:?}");
        assert_eq!(p.point.recall, Some(1.0));
        // <= 10 per 100: none may alert: 0.61 ..= 0.90 (recall .5).
        let p = exact_pick(&bp, "recall_at_fa:10".parse().unwrap()).unwrap();
        assert_eq!((p.threshold, p.point.recall), (0.75, Some(0.5)));
        for bad in [
            "recall_at_fa:0",
            "recall_at_fa:101",
            "recall_at_fa:x",
            "recall_at_fa:",
        ] {
            assert!(bad.parse::<Objective>().is_err(), "{bad}");
        }
        let p = exact_pick(&bp, o).unwrap();
        assert_eq!(
            p.point.false_alerts,
            Some(super::super::metrics::FalseAlerts {
                empty_frames: 4,
                alerts: 1
            })
        );
        assert_eq!(
            o.describe(),
            "recall at \u{2264} 25 false alerts per 100 empty frames"
        );
        assert!(o.is_alerts() && !o.is_frame());
        assert_eq!(serde_json::to_string(&o).unwrap(), "\"recall_at_fa:25\"");
        assert_eq!(
            "recall_at_fa:2.5".parse::<Objective>().unwrap(),
            Objective::RecallAtFa(2.5)
        );
        // Unreachable (a detection at confidence 1 on an empty frame): the fewest false
        // alerts (1 of 5 from 0.61 up), then the highest recall (.5 up to 0.90), unmet.
        let mut loud = imgs.clone();
        loud.push(frame_images(&[(&[], &[("car", 1.0)])]).remove(0));
        let refs: Vec<&SweepImage> = loud.iter().collect();
        let bp = Breakpoints::new(&refs, SweepScope::All);
        let p = exact_pick(&bp, "recall_at_fa:10".parse().unwrap()).unwrap();
        assert!(!p.met);
        assert_eq!((p.threshold, p.point.recall), (0.9, Some(0.5)));
        assert!(p.note.as_deref().unwrap().contains("not reached"), "{p:?}");
        // No empty frame: undefined.
        let busy: Vec<&SweepImage> = imgs.iter().take(2).collect();
        assert!(exact_pick(&Breakpoints::new(&busy, SweepScope::All), o).is_none());
    }

    /// B10: a pick fitted on pseudo ground truth (another model's detections) is never
    /// written into the config.
    #[test]
    fn picks_fitted_on_pseudo_labels_are_not_applied() {
        let mut a = test_advice(0.3);
        assert_eq!(a.apply_value(), Some(0.3));
        assert_eq!(a.not_on_labels(), None);
        a.relative = true;
        assert_eq!(a.apply_value(), None);
        assert_eq!(a.not_on_labels(), Some(NOT_ON_LABELS));
        // A device row's pick from a curve of pseudo ground truth.
        let curve = |relative: bool| Curve {
            key: "overall".into(),
            images: 2,
            gt: 5,
            relative,
            points: a.curve.clone(),
            configured: a.configured,
            picks: vec![a.best.clone().unwrap()],
            exact: vec![],
            roc_auc: None,
        };
        let real = curve(false).best(Objective::F1).unwrap();
        let pseudo = curve(true).best(Objective::F1).unwrap();
        assert!(!real.relative && pseudo.relative);
        let mut b = test_advice(0.3);
        b.by_device = vec![
            DevicePick {
                device: "openvino:cpu".into(),
                best: Some(real),
            },
            DevicePick {
                device: "ort:coreml".into(),
                best: Some(pseudo.clone()),
            },
        ];
        assert_eq!(b.apply_value_on("openvino:cpu"), Some(0.3));
        assert_eq!(b.apply_value_on("ort:coreml"), None);
        b.best = Some(pseudo);
        assert_eq!(b.apply_value(), None);
        assert_eq!(b.not_on_labels(), Some(NOT_ON_LABELS));
    }
}
