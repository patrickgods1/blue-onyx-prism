//! The confidence pick on the 0.01 grid with targets judged on their 95% Wilson bound.
//!
//! [`value`] is what an objective maximizes (F-beta; recall for `precision:p` and
//! `recall_at_fa:N`; precision for `recall:r`; frame TPR for `fpr:x`; Youden's J). [`bound`] is
//! the bound a target is judged on ([`super::stats::wilson`]): precision's lower bound for
//! `precision:p`, recall's lower bound for `recall:r`, the upper bound of alerts / empty frames
//! x 100 for `recall_at_fa:N`, the frame FPR's upper bound for `fpr:x`. A grid value meets the
//! target when its bound does ([`met`]); [`score`] is the value there and None elsewhere.
//!
//! [`pick`] applies the plateau rule of [`crate::benchmark::threshold::grid_run`] (values
//! within [`PLATEAU`] of the best, the middle of the widest run, ties to the higher run) to the
//! scores. When no grid value meets a target, the value closest to it on its bound is picked
//! (the most precise by the lower bound, ...; then the maximized metric; then the higher
//! threshold) with `met: false`: shown, never applied. A curve on which every grid value scores
//! within the plateau is flat ([`GridPick::flat`]): the pick says nothing about the confidence.

use super::stats::{lower, upper};
use super::tally::At;
use crate::benchmark::threshold::{Objective, PLATEAU, grid_run};

/// Scores this close count as equal.
const TIE: f64 = 1e-12;

/// The value `objective` maximizes at `a` (whether its target is met or not).
pub fn value(objective: Objective, a: &At) -> Option<f64> {
    match objective {
        Objective::F1 => a.f_beta(1.0),
        Objective::F2 => a.f_beta(2.0),
        Objective::F05 => a.f_beta(0.5),
        Objective::Precision(_) | Objective::RecallAtFa(_) => a.recall(),
        Objective::Recall(_) => a.precision(),
        Objective::Youden => a.youden(),
        Objective::Fpr(_) => a.tpr(),
    }
}

/// The bound a target is judged on at `a` (see the module docs); None without a target or
/// without the counts it needs. FA/100 is in false alerts per 100 empty frames.
pub fn bound(objective: Objective, a: &At) -> Option<f64> {
    match objective {
        Objective::Precision(_) => lower(a.tp, a.tp + a.fp),
        Objective::Recall(_) => lower(a.tp, a.tp + a.fn_),
        Objective::RecallAtFa(_) => upper(a.empty_alerts, a.empty).map(|u| 100.0 * u),
        Objective::Fpr(_) => upper(a.neg_alerts, a.neg),
        _ => None,
    }
}

/// The target of `objective` (None for F1, F2, F0.5, Youden).
pub fn target(objective: Objective) -> Option<f64> {
    match objective {
        Objective::Precision(t)
        | Objective::Recall(t)
        | Objective::RecallAtFa(t)
        | Objective::Fpr(t) => Some(t),
        _ => None,
    }
}

/// Whether the target is met at `a` on its bound; None without a target.
pub fn met(objective: Objective, a: &At) -> Option<bool> {
    let t = target(objective)?;
    let b = bound(objective, a);
    Some(match objective {
        Objective::Precision(_) | Objective::Recall(_) => b.is_some_and(|b| b >= t - TIE),
        _ => b.is_some_and(|b| b <= t + TIE),
    })
}

/// The objective's score at `a`: [`value`] where the target is met (or there is none).
pub fn score(objective: Objective, a: &At) -> Option<f64> {
    if met(objective, a) == Some(false) {
        return None;
    }
    value(objective, a)
}

/// Enough to pick at all: objects for the box objectives, positive and negative frames for the
/// frame ones, empty frames for `recall_at_fa`.
pub fn pickable(objective: Objective, a: &At) -> bool {
    match objective {
        Objective::Youden | Objective::Fpr(_) => a.pos > 0 && a.neg > 0,
        Objective::RecallAtFa(_) => a.objects() > 0 && a.empty > 0,
        _ => a.objects() > 0,
    }
}

/// A pick on the grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPick {
    /// The picked grid index.
    pub index: usize,
    /// The best-scoring index (ties: the highest).
    pub best: usize,
    /// The plateau `[first, last]` the pick is the middle of (`[index, index]` for a fallback).
    pub first: usize,
    pub last: usize,
    /// The target is met there on its bound (true without a target).
    pub met: bool,
    /// Every grid value scores within the plateau: the curve is flat.
    pub flat: bool,
}

/// The pick for `objective` over `points` (one per grid value, ascending; see the module docs).
/// None when the points cannot be picked on ([`pickable`]).
pub fn pick(objective: Objective, points: &[At]) -> Option<GridPick> {
    if !points.first().is_some_and(|a| pickable(objective, a)) {
        return None;
    }
    let scores: Vec<Option<f64>> = points.iter().map(|a| score(objective, a)).collect();
    if let Some(run) = grid_run(&scores, PLATEAU) {
        return Some(GridPick {
            index: run.middle(),
            best: run.best,
            first: run.first,
            last: run.last,
            met: true,
            flat: run.first == 0 && run.last + 1 == points.len(),
        });
    }
    // No grid value meets the target: the closest on its bound, then the maximized value, then
    // the higher threshold.
    let raises = matches!(objective, Objective::Precision(_) | Objective::Recall(_));
    let closeness = |a: &At| bound(objective, a).map(|b| if raises { b } else { -b });
    let mut best: Option<(usize, f64, f64)> = None;
    for (i, a) in points.iter().enumerate() {
        let Some(c) = closeness(a) else { continue };
        let v = value(objective, a).unwrap_or(0.0);
        let better =
            best.is_none_or(|(_, bc, bv)| c > bc + TIE || ((c - bc).abs() <= TIE && v >= bv - TIE));
        if better {
            best = Some((i, c, v));
        }
    }
    let (i, _, _) = best?;
    Some(GridPick {
        index: i,
        best: i,
        first: i,
        last: i,
        met: false,
        flat: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(tp: usize, fp: usize, fn_: usize) -> At {
        At {
            tp,
            fp,
            fn_,
            ..At::default()
        }
    }

    #[test]
    fn targets_are_judged_on_the_wilson_bound() {
        let o = Objective::Precision(0.9);
        // 95 / 100 precise: the lower bound is 0.888, below 0.9.
        let a = at(95, 5, 10);
        assert!(bound(o, &a).unwrap() < 0.9);
        assert_eq!(met(o, &a), Some(false));
        assert_eq!(score(o, &a), None);
        // 970 / 1000: the bound clears it.
        let a = at(970, 30, 100);
        assert_eq!(met(o, &a), Some(true));
        assert_eq!(score(o, &a), a.recall());
        let fa = Objective::RecallAtFa(5.0);
        let a = At {
            tp: 5,
            fn_: 5,
            empty: 100,
            empty_alerts: 1,
            ..At::default()
        };
        // 1 / 100 alerts: the upper bound is 5.4 per 100, above 5.
        assert!(bound(fa, &a).unwrap() > 5.0);
        assert_eq!(met(fa, &a), Some(false));
        assert_eq!(met(Objective::F1, &a), None);
        assert_eq!(score(Objective::F2, &a), a.f_beta(2.0));
    }

    #[test]
    fn plateau_pick_flat_curve_and_fallback() {
        // F1 peaks on indices 3..=5.
        let pts: Vec<At> = [1, 2, 5, 9, 9, 9, 4, 1]
            .iter()
            .map(|&tp| at(tp, 2, 10 - tp))
            .collect();
        let p = pick(Objective::F1, &pts).unwrap();
        assert_eq!(
            (p.first, p.last, p.index, p.met, p.flat),
            (3, 5, 4, true, false)
        );
        // Every value equal: flat, the middle of the grid.
        let flat: Vec<At> = (0..9).map(|_| at(5, 1, 5)).collect();
        let p = pick(Objective::F2, &flat).unwrap();
        assert!(p.flat);
        assert_eq!(p.index, 4);
        // Precision 0.99 is never reached on the bound: the most precise by its lower bound
        // (40/44: 0.79, above 10/10: 0.72 and 1/1: 0.21).
        let pts = vec![at(50, 50, 0), at(40, 4, 10), at(10, 0, 40), at(1, 0, 49)];
        let p = pick(Objective::Precision(0.99), &pts).unwrap();
        assert!(!p.met);
        assert_eq!(p.index, 1);
        assert_eq!(pick(Objective::F1, &[At::default()]), None);
    }
}
