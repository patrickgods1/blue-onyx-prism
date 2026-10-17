//! Re-rank saved results for new settings without a run: the page's "What matters most" saves
//! the threshold objective, accuracy weight and accuracy metric, and [`rescore`] re-grades every
//! stored run with them and recomputes each model's threshold advice for the new objective.
//!
//! The advice is exact when the stored predictions (`benchmark-preds.json`) have the device's
//! run: the exact pick is recomputed from them like the confidence search does
//! ([`super::search::exact_overall_pick`]) and added to the device's overall curve, so the
//! advice uses it. Without stored predictions (results of older versions) it falls back to the
//! 0.05 grid of the stored curve ([`super::threshold::Curve::best`]).

use super::BenchmarkResults;
use super::grade::{AccuracyMetric, Weights};
use super::search::StoredPreds;
use super::threshold::Objective;

/// What [`rescore`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rescored {
    /// The results changed (grades, recommendation or advice).
    pub changed: bool,
    /// Device curves that got an exact pick from the stored predictions.
    pub exact: usize,
    /// Device curves with only the 0.05 grid for the new objective.
    pub grid: usize,
}

/// Re-grade `results` on `metric` with `weights` and recompute the threshold advice of every
/// model whose objective is not `objective` (see the module docs).
pub fn rescore(
    results: &mut BenchmarkResults,
    preds: Option<&StoredPreds>,
    metric: AccuracyMetric,
    weights: Weights,
    objective: Objective,
) -> Rescored {
    let before = results.clone();
    let mut out = Rescored::default();
    let key = objective.to_string();
    let sets = results.image_sets.clone();
    for m in &mut results.models {
        if m.threshold_objective == objective
            && m.threshold.as_ref().is_none_or(|a| a.objective == key)
        {
            continue;
        }
        let runs = preds.map(|p| p.runs_of(&m.model)).unwrap_or_default();
        for d in &mut m.devices {
            let device = d.device.clone();
            let ok = d.ok();
            let Some(sweep) = d
                .run
                .as_mut()
                .filter(|_| ok)
                .and_then(|r| r.accuracy.as_mut())
                .and_then(|a| a.sweep.as_mut())
            else {
                continue;
            };
            if sweep.overall.picks.iter().any(|p| p.objective == key) {
                out.exact += 1;
                continue;
            }
            let pick = runs
                .iter()
                .find(|r| r.device == device)
                .and_then(|r| super::search::exact_overall_pick(&sets, r, objective));
            match pick {
                Some(p) => {
                    sweep.overall.picks.push(p);
                    out.exact += 1;
                }
                None => out.grid += 1,
            }
        }
        m.threshold_objective = objective;
        m.finish();
    }
    results.regrade(metric, weights);
    out.changed = *results != before;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::report::test_support::{dev, scene, with_sweep};

    /// The search fixture's model with two scored devices (CoreML faster, recommended).
    fn fixture() -> (BenchmarkResults, StoredPreds) {
        let (mut results, preds) = crate::benchmark::search::tests::fixture();
        let curve = [(0.9, true), (0.6, true), (0.4, true), (0.3, false)];
        let cpu = with_sweep(dev("openvino:cpu", 50.0, scene()), 3, &curve);
        let coreml = with_sweep(dev("ort:coreml", 10.0, scene()), 3, &curve);
        let m = &mut results.models[0];
        m.datasets = vec!["night-set".into(), "pseudo-set".into()];
        m.devices = vec![cpu, coreml];
        m.finish();
        (results, preds)
    }

    /// A new objective gets an exact pick from the stored predictions (equal to the search's),
    /// a new metric re-grades, and the same settings change nothing.
    #[test]
    fn rescore_recomputes_advice_and_grades() {
        let (mut results, preds) = fixture();
        assert_eq!(results.models[0].recommended.as_deref(), Some("ort:coreml"));
        let metric = results.accuracy_metric();
        let weights = Weights::default();
        let same = rescore(&mut results, Some(&preds), metric, weights, Objective::F1);
        assert!(!same.changed, "{same:?}");

        // Precision >= 0.8 is not among the stored picks: computed from the predictions.
        let target = Objective::Precision(0.8);
        let r = rescore(&mut results, Some(&preds), metric, weights, target);
        assert!(r.changed, "{r:?}");
        assert_eq!((r.exact, r.grid), (2, 0), "{r:?}");
        let m = &results.models[0];
        assert_eq!(m.threshold_objective, target);
        let a = m.threshold.as_ref().unwrap();
        assert_eq!(a.objective, target.to_string());
        assert_eq!(a.device, "ort:coreml");
        let best = a.best.as_ref().unwrap();
        // Same as the search over the stored predictions of that device.
        let req = crate::benchmark::search::SearchRequest {
            models: vec![m.model.clone()],
            device: Some(a.device.clone()),
            objective: target,
            ..Default::default()
        };
        let out = crate::benchmark::search::search(&results, &preds, &req, &|_| None).unwrap();
        let searched = out.results[0].best.as_ref().unwrap();
        assert_eq!(best.threshold, searched.threshold);
        assert_eq!(best.exact_threshold, searched.exact_threshold);
        assert_eq!(best.plateau, searched.plateau);
        // Once added, the pick is reused (no new computation, nothing changes).
        let again = rescore(&mut results, Some(&preds), metric, weights, target);
        assert!(!again.changed);

        // Without stored predictions: the grid.
        let (mut plain, _) = fixture();
        let r = rescore(&mut plain, None, metric, weights, Objective::Precision(0.7));
        assert_eq!((r.exact, r.grid), (0, 2), "{r:?}");
        assert_eq!(
            plain.models[0].threshold.as_ref().unwrap().objective,
            "precision:0.7"
        );

        // Another accuracy weight re-grades (the grades change, the objective does not).
        let speed_first = Weights {
            accuracy: 0.0,
            speed: 1.0,
        };
        let before = results.models[0].devices[0].run.as_ref().unwrap().grades;
        let r = rescore(&mut results, Some(&preds), metric, speed_first, target);
        let after = results.models[0].devices[0].run.as_ref().unwrap().grades;
        assert_eq!(r.changed, before != after);
    }
}
