//! The saved results seen through the config now.
//!
//! `benchmark.json` records each model's threshold advice at the confidence threshold the run
//! was configured with ([`ThresholdAdvice::configured`]) and the device the config resolved to
//! then ([`ModelResult::configured`]). After "Apply threshold" or a device change those are
//! history: the Benchmark page, `/v1/benchmark` and the exported reports compare the best
//! threshold with the one configured *now*. [`rebase`] fills [`ThresholdAdvice::now`] (and each
//! group's `now`) at the model's current effective threshold, rewrites the threshold part of
//! the recommendation text, and sets [`ModelResult::configured`] to the device the config
//! resolves to now. Nothing rebased is saved.
//!
//! The counts at the new threshold are exact: from the stored predictions
//! ([`super::search::points_at`], checked against the run's own counts at its threshold, so
//! predictions of another run are never mixed in), else from a point the results store at
//! exactly that threshold (0.05 grid, picks), else left unknown ([`NowSource::None`]).

use super::report::{BenchmarkResults, ImageSetInfo, ModelResult};
use super::search::{StoredPreds, points_at};
use super::threshold::{NowSource, NowThreshold, ThresholdAdvice};
use crate::config::{Config, ModelConfig};
use crate::registry::normalize_name;
use std::collections::{BTreeMap, HashMap};

/// What the config says about a model now.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelNow {
    /// Confidence threshold of a device without a profile (the model's own, else the global
    /// one).
    pub threshold: f32,
    /// What the loaded model runs with, when known (differs while a restart is pending).
    pub running: Option<f32>,
    /// Canonical spec of the device the config resolves to ([`super::configured_device`]).
    pub device: Option<String>,
    /// The model's confidence profiles ([`ModelConfig::device_thresholds`]): an advice is
    /// compared with the profile of its own device ([`ModelNow::threshold_for`]).
    pub device_thresholds: BTreeMap<String, f32>,
    /// The fingerprint of its serving settings now ([`super::variant::served_fingerprint`]):
    /// an As served result with another one is stale ([`ModelResult::stale_rules`]).
    pub served: Option<String>,
}

impl ModelNow {
    /// No profiles: every device uses `threshold`.
    pub fn new(threshold: f32, running: Option<f32>, device: Option<String>) -> Self {
        Self {
            threshold,
            running,
            device,
            device_thresholds: BTreeMap::new(),
            served: None,
        }
    }

    /// With the confidence profiles `map` ([`ModelConfig::device_thresholds`]).
    #[must_use]
    pub fn with_profiles(mut self, map: BTreeMap<String, f32>) -> Self {
        self.device_thresholds = map;
        self
    }

    /// The threshold the config gives the model on `device` (a spec): its profile for that
    /// device ([`crate::config::lookup_device_threshold`]), else [`ModelNow::threshold`].
    pub fn threshold_for(&self, device: &str) -> f32 {
        crate::config::lookup_device_threshold(&self.device_thresholds, device)
            .unwrap_or(self.threshold)
    }
}

/// Normalized model name -> [`ModelNow`] of every configured model. `device` resolves a model's
/// configured device (usually [`super::configured_device`] with the model's [`crate::backend::Selection`]);
/// `running` gives the threshold a loaded model runs with.
pub fn from_config(
    cfg: &Config,
    device: &dyn Fn(&ModelConfig) -> Option<String>,
    running: &dyn Fn(&str) -> Option<f32>,
) -> HashMap<String, ModelNow> {
    cfg.models
        .iter()
        .map(|m| {
            let name = m.effective_name();
            (
                normalize_name(&name),
                ModelNow {
                    threshold: m.confidence_threshold.unwrap_or(cfg.confidence_threshold),
                    running: running(&name),
                    device: device(m),
                    device_thresholds: m.device_thresholds.clone(),
                    served: Some(super::variant::served_fingerprint(cfg, m)),
                },
            )
        })
        .collect()
}

/// Rebase every model of `results` that the config has (see the module docs); an As served
/// result (`<model>+served`) is rebased on its model and marked [`ModelResult::stale_rules`]
/// when the model's serving settings changed since its run.
pub fn rebase(
    results: &mut BenchmarkResults,
    preds: Option<&StoredPreds>,
    now: &HashMap<String, ModelNow>,
) {
    let BenchmarkResults {
        models, image_sets, ..
    } = results;
    for m in models {
        let served = super::variant::is_served(&m.model);
        let key = normalize_name(super::variant::base_name(&m.model));
        if let Some(n) = now.get(&key) {
            if served {
                m.stale_rules = Some(
                    m.served.as_ref().map(|s| &s.fingerprint) != n.served.as_ref()
                        || m.served.is_none(),
                );
            }
            rebase_model(m, image_sets, preds, n);
        }
    }
}

/// Rebase one model: `sets` (the results' image sets, with ground truth) and `preds` give the
/// exact counts at a threshold the run did not use.
pub fn rebase_model(
    m: &mut ModelResult,
    sets: &[ImageSetInfo],
    preds: Option<&StoredPreds>,
    n: &ModelNow,
) {
    if let Some(d) = &n.device {
        m.configured = Some(d.clone());
    }
    if let Some(a) = m.threshold.as_mut() {
        m.recommendation = rebase_advice(a, &m.model, &m.recommendation, sets, preds, n);
    }
}

/// Fill [`ThresholdAdvice::now`] of `model`'s advice for `n`, at the threshold the config gives
/// the advice's device ([`ModelNow::threshold_for`] of [`ThresholdAdvice::device`]: its
/// confidence profile, else the model's / global threshold), since the curves are that
/// device's; returns `recommendation` (which ends with the advice's summary, see
/// `ModelResult::finish`) with that summary rewritten.
pub fn rebase_advice(
    a: &mut ThresholdAdvice,
    model: &str,
    recommendation: &str,
    sets: &[ImageSetInfo],
    preds: Option<&StoredPreds>,
    n: &ModelNow,
) -> String {
    let old = a.summary();
    let t = n.threshold_for(&a.device);
    let running = n.running.filter(|r| (r - t).abs() > 1e-6);
    a.now = None;
    if (t - a.configured.threshold).abs() < 1e-6 {
        a.now = Some(NowThreshold {
            threshold: t,
            running,
            point: Some(a.configured),
            source: NowSource::Run,
        });
        for g in a
            .by_dataset
            .iter_mut()
            .chain(&mut a.by_tag)
            .chain(&mut a.per_class)
        {
            g.now = Some(g.configured);
        }
    } else {
        let exact = preds
            .and_then(|p| p.runs_of(model).into_iter().find(|r| r.device == a.device))
            .and_then(|r| points_at(sets, r, t))
            .filter(|p| {
                p.at_run.counts == a.configured.counts
                    && (p.at_run.threshold - a.configured.threshold).abs() < 1e-6
            });
        let (point, source) = match &exact {
            Some(p) => (Some(p.overall), NowSource::Preds),
            None => match a.known_point_at(t) {
                Some(p) => (Some(p), NowSource::Curve),
                None => (None, NowSource::None),
            },
        };
        for (groups, found) in [
            (&mut a.by_dataset, exact.as_ref().map(|p| &p.by_dataset)),
            (&mut a.by_tag, exact.as_ref().map(|p| &p.by_tag)),
            (&mut a.per_class, exact.as_ref().map(|p| &p.per_class)),
        ] {
            for g in groups {
                g.now = found.and_then(|f| f.get(&g.key)).copied();
            }
        }
        a.now = Some(NowThreshold {
            threshold: t,
            running,
            point,
            source,
        });
    }
    match (old, a.summary()) {
        (Some(old), Some(new)) => match recommendation.strip_suffix(&old) {
            Some(head) => format!("{head}{new}"),
            None => recommendation.to_string(),
        },
        _ => recommendation.to_string(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::benchmark::search::{SearchRequest, search};
    use crate::benchmark::threshold::{Objective, ThresholdAdvice};

    /// The search fixture with model `m`'s threshold advice on its recommended device built as
    /// a benchmark run builds it (configured 0.50, best 0.35 for F1) and a recommendation that
    /// ends with the advice summary.
    pub(crate) fn advised_fixture() -> (BenchmarkResults, StoredPreds) {
        let (mut results, preds) = crate::benchmark::search::tests::fixture();
        let r = search(&results, &preds, &SearchRequest::default(), &|_| None)
            .unwrap()
            .results
            .remove(0);
        let a = ThresholdAdvice {
            objective: Objective::F1.to_string(),
            device: r.device.clone(),
            relative: r.relative,
            configured: r.configured,
            best: r.best.clone(),
            alternatives: r.alternatives.clone(),
            curve: r.grid.clone(),
            exact: r.exact.clone(),
            by_dataset: r
                .by_dataset
                .iter()
                .map(|g| crate::benchmark::threshold::GroupPick {
                    now: None,
                    ..g.clone()
                })
                .collect(),
            by_tag: vec![],
            per_class: r
                .per_class
                .iter()
                .map(|g| crate::benchmark::threshold::GroupPick {
                    now: None,
                    ..g.clone()
                })
                .collect(),
            by_device: vec![],
            roc: r.roc.clone(),
            now: None,
        };
        let m = &mut results.models[0];
        m.recommendation = format!(
            "fastest agreeing device; {}",
            a.summary().expect("best threshold")
        );
        m.threshold = Some(a);
        (results, preds)
    }

    fn now(t: f32) -> ModelNow {
        ModelNow::new(t, None, None)
    }

    #[test]
    fn rebase_evaluates_the_configured_threshold_now_exactly() {
        let (results, preds) = advised_fixture();
        // Unchanged config: the run's own counts.
        let mut r = results.clone();
        rebase_model(
            &mut r.models[0],
            &results.image_sets,
            Some(&preds),
            &now(0.5),
        );
        let a = r.models[0].threshold.as_ref().unwrap();
        let n = a.now.as_ref().unwrap();
        assert_eq!((n.source, n.threshold), (NowSource::Run, 0.5));
        assert_eq!(n.point, Some(a.configured));
        assert!(a.by_dataset.iter().all(|g| g.now == Some(g.configured)));
        assert_eq!(r.models[0].recommendation, results.models[0].recommendation);

        // After "Apply threshold" (0.35, the best): exact from the stored predictions, equal
        // to the search's evaluation at the threshold configured now.
        let s = search(&results, &preds, &SearchRequest::default(), &|_| Some(0.35))
            .unwrap()
            .results
            .remove(0);
        let mut r = results.clone();
        let n35 = ModelNow {
            running: Some(0.5),
            ..now(0.35)
        };
        rebase_model(&mut r.models[0], &results.image_sets, Some(&preds), &n35);
        let a = r.models[0].threshold.as_ref().unwrap();
        let n = a.now.as_ref().unwrap();
        assert_eq!(
            (n.source, n.threshold, n.running),
            (NowSource::Preds, 0.35, Some(0.5))
        );
        assert_eq!(n.point, Some(s.now));
        // At 0.35 every object is found and the FP (.3) is filtered: P = R = F1 = 1.
        let p = n.point.unwrap();
        assert_eq!((p.counts.tp, p.counts.fp, p.counts.fn_), (3, 0, 0));
        assert_eq!(p.f1, Some(1.0));
        assert_eq!(a.now_point(), Some(&p));
        for (g, sg) in a.by_dataset.iter().zip(&s.by_dataset) {
            assert_eq!(g.now, sg.now, "{}", g.key);
        }
        for (g, sg) in a.per_class.iter().zip(&s.per_class) {
            assert_eq!(g.now, sg.now, "{}", g.key);
        }
        // The recommendation now compares with the configured 0.35 (= the best).
        let rec = &r.models[0].recommendation;
        assert!(rec.starts_with("fastest agreeing device; "), "{rec}");
        assert!(rec.contains("(the configured threshold)"), "{rec}");
        assert!(rec.contains("running 0.50 until a restart"), "{rec}");
        assert!(!rec.contains("at the configured 0.50"), "{rec}");

        // Without stored predictions: a stored grid point is exact (0.35 is on the grid) ...
        let mut r = results.clone();
        rebase_model(&mut r.models[0], &results.image_sets, None, &now(0.35));
        let n = r.models[0].threshold.as_ref().unwrap().now.clone().unwrap();
        assert_eq!(
            (n.source, n.point.unwrap().counts),
            (NowSource::Curve, p.counts)
        );
        assert!(
            r.models[0]
                .threshold
                .as_ref()
                .unwrap()
                .by_dataset
                .iter()
                .all(|g| g.now.is_none())
        );
        // ... an off-grid threshold is not evaluated (never interpolated).
        let mut r = results.clone();
        rebase_model(&mut r.models[0], &results.image_sets, None, &now(0.37));
        let a = r.models[0].threshold.as_ref().unwrap();
        let n = a.now.as_ref().unwrap();
        assert_eq!((n.source, n.point), (NowSource::None, None));
        assert_eq!(a.now_point(), None);
        assert!(
            r.models[0]
                .recommendation
                .contains("(configured 0.37, not evaluated)"),
            "{}",
            r.models[0].recommendation
        );
        // An off-grid threshold with predictions: exact.
        let mut r = results.clone();
        rebase_model(
            &mut r.models[0],
            &results.image_sets,
            Some(&preds),
            &now(0.42),
        );
        let n = r.models[0].threshold.as_ref().unwrap().now.clone().unwrap();
        assert_eq!(n.source, NowSource::Preds);
        // .85 and .6 TP, .4 car below: 2 TP, 0 FP, 1 FN.
        let c = n.point.unwrap().counts;
        assert_eq!((c.tp, c.fp, c.fn_), (2, 0, 1));

        // Stored predictions of another run (counts at the run's threshold differ) are not
        // mixed in.
        let mut other = preds.clone();
        for run in &mut other.runs {
            for img in &mut run.images {
                img.preds.clear();
            }
        }
        let mut r = results.clone();
        rebase_model(
            &mut r.models[0],
            &results.image_sets,
            Some(&other),
            &now(0.37),
        );
        let n = r.models[0].threshold.as_ref().unwrap().now.clone().unwrap();
        assert_eq!(n.source, NowSource::None);
    }

    #[test]
    fn rebase_from_config_sets_device_and_threshold() {
        let (mut results, preds) = advised_fixture();
        // What is saved never carries `now`.
        let text = serde_json::to_string(&results).unwrap();
        assert!(!text.contains("\"now\""));
        let mut cfg = Config {
            confidence_threshold: 0.5,
            ..Default::default()
        };
        cfg.models.push(ModelConfig {
            name: Some("M".into()),
            path: "m.onnx".into(),
            confidence_threshold: Some(0.35),
            ..Default::default()
        });
        let map = from_config(&cfg, &|_| Some("openvino:cpu".into()), &|_| Some(0.5));
        rebase(&mut results, Some(&preds), &map);
        let m = &results.models[0];
        assert_eq!(m.configured.as_deref(), Some("openvino:cpu"));
        let n = m.threshold.as_ref().unwrap().now.as_ref().unwrap();
        assert_eq!((n.threshold, n.running), (0.35, Some(0.5)));
        // Served: `now` on the advice and its groups.
        let v = serde_json::to_value(m).unwrap();
        assert_eq!(v["threshold"]["now"]["source"], "preds");
        assert!(v["threshold"]["by_dataset"][0]["now"]["f1"].is_number());
    }
    #[test]
    fn rebase_uses_the_profile_of_the_advice_device() {
        let (results, preds) = advised_fixture();
        let device = results.models[0].threshold.as_ref().unwrap().device.clone();
        assert_eq!(device, "ort:coreml");
        let at = |n: &ModelNow| {
            let mut r = results.clone();
            rebase_model(&mut r.models[0], &results.image_sets, Some(&preds), n);
            r.models[0].threshold.clone().unwrap().now.unwrap()
        };
        // A profile for the advice's device (the curves' device) replaces the model threshold.
        let profiled = now(0.5).with_profiles([("ort:coreml".to_string(), 0.35)].into());
        assert_eq!(profiled.threshold_for("ort:coreml"), 0.35);
        assert_eq!(profiled.threshold_for("openvino:cpu"), 0.5);
        let n = at(&profiled);
        assert_eq!((n.threshold, n.source), (0.35, NowSource::Preds));
        let p = n.point.unwrap().counts;
        assert_eq!((p.tp, p.fp, p.fn_), (3, 0, 0));
        // A profile of another device does not: the model's 0.50, the run's own counts.
        let other = now(0.5).with_profiles([("openvino:cpu".to_string(), 0.35)].into());
        let n = at(&other);
        assert_eq!((n.threshold, n.source), (0.5, NowSource::Run));

        // from_config carries the profiles (any spelling the config canonicalized).
        let mut cfg = Config::default();
        cfg.models.push(ModelConfig {
            name: Some("M".into()),
            path: "m.onnx".into(),
            device_thresholds: [("ort:coreml".to_string(), 0.35)].into(),
            ..Default::default()
        });
        let map = from_config(&cfg, &|_| None, &|_| None);
        let mut r = results.clone();
        rebase(&mut r, Some(&preds), &map);
        let n = r.models[0].threshold.as_ref().unwrap().now.clone().unwrap();
        assert_eq!(n.threshold, 0.35);
        assert!(
            r.models[0]
                .recommendation
                .contains("(the configured threshold)"),
            "{}",
            r.models[0].recommendation
        );
    }
}

#[cfg(test)]
mod served_tests {
    use super::*;
    use crate::benchmark::report::ServedSettings;
    use crate::benchmark::report::test_support::{dev, model, scene};
    use crate::benchmark::variant::{base_name, is_served, served_fingerprint, served_name};

    fn cfg(dir: &std::path::Path) -> Config {
        let entry = |n: &str| {
            std::fs::write(dir.join(format!("{n}.onnx")), b"x").unwrap();
            ModelConfig {
                name: Some(n.into()),
                path: dir.join(format!("{n}.onnx")),
                ..Default::default()
            }
        };
        let mut m = entry("m");
        m.night_model = Some("night".into());
        Config {
            models: vec![m, entry("night")],
            ..Config::default()
        }
    }

    fn result(name: &str, p50: f64) -> ModelResult {
        let mut r = model(vec![dev("openvino:cpu", p50, scene())], None);
        r.model = name.into();
        r
    }

    /// As served results live beside the bare ones under their own name: a new bare run
    /// replaces only the bare result, the bare result is what `find` (the `auto` order, Apply)
    /// reads, and the names map back to the model.
    #[test]
    fn served_and_bare_results_are_kept_apart() {
        assert_eq!(served_name("IPcam-general"), "IPcam-general+served");
        assert_eq!(base_name("IPcam-general+served"), "IPcam-general");
        assert_eq!(base_name("IPcam-general+SERVED"), "IPcam-general");
        assert_eq!(
            base_name("IPcam-general+pre:clahe"),
            "IPcam-general+pre:clahe"
        );
        assert_eq!(base_name("+served"), "+served");
        assert!(is_served("m+served") && !is_served("m") && !is_served("m+filters"));
        let hw = crate::benchmark::report::HardwareSummary::default();
        let saved = BenchmarkResults::new(
            hw.clone(),
            Default::default(),
            vec![result("m", 10.0), result("m+served", 30.0)],
        );
        let again = BenchmarkResults::new(hw, Default::default(), vec![result("m", 12.0)]);
        let merged = BenchmarkResults::merge(Some(saved), again);
        assert_eq!(merged.models.len(), 2);
        let p50 = |n: &str| {
            merged.find(n).unwrap().devices[0]
                .run
                .as_ref()
                .unwrap()
                .stages_ms
                .total
                .p50
        };
        assert_eq!(p50("m"), 12.0);
        assert_eq!(p50("m+served"), 30.0);
    }

    /// `GET /v1/benchmark` marks an As served result stale when the model's serving settings
    /// changed since its run (its filter rules, its night model's settings), not when only its
    /// own day threshold did (what the sweep varies); a bare result never carries the flag.
    #[test]
    fn a_changed_serving_rule_marks_the_served_result_stale() {
        let dir = std::env::temp_dir().join(format!("bop-stale-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = cfg(&dir);
        let fp = served_fingerprint(&config, &config.models[0]);
        let saved = {
            let mut s = result("m+served", 30.0);
            s.served = Some(ServedSettings {
                fingerprint: fp.clone(),
                rules: String::new(),
            });
            BenchmarkResults::new(
                Default::default(),
                Default::default(),
                vec![result("m", 10.0), s],
            )
        };
        let stale = |c: &Config| {
            let mut r = saved.clone();
            rebase(&mut r, None, &from_config(c, &|_| None, &|_| None));
            (r.models[0].stale_rules, r.models[1].stale_rules)
        };
        assert_eq!(stale(&config), (None, Some(false)));
        // A new day threshold or device profile: not stale.
        let mut c = config.clone();
        c.models[0].confidence_threshold = Some(0.7);
        c.models[0].device_thresholds = [("openvino:cpu".to_string(), 0.6)].into();
        assert_eq!(stale(&c), (None, Some(false)));
        // A filter rule, the night threshold, enhancement, the night model's settings: stale.
        for edit in [
            (|c: &mut Config| {
                c.models[0].label_thresholds = Some([("person".to_string(), 0.6)].into())
            }) as fn(&mut Config),
            |c| c.models[0].night_threshold = Some(0.3),
            |c| c.models[0].preprocess = Some(crate::model::enhance::Preprocess::CLAHE),
            |c| c.models[1].confidence_threshold = Some(0.2),
            |c| c.models[1].enabled = false,
            |c| c.night_confidence_threshold = Some(0.5),
            |c| c.models[0].night_model = None,
        ] {
            let mut c = config.clone();
            edit(&mut c);
            assert_eq!(stale(&c), (None, Some(true)));
        }
        // The answer carries it; the saved file never does.
        let mut r = saved.clone();
        let mut c = config.clone();
        c.models[0].night_threshold = Some(0.3);
        rebase(&mut r, None, &from_config(&c, &|_| None, &|_| None));
        let v = serde_json::to_value(&r.models[1]).unwrap();
        assert_eq!(v["stale_rules"], true);
        let back: ModelResult = serde_json::from_value(v).unwrap();
        assert_eq!(back.stale_rules, None);
        assert!(
            !serde_json::to_string(&saved)
                .unwrap()
                .contains("stale_rules")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
