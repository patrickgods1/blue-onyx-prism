//! The confidence profiles view of `GET /v1/benchmark` (`profiles`, `goal`): per benchmarked
//! model and per device that ran, the measured speed and accuracy, the device's best confidence
//! threshold ([`crate::benchmark::threshold::ThresholdAdvice::apply_value_on`]) and the profile
//! the config holds for it ([`crate::config::ModelConfig::device_thresholds`]). The measured
//! part ([`ModelFacts`]) comes from the saved results and is cached with them; [`profiles`]
//! joins it with the config and the loaded devices on every request. Pure, no I/O.

use crate::benchmark::BenchmarkResults;
use crate::benchmark::threshold::Objective;
use crate::config::{BenchmarkConfig, Config, ModelConfig, ThresholdSource, thr};
use serde::Serialize;

/// What the saved results say about one device of a model (see [`ProfileDevice`]).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DeviceFacts {
    pub device: String,
    pub label: String,
    pub recommended: bool,
    pub ok: bool,
    pub error: Option<String>,
    pub fell_back: bool,
    pub ms: Option<f64>,
    pub finds: Option<f64>,
    pub correct: Option<f64>,
    pub best: Option<f32>,
}

/// The measured part of one model's profiles.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct ModelFacts {
    pub model: String,
    pub devices: Vec<DeviceFacts>,
}

/// The measured facts of every model of `results`, devices in the order they ran.
pub(super) fn facts(results: &BenchmarkResults) -> Vec<ModelFacts> {
    results
        .models
        .iter()
        .map(|m| ModelFacts {
            model: m.model.clone(),
            devices: m
                .devices
                .iter()
                .map(|d| {
                    let advice = m.threshold.as_ref();
                    // The device's own pick (exact spec first, then the same device).
                    let pick = advice.and_then(|a| {
                        a.by_device
                            .iter()
                            .find(|p| p.device == d.device)
                            .or_else(|| {
                                a.by_device
                                    .iter()
                                    .find(|p| crate::config::same_device(&p.device, &d.device))
                            })
                            .and_then(|p| p.best.as_ref())
                    });
                    let run = d.run.as_ref();
                    // The time per image the page shows: the graded p50, else the measured one.
                    let ms = run.and_then(|r| {
                        r.grades
                            .as_ref()
                            .map(|g| g.p50_ms)
                            .filter(|v| v.is_finite() && *v < 1e300)
                            .or(Some(r.stages_ms.total.p50))
                            .filter(|v| v.is_finite())
                    });
                    DeviceFacts {
                        device: d.device.clone(),
                        label: d.label.clone(),
                        recommended: m.recommended.as_deref() == Some(d.device.as_str()),
                        ok: d.ok(),
                        error: d.error.clone(),
                        fell_back: run.is_some_and(|r| r.fell_back),
                        ms,
                        finds: pick.and_then(|p| p.point.recall),
                        correct: pick.and_then(|p| p.point.precision),
                        best: advice.and_then(|a| a.apply_value_on(&d.device)),
                    }
                })
                .collect(),
        })
        .collect()
}

/// One device of a model in `profiles` (THE CONTRACT the Benchmark page codes against).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ProfileDevice {
    pub device: String,
    pub label: String,
    pub recommended: bool,
    /// The device the model runs on (loaded), else the one the config resolves to.
    pub now: bool,
    pub ok: bool,
    pub error: Option<String>,
    pub fell_back: bool,
    /// Time per image, p50 (ms).
    pub ms: Option<f64>,
    /// Recall at the device's best threshold (0..1).
    pub finds: Option<f64>,
    /// Precision at the device's best threshold (0..1).
    pub correct: Option<f64>,
    /// The device's best threshold when it may be applied.
    #[serde(serialize_with = "thr::ser_opt")]
    pub best: Option<f32>,
    /// The config's profile for this device.
    #[serde(serialize_with = "thr::ser_opt")]
    pub saved: Option<f32>,
    /// `best` is known and `saved` is not it.
    pub differs: bool,
}

/// One model in `profiles`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ProfileModel {
    pub model: String,
    /// The model is in the config.
    pub configured: bool,
    /// The model's configured `device` (a spec or `auto`), `global` when it follows the global
    /// device (or is not configured).
    pub device: String,
    /// Spec of the device the model is loaded on.
    pub running: Option<String>,
    /// The threshold of devices without a profile.
    #[serde(serialize_with = "thr::ser")]
    pub fallback: f32,
    /// `model` or `global`.
    pub fallback_source: &'static str,
    /// `saved`, `unsaved`, `differs` or `none` ([`status_of`]).
    pub status: &'static str,
    pub devices: Vec<ProfileDevice>,
}

/// `saved` when every device with a best threshold has it saved, `differs` when the config has
/// a profile for some of them but not every best is saved, `unsaved` when none of them has a
/// profile, `none` when no device has a best threshold.
pub(super) fn status_of(devices: &[ProfileDevice]) -> &'static str {
    let with_best: Vec<&ProfileDevice> = devices.iter().filter(|d| d.best.is_some()).collect();
    if with_best.is_empty() {
        "none"
    } else if with_best.iter().all(|d| !d.differs) {
        "saved"
    } else if with_best.iter().any(|d| d.saved.is_some()) {
        "differs"
    } else {
        "unsaved"
    }
}

/// Join the measured `facts` with the config: `running` gives a model's loaded device spec,
/// `planned` the device the config resolves a configured model to.
pub(super) fn profiles(
    facts: &[ModelFacts],
    cfg: &Config,
    running: &dyn Fn(&str) -> Option<String>,
    planned: &dyn Fn(&ModelConfig) -> Option<String>,
) -> Vec<ProfileModel> {
    facts
        .iter()
        .map(|f| {
            let mc = crate::benchmark::find_model(cfg, &f.model);
            let running = mc.and_then(|m| running(&m.effective_name()));
            let now = running.clone().or_else(|| mc.and_then(planned));
            let (fallback, source) = match mc {
                Some(m) => m.threshold_on(None, cfg.confidence_threshold),
                None => (cfg.confidence_threshold, ThresholdSource::Global),
            };
            let devices: Vec<ProfileDevice> = f
                .devices
                .iter()
                .map(|d| {
                    let saved = mc.and_then(|m| {
                        crate::config::lookup_device_threshold(&m.device_thresholds, &d.device)
                    });
                    ProfileDevice {
                        device: d.device.clone(),
                        label: d.label.clone(),
                        recommended: d.recommended,
                        now: now.as_deref().is_some_and(|n| {
                            n == d.device || crate::config::same_device(n, &d.device)
                        }),
                        ok: d.ok,
                        error: d.error.clone(),
                        fell_back: d.fell_back,
                        ms: d.ms,
                        finds: d.finds,
                        correct: d.correct,
                        best: d.best,
                        saved,
                        differs: d
                            .best
                            .is_some_and(|b| saved.is_none_or(|s| (s - b).abs() > 1e-6)),
                    }
                })
                .collect();
            ProfileModel {
                model: mc.map_or_else(|| f.model.clone(), ModelConfig::effective_name),
                configured: mc.is_some(),
                device: mc
                    .and_then(|m| m.device.clone())
                    .unwrap_or_else(|| "global".into()),
                running,
                fallback,
                fallback_source: match source {
                    ThresholdSource::Model => "model",
                    _ => "global",
                },
                status: status_of(&devices),
                devices,
            }
        })
        .collect()
}

/// The plain name of the benchmark goal (the Benchmark page's "What matters most" presets):
/// "Balanced", "Miss fewer objects", "Fewer false alerts", "Fastest / lightest", else "Custom".
pub(super) fn goal_label(b: &BenchmarkConfig) -> &'static str {
    let w = b.weights.accuracy_share();
    [
        (Objective::F1, 0.6, "Balanced"),
        (Objective::F2, 0.7, "Miss fewer objects"),
        (Objective::Precision(0.9), 0.5, "Fewer false alerts"),
        (Objective::F1, 0.2, "Fastest / lightest"),
    ]
    .into_iter()
    .find(|(o, pw, _)| *o == b.threshold_objective && (w - pw).abs() < 0.011)
    .map_or("Custom", |(_, _, l)| l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::grade::Weights;

    fn dev(device: &str, best: Option<f32>) -> DeviceFacts {
        DeviceFacts {
            device: device.into(),
            label: device.into(),
            recommended: false,
            ok: true,
            error: None,
            fell_back: false,
            ms: Some(10.0),
            finds: Some(0.8),
            correct: Some(0.9),
            best,
        }
    }

    #[test]
    fn statuses_and_now() {
        let facts = vec![ModelFacts {
            model: "M".into(),
            devices: vec![
                dev("ort:coreml", Some(0.3)),
                dev("openvino:cpu", Some(0.34)),
                dev("ort:cpu", None),
            ],
        }];
        let mut cfg = Config::default();
        cfg.models.push(ModelConfig {
            name: Some("m".into()),
            path: "m.onnx".into(),
            ..Default::default()
        });
        let run = |cfg: &Config, running: Option<&str>| {
            profiles(&facts, cfg, &|_| running.map(str::to_string), &|_| {
                Some("CPU".into())
            })
            .remove(0)
        };
        let p = run(&cfg, None);
        assert_eq!(
            (p.status, p.configured, p.model.as_str()),
            ("unsaved", true, "m")
        );
        assert_eq!((p.fallback, p.fallback_source), (0.5, "global"));
        assert_eq!(p.device, "global");
        // Planned: the config resolves to CPU (= openvino:cpu).
        assert_eq!(
            p.devices.iter().map(|d| d.now).collect::<Vec<_>>(),
            [false, true, false]
        );
        assert!(p.devices[0].differs && !p.devices[2].differs);
        // Loaded on CoreML: that one is now.
        let p = run(&cfg, Some("ort:coreml"));
        assert_eq!(p.running.as_deref(), Some("ort:coreml"));
        assert!(p.devices[0].now && !p.devices[1].now);

        cfg.models[0].confidence_threshold = Some(0.4);
        cfg.models[0].device = Some("auto".into());
        cfg.models[0].device_thresholds = [("ort:coreml".to_string(), 0.3)].into();
        let p = run(&cfg, None);
        assert_eq!((p.fallback, p.fallback_source), (0.4, "model"));
        assert_eq!(p.device, "auto");
        assert_eq!(p.status, "differs");
        assert_eq!(p.devices[0].saved, Some(0.3));
        assert!(!p.devices[0].differs);
        cfg.models[0]
            .device_thresholds
            .insert("openvino:cpu".into(), 0.34);
        assert_eq!(run(&cfg, None).status, "saved");
        cfg.models[0]
            .device_thresholds
            .insert("openvino:cpu".into(), 0.5);
        assert_eq!(run(&cfg, None).status, "differs");

        // No best anywhere: none. Not configured: listed, global fallback.
        let facts = vec![ModelFacts {
            model: "gone".into(),
            devices: vec![dev("ort:cpu", None)],
        }];
        let p = profiles(&facts, &cfg, &|_| None, &|_| None).remove(0);
        assert_eq!((p.status, p.configured), ("none", false));
        assert_eq!((p.fallback, p.fallback_source), (0.5, "global"));
        assert!(!p.devices[0].now);
    }

    #[test]
    fn goal_labels_follow_the_presets() {
        let b = |o: Objective, a: f64| BenchmarkConfig {
            threshold_objective: o,
            weights: Weights {
                accuracy: a,
                speed: 1.0 - a,
            },
            ..Default::default()
        };
        assert_eq!(goal_label(&b(Objective::F1, 0.6)), "Balanced");
        assert_eq!(goal_label(&b(Objective::F2, 0.7)), "Miss fewer objects");
        assert_eq!(
            goal_label(&b(Objective::Precision(0.9), 0.5)),
            "Fewer false alerts"
        );
        assert_eq!(goal_label(&b(Objective::F1, 0.2)), "Fastest / lightest");
        assert_eq!(goal_label(&b(Objective::F2, 0.5)), "Custom");
        assert_eq!(goal_label(&b(Objective::F1, 0.55)), "Custom");
    }
}
