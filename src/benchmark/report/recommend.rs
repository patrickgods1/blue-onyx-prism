//! Device recommendation: rank a model's device results (accuracy tie, then speed), pick the
//! recommended device, and build the measured `auto` device orders from stored results.
//! Pure functions over `DeviceResult`; nothing here reads or writes `benchmark.json`.

use super::*;

/// Measured `auto` orders of `models` (keyed by `config.data_path(&m.path)`, what the registry
/// plans with) from `results` on `hw`; see [`BenchmarkResults::auto_order`]. Models are matched
/// by name, like `--apply`.
pub fn auto_orders<'a>(
    results: Option<&BenchmarkResults>,
    config: &Config,
    models: impl IntoIterator<Item = &'a ModelConfig>,
    hw: &HardwareInfo,
) -> HashMap<PathBuf, Vec<Device>> {
    let Some(results) = results else {
        return HashMap::new();
    };
    models
        .into_iter()
        .filter_map(|m| {
            let order = results.auto_order(&m.effective_name(), hw)?;
            Some((config.data_path(&m.path), order))
        })
        .collect()
}

/// AP50 accuracy scores (0..1) this close to a model's best count as equally accurate when
/// picking its device. Each metric has its own tie on its own scale
/// ([`AccuracyMetric::tie`]); this is the AP50 one.
pub const ACCURACY_TIE: f64 = 0.015;

/// The recommended device of a model and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recommendation {
    pub device: Option<String>,
    pub reason: String,
}

/// The eligible devices of a sweep with their grades, and which of them are as accurate as the
/// best: shared by [`recommend`] (the first of [`Ranked::fastest_accurate`]) and [`speed_order`]
/// (all of them, the measured `auto` order).
struct Ranked<'a> {
    /// Eligible devices ([`DeviceResult::eligible`]) that have grades or a p50, in sweep order.
    eligible: Vec<(&'a DeviceResult, Grades)>,
    /// The grading metric of the sweep (its tie decides "as accurate").
    metric: AccuracyMetric,
    /// Best accuracy score among `eligible`; None when nothing was scored.
    top_acc: Option<f64>,
}

impl<'a> Ranked<'a> {
    fn of(devices: &'a [DeviceResult]) -> Self {
        let eligible: Vec<(&DeviceResult, Grades)> = devices
            .iter()
            .filter(|d| d.eligible())
            .filter_map(|d| {
                let g = d.grades().copied().or_else(|| {
                    d.total_p50()
                        .map(|p| Grades::new(None, p, Default::default(), false))
                })?;
                Some((d, g))
            })
            .collect();
        // Devices of one model whose detections agree differ in accuracy only by numeric noise
        // (OpenVINO's f16 CPU inference on ARM moves AP by ~1 point), so accuracy within
        // ACCURACY_TIE of the best counts as equal and the fastest of those wins. Letter grades
        // are too coarse here: 14 ms and 35 ms are both speed A.
        // Every run of one sweep is graded on one metric; the tie is on its scale.
        let metric = eligible
            .iter()
            .find(|(_, g)| g.accuracy_score.is_some())
            .map_or(AccuracyMetric::Ap50, |(_, g)| g.metric);
        let top_acc = eligible
            .iter()
            .filter_map(|(_, g)| g.accuracy_score)
            .max_by(f64::total_cmp);
        Self {
            eligible,
            metric,
            top_acc,
        }
    }

    fn as_accurate(&self, g: &Grades) -> bool {
        match (self.top_acc, g.accuracy_score) {
            (Some(top), Some(a)) => a >= top - self.metric.tie(),
            (Some(_), None) => false,
            (None, _) => true,
        }
    }

    /// The eligible devices as accurate as the best, fastest full-request p50 first (ties:
    /// [`Grades::rank_cmp`], then sweep order).
    fn fastest_accurate(&self) -> impl Iterator<Item = (&'a DeviceResult, Grades)> {
        let mut v: Vec<(&'a DeviceResult, Grades)> = self
            .eligible
            .iter()
            .filter(|(_, g)| self.as_accurate(g))
            .copied()
            .collect();
        v.sort_by(|a, b| {
            a.1.p50_ms
                .total_cmp(&b.1.p50_ms)
                .then_with(|| Grades::rank_cmp(&a.1, &b.1))
        });
        v.into_iter()
    }
}

/// The measured `auto` order of one sweep: the devices [`recommend`] chooses among (ran on the
/// device itself, detections agree with the CPU reference or no reference ran, as accurate as
/// the best), fastest full-request p50 first. The recommended device (without the
/// configured-device noise rule) is the first. Specs as recorded ("ort:directml:0").
pub fn speed_order(devices: &[DeviceResult]) -> Vec<&str> {
    Ranked::of(devices)
        .fastest_accurate()
        .map(|(d, _)| d.device.as_str())
        .collect()
}

/// The best device among those that ran and whose detections agree with the CPU reference:
/// the fastest full-request p50 among those as accurate as the best (within the grading
/// metric's tie, [`AccuracyMetric::tie`]: [`ACCURACY_TIE`] for AP50, 0.01 for ROC AUC). When the
/// `configured` device is eligible, as accurate ([`AccuracyMetric::keep_tie`]) and within
/// `margin` (fraction, e.g. `NOISE_MARGIN`) of the best p50, it is kept: the difference is
/// noise.
pub fn recommend(
    devices: &[DeviceResult],
    configured: Option<&str>,
    margin: f64,
) -> Recommendation {
    let ranked = Ranked::of(devices);
    let (eligible, metric) = (&ranked.eligible, ranked.metric);
    let Some((best, bg)) = ranked.fastest_accurate().next() else {
        let reason = if devices.iter().any(DeviceResult::ok) {
            "no device's detections agree with the CPU reference".to_string()
        } else {
            "no device could run this model".to_string()
        };
        return Recommendation {
            device: None,
            reason,
        };
    };
    let excluded: Vec<&str> = devices
        .iter()
        .filter(|d| d.ok() && !d.eligible())
        .map(|d| d.device.as_str())
        .collect();
    let excluded = if excluded.is_empty() {
        String::new()
    } else {
        format!(
            "; excluded (detections disagree with the reference): {}",
            excluded.join(", ")
        )
    };
    let grade_text = |g: &Grades| match g.accuracy {
        Some(a) => format!(
            "overall {} (accuracy {a} by {}, speed {})",
            g.overall,
            g.metric.label(),
            g.speed
        ),
        None => format!("speed {}", g.speed),
    };
    if let Some(cur) = configured
        && cur != best.device
        && let Some((d, g)) = eligible.iter().find(|(d, _)| d.device == cur)
        && g.p50_ms <= bg.p50_ms * (1.0 + margin)
        && match (g.accuracy_score, bg.accuracy_score) {
            (Some(a), Some(b)) => a >= b - metric.keep_tie(),
            (None, None) => true,
            _ => false,
        }
    {
        return Recommendation {
            device: Some(d.device.clone()),
            reason: format!(
                "the configured device {} ({:.1} ms) is within {:.0}% of the best, {} ({:.1} ms){excluded}",
                d.device,
                g.p50_ms,
                margin * 100.0,
                best.device,
                bg.p50_ms
            ),
        };
    }
    // Compared with the fastest other eligible device.
    let runner_up = eligible
        .iter()
        .filter(|(d, _)| d.device != best.device)
        .min_by(|a, b| a.1.p50_ms.total_cmp(&b.1.p50_ms));
    let vs = match runner_up {
        Some((d, g)) if g.p50_ms >= bg.p50_ms => format!(
            ", {:.2}x faster than {} ({:.1} ms)",
            g.p50_ms / bg.p50_ms.max(1e-9),
            d.device,
            g.p50_ms
        ),
        Some((d, g)) => format!(
            "; {} is {:.2}x faster ({:.1} ms) but less accurate",
            d.device,
            bg.p50_ms / g.p50_ms.max(1e-9),
            g.p50_ms
        ),
        None => String::new(),
    };
    Recommendation {
        device: Some(best.device.clone()),
        reason: format!(
            "{}; full request p50 {:.1} ms{vs}{excluded}",
            grade_text(&bg),
            bg.p50_ms
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;

    #[test]
    fn recommends_fastest_agreeing_device() {
        let m = model(
            vec![
                dev("ort:coreml", 105.0, scene()),
                dev("openvino:cpu", 37.0, scene()),
                dev("ort:cpu", 60.0, scene()),
            ],
            Some("ort:coreml"),
        );
        assert_eq!(m.reference.as_deref(), Some("openvino:cpu"));
        assert_eq!(m.recommended.as_deref(), Some("openvino:cpu"));
        assert!(
            m.recommendation.contains("1.62x faster than ort:cpu"),
            "{}",
            m.recommendation
        );
        let cpu = &m.devices[1];
        assert_eq!(cpu.agreement().unwrap().verdict, Verdict::Reference);
        assert_eq!(m.devices[0].agreement().unwrap().verdict, Verdict::Agrees);
        assert_eq!(m.fastest().unwrap().device, "openvino:cpu");
    }

    #[test]
    fn disagreeing_device_is_never_recommended() {
        let m = model(
            vec![
                dev("ort:coreml", 20.0, vec![]),
                dev("openvino:cpu", 89.0, scene()),
                dev("ort:cpu", 70.0, scene()),
            ],
            None,
        );
        assert_eq!(
            m.devices[0].agreement().unwrap().verdict,
            Verdict::Disagrees
        );
        assert!(!m.devices[0].eligible());
        assert_eq!(m.recommended.as_deref(), Some("ort:cpu"));
        assert!(
            m.recommendation.contains("excluded"),
            "{}",
            m.recommendation
        );
        // ...even though it is the fastest row.
        assert_eq!(m.fastest().unwrap().device, "ort:coreml");
    }

    #[test]
    fn failed_and_fallback_runs_are_excluded() {
        let mut fb = dev("openvino:gpu", 5.0, scene());
        fb.run.as_mut().unwrap().fell_back = true;
        let m = model(
            vec![failed("ort:coreml"), fb, dev("openvino:cpu", 40.0, scene())],
            None,
        );
        assert_eq!(m.recommended.as_deref(), Some("openvino:cpu"));
        let m = model(vec![failed("ort:coreml"), failed("openvino:cpu")], None);
        assert_eq!(m.recommended, None);
        assert_eq!(m.reference, None);
        assert!(m.recommendation.contains("no device could run"));
    }

    #[test]
    fn noise_margin_keeps_the_configured_device() {
        // Configured ort:coreml is 4% slower than the fastest: kept.
        let devices = || {
            vec![
                dev("ort:coreml", 41.6, scene()),
                dev("openvino:cpu", 40.0, scene()),
            ]
        };
        let m = model(devices(), Some("ort:coreml"));
        assert_eq!(m.recommended.as_deref(), Some("ort:coreml"));
        assert!(
            m.recommendation.contains("within 5%"),
            "{}",
            m.recommendation
        );
        // Without a configured device (or configured elsewhere) the fastest wins.
        assert_eq!(
            model(devices(), None).recommended.as_deref(),
            Some("openvino:cpu")
        );
        // 6% slower: switch.
        let m = model(
            vec![
                dev("ort:coreml", 42.4, scene()),
                dev("openvino:cpu", 40.0, scene()),
            ],
            Some("ort:coreml"),
        );
        assert_eq!(m.recommended.as_deref(), Some("openvino:cpu"));
        // A configured device that disagrees is not kept even when within the margin.
        let m = model(
            vec![
                dev("ort:coreml", 40.5, vec![]),
                dev("openvino:cpu", 40.0, scene()),
            ],
            Some("ort:coreml"),
        );
        assert_eq!(m.recommended.as_deref(), Some("openvino:cpu"));
    }

    #[test]
    fn ort_cpu_is_the_reference_without_openvino() {
        let m = model(
            vec![
                dev("ort:coreml", 44.0, scene()),
                dev("ort:cpu", 89.0, scene()),
            ],
            None,
        );
        assert_eq!(m.reference.as_deref(), Some("ort:cpu"));
        assert_eq!(m.recommended.as_deref(), Some("ort:coreml"));
        // No CPU device at all: unchecked, still recommended, with a note.
        let m = model(vec![dev("ort:coreml", 44.0, scene())], None);
        assert_eq!(m.reference, None);
        assert_eq!(m.recommended.as_deref(), Some("ort:coreml"));
        assert!(
            m.recommendation.contains("not checked"),
            "{}",
            m.recommendation
        );
    }

    #[test]
    fn grades_drive_recommendation_and_ranking() {
        // ort:coreml is faster (A speed) but less accurate (C): 0.6*2 + 0.4*4 = 2.8;
        // openvino:cpu: accuracy A, speed B: 0.6*4 + 0.4*3 = 3.6 -> wins.
        let m = model(
            vec![
                graded("ort:coreml", 40.0, Some(0.55)),
                graded("openvino:cpu", 60.0, Some(0.75)),
            ],
            None,
        );
        assert_eq!(m.recommended.as_deref(), Some("openvino:cpu"));
        assert!(
            m.recommendation.starts_with("overall A"),
            "{}",
            m.recommendation
        );
        assert_eq!(m.recommended_grades().unwrap().overall, Grade::A);
        // Configured device kept only when as accurate.
        let m = model(
            vec![
                graded("ort:coreml", 61.0, Some(0.745)),
                graded("openvino:cpu", 60.0, Some(0.75)),
            ],
            Some("ort:coreml"),
        );
        assert_eq!(m.recommended.as_deref(), Some("ort:coreml"));

        // Same letter grades and accuracy within noise (yolo26n: OpenVINO f16 on ARM is ~1 AP
        // off): the clearly faster device wins, and the reason reads as a speedup.
        let m = model(
            vec![
                graded("ort:coreml", 14.4, Some(0.481)),
                graded("openvino:cpu", 35.0, Some(0.490)),
                graded("ort:cpu", 33.7, Some(0.481)),
            ],
            None,
        );
        assert_eq!(m.recommended.as_deref(), Some("ort:coreml"));
        assert!(
            m.recommendation.contains("2.34x faster than ort:cpu"),
            "{}",
            m.recommendation
        );
        // A faster but clearly less accurate device loses, and the reason says so.
        let m = model(
            vec![
                graded("ort:coreml", 40.0, Some(0.55)),
                graded("openvino:cpu", 60.0, Some(0.75)),
            ],
            None,
        );
        assert!(
            m.recommendation
                .contains("ort:coreml is 1.50x faster (40.0 ms) but less accurate"),
            "{}",
            m.recommendation
        );

        let mut a = model(vec![graded("openvino:cpu", 30.0, Some(0.5))], None);
        a.model = "small".into();
        let mut b = model(vec![graded("ort:coreml", 45.0, Some(0.8))], None);
        b.model = "big".into();
        let r = BenchmarkResults::new(Default::default(), Default::default(), vec![a, b]);
        let rank = r.ranking();
        assert_eq!(rank.len(), 2);
        assert_eq!((rank[0].rank, rank[0].model.as_str()), (1, "big"));
        assert_eq!(rank[0].overall, Grade::A);
        assert_eq!(rank[1].device, "openvino:cpu");
    }

    #[test]
    fn roc_auc_metric_drives_tie_ranking_and_text() {
        // AUC within 0.01 of the best counts as equal (the AUC scale): the faster device wins.
        let m = model(
            vec![
                graded_auc("ort:coreml", 14.0, Some(0.952)),
                graded_auc("openvino:cpu", 35.0, Some(0.960)),
            ],
            None,
        );
        assert_eq!(m.recommended.as_deref(), Some("ort:coreml"));
        assert!(
            m.recommendation.contains("accuracy A by ROC AUC"),
            "{}",
            m.recommendation
        );
        // 0.015 apart is a real difference on the AUC scale (it would tie for AP50).
        let m = model(
            vec![
                graded_auc("ort:coreml", 14.0, Some(0.945)),
                graded_auc("openvino:cpu", 35.0, Some(0.960)),
            ],
            None,
        );
        assert_eq!(m.recommended.as_deref(), Some("openvino:cpu"));
        let ap = model(
            vec![
                graded("ort:coreml", 14.0, Some(0.745)),
                graded("openvino:cpu", 35.0, Some(0.760)),
            ],
            None,
        );
        assert_eq!(ap.recommended.as_deref(), Some("ort:coreml"));
        // Ranking rows say which metric graded them.
        let r = BenchmarkResults::new(Default::default(), Default::default(), vec![m.clone()]);
        let rows = r.ranking();
        assert_eq!(rows[0].metric, AccuracyMetric::RocAuc);
        assert_eq!(rows[0].accuracy_score, Some(0.960));
        assert_eq!(r.accuracy_metric(), AccuracyMetric::RocAuc);
    }

    #[test]
    fn speed_order_is_fastest_first_among_the_recommendable() {
        // Faster first; the CPU reference is in it too (the plan keeps the CPU last).
        let m = model(
            vec![
                dev("openvino:gpu.0", 41.6, scene()),
                dev("openvino:cpu", 80.0, scene()),
                dev("ort:directml:0", 10.5, scene()),
                dev("ort:directml:1", 109.2, scene()),
            ],
            None,
        );
        assert_eq!(
            speed_order(&m.devices),
            [
                "ort:directml:0",
                "openvino:gpu.0",
                "openvino:cpu",
                "ort:directml:1"
            ]
        );
        assert_eq!(m.recommended.as_deref(), Some(speed_order(&m.devices)[0]));

        // Failed runs, fallback runs and devices that disagree with the reference are out.
        let mut fb = dev("openvino:gpu.0", 5.0, scene());
        fb.run.as_mut().unwrap().fell_back = true;
        let m = model(
            vec![
                failed("ort:directml:0"),
                fb,
                dev("ort:directml:1", 3.0, vec![]),
                dev("openvino:cpu", 80.0, scene()),
            ],
            None,
        );
        assert!(!m.devices[2].eligible());
        assert_eq!(speed_order(&m.devices), ["openvino:cpu"]);

        // Accuracy: within the tie of the best counts as equal (speed decides); clearly less
        // accurate devices are out, however fast. No reference ran: all are checked by grades.
        let m = model(
            vec![
                graded("ort:directml:0", 10.0, Some(0.480)),
                graded("openvino:gpu.0", 40.0, Some(0.490)),
                graded("ort:directml:1", 5.0, Some(0.40)),
            ],
            None,
        );
        assert_eq!(m.reference, None);
        assert_eq!(
            speed_order(&m.devices),
            ["ort:directml:0", "openvino:gpu.0"]
        );

        // Nothing ran: empty (the static ranking stays).
        let m = model(vec![failed("ort:directml:0")], None);
        assert!(speed_order(&m.devices).is_empty());
    }

    #[test]
    fn auto_order_needs_the_same_gpus() {
        use crate::backend::detect::{GpuAdapter, GpuVendor};
        let gtx = GpuAdapter {
            vendor: GpuVendor::Nvidia,
            name: "NVIDIA GeForce GTX 1070 Ti".into(),
            vram_mb: 8060,
            index: 0,
            discrete: true,
            cuda: None,
        };
        let uhd = GpuAdapter {
            vendor: GpuVendor::Intel,
            name: "Intel(R) UHD Graphics 630".into(),
            vram_mb: 128,
            index: 1,
            discrete: false,
            cuda: None,
        };
        let hw = HardwareInfo::new("windows", "x86_64", vec![gtx.clone(), uhd.clone()]);
        // Recorded by an older version without the CUDA facts: still the same GPUs.
        let recorded = HardwareSummary {
            os: "windows".into(),
            arch: "x86_64".into(),
            cpu: "Intel(R) Core(TM) i5-8600K CPU @ 3.60GHz".into(),
            gpus: vec![
                "#0 NVIDIA GeForce GTX 1070 Ti (NVIDIA, discrete, 8060 MB VRAM)".into(),
                "#1 Intel(R) UHD Graphics 630 (Intel, integrated, 128 MB VRAM)".into(),
            ],
        };
        assert!(recorded.same_gpus(&hw));
        let mut m = model(
            vec![
                dev("openvino:gpu.0", 41.6, scene()),
                dev("ort:directml:0", 10.5, scene()),
            ],
            None,
        );
        m.model = "IPcam-general".into();
        let mut failed_only = model(vec![failed("openvino:gpu.0")], None);
        failed_only.model = "yolo26n".into();
        let results =
            BenchmarkResults::new(recorded.clone(), Default::default(), vec![m, failed_only]);
        let d = |s: &str| *crate::backend::spec::parse(s).unwrap().device().unwrap();
        assert_eq!(
            results.auto_order("ipcam-general.onnx", &hw),
            Some(vec![d("ort:directml:0"), d("openvino:gpu.0")])
        );
        assert_eq!(results.auto_order("yolo26n", &hw), None);
        assert_eq!(results.auto_order("other", &hw), None);

        // Keyed by the resolved model path, for the models asked about only.
        let mut cfg = Config::default();
        let entry = |name: &str, path: &str| ModelConfig {
            name: Some(name.into()),
            path: path.into(),
            ..ModelConfig::default()
        };
        cfg.models = vec![
            entry("IPcam-general", "models/IPcam-general.onnx"),
            entry("yolo26n", "models/yolo26n.xml"),
        ];
        let orders = auto_orders(Some(&results), &cfg, &cfg.models, &hw);
        assert_eq!(orders.len(), 1);
        assert_eq!(
            orders[&cfg.data_path(Path::new("models/IPcam-general.onnx"))][0],
            d("ort:directml:0")
        );
        assert!(auto_orders(None, &cfg, &cfg.models, &hw).is_empty());

        // Another GPU set (a card added, swapped or reordered): results are stale.
        for gpus in [
            vec![uhd.clone()],
            vec![
                GpuAdapter {
                    index: 0,
                    ..uhd.clone()
                },
                GpuAdapter {
                    index: 1,
                    ..gtx.clone()
                },
            ],
            vec![
                GpuAdapter {
                    name: "NVIDIA GeForce RTX 3060".into(),
                    ..gtx.clone()
                },
                uhd.clone(),
            ],
        ] {
            let other = HardwareInfo::new("windows", "x86_64", gpus);
            assert!(!recorded.same_gpus(&other));
            assert_eq!(results.auto_order("IPcam-general", &other), None);
        }
        assert!(!recorded.same_gpus(&HardwareInfo::new("linux", "x86_64", hw.gpus.clone())));
    }
}
