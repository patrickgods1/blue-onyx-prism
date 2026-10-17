//! Shared fixtures for the unit tests of `report` and `report::recommend` (synthetic runs and
//! device results).

use super::*;
use crate::benchmark::{StageStats, Stats};

pub(crate) fn det(label: &str, conf: f32, b: [usize; 4]) -> Detection {
    Detection {
        label: label.into(),
        confidence: conf,
        x_min: b[0],
        y_min: b[1],
        x_max: b[2],
        y_max: b[3],
    }
}

pub(crate) fn scene() -> Vec<Detection> {
    vec![
        det("dog", 0.90, [0, 0, 100, 100]),
        det("car", 0.80, [200, 200, 300, 300]),
        det("bicycle", 0.70, [400, 0, 500, 100]),
    ]
}

pub(crate) fn run(spec: &str, p50: f64, dets: Vec<Detection>) -> RunResult {
    let total = Stats {
        min: p50,
        mean: p50,
        p50,
        p95: p50,
        max: p50,
    };
    RunResult {
        model: "m".into(),
        path: "m.onnx".into(),
        family: "yolo5".into(),
        requested_device: spec.into(),
        device: spec.into(),
        device_name: String::new(),
        spec: spec.into(),
        fell_back: false,
        input: "640x640".into(),
        image: "x".into(),
        image_size: "1x1".into(),
        images: 1,
        compile_ms: 10.0,
        cache: "disabled".into(),
        warmup_iterations: 0,
        warmup_ms: 0.0,
        repeat: 1,
        stages_ms: StageStats {
            infer: total,
            total,
            ..Default::default()
        },
        throughput_fps: 1000.0 / p50,
        detections: dets,
        by_resolution: Vec::new(),
        accuracy: None,
        accuracy_note: None,
        grades: Some(Grades::new(None, p50, Default::default(), false)),
        agreement: None,
        per_image: Vec::new(),
        eval_preds: Vec::new(),
    }
}

pub(crate) fn dev(spec: &str, p50: f64, dets: Vec<Detection>) -> DeviceResult {
    let d = crate::backend::spec::parse(spec).unwrap();
    DeviceResult::ran(d.device().unwrap(), spec, run(spec, p50, dets))
}

pub(crate) fn failed(spec: &str) -> DeviceResult {
    let d = crate::backend::spec::parse(spec).unwrap();
    DeviceResult::failed(d.device().unwrap(), spec, "compile failed".into())
}

pub(crate) fn model(devices: Vec<DeviceResult>, configured: Option<&str>) -> ModelResult {
    let mut m = ModelResult {
        model: "m".into(),
        path: "m.onnx".into(),
        family: "yolo5".into(),
        timestamp: now(),
        run_id: None,
        image: "x".into(),
        datasets: vec!["sample".into()],
        images: 1,
        warmup: 0,
        repeat: 1,
        configured: configured.map(str::to_string),
        reference: None,
        devices,
        skipped: Vec::new(),
        recommended: None,
        recommendation: String::new(),
        error: None,
        threshold_objective: Objective::default(),
        threshold: None,
    };
    m.finish();
    m
}

pub(crate) fn graded(spec: &str, p50: f64, acc: Option<f64>) -> DeviceResult {
    let mut d = dev(spec, p50, scene());
    d.run.as_mut().unwrap().grades = Some(Grades::new(acc, p50, Default::default(), false));
    d
}

pub(crate) fn graded_auc(spec: &str, p50: f64, auc: Option<f64>) -> DeviceResult {
    let mut d = dev(spec, p50, scene());
    d.run.as_mut().unwrap().grades = Some(Grades::with_metric(
        AccuracyMetric::RocAuc,
        auc,
        p50,
        Default::default(),
        false,
    ));
    d
}

/// A run with an accuracy report whose sweep comes from `preds` (confidence, TP) over
/// `objects` objects.
pub(crate) fn with_sweep(
    mut d: DeviceResult,
    objects: usize,
    preds: &[(f32, bool)],
) -> DeviceResult {
    use super::super::metrics::{Breakpoints, SweepImage, SweepPred, SweepScope};
    use super::super::threshold::{Curve, ThresholdSweep, exact_pick, objectives};
    let img = SweepImage {
        gt: vec![("person".into(), false); objects],
        preds: preds
            .iter()
            .map(|&(confidence, tp)| SweepPred {
                confidence,
                class: "person".into(),
                tp,
                small: false,
            })
            .collect(),
        frames: vec![],
    };
    let bp = Breakpoints::new(&[&img], SweepScope::All);
    let curve = |key: &str| Curve {
        key: key.into(),
        images: 1,
        gt: objects,
        relative: false,
        points: super::super::metrics::SWEEP_THRESHOLDS
            .iter()
            .map(|&t| bp.point_at(t))
            .collect(),
        configured: bp.point_at(0.5),
        picks: objectives(Objective::F1)
            .iter()
            .filter_map(|o| exact_pick(&bp, *o))
            .collect(),
        exact: bp.decimated(50),
        roc_auc: None,
    };
    let run = d.run.as_mut().unwrap();
    run.accuracy = Some(super::super::AccuracyReport {
        ground_truth: "ground truth".into(),
        relative: false,
        classes: vec!["person".into()],
        threshold: 0.5,
        overall: Default::default(),
        by_dataset: vec![],
        by_tag: vec![],
        all_classes: None,
        sweep: Some(ThresholdSweep {
            configured: 0.5,
            overall: curve("overall"),
            by_dataset: vec![curve("exdark-night")],
            by_tag: vec![],
            per_class: vec![curve("person")],
            roc: None,
        }),
        frame_roc: None,
    });
    d
}
