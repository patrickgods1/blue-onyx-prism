//! Unit tests of the run history: snapshots, file, cap and pin, seeding, comparability,
//! verdict edges, trust and best trade-offs (synthetic entries, no inference).

use super::compare::*;
use super::trust::*;
use super::*;
use crate::benchmark::report::test_support::{dev, model, scene, with_sweep};

/// A device row that ran: p50 ms, recall over `objects` objects.
fn snap(device: &str, p50: f64, recall: Option<f64>, objects: usize) -> DeviceSnap {
    DeviceSnap {
        device: device.into(),
        label: device.into(),
        runtime: runtime_of(device).into(),
        total_p50_ms: Some(p50),
        total_p95_ms: Some(p50 * 1.2),
        infer_p50_ms: Some(p50 * 0.8),
        fps: Some(1000.0 / p50),
        recall,
        ground_truth: Some(objects),
        recommended: true,
        ..Default::default()
    }
}

fn entry(run_id: &str, devices: Vec<DeviceSnap>) -> HistoryEntry {
    HistoryEntry {
        run_id: run_id.into(),
        timestamp: run_id.into(),
        model: "IPcam-general".into(),
        pinned: false,
        settings: RunSettings {
            datasets: vec!["coco-cctv".into()],
            images: 120,
            repeat: 3,
            warmup: 1,
            objective: "f1".into(),
            metric: "ap50".into(),
            reference: None,
            labels: Labels::Human,
        },
        hardware_key: "windows/x86_64/i5/#0 UHD 630".into(),
        hardware: HardwareSummary {
            os: "windows".into(),
            arch: "x86_64".into(),
            cpu: "i5".into(),
            gpus: vec!["#0 UHD 630 (Intel, integrated, 128 MB VRAM)".into()],
        },
        runtimes: RuntimeVersions {
            openvino: Some("2026.0".into()),
            onnxruntime: Some("1.24.4".into()),
            onnxruntime_flavor: Some("cpu".into()),
        },
        devices,
        threshold: None,
        trained_on_share: None,
    }
}

fn thr(best: f32, plateau: Option<[f32; 2]>, score: Option<f64>) -> ThresholdSnap {
    ThresholdSnap {
        objective: "f1".into(),
        configured: 0.5,
        best: Some(best),
        plateau,
        exact_score: score,
        device: "openvino:cpu".into(),
    }
}

fn metric<'a>(c: &'a Changes, name: &str) -> &'a MetricChange {
    c.metrics
        .iter()
        .find(|m| m.metric == name)
        .unwrap_or_else(|| panic!("no {name} in {c:?}"))
}

#[test]
fn speed_verdicts_at_the_noise_floor() {
    // 5% of 100 ms = 5 ms: 4.9 ms is noise, 5.1 ms is a change (either way).
    let c = speed_change("p50_ms", "d", Some(100.0), Some(95.1), false);
    assert_eq!(c.verdict, Verdict::Same, "{}", c.why);
    assert!(c.why.contains("within the 5% / 0.5 ms"), "{}", c.why);
    let c = speed_change("p50_ms", "d", Some(100.0), Some(94.9), false);
    assert_eq!(c.verdict, Verdict::Better, "{}", c.why);
    assert_eq!(c.why, "5% faster (more than the 5% / 0.5 ms noise floor)");
    assert!((c.rel.unwrap() + 0.051).abs() < 1e-12);
    assert!((c.delta.unwrap() + 5.1).abs() < 1e-9);
    let c = speed_change("p50_ms", "d", Some(100.0), Some(105.1), false);
    assert_eq!(c.verdict, Verdict::Worse);
    assert!(c.why.starts_with("5% slower"), "{}", c.why);
    // Exactly the floor counts.
    let c = speed_change("p50_ms", "d", Some(100.0), Some(105.0), false);
    assert_eq!(c.verdict, Verdict::Worse);
    // Fast models: the 0.5 ms floor rules (5% of 4 ms is 0.2 ms).
    let c = speed_change("p50_ms", "d", Some(4.0), Some(3.6), false);
    assert_eq!(c.verdict, Verdict::Same, "10% but only 0.4 ms: {}", c.why);
    let c = speed_change("p50_ms", "d", Some(4.0), Some(3.5), false);
    assert_eq!(c.verdict, Verdict::Better);
    // Images per second: higher is better, relative rule only.
    let c = speed_change("fps", "d", Some(10.0), Some(10.6), true);
    assert_eq!(c.verdict, Verdict::Better);
    let c = speed_change("fps", "d", Some(10.0), Some(10.4), true);
    assert_eq!(c.verdict, Verdict::Same);
    assert!(c.why.contains("5% noise floor"), "{}", c.why);
    // Missing values: same, with the reason.
    let c = speed_change("p50_ms", "d", None, Some(10.0), false);
    assert_eq!((c.verdict, c.delta), (Verdict::Same, None));
    assert!(c.why.contains("earlier"), "{}", c.why);
}

#[test]
fn accuracy_verdicts_at_two_standard_errors() {
    // p = 0.5, n = 100: 2 * sqrt(0.25 / 100) = 0.1.
    assert!((proportion_noise(0.5, 100) - 0.1).abs() < 1e-12);
    let c = proportion_change("recall", "d", Some(0.5), Some(0.599), Some(100), "objects");
    assert_eq!(c.verdict, Verdict::Same, "{}", c.why);
    assert!(
        c.why.starts_with("within normal variation for 100 objects"),
        "{}",
        c.why
    );
    let c = proportion_change("recall", "d", Some(0.5), Some(0.601), Some(100), "objects");
    assert_eq!(c.verdict, Verdict::Better, "{}", c.why);
    assert!(c.why.contains("10.1 points"), "{}", c.why);
    let c = proportion_change("recall", "d", Some(0.5), Some(0.399), Some(100), "objects");
    assert_eq!(c.verdict, Verdict::Worse);
    // More objects, smaller noise: 2 * sqrt(0.25 / 10000) = 0.01.
    let c = proportion_change(
        "recall",
        "d",
        Some(0.5),
        Some(0.515),
        Some(10_000),
        "objects",
    );
    assert_eq!(c.verdict, Verdict::Better);
    // A perfect earlier score: at least one object's worth (1/50 = 0.02).
    assert!((proportion_noise(1.0, 50) - 0.02).abs() < 1e-12);
    let c = proportion_change(
        "precision",
        "d",
        Some(1.0),
        Some(0.99),
        Some(50),
        "detections",
    );
    assert_eq!(c.verdict, Verdict::Same, "{}", c.why);
    let c = proportion_change(
        "precision",
        "d",
        Some(1.0),
        Some(0.96),
        Some(50),
        "detections",
    );
    assert_eq!(c.verdict, Verdict::Worse);
    // No objects or a missing value: same, naming what is missing.
    let c = proportion_change("ap50", "d", Some(0.5), Some(0.9), Some(0), "objects");
    assert_eq!(c.verdict, Verdict::Same);
    assert!(c.why.contains("no objects"), "{}", c.why);
    let c = proportion_change("ap50", "d", Some(0.5), None, Some(10), "objects");
    assert_eq!((c.verdict, c.delta), (Verdict::Same, None));
    assert!(c.why.contains("later"), "{}", c.why);
}

#[test]
fn threshold_inside_the_plateau_is_rounding() {
    let a = thr(0.41, Some([0.38, 0.47]), Some(0.881));
    let c = threshold_change(Some(&a), Some(&thr(0.42, Some([0.39, 0.47]), Some(0.882)))).unwrap();
    assert_eq!(c.verdict, Verdict::Same);
    assert!(c.why.starts_with("no real change (rounding)"), "{}", c.why);
    assert!(
        c.why.contains("0.38-0.47") && c.why.contains("0.2 points"),
        "{}",
        c.why
    );
    assert!((c.delta.unwrap() - 0.01).abs() < 1e-6);
    assert!(c.device.is_none());
    // The plateau's edges are inside.
    let c = threshold_change(Some(&a), Some(&thr(0.47, None, Some(0.95)))).unwrap();
    assert_eq!(c.verdict, Verdict::Same);
    // Outside: by the best score, past the 0.2-point tolerance.
    let c = threshold_change(Some(&a), Some(&thr(0.55, None, Some(0.90)))).unwrap();
    assert_eq!(c.verdict, Verdict::Better, "{}", c.why);
    let c = threshold_change(Some(&a), Some(&thr(0.55, None, Some(0.86)))).unwrap();
    assert_eq!(c.verdict, Verdict::Worse);
    let c = threshold_change(Some(&a), Some(&thr(0.55, None, Some(0.882)))).unwrap();
    assert_eq!(c.verdict, Verdict::Same, "{}", c.why);
    assert!(c.why.contains("tolerance"), "{}", c.why);
    // No score: same, said so.
    let c = threshold_change(Some(&thr(0.4, None, None)), Some(&thr(0.6, None, None))).unwrap();
    assert_eq!(c.verdict, Verdict::Same);
    // Another objective: not comparable.
    let mut f2 = thr(0.3, None, Some(0.9));
    f2.objective = "f2".into();
    let c = threshold_change(Some(&a), Some(&f2)).unwrap();
    assert_eq!((c.verdict, c.delta), (Verdict::NotComparable, None));
    // Unchanged; one side missing; neither.
    let c = threshold_change(Some(&a), Some(&a)).unwrap();
    assert_eq!(c.why, "unchanged at 0.41");
    assert!(
        threshold_change(None, Some(&a))
            .unwrap()
            .why
            .contains("earlier")
    );
    assert!(threshold_change(None, None).is_none());
}

#[test]
fn comparability_reasons() {
    let a = entry("r1", vec![snap("openvino:cpu", 20.0, Some(0.8), 200)]);
    assert_eq!(comparable(&a, &a.clone()), Ok(()));
    let mut b = a.clone();
    b.settings.datasets = vec!["bmd45-cctv".into()];
    b.settings.images = 20;
    b.settings.reference = Some("yolo26s".into());
    let r = comparable(&a, &b).unwrap_err();
    assert_eq!(
        r,
        [
            "different datasets: coco-cctv -> bmd45-cctv",
            "different image count: 120 -> 20",
            "different reference model: none -> yolo26s",
        ]
    );
    // Hardware: another CPU; GPU facts (VRAM) alone do not matter.
    let mut b = a.clone();
    b.hardware.gpus = vec!["#0 UHD 630 (Intel, integrated, 256 MB VRAM)".into()];
    assert_eq!(comparable(&a, &b), Ok(()));
    b.hardware.cpu = "Ryzen".into();
    assert!(comparable(&a, &b).unwrap_err()[0].contains("the CPU changed"));
    // Runtimes: only those the devices used.
    let mut b = a.clone();
    b.runtimes.onnxruntime = Some("1.31.0".into());
    assert_eq!(comparable(&a, &b), Ok(()), "no ort device ran");
    b.runtimes.openvino = Some("2026.1".into());
    assert_eq!(
        comparable(&a, &b).unwrap_err(),
        ["OpenVINO changed: 2026.0 -> 2026.1"]
    );
    let mut ort_a = entry("r1", vec![snap("ort:cpu", 20.0, None, 0)]);
    ort_a.runtimes.openvino = None;
    let mut ort_b = ort_a.clone();
    ort_b.runtimes.onnxruntime_flavor = Some("cuda".into());
    assert_eq!(
        comparable(&ort_a, &ort_b).unwrap_err(),
        ["ONNX Runtime changed: 1.24.4 (cpu) -> 1.24.4 (cuda)"]
    );
    // No device in common.
    let b = entry("r2", vec![snap("ort:cpu", 20.0, None, 0)]);
    assert_eq!(
        comparable(&a, &b).unwrap_err(),
        ["no device ran in both runs"]
    );
    // A device that fell back does not count as having run.
    let mut fb = a.clone();
    fb.devices[0].fell_back = true;
    assert!(comparable(&a, &fb).is_err());
}

#[test]
fn compare_lists_changes_and_summarizes() {
    let mut a = entry("r1", vec![snap("openvino:cpu", 100.0, Some(0.80), 300)]);
    a.threshold = Some(thr(0.41, Some([0.38, 0.47]), Some(0.88)));
    let mut b = entry("r2", vec![snap("openvino:cpu", 91.0, Some(0.81), 300)]);
    b.threshold = Some(thr(0.42, Some([0.39, 0.47]), Some(0.881)));
    let c = compare(&a, &b);
    assert!(c.comparable && c.reasons.is_empty());
    assert_eq!(c.baseline.as_ref().unwrap().run_id, "r1");
    assert_eq!(metric(&c, "p50_ms").verdict, Verdict::Better);
    assert_eq!(metric(&c, "recall").verdict, Verdict::Same);
    assert_eq!(metric(&c, "threshold").verdict, Verdict::Same);
    // Metrics neither run has are left out.
    assert!(c.metrics.iter().all(|m| m.metric != "ap50"));
    assert_eq!(
        c.summary,
        "9% faster; accuracy about the same; confidence 0.41 -> 0.42, no real change (rounding)"
    );
    // Not comparable: reasons, no numbers.
    let mut other = b.clone();
    other.settings.images = 20;
    let c = compare(&a, &other);
    assert!(!c.comparable && c.metrics.is_empty());
    assert_eq!(
        c.summary,
        "not comparable with the earlier run: different image count: 120 -> 20"
    );
    // Serialized names follow the contract.
    let v = serde_json::to_value(compare(&a, &b)).unwrap();
    assert_eq!(v["metrics"][0]["verdict"], "better");
    assert_eq!(v["metrics"][0]["device"], "openvino:cpu");
    assert!(v["baseline"]["pinned"].is_boolean());
    let nc = serde_json::to_value(Verdict::NotComparable).unwrap();
    assert_eq!(nc, "not_comparable");
}

/// Images per second of a pipelined run never compares with one measured one image at a time
/// (older versions); two pipelined runs compare as before.
#[test]
fn fps_compares_only_the_same_way_of_measuring() {
    let a = entry("r1", vec![snap("openvino:cpu", 100.0, Some(0.80), 300)]);
    let mut fast = snap("openvino:cpu", 100.0, Some(0.80), 300);
    fast.fps = Some(14.0);
    fast.pipelined = true;
    let b = entry("r2", vec![fast.clone()]);
    let c = compare(&a, &b);
    let m = metric(&c, "fps");
    assert_eq!(
        (m.verdict, m.delta, m.rel),
        (Verdict::NotComparable, None, None)
    );
    assert_eq!(
        m.why,
        "images per second measured another way (one image at a time -> pipelined)"
    );
    assert_eq!(
        metric(&c, "p50_ms").verdict,
        Verdict::Same,
        "latency still compares"
    );
    let mut a2 = a.clone();
    a2.devices[0].pipelined = true;
    assert_eq!(metric(&compare(&a2, &b), "fps").verdict, Verdict::Better);
    // Older entries without the key load as not pipelined; a pipelined one writes it.
    let v = serde_json::to_value(&fast).unwrap();
    assert_eq!(v["pipelined"], true);
    let old: DeviceSnap = serde_json::from_str(r#"{"device":"openvino:cpu","fps":10.0}"#).unwrap();
    assert!(!old.pipelined);
    assert!(
        serde_json::to_value(&old)
            .unwrap()
            .get("pipelined")
            .is_none()
    );
}

#[test]
fn compare_with_history_picks_the_baseline() {
    let d = |p50: f64| vec![snap("openvino:cpu", p50, Some(0.8), 300)];
    let current = entry("r4", d(50.0));
    assert_eq!(compare_with_history(&[], &current), Changes::first_run());
    // Only the current entry itself: first run.
    assert!(
        compare_with_history(std::slice::from_ref(&current), &current)
            .baseline
            .is_none()
    );
    let r1 = entry("r1", d(100.0));
    let r2 = entry("r2", d(60.0));
    let mut r3 = entry("r3", d(55.0));
    r3.settings.images = 20; // not comparable
    let list = vec![r1.clone(), r2.clone(), r3.clone(), current.clone()];
    // The newest comparable earlier run (r3 differs).
    let c = compare_with_history(&list, &current);
    assert_eq!(c.baseline.unwrap().run_id, "r2");
    assert!(c.comparable);
    // A pinned baseline wins.
    let mut pinned = list.clone();
    pinned[0].pinned = true;
    let c = compare_with_history(&pinned, &current);
    let base = c.baseline.unwrap();
    assert_eq!((base.run_id.as_str(), base.pinned), ("r1", true));
    // Nothing comparable: the newest earlier one, not comparable.
    let only = vec![r3.clone(), current.clone()];
    let c = compare_with_history(&only, &current);
    assert_eq!(c.baseline.unwrap().run_id, "r3");
    assert!(!c.comparable);
    assert_eq!(c.reasons, ["different image count: 20 -> 120"]);
    // The current entry not in the list (status view of a result): every entry is earlier.
    let c = compare_with_history(&[r1, r2], &current);
    assert_eq!(c.baseline.unwrap().run_id, "r2");
    // Entries after the current one are not its baseline.
    let c = compare_with_history(&[current.clone(), r3], &current);
    assert!(c.baseline.is_none());
}

#[test]
fn pareto_marks_best_tradeoffs() {
    // (p50, score): b is dominated by a (faster and more accurate); c is the most accurate;
    // d ties a exactly (both stay).
    let rows = [
        (10.0, Some(0.7)),
        (20.0, Some(0.6)),
        (30.0, Some(0.9)),
        (10.0, Some(0.7)),
    ];
    assert_eq!(pareto(&rows), [true, false, true, true]);
    // Same speed, better score dominates; same score, faster dominates.
    assert_eq!(
        pareto(&[(10.0, Some(0.7)), (10.0, Some(0.8))]),
        [false, true]
    );
    assert_eq!(
        pareto(&[(9.0, Some(0.7)), (10.0, Some(0.7))]),
        [true, false]
    );
    // Unscored rows are never best while a row has a score; none scored: the fastest.
    assert_eq!(pareto(&[(5.0, None), (10.0, Some(0.5))]), [false, true]);
    assert_eq!(
        pareto(&[(5.0, None), (10.0, None), (5.0, None)]),
        [true, false, true]
    );
    // A non-finite p50 is the slowest.
    assert_eq!(
        pareto(&[(f64::NAN, Some(0.9)), (10.0, Some(0.9))]),
        [false, true]
    );
    assert!(pareto(&[]).is_empty());
}

#[test]
fn trust_levels() {
    let t = |objects: Option<usize>, relative: bool, repeat: usize| {
        trust(&TrustInput {
            objects,
            relative,
            reference: relative.then(|| "yolo26s".to_string()),
            repeat,
            ..Default::default()
        })
    };
    assert_eq!(t(Some(300), false, 3).level, TrustLevel::High);
    assert!(t(Some(300), false, 3).reasons.is_empty());
    assert_eq!(t(Some(299), false, 3).level, TrustLevel::Medium);
    assert_eq!(t(Some(100), false, 3).level, TrustLevel::Medium);
    let low = t(Some(99), false, 3);
    assert_eq!(low.level, TrustLevel::Low);
    assert_eq!(low.reasons, ["only 99 labelled objects (aim for 300+)"]);
    // Reference labels and a single timed run cap at medium; every reason is listed.
    let r = t(Some(500), true, 1);
    assert_eq!(r.level, TrustLevel::Medium);
    assert_eq!(r.reasons.len(), 2, "{:?}", r.reasons);
    assert!(
        r.reasons[0].contains("reference model (yolo26s)"),
        "{:?}",
        r.reasons
    );
    assert!(
        r.reasons[1].starts_with("1 timed run per image"),
        "{:?}",
        r.reasons
    );
    // The minimum over the rules.
    assert_eq!(t(Some(50), true, 1).level, TrustLevel::Low);
    // No accuracy: speed rules only, with the reason.
    let n = t(None, false, 3);
    assert_eq!(n.level, TrustLevel::High);
    assert_eq!(
        n.reasons,
        ["no accuracy data (no ground truth or reference model)"]
    );
    assert_eq!(t(None, false, 1).level, TrustLevel::Medium);
    // Fell back or did not run: low.
    let f = trust(&TrustInput {
        objects: Some(1000),
        repeat: 5,
        fell_back: Some(("openvino:cpu".into(), "openvino:gpu".into())),
        ..Default::default()
    });
    assert_eq!(f.level, TrustLevel::Low);
    assert_eq!(f.reasons, ["ran on openvino:cpu instead of openvino:gpu"]);
    let e = trust(&TrustInput {
        error: Some("compile failed".into()),
        ..Default::default()
    });
    assert_eq!((e.level, e.reasons.len()), (TrustLevel::Low, 1));
    assert_eq!(serde_json::to_value(TrustLevel::Medium).unwrap(), "medium");
    assert!(TrustLevel::Low < TrustLevel::Medium && TrustLevel::Medium < TrustLevel::High);
}

#[test]
fn model_and_device_trust_from_results() {
    let mut d = with_sweep(dev("openvino:cpu", 20.0, scene()), 3, &[(0.9, true)]);
    {
        let a = d.run.as_mut().unwrap().accuracy.as_mut().unwrap();
        a.overall.counts.tp = 150;
        a.overall.counts.fn_ = 30;
        a.relative = true;
        a.ground_truth = "relative to rt-detrv2-x on openvino:cpu, not ground truth".into();
    }
    d.run.as_mut().unwrap().repeat = 3;
    let m = model(vec![d, dev("ort:cpu", 30.0, scene())], None);
    let t = model_trust(&m);
    assert_eq!(t.level, TrustLevel::Medium);
    assert!(
        t.reasons[0].starts_with("180 labelled objects"),
        "{:?}",
        t.reasons
    );
    assert!(t.reasons[1].contains("(rt-detrv2-x)"), "{:?}", t.reasons);
    // The unscored device with one timed run.
    let t = device_trust(&m, &m.devices[1]);
    assert_eq!(t.level, TrustLevel::Medium);
    let failed = crate::benchmark::ModelResult::failed("x", "", "file missing".into());
    assert_eq!(model_trust(&failed).level, TrustLevel::Low);
}

#[test]
fn append_caps_and_pins() {
    let mut h = History::default();
    for i in 0..25 {
        h.append(entry(&format!("r{i:02}"), vec![]));
    }
    let ids: Vec<&str> = h
        .entries("ipcam-general")
        .iter()
        .map(|e| e.run_id.as_str())
        .collect();
    assert_eq!(ids.len(), MAX_ENTRIES_PER_MODEL);
    assert_eq!((ids[0], ids[19]), ("r05", "r24"));
    // A pinned baseline stays and does not count toward the cap.
    h.pin("IPCAM-GENERAL", "r05", true).unwrap();
    for i in 25..30 {
        h.append(entry(&format!("r{i:02}"), vec![]));
    }
    let list = h.entries("ipcam-general");
    assert_eq!(list.len(), MAX_ENTRIES_PER_MODEL + 1);
    assert_eq!(list[0].run_id, "r05");
    assert!(list[0].pinned);
    assert_eq!(list.iter().filter(|e| e.pinned).count(), 1);
    // Pinning another unpins it: 20 unpinned (r05 and r10..r28) plus the new pin.
    h.pin("ipcam-general", "r29", true).unwrap();
    let list = h.entries("ipcam-general");
    assert_eq!(list.len(), MAX_ENTRIES_PER_MODEL + 1);
    assert!(list.iter().all(|e| e.pinned == (e.run_id == "r29")));
    // The next run drops the oldest unpinned one, the old baseline.
    h.append(entry("r30", vec![]));
    assert!(h.find("ipcam-general", "r05").is_none());
    assert_eq!(h.entries("ipcam-general").len(), MAX_ENTRIES_PER_MODEL + 1);
    // Unpinning puts the model over the cap: the oldest goes.
    h.pin("ipcam-general", "r29", false).unwrap();
    assert_eq!(h.entries("ipcam-general").len(), MAX_ENTRIES_PER_MODEL);
    assert!(h.find("ipcam-general", "r10").is_none());
    h.pin("ipcam-general", "r29", true).unwrap();
    // Re-recording a run replaces it and keeps its pin.
    h.append(entry("r29", vec![snap("openvino:cpu", 1.0, None, 0)]));
    let e = h.entries("ipcam-general").last().unwrap();
    assert_eq!(
        (e.run_id.as_str(), e.pinned, e.devices.len()),
        ("r29", true, 1)
    );
    // Unknown runs and models are errors naming them.
    assert!(
        h.pin("ipcam-general", "nope", true)
            .unwrap_err()
            .to_string()
            .contains("nope")
    );
    assert!(h.pin("other", "r29", true).is_err());
    // Delete one, a model, everything.
    h.delete("ipcam-general", "r29").unwrap();
    assert!(h.delete("ipcam-general", "r29").is_err());
    let mut other = entry("x1", vec![]);
    other.model = "yolo26s".into();
    h.append(other);
    assert_eq!(h.delete_model("ipcam-general"), MAX_ENTRIES_PER_MODEL - 1);
    assert_eq!(h.delete_model("ipcam-general"), 0);
    assert_eq!(h.clear(), 1);
    assert!(h.models.is_empty());
}

/// Results with one finished model (two devices) and one that failed.
fn results_fixture(run_id: Option<&str>) -> BenchmarkResults {
    let mut ok = model(
        vec![
            dev("openvino:cpu", 20.0, scene()),
            dev("ort:cpu", 30.0, scene()),
        ],
        None,
    );
    ok.model = "IPcam-general".into();
    ok.run_id = run_id.map(str::to_string);
    let failed = crate::benchmark::ModelResult::failed("gone", "", "file missing".into());
    BenchmarkResults::new(
        HardwareSummary {
            os: "windows".into(),
            arch: "x86_64".into(),
            cpu: "i5".into(),
            gpus: vec!["#0 UHD 630 (Intel, integrated, 128 MB VRAM)".into()],
        },
        RuntimeVersions {
            openvino: Some("2026.0".into()),
            ..Default::default()
        },
        vec![ok, failed],
    )
}

#[test]
fn snapshots_and_seeding() {
    let r = results_fixture(Some("2026-10-16T14:02:11.000Z"));
    let e = HistoryEntry::of(&r.models[0], &r);
    assert_eq!(e.run_id, "2026-10-16T14:02:11.000Z");
    assert_eq!(e.hardware_key, "windows/x86_64/i5/#0 UHD 630");
    assert_eq!(e.runtimes.openvino.as_deref(), Some("2026.0"));
    assert_eq!(e.devices.len(), 2);
    let d = &e.devices[0];
    assert_eq!(
        (d.runtime.as_str(), d.total_p50_ms),
        ("openvino", Some(20.0))
    );
    assert_eq!(e.devices[1].runtime, "onnxruntime");
    assert!(d.recommended && !e.devices[1].recommended);
    assert_eq!(d.grade.as_deref(), Some("A"));
    assert_eq!(e.settings.labels, Labels::None);
    // Serialized like the contract (`fn`, nulls present).
    let v = serde_json::to_value(&e).unwrap();
    assert!(v["devices"][0]["fn"].is_null() && v["devices"][0]["ap50"].is_null());
    assert!(v["threshold"].is_null());

    // Seeding: the failed model is not recorded.
    let seeds = seed_entries(&r);
    assert_eq!(seeds.len(), 1);
    let h = History::seeded(Some(&r));
    assert_eq!(h.entries("ipcam-general").len(), 1);
    assert!(h.entries("gone").is_empty());
    assert!(History::seeded(None).models.is_empty());

    // Files of older versions: no `run_id`, the start time stands in.
    let mut v = serde_json::to_value(&r.models[0]).unwrap();
    v.as_object_mut().unwrap().remove("run_id");
    let old: crate::benchmark::ModelResult = serde_json::from_value(v).unwrap();
    assert_eq!(old.run_id, None);
    assert_eq!(run_key(&old), old.timestamp);
    assert_eq!(HistoryEntry::from(&old).run_id, old.timestamp);
}

#[test]
fn labels_and_reference_from_the_datasets() {
    let (results, _) = crate::benchmark::search::tests::fixture();
    let mut m = model(vec![dev("openvino:cpu", 20.0, scene())], None);
    m.datasets = vec!["night-set".into(), "pseudo-set".into()];
    let e = HistoryEntry::of(&m, &results);
    assert_eq!(e.settings.labels, Labels::Mixed);
    assert_eq!(e.settings.reference.as_deref(), Some("ref"));
    m.datasets = vec!["pseudo-set".into()];
    assert_eq!(
        HistoryEntry::of(&m, &results).settings.labels,
        Labels::Reference
    );
    m.datasets = vec!["night-set".into()];
    let e = HistoryEntry::of(&m, &results);
    assert_eq!(
        (e.settings.labels, e.settings.reference),
        (Labels::Human, None)
    );
    assert_eq!(
        super::reference_of_desc("relative to a on x, b on y, not ground truth").as_deref(),
        Some("a")
    );
    assert_eq!(super::reference_of_desc("ground truth"), None);
}

fn temp_dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("bop-hist-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn file_round_trip_and_lenient_load() {
    let dir = temp_dir();
    let results = dir.join("benchmark.json");
    let path = history_path(&results);
    assert_eq!(path, dir.join(HISTORY_FILE));
    assert!(History::load(&path).unwrap().is_none());
    let mut h = History::default();
    h.append(entry(
        "r1",
        vec![snap("openvino:cpu", 10.0, Some(0.5), 100)],
    ));
    h.append(entry("r2", vec![]));
    h.save(&path).unwrap();
    let back = History::load(&path).unwrap().unwrap();
    assert_eq!(back.models, h.models);
    assert_eq!(back.version, 1);
    // One bad entry: dropped with a warning naming it, the rest loads.
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    v["models"]["IPcam-general"][0]["devices"] = serde_json::json!("oops");
    std::fs::write(&path, v.to_string()).unwrap();
    let (back, warnings) = History::load_with_warnings(&path).unwrap();
    assert_eq!(back.unwrap().entries("ipcam-general").len(), 1);
    assert_eq!(warnings.len(), 1);
    assert!(
        warnings[0].contains("#1 of 'IPcam-general'"),
        "{}",
        warnings[0]
    );
    // Garbage: an error for a plain load, moved aside (not overwritten) by an update.
    std::fs::write(&path, "{nope").unwrap();
    assert!(History::load(&path).is_err());
    assert!(History::load_or_seed(&path, None).models.is_empty());
    let ((), warnings) = History::update(&path, None, |h| {
        h.append(entry("r3", vec![]));
        Ok(())
    })
    .unwrap();
    assert!(warnings[0].contains("moved it aside"), "{warnings:?}");
    assert_eq!(
        History::load(&path)
            .unwrap()
            .unwrap()
            .entries("IPcam-general")
            .len(),
        1
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn record_seeds_appends_once_per_model_and_skips_failures() {
    let dir = temp_dir();
    let results_file = dir.join("benchmark.json");
    let path = history_path(&results_file);
    // The results before the run (no run id: an older version wrote them) and the new run.
    let before = results_fixture(None);
    let mut after = results_fixture(Some("2026-10-16T15:00:00.000Z"));
    after.models[0].devices[0]
        .run
        .as_mut()
        .unwrap()
        .stages_ms
        .total
        .p50 = 15.0;

    // A failed model alone records nothing (no file).
    let w = record(
        &results_file,
        &seed_entries(&before),
        &after,
        &["gone".into()],
    );
    assert!(w.is_empty() && !path.exists(), "{w:?}");

    // The first record seeds the earlier result as the baseline, then appends the run.
    let w = record(
        &results_file,
        &seed_entries(&before),
        &after,
        &["IPcam-general".into(), "gone".into()],
    );
    assert!(w.is_empty(), "{w:?}");
    let h = History::load(&path).unwrap().unwrap();
    let list = h.entries("ipcam-general");
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].run_id, before.models[0].timestamp);
    assert_eq!(list[1].run_id, "2026-10-16T15:00:00.000Z");
    assert!(h.entries("gone").is_empty());
    // The new run against its seeded baseline: 25% faster on the CPU.
    let c = compare_with_history(list, &list[1]);
    assert!(c.comparable, "{c:?}");
    assert_eq!(metric(&c, "p50_ms").verdict, Verdict::Better);

    // Saving the same run again (per-model saves of one run) does not add an entry; a new run
    // appends one.
    record(
        &results_file,
        &seed_entries(&after),
        &after,
        &["IPcam-general".into()],
    );
    assert_eq!(
        History::load(&path)
            .unwrap()
            .unwrap()
            .entries("ipcam-general")
            .len(),
        2
    );
    let mut third = after.clone();
    third.models[0].run_id = Some("2026-10-16T16:00:00.000Z".into());
    record(
        &results_file,
        &seed_entries(&after),
        &third,
        &["IPcam-general".into()],
    );
    assert_eq!(
        History::load(&path)
            .unwrap()
            .unwrap()
            .entries("ipcam-general")
            .len(),
        3
    );

    // A history that cannot be written is a warning, never an error.
    let blocked = dir.join("file");
    std::fs::write(&blocked, "x").unwrap();
    let w = record(
        &blocked.join("benchmark.json"),
        &[],
        &third,
        &["IPcam-general".into()],
    );
    assert_eq!(w.len(), 1);
    assert!(
        w[0].contains("was not updated") && w[0].contains("writable"),
        "{}",
        w[0]
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn view_annotates_status_json() {
    let mut r = results_fixture(Some("2026-10-16T15:00:00.000Z"));
    r.models[1].run_id = Some("2026-10-16T14:00:00Z".into());
    assert_eq!(
        view::latest_run_id(&r).as_deref(),
        Some("2026-10-16T15:00:00.000Z")
    );
    // Later by time, not by text ("15:00:00Z" sorts after "15:00:00.500Z" as text).
    assert_eq!(
        view::time_cmp("2026-10-16T15:00:00.500Z", "2026-10-16T15:00:00Z"),
        std::cmp::Ordering::Greater
    );
    let mut h = History::default();
    let mut earlier = HistoryEntry::of(&r.models[0], &r);
    earlier.run_id = "2026-10-16T13:00:00.000Z".into();
    earlier.devices[0].total_p50_ms = Some(40.0);
    h.append(earlier);
    let mut v = serde_json::to_value(r.summary()).unwrap();
    view::annotate(&mut v, &r, &h);
    let m = &v["models"][0];
    assert_eq!(m["this_run"], true);
    assert_eq!(m["run_id"], "2026-10-16T15:00:00.000Z");
    assert_eq!(m["trust"]["level"], "medium");
    assert_eq!(m["changes"]["comparable"], true);
    assert_eq!(
        m["changes"]["baseline"]["run_id"],
        "2026-10-16T13:00:00.000Z"
    );
    assert!(
        m["changes"]["summary"]
            .as_str()
            .unwrap()
            .starts_with("50% faster")
    );
    assert_eq!(m["devices"][1]["runtime"], "onnxruntime");
    assert!(m["devices"][0]["trust"]["level"].is_string());
    // The failed model: not this run, no changes.
    let f = &v["models"][1];
    assert_eq!(
        (f["this_run"].clone(), f["changes"].clone()),
        (false.into(), serde_json::Value::Null)
    );
    let n = view::noise_json();
    assert_eq!(n["speed_rel"], 0.05);
    assert_eq!(n["trust_medium_objects"], 300);
    assert_eq!(n["min_class_gt"], 20);
    assert!((n["plateau_points"].as_f64().unwrap() - 0.2).abs() < 1e-9);
}

#[test]
fn ranking_rows_carry_runtime_tradeoff_and_trust() {
    let r = results_fixture(Some("2026-10-16T15:00:00.000Z"));
    let ranking = r.ranking();
    assert_eq!(ranking.len(), 1);
    let row = &ranking[0];
    assert_eq!(
        (row.device.as_str(), row.runtime.as_str()),
        ("openvino:cpu", "openvino")
    );
    assert!(row.best_tradeoff && row.this_run);
    assert_eq!(row.run_id.as_deref(), Some("2026-10-16T15:00:00.000Z"));
    assert_eq!(row.trust.level, TrustLevel::Medium);
    // Every device: the slower unscored CPU of ONNX Runtime is not a best trade-off.
    let points = r.points();
    assert_eq!(points.len(), 2);
    assert_eq!(points[0].device, "openvino:cpu");
    assert!(points[0].best_tradeoff && !points[1].best_tradeoff);
    assert_eq!(points[1].rank, 2);
    let v = serde_json::to_value(&points[1]).unwrap();
    assert!(v["recall"].is_null() && v["threshold"].is_null());
}

/// A device row whose AP averages `classes` (`(name, gt, ap50)`; AP50-95 = AP50 / 2) with
/// `ap_min_gt` 20 (None: an entry of an older version, no per-class values).
fn ap_snap(classes: &[(&str, usize, f64)], min_gt: Option<usize>, ap50: f64) -> DeviceSnap {
    let objects = classes.iter().map(|c| c.1).sum();
    let mut s = snap("openvino:cpu", 20.0, Some(0.8), objects);
    s.ap50 = Some(ap50);
    s.ap50_95 = Some(ap50 / 2.0);
    s.ap_min_gt = min_gt;
    if min_gt.is_some() {
        s.ap_classes = classes
            .iter()
            .map(|&(class, gt, ap)| ClassAp {
                class: class.into(),
                gt,
                ap50: Some(ap),
                ap50_95: Some(ap / 2.0),
            })
            .collect();
    }
    s
}

#[test]
fn macro_noise_hand_computed() {
    // Two classes at p 0.5 with 100 objects each: 2 * sqrt(0.0025 + 0.0025) / 2 = 0.0707.
    let n = macro_noise(&[(0.5, 100), (0.5, 100)]);
    assert!((n - (0.005f64).sqrt()).abs() < 1e-12, "{n}");
    // Perfect classes: the floor, one object of the smallest class: 1 / (2 * 20).
    assert!((macro_noise(&[(1.0, 950), (1.0, 20)]) - 1.0 / 40.0).abs() < 1e-12);
    // Classes without objects are left out; nothing at all is no noise.
    assert!((macro_noise(&[(1.0, 50), (0.3, 0)]) - 1.0 / 50.0).abs() < 1e-12);
    assert_eq!(macro_noise(&[]), 0.0);
}

/// The investigation's held-out set: person 178, car 953, truck 34, dog 7, bicycle 2, cat 2.
/// A dog going from 0.129 to 1.0 moved the old all-class mean by 14.5 points, "better" against
/// the noise of 1176 objects; averaged over the classes with 20 objects or more it moves nothing.
#[test]
fn a_rare_class_flip_is_no_longer_better() {
    let strong = [
        ("person", 178, 0.933),
        ("car", 953, 0.966),
        ("truck", 34, 0.947),
    ];
    let old_mean = |dog: f64| (0.933 + 0.966 + 0.947 + 0.505 + dog + 0.0) / 6.0;
    // Two entries of older versions: the old rule over every object (kept as it was).
    let mut a_old = ap_snap(&strong, None, old_mean(0.129));
    a_old.ground_truth = Some(1176);
    let mut b_old = a_old.clone();
    b_old.ap50 = Some(old_mean(1.0));
    let c = compare(
        &entry("r1", vec![a_old.clone()]),
        &entry("r2", vec![b_old.clone()]),
    );
    assert_eq!(metric(&c, "ap50").verdict, Verdict::Better, "the old rule");
    // The same runs scored with the supported-class rule: the dog is not in the average.
    let mean = (0.933 + 0.966 + 0.947) / 3.0;
    let a = ap_snap(&strong, Some(20), mean);
    let b = ap_snap(&strong, Some(20), mean);
    let c = compare(&entry("r1", vec![a.clone()]), &entry("r2", vec![b]));
    let m = metric(&c, "ap50");
    assert_eq!(m.verdict, Verdict::Same, "{}", m.why);
    assert!(
        m.why
            .contains("for the average of person (178 objects), car (953 objects)"),
        "{}",
        m.why
    );
    // A real change of the averaged classes still counts: truck 0.947 -> 0.70 moves the mean by
    // 8.2 points, over the noise (2 * sqrt(sum p (1 - p) / n) / 3 = 2.9 points).
    let mut worse = strong;
    worse[2].2 = 0.70;
    let b = ap_snap(&worse, Some(20), (0.933 + 0.966 + 0.70) / 3.0);
    let c = compare(&entry("r1", vec![a.clone()]), &entry("r2", vec![b]));
    let m = metric(&c, "ap50");
    assert_eq!(m.verdict, Verdict::Worse, "{}", m.why);
    assert!(
        m.why.contains("normal variation of 2.9 points"),
        "{}",
        m.why
    );
    // Two rare classes alone (fell back): a flip of one of 2 dog boxes (0.5 -> 1.0, the mean
    // +25 points) is within the macro noise (2 * sqrt(0.125 + 0.125) / 2 = 50 points).
    let rare = [("dog", 2, 0.5), ("cat", 2, 0.5)];
    let mut ra = ap_snap(&rare, Some(20), 0.5);
    ra.ap_fell_back = true;
    let mut rb = ap_snap(&[("dog", 2, 1.0), ("cat", 2, 0.5)], Some(20), 0.75);
    rb.ap_fell_back = true;
    let c = compare(&entry("r1", vec![ra]), &entry("r2", vec![rb]));
    assert_eq!(metric(&c, "ap50").verdict, Verdict::Same);
}

/// Runs that average other classes are not comparable on AP alone; the rest still compares.
#[test]
fn another_ap_basis_is_not_comparable_for_ap_only() {
    let two = [("person", 178, 0.93), ("car", 953, 0.96)];
    let three = [
        ("person", 178, 0.93),
        ("car", 953, 0.96),
        ("truck", 34, 0.9),
    ];
    let a = ap_snap(&two, Some(20), 0.945);
    let mut b = ap_snap(&three, Some(20), 0.93);
    b.recall = Some(0.95);
    let c = compare(&entry("r1", vec![a.clone()]), &entry("r2", vec![b.clone()]));
    assert!(c.comparable);
    for name in ["ap50", "ap50_95"] {
        let m = metric(&c, name);
        assert_eq!((m.verdict, m.delta), (Verdict::NotComparable, None));
    }
    assert_eq!(
        metric(&c, "ap50").why,
        "AP50 averaged over other classes (person, car -> person, car, truck)"
    );
    assert_eq!(metric(&c, "recall").verdict, Verdict::Better);
    assert_eq!(c.summary, "speed about the same; more accurate");
    // An entry of an older version beside a new one: "every class".
    let old = ap_snap(&two, None, 0.9);
    let m = metric(
        &compare(&entry("r1", vec![old]), &entry("r2", vec![a.clone()])),
        "ap50",
    )
    .clone();
    assert_eq!(m.verdict, Verdict::NotComparable);
    assert_eq!(
        m.why,
        "AP50 averaged over other classes (every class -> person, car)"
    );
    // The same classes in another order are the same basis; another minimum is not.
    let swapped = ap_snap(
        &[("car", 953, 0.96), ("person", 178, 0.93)],
        Some(20),
        0.945,
    );
    let c = compare(&entry("r1", vec![a.clone()]), &entry("r2", vec![swapped]));
    assert_eq!(metric(&c, "ap50").verdict, Verdict::Same);
    let mut min30 = a.clone();
    min30.ap_min_gt = Some(30);
    let m = metric(
        &compare(&entry("r1", vec![a.clone()]), &entry("r2", vec![min30])),
        "ap50",
    )
    .clone();
    assert_eq!(m.verdict, Verdict::NotComparable);
    assert_eq!(
        m.why,
        "AP50 averaged over classes with at least 20 objects -> at least 30 objects"
    );
}

/// False person alerts on frames without a person: a rate where lower is better.
#[test]
fn person_false_alerts_lower_is_better() {
    let with = |alerts: usize| {
        let mut s = snap("openvino:cpu", 20.0, Some(0.8), 300);
        s.person_free_frames = Some(200);
        s.person_false_alerts = Some(alerts);
        s
    };
    // 10 of 200 (5%) -> 2 of 200 (1%): 4 points down, noise 2 * sqrt(.05 * .95 / 200) = 3.1.
    let c = compare(&entry("r1", vec![with(10)]), &entry("r2", vec![with(2)]));
    let m = metric(&c, "person_false_alerts");
    assert_eq!(m.verdict, Verdict::Better, "{}", m.why);
    assert_eq!(
        m.why,
        "person false-alert rate down 4.0 points (more than the normal variation of 3.1 points \
         for 200 frames without a person)"
    );
    assert_eq!((m.before, m.after), (Some(0.05), Some(0.01)));
    // Counted with the accuracy measures.
    assert_eq!(c.summary, "speed about the same; more accurate");
    let m = metric(
        &compare(&entry("r1", vec![with(10)]), &entry("r2", vec![with(20)])),
        "person_false_alerts",
    )
    .clone();
    assert_eq!(m.verdict, Verdict::Worse, "{}", m.why);
    let m = metric(
        &compare(&entry("r1", vec![with(10)]), &entry("r2", vec![with(13)])),
        "person_false_alerts",
    )
    .clone();
    assert_eq!(m.verdict, Verdict::Same, "{}", m.why);
    // Lower-better directly: a rise is worse.
    let c = lower_proportion_change("x", "d", Some(0.1), Some(0.5), Some(100), "frames");
    assert_eq!(c.verdict, Verdict::Worse);
    // Neither run has it (older entries, no person class): not listed.
    let plain = snap("openvino:cpu", 20.0, Some(0.8), 300);
    let c = compare(&entry("r1", vec![plain.clone()]), &entry("r2", vec![plain]));
    assert!(c.metrics.iter().all(|m| m.metric != "person_false_alerts"));
}

/// The snapshot of a result records the AP's classes with their values and the person counts;
/// entries of older versions load without them.
#[test]
fn snapshots_record_the_ap_basis_and_person_alerts() {
    use crate::benchmark::metrics::{ApBasis, ClassAlerts, ClassMetrics, Summary};
    let mut m = model(vec![dev("openvino:cpu", 20.0, scene())], None);
    let class = |name: &str, gt: usize, ap: f64| ClassMetrics {
        class: name.into(),
        gt,
        ap50: Some(ap),
        ap50_95: Some(ap / 2.0),
        precision: None,
        recall: None,
    };
    m.devices[0].run.as_mut().unwrap().accuracy = Some(crate::benchmark::AccuracyReport {
        ground_truth: "ground truth".into(),
        relative: false,
        classes: vec!["person".into(), "car".into(), "dog".into()],
        threshold: 0.5,
        overall: Summary {
            ap50: Some(0.95),
            per_class: vec![
                class("person", 178, 0.93),
                class("car", 953, 0.97),
                class("dog", 7, 0.13),
            ],
            ap_basis: Some(ApBasis {
                min_gt: 20,
                classes: vec!["person".into(), "car".into()],
                fell_back: false,
            }),
            class_alerts: vec![ClassAlerts {
                class: "person".into(),
                frames_without: 210,
                alerts: 3,
            }],
            ..Default::default()
        },
        by_dataset: vec![],
        by_tag: vec![],
        all_classes: None,
        sweep: None,
        frame_roc: None,
    });
    let d = &HistoryEntry::from(&m).devices[0];
    assert_eq!(d.ap_min_gt, Some(20));
    assert!(!d.ap_fell_back);
    assert_eq!(
        d.ap_classes,
        [
            ClassAp {
                class: "person".into(),
                gt: 178,
                ap50: Some(0.93),
                ap50_95: Some(0.465)
            },
            ClassAp {
                class: "car".into(),
                gt: 953,
                ap50: Some(0.97),
                ap50_95: Some(0.485)
            },
        ]
    );
    assert_eq!(
        (d.person_free_frames, d.person_false_alerts),
        (Some(210), Some(3))
    );
    assert!((d.person_alert_rate().unwrap() - 3.0 / 210.0).abs() < 1e-12);
    let v = serde_json::to_value(d).unwrap();
    assert_eq!(v["ap_classes"][1]["gt"], 953);
    // An entry of an older version: none of them, and none written back.
    let old: DeviceSnap = serde_json::from_str(r#"{"device":"openvino:cpu","ap50":0.58}"#).unwrap();
    assert_eq!((old.ap_min_gt, old.person_free_frames), (None, None));
    assert!(old.ap_classes.is_empty() && !old.ap_fell_back);
    let back = serde_json::to_value(&old).unwrap();
    for k in [
        "ap_min_gt",
        "ap_classes",
        "ap_fell_back",
        "person_free_frames",
        "person_false_alerts",
    ] {
        assert!(back.get(k).is_none(), "{k}");
    }
}
