//! Whether two models' results can be set side by side ([`comparable_models`]): the model-vs-model
//! twin of [`crate::benchmark::history::comparable`], which compares two runs of one model.
//!
//! Two questions, answered apart, because a model on other hardware still finds the same objects:
//! - accuracy needs the same images and labels: the same datasets (in any order), image count,
//!   kind of labels and reference model. Hardware and runtimes do not matter;
//! - speed needs the same machine and runtime versions, and a device both ran on. Datasets matter
//!   only through accuracy: speed differs by image size, so the page times both on one image list.

use crate::benchmark::history::{HistoryEntry, Labels};
use crate::benchmark::report::{HardwareSummary, RuntimeVersions};
use serde::Serialize;

/// One side of a comparison: can it be made, and if not why (plain words, every reason).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Side {
    pub comparable: bool,
    pub reasons: Vec<String>,
}

impl Side {
    fn of(reasons: Vec<String>) -> Self {
        Self {
            comparable: reasons.is_empty(),
            reasons,
        }
    }
}

/// [`comparable_models`] of models `a` and `b`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelComparison {
    pub a: String,
    pub b: String,
    pub accuracy: Side,
    pub speed: Side,
}

fn sorted(v: &[String]) -> Vec<&str> {
    let mut s: Vec<&str> = v.iter().map(String::as_str).collect();
    s.sort_unstable();
    s
}

fn list(v: &[&str]) -> String {
    if v.is_empty() {
        "none".into()
    } else {
        v.join(", ")
    }
}

fn labels_word(l: Labels) -> &'static str {
    match l {
        Labels::Human => "human labels",
        Labels::Reference => "a reference model's labels",
        Labels::Mixed => "some human, some reference labels",
        Labels::None => "no labels",
    }
}

/// Why the accuracy of `a` and `b` cannot be compared (empty: it can).
pub fn accuracy_reasons(a: &HistoryEntry, b: &HistoryEntry) -> Vec<String> {
    let (sa, sb) = (&a.settings, &b.settings);
    let (da, db) = (sorted(&sa.datasets), sorted(&sb.datasets));
    let mut out = Vec::new();
    if da != db {
        out.push(format!(
            "different datasets: {} scored on {}, {} on {}",
            a.model,
            list(&da),
            b.model,
            list(&db)
        ));
    }
    if sa.images != sb.images {
        out.push(format!(
            "different image count: {} on {}, {} on {}",
            a.model, sa.images, b.model, sb.images
        ));
    }
    if sa.labels != sb.labels {
        out.push(format!(
            "different labels: {} with {}, {} with {}",
            a.model,
            labels_word(sa.labels),
            b.model,
            labels_word(sb.labels)
        ));
    }
    if sa.reference != sb.reference {
        out.push(format!(
            "different reference model: {} against {}, {} against {}",
            a.model,
            sa.reference.as_deref().unwrap_or("none"),
            b.model,
            sb.reference.as_deref().unwrap_or("none")
        ));
    }
    out
}

fn hardware_reason(
    a: &HardwareSummary,
    b: &HardwareSummary,
    key_a: &str,
    key_b: &str,
) -> Option<String> {
    a.difference(b).or_else(|| {
        (*a == HardwareSummary::default() && *b == HardwareSummary::default() && key_a != key_b)
            .then(|| format!("different hardware: {key_a} and {key_b}"))
    })
}

fn ort(r: &RuntimeVersions) -> String {
    format!(
        "{}{}",
        r.onnxruntime.as_deref().unwrap_or("none"),
        r.onnxruntime_flavor
            .as_deref()
            .map(|f| format!(" ({f})"))
            .unwrap_or_default()
    )
}

/// Why the speed of `a` and `b` cannot be compared (empty: it can).
pub fn speed_reasons(a: &HistoryEntry, b: &HistoryEntry) -> Vec<String> {
    let mut out = Vec::new();
    out.extend(hardware_reason(
        &a.hardware,
        &b.hardware,
        &a.hardware_key,
        &b.hardware_key,
    ));
    let uses = |e: &HistoryEntry, rt: &str| e.devices.iter().any(|d| d.ran() && d.runtime == rt);
    let (ra, rb) = (&a.runtimes, &b.runtimes);
    if uses(a, "openvino") && uses(b, "openvino") && ra.openvino != rb.openvino {
        out.push(format!(
            "different OpenVINO versions: {} and {}",
            ra.openvino.as_deref().unwrap_or("none"),
            rb.openvino.as_deref().unwrap_or("none")
        ));
    }
    if uses(a, "onnxruntime") && uses(b, "onnxruntime") && ort(ra) != ort(rb) {
        out.push(format!(
            "different ONNX Runtime versions: {} and {}",
            ort(ra),
            ort(rb)
        ));
    }
    let shared = a
        .devices
        .iter()
        .filter(|d| d.ran())
        .any(|d| b.device(&d.device).is_some_and(|x| x.ran()));
    if !shared {
        let names = |e: &HistoryEntry| {
            let v: Vec<&str> = e
                .devices
                .iter()
                .filter(|d| d.ran())
                .map(|d| d.device.as_str())
                .collect();
            list(&v)
        };
        out.push(format!(
            "no device in common: {} ran on {}, {} on {}",
            a.model,
            names(a),
            b.model,
            names(b)
        ));
    }
    out
}

/// Whether the results of models `a` and `b` compare (see the module docs). Pure.
pub fn comparable_models(a: &HistoryEntry, b: &HistoryEntry) -> ModelComparison {
    ModelComparison {
        a: a.model.clone(),
        b: b.model.clone(),
        accuracy: Side::of(accuracy_reasons(a, b)),
        speed: Side::of(speed_reasons(a, b)),
    }
}

/// Every pair of `entries`, each once, in the given order.
pub fn all_pairs(entries: &[HistoryEntry]) -> Vec<ModelComparison> {
    let mut out = Vec::new();
    for (i, a) in entries.iter().enumerate() {
        for b in entries.iter().skip(i + 1) {
            out.push(comparable_models(a, b));
        }
    }
    out
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::benchmark::history::{DeviceSnap, RunSettings};

    pub fn device(spec: &str, runtime: &str) -> DeviceSnap {
        DeviceSnap {
            device: spec.into(),
            runtime: runtime.into(),
            total_p50_ms: Some(10.0),
            ..DeviceSnap::default()
        }
    }

    pub fn entry(model: &str, datasets: &[&str], images: usize) -> HistoryEntry {
        HistoryEntry {
            run_id: "r".into(),
            timestamp: String::new(),
            model: model.into(),
            pinned: false,
            settings: RunSettings {
                datasets: datasets.iter().map(ToString::to_string).collect(),
                images,
                labels: Labels::Human,
                ..RunSettings::default()
            },
            hardware_key: "mac/arm64/m1".into(),
            hardware: HardwareSummary::default(),
            runtimes: RuntimeVersions {
                openvino: Some("2026.0".into()),
                onnxruntime: Some("1.24.4".into()),
                onnxruntime_flavor: Some("coreml".into()),
            },
            devices: vec![device("ort:coreml", "onnxruntime")],
            threshold: None,
            trained_on_share: None,
        }
    }

    #[test]
    fn same_images_compare_on_both_sides() {
        let c = comparable_models(&entry("a", &["x", "y"], 10), &entry("b", &["y", "x"], 10));
        assert!(c.accuracy.comparable, "{c:?}");
        assert!(c.speed.comparable, "{c:?}");
        assert!(c.accuracy.reasons.is_empty() && c.speed.reasons.is_empty());
    }

    #[test]
    fn other_datasets_or_counts_stop_accuracy_with_every_reason() {
        let c = comparable_models(&entry("a", &["x"], 10), &entry("b", &["y"], 20));
        assert!(!c.accuracy.comparable);
        assert_eq!(c.accuracy.reasons.len(), 2, "{c:?}");
        assert!(c.accuracy.reasons[0].contains("different datasets: a scored on x, b on y"));
        assert!(c.accuracy.reasons[1].contains("different image count"));
        // Speed is judged apart: it does not fail because the images differ.
        assert!(c.speed.comparable, "{c:?}");
    }

    #[test]
    fn labels_and_reference_stop_accuracy() {
        let mut b = entry("b", &["x"], 10);
        b.settings.labels = Labels::Reference;
        b.settings.reference = Some("rt-detrv2-x".into());
        let c = comparable_models(&entry("a", &["x"], 10), &b);
        assert!(!c.accuracy.comparable);
        assert!(c.accuracy.reasons.iter().any(|r| r.contains("labels")));
        assert!(c.accuracy.reasons.iter().any(|r| r.contains("rt-detrv2-x")));
    }

    #[test]
    fn hardware_flags_only_speed() {
        let (mut a, mut b) = (entry("a", &["x"], 10), entry("b", &["x"], 10));
        a.hardware.cpu = "Intel".into();
        b.hardware.cpu = "Apple M1".into();
        let c = comparable_models(&a, &b);
        assert!(c.accuracy.comparable, "accuracy stays comparable: {c:?}");
        assert!(!c.speed.comparable, "{c:?}");
    }

    #[test]
    fn runtime_versions_flag_only_speed_and_only_runtimes_both_use() {
        let (a, mut b) = (entry("a", &["x"], 10), entry("b", &["x"], 10));
        b.runtimes.onnxruntime = Some("1.31.0".into());
        let c = comparable_models(&a, &b);
        assert!(c.accuracy.comparable);
        assert!(c.speed.reasons[0].contains("ONNX Runtime"), "{c:?}");
        // OpenVINO differs too, but neither model ran on it.
        let (a2, mut b2) = (entry("a", &["x"], 10), entry("b", &["x"], 10));
        b2.runtimes.openvino = Some("2027.0".into());
        assert!(comparable_models(&a2, &b2).speed.comparable);
    }

    #[test]
    fn no_shared_device_flags_only_speed() {
        let (a, mut b) = (entry("a", &["x"], 10), entry("b", &["x"], 10));
        b.devices = vec![device("openvino:cpu", "openvino")];
        let c = comparable_models(&a, &b);
        assert!(c.accuracy.comparable);
        assert!(c.speed.reasons[0].contains("no device in common"), "{c:?}");
        // A device that failed does not count as shared.
        let mut d = entry("d", &["x"], 10);
        d.devices[0].error = Some("failed".into());
        assert!(!comparable_models(&a, &d).speed.comparable);
    }

    #[test]
    fn every_pair_once() {
        let e = [entry("a", &[], 0), entry("b", &[], 0), entry("c", &[], 0)];
        let p: Vec<(String, String)> = all_pairs(&e).into_iter().map(|c| (c.a, c.b)).collect();
        assert_eq!(p.len(), 3);
        assert_eq!(p[2], ("b".to_string(), "c".to_string()));
    }
}
