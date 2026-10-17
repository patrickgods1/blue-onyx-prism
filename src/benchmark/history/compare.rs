//! Whether two runs of a model can be compared ([`comparable`]) and what changed between them
//! ([`compare`]): per device and metric a [`MetricChange`] with a better / worse / same verdict
//! that only counts a change past the noise of the measurement, and a one-sentence summary.
//!
//! The rules are the named constants below (the page's glossary quotes them through
//! `GET /v1/benchmark`'s `noise`):
//! - speed: a change counts at `|delta| >= max(SPEED_REL * before, SPEED_ABS_MS)` (frames per
//!   second: the relative part only). [`SPEED_REL`] is the device-keep rule of the
//!   recommendation (`report::NOISE_MARGIN`);
//! - precision, recall, F1, AP, ROC AUC: a change counts past `SE_FACTOR` standard errors of a
//!   proportion, `SE_FACTOR * sqrt(p (1 - p) / n)` with `p` the earlier value and `n` the
//!   objects behind it, and never below one object's worth (`1 / n`, so a perfect earlier score
//!   does not make every single miss count). Rough: the counts are not independent samples;
//! - the best confidence threshold: no real change when the new value lies inside the earlier
//!   run's near-optimal range (it is that range's midpoint rounded down to 0.01, so small
//!   confidence shifts move it across a 0.01 boundary); otherwise better or worse by the best
//!   score, by more than [`super::super::threshold::PLATEAU`].

use super::{DeviceSnap, HistoryEntry, same_model};
use crate::benchmark::threshold::{Objective, PLATEAU};
use serde::{Deserialize, Serialize};

/// A speed change counts when it is at least this fraction of the earlier value...
pub const SPEED_REL: f64 = crate::benchmark::report::NOISE_MARGIN;
/// ...and at least this many milliseconds.
pub const SPEED_ABS_MS: f64 = 0.5;
/// An accuracy change counts past this many standard errors.
pub const SE_FACTOR: f64 = 2.0;

/// What a change means for the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Better,
    Worse,
    Same,
    NotComparable,
}

/// One metric of one device (or the model's threshold) before and after.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricChange {
    /// `p50_ms`, `infer_p50_ms`, `fps`, `ap50`, `ap50_95`, `roc_auc`, `precision`, `recall`,
    /// `f1`, `threshold`.
    pub metric: String,
    /// None for `threshold` (model level).
    pub device: Option<String>,
    pub before: Option<f64>,
    pub after: Option<f64>,
    /// `after - before` in raw units.
    pub delta: Option<f64>,
    /// `delta / before` for the speed metrics.
    pub rel: Option<f64>,
    pub verdict: Verdict,
    /// A plain sentence, always set.
    pub why: String,
}

/// The run a comparison is against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Baseline {
    pub run_id: String,
    pub timestamp: String,
    pub pinned: bool,
}

/// What changed between two runs of a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Changes {
    /// None: there is no earlier run.
    pub baseline: Option<Baseline>,
    pub comparable: bool,
    /// Why the runs are not comparable (empty when they are).
    pub reasons: Vec<String>,
    /// Empty when not comparable or without a baseline.
    pub metrics: Vec<MetricChange>,
    /// One plain sentence for the banner.
    pub summary: String,
}

impl Changes {
    /// No earlier run.
    pub fn first_run() -> Self {
        Self {
            baseline: None,
            comparable: false,
            reasons: Vec::new(),
            metrics: Vec::new(),
            summary: "first recorded run of this model".into(),
        }
    }
}

fn list(v: &[String]) -> String {
    if v.is_empty() {
        "none".into()
    } else {
        v.join(", ")
    }
}

fn opt(v: &Option<String>) -> &str {
    v.as_deref().unwrap_or("none")
}

/// Whether a device row uses a runtime.
fn uses(e: &HistoryEntry, runtime: &str) -> bool {
    e.devices.iter().any(|d| d.runtime == runtime)
}

/// Whether `a` (earlier) and `b` (later) measured the same thing: the same model, datasets,
/// image count, labels and reference model, hardware ([`crate::benchmark::report::
/// HardwareSummary::difference`], like `benchmark.json`'s merge), the versions of the runtimes
/// their devices used, and at least one device that ran in both. The error lists every reason,
/// in plain words.
pub fn comparable(a: &HistoryEntry, b: &HistoryEntry) -> Result<(), Vec<String>> {
    let mut reasons = Vec::new();
    let (sa, sb) = (&a.settings, &b.settings);
    if !same_model(&a.model, &b.model) {
        reasons.push(format!("different models: {} -> {}", a.model, b.model));
    }
    if sa.datasets != sb.datasets {
        reasons.push(format!(
            "different datasets: {} -> {}",
            list(&sa.datasets),
            list(&sb.datasets)
        ));
    }
    if sa.images != sb.images {
        reasons.push(format!(
            "different image count: {} -> {}",
            sa.images, sb.images
        ));
    }
    if sa.labels != sb.labels {
        let l = |x: super::Labels| {
            serde_json::to_value(x)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default()
        };
        reasons.push(format!(
            "different labels: {} -> {}",
            l(sa.labels),
            l(sb.labels)
        ));
    }
    if sa.reference != sb.reference {
        reasons.push(format!(
            "different reference model: {} -> {}",
            opt(&sa.reference),
            opt(&sb.reference)
        ));
    }
    if let Some(d) = a.hardware.difference(&b.hardware) {
        reasons.push(d);
    } else if a.hardware == Default::default()
        && b.hardware == Default::default()
        && a.hardware_key != b.hardware_key
    {
        reasons.push(format!(
            "the hardware changed ({} -> {})",
            a.hardware_key, b.hardware_key
        ));
    }
    let (ra, rb) = (&a.runtimes, &b.runtimes);
    if uses(a, "openvino") && uses(b, "openvino") && ra.openvino != rb.openvino {
        reasons.push(format!(
            "OpenVINO changed: {} -> {}",
            opt(&ra.openvino),
            opt(&rb.openvino)
        ));
    }
    if uses(a, "onnxruntime")
        && uses(b, "onnxruntime")
        && (ra.onnxruntime != rb.onnxruntime || ra.onnxruntime_flavor != rb.onnxruntime_flavor)
    {
        let v = |r: &crate::benchmark::report::RuntimeVersions| {
            format!(
                "{}{}",
                opt(&r.onnxruntime),
                r.onnxruntime_flavor
                    .as_deref()
                    .map(|f| format!(" ({f})"))
                    .unwrap_or_default()
            )
        };
        reasons.push(format!("ONNX Runtime changed: {} -> {}", v(ra), v(rb)));
    }
    if reasons.is_empty() && common_devices(a, b).is_empty() {
        reasons.push("no device ran in both runs".into());
    }
    if reasons.is_empty() {
        Ok(())
    } else {
        Err(reasons)
    }
}

/// The devices that ran (on themselves) in both runs, in `b`'s order.
fn common_devices<'a>(
    a: &'a HistoryEntry,
    b: &'a HistoryEntry,
) -> Vec<(&'a DeviceSnap, &'a DeviceSnap)> {
    b.devices
        .iter()
        .filter(|d| d.ran())
        .filter_map(|db| Some((a.device(&db.device).filter(|d| d.ran())?, db)))
        .collect()
}

fn pct(v: f64) -> String {
    format!("{:.0}%", v * 100.0)
}

/// Points (percent of 1) with one decimal.
fn points(v: f64) -> String {
    format!("{:.1}", v * 100.0)
}

/// A speed metric: lower is better for ms (`higher_better` false), higher for fps.
pub fn speed_change(
    metric: &str,
    device: &str,
    before: Option<f64>,
    after: Option<f64>,
    higher_better: bool,
) -> MetricChange {
    let mut c = MetricChange {
        metric: metric.to_string(),
        device: Some(device.to_string()),
        before,
        after,
        delta: None,
        rel: None,
        verdict: Verdict::Same,
        why: String::new(),
    };
    let (Some(b), Some(a)) = (
        before.filter(|v| v.is_finite()),
        after.filter(|v| v.is_finite()),
    ) else {
        c.why = format!(
            "no {} in the {} run",
            metric_name(metric),
            if before.is_none() { "earlier" } else { "later" }
        );
        return c;
    };
    let delta = a - b;
    c.delta = Some(delta);
    c.rel = (b > 0.0).then(|| delta / b);
    let (noise, floor) = if higher_better {
        (SPEED_REL * b, format!("{} noise floor", pct(SPEED_REL)))
    } else {
        (
            (SPEED_REL * b).max(SPEED_ABS_MS),
            format!("{} / {SPEED_ABS_MS} ms noise floor", pct(SPEED_REL)),
        )
    };
    let rel = c.rel.unwrap_or(0.0).abs();
    let improved = if higher_better {
        delta > 0.0
    } else {
        delta < 0.0
    };
    let word = if improved { "faster" } else { "slower" };
    if delta.abs() >= noise && delta != 0.0 {
        c.verdict = if improved {
            Verdict::Better
        } else {
            Verdict::Worse
        };
        c.why = format!("{} {word} (more than the {floor})", pct(rel));
    } else if delta == 0.0 {
        c.why = format!("unchanged (within the {floor})");
    } else {
        c.why = format!("about the same: {} {word}, within the {floor}", pct(rel));
    }
    c
}

/// A proportion metric (higher is better) behind `n` objects / detections / frames.
pub fn proportion_change(
    metric: &str,
    device: &str,
    before: Option<f64>,
    after: Option<f64>,
    n: Option<usize>,
    unit: &str,
) -> MetricChange {
    let mut c = MetricChange {
        metric: metric.to_string(),
        device: Some(device.to_string()),
        before,
        after,
        delta: None,
        rel: None,
        verdict: Verdict::Same,
        why: String::new(),
    };
    let name = metric_name(metric);
    let (Some(b), Some(a)) = (
        before.filter(|v| v.is_finite()),
        after.filter(|v| v.is_finite()),
    ) else {
        c.why = format!(
            "no {name} in the {} run",
            if before.is_none() { "earlier" } else { "later" }
        );
        return c;
    };
    let delta = a - b;
    c.delta = Some(delta);
    let Some(n) = n.filter(|n| *n > 0) else {
        c.why = format!("no {unit} behind the {name}, so a change cannot be judged");
        return c;
    };
    let noise = proportion_noise(b, n);
    let dir = if delta > 0.0 { "up" } else { "down" };
    if delta.abs() > noise {
        c.verdict = if delta > 0.0 {
            Verdict::Better
        } else {
            Verdict::Worse
        };
        c.why = format!(
            "{name} {dir} {} points (more than the normal variation of {} points for {n} {unit})",
            points(delta.abs()),
            points(noise)
        );
    } else {
        c.why = format!(
            "within normal variation for {n} {unit} ({name} {dir} {} points, noise {} points)",
            points(delta.abs()),
            points(noise)
        );
    }
    c
}

/// `SE_FACTOR * sqrt(p (1 - p) / n)`, at least one of `n` (`1 / n`).
pub fn proportion_noise(p: f64, n: usize) -> f64 {
    let n = n as f64;
    let p = p.clamp(0.0, 1.0);
    (SE_FACTOR * (p * (1.0 - p) / n).sqrt()).max(1.0 / n)
}

fn metric_name(metric: &str) -> &'static str {
    match metric {
        "p50_ms" => "time per image (p50)",
        "infer_p50_ms" => "inference time (p50)",
        "fps" => "images per second",
        "ap50" => "AP50",
        "ap50_95" => "AP50-95",
        "roc_auc" => "ROC AUC",
        "precision" => "precision",
        "recall" => "recall",
        "f1" => "F1",
        "threshold" => "best confidence threshold",
        _ => "value",
    }
}

/// The metrics of one device present in both runs.
fn device_changes(a: &DeviceSnap, b: &DeviceSnap) -> Vec<MetricChange> {
    let dev = b.device.as_str();
    let objects = a.ground_truth;
    let detections = match (a.tp, a.fp) {
        (Some(tp), Some(fp)) => Some(tp + fp),
        _ => None,
    };
    let mut v = vec![
        speed_change("p50_ms", dev, a.total_p50_ms, b.total_p50_ms, false),
        speed_change("infer_p50_ms", dev, a.infer_p50_ms, b.infer_p50_ms, false),
        speed_change("fps", dev, a.fps, b.fps, true),
    ];
    let acc = [
        ("ap50", a.ap50, b.ap50, objects, "objects"),
        ("ap50_95", a.ap50_95, b.ap50_95, objects, "objects"),
        (
            "roc_auc",
            a.roc_auc,
            b.roc_auc,
            a.frames.or(objects),
            if a.frames.is_some() {
                "frames"
            } else {
                "objects"
            },
        ),
        (
            "precision",
            a.precision,
            b.precision,
            detections,
            "detections",
        ),
        ("recall", a.recall, b.recall, objects, "objects"),
        ("f1", a.f1, b.f1, objects, "objects"),
    ];
    for (m, x, y, n, unit) in acc {
        // A metric neither run has (no ground truth, no frames) says nothing.
        if x.is_none() && y.is_none() {
            continue;
        }
        v.push(proportion_change(m, dev, x, y, n, unit));
    }
    v
}

/// The best-threshold change of the model, or None when neither run has advice.
pub fn threshold_change(
    a: Option<&super::ThresholdSnap>,
    b: Option<&super::ThresholdSnap>,
) -> Option<MetricChange> {
    if a.is_none() && b.is_none() {
        return None;
    }
    let before = a.and_then(|t| t.best).map(f64::from);
    let after = b.and_then(|t| t.best).map(f64::from);
    let mut c = MetricChange {
        metric: "threshold".into(),
        device: None,
        before,
        after,
        delta: None,
        rel: None,
        verdict: Verdict::Same,
        why: String::new(),
    };
    let (Some(ta), Some(tb)) = (a, b) else {
        c.why = format!(
            "no best confidence threshold in the {} run",
            if a.is_none() { "earlier" } else { "later" }
        );
        return Some(c);
    };
    if ta.objective != tb.objective {
        c.verdict = Verdict::NotComparable;
        c.why = format!(
            "the threshold goal changed ({} -> {}), so the picks are not comparable",
            ta.objective, tb.objective
        );
        return Some(c);
    }
    let (Some(x), Some(y)) = (ta.best, tb.best) else {
        c.why = format!(
            "no best confidence threshold in the {} run",
            if ta.best.is_none() {
                "earlier"
            } else {
                "later"
            }
        );
        return Some(c);
    };
    let delta = f64::from(y) - f64::from(x);
    c.delta = Some(delta);
    let goal = ta
        .objective
        .parse::<Objective>()
        .unwrap_or_default()
        .describe();
    let score_diff = match (ta.exact_score, tb.exact_score) {
        (Some(sa), Some(sb)) => Some(sb - sa),
        _ => None,
    };
    if delta.abs() < 1e-6 {
        c.why = format!("unchanged at {y:.2}");
    } else if let Some([lo, hi]) = ta.plateau
        && y >= lo - 1e-6
        && y <= hi + 1e-6
    {
        c.why = format!(
            "no real change (rounding): {y:.2} is inside the earlier near-optimal range \
             {lo:.2}-{hi:.2}, where every threshold scores within {} points of the best; the \
             pick is that range's midpoint rounded down to 0.01",
            points(PLATEAU)
        );
    } else if let Some(d) = score_diff {
        if d.abs() <= PLATEAU {
            c.why = format!(
                "the best {goal} moved by {} points, within the {}-point tolerance",
                points(d.abs()),
                points(PLATEAU)
            );
        } else {
            c.verdict = if d > 0.0 {
                Verdict::Better
            } else {
                Verdict::Worse
            };
            c.why = format!(
                "the best {goal} went {} {} points (to {})",
                if d > 0.0 { "up" } else { "down" },
                points(d.abs()),
                points(tb.exact_score.unwrap_or_default())
            );
        }
    } else {
        c.why = format!("{x:.2} -> {y:.2}; no score recorded to judge it");
    }
    Some(c)
}

/// What changed from `a` (earlier) to `b` (later). Not comparable: the reasons and no
/// metrics.
pub fn compare(a: &HistoryEntry, b: &HistoryEntry) -> Changes {
    let baseline = Some(Baseline {
        run_id: a.run_id.clone(),
        timestamp: a.timestamp.clone(),
        pinned: a.pinned,
    });
    if let Err(reasons) = comparable(a, b) {
        let summary = format!(
            "not comparable with the earlier run: {}",
            reasons.first().map_or("", String::as_str)
        );
        return Changes {
            baseline,
            comparable: false,
            reasons,
            metrics: Vec::new(),
            summary,
        };
    }
    let mut metrics: Vec<MetricChange> = common_devices(a, b)
        .into_iter()
        .flat_map(|(x, y)| device_changes(x, y))
        .collect();
    metrics.extend(threshold_change(a.threshold.as_ref(), b.threshold.as_ref()));
    let summary = summary(b, &metrics);
    Changes {
        baseline,
        comparable: true,
        reasons: Vec::new(),
        metrics,
        summary,
    }
}

/// "9% faster; accuracy about the same; confidence 0.41 -> 0.42, no real change (rounding)",
/// on `b`'s main device ([`HistoryEntry::main_device`], else the first compared one).
fn summary(b: &HistoryEntry, metrics: &[MetricChange]) -> String {
    let main = b
        .main_device()
        .map(|d| d.device.clone())
        .filter(|d| metrics.iter().any(|m| m.device.as_deref() == Some(d)))
        .or_else(|| metrics.iter().find_map(|m| m.device.clone()));
    let on = |m: &MetricChange| m.device.as_deref() == main.as_deref();
    let mut parts = Vec::new();
    if let Some(s) = metrics.iter().find(|m| on(m) && m.metric == "p50_ms") {
        let rel = s.rel.unwrap_or(0.0).abs();
        parts.push(match s.verdict {
            Verdict::Better => format!("{} faster", pct(rel)),
            Verdict::Worse => format!("{} slower", pct(rel)),
            _ => "speed about the same".to_string(),
        });
    }
    let acc: Vec<&MetricChange> = metrics
        .iter()
        .filter(|m| on(m) && matches!(m.metric.as_str(), "ap50" | "precision" | "recall" | "f1"))
        .collect();
    if !acc.is_empty() {
        let better = acc.iter().any(|m| m.verdict == Verdict::Better);
        let worse = acc.iter().any(|m| m.verdict == Verdict::Worse);
        parts.push(
            match (better, worse) {
                (true, false) => "more accurate",
                (false, true) => "less accurate",
                (true, true) => "accuracy mixed (some measures better, some worse)",
                _ => "accuracy about the same",
            }
            .to_string(),
        );
    }
    if let Some(t) = metrics.iter().find(|m| m.metric == "threshold")
        && let (Some(x), Some(y)) = (t.before, t.after)
        && (x - y).abs() > 1e-6
    {
        parts.push(match t.verdict {
            Verdict::Same if t.why.starts_with("no real change") => {
                format!("confidence {x:.2} -> {y:.2}, no real change (rounding)")
            }
            Verdict::Better => format!("confidence {x:.2} -> {y:.2}, a better best score"),
            Verdict::Worse => format!("confidence {x:.2} -> {y:.2}, a lower best score"),
            Verdict::NotComparable => format!("confidence {x:.2} -> {y:.2}, another goal"),
            Verdict::Same => format!("confidence {x:.2} -> {y:.2}, about the same score"),
        });
    }
    if parts.is_empty() {
        "nothing to compare".into()
    } else {
        parts.join("; ")
    }
}

/// The changes of `current` against the history of its model (`entries`, oldest first): the
/// pinned baseline, else the newest earlier entry that is comparable, else the newest earlier
/// one (then not comparable). Earlier = before `current` in the list when it is there, else
/// every entry with another run id.
pub fn compare_with_history(entries: &[HistoryEntry], current: &HistoryEntry) -> Changes {
    let earlier: &[HistoryEntry] = match entries.iter().position(|e| e.run_id == current.run_id) {
        Some(i) => entries.get(..i).unwrap_or(&[]),
        None => entries,
    };
    let candidates: Vec<&HistoryEntry> = earlier
        .iter()
        .filter(|e| e.run_id != current.run_id)
        .collect();
    let pinned = entries
        .iter()
        .find(|e| e.pinned && e.run_id != current.run_id);
    let base = pinned
        .or_else(|| {
            candidates
                .iter()
                .rev()
                .find(|e| comparable(e, current).is_ok())
                .copied()
        })
        .or_else(|| candidates.last().copied());
    match base {
        Some(b) => compare(b, current),
        None => Changes::first_run(),
    }
}
