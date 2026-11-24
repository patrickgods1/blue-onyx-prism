//! Whether two runs of a model can be compared ([`comparable`]) and what changed between them
//! ([`compare`]): per device and metric a [`MetricChange`] with a better / worse / same verdict
//! that only counts a change past the noise of the measurement, and a one-sentence summary.
//!
//! The rules are the named constants below (the page's glossary quotes them through
//! `GET /v1/benchmark`'s `noise`):
//! - speed: a change counts at `|delta| >= max(SPEED_REL * before, SPEED_ABS_MS)` (frames per
//!   second: the relative part only). [`SPEED_REL`] is the device-keep rule of the
//!   recommendation (`report::NOISE_MARGIN`);
//! - precision, recall, F1, ROC AUC: a change counts past `SE_FACTOR` standard errors of a
//!   proportion, `SE_FACTOR * sqrt(p (1 - p) / n)` with `p` the earlier value and `n` the
//!   objects behind it, and never below one object's worth (`1 / n`, so a perfect earlier score
//!   does not make every single miss count). Rough: the counts are not independent samples;
//! - AP50 and AP50-95 are class averages (over the classes with at least
//!   [`crate::benchmark::metrics::MIN_CLASS_GT`] objects, [`DeviceSnap::ap_classes`]): a change
//!   counts past [`macro_noise`] of the earlier run's per-class values, the standard error of a
//!   mean of `K` proportions, `SE_FACTOR * sqrt(sum p (1 - p) / n) / K`, never below one object
//!   of the smallest class (`1 / (K min n)`). Judging it by every object instead let a dog of
//!   2 boxes flipping from missed to found read as "better". Two runs that average other classes
//!   (another class set or minimum) are not comparable on AP alone, the rest still is; two
//!   entries of older versions (no `ap_min_gt`) keep the proportion rule over every object;
//! - false person alerts on frames without a person ([`DeviceSnap::person_free_frames`]): the
//!   proportion rule over those frames, lower is better; counted with the accuracy measures;
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
    /// `f1`, `person_false_alerts` (a rate), `threshold`.
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
    // The same sets in another order are the same images (QA-065); the text keeps each order.
    let set_of = |v: &[String]| {
        let mut v = v.to_vec();
        v.sort();
        v.dedup();
        v
    };
    if set_of(&sa.datasets) != set_of(&sb.datasets) {
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
    directed_proportion_change(metric, device, before, after, n, unit, true)
}

/// [`proportion_change`] of a rate where lower is better (false alerts).
pub fn lower_proportion_change(
    metric: &str,
    device: &str,
    before: Option<f64>,
    after: Option<f64>,
    n: Option<usize>,
    unit: &str,
) -> MetricChange {
    directed_proportion_change(metric, device, before, after, n, unit, false)
}

fn directed_proportion_change(
    metric: &str,
    device: &str,
    before: Option<f64>,
    after: Option<f64>,
    n: Option<usize>,
    unit: &str,
    higher_better: bool,
) -> MetricChange {
    let name = metric_name(metric);
    judged(metric, device, before, after, higher_better, |b| {
        let n = n
            .filter(|n| *n > 0)
            .ok_or_else(|| format!("no {unit} behind the {name}, so a change cannot be judged"))?;
        Ok((proportion_noise(b, n), format!("for {n} {unit}")))
    })
}

/// A metric change judged against a noise: `noise(before)` gives the noise and the words of what
/// it is "for" (`for 300 objects`), or why no noise can be judged (verdict Same).
fn judged(
    metric: &str,
    device: &str,
    before: Option<f64>,
    after: Option<f64>,
    higher_better: bool,
    noise: impl FnOnce(f64) -> Result<(f64, String), String>,
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
    let (noise, scope) = match noise(b) {
        Ok(x) => x,
        Err(why) => {
            c.why = why;
            return c;
        }
    };
    let dir = if delta > 0.0 { "up" } else { "down" };
    if delta.abs() > noise {
        c.verdict = if (delta > 0.0) == higher_better {
            Verdict::Better
        } else {
            Verdict::Worse
        };
        c.why = format!(
            "{name} {dir} {} points (more than the normal variation of {} points {scope})",
            points(delta.abs()),
            points(noise)
        );
    } else {
        c.why = format!(
            "within normal variation {scope} ({name} {dir} {} points, noise {} points)",
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

/// The noise of a mean of `K` per-class proportions `(p, n)` (a class average such as AP50):
/// `max(SE_FACTOR * sqrt(sum p (1 - p) / n) / K, 1 / (K * min n))`. The standard error of a mean
/// of independent proportions is `sqrt(sum p (1 - p) / n) / K`; the floor is one object of the
/// smallest class, which moves the mean by `1 / (K n)`. Classes with `n` 0 are left out; 0
/// without any class.
pub fn macro_noise(classes: &[(f64, usize)]) -> f64 {
    let used: Vec<(f64, f64)> = classes
        .iter()
        .filter(|(p, n)| *n > 0 && p.is_finite())
        .map(|&(p, n)| (p.clamp(0.0, 1.0), n as f64))
        .collect();
    let Some(min_n) = used.iter().map(|u| u.1).reduce(f64::min) else {
        return 0.0;
    };
    let k = used.len() as f64;
    let var: f64 = used.iter().map(|(p, n)| p * (1.0 - p) / n).sum();
    (SE_FACTOR * var.sqrt() / k).max(1.0 / (k * min_n))
}

/// What an AP of `s` averages, in words: `every class` (an entry of an older version), `car,
/// person`, or `dog, cat (none had 20 objects)`.
fn basis_words(s: &DeviceSnap) -> String {
    match s.ap_min_gt {
        None => "every class".into(),
        Some(min) => {
            let names: Vec<&str> = s.ap_classes.iter().map(|c| c.class.as_str()).collect();
            if s.ap_fell_back {
                format!("every class ({}; none had {min} objects)", names.join(", "))
            } else {
                names.join(", ")
            }
        }
    }
}

/// The classes of `s`'s AP average, sorted (None: an entry of an older version).
fn basis_key(s: &DeviceSnap) -> Option<(usize, bool, Vec<&str>)> {
    let min = s.ap_min_gt?;
    let mut names: Vec<&str> = s.ap_classes.iter().map(|c| c.class.as_str()).collect();
    names.sort_unstable();
    Some((min, s.ap_fell_back, names))
}

/// An AP change (`ap50` or `ap50_95`, `before` / `after` its values in `a` / `b`). Two entries of
/// older versions: [`proportion_change`] behind every object. One that averages other classes
/// (another [`DeviceSnap::ap_min_gt`] or class set): not comparable. The same classes: judged by
/// [`macro_noise`] over the earlier run's per-class values and objects.
fn ap_change(metric: &str, a: &DeviceSnap, b: &DeviceSnap) -> MetricChange {
    let pick = |s: &DeviceSnap| if metric == "ap50" { s.ap50 } else { s.ap50_95 };
    let (before, after) = (pick(a), pick(b));
    let dev = b.device.as_str();
    let (ka, kb) = (basis_key(a), basis_key(b));
    if ka.is_none() && kb.is_none() {
        return proportion_change(metric, dev, before, after, a.ground_truth, "objects");
    }
    let name = metric_name(metric);
    if ka != kb {
        let (wa, wb) = (basis_words(a), basis_words(b));
        let why = if wa == wb {
            format!(
                "{name} averaged over classes with at least {} objects -> at least {} objects",
                a.ap_min_gt.map_or_else(|| "any".into(), |m| m.to_string()),
                b.ap_min_gt.map_or_else(|| "any".into(), |m| m.to_string())
            )
        } else {
            format!("{name} averaged over other classes ({wa} -> {wb})")
        };
        return MetricChange {
            metric: metric.to_string(),
            device: Some(dev.to_string()),
            before,
            after,
            delta: None,
            rel: None,
            verdict: Verdict::NotComparable,
            why,
        };
    }
    let classes: Vec<(f64, usize, &str)> = a
        .ap_classes
        .iter()
        .filter_map(|c| {
            let p = if metric == "ap50" { c.ap50 } else { c.ap50_95 };
            Some((p?, c.gt, c.class.as_str()))
        })
        .filter(|(p, n, _)| *n > 0 && p.is_finite())
        .collect();
    if classes.is_empty() {
        return proportion_change(metric, dev, before, after, a.ground_truth, "objects");
    }
    judged(metric, dev, before, after, true, |_| {
        let noise = macro_noise(&classes.iter().map(|c| (c.0, c.1)).collect::<Vec<_>>());
        let list: Vec<String> = classes
            .iter()
            .map(|(_, n, name)| format!("{name} ({n} objects)"))
            .collect();
        Ok((noise, format!("for the average of {}", list.join(", "))))
    })
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
        "person_false_alerts" => "person false-alert rate",
        "threshold" => "best confidence threshold",
        "goal" => "goal metric",
        "live_requests_per_min" => "live load",
        "burst_p95_ms" => "burst p95",
        _ => "value",
    }
}

/// Images per second of one device: not comparable when one run read ahead and the other did
/// not (the throughput of a pipelined run is higher by construction).
fn fps_change(dev: &str, a: &DeviceSnap, b: &DeviceSnap) -> MetricChange {
    let mut c = speed_change("fps", dev, a.fps, b.fps, true);
    if a.pipelined != b.pipelined && c.delta.is_some() {
        let how = |p: bool| {
            if p {
                "pipelined"
            } else {
                "one image at a time"
            }
        };
        c.verdict = Verdict::NotComparable;
        c.delta = None;
        c.rel = None;
        c.why = format!(
            "images per second measured another way ({} -> {})",
            how(a.pipelined),
            how(b.pipelined)
        );
    }
    c
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
        fps_change(dev, a, b),
    ];
    for m in ["ap50", "ap50_95"] {
        let (x, y) = if m == "ap50" {
            (a.ap50, b.ap50)
        } else {
            (a.ap50_95, b.ap50_95)
        };
        // A metric neither run has (no ground truth) says nothing.
        if x.is_some() || y.is_some() {
            v.push(ap_change(m, a, b));
        }
    }
    let acc = [
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
    if let Some(c) = goal_change(dev, a, b) {
        v.push(c);
    }
    if a.live_requests_per_min.is_some() || b.live_requests_per_min.is_some() {
        v.push(live_load_change(dev, a, b));
    }
    if a.burst_p95_ms.is_some() || b.burst_p95_ms.is_some() {
        v.push(burst_change(dev, a, b));
    }
    // False person alerts on frames without a person: a rate, lower is better.
    let (ra, rb) = (a.person_alert_rate(), b.person_alert_rate());
    if ra.is_some() || rb.is_some() {
        v.push(lower_proportion_change(
            "person_false_alerts",
            dev,
            ra,
            rb,
            a.person_free_frames,
            "frames without a person",
        ));
    }
    v
}

/// The goal metric of two runs of one device ([`DeviceSnap::goal`]): better or worse only when
/// their 95% intervals do not overlap (two runs are not paired: other frames, other labels, so
/// the intervals' overlap is the conservative rule); not comparable for two different goals;
/// None when neither run has one.
pub fn goal_change(dev: &str, a: &DeviceSnap, b: &DeviceSnap) -> Option<MetricChange> {
    let (ga, gb) = (a.goal.as_ref(), b.goal.as_ref());
    if ga.is_none() && gb.is_none() {
        return None;
    }
    let name = |g: &crate::benchmark::eval::tune::GoalSnap| {
        g.optimize_for
            .parse::<crate::benchmark::eval::Optimize>()
            .map(|o| o.short())
            .unwrap_or_else(|_| g.optimize_for.clone())
    };
    let mut c = MetricChange {
        metric: "goal".to_string(),
        device: Some(dev.to_string()),
        before: ga.map(|g| g.value),
        after: gb.map(|g| g.value),
        delta: None,
        rel: None,
        verdict: Verdict::Same,
        why: String::new(),
    };
    let (Some(x), Some(y)) = (ga, gb) else {
        c.why = format!(
            "no goal metric in the {} run",
            if ga.is_none() { "earlier" } else { "later" }
        );
        return Some(c);
    };
    if x.optimize_for != y.optimize_for || x.classes != y.classes {
        c.verdict = Verdict::NotComparable;
        c.why = format!("another goal ({} -> {})", name(x), name(y));
        return Some(c);
    }
    let delta = y.value - x.value;
    c.delta = Some(delta);
    let iv = |g: &crate::benchmark::eval::tune::GoalSnap| Some((g.low?, g.high?));
    let (Some((xl, xh)), Some((yl, yh))) = (iv(x), iv(y)) else {
        c.why = format!(
            "{} without an interval in one of the runs: not judged",
            name(x)
        );
        return Some(c);
    };
    if yl > xh || yh < xl {
        c.verdict = if delta > 0.0 {
            Verdict::Better
        } else {
            Verdict::Worse
        };
        c.why = format!(
            "{} {:.3} -> {:.3}: the 95% intervals ({:.3}-{:.3}, {:.3}-{:.3}) do not overlap",
            name(x),
            x.value,
            y.value,
            xl,
            xh,
            yl,
            yh
        );
    } else {
        c.why = format!(
            "{} {:.3} -> {:.3}, within the overlapping 95% intervals",
            name(x),
            x.value,
            y.value
        );
    }
    Some(c)
}

/// The burst p95 of two runs of one device: a speed change when both ran the same load, not
/// comparable under another load (more cameras wait longer by construction).
fn burst_change(dev: &str, a: &DeviceSnap, b: &DeviceSnap) -> MetricChange {
    let mut c = speed_change("burst_p95_ms", dev, a.burst_p95_ms, b.burst_p95_ms, false);
    if let (Some(x), Some(y)) = (&a.burst_load, &b.burst_load)
        && x != y
        && c.delta.is_some()
    {
        c.verdict = Verdict::NotComparable;
        c.delta = None;
        c.rel = None;
        c.why = format!("burst under another load ({x} -> {y})");
    }
    c
}

/// The live load of two runs of one device: shown, never a verdict (a busier run is slower and
/// noisier).
fn live_load_change(dev: &str, a: &DeviceSnap, b: &DeviceSnap) -> MetricChange {
    let (x, y) = (a.live_requests_per_min, b.live_requests_per_min);
    let txt = |v: Option<f64>| v.map_or_else(|| "not recorded".to_string(), |v| format!("{v:.1}"));
    MetricChange {
        metric: "live_requests_per_min".to_string(),
        device: Some(dev.to_string()),
        before: x,
        after: y,
        delta: x.zip(y).map(|(x, y)| y - x),
        rel: None,
        verdict: Verdict::Same,
        why: format!(
            "detection requests answered while it ran, per minute: {} -> {} (a busier run is \
             slower and noisier)",
            txt(x),
            txt(y)
        ),
    }
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
        .filter(|m| {
            on(m)
                && matches!(
                    m.metric.as_str(),
                    "ap50" | "precision" | "recall" | "f1" | "person_false_alerts" | "goal"
                )
        })
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
    // QA-066: a pinned entry after `current` is no baseline of it (the change would run
    // backwards); the newest pinned earlier one.
    let pinned = candidates.iter().rev().find(|e| e.pinned).copied();
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
