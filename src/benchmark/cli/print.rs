//! Human-readable output of the benchmark command line: the report structures printed as JSON
//! with `--json`, and every table (devices, accuracy breakdown, thresholds, ranking, search).
//! Printing only: nothing here runs inference or touches the config file.

use crate::benchmark::grade::AccuracyMetric;
use crate::benchmark::images::ImageSet;
use crate::benchmark::report::{RuntimeVersions, Verdict};
use crate::benchmark::threshold::{Objective, ThresholdAdvice};
use crate::benchmark::{Comparison, MATCH_IOU, ModelResult, RunResult, Stats};
use crate::config_form::{DeviceChange, ProfileChange};
use anyhow::{Result, bail};
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Clone, Serialize)]
pub(super) struct ModelReport {
    pub(super) primary: RunResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) cpu: Option<RunResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) comparison: Option<Comparison>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) note: Option<String>,
    /// `--all-devices`: one run per runnable device option (`primary` is the recommended one, or
    /// the fastest by total p50 when none is recommended).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) all_devices: Vec<RunResult>,
    /// `--all-devices`: options that cannot run or failed for this model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) skipped_devices: Vec<String>,
    /// `--all-devices`: the recommended device and why.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) recommended: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) recommendation: Option<String>,
    /// `--all-devices`: device whose detections the others were compared with.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) reference: Option<String>,
    /// `--all-devices`: the best confidence threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) threshold: Option<ThresholdAdvice>,
}

#[derive(Debug, Clone, Serialize)]
pub(super) struct Report {
    pub(super) version: &'static str,
    pub(super) openvino_version: String,
    pub(super) available_devices: Vec<String>,
    pub(super) models: Vec<ModelReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) errors: Vec<String>,
    /// `--all-devices`: where the results were saved.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) results_file: Option<String>,
    /// `--apply`: per-model device changes written to the config file.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) applied: Vec<DeviceChange>,
    /// `--apply-threshold`: per-device confidence profile changes written to the config file.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) applied_thresholds: Vec<ProfileChange>,
    /// Datasets used (ids) and warnings about them.
    pub(super) datasets: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) warnings: Vec<String>,
    /// `--all-devices`: models ranked on their recommended devices.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) ranking: Vec<crate::benchmark::report::RankRow>,
}

/// The banner of a table run: versions, devices, cache, datasets, iterations, then the
/// warnings and errors so far (stderr).
pub(super) fn print_header(
    report: &Report,
    versions: &RuntimeVersions,
    cache_dir: Option<&Path>,
    sets: &[ImageSet],
    warmup: usize,
    repeat: usize,
) {
    println!(
        "Blue Onyx Prism {} benchmark | OpenVINO {} | ONNX Runtime {} | devices {:?} | cache {}",
        report.version,
        report.openvino_version,
        versions
            .onnxruntime
            .as_deref()
            .map(|v| match &versions.onnxruntime_flavor {
                Some(f) => format!("{v} ({f})"),
                None => v.to_string(),
            })
            .unwrap_or_else(|| "not loaded".into()),
        report.available_devices,
        cache_dir
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "disabled".into())
    );
    let names: Vec<String> = sets
        .iter()
        .map(|s| {
            format!(
                "{} ({} images, {})",
                s.id,
                s.images.len(),
                s.ground_truth.describe()
            )
        })
        .collect();
    println!("datasets: {}", names.join("; "));
    println!("warmup {warmup} | {repeat} timed run(s) per image");
    for w in &report.warnings {
        eprintln!("warning: {w}");
    }
    for e in &report.errors {
        eprintln!("error: {e}");
    }
}

impl Report {
    /// An empty report of a run over `sets`, with the errors and warnings so far.
    pub(super) fn new(
        info: crate::backend::RuntimeInfo,
        errors: Vec<String>,
        sets: &[ImageSet],
        warnings: Vec<String>,
    ) -> Self {
        Self {
            version: crate::VERSION,
            openvino_version: info.openvino_version,
            available_devices: info.available_devices,
            models: Vec::new(),
            errors,
            results_file: None,
            applied: Vec::new(),
            applied_thresholds: Vec::new(),
            datasets: sets.iter().map(|s| s.id.clone()).collect(),
            warnings,
            ranking: Vec::new(),
        }
    }

    /// Record a warning, printed on stderr unless `quiet`.
    pub(super) fn warn(&mut self, quiet: bool, w: String) {
        if !quiet {
            eprintln!("warning: {w}");
        }
        self.warnings.push(w);
    }

    /// Record an error, printed on stderr unless `quiet`.
    pub(super) fn error(&mut self, quiet: bool, msg: String) {
        if !quiet {
            eprintln!("error: {msg}");
        }
        self.errors.push(msg);
    }
}

/// The heading of a threshold search and one summary row per model.
pub(super) fn print_search_summary(
    req: &crate::benchmark::search::SearchRequest,
    out: &crate::benchmark::search::SearchOutput,
) {
    let mut filters = Vec::new();
    if !req.datasets.is_empty() {
        filters.push(format!("datasets {}", req.datasets.join(", ")));
    }
    if !req.tags.is_empty() {
        filters.push(format!("tags {}", req.tags.join(" + ")));
    }
    if !req.classes.is_empty() {
        filters.push(format!("classes {}", req.classes.join(", ")));
    }
    println!(
        "== Confidence threshold search ({}; IoU {}; {}; {:.1} ms, no inference)",
        req.objective.describe(),
        req.iou,
        if filters.is_empty() {
            "all images".to_string()
        } else {
            filters.join("; ")
        },
        out.elapsed_ms
    );
    println!(
        "  {:<20} {:<14} {:>5} {:>5} {:>5} {:>5}   {:>5} {:>5} {:>5} {:>5} {:>6}  {:>4} {:>4}  {:>5} {:>11} {:>11}",
        "model",
        "device",
        "conf",
        "P",
        "R",
        "F1",
        "best",
        "P",
        "R",
        "F1",
        "dF1",
        "imgs",
        "objs",
        "AUC",
        "TPR/FPR cf",
        "TPR/FPR bs"
    );
    for r in &out.results {
        // "conf": the threshold configured now (the gain is against it).
        let c = &r.now;
        let b = r.best.as_ref();
        println!(
            "  {:<20} {:<14} {:>5.2} {:>5} {:>5} {:>5}   {:>5} {:>5} {:>5} {:>5} {:>6}  {:>4} {:>4}  {:>5} {:>11} {:>11}{}",
            r.model,
            r.device,
            c.threshold,
            pct1(c.precision),
            pct1(c.recall),
            pct1(c.f1),
            b.map_or("-".into(), |b| format!(
                "{:.2}{}",
                b.threshold,
                if b.met { "" } else { "!" }
            )),
            pct1(b.and_then(|b| b.point.precision)),
            pct1(b.and_then(|b| b.point.recall)),
            pct1(b.and_then(|b| b.point.f1)),
            match (b.and_then(|b| b.point.f1), c.f1) {
                (Some(x), Some(y)) => format!("{:+.1}", (x - y) * 100.0),
                _ => "-".into(),
            },
            r.images,
            r.gt,
            pct1(r.roc.as_ref().and_then(|x| x.auc)),
            tpr_fpr(c),
            b.map_or("-".into(), |b| tpr_fpr(&b.point)),
            if r.relative { "  (relative)" } else { "" }
        );
    }
}

/// The threshold table of one searched model (nothing without a best threshold), with the
/// best thresholds and frame ROC AUCs per dataset / class.
pub(super) fn print_search_detail(r: &crate::benchmark::search::SearchResult) {
    let Some(b) = &r.best else { return };
    println!();
    println!(
        "  {} on {}: best {:.2} (exact optimum {:.3}, near-optimal range {}){}",
        r.model,
        r.device,
        b.threshold,
        b.exact_threshold.unwrap_or(b.threshold),
        b.plateau
            .map_or("-".into(), |[lo, hi]| format!("{lo:.3}..{hi:.3}")),
        b.note.as_ref().map_or(String::new(), |n| format!("; {n}"))
    );
    println!(
        "    {:>5} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6} {:>6}",
        "conf", "P", "R", "F1", "F2", "FP/img", "TPR", "FPR"
    );
    let mut extra = vec![r.configured, b.point];
    if let Some(c) = r.current {
        extra.push(c);
    }
    for p in crate::benchmark::threshold::table_points(&r.grid, &extra) {
        let mut marks = Vec::new();
        if (b.threshold - p.threshold).abs() < 1e-6 {
            marks.push("best");
        }
        if (r.configured.threshold - p.threshold).abs() < 1e-6 {
            marks.push("configured");
        }
        if r.current
            .is_some_and(|c| (c.threshold - p.threshold).abs() < 1e-6)
        {
            marks.push("in config now");
        }
        println!(
            "    {:>5.2} {:>6} {:>6} {:>6} {:>6} {:>7.2} {:>6} {:>6}{}",
            p.threshold,
            pct1(p.precision),
            pct1(p.recall),
            pct1(p.f1),
            pct1(p.f2),
            p.fp_per_image,
            pct1(p.frame.and_then(|f| f.tpr)),
            pct1(p.frame.and_then(|f| f.fpr)),
            if marks.is_empty() {
                String::new()
            } else {
                format!("  <- {}", marks.join(", "))
            }
        );
    }
    let groups: Vec<String> = r
        .by_dataset
        .iter()
        .chain(&r.per_class)
        .filter_map(|g| Some(format!("{} {:.2}", g.key, g.best.as_ref()?.threshold)))
        .collect();
    if !groups.is_empty() {
        println!("    best by dataset/class: {}", groups.join(", "));
    }
    let aucs: Vec<String> = r
        .by_dataset
        .iter()
        .filter_map(|g| Some(format!("{} {}", g.key, pct1(Some(g.roc_auc?)))))
        .collect();
    if !aucs.is_empty() {
        println!("    frame ROC AUC by dataset: {}", aucs.join(", "));
    }
}

/// How the server threshold relates to Blue Iris's own confidence settings.
pub(crate) const BLUE_IRIS_NOTE: &str = "note: Blue Iris can send its own min_confidence with each request (it then replaces the server threshold for that request), and each camera's minimum confidence filters the returned objects again on top.";

pub(super) fn pct1(v: Option<f64>) -> String {
    v.map_or("-".to_string(), |v| format!("{:.1}", v * 100.0))
}

/// "91.0/6.2" (frame TPR / FPR in %), "-" without frame rates.
pub(super) fn tpr_fpr(p: &crate::benchmark::metrics::ThresholdPoint) -> String {
    p.frame
        .as_ref()
        .map_or("-".into(), |f| format!("{}/{}", pct1(f.tpr), pct1(f.fpr)))
}

/// "person 97.1 (120+/80-), vehicle 93.0 (..)".
pub(super) fn class_aucs(roc: &crate::benchmark::roc::FrameRoc) -> String {
    roc.per_class
        .iter()
        .map(|c| {
            format!(
                "{} {} ({}+/{}-)",
                c.class,
                pct1(c.auc),
                c.positives,
                c.negatives
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Frame-level ROC per model on the advice device: AUC, and frame TPR / FPR at the configured
/// and the best threshold.
pub(super) fn print_frame_roc(swept: &[ModelResult]) {
    let rows: Vec<(
        &ModelResult,
        &ThresholdAdvice,
        &crate::benchmark::roc::FrameRoc,
    )> = swept
        .iter()
        .filter_map(|m| {
            let a = m.threshold.as_ref()?;
            Some((m, a, a.roc.as_ref()?))
        })
        .collect();
    if rows.is_empty() {
        return;
    }
    println!();
    println!(
        "== Frame-level ROC per model (a frame alerts when its top confidence >= the threshold; %)"
    );
    println!(
        "  {:<20} {:<14} {:>5} {:>5}  {:>5} {:>11}  {:>5} {:>11}  per class",
        "model", "device", "AUC", "micro", "conf", "TPR/FPR", "best", "TPR/FPR"
    );
    for (m, a, roc) in rows {
        println!(
            "  {:<20} {:<14} {:>5} {:>5}  {:>5.2} {:>11}  {:>5} {:>11}  {}",
            m.model,
            a.device,
            pct1(roc.auc),
            pct1(roc.micro_auc),
            a.configured.threshold,
            tpr_fpr(&a.configured),
            a.best
                .as_ref()
                .map_or("-".into(), |b| format!("{:.2}", b.threshold)),
            a.best.as_ref().map_or("-".into(), |b| tpr_fpr(&b.point)),
            class_aucs(roc)
        );
    }
    println!(
        "  (AUC from the 0.05+ predictions: frames below 0.05 score 0, which slightly understates it)"
    );
}

/// The best-threshold summary across models.
pub(super) fn print_thresholds(swept: &[ModelResult], objective: Objective) {
    let rows: Vec<(&ModelResult, &ThresholdAdvice)> = swept
        .iter()
        .filter_map(|m| m.threshold.as_ref().map(|a| (m, a)))
        .collect();
    if rows.is_empty() {
        return;
    }
    println!();
    println!(
        "== Best confidence threshold per model ({}; P/R/F1 in %)",
        objective.describe()
    );
    println!(
        "  {:<20} {:<14} {:>5} {:>5} {:>5} {:>5}   {:>5} {:>5} {:>5} {:>5}  {:>4} {:>4} {:>5}  night",
        "model", "device", "conf", "P", "R", "F1", "best", "P", "R", "F1", "F2", "P90", "dF1"
    );
    for (m, a) in rows {
        let c = &a.configured;
        let alt = |o: Objective| {
            a.alternatives
                .iter()
                .find(|p| p.objective == o.to_string())
                .map_or("-".to_string(), |p| {
                    format!("{:.2}{}", p.threshold, if p.met { "" } else { "!" })
                })
        };
        let night = a
            .by_dataset
            .iter()
            .chain(&a.by_tag)
            .find(|g| g.key.contains("night"))
            .and_then(|g| {
                g.best
                    .as_ref()
                    .map(|b| format!("{:.2} ({})", b.threshold, g.key))
            })
            .unwrap_or_else(|| "-".into());
        match &a.best {
            Some(b) => println!(
                "  {:<20} {:<14} {:>5.2} {:>5} {:>5} {:>5}   {:>5.2} {:>5} {:>5} {:>5}  {:>4} {:>4} {:>5}  {}{}",
                m.model,
                a.device,
                c.threshold,
                pct1(c.precision),
                pct1(c.recall),
                pct1(c.f1),
                b.threshold,
                pct1(b.point.precision),
                pct1(b.point.recall),
                pct1(b.point.f1),
                alt(Objective::F2),
                alt(Objective::Precision(0.9)),
                match (b.point.f1, c.f1) {
                    (Some(x), Some(y)) => format!("{:+.1}", (x - y) * 100.0),
                    _ => "-".into(),
                },
                night,
                if a.relative { "  (relative)" } else { "" }
            ),
            None => println!(
                "  {:<20} {:<14} no ground truth for a curve",
                m.model, a.device
            ),
        }
    }
    println!("  (! = precision 90% not reached; the most precise threshold is shown)");
}

/// The compact threshold table of one model's advice.
pub(super) fn print_threshold_table(a: &ThresholdAdvice) {
    let best = a.best.as_ref().map(|b| b.threshold);
    let mut extra = vec![a.configured];
    if let Some(b) = &a.best {
        extra.push(b.point);
    }
    println!(
        "  confidence threshold on {} ({}{}):",
        a.device,
        a.objective().describe(),
        if a.relative { ", relative" } else { "" }
    );
    println!(
        "    {:>5} {:>6} {:>6} {:>6} {:>6} {:>7} {:>6} {:>6}",
        "conf", "P", "R", "F1", "F2", "FP/img", "TPR", "FPR"
    );
    for p in crate::benchmark::threshold::table_points(&a.curve, &extra) {
        let mut marks = Vec::new();
        if best.is_some_and(|b| (b - p.threshold).abs() < 1e-6) {
            marks.push("best");
        }
        if (a.configured.threshold - p.threshold).abs() < 1e-6 {
            marks.push("configured");
        }
        println!(
            "    {:>5.2} {:>6} {:>6} {:>6} {:>6} {:>7.2} {:>6} {:>6}{}",
            p.threshold,
            pct1(p.precision),
            pct1(p.recall),
            pct1(p.f1),
            pct1(p.f2),
            p.fp_per_image,
            pct1(p.frame.and_then(|f| f.tpr)),
            pct1(p.frame.and_then(|f| f.fpr)),
            if marks.is_empty() {
                String::new()
            } else {
                format!("  <- {}", marks.join(", "))
            }
        );
    }
    let groups: Vec<String> = a
        .by_dataset
        .iter()
        .chain(&a.by_tag)
        .filter_map(|g| {
            let b = g.best.as_ref()?;
            Some(format!("{} {:.2}", g.key, b.threshold))
        })
        .collect();
    if !groups.is_empty() {
        println!("    best by dataset/tag: {}", groups.join(", "));
    }
    let classes: Vec<String> = a
        .per_class
        .iter()
        .filter_map(|g| Some(format!("{} {:.2}", g.key, g.best.as_ref()?.threshold)))
        .collect();
    if !classes.is_empty() {
        println!("    best by class: {}", classes.join(", "));
    }
}

pub(super) fn print_ranking(rows: &[crate::benchmark::report::RankRow]) {
    println!();
    let metric = rows.first().map_or(AccuracyMetric::Ap50, |r| r.metric);
    println!(
        "== Best model for this machine (each on its recommended device; accuracy graded by {})",
        metric.label()
    );
    println!(
        "  {:>2} {:<20} {:<16} {:>7} {:>8} {:>5} {:>9} {:>6} {:>6}",
        "#", "model", "device", "overall", "accuracy", "speed", "p50 ms", "AP50", "AUC"
    );
    for r in rows {
        println!(
            "  {:>2} {:<20} {:<16} {:>7} {:>8} {:>5} {:>9.1} {:>6} {:>6}",
            r.rank,
            r.model,
            r.device,
            r.overall.to_string(),
            r.accuracy.map_or("-".to_string(), |a| {
                if r.relative {
                    format!("{a}*")
                } else {
                    a.to_string()
                }
            }),
            r.speed.to_string(),
            r.p50_ms,
            r.ap50
                .map_or("-".to_string(), |v| format!("{:.1}", v * 100.0)),
            pct1(r.roc_auc)
        );
    }
}

/// The `--all-devices` JSON/table view of a sweep.
pub(super) fn model_report(m: &ModelResult) -> Result<ModelReport> {
    let runs: Vec<RunResult> = m
        .devices
        .iter()
        .filter(|d| d.ok())
        .filter_map(|d| d.run.clone())
        .collect();
    let mut skipped = m.skipped.clone();
    skipped.extend(
        m.devices
            .iter()
            .filter(|d| !d.ok())
            .map(|d| match (&d.error, &d.run) {
                (Some(e), _) => format!("{}: {e}", d.device),
                (None, Some(r)) => {
                    format!("{}: fell back to {}, result dropped", d.device, r.device)
                }
                (None, None) => format!("{}: did not run", d.device),
            }),
    );
    let primary = m
        .recommended
        .as_ref()
        .and_then(|rec| m.devices.iter().find(|d| &d.device == rec))
        .or_else(|| m.fastest())
        .and_then(|d| d.run.clone());
    let Some(primary) = primary else {
        bail!(
            "no device option could run this model ({})",
            skipped.join("; ")
        );
    };
    Ok(ModelReport {
        primary,
        cpu: None,
        comparison: None,
        note: None,
        all_devices: runs,
        skipped_devices: skipped,
        recommended: m.recommended.clone(),
        recommendation: Some(m.recommendation.clone()),
        reference: m.reference.clone(),
        threshold: m.threshold.clone(),
    })
}

/// Short agreement text for tables: "ref", "ok", "minor (2/3)", "BAD (0/3)", "-".
pub(crate) fn agreement_text(r: &RunResult) -> String {
    match &r.agreement {
        None => "-".to_string(),
        Some(a) => {
            let total = a
                .matched
                .saturating_add(a.only_device.max(a.only_reference));
            match a.verdict {
                Verdict::Reference => "ref".to_string(),
                Verdict::Agrees => "ok".to_string(),
                Verdict::Minor => format!("minor ({}/{total})", a.matched),
                Verdict::Disagrees => format!("BAD ({}/{total})", a.matched),
            }
        }
    }
}

/// Comparison table of an `--all-devices` model report, fastest first, with grades, then the
/// recommended device's accuracy breakdown.
pub(super) fn print_all_devices(result: &ModelResult, m: &ModelReport) {
    print_device_table(result, m);
    for s in &m.skipped_devices {
        println!("  skipped {s}");
    }
    match &m.recommended {
        Some(d) => println!("  recommended: {d} ({})", result.recommendation),
        None => println!("  recommended: none ({})", result.recommendation),
    }
    let p = &m.primary;
    print_accuracy_breakdown(p);
    if let Some(a) = &result.threshold {
        print_threshold_table(a);
    }
    if p.images > 1 {
        print_resolution_speeds(p);
    }
}

/// The heading and one row per device run of [`print_all_devices`], fastest first.
pub(super) fn print_device_table(result: &ModelResult, m: &ModelReport) {
    println!();
    println!(
        "== {}: all runnable devices ({} images x {} runs, ms)",
        m.primary.model, m.primary.images, m.primary.repeat
    );
    println!(
        "  {:<16} {:<26} {:>7} {:>9} {:>9} {:>9} {:>7} {:>6} {:>6} {:>6} {:>5} {:>5} {:<7} {:<12}",
        "requested",
        "device",
        "compile",
        "infer p50",
        "total p50",
        "total p95",
        "req/s",
        "AP50",
        "AP-95",
        "AUC",
        "P",
        "R",
        "O/A/S",
        "agreement"
    );
    let mut runs: Vec<&RunResult> = m.all_devices.iter().collect();
    runs.sort_by(|a, b| a.stages_ms.total.p50.total_cmp(&b.stages_ms.total.p50));
    let fastest = runs
        .first()
        .map(|r| r.stages_ms.total.p50)
        .unwrap_or_default();
    for r in &runs {
        let mut marks = Vec::new();
        if r.stages_ms.total.p50 == fastest && runs.len() > 1 {
            marks.push("fastest");
        }
        if m.recommended.as_deref() == Some(r.requested_device.as_str()) {
            marks.push("recommended");
        }
        if result.configured.as_deref() == Some(r.requested_device.as_str()) {
            marks.push("configured");
        }
        println!("{}", device_row(r, &marks));
    }
}

/// One row of [`print_device_table`] with its `marks` ("fastest", "recommended", ...).
pub(super) fn device_row(r: &RunResult, marks: &[&str]) -> String {
    let name = if r.device_name.is_empty() {
        r.device.clone()
    } else {
        format!("{} ({})", r.device, r.device_name)
    };
    let name: String = name.chars().take(26).collect();
    let acc = r.accuracy.as_ref().map(|a| &a.overall);
    let grades = r.grades.as_ref().map_or("-".to_string(), |g| {
        format!(
            "{}/{}{}/{}",
            g.overall,
            g.accuracy.map_or("-".to_string(), |a| a.to_string()),
            if g.relative { "*" } else { "" },
            g.speed
        )
    });
    format!(
        "  {:<16} {:<26} {:>7.0} {:>9.2} {:>9.2} {:>9.2} {:>7.1} {:>6} {:>6} {:>6} {:>5} {:>5} {:<7} {:<12}{}",
        r.requested_device,
        name,
        r.compile_ms,
        r.stages_ms.infer.p50,
        r.stages_ms.total.p50,
        r.stages_ms.total.p95,
        r.throughput_fps,
        pct1(acc.and_then(|a| a.ap50)),
        pct1(acc.and_then(|a| a.ap50_95)),
        pct1(r.accuracy.as_ref().and_then(|a| a.roc_auc())),
        pct1(acc.and_then(|a| a.precision)),
        pct1(acc.and_then(|a| a.recall)),
        grades,
        agreement_text(r),
        if marks.is_empty() {
            String::new()
        } else {
            format!("  <- {}", marks.join(", "))
        }
    )
}

/// The accuracy of `p` overall, per dataset and per tag (or why it is not scored).
pub(super) fn print_accuracy_breakdown(p: &RunResult) {
    match &p.accuracy {
        Some(a) => {
            println!(
                "  accuracy on {} ({}; P/R at {:.2}): {} | small/medium/large recall {}",
                p.requested_device,
                a.ground_truth,
                a.threshold,
                a.classes.join(", "),
                a.overall
                    .by_size
                    .iter()
                    .map(|b| format!("{} {}", b.bucket, pct1(b.recall)))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            for b in a.by_dataset.iter().chain(&a.by_tag) {
                println!(
                    "    {:<24} {:>4} img {:>5} obj  AP50 {:>5}  AUC {:>5}  P {:>5}  R {:>5}{}",
                    b.key,
                    b.images,
                    b.gt,
                    pct1(b.ap50),
                    pct1(b.roc_auc),
                    pct1(b.precision),
                    pct1(b.recall),
                    if b.relative { "  (relative)" } else { "" }
                );
            }
        }
        None => {
            if let Some(n) = &p.accuracy_note {
                println!("  accuracy not scored: {n}");
            }
        }
    }
}

/// Speed per resolution bucket of `p`: every bucket; one without images is n/a (it does not
/// enter any grade).
pub(super) fn print_resolution_speeds(p: &RunResult) {
    for bucket in crate::benchmark::images::RESOLUTION_BUCKETS {
        match p.by_resolution.iter().find(|b| b.bucket == bucket) {
            Some(b) => println!(
                "    res:{:<6} {:>4} img  pre {:>7.2}  infer {:>7.2}  post {:>6.2}  total p50 {:>7.2} p95 {:>7.2}",
                b.bucket, b.images, b.pre_p50, b.infer_p50, b.post_p50, b.total_p50, b.total_p95
            ),
            None => println!("    res:{bucket:<6}    0 img  n/a"),
        }
    }
}

pub(super) fn stats_row(name: &str, s: &Stats) -> String {
    format!(
        "  {name:<12} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
        s.min, s.mean, s.p50, s.p95, s.max
    )
}

pub(super) fn print_run(r: &RunResult) {
    println!();
    println!(
        "== {} [{}] on {} ({}){}",
        r.model,
        r.family,
        r.device,
        if r.device_name.is_empty() {
            "?"
        } else {
            &r.device_name
        },
        if r.fell_back {
            format!(" -- FELL BACK from {}", r.requested_device)
        } else {
            String::new()
        }
    );
    println!(
        "  path {} | input {} | image {}",
        r.path, r.input, r.image_size
    );
    println!(
        "  compile {:.0} ms (cache: {}) | warmup {} x in {:.1} ms",
        r.compile_ms, r.cache, r.warmup_iterations, r.warmup_ms
    );
    println!(
        "  {:<12} {:>9} {:>9} {:>9} {:>9} {:>9}   (ms, {} runs)",
        "stage", "min", "mean", "p50", "p95", "max", r.repeat
    );
    let s = &r.stages_ms;
    println!("{}", stats_row("decode", &s.decode));
    println!("{}", stats_row("preprocess", &s.preprocess));
    println!("{}", stats_row("infer", &s.infer));
    println!("{}", stats_row("postprocess", &s.postprocess));
    println!("{}", stats_row("total", &s.total));
    println!(
        "  throughput {:.1} img/s | {} detections (first image) | {} image(s): {}",
        r.throughput_fps,
        r.detections.len(),
        r.images,
        r.image
    );
    if let Some(a) = &r.accuracy {
        let pct = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{:.1}", v * 100.0));
        println!(
            "  accuracy ({}): AP50 {} AP50-95 {} frame ROC AUC {} P {} R {} F1 {}",
            a.ground_truth,
            pct(a.overall.ap50),
            pct(a.overall.ap50_95),
            pct(a.roc_auc()),
            pct(a.overall.precision),
            pct(a.overall.recall),
            pct(a.overall.f1)
        );
    }
    if let Some(g) = &r.grades {
        println!(
            "  grades: overall {} (accuracy {} by {}, speed {})",
            g.overall,
            g.accuracy.map_or("-".to_string(), |a| a.to_string()),
            g.metric.label(),
            g.speed
        );
    }
    for d in &r.detections {
        println!(
            "    {:<14} {:.3}  [{}, {}, {}, {}]",
            d.label, d.confidence, d.x_min, d.y_min, d.x_max, d.y_max
        );
    }
}

pub(super) fn print_model(m: &ModelReport) {
    print_run(&m.primary);
    if let Some(note) = &m.note {
        println!("  note: {note}");
    }
    let (Some(cpu), Some(c)) = (&m.cpu, &m.comparison) else {
        return;
    };
    print_run(cpu);
    let p = &m.primary;
    println!();
    println!("== {}: {} vs CPU (ms)", p.model, p.device);
    println!("  {:<12} {:>12} {:>12} {:>9}", "", p.device, "CPU", "ratio");
    let row = |name: &str, a: f64, b: f64| {
        let ratio = if a > 0.0 { b / a } else { 0.0 };
        println!("  {name:<12} {a:>12.2} {b:>12.2} {ratio:>8.2}x");
    };
    row("compile", p.compile_ms, cpu.compile_ms);
    row("infer p50", p.stages_ms.infer.p50, cpu.stages_ms.infer.p50);
    row(
        "infer mean",
        p.stages_ms.infer.mean,
        cpu.stages_ms.infer.mean,
    );
    row("total p50", p.stages_ms.total.p50, cpu.stages_ms.total.p50);
    row("total p95", p.stages_ms.total.p95, cpu.stages_ms.total.p95);
    println!(
        "  {:<12} {:>12.1} {:>12.1}",
        "img/s", p.throughput_fps, cpu.throughput_fps
    );
    println!(
        "  {:<12} {:>12} {:>12}",
        "detections",
        p.detections.len(),
        cpu.detections.len()
    );
    println!(
        "  matched {} (IoU >= {MATCH_IOU}, same label), only {} {}, only CPU {}; confidence diff max {:.4}, mean {:.4}",
        c.matched,
        p.device,
        c.only_primary,
        c.only_cpu,
        c.max_confidence_diff,
        c.mean_confidence_diff
    );
    if p.stages_ms.infer.p50 > cpu.stages_ms.infer.p50 && p.device != "CPU" {
        println!("  WARNING: {} inference is slower than CPU", p.device);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agreement_texts() {
        use crate::benchmark::report::Agreement;
        let mut r = RunResult {
            model: "m".into(),
            path: String::new(),
            family: String::new(),
            requested_device: String::new(),
            device: String::new(),
            device_name: String::new(),
            spec: String::new(),
            fell_back: false,
            input: String::new(),
            image: String::new(),
            image_size: String::new(),
            images: 1,
            compile_ms: 0.0,
            cache: String::new(),
            warmup_iterations: 0,
            warmup_ms: 0.0,
            repeat: 1,
            stages_ms: Default::default(),
            throughput_fps: 0.0,
            detections: Vec::new(),
            by_resolution: Vec::new(),
            accuracy: None,
            accuracy_note: None,
            grades: None,
            agreement: None,
            per_image: Vec::new(),
            eval_preds: Vec::new(),
        };
        assert_eq!(agreement_text(&r), "-");
        let a = |verdict, matched, only_device, only_reference| Agreement {
            reference: "openvino:cpu".into(),
            matched,
            only_device,
            only_reference,
            max_confidence_diff: 0.0,
            mean_confidence_diff: 0.0,
            score: 0.0,
            verdict,
        };
        r.agreement = Some(a(Verdict::Minor, 2, 0, 1));
        assert_eq!(agreement_text(&r), "minor (2/3)");
        r.agreement = Some(a(Verdict::Disagrees, 0, 0, 3));
        assert_eq!(agreement_text(&r), "BAD (0/3)");
        r.agreement = Some(a(Verdict::Reference, 3, 0, 0));
        assert_eq!(agreement_text(&r), "ref");
        // BM-07: counts near usize::MAX from a hand-edited benchmark.json do not overflow.
        r.agreement = Some(a(Verdict::Minor, usize::MAX, usize::MAX, 1));
        assert_eq!(
            agreement_text(&r),
            format!("minor ({}/{})", usize::MAX, usize::MAX)
        );
    }
}
