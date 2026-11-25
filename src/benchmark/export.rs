//! Standalone benchmark reports from [`BenchmarkResults`]: Markdown (`--report out.md`) and a
//! self-contained HTML page (`--report out.html`, inline CSS, light/dark, no external files;
//! its one script is the inlined table sorter `assets/sort.js`, so every table sorts by a click
//! on a header opened from disk too, numeric cells carry their value as `data-v`). Both show the machine, the datasets, the cross-model ranking, and per model
//! the device table with grades plus the recommended device's breakdown by dataset, tag and
//! resolution, and the confidence-threshold sweep (best threshold per model, a compact
//! P/R/F1/F2 table per model), and the frame-level ROC (AUC per class with the positive and
//! negative frame counts, frame TPR / FPR at the configured and the best threshold).

use super::grade::{
    ACCURACY_THRESHOLDS, AccuracyMetric, ROC_AUC_THRESHOLDS, SMALL_RECALL_WEIGHT,
    SPEED_THRESHOLDS_MS,
};
use super::metrics::ThresholdPoint;
use super::report::{BenchmarkResults, DeviceResult, ModelResult};
use super::threshold::{GroupPick, Pick, ThresholdAdvice, table_points};
use super::{AccuracyReport, Breakdown, RunResult};
use anyhow::{Context, Result, bail};
use std::fmt::Write as _;
use std::path::Path;

/// A table: title, header, rows of cells.
struct Table {
    title: String,
    head: Vec<String>,
    rows: Vec<Vec<String>>,
    /// The HTML report's sort values (Markdown ignores them).
    numbers: Numbers,
}

/// Sort values of a table's cells for the HTML report: `values[row][col]` is a cell's `data-v`
/// (None or a short row: the sorter reads the text), `desc[col]` a column whose first click puts
/// the highest value first (`data-sort-first="desc"`). Empty = none.
#[derive(Default)]
struct Numbers {
    values: Vec<Vec<Option<f64>>>,
    desc: Vec<bool>,
}

/// The table sorter of the pages (`PrismSort`), inlined into the HTML report.
const SORT_JS: &str = include_str!("../../assets/sort.js");

/// A report section: heading, paragraphs, tables.
struct Section {
    heading: String,
    level: u8,
    paragraphs: Vec<String>,
    tables: Vec<Table>,
}

fn pct(v: Option<f64>) -> String {
    v.map_or("-".into(), |v| format!("{:.1}", v * 100.0))
}

fn msf(v: f64) -> String {
    format!("{v:.1}")
}

fn grade_cells(r: &RunResult) -> [String; 3] {
    match &r.grades {
        Some(g) => [
            g.overall.to_string(),
            g.accuracy.map_or("-".into(), |a| {
                if g.relative {
                    format!("{a}*")
                } else {
                    a.to_string()
                }
            }),
            g.speed.to_string(),
        ],
        None => ["-".into(), "-".into(), "-".into()],
    }
}

fn device_row(m: &ModelResult, d: &DeviceResult) -> Vec<String> {
    let mut marks = Vec::new();
    if m.recommended.as_deref() == Some(d.device.as_str()) {
        marks.push("recommended");
    }
    if m.configured.as_deref() == Some(d.device.as_str()) {
        marks.push("configured");
    }
    if m.fastest().is_some_and(|f| f.device == d.device) {
        marks.push("fastest");
    }
    let mut device = d.device.clone();
    if !marks.is_empty() {
        device.push_str(&format!(" ({})", marks.join(", ")));
    }
    match (&d.run, &d.error) {
        (Some(r), _) if d.ok() => {
            let acc = r.accuracy.as_ref().map(|a| &a.overall);
            let [o, a, s] = grade_cells(r);
            vec![
                device,
                super::report::first_load_text(r.compile_ms, &r.cache),
                msf(r.stages_ms.infer.p50),
                msf(r.stages_ms.total.p50),
                msf(r.stages_ms.total.p95),
                r.stages_ms.total.p99_of().map_or("-".to_string(), msf),
                format!("{:.1}", r.throughput_fps),
                pct(acc.and_then(|s| s.ap50)),
                pct(acc.and_then(|s| s.ap50_95)),
                pct(r.accuracy.as_ref().and_then(|a| a.roc_auc())),
                pct(acc.and_then(|s| s.precision)),
                pct(acc.and_then(|s| s.recall)),
                o,
                a,
                s,
                super::cli::agreement_text(r),
            ]
        }
        (Some(r), _) => {
            let mut v = vec![device, format!("fell back to {}", r.device)];
            v.resize(16, String::new());
            v
        }
        (None, e) => {
            let mut v = vec![
                device,
                format!("failed: {}", e.as_deref().unwrap_or("did not run")),
            ];
            v.resize(16, String::new());
            v
        }
    }
}

fn breakdown_table<'a>(title: &str, rows: impl IntoIterator<Item = &'a Breakdown>) -> Table {
    Table {
        title: title.to_string(),
        numbers: Numbers::default(),
        head: [
            "", "images", "objects", "AP50", "AP50-95", "ROC AUC", "P", "R", "F1",
        ]
        .map(String::from)
        .to_vec(),
        rows: rows
            .into_iter()
            .map(|b| {
                vec![
                    if b.relative {
                        format!("{} (relative)", b.key)
                    } else {
                        b.key.clone()
                    },
                    b.images.to_string(),
                    b.gt.to_string(),
                    pct(b.ap50),
                    pct(b.ap50_95),
                    pct(b.roc_auc),
                    pct(b.precision),
                    pct(b.recall),
                    pct(b.f1),
                ]
            })
            .collect(),
    }
}

fn pick_cell(p: Option<&Pick>) -> String {
    match p {
        Some(p) if p.met => format!("{:.2}", p.threshold),
        Some(p) => format!("{:.2} (target not reached)", p.threshold),
        None => "-".into(),
    }
}

/// Frame TPR or FPR at a point, one sample per frame ("-" without frame rates).
fn tpr(p: Option<&ThresholdPoint>) -> String {
    pct(p.and_then(|p| p.alert_rates()).and_then(|f| f.tpr))
}

fn fpr(p: Option<&ThresholdPoint>) -> String {
    pct(p.and_then(|p| p.alert_rates()).and_then(|f| f.fpr))
}

/// "0.35", or "0.35 (running 0.50 until a restart)" while a restart is pending.
fn now_cell(a: &ThresholdAdvice) -> String {
    let t = a.now_threshold();
    match a.now.as_ref().and_then(|n| n.running) {
        Some(r) => format!("{t:.2} (running {r:.2} until a restart)"),
        None => format!("{t:.2}"),
    }
}

fn delta(a: Option<f64>, b: Option<f64>) -> String {
    match (a, b) {
        (Some(a), Some(b)) => format!("{:+.1}", (a - b) * 100.0),
        _ => "-".into(),
    }
}

/// The best-threshold summary across models.
fn threshold_summary(r: &BenchmarkResults) -> Option<Section> {
    let rows: Vec<(&ModelResult, &ThresholdAdvice)> = r
        .models
        .iter()
        .filter_map(|m| m.threshold.as_ref().map(|a| (m, a)))
        .collect();
    if rows.is_empty() {
        return None;
    }
    let alt =
        |a: &ThresholdAdvice, o: &str| pick_cell(a.alternatives.iter().find(|p| p.objective == o));
    let night = |a: &ThresholdAdvice| {
        a.by_dataset
            .iter()
            .chain(&a.by_tag)
            .filter(|g| g.key.contains("night"))
            .map(|g| format!("{} {}", g.key, pick_cell(g.best.as_ref())))
            .collect::<Vec<_>>()
            .join(", ")
    };
    Some(Section {
        heading: "Best confidence threshold".into(),
        level: 2,
        paragraphs: vec![
            "Per model on its recommended device: precision / recall / F1 (%, IoU 0.5) at the configured threshold and at the best one for the objective. The best threshold is exact on the 0.01 grid (every step from 0.05 is scored), the middle of the widest run of steps scoring within 0.002 of the best one; F2, precision:0.9 and youden show what those objectives would pick (youden over one sample per frame). A pick fitted on pseudo ground truth (another model's detections) is shown, never applied. Apply with `benchmark --apply-threshold` or the Benchmark page. The server threshold is what a model reports; Blue Iris may send its own min_confidence per request (it then replaces the server threshold for that request), and each camera's minimum confidence filters the returned objects again."
                .into(),
            "Frame level: a frame alerts for a class when its most confident prediction of the class reaches the threshold. ROC AUC is the macro mean over classes of the chance that a frame with the class scores higher than a frame without it (ties count half); TPR is the alert rate on frames with an object, FPR the false-alert rate on frames without one (classes pooled). Computed from the 0.05+ predictions: frames below 0.05 score 0, which can only understate the AUC slightly."
                .into(),
        ],
        tables: vec![Table {
            title: String::new(),
            numbers: Numbers::default(),
            head: [
                "model",
                "device",
                "objective",
                "configured",
                "P",
                "R",
                "F1",
                "best",
                "P",
                "R",
                "F1",
                "F1 gain",
                "f2 pick",
                "precision:0.9 pick",
                "youden pick",
                "ROC AUC",
                "frame TPR / FPR configured",
                "frame TPR / FPR best",
                "night",
            ]
            .map(String::from)
            .to_vec(),
            rows: rows
                .iter()
                .map(|(m, a)| {
                    // At the threshold configured now (see `super::current`).
                    let c = a.now_point();
                    let b = a.best.as_ref();
                    vec![
                        if a.relative {
                            format!("{} (relative)", m.model)
                        } else {
                            m.model.clone()
                        },
                        a.device.clone(),
                        a.objective.clone(),
                        now_cell(a),
                        pct(c.and_then(|c| c.precision)),
                        pct(c.and_then(|c| c.recall)),
                        pct(c.and_then(|c| c.f1)),
                        pick_cell(b),
                        pct(b.and_then(|b| b.point.precision)),
                        pct(b.and_then(|b| b.point.recall)),
                        pct(b.and_then(|b| b.point.f1)),
                        delta(b.and_then(|b| b.point.f1), c.and_then(|c| c.f1)),
                        alt(a, "f2"),
                        alt(a, "precision:0.9"),
                        alt(a, "youden"),
                        pct(a.roc.as_ref().and_then(|r| r.auc)),
                        format!("{} / {}", tpr(c), fpr(c)),
                        format!(
                            "{} / {}",
                            tpr(b.map(|b| &b.point)),
                            fpr(b.map(|b| &b.point))
                        ),
                        night(a),
                    ]
                })
                .collect(),
        }],
    })
}

fn group_table(title: &str, a: &ThresholdAdvice, groups: &[GroupPick]) -> Table {
    Table {
        title: title.to_string(),
        numbers: Numbers::default(),
        head: [
            "",
            "images",
            "objects",
            "best",
            "P",
            "R",
            "F1",
            "configured F1",
            "ROC AUC",
        ]
        .map(String::from)
        .to_vec(),
        rows: groups
            .iter()
            .map(|g| {
                let b = g.best.as_ref();
                vec![
                    if g.relative {
                        format!("{} (relative)", g.key)
                    } else {
                        g.key.clone()
                    },
                    g.images.to_string(),
                    g.gt.to_string(),
                    pick_cell(b),
                    pct(b.and_then(|b| b.point.precision)),
                    pct(b.and_then(|b| b.point.recall)),
                    pct(b.and_then(|b| b.point.f1)),
                    pct(a.group_now(g).and_then(|p| p.f1)),
                    pct(g.roc_auc),
                ]
            })
            .collect(),
    }
}

/// The "Confidence threshold" tables of one model.
fn threshold_tables(a: &ThresholdAdvice, paragraphs: &mut Vec<String>, tables: &mut Vec<Table>) {
    let obj = a.objective().describe();
    match &a.best {
        Some(b) => {
            let c = a.now_point();
            let mut p = format!(
                "Confidence threshold ({obj}, on {}): best {:.2} with P {} R {} F1 {}, configured {} with P {} R {} F1 {} (F1 {}).",
                a.device,
                b.threshold,
                pct(b.point.precision),
                pct(b.point.recall),
                pct(b.point.f1),
                now_cell(a),
                pct(c.and_then(|c| c.precision)),
                pct(c.and_then(|c| c.recall)),
                pct(c.and_then(|c| c.f1)),
                delta(b.point.f1, c.and_then(|c| c.f1)),
            );
            if let (Some(e), Some([lo, hi])) = (b.exact_threshold, b.plateau) {
                p.push_str(&format!(
                    " Best 0.01 step {e:.2}; near-optimal range {lo:.2}..{hi:.2}."
                ));
            }
            if let Some(why) = a.not_on_labels() {
                p.push_str(&format!(" Not applied: {why}."));
            }
            if let (Some(c), Some(f)) = (
                c.and_then(|c| c.alert_rates().copied()),
                b.point.alert_rates().copied(),
            ) {
                p.push_str(&format!(
                    " Frame TPR / FPR: {} / {} at the best, {} / {} at the configured.",
                    pct(f.tpr),
                    pct(f.fpr),
                    pct(c.tpr),
                    pct(c.fpr)
                ));
            }
            if let Some(n) = &b.note {
                p.push_str(&format!(" Note: {n}."));
            }
            paragraphs.push(p);
        }
        None => paragraphs.push(format!(
            "Confidence threshold ({obj}, on {}): no ground truth for a curve.",
            a.device
        )),
    }
    let best = a.best.as_ref().map(|b| b.threshold);
    let now = a.now_threshold();
    let mut extra: Vec<ThresholdPoint> = a.now_point().copied().into_iter().collect();
    if let Some(b) = &a.best {
        extra.push(b.point);
    }
    tables.push(Table {
        title: format!("Confidence threshold on {} (%, IoU 0.5)", a.device),
        numbers: Numbers::default(),
        head: ["confidence", "P", "R", "F1", "F2", "FP / image", ""]
            .map(String::from)
            .to_vec(),
        rows: table_points(&a.curve, &extra)
            .iter()
            .map(|p| {
                let mut marks = Vec::new();
                if best.is_some_and(|b| (b - p.threshold).abs() < 1e-6) {
                    marks.push("best");
                }
                if (now - p.threshold).abs() < 1e-6 {
                    marks.push("configured");
                }
                vec![
                    format!("{:.2}", p.threshold),
                    pct(p.precision),
                    pct(p.recall),
                    pct(p.f1),
                    pct(p.f2),
                    format!("{:.2}", p.fp_per_image),
                    marks.join(", "),
                ]
            })
            .collect(),
    });
    let groups: Vec<GroupPick> = a.by_dataset.iter().chain(&a.by_tag).cloned().collect();
    if !groups.is_empty() {
        tables.push(group_table("Best threshold by dataset and tag", a, &groups));
    }
    if !a.per_class.is_empty() {
        tables.push(group_table("Best threshold by class", a, &a.per_class));
    }
    if a.by_device.len() > 1 {
        tables.push(Table {
            title: "Best threshold by device".into(),
            numbers: Numbers::default(),
            head: ["device", "best", "P", "R", "F1"]
                .map(String::from)
                .to_vec(),
            rows: a
                .by_device
                .iter()
                .map(|d| {
                    let b = d.best.as_ref();
                    vec![
                        d.device.clone(),
                        pick_cell(b),
                        pct(b.and_then(|b| b.point.precision)),
                        pct(b.and_then(|b| b.point.recall)),
                        pct(b.and_then(|b| b.point.f1)),
                    ]
                })
                .collect(),
        });
    }
}

fn sections(r: &BenchmarkResults) -> Vec<Section> {
    let mut out = vec![machine_section(r), datasets_section(r)];
    let ranking = r.ranking();
    let metric = r.accuracy_metric();
    if !ranking.is_empty() {
        out.push(ranking_section(&ranking, metric));
    }
    if let Some(s) = threshold_summary(r) {
        out.push(s);
    }
    for m in &r.models {
        out.push(model_section(m));
    }
    out.push(grades_section(metric));
    out
}

/// Hardware and runtime versions.
fn machine_section(r: &BenchmarkResults) -> Section {
    let hw = &r.hardware;
    let rt = &r.runtimes;
    Section {
        heading: "Machine".into(),
        level: 2,
        paragraphs: vec![
            format!(
                "{} {} | CPU {} | GPUs: {}",
                hw.os,
                hw.arch,
                if hw.cpu.is_empty() { "?" } else { &hw.cpu },
                if hw.gpus.is_empty() {
                    "none".to_string()
                } else {
                    hw.gpus.join("; ")
                }
            ),
            format!(
                "OpenVINO {} | ONNX Runtime {}{} | Blue Onyx Prism {} | updated {}",
                rt.openvino.as_deref().unwrap_or("not loaded"),
                rt.onnxruntime.as_deref().unwrap_or("not loaded"),
                rt.onnxruntime_flavor
                    .as_deref()
                    .map(|f| format!(" ({f})"))
                    .unwrap_or_default(),
                r.version,
                crate::clock::display_stored(&r.timestamp)
            ),
        ],
        tables: vec![],
    }
}

/// The image sets with their ground truth and licenses.
fn datasets_section(r: &BenchmarkResults) -> Section {
    Section {
        heading: "Datasets".into(),
        level: 2,
        paragraphs: vec![],
        tables: vec![Table {
            title: String::new(),
            numbers: Numbers::default(),
            head: [
                "dataset",
                "images",
                "scored against",
                "license",
                "attribution",
            ]
            .map(String::from)
            .to_vec(),
            rows: r
                .image_sets
                .iter()
                .map(|s| {
                    vec![
                        format!("{} ({})", s.id, s.title),
                        s.images.len().to_string(),
                        s.ground_truth.describe(),
                        s.license.clone(),
                        s.attribution.clone(),
                    ]
                })
                .collect(),
        }],
    }
}

/// The models ranked on their recommended devices (`ranking` is not empty).
fn ranking_section(ranking: &[super::report::RankRow], metric: AccuracyMetric) -> Section {
    Section {
        heading: "Best model for this machine".into(),
        level: 2,
        paragraphs: vec![format!(
            "Each model on its recommended device, ranked by overall grade, then accuracy, then \
             speed; accuracy graded by {}. Models with no class in the datasets are not \
             scored for accuracy and rank last. Best: {} on {}.",
            metric.label(),
            ranking.first().map_or("-", |r| r.model.as_str()),
            ranking.first().map_or("-", |r| r.device.as_str())
        )],
        tables: vec![Table {
            title: String::new(),
            // Rank, the grades' points and score, p50, AP50, ROC AUC; the rows are written in rank
            // order, the table's own order.
            numbers: Numbers {
                values: ranking
                    .iter()
                    .map(|k| {
                        vec![
                            Some(k.rank as f64),
                            None,
                            None,
                            Some(k.overall_points),
                            k.accuracy_score,
                            Some(k.speed.points()),
                            Some(k.p50_ms).filter(|v| *v < f64::MAX),
                            k.ap50,
                            k.roc_auc,
                        ]
                    })
                    .collect(),
                desc: vec![false, false, false, true, true, true, false, true, true],
            },
            head: [
                "#", "model", "device", "overall", "accuracy", "speed", "p50 ms", "AP50", "ROC AUC",
            ]
            .map(String::from)
            .to_vec(),
            rows: ranking
                .iter()
                .map(|k| {
                    vec![
                        k.rank.to_string(),
                        k.model.clone(),
                        k.device.clone(),
                        if k.accuracy.is_none() {
                            format!("{} (speed only)", k.overall)
                        } else {
                            k.overall.to_string()
                        },
                        k.accuracy.map_or("not scored".into(), |a| {
                            if k.relative {
                                format!("{a}*")
                            } else {
                                a.to_string()
                            }
                        }),
                        k.speed.to_string(),
                        msf(k.p50_ms),
                        pct(k.ap50),
                        pct(k.roc_auc),
                    ]
                })
                .collect(),
        }],
    }
}

/// One model: run settings, recommendation, device table, the recommended device's accuracy
/// and speed breakdowns and the threshold tables.
fn model_section(m: &ModelResult) -> Section {
    let mut paragraphs = vec![format!(
        "{} | datasets {} ({} images) | {} timed run(s) per image after {} warm-up | {}",
        m.path,
        m.datasets.join(", "),
        m.images,
        m.repeat,
        m.warmup,
        crate::clock::display_stored(&m.timestamp)
    )];
    if let Some(e) = &m.error {
        paragraphs.push(format!("Not benchmarked: {e}"));
    }
    paragraphs.push(match &m.recommended {
        Some(d) => format!("Recommended device: {d} ({})", m.recommendation),
        None => format!("No recommendation: {}", m.recommendation),
    });
    if !m.skipped.is_empty() {
        paragraphs.push(format!("Not runnable here: {}", m.skipped.join("; ")));
    }
    paragraphs.extend(m.shared_load_warnings());
    let mut tables = vec![Table {
        title: "Devices".into(),
        numbers: Numbers::default(),
        head: [
            "device",
            "first load",
            "infer p50",
            "total p50",
            "total p95",
            "total p99",
            "req/s",
            "AP50",
            "AP50-95",
            "ROC AUC",
            "P",
            "R",
            "overall",
            "accuracy",
            "speed",
            "agreement",
        ]
        .map(String::from)
        .to_vec(),
        rows: m.devices.iter().map(|d| device_row(m, d)).collect(),
    }];
    if let Some(t) = burst_table(m) {
        paragraphs.push(
            "Burst test: cameras triggering together, each sending its images a fixed time \
             apart, three bursts per device on a temporary worker of the serving code; each \
             request is timed from when it was due (queue wait included), a trigger until its \
             last answer."
                .into(),
        );
        tables.push(t);
    }
    let best = m
        .recommended
        .as_ref()
        .and_then(|rec| m.devices.iter().find(|d| &d.device == rec))
        .and_then(|d| d.run.as_ref());
    if let Some(run) = best {
        if let Some(a) = &run.accuracy {
            accuracy_tables(run, a, &mut paragraphs, &mut tables);
        } else if let Some(n) = &run.accuracy_note {
            paragraphs.push(format!("Accuracy not scored: {n}."));
        }
        if !run.by_resolution.is_empty() {
            tables.push(resolution_table(run));
        }
    }
    if let Some(a) = &m.threshold {
        threshold_tables(a, &mut paragraphs, &mut tables);
    }
    Section {
        heading: m.model.clone(),
        level: 2,
        paragraphs,
        tables,
    }
}

/// The accuracy `a` of the recommended device's `run`: by dataset, tag, class, frame-level ROC
/// and object size.
fn accuracy_tables(
    run: &RunResult,
    a: &AccuracyReport,
    paragraphs: &mut Vec<String>,
    tables: &mut Vec<Table>,
) {
    paragraphs.push(format!(
        "Accuracy of {} ({}; classes {}; P/R at confidence {:.2}, IoU 0.5).",
        run.requested_device,
        a.ground_truth,
        a.classes.join(", "),
        a.threshold
    ));
    if let Some(b) = &a.overall.ap_basis {
        paragraphs.push(ap_basis_note(b));
    }
    tables.push(breakdown_table("By dataset", &a.by_dataset));
    tables.push(breakdown_table("By tag", a.shown_tags()));
    tables.push(Table {
        title: "By class".into(),
        numbers: Numbers::default(),
        head: ["class", "objects", "AP50", "AP50-95", "P", "R"]
            .map(String::from)
            .to_vec(),
        rows: a
            .overall
            .per_class
            .iter()
            .map(|c| {
                vec![
                    c.class.clone(),
                    c.gt.to_string(),
                    pct(c.ap50),
                    pct(c.ap50_95),
                    pct(c.precision),
                    pct(c.recall),
                ]
            })
            .collect(),
    });
    if let Some(roc) = &a.frame_roc {
        paragraphs.push(format!(
            "Frame-level ROC AUC of {}: macro {}, micro {} ({} positive and {} negative frame samples; a frame counts once per class).",
            run.requested_device,
            pct(roc.auc),
            pct(roc.micro_auc),
            roc.positives,
            roc.negatives
        ));
        if let Some(b) = &roc.auc_basis {
            paragraphs.push(format!(
                "The macro ROC AUC averages {} (counted in frames with the class).",
                b.describe()
            ));
        }
        tables.push(Table {
            title: "Frame-level ROC by class".into(),
            numbers: Numbers::default(),
            head: ["class", "frames with", "frames without", "ROC AUC"]
                .map(String::from)
                .to_vec(),
            rows: roc
                .per_class
                .iter()
                .map(|c| {
                    vec![
                        c.class.clone(),
                        c.positives.to_string(),
                        c.negatives.to_string(),
                        pct(c.auc),
                    ]
                })
                .collect(),
        });
    }
    tables.push(Table {
        title: "Recall by object size".into(),
        numbers: Numbers::default(),
        head: ["size", "objects", "recall"].map(String::from).to_vec(),
        rows: a
            .overall
            .by_size
            .iter()
            .map(|b| vec![b.bucket.clone(), b.gt.to_string(), pct(b.recall)])
            .collect(),
    });
}

/// The footnote naming the classes AP50 / AP50-95 average.
fn ap_basis_note(b: &super::metrics::ApBasis) -> String {
    if b.fell_back {
        format!(
            "AP50 and AP50-95 average {}, so they rest on few objects.",
            b.describe()
        )
    } else {
        format!(
            "AP50 and AP50-95 average {}; classes with fewer objects are scored in the By class \
             table only.",
            b.describe()
        )
    }
}

/// The burst test of each device of `m` that ran one ([`super::burst`]); None when none did.
fn burst_table(m: &ModelResult) -> Option<Table> {
    let ran: Vec<(&str, &super::burst::BurstResult)> = m
        .devices
        .iter()
        .filter_map(|d| Some((d.device.as_str(), d.run.as_ref()?.burst.as_ref()?)))
        .collect();
    if ran.is_empty() {
        return None;
    }
    let ok = |b: &super::burst::BurstResult| b.error.is_none() && b.answered > 0;
    let rows: Vec<Vec<String>> = ran
        .iter()
        .map(|(dev, b)| {
            let mut row = vec![dev.to_string(), b.load_text(), b.requests.to_string()];
            if !ok(b) {
                row.push(format!(
                    "not measured: {}",
                    b.error.as_deref().unwrap_or("no request was answered")
                ));
                row.resize(10, String::new());
                return row;
            }
            let refused = b.refused_total();
            row.extend([
                msf(b.latency_ms.p50),
                msf(b.latency_ms.p95),
                msf(b.latency_ms.p99),
                if b.triggers > b.triggers_incomplete {
                    msf(b.trigger_ms.p50)
                } else {
                    "-".into()
                },
                if b.triggers > b.triggers_incomplete {
                    msf(b.trigger_ms.p95)
                } else {
                    "-".into()
                },
                if refused == 0 {
                    "0".into()
                } else {
                    format!(
                        "{refused} ({})",
                        b.refused
                            .iter()
                            .map(|(k, v)| format!("{k} {v}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                },
                b.max_queue_depth.to_string(),
            ]);
            row
        })
        .collect();
    Some(Table {
        title: "Burst test (ms)".into(),
        numbers: Numbers::default(),
        head: [
            "device",
            "load (cameras x images x interval)",
            "requests",
            "request p50",
            "request p95",
            "request p99",
            "last answer p50",
            "last answer p95",
            "refused",
            "most waiting",
        ]
        .map(String::from)
        .to_vec(),
        rows,
    })
}

/// Speed per resolution bucket of `run` (every bucket; one without images is n/a).
fn resolution_table(run: &RunResult) -> Table {
    Table {
        title: format!("Speed by resolution on {} (ms)", run.requested_device),
        numbers: Numbers::default(),
        head: [
            "resolution",
            "images",
            "decode+pre p50",
            "infer p50",
            "post p50",
            "total p50",
            "total p95",
        ]
        .map(String::from)
        .to_vec(),
        rows: super::images::RESOLUTION_BUCKETS
            .iter()
            .map(
                |bucket| match run.by_resolution.iter().find(|b| b.bucket == *bucket) {
                    Some(b) => vec![
                        b.bucket.clone(),
                        b.images.to_string(),
                        msf(b.pre_p50),
                        msf(b.infer_p50),
                        msf(b.post_p50),
                        msf(b.total_p50),
                        msf(b.total_p95),
                    ],
                    None => {
                        let mut v = vec![bucket.to_string(), "0".into()];
                        v.resize(7, "n/a".into());
                        v
                    }
                },
            )
            .collect(),
    }
}

/// How the speed, accuracy and overall grades are computed.
fn grades_section(metric: AccuracyMetric) -> Section {
    Section {
        heading: "How grades are computed".into(),
        level: 2,
        paragraphs: vec![
            format!(
                "Speed: full request p50 (decode, preprocess, inference, postprocess) A < {} ms, B < {}, C < {}, D < {}, else F.",
                SPEED_THRESHOLDS_MS[0], SPEED_THRESHOLDS_MS[1], SPEED_THRESHOLDS_MS[2], SPEED_THRESHOLDS_MS[3]
            ),
            format!(
                "Accuracy is graded by {} (config benchmark.accuracy_metric).",
                metric.label()
            ),
            format!(
                "Accuracy, AP50 metric: AP@0.5 over the CCTV classes the model has (person, bicycle, car, motorcycle, bus, truck, dog, cat, bird, horse; IPcam 'vehicle' = car/truck/bus), averaged over the classes with at least {} ground-truth objects (every class when none has that many; rarer classes are scored per class only), blended {:.0}% with small-object recall when there are small objects: A >= {:.2}, B >= {:.2}, C >= {:.2}, D >= {:.2}, else F. * = relative to a reference model, not ground truth.",
                super::metrics::MIN_CLASS_GT,
                SMALL_RECALL_WEIGHT * 100.0,
                ACCURACY_THRESHOLDS[0], ACCURACY_THRESHOLDS[1], ACCURACY_THRESHOLDS[2], ACCURACY_THRESHOLDS[3]
            ),
            format!(
                "Accuracy, ROC AUC metric: macro frame-level ROC AUC over the same classes (those with at least {} frames with the class; no small-object component): A >= {:.2}, B >= {:.2}, C >= {:.2}, D >= {:.2}, else F (0.5 is chance).",
                super::metrics::MIN_CLASS_GT,
                ROC_AUC_THRESHOLDS[0], ROC_AUC_THRESHOLDS[1], ROC_AUC_THRESHOLDS[2], ROC_AUC_THRESHOLDS[3]
            ),
            "Overall: weighted mean of the grade points (A=4 .. F=0), default 60% accuracy, 40% speed (config benchmark.weights). Devices whose detections disagree with the CPU reference are never recommended.".into(),
        ],
        tables: vec![],
    }
}

/// Text for a Markdown heading, paragraph or table title: one line, inline HTML neutralized,
/// leading block syntax escaped and every `[` escaped, so no link or image (`![x](url)`, a
/// remote image a viewer would fetch) forms anywhere in the text; a `\` before ASCII
/// punctuation is escaped too, so the text's own `\[` cannot undo that (BM-08, QA-060: model
/// names, dataset ids and runtime error strings are not ours). Other backslashes (Windows
/// paths) stay as they are.
fn md_text(s: &str) -> String {
    md_escape(s, false)
}

/// Text for a Markdown table cell: [`md_text`] plus every `\` and `|` escaped (a `\` before
/// `|` would otherwise defeat the `\|`).
fn md_cell(s: &str) -> String {
    md_escape(s, true)
}

/// [`md_text`] (`cell` false) and [`md_cell`] (`cell` true).
fn md_escape(s: &str, cell: bool) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    let mut first = true;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\n' | '\r' => out.push(' '),
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\\' if cell || chars.peek().is_some_and(char::is_ascii_punctuation) => {
                out.push_str("\\\\");
            }
            '|' if cell || first => out.push_str("\\|"),
            '#' | '*' | '_' | '`' | '-' | '+' if first => {
                out.push('\\');
                out.push(c);
            }
            '[' => out.push_str("\\["),
            c => out.push(c),
        }
        first = first && c.is_whitespace();
    }
    out
}

/// Markdown report.
pub fn markdown(r: &BenchmarkResults) -> String {
    let mut out = String::from("# Blue Onyx Prism benchmark report\n\n");
    for s in sections(r) {
        let _ = writeln!(
            out,
            "{} {}\n",
            "#".repeat(s.level as usize),
            md_text(&s.heading)
        );
        for p in &s.paragraphs {
            let _ = writeln!(out, "{}\n", md_text(p));
        }
        for t in &s.tables {
            if t.rows.is_empty() {
                continue;
            }
            if !t.title.is_empty() {
                let _ = writeln!(out, "**{}**\n", md_text(&t.title));
            }
            let _ = writeln!(
                out,
                "| {} |",
                t.head
                    .iter()
                    .map(|h| md_cell(h))
                    .collect::<Vec<_>>()
                    .join(" | ")
            );
            let _ = writeln!(out, "|{}", " --- |".repeat(t.head.len()));
            for row in &t.rows {
                let _ = writeln!(
                    out,
                    "| {} |",
                    row.iter()
                        .map(|c| md_cell(c))
                        .collect::<Vec<_>>()
                        .join(" | ")
                );
            }
            out.push('\n');
        }
    }
    out
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Self-contained HTML report.
pub fn html(r: &BenchmarkResults) -> String {
    let mut out = String::from(
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <title>Blue Onyx Prism benchmark report</title><style>\
         :root{--bg:#f6f7f9;--fg:#1d2330;--muted:#5d6677;--card:#fff;--border:#d9dde5}\
         @media (prefers-color-scheme: dark){:root{--bg:#12151b;--fg:#dfe4ec;--muted:#949db0;--card:#1b2029;--border:#2d3442}}\
         body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}\
         main{max-width:1200px;margin:0 auto;padding:1.5rem}p{color:var(--muted)}\
         .scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--border);margin:.5rem 0 1rem}\
         th,td{padding:.35rem .6rem;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}\
         th{font-size:.85rem;color:var(--muted)}h3{font-size:1rem}\
         th.th-sortable{cursor:pointer;user-select:none}\
         th .th-sort{margin-left:.25rem;padding:0 .2rem;border:0;background:none;color:inherit;font:inherit;font-size:.8em;cursor:pointer}\
         th .th-sort .th-sort-arrow{opacity:.45}th .th-sort.on .th-sort-arrow{opacity:1}th .th-sort .th-sort-rank{font-weight:700}\
         .sort-tools{display:flex;justify-content:flex-end;gap:.6rem;font-size:.85rem}\
         .sort-tools[hidden],.sort-tools [hidden]{display:none}.sort-tools .sort-by{display:none}\
         @media (max-width:700px){.sort-tools{justify-content:flex-start}.sort-tools .sort-by{display:inline-flex}}\
         .sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}\
         </style></head><body><main>\n<h1>Blue Onyx Prism benchmark report</h1>\n",
    );
    let mut index = 0;
    for s in sections(r) {
        let _ = writeln!(out, "<h{l}>{}</h{l}>", html_escape(&s.heading), l = s.level);
        for p in &s.paragraphs {
            let _ = writeln!(out, "<p>{}</p>", html_escape(p));
        }
        for t in &s.tables {
            if t.rows.is_empty() {
                continue;
            }
            if !t.title.is_empty() {
                let _ = writeln!(out, "<h3>{}</h3>", html_escape(&t.title));
            }
            // Each table sorts on its own (PrismSort, attached by its `data-sort` key).
            index += 1;
            let _ = write!(
                out,
                "<div class=\"scroll\"><table data-sort=\"report-{index}\"><thead><tr>"
            );
            for (i, h) in t.head.iter().enumerate() {
                let first = if t.numbers.desc.get(i).copied().unwrap_or(false) {
                    " data-sort-first=\"desc\""
                } else {
                    ""
                };
                let _ = write!(out, "<th{first}>{}</th>", html_escape(h));
            }
            out.push_str("</tr></thead><tbody>\n");
            for (k, row) in t.rows.iter().enumerate() {
                out.push_str("<tr>");
                for (i, c) in row.iter().enumerate() {
                    let v = t
                        .numbers
                        .values
                        .get(k)
                        .and_then(|v| v.get(i).copied().flatten())
                        .filter(|v| v.is_finite());
                    match v {
                        Some(v) => {
                            let _ = write!(out, "<td data-v=\"{v}\">{}</td>", html_escape(c));
                        }
                        None => {
                            let _ = write!(out, "<td>{}</td>", html_escape(c));
                        }
                    }
                }
                out.push_str("</tr>\n");
            }
            out.push_str("</tbody></table></div>\n");
        }
    }
    // The sorter inline (opened from disk: no server to load it from); `</` escaped so nothing in
    // it can close the element early.
    out.push_str("</main>\n<script>\n");
    out.push_str(&SORT_JS.replace("</", "<\\/"));
    out.push_str("\n</script></body></html>\n");
    out
}

/// Write the report for `path` by its extension (`.html`/`.htm` or `.md`/`.markdown`).
pub fn write(r: &BenchmarkResults, path: &Path) -> Result<()> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let text = match ext.as_str() {
        "html" | "htm" => html(r),
        "md" | "markdown" => markdown(r),
        _ => bail!("--report {}: use a .html or .md file name", path.display()),
    };
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The report names the classes AP and the macro ROC AUC average.
    #[test]
    fn accuracy_paragraphs_name_the_averaged_classes() {
        use super::super::metrics::{ApBasis, Summary};
        let run = super::super::report::test_support::run("openvino:cpu", 10.0, vec![]);
        let basis = |classes: &[&str], fell_back: bool| ApBasis {
            min_gt: 20,
            classes: classes.iter().map(|c| c.to_string()).collect(),
            fell_back,
        };
        let a = AccuracyReport {
            ground_truth: "ground truth".into(),
            relative: false,
            classes: vec!["person".into(), "car".into(), "dog".into()],
            threshold: 0.5,
            overall: Summary {
                ap_basis: Some(basis(&["person", "car"], false)),
                ..Default::default()
            },
            by_dataset: vec![],
            by_tag: vec![],
            all_classes: None,
            sweep: None,
            frame_roc: Some(super::super::roc::FrameRoc {
                auc_basis: Some(basis(&["person", "dog"], true)),
                ..Default::default()
            }),
        };
        let (mut paragraphs, mut tables) = (Vec::new(), Vec::new());
        accuracy_tables(&run, &a, &mut paragraphs, &mut tables);
        let text = paragraphs.join("\n");
        assert!(
            text.contains(
                "AP50 and AP50-95 average over person, car (\u{2265} 20 objects each); classes \
                 with fewer objects are scored in the By class table only."
            ),
            "{text}"
        );
        assert!(
            text.contains(
                "The macro ROC AUC averages every class: none had 20 objects (counted in frames \
                 with the class)."
            ),
            "{text}"
        );
        assert!(
            grades_section(AccuracyMetric::Ap50).paragraphs[2]
                .contains("at least 20 ground-truth objects")
        );
    }

    /// A device's burst test is a table of the model's section (and its failure a row that
    /// says why); a model without one has no such table.
    #[test]
    fn the_report_shows_the_burst_test() {
        use super::super::burst::BurstResult;
        use super::super::report::test_support::{dev, model, scene};
        let mut d = dev("openvino:cpu", 20.0, scene());
        let mut b = BurstResult {
            cameras: 8,
            images: 4,
            interval_ms: 500,
            bursts: 3,
            requests: 96,
            answered: 94,
            triggers: 24,
            triggers_incomplete: 2,
            max_queue_depth: 12,
            ..BurstResult::default()
        };
        b.latency_ms.p95 = 480.0;
        b.trigger_ms.p95 = 1600.0;
        b.refused.insert("queue_full".into(), 2);
        d.run.as_mut().unwrap().burst = Some(b.clone());
        let mut failed = dev("ort:coreml", 10.0, scene());
        failed.run.as_mut().unwrap().burst = Some(BurstResult {
            error: Some("its worker did not load on ort:coreml".into()),
            ..b
        });
        let m = model(vec![d, failed], None);
        let r = BenchmarkResults::new(Default::default(), Default::default(), vec![m]);
        let md = markdown(&r);
        assert!(md.contains("Burst test (ms)"), "{md}");
        assert!(md.contains("8 x 4 x 500 ms"), "{md}");
        assert!(
            md.contains("| 480.0 |") && md.contains("| 1600.0 |"),
            "{md}"
        );
        assert!(md.contains("2 (queue_full 2)"), "{md}");
        assert!(
            md.contains("not measured: its worker did not load on ort:coreml"),
            "{md}"
        );
        assert!(html(&r).contains("Burst test (ms)"));
        let plain = model(vec![dev("openvino:cpu", 20.0, scene())], None);
        let r = BenchmarkResults::new(Default::default(), Default::default(), vec![plain]);
        assert!(!markdown(&r).contains("Burst test"));
    }

    #[test]
    fn reports_render_and_escape() {
        let mut m = ModelResult::failed("<b>m</b>", "m.onnx", "file | missing".into());
        m.datasets = vec!["sample".into()];
        let r = BenchmarkResults::new(Default::default(), Default::default(), vec![m])
            .with_sets(&[super::super::images::ImageSet::sample()]);
        let md = markdown(&r);
        assert!(md.starts_with("# Blue Onyx Prism benchmark report"));
        assert!(md.contains("## &lt;b&gt;m&lt;/b&gt;"), "{md}");
        assert!(!md.contains("<b>"), "{md}");
        assert!(md.contains("Not benchmarked: file | missing"));
        // BM-08: block syntax, inline HTML and newlines in untrusted text are neutralized.
        assert_eq!(md_text("# a\nb <c> & d"), "\\# a b &lt;c&gt; &amp; d");
        assert_eq!(md_text("  * list"), "  \\* list");
        assert_eq!(md_text("plain * mid"), "plain * mid");
        // QA-060: no image or link mid-text either.
        assert_eq!(
            md_text("dir:/x/![a](http://h/i.png) [l](http://h)"),
            "dir:/x/!\\[a](http://h/i.png) \\[l](http://h)"
        );
        assert_eq!(md_cell("e [x](u)"), "e \\[x](u)");
        // The text's own backslash cannot undo the escape; a Windows path stays readable.
        assert_eq!(md_text("a \\[x](u)"), "a \\\\\\[x](u)");
        assert_eq!(md_text("C:\\models\\m.onnx"), "C:\\models\\m.onnx");
        assert_eq!(md_cell("a \\| b"), "a \\\\\\| b");
        assert_eq!(md_cell("| x"), "\\| x");
        assert!(md.contains("| sample (Embedded sample"), "{md}");
        assert!(md.contains("How grades are computed"));
        let h = html(&r);
        assert!(h.contains("&lt;b&gt;m&lt;/b&gt;") && !h.contains("<b>m</b>"));
        assert!(h.contains("prefers-color-scheme"));
        let dir = std::env::temp_dir().join(format!("bop-rep-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        write(&r, &dir.join("r.md")).unwrap();
        write(&r, &dir.join("r.HTML")).unwrap();
        assert!(write(&r, &dir.join("r.txt")).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The HTML report's tables sort opened from disk: the sorter is inlined, every table has
    /// its own `data-sort` key, the ranking table's numbers carry `data-v` (its rows stay in rank
    /// order); Markdown is unchanged by it.
    #[test]
    fn the_html_report_sorts_its_tables() {
        use super::super::report::test_support::{graded, model};
        let mut a = model(vec![graded("openvino:cpu", 12.5, Some(0.9))], None);
        a.model = "first".into();
        let mut b = model(vec![graded("openvino:cpu", 40.0, Some(0.3))], None);
        b.model = "second".into();
        let r = BenchmarkResults::new(Default::default(), Default::default(), vec![b, a]);
        let h = html(&r);
        assert!(h.contains("<script>\n"), "the sorter is inline");
        assert!(h.contains("root.PrismSort = {"));
        assert!(h.contains("\n</script></body></html>\n"));
        // Nothing inside the script closes it early.
        let script = h.split("<script>").nth(1).unwrap();
        assert_eq!(
            script.matches("</").count(),
            3,
            "</script></body></html> only"
        );
        assert!(script.contains("PrismSort.attach"));
        assert!(h.contains("<table data-sort=\"report-1\">"));
        assert!(
            h.contains("<th data-sort-first=\"desc\">overall</th>"),
            "{h}"
        );
        let ranking = h
            .split("Best model for this machine")
            .nth(1)
            .and_then(|s| s.split("</table>").next())
            .unwrap();
        assert!(ranking.contains("<th>#</th>"));
        assert!(
            ranking.contains("<td data-v=\"1\">1</td><td>first</td>"),
            "{ranking}"
        );
        assert!(
            ranking.contains("<td data-v=\"2\">2</td><td>second</td>"),
            "{ranking}"
        );
        assert!(
            ranking.contains("<td data-v=\"12.5\">12.5</td>"),
            "{ranking}"
        );
        assert!(ranking.find("first").unwrap() < ranking.find("second").unwrap());
        let md = markdown(&r);
        assert!(!md.contains("data-v") && !md.contains("<script"), "{md}");
        assert!(md.contains("| 1 | first |"), "{md}");
    }

    #[test]
    fn threshold_sections() {
        use super::super::metrics::{Counts, SWEEP_THRESHOLDS, ThresholdPoint};
        use super::super::threshold::{GroupPick, Pick, ThresholdAdvice};
        let pt = |t: f32| {
            // Fewer detections as the threshold rises.
            let tp = (10.0 * (1.0 - t)) as usize;
            ThresholdPoint::new(
                t,
                Counts {
                    tp,
                    fp: (20.0 * (1.0 - t) * (1.0 - t)) as usize,
                    fn_: 10 - tp,
                },
                4,
            )
            .with_frame(Some(super::super::roc::FrameRates::new(
                (8.0 * (1.0 - t)) as usize,
                (4.0 * (1.0 - t)) as usize,
                8,
                4,
            )))
        };
        let pick = |t: f32, o: &str, met: bool| Pick {
            objective: o.into(),
            threshold: t,
            point: pt(t),
            met,
            note: (!met).then(|| "unreachable".into()),
            exact_threshold: Some(t + 0.004),
            exact_score: Some(0.7),
            plateau: Some([t - 0.05, t + 0.05]),
            relative: false,
        };
        let mut m = ModelResult::failed("yolo26n", "m.onnx", String::new());
        m.error = None;
        m.threshold = Some(ThresholdAdvice {
            objective: "f1".into(),
            device: "ort:coreml".into(),
            relative: false,
            configured: pt(0.5),
            best: Some(pick(0.37, "f1", true)),
            alternatives: vec![
                pick(0.37, "f1", true),
                pick(0.21, "f2", true),
                pick(0.9, "precision:0.9", false),
                pick(0.33, "youden", true),
            ],
            curve: SWEEP_THRESHOLDS.iter().map(|&t| pt(t)).collect(),
            exact: vec![],
            by_dataset: vec![GroupPick {
                key: "exdark-night".into(),
                images: 2,
                gt: 5,
                relative: false,
                configured: pt(0.5),
                best: Some(pick(0.25, "f1", true)),
                roc_auc: Some(0.912),
                now: None,
            }],
            by_tag: vec![],
            per_class: vec![],
            by_device: vec![],
            roc: Some(super::super::roc::FrameRoc {
                auc: Some(0.934),
                ..Default::default()
            }),
            now: None,
        });
        let r = BenchmarkResults::new(Default::default(), Default::default(), vec![m]);
        let md = markdown(&r);
        assert!(md.contains("## Best confidence threshold"), "{md}");
        assert!(md.contains("| yolo26n | ort:coreml | f1 | 0.50 |"), "{md}");
        // f2, precision:0.9 and youden picks, AUC, frame TPR / FPR at configured (0.50: 4 of 8,
        // 2 of 4) and best (0.37: 5 of 8, 2 of 4), night.
        assert!(
            md.contains(
                "| 0.21 | 0.90 (target not reached) | 0.33 | 93.4 | 50.0 / 50.0 | 62.5 / 50.0 | exdark-night 0.25 |"
            ),
            "{md}"
        );
        assert!(
            md.contains("Frame TPR / FPR: 62.5 / 50.0 at the best"),
            "{md}"
        );
        assert!(md.contains("| configured F1 | ROC AUC |"), "{md}");
        assert!(md.contains("| 91.2 |"), "{md}");
        assert!(
            md.contains("ROC AUC metric: macro frame-level ROC AUC"),
            "{md}"
        );
        assert!(md.contains("**Confidence threshold on ort:coreml (%, IoU 0.5)**"));
        // 0.20..0.80 by 0.05 plus the off-grid best 0.37; marks on best and configured.
        assert!(md.contains("| 0.37 |") && md.contains("| best |"), "{md}");
        assert!(md.contains("| configured |"));
        assert!(!md.contains("| 0.15 |") && md.contains("| 0.80 |"));
        assert!(
            md.contains("Best 0.01 step 0.37; near-optimal range"),
            "{md}"
        );
        assert!(md.contains("min_confidence"));
        let h = html(&r);
        assert!(h.contains("<h2>Best confidence threshold</h2>"));
    }
}
