//! Day / night / mixed comparison of runs queued together (`TrainRequest.compare_group`): one
//! row per run with its held-out AP per scene ([`BenchSummary::by_scene`]) and speed, the best
//! value of each column, and a "routed" row per model when the group has a day run and a night
//! (or IR) run: the day run's day AP beside the night run's night / IR AP, as a service with
//! `night_model` routing would score. Scenes are scored separately, so no combined AP is made
//! up. Beside AP, the alerting figures of the post-training benchmark at its configured threshold
//! (every held-out frame): person recall, vehicle recall (car/truck/bus merged) and false alerts
//! on frames without any object (best = highest recall, lowest false-alert rate). Pure; the
//! Training page and `train run --scene ... --scene ...` render it. Runs that differ only by
//! training seed (`--seeds`, "Repeat with N seeds") also get a "mean of N seeds" summary
//! ([`SeedSummary`]: mean and range per column; never the best of a column).

use super::dataset::{Balance, Excluded, SceneFilter};
use super::job::{BenchSummary, SceneAp};
use serde::Serialize;
use std::collections::BTreeMap;

/// One finished (or failed) run of a group.
#[derive(Debug, Clone, PartialEq)]
pub struct CompareRun {
    /// The trained model's name.
    pub name: String,
    /// Model label (`yolo26n`, `rfdetr-nano`).
    pub model: String,
    pub scene: SceneFilter,
    /// The run's classes (head order).
    pub classes: Vec<String>,
    pub excluded: Excluded,
    /// How the run balanced its training frames (off for older runs).
    pub balance: Balance,
    /// Best epoch (1-based) and the epoch training stopped at, when known.
    pub best_epoch: Option<u32>,
    pub stopped_epoch: Option<u32>,
    /// The run's training seed (`TrainRequest.train_seed`; None = not a repeat).
    pub seed: Option<u64>,
    /// None while the run has no benchmark (running, failed, or nothing held out).
    pub bench: Option<BenchSummary>,
}

/// AP50-95 and AP50 of a column.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Default)]
pub struct Cell {
    pub ap50_95: Option<f64>,
    pub ap50: Option<f64>,
    pub images: usize,
}

impl From<&SceneAp> for Cell {
    fn from(a: &SceneAp) -> Self {
        Self {
            ap50_95: a.ap50_95,
            ap50: a.ap50,
            images: a.images,
        }
    }
}

/// One row of the table.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Row {
    /// Run name, or `<day run> + <night run>` for a routed row.
    pub name: String,
    pub model: String,
    /// Canonical scene filter (`day+night routed` for a routed row).
    pub scene: String,
    /// A routed row (copies of other rows' cells, not a run).
    pub routed: bool,
    /// The run's training seed, when it is one of several repeats.
    pub seed: Option<u64>,
    pub best_epoch: Option<u32>,
    pub stopped_epoch: Option<u32>,
    pub day: Cell,
    pub night: Cell,
    pub ir: Cell,
    /// Every held-out frame (None for a routed row).
    pub all: Cell,
    /// AP50-95 over the scenes the run trained on.
    pub in_scope: Option<f64>,
    pub ms_p50: Option<f64>,
    /// Confidence threshold of the alerting figures below (None for a routed row and runs of
    /// older versions, like the figures).
    pub threshold: Option<f32>,
    /// Recall of `person` at `threshold` over every held-out frame.
    pub person_recall: Option<f64>,
    /// Recall of vehicles (car/truck/bus/vehicle merged) at `threshold`.
    pub vehicle_recall: Option<f64>,
    /// False alerts on held-out frames without any object.
    pub false_alerts: Option<FalseAlerts>,
}

/// Empty frames (no scored object) that alerted at the threshold, of all empty frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct FalseAlerts {
    pub alerts: usize,
    pub empty_frames: usize,
}

impl FalseAlerts {
    /// From a benchmark summary; None when it has no empty-frame counts.
    pub fn of(b: &BenchSummary) -> Option<Self> {
        Some(Self {
            alerts: b.empty_frame_alerts?,
            empty_frames: b.empty_frames?,
        })
    }

    /// Share of the empty frames that alerted (None without empty frames).
    pub fn rate(&self) -> Option<f64> {
        (self.empty_frames > 0).then(|| self.alerts as f64 / self.empty_frames as f64)
    }

    /// `3/130`.
    pub fn text(&self) -> String {
        format!("{}/{}", self.alerts, self.empty_frames)
    }
}

/// Which way a column improves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Better {
    Higher,
    Lower,
}

/// Column ids of [`Comparison::best_per_column`] and the direction each improves in. A new
/// column is one entry here plus its arm in [`value`].
pub const COLUMNS: [(&str, Better); 12] = [
    ("day", Better::Higher),
    ("night", Better::Higher),
    ("ir", Better::Higher),
    ("all", Better::Higher),
    ("day_ap50", Better::Higher),
    ("night_ap50", Better::Higher),
    ("ir_ap50", Better::Higher),
    ("all_ap50", Better::Higher),
    ("ms_p50", Better::Lower),
    ("person_recall", Better::Higher),
    ("vehicle_recall", Better::Higher),
    // The false-alert rate (alerts / empty frames).
    ("false_alerts", Better::Lower),
];

/// Mean and range of one column over the runs of a [`SeedSummary`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Spread {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
}

/// "Mean of N seeds": the repeats of one model on one scene filter.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SeedSummary {
    /// `mean of 3 seeds`.
    pub label: String,
    pub model: String,
    pub scene: String,
    /// Runs summarised.
    pub runs: usize,
    /// Column id ([`COLUMNS`]) -> spread over the runs that have a value for it.
    pub columns: BTreeMap<String, Spread>,
}

fn seed_summaries(rows: &[Row]) -> Vec<SeedSummary> {
    let mut keys: Vec<(&str, &str)> = Vec::new();
    for r in rows.iter().filter(|r| !r.routed && r.seed.is_some()) {
        let k = (r.model.as_str(), r.scene.as_str());
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    let mut out = Vec::new();
    for (model, scene) in keys {
        let set: Vec<&Row> = rows
            .iter()
            .filter(|r| !r.routed && r.seed.is_some() && r.model == model && r.scene == scene)
            .collect();
        if set.len() < 2 {
            continue;
        }
        let mut columns = BTreeMap::new();
        for (col, _) in COLUMNS {
            let vals: Vec<f64> = set.iter().filter_map(|r| value(r, col)).collect();
            if vals.is_empty() {
                continue;
            }
            columns.insert(
                col.to_string(),
                Spread {
                    mean: vals.iter().sum::<f64>() / vals.len() as f64,
                    min: vals.iter().copied().fold(f64::INFINITY, f64::min),
                    max: vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                },
            );
        }
        out.push(SeedSummary {
            label: format!("mean of {} seeds", set.len()),
            model: model.to_string(),
            scene: scene.to_string(),
            runs: set.len(),
            columns,
        });
    }
    out
}

/// The comparison of a group.
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct Comparison {
    pub rows: Vec<Row>,
    /// Column id ([`COLUMNS`]: AP50-95 per scene, `<scene>_ap50`, `ms_p50`, `person_recall`,
    /// `vehicle_recall`, `false_alerts`) -> index of the best run row (highest AP / recall,
    /// lowest time / false-alert rate; ties: the first). Routed rows never count.
    pub best_per_column: BTreeMap<String, usize>,
    /// One summary per set of two or more runs of the same model and scene that differ only by
    /// training seed. Never counted for `best_per_column`.
    pub seed_summaries: Vec<SeedSummary>,
    /// False when the runs were not benchmarked on the same held-out frames.
    pub comparable: bool,
    pub warnings: Vec<String>,
}

fn row_of(r: &CompareRun) -> Row {
    let cell = |s: &str| -> Cell {
        r.bench
            .as_ref()
            .and_then(|b| b.by_scene.get(s))
            .map(Cell::from)
            .unwrap_or_default()
    };
    Row {
        name: r.name.clone(),
        model: r.model.clone(),
        scene: r.scene.as_str().to_string(),
        routed: false,
        seed: r.seed,
        best_epoch: r.best_epoch,
        stopped_epoch: r.stopped_epoch,
        day: cell("day"),
        night: cell("night"),
        ir: cell("ir"),
        all: cell("all"),
        in_scope: r
            .bench
            .as_ref()
            .and_then(|b| b.in_scope.as_ref())
            .and_then(|a| a.ap50_95),
        ms_p50: r.bench.as_ref().and_then(|b| b.ms_p50),
        threshold: r.bench.as_ref().and_then(|b| b.threshold),
        person_recall: r.bench.as_ref().and_then(|b| b.person_recall),
        vehicle_recall: r.bench.as_ref().and_then(|b| b.vehicle_recall),
        false_alerts: r.bench.as_ref().and_then(FalseAlerts::of),
    }
}

/// The value of column `c` of a row.
fn value(r: &Row, c: &str) -> Option<f64> {
    match c {
        "day" => r.day.ap50_95,
        "night" => r.night.ap50_95,
        "ir" => r.ir.ap50_95,
        "all" => r.all.ap50_95,
        "day_ap50" => r.day.ap50,
        "night_ap50" => r.night.ap50,
        "ir_ap50" => r.ir.ap50,
        "all_ap50" => r.all.ap50,
        "ms_p50" => r.ms_p50,
        "person_recall" => r.person_recall,
        "vehicle_recall" => r.vehicle_recall,
        "false_alerts" => r.false_alerts.and_then(|f| f.rate()),
        _ => None,
    }
    .filter(|v| v.is_finite())
}

/// Build the comparison of `runs` (in queue order).
pub fn compare(runs: &[CompareRun]) -> Comparison {
    let mut c = Comparison {
        comparable: true,
        ..Comparison::default()
    };
    // Same held-out frames?
    let prints: Vec<(&str, Option<&str>)> = runs
        .iter()
        .filter_map(|r| {
            r.bench
                .as_ref()
                .map(|b| (r.name.as_str(), b.heldout_fingerprint.as_deref()))
        })
        .collect();
    let known: Vec<&str> = prints.iter().filter_map(|(_, f)| *f).collect();
    if known.iter().any(|f| Some(f) != known.first()) {
        c.comparable = false;
        c.warnings.push(
            "the runs were benchmarked on different held-out frames (the exports changed between \
             them), so their AP is not comparable; queue the comparison again"
                .into(),
        );
    } else if !known.is_empty() && known.len() < prints.len() {
        let names: Vec<&str> = prints
            .iter()
            .filter(|(_, f)| f.is_none())
            .map(|(n, _)| *n)
            .collect();
        c.warnings.push(format!(
            "{} recorded no held-out fingerprint (an older version): the held-out frames may \
             differ",
            names.join(", ")
        ));
    }
    if let Some(first) = runs.first() {
        if runs.iter().any(|r| r.classes != first.classes) {
            c.warnings.push(
                "the runs train different classes: AP is averaged over different classes".into(),
            );
        }
        if runs.iter().any(|r| r.excluded != first.excluded) {
            c.warnings.push(
                "the runs treat objects of untrained classes differently (background / frame \
                 left out)"
                    .into(),
            );
        }
        let mut thresholds: Vec<f32> = Vec::new();
        for t in runs
            .iter()
            .filter_map(|r| r.bench.as_ref().and_then(|b| b.threshold))
        {
            if !thresholds.contains(&t) {
                thresholds.push(t);
            }
        }
        if thresholds.len() > 1 {
            let list: Vec<String> = thresholds.iter().map(|t| format!("{t:.2}")).collect();
            c.warnings.push(format!(
                "the runs were benchmarked at different confidence thresholds ({}): their                  recall and false alerts are not comparable",
                list.join(" / ")
            ));
        }
        if runs.iter().any(|r| r.balance != first.balance) {
            let kinds = runs.iter().map(|r| r.balance.describe());
            let mut uniq: Vec<String> = Vec::new();
            for k in kinds {
                if !uniq.contains(&k) {
                    uniq.push(k);
                }
            }
            c.warnings.push(format!(
                "the runs balance their training frames differently ({}): AP differences may \
                 come from the balancing",
                uniq.join(" / ")
            ));
        }
    }
    c.rows = runs.iter().map(row_of).collect();
    for (col, better) in COLUMNS {
        let lower = better == Better::Lower;
        let mut best: Option<(usize, f64)> = None;
        for (i, r) in c.rows.iter().enumerate() {
            let Some(v) = value(r, col) else { continue };
            let better = match best {
                None => true,
                Some((_, b)) if lower => v < b,
                Some((_, b)) => v > b,
            };
            if better {
                best = Some((i, v));
            }
        }
        if let Some((i, _)) = best {
            c.best_per_column.insert(col.to_string(), i);
        }
    }
    c.seed_summaries = seed_summaries(&c.rows);
    // Routed rows, per model: the day run's day AP, the night run's night / IR AP.
    let mut models: Vec<&str> = Vec::new();
    for r in runs {
        if !models.contains(&r.model.as_str()) {
            models.push(&r.model);
        }
    }
    for m in models {
        let of_model = || {
            runs.iter()
                .filter(move |r| r.model == m && r.bench.is_some())
        };
        let day = of_model().find(|r| r.scene == SceneFilter::DAY);
        let night = of_model()
            .find(|r| r.scene == SceneFilter::NIGHT_IR)
            .or_else(|| of_model().find(|r| r.scene.is_night()));
        let (Some(d), Some(n)) = (day, night) else {
            continue;
        };
        let (dr, nr) = (row_of(d), row_of(n));
        c.rows.push(Row {
            name: format!("{} + {}", d.name, n.name),
            model: m.to_string(),
            scene: format!("day + {} routed", n.scene),
            routed: true,
            seed: None,
            best_epoch: None,
            stopped_epoch: None,
            day: dr.day,
            night: nr.night,
            ir: nr.ir,
            all: Cell::default(),
            in_scope: None,
            ms_p50: None,
            threshold: None,
            person_recall: None,
            vehicle_recall: None,
            false_alerts: None,
        });
    }
    c
}

fn fmt_ap(v: Option<f64>) -> String {
    v.map_or_else(|| "-".into(), |v| format!("{v:.3}"))
}

impl Comparison {
    /// A plain-text table (the CLI): AP50-95 per scene (`*` = best of the column), in-scope AP,
    /// p50 time, then the warnings.
    pub fn render_text(&self) -> String {
        let best = |i: usize, col: &str| self.best_per_column.get(col) == Some(&i);
        let mut table: Vec<Vec<String>> = vec![
            [
                "run",
                "model",
                "scene",
                "epochs",
                "day",
                "night",
                "ir",
                "all",
                "in-scope",
                "p50 ms",
                "person R",
                "vehicle R",
                "false alerts",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        ];
        for (i, r) in self.rows.iter().enumerate() {
            let mark = |col: &str| if !r.routed && best(i, col) { "*" } else { "" };
            let ap = |col: &str, v: Option<f64>| format!("{}{}", fmt_ap(v), mark(col));
            let recall = |col: &str, v: Option<f64>| {
                v.map_or_else(|| "-".into(), |v| format!("{v:.2}{}", mark(col)))
            };
            let epochs = match (r.best_epoch, r.stopped_epoch) {
                (Some(b), Some(s)) => format!("best {b}/{s}"),
                (Some(b), None) => format!("best {b}"),
                (None, Some(s)) => format!("{s}"),
                (None, None) => "-".into(),
            };
            let ms = r
                .ms_p50
                .map_or_else(|| "-".into(), |v| format!("{v:.1}{}", mark("ms_p50")));
            let alerts = r.false_alerts.map_or_else(
                || "-".into(),
                |f| format!("{}{}", f.text(), mark("false_alerts")),
            );
            let name = match r.seed {
                Some(seed) => format!("{} (seed {seed})", r.name),
                None => r.name.clone(),
            };
            table.push(vec![
                name,
                r.model.clone(),
                r.scene.clone(),
                epochs,
                ap("day", r.day.ap50_95),
                ap("night", r.night.ap50_95),
                ap("ir", r.ir.ap50_95),
                ap("all", r.all.ap50_95),
                fmt_ap(r.in_scope),
                ms,
                recall("person_recall", r.person_recall),
                recall("vehicle_recall", r.vehicle_recall),
                alerts,
            ]);
        }
        for sm in &self.seed_summaries {
            let ap = |col: &str| {
                sm.columns.get(col).map_or_else(
                    || "-".into(),
                    |s| format!("{:.3} ({:.3}-{:.3})", s.mean, s.min, s.max),
                )
            };
            let recall = |col: &str| {
                sm.columns.get(col).map_or_else(
                    || "-".into(),
                    |s| format!("{:.2} ({:.2}-{:.2})", s.mean, s.min, s.max),
                )
            };
            let pct = |col: &str| {
                sm.columns.get(col).map_or_else(
                    || "-".into(),
                    |s| {
                        format!(
                            "{:.0}% ({:.0}-{:.0}%)",
                            s.mean * 100.0,
                            s.min * 100.0,
                            s.max * 100.0
                        )
                    },
                )
            };
            let ms = sm.columns.get("ms_p50").map_or_else(
                || "-".into(),
                |s| format!("{:.1} ({:.1}-{:.1})", s.mean, s.min, s.max),
            );
            table.push(vec![
                sm.label.clone(),
                sm.model.clone(),
                sm.scene.clone(),
                "-".into(),
                ap("day"),
                ap("night"),
                ap("ir"),
                ap("all"),
                "-".into(),
                ms,
                recall("person_recall"),
                recall("vehicle_recall"),
                pct("false_alerts"),
            ]);
        }
        let widths: Vec<usize> = (0..table.first().map_or(0, Vec::len))
            .map(|k| {
                table
                    .iter()
                    .map(|row| row.get(k).map_or(0, |c| c.chars().count()))
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        let mut out = String::from("Held-out AP50-95 per scene");
        let mut thresholds: Vec<f32> = self.rows.iter().filter_map(|r| r.threshold).collect();
        thresholds.dedup();
        match thresholds.as_slice() {
            [] => {}
            [t] => out.push_str(&format!(
                "; recall and false alerts (empty frames that alert) at {t:.2}"
            )),
            _ => out.push_str(
                "; recall and false alerts (empty frames that alert) at each run's threshold",
            ),
        }
        out.push_str(" (* = best)\n");
        for row in &table {
            let line: Vec<String> = row
                .iter()
                .zip(&widths)
                .map(|(c, w)| format!("{c:<w$}"))
                .collect();
            out.push_str(line.join("  ").trim_end());
            out.push('\n');
        }
        for w in &self.warnings {
            out.push_str(&format!("warning: {w}\n"));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ap(v: f64) -> SceneAp {
        SceneAp {
            ap50: Some(v + 0.2),
            ap50_95: Some(v),
            images: 10,
        }
    }

    fn run(name: &str, scene: SceneFilter, day: f64, night: f64, ir: f64, ms: f64) -> CompareRun {
        let by_scene = [
            ("day", day),
            ("night", night),
            ("ir", ir),
            ("all", (day + night + ir) / 3.0),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), ap(v)))
        .collect();
        CompareRun {
            name: name.into(),
            seed: None,
            model: "yolo26n".into(),
            scene,
            classes: vec!["person".into()],
            excluded: Excluded::Background,
            balance: Balance::default(),
            best_epoch: Some(12),
            stopped_epoch: Some(27),
            bench: Some(BenchSummary {
                ms_p50: Some(ms),
                by_scene,
                in_scope: Some(ap(day)),
                heldout_fingerprint: Some("abc".into()),
                ..BenchSummary::default()
            }),
        }
    }

    #[test]
    fn rows_best_columns_and_the_routed_row() {
        let runs = [
            run("day-a", SceneFilter::DAY, 0.6, 0.2, 0.1, 10.0),
            run("night-a", SceneFilter::NIGHT_IR, 0.3, 0.5, 0.45, 11.0),
            run("custom-a", SceneFilter::ALL_SCENES, 0.55, 0.4, 0.5, 9.0),
        ];
        let c = compare(&runs);
        assert!(c.comparable);
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);
        assert_eq!(c.rows.len(), 4);
        assert_eq!(c.best_per_column["day"], 0);
        assert_eq!(c.best_per_column["night"], 1);
        assert_eq!(c.best_per_column["ir"], 2);
        assert_eq!(c.best_per_column["ms_p50"], 2);
        assert_eq!(c.best_per_column["night_ap50"], 1);
        let routed = &c.rows[3];
        assert!(routed.routed);
        assert_eq!(routed.name, "day-a + night-a");
        assert_eq!(routed.scene, "day + night+ir routed");
        assert_eq!(routed.day.ap50_95, Some(0.6));
        assert_eq!(routed.night.ap50_95, Some(0.5));
        assert_eq!(routed.ir.ap50_95, Some(0.45));
        assert_eq!(routed.all.ap50_95, None, "no invented combined AP");
        assert_eq!(c.rows[0].best_epoch, Some(12));
        let t = c.render_text();
        assert!(t.contains("0.600*"), "{t}");
        assert!(t.contains("best 12/27"), "{t}");
        assert!(t.contains("day-a + night-a"), "{t}");
    }

    /// `run` with alerting figures at threshold 0.5.
    fn alerting(mut r: CompareRun, person: f64, vehicle: f64, alerts: usize) -> CompareRun {
        let b = r.bench.as_mut().unwrap();
        b.threshold = Some(0.5);
        b.person_recall = Some(person);
        b.vehicle_recall = Some(vehicle);
        b.empty_frames = Some(130);
        b.empty_frame_alerts = Some(alerts);
        r
    }

    #[test]
    fn seeds_of_one_model_and_scene_get_a_mean_and_range_row() {
        let seeded = |name: &str, seed: u64, day: f64, ms: f64| {
            let mut r = run(name, SceneFilter::DAY, day, 0.2, 0.1, ms);
            r.seed = Some(seed);
            r
        };
        let runs = [
            seeded("a", 0, 0.40, 10.0),
            seeded("b", 1, 0.50, 12.0),
            seeded("c", 2, 0.60, 14.0),
            // Alone in its scene: no summary.
            {
                let mut r = run("d", SceneFilter::NIGHT_IR, 0.1, 0.5, 0.4, 10.0);
                r.seed = Some(0);
                r
            },
        ];
        let c = compare(&runs);
        // Four runs and the routed day + night row; summaries are not rows.
        assert_eq!(c.rows.len(), 5);
        assert_eq!(c.rows[1].seed, Some(1));
        assert_eq!(c.seed_summaries.len(), 1);
        let s = &c.seed_summaries[0];
        assert_eq!(s.label, "mean of 3 seeds");
        assert_eq!(s.runs, 3);
        let day = s.columns["day"];
        assert!((day.mean - 0.5).abs() < 1e-9 && day.min == 0.4 && day.max == 0.6);
        let ms = s.columns["ms_p50"];
        assert!((ms.mean - 12.0).abs() < 1e-9 && ms.min == 10.0 && ms.max == 14.0);
        // The best of a column is a run, never the summary.
        assert_eq!(c.best_per_column["day"], 2);
        let text = c.render_text();
        assert!(text.contains("a (seed 0)"), "{text}");
        assert!(text.contains("mean of 3 seeds"), "{text}");
        assert!(text.contains("0.500 (0.400-0.600)"), "{text}");
        // Runs without a seed make no summary.
        let plain = compare(&[
            run("x", SceneFilter::DAY, 0.4, 0.2, 0.1, 10.0),
            run("y", SceneFilter::DAY, 0.5, 0.2, 0.1, 10.0),
        ]);
        assert!(plain.seed_summaries.is_empty());
    }

    #[test]
    fn recall_and_false_alert_columns() {
        let runs = [
            alerting(
                run("day-a", SceneFilter::DAY, 0.6, 0.2, 0.1, 10.0),
                0.39,
                0.8,
                9,
            ),
            alerting(
                run("night-a", SceneFilter::NIGHT_IR, 0.3, 0.5, 0.45, 11.0),
                0.7,
                0.6,
                2,
            ),
            // An older run: no figures, never the best, dashes.
            run("old-a", SceneFilter::ALL_SCENES, 0.1, 0.1, 0.1, 12.0),
        ];
        let c = compare(&runs);
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);
        assert_eq!(c.best_per_column["person_recall"], 1);
        assert_eq!(c.best_per_column["vehicle_recall"], 0);
        assert_eq!(c.best_per_column["false_alerts"], 1, "lowest rate");
        let fa = c.rows[0].false_alerts.unwrap();
        assert_eq!((fa.text(), fa.rate()), ("9/130".into(), Some(9.0 / 130.0)));
        assert_eq!(c.rows[2].person_recall, None);
        assert_eq!(c.rows[2].false_alerts, None);
        // The routed row has none of them (they cover every held-out frame).
        let routed = c.rows.last().unwrap();
        assert!(routed.routed);
        assert_eq!(
            (routed.person_recall, routed.false_alerts, routed.threshold),
            (None, None, None)
        );
        let t = c.render_text();
        assert!(t.contains("at 0.50"), "{t}");
        assert!(t.contains("person R"), "{t}");
        assert!(t.contains("0.70*"), "{t}");
        assert!(t.contains("0.80*"), "{t}");
        assert!(t.contains("2/130*"), "{t}");
        assert!(t.contains(" 9/130\n"), "{t}");
        // A rate is compared, not the count: 3 of 300 beats 2 of 130.
        let mut more = runs.clone();
        let b = more[0].bench.as_mut().unwrap();
        b.empty_frames = Some(300);
        b.empty_frame_alerts = Some(3);
        assert_eq!(compare(&more).best_per_column["false_alerts"], 0);
        // No empty frames: no rate, no best.
        for r in more.iter_mut().take(2) {
            r.bench.as_mut().unwrap().empty_frames = Some(0);
            r.bench.as_mut().unwrap().empty_frame_alerts = Some(0);
        }
        assert!(!compare(&more).best_per_column.contains_key("false_alerts"));
        // Different thresholds warn.
        let mut other = runs.clone();
        other[1].bench.as_mut().unwrap().threshold = Some(0.4);
        let c = compare(&other);
        assert!(
            c.warnings
                .iter()
                .any(|w| w.contains("different confidence thresholds (0.50 / 0.40)")),
            "{:?}",
            c.warnings
        );
        assert!(c.render_text().contains("at each run's threshold"));
        // Serialized for `GET /v1/training/compare`.
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["rows"][0]["false_alerts"]["empty_frames"], 130);
        assert_eq!(v["rows"][0]["person_recall"], 0.39);
    }

    #[test]
    fn no_routed_row_without_a_day_and_a_night_run() {
        let c = compare(&[
            run("day-a", SceneFilter::DAY, 0.6, 0.2, 0.1, 10.0),
            run("custom-a", SceneFilter::ALL_SCENES, 0.5, 0.4, 0.5, 9.0),
        ]);
        assert!(c.rows.iter().all(|r| !r.routed));
        // A night-only run counts as the night side.
        let c = compare(&[
            run("day-a", SceneFilter::DAY, 0.6, 0.2, 0.1, 10.0),
            run(
                "n-a",
                SceneFilter::new(false, true, false),
                0.1,
                0.5,
                0.2,
                10.0,
            ),
        ]);
        assert!(c.rows.last().unwrap().routed);
    }

    #[test]
    fn mismatches_warn() {
        let mut b = run("night-a", SceneFilter::NIGHT_IR, 0.3, 0.5, 0.4, 11.0);
        b.bench.as_mut().unwrap().heldout_fingerprint = Some("other".into());
        b.classes.push("car".into());
        b.excluded = Excluded::DropFrame;
        b.balance
            .axes
            .insert(crate::training::dataset::BalanceAxis::Class);
        let c = compare(&[run("day-a", SceneFilter::DAY, 0.6, 0.2, 0.1, 10.0), b]);
        assert!(!c.comparable);
        assert_eq!(c.warnings.len(), 4, "{:?}", c.warnings);
        assert!(
            c.warnings
                .iter()
                .any(|w| w.contains("balance") && w.contains("off / class, strength 0.5")),
            "{:?}",
            c.warnings
        );
        assert!(
            c.render_text()
                .contains("warning: the runs were benchmarked")
        );
        // A run without a benchmark is a row of dashes and never the best.
        let mut x = run("x", SceneFilter::DAY, 0.9, 0.9, 0.9, 1.0);
        x.bench = None;
        let c = compare(&[x, run("y", SceneFilter::DAY, 0.1, 0.1, 0.1, 5.0)]);
        assert!(c.comparable);
        assert_eq!(c.best_per_column["day"], 1);
        assert_eq!(c.rows[0].day.ap50_95, None);
    }
}
