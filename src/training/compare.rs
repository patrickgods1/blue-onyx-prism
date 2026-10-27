//! Day / night / mixed comparison of runs queued together (`TrainRequest.compare_group`): one
//! row per run with its held-out AP per scene ([`BenchSummary::by_scene`]) and speed, the best
//! value of each column, and a "routed" row per model when the group has a day run and a night
//! (or IR) run: the day run's day AP beside the night run's night / IR AP, as a service with
//! `night_model` routing would score. Scenes are scored separately, so no combined AP is made
//! up. Pure; the Training page and `train run --scene ... --scene ...` render it.

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
}

/// Column ids of [`Comparison::best_per_column`].
pub const COLUMNS: [&str; 9] = [
    "day",
    "night",
    "ir",
    "all",
    "day_ap50",
    "night_ap50",
    "ir_ap50",
    "all_ap50",
    "ms_p50",
];

/// The comparison of a group.
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct Comparison {
    pub rows: Vec<Row>,
    /// Column id ([`COLUMNS`]: AP50-95 per scene, `<scene>_ap50`, `ms_p50`) -> index of the
    /// best run row (highest AP, lowest time; ties: the first). Routed rows never count.
    pub best_per_column: BTreeMap<String, usize>,
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
    for col in COLUMNS {
        let lower = col == "ms_p50";
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
            best_epoch: None,
            stopped_epoch: None,
            day: dr.day,
            night: nr.night,
            ir: nr.ir,
            all: Cell::default(),
            in_scope: None,
            ms_p50: None,
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
                "run", "model", "scene", "epochs", "day", "night", "ir", "all", "in-scope",
                "p50 ms",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        ];
        for (i, r) in self.rows.iter().enumerate() {
            let ap = |col: &str, v: Option<f64>| {
                let mark = if !r.routed && best(i, col) { "*" } else { "" };
                format!("{}{mark}", fmt_ap(v))
            };
            let epochs = match (r.best_epoch, r.stopped_epoch) {
                (Some(b), Some(s)) => format!("best {b}/{s}"),
                (Some(b), None) => format!("best {b}"),
                (None, Some(s)) => format!("{s}"),
                (None, None) => "-".into(),
            };
            let ms = r.ms_p50.map_or_else(
                || "-".into(),
                |v| {
                    let mark = if !r.routed && best(i, "ms_p50") {
                        "*"
                    } else {
                        ""
                    };
                    format!("{v:.1}{mark}")
                },
            );
            table.push(vec![
                r.name.clone(),
                r.model.clone(),
                r.scene.clone(),
                epochs,
                ap("day", r.day.ap50_95),
                ap("night", r.night.ap50_95),
                ap("ir", r.ir.ap50_95),
                ap("all", r.all.ap50_95),
                fmt_ap(r.in_scope),
                ms,
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
        let mut out = String::from("Held-out AP50-95 per scene (* = best)\n");
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
