//! Day / night / mixed comparison of runs queued together (`TrainRequest.compare_group`): one
//! row per run with its held-out AP per scene ([`BenchSummary::by_scene`]) and speed, the best
//! value of each column, and a "routed" row per model when the group has a day run and a night
//! (or IR) run: the day run's day AP beside the night run's night / IR AP, as a service with
//! `night_model` routing would score. Scenes are scored separately, so no combined AP is made
//! up. Beside AP, the alerting figures of the post-training benchmark at its configured threshold
//! (every held-out frame): person recall, vehicle recall (car/truck/bus merged) and false alerts
//! on frames without any object, and false person alerts on frames without a person (a parked car
//! does not hide them; best = highest recall, lowest false-alert rate). A warning names runs whose
//! AP averages different classes (a class counts with at least
//! [`crate::benchmark::metrics::MIN_CLASS_GT`] held-out objects). Pure; the
//! Train tab and `train run --scene ... --scene ...` render it. Runs that differ only by
//! training seed (`--seeds`, "Repeat with N seeds") also get a "mean of N seeds" summary
//! ([`SeedSummary`]: mean and range per column; never the best of a column). A "random picks"
//! column holds AP50-95 on the held-out frames the Review queue picked at random
//! ([`BenchSummary::random`], the least biased score), shown only when every benchmarked run has
//! at least [`MIN_RANDOM_HELDOUT`] of them ([`Comparison::random_column`]).

use super::dataset::{Balance, Excluded, IgnoreMode, SceneFilter};
use super::job::{BenchSummary, SceneAp, SubsetScore};
use super::recipe::SelectMetric;
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
    /// Its class merges (`TrainRequest.rename`; an empty target drops the label).
    pub rename: BTreeMap<String, String>,
    pub excluded: Excluded,
    /// What its training frames with ignore regions became (`RunSettings::ignore_mode`); None
    /// while the run has not started training.
    pub ignore: Option<IgnoreMode>,
    /// The share of empty training frames it kept (`RunSettings::negative_share`); None =
    /// unknown (not started, or a record from before it was recorded).
    pub negative_share: Option<f32>,
    /// The most frames per static spot (`RunSettings::static_cap`, `Some(0)` = off); None =
    /// unknown.
    pub static_cap: Option<u32>,
    /// How the run balanced its training frames (off for older runs).
    pub balance: Balance,
    /// Best epoch (1-based) and the epoch training stopped at, when known.
    pub best_epoch: Option<u32>,
    pub stopped_epoch: Option<u32>,
    /// The run's training seed (`TrainRequest.train_seed`; None = not a repeat).
    pub seed: Option<u64>,
    /// The metric that picked its best epoch (`RunRecord::select_metric`).
    pub select_metric: SelectMetric,
    /// The classes its best epoch was picked over (the trainer's `support` event: classes with at
    /// least `MIN_CLASS_GT` validation boxes); None = every class (no such class, or a run from
    /// before the rule).
    pub select_classes: Option<Vec<String>>,
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
    /// The metric that picked `best_epoch` (None for a routed row).
    pub select_metric: Option<SelectMetric>,
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
    /// False person alerts on held-out frames without a person (other objects allowed: a
    /// parked car does not hide them); None for a routed row and runs of older versions.
    pub person_false_alerts: Option<PersonAlerts>,
    /// The held-out frames the Review queue picked at random ([`BenchSummary::random`]; None for a
    /// routed row, a run without any and runs of older versions). Its AP50-95 is the `random`
    /// column, shown only with [`Comparison::random_column`].
    pub random: Option<SubsetScore>,
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

/// Frames without a person that got a person detection at the threshold, of all such frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PersonAlerts {
    pub alerts: usize,
    pub frames: usize,
}

impl PersonAlerts {
    /// From a benchmark summary; None when it has no person counts.
    pub fn of(b: &BenchSummary) -> Option<Self> {
        Some(Self {
            alerts: b.person_false_alerts?,
            frames: b.person_free_frames?,
        })
    }

    /// Share of the frames without a person that alerted (None without such frames).
    pub fn rate(&self) -> Option<f64> {
        (self.frames > 0).then(|| self.alerts as f64 / self.frames as f64)
    }

    /// `2/210`.
    pub fn text(&self) -> String {
        format!("{}/{}", self.alerts, self.frames)
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
pub const COLUMNS: [(&str, Better); 14] = [
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
    // False person alerts / frames without a person.
    ("person_false_alerts", Better::Lower),
    // AP50-95 on the held-out random picks (only with `Comparison::random_column`).
    ("random", Better::Higher),
];

/// Fewest held-out random picks ([`BenchSummary::random`]) every benchmarked run needs before the
/// Comparison shows its `random` column: AP over fewer frames moves too much from frame to frame.
pub const MIN_RANDOM_HELDOUT: usize = 50;

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

fn seed_summaries(rows: &[Row], random_column: bool) -> Vec<SeedSummary> {
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
        for (col, _) in COLUMNS
            .into_iter()
            .filter(|(c, _)| random_column || *c != "random")
        {
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
    /// Whether the `random` column is shown: every benchmarked run has at least
    /// [`MIN_RANDOM_HELDOUT`] held-out random picks. Without it the column is neither shown nor
    /// in `best_per_column` or the seed summaries (the rows still carry `random`).
    pub random_column: bool,
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
        select_metric: Some(r.select_metric),
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
        person_false_alerts: r.bench.as_ref().and_then(PersonAlerts::of),
        random: r.bench.as_ref().and_then(|b| b.random.clone()),
    }
}

impl Row {
    /// " by F2" beside the best epoch of a run not selected by mAP (empty otherwise).
    pub fn selected_by(&self) -> String {
        self.select_metric
            .filter(|m| !m.is_default())
            .map_or_else(String::new, |m| format!(" by {}", m.short()))
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
        "person_false_alerts" => r.person_false_alerts.and_then(|f| f.rate()),
        "random" => r.random.as_ref().and_then(|s| s.ap50_95),
        _ => None,
    }
    .filter(|v| v.is_finite())
}

/// The figures of the Models tab's "Best first" order, in order: person recall, vehicle recall,
/// AP50-95 on every held-out frame (higher is better each). What alerting sees comes first: a
/// missed person is the costliest error of a camera model.
pub type BestKeys = [Option<f64>; 3];

/// [`BestKeys`] of a run's held-out benchmark (all None without one).
pub fn best_keys(b: Option<&BenchSummary>) -> BestKeys {
    let Some(b) = b else {
        return [None; 3];
    };
    [
        b.person_recall,
        b.vehicle_recall,
        b.by_scene.get("all").and_then(|a| a.ap50_95),
    ]
}

/// Indices of `keys` best first: by each figure of [`BestKeys`] in turn, highest first, a
/// missing (or not finite) figure after every present one; rows equal on all three keep their
/// given order. Plain values, no noise rule: these are one run's figures on one held-out set,
/// with no repeat to measure a noise from.
pub fn best_order_by(keys: &[BestKeys]) -> Vec<usize> {
    use std::cmp::Ordering;
    // Highest first, a missing figure last.
    let one = |x: Option<f64>, y: Option<f64>| match (
        x.filter(|v| v.is_finite()),
        y.filter(|v| v.is_finite()),
    ) {
        (Some(x), Some(y)) => y.total_cmp(&x),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    };
    let mut idx: Vec<(usize, &BestKeys)> = keys.iter().enumerate().collect();
    idx.sort_by(|(_, a), (_, b)| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| one(*x, *y))
            .find(|o| o.is_ne())
            .unwrap_or(Ordering::Equal)
    });
    idx.into_iter().map(|(i, _)| i).collect()
}

/// The run rows of a comparison best first (indices into `rows`; [`best_order_by`] over person
/// R, vehicle R, all AP50-95): the Comparison card's `data-best`. Routed rows are left out
/// (they copy other rows' cells and stay pinned under the runs).
pub fn best_order(rows: &[Row]) -> Vec<usize> {
    let runs: Vec<(usize, BestKeys)> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| !r.routed)
        .map(|(i, r)| (i, [r.person_recall, r.vehicle_recall, r.all.ap50_95]))
        .collect();
    let keys: Vec<BestKeys> = runs.iter().map(|(_, k)| *k).collect();
    best_order_by(&keys)
        .into_iter()
        .filter_map(|k| runs.get(k).map(|(i, _)| *i))
        .collect()
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
        if runs.iter().any(|r| r.rename != first.rename) {
            let mut seen: Vec<(&String, &String)> = Vec::new();
            for r in runs {
                for e in &r.rename {
                    if !seen.contains(&e) {
                        seen.push(e);
                    }
                }
            }
            let parts: Vec<String> = seen
                .iter()
                .filter_map(|(from, to)| {
                    let n = runs
                        .iter()
                        .filter(|r| r.rename.get(*from) == Some(*to))
                        .count();
                    (n < runs.len()).then(|| {
                        let to = if to.is_empty() {
                            "dropped"
                        } else {
                            to.as_str()
                        };
                        format!("{from} \u{2192} {to} on {n} of {}", runs.len())
                    })
                })
                .collect();
            c.warnings.push(format!(
                "the runs merge classes differently ({})",
                parts.join(", ")
            ));
        }
        if runs.iter().any(|r| r.excluded != first.excluded) {
            c.warnings.push(
                "the runs treat objects of untrained classes differently (background / frame \
                 left out)"
                    .into(),
            );
        }
        let mut modes: Vec<IgnoreMode> = Vec::new();
        for m in runs.iter().filter_map(|r| r.ignore) {
            if !modes.contains(&m) {
                modes.push(m);
            }
        }
        if modes.len() > 1 {
            let names: Vec<&str> = modes.iter().map(|m| m.as_str()).collect();
            c.warnings.push(format!(
                "the runs treat frames with an unsure or too-small object differently ({}): \
                 their training and validation frames differ, so AP differences may come from \
                 that",
                names.join(" / ")
            ));
        }
        let mut shares: Vec<String> = Vec::new();
        for n in runs.iter().filter_map(|r| r.negative_share) {
            let pct = format!("{:.0}%", n * 100.0);
            if !shares.contains(&pct) {
                shares.push(pct);
            }
        }
        if shares.len() > 1 {
            c.warnings.push(format!(
                "the runs keep different shares of empty training frames ({}): their training frames differ, so differences in false alerts may come from that",
                shares.join(" / ")
            ));
        }
        let mut caps: Vec<u32> = Vec::new();
        for n in runs.iter().filter_map(|r| r.static_cap) {
            if !caps.contains(&n) {
                caps.push(n);
            }
        }
        if caps.len() > 1 {
            let list: Vec<String> = caps
                .iter()
                .map(|n| {
                    if *n == 0 {
                        "off".to_string()
                    } else {
                        n.to_string()
                    }
                })
                .collect();
            c.warnings.push(format!(
                "the runs cap static objects differently ({}): their training frames differ, so AP differences may come from that",
                list.join(" / ")
            ));
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
                "the runs were benchmarked at different confidence thresholds ({}): their \
                 recall and false alerts are not comparable",
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
        // The classes each run's AP averages (only runs that recorded them).
        let mut bases: Vec<String> = Vec::new();
        for b in runs
            .iter()
            .filter_map(|r| r.bench.as_ref().and_then(|b| b.ap_basis.as_ref()))
        {
            let mut names = b.classes.clone();
            names.sort();
            let words = names.join(", ");
            if !bases.contains(&words) {
                bases.push(words);
            }
        }
        if bases.len() > 1 {
            c.warnings.push(format!(
                "the runs' AP averages different classes ({}): a class counts in the average only \
                 with at least {} held-out objects, so their AP columns are not comparable",
                bases.join(" / "),
                crate::benchmark::metrics::MIN_CLASS_GT
            ));
        }
        let mut picked: Vec<String> = Vec::new();
        for r in runs {
            let v = r
                .select_classes
                .as_ref()
                .map_or_else(|| "every class".to_string(), |c| c.join(", "));
            if !picked.contains(&v) {
                picked.push(v);
            }
        }
        if picked.len() > 1 {
            c.warnings.push(format!(
                "the runs picked their best epoch over different classes ({}): a class counts \
                 only with at least {} validation boxes, so their best epochs were chosen on \
                 different evidence",
                picked.join(" / "),
                crate::benchmark::metrics::MIN_CLASS_GT
            ));
        }
        if runs.iter().any(|r| r.select_metric != first.select_metric) {
            let mut names: Vec<&str> = Vec::new();
            for m in runs.iter().map(|r| r.select_metric.short()) {
                if !names.contains(&m) {
                    names.push(m);
                }
            }
            c.warnings.push(format!(
                "the runs picked their best epoch by different metrics ({}): AP differences may \
                 come from that choice",
                names.join(" / ")
            ));
        }
    }
    c.rows = runs.iter().map(row_of).collect();
    // The random picks: shown when every benchmarked run has enough; counts that differ mean
    // the exports marked other frames as picks (the frames themselves are the fingerprint's).
    let picks: Vec<Option<SubsetScore>> = c
        .rows
        .iter()
        .zip(runs)
        .filter(|(_, r)| r.bench.is_some())
        .map(|(row, _)| row.random.clone())
        .collect();
    c.random_column = !picks.is_empty()
        && picks
            .iter()
            .all(|p| p.as_ref().is_some_and(|p| p.images >= MIN_RANDOM_HELDOUT));
    let mut counts: Vec<usize> = Vec::new();
    for n in picks.iter().flatten().map(|p| p.images) {
        if !counts.contains(&n) {
            counts.push(n);
        }
    }
    if counts.len() > 1 {
        let list: Vec<String> = counts.iter().map(usize::to_string).collect();
        c.warnings.push(format!(
            "the runs have different numbers of random picks among their held-out frames ({}): \
             the label files changed between their exports, so their random-pick scores are not \
             comparable; queue the comparison again",
            list.join(" / ")
        ));
    }
    for (col, better) in COLUMNS {
        if col == "random" && !c.random_column {
            continue;
        }
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
    c.seed_summaries = seed_summaries(&c.rows, c.random_column);
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
            select_metric: None,
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
            person_false_alerts: None,
            random: None,
        });
    }
    c
}

fn fmt_ap(v: Option<f64>) -> String {
    v.map_or_else(|| "-".into(), |v| format!("{v:.3}"))
}

impl Comparison {
    /// A plain-text table (the CLI): AP50-95 per scene (`*` = best of the column), in-scope AP,
    /// AP50-95 on the random picks (with [`Comparison::random_column`]), p50 time, then the
    /// warnings.
    pub fn render_text(&self) -> String {
        let best = |i: usize, col: &str| self.best_per_column.get(col) == Some(&i);
        // The random picks' column goes after `in-scope` when shown.
        const RANDOM_AT: usize = 9;
        let with_random = |mut cells: Vec<String>, random: String| {
            if self.random_column {
                cells.insert(RANDOM_AT, random);
            }
            cells
        };
        let header = [
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
            "person FA",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mut table: Vec<Vec<String>> = vec![with_random(header, "random".into())];
        for (i, r) in self.rows.iter().enumerate() {
            let mark = |col: &str| if !r.routed && best(i, col) { "*" } else { "" };
            let ap = |col: &str, v: Option<f64>| format!("{}{}", fmt_ap(v), mark(col));
            let recall = |col: &str, v: Option<f64>| {
                v.map_or_else(|| "-".into(), |v| format!("{v:.2}{}", mark(col)))
            };
            let by = r.selected_by();
            let epochs = match (r.best_epoch, r.stopped_epoch) {
                (Some(b), Some(s)) => format!("best {b}{by}/{s}"),
                (Some(b), None) => format!("best {b}{by}"),
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
            let person_alerts = r.person_false_alerts.map_or_else(
                || "-".into(),
                |f| format!("{}{}", f.text(), mark("person_false_alerts")),
            );
            let name = match r.seed {
                Some(seed) => format!("{} (seed {seed})", r.name),
                None => r.name.clone(),
            };
            let random = ap("random", r.random.as_ref().and_then(|s| s.ap50_95));
            table.push(with_random(
                vec![
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
                    person_alerts,
                ],
                random,
            ));
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
            table.push(with_random(
                vec![
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
                    pct("person_false_alerts"),
                ],
                ap("random"),
            ));
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
                "; recall and false alerts (empty frames that alert; person FA: frames without a \
                 person that alert for one) at {t:.2}"
            )),
            _ => out.push_str(
                "; recall and false alerts (empty frames that alert; person FA: frames without a \
                 person that alert for one) at each run's threshold",
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
            ap_basis: None,
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
            rename: BTreeMap::new(),
            excluded: Excluded::Background,
            ignore: Some(IgnoreMode::Background),
            negative_share: Some(0.15),
            static_cap: Some(20),
            balance: Balance::default(),
            best_epoch: Some(12),
            stopped_epoch: Some(27),
            select_metric: SelectMetric::Map,
            select_classes: None,
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
    fn best_first_is_person_then_vehicle_recall_then_ap_with_missing_last() {
        let k = |p: Option<f64>, v: Option<f64>, a: Option<f64>| -> BestKeys { [p, v, a] };
        let keys = [
            k(Some(0.8), Some(0.5), Some(0.30)),
            k(Some(0.9), Some(0.1), Some(0.10)),
            k(Some(0.8), Some(0.7), Some(0.20)),
            k(None, Some(0.99), Some(0.99)),
            k(Some(0.8), Some(0.7), Some(0.25)),
            k(Some(f64::NAN), None, None),
            k(Some(0.8), None, Some(0.9)),
            k(Some(0.8), Some(0.5), Some(0.30)),
        ];
        // Person R first; equal person R: vehicle R; equal both: AP50-95; a missing figure after
        // the present ones (NaN counts as missing); an exact tie keeps the given order.
        assert_eq!(best_order_by(&keys), [1, 4, 2, 0, 7, 6, 3, 5]);
        assert!(best_order_by(&[]).is_empty());
        // From a benchmark summary: person R, vehicle R, AP50-95 on every held-out frame.
        let b = alerting(run("x", SceneFilter::DAY, 0.6, 0.3, 0.3, 10.0), 0.7, 0.4, 3).bench;
        let got = best_keys(b.as_ref());
        assert_eq!((got[0], got[1]), (Some(0.7), Some(0.4)));
        assert!((got[2].unwrap() - 0.4).abs() < 1e-9);
        assert_eq!(best_keys(None), [None; 3]);
    }

    #[test]
    fn the_comparison_best_order_leaves_routed_rows_out() {
        let runs = [
            alerting(
                run("day-a", SceneFilter::DAY, 0.6, 0.2, 0.1, 10.0),
                0.5,
                0.5,
                1,
            ),
            alerting(
                run("night-a", SceneFilter::NIGHT_IR, 0.3, 0.5, 0.45, 11.0),
                0.9,
                0.2,
                1,
            ),
            run("custom-a", SceneFilter::ALL_SCENES, 0.55, 0.4, 0.5, 9.0),
        ];
        let c = compare(&runs);
        assert!(c.rows[3].routed);
        assert_eq!(best_order(&c.rows), [1, 0, 2]);
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
        assert!(
            t.lines()
                .any(|l| l.contains(" 9/130 ") && !l.contains("9/130*")),
            "{t}"
        );
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
            c.warnings.iter().any(|w| w
                == "the runs were benchmarked at different confidence thresholds \
                               (0.50 / 0.40): their recall and false alerts are not comparable"),
            "{:?}",
            c.warnings
        );
        assert!(c.render_text().contains("at each run's threshold"));
        // Serialized for `GET /v1/training/compare`.
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["rows"][0]["false_alerts"]["empty_frames"], 130);
        assert_eq!(v["rows"][0]["person_recall"], 0.39);
    }

    /// False person alerts on frames without a person: a lower-is-better rate column.
    #[test]
    fn person_false_alert_column() {
        let with = |name: &str, alerts: usize, frames: usize| {
            let mut r = alerting(
                run(name, SceneFilter::DAY, 0.5, 0.2, 0.1, 10.0),
                0.8,
                0.8,
                1,
            );
            let b = r.bench.as_mut().unwrap();
            b.person_false_alerts = Some(alerts);
            b.person_free_frames = Some(frames);
            r
        };
        // 3 of 300 (1%) beats 2 of 100 (2%): the rate, not the count.
        let c = compare(&[
            with("a", 2, 100),
            with("b", 3, 300),
            run("old", SceneFilter::DAY, 0.1, 0.1, 0.1, 9.0),
        ]);
        assert_eq!(c.best_per_column["person_false_alerts"], 1);
        let pa = c.rows[1].person_false_alerts.unwrap();
        assert_eq!((pa.text(), pa.rate()), ("3/300".into(), Some(0.01)));
        assert_eq!(c.rows[2].person_false_alerts, None, "an older run");
        let t = c.render_text();
        assert!(t.contains("person FA"), "{t}");
        assert!(t.contains("3/300*"), "{t}");
        assert!(t.contains("person FA: frames without a person"), "{t}");
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["rows"][0]["person_false_alerts"]["frames"], 100);
        // Seed repeats get its mean and range.
        let mut s0 = with("s0", 2, 100);
        s0.seed = Some(0);
        let mut s1 = with("s1", 4, 100);
        s1.seed = Some(1);
        let c = compare(&[s0, s1]);
        let sp = c.seed_summaries[0].columns["person_false_alerts"];
        assert!((sp.mean - 0.03).abs() < 1e-12 && sp.min == 0.02 && sp.max == 0.04);
    }

    /// Runs whose AP averages different classes are warned about; unrecorded classes say nothing.
    #[test]
    fn runs_averaging_different_classes_warn() {
        use crate::benchmark::metrics::ApBasis;
        let based = |name: &str, classes: &[&str]| {
            let mut r = run(name, SceneFilter::DAY, 0.5, 0.2, 0.1, 10.0);
            r.bench.as_mut().unwrap().ap_basis = Some(ApBasis {
                min_gt: 20,
                classes: classes.iter().map(|c| c.to_string()).collect(),
                fell_back: false,
            });
            r
        };
        let c = compare(&[
            based("a", &["person", "car"]),
            based("b", &["car", "person", "truck"]),
        ]);
        assert_eq!(
            c.warnings,
            [
                "the runs' AP averages different classes (car, person / car, person, truck): a \
                 class counts in the average only with at least 20 held-out objects, so their AP \
                 columns are not comparable"
            ]
        );
        // The same classes in another order, or a run of an older version: no warning.
        assert!(
            compare(&[
                based("a", &["person", "car"]),
                based("b", &["car", "person"])
            ])
            .warnings
            .is_empty()
        );
        let old = run("old", SceneFilter::DAY, 0.5, 0.2, 0.1, 10.0);
        assert!(
            compare(&[based("a", &["person", "car"]), old])
                .warnings
                .is_empty()
        );
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
        // The ignore mode differs: said once; a run not started yet (no mode) says nothing.
        let mut d = run("night-b", SceneFilter::NIGHT_IR, 0.3, 0.5, 0.4, 11.0);
        d.ignore = Some(IgnoreMode::DropFrame);
        let mut queued = run("night-c", SceneFilter::NIGHT_IR, 0.3, 0.5, 0.4, 11.0);
        queued.ignore = None;
        let day = run("day-a", SceneFilter::DAY, 0.6, 0.2, 0.1, 10.0);
        let c = compare(&[day.clone(), d, queued.clone()]);
        assert_eq!(
            c.warnings,
            [
                "the runs treat frames with an unsure or too-small object differently \
              (background / drop-frame): their training and validation frames differ, so AP \
              differences may come from that"
            ]
        );
        assert!(compare(&[day.clone(), queued.clone()]).warnings.is_empty());
        // Empty-frame share and static cap: said once each, only when known values differ.
        let mut s25 = run("day-b", SceneFilter::DAY, 0.6, 0.2, 0.1, 10.0);
        s25.negative_share = Some(0.25);
        s25.static_cap = Some(0);
        let c = compare(&[day.clone(), s25.clone()]);
        assert_eq!(
            c.warnings,
            [
                "the runs keep different shares of empty training frames (15% / 25%): their training frames differ, so differences in false alerts may come from that",
                "the runs cap static objects differently (20 / off): their training frames differ, so AP differences may come from that"
            ]
        );
        let mut unknown = day.clone();
        unknown.negative_share = None;
        unknown.static_cap = None;
        assert!(compare(&[unknown.clone(), day.clone()]).warnings.is_empty());
        assert!(compare(&[unknown, s25]).warnings.is_empty());
        let mut same = day.clone();
        same.name = "day-c".into();
        assert!(compare(&[day, same]).warnings.is_empty());
        // A run without a benchmark is a row of dashes and never the best.
        let mut x = run("x", SceneFilter::DAY, 0.9, 0.9, 0.9, 1.0);
        x.bench = None;
        let c = compare(&[x, run("y", SceneFilter::DAY, 0.1, 0.1, 0.1, 5.0)]);
        assert!(c.comparable);
        assert_eq!(c.best_per_column["day"], 1);
        assert_eq!(c.rows[0].day.ap50_95, None);
    }

    /// `run` with `n` held-out random picks scored at AP50-95 `ap`.
    fn picked(mut r: CompareRun, n: usize, ap: f64) -> CompareRun {
        r.bench.as_mut().unwrap().random = Some(SubsetScore {
            images: n,
            gt: 2 * n,
            ap50: Some(ap + 0.2),
            ap50_95: Some(ap),
            recall: Some(0.8),
            ap_basis: None,
        });
        r
    }

    #[test]
    fn the_random_picks_column_needs_enough_frames_in_every_run() {
        let a = || run("day-a", SceneFilter::DAY, 0.6, 0.2, 0.1, 10.0);
        let b = || run("all-a", SceneFilter::ALL_SCENES, 0.5, 0.4, 0.5, 9.0);
        // Every run at 50 or more: shown, with a best.
        let c = compare(&[picked(a(), 60, 0.31), picked(b(), 60, 0.42)]);
        assert!(c.random_column);
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);
        assert_eq!(c.best_per_column["random"], 1);
        assert_eq!(c.rows[0].random.as_ref().unwrap().images, 60);
        let t = c.render_text();
        assert!(t.contains("in-scope  random"), "{t}");
        assert!(t.contains("0.420*"), "{t}");
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["random_column"], true);
        assert_eq!(v["rows"][1]["random"]["ap50_95"], 0.42);
        // Exactly the minimum is enough.
        let c = compare(&[
            picked(a(), MIN_RANDOM_HELDOUT, 0.31),
            picked(b(), MIN_RANDOM_HELDOUT, 0.42),
        ]);
        assert!(c.random_column);
        // One run under it: hidden everywhere, the rows keep their figures.
        let c = compare(&[picked(a(), 60, 0.31), picked(b(), 49, 0.42)]);
        assert!(!c.random_column);
        assert!(!c.best_per_column.contains_key("random"));
        let t = c.render_text();
        assert!(
            t.lines().nth(1).is_some_and(|h| !h.contains("random")),
            "{t}"
        );
        assert!(!t.contains("0.420"), "{t}");
        assert_eq!(c.rows[1].random.as_ref().unwrap().images, 49);
        assert_eq!(c.warnings.len(), 1, "60 / 49 differ: {:?}", c.warnings);
        // A run without any (an older version, or no random pick held out): hidden.
        let c = compare(&[picked(a(), 60, 0.31), b()]);
        assert!(!c.random_column);
        assert!(!compare(&[a(), b()]).random_column);
        // A run not benchmarked yet hides nothing; the routed row has none.
        let mut queued = b();
        queued.bench = None;
        let night = picked(
            run("night-a", SceneFilter::NIGHT_IR, 0.3, 0.5, 0.45, 11.0),
            60,
            0.2,
        );
        let c = compare(&[picked(a(), 60, 0.31), night, queued]);
        assert!(c.random_column);
        let routed = c.rows.last().unwrap();
        assert!(routed.routed && routed.random.is_none());
        assert_eq!(c.best_per_column["random"], 0);
        // Seed repeats: the column joins their mean only when shown.
        let seeded = |n: usize, seed: u64, ap: f64| {
            let mut r = picked(a(), n, ap);
            r.seed = Some(seed);
            r
        };
        let c = compare(&[seeded(60, 0, 0.3), seeded(60, 1, 0.4)]);
        let spread = c.seed_summaries[0].columns["random"];
        assert!((spread.mean - 0.35).abs() < 1e-9, "{spread:?}");
        let c = compare(&[seeded(60, 0, 0.3), seeded(40, 1, 0.4)]);
        assert!(!c.seed_summaries[0].columns.contains_key("random"));
    }

    #[test]
    fn differing_random_pick_counts_warn() {
        let a = picked(run("day-a", SceneFilter::DAY, 0.6, 0.2, 0.1, 10.0), 60, 0.3);
        let b = picked(run("day-b", SceneFilter::DAY, 0.5, 0.2, 0.1, 10.0), 58, 0.4);
        let c = compare(&[a.clone(), b]);
        assert_eq!(
            c.warnings,
            [
                "the runs have different numbers of random picks among their held-out frames \
                 (60 / 58): the label files changed between their exports, so their random-pick \
                 scores are not comparable; queue the comparison again"
            ]
        );
        assert!(c.comparable, "the rest of the table still compares");
        assert!(c.random_column, "both have enough: shown, with the warning");
        // A run without random picks is no count.
        let plain = run("day-c", SceneFilter::DAY, 0.5, 0.2, 0.1, 10.0);
        assert!(compare(&[a, plain]).warnings.is_empty());
    }

    #[test]
    fn runs_picking_over_different_classes_warn() {
        let a = run("a", SceneFilter::DAY, 0.5, 0.5, 0.5, 10.0);
        let mut b = run("b", SceneFilter::DAY, 0.5, 0.5, 0.5, 10.0);
        let same = compare(&[a.clone(), b.clone()]);
        assert!(
            !same
                .warnings
                .iter()
                .any(|w| w.contains("over different classes"))
        );
        b.select_classes = Some(vec!["car".into(), "person".into()]);
        let c = compare(&[a, b]);
        let w = c
            .warnings
            .iter()
            .find(|w| w.contains("over different classes"))
            .expect("a warning");
        assert!(w.contains("every class / car, person"), "{w}");
        assert!(w.contains("20 validation boxes"), "{w}");
    }

    #[test]
    fn runs_selected_by_different_metrics_are_marked() {
        let same = compare(&[
            run("a", SceneFilter::DAY, 0.5, 0.5, 0.5, 10.0),
            run("b", SceneFilter::DAY, 0.4, 0.4, 0.4, 10.0),
        ]);
        assert!(same.warnings.is_empty(), "{:?}", same.warnings);
        assert_eq!(same.rows[0].selected_by(), "");
        assert!(same.render_text().contains("best 12/27"));
        let mut b = run("b", SceneFilter::DAY, 0.4, 0.4, 0.4, 10.0);
        b.select_metric = SelectMetric::F2;
        let c = compare(&[run("a", SceneFilter::DAY, 0.5, 0.5, 0.5, 10.0), b]);
        assert_eq!(c.warnings.len(), 1);
        assert!(
            c.warnings[0].contains("different metrics (mAP / F2)"),
            "{:?}",
            c.warnings
        );
        assert_eq!(c.rows[1].select_metric, Some(SelectMetric::F2));
        assert_eq!(c.rows[1].selected_by(), " by F2");
        let text = c.render_text();
        assert!(text.contains("best 12 by F2/27"), "{text}");
        let json = serde_json::to_value(&c).unwrap();
        assert_eq!(json["rows"][1]["select_metric"], "f2");
    }
}
