//! The Import, Auto-label, Train and Models tabs' page ([`TrainingTemplate`], `train/training.html`):
//! [`view`] reads the run records, the capability and the config, and [`view_with`] builds the
//! template from pure stages, one per card: the environment line and GPU choices ([`env_view`]),
//! the family / size grid ([`family_rows`]), the large-download confirmations
//! ([`large_downloads`]), the run queue ([`run_rows`]), the finished runs in comparable groups
//! ([`result_items`], [`result_groups`]), the Comparison cards ([`compare_cards`]) and the dataset
//! list ([`dataset_rows`]).

use super::{EXTRA_SETS, runs, tasks};
use crate::resources::catalog::{self, Platform};
use crate::resources::export::train::{self as env, TrainPaths};
use crate::server::AppState;
use crate::server::train_area::tab_title;
use crate::training::capability::Capability;
use crate::training::compare::{self, CompareRun, Comparison};
use crate::training::dataset::{Balance, BalanceAxis, SceneFilter};
use crate::training::job::BenchSummary;
use crate::training::recipe;
use crate::training::service::{RunRecord, RunState};
use crate::training::{bench, estimate, retry};
use askama::Template;
use std::collections::BTreeMap;

pub(super) struct SizeCell {
    pub(super) label: String,
    pub(super) size: &'static str,
    pub(super) fits: bool,
    /// The full estimate (the cell's tooltip).
    pub(super) estimate: String,
    /// The compact one shown in the cell (`5.4 GB · batch 32 · 20 s`).
    pub(super) short: String,
    /// `light` | `medium` | `heavy` (the legend explains them).
    pub(super) weight: &'static str,
    pub(super) latency: &'static str,
    pub(super) why: String,
}

pub(super) struct FamilyRow {
    pub(super) title: &'static str,
    pub(super) license: &'static str,
    pub(super) available: bool,
    pub(super) why: String,
    pub(super) sizes: Vec<SizeCell>,
    /// Empty cells after the sizes, so every row of the grid has as many cells.
    pub(super) pad: usize,
    /// The family's starting point (`100 ep · patience 15 · lr 1e-4`, [`recipe::recommended`],
    /// [`recipe::default_lr_label`]): what blank Epochs / Patience / Learning rate fields mean.
    pub(super) recommended: String,
    /// The family's default learning rate (`data-default-lr` of its size checkboxes: the page's
    /// Learning rate placeholder follows the ticked family).
    pub(super) default_lr: &'static str,
}

/// One option of the LR schedule select (`""` = the family's default).
pub(super) struct LrScheduleChoice {
    pub(super) value: &'static str,
    pub(super) title: &'static str,
    pub(super) selected: bool,
}

/// One option of the "Select best by" select.
pub(super) struct SelectChoice {
    pub(super) value: &'static str,
    pub(super) title: &'static str,
    pub(super) hint: &'static str,
    pub(super) selected: bool,
}

/// A licence notice shared by some families (shown once under the size grid).
pub(super) struct NoticeRow {
    pub(super) families: String,
    pub(super) notice: &'static str,
}

/// One inference-latency class of the legend (`light`, its explanation).
pub(super) struct LatencyRow {
    pub(super) weight: &'static str,
    pub(super) text: &'static str,
}

pub(super) struct RunRow {
    pub(super) id: String,
    pub(super) name: String,
    pub(super) model: String,
    pub(super) state: &'static str,
    pub(super) busy: bool,
    /// Finish now applies: running (or paused) in its training stage, not stopped yet.
    pub(super) can_finish: bool,
    /// The Stop now / Continue notice while the run trains past an early-stop point (validation
    /// still improving); empty otherwise.
    pub(super) extend: String,
    /// The epoch Stop now keeps, and the point's extension number (Continue dismisses it).
    pub(super) extend_keep: u32,
    pub(super) extend_no: u32,
    pub(super) percent: u8,
    pub(super) describe: String,
    pub(super) dataset: String,
    /// The dataset's warnings, shown on the row.
    pub(super) warnings: Vec<String>,
    /// A failed run with a checkpoint: the Retry export button.
    pub(super) can_retry: bool,
    /// What the retry exports when training did not end, else empty.
    pub(super) retry_note: String,
    pub(super) files: RunFiles,
    pub(super) log_tail: String,
}

/// What a finished run leaves on disk.
pub(super) struct RunFiles {
    /// "1.2 GB" (bytes deleting the folder frees), empty when there is nothing.
    pub(super) size: String,
    /// "1.2 GB on disk, 900 MB of it checkpoints", empty when there is nothing.
    pub(super) text: String,
}

impl RunFiles {
    /// The files of `r`; nothing is measured while it is busy.
    fn of(r: &RunRecord) -> Self {
        let total = if r.state.is_busy() {
            0
        } else {
            retry::disk_use(&r.run_dir)
        };
        if total == 0 {
            return Self {
                size: String::new(),
                text: String::new(),
            };
        }
        let size = catalog::format_size(total);
        let work = retry::disk_use(&retry::work_dir(&r.run_dir));
        let text = if work > 0 {
            format!(
                "Run files: {size} on disk, {} of it checkpoints",
                catalog::format_size(work)
            )
        } else {
            format!("Run files: {size} on disk")
        };
        Self { size, text }
    }

    /// The Remove button's confirmation. The installed model is not in the run's folder.
    fn confirm(&self, r: &RunRecord) -> String {
        let model = if r.output.is_some() {
            " The installed model stays, but its held-out frames go, so it can no longer be \
             benchmarked on them."
        } else {
            ""
        };
        format!(
            "Delete the files of run {} ({}) from this machine?{model}",
            r.name, self.size
        )
    }
}

pub(super) struct ResultRow {
    pub(super) id: String,
    pub(super) name: String,
    pub(super) family: String,
    pub(super) license: String,
    pub(super) classes: String,
    /// "merged: truck \u{2192} car" (empty without merges).
    pub(super) merges: String,
    pub(super) benchmark: String,
    /// `/benchmark/compare?set=...`: Compare on the run's held-out images.
    pub(super) compare_href: String,
    /// The new model and the current default on the same frames, in words.
    pub(super) compare: String,
    /// The share of the benchmark that is its own training frames, in words (empty: none).
    pub(super) trained_on: String,
    /// It is the default model now.
    pub(super) is_default: bool,
    pub(super) finished: String,
    pub(super) in_config: bool,
    pub(super) warnings: Vec<String>,
    pub(super) files: RunFiles,
    /// The Remove button's confirmation.
    pub(super) confirm: String,
    /// Held-out person recall, vehicle recall and AP50-95 on every held-out frame (a dash
    /// without a benchmark): the sortable columns beside the sentence.
    pub(super) person_r: String,
    pub(super) vehicle_r: String,
    pub(super) ap: String,
    /// Its place in its group's Best first order ([`compare::best_order_by`], `data-best`).
    pub(super) best: usize,
}

/// Runs whose held-out scores compare: the same held-out frames
/// ([`BenchSummary::heldout_fingerprint`]), trained classes and class merges.
type HeldoutKey = (String, Vec<String>, BTreeMap<String, String>);

/// [`HeldoutKey`] of a finished run; None without a benchmark or a fingerprint (a run of an
/// older version): such a run compares with nothing and is a group of its own.
fn heldout_key(r: &RunRecord) -> Option<HeldoutKey> {
    let fp = r.benchmark.as_ref()?.heldout_fingerprint.clone()?;
    let classes = r.output.as_ref()?.classes.clone();
    Some((fp, classes, r.request.rename.clone()))
}

/// One group of the Models tab's results list: the runs of one [`HeldoutKey`] (or one run
/// that compares with nothing), best first.
pub(super) struct ResultGroup {
    /// The group's header row ("Held-out set 3f2a1c..., 4 runs ...").
    pub(super) head: String,
    pub(super) rows: Vec<ResultRow>,
}

/// One finished run for [`result_groups`].
pub(super) struct GroupItem {
    pub(super) key: Option<HeldoutKey>,
    pub(super) benchmarked: bool,
    pub(super) keys: compare::BestKeys,
    pub(super) row: ResultRow,
}

/// The results list in groups of runs whose held-out scores compare (`items` newest first): the
/// groups in the order of their newest run, each best first ([`compare::best_order_by`]: person
/// R, vehicle R, AP50-95; equal ones newest first), `best` set to the place in the group.
fn result_groups(items: Vec<GroupItem>) -> Vec<ResultGroup> {
    let mut groups: Vec<(Option<HeldoutKey>, bool, Vec<GroupItem>)> = Vec::new();
    for it in items {
        match groups.iter_mut().find(|g| g.0.is_some() && g.0 == it.key) {
            Some(g) => g.2.push(it),
            None => groups.push((it.key.clone(), it.benchmarked, vec![it])),
        }
    }
    // A held-out set two groups share (other classes or merges) names what differs.
    let shared = |fp: &str| {
        groups
            .iter()
            .filter(|g| g.0.as_ref().is_some_and(|k| k.0 == fp))
            .count()
            > 1
    };
    let heads: Vec<String> = groups
        .iter()
        .map(|(key, benchmarked, items)| match key {
            Some((fp, classes, rename)) => {
                let short: String = fp.chars().take(6).collect();
                let n = items.len();
                let mut head = format!(
                    "Held-out set {short}\u{2026}, {n} run{}",
                    if n == 1 { "" } else { "s" }
                );
                if shared(fp) {
                    head.push_str(&format!(", classes {}", classes.join(", ")));
                    let merges = merges_text(rename);
                    if !merges.is_empty() {
                        head.push_str(&format!(" ({merges})"));
                    }
                }
                if n > 1 {
                    head.push_str(" \u{2014} scores compare only inside a group");
                }
                head
            }
            None if *benchmarked => {
                "Held-out set not recorded (a run of an older version): its scores compare with \
                 no other run"
                    .to_string()
            }
            None => "Not benchmarked".to_string(),
        })
        .collect();
    groups
        .into_iter()
        .zip(heads)
        .map(|((_, _, items), head)| {
            let keys: Vec<compare::BestKeys> = items.iter().map(|i| i.keys).collect();
            let order = compare::best_order_by(&keys);
            let mut slots: Vec<Option<GroupItem>> = items.into_iter().map(Some).collect();
            let rows = order
                .into_iter()
                .enumerate()
                .filter_map(|(place, i)| {
                    let mut row = slots.get_mut(i)?.take()?.row;
                    row.best = place + 1;
                    Some(row)
                })
                .collect();
            ResultGroup { head, rows }
        })
        .collect()
}

/// A recall as the Models tab shows it (`0.39`; a dash when unknown).
fn recall_text(v: Option<f64>) -> String {
    v.filter(|v| v.is_finite())
        .map_or_else(|| "\u{2014}".into(), |v| format!("{v:.2}"))
}

/// One GPU of the page's GPU selects (a PC with more than one usable NVIDIA GPU).
pub(super) struct GpuChoice {
    /// `cuda:N` (the form's `gpu`).
    pub(super) value: String,
    /// `NVIDIA GeForce RTX 3090 (cuda:1, 24 GB, PyTorch cu126)`.
    pub(super) label: String,
}

/// One checkbox of the Train form's Benchmark devices fieldset.
pub(super) struct BenchDeviceChoice {
    /// Canonical spec (`ort:cuda:0`; the form's `benchmark_device`).
    pub(super) value: String,
    pub(super) label: String,
    pub(super) checked: bool,
    /// Saved but not runnable on this machine now (kept so a save does not drop it silently).
    pub(super) unavailable: bool,
}

/// The Benchmark devices checkboxes: every runnable device of this machine, ticked when in
/// `saved` (`training.benchmark_devices`), then saved ones that cannot run here; and what none
/// ticked means (the `auto` pick for a model without its own device).
fn bench_device_choices(state: &AppState, saved: &[String]) -> (Vec<BenchDeviceChoice>, String) {
    let sel = crate::server::devices::devices_snapshot(state).selection;
    let mut v: Vec<BenchDeviceChoice> = sel
        .options
        .iter()
        .filter(|o| o.runnable)
        .map(|o| {
            let value = o.spec.to_string();
            BenchDeviceChoice {
                checked: saved.contains(&value),
                label: o.label.clone(),
                value,
                unavailable: false,
            }
        })
        .collect();
    for d in saved {
        if !v.iter().any(|c| &c.value == d) {
            v.push(BenchDeviceChoice {
                value: d.clone(),
                label: "not runnable on this machine now".into(),
                checked: true,
                unavailable: true,
            });
        }
    }
    let auto = sel
        .auto_pick()
        .map(|o| format!("{} ({})", o.spec, o.label))
        .unwrap_or_else(|| "none runnable".into());
    (v, auto)
}

/// One checkbox of the Train form's Runs fieldset.
pub(super) struct SceneChoice {
    /// The canonical filter (`all`, `day`, `night+ir`, `night`, `ir`; the form's `scene`).
    pub(super) value: &'static str,
    pub(super) title: &'static str,
    pub(super) describe: &'static str,
    /// The model-name prefix (`custom`, `day`, `night`, ...), for the runs line.
    pub(super) prefix: &'static str,
    /// Shown at once (else under "Other").
    pub(super) preset: bool,
}

/// The Runs checkboxes: the presets (mixed, day, night + IR), then night only and IR only.
pub(super) const SCENE_CHOICES: [(SceneFilter, &str, bool); 5] = [
    (SceneFilter::ALL_SCENES, "All scenes (mixed)", true),
    (SceneFilter::DAY, "Day", true),
    (SceneFilter::NIGHT_IR, "Night + IR", true),
    (SceneFilter::new(false, true, false), "Night only", false),
    (SceneFilter::new(false, false, true), "IR only", false),
];

/// One cell of a comparison table (AP50-95, AP50 beneath).
pub(super) struct CompareCell {
    pub(super) ap: String,
    pub(super) ap50: String,
    pub(super) images: usize,
    pub(super) best: bool,
}

pub(super) struct CompareRowView {
    pub(super) name: String,
    /// The training seed of a repeat (shown as `seed 1` beside the model).
    pub(super) seed: Option<u64>,
    pub(super) model: String,
    pub(super) scene: String,
    pub(super) routed: bool,
    /// The run's state while it has no benchmark (`running`, `failed`); empty when benchmarked.
    pub(super) state: String,
    /// `best 12 / stopped 27`.
    pub(super) epochs: String,
    /// Day, night, IR, all.
    pub(super) cells: Vec<CompareCell>,
    pub(super) in_scope: String,
    pub(super) ms: String,
    pub(super) ms_best: bool,
    /// Person recall, vehicle recall, false alerts ([`ALERT_COLUMNS`]).
    pub(super) alerts: Vec<AlertCell>,
    /// AP50-95 on the held-out random picks (shown with [`CompareCard::random`]).
    pub(super) random: AlertCell,
    /// Its place in the card's Best first order ([`compare::best_order`], `data-best`); None for
    /// a routed row.
    pub(super) best: Option<usize>,
}

/// The "mean of N seeds" row of a Comparison card.
pub(super) struct SeedSummaryView {
    pub(super) label: String,
    pub(super) model: String,
    pub(super) scene: String,
    /// Day, night, IR, all, in scope, random picks (only with [`CompareCard::random`]), p50 ms,
    /// person R, vehicle R, false alerts, person false alerts: `mean (min-max)`.
    pub(super) cells: Vec<String>,
}

/// `0.412 (0.398-0.431)` of a [`compare::Spread`] with `decimals` places (`scale` 100 = percent).
fn spread_text(s: Option<&compare::Spread>, decimals: usize, scale: f64, unit: &str) -> String {
    s.map_or_else(
        || "\u{2014}".into(),
        |s| {
            format!(
                "{:.d$}{unit} ({:.d$}EN\u{2014}{:.d$}{unit})",
                s.mean * scale,
                s.min * scale,
                s.max * scale,
                d = decimals
            )
        },
    )
}

fn seed_summary_view(sm: &compare::SeedSummary, random: bool) -> SeedSummaryView {
    let col =
        |c: &str, d: usize, scale: f64, unit: &str| spread_text(sm.columns.get(c), d, scale, unit);
    let mut cells = vec![
        col("day", 3, 1.0, ""),
        col("night", 3, 1.0, ""),
        col("ir", 3, 1.0, ""),
        col("all", 3, 1.0, ""),
        "\u{2014}".into(),
        col("ms_p50", 1, 1.0, ""),
        col("person_recall", 2, 1.0, ""),
        col("vehicle_recall", 2, 1.0, ""),
        col("false_alerts", 1, 100.0, "%"),
        col("person_false_alerts", 1, 100.0, "%"),
    ];
    if random {
        // After `in scope`, as in the table's head.
        cells.insert(5, col("random", 3, 1.0, ""));
    }
    SeedSummaryView {
        label: sm.label.clone(),
        model: sm.model.clone(),
        scene: sm.scene.clone(),
        cells,
    }
}

/// Text and hover text of a comparison row's random-picks cell: AP50-95, and what it covers.
fn random_cell(row: &compare::Row) -> (String, String) {
    let Some(s) = row
        .random
        .as_ref()
        .filter(|s| s.ap50_95.is_some_and(f64::is_finite))
    else {
        return ("\u{2014}".into(), String::new());
    };
    let mut title = format!(
        "AP50-95 on {} held-out frames picked at random ({} objects)",
        s.images, s.gt
    );
    if let Some(a) = s.ap50.filter(|a| a.is_finite()) {
        title.push_str(&format!(", AP50 {a:.3}"));
    }
    if let Some(r) = s.recall.filter(|r| r.is_finite()) {
        let at = row
            .threshold
            .map_or_else(String::new, |t| format!(" at threshold {t:.2}"));
        title.push_str(&format!(", recall {r:.2}{at}"));
    }
    (ap_text(s.ap50_95), title)
}

/// The tip of the Comparison card's random-picks column head (`frames`, `objects` of the first
/// run that has them; runs whose counts differ are warned about).
fn random_tip(frames: usize, objects: usize) -> String {
    format!(
        "AP50-95 on held-out frames the queue picked at random ({frames} frames, {objects} \
         objects): nobody chose them, so this is the least biased score. Hidden under {} frames.",
        compare::MIN_RANDOM_HELDOUT
    )
}

/// One alerting cell of a comparison row: `0.39`, `3/130`, a dash when unknown.
pub(super) struct AlertCell {
    pub(super) text: String,
    /// Hover text (the threshold, the false-alert rate).
    pub(super) title: String,
    pub(super) best: bool,
}

/// The alerting columns of the Comparison card, in table order ([`compare::COLUMNS`] ids).
const ALERT_COLUMNS: [&str; 4] = [
    "person_recall",
    "vehicle_recall",
    "false_alerts",
    "person_false_alerts",
];

/// The Comparison card of one `compare_group`.
pub(super) struct CompareCard {
    pub(super) group: String,
    /// The latest group (and the latest with a result) starts open; older ones collapsed.
    pub(super) open: bool,
    pub(super) comparable: bool,
    pub(super) warnings: Vec<String>,
    pub(super) rows: Vec<CompareRowView>,
    /// "Mean of N seeds" rows (runs that differ only by training seed).
    pub(super) seed_summaries: Vec<SeedSummaryView>,
    /// The random-picks column is shown ([`compare::Comparison::random_column`]).
    pub(super) random: bool,
    /// Its head's tip ([`random_tip`]); empty when hidden.
    pub(super) random_tip: String,
}

/// One axis checkbox of the Train form's Balance fieldset.
pub(super) struct BalanceChoice {
    /// `class`, `scene`, `camera`, `camera-day` (the form's `balance`).
    pub(super) value: &'static str,
    pub(super) title: &'static str,
    pub(super) hint: &'static str,
    pub(super) checked: bool,
}

/// The Balance fieldset's defaults: `training.balance` when set, else off.
pub(super) struct BalanceView {
    pub(super) axes: Vec<BalanceChoice>,
    pub(super) on: bool,
    pub(super) gentle: bool,
    pub(super) full: bool,
    /// A configured strength other than 0.5 and 1 (a third radio, checked).
    pub(super) custom: Option<String>,
    pub(super) max_repeat: u32,
    pub(super) most: u32,
    pub(super) cap_on: bool,
    pub(super) cap: String,
}

impl BalanceView {
    fn of(b: Option<&Balance>) -> Self {
        use crate::training::dataset::balance::{DEFAULT_CAP, MAX_REPEAT};
        let d = Balance::default();
        let b = b.unwrap_or(&d);
        let s = b.strength;
        Self {
            axes: BalanceAxis::ALL
                .iter()
                .map(|&a| BalanceChoice {
                    value: a.as_str(),
                    title: match a {
                        BalanceAxis::Class => "Class",
                        BalanceAxis::Scene => "Scene",
                        BalanceAxis::Camera => "Camera",
                        BalanceAxis::CameraDay => "Camera + day",
                    },
                    hint: a.describe(),
                    checked: b.axes.contains(&a),
                })
                .collect(),
            on: !b.is_off(),
            gentle: s == 0.5,
            full: s == 1.0,
            custom: (s != 0.5 && s != 1.0).then(|| crate::config::clean_f32(s).to_string()),
            max_repeat: b.max_repeat,
            most: MAX_REPEAT,
            cap_on: b.cap_dominant.is_some(),
            cap: crate::config::clean_f32(b.cap_dominant.unwrap_or(DEFAULT_CAP)).to_string(),
        }
    }
}

pub(super) struct DatasetRow {
    pub(super) path: String,
    /// When it was exported, or the folder's name ([`crate::server::train_area::ExportRow`]).
    pub(super) when: String,
    /// Its frames, when it has an export report.
    pub(super) frames: Option<u64>,
    /// "17 train, 9 held out".
    pub(super) split: String,
    /// "person 27, bicycle 1".
    pub(super) classes: String,
    pub(super) exists: bool,
    /// In `training.datasets` (else found under `<training dir>/exports`).
    pub(super) saved: bool,
}

impl DatasetRow {
    fn of(path: String, dir: &std::path::Path, exists: bool, saved: bool) -> Self {
        let r = crate::server::train_area::export_row(dir);
        Self {
            path,
            when: r.when,
            frames: r.frames,
            split: r.split,
            classes: r.classes,
            exists,
            saved,
        }
    }
}

#[derive(Template)]
#[template(path = "train/training.html")]
pub(in crate::server) struct TrainingTemplate {
    pub(super) nav: &'static str,
    /// The tab of the Train area this renders ([`crate::server::train_area::TABS`]).
    pub(super) tab: &'static str,
    pub(super) tab_title: &'static str,
    pub(super) version: &'static str,
    pub(super) message: Option<String>,
    pub(super) error: Option<String>,
    pub(super) available: bool,
    pub(super) summary: String,
    /// Where the open-vocabulary labelers run (PyTorch on Metal or CUDA), or why they cannot.
    pub(super) labeler_device: String,
    pub(super) reasons: Vec<String>,
    pub(super) env_ready: bool,
    pub(super) env_note: String,
    /// "the training environment (about 450 MB)": what the Train confirmation names.
    pub(super) env_download: String,
    pub(super) allow_large: bool,
    /// What an import may need over the large-download gate (`None` = the confirmation is not
    /// shown: nothing large is missing, large downloads are allowed or there is no GPU).
    pub(super) import_large: Option<String>,
    /// The same for the auto-labelers.
    pub(super) autolabel_large: Option<String>,
    /// The GPUs the PyTorch jobs may run on, the automatic pick first; empty = nothing to choose
    /// (one GPU, Apple silicon, none): the selects are not shown.
    gpu_choices: Vec<GpuChoice>,
    /// What a GPU on another PyTorch build than the automatic pick's would download
    /// (`PyTorch cu130, about 3.1 GB, for NVIDIA GeForce RTX 5090`), when not set up.
    other_env: Option<String>,
    pub(super) datasets: Vec<DatasetRow>,
    pub(super) families: Vec<FamilyRow>,
    /// Most sizes of a family (the grid's size columns).
    pub(super) size_cols: usize,
    pub(super) notices: Vec<NoticeRow>,
    pub(super) latencies: Vec<LatencyRow>,
    /// `training.epochs`, blank = each family's recommendation.
    pub(super) epochs: String,
    /// `training.patience`, blank = each family's recommendation.
    pub(super) patience: String,
    /// `training.lr`, blank = each family's default.
    pub(super) lr: String,
    /// Every family's default learning rate (the field's hint).
    pub(super) lr_defaults: String,
    /// The LR schedule select, `training.lr_schedule` selected.
    pub(super) lr_schedules: Vec<LrScheduleChoice>,
    /// Which schedules each family takes and its final factor (the Advanced hint).
    pub(super) lr_schedule_hint: String,
    /// `training.lr_min_factor`, blank = each family's default.
    pub(super) lr_min_factor: String,
    /// "Select best by": [`recipe::SELECT_METRICS`], `training.select_metric` selected.
    pub(super) select_metrics: Vec<SelectChoice>,
    /// The selected choice's hint.
    pub(super) select_hint: &'static str,
    pub(super) scenes: Vec<SceneChoice>,
    pub(super) balance: BalanceView,
    /// Day / night / mixed comparisons of runs queued together, the latest first.
    pub(super) comparisons: Vec<CompareCard>,
    pub(super) public_available: bool,
    /// The public night subset is downloaded (else mixing it in needs the large-download
    /// confirmation).
    pub(super) public_installed: bool,
    /// Its download size (`926 MB`).
    pub(super) public_size: String,
    pub(super) extra_sets: Vec<&'static str>,
    /// The Benchmark devices checkboxes ([`bench_device_choices`]).
    pub(super) bench_devices: Vec<BenchDeviceChoice>,
    /// What none ticked runs on: the device `auto` picks (a model without its own device).
    pub(super) bench_auto: String,
    pub(super) runs: Vec<RunRow>,
    /// Finished runs in groups whose held-out scores compare ([`result_groups`]).
    pub(super) results: Vec<ResultGroup>,
    pub(super) models: Vec<String>,
    /// The models folder (where a trained model goes by default).
    pub(super) models_dir: String,
    pub(super) default_model: String,
    /// The last default switch, while it can be reverted.
    pub(super) revert: Option<crate::training::switch::RevertView>,
    /// `{model: verifier}` as JSON: the Use card's confirm names the one replaced.
    pub(super) verifiers: String,
    pub(super) training_dir: String,
    /// `training.dir` as configured (empty = the default folder).
    pub(super) training_dir_setting: String,
    /// The default training folder (`<data root>/training`).
    pub(super) training_dir_default: String,
    /// Import and auto-label jobs.
    pub(super) tasks: Vec<crate::server::training_tasks::TaskRow>,
    /// A job is queued or running (its table starts open).
    pub(super) tasks_busy: bool,
    /// A run or task is queued or running (the training folder cannot change).
    pub(super) jobs_busy: bool,
    pub(super) footage: Vec<crate::server::training_tasks::FootageRow>,
    pub(super) dataset_dir: String,
    /// Imports listed in the folder auto-label runs on (`collect.resolved_review_dir`, the
    /// Review tab's folder), so their Auto-label buttons find the frames.
    pub(super) imports: Vec<crate::training::import::write::ImportLine>,
    /// That folder.
    pub(super) imports_dir: String,
}

/// Finish now applies to `r`: running or paused in its training stage, not stopped and not
/// exporting yet ([`TrainingService::finish`] checks the same).
fn can_finish(r: &RunRecord) -> bool {
    r.state.is_active()
        && !r.retrying
        && r.stage == Some(crate::training::job::TrainStage::Train)
        && r.progress
            .as_ref()
            .is_none_or(|p| p.stopped.is_none() && p.phase.is_none())
}

/// The Stop now / Continue notice of `r`: while it can still finish and trains past an
/// early-stop point.
fn extension(r: &RunRecord) -> Option<&crate::training::progress::ExtensionNotice> {
    if !can_finish(r) {
        return None;
    }
    r.progress.as_ref()?.extension.as_ref()
}

/// The Benchmark's Compare step on the images a run was benchmarked on: its held-out set comes
/// first in `datasets`. Without a benchmark the step opens on its default set.
pub(super) fn compare_href(b: Option<&BenchSummary>) -> String {
    match b.and_then(|b| b.datasets.first()) {
        Some(set) => format!(
            "/benchmark/compare?set={}",
            crate::server::config_page::encode_uri_component(set)
        ),
        None => "/benchmark/compare".to_string(),
    }
}

pub(super) fn bench_text(b: Option<&BenchSummary>, err: Option<&str>) -> String {
    match (b, err) {
        (Some(b), _) => {
            let mut s = format!(
                "AP50 {}{} \u{b7} {} on {} \u{b7} {} images ({})",
                b.ap50.map_or_else(|| "n/a".into(), |a| format!("{a:.3}")),
                b.ap_basis
                    .as_ref()
                    .filter(|_| b.ap50.is_some())
                    .map_or_else(String::new, basis_short),
                b.ms_p50
                    .map_or_else(|| "?".into(), |m| format!("{m:.1} ms")),
                b.device.as_deref().unwrap_or("?"),
                b.images,
                b.datasets.join(", ")
            );
            // AP50-95 per scene of the held-out frames, and over the scenes it trained on.
            let scenes: Vec<String> = ["day", "night", "ir"]
                .iter()
                .filter_map(|k| {
                    let a = b.by_scene.get(*k).filter(|a| a.images > 0)?;
                    Some(format!("{} {}", scene_name(k), ap_text(a.ap50_95)))
                })
                .collect();
            if !scenes.is_empty() {
                s.push_str(&format!(" \u{b7} AP50-95 {}", scenes.join(" / ")));
            }
            if let Some(a) = b.in_scope.as_ref().and_then(|a| a.ap50_95) {
                s.push_str(&format!(" \u{b7} in scope {a:.3}"));
            }
            // Alerting at the configured threshold (only the figures present).
            let mut alerting = Vec::new();
            if let Some(r) = b.person_recall.filter(|r| r.is_finite()) {
                alerting.push(format!("person recall {r:.2}"));
            }
            if let Some(r) = b.vehicle_recall.filter(|r| r.is_finite()) {
                alerting.push(format!("vehicle recall {r:.2}"));
            }
            if let (Some(a), Some(e)) = (b.empty_frame_alerts, b.empty_frames.filter(|e| *e > 0)) {
                alerting.push(format!("false alerts {a}/{e} empty frames"));
            }
            if let (Some(a), Some(f)) = (
                b.person_false_alerts,
                b.person_free_frames.filter(|f| *f > 0),
            ) {
                alerting.push(format!("person alerts {a}/{f} frames without a person"));
            }
            if !alerting.is_empty() {
                s.push_str(&format!(" \u{b7} {}", alerting.join(" \u{b7} ")));
                if let Some(t) = b.threshold {
                    s.push_str(&format!(" @ {t:.2}"));
                }
            }
            s
        }
        (None, Some(e)) => format!("benchmark failed: {e}"),
        (None, None) => "not benchmarked".into(),
    }
}

/// ` (car, person)`: the classes an AP averages, or ` (every class: none had 20 objects)`.
fn basis_short(b: &crate::benchmark::metrics::ApBasis) -> String {
    if b.fell_back {
        format!(" ({})", b.describe())
    } else {
        format!(" ({})", b.classes.join(", "))
    }
}

/// `day`, `night`, `IR`.
fn scene_name(k: &str) -> &str {
    if k == "ir" { "IR" } else { k }
}

/// An AP as shown (`0.512`), a dash when unknown.
fn ap_text(v: Option<f64>) -> String {
    v.filter(|v| v.is_finite())
        .map_or_else(|| "\u{2014}".into(), |v| format!("{v:.3}"))
}

/// The runs of each `compare_group` in `records` (queue order), groups by their latest run,
/// the newest group first; only groups of two runs or more.
pub(super) fn compare_groups(records: &[RunRecord]) -> Vec<(String, Vec<&RunRecord>)> {
    let mut groups: Vec<(String, Vec<&RunRecord>, usize)> = Vec::new();
    for (i, r) in records.iter().enumerate() {
        let Some(g) = r.request.compare_group.as_deref() else {
            continue;
        };
        match groups.iter_mut().find(|(k, _, _)| k == g) {
            Some((_, v, last)) => {
                v.push(r);
                *last = i;
            }
            None => groups.push((g.to_string(), vec![r], i)),
        }
    }
    groups.retain(|(_, v, _)| v.len() >= 2);
    groups.sort_by_key(|g| std::cmp::Reverse(g.2));
    groups.into_iter().map(|(g, v, _)| (g, v)).collect()
}

/// A run record as a row of [`compare::compare`].
fn compare_run(r: &RunRecord) -> CompareRun {
    let p = r.progress.as_ref();
    let stop = p.and_then(|p| p.stopped.as_ref());
    CompareRun {
        name: r.name.clone(),
        model: r.request.recipe.label(),
        scene: r.request.scene,
        classes: r.request.classes.clone().unwrap_or_default(),
        rename: r.request.rename.clone(),
        excluded: r.request.excluded,
        ignore: p
            .and_then(|p| p.settings.as_ref())
            .map(crate::training::progress::RunSettings::ignore_mode),
        negative_share: p
            .and_then(|p| p.settings.as_ref())
            .and_then(|s| s.negative_share),
        static_cap: p
            .and_then(|p| p.settings.as_ref())
            .and_then(|s| s.static_cap),
        workers: p.and_then(|p| p.settings.as_ref()).and_then(|s| s.workers),
        balance: r.request.balance.clone(),
        best_epoch: stop
            .map(|s| s.best_epoch)
            .or_else(|| p.and_then(|p| p.best.as_ref()).map(|b| b.epoch)),
        // A run that trained to its end stopped at its last epoch.
        stopped_epoch: stop.map(|s| s.epoch).or_else(|| {
            p.filter(|p| p.epoch > 0 && (r.state == RunState::Done || p.phase.is_some()))
                .map(|p| p.epoch)
        }),
        seed: r.request.train_seed,
        select_metric: r.select_metric(),
        select_classes: crate::training::progress::selected_classes(p),
        bench: r.benchmark.clone(),
    }
}

/// The comparison of `group` in `records`: its runs and the table; None when no run has it.
pub(super) fn group_comparison<'a>(
    records: &'a [RunRecord],
    group: &str,
) -> Option<(Vec<&'a RunRecord>, Comparison)> {
    let runs: Vec<&RunRecord> = records
        .iter()
        .filter(|r| r.request.compare_group.as_deref() == Some(group))
        .collect();
    if runs.is_empty() {
        return None;
    }
    let c = compare::compare(&runs.iter().map(|r| compare_run(r)).collect::<Vec<_>>());
    Some((runs, c))
}

/// The Comparison cards of `records` (see [`compare_groups`]); older groups start collapsed.
pub(super) fn compare_cards(records: &[RunRecord]) -> Vec<CompareCard> {
    let groups = compare_groups(records);
    // Open: the newest group, and the newest one with a result when that one has none yet.
    let with_results = groups
        .iter()
        .position(|(_, runs)| runs.iter().any(|r| r.benchmark.is_some()));
    groups
        .into_iter()
        .enumerate()
        .map(|(i, (group, runs))| {
            let c = compare::compare(&runs.iter().map(|r| compare_run(r)).collect::<Vec<_>>());
            let best = |row: usize, col: &str| c.best_per_column.get(col) == Some(&row);
            let order = compare::best_order(&c.rows);
            let rows = c
                .rows
                .iter()
                .enumerate()
                .map(|(k, row)| {
                    let state = runs
                        .get(k)
                        .filter(|r| !row.routed && r.benchmark.is_none())
                        .map(|r| {
                            if r.state == RunState::Done {
                                "not benchmarked".to_string()
                            } else {
                                r.state.as_str().to_string()
                            }
                        })
                        .unwrap_or_default();
                    let cells = [
                        ("day", &row.day),
                        ("night", &row.night),
                        ("ir", &row.ir),
                        ("all", &row.all),
                    ]
                    .into_iter()
                    .map(|(col, cell)| CompareCell {
                        ap: ap_text(cell.ap50_95),
                        ap50: cell.ap50.map_or_else(String::new, |a| format!("{a:.3}")),
                        images: cell.images,
                        best: !row.routed && best(k, col),
                    })
                    .collect();
                    let by = row.selected_by();
                    let epochs = match (row.best_epoch, row.stopped_epoch) {
                        (Some(b), Some(s)) => format!("best {b}{by} / stopped {s}"),
                        (Some(b), None) => format!("best {b}{by}"),
                        (None, Some(s)) => format!("stopped {s}"),
                        (None, None) => "\u{2014}".into(),
                    };
                    CompareRowView {
                        best: order.iter().position(|&i| i == k).map(|p| p + 1),
                        name: row.name.clone(),
                        seed: row.seed,
                        model: row.model.clone(),
                        scene: row.scene.clone(),
                        routed: row.routed,
                        state,
                        epochs,
                        cells,
                        in_scope: ap_text(row.in_scope),
                        ms: row
                            .ms_p50
                            .map_or_else(|| "\u{2014}".into(), |m| format!("{m:.1}")),
                        ms_best: !row.routed && best(k, "ms_p50"),
                        alerts: ALERT_COLUMNS
                            .iter()
                            .map(|col| {
                                let (text, title) = alert_cell(row, col);
                                AlertCell {
                                    text,
                                    title,
                                    best: !row.routed && best(k, col),
                                }
                            })
                            .collect(),
                        random: {
                            let (text, title) = random_cell(row);
                            AlertCell {
                                text,
                                title,
                                best: !row.routed && best(k, "random"),
                            }
                        },
                    }
                })
                .collect();
            let random_tip = c
                .rows
                .iter()
                .find_map(|r| r.random.as_ref())
                .filter(|_| c.random_column)
                .map_or_else(String::new, |s| random_tip(s.images, s.gt));
            CompareCard {
                group,
                open: i == 0 || Some(i) == with_results,
                comparable: c.comparable,
                warnings: c.warnings,
                rows,
                seed_summaries: c
                    .seed_summaries
                    .iter()
                    .map(|sm| seed_summary_view(sm, c.random_column))
                    .collect(),
                random: c.random_column,
                random_tip,
            }
        })
        .collect()
}

/// Text and hover text of alerting column `col` of a comparison row (a dash when the run has
/// no such figure: not benchmarked, an older version, a routed row).
fn alert_cell(row: &compare::Row, col: &str) -> (String, String) {
    let at = row
        .threshold
        .map_or_else(String::new, |t| format!(" at threshold {t:.2}"));
    let dash = || ("\u{2014}".to_string(), String::new());
    match col {
        "person_recall" | "vehicle_recall" => {
            let v = if col == "person_recall" {
                row.person_recall
            } else {
                row.vehicle_recall
            };
            v.filter(|v| v.is_finite()).map_or_else(dash, |v| {
                (format!("{v:.2}"), format!("recall{at}, IoU 0.5"))
            })
        }
        "person_false_alerts" => row.person_false_alerts.map_or_else(dash, |f| {
            let rate = f
                .rate()
                .map_or_else(String::new, |r| format!(" ({:.1}%)", r * 100.0));
            (
                f.text(),
                format!(
                    "{} of {} frames without a person had a person detection{rate}{at}",
                    f.alerts, f.frames
                ),
            )
        }),
        "false_alerts" => row.false_alerts.map_or_else(dash, |f| {
            let rate = f
                .rate()
                .map_or_else(String::new, |r| format!(" ({:.1}%)", r * 100.0));
            (
                f.text(),
                format!(
                    "{} of {} frames without any object alerted{rate}{at}",
                    f.alerts, f.empty_frames
                ),
            )
        }),
        _ => dash(),
    }
}

/// `light: runs on a CPU ...` -> (`light`, `runs on a CPU ...`).
pub(super) fn latency_parts(hint: &'static str) -> (&'static str, &'static str) {
    hint.split_once(": ").unwrap_or(("", hint))
}

/// The page of `tab` with `r`'s message (`Ok`) or error (`Err`), in the caller's words: the end of
/// a form action that answers with the page either way.
pub(in crate::server) fn flash(
    state: &AppState,
    tab: &'static str,
    r: Result<String, String>,
) -> axum::response::Response {
    match r {
        Ok(m) => crate::server::render(&view(state, tab, Some(m), None)),
        Err(e) => crate::server::render(&view(state, tab, None, Some(e))),
    }
}

pub(in crate::server) fn view(
    state: &AppState,
    tab: &'static str,
    message: Option<String>,
    error: Option<String>,
) -> TrainingTemplate {
    let all = runs(state);
    let cap = crate::training::capability_with_runs(&all);
    view_with(state, tab, message, error, all, cap)
}

/// [`view`] of the run records `all` on `cap` (tests pass their own, so the page does not
/// depend on the machine they run on). The stages, in the order they read files and the queue.
pub(super) fn view_with(
    state: &AppState,
    tab: &'static str,
    message: Option<String>,
    error: Option<String>,
    all: Vec<RunRecord>,
    cap: Capability,
) -> TrainingTemplate {
    let config = state.config_read().clone();
    let dataset_dir = crate::collect::CollectSettings::for_config(&config).dir;
    let (bench_devices, bench_auto) =
        bench_device_choices(state, &config.training.benchmark_devices);
    let imports_dir = config.collect.resolved_review_dir(&config.data_root());
    let paths = TrainPaths::for_config(&config);
    let env = env_view(&cap, &paths, Platform::current());
    let (families, size_cols) = family_rows(&cap);
    let notices = notice_rows(&cap);
    let latencies = latency_rows(&cap);
    let (import_large, autolabel_large) =
        large_downloads(&cap, config.allow_large_downloads, &env, &paths);
    let run_rows = run_rows(&all);
    let names: Vec<String> = config.models.iter().map(|m| m.effective_name()).collect();
    let current = crate::training::switch::current_default(&config).unwrap_or_default();
    let results = result_groups(result_items(&all, &names, &current));
    let default_model = config
        .effective_default_index()
        .and_then(|i| config.models.get(i))
        .map(|m| m.effective_name())
        .unwrap_or_default();
    let task_rows = crate::server::training_tasks::task_rows(&tasks(state));
    let tasks_busy = task_rows.iter().any(|t| t.busy);
    let jobs_busy = tasks_busy || all.iter().any(|r| r.state.is_busy());
    let (select_metrics, select_hint) = select_choices(&config.training);
    TrainingTemplate {
        nav: "train",
        tab,
        tab_title: tab_title(tab),
        version: crate::VERSION,
        message,
        error,
        available: cap.available,
        summary: cap.summary(),
        labeler_device: cap.accelerator.as_ref().map_or_else(
            || "not available: needs an NVIDIA GPU or Apple silicon".to_string(),
            |a| format!("PyTorch on {}", a.torch_backend()),
        ),
        reasons: cap.reasons.clone(),
        env_ready: env.ready,
        env_note: env.note,
        env_download: env.download,
        gpu_choices: env.gpu_choices,
        other_env: env.other_env,
        allow_large: config.allow_large_downloads,
        import_large,
        autolabel_large,
        datasets: dataset_rows(&config),
        families,
        size_cols,
        notices,
        latencies,
        epochs: blank_or(config.training.epochs),
        patience: blank_or(config.training.patience),
        lr: blank_or(config.training.lr),
        lr_defaults: lr_defaults_text(),
        lr_schedules: lr_schedule_choices(config.training.lr_schedule),
        lr_schedule_hint: lr_schedule_hint_text(),
        lr_min_factor: blank_or(config.training.lr_min_factor),
        select_metrics,
        select_hint,
        scenes: scene_choices(),
        balance: BalanceView::of(config.training.balance.as_ref()),
        comparisons: compare_cards(&all),
        public_available: catalog::lila_night().is_some(),
        public_installed: crate::training::public_subset_installed(&config.data_root()),
        public_size: catalog::lila_night()
            .map(|r| catalog::format_size(r.size()))
            .unwrap_or_default(),
        extra_sets: EXTRA_SETS.to_vec(),
        bench_devices,
        bench_auto,
        runs: run_rows,
        results,
        models: names,
        default_model,
        revert: crate::training::switch::revert_view(&config),
        verifiers: serde_json::to_string(&crate::training::switch::verifiers(&config))
            .unwrap_or_default(),
        training_dir: crate::training::training_dir(&config).display().to_string(),
        training_dir_setting: config
            .training
            .dir
            .as_ref()
            .map(|d| d.display().to_string())
            .unwrap_or_default(),
        training_dir_default: crate::training::config::TrainingConfig::default()
            .dir_in(&config.data_root())
            .display()
            .to_string(),
        models_dir: paths.export.models.display().to_string(),
        tasks: task_rows,
        tasks_busy,
        jobs_busy,
        footage: crate::server::training_tasks::footage(),
        imports: recent_imports(&imports_dir),
        imports_dir: imports_dir.display().to_string(),
        dataset_dir: dataset_dir.display().to_string(),
    }
}

/// The environment line, the GPU choices and the other lanes' downloads ([`env_view`]).
#[derive(Default)]
struct EnvView {
    /// The training environment of the automatic pick's lane is set up.
    ready: bool,
    note: String,
    /// "the training environment (about 450 MB)": what the Train confirmation names; empty
    /// without an accelerator.
    download: String,
    gpu_choices: Vec<GpuChoice>,
    /// What a GPU on another PyTorch build than the automatic pick's would download, when not set
    /// up ([`TrainingTemplate::other_env`]).
    other_env: Option<String>,
}

/// The environment card of `cap` with its environments under `paths` on `platform`: whether the
/// automatic pick's environment is set up (else what its first run downloads), the GPU choices
/// (only with more than one usable GPU) and what a chosen GPU on another lane would download (a
/// chosen GPU on another lane sets up that lane's environment, which replaces the other one).
fn env_view(cap: &Capability, paths: &TrainPaths, platform: Platform) -> EnvView {
    let (ready, note) = match &cap.accelerator {
        Some(a) => {
            let ready = env::train_env_ready(paths, platform, a.lane());
            let (dl, disk) = a.lane().estimate();
            let note = if ready {
                format!(
                    "ready (PyTorch {}, {})",
                    a.lane().as_str(),
                    paths.export.env.display()
                )
            } else {
                format!(
                    "not set up: the first run downloads about {} (PyTorch {} and the trainers, \
                     {} on disk under {})",
                    catalog::format_size(dl),
                    a.lane().as_str(),
                    catalog::format_size(disk),
                    paths.export.tools.display()
                )
            };
            (ready, note)
        }
        None => (false, "not available on this machine".into()),
    };
    let download = cap.accelerator.as_ref().map_or_else(String::new, |a| {
        format!(
            "the training environment (about {})",
            catalog::format_size(a.lane().estimate().0)
        )
    });
    let gpu_choices: Vec<GpuChoice> = if cap.has_gpu_choice() {
        cap.cuda_options
            .iter()
            .map(|a| GpuChoice {
                value: a.device_arg(),
                label: a.choice_label(),
            })
            .collect()
    } else {
        Vec::new()
    };
    let other_env = {
        let auto_lane = cap.accelerator.as_ref().map(|a| a.lane());
        let mut parts: Vec<String> = Vec::new();
        for a in cap.cuda_options.iter().skip(1) {
            let lane = a.lane();
            if Some(lane) == auto_lane {
                continue;
            }
            let name = a.name();
            let tag = format!("PyTorch {}", lane.as_str());
            match parts.iter_mut().find(|p| p.starts_with(&tag)) {
                Some(p) => {
                    p.push_str(", ");
                    p.push_str(name);
                }
                None if !env::train_env_ready(paths, platform, lane) => parts.push(format!(
                    "{tag}, about {}, for {name}",
                    catalog::format_size(lane.estimate().0)
                )),
                None => {}
            }
        }
        (!parts.is_empty()).then(|| parts.join("; "))
    };
    EnvView {
        ready,
        note,
        download,
        gpu_choices,
        other_env,
    }
}

/// The family / size grid of `cap`, and its number of size columns (the most sizes of a family;
/// shorter rows are padded).
fn family_rows(cap: &Capability) -> (Vec<FamilyRow>, usize) {
    let size_cols = cap
        .families
        .iter()
        .map(|f| f.sizes.len())
        .max()
        .unwrap_or(0);
    let families = cap
        .families
        .iter()
        .map(|f| FamilyRow {
            title: f.title,
            license: f.license,
            available: f.available,
            why: f.why.clone().unwrap_or_default(),
            pad: size_cols.saturating_sub(f.sizes.len()),
            recommended: {
                let r = recipe::recommended(f.family);
                format!(
                    "{} ep \u{b7} patience {} \u{b7} lr {}",
                    r.epochs,
                    r.patience,
                    recipe::default_lr_label(f.family)
                )
            },
            default_lr: recipe::default_lr_label(f.family),
            sizes: f
                .sizes
                .iter()
                .map(|z| {
                    let (estimate, short) = match (z.vram_mb, z.epoch_s_per_100, z.suggested_batch)
                    {
                        (Some(v), Some(t), Some(b)) => {
                            let gb = v as f64 / 1024.0;
                            let t = estimate::format_duration(t);
                            (
                                format!("~{gb:.1} GB at batch {b}, ~{t} / epoch / 100 images"),
                                format!("{gb:.1} GB \u{b7} batch {b} \u{b7} {t}"),
                            )
                        }
                        _ => (String::new(), String::new()),
                    };
                    SizeCell {
                        label: f.family.base_name(z.size),
                        size: z.size,
                        fits: z.fits,
                        estimate,
                        short,
                        weight: latency_parts(z.latency).0,
                        latency: z.latency,
                        why: z.why.clone().unwrap_or_default(),
                    }
                })
                .collect(),
        })
        .collect();
    (families, size_cols)
}

/// The licence notices of `cap`'s available families, each once, with the families it covers.
fn notice_rows(cap: &Capability) -> Vec<NoticeRow> {
    let mut notices: Vec<NoticeRow> = Vec::new();
    for f in cap.families.iter().filter(|f| f.available) {
        let n = f.family.notice();
        match notices.iter_mut().find(|r| r.notice == n) {
            Some(r) => {
                r.families.push_str(", ");
                r.families.push_str(f.title);
            }
            None => notices.push(NoticeRow {
                families: f.title.to_string(),
                notice: n,
            }),
        }
    }
    notices
}

/// The latency legend of `cap`'s sizes: each class once, light, medium, then heavy.
fn latency_rows(cap: &Capability) -> Vec<LatencyRow> {
    let mut latencies: Vec<LatencyRow> = Vec::new();
    for z in cap.families.iter().flat_map(|f| f.sizes.iter()) {
        let (weight, text) = latency_parts(z.latency);
        if !latencies.iter().any(|l| l.weight == weight) {
            latencies.push(LatencyRow { weight, text });
        }
    }
    latencies.sort_by_key(|l| match l.weight {
        "light" => 0,
        "medium" => 1,
        _ => 2,
    });
    latencies
}

/// What an import and an auto-label run may need over the large-download gate
/// ([`TrainingTemplate::import_large`], [`TrainingTemplate::autolabel_large`]): None for both
/// without an accelerator or when large downloads are allowed.
fn large_downloads(
    cap: &Capability,
    allow_large: bool,
    env: &EnvView,
    paths: &TrainPaths,
) -> (Option<String>, Option<String>) {
    let large = |parts: Vec<String>| (!parts.is_empty()).then(|| parts.join("; "));
    match (&cap.accelerator, allow_large) {
        (Some(_), false) => {
            let env_part = (!env.ready).then(|| env.download.clone());
            let missing = |keys: &[&str]| {
                let list: Vec<_> = keys
                    .iter()
                    .filter_map(|k| catalog::autolabel_weights(k))
                    .chain(
                        keys.contains(&"rfdetr-large")
                            .then(|| catalog::base_weights(catalog::TrainFamily::Rfdetr, "large"))
                            .flatten(),
                    )
                    .map(|r| (r, crate::training::pyenv::weights_dir(paths, r)))
                    .collect();
                crate::training::pyenv::missing_bytes(&list)
            };
            let mut import = env_part.iter().cloned().collect::<Vec<_>>();
            let siglip = missing(&["siglip2-base"]);
            if siglip > 0 {
                import.push(format!(
                    "SigLIP 2 ({}) for the SigLIP selection",
                    catalog::format_size(siglip)
                ));
            }
            let mut label = env_part.into_iter().collect::<Vec<_>>();
            if let Some(o) = &env.other_env {
                label.push(format!(
                    "the training environment of another GPU if you choose it ({o})"
                ));
            }
            let defaults = missing(&[
                "mm-gdino-tiny",
                "rfdetr-large",
                "qwen3-vl-2b",
                "siglip2-base",
            ]);
            let others = missing(&[
                "mm-gdino-base",
                "mm-gdino-large",
                "qwen3-vl-4b",
                "qwen3-vl-8b",
            ]);
            // Never a reason for the box on its own: the page offers SAM 3 once it is installed.
            let sam3 = if missing(&[catalog::SAM3_KEY]) > 0 {
                "; SAM 3 has its own download under Advanced"
            } else {
                ""
            };
            if defaults > 0 {
                label.push(format!(
                    "the labeler weights ({} for the default set, more for larger sizes{sam3})",
                    catalog::format_size(defaults)
                ));
            } else if others > 0 {
                label.push(format!(
                    "the weights of a larger labeler size if you pick one{sam3}"
                ));
            }
            (large(import), large(label))
        }
        _ => (None, None),
    }
}

/// The dataset's warnings of `r` (none without a dataset report).
fn warnings_of(r: &RunRecord) -> Vec<String> {
    r.dataset
        .as_ref()
        .map_or_else(Vec::new, |d| d.warnings.clone())
}

/// The run queue: every run of `all` not done, newest first.
fn run_rows(all: &[RunRecord]) -> Vec<RunRow> {
    all.iter()
        .rev()
        .filter(|r| r.state != RunState::Done)
        .map(|r| RunRow {
            warnings: warnings_of(r),
            can_retry: retry::retry_checkpoint(r).is_ok(),
            retry_note: if retry::retry_checkpoint(r).is_ok() {
                retry::partial_note(r)
            } else {
                String::new()
            },
            files: RunFiles::of(r),
            id: r.id.clone(),
            name: r.name.clone(),
            model: r.request.recipe.label(),
            state: r.state.as_str(),
            busy: r.state.is_busy(),
            can_finish: can_finish(r),
            extend: extension(r).map_or_else(String::new, |n| n.text.clone()),
            extend_keep: extension(r).map_or(0, |n| n.keep_epoch),
            extend_no: extension(r).map_or(0, |n| n.extension),
            percent: r.percent,
            describe: r.describe(),
            dataset: r.dataset.as_ref().map_or_else(String::new, |d| {
                let mut t = format!(
                    "{} train / {} val / {} held out",
                    d.train_images, d.valid_images, d.heldout_images
                );
                if let Some(b) = &d.balance_summary {
                    t.push_str(&format!(" \u{b7} balanced: {b}"));
                }
                t
            }),
            log_tail: r
                .log
                .iter()
                .rev()
                .take(6)
                .rev()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n"),
        })
        .collect()
}

/// The finished runs of `all` that installed a model, newest first, for [`result_groups`]:
/// `names` are the configured models, `current` the default model's name.
fn result_items(all: &[RunRecord], names: &[String], current: &str) -> Vec<GroupItem> {
    all.iter()
        .rev()
        .filter(|r| r.state == RunState::Done)
        .filter_map(|r| {
            let o = r.output.as_ref()?;
            let keys = compare::best_keys(r.benchmark.as_ref());
            Some(GroupItem {
                key: heldout_key(r),
                benchmarked: r.benchmark.is_some(),
                keys,
                row: ResultRow {
                    person_r: recall_text(keys[0]),
                    vehicle_r: recall_text(keys[1]),
                    ap: ap_text(keys[2]),
                    best: 0,
                    id: r.id.clone(),
                    name: o.name.clone(),
                    family: format!("{} ({})", r.request.recipe.label(), o.family),
                    license: o.license.clone(),
                    classes: o.classes.join(", "),
                    merges: merges_text(&r.request.rename),
                    benchmark: bench_text(r.benchmark.as_ref(), r.benchmark_error.as_deref()),
                    compare_href: compare_href(r.benchmark.as_ref()),
                    compare: bench::compare_text(r.benchmark.as_ref(), r.baseline.as_ref()),
                    trained_on: r
                        .benchmark
                        .as_ref()
                        .and_then(|b| b.trained_on_share)
                        .and_then(bench::trained_chip)
                        .unwrap_or_default(),
                    is_default: names.iter().any(|n| n.eq_ignore_ascii_case(&o.name))
                        && o.name.eq_ignore_ascii_case(current),
                    finished: r
                        .finished
                        .as_deref()
                        .map(crate::clock::display_stored)
                        .unwrap_or_default(),
                    in_config: names.iter().any(|n| n.eq_ignore_ascii_case(&o.name)),
                    warnings: warnings_of(r),
                    confirm: RunFiles::of(r).confirm(r),
                    files: RunFiles::of(r),
                },
            })
        })
        .collect()
}

/// The Datasets list: `training.datasets` as saved, then the exports found under the training
/// folder that are not saved.
fn dataset_rows(config: &crate::config::Config) -> Vec<DatasetRow> {
    let mut v: Vec<DatasetRow> = config
        .training
        .datasets
        .iter()
        .map(|d| {
            let dir = config.data_path(d);
            DatasetRow::of(d.display().to_string(), &dir, dir.is_dir(), true)
        })
        .collect();
    for d in crate::training::discover_exports(config) {
        let path = d.display().to_string();
        if !v.iter().any(|r| r.path == path) {
            v.push(DatasetRow::of(path, &d, true, false));
        }
    }
    v
}

/// An optional setting as its field shows it: blank when unset.
fn blank_or<T: ToString>(v: Option<T>) -> String {
    v.map(|e| e.to_string()).unwrap_or_default()
}

/// The LR schedule select: `default` (`""`), then every schedule, `selected` marked.
fn lr_schedule_choices(selected: Option<recipe::LrSchedule>) -> Vec<LrScheduleChoice> {
    std::iter::once(("", "default"))
        .chain(
            recipe::LrSchedule::ALL
                .iter()
                .map(|s| (s.as_str(), s.as_str())),
        )
        .map(|(value, title)| LrScheduleChoice {
            value,
            title,
            selected: selected.map_or("", recipe::LrSchedule::as_str) == value,
        })
        .collect()
}

/// The "Select best by" select ([`recipe::SELECT_METRICS`]) with `training`'s metric selected,
/// and the selected one's hint.
fn select_choices(
    training: &crate::training::config::TrainingConfig,
) -> (Vec<SelectChoice>, &'static str) {
    let metric = training.select_metric_for(None);
    let choices = recipe::SELECT_METRICS
        .iter()
        .map(|o| SelectChoice {
            value: o.metric.id(),
            title: o.label,
            hint: o.hint,
            selected: metric == o.metric,
        })
        .collect();
    let hint = recipe::SELECT_METRICS
        .iter()
        .find(|o| metric == o.metric)
        .map_or("", |o| o.hint);
    (choices, hint)
}

/// The Runs checkboxes ([`SCENE_CHOICES`]).
fn scene_choices() -> Vec<SceneChoice> {
    SCENE_CHOICES
        .iter()
        .map(|(s, title, preset)| SceneChoice {
            value: s.as_str(),
            title,
            describe: s.describe(),
            prefix: s.label(),
            preset: *preset,
        })
        .collect()
}

/// The ten newest imports listed in `dir`, newest first.
fn recent_imports(dir: &std::path::Path) -> Vec<crate::training::import::write::ImportLine> {
    let mut v = crate::training::import::write::read_imports(dir);
    v.reverse();
    v.truncate(10);
    v
}

/// "merged: truck -> car; dropped: tree" for the Models tab; empty without merges.
pub(super) fn merges_text(rename: &BTreeMap<String, String>) -> String {
    let merged: Vec<String> = rename
        .iter()
        .filter(|(_, to)| !to.is_empty())
        .map(|(from, to)| format!("{from} \u{2192} {to}"))
        .collect();
    let dropped: Vec<&str> = rename
        .iter()
        .filter(|(_, to)| to.is_empty())
        .map(|(from, _)| from.as_str())
        .collect();
    let mut parts = Vec::new();
    if !merged.is_empty() {
        parts.push(format!("merged: {}", merged.join(", ")));
    }
    if !dropped.is_empty() {
        parts.push(format!("dropped: {}", dropped.join(", ")));
    }
    parts.join("; ")
}

/// One family of each trainer, with the families it stands for (hint texts).
const LR_GROUPS: [(catalog::TrainFamily, &str); 3] = [
    (catalog::TrainFamily::Yolo26, "YOLO"),
    (catalog::TrainFamily::Rfdetr, "RF-DETR"),
    (catalog::TrainFamily::Rtdetrv2, "RT-DETRv2 / D-FINE"),
];

/// "YOLO: auto (optimizer picks); RF-DETR: 1e-4 (5e-5 under 1000 unique frames); RT-DETRv2 /
/// D-FINE: 1e-4".
fn lr_defaults_text() -> String {
    LR_GROUPS
        .iter()
        .map(|(f, t)| format!("{t}: {}", recipe::default_lr_label(*f)))
        .collect::<Vec<_>>()
        .join("; ")
}

/// "YOLO: linear (default) or cosine, final factor 0.01; ...".
fn lr_schedule_hint_text() -> String {
    LR_GROUPS
        .iter()
        .map(|(f, t)| {
            let s = recipe::lr_schedules(*f);
            let others: Vec<&str> = s
                .allowed
                .iter()
                .filter(|x| **x != s.default)
                .map(|x| x.as_str())
                .collect();
            format!(
                "{t}: {} (default) or {}, final factor {}",
                s.default,
                others.join(" or "),
                s.min_factor
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::detect::{
        ComputeCapability, CudaInfo, GpuAdapter, GpuVendor, HardwareInfo, apple_gpu,
    };
    use crate::training::capability::capability;

    fn nvidia(name: &str, cc: (u32, u32), vram_mb: u64, ordinal: u32) -> GpuAdapter {
        GpuAdapter {
            vendor: GpuVendor::Nvidia,
            name: name.into(),
            vram_mb,
            index: ordinal,
            discrete: true,
            cuda: Some(CudaInfo {
                ordinal,
                compute_capability: ComputeCapability::new(cc.0, cc.1),
            }),
        }
    }

    /// An RTX 5090 (cu130, the automatic pick) and two cu126 cards, on a CUDA 13 driver.
    fn two_lanes() -> Capability {
        capability(
            &HardwareInfo::new(
                "windows",
                "x86_64",
                vec![
                    nvidia("RTX 5090", (12, 0), 32768, 0),
                    nvidia("RTX 3090", (8, 6), 24576, 1),
                    nvidia("GTX 1060", (6, 1), 6144, 2),
                ],
            )
            .with_cuda_driver(13, 0),
            Some(32.0),
        )
    }

    fn mac() -> Capability {
        capability(
            &HardwareInfo::new("macos", "aarch64", vec![apple_gpu("Apple M1")]),
            Some(16.0),
        )
    }

    fn paths(root: &std::path::Path) -> TrainPaths {
        TrainPaths::new(root, root.join("models"))
    }

    #[test]
    fn env_view_without_an_accelerator_offers_nothing() {
        let root = crate::test_dirs::TempDir::new("train-view-env-none");
        let cap = capability(&HardwareInfo::new("linux", "x86_64", vec![]), None);
        let v = env_view(&cap, &paths(&root), Platform::current());
        assert!(!v.ready);
        assert_eq!(v.note, "not available on this machine");
        assert_eq!(v.download, "");
        assert!(v.gpu_choices.is_empty());
        assert_eq!(v.other_env, None);
        assert_eq!(
            large_downloads(&cap, false, &EnvView::default(), &paths(&root)),
            (None, None)
        );
    }

    #[test]
    fn env_view_names_the_download_and_the_other_lanes() {
        let root = crate::test_dirs::TempDir::new("train-view-env-lanes");
        let p = paths(&root);
        let cap = two_lanes();
        let v = env_view(&cap, &p, Platform::current());
        let auto = cap.accelerator.as_ref().unwrap().lane();
        let (dl, disk) = auto.estimate();
        assert!(!v.ready);
        assert_eq!(
            v.note,
            format!(
                "not set up: the first run downloads about {} (PyTorch {} and the trainers, {} on \
                 disk under {})",
                catalog::format_size(dl),
                auto.as_str(),
                catalog::format_size(disk),
                p.export.tools.display()
            )
        );
        assert_eq!(
            v.download,
            format!(
                "the training environment (about {})",
                catalog::format_size(dl)
            )
        );
        let values: Vec<&str> = v.gpu_choices.iter().map(|g| g.value.as_str()).collect();
        assert_eq!(values, ["cuda:0", "cuda:1", "cuda:2"]);
        // Both cu126 cards share one entry; the automatic pick's lane is not listed.
        let other = cap.cuda_options[1].lane();
        assert_ne!(other, auto);
        assert_eq!(
            v.other_env.as_deref(),
            Some(
                format!(
                    "PyTorch {}, about {}, for RTX 3090, GTX 1060",
                    other.as_str(),
                    catalog::format_size(other.estimate().0)
                )
                .as_str()
            )
        );
        // Large downloads allowed: no confirmation.
        assert_eq!(large_downloads(&cap, true, &v, &p), (None, None));
        // Not allowed: the import names the environment, the auto-label run the other GPU too.
        let (import, label) = large_downloads(&cap, false, &v, &p);
        assert!(import.unwrap().starts_with(&v.download));
        let label = label.unwrap();
        assert!(label.starts_with(&v.download), "{label}");
        assert!(
            label.contains("the training environment of another GPU if you choose it (PyTorch"),
            "{label}"
        );
    }

    #[test]
    fn one_gpu_has_no_choices_and_no_other_lane() {
        let root = crate::test_dirs::TempDir::new("train-view-env-mac");
        let v = env_view(&mac(), &paths(&root), Platform::current());
        assert!(v.gpu_choices.is_empty());
        assert_eq!(v.other_env, None);
        assert!(v.note.starts_with("not set up: "), "{}", v.note);
    }

    #[test]
    fn family_rows_pad_to_the_widest_family() {
        let cap = mac();
        let (rows, cols) = family_rows(&cap);
        assert_eq!(rows.len(), cap.families.len());
        assert_eq!(
            cols,
            cap.families.iter().map(|f| f.sizes.len()).max().unwrap()
        );
        for (row, f) in rows.iter().zip(&cap.families) {
            assert_eq!(row.title, f.title);
            assert_eq!(row.sizes.len() + row.pad, cols);
            assert_eq!(row.default_lr, recipe::default_lr_label(f.family));
            let r = recipe::recommended(f.family);
            assert_eq!(
                row.recommended,
                format!(
                    "{} ep \u{b7} patience {} \u{b7} lr {}",
                    r.epochs, r.patience, row.default_lr
                )
            );
            for (cell, z) in row.sizes.iter().zip(&f.sizes) {
                assert_eq!(cell.label, f.family.base_name(z.size));
                assert_eq!(cell.weight, latency_parts(z.latency).0);
                // An estimate is shown with its short form, or neither.
                assert_eq!(cell.estimate.is_empty(), cell.short.is_empty());
            }
        }
    }

    #[test]
    fn notices_and_latencies_are_listed_once_in_order() {
        let cap = mac();
        let notices = notice_rows(&cap);
        let mut seen = Vec::new();
        for n in &notices {
            assert!(!seen.contains(&n.notice), "{}", n.notice);
            seen.push(n.notice);
        }
        for f in cap.families.iter().filter(|f| f.available) {
            let row = notices
                .iter()
                .find(|n| n.notice == f.family.notice())
                .unwrap();
            assert!(row.families.split(", ").any(|t| t == f.title));
        }
        let weights: Vec<&str> = latency_rows(&cap).iter().map(|l| l.weight).collect();
        let rank = |w: &str| match w {
            "light" => 0,
            "medium" => 1,
            _ => 2,
        };
        assert!(
            weights.windows(2).all(|w| rank(w[0]) < rank(w[1])),
            "{weights:?}"
        );
    }

    #[test]
    fn run_rows_and_result_items_split_the_records_newest_first() {
        use crate::server::training::tests::record;
        let mut done = record(
            "d1",
            SceneFilter::DAY,
            None,
            RunState::Done,
            Some((0.5, 0.4, 0.3)),
        );
        done.output = Some(crate::training::job::TrainOutput {
            name: "Mine".into(),
            onnx: "Mine.onnx".into(),
            yaml: "Mine.yaml".into(),
            provenance: "Mine.provenance.json".into(),
            family: crate::model::ModelFamilyKind::Yolo26,
            classes: vec!["person".into(), "car".into()],
            license: "AGPL-3.0".into(),
            heldout: None,
            device: "mps".into(),
            train_seconds: 1.0,
        });
        let no_model = record("d2", SceneFilter::DAY, None, RunState::Done, None);
        let mut failed = record("f1", SceneFilter::NIGHT_IR, None, RunState::Failed, None);
        failed.log = (1..=8).map(|i| format!("line {i}")).collect();
        let queued = record("q1", SceneFilter::DAY, None, RunState::Queued, None);
        let all = vec![done, failed, no_model, queued];
        let rows = run_rows(&all);
        let ids: Vec<&str> = rows.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["q1", "f1"]);
        assert_eq!(
            rows[1].log_tail,
            "line 3\nline 4\nline 5\nline 6\nline 7\nline 8"
        );
        assert!(rows[0].busy && !rows[1].busy);
        let names = vec!["mine".to_string(), "other".to_string()];
        let items = result_items(&all, &names, "MINE");
        assert_eq!(
            items.len(),
            1,
            "a finished run without a model is no result"
        );
        let row = &items[0].row;
        assert_eq!(row.id, "d1");
        assert_eq!(row.classes, "person, car");
        assert!(row.in_config && row.is_default);
        // AP50-95 on every held-out frame (the `all` scene: (0.5 + 0.4 + 0.3) / 3).
        assert_eq!(row.ap, "0.400");
        assert!(items[0].benchmarked);
        let not_default = result_items(&all, &names, "other");
        assert!(not_default[0].row.in_config && !not_default[0].row.is_default);
        assert!(!result_items(&all, &[], "Mine")[0].row.in_config);
    }

    #[test]
    fn form_defaults_show_blank_for_unset() {
        assert_eq!(blank_or::<u32>(None), "");
        assert_eq!(blank_or(Some(0.0005_f32)), "0.0005");
        let choices = lr_schedule_choices(None);
        assert_eq!(choices[0].value, "");
        assert_eq!(choices.iter().filter(|c| c.selected).count(), 1);
        assert!(choices[0].selected);
        let s = recipe::LrSchedule::ALL[0];
        let choices = lr_schedule_choices(Some(s));
        let picked: Vec<&str> = choices
            .iter()
            .filter(|c| c.selected)
            .map(|c| c.value)
            .collect();
        assert_eq!(picked, [s.as_str()]);
        let (metrics, hint) = select_choices(&Default::default());
        assert_eq!(metrics.iter().filter(|m| m.selected).count(), 1);
        assert_eq!(metrics.iter().find(|m| m.selected).unwrap().hint, hint);
        let scenes = scene_choices();
        assert_eq!(scenes.len(), SCENE_CHOICES.len());
        assert_eq!(scenes[0].value, "all");
    }
}
