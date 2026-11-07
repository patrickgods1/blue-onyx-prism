//! The training script's progress events (`@train {json}` lines on its stdout, see
//! `scripts/train_model.py`) and the run's progress derived from them ([`TrainProgress`]:
//! percent of the training stage, epoch, ETA, every scalar metric per epoch in
//! [`TrainProgress::history`], the best epoch and the convergence stop, whose rule
//! [`convergence_run`] mirrors the script's `Convergence`, with its smoothing window and the
//! extensions while the training loss still falls ([`loss_falling`], [`ConvergePoint`]; the page's
//! Stop now / Continue notice is [`TrainProgress::extension`]) and the anneal before the stop
//! ([`Anneal`]: a lowered learning rate for a few epochs, then the stop). Raw metric keys of the trainers (rfdetr,
//! Ultralytics, transformers) map to stable ids ([`canonical`], described by [`METRICS`]);
//! unknown keys are kept under their raw name. Pure.

use super::recipe::SelectMetric;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::BTreeMap;

/// Marker before every event.
pub const MARKER: &str = "@train ";

/// One event of the training script.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TrainEvent {
    /// `--self-check`: what PyTorch sees.
    SelfCheck {
        #[serde(default)]
        cuda: bool,
        #[serde(default)]
        mps: bool,
        #[serde(default)]
        cuda_ok: Option<bool>,
        #[serde(default)]
        torch: Option<String>,
        #[serde(default)]
        devices: Vec<serde_json::Value>,
    },
    Start {
        #[serde(default)]
        device: String,
        #[serde(default)]
        train_images: usize,
        #[serde(default)]
        valid_images: usize,
        #[serde(default)]
        epochs: u32,
        /// The input size the trainer was given.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        imgsz: Option<u32>,
        /// The batch size the trainer was given (after an out-of-memory retry, the halved one).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        batch: Option<u32>,
    },
    Batch {
        epoch: u32,
        epochs: u32,
        batch: u32,
        batches: u32,
        /// The main parameter group's learning rate after this batch (None: unreadable, or an
        /// older script).
        #[serde(
            default,
            deserialize_with = "finite_opt",
            skip_serializing_if = "Option::is_none"
        )]
        lr: Option<f64>,
    },
    Epoch {
        epoch: u32,
        epochs: u32,
        #[serde(default)]
        loss: Option<f64>,
        #[serde(default)]
        map50: Option<f64>,
        #[serde(default)]
        map: Option<f64>,
        #[serde(default)]
        elapsed_s: Option<f64>,
        /// Every scalar metric of the epoch, raw key -> value ([`canonical`] maps the keys).
        #[serde(
            default,
            deserialize_with = "finite_metrics",
            skip_serializing_if = "BTreeMap::is_empty"
        )]
        metrics: BTreeMap<String, f64>,
    },
    Eval {
        #[serde(default)]
        epoch: u32,
        #[serde(default)]
        val_loss: Option<f64>,
        #[serde(
            default,
            deserialize_with = "finite_metrics",
            skip_serializing_if = "BTreeMap::is_empty"
        )]
        metrics: BTreeMap<String, f64>,
    },
    Phase {
        phase: String,
        #[serde(default)]
        detail: String,
    },
    /// The convergence rule (`converged`) or a finish request (`finish`, the stop file) ended
    /// training after `epoch`; the best checkpoint is exported.
    Stop {
        reason: String,
        epoch: u32,
        #[serde(default)]
        best_epoch: u32,
        #[serde(default)]
        best: Option<f64>,
    },
    /// The rule reached a stop point while the training loss still fell: training goes on.
    ConvergePoint(ConvergePoint),
    /// The rule reached a stop point and anneals: a lowered learning rate until `until`, then
    /// the stop.
    Anneal(Anneal),
    /// The selected metric's source failed: the run goes on by mAP.
    SelectFallback(SelectFallback),
    Done {
        onnx: String,
        yaml: String,
        family: String,
        #[serde(default)]
        classes: Vec<String>,
        #[serde(default)]
        imgsz: u32,
        #[serde(default)]
        seconds: Option<f64>,
        #[serde(default)]
        device: Option<String>,
    },
}

/// The event in `line`, if it carries one (anywhere in the line: a progress bar may precede it
/// on the same line). Unknown events and bad JSON are None.
pub fn parse_line(line: &str) -> Option<TrainEvent> {
    let at = line.find(MARKER)?;
    let json = line.get(at + MARKER.len()..)?.trim();
    serde_json::from_str(json).ok()
}

/// Longest metric key kept.
pub const MAX_METRIC_KEY: usize = 64;
/// Most metrics kept per epoch (known ids first, then raw keys in order).
pub const MAX_METRICS_PER_EPOCH: usize = 64;
/// Most epochs kept in [`TrainProgress::history`] (the oldest go first).
pub const MAX_HISTORY: usize = super::config::MAX_EPOCHS as usize;

/// How the page prints a metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricFormat {
    /// Three decimals (`0.711`).
    Fixed3,
    /// Scientific (`1.0e-4`), learning rates.
    Sci,
    /// Percent of 0..1 (`71.1%`).
    Pct,
    /// Integer.
    Int,
    /// Gigabytes, one decimal.
    Gb,
    /// Per second, one decimal.
    Rate,
}

/// One canonical metric id (`GET /v1/training` `metrics`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MetricInfo {
    pub id: &'static str,
    /// "mAP 50-95".
    pub label: &'static str,
    /// Column header: "mAP".
    pub short: &'static str,
    /// Larger is better (true), smaller (false), neither (None: learning rate, memory).
    pub higher_better: Option<bool>,
    pub format: MetricFormat,
}

const fn m(
    id: &'static str,
    label: &'static str,
    short: &'static str,
    higher_better: Option<bool>,
    format: MetricFormat,
) -> MetricInfo {
    MetricInfo {
        id,
        label,
        short,
        higher_better,
        format,
    }
}

/// Every canonical metric id, in the page's default order.
pub const METRICS: &[MetricInfo] = {
    use MetricFormat::*;
    const UP: Option<bool> = Some(true);
    const DOWN: Option<bool> = Some(false);
    &[
        m("map", "mAP 50-95", "mAP", UP, Fixed3),
        m("map50", "mAP 50", "mAP50", UP, Fixed3),
        m("map75", "mAP 75", "mAP75", UP, Fixed3),
        m("mar", "mAR (max detections)", "mAR", UP, Fixed3),
        m("iou", "Mean IoU of matched boxes", "IoU", UP, Fixed3),
        m(
            "precision",
            "Precision (at the best F1 threshold)",
            "P",
            UP,
            Fixed3,
        ),
        m(
            "recall",
            "Recall (at the best F1 threshold)",
            "R",
            UP,
            Fixed3,
        ),
        m("f1", "F1 (macro, best threshold)", "F1", UP, Fixed3),
        m(
            "f1_thr",
            "Best F1 confidence threshold",
            "F1 thr",
            None,
            Fixed3,
        ),
        m(
            "f2",
            "F2 (recall-weighted, best threshold)",
            "F2",
            UP,
            Fixed3,
        ),
        m(
            "f2_thr",
            "Best F2 confidence threshold",
            "F2 thr",
            None,
            Fixed3,
        ),
        m(
            "f2_precision",
            "Precision at the best F2 threshold",
            "F2 P",
            UP,
            Fixed3,
        ),
        m(
            "f2_recall",
            "Recall at the best F2 threshold",
            "F2 R",
            UP,
            Fixed3,
        ),
        m(
            "f0_5",
            "F0.5 (precision-weighted, best threshold)",
            "F0.5",
            UP,
            Fixed3,
        ),
        m(
            "f0_5_thr",
            "Best F0.5 confidence threshold",
            "F0.5 thr",
            None,
            Fixed3,
        ),
        m(
            "f0_5_precision",
            "Precision at the best F0.5 threshold",
            "F0.5 P",
            UP,
            Fixed3,
        ),
        m(
            "f0_5_recall",
            "Recall at the best F0.5 threshold",
            "F0.5 R",
            UP,
            Fixed3,
        ),
        m("loss", "Training loss", "Loss", DOWN, Fixed3),
        m("val_loss", "Validation loss", "Val loss", DOWN, Fixed3),
        m("lr", "Learning rate", "LR", None, Sci),
        m("mem_gb", "Peak GPU memory (GB)", "Mem", None, Gb),
        m("img_s", "Training images per second", "img/s", UP, Rate),
        m("ema_map", "mAP 50-95 (EMA)", "EMA mAP", UP, Fixed3),
        m("ema_map50", "mAP 50 (EMA)", "EMA mAP50", UP, Fixed3),
        m("ema_map75", "mAP 75 (EMA)", "EMA mAP75", UP, Fixed3),
        m("ema_mar", "mAR (EMA)", "EMA mAR", UP, Fixed3),
    ]
};

/// The [`MetricInfo`] of `id`.
pub fn metric(id: &str) -> Option<&'static MetricInfo> {
    METRICS.iter().find(|m| m.id == id)
}

/// The stable id of a trainer's raw metric key (rfdetr `val/mAP_50_95`, Ultralytics
/// `metrics/mAP50-95(B)`, transformers `eval_loss`, the script's own `f2`), None for keys
/// without one (they are kept under their raw name). rfdetr's `val/ema_*` keys map to `ema_<id>`.
pub fn canonical(raw: &str) -> Option<&'static str> {
    if let Some(rest) = raw.strip_prefix("val/ema_") {
        let id = canonical(&format!("val/{rest}"))?;
        return METRICS
            .iter()
            .find(|m| m.id.strip_prefix("ema_") == Some(id))
            .map(|m| m.id);
    }
    if let Some(m) = metric(raw) {
        return Some(m.id);
    }
    Some(match raw {
        // rfdetr (Lightning callback metrics).
        "val/mAP_50_95" => "map",
        "val/mAP_50" => "map50",
        "val/mAP_75" => "map75",
        "val/mAR" => "mar",
        "val/F1" => "f1",
        "val/precision" => "precision",
        "val/recall" => "recall",
        "val/loss" => "val_loss",
        "train/loss" | "train/loss_epoch" => "loss",
        "train/lr" => "lr",
        // Ultralytics.
        "metrics/mAP50-95(B)" => "map",
        "metrics/mAP50(B)" => "map50",
        "metrics/mAP75(B)" => "map75",
        "metrics/precision(B)" => "precision",
        "metrics/recall(B)" => "recall",
        "lr/pg0" => "lr",
        // transformers (Trainer logs and the script's compute_metrics).
        "eval_loss" => "val_loss",
        "learning_rate" => "lr",
        "eval_map" => "map",
        "eval_map_50" => "map50",
        "eval_map_75" => "map75",
        "eval_mar_100" => "mar",
        "eval_precision" => "precision",
        "eval_recall" => "recall",
        "eval_f1" => "f1",
        "eval_f1_thr" => "f1_thr",
        "eval_f2" => "f2",
        "eval_f2_thr" => "f2_thr",
        "eval_f0_5" => "f0_5",
        _ => return None,
    })
}

/// Counters a trainer logs beside its metrics (transformers' `epoch`, Lightning's `step`); the
/// history keys points by epoch already.
const COUNTER_KEYS: &[&str] = &["epoch", "step", "global_step", "eval_epoch"];

/// A finite number, else None (a `null`, a string or a NaN does not lose the event).
fn finite_opt<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
    let v: Option<serde_json::Value> = Option::deserialize(d)?;
    Ok(v.and_then(|v| v.as_f64()).filter(|f| f.is_finite()))
}

/// A JSON object of metrics, keeping finite numbers only (a `null`, a string or a NaN the
/// script let through does not lose the whole event) and no [`COUNTER_KEYS`].
fn finite_metrics<'de, D: Deserializer<'de>>(d: D) -> Result<BTreeMap<String, f64>, D::Error> {
    let raw: Option<BTreeMap<String, serde_json::Value>> = Option::deserialize(d)?;
    Ok(raw
        .unwrap_or_default()
        .into_iter()
        .filter(|(k, _)| !COUNTER_KEYS.contains(&k.as_str()))
        .filter_map(|(k, v)| v.as_f64().filter(|f| f.is_finite()).map(|f| (k, f)))
        .collect())
}

/// One finished epoch's metrics (canonical ids where known, raw keys otherwise).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct EpochPoint {
    pub epoch: u32,
    /// Seconds since training started when the epoch was reported.
    pub elapsed_s: f64,
    pub metrics: BTreeMap<String, f64>,
}

/// The best epoch on the selected metric ([`SelectMetric`]), by default the primary one (`map`,
/// else `map50`, else the lowest `val_loss`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Best {
    pub metric: String,
    pub epoch: u32,
    pub value: f64,
}

/// A best epoch's metric as the status lines name it.
fn best_label(id: &str) -> &str {
    match id {
        "map" => "mAP50-95",
        "map50" => "mAP50",
        "val_loss" => "val loss",
        other => metric(other).map_or(other, |m| m.short),
    }
}

/// The primary metrics, in order of preference, with "larger is better".
const PRIMARY: [(&str, bool); 3] = [("map", true), ("map50", true), ("val_loss", false)];

/// The best epoch of `history` on `select` (see [`Best`]); the earliest epoch wins a tie.
pub fn best_of(history: &[EpochPoint], select: SelectMetric) -> Option<Best> {
    if !select.is_default() {
        let mut best: Option<Best> = None;
        for (epoch, v) in rule_values(history, select) {
            let Some(v) = v else { continue };
            if best.as_ref().is_none_or(|b| v > b.value) {
                best = Some(Best {
                    metric: select.id().to_string(),
                    epoch,
                    value: v,
                });
            }
        }
        if best.is_some() {
            return best;
        }
    }
    PRIMARY.iter().find_map(|&(id, up)| {
        let mut best: Option<Best> = None;
        for p in history {
            let Some(&v) = p.metrics.get(id) else {
                continue;
            };
            let better = best
                .as_ref()
                .is_none_or(|b| if up { v > b.value } else { v < b.value });
            if better {
                best = Some(Best {
                    metric: id.to_string(),
                    epoch: p.epoch,
                    value: v,
                });
            }
        }
        best
    })
}

/// The value of the convergence rule and the best epoch for each row of `history`, as the
/// script's `pick_metric`: `map` (`map50` when no epoch has it), `map50`, or the macro F / mean
/// IoU, which a validated epoch without one (nothing matched) counts as 0.
pub fn rule_values(history: &[EpochPoint], select: SelectMetric) -> Vec<(u32, Option<f64>)> {
    let map = select == SelectMetric::Map;
    let id = if map && !history.iter().any(|p| p.metrics.contains_key("map")) {
        "map50"
    } else {
        select.id()
    };
    let zero = !matches!(select, SelectMetric::Map | SelectMetric::Map50);
    history
        .iter()
        .map(|p| {
            let validated = p.metrics.contains_key("map") || p.metrics.contains_key("map50");
            let v = p.metrics.get(id).copied();
            (p.epoch, v.or((zero && validated).then_some(0.0)))
        })
        .collect()
}

/// Why training ended before its last epoch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stop {
    /// `converged` (no improvement for the patience) or `finish` (asked to finish now).
    pub reason: String,
    pub epoch: u32,
    pub best_epoch: u32,
}

/// The script's `converge_point` event: the rule reached a stop point at `epoch` while the
/// training loss still fell, so the run goes on to `next_check` (the stop point after it).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ConvergePoint {
    pub epoch: u32,
    /// The epoch whose checkpoint a stop now keeps (highest raw value of the selected metric).
    pub best_epoch: u32,
    pub best: Option<f64>,
    pub loss_falling: bool,
    /// Mean training loss of the earlier / later half of the patience window ([`loss_halves`]).
    pub loss_from: Option<f64>,
    pub loss_to: Option<f64>,
    /// 1 for the first extension, at most [`MAX_EXTENSIONS`].
    pub extension: u32,
    pub next_check: u32,
}

/// The script's `anneal` event: the rule reached a stop point at `epoch` (no extension left
/// or the loss no longer falling) and trains on with a lowered learning rate until `until`, where
/// it stops.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Anneal {
    pub epoch: u32,
    pub until: u32,
    /// The epoch whose checkpoint a stop now keeps (highest raw value so far).
    pub best_epoch: u32,
    pub best: Option<f64>,
    pub loss_falling: bool,
    pub extensions: u32,
}

/// The script's `select_fallback` event: `metric`'s source failed at `epoch`, so from there the
/// run selects by mAP; the checkpoint and the rule start again at that epoch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SelectFallback {
    pub metric: String,
    pub epoch: u32,
    pub reason: String,
}

/// Extensions past a stop point while the training loss still falls (the script's
/// `MAX_EXTENSIONS`).
pub const MAX_EXTENSIONS: u32 = 2;
/// Relative drop of the mean training loss that counts as still falling (`LOSS_FALL`).
pub const LOSS_FALL: f64 = 0.02;

/// Left-to-right sum / count, as the script's `mean` (no compensated summation).
fn mean(v: &[f64]) -> f64 {
    let mut s = 0.0;
    for x in v {
        s += x;
    }
    s / v.len() as f64
}

/// Mean of the earlier and of the later `n / 2` finite values of `losses` (an odd middle value
/// counts in neither); None below 4 values. The script's `loss_halves`.
pub fn loss_halves(losses: &[Option<f64>]) -> Option<(f64, f64)> {
    let v: Vec<f64> = losses
        .iter()
        .flatten()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    let half = v.len() / 2;
    if half < 2 {
        return None;
    }
    Some((mean(v.get(..half)?), mean(v.get(v.len() - half..)?)))
}

/// Whether the training loss still falls over `losses` (in epoch order): the later half's mean
/// is below the earlier half's by more than [`LOSS_FALL`] of it ([`loss_halves`]; too few
/// values: false). The twin of the script's `loss_falling` (parity-tested).
pub fn loss_falling(losses: &[Option<f64>]) -> bool {
    loss_halves(losses).is_some_and(|(early, late)| late < early - LOSS_FALL * early.abs())
}

/// The convergence rule's settings (the script's `Convergence` arguments).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Rule {
    /// Epochs without a gain before a stop point (0 = never).
    pub patience: u32,
    /// No stop point before this epoch.
    pub min_epochs: u32,
    /// Smallest gain of the smoothed value that counts.
    pub min_delta: f64,
    /// The gain is measured on the mean of the last `smooth` values (0 and 1 = the raw value).
    pub smooth: u32,
    /// The run's epochs (0 = unknown): no extension at the last one.
    pub epochs: u32,
    /// Epochs trained with a lowered learning rate after the first stop point that would stop
    /// (at most up to `epochs`; 0 = stop there).
    pub anneal: u32,
}

/// Where [`convergence_run`] stopped and the stop points it went past.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ConvergenceRun {
    pub stop: Option<u32>,
    /// Epochs of the `converge_point`s (loss still falling: extended).
    pub points: Vec<u32>,
    /// The stop point that started the anneal (the `anneal` event's epoch).
    pub anneal: Option<u32>,
}

/// The convergence rule over `(epoch, value of the selected metric ([`rule_values`]), training
/// loss)` in epoch order (a repeated or older epoch is ignored): a stop point once `epoch >=
/// min_epochs` and `patience` epochs passed since the last gain of more than `min_delta` in the
/// mean of the last `smooth` values (or since the last extension). At a stop point the run goes
/// on while the loss still falls over the last `patience` epochs ([`loss_falling`]), at most
/// [`MAX_EXTENSIONS`] times and before its last epoch; else, with `anneal` > 0 and before its
/// last epoch, it anneals until `epoch + anneal` (at most `epochs`; later stop points are
/// ignored) and stops there; else it stops. The twin of the script's `Convergence`
/// (parity-tested).
pub fn convergence_run(epochs: &[(u32, Option<f64>, Option<f64>)], rule: &Rule) -> ConvergenceRun {
    let mut t = Tracker::new(rule.min_delta, rule.smooth);
    let mut losses: Vec<Option<f64>> = Vec::new();
    let (mut restart, mut out) = (0, ConvergenceRun::default());
    let mut anneal_until = 0;
    for &(epoch, v, loss) in epochs {
        if t.update(epoch, v).is_none() {
            continue;
        }
        losses.push(loss.filter(|l| l.is_finite()));
        if anneal_until > 0 {
            if epoch >= anneal_until {
                out.stop = Some(epoch);
                break;
            }
            continue;
        }
        if rule.patience == 0
            || t.best.is_none()
            || epoch < rule.min_epochs
            || epoch - t.best_epoch.max(restart) < rule.patience
        {
            continue;
        }
        let recent = losses
            .get(losses.len().saturating_sub(rule.patience as usize)..)
            .unwrap_or_default();
        let room = rule.epochs == 0 || epoch < rule.epochs;
        if loss_falling(recent) && out.points.len() < MAX_EXTENSIONS as usize && room {
            restart = epoch;
            out.points.push(epoch);
        } else if rule.anneal > 0 && room {
            let until = epoch.saturating_add(rule.anneal);
            anneal_until = if rule.epochs == 0 {
                until
            } else {
                until.min(rule.epochs)
            };
            out.anneal = Some(epoch);
        } else {
            out.stop = Some(epoch);
            break;
        }
    }
    out
}

/// The epoch [`convergence_run`] stops at without training losses (so without extensions) for
/// the per-epoch `values`, after an anneal of `anneal` epochs; None when it does not stop.
pub fn convergence_decision(
    values: &[(u32, f64)],
    patience: u32,
    min_epochs: u32,
    min_delta: f64,
    smooth: u32,
    anneal: u32,
) -> Option<u32> {
    let epochs: Vec<_> = values.iter().map(|&(e, v)| (e, Some(v), None)).collect();
    let rule = Rule {
        patience,
        min_epochs,
        min_delta,
        smooth,
        epochs: 0,
        anneal,
    };
    convergence_run(&epochs, &rule).stop
}

/// The rule's state: the last epoch seen, the last gain of the smoothed value and the highest
/// raw value.
#[derive(Debug, Default)]
struct Tracker {
    last: u32,
    best: Option<f64>,
    best_epoch: u32,
    top: Option<f64>,
    top_epoch: u32,
    min_delta: f64,
    smooth: usize,
    window: Vec<f64>,
}

impl Tracker {
    fn new(min_delta: f64, smooth: u32) -> Self {
        Self {
            min_delta,
            smooth: usize::try_from(smooth.max(1)).unwrap_or(1),
            ..Self::default()
        }
    }

    /// Fold one epoch in; None for a repeated or older epoch.
    fn update(&mut self, epoch: u32, v: Option<f64>) -> Option<()> {
        if epoch <= self.last {
            return None;
        }
        self.last = epoch;
        if let Some(v) = v.filter(|v| v.is_finite()) {
            self.window.push(v);
            if self.window.len() > self.smooth {
                self.window.remove(0);
            }
            let m = mean(&self.window);
            if self.best.is_none_or(|b| m > b + self.min_delta) {
                self.best = Some(m);
                self.best_epoch = epoch;
            }
            if self.top.is_none_or(|t| v > t) {
                self.top = Some(v);
                self.top_epoch = epoch;
            }
        }
        Some(())
    }
}

/// The page's notice while a run is in an extension ([`TrainProgress::extension`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ExtensionNotice {
    /// "Early stop reached at epoch 15 (best epoch 5, mAP 0.835); training loss is still falling
    /// (3.569 -> 3.315). Continuing to epoch 25."
    pub text: String,
    /// The epoch a stop now keeps.
    pub keep_epoch: u32,
    /// The point's extension number (the page's Continue dismisses one point).
    pub extension: u32,
}

/// The settings a run trains with, set by the job when the trainer starts (the page's run
/// settings line; `GET /v1/training` `runs[].progress.settings`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSettings {
    /// Learning rate (`--lr`); None = the trainer's own (YOLO: the optimizer picks).
    pub lr: Option<f32>,
    /// RF-DETR's backbone learning rate (`--lr-encoder`); None = none or the trainer's own.
    pub lr_encoder: Option<f32>,
    /// The schedule in effect (the family's default when none is set).
    pub lr_schedule: super::recipe::LrSchedule,
    /// The final learning rate factor in effect (the family's default when none is set).
    pub lr_min_factor: f32,
    /// What picks the exported epoch and drives the convergence rule (a record from before the
    /// setting: `map`).
    #[serde(default)]
    pub select_metric: SelectMetric,
    /// Epochs (a cap while early stopping is on).
    pub epochs: u32,
    /// Patience in effect (0 = off), after a small validation split doubled it.
    pub patience: u32,
    /// The recommended patience was doubled for a small validation split.
    pub patience_doubled: bool,
    /// The batch size asked for (the progress' `batch_size` is the one training).
    pub batch: u32,
    /// Data loader workers of this attempt.
    pub workers: Option<u32>,
    /// The trainer's seed of a "Repeat with N seeds" run; None = the options' seed.
    pub seed: Option<u64>,
    /// The scene filter (`all`, `day`, `night`, `day+night`, ... `SceneFilter::as_str`).
    pub scene: Option<String>,
    /// The GPU's name.
    pub gpu: Option<String>,
}

/// What the run reports while the script trains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct TrainProgress {
    pub device: Option<String>,
    pub epoch: u32,
    pub epochs: u32,
    pub batch: u32,
    pub batches: u32,
    pub loss: Option<f64>,
    pub val_loss: Option<f64>,
    pub map50: Option<f64>,
    /// Seconds since the first batch event.
    pub elapsed_s: f64,
    /// Estimated seconds left (None until one batch is done).
    pub eta_s: Option<f64>,
    /// `export` once the script exports.
    pub phase: Option<String>,
    /// Loss per finished epoch.
    pub losses: Vec<f64>,
    /// Every metric per epoch, merged by epoch (an `eval` event fills its epoch's row), at most
    /// [`MAX_HISTORY`] points.
    pub history: Vec<EpochPoint>,
    /// The best epoch so far ([`best_of`]).
    pub best: Option<Best>,
    /// Why this attempt restarted ("out of GPU memory at batch 16, retrying with batch 8").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// The run's patience (0 = off), set from the recipe when it starts.
    pub patience: u32,
    /// The run's min_delta (the convergence rule's), set with `patience`.
    #[serde(skip_serializing_if = "is_zero")]
    pub min_delta: f64,
    /// The run's minimum epochs (the rule never stops before), set with `patience`.
    #[serde(skip_serializing_if = "is_zero_u32")]
    pub min_epochs: u32,
    /// The rule's smoothing window (mean of the last N values; 0 = 1), set with `patience`.
    #[serde(skip_serializing_if = "is_zero_u32")]
    pub smooth: u32,
    /// Epochs since the last improvement of the rule's metric ([`Self::select_metric`]; mAP 50-95,
    /// else mAP 50, by default), or since the last extension: the run reaches a stop point when
    /// this reaches `patience` (past the minimum epochs).
    pub since_best: u32,
    /// The last stop point the run went past because the training loss still fell.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub converge_point: Option<ConvergePoint>,
    /// The anneal the run is in (or ended with): a lowered learning rate until `until`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anneal: Option<Anneal>,
    /// The page's Stop now / Continue notice while the run trains past `converge_point` or
    /// anneals.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extension: Option<ExtensionNotice>,
    /// Training ended early (converged or finished on request).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopped: Option<Stop>,
    /// The current learning rate: the latest batch event's, else the latest epoch's `lr`.
    pub lr: Option<f64>,
    /// The epoch the convergence rule stops after when nothing improves from now on
    /// ([`Self::stop_epoch`]); None while early stopping is off and once the run stopped.
    pub stop_epoch: Option<u32>,
    /// The batch size training (the start event's; the job sets the attempt's beforehand).
    pub batch_size: Option<u32>,
    /// The input size training (the start event's).
    pub imgsz: Option<u32>,
    /// The run's settings (None for a run recorded before they were kept).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub settings: Option<RunSettings>,
    /// The selected metric's source failed: the run went on by mAP.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub select_fallback: Option<SelectFallback>,
}

fn is_zero(v: &f64) -> bool {
    *v == 0.0
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

impl TrainProgress {
    /// Fold `ev` in; `elapsed_s` is the wall time since training started.
    pub fn apply(&mut self, ev: &TrainEvent, elapsed_s: f64) {
        self.elapsed_s = elapsed_s;
        match ev {
            TrainEvent::Start {
                device,
                epochs,
                imgsz,
                batch,
                ..
            } => {
                self.device = Some(device.clone());
                self.epochs = *epochs;
                if imgsz.is_some() {
                    self.imgsz = *imgsz;
                }
                if batch.is_some() {
                    self.batch_size = *batch;
                }
            }
            TrainEvent::Batch {
                epoch,
                epochs,
                batch,
                batches,
                lr,
            } => {
                self.epoch = *epoch;
                self.epochs = *epochs;
                self.batch = *batch;
                self.batches = *batches;
                if let Some(lr) = lr.filter(|v| v.is_finite()) {
                    self.lr = Some(lr);
                }
            }
            TrainEvent::Epoch {
                epoch,
                epochs,
                loss,
                map50,
                map,
                elapsed_s: at,
                metrics,
            } => {
                self.epoch = *epoch;
                self.epochs = *epochs;
                self.batch = self.batches;
                if let Some(l) = loss {
                    self.loss = Some(*l);
                    self.losses.push(*l);
                }
                if map50.is_some() {
                    self.map50 = *map50;
                }
                let named = [("loss", *loss), ("map50", *map50), ("map", *map)];
                self.merge(*epoch, at.unwrap_or(elapsed_s), metrics, &named);
                if let Some(&lr) = self
                    .history
                    .iter()
                    .find(|p| p.epoch == *epoch)
                    .and_then(|p| p.metrics.get("lr"))
                {
                    self.lr = Some(lr);
                }
            }
            TrainEvent::Eval {
                epoch,
                val_loss,
                metrics,
            } => {
                if val_loss.is_some() {
                    self.val_loss = *val_loss;
                }
                let e = if *epoch == 0 { self.epoch } else { *epoch };
                self.merge(e, elapsed_s, metrics, &[("val_loss", *val_loss)]);
            }
            TrainEvent::Phase { phase, .. } => self.phase = Some(phase.clone()),
            TrainEvent::Stop {
                reason,
                epoch,
                best_epoch,
                ..
            } => {
                self.stopped = Some(Stop {
                    reason: reason.clone(),
                    epoch: *epoch,
                    best_epoch: *best_epoch,
                });
            }
            TrainEvent::ConvergePoint(p) => {
                self.converge_point = Some(p.clone());
                self.since_best = self.since_improvement();
            }
            TrainEvent::Anneal(a) => self.anneal = Some(a.clone()),
            TrainEvent::SelectFallback(f) => {
                self.select_fallback = Some(f.clone());
                self.best = best_of(self.rule_history(), self.select_metric());
                self.since_best = self.since_improvement();
            }
            TrainEvent::SelfCheck { .. } | TrainEvent::Done { .. } => {}
        }
        self.extension = self.extension_notice();
        let early_stop = self.patience > 0 && (self.epochs == 0 || self.patience < self.epochs);
        self.stop_epoch = (early_stop && self.stopped.is_none()).then(|| self.stop_epoch());
        let f = self.fraction();
        self.eta_s = (f > 0.0 && f < 1.0).then(|| elapsed_s * (1.0 - f) / f);
        if self.phase.is_some() {
            self.eta_s = None;
        }
    }

    /// Fold one event's metrics into `epoch`'s row of the history (created when new): raw keys
    /// under their [`canonical`] id (the first key of an id wins), the event's named fields
    /// (`loss`, `map50`, `map`, `val_loss`) over them.
    fn merge(
        &mut self,
        epoch: u32,
        elapsed_s: f64,
        raw: &BTreeMap<String, f64>,
        named: &[(&str, Option<f64>)],
    ) {
        if epoch == 0 {
            return;
        }
        let mut known: BTreeMap<String, f64> = BTreeMap::new();
        let mut other: Vec<(String, f64)> = Vec::new();
        for (k, &v) in raw {
            if !v.is_finite() || k.is_empty() || k.chars().count() > MAX_METRIC_KEY {
                continue;
            }
            match canonical(k) {
                // A key that is the id itself (the script's explicit `lr` beside Ultralytics'
                // `lr/pg0`, its bias group on older versions) wins over the trainer's raw keys.
                Some(id) if id == k => {
                    known.insert(id.to_string(), v);
                }
                Some(id) => {
                    known.entry(id.to_string()).or_insert(v);
                }
                None => other.push((k.clone(), v)),
            }
        }
        // The event's own fields are the script's choice among the trainer's keys: they win.
        for (id, v) in named {
            if let Some(v) = v.filter(|v| v.is_finite()) {
                known.insert((*id).to_string(), v);
            }
        }
        if known.is_empty() && other.is_empty() {
            return;
        }
        let at = match self.history.iter().position(|p| p.epoch >= epoch) {
            Some(i) if self.history.get(i).is_some_and(|p| p.epoch == epoch) => i,
            Some(i) => {
                self.history.insert(i, EpochPoint::default());
                i
            }
            None => {
                self.history.push(EpochPoint::default());
                self.history.len() - 1
            }
        };
        let newest = at + 1 == self.history.len();
        let Some(p) = self.history.get_mut(at) else {
            return;
        };
        p.epoch = epoch;
        p.elapsed_s = p.elapsed_s.max(elapsed_s);
        for (k, v) in known.into_iter().chain(other) {
            if p.metrics.len() < MAX_METRICS_PER_EPOCH || p.metrics.contains_key(&k) {
                p.metrics.insert(k, v);
            }
        }
        // The newest epoch's values are the run's current ones.
        if newest {
            if let Some(&v) = p.metrics.get("map50") {
                self.map50 = Some(v);
            }
            if let Some(&v) = p.metrics.get("val_loss") {
                self.val_loss = Some(v);
            }
        }
        while self.history.len() > MAX_HISTORY {
            self.history.remove(0);
        }
        self.best = best_of(self.rule_history(), self.select_metric());
        self.since_best = self.since_improvement();
    }

    /// The metric the run selects by: its setting, `map` from a [`SelectFallback`] on (and for a
    /// record from before the setting).
    pub fn select_metric(&self) -> SelectMetric {
        match &self.settings {
            Some(s) if self.select_fallback.is_none() => s.select_metric,
            _ => SelectMetric::Map,
        }
    }

    /// The history the best epoch and the rule see: all of it, or from a [`SelectFallback`] on
    /// (the script forgets the earlier values there).
    fn rule_history(&self) -> &[EpochPoint] {
        let from = self.select_fallback.as_ref().map_or(0, |f| f.epoch);
        let at = self.history.partition_point(|p| p.epoch < from);
        self.history.get(at..).unwrap_or_default()
    }

    /// The rule's state over the history ([`rule_values`] of the selected metric).
    fn rule_state(&self) -> Tracker {
        let mut t = Tracker::new(self.min_delta, self.smooth);
        for (epoch, v) in rule_values(self.rule_history(), self.select_metric()) {
            t.update(epoch, v);
        }
        t
    }

    /// Epochs since the rule's last improvement over the history, or since the last extension
    /// ([`Self::converge_point`]); 0 without a value.
    fn since_improvement(&self) -> u32 {
        let t = self.rule_state();
        if t.best.is_none() {
            return 0;
        }
        let restart = self.converge_point.as_ref().map_or(0, |p| p.epoch);
        t.last.saturating_sub(t.best_epoch.max(restart))
    }

    /// The Stop now / Continue notice: while training past the last `converge_point` or in the
    /// anneal, not stopped or exporting. The kept epoch is the highest raw value so far (the
    /// exported checkpoint).
    fn extension_notice(&self) -> Option<ExtensionNotice> {
        if self.stopped.is_some() || self.phase.is_some() {
            return None;
        }
        let t = self.rule_state();
        let label = self.select_metric().short();
        let kept = |epoch: u32, value: Option<f64>| {
            let (keep, best) = match t.top {
                Some(v) if t.top_epoch > 0 => (t.top_epoch, Some(v)),
                _ => (epoch, value),
            };
            (
                keep,
                best.map_or_else(String::new, |b| format!(", {label} {b:.3}")),
            )
        };
        if let Some(a) = &self.anneal {
            let (keep, best) = kept(a.best_epoch, a.best);
            return Some(ExtensionNotice {
                text: format!(
                    "Early stop reached at epoch {} (best epoch {keep}{best}). Lowering the \
                     learning rate until epoch {}, then stopping.",
                    a.epoch, a.until
                ),
                keep_epoch: keep,
                // Its own number: a Continue on an earlier extension does not hide it.
                extension: MAX_EXTENSIONS + 1,
            });
        }
        let p = self.converge_point.as_ref()?;
        let (keep, best) = kept(p.best_epoch, p.best);
        let loss = match (p.loss_from, p.loss_to) {
            (Some(a), Some(b)) => format!(" ({a:.3} \u{2192} {b:.3})"),
            _ => String::new(),
        };
        Some(ExtensionNotice {
            text: format!(
                "Early stop reached at epoch {} (best epoch {keep}{best}); training loss is still \
                 falling{loss}. Continuing to epoch {}.",
                p.epoch,
                self.stop_epoch()
            ),
            keep_epoch: keep,
            extension: p.extension,
        })
    }

    /// The epoch the rule stops after when nothing improves from now on:
    /// `max(min_epochs, last improvement + patience)`, at most the run's epochs.
    pub fn stop_epoch(&self) -> u32 {
        if let Some(a) = &self.anneal {
            return a.until;
        }
        let last = self.history.last().map_or(self.epoch, |p| p.epoch);
        let improved = last.saturating_sub(self.since_best);
        let n = self.min_epochs.max(improved.saturating_add(self.patience));
        if self.epochs > 0 {
            n.min(self.epochs)
        } else {
            n
        }
    }

    /// One short line for the Jobs tab: `epoch 12 of 50, best mAP50-95 0.402 at epoch 9`. None
    /// before the epochs are known.
    pub fn brief(&self) -> Option<String> {
        if self.phase.is_some() {
            return Some("exporting the trained model".to_string());
        }
        if let Some(t) = self.stop_text() {
            return Some(t);
        }
        if self.epochs == 0 {
            return None;
        }
        let mut s = format!("epoch {} of {}", self.epoch.max(1), self.epochs);
        if let Some(b) = &self.best {
            let label = best_label(&b.metric);
            s.push_str(&format!(
                ", best {label} {:.3} at epoch {}",
                b.value, b.epoch
            ));
        }
        Some(s)
    }

    /// Seconds of training left, least to most: the run stopping at [`Self::stop_epoch`] (nothing
    /// improves from now on) and the run reaching the epoch cap. `paused_s` of the elapsed time
    /// did no work, so it is the training stage's pauses only. None before the first batch and once the epochs are over.
    pub fn time_left(&self, paused_s: f64) -> Option<(f64, f64)> {
        let f = self.fraction();
        if self.phase.is_some() || f <= 0.0 || f >= 1.0 {
            return None;
        }
        let epochs = f64::from(self.epochs);
        let done = f * epochs;
        let per_epoch = (self.elapsed_s - paused_s).max(0.0) / done;
        let cap = (epochs - done) * per_epoch;
        let stop = self
            .stop_epoch
            .map_or(cap, |e| (f64::from(e) - done).max(0.0) * per_epoch);
        Some((stop.min(cap), cap))
    }

    /// Share of the training done, 0..1: finished epochs plus the share of the current one.
    pub fn fraction(&self) -> f64 {
        if self.stopped.is_some() {
            return 1.0;
        }
        if self.epochs == 0 {
            return 0.0;
        }
        let in_epoch = if self.batches > 0 {
            f64::from(self.batch.min(self.batches)) / f64::from(self.batches)
        } else {
            0.0
        };
        let done = f64::from(self.epoch.saturating_sub(1)) + in_epoch;
        (done / f64::from(self.epochs)).clamp(0.0, 1.0)
    }

    /// "converged at epoch 24 (best 12)" / "finished on request at epoch 7 (best 6)".
    pub fn stop_text(&self) -> Option<String> {
        let s = self.stopped.as_ref()?;
        Some(if s.reason == "finish" {
            format!(
                "finished on request at epoch {} (best {})",
                s.epoch, s.best_epoch
            )
        } else {
            format!("converged at epoch {} (best {})", s.epoch, s.best_epoch)
        })
    }

    /// "epoch 3/50, batch 12/40, loss 2.31, about 40 min left".
    pub fn describe(&self) -> String {
        if let Some(p) = &self.phase {
            let stop = self
                .stop_text()
                .map_or_else(String::new, |t| format!(" ({t})"));
            return format!("{p}: exporting the trained model to ONNX and checking it{stop}");
        }
        if let Some(t) = self.stop_text() {
            return t;
        }
        let note = self
            .note
            .as_ref()
            .map_or_else(String::new, |n| format!(" ({n})"));
        if self.epochs == 0 {
            return format!("starting{note}");
        }
        let mut s = format!("epoch {}/{}", self.epoch.max(1), self.epochs);
        if self.batches > 0 && self.batch < self.batches {
            s.push_str(&format!(", batch {}/{}", self.batch, self.batches));
        }
        if let Some(l) = self.loss {
            s.push_str(&format!(", loss {l:.3}"));
        }
        if let Some(m) = self.map50 {
            s.push_str(&format!(", val mAP50 {m:.3}"));
        }
        if let Some(b) = &self.best {
            let label = best_label(&b.metric);
            s.push_str(&format!(
                ", best {label} {:.3} @ epoch {}",
                b.value, b.epoch
            ));
        }
        if let Some(a) = &self.anneal {
            s.push_str(&format!(
                ", lowering the learning rate until epoch {}, then stopping",
                a.until
            ));
        } else if self.patience > 0 && self.patience < self.epochs && self.since_best > 0 {
            s.push_str(&format!(
                ", {} without improvement (stops after epoch {} if no gain)",
                plural(self.since_best, "epoch"),
                self.stop_epoch()
            ));
        }
        if let Some(p) = self
            .converge_point
            .as_ref()
            .filter(|_| self.anneal.is_none())
        {
            s.push_str(&format!(
                ", extended past epoch {} (loss still falling, {} of {MAX_EXTENSIONS})",
                p.epoch, p.extension
            ));
        }
        if let Some(lr) = self.lr {
            s.push_str(&format!(", lr {}", sci(lr)));
        }
        if let Some(e) = self.eta_s {
            s.push_str(&format!(
                ", about {} left",
                super::estimate::format_duration(e)
            ));
        }
        s.push_str(&note);
        s
    }
}

/// `2.3e-4`: two significant digits (`1e-4` when the second is 0).
pub fn sci(v: f64) -> String {
    if v == 0.0 || !v.is_finite() {
        return format!("{v}");
    }
    let s = format!("{v:.1e}");
    match s.split_once('e') {
        Some((m, e)) => format!("{}e{e}", m.strip_suffix(".0").unwrap_or(m)),
        None => s,
    }
}

fn plural(n: u32, word: &str) -> String {
    format!("{n} {word}{}", if n == 1 { "" } else { "s" })
}

/// Running the repository's Python scripts from tests. The one place that decides what a missing
/// interpreter means: a local run skips, a run with `BOP_REQUIRE_PYTHON=1` (CI) fails, so a green
/// CI cannot mean "nothing ran".
#[cfg(test)]
pub(crate) mod test_python {
    use std::io::Write as _;
    use std::process::{Command, Output, Stdio};

    const REQUIRE_ENV: &str = "BOP_REQUIRE_PYTHON";

    const HAS_PYTHON: &str = "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)";
    const HAS_NUMPY: &str = "import numpy";

    /// Whether the variable's value asks for it: anything but unset, empty or `0`.
    fn is_required(value: Option<&std::ffi::OsStr>) -> bool {
        value.is_some_and(|v| !v.is_empty() && v != "0")
    }

    /// The first of `python3`, `python` that runs `probe` with `-I`, as the tests run it. `-I`
    /// leaves out the user's site-packages: a package counts only in the interpreter's own.
    fn find(probe: &str) -> Option<&'static str> {
        ["python3", "python"].into_iter().find(|p| {
            Command::new(p)
                .args(["-I", "-c", probe])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        })
    }

    /// What a test does about a missing tool: panic when it is required, else say so and go on.
    fn missing(required: bool, test: &str, tool: &str) {
        assert!(
            !required,
            "{test}: {tool} not found with `python3` or `python`, and {REQUIRE_ENV} forbids \
             skipping"
        );
        eprintln!("{test} skipped: no {tool}");
    }

    fn found(probe: &str, test: &str, tool: &str) -> Option<&'static str> {
        let hit = find(probe);
        if hit.is_none() {
            let value = std::env::var_os(REQUIRE_ENV);
            missing(is_required(value.as_deref()), test, tool);
        }
        hit
    }

    /// A Python 3.8+, or None when the test should skip (never under `BOP_REQUIRE_PYTHON`).
    pub(crate) fn python3(test: &str) -> Option<&'static str> {
        found(HAS_PYTHON, test, "Python 3.8+")
    }

    /// As [`python3`], with numpy in the interpreter's own site-packages.
    pub(crate) fn python3_with_numpy(test: &str) -> Option<&'static str> {
        found(HAS_NUMPY, test, "Python with numpy")
    }

    /// Run `python args...` with `input` on stdin; its output, or a panic stating why it failed.
    /// The child's stderr is kept for that panic, and a pipe the child closed is not the
    /// failure: a child that dies early must be reported by what it said, not as a broken pipe in
    /// the test (scripts are larger than a pipe's buffer). The input goes in from a thread, so
    /// a child that writes much before reading cannot deadlock against this one.
    pub(crate) fn run(python: &str, args: &[&str], input: &str) -> Output {
        let mut child = Command::new(python)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("cannot start {python}: {e}"));
        let mut stdin = child.stdin.take().expect("stdin is piped");
        let input = input.to_string();
        let writer = std::thread::spawn(move || stdin.write_all(input.as_bytes()));
        let out = child
            .wait_with_output()
            .unwrap_or_else(|e| panic!("waiting for {python}: {e}"));
        let wrote = writer.join().expect("the stdin writer does not panic");
        let said = || {
            format!(
                "\n--- stderr ---\n{}\n--- stdout ---\n{}",
                String::from_utf8_lossy(&out.stderr),
                String::from_utf8_lossy(&out.stdout)
            )
        };
        assert!(
            out.status.success(),
            "{python} failed: {}{}",
            out.status,
            said()
        );
        if let Err(e) = wrote {
            panic!(
                "{python} exited with success without reading its input ({e}){}",
                said()
            );
        }
        out
    }

    mod tests {
        use super::*;

        #[test]
        fn only_a_set_value_requires() {
            assert!(!is_required(None));
            assert!(!is_required(Some("".as_ref())));
            assert!(!is_required(Some("0".as_ref())));
            assert!(is_required(Some("1".as_ref())));
            assert!(is_required(Some("true".as_ref())));
        }

        #[test]
        fn a_missing_tool_is_a_skip_locally() {
            missing(false, "this test", "tool");
        }

        #[test]
        #[should_panic(expected = "this test: tool not found")]
        fn a_missing_tool_is_a_failure_when_required() {
            missing(true, "this test", "tool");
        }

        /// The child that dies with the input unread, larger than any pipe's buffer: the panic
        /// states the child's stderr and status, not a broken pipe.
        #[test]
        fn a_child_that_exits_early_is_reported_by_its_stderr() {
            let Some(python) = python3("a_child_that_exits_early") else {
                return;
            };
            let code = "import sys; sys.stderr.write('the real reason\\n'); sys.exit(7)";
            let input = "x".repeat(4 << 20);
            let panic = std::panic::catch_unwind(|| run(python, &["-I", "-c", code], &input))
                .expect_err("the child failed");
            let text = panic.downcast_ref::<String>().expect("a formatted panic");
            assert!(text.contains("the real reason"), "{text}");
            assert!(text.contains('7'), "{text}");
            assert!(!text.contains("roken pipe"), "{text}");
        }

        /// Exiting 0 with the input unread is a failure too: the script never saw what it tests.
        #[test]
        fn a_child_that_ignores_its_input_is_reported() {
            let Some(python) = python3("a_child_that_ignores_its_input") else {
                return;
            };
            let input = "x".repeat(4 << 20);
            let panic =
                std::panic::catch_unwind(|| run(python, &["-I", "-c", "print('done')"], &input))
                    .expect_err("the child read nothing");
            let text = panic.downcast_ref::<String>().expect("a formatted panic");
            assert!(text.contains("without reading its input"), "{text}");
            assert!(text.contains("done"), "{text}");
        }

        #[test]
        fn input_and_output_larger_than_a_pipe_do_not_deadlock() {
            let Some(python) = python3("input_and_output_larger_than_a_pipe") else {
                return;
            };
            // Writes everything before reading on: no reader in the test would hang here.
            let code = "import sys\nsys.stdout.write('y' * (4 << 20))\nsys.stdout.flush()\n\
                        print(len(sys.stdin.read()))";
            let out = run(python, &["-I", "-c", code], &"x".repeat(4 << 20));
            let text = String::from_utf8_lossy(&out.stdout);
            assert!(text.trim_end().ends_with(&(4 << 20).to_string()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_parse_anywhere_in_a_line() {
        let l = "  1/1 3.3G 1.5 4.9: 20% ━━ 1/5 3.3s/it@train {\"epoch\":1,\"epochs\":3,\"batch\":2,\"batches\":5,\"event\":\"batch\"}";
        assert_eq!(
            parse_line(l),
            Some(TrainEvent::Batch {
                epoch: 1,
                epochs: 3,
                batch: 2,
                batches: 5,
                lr: None,
            })
        );
        let e = parse_line(
            r#"@train {"epoch":2,"epochs":3,"loss":6.1,"map50":0.02,"map":0.01,"elapsed_s":7.9,"event":"epoch"}"#,
        );
        assert!(
            matches!(e, Some(TrainEvent::Epoch { epoch: 2, loss: Some(l), .. }) if (l - 6.1).abs() < 1e-9)
        );
        let done = parse_line(r#"@train {"onnx":"/m/x.onnx","yaml":"/m/x.yaml","family":"rfdetr","classes":["a"],"imgsz":384,"seconds":52.9,"device":"mps","event":"done"}"#).unwrap();
        assert!(
            matches!(done, TrainEvent::Done { ref family, imgsz: 384, .. } if family == "rfdetr")
        );
        let sc = parse_line(r#"@train {"trainer":"rfdetr","torch":"2.14.0","cuda":false,"mps":true,"event":"self_check"}"#).unwrap();
        assert!(matches!(
            sc,
            TrainEvent::SelfCheck {
                mps: true,
                cuda: false,
                ..
            }
        ));
        assert_eq!(parse_line("Epoch 1/3 loss 2"), None);
        assert_eq!(parse_line(r#"@train {"event":"telemetry"}"#), None);
        assert_eq!(parse_line("@train {broken"), None);
    }

    #[test]
    fn the_brief_line_keeps_only_the_epoch_and_the_best() {
        let mut p = TrainProgress::default();
        assert_eq!(p.brief(), None);
        p.epoch = 12;
        p.epochs = 50;
        p.batch = 310;
        p.batches = 1204;
        p.loss = Some(0.42);
        assert_eq!(p.brief().unwrap(), "epoch 12 of 50");
        p.best = Some(Best {
            metric: "map".into(),
            epoch: 9,
            value: 0.4021,
        });
        assert_eq!(
            p.brief().unwrap(),
            "epoch 12 of 50, best mAP50-95 0.402 at epoch 9"
        );
        p.phase = Some("export".into());
        assert_eq!(p.brief().unwrap(), "exporting the trained model");
    }

    #[test]
    fn time_left_runs_from_the_early_stop_to_the_epoch_cap_without_the_paused_time() {
        let p = TrainProgress {
            epoch: 11,
            epochs: 50,
            elapsed_s: 1100.0,
            stop_epoch: Some(20),
            ..TrainProgress::default()
        };
        // 10 finished epochs, 110 s each.
        assert_eq!(p.time_left(0.0), Some((1100.0, 4400.0)));
        // 500 s of it were paused: 60 s per epoch.
        assert_eq!(p.time_left(500.0), Some((600.0, 2400.0)));
        // No early stop: one answer.
        let off = TrainProgress {
            stop_epoch: None,
            ..p.clone()
        };
        assert_eq!(off.time_left(0.0), Some((4400.0, 4400.0)));
        // Past the rule's epoch: the run can stop now.
        let past = TrainProgress {
            stop_epoch: Some(5),
            ..p.clone()
        };
        assert_eq!(past.time_left(0.0), Some((0.0, 4400.0)));
        // Nothing to measure yet, and nothing left once exporting.
        assert_eq!(TrainProgress::default().time_left(0.0), None);
        let export = TrainProgress {
            phase: Some("export".into()),
            ..p
        };
        assert_eq!(export.time_left(0.0), None);
    }

    #[test]
    fn progress_and_eta_follow_the_batches() {
        let mut p = TrainProgress::default();
        p.apply(
            &TrainEvent::Start {
                device: "cuda:0".into(),
                train_images: 100,
                valid_images: 10,
                epochs: 4,
                imgsz: Some(640),
                batch: Some(8),
            },
            0.0,
        );
        assert_eq!(p.fraction(), 0.0);
        assert_eq!(p.eta_s, None);
        assert_eq!((p.batch_size, p.imgsz), (Some(8), Some(640)));
        p.apply(
            &TrainEvent::Batch {
                epoch: 1,
                epochs: 4,
                batch: 5,
                batches: 10,
                lr: None,
            },
            10.0,
        );
        assert!((p.fraction() - 0.125).abs() < 1e-12);
        // 12.5% in 10 s: 70 s left.
        assert!((p.eta_s.unwrap() - 70.0).abs() < 1e-9);
        p.apply(
            &TrainEvent::Epoch {
                epoch: 1,
                epochs: 4,
                loss: Some(2.5),
                map50: Some(0.3),
                map: None,
                elapsed_s: None,
                metrics: BTreeMap::new(),
            },
            20.0,
        );
        assert!((p.fraction() - 0.25).abs() < 1e-12);
        assert_eq!(p.losses, [2.5]);
        assert!(
            p.describe().starts_with(
                "epoch 1/4, loss 2.500, val mAP50 0.300, best mAP50 0.300 @ epoch 1, about 60 s left"
            ),
            "{}",
            p.describe()
        );
        p.apply(
            &TrainEvent::Batch {
                epoch: 2,
                epochs: 4,
                batch: 3,
                batches: 10,
                lr: None,
            },
            26.0,
        );
        assert!(p.describe().contains("batch 3/10"), "{}", p.describe());
        p.apply(
            &TrainEvent::Phase {
                phase: "export".into(),
                detail: String::new(),
            },
            90.0,
        );
        assert_eq!(p.eta_s, None);
        assert!(p.describe().contains("exporting"));
    }

    #[test]
    fn learning_rate_input_size_batch_and_stop_epoch_follow_the_events() {
        let with = parse_line(
            r#"@train {"event":"batch","epoch":1,"epochs":20,"batch":4,"batches":10,"lr":0.00023}"#,
        )
        .unwrap();
        assert!(
            matches!(with, TrainEvent::Batch { lr: Some(v), .. } if (v - 0.00023).abs() < 1e-15)
        );
        let without =
            parse_line(r#"@train {"event":"batch","epoch":1,"epochs":20,"batch":5,"batches":10}"#)
                .unwrap();
        assert!(matches!(without, TrainEvent::Batch { lr: None, .. }));
        // A null or a string the script let through does not lose the event.
        for bad in ["null", "\"x\""] {
            let l = format!(
                r#"@train {{"event":"batch","epoch":1,"epochs":20,"batch":6,"batches":10,"lr":{bad}}}"#
            );
            assert!(
                matches!(parse_line(&l), Some(TrainEvent::Batch { lr: None, .. })),
                "{l}"
            );
        }
        // The script's start event (extra keys of device_info ignored).
        let start = parse_line(
            r#"@train {"event":"start","trainer":"ultralytics","family":"yolo26","device":"cuda:0","classes":["person"],"train_images":100,"valid_images":20,"epochs":20,"imgsz":640,"batch":8,"cuda":true}"#,
        )
        .unwrap();
        let mut p = TrainProgress {
            patience: 5,
            min_epochs: 3,
            min_delta: 0.001,
            smooth: 1,
            batch_size: Some(16),
            ..TrainProgress::default()
        };
        p.apply(&start, 0.0);
        assert_eq!((p.batch_size, p.imgsz, p.epochs), (Some(8), Some(640), 20));
        // An older script's start event keeps the job's batch.
        let mut q = TrainProgress {
            batch_size: Some(4),
            ..TrainProgress::default()
        };
        q.apply(
            &parse_line(r#"@train {"event":"start","device":"mps","epochs":3}"#).unwrap(),
            0.0,
        );
        assert_eq!((q.batch_size, q.imgsz), (Some(4), None));
        assert_eq!(p.lr, None);
        p.apply(&with, 1.0);
        assert_eq!(p.lr, Some(0.00023));
        assert!(p.describe().contains(", lr 2.3e-4"), "{}", p.describe());
        p.apply(&without, 2.0);
        assert_eq!(p.lr, Some(0.00023), "a batch without lr keeps the last one");
        // The script's explicit `lr` (main group) wins over Ultralytics' `lr/pg0`.
        let ep = |e: u32, metrics: &[(&str, f64)]| TrainEvent::Epoch {
            epoch: e,
            epochs: 20,
            loss: Some(1.0),
            map50: None,
            map: None,
            elapsed_s: None,
            metrics: metrics.iter().map(|(k, v)| ((*k).into(), *v)).collect(),
        };
        p.apply(
            &ep(
                1,
                &[
                    ("lr/pg0", 0.07),
                    ("lr", 0.0002),
                    ("metrics/mAP50-95(B)", 0.5),
                ],
            ),
            3.0,
        );
        assert_eq!(p.history[0].metrics["lr"], 0.0002);
        assert_eq!(p.lr, Some(0.0002));
        // Alone, a raw key is the lr.
        p.apply(
            &ep(2, &[("lr/pg0", 0.0001), ("metrics/mAP50-95(B)", 0.4)]),
            4.0,
        );
        assert_eq!(p.lr, Some(0.0001));
        // Best epoch 1 + patience 5 (at least min_epochs 3).
        assert_eq!(p.stop_epoch, Some(6));
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v["stop_epoch"], 6);
        assert_eq!(v["lr"], 0.0001);
        assert_eq!(
            (v["batch_size"].clone(), v["imgsz"].clone()),
            (8.into(), 640.into())
        );
        assert!(v.get("settings").is_none());
        // Early stopping off, or stopped: none.
        let mut off = TrainProgress::default();
        off.apply(&ep(1, &[("map", 0.4)]), 1.0);
        assert_eq!(off.stop_epoch, None);
        p.apply(
            &TrainEvent::Stop {
                reason: "converged".into(),
                epoch: 6,
                best_epoch: 1,
                best: Some(0.5),
            },
            9.0,
        );
        assert_eq!(p.stop_epoch, None);
        // A run recorded before these fields existed.
        let old: TrainProgress =
            serde_json::from_str(r#"{"epoch":3,"epochs":10,"patience":4}"#).unwrap();
        assert_eq!(
            (old.lr, old.stop_epoch, old.batch_size, old.imgsz),
            (None, None, None, None)
        );
        assert!(old.settings.is_none());
        // The settings' JSON shape (the page's run settings line).
        let s = RunSettings {
            lr: Some(5e-5),
            lr_encoder: Some(7.5e-5),
            lr_schedule: crate::training::recipe::LrSchedule::Cosine,
            lr_min_factor: 0.1,
            select_metric: SelectMetric::F2,
            epochs: 100,
            patience: 20,
            patience_doubled: true,
            batch: 4,
            workers: Some(3),
            seed: Some(2),
            scene: Some("night+ir".into()),
            gpu: Some("NVIDIA GeForce GTX 1070 Ti".into()),
        };
        let v = serde_json::to_value(TrainProgress {
            settings: Some(s.clone()),
            ..TrainProgress::default()
        })
        .unwrap();
        assert_eq!(v["settings"]["lr_schedule"], "cosine");
        assert_eq!(v["settings"]["select_metric"], "f2");
        assert_eq!(v["settings"]["patience_doubled"], true);
        assert_eq!(v["settings"]["scene"], "night+ir");
        let back: RunSettings = serde_json::from_value(v["settings"].clone()).unwrap();
        assert_eq!(back, s);
        assert_eq!(sci(0.00023), "2.3e-4");
        assert_eq!(sci(0.0001), "1e-4");
        assert_eq!(sci(0.01), "1e-2");
    }

    fn epoch(e: u32, metrics: &[(&str, f64)]) -> TrainEvent {
        TrainEvent::Epoch {
            epoch: e,
            epochs: 3,
            loss: None,
            map50: None,
            map: None,
            elapsed_s: Some(f64::from(e) * 10.0),
            metrics: metrics
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect(),
        }
    }

    #[test]
    fn epoch_events_carry_metrics_and_drop_non_numbers() {
        let e = parse_line(
            r#"@train {"event":"epoch","epoch":1,"epochs":3,"loss":4.2,"metrics":{"val/mAP_50_95":0.711,"val/F1":0.8,"bad":null,"s":"x","val/AP/raccoon":0.5,"epoch":1.0}}"#,
        )
        .unwrap();
        let TrainEvent::Epoch { metrics, .. } = &e else {
            panic!("{e:?}")
        };
        assert_eq!(metrics.len(), 3, "{metrics:?}");
        // Old events without metrics still parse.
        assert!(matches!(
            parse_line(r#"@train {"event":"eval","epoch":2,"val_loss":1.5}"#),
            Some(TrainEvent::Eval { epoch: 2, val_loss: Some(_), ref metrics }) if metrics.is_empty()
        ));
        let mut p = TrainProgress::default();
        p.apply(&e, 5.0);
        let row = &p.history[0];
        assert_eq!(row.epoch, 1);
        assert_eq!(row.metrics["map"], 0.711);
        assert_eq!(row.metrics["f1"], 0.8);
        assert_eq!(row.metrics["loss"], 4.2);
        // Unknown keys stay under their raw name.
        assert_eq!(row.metrics["val/AP/raccoon"], 0.5);
        assert_eq!(
            p.best,
            Some(Best {
                metric: "map".into(),
                epoch: 1,
                value: 0.711
            })
        );
        assert!(
            p.describe().contains("best mAP50-95 0.711 @ epoch 1"),
            "{}",
            p.describe()
        );
    }

    #[test]
    fn raw_keys_map_to_stable_ids() {
        for (raw, id) in [
            // rfdetr
            ("val/mAP_50_95", "map"),
            ("val/mAP_50", "map50"),
            ("val/mAP_75", "map75"),
            ("val/mAR", "mar"),
            ("val/F1", "f1"),
            ("val/precision", "precision"),
            ("val/recall", "recall"),
            ("val/loss", "val_loss"),
            ("train/loss", "loss"),
            ("train/lr", "lr"),
            ("val/ema_mAP_50_95", "ema_map"),
            ("val/ema_mAP_50", "ema_map50"),
            ("val/ema_mAR", "ema_mar"),
            // Ultralytics
            ("metrics/mAP50-95(B)", "map"),
            ("metrics/mAP50(B)", "map50"),
            ("metrics/precision(B)", "precision"),
            ("metrics/recall(B)", "recall"),
            ("lr/pg0", "lr"),
            // transformers
            ("eval_loss", "val_loss"),
            ("learning_rate", "lr"),
            ("eval_map", "map"),
            ("eval_map_50", "map50"),
            // the script's own
            ("f2", "f2"),
            ("iou", "iou"),
            ("f2_thr", "f2_thr"),
            ("mem_gb", "mem_gb"),
            ("img_s", "img_s"),
        ] {
            assert_eq!(canonical(raw), Some(id), "{raw}");
        }
        for raw in [
            "val/ema_F1_bogus",
            "val/AP/person",
            "val/box_loss",
            "",
            "val/ema_",
        ] {
            assert_eq!(canonical(raw), None, "{raw}");
        }
        // Every id is described once.
        let mut ids: Vec<&str> = METRICS.iter().map(|m| m.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), METRICS.len());
        let j = serde_json::to_value(METRICS).unwrap();
        assert_eq!(j[0]["id"], "map");
        assert_eq!(j[0]["format"], "fixed3");
        assert_eq!(j[0]["higher_better"], true);
        let lr = METRICS.iter().position(|m| m.id == "lr").unwrap();
        assert_eq!(j[lr]["format"], "sci");
        assert!(j[lr]["higher_better"].is_null());
    }

    /// `iou` lands in the history and never decides the best epoch or the convergence rule.
    #[test]
    fn mean_iou_is_shown_but_decides_nothing() {
        let mut p = TrainProgress::default();
        p.apply(&epoch(1, &[("val/mAP_50_95", 0.5), ("iou", 0.9)]), 10.0);
        p.apply(&epoch(2, &[("val/mAP_50_95", 0.6), ("iou", 0.7)]), 20.0);
        assert_eq!(p.history[0].metrics["iou"], 0.9);
        assert_eq!(p.history[1].metrics["iou"], 0.7);
        assert_eq!(
            p.best,
            Some(Best {
                metric: "map".into(),
                epoch: 2,
                value: 0.6
            })
        );
        // Only an `iou`: no best epoch.
        let mut p = TrainProgress::default();
        p.apply(&epoch(1, &[("iou", 0.9)]), 10.0);
        assert_eq!(p.best, None);
    }

    #[test]
    fn eval_events_fill_the_same_epoch_and_best_follows_the_primary_metric() {
        let mut p = TrainProgress::default();
        // transformers: the epoch's loss, then its evaluation.
        p.apply(&epoch(1, &[("loss", 3.0), ("learning_rate", 1e-4)]), 10.0);
        p.apply(
            &TrainEvent::Eval {
                epoch: 1,
                val_loss: Some(2.0),
                metrics: [("eval_loss".to_string(), 2.0)].into(),
            },
            11.0,
        );
        assert_eq!(p.history.len(), 1);
        assert_eq!(p.history[0].metrics["val_loss"], 2.0);
        assert_eq!(p.history[0].metrics["lr"], 1e-4);
        assert_eq!(p.history[0].elapsed_s, 11.0);
        assert_eq!(p.best.as_ref().unwrap().metric, "val_loss");
        p.apply(&epoch(2, &[("loss", 2.5)]), 20.0);
        // An eval without an epoch goes to the current one.
        p.apply(
            &TrainEvent::Eval {
                epoch: 0,
                val_loss: Some(1.5),
                metrics: BTreeMap::new(),
            },
            21.0,
        );
        p.apply(&epoch(3, &[("loss", 2.0)]), 30.0);
        p.apply(
            &TrainEvent::Eval {
                epoch: 3,
                val_loss: Some(1.8),
                metrics: BTreeMap::new(),
            },
            31.0,
        );
        // Lowest val loss: epoch 2.
        assert_eq!(
            p.best,
            Some(Best {
                metric: "val_loss".into(),
                epoch: 2,
                value: 1.5
            })
        );
        assert_eq!(p.val_loss, Some(1.8));
        // mAP outranks the val loss once any epoch has it.
        p.history[0].metrics.insert("map".into(), 0.2);
        assert_eq!(
            best_of(&p.history, SelectMetric::Map).unwrap().metric,
            "map"
        );
        // map50 when there is no mAP50-95; a tie keeps the earlier epoch.
        let h = [
            EpochPoint {
                epoch: 1,
                elapsed_s: 1.0,
                metrics: [("map50".to_string(), 0.4)].into(),
            },
            EpochPoint {
                epoch: 2,
                elapsed_s: 2.0,
                metrics: [("map50".to_string(), 0.4)].into(),
            },
        ];
        assert_eq!(best_of(&h, SelectMetric::Map).unwrap().epoch, 1);
        assert_eq!(best_of(&[], SelectMetric::Map), None);
    }

    #[test]
    fn history_is_merged_in_epoch_order_and_capped() {
        let mut p = TrainProgress::default();
        p.apply(&epoch(2, &[("f1", 0.2)]), 1.0);
        p.apply(&epoch(1, &[("f1", 0.1)]), 2.0);
        assert_eq!(
            p.history.iter().map(|h| h.epoch).collect::<Vec<_>>(),
            [1, 2]
        );
        // Over-long keys are dropped; a row keeps at most MAX_METRICS_PER_EPOCH keys.
        let many: Vec<(String, f64)> = (0..100).map(|i| (format!("k{i:03}"), 1.0)).collect();
        let long = "x".repeat(MAX_METRIC_KEY + 1);
        let mut raw: BTreeMap<String, f64> = many.into_iter().collect();
        raw.insert(long.clone(), 1.0);
        raw.insert("val/mAP_50".into(), 0.5);
        p.apply(
            &TrainEvent::Epoch {
                epoch: 3,
                epochs: 3,
                loss: None,
                map50: None,
                map: None,
                elapsed_s: None,
                metrics: raw,
            },
            3.0,
        );
        let row = &p.history[2];
        assert_eq!(row.metrics.len(), MAX_METRICS_PER_EPOCH);
        assert_eq!(row.metrics["map50"], 0.5, "known ids come first");
        assert!(!row.metrics.contains_key(&long));
        assert_eq!(p.map50, Some(0.5));
        for e in 4..(MAX_HISTORY as u32 + 10) {
            p.apply(&epoch(e, &[("loss", 1.0)]), 4.0);
        }
        assert_eq!(p.history.len(), MAX_HISTORY);
        assert_eq!(p.history[0].epoch, 10);
    }

    #[test]
    fn old_records_without_history_still_load() {
        let old = r#"{"device":"mps","epoch":1,"epochs":50,"batch":10,"batches":10,"loss":4.19,
            "val_loss":null,"map50":0.86,"elapsed_s":2117.9,"eta_s":100.0,"phase":null,"losses":[4.19]}"#;
        let p: TrainProgress = serde_json::from_str(old).unwrap();
        assert!(p.history.is_empty());
        assert_eq!(p.best, None);
        assert_eq!(p.losses, [4.19]);
        let j = serde_json::to_value(&p).unwrap();
        assert!(j["history"].as_array().unwrap().is_empty());
        assert!(j["best"].is_null());
    }

    #[test]
    fn stop_events_end_training_and_describe_the_run() {
        let ev = parse_line(
            r#"@train {"event":"stop","reason":"converged","epoch":24,"best_epoch":12,"best":0.41}"#,
        )
        .unwrap();
        assert_eq!(
            ev,
            TrainEvent::Stop {
                reason: "converged".into(),
                epoch: 24,
                best_epoch: 12,
                best: Some(0.41)
            }
        );
        let mut p = TrainProgress {
            patience: 3,
            min_delta: 0.001,
            ..TrainProgress::default()
        };
        let map = |e: u32, v: f64| {
            let mut ev = epoch(e, &[("map", v)]);
            if let TrainEvent::Epoch { epochs, .. } = &mut ev {
                *epochs = 10;
            }
            ev
        };
        p.apply(&map(1, 0.30), 1.0);
        p.apply(&map(2, 0.40), 2.0);
        // Below min_delta: no improvement for the rule (the best epoch still moves).
        p.apply(&map(3, 0.4005), 3.0);
        assert_eq!(p.since_best, 1);
        assert_eq!(p.best.as_ref().unwrap().epoch, 3);
        p.apply(&map(4, 0.39), 4.0);
        assert_eq!(p.since_best, 2);
        assert!(
            p.describe()
                .contains("2 epochs without improvement (stops after epoch 5 if no gain)"),
            "{}",
            p.describe()
        );
        // The minimum epochs win over an earlier stale stop.
        let floor = TrainProgress {
            min_epochs: 8,
            ..p.clone()
        };
        assert_eq!(floor.stop_epoch(), 8);
        assert!(
            floor
                .describe()
                .contains("(stops after epoch 8 if no gain)"),
            "{}",
            floor.describe()
        );
        assert!(p.fraction() < 1.0);
        p.apply(
            &TrainEvent::Stop {
                reason: "finish".into(),
                epoch: 4,
                best_epoch: 3,
                best: Some(0.4005),
            },
            5.0,
        );
        assert_eq!(p.fraction(), 1.0);
        assert_eq!(p.eta_s, None);
        assert_eq!(p.describe(), "finished on request at epoch 4 (best 3)");
        p.apply(
            &TrainEvent::Phase {
                phase: "export".into(),
                detail: String::new(),
            },
            6.0,
        );
        assert!(
            p.describe()
                .ends_with("(finished on request at epoch 4 (best 3))"),
            "{}",
            p.describe()
        );
        p.stopped.as_mut().unwrap().reason = "converged".into();
        assert_eq!(p.stop_text().unwrap(), "converged at epoch 4 (best 3)");
        // Round trip; an old record has none of the new fields.
        let back: TrainProgress =
            serde_json::from_value(serde_json::to_value(&p).unwrap()).unwrap();
        assert_eq!(back, p);
        let old: TrainProgress = serde_json::from_str(r#"{"epoch":1,"epochs":5}"#).unwrap();
        assert_eq!((old.stopped, old.since_best, old.patience), (None, 0, 0));
    }

    /// One fixed sequence: `(epoch, mAP, training loss)`, the rule, and where it stops plus the
    /// stop points it extends past.
    struct Case {
        epochs: Vec<(u32, Option<f64>, Option<f64>)>,
        rule: Rule,
        want: ConvergenceRun,
    }

    fn case(
        values: &[(u32, f64)],
        patience: u32,
        min_epochs: u32,
        delta: f64,
        smooth: u32,
    ) -> Case {
        Case {
            epochs: values.iter().map(|&(e, v)| (e, Some(v), None)).collect(),
            rule: Rule {
                patience,
                min_epochs,
                min_delta: delta,
                smooth,
                epochs: 0,
                anneal: 0,
            },
            want: ConvergenceRun::default(),
        }
    }

    impl Case {
        fn stops(mut self, stop: Option<u32>, points: &[u32]) -> Self {
            self.want = ConvergenceRun {
                stop,
                points: points.to_vec(),
                anneal: None,
            };
            self
        }

        /// Anneals `epochs` at the first stop point that would stop; it started at `from`.
        fn anneals(mut self, epochs: u32, from: Option<u32>) -> Self {
            self.rule.anneal = epochs;
            self.want.anneal = from;
            self
        }

        /// The training loss per entry (in order), and the run's epochs.
        fn losses(mut self, losses: &[Option<f64>], epochs: u32) -> Self {
            for (e, l) in self.epochs.iter_mut().zip(losses) {
                e.2 = *l;
            }
            self.rule.epochs = epochs;
            self
        }
    }

    /// Validation mAP50-95 and training loss of run 20261030-010233-b69d (RF-DETR nano, 16
    /// validation images): stopped at epoch 15 on the epoch-5 peak while the loss still fell.
    const INCIDENT_MAP: [f64; 15] = [
        0.7343, 0.7732, 0.7879, 0.7579, 0.8350, 0.7320, 0.8194, 0.8155, 0.7966, 0.7669, 0.7535,
        0.7368, 0.8009, 0.8202, 0.8188,
    ];
    const INCIDENT_LOSS: [f64; 15] = [
        4.582, 4.141, 3.842, 3.680, 3.618, 3.630, 3.529, 3.547, 3.655, 3.483, 3.396, 3.167, 3.322,
        3.156, 3.535,
    ];

    fn incident() -> Vec<(u32, f64)> {
        (1..).zip(INCIDENT_MAP).collect()
    }

    fn incident_losses() -> Vec<Option<f64>> {
        INCIDENT_LOSS.iter().map(|&l| Some(l)).collect()
    }

    /// Fixed mAP sequences and rules, with the epoch each stops at (None = runs to the end).
    fn sequences() -> Vec<Case> {
        let ramp: Vec<(u32, f64)> = (1..=30).map(|e| (e, f64::from(e) * 0.01)).collect();
        let plateau: Vec<(u32, f64)> = (1..=30)
            .map(|e| {
                (
                    e,
                    if e <= 6 {
                        f64::from(e) * 0.05
                    } else {
                        0.3 + f64::from(e) * 1e-4
                    },
                )
            })
            .collect();
        let early: Vec<(u32, f64)> =
            vec![(1, 0.5), (2, 0.4), (3, 0.4), (4, 0.4), (5, 0.6), (6, 0.5)];
        let noisy: Vec<(u32, f64)> = vec![
            (1, 0.1),
            (2, 0.2),
            (2, 0.9),
            (3, 0.2005),
            (4, 0.21),
            (5, 0.1),
            (6, 0.2),
            (7, 0.2),
        ];
        let flat: Vec<(u32, f64)> = (1..=40).map(|e| (e, 0.5)).collect();
        // Falls 10% per epoch; then flat after epoch 5.
        let falling: Vec<Option<f64>> = (0..40).map(|i| Some(4.0 * 0.9f64.powi(i))).collect();
        let flattening: Vec<Option<f64>> =
            (0..40).map(|i| Some(4.0 * 0.9f64.powi(i.min(4)))).collect();
        // Missing losses are skipped: epochs 2..5 hold 3 values, too few to fall.
        let gappy: Vec<Option<f64>> = (0..40)
            .map(|i| (i % 4 != 2).then(|| 4.0 * 0.9f64.powi(i)))
            .collect();
        vec![
            // smooth 1: the rule as it was.
            case(&ramp, 3, 10, 0.001, 1).stops(None, &[]),
            case(&plateau, 5, 10, 0.001, 1).stops(Some(16), &[]),
            case(&plateau, 0, 0, 0.0, 1).stops(None, &[]),
            case(&early, 2, 0, 0.0, 1).stops(Some(3), &[]),
            case(&early, 2, 5, 0.0, 1).stops(None, &[]),
            case(&noisy, 3, 0, 0.001, 1).stops(Some(7), &[]),
            // smooth 0 reads as 1.
            case(&plateau, 5, 10, 0.001, 0).stops(Some(16), &[]),
            // smooth 3: a steady ramp keeps gaining; the plateau's 3-mean gains last at epoch 8.
            case(&ramp, 3, 10, 0.001, 3).stops(None, &[]),
            case(&plateau, 5, 10, 0.001, 3).stops(Some(13), &[]),
            case(&early, 2, 0, 0.0, 3).stops(Some(3), &[]),
            // The incident: one noisy peak stops it at 15; the 3-mean still gains at 15.
            case(&incident(), 10, 10, 0.001, 1).stops(Some(15), &[]),
            case(&incident(), 10, 10, 0.001, 3).stops(None, &[]),
            // With its losses: still falling at 15, so it goes on.
            case(&incident(), 10, 10, 0.001, 1)
                .losses(&incident_losses(), 100)
                .stops(None, &[15]),
            // Loss still falling: two extensions, then the stop.
            case(&flat, 4, 0, 0.0, 1)
                .losses(&falling, 40)
                .stops(Some(13), &[5, 9]),
            // The loss flattens after the first extension: stop at the next point.
            case(&flat, 4, 0, 0.0, 1)
                .losses(&flattening, 40)
                .stops(Some(9), &[5]),
            // No epochs left at the stop point: stop.
            case(&flat, 4, 0, 0.0, 1)
                .losses(&falling, 5)
                .stops(Some(5), &[]),
            // Too few losses in the window: stop; later windows have enough.
            case(&flat, 4, 0, 0.0, 1)
                .losses(&gappy, 40)
                .stops(Some(5), &[]),
            case(&flat, 3, 0, 0.0, 1)
                .losses(&falling, 40)
                .stops(Some(4), &[]),
            // Anneal: the stop point at 5 trains 3 more epochs, then stops.
            case(&flat, 4, 0, 0.0, 1)
                .stops(Some(8), &[])
                .anneals(3, Some(5)),
            case(&plateau, 5, 10, 0.001, 1)
                .stops(Some(21), &[])
                .anneals(5, Some(16)),
            // Clipped by the run's epochs (7), and none at the last epoch (5).
            case(&flat, 4, 0, 0.0, 1)
                .losses(&[], 7)
                .stops(Some(7), &[])
                .anneals(5, Some(5)),
            case(&flat, 4, 0, 0.0, 1)
                .losses(&[], 5)
                .stops(Some(5), &[])
                .anneals(5, None),
            // Extensions first; the stop point after them anneals.
            case(&flat, 4, 0, 0.0, 1)
                .losses(&flattening, 40)
                .stops(Some(12), &[5])
                .anneals(3, Some(9)),
            case(&flat, 4, 0, 0.0, 1)
                .losses(&falling, 40)
                .stops(Some(16), &[5, 9])
                .anneals(3, Some(13)),
            // The sequence ends inside the anneal: no stop.
            case(&incident(), 10, 10, 0.001, 1)
                .stops(None, &[])
                .anneals(5, Some(15)),
            // Patience 0 never reaches a stop point, so never anneals.
            case(&plateau, 0, 0, 0.0, 1)
                .stops(None, &[])
                .anneals(5, None),
        ]
    }

    #[test]
    fn convergence_rule_stops_where_expected() {
        for (i, c) in sequences().into_iter().enumerate() {
            assert_eq!(convergence_run(&c.epochs, &c.rule), c.want, "sequence {i}");
        }
        assert_eq!(convergence_decision(&[], 1, 0, 0.0, 1, 0), None);
        // The incident: smooth 1 stops at 15, smooth 3 not by 15.
        assert_eq!(
            convergence_decision(&incident(), 10, 10, 0.001, 1, 0),
            Some(15)
        );
        assert_eq!(convergence_decision(&incident(), 10, 10, 0.001, 3, 0), None);
        // An anneal moves the stop: 15 + 4 is past the 15 epochs; a 40-epoch plateau stops at
        // 16 + 4.
        assert_eq!(convergence_decision(&incident(), 10, 10, 0.001, 1, 4), None);
        let plateau: Vec<(u32, f64)> = (1..=40).map(|e| (e, 0.5)).collect();
        assert_eq!(convergence_decision(&plateau, 5, 10, 0.0, 1, 0), Some(10));
        assert_eq!(convergence_decision(&plateau, 5, 10, 0.0, 1, 4), Some(14));
    }

    #[test]
    fn smoothing_moves_the_gain_not_the_kept_checkpoint() {
        let mut t = Tracker::new(0.001, 3);
        for (e, v) in incident() {
            t.update(e, Some(v));
        }
        // Raw top: the epoch-5 peak (the checkpoint kept); 3-mean gains last at epoch 15.
        assert_eq!((t.top_epoch, t.top), (5, Some(0.8350)));
        assert_eq!(t.best_epoch, 15);
        let gains: Vec<u32> = {
            let mut t = Tracker::new(0.001, 3);
            incident()
                .into_iter()
                .filter_map(|(e, v)| {
                    let before = t.best_epoch;
                    t.update(e, Some(v));
                    (t.best_epoch != before).then_some(e)
                })
                .collect()
        };
        assert_eq!(gains, [1, 2, 3, 4, 5, 7, 9, 15]);
    }

    /// `loss_falling` cases: `(losses, falling)`.
    fn loss_cases() -> Vec<(Vec<Option<f64>>, bool)> {
        let s = |v: &[f64]| v.iter().map(|&x| Some(x)).collect::<Vec<_>>();
        vec![
            (vec![], false),
            (s(&[4.0, 3.0, 2.0]), false),
            (s(&[4.0, 3.0, 2.0, 1.0]), true),
            (s(&[1.0, 1.0, 1.0, 1.0]), false),
            (s(&[1.0, 1.0, 0.985, 0.985]), false),
            (s(&[1.0, 1.0, 0.97, 0.97]), true),
            (s(&[1.0, 1.0, 1.1, 1.2]), false),
            // An odd middle value counts in neither half.
            (s(&[4.0, 4.0, 100.0, 3.0, 3.0]), true),
            // Missing values are skipped.
            (
                vec![Some(4.0), None, Some(4.0), Some(3.0), None, Some(3.0)],
                true,
            ),
            (vec![Some(4.0), None, None, Some(3.0), Some(2.0)], false),
            (incident_losses().split_off(5), true),
        ]
    }

    #[test]
    fn loss_falling_compares_the_halves() {
        for (i, (l, want)) in loss_cases().into_iter().enumerate() {
            assert_eq!(loss_falling(&l), want, "case {i}");
        }
        assert!(!loss_falling(&[Some(f64::NAN); 8]));
        let (a, b) = loss_halves(&incident_losses().split_off(5)).unwrap();
        assert!(
            (a - 3.5688).abs() < 1e-9 && (b - 3.3152).abs() < 1e-9,
            "{a} {b}"
        );
    }

    #[test]
    fn converge_points_parse_and_show_the_stop_now_notice() {
        let ev = parse_line(
            r#"@train {"epoch":15,"best_epoch":5,"best":0.835,"loss_falling":true,"loss_from":3.5688,"loss_to":3.3152,"extension":1,"next_check":25,"event":"converge_point"}"#,
        )
        .unwrap();
        let TrainEvent::ConvergePoint(cp) = &ev else {
            panic!("{ev:?}")
        };
        assert_eq!(
            (cp.epoch, cp.best_epoch, cp.extension, cp.next_check),
            (15, 5, 1, 25)
        );
        let mut p = TrainProgress {
            patience: 10,
            min_delta: 0.001,
            min_epochs: 10,
            ..TrainProgress::default()
        };
        for (e, v) in incident() {
            let mut ev = epoch(e, &[("map", v)]);
            if let TrainEvent::Epoch { epochs, .. } = &mut ev {
                *epochs = 100;
            }
            p.apply(&ev, f64::from(e));
        }
        assert_eq!((p.since_best, p.stop_epoch()), (10, 15));
        assert_eq!(p.extension, None);
        p.apply(&ev, 16.0);
        // The countdown restarts at the point.
        assert_eq!((p.since_best, p.stop_epoch()), (0, 25));
        let n = p.extension.clone().unwrap();
        assert_eq!(
            n.text,
            "Early stop reached at epoch 15 (best epoch 5, mAP 0.835); training loss is still \
             falling (3.569 \u{2192} 3.315). Continuing to epoch 25."
        );
        assert_eq!((n.keep_epoch, n.extension), (5, 1));
        assert!(
            p.describe()
                .contains("extended past epoch 15 (loss still falling, 1 of 2)"),
            "{}",
            p.describe()
        );
        // Round trip.
        let back: TrainProgress =
            serde_json::from_value(serde_json::to_value(&p).unwrap()).unwrap();
        assert_eq!(back, p);
        // A stop (Stop now, or the next point) ends the notice.
        p.apply(
            &TrainEvent::Stop {
                reason: "finish".into(),
                epoch: 17,
                best_epoch: 5,
                best: Some(0.835),
            },
            17.0,
        );
        assert_eq!(p.extension, None);
        // A stop event with the new fields still parses.
        assert!(matches!(
            parse_line(
                r#"@train {"event":"stop","reason":"converged","epoch":24,"best_epoch":12,"best":0.41,"loss_falling":false,"extensions":0}"#
            ),
            Some(TrainEvent::Stop { epoch: 24, .. })
        ));
    }

    #[test]
    fn an_anneal_parses_and_the_page_says_when_it_stops() {
        let ev = parse_line(
            r#"@train {"epoch":15,"until":20,"best_epoch":5,"best":0.835,"loss_falling":false,"extensions":1,"event":"anneal"}"#,
        )
        .unwrap();
        let TrainEvent::Anneal(a) = &ev else {
            panic!("{ev:?}")
        };
        assert_eq!(
            (a.epoch, a.until, a.best_epoch, a.extensions),
            (15, 20, 5, 1)
        );
        let mut p = TrainProgress {
            patience: 10,
            min_delta: 0.001,
            min_epochs: 10,
            ..TrainProgress::default()
        };
        for (e, v) in incident() {
            let mut ev = epoch(e, &[("map", v)]);
            if let TrainEvent::Epoch { epochs, .. } = &mut ev {
                *epochs = 100;
            }
            p.apply(&ev, f64::from(e));
        }
        // An earlier extension's notice gives way to the anneal's.
        p.apply(
            &TrainEvent::ConvergePoint(ConvergePoint {
                epoch: 5,
                best_epoch: 5,
                extension: 1,
                next_check: 15,
                ..ConvergePoint::default()
            }),
            15.0,
        );
        p.apply(&ev, 15.0);
        assert_eq!(p.stop_epoch(), 20);
        let n = p.extension.clone().unwrap();
        assert_eq!(
            n.text,
            "Early stop reached at epoch 15 (best epoch 5, mAP 0.835). Lowering the learning \
             rate until epoch 20, then stopping."
        );
        assert_eq!((n.keep_epoch, n.extension), (5, MAX_EXTENSIONS + 1));
        let d = p.describe();
        assert!(
            d.contains(", lowering the learning rate until epoch 20, then stopping")
                && !d.contains("without improvement")
                && !d.contains("extended past"),
            "{d}"
        );
        let back: TrainProgress =
            serde_json::from_value(serde_json::to_value(&p).unwrap()).unwrap();
        assert_eq!(back, p);
        // The stop after it (with the script's `anneal_until`) ends the notice.
        let stop = parse_line(
            r#"@train {"reason":"converged","epoch":20,"best_epoch":5,"best":0.835,"loss_falling":false,"extensions":1,"anneal_until":20,"event":"stop"}"#,
        )
        .unwrap();
        p.apply(&stop, 20.0);
        assert_eq!(p.extension, None);
        assert_eq!(p.stop_text().unwrap(), "converged at epoch 20 (best 5)");
    }

    /// The script's `Convergence` and `loss_falling` decide as [`convergence_run`] and
    /// [`loss_falling`] (skipped without Python 3).
    #[test]
    fn convergence_matches_the_training_script() {
        let script = include_str!("../../scripts/train_model.py");
        let cases: Vec<serde_json::Value> = sequences()
            .into_iter()
            .map(|c| {
                serde_json::json!({"epochs": c.epochs, "patience": c.rule.patience,
                    "min_epochs": c.rule.min_epochs, "min_delta": c.rule.min_delta,
                    "smooth": c.rule.smooth, "total": c.rule.epochs, "anneal": c.rule.anneal})
            })
            .collect();
        let losses: Vec<serde_json::Value> = loss_cases()
            .into_iter()
            .map(|(l, _)| serde_json::json!(l))
            .collect();
        let code = r#"
import json, sys
src, cases = sys.stdin.read().split("\n#CASES#\n")
cases = json.loads(cases)
ns = {"__name__": "train_model"}
exec(compile(src, "train_model.py", "exec"), ns)
out = []
for c in cases["runs"]:
    conv = ns["Convergence"](c["patience"], c["min_epochs"], c["min_delta"], None, c["smooth"], c["total"],
                             c["anneal"])
    stop, points, anneal = None, [], None
    for e, v, l in c["epochs"]:
        if conv.update(e, v, l):
            stop = e
            break
        if conv.point:
            points.append(e)
        if conv.annealing:
            anneal = e
    out.append({"stop": stop, "points": points, "anneal": anneal})
falling = [ns["loss_falling"](l) for l in cases["losses"]]
import pathlib, tempfile
d = pathlib.Path(tempfile.mkdtemp())
f = d / "stop-request"
c = ns["Convergence"](15, 10, 0.001, f)
fin = [c.update(1, 0.1)]
f.write_text("x")
fin += [c.update(2, 0.2), c.update(2, 0.3), c.update(3, 0.3), c.top_epoch]
f.unlink()
# The stop file finishes an anneal at once.
c = ns["Convergence"](2, 0, 0.0, f, 1, 0, 5)
af = [c.update(e, 0.5) for e in (1, 2, 3)] + [c.anneal_until]
f.write_text("x")
af += [c.update(4, 0.5)]
f.unlink()
d.rmdir()
# The anneal's learning rate bend, on a stand-in LambdaLR at step 10 over 10 steps to 0.1.
class Sched:
    last_epoch = 10
    lr_lambdas = [lambda s: 1.0, lambda s: 0.05, lambda s: 1.0 - s / 100.0]
sch = Sched()
ok = ns["bend_lambda_lr"](sch, 10, 0.1)
bend = [[f(s) for s in (8, 10, 12, 15, 20, 25)] for f in sch.lr_lambdas]
none = ns["bend_lambda_lr"](object(), 10, 0.1)
print(json.dumps({"runs": out, "falling": falling, "finish": fin, "anneal_finish": af, "bend": [ok, none, bend]}))
"#;
        let Some(python) = test_python::python3("convergence parity") else {
            return;
        };
        let input = format!(
            "{script}\n#CASES#\n{}",
            serde_json::json!({"runs": cases, "losses": losses})
        );
        let out = test_python::run(python, &["-I", "-c", code], &input);
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        // The stop file finishes before min_epochs, once.
        assert_eq!(
            v["finish"],
            serde_json::json!([null, "finish", null, null, 2])
        );
        // ... and an anneal at once (the stop point at 3 anneals until 8).
        assert_eq!(
            v["anneal_finish"],
            serde_json::json!([null, null, null, 8, "finish"])
        );
        type Run = (Option<u32>, Vec<u32>, Option<u32>);
        let got: Vec<Run> = v["runs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                serde_json::from_value(serde_json::json!([r["stop"], r["points"], r["anneal"]]))
                    .unwrap()
            })
            .collect();
        let want: Vec<Run> = sequences()
            .into_iter()
            .map(|c| {
                let r = convergence_run(&c.epochs, &c.rule);
                (r.stop, r.points, r.anneal)
            })
            .collect();
        assert_eq!(got, want);
        // The bend: a half cosine from the factor at step 10 down to 0.1 over 10 steps, never
        // above the schedule (0.05 stays), unchanged before; no lambdas, no bend.
        assert_eq!(v["bend"][0], true);
        assert_eq!(v["bend"][1], false);
        let bend = |f0: f64, lo: f64, s: f64| {
            let t = ((s - 10.0) / 10.0).clamp(0.0, 1.0);
            lo + (f0 - lo) * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())
        };
        let steps = [8.0, 10.0, 12.0, 15.0, 20.0, 25.0];
        let schedules: [&dyn Fn(f64) -> f64; 3] = [&|_| 1.0, &|_| 0.05, &|s| 1.0 - s / 100.0];
        for (i, orig) in schedules.iter().enumerate() {
            let f0 = orig(10.0);
            let lo = f0.min(0.1);
            for (j, &s) in steps.iter().enumerate() {
                let want = orig(s).min(bend(f0, lo, s));
                let got = v["bend"][2][i][j].as_f64().unwrap();
                assert!(
                    (got - want).abs() < 1e-12,
                    "lambda {i} step {s}: {got} != {want}"
                );
            }
        }
        assert!((v["bend"][2][0][2].as_f64().unwrap() - 0.914_057_647_468_726_3).abs() < 1e-12);
        assert_eq!(v["bend"][2][0][4], 0.1);
        assert_eq!(v["bend"][2][1][3], 0.05);
        let falling: Vec<bool> = serde_json::from_value(v["falling"].clone()).unwrap();
        let want: Vec<bool> = loss_cases().iter().map(|(l, _)| loss_falling(l)).collect();
        assert_eq!(falling, want);
    }

    /// The script's `match_detections(..., ious=)` on hand-computed boxes (a partial overlap, a
    /// box of exactly 0.5, one below it, a wrong class, two predictions on one ground truth), its
    /// return value unchanged, `mean_iou`, `IouMeter` and the trainer hooks' box readers (skipped
    /// without Python 3 or numpy).
    #[test]
    fn mean_iou_matches_hand_computed_boxes() {
        let script = include_str!("../../scripts/train_model.py");
        let code = r#"
import json, sys
import numpy as np
ns = {"__name__": "train_model"}
exec(compile(sys.stdin.read(), "train_model.py", "exec"), ns)
logs = []
ns["log"] = logs.append

def boxes(rows):
    return np.array(rows, dtype=np.float64).reshape(-1, 4)

def labels(rows):
    return np.array(rows, dtype=np.int64)

def scores(rows):
    return np.array(rows, dtype=np.float64)

preds = [
    # The 0.9 box takes the ground truth (IoU 0.8); the better 0.5 box finds it used. The 0.8 box
    # sits exactly on the class 1 ground truth under the wrong class. The class 1 box is at 0.4.
    (boxes([[0, 0, 10, 8], [0, 0, 10, 10], [20, 0, 30, 10], [20, 0, 30, 4]]), labels([0, 0, 0, 1]),
     scores([0.9, 0.5, 0.8, 0.7])),
    # IoU 0.6.
    (boxes([[0, 0, 10, 6]]), labels([1]), scores([0.6])),
    # Exactly the threshold, IoU 0.5.
    (boxes([[0, 0, 10, 5]]), labels([0]), scores([0.4])),
    # No ground truth in this image.
    (boxes([[0, 0, 10, 10]]), labels([0]), scores([0.9])),
    # No prediction in this image.
    (boxes([]), labels([]), scores([])),
]
gts = [
    (boxes([[0, 0, 10, 10], [20, 0, 30, 10]]), labels([0, 1])),
    (boxes([[0, 0, 10, 10]]), labels([1])),
    (boxes([[0, 0, 10, 10]]), labels([0])),
    (boxes([]), labels([])),
    (boxes([[0, 0, 10, 10]]), labels([0])),
]
ious = []
with_ious = ns["match_detections"](preds, gts, ious=ious)
plain = ns["match_detections"](preds, gts)
same = sorted(with_ious) == sorted(plain) and all(
    np.array_equal(with_ious[c][0], plain[c][0]) and np.array_equal(with_ious[c][1], plain[c][1])
    and with_ious[c][2] == plain[c][2] for c in plain)
mean = ns["mean_iou"](ious)

meter = ns["IouMeter"]()
meter.add(lambda: (preds, gts))
ok = meter.take()
after_take = meter.take()
def broken():
    raise KeyError("conf")
meter.add(lambda: (preds, gts))
meter.add(broken)
meter.add(broken)
meter.add(lambda: (preds, gts))
failed = meter.take()

# Stand-in tensor.
class T:
    def __init__(self, a):
        self.a = np.array(a)
    def detach(self):
        return self
    def float(self):
        return T(self.a.astype(np.float32))
    def cpu(self):
        return self
    def numpy(self):
        return self.a

pred = {"bboxes": T([[0, 0, 10, 8]]), "cls": T([2]), "conf": T([0.9])}
yolo = []
ns["match_detections"](*ns["ultralytics_boxes"](pred, {"bboxes": T([[0, 0, 10, 10]]), "cls": T([2])}), ious=yolo)
pred = {"boxes": T([[0, 0, 10, 8], [0, 0, 10, 6]]), "labels": T([2, 2]), "scores": T([0.9, 0.8])}
rfdetr = []
ns["match_detections"](*ns["rfdetr_boxes"]([pred], [{"boxes": T([[0, 0, 10, 10]]), "labels": T([2])}]), ious=rfdetr)
print(json.dumps({"ious": ious, "same": same, "mean": mean["iou"], "empty": ns["mean_iou"]([]), "ok": ok["iou"],
                  "after_take": after_take, "failed": failed, "logs": logs, "yolo": yolo, "rfdetr": rfdetr}))
"#;
        let Some(python) = test_python::python3_with_numpy("mean IoU") else {
            return;
        };
        let out = test_python::run(python, &["-I", "-c", code], script);
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        let close = |got: &serde_json::Value, want: f64, what: &str| {
            let got = got.as_f64().unwrap();
            assert!((got - want).abs() < 1e-6, "{what}: {got} != {want}");
        };
        let ious = v["ious"].as_array().unwrap();
        assert_eq!(ious.len(), 3, "{ious:?}");
        for (got, want) in ious.iter().zip([0.8, 0.6, 0.5]) {
            close(got, want, "matched IoU");
        }
        assert_eq!(v["same"], true);
        close(&v["mean"], (0.8 + 0.6 + 0.5) / 3.0, "mean");
        assert_eq!(v["empty"], serde_json::json!({}));
        close(&v["ok"], (0.8 + 0.6 + 0.5) / 3.0, "meter");
        // Reading empties the meter; a failed read logs once and leaves the metric out.
        assert_eq!(v["after_take"], serde_json::json!({}));
        assert_eq!(v["failed"], serde_json::json!({}));
        assert_eq!(v["logs"].as_array().unwrap().len(), 1, "{}", v["logs"]);
        assert!(
            v["logs"][0].as_str().unwrap().contains("mean IoU skipped"),
            "{}",
            v["logs"]
        );
        // The hooks' readers: both hand over boxes, classes and scores.
        close(&v["yolo"][0], 0.8, "ultralytics_boxes");
        assert_eq!(v["rfdetr"].as_array().unwrap().len(), 1);
        close(&v["rfdetr"][0], 0.8, "rfdetr_boxes");
    }

    /// The script's memory gate before the export (`memory_shortfall`, `wait_for_memory` with a
    /// fake clock): unknown values never block, the 3 GB / 4 GB limits, a wait that ends when
    /// memory frees, one that runs out, and a stop file that only counts when it appears during
    /// the wait (skipped without Python 3).
    #[test]
    fn export_memory_gate_of_the_training_script() {
        let script = include_str!("../../scripts/train_model.py");
        let code = r#"
import json, sys, pathlib, tempfile
ns = {"__name__": "train_model"}
exec(compile(sys.stdin.read(), "train_model.py", "exec"), ns)
G = 1 << 30
short = ns["memory_shortfall"]
out = {"short": [len(short(None, None)), len(short(3 * G - 1, None)), len(short(3 * G, 4 * G)),
                 len(short(3 * G - 1, 4 * G - 1)), len(short(None, 4 * G - 1))]}

def run(states, stop=None, make_stop_at=None):
    t = [0.0]
    it = iter(states)
    last = [None]
    def status():
        last[0] = next(it, last[0])
        return last[0]
    def sleep(s):
        t[0] += s
        if make_stop_at is not None and t[0] >= make_stop_at:
            stop.write_text("x")
    ok = ns["wait_for_memory"](stop, 300, 10, status=status, sleep=sleep, clock=lambda: t[0])
    return [ok, t[0]]

low, okm = {"ram": G, "commit": 8 * G}, {"ram": 4 * G, "commit": 8 * G}
out["freed"] = run([low, low, okm])
out["enough"] = run([okm])
out["timeout"] = run([low])
d = pathlib.Path(tempfile.mkdtemp())
f = d / "stop-request"
out["stop_during"] = run([low], f, 30)
out["stop_before"] = run([low], f)
f.unlink()
d.rmdir()
print("RESULT: " + json.dumps(out))
"#;
        let Some(python) = test_python::python3("export memory gate") else {
            return;
        };
        let out = test_python::run(python, &["-I", "-c", code], script);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let line = stdout
            .lines()
            .find_map(|l| l.strip_prefix("RESULT: "))
            .unwrap_or_else(|| panic!("no result in {stdout}"));
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(v["short"], serde_json::json!([0, 1, 0, 2, 1]));
        assert_eq!(v["freed"], serde_json::json!([true, 20.0]));
        assert_eq!(v["enough"], serde_json::json!([true, 0.0]));
        assert_eq!(v["timeout"], serde_json::json!([false, 300.0]));
        assert_eq!(v["stop_during"], serde_json::json!([false, 30.0]));
        assert_eq!(v["stop_before"], serde_json::json!([false, 300.0]));
        // The waits are reported as export phase events the service parses.
        assert!(
            stdout
                .lines()
                .filter_map(parse_line)
                .any(|e| matches!(e, TrainEvent::Phase { ref detail, .. } if detail.contains("waiting up to 5 min"))),
            "{stdout}"
        );
    }

    /// The script's learning-rate helpers and options (skipped without Python 3): Ultralytics'
    /// main group (not the bias group warming down from 0.1, not MuSGD's x3 head copy), unreadable
    /// optimizers give None, and `--lr-scheduler` / `--lr-min-factor` default to unset and are
    /// checked per trainer like [`crate::training::recipe::lr_schedule_error`]. A batch event with
    /// its `lr` parses here.
    #[test]
    fn learning_rate_helpers_and_options_of_the_training_script() {
        let script = include_str!("../../scripts/train_model.py");
        let code = r#"
import json, sys
ns = {"__name__": "train_model"}
exec(compile(sys.stdin.read(), "train_model.py", "exec"), ns)
class Opt:
    def __init__(self, groups):
        self.param_groups = groups
P = [object()]
warmup = Opt([
    {"params": P, "param_group": "weight", "lr": 0.0004, "initial_lr": 0.002},
    {"params": P, "param_group": "bn", "lr": 0.0004, "initial_lr": 0.002},
    {"params": P, "param_group": "bias", "lr": 0.08, "initial_lr": 0.002},
])
bias_first = Opt([warmup.param_groups[2], warmup.param_groups[0]])
muon = Opt([
    {"params": P, "param_group": "weight", "lr": 0.03, "initial_lr": 0.03},
    {"params": [], "param_group": "weight", "lr": 0.01, "initial_lr": 0.01},
    {"params": P, "param_group": "muon", "lr": 0.01, "initial_lr": 0.01},
    {"params": P, "param_group": "bias", "lr": 0.01, "initial_lr": 0.01},
])
plain = Opt([{"params": P, "lr": 5e-05}, {"params": P, "lr": 7.5e-05}])
class Sched:
    def get_last_lr(self):
        return [0.0001, 0.0002]
U = ns["ultralytics_lr"]
out = {
    "ultra": [U(warmup), U(bias_first), U(muon), U(None), U(Opt([{"params": P, "lr": float("nan")}]))],
    "group": [ns["group_lr"](plain), ns["group_lr"](plain, 1), ns["group_lr"](None)],
    "sched": [ns["scheduler_lr"](Sched()), ns["scheduler_lr"](None)],
}
base = ["--family", "yolo26", "--size", "n", "--name", "x", "--weights", "w", "--data", "d", "--out-dir", "o",
        "--work", "k", "--export-script", "e"]
def parse(*extra):
    sys.argv = ["train_model.py", *extra, *base]
    try:
        a = ns["parse_args"]()
        return [a.lr_scheduler, a.lr_min_factor]
    except SystemExit as e:
        return "error"
out["args"] = [
    parse("--trainer", "ultralytics"),
    parse("--trainer", "ultralytics", "--lr-scheduler", "cosine", "--lr-min-factor", "0.05"),
    parse("--trainer", "ultralytics", "--lr-scheduler", "step"),
    parse("--trainer", "rfdetr", "--lr-scheduler", "step", "--lr-min-factor", "0.2"),
    parse("--trainer", "rfdetr", "--lr-scheduler", "linear"),
    parse("--trainer", "transformers", "--lr-scheduler", "linear"),
    parse("--trainer", "transformers", "--lr-scheduler", "linear", "--lr-min-factor", "0.1"),
    parse("--trainer", "transformers", "--lr-min-factor", "1.5"),
]
sys.stdout.flush()
ns["event"]("batch", epoch=1, epochs=3, batch=2, batches=5, lr=U(warmup))
print("RESULT: " + json.dumps(out))
"#;
        let Some(python) = test_python::python3("learning rate helpers") else {
            return;
        };
        let out = test_python::run(python, &["-I", "-c", code], script);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let line = stdout
            .lines()
            .find_map(|l| l.strip_prefix("RESULT: "))
            .unwrap_or_else(|| panic!("no result in {stdout}"));
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        use serde_json::json;
        assert_eq!(v["ultra"], json!([0.0004, 0.0004, 0.01, null, null]));
        assert_eq!(v["group"], json!([5e-05, 7.5e-05, null]));
        assert_eq!(v["sched"], json!([0.0001, null]));
        assert_eq!(
            v["args"],
            json!([
                [null, null],
                ["cosine", 0.05],
                "error",
                ["step", 0.2],
                "error",
                ["linear", null],
                "error",
                "error"
            ])
        );
        let batch = stdout.lines().find_map(parse_line);
        assert_eq!(
            batch,
            Some(TrainEvent::Batch {
                epoch: 1,
                epochs: 3,
                batch: 2,
                batches: 5,
                lr: Some(0.0004)
            })
        );
    }

    fn select_settings(select: SelectMetric) -> RunSettings {
        RunSettings {
            lr: None,
            lr_encoder: None,
            lr_schedule: crate::training::recipe::LrSchedule::Linear,
            lr_min_factor: 0.01,
            select_metric: select,
            epochs: 30,
            patience: 3,
            patience_doubled: false,
            batch: 8,
            workers: None,
            seed: None,
            scene: None,
            gpu: None,
        }
    }

    /// `(epoch, loss, raw metrics)`.
    type SelectRow = (u32, f64, Vec<(&'static str, f64)>);

    /// A progress selecting by `select` that saw `rows`.
    fn selecting(select: SelectMetric, rows: &[SelectRow]) -> TrainProgress {
        let mut p = TrainProgress {
            patience: 3,
            min_epochs: 2,
            min_delta: 0.001,
            smooth: 1,
            settings: Some(select_settings(select)),
            ..TrainProgress::default()
        };
        for (e, loss, m) in rows {
            p.apply(
                &TrainEvent::Epoch {
                    epoch: *e,
                    epochs: 30,
                    loss: Some(*loss),
                    map50: None,
                    map: None,
                    elapsed_s: None,
                    metrics: m.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
                },
                f64::from(*e),
            );
        }
        p
    }

    /// mAP peaks at epoch 2, F2 at 4, F0.5 at 1, IoU at 3; F2 is missing at epoch 5 (nothing
    /// matched), the last row has no validation.
    fn select_rows() -> Vec<SelectRow> {
        let row = |map, map50, f1, f2, f05, iou| {
            vec![
                ("metrics/mAP50-95(B)", map),
                ("metrics/mAP50(B)", map50),
                ("f1", f1),
                ("f2", f2),
                ("f0_5", f05),
                ("iou", iou),
            ]
        };
        vec![
            (1, 3.0, row(0.30, 0.61, 0.50, 0.40, 0.70, 0.60)),
            (2, 2.9, row(0.50, 0.62, 0.52, 0.45, 0.60, 0.62)),
            (3, 2.8, row(0.40, 0.70, 0.60, 0.50, 0.50, 0.80)),
            (4, 2.7, row(0.45, 0.65, 0.65, 0.90, 0.40, 0.70)),
            (
                5,
                2.6,
                vec![("metrics/mAP50-95(B)", 0.44), ("metrics/mAP50(B)", 0.64)],
            ),
            (6, 2.5, vec![("val_loss", 1.0)]),
        ]
    }

    #[test]
    fn best_and_the_rule_follow_the_selected_metric() {
        let want = [
            (SelectMetric::Map, "map", 2, 0.50),
            (SelectMetric::Map50, "map50", 3, 0.70),
            (SelectMetric::F1, "f1", 4, 0.65),
            (SelectMetric::F2, "f2", 4, 0.90),
            (SelectMetric::F05, "f0_5", 1, 0.70),
            (SelectMetric::Iou, "iou", 3, 0.80),
        ];
        for (select, id, epoch, value) in want {
            let p = selecting(select, &select_rows());
            assert_eq!(p.select_metric(), select);
            assert_eq!(
                p.best,
                Some(Best {
                    metric: id.into(),
                    epoch,
                    value
                }),
                "{id}"
            );
            // The last row has no value, so it counts as an epoch without a gain.
            assert_eq!(p.since_best, 6 - epoch, "{id}");
        }
        let p = selecting(SelectMetric::F2, &select_rows());
        // A validated epoch without the key is 0 (the script's pick_metric); no validation, none.
        let v = rule_values(&p.history, SelectMetric::F2);
        assert_eq!(v[4], (5, Some(0.0)));
        assert_eq!(v[5], (6, None));
        assert_eq!(rule_values(&p.history, SelectMetric::Map)[5], (6, None));
        assert!(
            p.describe().contains("best F2 0.900 @ epoch 4"),
            "{}",
            p.describe()
        );
        // The notices name the metric and keep its best epoch; a default run's are as before.
        for (select, kept) in [
            (SelectMetric::F2, "best epoch 4, F2 0.900"),
            (SelectMetric::Iou, "best epoch 3, IoU 0.800"),
            (SelectMetric::Map, "best epoch 2, mAP 0.500"),
        ] {
            let mut p = selecting(select, &select_rows());
            p.apply(
                &TrainEvent::ConvergePoint(ConvergePoint {
                    epoch: 6,
                    best_epoch: 1,
                    best: Some(0.1),
                    loss_falling: true,
                    loss_from: Some(3.0),
                    loss_to: Some(2.5),
                    extension: 1,
                    next_check: 9,
                }),
                6.0,
            );
            let text = &p.extension.as_ref().unwrap().text;
            assert!(text.contains(kept), "{text}");
        }
    }

    #[test]
    fn a_record_without_the_setting_selects_by_map_as_before() {
        // A run recorded before the setting: settings without `select_metric`, or none at all.
        let mut s = serde_json::to_value(select_settings(SelectMetric::Map)).unwrap();
        s.as_object_mut().unwrap().remove("select_metric");
        let old: RunSettings = serde_json::from_value(s).unwrap();
        assert_eq!(old.select_metric, SelectMetric::Map);
        let p: TrainProgress = serde_json::from_value(serde_json::json!({
            "epoch": 2, "epochs": 5,
            "history": [{"epoch": 1, "metrics": {"map": 0.4, "f2": 0.9}},
                        {"epoch": 2, "metrics": {"map": 0.5, "f2": 0.1}}],
            "best": {"metric": "map", "epoch": 2, "value": 0.5},
            "settings": serde_json::to_value(&old).unwrap(),
        }))
        .unwrap();
        assert_eq!(p.select_metric(), SelectMetric::Map);
        assert_eq!(p.select_fallback, None);
        let mut none = p.clone();
        none.settings = None;
        for mut q in [p, none] {
            q.apply(
                &TrainEvent::Eval {
                    epoch: 2,
                    val_loss: Some(1.0),
                    metrics: BTreeMap::new(),
                },
                2.0,
            );
            assert_eq!(q.best.as_ref().unwrap().metric, "map");
            assert_eq!(q.best.as_ref().unwrap().epoch, 2);
            assert!(
                serde_json::to_value(&q)
                    .unwrap()
                    .get("select_fallback")
                    .is_none()
            );
        }
    }

    #[test]
    fn a_fallback_selects_by_map_from_its_epoch() {
        let mut p = selecting(SelectMetric::F2, &select_rows());
        assert_eq!(p.best.as_ref().unwrap().metric, "f2");
        let ev = parse_line(
            r#"@train {"metric":"f2","epoch":3,"reason":"the F-beta sweep failed","event":"select_fallback"}"#,
        )
        .unwrap();
        p.apply(&ev, 6.0);
        assert_eq!(p.select_metric(), SelectMetric::Map);
        // Epochs before it no longer count: mAP's own best of the rest, not epoch 2's 0.50.
        assert_eq!(
            p.best,
            Some(Best {
                metric: "map".into(),
                epoch: 4,
                value: 0.45
            })
        );
        assert_eq!(p.since_best, 2);
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v["select_fallback"]["metric"], "f2");
        let back: TrainProgress = serde_json::from_value(v).unwrap();
        assert_eq!(back.select_metric(), SelectMetric::Map);
    }

    /// `Convergence` fed by the script's `pick_metric` decides as [`convergence_run`] over
    /// [`rule_values`], with the same best epoch as [`best_of`], for every metric of
    /// `SELECT_METRICS`; both lists and the names agree; `Selection` falls back once (skipped
    /// without Python 3).
    #[test]
    fn selected_metric_matches_the_training_script() {
        use crate::training::recipe::SELECT_METRICS;
        // A long run: the metrics peak at different epochs, F2 and IoU go missing at some.
        let rows: Vec<SelectRow> = (1..=40u32)
            .map(|e| {
                let t = f64::from(e);
                let wave = |p: f64, q: f64| 0.5 + 0.3 * (t / p).sin() - t * q;
                let mut m = vec![
                    ("metrics/mAP50-95(B)", wave(5.0, 0.002)),
                    ("metrics/mAP50(B)", wave(4.0, 0.001)),
                    ("f1", wave(6.0, 0.0)),
                    ("f0_5", wave(3.0, 0.004)),
                ];
                if e % 7 != 0 {
                    m.push(("f2", wave(7.0, 0.003)));
                    m.push(("iou", wave(2.5, 0.0)));
                }
                (e, 4.0 - t * if e < 20 { 0.08 } else { 0.0 }, m)
            })
            .collect();
        let rule = Rule {
            patience: 6,
            min_epochs: 10,
            min_delta: 0.001,
            smooth: 3,
            epochs: 40,
            anneal: 3,
        };
        let raw: Vec<serde_json::Value> = rows
            .iter()
            .map(|(e, l, m)| {
                let metrics: BTreeMap<&str, f64> = m.iter().copied().collect();
                serde_json::json!({"epoch": e, "loss": l, "metrics": metrics})
            })
            .collect();
        let shorts: BTreeMap<&str, &str> = SELECT_METRICS
            .iter()
            .map(|o| (o.metric.id(), o.metric.short()))
            .collect();
        let ids: Vec<&str> = SELECT_METRICS.iter().map(|o| o.metric.id()).collect();
        let data = serde_json::json!({"rows": raw, "ids": ids, "patience": rule.patience,
            "min_epochs": rule.min_epochs, "min_delta": rule.min_delta, "smooth": rule.smooth,
            "total": rule.epochs, "anneal": rule.anneal});
        let code = r#"
import io, json, sys
from contextlib import redirect_stdout
src, data = sys.stdin.read().split("\n#CASES#\n")
d = json.loads(data)
ns = {"__name__": "train_model"}
exec(compile(src, "train_model.py", "exec"), ns)
Pick, Sel, Conv = ns["pick_metric"], ns["Selection"], ns["Convergence"]
out = {"ids": list(ns["SELECT_METRICS"]), "labels": ns["SELECT_LABELS"], "runs": {}}
for id in d["ids"]:
    c = Conv(d["patience"], d["min_epochs"], d["min_delta"], None, d["smooth"], d["total"], d["anneal"])
    stop, points, anneal = None, [], None
    for r in d["rows"]:
        decision = c.update(r["epoch"], Pick(r["metrics"], id), r["loss"])
        if c.point is not None:
            points.append(c.point["epoch"])
        if c.annealing is not None:
            anneal = c.annealing["epoch"]
        if decision:
            stop = r["epoch"]
            break
    out["runs"][id] = {"stop": stop, "points": points, "anneal": anneal, "top_epoch": c.top_epoch}
nan = float("nan")
out["pick"] = [
    Pick({"metrics/mAP50-95(B)": 0.5, "metrics/mAP50(B)": 0.7}, "map"),
    Pick({"metrics/mAP50(B)": 0.7}, "map"),
    Pick({"val/ema_mAP_50_95": 0.4}, "map"),
    Pick({"map": 0.3, "map50": 0.6}, "map"),
    Pick({}, "map"),
    Pick({"val/mAP_50": 0.6}, "map50"),
    Pick({"metrics/mAP50-95(B)": 0.5}, "map50"),
    Pick({"val/F1": 0.55}, "f1"),
    Pick({"f1": 0.25}, "f1"),
    Pick({"f2": 0.8, "map": 0.1}, "f2"),
    Pick({"f0_5": 0.2}, "f0_5"),
    Pick({"iou": 0.9}, "iou"),
    Pick({"map": 0.5}, "f2"),
    Pick({"map": 0.5}, "iou"),
    Pick({"f1": nan}, "f1"),
]
buf = io.StringIO()
with redirect_stdout(buf):
    c = Conv(2, 0, 0.0, None, 1, 0, 0, ns["SELECT_LABELS"]["f2"])
    s = Sel("f2", c)
    values = [s.value({"f2": 0.9, "map": 0.2}, 1, sweep_failed=False)]
    c.update(1, values[-1])
    before = [c.top, c.top_epoch, c.label]
    values.append(s.value({"map": 0.3}, 2, sweep_failed=True, iou_failed=True))
    after = [c.top, c.top_epoch, c.label, s.metric, s.requested, s.label]
    c.update(2, values[-1])
    values.append(s.value({"map": 0.25}, 3, sweep_failed=True))
    c.update(3, values[-1])
    final = [c.top, c.top_epoch]
text = buf.getvalue()
out["fallback"] = {"values": values, "before": before, "after": after, "final": final, "text": text}
buf = io.StringIO()
with redirect_stdout(buf):
    kept = [Sel("f1").value({"f1": 0.5}, 1, iou_failed=True), Sel("iou").value({"iou": 0.6}, 1, sweep_failed=True),
            Sel("map").value({"map": 0.7}, 1, sweep_failed=True, iou_failed=True),
            Sel("map50").value({"map50": 0.8}, 1, sweep_failed=True, iou_failed=True)]
    iou = Sel("iou")
    kept.append(iou.value({"map": 0.4, "iou": 0.9}, 4, iou_failed=True))
    kept.append(iou.metric)
out["kept"] = kept
out["kept_text"] = buf.getvalue()
def stop_text(label):
    buf = io.StringIO()
    c = Conv(2, 0, 0.0, None, 3, 0, 0, label)
    with redirect_stdout(buf):
        for e in range(1, 5):
            if c.step(e, 0.5, None):
                break
    return buf.getvalue()
out["stop"] = {"default": stop_text("mAP"), "f2": stop_text("F2")}
base = ["--family", "yolo26", "--size", "n", "--name", "x", "--weights", "w", "--data", "d", "--out-dir", "o",
        "--work", "k", "--export-script", "e", "--trainer", "ultralytics"]
def parse(*extra):
    sys.argv = ["train_model.py", *extra, *base]
    try:
        return ns["parse_args"]().select_metric
    except SystemExit:
        return "error"
with redirect_stdout(io.StringIO()):
    sys.stderr = io.StringIO()
    out["args"] = [parse(), parse("--select-metric", "f0_5"), parse("--select-metric", "f3")]
    sys.stderr = sys.__stderr__
print("RESULT: " + json.dumps(out))
"#;
        let Some(python) = test_python::python3("selected metric") else {
            return;
        };
        let script = include_str!("../../scripts/train_model.py");
        let input = format!("{script}\n#CASES#\n{data}");
        let out = test_python::run(python, &["-I", "-c", code], &input);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let line = stdout
            .lines()
            .find_map(|l| l.strip_prefix("RESULT: "))
            .unwrap_or_else(|| panic!("no result in {stdout}"));
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        use serde_json::json;
        assert_eq!(v["ids"], json!(ids));
        assert_eq!(v["labels"], json!(shorts));
        let mut stops = 0;
        for o in SELECT_METRICS {
            let id = o.metric.id();
            let mut p = selecting(o.metric, &rows);
            p.min_delta = rule.min_delta;
            let values = rule_values(&p.history, o.metric);
            let run_rows: Vec<_> = values
                .iter()
                .zip(&rows)
                .map(|(&(e, x), (_, l, _))| (e, x, Some(*l)))
                .collect();
            let r = convergence_run(&run_rows, &rule);
            let got = &v["runs"][id];
            assert_eq!(got["stop"], json!(r.stop), "{id}");
            assert_eq!(got["points"], json!(r.points), "{id}");
            assert_eq!(got["anneal"], json!(r.anneal), "{id}");
            stops += usize::from(r.stop.is_some());
            // The best epoch over what the script saw up to its stop.
            let seen = r.stop.unwrap_or(rule.epochs);
            let cut: Vec<EpochPoint> = p
                .history
                .iter()
                .filter(|h| h.epoch <= seen)
                .cloned()
                .collect();
            let best = best_of(&cut, o.metric).unwrap();
            assert_eq!(got["top_epoch"], json!(best.epoch), "{id}");
        }
        assert!(stops >= 3, "the rule should stop for most metrics: {v}");
        assert_eq!(
            v["pick"],
            json!([
                0.5, 0.7, 0.4, 0.3, null, 0.6, null, 0.55, 0.25, 0.8, 0.2, 0.9, 0.0, 0.0, 0.0
            ])
        );
        // The fall-back: the epoch's mAP, the rule starts again on it, once.
        let f = &v["fallback"];
        assert_eq!(f["values"], json!([0.9, 0.3, 0.25]));
        assert_eq!(f["before"], json!([0.9, 1, "F2"]));
        assert_eq!(f["after"], json!([null, 0, "mAP", "map", "f2", "mAP"]));
        assert_eq!(f["final"], json!([0.3, 2]));
        let text = f["text"].as_str().unwrap();
        assert_eq!(
            text.matches("--select-metric f2: the F-beta sweep failed")
                .count(),
            1
        );
        assert_eq!(text.matches("select_fallback").count(), 1);
        let ev = text.lines().find_map(parse_line).unwrap();
        assert_eq!(
            ev,
            TrainEvent::SelectFallback(SelectFallback {
                metric: "f2".into(),
                epoch: 2,
                reason: "the F-beta sweep failed".into()
            })
        );
        // A failed source that the metric does not read changes nothing.
        assert_eq!(v["kept"], json!([0.5, 0.6, 0.7, 0.8, 0.4, "map"]));
        let kept = v["kept_text"].as_str().unwrap();
        assert_eq!(kept.matches("select_fallback").count(), 1, "{kept}");
        assert!(
            kept.contains("--select-metric iou: the mean IoU failed"),
            "{kept}"
        );
        // The log names the metric; the default's text is as it always was.
        let d = v["stop"]["default"].as_str().unwrap();
        assert!(
            d.contains(
                "stopping after epoch 3: no mAP gain above 0 for 2 epochs (mAP averaged over the \
                 last 3 epochs); training loss no longer falling; best epoch 1 (mAP 0.5)"
            ),
            "{d}"
        );
        let f2 = v["stop"]["f2"].as_str().unwrap();
        assert!(
            f2.contains("no F2 gain above 0 for 2 epochs (F2 averaged over the last 3 epochs)"),
            "{f2}"
        );
        assert!(f2.contains("best epoch 1 (F2 0.5)"), "{f2}");
        assert_eq!(v["args"], json!(["map", "f0_5", "error"]));
    }
}
