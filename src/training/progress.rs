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
    },
    Batch {
        epoch: u32,
        epochs: u32,
        batch: u32,
        batches: u32,
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

/// The best epoch on the primary metric (`map`, else `map50`, else the lowest `val_loss`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Best {
    pub metric: String,
    pub epoch: u32,
    pub value: f64,
}

/// The primary metrics, in order of preference, with "larger is better".
const PRIMARY: [(&str, bool); 3] = [("map", true), ("map50", true), ("val_loss", false)];

/// The best epoch of `history` (see [`Best`]); the earliest epoch wins a tie.
pub fn best_of(history: &[EpochPoint]) -> Option<Best> {
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
    /// The epoch whose checkpoint a stop now keeps (highest raw mAP).
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
    /// The epoch whose checkpoint a stop now keeps (highest raw mAP so far).
    pub best_epoch: u32,
    pub best: Option<f64>,
    pub loss_falling: bool,
    pub extensions: u32,
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
    /// Smallest gain of the smoothed mAP that counts.
    pub min_delta: f64,
    /// The gain is measured on the mean of the last `smooth` mAP values (0 and 1 = the raw value).
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

/// The convergence rule over `(epoch, validation mAP, training loss)` in epoch order (a
/// repeated or older epoch is ignored): a stop point once `epoch >= min_epochs` and `patience`
/// epochs passed since the last gain of more than `min_delta` in the mean of the last `smooth`
/// mAP values (or since the last extension). At a stop point the run goes on while the loss
/// still falls over the last `patience` epochs ([`loss_falling`]), at most [`MAX_EXTENSIONS`]
/// times and before its last epoch; else, with `anneal` > 0 and before its last epoch, it
/// anneals until `epoch + anneal` (at most `epochs`; later stop points are ignored) and stops
/// there; else it stops. The twin of the script's `Convergence` (parity-tested).
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
/// the per-epoch validation mAP `values`, after an anneal of `anneal` epochs; None when it does
/// not stop.
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
    /// The rule's smoothing window (mean of the last N mAP values; 0 = 1), set with `patience`.
    #[serde(skip_serializing_if = "is_zero_u32")]
    pub smooth: u32,
    /// Epochs since the last improvement of the rule's metric (mAP 50-95, else mAP 50), or since
    /// the last extension: the run reaches a stop point when this reaches `patience` (past the
    /// minimum epochs).
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
            TrainEvent::Start { device, epochs, .. } => {
                self.device = Some(device.clone());
                self.epochs = *epochs;
            }
            TrainEvent::Batch {
                epoch,
                epochs,
                batch,
                batches,
            } => {
                self.epoch = *epoch;
                self.epochs = *epochs;
                self.batch = *batch;
                self.batches = *batches;
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
            TrainEvent::SelfCheck { .. } | TrainEvent::Done { .. } => {}
        }
        self.extension = self.extension_notice();
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
        self.best = best_of(&self.history);
        self.since_best = self.since_improvement();
    }

    /// The rule's state over the history (its metric: `map` when any epoch has it, else `map50`).
    fn rule_state(&self) -> Tracker {
        let id = if self.history.iter().any(|p| p.metrics.contains_key("map")) {
            "map"
        } else {
            "map50"
        };
        let mut t = Tracker::new(self.min_delta, self.smooth);
        for p in &self.history {
            t.update(p.epoch, p.metrics.get(id).copied());
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
        let kept = |epoch: u32, value: Option<f64>| {
            let (keep, best) = match t.top {
                Some(v) if t.top_epoch > 0 => (t.top_epoch, Some(v)),
                _ => (epoch, value),
            };
            (
                keep,
                best.map_or_else(String::new, |b| format!(", mAP {b:.3}")),
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
            let label = match b.metric.as_str() {
                "map" => "mAP50-95",
                "map50" => "mAP50",
                "val_loss" => "val loss",
                other => other,
            };
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

fn plural(n: u32, word: &str) -> String {
    format!("{n} {word}{}", if n == 1 { "" } else { "s" })
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
                batches: 5
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
    fn progress_and_eta_follow_the_batches() {
        let mut p = TrainProgress::default();
        p.apply(
            &TrainEvent::Start {
                device: "cuda:0".into(),
                train_images: 100,
                valid_images: 10,
                epochs: 4,
            },
            0.0,
        );
        assert_eq!(p.fraction(), 0.0);
        assert_eq!(p.eta_s, None);
        p.apply(
            &TrainEvent::Batch {
                epoch: 1,
                epochs: 4,
                batch: 5,
                batches: 10,
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
        assert_eq!(best_of(&p.history).unwrap().metric, "map");
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
        assert_eq!(best_of(&h).unwrap().epoch, 1);
        assert_eq!(best_of(&[]), None);
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

    /// A Python 3 to run the script's code with (`python3`, else `python`); None without one.
    fn python3() -> Option<&'static str> {
        ["python3", "python"].into_iter().find(|p| {
            std::process::Command::new(p)
                .args([
                    "-I",
                    "-c",
                    "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)",
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        })
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
        let Some(python) = python3() else {
            eprintln!("no Python 3 found; parity test skipped");
            return;
        };
        let mut child = std::process::Command::new(python)
            .args(["-I", "-c", code])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let input = format!(
            "{script}\n#CASES#\n{}",
            serde_json::json!({"runs": cases, "losses": losses})
        );
        {
            use std::io::Write as _;
            let mut stdin = child.stdin.take().unwrap();
            stdin.write_all(input.as_bytes()).unwrap();
        }
        let out = child.wait_with_output().unwrap();
        assert!(out.status.success(), "{python} failed");
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
        let Some(python) = python3() else {
            eprintln!("no Python 3 found; memory gate test skipped");
            return;
        };
        let mut child = std::process::Command::new(python)
            .args(["-I", "-c", code])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        {
            use std::io::Write as _;
            let mut stdin = child.stdin.take().unwrap();
            stdin.write_all(script.as_bytes()).unwrap();
        }
        let out = child.wait_with_output().unwrap();
        assert!(out.status.success(), "{python} failed");
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
}
