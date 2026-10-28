//! The training script's progress events (`@train {json}` lines on its stdout, see
//! `scripts/train_model.py`) and the run's progress derived from them ([`TrainProgress`]:
//! percent of the training stage, epoch, ETA, every scalar metric per epoch in
//! [`TrainProgress::history`], the best epoch and the convergence stop, whose rule
//! [`convergence_decision`] mirrors the script's `Convergence`). Raw metric keys of the trainers (rfdetr,
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

/// The epoch the convergence rule stops at for the per-epoch validation mAP `values` (in epoch
/// order; a repeated or older epoch is ignored), None when it does not stop. An improvement is a
/// value above the best so far by more than `min_delta`; the run stops once `epoch >= min_epochs`
/// and `patience` epochs passed since the last improvement (0 = never). The twin of the script's
/// `Convergence` (parity-tested).
pub fn convergence_decision(
    values: &[(u32, f64)],
    patience: u32,
    min_epochs: u32,
    min_delta: f64,
) -> Option<u32> {
    let mut t = Tracker {
        min_delta,
        ..Tracker::default()
    };
    values.iter().find_map(|&(epoch, v)| {
        t.update(epoch, Some(v))?;
        let stale = t.best.is_some() && epoch - t.best_epoch >= patience;
        (patience > 0 && epoch >= min_epochs && stale).then_some(epoch)
    })
}

/// The rule's state: the last epoch seen and the last improvement.
#[derive(Debug, Default)]
struct Tracker {
    last: u32,
    best: Option<f64>,
    best_epoch: u32,
    min_delta: f64,
}

impl Tracker {
    /// Fold one epoch in; None for a repeated or older epoch.
    fn update(&mut self, epoch: u32, v: Option<f64>) -> Option<()> {
        if epoch <= self.last {
            return None;
        }
        self.last = epoch;
        if let Some(v) = v.filter(|v| v.is_finite())
            && self.best.is_none_or(|b| v > b + self.min_delta)
        {
            self.best = Some(v);
            self.best_epoch = epoch;
        }
        Some(())
    }
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
    /// Epochs since the last improvement of the rule's metric (mAP 50-95, else mAP 50): the run
    /// stops when this reaches `patience` (past the minimum epochs).
    pub since_best: u32,
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
            TrainEvent::SelfCheck { .. } | TrainEvent::Done { .. } => {}
        }
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

    /// Epochs since the rule's last improvement over the history (its metric: `map` when any
    /// epoch has it, else `map50`; 0 without either).
    fn since_improvement(&self) -> u32 {
        let id = if self.history.iter().any(|p| p.metrics.contains_key("map")) {
            "map"
        } else {
            "map50"
        };
        let mut t = Tracker {
            min_delta: self.min_delta,
            ..Tracker::default()
        };
        for p in &self.history {
            t.update(p.epoch, p.metrics.get(id).copied());
        }
        if t.best.is_none() {
            return 0;
        }
        t.last.saturating_sub(t.best_epoch)
    }

    /// The epoch the rule stops after when nothing improves from now on:
    /// `max(min_epochs, last improvement + patience)`, at most the run's epochs.
    pub fn stop_epoch(&self) -> u32 {
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
        if self.patience > 0 && self.patience < self.epochs && self.since_best > 0 {
            s.push_str(&format!(
                ", {} without improvement (stops after epoch {} if no gain)",
                plural(self.since_best, "epoch"),
                self.stop_epoch()
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

    /// `(values, patience, min_epochs, min_delta, stop epoch)`.
    type Case = (Vec<(u32, f64)>, u32, u32, f64, Option<u32>);

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
        vec![
            (ramp.clone(), 3, 10, 0.001, None),
            (plateau.clone(), 5, 10, 0.001, Some(16)),
            (plateau, 0, 0, 0.0, None),
            (early.clone(), 2, 0, 0.0, Some(3)),
            (early, 2, 5, 0.0, None),
            (noisy, 3, 0, 0.001, Some(7)),
        ]
    }

    #[test]
    fn convergence_rule_stops_where_expected() {
        for (i, (v, patience, min_epochs, delta, want)) in sequences().into_iter().enumerate() {
            assert_eq!(
                convergence_decision(&v, patience, min_epochs, delta),
                want,
                "sequence {i}"
            );
        }
        assert_eq!(convergence_decision(&[], 1, 0, 0.0), None);
    }

    /// The script's `Convergence` decides as [`convergence_decision`] (skipped without python3).
    #[test]
    fn convergence_matches_the_training_script() {
        let script = include_str!("../../scripts/train_model.py");
        let cases: Vec<serde_json::Value> = sequences()
            .into_iter()
            .map(|(v, p, m, d, _)| serde_json::json!({"values": v, "patience": p, "min_epochs": m, "min_delta": d}))
            .collect();
        let code = r#"
import json, sys
src, cases = sys.stdin.read().split("\n#CASES#\n")
ns = {"__name__": "train_model"}
exec(compile(src, "train_model.py", "exec"), ns)
out = []
for c in json.loads(cases):
    conv = ns["Convergence"](c["patience"], c["min_epochs"], c["min_delta"], None)
    stop = None
    for e, v in c["values"]:
        if conv.update(e, v):
            stop = e
            break
    out.append(stop)
import pathlib, tempfile
d = pathlib.Path(tempfile.mkdtemp())
f = d / "stop-request"
c = ns["Convergence"](15, 10, 0.001, f)
fin = [c.update(1, 0.1)]
f.write_text("x")
fin += [c.update(2, 0.2), c.update(2, 0.3), c.update(3, 0.3), c.top_epoch]
f.unlink()
d.rmdir()
print(json.dumps({"stops": out, "finish": fin}))
"#;
        let mut child = match std::process::Command::new("python3")
            .args(["-I", "-c", code])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("python3 not found ({e}); parity test skipped");
                return;
            }
        };
        let input = format!("{script}\n#CASES#\n{}", serde_json::Value::from(cases));
        {
            use std::io::Write as _;
            let mut stdin = child.stdin.take().unwrap();
            stdin.write_all(input.as_bytes()).unwrap();
        }
        let out = child.wait_with_output().unwrap();
        assert!(out.status.success(), "python3 failed");
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        // The stop file finishes before min_epochs, once.
        assert_eq!(
            v["finish"],
            serde_json::json!([null, "finish", null, null, 2])
        );
        let got: Vec<Option<u32>> = serde_json::from_value(v["stops"].clone()).unwrap();
        let want: Vec<Option<u32>> = sequences()
            .into_iter()
            .map(|(v, p, m, d, _)| convergence_decision(&v, p, m, d))
            .collect();
        assert_eq!(got, want);
    }
}
