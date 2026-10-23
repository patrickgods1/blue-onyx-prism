//! The training script's progress events (`@train {json}` lines on its stdout, see
//! `scripts/train_model.py`) and the run's progress derived from them ([`TrainProgress`]:
//! percent of the training stage, epoch, ETA). Pure.

use serde::{Deserialize, Serialize};

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
    },
    Eval {
        #[serde(default)]
        epoch: u32,
        #[serde(default)]
        val_loss: Option<f64>,
    },
    Phase {
        phase: String,
        #[serde(default)]
        detail: String,
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
                ..
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
            }
            TrainEvent::Eval { val_loss, .. } => {
                if val_loss.is_some() {
                    self.val_loss = *val_loss;
                }
            }
            TrainEvent::Phase { phase, .. } => self.phase = Some(phase.clone()),
            TrainEvent::SelfCheck { .. } | TrainEvent::Done { .. } => {}
        }
        let f = self.fraction();
        self.eta_s = (f > 0.0 && f < 1.0).then(|| elapsed_s * (1.0 - f) / f);
        if self.phase.is_some() {
            self.eta_s = None;
        }
    }

    /// Share of the training done, 0..1: finished epochs plus the share of the current one.
    pub fn fraction(&self) -> f64 {
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

    /// "epoch 3/50, batch 12/40, loss 2.31, about 40 min left".
    pub fn describe(&self) -> String {
        if let Some(p) = &self.phase {
            return format!("{p}: exporting the trained model to ONNX and checking it");
        }
        if self.epochs == 0 {
            return "starting".into();
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
        if let Some(e) = self.eta_s {
            s.push_str(&format!(
                ", about {} left",
                super::estimate::format_duration(e)
            ));
        }
        s
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
            },
            20.0,
        );
        assert!((p.fraction() - 0.25).abs() < 1e-12);
        assert_eq!(p.losses, [2.5]);
        assert!(
            p.describe()
                .starts_with("epoch 1/4, loss 2.500, val mAP50 0.300, about 60 s left"),
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
}
