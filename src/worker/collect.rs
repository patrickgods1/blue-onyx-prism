//! The worker's side of the dataset collector ([`crate::collect`]): start a request's
//! [`Draft`] in `finish`, carry it through the verify lane, and hand it over after the reply.
//! Everything here is cheap bookkeeping on the worker or lane thread (a few small vectors and
//! a `Bytes` reference); decoding, hashing and writing happen on the collector thread.

use super::pipeline::Meta;
use crate::api::Prediction;
use crate::collect::record::{VerifyCandidate, VerifyInfo};
use crate::collect::{CollectSettings, Draft, DraftInput, Frame};
use crate::model::Detection;
use crate::model::filters::SuppressReason;
use crate::verify::{Outcome, VerifyConfig};
use std::sync::Arc;

/// A request's draft with what [`Job::submit`] needs.
#[derive(Debug)]
pub(super) struct Job {
    pub(super) draft: Draft,
    pub(super) image: bytes::Bytes,
    pub(super) settings: Arc<CollectSettings>,
}

impl Job {
    /// The job of a request this worker answers (collection on, not a verifier check, the
    /// upload's bytes kept); None otherwise.
    pub(super) fn start(meta: &Meta, input: &DraftInput<'_>) -> Option<Self> {
        let settings = &meta.live.collect;
        if !settings.enabled || meta.verify_crop {
            return None;
        }
        let image = meta.original.clone()?;
        Some(Self {
            draft: Draft::new(input),
            image,
            settings: settings.clone(),
        })
    }

    /// Hand the finished record to the collector (never blocks).
    pub(super) fn submit(mut self, predictions: &[Prediction]) {
        self.draft.set_predictions(predictions);
        crate::collect::submit(Frame {
            settings: self.settings,
            image: self.image,
            record: self.draft,
        });
    }
}

/// The detections `memory` removed: `before` minus `after`, where `after` keeps `before`'s
/// order (the static memory never reorders).
pub(super) fn removed_by_static(
    before: &[Detection],
    after: &[Detection],
) -> Vec<(Detection, SuppressReason)> {
    let mut out = Vec::new();
    let mut j = 0;
    for d in before {
        if after.get(j) == Some(d) {
            j += 1;
        } else {
            out.push((d.clone(), SuppressReason::Static));
        }
    }
    out
}

/// The verifier's part of a record: every candidate of `plan` (indices into `before`, the
/// predictions before the verifier) with its outcome (`outcomes` of
/// [`crate::verify::outcomes`]), and in mode `frame` what the verifier found on the frame.
pub(super) fn verify_info(
    before: &[Prediction],
    outcomes: &[(usize, Outcome)],
    found: &[Option<Vec<Prediction>>],
    cfg: &VerifyConfig,
    verifier: &str,
    failure: Option<&str>,
) -> VerifyInfo {
    let candidates = outcomes
        .iter()
        .filter_map(|(i, o)| {
            let p = before.get(*i)?;
            Some(VerifyCandidate {
                label: p.label.clone(),
                score: p.confidence,
                bbox: crate::verify::pred_box(p),
                outcome: o.as_str().to_string(),
            })
        })
        .collect();
    let found = match cfg.mode {
        crate::verify::VerifyMode::Frame => found
            .first()
            .and_then(Option::as_ref)
            .map(|f| f.iter().map(crate::collect::det_of_prediction).collect())
            .unwrap_or_default(),
        crate::verify::VerifyMode::Crop => Vec::new(),
    };
    VerifyInfo {
        verifier: verifier.to_string(),
        mode: cfg.mode.as_str().to_string(),
        failure: failure.map(str::to_string),
        candidates,
        found,
        min_confidence: cfg.min_confidence,
        iou: cfg.iou,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn d(x: f32) -> Detection {
        Detection {
            x1: x,
            y1: 0.0,
            x2: x + 1.0,
            y2: 1.0,
            score: 0.9,
            class_id: 0,
        }
    }

    #[test]
    fn static_removals_are_the_missing_ones_in_order() {
        let before = [d(1.0), d(2.0), d(3.0), d(4.0)];
        let after = [d(1.0), d(3.0)];
        let r = removed_by_static(&before, &after);
        assert_eq!(
            r,
            vec![
                (d(2.0), SuppressReason::Static),
                (d(4.0), SuppressReason::Static)
            ]
        );
        assert!(removed_by_static(&before, &before).is_empty());
    }

    #[test]
    fn lowering_the_threshold_then_cutting_keeps_the_same_boxes() {
        // Greedy NMS: a box at or above `thr` is only suppressed by a higher-scoring kept box,
        // so `nms(all at thr - margin)` cut at `thr` equals `nms(only >= thr)`. Checked on
        // pseudo-random overlapping boxes, with and without the candidate / output caps.
        use crate::model::nms::nms_capped;
        let mut seed = 0x1234_5678_u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f32 / (1u64 << 53) as f32
        };
        for round in 0..200 {
            let n = 5 + round % 60;
            let dets: Vec<Detection> = (0..n)
                .map(|_| {
                    let (x, y) = (rnd() * 100.0, rnd() * 100.0);
                    Detection {
                        x1: x,
                        y1: y,
                        x2: x + 5.0 + rnd() * 30.0,
                        y2: y + 5.0 + rnd() * 30.0,
                        score: rnd(),
                        class_id: (rnd() * 3.0) as usize,
                    }
                })
                .collect();
            let (thr, margin) = (0.5f32, 0.1f32);
            for (cands, max_det) in [(3000, 300), (10, 4)] {
                let mut only = dets.clone();
                only.retain(|x| x.score >= thr);
                nms_capped(&mut only, 0.45, cands, max_det);
                let mut low = dets.clone();
                low.retain(|x| x.score >= thr - margin);
                nms_capped(&mut low, 0.45, cands, max_det);
                low.retain(|x| x.score >= thr);
                // The caps keep the highest scores, which are the same boxes in both runs.
                assert_eq!(low, only, "round {round}, caps {cands}/{max_det}");
            }
        }
    }
}
