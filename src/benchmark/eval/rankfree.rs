//! The rank-only metrics (no threshold) with group weights, for their cluster bootstrap: AP50,
//! AP50-95 and the frame ROC AUC recomputed from the resampled images' match records.
//!
//! - AP: the records of [`crate::benchmark::metrics::evaluate`] (per class, the
//!   [`crate::benchmark::metrics::MAX_DETS`] most confident predictions per image, matched once
//!   per IoU of [`IOU_THRESHOLDS`], 101-point interpolated precision), kept with the group of
//!   their image. A record of a group drawn `w` times counts `w` times (the same as `w` copies
//!   of its image: copies of a record are adjacent in the confidence order, and the precision
//!   envelope only looks at the end of such a run). The classes averaged are those of the full
//!   sample ([`ApBasis`]), fixed across resamples; a class with no object in a resample is left
//!   out of that resample's mean. With every weight 1 this equals `evaluate`'s AP (tested).
//! - ROC AUC: one sample per frame ([`crate::benchmark::metrics::SweepImage::alert_sample`]:
//!   positive with a counted object, negative without one or an ignore region of one), scored by
//!   its highest counted-class confidence outside every ignore region (0 without one); the
//!   weighted Mann-Whitney statistic (a tie counts half).

use super::cv::Groups;
use crate::benchmark::metrics::{
    ApBasis, EvalImage, GtBox, IOU_THRESHOLDS, PredBox, SweepImage, SweepScope, cap, match_image,
};

/// AP inputs of a set of images (see the module docs).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ApData {
    pub classes: Vec<String>,
    /// Indices into `classes` of the averaged classes.
    pub basis: Vec<usize>,
    /// Per class, per IoU index: (confidence, true positive, group) by descending confidence.
    records: Vec<Vec<Vec<(f32, bool, u32)>>>,
    /// Per class, per group: non-ignored objects.
    npos: Vec<Vec<u32>>,
    /// IoU thresholds matched: 1 (AP50 only) or all of [`IOU_THRESHOLDS`].
    ious: usize,
}

impl ApData {
    /// The records of `images` (counted classes only) in `groups`; `all_ious` also matches
    /// IoU 0.55..0.95 (AP50-95).
    pub fn new(images: &[EvalImage], groups: &Groups, classes: &[String], all_ious: bool) -> Self {
        let ious = if all_ious { IOU_THRESHOLDS.len() } else { 1 };
        let mut records = vec![vec![Vec::new(); ious]; classes.len()];
        let mut npos = vec![vec![0u32; groups.len()]; classes.len()];
        for (img, &g) in images.iter().zip(&groups.of_image) {
            let capped = cap(&img.preds);
            let gid = u32::try_from(g).unwrap_or(u32::MAX);
            for (ci, class) in classes.iter().enumerate() {
                let preds: Vec<PredBox> = capped
                    .iter()
                    .filter(|p| p.label == *class)
                    .cloned()
                    .collect();
                let gt: Vec<GtBox> = img
                    .gt
                    .iter()
                    .filter(|b| b.label == *class)
                    .cloned()
                    .collect();
                if let Some(n) = npos.get_mut(ci).and_then(|v| v.get_mut(g)) {
                    *n += u32::try_from(gt.iter().filter(|b| !b.ignore).count()).unwrap_or(0);
                }
                for (ii, &t) in IOU_THRESHOLDS.iter().take(ious).enumerate() {
                    let m = match_image(&preds, &gt, t);
                    if let Some(r) = records.get_mut(ci).and_then(|v| v.get_mut(ii)) {
                        for (p, res) in preds.iter().zip(m.pred) {
                            if let Some(tp) = res {
                                r.push((p.confidence, tp, gid));
                            }
                        }
                    }
                }
            }
        }
        // Stable: equal confidences keep image order, as `evaluate` sorts.
        for per_iou in &mut records {
            for r in per_iou {
                r.sort_by(|a, b| b.0.total_cmp(&a.0));
            }
        }
        let gts: Vec<usize> = npos
            .iter()
            .map(|v| v.iter().map(|&n| n as usize).sum())
            .collect();
        let with_gt: Vec<usize> = (0..classes.len())
            .filter(|&i| gts.get(i).is_some_and(|&n| n > 0))
            .collect();
        let names: Vec<&str> = with_gt
            .iter()
            .filter_map(|&i| classes.get(i).map(String::as_str))
            .collect();
        let counts: Vec<usize> = with_gt
            .iter()
            .filter_map(|&i| gts.get(i).copied())
            .collect();
        let (_, idx) = ApBasis::of(&names, &counts);
        Self {
            classes: classes.to_vec(),
            basis: idx
                .iter()
                .filter_map(|&i| with_gt.get(i).copied())
                .collect(),
            records,
            npos,
            ious,
        }
    }

    /// AP of class `ci` at IoU index `ii` with group weights `w` (None = every weight 1).
    fn ap(&self, ci: usize, ii: usize, w: Option<&[u32]>) -> Option<f64> {
        let weight = |g: u32| w.map_or(1, |w| w.get(g as usize).copied().unwrap_or(0));
        let npos: u64 = self
            .npos
            .get(ci)?
            .iter()
            .enumerate()
            .map(|(g, &n)| u64::from(n) * u64::from(weight(u32::try_from(g).unwrap_or(u32::MAX))))
            .sum();
        if npos == 0 {
            return None;
        }
        let recs = self.records.get(ci)?.get(ii)?;
        let mut recall = Vec::with_capacity(recs.len());
        let mut precision = Vec::with_capacity(recs.len());
        let (mut tp, mut fp) = (0u64, 0u64);
        for &(_, is_tp, g) in recs {
            let k = u64::from(weight(g));
            if k == 0 {
                continue;
            }
            if is_tp {
                tp += k;
            } else {
                fp += k;
            }
            recall.push(tp as f64 / npos as f64);
            precision.push(tp as f64 / (tp + fp) as f64);
        }
        let mut right_max = f64::NEG_INFINITY;
        for p in precision.iter_mut().rev() {
            if right_max > *p {
                *p = right_max;
            } else {
                right_max = *p;
            }
        }
        let mut sum = 0.0;
        for k in 0..=100 {
            let r = f64::from(k) / 100.0;
            let idx = recall.partition_point(|&x| x < r - 1e-12);
            if let Some(p) = precision.get(idx) {
                sum += p;
            }
        }
        Some(sum / 101.0)
    }

    /// Mean AP50 over the basis classes with objects in this sample.
    pub fn ap50(&self, w: Option<&[u32]>) -> Option<f64> {
        mean(self.basis.iter().filter_map(|&c| self.ap(c, 0, w)))
    }

    /// Mean AP50-95 (None unless built with every IoU).
    pub fn ap50_95(&self, w: Option<&[u32]>) -> Option<f64> {
        if self.ious < IOU_THRESHOLDS.len() {
            return None;
        }
        mean(self.basis.iter().filter_map(|&c| {
            let v: Vec<f64> = (0..self.ious).filter_map(|i| self.ap(c, i, w)).collect();
            (!v.is_empty()).then(|| v.iter().sum::<f64>() / IOU_THRESHOLDS.len() as f64)
        }))
    }

    /// AP50 of one class (all weights 1).
    pub fn class_ap50(&self, class: &str) -> Option<f64> {
        let ci = self.classes.iter().position(|c| c == class)?;
        self.ap(ci, 0, None)
    }
}

fn mean(it: impl Iterator<Item = f64>) -> Option<f64> {
    let v: Vec<f64> = it.collect();
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

/// Frame samples for the ROC AUC: (score, positive, group), ascending score.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AucData {
    samples: Vec<(f32, bool, u32)>,
}

impl AucData {
    /// The samples of `images` (matched, with frame classes) in `groups`.
    pub fn new(images: &[SweepImage], groups: &Groups) -> Self {
        let mut samples: Vec<(f32, bool, u32)> = images
            .iter()
            .zip(&groups.of_image)
            .filter_map(|(img, &g)| {
                let a = img.alert_sample(SweepScope::All)?;
                (a.positive || !a.ignore_only)
                    .then(|| (a.score, a.positive, u32::try_from(g).unwrap_or(u32::MAX)))
            })
            .collect();
        samples.sort_by(|a, b| a.0.total_cmp(&b.0));
        Self { samples }
    }

    /// The weighted AUC (None without both kinds of frames).
    pub fn auc(&self, w: Option<&[u32]>) -> Option<f64> {
        let weight = |g: u32| f64::from(w.map_or(1, |w| w.get(g as usize).copied().unwrap_or(0)));
        let (mut below_neg, mut sum, mut pos_w, mut neg_w) = (0.0, 0.0, 0.0, 0.0);
        let mut i = 0;
        while i < self.samples.len() {
            let score = self.samples.get(i).map_or(0.0, |s| s.0);
            let (mut bp, mut bn) = (0.0, 0.0);
            // The group's first sample is always taken: a NaN score (never equal to itself)
            // forms its own group instead of looping forever (QA-188).
            let start = i;
            while let Some(&(s, pos, g)) = self.samples.get(i) {
                if i > start && s != score {
                    break;
                }
                if pos {
                    bp += weight(g);
                } else {
                    bn += weight(g);
                }
                i += 1;
            }
            sum += bp * below_neg + 0.5 * bp * bn;
            below_neg += bn;
            pos_w += bp;
            neg_w += bn;
        }
        (pos_w > 0.0 && neg_w > 0.0).then(|| sum / (pos_w * neg_w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::metrics::evaluate;

    /// QA-188: a NaN score ends (each NaN its own group); ties keep half credit, -0.0 and 0.0
    /// still tie (no `total_cmp` grouping).
    #[test]
    fn auc_with_nan_scores_terminates() {
        let data = |samples: Vec<(f32, bool, u32)>| {
            let mut samples = samples;
            samples.sort_by(|a, b| a.0.total_cmp(&b.0));
            AucData { samples }
        };
        let nan = data(vec![
            (0.9, true, 0),
            (f32::NAN, true, 0),
            (0.2, false, 0),
            (f32::NAN, false, 0),
        ]);
        // Sorted: 0.2 neg, 0.9 pos, NaN pos, NaN neg (one group each). Pairs (pos, neg) where
        // the positive ranks above: (0.9, 0.2), (NaN+, 0.2); 2 of 4.
        assert_eq!(nan.auc(None), Some(0.5));
        // Ties: half credit, -0.0 and 0.0 one group.
        let tie = data(vec![(0.0, true, 0), (-0.0, false, 0)]);
        assert_eq!(tie.auc(None), Some(0.5));
        let ranked = data(vec![(0.8, true, 0), (0.3, false, 0), (0.3, true, 1)]);
        // (0.8 > 0.3) = 1, (0.3 = 0.3) = 0.5: 1.5 of 2.
        assert_eq!(ranked.auc(None), Some(0.75));
        // Group weights: group 1 left out.
        assert_eq!(ranked.auc(Some(&[1, 0])), Some(1.0));
    }

    fn gt(label: &str, x: f32, ignore: bool) -> GtBox {
        GtBox {
            label: label.into(),
            bbox: [x, 0.0, x + 50.0, 50.0],
            ignore,
            crowd: false,
        }
    }

    fn p(label: &str, c: f32, x: f32) -> PredBox {
        PredBox {
            label: label.into(),
            confidence: c,
            bbox: [x, 0.0, x + 50.0, 50.0],
            window: None,
        }
    }

    fn images() -> Vec<EvalImage> {
        let mut out = Vec::new();
        for i in 0..30 {
            let f = i as f32;
            let mut gts = vec![gt("person", 0.0, false)];
            if i % 3 == 0 {
                gts.push(gt("car", 100.0, false));
            }
            if i % 7 == 0 {
                gts.push(gt("person", 300.0, true));
            }
            let mut preds = vec![p("person", 0.3 + (f * 0.37) % 0.6, 2.0 * (i % 4) as f32)];
            if i % 2 == 0 {
                preds.push(p(
                    "car",
                    0.2 + (f * 0.11) % 0.7,
                    100.0 + (i % 5) as f32 * 6.0,
                ));
            }
            if i % 5 == 0 {
                preds.push(p("person", 0.5, 600.0));
            }
            out.push(EvalImage {
                gt: gts,
                preds,
                annotated: None,
            });
        }
        out
    }

    #[test]
    fn unit_weights_equal_evaluate() {
        let imgs = images();
        let classes = vec!["person".to_string(), "car".to_string()];
        let groups = Groups::new((0..imgs.len()).map(|i| ["a", "b", "c", "d"][i % 4]));
        let d = ApData::new(&imgs, &groups, &classes, true);
        let s = evaluate(&imgs, &classes, 0.5);
        let close = |a: Option<f64>, b: Option<f64>| match (a, b) {
            (Some(a), Some(b)) => (a - b).abs() < 1e-12,
            (a, b) => a == b,
        };
        assert!(
            close(d.ap50(None), s.ap50),
            "{:?} {:?}",
            d.ap50(None),
            s.ap50
        );
        assert!(close(d.ap50_95(None), s.ap50_95));
        // Weights of 1 are no weights.
        let ones = vec![1u32; groups.len()];
        assert_eq!(d.ap50(Some(&ones)), d.ap50(None));
        // A group drawn twice equals its images duplicated.
        let mut w = vec![1u32; groups.len()];
        w[1] = 2;
        w[3] = 0;
        let dup: Vec<EvalImage> = imgs
            .iter()
            .enumerate()
            .flat_map(|(i, e)| {
                let n = w[groups.of_image[i]];
                std::iter::repeat_n(e.clone(), n as usize)
            })
            .collect();
        let s2 = evaluate(&dup, &classes, 0.5);
        assert!(
            close(d.ap50(Some(&w)), s2.ap50),
            "{:?} {:?}",
            d.ap50(Some(&w)),
            s2.ap50
        );
    }

    #[test]
    fn weighted_auc_equals_brute_force() {
        let imgs = images();
        let classes = vec!["person".to_string(), "car".to_string()];
        let mut sweep: Vec<SweepImage> = imgs
            .iter()
            .map(|e| SweepImage::of(e).with_frames(e, &classes))
            .collect();
        // Some frames without objects.
        for (i, e) in imgs.iter().enumerate().take(8) {
            let empty = EvalImage {
                gt: vec![],
                preds: e
                    .preds
                    .iter()
                    .map(|p| PredBox {
                        confidence: p.confidence * 0.8,
                        ..p.clone()
                    })
                    .take(i % 3)
                    .collect(),
                annotated: None,
            };
            sweep.push(SweepImage::of(&empty).with_frames(&empty, &classes));
        }
        let groups = Groups::new((0..sweep.len()).map(|i| ["a", "b", "c"][i % 3]));
        let d = AucData::new(&sweep, &groups);
        let w = [2u32, 0, 1];
        let samples: Vec<(f32, bool, f64)> = sweep
            .iter()
            .zip(&groups.of_image)
            .filter_map(|(s, &g)| {
                let a = s.alert_sample(SweepScope::All)?;
                (a.positive || !a.ignore_only).then(|| (a.score, a.positive, f64::from(w[g])))
            })
            .collect();
        let (mut num, mut pw, mut nw) = (0.0, 0.0, 0.0);
        for &(s, pos, wp) in &samples {
            if pos {
                pw += wp;
                for &(t, neg_pos, wn) in &samples {
                    if !neg_pos {
                        num += wp
                            * wn
                            * if s > t {
                                1.0
                            } else if s == t {
                                0.5
                            } else {
                                0.0
                            };
                    }
                }
            } else {
                nw += wp;
            }
        }
        let want = num / (pw * nw);
        assert!((d.auc(Some(&w)).unwrap() - want).abs() < 1e-12);
        assert!(d.auc(None).is_some());
        assert_eq!(AucData::default().auc(None), None);
    }

    /// S5: an As served box returned on two separate threshold stretches is stored once per
    /// stretch; AP counts it once (the copy is not a second, unmatched detection).
    #[test]
    fn a_box_stored_per_stretch_counts_once_in_ap() {
        use crate::benchmark::metrics::Window;
        let windowed = |w: [f32; 2]| PredBox {
            window: Some(Window::from(w)),
            ..p("person", 0.8, 100.0)
        };
        // Three persons found at 0.9, 0.8 (stored once per stretch) and 0.7: a copy counted as a
        // false positive before the 0.7 find would lower its precision.
        let imgs = vec![EvalImage {
            gt: vec![
                gt("person", 0.0, false),
                gt("person", 100.0, false),
                gt("person", 200.0, false),
            ],
            preds: vec![
                p("person", 0.9, 0.0),
                windowed([f32::NEG_INFINITY, 0.3]),
                windowed([0.5, 0.8]),
                p("person", 0.7, 200.0),
            ],
            annotated: None,
        }];
        let classes = vec!["person".to_string()];
        let groups = Groups::new(["a"]);
        let d = ApData::new(&imgs, &groups, &classes, false);
        assert_eq!(d.ap50(None), Some(1.0));
        assert_eq!(evaluate(&imgs, &classes, 0.5).ap50, Some(1.0));
    }
}
