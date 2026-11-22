//! Per-image counts on the 0.01 confidence grid ([`crate::benchmark::threshold::grid`]): what
//! the cross-validation, the bootstraps and the breakdowns sum.
//!
//! An image's [`ImageTally`] holds what does not depend on the threshold (its scored objects,
//! whether it is a positive frame, a negative frame or an empty frame) and, per grid value, its
//! true and false positives and whether it alerts. Built from the image matched once
//! ([`SweepImage`]): a plain run's prediction counts at every `t <= confidence`, one of a run with
//! serving rules on its windows (`after < t <= until`, [`crate::benchmark::metrics::PredBox::at`]),
//! so a `+served` result is counted exactly as served. Summed over images, the counts equal the
//! threshold sweep's ([`crate::benchmark::metrics::Breakpoints::point_at`]) at every grid value
//! (tested).
//!
//! Definitions (one source of truth for the evaluation):
//! - box counts: TP / FP / FN at IoU 0.5 of the counted classes (ignore regions neither);
//! - positive frame: a non-ignored object of a counted class; negative frame: none and no
//!   ignore region of one (a frame whose only counted boxes are ignore regions cannot be judged
//!   and is out of the frame rates); a frame alerts at `t` when one of its counted-class
//!   detections outside every ignore region is returned there;
//! - empty frame: no non-ignored counted object (ignore-only frames included); a false alert is
//!   an empty frame that alerts. FA/100 = 100 x false alerts / empty frames.

use crate::benchmark::metrics::{Counts, SweepImage, SweepScope};
use crate::benchmark::threshold::grid;

/// What one image adds at one grid value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Step {
    pub tp: u32,
    pub fp: u32,
    /// The frame alerts (1) or not (0).
    pub alerts: u8,
}

/// One image's counts over the grid.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ImageTally {
    /// Non-ignored objects of the counted classes.
    pub objects: u32,
    /// A positive frame (see the module docs).
    pub positive: bool,
    /// A negative frame: neither an object nor an ignore region of a counted class.
    pub negative: bool,
    /// A frame at all: its set labels a counted class (else it counts boxes only, never frames).
    pub frame: bool,
    /// Per grid value ([`grid`] order).
    pub steps: Vec<Step>,
}

impl ImageTally {
    /// The tally of `img` (matched once, with its frame classes: [`SweepImage::with_frames`]);
    /// `thresholds` is the grid. An image without frame classes still counts boxes, never as a
    /// frame.
    pub fn of(img: &SweepImage, thresholds: &[f32]) -> Self {
        let sample = img.alert_sample(SweepScope::All);
        let objects = u32::try_from(img.gt.len()).unwrap_or(u32::MAX);
        let (positive, negative) = match &sample {
            Some(a) => (a.positive, !a.positive && !a.ignore_only),
            None => (false, false),
        };
        let steps = thresholds
            .iter()
            .map(|&t| {
                let mut s = Step::default();
                for p in &img.preds {
                    if p.after < t && p.confidence >= t {
                        if p.tp {
                            s.tp += 1;
                        } else {
                            s.fp += 1;
                        }
                    }
                }
                s.alerts = u8::from(
                    sample
                        .as_ref()
                        .is_some_and(|a| a.spans.iter().any(|&(lo, hi)| lo < t && t <= hi)),
                );
                s
            })
            .collect();
        Self {
            objects,
            positive,
            negative,
            frame: sample.is_some(),
            steps,
        }
    }
}

/// Counts summed over images at one grid value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct At {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    /// Positive frames and those alerting.
    pub pos: usize,
    pub pos_alerts: usize,
    /// Negative frames and those alerting (false-positive frames).
    pub neg: usize,
    pub neg_alerts: usize,
    /// Empty frames and those alerting (false alerts).
    pub empty: usize,
    pub empty_alerts: usize,
}

impl At {
    /// `img` at grid index `k`, `w` times (a bootstrap weight).
    pub fn add_image(&mut self, img: &ImageTally, k: usize, w: usize) {
        let s = img.steps.get(k).copied().unwrap_or_default();
        let tp = s.tp as usize;
        self.tp += w * tp;
        self.fp += w * s.fp as usize;
        self.fn_ += w * (img.objects as usize).saturating_sub(tp);
        let alerts = usize::from(s.alerts);
        if img.positive {
            self.pos += w;
            self.pos_alerts += w * alerts;
        }
        if img.negative {
            self.neg += w;
            self.neg_alerts += w * alerts;
        }
        if img.frame && !img.positive {
            self.empty += w;
            self.empty_alerts += w * alerts;
        }
    }

    pub fn add(&mut self, o: &At) {
        self.add_scaled(o, 1);
    }

    /// `o` `w` times.
    pub fn add_scaled(&mut self, o: &At, w: usize) {
        self.tp += w * o.tp;
        self.fp += w * o.fp;
        self.fn_ += w * o.fn_;
        self.pos += w * o.pos;
        self.pos_alerts += w * o.pos_alerts;
        self.neg += w * o.neg;
        self.neg_alerts += w * o.neg_alerts;
        self.empty += w * o.empty;
        self.empty_alerts += w * o.empty_alerts;
    }

    pub fn counts(&self) -> Counts {
        Counts {
            tp: self.tp,
            fp: self.fp,
            fn_: self.fn_,
        }
    }

    pub fn objects(&self) -> usize {
        self.tp + self.fn_
    }

    /// None without detections.
    pub fn precision(&self) -> Option<f64> {
        self.counts().precision()
    }

    /// None without objects.
    pub fn recall(&self) -> Option<f64> {
        self.counts().recall()
    }

    /// [`Counts::f_beta`]: None without objects, 0 when none is found.
    pub fn f_beta(&self, beta: f64) -> Option<f64> {
        self.counts().f_beta(beta)
    }

    /// False alerts per 100 empty frames (None without empty frames).
    pub fn fa100(&self) -> Option<f64> {
        (self.empty > 0).then(|| 100.0 * self.empty_alerts as f64 / self.empty as f64)
    }

    /// Frame TPR (None without positive frames).
    pub fn tpr(&self) -> Option<f64> {
        (self.pos > 0).then(|| self.pos_alerts as f64 / self.pos as f64)
    }

    /// Frame FPR (None without negative frames).
    pub fn fpr(&self) -> Option<f64> {
        (self.neg > 0).then(|| self.neg_alerts as f64 / self.neg as f64)
    }

    /// Youden's J over frames (None without both kinds).
    pub fn youden(&self) -> Option<f64> {
        Some(self.tpr()? - self.fpr()?)
    }
}

/// Every image of `images` at every grid value: index `k` is [`grid`]`()[k]`.
pub fn sum_all<'a>(images: impl IntoIterator<Item = &'a ImageTally>, n: usize) -> Vec<At> {
    let mut out = vec![At::default(); n];
    for img in images {
        for (k, a) in out.iter_mut().enumerate() {
            a.add_image(img, k, 1);
        }
    }
    out
}

/// The tallies of `images` on the grid.
pub fn tallies(images: &[SweepImage]) -> Vec<ImageTally> {
    let g = grid();
    images.iter().map(|i| ImageTally::of(i, &g)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::metrics::{Breakpoints, EvalImage, GtBox, PredBox, Window};

    fn gt(label: &str, b: [f32; 4], ignore: bool) -> GtBox {
        GtBox {
            label: label.into(),
            bbox: b,
            ignore,
            crowd: false,
        }
    }

    fn p(label: &str, c: f32, b: [f32; 4]) -> PredBox {
        PredBox {
            label: label.into(),
            confidence: c,
            bbox: b,
            window: None,
        }
    }

    const A: [f32; 4] = [0.0, 0.0, 100.0, 100.0];
    const B: [f32; 4] = [200.0, 200.0, 300.0, 300.0];
    const C: [f32; 4] = [400.0, 0.0, 500.0, 100.0];

    pub(crate) fn sample_images() -> Vec<EvalImage> {
        let classes = Some(vec!["person".to_string(), "car".to_string()]);
        vec![
            // Two objects, one found at 0.9, a false positive at 0.4.
            EvalImage {
                gt: vec![gt("person", A, false), gt("car", B, false)],
                preds: vec![p("person", 0.9, A), p("car", 0.4, C), p("car", 0.07, B)],
                annotated: classes.clone(),
            },
            // Empty frame with a false alert at 0.62.
            EvalImage {
                gt: vec![],
                preds: vec![p("person", 0.62, A), p("person", 0.2, B)],
                annotated: classes.clone(),
            },
            // Ignore-only frame: an empty frame, but not a negative one.
            EvalImage {
                gt: vec![gt("person", A, true)],
                preds: vec![p("person", 0.8, A), p("car", 0.3, C)],
                annotated: classes.clone(),
            },
            // Windowed (served) predictions: returned only on (0.3, 0.7].
            EvalImage {
                gt: vec![gt("person", A, false)],
                preds: vec![PredBox {
                    window: Some(Window {
                        after: 0.3,
                        until: 0.7,
                    }),
                    ..p("person", 0.7, A)
                }],
                annotated: classes,
            },
            // A frame of a set that labels only cars: person boxes are no frames of it.
            EvalImage {
                gt: vec![gt("car", B, false)],
                preds: vec![p("car", 0.55, B)],
                annotated: Some(vec!["car".to_string()]),
            },
        ]
    }

    #[test]
    fn grid_sums_equal_the_sweep_at_every_grid_value() {
        let classes = ["person".to_string(), "car".to_string()];
        let imgs: Vec<SweepImage> = sample_images()
            .iter()
            .map(|e| {
                let fc: Vec<String> = classes.iter().filter(|c| e.annotates(c)).cloned().collect();
                SweepImage::of(e).with_frames(e, &fc)
            })
            .collect();
        let t = tallies(&imgs);
        let refs: Vec<&SweepImage> = imgs.iter().collect();
        let bp = Breakpoints::new(&refs, SweepScope::All);
        let g = grid();
        let sums = sum_all(&t, g.len());
        for (k, &thr) in g.iter().enumerate() {
            let want = bp.point_at(thr);
            let a = sums[k];
            assert_eq!(a.counts(), want.counts, "counts at {thr}");
            let f = want.per_frame.unwrap();
            assert_eq!(
                (a.pos + a.neg, a.pos_alerts + a.neg_alerts),
                (f.positives + f.negatives, f.tp + f.fp),
                "frames at {thr}"
            );
            assert_eq!((a.pos, a.pos_alerts), (f.positives, f.tp), "{thr}");
            assert_eq!((a.neg, a.neg_alerts), (f.negatives, f.fp), "{thr}");
            let fa = want.false_alerts.unwrap();
            assert_eq!(
                (a.empty, a.empty_alerts),
                (fa.empty_frames, fa.alerts),
                "{thr}"
            );
        }
        // Hand-checked at 0.5: found are the 0.9 person, the windowed person and the 0.55 car
        // (the 0.07 car is below), missed the first car; the 0.62 person on the empty frame is
        // the false positive and its false alert; the 0.8 person lies in the ignore region.
        let k = g.iter().position(|&x| x == 0.5).unwrap();
        let a = sums[k];
        assert_eq!((a.tp, a.fp, a.fn_), (3, 1, 1));
        assert_eq!((a.empty, a.empty_alerts), (2, 1));
        assert_eq!(a.fa100(), Some(50.0));
        let k = g.iter().position(|&x| x == 0.75).unwrap();
        assert_eq!(sums[k].tp, 1, "the served box is gone above its window");
    }

    #[test]
    fn rates_of_an_empty_tally_are_none() {
        let a = At::default();
        assert_eq!(
            (a.precision(), a.recall(), a.fa100(), a.tpr(), a.fpr()),
            (None, None, None, None, None)
        );
        assert_eq!(a.f_beta(1.0), None);
        let a = At {
            fn_: 3,
            ..At::default()
        };
        assert_eq!(a.f_beta(2.0), Some(0.0));
    }
}
