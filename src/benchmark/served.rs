//! What the service returns for one frame after the family's post-processing, at one threshold
//! ([`returned_at`]) and at every threshold of a sweep at once ([`windows`]): the benchmark's
//! scoring of a run with serving rules (`--served`, `--variants filters` / `verify:`) without
//! re-implementing any rule.
//!
//! [`returned_at`] is the worker's order (`worker::run::finish`, then the verify lane): the
//! family's detections at or above the threshold, the filter rules
//! ([`crate::model::filters::apply`], evidence from those detections only), the conversion to
//! predictions with the object filter ([`crate::model::to_predictions`]), then the verifier
//! ([`crate::verify::plan`] on that answer, [`crate::verify::outcomes`],
//! [`crate::verify::apply`]). The evaluation copy is the same without the object filter: a box
//! of a label the object filter drops is never a verifier candidate (the service does not even
//! return it) and stays unverified there. Static-box memory is not applied: it is per-camera
//! history of a live stream, which a set of independent images is not.
//!
//! B2 (why a sweep cannot just filter one low-threshold pass): with overlap rules the evidence
//! is the detections at or above the threshold, so a 0.07 `potted plant` drops a 0.8 `person`
//! at threshold 0.05 but not at 0.5; and the verifier checks the `max_crops` most confident
//! candidates of what is returned. What is returned at `t` is therefore not "the 0.05 answer
//! with confidence >= t". It changes only where the family's input changes, at the distinct
//! detection scores: on `(s_next, s]` between two consecutive scores the family returns the same
//! detections, so everything after it does too. [`windows`] runs [`returned_at`] once at every
//! distinct score at or above the floor and records for each box the stretches of thresholds at
//! which it is returned ([`Window`]; a box returned on two separate stretches is stored once per
//! stretch). A box whose stretch is "every threshold up to its own score" keeps no window, so a
//! run whose rules change nothing stores exactly what a plain run stores. Exact at every
//! threshold at or above the floor, not only on the 0.01 grid; the sweep
//! ([`super::metrics::SweepImage::with_iou`]) matches each stretch's boxes on their own.
//!
//! On a frame whose threshold does not follow the swept one ([`Threshold::Fixed`]: a night /
//! IR frame with `night_threshold`, a frame the night model answers) the answer at that
//! threshold is returned at every threshold ([`Window::ALWAYS`]).
//!
//! The verifier's verdict on a candidate depends on its crop (box, `pad`, frame) or on the whole
//! frame, never on the threshold; so each crop is asked once ([`Answers`] implementations cache)
//! and the threshold only decides which candidates are checked (the `max_crops` most confident
//! of what is returned there). A crop the verifier cannot answer leaves the candidate unverified
//! (kept), as the service's fail-open does.

use super::metrics::{PredBox, Window};
use crate::api::Prediction;
use crate::image_stats::Scene;
use crate::model::filters::FilterRules;
use crate::model::{Detection, PreprocessCtx};
use crate::verify::{CropRect, Outcome, VerifyConfig, VerifyMode};

/// What runs after the family's post-processing, as the worker applies it.
#[derive(Debug, Clone, Copy)]
pub struct PostRules<'a> {
    /// The model's labels (class id -> label).
    pub classes: &'a [String],
    /// The filter rules (None or all off: none).
    pub filters: Option<&'a FilterRules>,
    /// Only report these labels (None or empty: all).
    pub object_filter: Option<&'a [String]>,
    /// The verifier settings (None: no verifier).
    pub verify: Option<&'a VerifyConfig>,
}

impl PostRules<'_> {
    /// Nothing changes what a plain run returns at any threshold (no filter rule, no verifier).
    pub fn is_plain(&self) -> bool {
        self.filters.is_none_or(FilterRules::is_noop) && self.verify.is_none()
    }
}

/// The verifier's answers on one frame: what it reports on a crop (mode `crop`) or on the whole
/// frame (mode `frame`); None when it could not check (unverified, kept).
pub trait Answers {
    fn crop(&mut self, r: CropRect) -> Option<Vec<Prediction>>;
    fn frame(&mut self) -> Option<Vec<Prediction>>;
}

/// No verifier: every candidate unverified.
#[derive(Debug)]
pub struct NoAnswers;

impl Answers for NoAnswers {
    fn crop(&mut self, _: CropRect) -> Option<Vec<Prediction>> {
        None
    }
    fn frame(&mut self) -> Option<Vec<Prediction>> {
        None
    }
}

/// What the service returns at one threshold.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Returned {
    /// The answer (object filter applied, the verifier's rejections removed).
    pub predictions: Vec<Prediction>,
    /// The same without the object filter, each with the index of its detection in the input.
    pub eval: Vec<(usize, Prediction)>,
}

/// What the service returns for `dets` (the family's detections of one frame of `scene` at a
/// threshold at or below `t`, in post-processing order) at threshold `t` (see the module docs).
pub fn returned_at(
    dets: &[Detection],
    ctx: &PreprocessCtx,
    scene: Scene,
    t: f32,
    rules: &PostRules,
    answers: &mut dyn Answers,
) -> Returned {
    // The family at `t`: the detections at or above it, in order (its prefix property).
    let family: Vec<usize> = (0..dets.len())
        .filter(|&i| dets.get(i).is_some_and(|d| d.score >= t))
        .collect();
    let input: Vec<Detection> = family
        .iter()
        .filter_map(|&i| dets.get(i).cloned())
        .collect();
    let kept: Vec<usize> = match rules.filters.filter(|r| !r.is_noop()) {
        Some(r) => {
            let (out, _) = crate::model::filters::apply(
                input.clone(),
                rules.classes,
                ctx.orig_w,
                ctx.orig_h,
                scene,
                r,
            );
            subsequence(&input, &out)
                .into_iter()
                .filter_map(|k| family.get(k).copied())
                .collect()
        }
        None => family,
    };
    let mut eval: Vec<(usize, Prediction)> = Vec::new();
    // Indices into `eval` of the answer's predictions, in order.
    let mut answer: Vec<usize> = Vec::new();
    for i in kept {
        let Some(d) = dets.get(i) else { continue };
        let one = std::slice::from_ref(d);
        let Some(p) = crate::model::to_predictions(one, ctx, rules.classes, None).pop() else {
            continue;
        };
        if !crate::model::to_predictions(one, ctx, rules.classes, rules.object_filter).is_empty() {
            answer.push(eval.len());
        }
        eval.push((i, p));
    }
    let predictions: Vec<Prediction> = answer
        .iter()
        .filter_map(|&k| eval.get(k).map(|(_, p)| p.clone()))
        .collect();
    let Some(cfg) = rules.verify else {
        return Returned { predictions, eval };
    };
    let plan = crate::verify::plan(&predictions, ctx.orig_w, ctx.orig_h, scene, cfg);
    if plan.is_empty() {
        return Returned { predictions, eval };
    }
    let found: Vec<Option<Vec<Prediction>>> = match cfg.mode {
        VerifyMode::Crop => plan
            .checks
            .iter()
            .map(|c| c.crop.and_then(|r| answers.crop(r)))
            .collect(),
        VerifyMode::Frame => {
            if plan.checks.is_empty() {
                Vec::new()
            } else {
                vec![answers.frame()]
            }
        }
    };
    let outcomes = crate::verify::outcomes(&plan, &found, cfg);
    let rejected: Vec<usize> = outcomes
        .iter()
        .filter(|(_, o)| *o == Outcome::Rejected)
        .filter_map(|(k, _)| answer.get(*k).copied())
        .collect();
    let (predictions, _) = crate::verify::apply(predictions, &outcomes);
    let eval = eval
        .into_iter()
        .enumerate()
        .filter(|(k, _)| !rejected.contains(k))
        .map(|(_, e)| e)
        .collect();
    Returned { predictions, eval }
}

/// The positions in `all` of `sub`, a subsequence of it (greedy, in order).
fn subsequence(all: &[Detection], sub: &[Detection]) -> Vec<usize> {
    let mut out = Vec::with_capacity(sub.len());
    let mut j = 0;
    for s in sub {
        while let Some(a) = all.get(j) {
            j += 1;
            if a == s {
                out.push(j - 1);
                break;
            }
        }
    }
    out
}

/// The threshold of one frame in a sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Threshold {
    /// The swept threshold (a day frame; a night / IR frame without `night_threshold`).
    Sweep,
    /// This threshold whatever the sweep's (`night_threshold` on a night / IR frame, the night
    /// model's own on a frame it answers).
    Fixed(f32),
}

/// Every box the service returns for `dets` (the family's detections at `floor`, the lowest
/// threshold the sweep scores) at some threshold at or above `floor`, with the stretches of
/// thresholds it is returned at ([`PredBox::window`]; see the module docs), in the order of the
/// answer at the lowest threshold. Labels are the model's (not yet mapped to scored classes).
pub fn windows(
    dets: &[Detection],
    ctx: &PreprocessCtx,
    scene: Scene,
    threshold: Threshold,
    floor: f32,
    rules: &PostRules,
    answers: &mut dyn Answers,
) -> Vec<PredBox> {
    let pred = |p: &Prediction, window: Option<Window>| PredBox {
        window,
        ..super::run::to_pred(p)
    };
    if let Threshold::Fixed(t) = threshold {
        return returned_at(dets, ctx, scene, t, rules, answers)
            .eval
            .iter()
            .map(|(_, p)| pred(p, Some(Window::ALWAYS)))
            .collect();
    }
    if rules.is_plain() {
        return returned_at(dets, ctx, scene, floor, rules, answers)
            .eval
            .iter()
            .map(|(_, p)| pred(p, None))
            .collect();
    }
    // Distinct scores at or above the floor, highest first: the answer is the same on
    // (next lower score, score].
    let mut keys: Vec<f32> = dets
        .iter()
        .map(|d| d.score)
        .filter(|s| *s >= floor && s.is_finite())
        .collect();
    keys.sort_by(|a, b| b.total_cmp(a));
    keys.dedup();
    let answers_at: Vec<Returned> = keys
        .iter()
        .map(|&t| returned_at(dets, ctx, scene, t, rules, answers))
        .collect();
    // Per detection: its prediction and the stretches it is returned on (built from the top).
    let mut boxes: Vec<(usize, Prediction, Vec<Window>)> = Vec::new();
    for (k, r) in answers_at.iter().enumerate() {
        let lower = keys.get(k + 1).copied().unwrap_or(f32::NEG_INFINITY);
        let upper = keys.get(k).copied().unwrap_or(f32::NEG_INFINITY);
        for (i, p) in &r.eval {
            let entry = match boxes.iter_mut().position(|b| b.0 == *i) {
                Some(e) => e,
                None => {
                    boxes.push((*i, p.clone(), Vec::new()));
                    boxes.len() - 1
                }
            };
            let Some((_, _, spans)) = boxes.get_mut(entry) else {
                continue;
            };
            match spans.last_mut() {
                // Returned on the stretch just above too: one stretch.
                Some(w) if w.after == upper => w.after = lower,
                _ => spans.push(Window {
                    after: lower,
                    until: upper,
                }),
            }
        }
    }
    // The order of the answer at the lowest threshold, then the boxes only higher ones return.
    let order: Vec<usize> = answers_at
        .last()
        .map(|r| r.eval.iter().map(|(i, _)| *i).collect())
        .unwrap_or_default();
    boxes.sort_by_key(|b| order.iter().position(|i| *i == b.0).unwrap_or(usize::MAX));
    let mut out = Vec::new();
    for (i, p, spans) in boxes {
        let score = dets.get(i).map_or(f32::NAN, |d| d.score);
        for w in spans {
            let plain = w.after == f32::NEG_INFINITY && w.until == score;
            out.push(pred(&p, (!plain).then_some(w)));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::metrics::{
        Breakpoints, EvalImage, GtBox, SweepImage, SweepScope, image_counts, sweep_counts,
    };
    use crate::model::ResizeMode;
    use crate::model::filters::OverlapRule;
    use std::collections::BTreeMap;

    const W: u32 = 640;
    const H: u32 = 480;

    fn ctx() -> PreprocessCtx {
        PreprocessCtx {
            orig_w: W,
            orig_h: H,
            input_w: W,
            input_h: H,
            mode: ResizeMode::Stretch,
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        }
    }

    fn names() -> Vec<String> {
        ["person", "car", "potted plant", "dog"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
        fn pick(&mut self, n: usize) -> usize {
            ((self.f() * n as f32) as usize).min(n - 1)
        }
    }

    /// Detections in a few clusters, so overlap rules fire; coarse scores, so ties occur.
    fn random_dets(r: &mut Rng) -> Vec<Detection> {
        let n = 2 + r.pick(14);
        let centres: Vec<(f32, f32)> = (0..3)
            .map(|_| (60.0 + r.f() * 500.0, 60.0 + r.f() * 360.0))
            .collect();
        let mut v: Vec<Detection> = (0..n)
            .map(|_| {
                let (cx, cy) = centres[r.pick(3)];
                let (w, h) = (20.0 + r.f() * 80.0, 20.0 + r.f() * 120.0);
                let (dx, dy) = ((r.f() - 0.5) * 20.0, (r.f() - 0.5) * 20.0);
                Detection {
                    x1: cx + dx - w / 2.0,
                    y1: cy + dy - h / 2.0,
                    x2: cx + dx + w / 2.0,
                    y2: cy + dy + h / 2.0,
                    score: 0.05 + (r.f() * 0.95 * 40.0).floor() / 40.0,
                    class_id: r.pick(4),
                }
            })
            .collect();
        // Post-processing order: most confident first (as NMS emits them).
        v.sort_by(|a, b| b.score.total_cmp(&a.score));
        v
    }

    fn random_rules(r: &mut Rng) -> FilterRules {
        let some = |r: &mut Rng, p: f32| r.f() < p;
        let mut rules = FilterRules::default();
        if some(r, 0.3) {
            rules.night_confidence_threshold = Some(0.1 + r.f() * 0.5);
        }
        if some(r, 0.3) {
            rules
                .label_thresholds
                .insert(["person", "CAR"][r.pick(2)].into(), 0.1 + r.f() * 0.6);
        }
        if some(r, 0.3) {
            rules.night_label_thresholds = BTreeMap::from([("person".into(), 0.2 + r.f() * 0.5)]);
        }
        if some(r, 0.3) {
            rules.person_aspect = Some([0.3, 0.9]);
        }
        if some(r, 0.2) {
            rules.min_box_area_frac = Some(0.005);
        }
        for _ in 0..r.pick(3) {
            rules.suppress_overlaps.push(OverlapRule {
                drop: ["person", "car", "dog"][r.pick(3)].into(),
                if_overlaps: ["potted plant", "car", "person"][r.pick(3)].into(),
                iou: 0.2 + r.f() * 0.5,
                night_only: some(r, 0.3),
            });
        }
        rules
    }

    fn random_verify(r: &mut Rng) -> Option<VerifyConfig> {
        (r.f() < 0.6).then(|| VerifyConfig {
            labels: vec!["person".into(), "dog".into()],
            mode: if r.f() < 0.7 {
                VerifyMode::Crop
            } else {
                VerifyMode::Frame
            },
            max_crops: 1 + r.pick(3),
            night_only: r.f() < 0.3,
            ..VerifyConfig::with_model("v")
        })
    }

    /// A verifier that confirms a crop by a fixed function of the crop (the same crop, the
    /// same answer), and the frame by a fixed list. Counts its calls.
    struct FakeVerifier {
        calls: usize,
        frame: Vec<Prediction>,
    }

    impl Answers for FakeVerifier {
        fn crop(&mut self, r: CropRect) -> Option<Vec<Prediction>> {
            self.calls += 1;
            let h = (r.x * 31 + r.y * 17 + r.w * 7 + r.h) % 5;
            match h {
                // Unanswered (fail-open).
                0 => None,
                // A matching box over the whole crop: confirms when the reference box covers
                // most of the crop.
                1 | 2 => Some(vec![Prediction {
                    x_min: 0,
                    y_min: 0,
                    x_max: r.w as usize,
                    y_max: r.h as usize,
                    confidence: 0.9,
                    label: "person".into(),
                }]),
                _ => Some(Vec::new()),
            }
        }
        fn frame(&mut self) -> Option<Vec<Prediction>> {
            self.calls += 1;
            Some(self.frame.clone())
        }
    }

    fn plain(p: &Prediction) -> PredBox {
        super::super::run::to_pred(p)
    }

    fn key(v: &[PredBox]) -> Vec<String> {
        let mut k: Vec<String> = v
            .iter()
            .map(|p| format!("{} {:.4} {:?}", p.label, p.confidence, p.bbox))
            .collect();
        k.sort();
        k
    }

    /// The scored predictions at `t` from the windows equal what the service returns at `t`
    /// (B2), at every 0.01 grid value and at random thresholds in between, for random frames,
    /// scenes, filter rules and verifiers; and the sweep's counts from one windowed image
    /// ([`SweepImage::with_iou`]) equal the counts of the service's answer at `t` matched on its
    /// own.
    #[test]
    fn windows_equal_the_service_at_every_threshold() {
        let mut r = Rng(42);
        let classes = names();
        let mut grid = crate::benchmark::threshold::grid();
        for _ in 0..40 {
            grid.push(0.05 + r.f() * 0.95);
        }
        let mut windowed_seen = 0;
        let mut naive_wrong = 0;
        // Every image windowed, and per threshold every image's answer there (plain).
        let mut all_windowed: Vec<SweepImage> = Vec::new();
        let mut all_at: Vec<Vec<SweepImage>> = vec![Vec::new(); grid.len()];
        for case in 0..400 {
            let dets = random_dets(&mut r);
            let scene = [Scene::Day, Scene::Night, Scene::Ir][r.pick(3)];
            let filters = random_rules(&mut r);
            let verify = random_verify(&mut r);
            let object_filter: Vec<String> = if r.f() < 0.2 {
                vec!["person".into(), "car".into()]
            } else {
                Vec::new()
            };
            let rules = PostRules {
                classes: &classes,
                filters: Some(&filters),
                object_filter: Some(&object_filter),
                verify: verify.as_ref(),
            };
            let frame = vec![Prediction {
                x_min: 0,
                y_min: 0,
                x_max: W as usize / 2,
                y_max: H as usize,
                confidence: 0.8,
                label: "person".into(),
            }];
            let mut v = FakeVerifier {
                calls: 0,
                frame: frame.clone(),
            };
            let stored = windows(&dets, &ctx(), scene, Threshold::Sweep, 0.05, &rules, &mut v);
            windowed_seen += usize::from(stored.iter().any(|p| p.window.is_some()));
            // Ground truth near some boxes, a few ignore regions.
            let mut gt: Vec<GtBox> = Vec::new();
            for d in &dets {
                if r.f() < 0.5 {
                    gt.push(GtBox {
                        label: classes[d.class_id].clone(),
                        bbox: [d.x1 + 3.0, d.y1, d.x2, d.y2 + 2.0],
                        ignore: r.f() < 0.15,
                        crowd: false,
                    });
                }
            }
            let windowed = EvalImage {
                gt: gt.clone(),
                preds: stored.clone(),
                annotated: None,
            };
            let sweep = SweepImage::of(&windowed);
            let naive: Vec<PredBox> = returned_at(
                &dets,
                &ctx(),
                scene,
                0.05,
                &rules,
                &mut FakeVerifier {
                    calls: 0,
                    frame: frame.clone(),
                },
            )
            .eval
            .iter()
            .map(|(_, p)| plain(p))
            .collect();
            all_windowed.push(SweepImage::of(&windowed).with_frames(&windowed, &classes));
            for (g, &t) in grid.iter().enumerate() {
                let mut direct = FakeVerifier {
                    calls: 0,
                    frame: frame.clone(),
                };
                let want: Vec<PredBox> = returned_at(&dets, &ctx(), scene, t, &rules, &mut direct)
                    .eval
                    .iter()
                    .map(|(_, p)| plain(p))
                    .collect();
                let got: Vec<PredBox> = stored.iter().filter(|p| p.at(t)).cloned().collect();
                assert_eq!(key(&got), key(&want), "case {case} t {t}");
                // What the sweep scored before B2: the floor's answer cut at `t`.
                let old: Vec<PredBox> = naive
                    .iter()
                    .filter(|p| p.confidence >= t)
                    .cloned()
                    .collect();
                naive_wrong += usize::from(key(&old) != key(&want));
                // Scoring: the windowed image at `t` = the service's answer at `t`.
                let answer = EvalImage {
                    gt: gt.clone(),
                    preds: want,
                    annotated: None,
                };
                all_at[g].push(SweepImage::of(&answer).with_frames(&answer, &classes));
                let at_t = image_counts(&answer, t).0;
                assert_eq!(image_counts(&windowed, t).0, at_t, "case {case} t {t}");
                assert_eq!(
                    sweep_counts(&[&sweep], SweepScope::All, t),
                    at_t,
                    "case {case} t {t}"
                );
            }
            // One question per distinct crop at most (the bench's cache does the rest).
            assert!(v.calls <= dets.len().max(1) * 40, "case {case}");
        }
        // The whole sweep (box counts, frame alert rates, false alerts per empty frame, pooled
        // frame rates) over every image at once, against the answers at each threshold.
        let refs: Vec<&SweepImage> = all_windowed.iter().collect();
        for scope in [
            SweepScope::All,
            SweepScope::Class("person"),
            SweepScope::Small,
        ] {
            let bp = Breakpoints::new(&refs, scope);
            for (g, &t) in grid.iter().enumerate() {
                let at: Vec<&SweepImage> = all_at[g].iter().collect();
                let want = Breakpoints::new(&at, scope).point_at(t);
                let got = bp.point_at(t);
                assert_eq!(got.counts, want.counts, "{scope:?} t {t}");
                assert_eq!(got.per_frame, want.per_frame, "{scope:?} t {t}");
                assert_eq!(got.false_alerts, want.false_alerts, "{scope:?} t {t}");
                assert_eq!(got.frame, want.frame, "{scope:?} t {t}");
            }
        }
        assert!(
            windowed_seen > 50,
            "the rules must make windows: {windowed_seen}"
        );
        // The test can fail: cutting the lowest threshold's answer is wrong on many of them.
        assert!(naive_wrong > 100, "{naive_wrong}");
    }

    /// A fixed threshold (night threshold, night model): the answer at it at every threshold.
    #[test]
    fn a_fixed_threshold_returns_its_answer_at_every_threshold() {
        let mut r = Rng(7);
        let classes = names();
        let dets = random_dets(&mut r);
        let rules = PostRules {
            classes: &classes,
            filters: None,
            object_filter: None,
            verify: None,
        };
        let stored = windows(
            &dets,
            &ctx(),
            Scene::Night,
            Threshold::Fixed(0.3),
            0.05,
            &rules,
            &mut NoAnswers,
        );
        let want = returned_at(&dets, &ctx(), Scene::Night, 0.3, &rules, &mut NoAnswers);
        for t in crate::benchmark::threshold::grid() {
            let got: Vec<PredBox> = stored.iter().filter(|p| p.at(t)).cloned().collect();
            assert_eq!(got.len(), want.eval.len(), "t {t}");
        }
        assert!(stored.iter().all(|p| p.window == Some(Window::ALWAYS)));
    }

    /// B2 by hand: a 0.07 plant drops a 0.8 person at 0.05 only; at 0.5 the person is back,
    /// and a plain run's sweep would have lost it at every threshold.
    #[test]
    fn a_weak_overlap_box_suppresses_only_below_its_score() {
        let classes = names();
        let d = |score: f32, class_id: usize| Detection {
            x1: 100.0,
            y1: 100.0,
            x2: 150.0,
            y2: 200.0,
            score,
            class_id,
        };
        let dets = vec![d(0.8, 0), d(0.07, 2)];
        let filters = FilterRules {
            suppress_overlaps: vec![OverlapRule {
                drop: "person".into(),
                if_overlaps: "potted plant".into(),
                iou: 0.5,
                night_only: false,
            }],
            ..FilterRules::default()
        };
        let rules = PostRules {
            classes: &classes,
            filters: Some(&filters),
            object_filter: None,
            verify: None,
        };
        let stored = windows(
            &dets,
            &ctx(),
            Scene::Day,
            Threshold::Sweep,
            0.05,
            &rules,
            &mut NoAnswers,
        );
        let person = stored.iter().find(|p| p.label == "person").unwrap();
        assert_eq!(
            person.window,
            Some(Window {
                after: 0.07,
                until: 0.8
            })
        );
        assert!(!person.at(0.05) && !person.at(0.07) && person.at(0.08) && person.at(0.5));
        let plant = stored.iter().find(|p| p.label == "potted plant").unwrap();
        assert_eq!(plant.window, None, "returned up to its own score");
        // No rule that changes anything: what a plain run stores.
        let none = FilterRules::default();
        let plain_rules = PostRules {
            filters: Some(&none),
            ..rules
        };
        assert!(
            windows(
                &dets,
                &ctx(),
                Scene::Day,
                Threshold::Sweep,
                0.05,
                &plain_rules,
                &mut NoAnswers
            )
            .iter()
            .all(|p| p.window.is_none())
        );
    }
}
