//! The simulation of the plan's verification: a precision target picked on its 95% Wilson
//! bound holds on fresh frames far more often than one picked on the point estimate (the rule
//! before the evaluation core, [`crate::benchmark::threshold::exact_pick`]).
//!
//! A synthetic detector: every frame has one object, found with probability [`FOUND`] at a
//! confidence `0.25 + 0.75 sqrt(u)`, and 0, 1 or 2 false positives at `0.05 + 0.9 u^3` (u
//! uniform). Each trial draws [`FRAMES`] calibration frames, picks the confidence for
//! `precision:0.9` both ways on the same frames, and measures the true precision at each pick
//! on [`FRESH`] fresh frames.

use super::pick::pick;
use super::stats::Rng;
use super::tally::{sum_all, tallies};
use crate::benchmark::metrics::{Breakpoints, SweepImage, SweepPred, SweepScope};
use crate::benchmark::threshold::{Objective, exact_pick, grid};

const FOUND: f64 = 0.9;
const FRAMES: usize = 500;
const FRESH: usize = 20_000;
const TRIALS: u64 = 200;
const TARGET: f64 = 0.9;

/// One synthetic frame: (confidence, true positive) of its detections.
fn frame(rng: &mut Rng) -> Vec<(f32, bool)> {
    let mut d = Vec::new();
    if rng.unit() < FOUND {
        d.push(((0.25 + 0.75 * rng.unit().sqrt()) as f32, true));
    }
    let fps = (rng.unit() * 3.0).floor() as usize;
    for _ in 0..fps {
        d.push(((0.05 + 0.9 * rng.unit().powi(3)) as f32, false));
    }
    d
}

fn sweep_image(dets: &[(f32, bool)]) -> SweepImage {
    SweepImage {
        gt: vec![("person".into(), false)],
        preds: dets
            .iter()
            .map(|&(c, tp)| SweepPred {
                confidence: c,
                after: f32::NEG_INFINITY,
                class: "person".into(),
                tp,
                small: false,
            })
            .collect(),
        ..SweepImage::default()
    }
}

/// The true precision at `t` on fresh frames.
fn fresh_precision(rng: &mut Rng, t: f32) -> f64 {
    let (mut tp, mut fp) = (0usize, 0usize);
    for _ in 0..FRESH {
        for (c, is_tp) in frame(rng) {
            if c >= t {
                if is_tp {
                    tp += 1;
                } else {
                    fp += 1;
                }
            }
        }
    }
    tp as f64 / (tp + fp).max(1) as f64
}

#[test]
fn a_precision_target_on_its_bound_holds_on_fresh_frames() {
    let objective = Objective::Precision(TARGET);
    let g = grid();
    let (mut bound_holds, mut point_holds, mut trials) = (0u32, 0u32, 0u32);
    for seed in 0..TRIALS {
        let mut rng = Rng::new(0xC0FF_EE00 + seed);
        let frames: Vec<SweepImage> = (0..FRAMES).map(|_| sweep_image(&frame(&mut rng))).collect();
        // The new rule: the grid pick with the target judged on the Wilson lower bound.
        let t = tallies(&frames);
        let new = pick(objective, &sum_all(&t, g.len())).expect("a pick");
        // The old rule: the same grid, the target judged on the point estimate.
        let refs: Vec<&SweepImage> = frames.iter().collect();
        let old = exact_pick(&Breakpoints::new(&refs, SweepScope::All), objective).expect("a pick");
        if !(new.met && old.met) {
            continue;
        }
        trials += 1;
        let new_t = g[new.index];
        bound_holds += u32::from(fresh_precision(&mut rng, new_t) >= TARGET);
        point_holds += u32::from(fresh_precision(&mut rng, old.threshold) >= TARGET);
    }
    let rate = |h: u32| f64::from(h) / f64::from(trials.max(1));
    println!(
        "precision >= {TARGET} held on fresh frames: bound rule {:.1}%, point rule {:.1}% ({trials} \
         trials)",
        100.0 * rate(bound_holds),
        100.0 * rate(point_holds)
    );
    assert!(
        trials >= 190,
        "only {trials} trials met the target both ways"
    );
    assert!(
        rate(bound_holds) >= 0.90,
        "bound rule held {}",
        rate(bound_holds)
    );
    assert!(
        rate(point_holds) <= rate(bound_holds) - 0.2,
        "point rule held {} vs {}",
        rate(point_holds),
        rate(bound_holds)
    );
}
