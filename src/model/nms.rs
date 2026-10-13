//! Class-aware greedy non-maximum suppression.

use super::Detection;

/// Intersection over union of two axis-aligned boxes (`x1,y1,x2,y2`). Inverted boxes have
/// zero area; returns 0 when the union is empty.
pub fn iou(a: &Detection, b: &Detection) -> f32 {
    let iw = (a.x2.min(b.x2) - a.x1.max(b.x1)).max(0.0);
    let ih = (a.y2.min(b.y2) - a.y1.max(b.y1)).max(0.0);
    let inter = iw * ih;
    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

/// Lowest confidence threshold post-processing ever runs with. A request `min_confidence` (or a
/// configured `confidence_threshold`) below this uses the floor instead: with a near-zero
/// threshold every anchor of a YOLO head (25200 for YOLOv5, 8400 for YOLOv8) becomes an NMS
/// candidate. Scores under 1% are noise for a surveillance detector.
pub const MIN_CONFIDENCE_FLOOR: f32 = 0.01;

/// Most candidates that enter NMS (Ultralytics `max_nms` keeps the top 30000; its NMS then
/// sorts and keeps the best 3000 boxes per image). The highest-scoring ones are kept.
pub const MAX_NMS_CANDIDATES: usize = 3000;

/// Most detections reported per image after NMS (Ultralytics `max_det`).
pub const MAX_DETS: usize = 300;

/// Greedy NMS: sort by score (descending) and drop every box whose IoU with a higher-scoring
/// kept box of the same class is `>= iou_threshold`. The result stays sorted by score.
/// Detections with a non-finite coordinate or score are dropped first (they would otherwise
/// sort to the top and suppress nothing, or be suppressed by nothing).
pub fn nms(dets: &mut Vec<Detection>, iou_threshold: f32) {
    super::retain_finite(dets);
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    greedy_sorted(dets, iou_threshold, usize::MAX);
}

/// [`nms`] with bounded cost: keeps the `max_candidates` highest-scoring finite detections
/// (ties broken by input order, exactly as a stable sort would) in O(n), runs NMS on them,
/// and returns at most `max_det` boxes. For inputs with at most `max_candidates` finite
/// detections the result equals `nms` followed by `truncate(max_det)`.
/// Cost: O(n + k log k + k * min(k, max_det)) IoU checks for `k = min(n, max_candidates)`.
pub(crate) fn nms_capped(
    dets: &mut Vec<Detection>,
    iou_threshold: f32,
    max_candidates: usize,
    max_det: usize,
) {
    super::retain_finite(dets);
    keep_top_scores(dets, max_candidates);
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    greedy_sorted(dets, iou_threshold, max_det);
}

/// Keep the `k` highest-scoring detections in their input order (O(n) selection; equal
/// scores prefer the earlier detection).
fn keep_top_scores(dets: &mut Vec<Detection>, k: usize) {
    if dets.len() <= k {
        return;
    }
    if k == 0 {
        dets.clear();
        return;
    }
    let mut order: Vec<(f32, usize)> = dets.iter().map(|d| d.score).zip(0..).collect();
    order.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut keep = vec![false; dets.len()];
    for &(_, i) in order.iter().take(k) {
        if let Some(flag) = keep.get_mut(i) {
            *flag = true;
        }
    }
    let mut flags = keep.into_iter();
    dets.retain(|_| flags.next().unwrap_or(false));
}

/// Greedy suppression over detections already sorted by score (descending), stopping once
/// `max_det` boxes are kept. A box is dropped iff a kept, higher-ranked box of its class
/// overlaps it with IoU `>= iou_threshold` -- the same decision as suppressing forward from
/// every kept box, but each candidate is compared only with the kept boxes of its class.
fn greedy_sorted(dets: &mut Vec<Detection>, iou_threshold: f32, max_det: usize) {
    let mut kept: Vec<Detection> = Vec::new();
    let mut keep = vec![false; dets.len()];
    for (d, flag) in dets.iter().zip(keep.iter_mut()) {
        if kept.len() >= max_det {
            break;
        }
        let overlapped = kept
            .iter()
            .any(|k| k.class_id == d.class_id && iou(k, d) >= iou_threshold);
        if !overlapped {
            kept.push(d.clone());
            *flag = true;
        }
    }
    let mut flags = keep.into_iter();
    dets.retain(|_| flags.next().unwrap_or(false));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class_id: usize) -> Detection {
        Detection {
            x1,
            y1,
            x2,
            y2,
            score,
            class_id,
        }
    }

    #[test]
    fn iou_known_rects() {
        let a = d(0.0, 0.0, 10.0, 10.0, 1.0, 0);
        let b = d(5.0, 0.0, 15.0, 10.0, 1.0, 0);
        // inter 50, union 150
        assert!((iou(&a, &b) - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(iou(&a, &a), 1.0);
        let c = d(20.0, 20.0, 30.0, 30.0, 1.0, 0);
        assert_eq!(iou(&a, &c), 0.0);
        let inner = d(0.0, 0.0, 5.0, 5.0, 1.0, 0);
        assert!((iou(&a, &inner) - 0.25).abs() < 1e-6);
        let degenerate = d(1.0, 1.0, 1.0, 1.0, 1.0, 0);
        assert_eq!(iou(&degenerate, &degenerate), 0.0);
    }

    #[test]
    fn same_class_suppressed() {
        let a = d(0.0, 0.0, 100.0, 100.0, 0.7, 1);
        let b = d(0.0, 11.0, 100.0, 111.0, 0.9, 1); // inter 8900, union 11100 -> ~0.80
        assert!((iou(&a, &b) - 8900.0 / 11100.0).abs() < 1e-5);
        let mut v = vec![a, b.clone()];
        nms(&mut v, 0.5);
        assert_eq!(v, vec![b]);
    }

    #[test]
    fn different_classes_kept() {
        let a = d(0.0, 0.0, 100.0, 100.0, 0.7, 1);
        let b = d(0.0, 11.0, 100.0, 111.0, 0.9, 2);
        let mut v = vec![a, b];
        nms(&mut v, 0.5);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].score, 0.9);
    }

    #[test]
    fn chain_and_threshold() {
        // b overlaps a strongly; c overlaps b but barely a -> c survives once b is suppressed.
        let a = d(0.0, 0.0, 10.0, 10.0, 0.9, 0);
        let b = d(2.0, 0.0, 12.0, 10.0, 0.8, 0);
        let c = d(9.0, 0.0, 19.0, 10.0, 0.7, 0);
        let mut v = vec![c.clone(), b, a.clone()];
        nms(&mut v, 0.5);
        assert_eq!(v, vec![a, c]);

        let x = d(0.0, 0.0, 10.0, 10.0, 0.9, 0);
        let y = d(5.0, 0.0, 15.0, 10.0, 0.8, 0); // IoU 1/3
        let mut v = vec![x.clone(), y.clone()];
        nms(&mut v, 0.34);
        assert_eq!(v.len(), 2);
        let mut v = vec![x, y];
        nms(&mut v, 0.33);
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn non_finite_dropped_before_suppression() {
        let good = d(0.0, 0.0, 10.0, 10.0, 0.9, 0);
        let mut v = vec![
            d(f32::NAN, 0.0, 10.0, 10.0, 0.99, 0),
            d(0.0, 0.0, 10.0, 10.0, f32::INFINITY, 0),
            d(0.0, 0.0, 10.0, 10.0, f32::NAN, 0),
            d(0.0, 0.0, f32::INFINITY, 10.0, 0.95, 0),
            good.clone(),
        ];
        nms(&mut v, 0.5);
        assert_eq!(v, vec![good]);
    }

    /// The original forward-suppression NMS, kept as the reference.
    fn reference_nms(dets: &mut Vec<Detection>, iou_threshold: f32) {
        crate::model::retain_finite(dets);
        dets.sort_by(|a, b| b.score.total_cmp(&a.score));
        let n = dets.len();
        let mut suppressed = vec![false; n];
        for i in 0..n {
            if suppressed[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !suppressed[j]
                    && dets[j].class_id == dets[i].class_id
                    && iou(&dets[i], &dets[j]) >= iou_threshold
                {
                    suppressed[j] = true;
                }
            }
        }
        let mut flags = suppressed.into_iter();
        dets.retain(|_| !flags.next().unwrap_or(false));
    }

    /// Deterministic pseudo-random boxes (LCG), with repeated scores to exercise ties.
    fn random_dets(n: usize, seed: u64) -> Vec<Detection> {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            f32::from(u16::try_from(state >> 48).unwrap_or(0)) / 65535.0
        };
        (0..n)
            .map(|_| {
                let (x, y) = (next() * 600.0, next() * 600.0);
                let (w, h) = (10.0 + next() * 80.0, 10.0 + next() * 80.0);
                let score = (next() * 50.0).round() / 50.0;
                let class_id = usize::from(next() < 0.3);
                d(x, y, x + w, y + h, score, class_id)
            })
            .collect()
    }

    #[test]
    fn greedy_matches_reference() {
        for seed in 0..20 {
            let dets = random_dets(400, seed);
            for thr in [0.3, 0.45, 0.7] {
                let mut a = dets.clone();
                let mut b = dets.clone();
                nms(&mut a, thr);
                reference_nms(&mut b, thr);
                assert_eq!(a, b, "seed {seed} thr {thr}");
                // Under both caps, nms_capped equals nms + truncate.
                let mut c = dets.clone();
                nms_capped(&mut c, thr, MAX_NMS_CANDIDATES, MAX_DETS);
                b.truncate(MAX_DETS);
                assert_eq!(c, b, "seed {seed} thr {thr}");
            }
        }
    }

    #[test]
    fn capped_equals_reference_on_top_k() {
        // Above the candidate cap: identical to a stable sort, truncate to k, then NMS.
        for seed in 0..10 {
            let dets = random_dets(1000, seed);
            let mut expected = dets.clone();
            expected.sort_by(|a, b| b.score.total_cmp(&a.score));
            expected.truncate(150);
            reference_nms(&mut expected, 0.45);
            expected.truncate(40);
            let mut got = dets;
            nms_capped(&mut got, 0.45, 150, 40);
            assert_eq!(got, expected, "seed {seed}");
        }
    }

    #[test]
    fn keep_top_scores_keeps_best_in_input_order() {
        let mut v: Vec<Detection> = [0.2, 0.9, 0.5, 0.9, 0.1, 0.5, 0.7]
            .iter()
            .enumerate()
            .map(|(i, &s)| d(i as f32, 0.0, i as f32 + 1.0, 1.0, s, 0))
            .collect();
        keep_top_scores(&mut v, 4);
        let got: Vec<(f32, f32)> = v.iter().map(|d| (d.x1, d.score)).collect();
        // Ties at 0.5: the earlier one (index 2) wins.
        assert_eq!(got, vec![(1.0, 0.9), (2.0, 0.5), (3.0, 0.9), (6.0, 0.7)]);
        keep_top_scores(&mut v, 10);
        assert_eq!(v.len(), 4);
        keep_top_scores(&mut v, 0);
        assert!(v.is_empty());
    }

    #[test]
    fn capped_limits_and_constants() {
        assert_eq!(MIN_CONFIDENCE_FLOOR, 0.01);
        assert_eq!(MAX_NMS_CANDIDATES, 3000);
        assert_eq!(MAX_DETS, 300);
        // 10000 disjoint boxes: none suppress each other, so only the caps limit the output.
        let mut v: Vec<Detection> = (0..10_000)
            .map(|i| {
                let (x, y) = ((i % 100) as f32 * 20.0, (i / 100) as f32 * 20.0);
                d(x, y, x + 10.0, y + 10.0, i as f32 / 10_000.0, 0)
            })
            .collect();
        nms_capped(&mut v, 0.45, MAX_NMS_CANDIDATES, MAX_DETS);
        assert_eq!(v.len(), MAX_DETS);
        assert_eq!(v[0].score, 9_999.0 / 10_000.0);
        assert_eq!(v[MAX_DETS - 1].score, 9_700.0 / 10_000.0);
        // Non-finite detections never take a candidate slot.
        let mut v = vec![
            d(0.0, 0.0, 1.0, 1.0, f32::NAN, 0),
            d(0.0, 0.0, 1.0, 1.0, f32::INFINITY, 0),
            d(5.0, 5.0, 6.0, 6.0, 0.3, 0),
        ];
        nms_capped(&mut v, 0.45, 1, 1);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].score, 0.3);
    }

    /// Seeded xorshift64 (no dependency; deterministic across platforms).
    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        /// Uniform in [0, 1) with 24 bits.
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// 0..200 detections around a few anchors (so many overlap), 3 classes, scores on a coarse
    /// grid (many ties), plus zero-area, inverted, out-of-image and non-finite boxes, and pairs
    /// whose IoU is exactly 0.5 (`[x, y, x+w, y+h]` and `[x, y, x+2w, y+h]`).
    fn random_set(rng: &mut XorShift) -> Vec<Detection> {
        let n = rng.below(200) as usize;
        let anchors: Vec<(f32, f32)> = (0..1 + rng.below(6))
            .map(|_| (rng.unit() * 640.0, rng.unit() * 640.0))
            .collect();
        let mut v = Vec::with_capacity(n + 8);
        while v.len() < n {
            let (ax, ay) = anchors[rng.below(anchors.len() as u64) as usize];
            let x = ax + (rng.unit() - 0.5) * 60.0;
            let y = ay + (rng.unit() - 0.5) * 60.0;
            let w = 1.0 + rng.unit() * 120.0;
            let h = 1.0 + rng.unit() * 120.0;
            let score = match rng.below(4) {
                0 => [0.25, 0.5, 0.75][rng.below(3) as usize],
                _ => (rng.unit() * 20.0).floor() / 20.0,
            };
            let class_id = rng.below(3) as usize;
            let b = match rng.below(12) {
                0 => d(x, y, x, y + h, score, class_id),     // zero width
                1 => d(x, y, x + w, y, score, class_id),     // zero height
                2 => d(x + w, y + h, x, y, score, class_id), // inverted
                3 => d(-x - w, -y, -x, -y + h, score, class_id), // left of / above the image
                4 => d(x + 700.0, y, x + 700.0 + w, y + h, score, class_id), // right of it
                5 if rng.below(8) == 0 => d(x, y, x + w, y + h, f32::NAN, class_id),
                6 => {
                    // IoU exactly 0.5 with the next box (integer coordinates: exact in f32).
                    let (x, y, w, h) = (x.floor(), y.floor(), w.floor().max(1.0), h.floor());
                    v.push(d(x, y, x + 2.0 * w, y + h + 1.0, score, class_id));
                    d(x, y, x + w, y + h + 1.0, score, class_id)
                }
                7 => v
                    .last()
                    .cloned()
                    .unwrap_or_else(|| d(x, y, x + w, y + h, score, class_id)),
                _ => d(x, y, x + w, y + h, score, class_id),
            };
            v.push(b);
        }
        v
    }

    /// `sub` is a sub-multiset of `of`; returns `of` minus `sub`.
    fn multiset_minus(of: &[Detection], sub: &[Detection]) -> Vec<Detection> {
        let mut rest = of.to_vec();
        for s in sub {
            let i = rest
                .iter()
                .position(|r| r == s)
                .unwrap_or_else(|| panic!("{s:?} is not in the input"));
            rest.swap_remove(i);
        }
        rest
    }

    #[test]
    fn nms_properties_on_random_sets() {
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let mut exact_half_seen = 0usize;
        for case in 0..300 {
            let input = random_set(&mut rng);
            let finite: Vec<Detection> = input
                .iter()
                .filter(|d| {
                    [d.x1, d.y1, d.x2, d.y2, d.score]
                        .iter()
                        .all(|v| v.is_finite())
                })
                .cloned()
                .collect();
            for thr in [0.3, 0.45, 0.5, 0.7] {
                let mut out = input.clone();
                nms(&mut out, thr);
                let ctx = format!("case {case} thr {thr}");
                // (a) A subset of the finite input, sorted by score.
                let dropped = multiset_minus(&finite, &out);
                assert!(out.windows(2).all(|w| w[0].score >= w[1].score), "{ctx}");
                // (b) No two kept boxes of one class overlap with IoU >= the threshold.
                for (i, a) in out.iter().enumerate() {
                    for b in &out[i + 1..] {
                        assert!(
                            a.class_id != b.class_id || iou(a, b) < thr,
                            "{ctx}: kept {a:?} and {b:?}"
                        );
                    }
                }
                // (c) Every dropped box overlaps a kept box of its class (IoU >= threshold)
                // whose score is at least its own.
                for x in &dropped {
                    if thr == 0.5
                        && out
                            .iter()
                            .any(|k| k.class_id == x.class_id && iou(k, x) == 0.5)
                    {
                        exact_half_seen += 1;
                    }
                    assert!(
                        out.iter().any(|k| k.class_id == x.class_id
                            && iou(k, x) >= thr
                            && k.score >= x.score),
                        "{ctx}: {x:?} dropped without a suppressor"
                    );
                }
                // Idempotent: the kept set suppresses nothing further.
                let mut again = out.clone();
                nms(&mut again, thr);
                assert_eq!(again, out, "{ctx}");
                // (d) Below the candidate cap, nms_capped == nms followed by truncate(cap).
                for cap in [0, 1, 7, 50, MAX_DETS] {
                    let mut capped = input.clone();
                    nms_capped(&mut capped, thr, finite.len().max(1), cap);
                    let mut expected = out.clone();
                    expected.truncate(cap);
                    assert_eq!(capped, expected, "{ctx} cap {cap}");
                }
            }
        }
        // The IoU == threshold boundary (suppressed: `>=`) was exercised.
        assert!(exact_half_seen > 0);
    }
}
