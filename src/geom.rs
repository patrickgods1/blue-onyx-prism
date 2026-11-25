//! Geometry of `[x0, y0, x1, y1]` boxes in `f32`: area, IoU, the share of the smaller box inside
//! the other, the "proper box" test and the lexicographic box order. One copy for the inference
//! path ([`crate::verify::iou`]) and the review labels (re-exported as `collect::labels::iou` /
//! `inside_share`), std-only so neither side depends on the other.
//!
//! The page's twins (`iou`, `insideShare` in `assets/review-rules.js`) are parity-tested in
//! `tests/it/js_rules.rs`; keep the order of the `f32` operations as it is, so results stay
//! bit-identical for valid boxes.

use std::cmp::Ordering;

/// The area of a box, each side clamped at 0 (0 for an empty or inverted box).
pub fn area(b: &[f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

/// Width and height of the intersection of two boxes, each clamped at 0.
fn overlap(a: &[f32; 4], b: &[f32; 4]) -> (f32, f32) {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    (iw, ih)
}

/// Intersection over union of two boxes (0 for empty boxes).
pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let (iw, ih) = overlap(a, b);
    let inter = iw * ih;
    let union = area(a) + area(b) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// How much of the smaller of two boxes lies inside the other: their intersection over the
/// smaller area (0 for an empty box). 1 = one is wholly inside the other.
pub fn inside_share(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let (iw, ih) = overlap(a, b);
    let smaller = area(a).min(area(b));
    if smaller > 0.0 {
        iw * ih / smaller
    } else {
        0.0
    }
}

/// Finite coordinates of a box with a positive size.
pub fn proper_box(b: &[f32; 4]) -> bool {
    b.iter().all(|v| v.is_finite()) && b[2] > b[0] && b[3] > b[1]
}

/// Boxes in lexicographic order of their coordinates by `f32::total_cmp` (a total order: NaN and
/// -0.0 have their places), so lists sorted by box come out the same whatever their input order.
pub fn cmp_box(a: &[f32; 4], b: &[f32; 4]) -> Ordering {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.total_cmp(y))
        .find(|o| o.is_ne())
        .unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_clamps_each_side() {
        assert_eq!(area(&[0.0, 0.0, 10.0, 5.0]), 50.0);
        assert_eq!(area(&[10.0, 0.0, 0.0, 5.0]), 0.0);
        assert_eq!(area(&[0.0, 5.0, 10.0, 0.0]), 0.0);
        // Both sides inverted: still 0, never a positive product of two negatives.
        assert_eq!(area(&[10.0, 10.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn iou_and_inside_share_of_known_boxes() {
        let a = [0.0, 0.0, 10.0, 10.0];
        let b = [5.0, 0.0, 15.0, 10.0];
        // Intersection 50, union 150.
        assert_eq!(iou(&a, &b), 50.0 / 150.0);
        assert_eq!(iou(&a, &a), 1.0);
        assert_eq!(iou(&a, &[20.0, 20.0, 30.0, 30.0]), 0.0);
        assert_eq!(iou(&[0.0; 4], &[0.0; 4]), 0.0, "empty boxes");
        let inner = [2.0, 2.0, 4.0, 4.0];
        assert_eq!(inside_share(&a, &inner), 1.0);
        assert_eq!(inside_share(&inner, &a), 1.0, "symmetric");
        assert_eq!(inside_share(&a, &b), 0.5);
        assert_eq!(inside_share(&a, &[3.0, 3.0, 3.0, 9.0]), 0.0, "empty box");
    }

    // The same arithmetic as the copies this module replaced, bit for bit, on a grid of boxes.
    #[test]
    fn results_are_bit_identical_to_the_inline_formulas() {
        let old_iou = |a: [f32; 4], b: [f32; 4]| {
            let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
            let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
            let inter = iw * ih;
            let area = |r: [f32; 4]| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
            let union = area(a) + area(b) - inter;
            if union > 0.0 { inter / union } else { 0.0 }
        };
        let old_inside = |a: [f32; 4], b: [f32; 4]| {
            let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
            let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
            let area = |r: [f32; 4]| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
            let smaller = area(a).min(area(b));
            if smaller > 0.0 {
                iw * ih / smaller
            } else {
                0.0
            }
        };
        let v = [0.0f32, 0.1, 0.33, 0.5, 0.71, 1.0];
        let boxes: Vec<[f32; 4]> = v
            .iter()
            .flat_map(|&x0| v.iter().map(move |&y0| [x0, y0, x0 + 0.37, y0 + 0.29]))
            .chain([[0.2, 0.2, 0.1, 0.9], [0.5, 0.5, 0.5, 0.5]])
            .collect();
        for a in &boxes {
            for b in &boxes {
                assert_eq!(
                    iou(a, b).to_bits(),
                    old_iou(*a, *b).to_bits(),
                    "{a:?} {b:?}"
                );
                assert_eq!(
                    inside_share(a, b).to_bits(),
                    old_inside(*a, *b).to_bits(),
                    "{a:?} {b:?}"
                );
            }
        }
    }

    #[test]
    fn proper_box_wants_finite_positive_sides() {
        assert!(proper_box(&[0.0, 0.0, 1.0, 1.0]));
        assert!(!proper_box(&[0.0, 0.0, 0.0, 1.0]), "zero width");
        assert!(!proper_box(&[0.0, 1.0, 1.0, 0.5]), "inverted");
        assert!(!proper_box(&[0.0, 0.0, f32::NAN, 1.0]));
        assert!(!proper_box(&[f32::NEG_INFINITY, 0.0, 1.0, 1.0]));
    }

    #[test]
    fn cmp_box_is_lexicographic_and_total() {
        let a = [0.0, 0.0, 1.0, 1.0];
        assert_eq!(cmp_box(&a, &a), Ordering::Equal);
        assert_eq!(cmp_box(&a, &[0.0, 0.0, 1.0, 2.0]), Ordering::Less);
        assert_eq!(cmp_box(&[0.1, 0.0, 0.0, 0.0], &a), Ordering::Greater);
        // total_cmp: -0.0 sorts before 0.0, NaN after every number.
        assert_eq!(cmp_box(&[-0.0, 0.0, 1.0, 1.0], &a), Ordering::Less);
        assert_eq!(
            cmp_box(&[f32::NAN, 0.0, 0.0, 0.0], &[f32::INFINITY, 0.0, 0.0, 0.0]),
            Ordering::Greater
        );
    }
}
