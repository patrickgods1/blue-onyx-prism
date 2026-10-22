//! Zero-compute false-positive rules on a model's detections (night/IR thresholds, per-label
//! thresholds, person shape priors, overlap suppression). Pure: the worker applies [`apply`]
//! between the family postprocess and [`super::to_predictions`], and the benchmark calls the
//! same function, so the two cannot drift apart. Every rule defaults to off; the family has
//! already applied the base confidence threshold, so these rules only ever remove boxes.
//!
//! Order per detection: thresholds, then person aspect, then minimum area, then overlap rules.
//! Each removed box is counted once, under the first rule that removed it.

use super::Detection;
use crate::image_stats::Scene;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Default IoU of an [`OverlapRule`].
pub const DEFAULT_OVERLAP_IOU: f32 = 0.5;
/// An [`OverlapRule`] also fires when the `if_overlaps` box covers at least this share of the
/// `drop` box's area (a small "person" inside a large "potted plant" has a low IoU).
pub const OVERLAP_COVER_FRAC: f32 = 0.8;
/// The label the person shape prior applies to.
pub const PERSON: &str = "person";

fn default_overlap_iou() -> f32 {
    DEFAULT_OVERLAP_IOU
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Drop a `drop` box that overlaps an `if_overlaps` box: IoU >= `iou`, or the `if_overlaps`
/// box covers >= [`OVERLAP_COVER_FRAC`] of the `drop` box. Labels match case-insensitively.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlapRule {
    pub drop: String,
    pub if_overlaps: String,
    #[serde(
        default = "default_overlap_iou",
        serialize_with = "crate::config::thr::ser_clean"
    )]
    pub iou: f32,
    /// Only on night / IR frames.
    #[serde(default, skip_serializing_if = "is_false")]
    pub night_only: bool,
}

/// The resolved rules for one request (per-model values over the global ones).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FilterRules {
    /// Minimum score on night / IR frames (all labels).
    pub night_confidence_threshold: Option<f32>,
    /// Minimum score per label (case-insensitive keys).
    pub label_thresholds: BTreeMap<String, f32>,
    /// Minimum score per label on night / IR frames.
    pub night_label_thresholds: BTreeMap<String, f32>,
    /// Allowed width / height of a `person` box, `[min, max]`.
    pub person_aspect: Option<[f32; 2]>,
    /// Minimum box area as a share of the image area.
    pub min_box_area_frac: Option<f32>,
    pub suppress_overlaps: Vec<OverlapRule>,
}

impl FilterRules {
    /// No rule is set: [`apply`] returns its input unchanged.
    pub fn is_noop(&self) -> bool {
        self.night_confidence_threshold.is_none()
            && self.label_thresholds.is_empty()
            && self.night_label_thresholds.is_empty()
            && self.person_aspect.is_none()
            && self.min_box_area_frac.is_none()
            && self.suppress_overlaps.is_empty()
    }
}

/// Why a detection was removed (`detections_suppressed_total{reason}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SuppressReason {
    NightThreshold,
    LabelThreshold,
    Aspect,
    MinArea,
    Overlap,
    /// Static-box memory ([`crate::static_memory`]), counted by the worker.
    Static,
}

impl SuppressReason {
    pub const ALL: [SuppressReason; 6] = [
        SuppressReason::NightThreshold,
        SuppressReason::LabelThreshold,
        SuppressReason::Aspect,
        SuppressReason::MinArea,
        SuppressReason::Overlap,
        SuppressReason::Static,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            SuppressReason::NightThreshold => "night_threshold",
            SuppressReason::LabelThreshold => "label_threshold",
            SuppressReason::Aspect => "aspect",
            SuppressReason::MinArea => "min_area",
            SuppressReason::Overlap => "overlap",
            SuppressReason::Static => "static",
        }
    }
}

/// Removed detections per [`SuppressReason`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SuppressCounts([u32; SuppressReason::ALL.len()]);

impl SuppressCounts {
    pub fn get(&self, r: SuppressReason) -> u32 {
        self.0.get(r as usize).copied().unwrap_or(0)
    }

    pub fn add(&mut self, r: SuppressReason, n: u32) {
        if let Some(c) = self.0.get_mut(r as usize) {
            *c = c.saturating_add(n);
        }
    }

    pub fn total(&self) -> u32 {
        self.0.iter().fold(0u32, |a, b| a.saturating_add(*b))
    }

    /// `(reason, count)` for every reason with a count, for logs.
    pub fn nonzero(&self) -> impl Iterator<Item = (SuppressReason, u32)> + '_ {
        SuppressReason::ALL
            .into_iter()
            .map(|r| (r, self.get(r)))
            .filter(|(_, n)| *n > 0)
    }
}

/// The label of class `id`, as [`super::to_predictions`] reports it.
pub fn label_of(class_names: &[String], id: usize) -> std::borrow::Cow<'_, str> {
    match class_names.get(id) {
        Some(s) => std::borrow::Cow::Borrowed(s.as_str()),
        None => std::borrow::Cow::Owned(format!("class_{id}")),
    }
}

/// The value of `label` in `map`, matched case-insensitively.
fn lookup(map: &BTreeMap<String, f32>, label: &str) -> Option<f32> {
    map.get(label).copied().or_else(|| {
        map.iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(label))
            .map(|(_, v)| *v)
    })
}

/// A box clamped to the image: `(x1, y1, x2, y2)`.
fn clamped(d: &Detection, w: f32, h: f32) -> (f32, f32, f32, f32) {
    (
        d.x1.clamp(0.0, w),
        d.y1.clamp(0.0, h),
        d.x2.clamp(0.0, w),
        d.y2.clamp(0.0, h),
    )
}

/// Does `other` suppress `d` under `rule` (IoU or cover, see [`OverlapRule`])?
fn overlaps(d: &Detection, other: &Detection, iou: f32) -> bool {
    if super::nms::iou(d, other) >= iou {
        return true;
    }
    let iw = (d.x2.min(other.x2) - d.x1.max(other.x1)).max(0.0);
    let ih = (d.y2.min(other.y2) - d.y1.max(other.y1)).max(0.0);
    let area = (d.x2 - d.x1).max(0.0) * (d.y2 - d.y1).max(0.0);
    area > 0.0 && iw * ih >= OVERLAP_COVER_FRAC * area
}

/// Apply `rules` to `dets` of an `img_w` x `img_h` frame of `scene`; `class_names` resolve
/// the labels (as in [`super::to_predictions`]). Returns the kept detections in their input
/// order and how many each rule removed.
///
/// - thresholds: a box is removed when its score is below its label's `label_thresholds`
///   entry, or, on night / IR frames, below `night_confidence_threshold` or its label's
///   `night_label_thresholds` entry. These only raise the family's threshold; a value at or
///   below it does nothing. Counted as `night_threshold` when a night value removed it, else
///   `label_threshold`;
/// - `person_aspect`: a `person` box whose width / height is outside `[min, max]`;
/// - `min_box_area_frac`: a box whose area is below this share of the image area;
/// - `suppress_overlaps`: the evidence boxes (`if_overlaps`) are all input detections, also
///   those removed by an earlier rule or later by the object filter (a "potted plant" box is
///   evidence even when plants are not reported).
///
/// Non-finite or empty boxes pass through untouched ([`super::to_predictions`] drops them).
pub fn apply(
    dets: Vec<Detection>,
    class_names: &[String],
    img_w: u32,
    img_h: u32,
    scene: Scene,
    rules: &FilterRules,
) -> (Vec<Detection>, SuppressCounts) {
    let mut counts = SuppressCounts::default();
    if rules.is_noop() || dets.is_empty() {
        return (dets, counts);
    }
    let night = scene.is_night();
    let (w, h) = (img_w.max(1) as f32, img_h.max(1) as f32);
    let image_area = w * h;
    let overlap_rules: Vec<&OverlapRule> = rules
        .suppress_overlaps
        .iter()
        .filter(|r| night || !r.night_only)
        .collect();
    let labels: Vec<std::borrow::Cow<'_, str>> = dets
        .iter()
        .map(|d| label_of(class_names, d.class_id))
        .collect();
    let mut keep = Vec::with_capacity(dets.len());
    for (i, d) in dets.iter().enumerate() {
        let label = labels.get(i).map(|l| l.as_ref()).unwrap_or_default();
        let (x1, y1, x2, y2) = clamped(d, w, h);
        let (bw, bh) = (x2 - x1, y2 - y1);
        if !d.is_finite() || bw <= 0.0 || bh <= 0.0 {
            keep.push(d.clone());
            continue;
        }
        let reason = 'rules: {
            if let Some(t) = lookup(&rules.label_thresholds, label)
                && d.score < t
            {
                break 'rules Some(SuppressReason::LabelThreshold);
            }
            if night {
                let t = [
                    rules.night_confidence_threshold,
                    lookup(&rules.night_label_thresholds, label),
                ]
                .into_iter()
                .flatten()
                .fold(f32::NEG_INFINITY, f32::max);
                if d.score < t {
                    break 'rules Some(SuppressReason::NightThreshold);
                }
            }
            if let Some([lo, hi]) = rules.person_aspect
                && label.eq_ignore_ascii_case(PERSON)
            {
                let aspect = bw / bh;
                if aspect < lo || aspect > hi {
                    break 'rules Some(SuppressReason::Aspect);
                }
            }
            if let Some(frac) = rules.min_box_area_frac
                && bw * bh < frac * image_area
            {
                break 'rules Some(SuppressReason::MinArea);
            }
            for r in &overlap_rules {
                if !label.eq_ignore_ascii_case(&r.drop) {
                    continue;
                }
                let hit = dets.iter().enumerate().any(|(j, o)| {
                    j != i
                        && o.is_finite()
                        && labels
                            .get(j)
                            .is_some_and(|l| l.eq_ignore_ascii_case(&r.if_overlaps))
                        && overlaps(d, o, r.iou)
                });
                if hit {
                    break 'rules Some(SuppressReason::Overlap);
                }
            }
            None
        };
        match reason {
            Some(r) => counts.add(r, 1),
            None => keep.push(d.clone()),
        }
    }
    (keep, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        ["person", "car", "potted plant"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn det(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class_id: usize) -> Detection {
        Detection {
            x1,
            y1,
            x2,
            y2,
            score,
            class_id,
        }
    }

    /// A tall person box (aspect 0.5), a car and a plant.
    fn sample() -> Vec<Detection> {
        vec![
            det(100.0, 100.0, 150.0, 200.0, 0.55, 0),
            det(300.0, 300.0, 500.0, 400.0, 0.9, 1),
            det(600.0, 100.0, 700.0, 250.0, 0.7, 2),
        ]
    }

    fn run(
        dets: Vec<Detection>,
        scene: Scene,
        rules: &FilterRules,
    ) -> (Vec<Detection>, SuppressCounts) {
        apply(dets, &names(), 1000, 800, scene, rules)
    }

    #[test]
    fn defaults_are_a_no_op() {
        let rules = FilterRules::default();
        assert!(rules.is_noop());
        for scene in [Scene::Day, Scene::Night, Scene::Ir] {
            let (out, c) = run(sample(), scene, &rules);
            assert_eq!(out, sample());
            assert_eq!(c.total(), 0);
        }
    }

    #[test]
    fn night_threshold_applies_only_at_night() {
        let rules = FilterRules {
            night_confidence_threshold: Some(0.6),
            ..Default::default()
        };
        let (out, c) = run(sample(), Scene::Day, &rules);
        assert_eq!(out.len(), 3);
        assert_eq!(c.total(), 0);
        for scene in [Scene::Night, Scene::Ir] {
            let (out, c) = run(sample(), scene, &rules);
            assert_eq!(out.len(), 2, "the 0.55 person goes");
            assert!(out.iter().all(|d| d.class_id != 0));
            assert_eq!(c.get(SuppressReason::NightThreshold), 1);
        }
        // A lower night threshold never lowers the family's threshold (nothing to add back).
        let low = FilterRules {
            night_confidence_threshold: Some(0.1),
            ..Default::default()
        };
        assert_eq!(run(sample(), Scene::Ir, &low).0, sample());
    }

    #[test]
    fn label_thresholds_are_case_insensitive_with_night_variants() {
        let rules = FilterRules {
            label_thresholds: BTreeMap::from([("Car".to_string(), 0.95)]),
            night_label_thresholds: BTreeMap::from([("PERSON".to_string(), 0.6)]),
            ..Default::default()
        };
        let (out, c) = run(sample(), Scene::Day, &rules);
        assert_eq!(out.len(), 2);
        assert_eq!(c.get(SuppressReason::LabelThreshold), 1);
        let (out, c) = run(sample(), Scene::Night, &rules);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].class_id, 2);
        assert_eq!(c.get(SuppressReason::LabelThreshold), 1);
        assert_eq!(c.get(SuppressReason::NightThreshold), 1);
        assert_eq!(c.total(), 2);
    }

    #[test]
    fn person_aspect_only_for_person() {
        let rules = FilterRules {
            person_aspect: Some([0.2, 0.9]),
            ..Default::default()
        };
        // Tall person (0.5) kept; wide car untouched.
        assert_eq!(run(sample(), Scene::Day, &rules).0, sample());
        // A wide "person" (a rock): 2.0.
        let rock = det(100.0, 100.0, 300.0, 200.0, 0.8, 0);
        let (out, c) = run(vec![rock], Scene::Day, &rules);
        assert!(out.is_empty());
        assert_eq!(c.get(SuppressReason::Aspect), 1);
        // A sliver: 0.1.
        let (out, _) = run(vec![det(0.0, 0.0, 10.0, 100.0, 0.8, 0)], Scene::Ir, &rules);
        assert!(out.is_empty());
        // The aspect is measured on the box clamped to the image.
        let (out, _) = run(
            vec![det(-50.0, 0.0, 50.0, 100.0, 0.8, 0)],
            Scene::Day,
            &rules,
        );
        assert_eq!(out.len(), 1, "clamped to 50x100 (0.5)");
    }

    #[test]
    fn min_box_area() {
        let rules = FilterRules {
            min_box_area_frac: Some(0.01), // 8000 px of 1000x800
            ..Default::default()
        };
        let (out, c) = run(sample(), Scene::Day, &rules);
        // person 50x100 = 5000 < 8000 goes; car 20000 and plant 15000 stay.
        assert_eq!(out.len(), 2);
        assert_eq!(c.get(SuppressReason::MinArea), 1);
    }

    #[test]
    fn overlap_rule_iou_and_cover() {
        let rules = FilterRules {
            suppress_overlaps: vec![OverlapRule {
                drop: "Person".into(),
                if_overlaps: "potted plant".into(),
                iou: 0.5,
                night_only: false,
            }],
            ..Default::default()
        };
        // Same box as the plant: IoU 1.
        let mut dets = sample();
        dets.push(det(600.0, 100.0, 700.0, 250.0, 0.8, 0));
        let (out, c) = run(dets, Scene::Day, &rules);
        assert_eq!(out, sample(), "only the person on the plant goes");
        assert_eq!(c.get(SuppressReason::Overlap), 1);
        // A small person inside the plant: low IoU, but covered.
        let inner = det(620.0, 120.0, 650.0, 180.0, 0.8, 0);
        let plant = det(600.0, 100.0, 700.0, 250.0, 0.7, 2);
        assert!(crate::model::nms::iou(&inner, &plant) < 0.5);
        let (out, _) = run(vec![inner.clone(), plant.clone()], Scene::Day, &rules);
        assert_eq!(out, vec![plant.clone()]);
        // Half outside the plant: neither.
        let half = det(650.0, 100.0, 750.0, 250.0, 0.8, 0);
        let (out, _) = run(vec![half, plant.clone()], Scene::Day, &rules);
        assert_eq!(out.len(), 2);
        // The plant is evidence even when a threshold removes it.
        let strict = FilterRules {
            label_thresholds: BTreeMap::from([("potted plant".into(), 0.99)]),
            ..rules.clone()
        };
        let (out, c) = run(vec![inner, plant], Scene::Day, &strict);
        assert!(out.is_empty());
        assert_eq!(c.get(SuppressReason::Overlap), 1);
        assert_eq!(c.get(SuppressReason::LabelThreshold), 1);
    }

    #[test]
    fn overlap_rule_night_only() {
        let rules = FilterRules {
            suppress_overlaps: vec![OverlapRule {
                drop: "person".into(),
                if_overlaps: "potted plant".into(),
                iou: 0.5,
                night_only: true,
            }],
            ..Default::default()
        };
        let pair = vec![
            det(600.0, 100.0, 700.0, 250.0, 0.8, 0),
            det(600.0, 100.0, 700.0, 250.0, 0.7, 2),
        ];
        assert_eq!(run(pair.clone(), Scene::Day, &rules).0.len(), 2);
        assert_eq!(run(pair.clone(), Scene::Ir, &rules).0.len(), 1);
        assert_eq!(run(pair, Scene::Night, &rules).0.len(), 1);
    }

    #[test]
    fn non_finite_and_unknown_classes_pass_through() {
        let rules = FilterRules {
            min_box_area_frac: Some(0.5),
            label_thresholds: BTreeMap::from([("class_9".into(), 0.9)]),
            ..Default::default()
        };
        let nan = det(f32::NAN, 0.0, 10.0, 10.0, 0.9, 0);
        let (out, _) = run(vec![nan], Scene::Day, &rules);
        assert_eq!(out.len(), 1);
        // Unknown class ids resolve to `class_<id>` like the response.
        let (out, c) = run(
            vec![det(0.0, 0.0, 900.0, 700.0, 0.5, 9)],
            Scene::Day,
            &rules,
        );
        assert!(out.is_empty());
        assert_eq!(c.get(SuppressReason::LabelThreshold), 1);
    }

    #[test]
    fn overlap_rule_serde_defaults() {
        let r: OverlapRule =
            serde_json::from_str(r#"{"drop":"person","if_overlaps":"potted plant"}"#).unwrap();
        assert_eq!(r.iou, DEFAULT_OVERLAP_IOU);
        assert!(!r.night_only);
        assert_eq!(
            serde_json::to_string(&r).unwrap(),
            r#"{"drop":"person","if_overlaps":"potted plant","iou":0.5}"#
        );
    }

    #[test]
    fn counts() {
        let mut c = SuppressCounts::default();
        c.add(SuppressReason::Static, 2);
        c.add(SuppressReason::Aspect, 1);
        assert_eq!(c.total(), 3);
        assert_eq!(
            c.nonzero().collect::<Vec<_>>(),
            vec![(SuppressReason::Aspect, 1), (SuppressReason::Static, 2)]
        );
        assert_eq!(SuppressReason::MinArea.as_str(), "min_area");
    }
}
