//! Verifier cascade (`verify` in the config): a second model checks the primary model's
//! candidate boxes (by default night / IR `person` boxes) and a candidate the verifier does not
//! confirm is dropped. This module is pure: the config struct, which predictions are
//! candidates ([`plan`]), the crop geometry ([`crop_rect`], [`crop`]), the keep / reject rule
//! ([`verdict`]) and how outcomes change the predictions ([`apply`]). The worker
//! (`worker::verify`) and the benchmark variant `verify:<model>` both call these, so they make
//! the same decisions.
//!
//! Geometry, mode `crop`: a candidate box `[x1, y1, x2, y2]` (frame pixels, width `w`, height
//! `h`) is padded by `pad * w` left and right and `pad * h` above and below, rounded outward to
//! whole pixels and clamped to the frame ([`crop_rect`]). The verifier sees only that crop. Its
//! *reference box* is the candidate box in crop coordinates (`[x1 - cx, y1 - cy, x2 - cx,
//! y2 - cy]` for a crop at `(cx, cy)`), centred in the crop unless the frame edge clipped the
//! padding. Mode `frame`: the verifier sees the whole frame and the reference box is the
//! candidate box itself.
//!
//! Rule ([`verdict`]): the candidate is kept when the verifier reports a box whose label
//! matches (case-insensitive, after `aliases` maps the verifier's label to the primary's), with
//! confidence >= `min_confidence` and IoU >= `iou` with the reference box; otherwise it is
//! rejected. A candidate the verifier could not check (unavailable, busy, timed out, failed,
//! beyond `max_crops`) is kept as unverified (fail-open).

use crate::api::Prediction;
use crate::image::RgbImage;
use crate::image_stats::Scene;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Largest accepted `max_crops`.
pub const MAX_CROPS_LIMIT: usize = 16;
/// Largest accepted `pad` (share of the box size added on each side).
pub const MAX_PAD: f32 = 2.0;

/// What the verifier is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VerifyMode {
    /// One padded crop per candidate (cost grows with the number of candidates).
    #[default]
    Crop,
    /// The whole frame once; candidates are matched by IoU.
    Frame,
}

impl VerifyMode {
    pub fn as_str(self) -> &'static str {
        match self {
            VerifyMode::Crop => "crop",
            VerifyMode::Frame => "frame",
        }
    }
}

/// `verify` in the config (global default or per model; a model's value replaces the
/// global one, `"enabled": false` turns a global one off for that model).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VerifyConfig {
    pub enabled: bool,
    /// The configured model that verifies (never verifies itself).
    pub model: String,
    /// Labels whose boxes are verified (case-insensitive); others pass unchanged.
    pub labels: Vec<String>,
    /// Lowest verifier confidence that confirms a candidate.
    #[serde(serialize_with = "crate::config::thr::ser_clean")]
    pub min_confidence: f32,
    /// Lowest IoU between the verifier's box and the reference box.
    #[serde(serialize_with = "crate::config::thr::ser_clean")]
    pub iou: f32,
    /// Only verify on night / IR frames.
    pub night_only: bool,
    pub mode: VerifyMode,
    /// Crop padding per side, as a share of the box width / height (mode `crop`).
    #[serde(serialize_with = "crate::config::thr::ser_clean")]
    pub pad: f32,
    /// Most candidates verified per request (most confident first); the rest stay unverified.
    pub max_crops: usize,
    /// Verifier label -> primary label (`{"vehicle": "car"}` for MegaDetector).
    pub aliases: BTreeMap<String, String>,
}

impl Default for VerifyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: String::new(),
            labels: vec!["person".to_string()],
            min_confidence: 0.5,
            iou: 0.3,
            night_only: true,
            mode: VerifyMode::Crop,
            pad: 0.2,
            max_crops: 4,
            aliases: BTreeMap::new(),
        }
    }
}

impl VerifyConfig {
    /// The defaults with `model` as the verifier.
    pub fn with_model(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Self::default()
        }
    }

    /// Range checks (the model reference is checked by the config), errors naming `field`.
    pub fn validate(&self, field: &str) -> anyhow::Result<()> {
        use anyhow::bail;
        if !self.enabled {
            return Ok(());
        }
        if self.model.trim().is_empty() {
            bail!(
                "{field}.model: is empty; name the configured model that verifies, or set \
                 \"enabled\": false"
            );
        }
        if self.labels.is_empty() {
            bail!("{field}.labels: is empty; list the labels to verify, e.g. [\"person\"]");
        }
        if self.labels.iter().any(|l| l.trim().is_empty()) {
            bail!("{field}.labels: contains an empty label; remove it");
        }
        crate::config::check_threshold(&format!("{field}.min_confidence"), self.min_confidence)?;
        crate::config::check_threshold(&format!("{field}.iou"), self.iou)?;
        if !(self.pad.is_finite() && (0.0..=MAX_PAD).contains(&self.pad)) {
            bail!(
                "{field}.pad: must be between 0 and {MAX_PAD}, got {}; 0.2 adds a fifth of the \
                 box size on each side",
                self.pad
            );
        }
        crate::config::check_range(
            &format!("{field}.max_crops"),
            self.max_crops,
            1,
            MAX_CROPS_LIMIT,
        )?;
        if self
            .aliases
            .iter()
            .any(|(k, v)| k.trim().is_empty() || v.trim().is_empty())
        {
            bail!("{field}.aliases: has an empty label; use {{\"verifier label\": \"label\"}}");
        }
        Ok(())
    }

    /// One line for logs and the benchmark summary.
    pub fn describe(&self) -> String {
        let mut s = format!(
            "{} verifies {} ({}, conf >= {}, IoU >= {}{}",
            self.model,
            self.labels.join("/"),
            self.mode.as_str(),
            self.min_confidence,
            self.iou,
            if self.night_only { ", night / IR" } else { "" }
        );
        if self.mode == VerifyMode::Crop {
            s.push_str(&format!(", pad {}, up to {}", self.pad, self.max_crops));
        }
        s.push(')');
        s
    }

    fn is_candidate_label(&self, label: &str) -> bool {
        self.labels
            .iter()
            .any(|l| l.trim().eq_ignore_ascii_case(label))
    }
}

/// What happened to one candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The verifier confirmed it.
    Kept,
    /// The verifier answered and did not confirm it: dropped.
    Rejected,
    /// Not checked (verifier unavailable, busy, late, failed, or over `max_crops`): kept.
    Unverified,
}

impl Outcome {
    pub const ALL: [Outcome; 3] = [Outcome::Kept, Outcome::Rejected, Outcome::Unverified];

    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Kept => "kept",
            Outcome::Rejected => "rejected",
            Outcome::Unverified => "unverified",
        }
    }
}

/// Candidates per [`Outcome`] of one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Counts {
    pub kept: u32,
    pub rejected: u32,
    pub unverified: u32,
}

impl Counts {
    pub fn add(&mut self, o: Outcome) {
        let c = match o {
            Outcome::Kept => &mut self.kept,
            Outcome::Rejected => &mut self.rejected,
            Outcome::Unverified => &mut self.unverified,
        };
        *c = c.saturating_add(1);
    }

    pub fn get(&self, o: Outcome) -> u32 {
        match o {
            Outcome::Kept => self.kept,
            Outcome::Rejected => self.rejected,
            Outcome::Unverified => self.unverified,
        }
    }

    pub fn total(&self) -> u32 {
        self.kept + self.rejected + self.unverified
    }
}

/// A crop in frame pixels (`x + w <= frame width`, `y + h <= frame height`, `w, h >= 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CropRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// One candidate to check.
#[derive(Debug, Clone, PartialEq)]
pub struct Check {
    /// Index into the predictions passed to [`plan`].
    pub index: usize,
    pub label: String,
    /// The box the verifier's boxes are compared with, in the verifier's image coordinates
    /// (the crop for mode `crop`, the frame for mode `frame`).
    pub reference: [f32; 4],
    /// The crop to show the verifier (mode `crop`); None for mode `frame`.
    pub crop: Option<CropRect>,
}

/// What [`plan`] decided for one request.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Plan {
    /// Candidates to verify (most confident first, at most `max_crops`).
    pub checks: Vec<Check>,
    /// Candidates over `max_crops` (unverified).
    pub overflow: Vec<usize>,
}

impl Plan {
    pub fn is_empty(&self) -> bool {
        self.checks.is_empty() && self.overflow.is_empty()
    }
}

/// Frame pixel box of a prediction.
pub fn pred_box(p: &Prediction) -> [f32; 4] {
    [
        p.x_min as f32,
        p.y_min as f32,
        p.x_max as f32,
        p.y_max as f32,
    ]
}

/// True when `cfg` verifies anything on a frame of `scene`.
pub fn applies(cfg: &VerifyConfig, scene: Scene) -> bool {
    cfg.enabled && !(cfg.night_only && scene == Scene::Day)
}

/// The candidates of `preds` (predictions of a `width` x `height` frame of `scene`) under
/// `cfg`: labels in `cfg.labels`, not on a day frame when `night_only`; the `max_crops` most
/// confident are checked (ties: earlier first), the rest overflow. In mode `frame` every
/// candidate is checked (one verifier run covers them all).
pub fn plan(
    preds: &[Prediction],
    width: u32,
    height: u32,
    scene: Scene,
    cfg: &VerifyConfig,
) -> Plan {
    if !applies(cfg, scene) {
        return Plan::default();
    }
    let mut idx: Vec<usize> = preds
        .iter()
        .enumerate()
        .filter(|(_, p)| cfg.is_candidate_label(&p.label))
        .map(|(i, _)| i)
        .collect();
    // Stable: equal confidences keep their order.
    let conf = |i: usize| preds.get(i).map_or(f32::NEG_INFINITY, |p| p.confidence);
    idx.sort_by(|&a, &b| conf(b).total_cmp(&conf(a)));
    let limit = match cfg.mode {
        VerifyMode::Crop => cfg.max_crops.max(1),
        VerifyMode::Frame => usize::MAX,
    };
    let overflow = idx.split_off(idx.len().min(limit));
    let checks = idx
        .into_iter()
        .filter_map(|i| {
            let p = preds.get(i)?;
            let b = pred_box(p);
            let (reference, crop) = match cfg.mode {
                VerifyMode::Frame => (b, None),
                VerifyMode::Crop => {
                    let r = crop_rect(b, cfg.pad, width, height);
                    (reference_in_crop(b, r), Some(r))
                }
            };
            Some(Check {
                index: i,
                label: p.label.clone(),
                reference,
                crop,
            })
        })
        .collect();
    Plan { checks, overflow }
}

/// The crop of box `b` padded by `pad` (see the module docs) in a `width` x `height` frame.
pub fn crop_rect(b: [f32; 4], pad: f32, width: u32, height: u32) -> CropRect {
    let (fw, fh) = (width.max(1) as f32, height.max(1) as f32);
    let [x1, y1, x2, y2] = b;
    let (bw, bh) = ((x2 - x1).max(0.0), (y2 - y1).max(0.0));
    let pad = if pad.is_finite() { pad.max(0.0) } else { 0.0 };
    let lo = |v: f32, max: f32| v.floor().clamp(0.0, max - 1.0);
    let hi = |v: f32, max: f32| v.ceil().clamp(1.0, max);
    let cx0 = lo(x1 - pad * bw, fw);
    let cy0 = lo(y1 - pad * bh, fh);
    let cx1 = hi(x2 + pad * bw, fw).max(cx0 + 1.0);
    let cy1 = hi(y2 + pad * bh, fh).max(cy0 + 1.0);
    // Whole, in-range pixel values by construction (floor / ceil, clamped to the frame).
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the values are whole numbers in 0..=frame side (floor / ceil, clamped)"
    )]
    let px = |v: f32| v as u32;
    CropRect {
        x: px(cx0),
        y: px(cy0),
        w: px(cx1 - cx0),
        h: px(cy1 - cy0),
    }
}

/// Box `b` (frame pixels) in the coordinates of crop `r`.
pub fn reference_in_crop(b: [f32; 4], r: CropRect) -> [f32; 4] {
    let (x, y) = (r.x as f32, r.y as f32);
    [b[0] - x, b[1] - y, b[2] - x, b[3] - y]
}

/// The pixels of `r` (clamped to `img`).
pub fn crop(img: &RgbImage, r: CropRect) -> RgbImage {
    let x0 = r.x.min(img.width.saturating_sub(1));
    let y0 = r.y.min(img.height.saturating_sub(1));
    let w = r.w.min(img.width - x0).max(1);
    let h = r.h.min(img.height - y0).max(1);
    let stride = img.width as usize * 3;
    let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
    for row in y0..y0 + h {
        let start = row as usize * stride + x0 as usize * 3;
        if let Some(line) = img.rgb.get(start..start + w as usize * 3) {
            rgb.extend_from_slice(line);
        }
    }
    RgbImage {
        width: w,
        height: h,
        rgb,
    }
}

/// IoU of two `[x1, y1, x2, y2]` boxes (0 for empty ones).
pub fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = iw * ih;
    let area = |r: [f32; 4]| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
    let union = area(a) + area(b) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// True when the verifier's `found` label names the candidate's `label` (case-insensitive;
/// `aliases` maps a verifier label to a primary label, matched case-insensitively too).
pub fn label_matches(found: &str, label: &str, aliases: &BTreeMap<String, String>) -> bool {
    if found.eq_ignore_ascii_case(label) {
        return true;
    }
    aliases.iter().any(|(from, to)| {
        from.trim().eq_ignore_ascii_case(found) && to.trim().eq_ignore_ascii_case(label)
    })
}

/// The decision for one candidate of label `label` whose reference box is `reference`, given
/// what the verifier `found` on its image: [`Outcome::Kept`] or [`Outcome::Rejected`].
pub fn verdict(
    label: &str,
    reference: [f32; 4],
    found: &[Prediction],
    cfg: &VerifyConfig,
) -> Outcome {
    let confirmed = found.iter().any(|p| {
        p.confidence >= cfg.min_confidence
            && label_matches(&p.label, label, &cfg.aliases)
            && iou(pred_box(p), reference) >= cfg.iou
    });
    if confirmed {
        Outcome::Kept
    } else {
        Outcome::Rejected
    }
}

/// `preds` without the rejected candidates of `outcomes` (`(prediction index, outcome)`;
/// indices not listed are not candidates and stay), with the counts per outcome.
pub fn apply(preds: Vec<Prediction>, outcomes: &[(usize, Outcome)]) -> (Vec<Prediction>, Counts) {
    let mut counts = Counts::default();
    for (_, o) in outcomes {
        counts.add(*o);
    }
    let rejected = |i: usize| {
        outcomes
            .iter()
            .any(|&(j, o)| j == i && o == Outcome::Rejected)
    };
    let kept = preds
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !rejected(*i))
        .map(|(_, p)| p)
        .collect();
    (kept, counts)
}

/// The outcomes of a whole [`Plan`] from the verifier's answers: `found[k]` holds what the
/// verifier reported for `plan.checks[k]` (None: not checked), in mode `frame` the one frame
/// answer is used for every check. Overflow candidates are unverified.
pub fn outcomes(
    plan: &Plan,
    found: &[Option<Vec<Prediction>>],
    cfg: &VerifyConfig,
) -> Vec<(usize, Outcome)> {
    let mut out: Vec<(usize, Outcome)> = plan
        .checks
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let answer = match cfg.mode {
                VerifyMode::Crop => found.get(k),
                VerifyMode::Frame => found.first(),
            };
            let o = match answer {
                Some(Some(f)) => verdict(&c.label, c.reference, f, cfg),
                _ => Outcome::Unverified,
            };
            (c.index, o)
        })
        .collect();
    out.extend(plan.overflow.iter().map(|&i| (i, Outcome::Unverified)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(label: &str, conf: f32, b: [usize; 4]) -> Prediction {
        Prediction {
            x_min: b[0],
            y_min: b[1],
            x_max: b[2],
            y_max: b[3],
            confidence: conf,
            label: label.into(),
        }
    }

    #[test]
    fn crop_rect_pads_rounds_outward_and_clamps() {
        // 20x40 box, pad 0.2: 4 px left / right, 8 px above / below.
        let r = crop_rect([100.0, 100.0, 120.0, 140.0], 0.2, 640, 480);
        assert_eq!(
            r,
            CropRect {
                x: 96,
                y: 92,
                w: 28,
                h: 56
            }
        );
        assert_eq!(
            reference_in_crop([100.0, 100.0, 120.0, 140.0], r),
            [4.0, 8.0, 24.0, 48.0]
        );
        // Fractional padding rounds outward.
        let r = crop_rect([10.0, 10.0, 15.0, 15.0], 0.3, 100, 100);
        assert_eq!(
            r,
            CropRect {
                x: 8,
                y: 8,
                w: 9,
                h: 9
            }
        );
        // At the frame edge the padding is clipped: the box is no longer centred.
        let r = crop_rect([0.0, 0.0, 50.0, 100.0], 0.2, 60, 110);
        assert_eq!(
            r,
            CropRect {
                x: 0,
                y: 0,
                w: 60,
                h: 110
            }
        );
        assert_eq!(
            reference_in_crop([0.0, 0.0, 50.0, 100.0], r),
            [0.0, 0.0, 50.0, 100.0]
        );
        // Degenerate boxes still give a 1 px crop inside the frame.
        let r = crop_rect([99.0, 99.0, 99.0, 99.0], 0.2, 100, 100);
        assert_eq!(
            r,
            CropRect {
                x: 99,
                y: 99,
                w: 1,
                h: 1
            }
        );
        let r = crop_rect([5.0, 5.0, 9.0, 9.0], f32::NAN, 100, 100);
        assert_eq!(
            r,
            CropRect {
                x: 5,
                y: 5,
                w: 4,
                h: 4
            }
        );
    }

    #[test]
    fn crop_copies_the_pixels() {
        let img = RgbImage {
            width: 4,
            height: 3,
            rgb: (0..36).map(|v| v as u8).collect(),
        };
        let c = crop(
            &img,
            CropRect {
                x: 1,
                y: 1,
                w: 2,
                h: 2,
            },
        );
        assert_eq!((c.width, c.height), (2, 2));
        // Rows 1 and 2, pixels 1..3 (3 bytes each).
        assert_eq!(c.rgb, [15, 16, 17, 18, 19, 20, 27, 28, 29, 30, 31, 32]);
        // Out of range is clamped.
        let c = crop(
            &img,
            CropRect {
                x: 3,
                y: 2,
                w: 9,
                h: 9,
            },
        );
        assert_eq!((c.width, c.height, c.rgb.len()), (1, 1, 3));
    }

    #[test]
    fn iou_known_rects() {
        assert_eq!(iou([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]), 1.0);
        assert_eq!(iou([0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 20.0, 10.0]), 0.0);
        assert!((iou([0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]) - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(iou([0.0; 4], [0.0; 4]), 0.0);
        // The whole padded crop against its centred box: 1 / (1 + 2 pad)^2.
        let inner = [4.0, 8.0, 24.0, 48.0];
        let v = iou([0.0, 0.0, 28.0, 56.0], inner);
        assert!((v - 1.0 / 1.96).abs() < 1e-6, "{v}");
    }

    #[test]
    fn plan_selects_night_candidates_most_confident_first() {
        let cfg = VerifyConfig {
            max_crops: 2,
            labels: vec!["Person".into()],
            ..VerifyConfig::with_model("v")
        };
        let preds = vec![
            p("person", 0.6, [10, 10, 30, 50]),
            p("car", 0.9, [100, 100, 200, 150]),
            p("PERSON", 0.95, [50, 10, 70, 50]),
            p("person", 0.7, [80, 10, 90, 50]),
        ];
        let plan = plan(&preds, 640, 480, Scene::Ir, &cfg);
        assert_eq!(
            plan.checks.iter().map(|c| c.index).collect::<Vec<_>>(),
            [2, 3]
        );
        assert_eq!(plan.overflow, [0]);
        assert_eq!(
            plan.checks[0].crop,
            Some(crop_rect([50.0, 10.0, 70.0, 50.0], 0.2, 640, 480))
        );
        // Day frame with night_only: nothing; without: the same candidates.
        assert!(super::plan(&preds, 640, 480, Scene::Day, &cfg).is_empty());
        let all_day = VerifyConfig {
            night_only: false,
            ..cfg.clone()
        };
        assert_eq!(
            super::plan(&preds, 640, 480, Scene::Day, &all_day)
                .checks
                .len(),
            2
        );
        // Frame mode: every candidate, reference = the box itself, no crop.
        let frame = VerifyConfig {
            mode: VerifyMode::Frame,
            ..cfg.clone()
        };
        let fp = super::plan(&preds, 640, 480, Scene::Night, &frame);
        assert_eq!(fp.checks.len(), 3);
        assert!(fp.overflow.is_empty());
        assert_eq!(fp.checks[0].reference, [50.0, 10.0, 70.0, 50.0]);
        assert_eq!(fp.checks[0].crop, None);
        // Disabled: nothing.
        let off = VerifyConfig {
            enabled: false,
            ..cfg
        };
        assert!(super::plan(&preds, 640, 480, Scene::Ir, &off).is_empty());
    }

    #[test]
    fn verdict_needs_label_confidence_and_overlap() {
        let mut cfg = VerifyConfig::with_model("v");
        let reference = [4.0, 8.0, 24.0, 48.0];
        let hit = p("person", 0.8, [3, 7, 25, 49]);
        assert_eq!(
            verdict("person", reference, std::slice::from_ref(&hit), &cfg),
            Outcome::Kept
        );
        assert_eq!(
            verdict(
                "Person",
                reference,
                &[p("PERSON", 0.8, [3, 7, 25, 49])],
                &cfg
            ),
            Outcome::Kept
        );
        // Nothing found, too unsure, other label, elsewhere in the crop.
        assert_eq!(verdict("person", reference, &[], &cfg), Outcome::Rejected);
        assert_eq!(
            verdict(
                "person",
                reference,
                &[p("person", 0.49, [3, 7, 25, 49])],
                &cfg
            ),
            Outcome::Rejected
        );
        assert_eq!(
            verdict("person", reference, &[p("dog", 0.9, [3, 7, 25, 49])], &cfg),
            Outcome::Rejected
        );
        assert_eq!(
            verdict(
                "person",
                reference,
                &[p("person", 0.9, [20, 40, 28, 56])],
                &cfg
            ),
            Outcome::Rejected
        );
        // The whole crop still overlaps the centred box enough at the default IoU 0.3.
        assert_eq!(
            verdict(
                "person",
                reference,
                &[p("person", 0.9, [0, 0, 28, 56])],
                &cfg
            ),
            Outcome::Kept
        );
        cfg.iou = 0.6;
        assert_eq!(
            verdict(
                "person",
                reference,
                &[p("person", 0.9, [0, 0, 28, 56])],
                &cfg
            ),
            Outcome::Rejected
        );
        // Aliases map the verifier's label to the primary's.
        cfg.aliases = BTreeMap::from([("Vehicle".to_string(), "car".to_string())]);
        assert_eq!(
            verdict("car", reference, &[p("vehicle", 0.9, [4, 8, 24, 48])], &cfg),
            Outcome::Kept
        );
        assert_eq!(
            verdict(
                "truck",
                reference,
                &[p("vehicle", 0.9, [4, 8, 24, 48])],
                &cfg
            ),
            Outcome::Rejected
        );
        assert!(label_matches("car", "CAR", &BTreeMap::new()));
    }

    #[test]
    fn outcomes_and_apply_drop_only_rejected_candidates() {
        let cfg = VerifyConfig {
            max_crops: 2,
            ..VerifyConfig::with_model("v")
        };
        let preds = vec![
            p("person", 0.9, [10, 10, 30, 50]),
            p("car", 0.9, [100, 100, 200, 150]),
            p("person", 0.8, [300, 10, 320, 50]),
            p("person", 0.6, [400, 10, 420, 50]),
        ];
        let plan = plan(&preds, 640, 480, Scene::Ir, &cfg);
        let r0 = plan.checks[0].reference;
        let confirm = |r: [f32; 4]| {
            vec![p(
                "person",
                0.9,
                [r[0] as usize, r[1] as usize, r[2] as usize, r[3] as usize],
            )]
        };
        // First confirmed, second answered with nothing.
        let o = outcomes(&plan, &[Some(confirm(r0)), Some(Vec::new())], &cfg);
        assert_eq!(
            o,
            [
                (0, Outcome::Kept),
                (2, Outcome::Rejected),
                (3, Outcome::Unverified)
            ]
        );
        let (kept, counts) = apply(preds.clone(), &o);
        assert_eq!(
            kept.iter().map(|p| p.x_min).collect::<Vec<_>>(),
            [10, 100, 400],
            "the car is not a candidate and passes unchanged"
        );
        assert_eq!(
            counts,
            Counts {
                kept: 1,
                rejected: 1,
                unverified: 1
            }
        );
        assert_eq!(counts.total(), 3);
        // No answers (verifier unavailable): every candidate unverified, nothing dropped.
        let o = outcomes(&plan, &[None, None], &cfg);
        let (kept, counts) = apply(preds.clone(), &o);
        assert_eq!(kept, preds);
        assert_eq!(counts.unverified, 3);
        // Frame mode: one answer for all.
        let frame = VerifyConfig {
            mode: VerifyMode::Frame,
            ..cfg
        };
        let fplan = super::plan(&preds, 640, 480, Scene::Ir, &frame);
        let o = outcomes(
            &fplan,
            &[Some(vec![p("person", 0.7, [11, 11, 31, 51])])],
            &frame,
        );
        let (kept, counts) = apply(preds, &o);
        assert_eq!(kept.len(), 2);
        assert_eq!(
            counts,
            Counts {
                kept: 1,
                rejected: 2,
                unverified: 0
            }
        );
    }

    #[test]
    fn config_defaults_validation_and_serde() {
        let c: VerifyConfig = serde_json::from_str(r#"{"model": "yolo26s"}"#).unwrap();
        assert_eq!(c, VerifyConfig::with_model("yolo26s"));
        assert!(c.enabled && c.night_only);
        assert_eq!(c.mode, VerifyMode::Crop);
        c.validate("verify").unwrap();
        let text = serde_json::to_string(&c).unwrap();
        assert!(text.contains(r#""min_confidence":0.5"#), "{text}");
        assert!(text.contains(r#""pad":0.2"#), "{text}");
        assert!(text.contains(r#""mode":"crop""#), "{text}");
        let back: VerifyConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back, c);
        let f: VerifyConfig = serde_json::from_str(r#"{"model":"x","mode":"frame"}"#).unwrap();
        assert_eq!(f.mode, VerifyMode::Frame);
        assert!(serde_json::from_str::<VerifyConfig>(r#"{"mode":"tile"}"#).is_err());
        let bad = |c: VerifyConfig, want: &str| {
            let e = c.validate("verify").unwrap_err().to_string();
            assert!(e.contains(want), "{e}");
        };
        bad(VerifyConfig::default(), "verify.model: is empty");
        bad(
            VerifyConfig {
                labels: vec![],
                ..c.clone()
            },
            "verify.labels",
        );
        bad(
            VerifyConfig {
                labels: vec![" ".into()],
                ..c.clone()
            },
            "empty label",
        );
        bad(
            VerifyConfig {
                min_confidence: 1.5,
                ..c.clone()
            },
            "verify.min_confidence",
        );
        bad(
            VerifyConfig {
                iou: -0.1,
                ..c.clone()
            },
            "verify.iou",
        );
        bad(
            VerifyConfig {
                pad: 3.0,
                ..c.clone()
            },
            "verify.pad",
        );
        bad(
            VerifyConfig {
                max_crops: 0,
                ..c.clone()
            },
            "verify.max_crops",
        );
        bad(
            VerifyConfig {
                max_crops: 17,
                ..c.clone()
            },
            "verify.max_crops",
        );
        bad(
            VerifyConfig {
                aliases: BTreeMap::from([("".to_string(), "car".to_string())]),
                ..c.clone()
            },
            "verify.aliases",
        );
        // Disabled: nothing else is checked.
        VerifyConfig {
            enabled: false,
            ..VerifyConfig::default()
        }
        .validate("verify")
        .unwrap();
        assert!(
            c.describe().contains("yolo26s verifies person (crop"),
            "{}",
            c.describe()
        );
    }
}
