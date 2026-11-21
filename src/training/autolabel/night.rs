//! Night and IR frames in a labelers run (docs/TRAINING.md "Night and IR frames"): the two extras
//! `scripts/autolabel.py` applies to frames whose scene is `night` or `ir`, and what the frames'
//! marker says about them.
//!
//! - **The enhanced view** ([`AutolabelRequest::night_enhance`], default [`DEFAULT_NIGHT_ENHANCE`]):
//!   every labeler also runs on a copy enhanced by a [`Preprocess`] mode; the script's
//!   `enhance_array` is the byte-for-byte twin of [`crate::model::enhance::apply`] (the parity test
//!   below). Its boxes join the original's per labeler (one object of one kind found on both is one
//!   box, the higher score kept), the geometry is the same, so boxes map 1:1; the VLM and SigLIP 2
//!   crop the original, the stored frame. The copy is added, never put in the original's place:
//!   `docs/NIGHT_IR.md` measured CLAHE and gamma on every frame lowering the service models' person
//!   recall by 3 to 13 points on ExDark, so the union keeps every box the original gives and only
//!   adds those the copy gives. Each box records its views; [`super::triage`] never accepts a box
//!   the copy alone found or whose score came from it, nor hides one for its score, and says so in
//!   its reasons (`ENHANCED_ONLY_WHY`, `ENHANCED_RAISED_WHY`), so the verdicts on them can be
//!   counted. A request stored without the option reads as [`DEFAULT_NIGHT_ENHANCE`], as a new one.
//! - **MegaDetector v6** ([`AutolabelRequest::mdv6`], opt-in): `MDV6-yolov9-c.pt` ([`MDV6_MODEL`],
//!   the catalog's export pin, AGPL-3.0, 52 MB) through the training environment's Ultralytics,
//!   at [`MDV6_THRESHOLD`]. Its `animal` boxes are named by SigLIP 2 over the run's animal classes:
//!   the top species from 0.5 names the box; below it the box keeps that guess with `animal_guess`
//!   ("animal?" on the Review tab, with the scores). A box on one of the other labelers' boxes of the
//!   same object is a vote (`labelers["mdv6"]`, [`MDV6_KEY`]), so agreement counts it; any other is
//!   a box of its own. [`super::triage`] never accepts a box only MegaDetector found (its constants
//!   were tuned on MM-Grounding-DINO + RF-DETR, and a bulk accept reaches the exports), never hides
//!   one for its score, and never accepts one at a parked spot.
//!
//! The frames' marker names both on night and IR frames only: a run expects [`frame_labeler`] and
//! writes [`done_labeler`], what the script reports it did on the frame (an enhancement that failed
//! there leaves `night-<mode>` out, so the next run labels the frame again). A marker with more
//! extras than a run asks for covers it ([`marker_covers`]): after a MegaDetector run, a run without
//! it skips the frame.

use super::AutolabelRequest;
use crate::model::enhance::Preprocess;
use crate::resources::catalog::{self as res_catalog, Resource};

/// The enhancement of a new request: CLAHE (clip 2, 8 x 8 tiles), OpenCV's usual setting. It
/// brightens shadows locally without the global gamma's washed-out highlights (IR spotlights).
pub const DEFAULT_NIGHT_ENHANCE: Preprocess = Preprocess::CLAHE;

/// The catalog model whose weights MegaDetector runs from (the export pin's Zenodo file).
pub const MDV6_MODEL: &str = "mdv6-yolov9-c";
/// MegaDetector's key in a box's `labelers` (`scripts/autolabel.py` `MD_KEY`).
pub const MDV6_KEY: &str = "mdv6";
/// MegaDetector's score threshold: the value MegaDetector's own tools suggest for reviewing
/// camera-trap images (0.2); every box is reviewed, so a lower value only adds false ones.
pub const MDV6_THRESHOLD: f32 = 0.2;

/// Whether a frame of `scene` gets the night extras.
pub fn is_night(scene: &str) -> bool {
    matches!(scene, "night" | "ir")
}

/// MegaDetector's weights pin.
pub fn mdv6_weights() -> Option<&'static Resource> {
    res_catalog::export_model(MDV6_MODEL)
}

/// The enhancement a labelers run of `r` applies (None: off, or a run with a model, which runs in
/// this process without it).
pub fn enhancement(r: &AutolabelRequest) -> Option<Preprocess> {
    r.night_enhance
        .filter(|p| !p.is_none() && r.with_model.is_none())
}

/// The marker of a frame of `scene` in a labelers run of `r` whose labelers are `base`
/// ([`super::job::AutolabelJob`]'s list): on night and IR frames `+ mdv6-yolov9-c` and
/// `+ night-<mode>` when they apply, else `base`.
pub fn frame_labeler(base: &str, r: &AutolabelRequest, scene: &str) -> String {
    if !is_night(scene) || r.with_model.is_some() {
        return base.to_string();
    }
    let mut s = base.to_string();
    if r.mdv6 {
        s.push_str(" + ");
        s.push_str(MDV6_MODEL);
    }
    if let Some(p) = enhancement(r) {
        s.push_str(&format!(" + night-{p}"));
    }
    s
}

/// The marker a labelers run of `r` writes on a frame of `scene` whose labelers are `base`, from what
/// the script reports for it: `enhanced` (None: a line of a script older than the option, which
/// falls back to the request, [`frame_labeler`]) and whether MegaDetector ran (`mdv6_ran`).
pub fn done_labeler(
    base: &str,
    r: &AutolabelRequest,
    scene: &str,
    enhanced: Option<bool>,
    mdv6_ran: bool,
) -> String {
    let Some(enhanced) = enhanced else {
        return frame_labeler(base, r, scene);
    };
    if !is_night(scene) || r.with_model.is_some() {
        return base.to_string();
    }
    let mut s = base.to_string();
    if mdv6_ran {
        s.push_str(" + ");
        s.push_str(MDV6_MODEL);
    }
    if enhanced && let Some(p) = enhancement(r) {
        s.push_str(&format!(" + night-{p}"));
    }
    s
}

/// A marker part that is a night extra, not a labeler.
fn is_extra(part: &str) -> bool {
    part == MDV6_MODEL || part.starts_with("night-")
}

/// Whether a frame whose marker says `found` needs no run that would write `wanted`: the same
/// labelers in the same order, and every night extra of `wanted` in `found` (more is fine).
pub fn marker_covers(found: &str, wanted: &str) -> bool {
    let split = |m: &str| -> (Vec<String>, Vec<String>) {
        m.split(" + ")
            .map(str::to_string)
            .partition(|p| !is_extra(p))
    };
    let (found_base, found_extra) = split(found);
    let (wanted_base, wanted_extra) = split(wanted);
    found_base == wanted_base && wanted_extra.iter().all(|e| found_extra.contains(e))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::image::RgbImage;
    use crate::model::enhance;

    #[test]
    fn the_marker_names_the_night_extras_on_night_frames_only() {
        let mut r = AutolabelRequest::new(super::super::FrameFilter::default());
        let base = "mm-gdino-tiny + rfdetr-large";
        assert_eq!(
            r.night_enhance,
            Some(DEFAULT_NIGHT_ENHANCE),
            "on by default"
        );
        assert!(!r.mdv6, "MegaDetector is opt-in");
        assert_eq!(frame_labeler(base, &r, "day"), base);
        assert_eq!(
            frame_labeler(base, &r, "ir"),
            "mm-gdino-tiny + rfdetr-large + night-clahe"
        );
        r.mdv6 = true;
        r.night_enhance = Some(Preprocess::Gamma(0.6));
        assert_eq!(
            frame_labeler(base, &r, "night"),
            "mm-gdino-tiny + rfdetr-large + mdv6-yolov9-c + night-gamma:0.6"
        );
        r.night_enhance = Some(Preprocess::None);
        assert_eq!(
            frame_labeler(base, &r, "night"),
            "mm-gdino-tiny + rfdetr-large + mdv6-yolov9-c"
        );
        r.night_enhance = None;
        r.mdv6 = false;
        assert_eq!(frame_labeler(base, &r, "night"), base);
        // A run with a model has its own marker and no extras.
        let mut m = AutolabelRequest::new(super::super::FrameFilter::default());
        m.with_model = Some("yolo".into());
        assert_eq!(enhancement(&m), None);
        assert_eq!(
            frame_labeler("model:yolo@auto", &m, "ir"),
            "model:yolo@auto"
        );
    }

    #[test]
    fn the_written_marker_follows_what_the_script_did() {
        let mut r = AutolabelRequest::new(super::super::FrameFilter::default());
        r.mdv6 = true;
        let base = "mm-gdino-tiny + rfdetr-large";
        let all = "mm-gdino-tiny + rfdetr-large + mdv6-yolov9-c + night-clahe";
        assert_eq!(done_labeler(base, &r, "ir", Some(true), true), all);
        // The enhancement failed on this frame, MegaDetector was not loaded: neither is named.
        assert_eq!(
            done_labeler(base, &r, "ir", Some(false), true),
            "mm-gdino-tiny + rfdetr-large + mdv6-yolov9-c"
        );
        assert_eq!(done_labeler(base, &r, "night", Some(false), false), base);
        // A day frame never names them; an older script's line falls back to the request.
        assert_eq!(done_labeler(base, &r, "day", Some(true), true), base);
        assert_eq!(done_labeler(base, &r, "ir", None, false), all);
    }

    #[test]
    fn a_marker_with_more_extras_covers_a_run_with_fewer() {
        let base = "mm-gdino-tiny + rfdetr-large";
        let enh = "mm-gdino-tiny + rfdetr-large + night-clahe";
        let all = "mm-gdino-tiny + rfdetr-large + mdv6-yolov9-c + night-clahe";
        assert!(marker_covers(all, enh) && marker_covers(all, base) && marker_covers(enh, enh));
        assert!(!marker_covers(enh, all), "MegaDetector never ran there");
        assert!(!marker_covers(base, enh), "the copy was never looked at");
        assert!(!marker_covers(
            "mm-gdino-tiny + rfdetr-large + night-gamma:0.6",
            enh
        ));
        assert!(!marker_covers("mm-gdino-base + rfdetr-large", base));
        assert!(!marker_covers(
            base,
            "mm-gdino-tiny + rfdetr-large + qwen3-vl-2b"
        ));
        assert!(marker_covers("model:yolo@auto", "model:yolo@auto"));
        assert!(!marker_covers("model:yolo@auto", "model:yolo@openvino:cpu"));
    }

    #[test]
    fn megadetector_runs_from_the_pinned_export_weights() {
        let r = mdv6_weights().unwrap();
        assert_eq!(r.model_name(), Some(MDV6_MODEL));
        assert_eq!(r.parts.len(), 1);
        assert!(r.parts[0].file_name.ends_with(".pt"));
        assert!(!r.is_large(), "{} bytes", r.size());
        assert_eq!(r.license, "AGPL-3.0");
    }

    /// Deterministic pixels: a small LCG, scaled into `lo..lo + span`.
    fn image(w: u32, h: u32, seed: u32, lo: u8, span: u8, mono: bool) -> RgbImage {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            lo.saturating_add(u8::try_from((s >> 24) % u32::from(span.max(1))).unwrap())
        };
        let mut rgb = Vec::new();
        for _ in 0..w * h {
            if mono {
                let v = next();
                rgb.extend([v, v, v]);
            } else {
                rgb.extend([next(), next(), next()]);
            }
        }
        RgbImage {
            width: w,
            height: h,
            rgb,
        }
    }

    /// `scripts/autolabel.py`'s `enhance_array` gives the same bytes as [`enhance::apply`] for every
    /// mode, on colour and monochrome frames, dark and low-contrast ones, black pixels, frames
    /// smaller than the tile grid and odd sizes. Skipped without a Python with numpy unless
    /// `BOP_REQUIRE_PYTHON` is set.
    #[test]
    fn the_script_enhances_exactly_like_the_service() {
        use crate::training::progress::test_python;
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD;
        let mut dark = image(41, 29, 7, 0, 40, false);
        // Black pixels in a colour frame take the new gray.
        for px in dark.rgb.chunks_mut(3).step_by(5) {
            px.copy_from_slice(&[0, 0, 0]);
        }
        let images = [
            dark,
            image(64, 48, 3, 90, 30, false),
            image(37, 23, 11, 10, 60, true),
            image(5, 3, 5, 0, 255, false),
            image(1, 1, 9, 20, 10, false),
            image(200, 120, 13, 0, 255, false),
            image(97, 61, 17, 30, 20, true),
        ];
        let modes = [
            "clahe",
            "clahe:3,4",
            "clahe:1.5,2",
            "clahe:0.5,64",
            "grayscale",
            "gamma:0.6",
            "gamma:2.2",
            "none",
        ];
        let mut cases = Vec::new();
        let mut want = Vec::new();
        for img in &images {
            for m in modes {
                let mode: Preprocess = m.parse().unwrap();
                let mut out = img.clone();
                enhance::apply(&mut out, mode);
                cases.push(serde_json::json!({
                    "mode": mode.to_string(), "width": img.width, "height": img.height,
                    "rgb": b64.encode(&img.rgb),
                }));
                want.push((m, img.width, img.height, out.rgb));
            }
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let script = root.join("scripts/autolabel.py");
        let Some(py) = test_python::python3_with_numpy("autolabel.py enhancement twin") else {
            return;
        };
        let input = serde_json::json!({ "cases": cases }).to_string();
        let out = test_python::run(
            py,
            &["-I", script.to_str().unwrap(), "--enhance-parity"],
            &input,
        );
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        let got = v["out"].as_array().unwrap();
        assert_eq!(got.len(), want.len());
        for (g, (m, w, h, bytes)) in got.iter().zip(&want) {
            let g = b64.decode(g.as_str().unwrap()).unwrap();
            let diff = g.iter().zip(bytes).filter(|(a, b)| a != b).count();
            assert!(
                g.len() == bytes.len() && diff == 0,
                "{m} on {w}x{h}: {diff} of {} bytes differ",
                bytes.len()
            );
        }
    }
}
