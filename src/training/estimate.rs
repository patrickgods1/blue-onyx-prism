//! Rough VRAM and time estimates per family and size ([`estimate`]), and the largest batch that
//! fits a GPU ([`suggest_batch`]). **Estimates, not measurements**: the constants below are round
//! numbers from public reports (Ultralytics and Roboflow issue threads, transformers fine-tuning
//! notes) and the 1-epoch smoke runs on an Apple M1 (docs/TRAINING.md); they are good to a factor
//! of about 1.5 and only meant to hide choices that cannot fit and to set expectations.
//!
//! Model: `vram = base + per_image x batch x (side / default)^2` MB, and
//! `seconds per epoch = images / (images per second x gpu factor) x (imgsz / default)^2`, where
//! images per second are a GTX 1080 class (Pascal) GPU's at the default size and the GPU factor
//! scales them by generation ([`gpu_factor`]). On Apple MPS the per-second numbers are the M1's.
//! `side` is the input size, except for RF-DETR trained multi-scale: rfdetr resizes every sample
//! of a batch to its largest scale ([`rfdetr_max_side`], 544 px for a 384 px nano), so memory
//! follows that side (time follows the input size: the per-second numbers were measured with
//! multi-scale on).

use crate::backend::detect::ComputeCapability;
use crate::resources::catalog::TrainFamily;
use serde::Serialize;

/// Cost constants of one family and size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cost {
    /// Fixed VRAM (weights, optimizer state, CUDA context), MB.
    pub base_mb: u32,
    /// VRAM per image of a batch at the default input size, MB.
    pub per_image_mb: u32,
    /// Training images per second at the default size on a GTX 1080 class GPU.
    pub pascal_img_s: f32,
    /// The same on an Apple M1 (MPS); 0 = not trainable there.
    pub m1_img_s: f32,
}

const fn c(base_mb: u32, per_image_mb: u32, pascal_img_s: f32, m1_img_s: f32) -> Cost {
    Cost {
        base_mb,
        per_image_mb,
        pascal_img_s,
        m1_img_s,
    }
}

/// The cost table. M1 numbers of yolo26n, rfdetr-nano, rfdetr-base and rt-detrv2-s were measured
/// (1 epoch, 31-38 images, docs/TRAINING.md; RF-DETR trains multi-scale above its input size), the
/// other sizes scaled by their compute.
pub fn cost(family: TrainFamily, size: &str) -> Option<Cost> {
    use TrainFamily::*;
    Some(match (family, size) {
        (Yolo26 | Yolo11 | Yolov8, "n") => c(700, 150, 90.0, 5.0),
        (Yolo26 | Yolo11 | Yolov8, "s") => c(900, 260, 60.0, 2.5),
        (Yolo26 | Yolo11 | Yolov8, "m") => c(1200, 450, 30.0, 1.2),
        (Yolo26 | Yolo11 | Yolov8, "l") => c(1400, 560, 22.0, 0.9),
        (Yolo26 | Yolo11 | Yolov8, "x") => c(1800, 800, 13.0, 0.5),
        // Calibrated on an M1 16 GB run (batch 16 at 384 px, multi-scale up to 544 px: over
        // 20.4 GB when it ran out of memory, 1500 + 16 x 650 x (544/384)^2 = 22.4 GB).
        (Rfdetr, "nano") => c(1500, 650, 25.0, 1.0),
        (Rfdetr, "small") => c(1800, 650, 18.0, 0.7),
        (Rfdetr, "medium") => c(2000, 800, 14.0, 0.55),
        (Rfdetr, "base") => c(2000, 850, 12.0, 0.05),
        (Rfdetr, "large") => c(2500, 1100, 8.0, 0.05),
        (Rtdetrv2, "s") => c(1200, 500, 30.0, 1.1),
        (Rtdetrv2, "m") => c(1500, 650, 22.0, 0.8),
        (Rtdetrv2, "l") => c(1800, 850, 16.0, 0.6),
        (Rtdetrv2, "x") => c(2500, 1200, 9.0, 0.35),
        (Dfine, "s") => c(1100, 450, 32.0, 0.0),
        (Dfine, "m") => c(1400, 600, 22.0, 0.0),
        (Dfine, "l") => c(1800, 800, 15.0, 0.0),
        (Dfine, "x") => c(2400, 1100, 9.0, 0.0),
        _ => return None,
    })
}

/// Speed of a CUDA GPU relative to a GTX 1080 (Pascal), by compute capability: Volta 2.0, Turing
/// 1.6, A100 4.0, Ampere consumer 2.2, Ada 3.2, Hopper 6.0, Blackwell 4.5.
pub fn gpu_factor(cc: ComputeCapability) -> f32 {
    match (cc.major, cc.minor) {
        (..=6, _) => 1.0,
        (7, 0..=4) => 2.0,
        (7, _) => 1.6,
        (8, 0) => 4.0,
        (8, 9..) => 3.2,
        (8, _) => 2.2,
        (9, _) => 6.0,
        _ => 4.5,
    }
}

/// Where the estimate is for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Target {
    Cuda(ComputeCapability),
    Mps,
}

/// One estimate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Estimate {
    /// GPU memory the run needs, MB.
    pub vram_mb: u64,
    /// Seconds per epoch over `images` training images.
    pub epoch_s: f64,
    /// The whole run (`epochs` epochs), seconds.
    pub total_s: f64,
}

/// RF-DETR's patch size and window count per size (rfdetr's `RFDETR*Config`: base uses the
/// DINOv2 14 px patches with 4 windows, every other size 16 px with 2).
pub fn rfdetr_patch_windows(size: &str) -> (u32, u32) {
    if size == "base" { (14, 4) } else { (16, 2) }
}

/// The largest side rfdetr trains an `imgsz` RF-DETR at with its default multi-scale
/// (`expanded_scales`): `(imgsz / (patch x windows) + 5) x patch x windows`
/// (`rfdetr.datasets.coco.compute_multi_scale_scales`; nano 384 -> 544).
pub fn rfdetr_max_side(size: &str, imgsz: u32) -> u32 {
    let (patch, windows) = rfdetr_patch_windows(size);
    let block = patch * windows;
    (imgsz / block + 5) * block
}

/// Estimate a run (see the module docs), RF-DETR with its default multi-scale training. None
/// when the size is unknown or the family does not train on `target`.
pub fn estimate(
    family: TrainFamily,
    size: &str,
    imgsz: u32,
    batch: u32,
    images: usize,
    epochs: u32,
    target: Target,
) -> Option<Estimate> {
    estimate_with(family, size, imgsz, batch, images, epochs, target, true)
}

/// [`estimate`] with RF-DETR's multi-scale training on or off (ignored by the other families).
#[allow(clippy::too_many_arguments)]
pub fn estimate_with(
    family: TrainFamily,
    size: &str,
    imgsz: u32,
    batch: u32,
    images: usize,
    epochs: u32,
    target: Target,
    multi_scale: bool,
) -> Option<Estimate> {
    let cost = cost(family, size)?;
    let ratio2 = |side: u32| {
        let r = f64::from(side) / f64::from(family.default_imgsz(size));
        r * r
    };
    let scale = ratio2(imgsz);
    let mem_side = if family == TrainFamily::Rfdetr && multi_scale {
        rfdetr_max_side(size, imgsz)
    } else {
        imgsz
    };
    let vram = f64::from(cost.base_mb)
        + f64::from(cost.per_image_mb) * f64::from(batch) * ratio2(mem_side);
    let speed = match target {
        Target::Cuda(cc) => f64::from(cost.pascal_img_s) * f64::from(gpu_factor(cc)),
        Target::Mps => f64::from(cost.m1_img_s),
    };
    if speed <= 0.0 {
        return None;
    }
    let epoch_s = images as f64 / speed * scale;
    Some(Estimate {
        vram_mb: to_u64(vram),
        epoch_s,
        total_s: epoch_s * f64::from(epochs),
    })
}

/// The largest batch of 1, 2, 4, ..., 64 whose estimate fits `vram_mb` (90% of it, for the
/// allocator's slack); None when even batch 1 does not. RF-DETR with multi-scale on.
pub fn suggest_batch(family: TrainFamily, size: &str, imgsz: u32, vram_mb: u64) -> Option<u32> {
    suggest_batch_with(family, size, imgsz, vram_mb, true)
}

/// [`suggest_batch`] with RF-DETR's multi-scale training on or off.
pub fn suggest_batch_with(
    family: TrainFamily,
    size: &str,
    imgsz: u32,
    vram_mb: u64,
    multi_scale: bool,
) -> Option<u32> {
    let usable = vram_mb.saturating_mul(9) / 10;
    [64u32, 32, 16, 8, 4, 2, 1].into_iter().find(|b| {
        estimate_with(
            family,
            size,
            imgsz,
            *b,
            1,
            1,
            Target::Cuda(ComputeCapability::new(8, 6)),
            multi_scale,
        )
        .is_some_and(|e| e.vram_mb <= usable)
    })
}

/// The largest batch of 1, 2, 4, ..., 64 that fits `vram_mb` (90% of it) by a measured peak:
/// an earlier run of the same model, input size and multi-scale setting on this GPU peaked at
/// `peak_mb` with `batch` images. Linear in the batch over the fixed part `base_mb` (at most half
/// the peak, so a small run still leaves a per-image cost). None when even batch 1 does not fit.
pub fn batch_from_peak(base_mb: u32, batch: u32, peak_mb: f64, vram_mb: u64) -> Option<u32> {
    if batch == 0 || !peak_mb.is_finite() || peak_mb <= 0.0 {
        return None;
    }
    let base = f64::from(base_mb).min(peak_mb * 0.5);
    let per_image = (peak_mb - base) / f64::from(batch);
    let usable = vram_mb.saturating_mul(9) / 10;
    [64u32, 32, 16, 8, 4, 2, 1]
        .into_iter()
        .find(|b| to_u64(base + per_image * f64::from(*b)) <= usable)
}

/// Inference speed to expect from a size on the service's devices (shown next to the choice).
pub fn latency_hint(family: TrainFamily, size: &str) -> &'static str {
    match (family, size) {
        (_, "n" | "nano" | "s" | "small") => {
            "light: runs on a CPU or an Intel iGPU (UHD 630) at camera rates"
        }
        (_, "m" | "medium" | "base") => {
            "medium: about 2-3x the small model's time; a discrete GPU is best"
        }
        _ => "heavy: needs a discrete GPU for camera streams",
    }
}

/// `f.round()` as `u64`, clamped at 0 and `u64::MAX`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "clamped to the u64 range before the cast"
)]
pub(crate) fn to_u64(f: f64) -> u64 {
    if f.is_nan() || f <= 0.0 {
        0
    } else if f >= u64::MAX as f64 {
        u64::MAX
    } else {
        f.round() as u64
    }
}

/// "45 s", "12 min", "3.5 h".
pub fn format_duration(s: f64) -> String {
    if s < 90.0 {
        format!("{} s", to_u64(s))
    } else if s < 5400.0 {
        format!("{} min", to_u64(s / 60.0))
    } else {
        format!("{:.1} h", s / 3600.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_family_and_size_has_a_cost() {
        for f in TrainFamily::ALL {
            for s in f.sizes() {
                let c = cost(f, s).unwrap_or_else(|| panic!("{f} {s}"));
                assert!(c.base_mb > 0 && c.per_image_mb > 0 && c.pascal_img_s > 0.0);
                assert_eq!(c.m1_img_s > 0.0, f.mps_ok(), "{f} {s}");
            }
        }
        assert!(cost(TrainFamily::Yolo26, "nano").is_none());
    }

    #[test]
    fn larger_sizes_cost_more() {
        for f in TrainFamily::ALL {
            let sizes = f.sizes();
            for w in sizes.windows(2) {
                let (a, b) = (cost(f, w[0]).unwrap(), cost(f, w[1]).unwrap());
                assert!(b.per_image_mb >= a.per_image_mb, "{f} {w:?}");
                assert!(b.pascal_img_s <= a.pascal_img_s, "{f} {w:?}");
            }
        }
    }

    #[test]
    fn estimate_scales_with_batch_size_and_resolution() {
        let cc = Target::Cuda(ComputeCapability::new(6, 1));
        let e = estimate(TrainFamily::Yolo26, "n", 640, 16, 900, 10, cc).unwrap();
        assert_eq!(e.vram_mb, 700 + 150 * 16);
        assert!((e.epoch_s - 10.0).abs() < 1e-9, "{e:?}");
        assert!((e.total_s - 100.0).abs() < 1e-9);
        // Twice the side: four times the per-image memory and time.
        let big = estimate(TrainFamily::Yolo26, "n", 1280, 16, 900, 10, cc).unwrap();
        assert_eq!(big.vram_mb, 700 + 150 * 16 * 4);
        assert!((big.epoch_s - 40.0).abs() < 1e-9);
        // Ada is 3.2x a Pascal card.
        let ada = estimate(
            TrainFamily::Yolo26,
            "n",
            640,
            16,
            900,
            10,
            Target::Cuda(ComputeCapability::new(8, 9)),
        )
        .unwrap();
        assert!((ada.epoch_s - 10.0 / f64::from(3.2f32)).abs() < 1e-9);
        // M1, measured constants.
        let m1 = estimate(TrainFamily::Rfdetr, "nano", 384, 4, 38, 1, Target::Mps).unwrap();
        assert!((m1.epoch_s - 38.0).abs() < 1e-9);
        assert!(estimate(TrainFamily::Dfine, "s", 640, 4, 10, 1, Target::Mps).is_none());
    }

    #[test]
    fn batch_suggestion_fits_the_card() {
        // 8 GB: YOLO26 s at 640 fits batch 16 (900 + 260 x 16 = 5060 MB), not 32.
        assert_eq!(suggest_batch(TrainFamily::Yolo26, "s", 640, 8192), Some(16));
        // 4 GB GTX 1650 class: RF-DETR large does not fit at all.
        assert_eq!(suggest_batch(TrainFamily::Rfdetr, "large", 704, 3000), None);
        // RF-DETR nano 384 trains at up to 544 px: 1500 + 650 x 2 x 2.007 = 4109 MB > 3686.
        assert_eq!(
            suggest_batch(TrainFamily::Rfdetr, "nano", 384, 4096),
            Some(1)
        );
        assert_eq!(
            suggest_batch_with(TrainFamily::Rfdetr, "nano", 384, 4096, false),
            Some(2)
        );
        assert_eq!(
            suggest_batch(TrainFamily::Yolo26, "n", 640, 100_000),
            Some(64)
        );
    }

    /// Run 20261027-071956-7f1c: RF-DETR nano 384 at batch 16 on a 16 GB M1 (the page suggested
    /// 16) needed over 20 GB and ran out of memory; the share of a 16 GB Mac now suggests 4.
    #[test]
    fn rfdetr_nano_on_a_16_gb_mac_suggests_a_batch_that_fits() {
        let mem = crate::training::estimate::to_u64(
            16.0 * 1024.0 * crate::training::capability::MPS_MEMORY_SHARE,
        );
        let b = suggest_batch(TrainFamily::Rfdetr, "nano", 384, mem).unwrap();
        assert!(b <= 4, "{b}");
        assert_eq!(b, 4);
        let e = estimate(TrainFamily::Rfdetr, "nano", 384, 16, 1, 1, Target::Mps).unwrap();
        assert!(e.vram_mb > 20_000, "{e:?}");
        // Multi-scale off: memory at the input size.
        assert_eq!(
            suggest_batch_with(TrainFamily::Rfdetr, "nano", 384, mem, false),
            Some(8)
        );
        let off = estimate_with(
            TrainFamily::Rfdetr,
            "nano",
            384,
            4,
            1,
            1,
            Target::Mps,
            false,
        );
        assert_eq!(off.unwrap().vram_mb, 1500 + 650 * 4);
        // Time follows the input size either way.
        let on = estimate(TrainFamily::Rfdetr, "nano", 384, 4, 38, 1, Target::Mps).unwrap();
        assert!((on.epoch_s - 38.0).abs() < 1e-9);
    }

    #[test]
    fn rfdetr_multi_scale_sides_follow_rfdetr() {
        // compute_multi_scale_scales(resolution, expanded_scales=True, patch, windows)[-1]
        assert_eq!(rfdetr_max_side("nano", 384), 544);
        assert_eq!(rfdetr_max_side("small", 512), 672);
        assert_eq!(rfdetr_max_side("medium", 576), 736);
        assert_eq!(rfdetr_max_side("base", 560), 840);
        assert_eq!(rfdetr_max_side("large", 704), 864);
        // Other families ignore multi-scale.
        let cc = Target::Cuda(ComputeCapability::new(8, 6));
        assert_eq!(
            estimate_with(TrainFamily::Yolo26, "n", 640, 8, 1, 1, cc, true),
            estimate_with(TrainFamily::Yolo26, "n", 640, 8, 1, 1, cc, false)
        );
    }

    #[test]
    fn a_measured_peak_sets_the_batch() {
        // 4 images peaked at 7 GB of a 16 GB share (9830 MB, 8847 usable): per image
        // (7168 - 1500) / 4 = 1417 MB; batch 4 fits, 8 (12836 MB) does not.
        assert_eq!(batch_from_peak(1500, 4, 7168.0, 9830), Some(4));
        // Lower measured peak than the estimate: a larger batch.
        assert_eq!(batch_from_peak(1500, 4, 3000.0, 9830), Some(16));
        // Base at most half the peak: 1 image at 2000 MB costs 1000 MB per image.
        assert_eq!(batch_from_peak(1500, 1, 2000.0, 4000), Some(2));
        assert_eq!(batch_from_peak(1500, 4, 50_000.0, 9830), None);
        assert_eq!(batch_from_peak(1500, 0, 1000.0, 9830), None);
        assert_eq!(batch_from_peak(1500, 4, f64::NAN, 9830), None);
    }

    #[test]
    fn gpu_generations_order() {
        let f = |a, b| gpu_factor(ComputeCapability::new(a, b));
        assert_eq!(f(6, 1), 1.0);
        assert_eq!(f(7, 5), 1.6);
        assert_eq!(f(8, 6), 2.2);
        assert_eq!(f(8, 9), 3.2);
        assert_eq!(f(12, 0), 4.5);
    }

    #[test]
    fn durations_read_well() {
        assert_eq!(format_duration(42.4), "42 s");
        assert_eq!(format_duration(600.0), "10 min");
        assert_eq!(format_duration(4.0 * 3600.0), "4.0 h");
        assert_eq!(to_u64(-3.0), 0);
        assert_eq!(to_u64(f64::NAN), 0);
    }
}
