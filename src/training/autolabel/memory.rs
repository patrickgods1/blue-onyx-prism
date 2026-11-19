//! GPU memory of a labelers run: what each labeler takes on the card ([`labelers_on_gpu`]), the
//! options a card cannot hold at all ([`too_big`], refused by [`super::plan_autolabel`] and
//! disabled on the Auto-label tab), and the free-memory warning before a run ([`gpu_warning`]).
//!
//! `scripts/autolabel.py` loads every chosen labeler on one device and keeps them there for the
//! run, so they add up. A labeler takes about its checkpoint as stored (the script loads each at
//! the checkpoint's precision: MM-Grounding-DINO, RF-DETR and SAM 3 in f32, Qwen3-VL in bf16 / f16);
//! SigLIP 2 ranks a few crops per frame on the CPU when other models use the GPU and there are at
//! most 12 animal classes (the script's `siglip_device`), so it takes nothing there. An import's
//! embeddings put SigLIP 2's vision tower alone on the GPU ([`embed_need`]). The sizes come from the catalog pins, so a new pin
//! moves them.
//!
//! Host memory commit (Windows) is apart: [`commit_need`] is what a labelers run commits on the
//! host, for the check before it starts ([`crate::training::commit`]); [`embed_commit_need`] an
//! import's SigLIP 2 selection.

use super::{AutolabelRequest, GdinoSize, VlmSize};
use crate::backend::vram::{self, JobNeed, gb};
use crate::resources::catalog::{self as res_catalog, Resource};
use crate::training::capability::Accelerator;
use crate::training::commit::Need;

/// CUDA context and the allocator's reserve beside the weights, MB: torch's context alone is
/// 300-500 MB.
pub const CONTEXT_MB: u64 = 600;

/// Activations, cuDNN workspaces and allocator fragmentation on top of the weights while a run
/// labels, percent of the weights; for the free-memory warning only.
pub const WORKING_PERCENT: u64 = 20;

const MIB: u64 = 1024 * 1024;

/// One labeler as it sits on the GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnGpu {
    /// `Qwen3-VL 4B`.
    pub name: String,
    /// The weights as loaded, MB.
    pub weights_mb: u64,
}

/// The display name of a labeler checkpoint key.
fn title(key: &str) -> String {
    match key {
        "mm-gdino-tiny" => "MM-Grounding-DINO tiny".into(),
        "mm-gdino-base" => "MM-Grounding-DINO base".into(),
        "mm-gdino-large" => "MM-Grounding-DINO large".into(),
        "qwen3-vl-2b" => "Qwen3-VL 2B".into(),
        "qwen3-vl-4b" => "Qwen3-VL 4B".into(),
        "qwen3-vl-8b" => "Qwen3-VL 8B".into(),
        "sam3" => "SAM 3".into(),
        "rfdetr-large" => "RF-DETR L".into(),
        other => other.into(),
    }
}

fn on_gpu(key: &str, res: Option<&'static Resource>) -> Option<OnGpu> {
    Some(OnGpu {
        name: title(key),
        weights_mb: res?.size().div_ceil(MIB),
    })
}

/// The labelers `r` puts on the GPU, in load order (none for a run with a model, which runs in
/// this process on its own device). SigLIP 2 is left out: it runs on the CPU beside them.
pub fn labelers_on_gpu(r: &AutolabelRequest) -> Vec<OnGpu> {
    if r.with_model.is_some() {
        return Vec::new();
    }
    let mut v = Vec::new();
    if r.sam3 {
        v.extend(on_gpu("sam3", crate::resources::hf_token::sam3_resource()));
    }
    if let Some(g) = r.gdino {
        v.extend(on_gpu(g.key(), res_catalog::autolabel_weights(g.key())));
    }
    if r.rfdetr {
        v.extend(on_gpu(
            "rfdetr-large",
            res_catalog::base_weights(res_catalog::TrainFamily::Rfdetr, "large"),
        ));
    }
    if let Some(q) = r.vlm {
        v.extend(on_gpu(q.key(), res_catalog::autolabel_weights(q.key())));
    }
    v
}

/// What one option needs on a card at the least: its weights and the context. An option over
/// the card's memory cannot load whatever else runs.
pub fn option_need_mb(weights_mb: u64) -> u64 {
    weights_mb.saturating_add(CONTEXT_MB)
}

/// The GPU memory one VLM / open-vocabulary option needs at the least ([`option_need_mb`]),
/// by the form's value (`2b`, `4b`, `8b`, `tiny`, `base`, `large`, `sam3`); None for an unknown
/// value. What the Auto-label tab's options carry as `data-need-mb`.
pub fn form_option_need_mb(value: &str) -> Option<u64> {
    let res = if value == "sam3" {
        crate::resources::hf_token::sam3_resource()
    } else if let Some(v) = VlmSize::parse(value) {
        res_catalog::autolabel_weights(v.key())
    } else {
        res_catalog::autolabel_weights(GdinoSize::parse(value)?.key())
    }?;
    Some(option_need_mb(res.size().div_ceil(MIB)))
}

/// Why `r` cannot run on a card of `memory_mb` (0 = unknown: nothing is refused): a labeler
/// whose weights alone do not fit, with what would. Pure.
pub fn too_big(r: &AutolabelRequest, memory_mb: u64, gpu: &str) -> Option<String> {
    if memory_mb == 0 {
        return None;
    }
    let big: Vec<String> = labelers_on_gpu(r)
        .into_iter()
        .filter(|l| option_need_mb(l.weights_mb) > memory_mb)
        .map(|l| {
            format!(
                "{} needs at least {} of GPU memory",
                l.name,
                gb(option_need_mb(l.weights_mb))
            )
        })
        .collect();
    if big.is_empty() {
        return None;
    }
    let smaller = smaller_vlm(memory_mb).map_or_else(
        || "turn the VLM check off".to_string(),
        |v| format!("choose {} or turn the VLM check off", title(v.key())),
    );
    Some(format!(
        "{} and {gpu} has {}: it cannot load there. For the VLM check {smaller}; choose another \
         GPU, or label with a model instead (--with-model)",
        big.join(", "),
        gb(memory_mb)
    ))
}

/// The largest Qwen3-VL that fits `memory_mb` alone.
fn smaller_vlm(memory_mb: u64) -> Option<VlmSize> {
    [VlmSize::Qwen8b, VlmSize::Qwen4b, VlmSize::Qwen2b]
        .into_iter()
        .find(|v| form_option_need_mb(v.key()).is_some_and(|n| n <= memory_mb))
}

/// What the run needs on its GPU for the warning: the labelers' weights plus [`WORKING_PERCENT`]
/// plus [`CONTEXT_MB`]. None when nothing goes on the GPU.
pub fn job_need(r: &AutolabelRequest) -> Option<JobNeed> {
    let on = labelers_on_gpu(r);
    if on.is_empty() {
        return None;
    }
    let weights: u64 = on.iter().map(|l| l.weights_mb).sum();
    let working = weights.saturating_mul(WORKING_PERCENT) / 100;
    Some(JobNeed {
        what: "this auto-label run".into(),
        need_mb: weights.saturating_add(working).saturating_add(CONTEXT_MB),
        detail: format!(
            "{} together, loaded at once",
            on.iter()
                .map(|l| format!("{} {}", l.name, gb(l.weights_mb)))
                .collect::<Vec<_>>()
                .join(" + ")
        ),
        smaller: "a smaller VLM, or the VLM check off".into(),
    })
}

/// Host memory commit (Windows) of `scripts/autolabel.py` with RF-DETR L, bytes: Python, PyTorch,
/// the CUDA context and RF-DETR L. Measured: 2.6 GB for the overlap check's `--zoom-pairs`, the same
/// script and model ([`crate::training::overlaps::zoom::ZOOM_COMMIT_BYTES`]). A run without RF-DETR
/// L counts it too, at most ~0.3 GB high (its checkpoint is 136 MB).
pub const LABELERS_BASE_COMMIT_BYTES: u64 = crate::training::overlaps::zoom::ZOOM_COMMIT_BYTES;

/// Host commit SigLIP 2 adds (the whole model, loaded in float32), bytes. Measured on the GTX 1070
/// Ti box (2026-11-18, private bytes of the labeler's process tree): an RF-DETR L + SigLIP 2 run
/// took about 4.6 GB at load, then stayed flat over 220 frames (no leak); less
/// [`LABELERS_BASE_COMMIT_BYTES`].
pub const SIGLIP_COMMIT_BYTES: u64 = 2 << 30;

/// The least host commit SAM 3 adds, bytes: a lower bound only. The same box, the same day: a SAM
/// 3 + RF-DETR L run was killed while still loading SAM 3, at >= 7.6 GB; less
/// [`LABELERS_BASE_COMMIT_BYTES`]. Its estimate is the larger of this and
/// [`UNMEASURED_CHECKPOINT_FACTOR`] x its checkpoint (3.2 GB x 2 = 6.4 GB).
pub const SAM3_COMMIT_FLOOR_BYTES: u64 = 5 << 30;

/// Host commit of a labeler with no full measurement (MM-Grounding-DINO, Qwen3-VL; SAM 3 above its
/// floor) as a multiple of its checkpoint as stored (the precision the script loads it at): a
/// guess (GB here are GiB, as in the messages). The script builds each model on the host
/// (`from_pretrained`) before `.to(device)`; the measured ones sit at 1.4x (SigLIP 2: 2.0 GB for a
/// 1.4 GB checkpoint) and above 1.55x (SAM 3: still loading at 5.0 GB for 3.2 GB). With it the
/// default run (MM-Grounding-DINO tiny 1.3, RF-DETR L, Qwen3-VL 2B 7.9, SigLIP 2) is about 13.8 GB
/// and the SAM 3 + RF-DETR L + Qwen3-VL 2B + SigLIP 2 run about 19 GB, which is consistent with
/// that run growing the box's paging file from about 7.8 GB to its 24 GB maximum (a commit limit of
/// about 39.8 GB with ~18.6 GB in use by everything else: up to ~21 GB for the run).
pub const UNMEASURED_CHECKPOINT_FACTOR: u64 = 2;

/// Host commit of one labeler by catalog key ([`UNMEASURED_CHECKPOINT_FACTOR`] x its checkpoint),
/// with SAM 3 at least [`SAM3_COMMIT_FLOOR_BYTES`]; None when the build has no pin for it.
fn labeler_commit(key: &str, res: Option<&'static Resource>) -> Option<(String, u64)> {
    let estimate = res?.size().saturating_mul(UNMEASURED_CHECKPOINT_FACTOR);
    let bytes = if key == "sam3" {
        estimate.max(SAM3_COMMIT_FLOOR_BYTES)
    } else {
        estimate
    };
    Some((title(key), bytes))
}

/// The host commit (Windows) a labelers run of `r` on `classes` takes at its peak, for the check
/// before it starts ([`crate::training::commit`]): [`LABELERS_BASE_COMMIT_BYTES`] plus each
/// labeler it loads (they stay loaded for the whole run). SigLIP 2 counts only with two or more
/// animal classes, as the script loads it (`siglip_device`). None for a run with a model, which
/// labels in this process.
pub fn commit_need(r: &AutolabelRequest, classes: &[String]) -> Option<Need> {
    if r.with_model.is_some() {
        return None;
    }
    let mut parts: Vec<(String, u64)> = Vec::new();
    if r.sam3 {
        parts.extend(labeler_commit(
            "sam3",
            crate::resources::hf_token::sam3_resource(),
        ));
    }
    if let Some(g) = r.gdino {
        parts.extend(labeler_commit(
            g.key(),
            res_catalog::autolabel_weights(g.key()),
        ));
    }
    if let Some(q) = r.vlm {
        parts.extend(labeler_commit(
            q.key(),
            res_catalog::autolabel_weights(q.key()),
        ));
    }
    let animals = classes
        .iter()
        .filter(|c| super::Category::of(c) == super::Category::Animal)
        .count();
    if r.siglip && animals >= 2 {
        parts.push(("SigLIP 2".into(), SIGLIP_COMMIT_BYTES));
    }
    let mut by_size = parts.clone();
    by_size.sort_by_key(|a| std::cmp::Reverse(a.1));
    let optional: Vec<String> = by_size.into_iter().map(|(n, _)| n).collect();
    let base = if r.rfdetr {
        "PyTorch and RF-DETR L"
    } else {
        "PyTorch"
    };
    let bytes = parts
        .iter()
        .map(|(_, b)| *b)
        .fold(LABELERS_BASE_COMMIT_BYTES, u64::saturating_add);
    let mut detail = vec![format!("{base} {}", gb_text(LABELERS_BASE_COMMIT_BYTES))];
    detail.extend(parts.iter().map(|(n, b)| format!("{n} {}", gb_text(*b))));
    Some(Need {
        what: "this auto-label run".into(),
        bytes,
        detail: detail.join(" + "),
        lighter: (!optional.is_empty()).then(|| {
            format!(
                "turn off {} (or label with a trained model, --with-model)",
                optional.join(" or ")
            )
        }),
    })
}

/// Host commit (Windows) of an import's SigLIP 2 selection (`scripts/import_footage.py embed`,
/// which loads the whole model in float32 on the host and moves its vision tower to the GPU),
/// bytes: counted as the measured RF-DETR L + SigLIP 2 run ([`LABELERS_BASE_COMMIT_BYTES`] +
/// [`SIGLIP_COMMIT_BYTES`], 4.6 GB), not measured on its own; RF-DETR L's share makes it ~0.3 GB
/// high.
pub const EMBED_COMMIT_BYTES: u64 = LABELERS_BASE_COMMIT_BYTES + SIGLIP_COMMIT_BYTES;

/// [`EMBED_COMMIT_BYTES`] as the import's [`Need`].
pub fn embed_commit_need() -> Need {
    Need {
        what: "this import's SigLIP 2 selection".into(),
        bytes: EMBED_COMMIT_BYTES,
        detail: format!("PyTorch and SigLIP 2 {}", gb_text(EMBED_COMMIT_BYTES)),
        lighter: Some("use the dHash selection (--select dhash), which loads no model".into()),
    }
}

/// `2.6 GB`.
fn gb_text(bytes: u64) -> String {
    format!("{:.1} GB", crate::training::gib(bytes))
}

/// SigLIP 2's vision tower in float32, MB (92.9 M parameters, measured): what an import's
/// embeddings put on the GPU (`scripts/import_footage.py embed`, half of it in float16 on
/// compute capability 7.0 and newer).
pub const SIGLIP_VISION_MB: u64 = 354;

/// Activations of one embedding batch (16 images at 224 px), MB, rounded up.
pub const EMBED_BATCH_MB: u64 = 200;

/// What an import's SigLIP 2 embeddings need on CUDA device `cc` major `cc_major` (None:
/// unknown, float32 assumed), for [`vram::check_cuda_job`] before that step.
pub fn embed_need(cc_major: Option<u32>) -> JobNeed {
    let weights = if cc_major.is_some_and(|m| m >= 7) {
        SIGLIP_VISION_MB / 2
    } else {
        SIGLIP_VISION_MB
    };
    JobNeed {
        what: "this import's SigLIP 2 embeddings".into(),
        need_mb: weights + EMBED_BATCH_MB + CONTEXT_MB,
        detail: format!("its vision tower {} and a batch of 16", gb(weights)),
        smaller: "the dHash selection (--select dhash), which needs no GPU".into(),
    }
}

/// The warning before a labelers run of `r` on `accel` when the card is short of free memory
/// now; None when it fits, cannot be read, or on Apple silicon (unified memory).
pub fn gpu_warning(r: &AutolabelRequest, accel: &Accelerator) -> Option<String> {
    match accel {
        Accelerator::Cuda { ordinal, .. } => vram::check_cuda_job(*ordinal, &job_need(r)?),
        Accelerator::Mps { .. } => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::FrameFilter;
    use super::*;

    fn request() -> AutolabelRequest {
        AutolabelRequest::new(FrameFilter::default())
    }

    fn mb(key: &str) -> u64 {
        res_catalog::autolabel_weights(key)
            .unwrap()
            .size()
            .div_ceil(MIB)
    }

    #[test]
    fn the_default_run_puts_three_labelers_on_the_gpu() {
        let names: Vec<String> = labelers_on_gpu(&request())
            .into_iter()
            .map(|l| l.name)
            .collect();
        assert_eq!(
            names,
            ["MM-Grounding-DINO tiny", "RF-DETR L", "Qwen3-VL 2B"]
        );
        let mut sam = request();
        sam.gdino = None;
        sam.sam3 = true;
        assert_eq!(labelers_on_gpu(&sam)[0].name, "SAM 3");
        let mut model = request();
        model.with_model = Some("IPcam-general".into());
        assert!(labelers_on_gpu(&model).is_empty());
        assert_eq!(job_need(&model), None);
        let need = job_need(&request()).unwrap();
        let weights: u64 = labelers_on_gpu(&request())
            .iter()
            .map(|l| l.weights_mb)
            .sum();
        assert_eq!(
            need.need_mb,
            weights + weights * WORKING_PERCENT / 100 + CONTEXT_MB
        );
        assert!(need.detail.contains("Qwen3-VL 2B"), "{}", need.detail);
    }

    #[test]
    fn options_a_card_cannot_hold_are_refused_with_what_fits() {
        let mut r = request();
        // Unknown memory: nothing refused.
        r.vlm = Some(VlmSize::Qwen8b);
        assert_eq!(too_big(&r, 0, "GPU"), None);
        // Qwen3-VL 8B (17.5 GB) on an 8 GB card.
        let e = too_big(&r, 8192, "NVIDIA GeForce GTX 1070 Ti (cuda:0)").unwrap();
        assert!(e.starts_with("Qwen3-VL 8B needs at least "), "{e}");
        assert!(
            e.contains("NVIDIA GeForce GTX 1070 Ti (cuda:0) has 8.0 GB"),
            "{e}"
        );
        assert!(
            e.contains("choose Qwen3-VL 2B or turn the VLM check off"),
            "{e}"
        );
        // 4B (8.9 GB) does not fit 8 GB either; it does fit 12 GB, where 8B does not.
        r.vlm = Some(VlmSize::Qwen4b);
        assert!(too_big(&r, 8192, "g").is_some());
        assert_eq!(too_big(&r, 12288, "g"), None);
        r.vlm = Some(VlmSize::Qwen8b);
        assert!(
            too_big(&r, 12288, "g")
                .unwrap()
                .contains("choose Qwen3-VL 4B")
        );
        assert_eq!(too_big(&r, 24576, "g"), None);
        // The default run fits an 8 GB card; on a 4 GB one 2B does not and nothing smaller exists.
        let d = request();
        assert_eq!(too_big(&d, 8192, "g"), None);
        let small = too_big(&d, 4096, "g").unwrap();
        assert!(small.contains("Qwen3-VL 2B needs"), "{small}");
        assert!(
            small.contains("For the VLM check turn the VLM check off"),
            "{small}"
        );
    }

    #[test]
    fn form_values_carry_the_catalog_sizes() {
        assert_eq!(
            form_option_need_mb("2b"),
            Some(mb("qwen3-vl-2b") + CONTEXT_MB)
        );
        assert_eq!(
            form_option_need_mb("8b"),
            Some(mb("qwen3-vl-8b") + CONTEXT_MB)
        );
        assert_eq!(
            form_option_need_mb("tiny"),
            Some(mb("mm-gdino-tiny") + CONTEXT_MB)
        );
        assert!(form_option_need_mb("sam3").unwrap() > 3000);
        assert_eq!(form_option_need_mb("off"), None);
    }

    /// The Auto-label tab's options carry the needs this module refuses by, so the page disables
    /// exactly what a request would be refused for.
    #[test]
    fn the_page_options_carry_the_same_needs() {
        let html = include_str!("../../../templates/train/autolabel.html");
        for value in ["tiny", "base", "large", "sam3", "2b", "4b", "8b"] {
            let need = form_option_need_mb(value).unwrap();
            let want = format!("value=\"{value}\" data-need-mb=\"{need}\"");
            assert!(html.contains(&want), "{want} missing from autolabel.html");
        }
    }

    #[test]
    fn embeddings_need_the_vision_tower_only() {
        let pascal = embed_need(Some(6));
        assert_eq!(
            pascal.need_mb,
            SIGLIP_VISION_MB + EMBED_BATCH_MB + CONTEXT_MB
        );
        assert_eq!(embed_need(None).need_mb, pascal.need_mb);
        let turing = embed_need(Some(7));
        assert_eq!(
            turing.need_mb,
            SIGLIP_VISION_MB / 2 + EMBED_BATCH_MB + CONTEXT_MB
        );
        assert!(turing.smaller.contains("dhash"), "{}", turing.smaller);
    }

    #[test]
    fn apple_silicon_gets_no_free_memory_warning() {
        let mps = Accelerator::Mps {
            name: "Apple M1".into(),
            memory_mb: 1,
        };
        assert_eq!(gpu_warning(&request(), &mps), None);
    }

    const GIB: u64 = 1 << 30;

    fn classes(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    /// A request with only the given labelers.
    fn only(
        sam3: bool,
        gdino: Option<GdinoSize>,
        vlm: Option<VlmSize>,
        siglip: bool,
    ) -> AutolabelRequest {
        let mut r = request();
        r.sam3 = sam3;
        r.gdino = gdino;
        r.vlm = vlm;
        r.siglip = siglip;
        r
    }

    #[test]
    fn commit_needs_add_up_the_labelers_a_run_loads() {
        let animals = classes(&["person", "car", "dog", "cat"]);
        // RF-DETR L + SigLIP 2: the measured 4.6 GB.
        let rf_siglip = commit_need(&only(false, None, None, true), &animals).unwrap();
        assert_eq!(
            rf_siglip.bytes,
            LABELERS_BASE_COMMIT_BYTES + SIGLIP_COMMIT_BYTES
        );
        assert_eq!(rf_siglip.bytes, EMBED_COMMIT_BYTES);
        assert_eq!(
            rf_siglip.detail,
            "PyTorch and RF-DETR L 2.6 GB + SigLIP 2 2.0 GB"
        );
        assert_eq!(
            rf_siglip.lighter.as_deref(),
            Some("turn off SigLIP 2 (or label with a trained model, --with-model)")
        );
        // SigLIP 2 is not loaded with fewer than two animal classes: RF-DETR L alone.
        let one_animal = commit_need(&only(false, None, None, true), &classes(&["person", "dog"]));
        assert_eq!(one_animal.unwrap().bytes, LABELERS_BASE_COMMIT_BYTES);
        let rf = commit_need(&only(false, None, None, false), &animals).unwrap();
        assert_eq!((rf.bytes, rf.lighter), (LABELERS_BASE_COMMIT_BYTES, None));
        // SAM 3: twice its 3.2 GB checkpoint, above its 5.0 GB floor.
        let sam = commit_need(&only(true, None, None, false), &animals).unwrap();
        let sam3 = crate::resources::hf_token::sam3_resource().unwrap().size();
        assert_eq!(sam.bytes, LABELERS_BASE_COMMIT_BYTES + 2 * sam3);
        assert!(2 * sam3 > SAM3_COMMIT_FLOOR_BYTES);
        assert!(sam.detail.ends_with(" + SAM 3 6.4 GB"), "{}", sam.detail);
        // The default run: MM-Grounding-DINO tiny, RF-DETR L, Qwen3-VL 2B, SigLIP 2, about 13.8 GB.
        let d = commit_need(&request(), &animals).unwrap();
        assert_eq!(
            d.bytes,
            LABELERS_BASE_COMMIT_BYTES
                + 2 * mb_bytes("mm-gdino-tiny")
                + 2 * mb_bytes("qwen3-vl-2b")
                + SIGLIP_COMMIT_BYTES
        );
        assert!((13 * GIB..14 * GIB).contains(&d.bytes), "{}", d.detail);
        assert_eq!(
            d.lighter.as_deref(),
            Some(
                "turn off Qwen3-VL 2B or SigLIP 2 or MM-Grounding-DINO tiny (or label with a \
                 trained model, --with-model)"
            )
        );
        // The run that filled C:: SAM 3, RF-DETR L, Qwen3-VL 2B, SigLIP 2, about 19 GB.
        let full = commit_need(&only(true, None, Some(VlmSize::Qwen2b), true), &animals).unwrap();
        assert!(
            (18 * GIB..20 * GIB).contains(&full.bytes),
            "{}",
            full.detail
        );
        // A run with a model labels in this process.
        let mut model = request();
        model.with_model = Some("IPcam-general".into());
        assert_eq!(commit_need(&model, &animals), None);
    }

    fn mb_bytes(key: &str) -> u64 {
        res_catalog::autolabel_weights(key).unwrap().size()
    }

    /// The decision on the dev box (commit 23.65 GB limit, ~5.05 GB free, `c:\pagefile.sys 2935
    /// 24576`, 15.2 GB free on C:, so no growth under the 16 GB margin) for each labeler set.
    #[test]
    fn this_box_runs_rfdetr_alone_and_refuses_the_rest() {
        use crate::training::commit::{refusal, tests::this_box};
        let animals = classes(&["person", "car", "dog", "cat"]);
        let h = this_box();
        let refused = |r: &AutolabelRequest| refusal(&commit_need(r, &animals).unwrap(), Some(&h));
        // RF-DETR L alone: 2.6 + 1 spare fits 5.05.
        assert_eq!(refused(&only(false, None, None, false)), None);
        // RF-DETR L + SigLIP 2: 4.6 + 1 does not (that run grew the paging file by 2 GB from here).
        let e = refused(&only(false, None, None, true)).unwrap();
        assert!(
            e.starts_with("this auto-label run needs about 5.6 GB"),
            "{e}"
        );
        // Every SAM 3 run.
        for r in [
            only(true, None, None, false),
            only(true, None, Some(VlmSize::Qwen2b), true),
        ] {
            let e = refused(&r).unwrap();
            assert!(e.contains("SAM 3"), "{e}");
            assert!(
                e.contains("the paging file on C: can grow only while 16 GB"),
                "{e}"
            );
        }
        // The example in docs/TRAINING.md ("Memory commit check before a job").
        assert_eq!(
            refused(&only(true, None, None, false)).unwrap(),
            "this auto-label run needs about 10.0 GB of memory commit (PyTorch and RF-DETR L 2.6 \
             GB + SAM 3 6.4 GB, 1.0 GB kept spare); 5.0 GB is free and the paging file on C: can \
             grow only while 16 GB stays free there (15.2 GB free now). Free space on C:, put the \
             paging file on another drive, or turn off SAM 3 (or label with a trained model, \
             --with-model), then start the job again."
        );
        // The default run.
        assert!(refused(&request()).is_some());
    }
}
