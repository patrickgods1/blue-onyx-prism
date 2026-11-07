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

use super::{AutolabelRequest, GdinoSize, VlmSize};
use crate::backend::vram::{self, JobNeed, gb};
use crate::resources::catalog::{self as res_catalog, Resource};
use crate::training::capability::Accelerator;

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
}
