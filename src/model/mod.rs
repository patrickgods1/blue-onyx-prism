//! Model families: how to build the input tensor(s) for a model and how to turn its raw
//! outputs into detections in original-image pixel coordinates.
//!
//! Everything here is pure CPU code with no OpenVINO dependency so it is unit-testable.

pub mod classes;
pub mod detr;
pub mod nms;
pub mod preprocess;
pub mod rtdetr;
pub mod yolo26;
pub mod yolo5;
pub mod yolo8;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelFamilyKind {
    /// Pick from the model's output shapes at load time.
    #[default]
    Auto,
    /// Ultralytics YOLO26 exported end-to-end (`nms=False`): output `[1, 300, 6]`.
    Yolo26,
    /// YOLOv5 (incl. MikeLud ipcam models): output `[1, N, 5+C]`, needs NMS.
    Yolo5,
    /// YOLOv8/11 raw: output `[1, 4+C, 8400]`, needs NMS.
    Yolo8,
    /// RT-DETRv2 ONNX: inputs `images` + `orig_target_sizes`, outputs `labels/boxes/scores`.
    RtDetr,
    /// Hugging Face DETR-style export (D-FINE, RF-DETR): input `pixel_values`, outputs
    /// `logits [1,Q,C]` + `pred_boxes [1,Q,4]` (normalized cx,cy,w,h). Stretch, x/255; ImageNet
    /// mean/std normalization only for 91-class (COCO category id) heads, i.e. RF-DETR.
    Detr,
    /// RF-DETR (same ports as [`ModelFamilyKind::Detr`]) with ImageNet mean/std normalization
    /// regardless of the class count, for fine-tuned RF-DETR models.
    RfDetr,
}

impl std::fmt::Display for ModelFamilyKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl ModelFamilyKind {
    /// Every family, in the order shown in help texts and UI selects.
    pub const ALL: [ModelFamilyKind; 7] = [
        ModelFamilyKind::Auto,
        ModelFamilyKind::Yolo26,
        ModelFamilyKind::Yolo5,
        ModelFamilyKind::Yolo8,
        ModelFamilyKind::RtDetr,
        ModelFamilyKind::Detr,
        ModelFamilyKind::RfDetr,
    ];

    /// Lowercase name, as in `Display`, serde and `--family`.
    pub fn name(self) -> &'static str {
        match self {
            ModelFamilyKind::Auto => "auto",
            ModelFamilyKind::Yolo26 => "yolo26",
            ModelFamilyKind::Yolo5 => "yolo5",
            ModelFamilyKind::Yolo8 => "yolo8",
            ModelFamilyKind::RtDetr => "rtdetr",
            ModelFamilyKind::Detr => "detr",
            ModelFamilyKind::RfDetr => "rfdetr",
        }
    }

    /// One-line description for CLI help (`--family`).
    pub fn description(self) -> &'static str {
        match self {
            ModelFamilyKind::Auto => "Pick from the model's output shapes at load time",
            ModelFamilyKind::Yolo26 => {
                "Ultralytics YOLO26 exported end-to-end (`nms=False`): output `[1, 300, 6]`"
            }
            ModelFamilyKind::Yolo5 => {
                "YOLOv5 (incl. MikeLud ipcam models): output `[1, N, 5+C]`, needs NMS"
            }
            ModelFamilyKind::Yolo8 => "YOLOv8/11 raw: output `[1, 4+C, 8400]`, needs NMS",
            ModelFamilyKind::RtDetr => {
                "RT-DETRv2 ONNX: inputs `images` + `orig_target_sizes`, outputs `labels/boxes/scores`"
            }
            ModelFamilyKind::Detr => {
                "Hugging Face DETR-style export (D-FINE, RF-DETR): input `pixel_values`, outputs \
                 `logits [1,Q,C]` + `pred_boxes [1,Q,4]` (normalized cx,cy,w,h). Stretch, x/255; \
                 ImageNet mean/std normalization only for 91-class (COCO category id) heads, i.e. RF-DETR"
            }
            ModelFamilyKind::RfDetr => {
                "RF-DETR (same ports as `detr`) with ImageNet mean/std normalization regardless \
                 of the class count, for fine-tuned RF-DETR models"
            }
        }
    }
}

/// Parses the lowercase names used by `Display` and serde (`auto`, `yolo26`, ..., `rfdetr`).
impl std::str::FromStr for ModelFamilyKind {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        ModelFamilyKind::ALL
            .into_iter()
            .find(|k| k.name() == s)
            .ok_or_else(|| {
                let valid: Vec<&str> = ModelFamilyKind::ALL.iter().map(|k| k.name()).collect();
                anyhow::anyhow!("unknown model family '{s}'; valid: {}", valid.join(", "))
            })
    }
}

/// How the source image is fitted into the model input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResizeMode {
    /// Keep aspect ratio, pad with gray (114) and center. Used by YOLO families.
    Letterbox,
    /// Stretch to the input size. Used by RT-DETR.
    Stretch,
}

/// Per-channel normalization applied after scaling to `0..1`: `(v - mean) / std` (RGB order).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normalization {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Normalization {
    /// ImageNet statistics (DINOv2 backbones such as RF-DETR).
    pub const IMAGENET: Normalization = Normalization {
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
    };
}

/// Geometry recorded during preprocessing so boxes can be mapped back to the original image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreprocessCtx {
    pub orig_w: u32,
    pub orig_h: u32,
    pub input_w: u32,
    pub input_h: u32,
    pub mode: ResizeMode,
    /// Letterbox: uniform scale applied to the source. Stretch: unused (1.0).
    pub scale: f32,
    /// Letterbox: left/top padding in input pixels. Stretch: 0.
    pub pad_x: f32,
    pub pad_y: f32,
}

impl PreprocessCtx {
    /// Map a box in model-input pixel coordinates back to original-image coordinates (unclamped).
    pub fn to_original(&self, x1: f32, y1: f32, x2: f32, y2: f32) -> (f32, f32, f32, f32) {
        match self.mode {
            ResizeMode::Letterbox => {
                let s = self.scale.max(f32::EPSILON);
                (
                    (x1 - self.pad_x) / s,
                    (y1 - self.pad_y) / s,
                    (x2 - self.pad_x) / s,
                    (y2 - self.pad_y) / s,
                )
            }
            ResizeMode::Stretch => {
                let sx = self.orig_w as f32 / self.input_w.max(1) as f32;
                let sy = self.orig_h as f32 / self.input_h.max(1) as f32;
                (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
            }
        }
    }
}

/// A raw output tensor copied out of the inference request. Backends refill the same buffers
/// request after request ([`Self::f32_mut`] and friends keep the allocation), so steady state
/// does not allocate.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputBuf {
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
}

impl Default for OutputBuf {
    fn default() -> Self {
        OutputBuf::F32(Vec::new())
    }
}

impl OutputBuf {
    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            OutputBuf::F32(v) => v.len(),
            OutputBuf::I64(v) => v.len(),
            OutputBuf::I32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The buffer as an empty f32 vector, keeping its capacity when it already holds f32.
    pub fn f32_mut(&mut self) -> &mut Vec<f32> {
        if !matches!(self, OutputBuf::F32(_)) {
            *self = OutputBuf::F32(Vec::new());
        }
        let OutputBuf::F32(v) = self else {
            unreachable!()
        };
        v.clear();
        v
    }

    /// The buffer as an empty i64 vector, keeping its capacity when it already holds i64.
    pub fn i64_mut(&mut self) -> &mut Vec<i64> {
        if !matches!(self, OutputBuf::I64(_)) {
            *self = OutputBuf::I64(Vec::new());
        }
        let OutputBuf::I64(v) = self else {
            unreachable!()
        };
        v.clear();
        v
    }

    /// The buffer as an empty i32 vector, keeping its capacity when it already holds i32.
    pub fn i32_mut(&mut self) -> &mut Vec<i32> {
        if !matches!(self, OutputBuf::I32(_)) {
            *self = OutputBuf::I32(Vec::new());
        }
        let OutputBuf::I32(v) = self else {
            unreachable!()
        };
        v.clear();
        v
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct NamedOutput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: OutputBuf,
}

impl NamedOutput {
    pub fn as_f32(&self) -> Option<&[f32]> {
        match &self.data {
            OutputBuf::F32(v) => Some(v),
            _ => None,
        }
    }

    /// Integer outputs as i64 regardless of storage width: the stored slice for i64, else the
    /// values widened into `scratch` (owned by the caller, reused across calls). None for f32.
    pub fn as_i64<'a>(&'a self, scratch: &'a mut Vec<i64>) -> Option<&'a [i64]> {
        match &self.data {
            OutputBuf::I64(v) => Some(v),
            OutputBuf::I32(v) => {
                scratch.clear();
                scratch.extend(v.iter().map(|&x| i64::from(x)));
                Some(scratch)
            }
            OutputBuf::F32(_) => None,
        }
    }

    /// Prepare `outs` to receive `n` outputs: keeps the existing entries (and their buffers),
    /// adds empty ones or drops extra ones.
    pub fn resize_slots(outs: &mut Vec<NamedOutput>, n: usize) {
        outs.resize_with(n, NamedOutput::default);
    }
}

/// An additional (non-image) input tensor a family needs per request.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtraInput {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: ExtraData,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExtraData {
    I64(Vec<i64>),
    I32(Vec<i32>),
    F32(Vec<f32>),
}

/// Shape/type description of a model port, used for family auto-detection and validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortSpec {
    pub name: String,
    /// -1 for dynamic dimensions.
    pub shape: Vec<i64>,
    pub elem: PortElem,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortElem {
    F32,
    F16,
    I64,
    I32,
    U8,
    Other,
}

/// A detection in original-image pixel coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub class_id: usize,
}

impl Detection {
    /// Every coordinate and the score is a finite number. Non-finite outputs (fp16 overflow,
    /// a broken export) must never reach NMS or the response (see `to_predictions`).
    pub fn is_finite(&self) -> bool {
        [self.x1, self.y1, self.x2, self.y2, self.score]
            .iter()
            .all(|v| v.is_finite())
    }
}

/// Drop every detection with a non-finite coordinate or score.
pub fn retain_finite(dets: &mut Vec<Detection>) {
    dets.retain(Detection::is_finite);
}

/// Largest class index a model output is allowed to name (anything above is a broken export).
pub const MAX_CLASS_INDEX: f32 = 1_000_000.0;

/// A class id emitted as a float: finite, non-negative and plausible, else None.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "checked finite and clamped to 0..=MAX_CLASS_INDEX before the cast"
)]
pub(crate) fn class_index(v: f32) -> Option<usize> {
    let r = v.round();
    (r.is_finite() && (0.0..=MAX_CLASS_INDEX).contains(&r)).then_some(r as u32 as usize)
}

/// Thresholds applied during post-processing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PostParams {
    pub confidence_threshold: f32,
    pub nms_iou: f32,
}

/// Behaviour of one model family. Implementations are stateless apart from the class count.
pub trait Family: Send + Sync {
    fn kind(&self) -> ModelFamilyKind;
    fn resize_mode(&self) -> ResizeMode;
    /// Mean/std normalization of the `0..1` input, if the model expects it (see
    /// [`preprocess::Preprocessor::for_family`]).
    fn normalization(&self) -> Option<Normalization> {
        None
    }
    /// Extra inputs besides the image tensor (RT-DETR's `orig_target_sizes`).
    fn extra_inputs(&self, _ctx: &PreprocessCtx) -> Vec<ExtraInput> {
        Vec::new()
    }
    /// Convert raw outputs to detections in original-image coordinates (unclamped).
    fn postprocess(
        &self,
        outputs: &[NamedOutput],
        ctx: &PreprocessCtx,
        params: &PostParams,
    ) -> Result<Vec<Detection>>;
}

/// Pick a family implementation. `Auto` inspects the port specs.
pub fn make_family(
    kind: ModelFamilyKind,
    inputs: &[PortSpec],
    outputs: &[PortSpec],
    num_classes: usize,
) -> Result<Box<dyn Family>> {
    let kind = if kind == ModelFamilyKind::Auto {
        detect_family(inputs, outputs, num_classes)?
    } else {
        kind
    };
    Ok(match kind {
        ModelFamilyKind::Yolo26 => Box::new(yolo26::Yolo26::new(num_classes)),
        ModelFamilyKind::Yolo5 => Box::new(yolo5::Yolo5::new(num_classes)),
        ModelFamilyKind::Yolo8 => Box::new(yolo8::Yolo8::new(num_classes)),
        ModelFamilyKind::RtDetr => Box::new(rtdetr::RtDetr::new(num_classes)),
        ModelFamilyKind::Detr => Box::new(detr::Detr::from_ports(outputs, num_classes, false)),
        ModelFamilyKind::RfDetr => Box::new(detr::Detr::from_ports(outputs, num_classes, true)),
        ModelFamilyKind::Auto => unreachable!(),
    })
}

/// Heuristic family detection from port shapes:
/// - outputs `logits` + `pred_boxes` (input `pixel_values`) -> Detr (D-FINE, RF-DETR)
/// - an input named `orig_target_sizes`, or 3 outputs including `boxes` -> RtDetr
/// - single output `[.., 4+C, A]` with A > 4+C -> Yolo8
/// - single output `[.., N, 6]` -> Yolo26 (end-to-end), except a 1-class model with N > 1000 or
///   dynamic N, which is a 1-class YOLOv5 (`[.., A, 5+1]`)
/// - single output `[.., A, 5+C]` -> Yolo5
pub fn detect_family(
    inputs: &[PortSpec],
    outputs: &[PortSpec],
    num_classes: usize,
) -> Result<ModelFamilyKind> {
    if detr::is_detr(inputs, outputs) {
        return Ok(ModelFamilyKind::Detr);
    }
    if inputs.iter().any(|p| p.name == "orig_target_sizes")
        || (outputs.len() == 3 && outputs.iter().any(|p| p.name == "boxes"))
    {
        return Ok(ModelFamilyKind::RtDetr);
    }
    if let [only] = outputs
        && let [.., prev, last] = only.shape.as_slice()
    {
        let (prev, last) = (*prev, *last);
        let classes = i64::try_from(num_classes).ok().filter(|&c| c > 0);
        {
            // Class-count rules first: a 1-class YOLOv5 export (`[1, 25200, 6]`) has the same
            // last dim as YOLO26's `[1, 300, 6]`. YOLO26/YOLOv10 keep a fixed top-k of 300 rows,
            // YOLOv5 has thousands of anchors (1008 even at 128 px) or a dynamic dim.
            if classes.is_some_and(|c| prev == 4 + c) && last > prev {
                return Ok(ModelFamilyKind::Yolo8);
            }
            if last == 6 {
                let yolo5_rows = !(0..=1000).contains(&prev);
                return Ok(if classes == Some(1) && yolo5_rows {
                    ModelFamilyKind::Yolo5
                } else {
                    ModelFamilyKind::Yolo26
                });
            }
            if classes.is_some_and(|c| last == 5 + c) {
                return Ok(ModelFamilyKind::Yolo5);
            }
            // Fallbacks when the class count is unknown.
            if prev < last && (5..=1000).contains(&prev) {
                return Ok(ModelFamilyKind::Yolo8);
            }
            if (6..=1000).contains(&last) {
                return Ok(ModelFamilyKind::Yolo5);
            }
        }
    }
    anyhow::bail!(
        "could not auto-detect the model family from its outputs {:?}; the model is not a \
         YOLOv5/v8/26, DETR or RT-DETR export, or its classes file has another class count; set \
         `family` for this model (yolo5, yolo8, yolo26, detr, rfdetr, rtdetr)",
        outputs
            .iter()
            .map(|o| (&o.name, &o.shape))
            .collect::<Vec<_>>()
    )
}

/// Input size `(w, h)` to use when the model's image input has dynamic height/width:
/// 1. `catalog_hint`: the download catalog's size for this model, looked up by the caller from
///    the file name (e.g. `rfdetr-base.onnx`); the model layer does not depend on the catalog,
/// 2. `size` from a Hugging Face `preprocessor_config.json` when the model sits in a repo's `onnx/`
///    directory (`<repo>/onnx/model.onnx` + `<repo>/preprocessor_config.json`),
/// 3. 560 for RF-DETR-style outputs (`logits` with 91 COCO category ids), the RF-DETR base size,
/// 4. 640.
pub fn dynamic_input_size(
    path: &std::path::Path,
    outputs: &[PortSpec],
    catalog_hint: Option<u32>,
) -> (u32, u32) {
    if let Some(s) = catalog_hint {
        return (s, s);
    }
    if let Some(dir) = path.parent()
        && dir.file_name().is_some_and(|d| d == "onnx")
        && let Some(repo) = dir.parent()
        && let Some(size) = hf_preprocessor_size(&repo.join("preprocessor_config.json"))
    {
        return size;
    }
    if detr::logits_classes(outputs) == Some(detr::COCO91_CLASSES) {
        return (detr::RFDETR_DEFAULT_SIZE, detr::RFDETR_DEFAULT_SIZE);
    }
    (DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE)
}

/// Default side of a square model input.
pub const DEFAULT_INPUT_SIZE: u32 = 640;

/// `size` of a Hugging Face `preprocessor_config.json`: `{"height", "width"}`, or
/// `{"shortest_edge"}` / `{"longest_edge"}` as a square.
pub fn hf_preprocessor_size(path: &std::path::Path) -> Option<(u32, u32)> {
    let text = std::fs::read_to_string(path).ok()?;
    parse_hf_preprocessor_size(&text)
}

/// See [`hf_preprocessor_size`].
pub fn parse_hf_preprocessor_size(text: &str) -> Option<(u32, u32)> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    let size = v.get("size")?;
    let dim = |k: &str| {
        size.get(k)
            .and_then(|x| x.as_u64())
            .and_then(|x| u32::try_from(x).ok())
            .filter(|&x| (32..=4096).contains(&x))
    };
    match (dim("width"), dim("height")) {
        (Some(w), Some(h)) => Some((w, h)),
        _ => dim("shortest_edge")
            .or_else(|| dim("longest_edge"))
            .map(|s| (s, s)),
    }
}

/// Common post-step: class filter (case-insensitive), clamp to the image, round to pixels.
/// Confidence thresholds are applied by the families themselves. Detections with a non-finite
/// coordinate or score are dropped and the score is clamped to `0..=1`, so the JSON never holds
/// `null` or a `(0,0,0,0)` box (PL-02).
pub fn to_predictions(
    dets: &[Detection],
    ctx: &PreprocessCtx,
    class_names: &[String],
    filter: Option<&[String]>,
) -> Vec<crate::api::Prediction> {
    let w = ctx.orig_w.max(1) as f32;
    let h = ctx.orig_h.max(1) as f32;
    dets.iter()
        .filter_map(|d| {
            let label = class_names
                .get(d.class_id)
                .cloned()
                .unwrap_or_else(|| format!("class_{}", d.class_id));
            if let Some(f) = filter
                && !f.is_empty()
                && !f.iter().any(|x| x.eq_ignore_ascii_case(&label))
            {
                return None;
            }
            if !d.is_finite() {
                return None;
            }
            let x1 = d.x1.clamp(0.0, w);
            let y1 = d.y1.clamp(0.0, h);
            let x2 = d.x2.clamp(0.0, w);
            let y2 = d.y2.clamp(0.0, h);
            if x2 <= x1 || y2 <= y1 {
                return None;
            }
            Some(crate::api::Prediction {
                x_min: pixel(x1),
                y_min: pixel(y1),
                x_max: pixel(x2),
                y_max: pixel(y2),
                confidence: d.score.clamp(0.0, 1.0),
                label,
            })
        })
        .collect()
}

/// A finite coordinate already clamped to `0..=side` (a `u32` image side) as a pixel index.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "caller clamps to 0..=image side (u32), so the rounded value fits usize"
)]
fn pixel(v: f32) -> usize {
    v.round().max(0.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn port(name: &str, shape: &[i64]) -> PortSpec {
        PortSpec {
            name: name.into(),
            shape: shape.to_vec(),
            elem: PortElem::F32,
        }
    }

    #[test]
    fn family_detection() {
        assert_eq!(
            detect_family(
                &[port("images", &[1, 3, 640, 640])],
                &[port("output0", &[1, 300, 6])],
                80
            )
            .unwrap(),
            ModelFamilyKind::Yolo26
        );
        assert_eq!(
            detect_family(
                &[port("images", &[1, 3, 640, 640])],
                &[port("output0", &[1, 84, 8400])],
                80
            )
            .unwrap(),
            ModelFamilyKind::Yolo8
        );
        assert_eq!(
            detect_family(
                &[port("images", &[1, 3, 640, 640])],
                &[port("output", &[1, 25200, 8])],
                3
            )
            .unwrap(),
            ModelFamilyKind::Yolo5
        );
        // PL-05: a 1-class YOLOv5 export has rows `cx,cy,w,h,obj,cls0` (last dim 6, like YOLO26).
        let img = [port("images", &[1, 3, 640, 640])];
        for rows in [25200, 6300, -1] {
            assert_eq!(
                detect_family(&img, &[port("output", &[1, rows, 6])], 1).unwrap(),
                ModelFamilyKind::Yolo5,
                "rows {rows}"
            );
        }
        // A 1-class YOLO26 keeps its fixed 300 rows; a 1-class YOLOv8 is `[1, 5, 8400]`.
        assert_eq!(
            detect_family(&img, &[port("output0", &[1, 300, 6])], 1).unwrap(),
            ModelFamilyKind::Yolo26
        );
        assert_eq!(
            detect_family(&img, &[port("output0", &[1, 5, 8400])], 1).unwrap(),
            ModelFamilyKind::Yolo8
        );
        // 2-class YOLOv8 `[1, 6, 8400]` is not YOLO26 either.
        assert_eq!(
            detect_family(&img, &[port("output0", &[1, 6, 8400])], 2).unwrap(),
            ModelFamilyKind::Yolo8
        );
        assert_eq!(
            detect_family(&img, &[port("output0", &[1, 300, 6])], 0).unwrap(),
            ModelFamilyKind::Yolo26
        );
        assert_eq!(
            detect_family(
                &[
                    port("images", &[1, 3, 640, 640]),
                    port("orig_target_sizes", &[1, 2])
                ],
                &[
                    port("labels", &[1, 300]),
                    port("boxes", &[1, 300, 4]),
                    port("scores", &[1, 300])
                ],
                80
            )
            .unwrap(),
            ModelFamilyKind::RtDetr
        );
    }

    #[test]
    fn detr_family_detection() {
        let dfine_out = [
            port("logits", &[-1, 300, 80]),
            port("pred_boxes", &[-1, 300, 4]),
        ];
        let rf_out = [
            port("pred_boxes", &[-1, 300, 4]),
            port("logits", &[-1, 300, 91]),
        ];
        let pix = [port("pixel_values", &[-1, 3, -1, -1])];
        for out in [&dfine_out[..], &rf_out[..]] {
            assert_eq!(detect_family(&pix, out, 80).unwrap(), ModelFamilyKind::Detr);
        }
        // Normalization: on for the 91-class (RF-DETR) head or when forced.
        let fam = |k, out: &[PortSpec]| make_family(k, &pix, out, 80).unwrap();
        assert_eq!(fam(ModelFamilyKind::Auto, &dfine_out).normalization(), None);
        assert_eq!(
            fam(ModelFamilyKind::Auto, &dfine_out).kind(),
            ModelFamilyKind::Detr
        );
        let rf = fam(ModelFamilyKind::Auto, &rf_out);
        assert_eq!(rf.normalization(), Some(Normalization::IMAGENET));
        assert_eq!(rf.kind(), ModelFamilyKind::RfDetr);
        assert_eq!(rf.resize_mode(), ResizeMode::Stretch);
        assert_eq!(
            fam(ModelFamilyKind::RfDetr, &dfine_out).normalization(),
            Some(Normalization::IMAGENET)
        );
        // Not DETR: missing `pred_boxes`, or several inputs without `pixel_values`.
        assert!(!detr::is_detr(&pix, &dfine_out[..1]));
        let two = [port("a", &[1, 3, 640, 640]), port("b", &[1, 2])];
        assert!(!detr::is_detr(&two, &dfine_out));
        let k: ModelFamilyKind = serde_json::from_str(r#""rfdetr""#).unwrap();
        assert_eq!(k, ModelFamilyKind::RfDetr);
        let k: ModelFamilyKind = serde_json::from_str(r#""detr""#).unwrap();
        assert_eq!(
            (k, k.to_string()),
            (ModelFamilyKind::Detr, "detr".to_string())
        );
        assert_eq!(ModelFamilyKind::RfDetr.to_string(), "rfdetr");
    }

    #[test]
    fn family_from_str_round_trips() {
        for k in ModelFamilyKind::ALL {
            assert_eq!(k.to_string().parse::<ModelFamilyKind>().unwrap(), k);
            let json = serde_json::to_string(&k).unwrap();
            assert_eq!(json, format!("\"{k}\""));
            assert!(!k.description().is_empty());
        }
        let err = "bogus".parse::<ModelFamilyKind>().unwrap_err().to_string();
        assert!(
            err.contains("bogus") && err.contains("auto, yolo26, yolo5"),
            "{err}"
        );
        assert!("YOLO5".parse::<ModelFamilyKind>().is_err());
    }

    #[test]
    fn dynamic_input_sizes() {
        use std::path::Path;
        let none: [PortSpec; 0] = [];
        let rf_out = [
            port("pred_boxes", &[-1, 300, 4]),
            port("logits", &[-1, 300, 91]),
        ];
        // A catalog hint wins over everything else.
        assert_eq!(
            dynamic_input_size(Path::new("models/rfdetr-nano.onnx"), &none, Some(384)),
            (384, 384)
        );
        assert_eq!(
            dynamic_input_size(Path::new("x/RFDETR-MEDIUM.onnx"), &rf_out, Some(576)),
            (576, 576)
        );
        // Unknown RF-DETR-like model: base size; anything else: 640.
        assert_eq!(
            dynamic_input_size(Path::new("m/custom.onnx"), &rf_out, None),
            (560, 560)
        );
        assert_eq!(
            dynamic_input_size(Path::new("m/custom.onnx"), &none, None),
            (640, 640)
        );
        // Hugging Face repo layout.
        let repo = std::env::temp_dir().join(format!("bo-hf-repo-{}", std::process::id()));
        std::fs::create_dir_all(repo.join("onnx")).unwrap();
        std::fs::write(
            repo.join("preprocessor_config.json"),
            r#"{"do_normalize":false,"size":{"height":512,"width":448}}"#,
        )
        .unwrap();
        let model = repo.join("onnx").join("model.onnx");
        assert_eq!(dynamic_input_size(&model, &rf_out, None), (448, 512));
        // Only next to an `onnx/` directory.
        assert_eq!(
            dynamic_input_size(&repo.join("model.onnx"), &none, None),
            (640, 640)
        );
        std::fs::remove_dir_all(&repo).ok();

        assert_eq!(
            parse_hf_preprocessor_size(r#"{"size":{"shortest_edge":800}}"#),
            Some((800, 800))
        );
        assert_eq!(
            parse_hf_preprocessor_size(r#"{"size":{"height":0,"width":5}}"#),
            None
        );
        assert_eq!(parse_hf_preprocessor_size("{}"), None);
        assert_eq!(parse_hf_preprocessor_size("not json"), None);
    }

    #[test]
    fn output_buffers_are_reused() {
        let mut b = OutputBuf::default();
        b.f32_mut().extend_from_slice(&[1.0, 2.0, 3.0]);
        let cap = match &b {
            OutputBuf::F32(v) => v.capacity(),
            _ => unreachable!(),
        };
        let v = b.f32_mut();
        assert!(
            v.is_empty() && v.capacity() == cap,
            "same allocation, cleared"
        );
        v.push(4.0);
        assert_eq!(b, OutputBuf::F32(vec![4.0]));
        b.i32_mut().extend_from_slice(&[7, -1]);
        assert_eq!(b.len(), 2);
        let out = NamedOutput {
            name: "labels".into(),
            shape: vec![1, 2],
            data: b,
        };
        let mut scratch = Vec::new();
        assert_eq!(out.as_i64(&mut scratch), Some(&[7i64, -1][..]));
        let out = NamedOutput {
            data: OutputBuf::I64(vec![5]),
            ..out
        };
        assert_eq!(out.as_i64(&mut scratch), Some(&[5i64][..]));
        let f = NamedOutput::default();
        assert!(f.as_i64(&mut scratch).is_none() && f.data.is_empty());
        let mut outs = vec![NamedOutput::default(); 3];
        NamedOutput::resize_slots(&mut outs, 1);
        assert_eq!(outs.len(), 1);
    }

    #[test]
    fn letterbox_unmapping() {
        // 1920x1080 into 640x640: scale 1/3, pad_y = (640-360)/2 = 140
        let ctx = PreprocessCtx {
            orig_w: 1920,
            orig_h: 1080,
            input_w: 640,
            input_h: 640,
            mode: ResizeMode::Letterbox,
            scale: 1.0 / 3.0,
            pad_x: 0.0,
            pad_y: 140.0,
        };
        let (x1, y1, x2, y2) = ctx.to_original(0.0, 140.0, 640.0, 500.0);
        assert!((x1 - 0.0).abs() < 1e-3 && (y1 - 0.0).abs() < 1e-3);
        assert!((x2 - 1920.0).abs() < 1e-2 && (y2 - 1080.0).abs() < 1e-2);
    }

    #[test]
    fn predictions_filter_and_clamp() {
        let ctx = PreprocessCtx {
            orig_w: 100,
            orig_h: 50,
            input_w: 640,
            input_h: 640,
            mode: ResizeMode::Stretch,
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        };
        let dets = vec![
            Detection {
                x1: -5.0,
                y1: 1.0,
                x2: 20.0,
                y2: 80.0,
                score: 0.9,
                class_id: 0,
            },
            Detection {
                x1: 1.0,
                y1: 1.0,
                x2: 2.0,
                y2: 2.0,
                score: 0.8,
                class_id: 1,
            },
        ];
        let names = vec!["person".to_string(), "car".to_string()];
        let filt = vec!["Person".to_string()];
        let p = to_predictions(&dets, &ctx, &names, Some(&filt));
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].label, "person");
        assert_eq!((p[0].x_min, p[0].y_max), (0, 50));
        assert_eq!(to_predictions(&dets, &ctx, &names, None).len(), 2);
    }

    #[test]
    fn predictions_drop_non_finite_and_clamp_score() {
        let ctx = PreprocessCtx {
            orig_w: 100,
            orig_h: 50,
            input_w: 640,
            input_h: 640,
            mode: ResizeMode::Stretch,
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        };
        let d = |x1, y1, x2, y2, score| Detection {
            x1,
            y1,
            x2,
            y2,
            score,
            class_id: 0,
        };
        let dets = vec![
            d(f32::NAN, 0.0, 10.0, 10.0, 0.9),
            d(0.0, f32::INFINITY, 10.0, 10.0, 0.9),
            d(0.0, 0.0, f32::NEG_INFINITY, 10.0, 0.9),
            d(0.0, 0.0, 10.0, f32::NAN, 0.9),
            d(0.0, 0.0, 10.0, 10.0, f32::NAN),
            d(0.0, 0.0, 10.0, 10.0, f32::INFINITY),
            d(0.0, 0.0, 10.0, 10.0, 1.5),
            d(-1e30, -1e30, 1e30, 1e30, -0.5),
        ];
        let p = to_predictions(&dets, &ctx, &["a".to_string()], None);
        assert_eq!(p.len(), 2);
        assert_eq!((p[0].x_max, p[0].y_max, p[0].confidence), (10, 10, 1.0));
        assert_eq!((p[1].x_max, p[1].y_max, p[1].confidence), (100, 50, 0.0));
        let json = serde_json::to_string(&p).unwrap();
        assert!(!json.contains("null"), "{json}");
        assert_eq!(class_index(2.4), Some(2));
        assert_eq!(class_index(-0.4), Some(0));
        assert_eq!(class_index(-1.0), None);
        assert_eq!(class_index(f32::NAN), None);
        assert_eq!(class_index(f32::INFINITY), None);
        assert_eq!(class_index(1e9), None);
        let mut v = dets.clone();
        retain_finite(&mut v);
        assert_eq!(v.len(), 2);
    }
}
