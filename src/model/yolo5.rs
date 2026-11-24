//! YOLOv5 (incl. MikeLud ipcam models): one output `[1, N, 5+C]` with rows
//! `cx, cy, w, h, obj, cls_0..cls_C` in model-input pixels. Needs NMS.

use anyhow::Result;

use super::nms::{MAX_DETS, MAX_NMS_CANDIDATES, nms_capped};
use super::yolo26::{matrix_dims, meets, primary_f32_output};
use super::{
    Detection, Family, ModelFamilyKind, NamedOutput, PostParams, PreprocessCtx, ResizeMode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Yolo5 {
    pub num_classes: usize,
}

impl Yolo5 {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl Family for Yolo5 {
    fn kind(&self) -> ModelFamilyKind {
        ModelFamilyKind::Yolo5
    }
    fn resize_mode(&self) -> ResizeMode {
        ResizeMode::Letterbox
    }
    fn postprocess(
        &self,
        outputs: &[NamedOutput],
        ctx: &PreprocessCtx,
        params: &PostParams,
    ) -> Result<Vec<Detection>> {
        let (out, data) = primary_f32_output("yolo5", outputs)?;
        let (_, cols) = matrix_dims("yolo5", "[1, N, 5+C]", out, data.len())?;
        if cols < 6 {
            anyhow::bail!(
                "yolo5: output `{}` has shape {:?}, expected [1, N, 5+C] with C >= 1",
                out.name,
                out.shape
            );
        }
        if self.num_classes > 0 && cols != 5 + self.num_classes {
            anyhow::bail!(
                "yolo5: output `{}` has shape {:?}, expected [1, N, {}] for {} classes; the \
                 classes file lists another number of classes than the model was trained with",
                out.name,
                out.shape,
                5 + self.num_classes,
                self.num_classes
            );
        }
        let thr = params.confidence_threshold;
        let mut dets = Vec::new();
        for row in data.chunks_exact(cols) {
            // `cols >= 6` was checked above, so every row has the 5 box fields and >= 1 class.
            let &[cx, cy, w, h, obj, ref class_scores @ ..] = row else {
                continue;
            };
            // Early out: conf = obj * max_cls <= obj when scores are in 0..1.
            if !meets(obj, thr) {
                continue;
            }
            let (cls, best) = argmax(class_scores);
            let conf = obj * best;
            // `inf * x` and `NaN` never pass: the score must be a finite number.
            if !conf.is_finite() || !meets(conf, thr) {
                continue;
            }
            let (x1, y1, x2, y2) =
                ctx.to_original(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0);
            dets.push(Detection {
                x1,
                y1,
                x2,
                y2,
                score: conf,
                class_id: cls,
            });
        }
        nms_capped(&mut dets, params.nms_iou, MAX_NMS_CANDIDATES, MAX_DETS);
        Ok(dets)
    }
}

/// Index and value of the maximum (first wins on ties; NaN never wins).
pub(crate) fn argmax(xs: &[f32]) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in xs.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::OutputBuf;

    // QA-043: `family: auto` now picks Yolo5 for a static `[1, 25200, 6]` output whatever the class
    // count; a 1-class export given the 80 COCO names (no classes file) must fail with a message
    // that names the cause, never decode as boxes.
    #[test]
    fn six_columns_with_80_class_names_fail_naming_the_classes_file() {
        let out = NamedOutput {
            name: "output".into(),
            shape: vec![1, 25200, 6],
            data: OutputBuf::F32(vec![0.0; 25200 * 6]),
        };
        let ctx = PreprocessCtx {
            orig_w: 640,
            orig_h: 640,
            input_w: 640,
            input_h: 640,
            mode: ResizeMode::Letterbox,
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        };
        let params = PostParams {
            confidence_threshold: 0.5,
            nms_iou: 0.45,
        };
        let err = Yolo5::new(80)
            .postprocess(&[out], &ctx, &params)
            .expect_err("85 columns expected for 80 classes");
        assert!(
            err.to_string()
                .contains("classes file lists another number of classes"),
            "{err}"
        );
    }
}
