//! YOLOv8/11 raw head: one output `[1, 4+C, A]`, channel-major (row 0..4 = cx, cy, w, h;
//! rows 4.. = class scores) in model-input pixels. Needs NMS.

use anyhow::Result;

use super::nms::{MAX_DETS, MAX_NMS_CANDIDATES, nms_capped};
use super::yolo26::{matrix_dims, meets, primary_f32_output};
use super::{
    Detection, Family, ModelFamilyKind, NamedOutput, PostParams, PreprocessCtx, ResizeMode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Yolo8 {
    pub num_classes: usize,
}

impl Yolo8 {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl Family for Yolo8 {
    fn kind(&self) -> ModelFamilyKind {
        ModelFamilyKind::Yolo8
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
        let (out, data) = primary_f32_output("yolo8", outputs)?;
        let (channels, anchors) = matrix_dims("yolo8", "[1, 4+C, A]", out, data.len())?;
        if channels < 5 {
            anyhow::bail!(
                "yolo8: output `{}` has shape {:?}, expected [1, 4+C, A] with C >= 1",
                out.name,
                out.shape
            );
        }
        if self.num_classes > 0 && channels != 4 + self.num_classes {
            anyhow::bail!(
                "yolo8: output `{}` has shape {:?}, expected [1, {}, A] for {} classes; the \
                 classes file lists another number of classes than the model was trained with",
                out.name,
                out.shape,
                4 + self.num_classes,
                self.num_classes
            );
        }
        let num_classes = channels - 4;
        let thr = params.confidence_threshold;

        // Pass 1: per-anchor best (score, class), walking the class rows contiguously.
        let mut best = vec![(f32::NEG_INFINITY, 0usize); anchors];
        for (c, row) in data
            .chunks_exact(anchors)
            .skip(4)
            .take(num_classes)
            .enumerate()
        {
            for (b, &v) in best.iter_mut().zip(row) {
                if v > b.0 {
                    *b = (v, c);
                }
            }
        }

        let mut planes = data.chunks_exact(anchors);
        let (Some(cxs), Some(cys), Some(ws), Some(hs)) =
            (planes.next(), planes.next(), planes.next(), planes.next())
        else {
            anyhow::bail!(
                "yolo8: output `{}` shape {:?} has fewer than 4 box rows",
                out.name,
                out.shape
            );
        };
        let mut dets = Vec::new();
        let boxes = cxs.iter().zip(cys).zip(ws).zip(hs).zip(&best);
        for ((((&cx, &cy), &w), &h), &(conf, class_id)) in boxes {
            if !meets(conf, thr) {
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
                class_id,
            });
        }
        nms_capped(&mut dets, params.nms_iou, MAX_NMS_CANDIDATES, MAX_DETS);
        Ok(dets)
    }
}
