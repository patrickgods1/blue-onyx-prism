//! Ultralytics YOLO26 exported end-to-end: one output `[1, N, 6]` with rows
//! `x1, y1, x2, y2, conf, cls` in model-input pixels. NMS-free.

use anyhow::Result;

use super::output::{matrix_dims, meets, primary_f32_output};
use super::{
    Detection, Family, ModelFamilyKind, NamedOutput, PostParams, PreprocessCtx, ResizeMode,
    class_index, retain_finite,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Yolo26 {
    pub num_classes: usize,
}

impl Yolo26 {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl Family for Yolo26 {
    fn kind(&self) -> ModelFamilyKind {
        ModelFamilyKind::Yolo26
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
        let (out, data) = primary_f32_output("yolo26", outputs)?;
        let (_, cols) = matrix_dims("yolo26", "[1, N, 6]", out, data.len())?;
        if cols != 6 {
            anyhow::bail!(
                "yolo26: output `{}` has shape {:?}, expected [1, N, 6]",
                out.name,
                out.shape
            );
        }
        let mut dets = Vec::new();
        for &[bx1, by1, bx2, by2, conf, cls] in data.as_chunks::<6>().0 {
            if !meets(conf, params.confidence_threshold) {
                continue;
            }
            let Some(class_id) = class_index(cls) else {
                continue;
            };
            let (x1, y1, x2, y2) = ctx.to_original(bx1, by1, bx2, by2);
            dets.push(Detection {
                x1,
                y1,
                x2,
                y2,
                score: conf,
                class_id,
            });
        }
        retain_finite(&mut dets);
        Ok(dets)
    }
}
