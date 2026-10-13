//! RT-DETR(v2) ONNX: inputs `images` + i64 `orig_target_sizes` `[1, 2]`; outputs
//! `labels` `[1, N]` (i64/i32), `boxes` `[1, N, 4]` (xyxy, f32), `scores` `[1, N]` (f32).
//! NMS-free.
//!
//! We feed the *input* size as `orig_target_sizes`, so boxes come back in model-input pixels
//! and `PreprocessCtx::to_original` (stretch) maps them to the original image.

use anyhow::Result;

use super::yolo26::meets;
use super::{
    Detection, ExtraData, ExtraInput, Family, ModelFamilyKind, NamedOutput, PostParams,
    PreprocessCtx, ResizeMode, retain_finite,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtDetr {
    pub num_classes: usize,
}

impl RtDetr {
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

fn find<'a>(outputs: &'a [NamedOutput], name: &str) -> Result<&'a NamedOutput> {
    outputs.iter().find(|o| o.name == name).ok_or_else(|| {
        anyhow::anyhow!(
            "rtdetr: missing output `{name}`; got {:?}",
            outputs
                .iter()
                .map(|o| (&o.name, &o.shape))
                .collect::<Vec<_>>()
        )
    })
}

/// Number of queries for `[1, N, ..tail]` or `[N, ..tail]`.
fn queries(out: &NamedOutput, tail: &[usize], expected: &str) -> Result<usize> {
    match out.shape.as_slice() {
        [1, n, rest @ ..] if rest == tail => Ok(*n),
        [n, rest @ ..] if rest == tail => Ok(*n),
        _ => anyhow::bail!(
            "rtdetr: output `{}` has shape {:?}, expected {expected}",
            out.name,
            out.shape
        ),
    }
}

impl Family for RtDetr {
    fn kind(&self) -> ModelFamilyKind {
        ModelFamilyKind::RtDetr
    }
    fn resize_mode(&self) -> ResizeMode {
        ResizeMode::Stretch
    }
    fn extra_inputs(&self, ctx: &PreprocessCtx) -> Vec<ExtraInput> {
        vec![ExtraInput {
            name: "orig_target_sizes".into(),
            shape: vec![1, 2],
            data: ExtraData::I64(vec![i64::from(ctx.input_w), i64::from(ctx.input_h)]),
        }]
    }
    fn postprocess(
        &self,
        outputs: &[NamedOutput],
        ctx: &PreprocessCtx,
        params: &PostParams,
    ) -> Result<Vec<Detection>> {
        let labels_o = find(outputs, "labels")?;
        let boxes_o = find(outputs, "boxes")?;
        let scores_o = find(outputs, "scores")?;

        let n_labels = queries(labels_o, &[], "[1, N]")?;
        let n_boxes = queries(boxes_o, &[4], "[1, N, 4]")?;
        let n_scores = queries(scores_o, &[], "[1, N]")?;
        if n_labels != n_boxes || n_labels != n_scores {
            anyhow::bail!(
                "rtdetr: inconsistent query counts: labels {:?}, boxes {:?}, scores {:?}",
                labels_o.shape,
                boxes_o.shape,
                scores_o.shape
            );
        }
        let n = n_labels;

        let mut scratch = Vec::new();
        let labels = labels_o.as_i64(&mut scratch).ok_or_else(|| {
            anyhow::anyhow!("rtdetr: output `labels` must be i64 or i32, got f32")
        })?;
        let boxes = boxes_o
            .as_f32()
            .ok_or_else(|| anyhow::anyhow!("rtdetr: output `boxes` must be f32"))?;
        let scores = scores_o
            .as_f32()
            .ok_or_else(|| anyhow::anyhow!("rtdetr: output `scores` must be f32"))?;
        if labels.len() != n || Some(boxes.len()) != n.checked_mul(4) || scores.len() != n {
            anyhow::bail!(
                "rtdetr: buffer lengths (labels {}, boxes {}, scores {}) do not match shapes {:?} {:?} {:?}",
                labels.len(),
                boxes.len(),
                scores.len(),
                labels_o.shape,
                boxes_o.shape,
                scores_o.shape
            );
        }

        let mut dets = Vec::new();
        let rows = scores.iter().zip(labels).zip(boxes.as_chunks::<4>().0);
        for ((&score, &label), &[bx1, by1, bx2, by2]) in rows {
            if !meets(score, params.confidence_threshold) {
                continue;
            }
            // Negative labels (padding) and absurd ones are skipped.
            let Some(class_id) = usize::try_from(label).ok().filter(|&c| c <= 1_000_000) else {
                continue;
            };
            let (x1, y1, x2, y2) = ctx.to_original(bx1, by1, bx2, by2);
            dets.push(Detection {
                x1,
                y1,
                x2,
                y2,
                score,
                class_id,
            });
        }
        retain_finite(&mut dets);
        Ok(dets)
    }
}
