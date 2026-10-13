//! Image preprocessing: fit an RGB8 image into the model input (letterbox or stretch) and
//! convert it to a CHW `f32` tensor in `0..1`, optionally mean/std normalized per channel.

use anyhow::{Context, Result, bail};
use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};

use super::{Family, Normalization, PreprocessCtx, ResizeMode};

/// Gray value used for letterbox padding (Ultralytics convention).
pub const PAD_VALUE: u8 = 114;

/// Largest accepted model input side, in pixels.
pub const MAX_INPUT_SIDE: u32 = 8192;
/// Largest accepted input tensor (`3 * w * h` f32 values), in bytes: 256 MiB.
pub const MAX_INPUT_TENSOR_BYTES: u64 = 256 << 20;

/// Reject a model input size that is zero, wider/taller than [`MAX_INPUT_SIDE`] or whose f32
/// CHW tensor would exceed [`MAX_INPUT_TENSOR_BYTES`]. The size comes from the model file, so a
/// broken or hostile export must fail the model instead of aborting the process on allocation.
pub fn check_input_size(input_w: u32, input_h: u32) -> Result<()> {
    if input_w == 0 || input_h == 0 {
        bail!("invalid model input size {input_w}x{input_h}");
    }
    if input_w > MAX_INPUT_SIDE || input_h > MAX_INPUT_SIDE {
        bail!("model input size {input_w}x{input_h} exceeds the {MAX_INPUT_SIDE} pixel side limit");
    }
    let bytes = u64::from(input_w) * u64::from(input_h) * 3 * 4;
    if bytes > MAX_INPUT_TENSOR_BYTES {
        bail!(
            "model input size {input_w}x{input_h} needs a {bytes} byte tensor, over the \
             {MAX_INPUT_TENSOR_BYTES} byte limit"
        );
    }
    Ok(())
}

/// Reusable preprocessor. Buffers (resized image, output tensor, resizer scratch) are kept
/// across calls so steady-state operation does not allocate. Nothing is allocated until the
/// first [`Self::run`]; an input size that fails [`check_input_size`] makes `run` return an
/// error (use [`Self::try_new`] / [`Self::try_for_family`] to reject it at load time).
pub struct Preprocessor {
    input_w: u32,
    input_h: u32,
    mode: ResizeMode,
    resizer: Resizer,
    options: ResizeOptions,
    /// Resized HWC RGB8 content (only the content area, not the padding).
    resized: Vec<u8>,
    /// CHW f32 output tensor, `3 * input_w * input_h`.
    tensor: Vec<f32>,
    /// Per channel (R, G, B): `lut[c][v] == v as f32 / 255.0`, or with normalization
    /// `(v / 255 - mean[c]) / std[c]` (computed in f64, rounded once).
    lut: [[f32; 256]; 3],
    normalization: Option<Normalization>,
}

impl std::fmt::Debug for Preprocessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Preprocessor")
            .field("input_w", &self.input_w)
            .field("input_h", &self.input_h)
            .field("mode", &self.mode)
            .field("normalization", &self.normalization)
            .finish_non_exhaustive()
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "f64 -> f32 of values in roughly -3..3: only precision is lost, by design"
)]
fn build_lut(norm: Option<Normalization>) -> [[f32; 256]; 3] {
    let mut lut = [[0f32; 256]; 3];
    let stats = norm.map(|n| {
        n.mean
            .iter()
            .zip(&n.std)
            .map(|(&m, &s)| (f64::from(m), f64::from(s)))
            .collect::<Vec<_>>()
    });
    for (c, table) in lut.iter_mut().enumerate() {
        let stats = stats.as_ref().and_then(|v| v.get(c).copied());
        for (i, v) in table.iter_mut().enumerate() {
            // `i` is 0..256, exact in f64 (and f32).
            let x = i as f64 / 255.0;
            *v = match stats {
                None => x as f32,
                Some((mean, std)) => ((x - mean) / std) as f32,
            };
        }
    }
    lut
}

impl Preprocessor {
    pub fn new(input_w: u32, input_h: u32, mode: ResizeMode) -> Self {
        Self {
            input_w,
            input_h,
            mode,
            resizer: Resizer::new(),
            options: ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear)),
            resized: Vec::new(),
            tensor: Vec::new(),
            lut: build_lut(None),
            normalization: None,
        }
    }

    /// [`Self::new`] after [`check_input_size`].
    pub fn try_new(input_w: u32, input_h: u32, mode: ResizeMode) -> Result<Self> {
        check_input_size(input_w, input_h)?;
        Ok(Self::new(input_w, input_h, mode))
    }

    /// Preprocessor for `family`: its resize mode and normalization.
    pub fn for_family(input_w: u32, input_h: u32, family: &dyn Family) -> Self {
        Self::new(input_w, input_h, family.resize_mode()).with_normalization(family.normalization())
    }

    /// [`Self::for_family`] after [`check_input_size`].
    pub fn try_for_family(input_w: u32, input_h: u32, family: &dyn Family) -> Result<Self> {
        check_input_size(input_w, input_h)?;
        Ok(Self::for_family(input_w, input_h, family))
    }

    /// Normalize each channel as `(v / 255 - mean) / std` (None = plain `0..1`).
    pub fn with_normalization(mut self, normalization: Option<Normalization>) -> Self {
        self.lut = build_lut(normalization);
        self.normalization = normalization;
        self
    }

    pub fn normalization(&self) -> Option<Normalization> {
        self.normalization
    }

    /// Model input size `(w, h)`.
    pub fn input_size(&self) -> (u32, u32) {
        (self.input_w, self.input_h)
    }

    pub fn mode(&self) -> ResizeMode {
        self.mode
    }

    /// `rgb` is tightly packed RGB8, `w*h*3` bytes. Returns CHW f32 in 0..1 (or normalized)
    /// (length `3*input_w*input_h`) and the geometry needed to map boxes back.
    pub fn run(&mut self, rgb: &[u8], w: u32, h: u32) -> Result<(&[f32], PreprocessCtx)> {
        let mut tensor = std::mem::take(&mut self.tensor);
        let r = self.run_into(rgb, w, h, &mut tensor);
        self.tensor = tensor;
        r.map(|ctx| (&self.tensor[..], ctx))
    }

    /// [`Self::run`] into a caller-owned tensor buffer (resized to `3*input_w*input_h`, its
    /// allocation reused), e.g. a pooled buffer that outlives this call.
    pub fn run_into(
        &mut self,
        rgb: &[u8],
        w: u32,
        h: u32,
        out: &mut Vec<f32>,
    ) -> Result<PreprocessCtx> {
        if w == 0 || h == 0 {
            bail!("cannot preprocess an empty image ({w}x{h})");
        }
        check_input_size(self.input_w, self.input_h)?;
        let expected = (w as usize)
            .checked_mul(h as usize)
            .and_then(|p| p.checked_mul(3))
            .ok_or_else(|| anyhow::anyhow!("image {w}x{h} is too large"))?;
        if rgb.len() != expected {
            bail!(
                "RGB buffer has {} bytes, expected {expected} for {w}x{h}x3",
                rgb.len()
            );
        }
        let (iw, ih) = (self.input_w, self.input_h);

        let (new_w, new_h, scale, pad_x, pad_y) = match self.mode {
            ResizeMode::Letterbox => {
                let scale = (f64::from(iw) / f64::from(w)).min(f64::from(ih) / f64::from(h));
                let new_w = fit_side(f64::from(w) * scale, iw);
                let new_h = fit_side(f64::from(h) * scale, ih);
                let pad_x = (iw - new_w) / 2;
                let pad_y = (ih - new_h) / 2;
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "scale is a ratio of two u32 sides; f32 precision is enough for box mapping"
                )]
                let scale = scale as f32;
                (new_w, new_h, scale, pad_x, pad_y)
            }
            ResizeMode::Stretch => (iw, ih, 1.0f32, 0, 0),
        };

        // Resize (or pass through) to the content size. `new_w <= iw`, `new_h <= ih`, so the
        // buffer is bounded by the (checked) input tensor size.
        let content: &[u8] = if new_w == w && new_h == h {
            rgb
        } else {
            let len = new_w as usize * new_h as usize * 3;
            self.resized.resize(len, 0);
            let src = ImageRef::new(w, h, rgb, PixelType::U8x3).context("source image view")?;
            let dst_buf = self
                .resized
                .get_mut(..len)
                .ok_or_else(|| anyhow::anyhow!("resize buffer too small"))?;
            let mut dst = Image::from_slice_u8(new_w, new_h, dst_buf, PixelType::U8x3)
                .context("resize destination")?;
            self.resizer
                .resize(&src, &mut dst, &self.options)
                .context("resizing image")?;
            self.resized
                .get(..len)
                .ok_or_else(|| anyhow::anyhow!("resize buffer too small"))?
        };

        let plane = iw as usize * ih as usize;
        out.resize(3 * plane, 0.0);
        if new_w != iw || new_h != ih {
            for (p, lut) in out.chunks_exact_mut(plane).zip(&self.lut) {
                p.fill(lut.get(usize::from(PAD_VALUE)).copied().unwrap_or(0.0));
            }
        }

        // HWC u8 -> CHW f32 in one pass over the content rows.
        let (r_plane, rest) = out.split_at_mut(plane);
        let (g_plane, b_plane) = rest.split_at_mut(plane);
        let row_bytes = new_w as usize * 3;
        let [lr, lg, lb] = &self.lut;
        for (y, src_row) in content.chunks_exact(row_bytes).enumerate() {
            let start = (y + pad_y as usize) * iw as usize + pad_x as usize;
            let end = start + new_w as usize;
            let (Some(r), Some(g), Some(b)) = (
                r_plane.get_mut(start..end),
                g_plane.get_mut(start..end),
                b_plane.get_mut(start..end),
            ) else {
                bail!("letterbox row {y} ({start}..{end}) falls outside the {iw}x{ih} input");
            };
            let px = src_row.as_chunks::<3>().0.iter();
            #[allow(clippy::indexing_slicing, reason = "u8 values index 256-entry tables")]
            for (((&[pr, pg, pb], r), g), b) in
                px.zip(r.iter_mut()).zip(g.iter_mut()).zip(b.iter_mut())
            {
                *r = lr[usize::from(pr)];
                *g = lg[usize::from(pg)];
                *b = lb[usize::from(pb)];
            }
        }

        let ctx = PreprocessCtx {
            orig_w: w,
            orig_h: h,
            input_w: iw,
            input_h: ih,
            mode: self.mode,
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
        };
        Ok(ctx)
    }
}

/// `side.round()` clamped into `1..=max` (so it fits `u32`).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "clamped to 1..=max (a u32) before the cast; NaN clamps to NaN and `as` maps it to 0, then max(1)"
)]
fn fit_side(side: f64, max: u32) -> u32 {
    (side.round().clamp(1.0, f64::from(max)) as u32).clamp(1, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, c: [u8; 3]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..w * h {
            v.extend_from_slice(&c);
        }
        v
    }

    #[test]
    fn letterbox_1080p() {
        let color = [200u8, 50, 10];
        let img = solid(1920, 1080, color);
        let mut p = Preprocessor::new(640, 640, ResizeMode::Letterbox);
        let (t, ctx) = p.run(&img, 1920, 1080).unwrap();
        assert_eq!(t.len(), 3 * 640 * 640);
        assert_eq!(ctx.pad_x, 0.0);
        assert_eq!(ctx.pad_y, 140.0);
        assert!((ctx.scale - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(
            (ctx.orig_w, ctx.orig_h, ctx.input_w, ctx.input_h),
            (1920, 1080, 640, 640)
        );
        let plane = 640 * 640;
        let pad = 114.0 / 255.0;
        for (c, &cv) in color.iter().enumerate() {
            let want = cv as f32 / 255.0;
            for y in 0..640 {
                for x in 0..640 {
                    let v = t[c * plane + y * 640 + x];
                    if !(140..500).contains(&y) {
                        assert_eq!(v, pad, "pad c={c} y={y} x={x}");
                    } else {
                        assert_eq!(v, want, "content c={c} y={y} x={x}");
                    }
                }
            }
        }
    }

    #[test]
    fn passthrough_same_size() {
        let img: Vec<u8> = (0..640 * 640 * 3).map(|i| (i * 7 % 251) as u8).collect();
        let mut p = Preprocessor::new(640, 640, ResizeMode::Letterbox);
        let (t, ctx) = p.run(&img, 640, 640).unwrap();
        assert_eq!((ctx.scale, ctx.pad_x, ctx.pad_y), (1.0, 0.0, 0.0));
        let plane = 640 * 640;
        for (i, px) in img.as_chunks::<3>().0.iter().enumerate() {
            for (c, &v) in px.iter().enumerate() {
                assert_eq!(t[c * plane + i], v as f32 / 255.0);
            }
        }
    }

    #[test]
    fn stretch_fills_tensor() {
        let color = [10u8, 20, 30];
        let img = solid(100, 50, color);
        let mut p = Preprocessor::new(640, 640, ResizeMode::Stretch);
        let (t, ctx) = p.run(&img, 100, 50).unwrap();
        assert_eq!((ctx.scale, ctx.pad_x, ctx.pad_y), (1.0, 0.0, 0.0));
        assert_eq!(ctx.mode, ResizeMode::Stretch);
        assert_eq!((ctx.orig_w, ctx.orig_h), (100, 50));
        let plane = 640 * 640;
        for (c, &cv) in color.iter().enumerate() {
            let want = cv as f32 / 255.0;
            for &v in &t[c * plane..(c + 1) * plane] {
                assert_eq!(v, want);
            }
        }
    }

    #[test]
    fn imagenet_normalization_per_channel() {
        let n = Normalization::IMAGENET;
        let color = [255u8, 0, 128];
        let img = solid(8, 4, color);
        let mut p = Preprocessor::new(16, 16, ResizeMode::Letterbox).with_normalization(Some(n));
        assert_eq!(p.normalization(), Some(n));
        let (t, ctx) = p.run(&img, 8, 4).unwrap();
        assert_eq!((ctx.pad_x, ctx.pad_y), (0.0, 4.0));
        let plane = 16 * 16;
        let want =
            |c: usize, v: u8| ((v as f64 / 255.0 - n.mean[c] as f64) / n.std[c] as f64) as f32;
        for c in 0..3 {
            // Padding row 0 is normalized gray, content row 4 the normalized color.
            assert_eq!(t[c * plane], want(c, PAD_VALUE));
            assert_eq!(t[c * plane + 4 * 16 + 3], want(c, color[c]));
        }
        assert!((t[0] - (1.0 / 255.0 * 114.0 - 0.485) / 0.229).abs() < 1e-6);
        assert!((t[4 * 16] - (1.0 - 0.485) / 0.229).abs() < 1e-6);
        assert!((t[plane + 4 * 16] - (0.0 - 0.456) / 0.224).abs() < 1e-6);
        // Back to plain 0..1.
        let mut p = p.with_normalization(None);
        let (t, _) = p.run(&img, 8, 4).unwrap();
        assert_eq!(t[4 * 16], 1.0);
        assert_eq!(t[2 * plane + 4 * 16], 128.0 / 255.0);
    }

    #[test]
    fn buffers_reused_and_errors() {
        let mut p = Preprocessor::new(64, 64, ResizeMode::Letterbox);
        let a = solid(128, 32, [255, 255, 255]);
        let ptr1 = p.run(&a, 128, 32).unwrap().0.as_ptr();
        let b = solid(32, 128, [0, 0, 0]);
        let (t, ctx) = p.run(&b, 32, 128).unwrap();
        assert_eq!(t.as_ptr(), ptr1);
        assert_eq!((ctx.pad_x, ctx.pad_y), (24.0, 0.0));
        assert_eq!(ctx.scale, 0.5);
        // Left padding column is gray, content (x 24..40) is black.
        assert_eq!(t[0], 114.0 / 255.0);
        assert_eq!(t[30], 0.0);
        assert_eq!(t[40], 114.0 / 255.0);
        assert!(p.run(&b, 33, 128).is_err());
        assert!(p.run(&[], 0, 0).is_err());
    }

    #[test]
    fn input_size_limits() {
        assert!(check_input_size(640, 640).is_ok());
        assert!(check_input_size(4096, 4096).is_ok());
        assert!(check_input_size(0, 640).is_err());
        let e = check_input_size(8193, 32).unwrap_err().to_string();
        assert!(e.contains("8193x32") && e.contains("8192"), "{e}");
        let e = check_input_size(8192, 8192).unwrap_err().to_string();
        assert!(e.contains("byte"), "{e}");
        assert!(Preprocessor::try_new(16384, 16384, ResizeMode::Letterbox).is_err());
        assert!(Preprocessor::try_new(640, 640, ResizeMode::Letterbox).is_ok());
        // `new` never allocates; the oversize is rejected by `run` instead of aborting.
        let mut p = Preprocessor::new(16384, 16384, ResizeMode::Letterbox);
        let img = solid(4, 4, [0, 0, 0]);
        let e = p.run(&img, 4, 4).unwrap_err().to_string();
        assert!(e.contains("16384x16384"), "{e}");
        assert_eq!(fit_side(f64::NAN, 10), 1);
        assert_eq!(fit_side(1e12, 10), 10);
        assert_eq!(fit_side(3.4, 10), 3);
    }

    #[test]
    fn run_into_matches_run_on_a_dirty_buffer() {
        let mut p = Preprocessor::new(64, 48, ResizeMode::Letterbox)
            .with_normalization(Some(Normalization::IMAGENET));
        let img: Vec<u8> = (0..100 * 30 * 3).map(|i| (i * 7 % 256) as u8).collect();
        let (want, want_ctx) = p.run(&img, 100, 30).map(|(t, c)| (t.to_vec(), c)).unwrap();
        let mut out = vec![f32::NAN; 5];
        let ctx = p.run_into(&img, 100, 30, &mut out).unwrap();
        assert_eq!(ctx, want_ctx);
        assert_eq!(out.len(), want.len());
        assert!(
            out.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        // Reused with stale values from a bigger image: still exact.
        out.iter_mut().for_each(|v| *v = 9.0);
        p.run_into(&img, 100, 30, &mut out).unwrap();
        assert!(
            out.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
}
