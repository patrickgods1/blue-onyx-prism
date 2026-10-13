//! Image decode/encode and annotation (bounding boxes + labels).
//!
//! Text is rasterized directly with `ab_glyph` and boxes are drawn on the raw RGB buffer, so
//! no optional `imageproc` features are required. Label font: Roboto Mono (Apache-2.0), see
//! `assets/roboto-mono-stripped.ttf` (taken from blue-onyx).

use crate::api::Prediction;
use ab_glyph::{Font, FontArc, PxScale, ScaleFont};
use anyhow::{Context, Result, anyhow, bail};
use std::path::Path;
use zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions};
use zune_jpeg::JpegDecoder;

static FONT_BYTES: &[u8] = include_bytes!("../assets/roboto-mono-stripped.ttf");

/// Default of `max_image_pixels` (config): 50 megapixels, about 8700 x 5700.
pub const MAX_IMAGE_PIXELS_DEFAULT: u64 = 50_000_000;
static MAX_IMAGE_PIXELS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(MAX_IMAGE_PIXELS_DEFAULT);

/// Set the process-wide decode ceiling (config `max_image_pixels`); 0 restores the default.
pub fn set_max_pixels(pixels: u64) {
    let v = if pixels == 0 {
        MAX_IMAGE_PIXELS_DEFAULT
    } else {
        pixels
    };
    MAX_IMAGE_PIXELS.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// The current decode ceiling in pixels (width x height).
pub fn max_pixels() -> u64 {
    #[cfg(test)]
    if let Some(v) = TEST_LIMIT.with(std::cell::Cell::get) {
        return v;
    }
    MAX_IMAGE_PIXELS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
thread_local! {
    /// Per-thread ceiling for tests: a small limit set through the process-wide value made
    /// concurrently running decode tests fail.
    static TEST_LIMIT: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// Tightly packed RGB8 image (`rgb.len() == width * height * 3`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RgbImage {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// Error for an image rejected by the decode ceiling (`max_image_pixels`) or a zero size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SizeRejected(pub String);

impl std::fmt::Display for SizeRejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SizeRejected {}

/// Reject a zero-sized image or one over [`max_pixels`] before anything is allocated.
fn check_dimensions(w: u64, h: u64) -> Result<()> {
    if w == 0 || h == 0 {
        return Err(SizeRejected(format!("image has a zero dimension ({w}x{h})")).into());
    }
    let limit = max_pixels();
    let pixels = w.saturating_mul(h);
    if pixels > limit {
        return Err(SizeRejected(format!(
            "image {w}x{h} ({pixels} pixels) exceeds the decode limit of {limit} pixels \
             (max_image_pixels)"
        ))
        .into());
    }
    Ok(())
}

/// `w * h * 3`, the byte length of a packed RGB8 image, or an error on overflow.
fn rgb_len(w: u32, h: u32) -> Result<usize> {
    (w as usize)
        .checked_mul(h as usize)
        .and_then(|p| p.checked_mul(3))
        .ok_or_else(|| anyhow!("image {w}x{h} does not fit in memory"))
}

/// Decode JPEG (zune-jpeg) or any format the `image` crate understands (PNG, ...).
///
/// The header is read first: a zero-sized image or one whose pixel count exceeds
/// [`max_pixels`] is rejected before any pixel buffer is allocated.
pub fn decode(bytes: &[u8]) -> Result<RgbImage> {
    if bytes.starts_with(&[0xFF, 0xD8]) {
        match decode_jpeg(bytes) {
            Ok(img) => return Ok(img),
            Err(e) if e.is::<SizeRejected>() => return Err(e),
            Err(e) => {
                tracing::debug!("zune-jpeg failed ({e:#}), falling back to image crate");
            }
        }
    }
    decode_with_image_crate(bytes)
}

fn decode_with_image_crate(bytes: &[u8]) -> Result<RgbImage> {
    use image::ImageReader;
    use std::io::Cursor;

    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("detecting image format")?;
    let format = reader
        .format()
        .ok_or_else(|| anyhow!("unrecognized image format"))?;
    let (w, h) = reader
        .into_dimensions()
        .context("reading image dimensions")?;
    check_dimensions(u64::from(w), u64::from(h))?;
    let expected = rgb_len(w, h)?;

    let mut reader = ImageReader::new(Cursor::new(bytes));
    reader.set_format(format);
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(w);
    limits.max_image_height = Some(h);
    // Room for the decoder's working set plus a 16-bit RGBA frame of the ceiling.
    limits.max_alloc = Some(max_pixels().saturating_mul(16));
    reader.limits(limits);
    let rgb = reader.decode().context("decoding image")?.into_rgb8();
    if (rgb.width(), rgb.height()) != (w, h) {
        bail!(
            "decoded image is {}x{}, header said {w}x{h}",
            rgb.width(),
            rgb.height()
        );
    }
    let rgb = rgb.into_raw();
    if rgb.len() != expected {
        bail!("unexpected RGB buffer size {} for {w}x{h}", rgb.len());
    }
    Ok(RgbImage {
        width: w,
        height: h,
        rgb,
    })
}

fn decode_jpeg(bytes: &[u8]) -> Result<RgbImage> {
    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    decoder
        .decode_headers()
        .map_err(|e| anyhow!("JPEG header decode failed: {e:?}"))?;
    let (w, h) = decoder
        .dimensions()
        .ok_or_else(|| anyhow!("JPEG has no dimensions"))?;
    let w = u32::try_from(w).map_err(|_| anyhow!("JPEG width {w} is out of range"))?;
    let h = u32::try_from(h).map_err(|_| anyhow!("JPEG height {h} is out of range"))?;
    check_dimensions(u64::from(w), u64::from(h))?;
    let expected = rgb_len(w, h)?;
    let rgb = decoder
        .decode()
        .map_err(|e| anyhow!("JPEG decode failed: {e:?}"))?;
    if rgb.len() != expected {
        bail!("unexpected JPEG output size {} for {w}x{h}", rgb.len());
    }
    Ok(RgbImage {
        width: w,
        height: h,
        rgb,
    })
}

/// Encode as baseline JPEG.
pub fn encode_jpeg(img: &RgbImage, quality: u8) -> Result<Vec<u8>> {
    if img.width == 0 || img.height == 0 {
        bail!("cannot encode an empty image");
    }
    let (Ok(w16), Ok(h16)) = (u16::try_from(img.width), u16::try_from(img.height)) else {
        bail!("image too large for JPEG ({}x{})", img.width, img.height);
    };
    if img.rgb.len() != rgb_len(img.width, img.height)? {
        bail!("RGB buffer size does not match dimensions");
    }
    let mut out = Vec::new();
    let enc = jpeg_encoder::Encoder::new(&mut out, quality.clamp(1, 100));
    enc.encode(&img.rgb, w16, h16, jpeg_encoder::ColorType::Rgb)
        .map_err(|e| anyhow!("JPEG encode failed: {e}"))?;
    Ok(out)
}

const BOX_COLOR: [u8; 3] = [255, 0, 0];
const BAR_COLOR: [u8; 3] = [170, 0, 0];
const TEXT_COLOR: [u8; 3] = [255, 255, 255];

/// Round a geometry value to an integer; NaN maps to 0 and huge values saturate at +-1e9.
#[allow(
    clippy::cast_possible_truncation,
    reason = "clamped to +-1e9 (fits i64); NaN clamps to NaN and `as` maps it to 0"
)]
fn f2i(v: f32) -> i64 {
    v.round().clamp(-1e9, 1e9) as i64
}

/// Clamp `v` into `0..=max` and convert to `usize` (never negative after the clamp).
fn clamp_index(v: i64, max: u32) -> usize {
    usize::try_from(v.clamp(0, i64::from(max))).unwrap_or(0)
}

fn fill_rect(img: &mut RgbImage, x0: i64, y0: i64, x1: i64, y1: i64, color: [u8; 3]) {
    // Half-open [x0, x1) x [y0, y1), clamped to the image.
    let x0 = clamp_index(x0, img.width);
    let x1 = clamp_index(x1, img.width);
    let y0 = clamp_index(y0, img.height);
    let y1 = clamp_index(y1, img.height);
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    let Some(row_bytes) = (img.width as usize).checked_mul(3) else {
        return;
    };
    for row in img.rgb.chunks_exact_mut(row_bytes).skip(y0).take(y1 - y0) {
        if let Some(span) = row.get_mut(x0 * 3..x1 * 3) {
            for px in span.as_chunks_mut::<3>().0 {
                *px = color;
            }
        }
    }
}

fn blend_pixel(img: &mut RgbImage, x: i64, y: i64, color: [u8; 3], coverage: f32) {
    if x < 0 || y < 0 || x >= i64::from(img.width) || y >= i64::from(img.height) {
        return;
    }
    let (x, y) = (clamp_index(x, img.width), clamp_index(y, img.height));
    let i = (y * img.width as usize + x) * 3;
    let Some(px3) = img.rgb.get_mut(i..i + 3) else {
        return;
    };
    let a = coverage.clamp(0.0, 1.0);
    for (px, &src) in px3.iter_mut().zip(color.iter()) {
        let dst = f32::from(*px);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a linear blend of two 0..=255 values with a 0..=1 weight, clamped again"
        )]
        let v = (dst + (f32::from(src) - dst) * a).round().clamp(0.0, 255.0) as u8;
        *px = v;
    }
}

fn draw_text(img: &mut RgbImage, font: &FontArc, px: f32, x: i64, y: i64, max_x: i64, text: &str) {
    let scaled = font.as_scaled(PxScale::from(px));
    let mut caret = x as f32;
    let baseline = y as f32 + scaled.ascent();
    for ch in text.chars() {
        let gid = scaled.glyph_id(ch);
        let glyph =
            gid.with_scale_and_position(PxScale::from(px), ab_glyph::point(caret, baseline));
        caret += scaled.h_advance(gid);
        if f2i(caret) > max_x {
            break;
        }
        if let Some(outlined) = font.outline_glyph(glyph) {
            let b = outlined.px_bounds();
            outlined.draw(|gx, gy, cov| {
                blend_pixel(
                    img,
                    f2i(b.min.x) + i64::from(gx),
                    f2i(b.min.y) + i64::from(gy),
                    TEXT_COLOR,
                    cov,
                );
            });
        }
    }
}

/// Return a copy of `img` with a red rectangle and a `label  NN%` bar for every prediction.
/// Boxes outside the image are clamped; degenerate boxes are skipped for the rectangle.
pub fn draw_predictions(img: &RgbImage, preds: &[Prediction]) -> RgbImage {
    let mut out = img.clone();
    if out.width == 0
        || out.height == 0
        || Some(out.rgb.len()) != rgb_len(out.width, out.height).ok()
    {
        return out;
    }
    let font = FontArc::try_from_slice(FONT_BYTES).ok();
    let scale = (out.width.max(out.height) as f32 / 640.0).max(0.5);
    let thickness = f2i(scale.ceil()).max(1);
    let font_px = (16.0 * scale).ceil().max(12.0);
    let bar_h = f2i(font_px);
    let coord = |v: usize| i64::try_from(v).unwrap_or(i64::MAX / 4);

    for p in preds {
        let (x0, y0, x1, y1) = (
            coord(p.x_min),
            coord(p.y_min),
            coord(p.x_max),
            coord(p.y_max),
        );
        if x1 > x0 && y1 > y0 {
            let t = thickness
                .min((x1 - x0 + 1) / 2)
                .min((y1 - y0 + 1) / 2)
                .max(1);
            fill_rect(&mut out, x0, y0, x1, y0 + t, BOX_COLOR);
            fill_rect(&mut out, x0, y1 - t, x1, y1, BOX_COLOR);
            fill_rect(&mut out, x0, y0, x0 + t, y1, BOX_COLOR);
            fill_rect(&mut out, x1 - t, y0, x1, y1, BOX_COLOR);
        }
        if let Some(font) = &font {
            // Keep the bar on screen even when the box touches the top edge.
            let bar_w = (x1 - x0).max(bar_h * 6);
            let bar_y = y0.clamp(0, (i64::from(out.height) - bar_h).max(0));
            fill_rect(&mut out, x0, bar_y, x0 + bar_w, bar_y + bar_h, BAR_COLOR);
            let label = format!("{}   {:.0}%", p.label, p.confidence * 100.0);
            draw_text(&mut out, font, font_px - 1.0, x0, bar_y, x0 + bar_w, &label);
        }
    }
    out
}

/// Longest file stem [`save_annotated`] keeps from a client-supplied name.
const MAX_STEM_LEN: usize = 64;

/// Windows reserved device names: `CON.jpg` (any extension, any case) opens the device.
fn is_windows_reserved(stem: &str) -> bool {
    let base = stem.split('.').next().unwrap_or(stem).to_ascii_uppercase();
    match base.as_str() {
        "CON" | "PRN" | "AUX" | "NUL" => true,
        _ => {
            let b = base.as_bytes();
            b.len() == 4
                && (base.starts_with("COM") || base.starts_with("LPT"))
                && b.get(3).is_some_and(u8::is_ascii_digit)
        }
    }
}

/// The file stem of an untrusted (client-supplied) name, safe on every platform: characters
/// outside `[A-Za-z0-9._-]` become `_`, leading/trailing dots are dropped, at most
/// [`MAX_STEM_LEN`] characters; `None` when nothing usable is left or it names a Windows device.
fn safe_stem(name: &str) -> Option<String> {
    // Both separators, whatever the host OS (a Windows client may send `C:\cam\x.jpg`).
    let file = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let stem = match file.rfind('.') {
        Some(i) if i > 0 => file.get(..i).unwrap_or(file),
        _ => file,
    };
    let clean: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .take(MAX_STEM_LEN)
        .collect();
    let clean = clean.trim_matches('.');
    (!clean.is_empty() && !is_windows_reserved(clean)).then(|| clean.to_string())
}

/// Write `<stem>_od.jpg` (annotated) into `dir`, plus `<stem>.jpg` (unannotated) when
/// `save_ref` is set. `name` is untrusted (client-supplied): only a sanitized form of its file
/// stem is used ([`safe_stem`]), else a random `image_<uuid>` name.
pub fn save_annotated(
    dir: &Path,
    name: &str,
    img: &RgbImage,
    preds: &[Prediction],
    save_ref: bool,
) -> Result<()> {
    let stem = safe_stem(name).unwrap_or_else(|| format!("image_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;

    let annotated = draw_predictions(img, preds);
    let od_path = dir.join(format!("{stem}_od.jpg"));
    std::fs::write(&od_path, encode_jpeg(&annotated, 95)?)
        .with_context(|| format!("writing {}", od_path.display()))?;
    if save_ref {
        let ref_path = dir.join(format!("{stem}.jpg"));
        std::fs::write(&ref_path, encode_jpeg(img, 95)?)
            .with_context(|| format!("writing {}", ref_path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(w: u32, h: u32) -> RgbImage {
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                rgb.extend_from_slice(&[(x * 4) as u8, (y * 4) as u8, 128]);
            }
        }
        RgbImage {
            width: w,
            height: h,
            rgb,
        }
    }

    #[test]
    fn jpeg_roundtrip() {
        let img = synthetic(64, 48);
        let bytes = encode_jpeg(&img, 95).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!((back.width, back.height), (64, 48));
        assert_eq!(back.rgb.len(), 64 * 48 * 3);
        // Lossy: check a mid pixel is close.
        let i = (24 * 64 + 32) * 3;
        for c in 0..3 {
            assert!((back.rgb[i + c] as i32 - img.rgb[i + c] as i32).abs() < 24);
        }
    }

    #[test]
    fn png_decodes() {
        let img = synthetic(8, 8);
        let buf = image::RgbImage::from_raw(8, 8, img.rgb.clone()).unwrap();
        let mut png = Vec::new();
        buf.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        assert_eq!(decode(&png).unwrap(), img);
    }

    #[test]
    fn draw_with_out_of_range_boxes() {
        let img = synthetic(32, 32);
        let preds = vec![
            Prediction {
                x_min: 0,
                y_min: 0,
                x_max: 5000,
                y_max: 5000,
                confidence: 0.9,
                label: "big".into(),
            },
            Prediction {
                x_min: 100,
                y_min: 100,
                x_max: 200,
                y_max: 200,
                confidence: 0.5,
                label: "off".into(),
            },
            Prediction {
                x_min: 10,
                y_min: 10,
                x_max: 10,
                y_max: 10,
                confidence: 0.1,
                label: "empty".into(),
            },
            Prediction {
                x_min: 20,
                y_min: 20,
                x_max: 5,
                y_max: 5,
                confidence: 0.1,
                label: "inverted".into(),
            },
        ];
        let out = draw_predictions(&img, &preds);
        assert_eq!(out.rgb.len(), img.rgb.len());
        assert_ne!(out.rgb, img.rgb);
        let none = draw_predictions(&RgbImage::default(), &preds);
        assert!(none.rgb.is_empty());
    }

    #[test]
    fn save_writes_files() {
        let dir = std::env::temp_dir().join(format!("bo_img_test_{}", uuid::Uuid::new_v4()));
        let img = synthetic(32, 32);
        save_annotated(&dir, "../evil/cam1.jpg", &img, &[], true).unwrap();
        assert!(dir.join("cam1_od.jpg").exists());
        assert!(dir.join("cam1.jpg").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn saved_names_are_sanitized() {
        assert_eq!(safe_stem("cam1.jpg").as_deref(), Some("cam1"));
        assert_eq!(safe_stem("../evil/cam1.jpg").as_deref(), Some("cam1"));
        assert_eq!(safe_stem("C:\\x\\cam 2.jpg").as_deref(), Some("cam_2"));
        assert_eq!(safe_stem("a:b*c?.jpg").as_deref(), Some("a_b_c_"));
        assert_eq!(safe_stem("caméra.jpg").as_deref(), Some("cam_ra"));
        assert_eq!(safe_stem("v1.2.3.jpg").as_deref(), Some("v1.2.3"));
        assert_eq!(safe_stem(".hidden").as_deref(), Some("hidden"));
        let long = format!("{}.jpg", "x".repeat(200));
        assert_eq!(safe_stem(&long).map(|s| s.len()), Some(MAX_STEM_LEN));
        for bad in [
            "",
            ".",
            "..",
            "...",
            "/",
            "CON.jpg",
            "con",
            "Nul.png",
            "aux",
            "PRN.x.jpg",
            "COM1.jpg",
            "lpt9",
            "COM0.jpg",
            "con..jpg",
        ] {
            assert_eq!(safe_stem(bad), None, "{bad:?}");
        }
        for ok in ["CONSOLE.jpg", "COM10.jpg", "LPTX.jpg", "con_od.jpg", "NULL"] {
            assert!(safe_stem(ok).is_some(), "{ok:?}");
        }
    }

    #[test]
    fn reserved_name_falls_back_to_uuid() {
        let dir = std::env::temp_dir().join(format!("bo_img_test_{}", uuid::Uuid::new_v4()));
        let img = synthetic(8, 8);
        save_annotated(&dir, "CON.jpg", &img, &[], true).unwrap();
        let names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names.len(), 2, "{names:?}");
        assert!(names.iter().all(|n| n.starts_with("image_")), "{names:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn png_bytes(w: u32, h: u32) -> Vec<u8> {
        let img = synthetic(w, h);
        let buf = image::RgbImage::from_raw(w, h, img.rgb).unwrap();
        let mut png = Vec::new();
        buf.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        png
    }

    /// A PNG signature + IHDR claiming `w x h` (no pixel data): enough for the header read.
    fn png_header_only(w: u32, h: u32) -> Vec<u8> {
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(b"IHDR");
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB
        let crc = |data: &[u8]| {
            // CRC-32 (ISO 3309), as the PNG spec requires over type + data.
            let mut c: u32 = 0xFFFF_FFFF;
            for &b in data {
                c ^= u32::from(b);
                for _ in 0..8 {
                    c = if c & 1 != 0 {
                        0xEDB8_8320 ^ (c >> 1)
                    } else {
                        c >> 1
                    };
                }
            }
            !c
        };
        let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        out.extend_from_slice(&13u32.to_be_bytes());
        out.extend_from_slice(&ihdr);
        out.extend_from_slice(&crc(&ihdr).to_be_bytes());
        // An empty IDAT chunk: the header reader stops here, before any pixel data.
        out.extend_from_slice(&0u32.to_be_bytes());
        out.extend_from_slice(b"IDAT");
        out.extend_from_slice(&crc(b"IDAT").to_be_bytes());
        out
    }

    #[test]
    fn zero_size_png_rejected() {
        // The PNG decoder itself refuses a zero dimension; our own check covers decoders
        // that do not.
        assert!(decode(&png_header_only(0, 10)).is_err());
        let e = check_dimensions(0, 10).unwrap_err();
        assert!(e.is::<SizeRejected>(), "{e:#}");
        assert!(e.to_string().contains("0x10"), "{e}");
        assert!(check_dimensions(10, 0).is_err());
        assert!(check_dimensions(10, 10).is_ok());
        assert!(check_dimensions(u64::MAX, 2).is_err());
    }

    #[test]
    fn oversize_header_rejected_before_decoding() {
        // 20000 x 20000 = 400 MP > the 50 MP default; only the header exists, so a decoder that
        // reached the pixel data would fail with a different (truncation) error.
        let e = decode(&png_header_only(20000, 20000)).unwrap_err();
        assert!(e.is::<SizeRejected>(), "{e:#}");
        let msg = e.to_string();
        assert!(
            msg.contains("20000x20000") && msg.contains("50000000"),
            "{msg}"
        );
        // Same for a JPEG whose SOF0 header claims 9000x9000 (81 MP): patch a real 8x8 file.
        let mut jpg = encode_jpeg(&synthetic(8, 8), 90).unwrap();
        let sof = jpg.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        jpg[sof + 5..sof + 7].copy_from_slice(&9000u16.to_be_bytes());
        jpg[sof + 7..sof + 9].copy_from_slice(&9000u16.to_be_bytes());
        let e = decode(&jpg).unwrap_err();
        assert!(e.is::<SizeRejected>(), "{e:#}");
        assert!(e.to_string().contains("9000x9000"), "{e}");
    }

    #[test]
    fn max_pixels_setting_is_honoured() {
        // Small limits go through this thread's override: the process-wide value is shared
        // with every decode test running in parallel.
        let png = png_bytes(8, 8);
        let jpg = encode_jpeg(&synthetic(8, 8), 90).unwrap();
        TEST_LIMIT.with(|l| l.set(Some(63)));
        assert_eq!(max_pixels(), 63);
        let e = decode(&png).unwrap_err();
        assert!(
            e.is::<SizeRejected>() && e.to_string().contains("8x8"),
            "{e}"
        );
        assert!(decode(&jpg).unwrap_err().is::<SizeRejected>());
        TEST_LIMIT.with(|l| l.set(Some(64)));
        assert_eq!(decode(&png).unwrap(), synthetic(8, 8));
        assert_eq!(decode(&jpg).unwrap().width, 8);
        TEST_LIMIT.with(|l| l.set(None));
        // The process-wide setter, with values no other test's images come near.
        set_max_pixels(MAX_IMAGE_PIXELS_DEFAULT * 2);
        assert_eq!(max_pixels(), MAX_IMAGE_PIXELS_DEFAULT * 2);
        set_max_pixels(0);
        assert_eq!(max_pixels(), MAX_IMAGE_PIXELS_DEFAULT);
        assert!(decode(&png_bytes(16, 16)).is_ok());
    }

    #[test]
    fn encode_rejects_bad_buffers() {
        let mut img = synthetic(4, 4);
        img.rgb.pop();
        assert!(encode_jpeg(&img, 90).is_err());
        let huge = RgbImage {
            width: 70_000,
            height: 1,
            rgb: Vec::new(),
        };
        assert!(
            encode_jpeg(&huge, 90)
                .unwrap_err()
                .to_string()
                .contains("too large")
        );
    }
}
