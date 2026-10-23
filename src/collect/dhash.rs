//! 64-bit difference hash (dHash) of a frame for near-duplicate detection: the luma is
//! box-averaged down to 9 x 8 cells and bit `8 * row + col` is set when cell `col` is brighter
//! than cell `col + 1` of the same row. Two frames of a still scene differ in a few bits; a
//! moved object or another scene flips many. Pure, plus a grayscale JPEG decode for the
//! collector thread ([`of_encoded`]).

use anyhow::{Result, anyhow};

/// Columns of the downscale (one more than the bits per row).
const COLS: usize = 9;
const ROWS: usize = 8;

/// dHash of a `w` x `h` 8-bit luma plane (`luma.len() >= w * h`). Each cell is the exact mean
/// of its pixels (cell edges at `floor(i * w / 9)`); an image narrower than 9 or shorter than
/// 8 pixels repeats pixels so every cell has at least one.
pub fn of_luma(luma: &[u8], w: usize, h: usize) -> u64 {
    if w == 0 || h == 0 || luma.len() < w.saturating_mul(h) {
        return 0;
    }
    let edges = |n: usize, cells: usize| -> Vec<(usize, usize)> {
        (0..cells)
            .map(|i| {
                let a = i * n / cells;
                let b = ((i + 1) * n / cells).max(a + 1).min(n.max(1));
                (a.min(n - 1), b)
            })
            .collect()
    };
    let xs = edges(w, COLS);
    let ys = edges(h, ROWS);
    let mut cells = [[0f64; COLS]; ROWS];
    for (r, &(y0, y1)) in ys.iter().enumerate() {
        for (c, &(x0, x1)) in xs.iter().enumerate() {
            let mut sum = 0u64;
            for y in y0..y1 {
                let row = luma.get(y * w + x0..y * w + x1).unwrap_or_default();
                sum += row.iter().map(|&v| u64::from(v)).sum::<u64>();
            }
            let n = ((y1 - y0) * (x1 - x0)).max(1);
            if let Some(cell) = cells.get_mut(r).and_then(|row| row.get_mut(c)) {
                *cell = sum as f64 / n as f64;
            }
        }
    }
    let mut hash = 0u64;
    for (r, row) in cells.iter().enumerate() {
        for c in 0..COLS - 1 {
            let (a, b) = (row.get(c).copied(), row.get(c + 1).copied());
            if let (Some(a), Some(b)) = (a, b)
                && a > b
            {
                hash |= 1u64 << (r * (COLS - 1) + c);
            }
        }
    }
    hash
}

/// dHash of a packed RGB8 image (BT.601 luma).
pub fn of_rgb(img: &crate::image::RgbImage) -> u64 {
    let luma: Vec<u8> = img
        .rgb
        .as_chunks::<3>()
        .0
        .iter()
        .map(|[r, g, b]| crate::image::luma601(*r, *g, *b))
        .collect();
    of_luma(&luma, img.width as usize, img.height as usize)
}

/// dHash and size of an encoded image: a JPEG is decoded straight to luma (no colour
/// conversion), anything else through [`crate::image::decode`].
pub fn of_encoded(bytes: &[u8]) -> Result<(u64, u32, u32)> {
    if bytes.starts_with(&[0xFF, 0xD8]) {
        use zune_core::bytestream::ZCursor;
        use zune_core::colorspace::ColorSpace;
        use zune_core::options::DecoderOptions;
        let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::Luma);
        let mut d = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes), options);
        if let Ok(luma) = d.decode()
            && let Some((w, h)) = d.dimensions()
            && luma.len() >= w * h
        {
            let (w32, h32) = (
                u32::try_from(w).map_err(|_| anyhow!("width out of range"))?,
                u32::try_from(h).map_err(|_| anyhow!("height out of range"))?,
            );
            return Ok((of_luma(&luma, w, h), w32, h32));
        }
    }
    let img = crate::image::decode(bytes)?;
    Ok((of_rgb(&img), img.width, img.height))
}

/// Bits that differ.
pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// 16 lowercase hex digits.
pub fn hex(h: u64) -> String {
    format!("{h:016x}")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::cast_possible_truncation)]
mod tests {
    use super::*;
    use crate::image::RgbImage;

    fn gradient(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> Vec<u8> {
        (0..w * h).map(|i| f(i % w, i / w)).collect()
    }

    #[test]
    fn decreasing_rows_set_every_bit_and_increasing_none() {
        let (w, h) = (90, 80);
        let down = gradient(w, h, |x, _| 255 - (x * 255 / w) as u8);
        assert_eq!(of_luma(&down, w, h), u64::MAX);
        let up = gradient(w, h, |x, _| (x * 255 / w) as u8);
        assert_eq!(of_luma(&up, w, h), 0);
        let flat = vec![128u8; w * h];
        assert_eq!(of_luma(&flat, w, h), 0);
    }

    #[test]
    fn exact_cell_means_on_a_9x8_image() {
        // One pixel per cell: bit (r, c) set iff p[r][c] > p[r][c+1].
        let mut px = vec![0u8; 72];
        px[0] = 10; // row 0: 10 > 0 -> bit 0
        px[9 + 3] = 5; // row 1, col 3: 5 > 0 -> bit 8 + 3; col 2: 0 > 5 no
        let h = of_luma(&px, 9, 8);
        assert_eq!(h, 1 | (1 << 11));
    }

    #[test]
    fn tiny_and_bad_inputs_do_not_panic() {
        assert_eq!(of_luma(&[], 0, 0), 0);
        assert_eq!(of_luma(&[1, 2], 2, 1), of_luma(&[1, 2], 2, 1));
        assert_eq!(of_luma(&[1], 5, 5), 0);
        let _ = of_luma(&[200, 10, 5], 3, 1);
    }

    #[test]
    fn small_noise_is_close_and_another_scene_is_far() {
        let (w, h) = (160usize, 120usize);
        let scene =
            |x: usize, y: usize| ((x * 7 + y * 3) % 200) as u8 + if x > 80 { 40 } else { 0 };
        let a = gradient(w, h, scene);
        let b: Vec<u8> = a
            .iter()
            .enumerate()
            .map(|(i, v)| v.saturating_add((i % 3) as u8))
            .collect();
        let c = gradient(w, h, |x, y| ((w - x) * 2 + y) as u8);
        let (ha, hb, hc) = (of_luma(&a, w, h), of_luma(&b, w, h), of_luma(&c, w, h));
        assert!(hamming(ha, hb) <= 6, "{}", hamming(ha, hb));
        assert!(hamming(ha, hc) > 6, "{}", hamming(ha, hc));
        assert_eq!(hex(0xab), "00000000000000ab");
    }

    #[test]
    fn encoded_jpeg_and_png_hash_like_the_rgb_frame() {
        let (w, h) = (96u32, 64u32);
        let img = RgbImage {
            width: w,
            height: h,
            rgb: (0..w * h)
                .flat_map(|i| {
                    let x = i % w;
                    let v = (255 - x * 255 / w) as u8;
                    [v, v, v]
                })
                .collect(),
        };
        let direct = of_rgb(&img);
        let jpeg = crate::image::encode_jpeg(&img, 95).unwrap();
        let (hj, jw, jh) = of_encoded(&jpeg).unwrap();
        assert_eq!((jw, jh), (w, h));
        assert!(hamming(direct, hj) <= 2, "{}", hamming(direct, hj));
        assert!(of_encoded(b"not an image").is_err());
    }
}
