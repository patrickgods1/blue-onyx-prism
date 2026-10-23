//! Optional image enhancement before the model's letterbox / stretch ([`Preprocess`]): BT.601
//! grayscale, a gamma curve, or CLAHE (contrast-limited adaptive histogram equalization) on
//! luma. Pure, on the decoded [`RgbImage`]; the prep pool ([`crate::prep`]) applies it per
//! request after the scene statistics are taken from the original frame, and the benchmark
//! calls the same [`apply`]. Off by default: published results show enhancement before a
//! detector trained on normal images helps some models a little and hurts others, so measure
//! it with `benchmark --variants pre:<mode>` first.
//!
//! String form (config and CLI): `none`, `grayscale`, `gamma:<g>` (`out = 255 (in/255)^g`, so
//! `g < 1` brightens shadows), `clahe[:<clip>[,<tiles>]]` (clip limit as a multiple of the
//! mean bin height, default [`DEFAULT_CLAHE_CLIP`]; `tiles` x `tiles` grid, default
//! [`DEFAULT_CLAHE_TILES`]).
//!
//! CLAHE follows OpenCV's algorithm: one 256-bin luma histogram per tile, each bin clipped at
//! `max(1, clip * tile_area / 256)` with the excess spread evenly over all bins (the remainder
//! one count per bin at an even stride), the tile's cumulative histogram scaled to 0..255 as its
//! map, and each pixel mapped by bilinear interpolation between the maps of the four nearest
//! tile centres (pixels outside the outer centres use the nearest ones). A monochrome frame gets
//! the new luma in all three channels; a colour frame keeps its hue: every channel is scaled by
//! `new / old` luma (a black pixel becomes the new gray), clamped to 255.

use crate::image::{RgbImage, luma601};
use crate::image_stats::Scene;
use serde::{Deserialize, Serialize};

/// Default CLAHE clip limit (OpenCV's usual 2.0).
pub const DEFAULT_CLAHE_CLIP: f32 = 2.0;
/// Default CLAHE tiles per axis (8 x 8, as OpenCV).
pub const DEFAULT_CLAHE_TILES: u8 = 8;
/// Largest accepted gamma exponent (and the reciprocal is the smallest).
pub const MAX_GAMMA: f32 = 10.0;
/// Largest accepted CLAHE clip limit.
pub const MAX_CLAHE_CLIP: f32 = 64.0;
/// Largest accepted CLAHE tiles per axis.
pub const MAX_CLAHE_TILES: u8 = 64;

/// Accepted forms, for error messages.
pub const PREPROCESS_FORMS: &str = "none, grayscale, gamma:<g> (e.g. gamma:0.6), clahe, clahe:<clip> or clahe:<clip>,<tiles> (e.g. clahe:2,8)";

/// One enhancement mode (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Preprocess {
    #[default]
    None,
    /// BT.601 luma in all three channels.
    Grayscale,
    /// `out = 255 (in/255)^g` on every channel.
    Gamma(f32),
    /// CLAHE on luma.
    Clahe { clip: f32, tiles: u8 },
}

impl Preprocess {
    /// CLAHE with the default clip and tiles.
    pub const CLAHE: Preprocess = Preprocess::Clahe {
        clip: DEFAULT_CLAHE_CLIP,
        tiles: DEFAULT_CLAHE_TILES,
    };

    pub fn is_none(&self) -> bool {
        matches!(self, Preprocess::None)
    }
}

impl std::fmt::Display for Preprocess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Preprocess::None => f.write_str("none"),
            Preprocess::Grayscale => f.write_str("grayscale"),
            Preprocess::Gamma(g) => write!(f, "gamma:{g}"),
            Preprocess::Clahe { clip, tiles }
                if clip == DEFAULT_CLAHE_CLIP && tiles == DEFAULT_CLAHE_TILES =>
            {
                f.write_str("clahe")
            }
            Preprocess::Clahe { clip, tiles } => write!(f, "clahe:{clip},{tiles}"),
        }
    }
}

fn parse_num(what: &str, s: &str, text: &str) -> Result<f32, String> {
    let v: f32 = s
        .trim()
        .parse()
        .map_err(|_| format!("preprocess '{text}': {what} '{s}' is not a number"))?;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("preprocess '{text}': {what} must be finite"))
    }
}

impl std::str::FromStr for Preprocess {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        let s = text.trim().to_ascii_lowercase();
        let (head, arg) = match s.split_once(':') {
            Some((h, a)) => (h.trim(), Some(a.trim())),
            None => (s.as_str(), None),
        };
        match (head, arg) {
            ("none", None) => Ok(Preprocess::None),
            ("grayscale", None) => Ok(Preprocess::Grayscale),
            ("gamma", Some(a)) => {
                let g = parse_num("gamma", a, text)?;
                if g <= 0.0 || g > MAX_GAMMA || g < 1.0 / MAX_GAMMA {
                    return Err(format!(
                        "preprocess '{text}': gamma must be between {} and {MAX_GAMMA} \
                         (below 1 brightens shadows, e.g. gamma:0.6)",
                        1.0 / MAX_GAMMA
                    ));
                }
                Ok(Preprocess::Gamma(g))
            }
            ("clahe", None) => Ok(Preprocess::CLAHE),
            ("clahe", Some(a)) => {
                let (c, t) = match a.split_once(',') {
                    Some((c, t)) => (c, Some(t)),
                    None => (a, None),
                };
                let clip = parse_num("clip", c, text)?;
                if clip <= 0.0 || clip > MAX_CLAHE_CLIP {
                    return Err(format!(
                        "preprocess '{text}': clip must be above 0 and at most {MAX_CLAHE_CLIP} \
                         (2 is usual)"
                    ));
                }
                let tiles = match t {
                    None => DEFAULT_CLAHE_TILES,
                    Some(t) => t
                        .trim()
                        .parse::<u8>()
                        .ok()
                        .filter(|n| (1..=MAX_CLAHE_TILES).contains(n))
                        .ok_or_else(|| {
                            format!(
                                "preprocess '{text}': tiles must be a whole number from 1 to \
                                 {MAX_CLAHE_TILES} (8 is usual)"
                            )
                        })?,
                };
                Ok(Preprocess::Clahe { clip, tiles })
            }
            _ => Err(format!(
                "preprocess '{text}' is not a mode; use one of: {PREPROCESS_FORMS}"
            )),
        }
    }
}

impl TryFrom<String> for Preprocess {
    type Error = String;
    fn try_from(s: String) -> Result<Self, String> {
        s.parse()
    }
}

impl From<Preprocess> for String {
    fn from(p: Preprocess) -> String {
        p.to_string()
    }
}

/// The modes of one model: `preprocess` on every frame, replaced by `night_preprocess` (when
/// set) on night / IR frames.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PreprocessModes {
    pub preprocess: Preprocess,
    pub night_preprocess: Option<Preprocess>,
}

impl PreprocessModes {
    /// The mode for a frame of `scene`.
    pub fn for_scene(&self, scene: Scene) -> Preprocess {
        match self.night_preprocess {
            Some(n) if scene.is_night() => n,
            _ => self.preprocess,
        }
    }

    /// Nothing to do on any frame.
    pub fn is_noop(&self) -> bool {
        self.preprocess.is_none() && self.night_preprocess.is_none_or(|p| p.is_none())
    }
}

/// Enhance `img` in place with `mode` (see the module docs). A buffer shorter than
/// `width * height * 3` is processed up to its last whole pixel row.
pub fn apply(img: &mut RgbImage, mode: Preprocess) {
    match mode {
        Preprocess::None => {}
        Preprocess::Grayscale => img.to_gray3(),
        Preprocess::Gamma(g) => {
            let lut = gamma_lut(g);
            #[allow(clippy::indexing_slicing, reason = "u8 values index a 256-entry table")]
            for v in &mut img.rgb {
                *v = lut[usize::from(*v)];
            }
        }
        Preprocess::Clahe { clip, tiles } => clahe(img, clip, tiles),
    }
}

/// `lut[v] = round(255 (v/255)^g)`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is clamped to 0..=255 first"
)]
pub fn gamma_lut(g: f32) -> [u8; 256] {
    let g = f64::from(g);
    let mut lut = [0u8; 256];
    for (v, out) in lut.iter_mut().enumerate() {
        let y = 255.0 * (v as f64 / 255.0).powf(g);
        // In 0..=255 for any positive g.
        *out = y.round().clamp(0.0, 255.0) as u8;
    }
    lut
}

/// Tile bounds along one axis of `len` pixels split into `n` tiles: tile `i` is
/// `[i * len / n, (i + 1) * len / n)`.
fn tile_start(i: usize, len: usize, n: usize) -> usize {
    i * len / n
}

/// Per pixel along one axis: the two tiles to blend and the weight of the second (0..=256).
/// `1 <= n <= len` and `n <= MAX_CLAHE_TILES` (callers clamp), so every tile index fits u16
/// and every range lies inside `0..len`.
#[allow(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "tile indices < n <= 64; tile bounds within 0..=len; weights clamped to 0..=256"
)]
fn axis_weights(len: usize, n: usize) -> (Vec<u16>, Vec<(u16, u16, u16)>) {
    // Tile of every pixel (histogram pass).
    let mut tile_of = vec![0u16; len];
    for i in 0..n {
        let (a, b) = (tile_start(i, len, n), tile_start(i + 1, len, n));
        for t in &mut tile_of[a..b] {
            *t = i as u16;
        }
    }
    // Centre of every tile (pixel units, pixel x covers [x, x+1), its centre is x + 0.5).
    let centres: Vec<f32> = (0..n)
        .map(|i| (tile_start(i, len, n) + tile_start(i + 1, len, n)) as f32 / 2.0)
        .collect();
    let last = n - 1;
    let mut blend = Vec::with_capacity(len);
    let mut i = 0usize;
    for x in 0..len {
        let p = x as f32 + 0.5;
        if p <= centres[0] {
            blend.push((0, 0, 0));
        } else if p >= centres[last] {
            blend.push((last as u16, last as u16, 0));
        } else {
            while i + 1 < last && p >= centres[i + 1] {
                i += 1;
            }
            let w = (p - centres[i]) / (centres[i + 1] - centres[i]);
            let w = (w * 256.0).round().clamp(0.0, 256.0) as u16;
            blend.push((i as u16, (i + 1) as u16, w));
        }
    }
    (tile_of, blend)
}

/// The tile map of a histogram of `area` pixels with clip limit `clip` (see the module docs).
#[allow(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bins < 256 by the loop bound; the clip limit is a positive count; cdf <= area \
              so the map value is <= 255"
)]
fn tile_lut(hist: &mut [u32; 256], area: u32, clip: f32) -> [u8; 256] {
    let mut lut = [0u8; 256];
    if area == 0 {
        return lut;
    }
    let limit = ((clip * area as f32 / 256.0).round() as u32).max(1);
    let mut excess = 0u32;
    for h in hist.iter_mut() {
        if *h > limit {
            excess += *h - limit;
            *h = limit;
        }
    }
    let batch = excess / 256;
    let residual = (excess % 256) as usize;
    for h in hist.iter_mut() {
        *h += batch;
    }
    if let Some(step) = 256usize.checked_div(residual) {
        let step = step.max(1);
        let mut left = residual;
        let mut k = 0;
        while k < 256 && left > 0 {
            hist[k] += 1;
            left -= 1;
            k += step;
        }
    }
    let mut cdf = 0u64;
    let area = u64::from(area);
    for (h, out) in hist.iter().zip(lut.iter_mut()) {
        cdf += u64::from(*h);
        // cdf <= area, so the value is in 0..=255.
        *out = ((cdf * 255 + area / 2) / area).min(255) as u8;
    }
    lut
}

/// CLAHE on the luma of `img` (see the module docs).
#[allow(
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "rows are bounded by the buffer (`h * w * 3 <= len`), tile indices come from \
              axis_weights (< tx, ty), u8 values index 256-entry maps, blended values are \
              clamped to 255, tile areas fit u32 (a frame is at most max_image_pixels)"
)]
fn clahe(img: &mut RgbImage, clip: f32, tiles: u8) {
    let w = img.width as usize;
    let rows = if w == 0 {
        0
    } else {
        (img.height as usize).min(img.rgb.len() / (w * 3))
    };
    let h = rows;
    if w == 0 || h == 0 {
        return;
    }
    let tx = usize::from(tiles.max(1)).min(w);
    let ty = usize::from(tiles.max(1)).min(h);
    let (col_tile, col_blend) = axis_weights(w, tx);
    let (row_tile, row_blend) = axis_weights(h, ty);
    let rgb = &mut img.rgb[..w * h * 3];

    // Pass 1: luma, monochrome check, tile histograms.
    let mut luma = vec![0u8; w * h];
    let mut hists = vec![[0u32; 256]; tx * ty];
    let mut mono = true;
    for y in 0..h {
        let src = &rgb[y * w * 3..(y + 1) * w * 3];
        let dst = &mut luma[y * w..(y + 1) * w];
        let hrow = &mut hists[usize::from(row_tile[y]) * tx..(usize::from(row_tile[y]) + 1) * tx];
        for ((px, l), t) in src.as_chunks::<3>().0.iter().zip(dst).zip(&col_tile) {
            let [r, g, b] = *px;
            mono &= r == g && g == b;
            let v = luma601(r, g, b);
            *l = v;
            hrow[usize::from(*t)][usize::from(v)] += 1;
        }
    }
    let luts: Vec<[u8; 256]> = hists
        .iter_mut()
        .enumerate()
        .map(|(k, hist)| {
            let (i, j) = (k % tx, k / tx);
            let area = (tile_start(i + 1, w, tx) - tile_start(i, w, tx))
                * (tile_start(j + 1, h, ty) - tile_start(j, h, ty));
            tile_lut(hist, area as u32, clip)
        })
        .collect();

    // 1 / old luma for the colour scaling.
    let mut recip = [0f32; 256];
    for (v, r) in recip.iter_mut().enumerate().skip(1) {
        *r = 1.0 / v as f32;
    }

    // Pass 2: interpolate the maps and write back.
    for y in 0..h {
        let (r1, r2, wy) = row_blend[y];
        let wy = u32::from(wy);
        let top = &luts[usize::from(r1) * tx..(usize::from(r1) + 1) * tx];
        let bot = &luts[usize::from(r2) * tx..(usize::from(r2) + 1) * tx];
        let lrow = &luma[y * w..(y + 1) * w];
        let px = rgb[y * w * 3..(y + 1) * w * 3].as_chunks_mut::<3>().0;
        for ((p, &v), &(c1, c2, wx)) in px.iter_mut().zip(lrow).zip(&col_blend) {
            let (c1, c2, wx) = (usize::from(c1), usize::from(c2), u32::from(wx));
            let vi = usize::from(v);
            let a = u32::from(top[c1][vi]);
            let b = u32::from(top[c2][vi]);
            let c = u32::from(bot[c1][vi]);
            let d = u32::from(bot[c2][vi]);
            let t = a * (256 - wx) + b * wx;
            let u = c * (256 - wx) + d * wx;
            let nv = ((t * (256 - wy) + u * wy + 32768) >> 16).min(255) as u8;
            if mono {
                *p = [nv, nv, nv];
            } else if v == 0 {
                *p = [
                    p[0].saturating_add(nv),
                    p[1].saturating_add(nv),
                    p[2].saturating_add(nv),
                ];
            } else {
                let k = f32::from(nv) * recip[vi];
                for ch in p.iter_mut() {
                    *ch = (f32::from(*ch) * k + 0.5).min(255.0) as u8;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> RgbImage {
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = f(x, y);
                rgb.extend([v, v, v]);
            }
        }
        RgbImage {
            width: w,
            height: h,
            rgb,
        }
    }

    #[test]
    fn parse_and_display_round_trip() {
        for (text, want) in [
            ("none", Preprocess::None),
            ("grayscale", Preprocess::Grayscale),
            (" Gamma:0.6 ", Preprocess::Gamma(0.6)),
            ("clahe", Preprocess::CLAHE),
            (
                "clahe:3",
                Preprocess::Clahe {
                    clip: 3.0,
                    tiles: 8,
                },
            ),
            (
                "clahe:1.5,4",
                Preprocess::Clahe {
                    clip: 1.5,
                    tiles: 4,
                },
            ),
        ] {
            let p: Preprocess = text.parse().unwrap();
            assert_eq!(p, want, "{text}");
            let back: Preprocess = p.to_string().parse().unwrap();
            assert_eq!(back, p);
        }
        assert_eq!(Preprocess::Gamma(0.6).to_string(), "gamma:0.6");
        assert_eq!(Preprocess::CLAHE.to_string(), "clahe");
        assert_eq!(
            Preprocess::Clahe {
                clip: 2.0,
                tiles: 4
            }
            .to_string(),
            "clahe:2,4"
        );
        for bad in [
            "",
            "gamma",
            "gamma:0",
            "gamma:-1",
            "gamma:nan",
            "gamma:11",
            "clahe:0",
            "clahe:2,0",
            "clahe:2,65",
            "clahe:2,x",
            "sharpen",
            "none:1",
        ] {
            let e = bad.parse::<Preprocess>().unwrap_err();
            assert!(e.contains("preprocess"), "{bad}: {e}");
        }
    }

    #[test]
    fn serde_uses_the_string_form() {
        let v = serde_json::to_string(&Preprocess::Gamma(0.6)).unwrap();
        assert_eq!(v, r#""gamma:0.6""#);
        let p: Preprocess = serde_json::from_str(r#""clahe:2.5,6""#).unwrap();
        assert_eq!(
            p,
            Preprocess::Clahe {
                clip: 2.5,
                tiles: 6
            }
        );
        assert!(serde_json::from_str::<Preprocess>(r#""blur""#).is_err());
    }

    #[test]
    fn modes_pick_night_on_night_and_ir() {
        let m = PreprocessModes {
            preprocess: Preprocess::Gamma(0.8),
            night_preprocess: Some(Preprocess::CLAHE),
        };
        assert_eq!(m.for_scene(Scene::Day), Preprocess::Gamma(0.8));
        assert_eq!(m.for_scene(Scene::Night), Preprocess::CLAHE);
        assert_eq!(m.for_scene(Scene::Ir), Preprocess::CLAHE);
        let day_only = PreprocessModes {
            preprocess: Preprocess::Grayscale,
            night_preprocess: None,
        };
        assert_eq!(day_only.for_scene(Scene::Ir), Preprocess::Grayscale);
        assert!(PreprocessModes::default().is_noop());
        assert!(
            PreprocessModes {
                night_preprocess: Some(Preprocess::None),
                ..Default::default()
            }
            .is_noop()
        );
        assert!(!m.is_noop());
    }

    #[test]
    fn gamma_one_is_identity_and_below_one_brightens() {
        let lut = gamma_lut(1.0);
        assert!(lut.iter().enumerate().all(|(i, v)| usize::from(*v) == i));
        let lut = gamma_lut(0.5);
        assert_eq!((lut[0], lut[255]), (0, 255));
        // 255 * sqrt(64/255) = 127.75
        assert_eq!(lut[64], 128);
        assert!((1..255).all(|v| lut[v] >= v as u8));
        let mut img = RgbImage {
            width: 1,
            height: 1,
            rgb: vec![64, 0, 255],
        };
        apply(&mut img, Preprocess::Gamma(0.5));
        assert_eq!(img.rgb, [128, 0, 255]);
    }

    #[test]
    fn grayscale_is_bt601_luma() {
        let mut img = RgbImage {
            width: 2,
            height: 1,
            rgb: vec![255, 0, 0, 10, 200, 30],
        };
        apply(&mut img, Preprocess::Grayscale);
        let a = luma601(255, 0, 0);
        let b = luma601(10, 200, 30);
        assert_eq!(img.rgb, [a, a, a, b, b, b]);
    }

    #[test]
    fn tile_lut_equalizes_and_clips() {
        // Two equal populations at 0 and 255 with no clipping: the map is the scaled CDF.
        let mut h = [0u32; 256];
        h[0] = 50;
        h[255] = 50;
        let lut = tile_lut(&mut h, 100, 1000.0);
        assert_eq!(lut[0], 128); // round(50 * 255 / 100) = 127.5 -> 128
        assert_eq!(lut[254], 128);
        assert_eq!(lut[255], 255);

        // One bin of 256 pixels with clip 1: limit = 1, excess 255 = 0 per bin + 255 residual
        // spread one per bin from bin 0 (stride 1): every bin 1 except bin 255 ... bin 0 gets
        // 1 + 1 = 2.
        let mut h = [0u32; 256];
        h[10] = 256;
        let lut = tile_lut(&mut h, 256, 1.0);
        assert_eq!(h.iter().sum::<u32>(), 256, "counts preserved");
        assert_eq!(h[0], 1);
        assert_eq!(h[10], 2, "clipped to 1 plus one residual count");
        assert_eq!(h[255], 0);
        assert_eq!(lut[0], 1);
        assert_eq!(lut[254], 255);
        // A near-identity ramp: equalization does not create an extreme jump.
        assert!(lut.windows(2).all(|p| p[1] >= p[0]));
    }

    #[test]
    fn clahe_on_a_flat_frame_maps_through_its_histogram() {
        // A flat frame: every tile is one bin of `area` pixels. With clip 2 and an 8x8 tile of
        // 64 px the limit is max(1, round(2 * 64 / 256)) = 1: 63 excess -> 0 per bin + 63
        // residual at stride 4 (bins 0, 4, ..., 248). Value 100 then maps to
        // round(cdf(100) * 255 / 64): bins 0..=100 hold 26 residual counts (0, 4, .., 100)
        // plus the clipped 1 at bin 100 = 27 -> round(27 * 255 / 64) = 108.
        let mut img = gray(64, 64, |_, _| 100);
        apply(&mut img, Preprocess::CLAHE);
        assert!(img.rgb.iter().all(|&v| v == 108), "{:?}", &img.rgb[..9]);
    }

    #[test]
    fn clahe_stretches_low_contrast_and_keeps_order() {
        // A dark ramp 20..=51 over x: CLAHE spreads it over a wider range and keeps it
        // monotonic per row (interpolation between monotonic maps).
        let mut img = gray(256, 64, |x, _| 20 + (x / 8) as u8);
        apply(
            &mut img,
            Preprocess::Clahe {
                clip: 40.0,
                tiles: 1,
            },
        );
        let row: Vec<u8> = img.rgb[..256 * 3].chunks(3).map(|p| p[0]).collect();
        assert!(row.windows(2).all(|p| p[1] >= p[0]), "{row:?}");
        let (lo, hi) = (row[0], row[255]);
        assert!(hi - lo > 200, "{lo}..{hi}");
        // Gray stays gray.
        assert!(img.rgb.chunks(3).all(|p| p[0] == p[1] && p[1] == p[2]));
    }

    /// Straightforward float CLAHE on a gray image (same tile maps), for comparison.
    fn clahe_ref(img: &RgbImage, clip: f32, tiles: u8) -> Vec<u8> {
        let (w, h) = (img.width as usize, img.height as usize);
        let (tx, ty) = (usize::from(tiles).min(w), usize::from(tiles).min(h));
        let bound = |i: usize, len: usize, n: usize| i * len / n;
        let luma: Vec<u8> = img.rgb.chunks(3).map(|p| p[0]).collect();
        let mut maps = Vec::new();
        for j in 0..ty {
            for i in 0..tx {
                let mut hist = [0u32; 256];
                for y in bound(j, h, ty)..bound(j + 1, h, ty) {
                    for x in bound(i, w, tx)..bound(i + 1, w, tx) {
                        hist[usize::from(luma[y * w + x])] += 1;
                    }
                }
                let area = (bound(i + 1, w, tx) - bound(i, w, tx))
                    * (bound(j + 1, h, ty) - bound(j, h, ty));
                maps.push(tile_lut(&mut hist, area as u32, clip));
            }
        }
        // Tile index and weight of the next tile for a pixel centre `p` along an axis.
        let pick = |p: f32, len: usize, n: usize| -> (usize, usize, f32) {
            let c = |i: usize| (bound(i, len, n) + bound(i + 1, len, n)) as f32 / 2.0;
            if p <= c(0) {
                return (0, 0, 0.0);
            }
            if p >= c(n - 1) {
                return (n - 1, n - 1, 0.0);
            }
            let i = (0..n - 1).find(|&i| p < c(i + 1)).unwrap();
            (i, i + 1, (p - c(i)) / (c(i + 1) - c(i)))
        };
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            let (j1, j2, wy) = pick(y as f32 + 0.5, h, ty);
            for x in 0..w {
                let (i1, i2, wx) = pick(x as f32 + 0.5, w, tx);
                let v = usize::from(luma[y * w + x]);
                let m = |j: usize, i: usize| f32::from(maps[j * tx + i][v]);
                let top = m(j1, i1) * (1.0 - wx) + m(j1, i2) * wx;
                let bot = m(j2, i1) * (1.0 - wx) + m(j2, i2) * wx;
                out.push((top * (1.0 - wy) + bot * wy).round() as u8);
            }
        }
        out
    }

    #[test]
    fn clahe_matches_a_float_reference() {
        // Uneven tiles (101 x 67 into 4 x 4 and 8 x 8), varied content per region.
        let src = gray(101, 67, |x, y| {
            if x < 50 {
                ((x * 7 + y * 3) % 40) as u8 + 10
            } else {
                ((x * 13 + y * 5) % 120) as u8 + 120
            }
        });
        for (clip, tiles) in [(2.0, 4), (1.0, 8), (8.0, 3), (2.0, 1)] {
            let want = clahe_ref(&src, clip, tiles);
            let mut img = src.clone();
            apply(&mut img, Preprocess::Clahe { clip, tiles });
            let got: Vec<u8> = img.rgb.chunks(3).map(|p| p[0]).collect();
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            // Fixed-point (1/256) weights against f32: at most one level apart.
            assert!(worst <= 1, "clip {clip} tiles {tiles}: off by {worst}");
            assert!(img.rgb.chunks(3).all(|p| p[0] == p[1] && p[1] == p[2]));
        }
    }

    #[test]
    fn clahe_handles_tiny_and_short_buffers() {
        let mut one = gray(1, 1, |_, _| 77);
        apply(&mut one, Preprocess::CLAHE);
        assert_eq!(one.rgb.len(), 3);
        let mut empty = RgbImage::default();
        apply(&mut empty, Preprocess::CLAHE);
        // Buffer shorter than the header says: only whole rows are touched, no panic.
        let mut short = gray(10, 10, |x, _| x as u8 * 20);
        short.rgb.truncate(10 * 3 * 4 + 5);
        apply(&mut short, Preprocess::CLAHE);
        assert_eq!(short.rgb.len(), 10 * 3 * 4 + 5);
    }

    /// `cargo test --release --lib enhance::tests::bench_1080p -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_1080p() {
        let (w, h) = (1920u32, 1080u32);
        let colour = RgbImage {
            width: w,
            height: h,
            rgb: (0..w * h * 3)
                .map(|i| ((i.wrapping_mul(2654435761)) >> 24) as u8 / 4)
                .collect(),
        };
        let mono = gray(w, h, |x, y| ((x ^ y) & 63) as u8);
        for (name, src) in [("colour", &colour), ("mono", &mono)] {
            for mode in [
                Preprocess::CLAHE,
                Preprocess::Gamma(0.6),
                Preprocess::Grayscale,
            ] {
                let mut best = f64::MAX;
                for _ in 0..10 {
                    let mut img = src.clone();
                    let t = std::time::Instant::now();
                    apply(&mut img, mode);
                    best = best.min(t.elapsed().as_secs_f64() * 1e3);
                    std::hint::black_box(&img);
                }
                println!("{name} {mode}: {best:.2} ms (best of 10)");
            }
        }
    }
}
