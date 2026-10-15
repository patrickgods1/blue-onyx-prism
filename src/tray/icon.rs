//! The tray icon, drawn at runtime (no asset files): a two-tone prism (diamond) glyph with a
//! status dot in the lower right corner, cut out of the glyph by a transparent ring so it reads
//! on light and dark menu bars. Pure RGBA output, anti-aliased by 4x4 supersampling in integer
//! math.

use crate::tray::model::IconColor;

/// Edge length the app renders (Windows scales it; macOS shows it at 16-22 pt).
pub const ICON_SIZE: u32 = 32;

const PRISM_LIGHT: [u8; 3] = [126, 154, 196];
const PRISM_DARK: [u8; 3] = [70, 95, 140];

/// RGB of a status dot.
pub fn dot_rgb(color: IconColor) -> [u8; 3] {
    match color {
        IconColor::Green => [52, 199, 89],
        IconColor::Amber => [255, 176, 0],
        IconColor::Red => [235, 64, 52],
        IconColor::Grey => [142, 142, 147],
    }
}

/// What one sample point hits.
fn sample(color: IconColor, size: i64, sx: i64, sy: i64) -> Option<[u8; 3]> {
    // Coordinates in 1/8 pixel units: the image spans 0..8*size.
    let (dx, dy) = (sx - 6 * size, sy - 6 * size);
    let dist2 = dx * dx + dy * dy;
    let dot_r = 2 * size; // 1/4 of the edge
    let ring = (3 * size / 8).max(8); // >= 1 px transparent gap around the dot
    if dist2 <= dot_r * dot_r {
        return Some(dot_rgb(color));
    }
    if dist2 <= (dot_r + ring) * (dot_r + ring) {
        return None;
    }
    // Diamond centered in the image, half-width 3/8, half-height 15/32 of the edge:
    // |x|/a + |y|/b <= 1 with a = 3S, b = 15S/4 (units) <=> 15|x| + 12|y| <= 45S.
    let (cx, cy) = (sx - 4 * size, sy - 4 * size);
    if 15 * cx.abs() + 12 * cy.abs() <= 45 * size {
        Some(if cx < 0 { PRISM_LIGHT } else { PRISM_DARK })
    } else {
        None
    }
}

/// RGBA pixels (row-major, `size * size * 4` bytes, straight alpha) of the icon.
pub fn icon_rgba(color: IconColor, size: u32) -> Vec<u8> {
    let s = i64::from(size);
    let mut out = Vec::with_capacity(usize::try_from(size * size * 4).unwrap_or(0));
    for y in 0..s {
        for x in 0..s {
            let (mut n, mut r, mut g, mut b) = (0u32, 0u32, 0u32, 0u32);
            for i in 0..4 {
                for j in 0..4 {
                    if let Some([pr, pg, pb]) =
                        sample(color, s, 8 * x + 1 + 2 * i, 8 * y + 1 + 2 * j)
                    {
                        n += 1;
                        r += u32::from(pr);
                        g += u32::from(pg);
                        b += u32::from(pb);
                    }
                }
            }
            let px = if n == 0 {
                [0, 0, 0, 0]
            } else {
                let avg = |v: u32| u8::try_from((v + n / 2) / n).unwrap_or(u8::MAX);
                let alpha = u8::try_from((n * 255 + 8) / 16).unwrap_or(u8::MAX);
                [avg(r), avg(g), avg(b), alpha]
            };
            out.extend_from_slice(&px);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn px(img: &[u8], size: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * size + x) * 4) as usize;
        [img[i], img[i + 1], img[i + 2], img[i + 3]]
    }

    #[test]
    fn dimensions_and_transparent_corners() {
        for size in [16, 22, 32, 64] {
            let img = icon_rgba(IconColor::Green, size);
            assert_eq!(img.len(), (size * size * 4) as usize);
            assert_eq!(px(&img, size, 0, 0)[3], 0);
            assert_eq!(px(&img, size, size - 1, 0)[3], 0);
            assert_eq!(px(&img, size, 0, size - 1)[3], 0);
            // The glyph covers the center fully.
            assert_eq!(px(&img, size, size / 2 - 2, size / 2 - 2)[3], 255);
        }
        assert!(icon_rgba(IconColor::Red, 0).is_empty());
    }

    #[test]
    fn dot_has_the_status_color() {
        for c in [
            IconColor::Green,
            IconColor::Amber,
            IconColor::Red,
            IconColor::Grey,
        ] {
            let img = icon_rgba(c, ICON_SIZE);
            let [r, g, b, a] = px(&img, ICON_SIZE, 24, 24);
            assert_eq!([r, g, b], dot_rgb(c), "{c:?}");
            assert_eq!(a, 255);
        }
    }

    #[test]
    fn prism_is_two_tone_and_ring_is_cut_out() {
        let img = icon_rgba(IconColor::Green, ICON_SIZE);
        assert_eq!(px(&img, 32, 12, 12), [126, 154, 196, 255]);
        assert_eq!(px(&img, 32, 19, 10), [70, 95, 140, 255]);
        // Between the dot (radius 8 px around 24,24, 1.5 px ring) and the glyph: transparent gap.
        let [_, _, _, a] = px(&img, 32, 17, 17);
        assert!(a < 255);
        let edge = px(&img, 32, 16, 1);
        assert!(edge[3] > 0 && edge[3] < 255, "anti-aliased tip {edge:?}");
    }

    #[test]
    fn colors_differ_only_in_the_dot() {
        let a = icon_rgba(IconColor::Green, 32);
        let b = icon_rgba(IconColor::Red, 32);
        assert_ne!(a, b);
        assert_eq!(px(&a, 32, 10, 16), px(&b, 32, 10, 16));
    }
}
