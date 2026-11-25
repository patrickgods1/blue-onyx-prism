//! Cheap scene statistics of a decoded frame: mean luma and colour spread on a strided sample
//! of about [`SAMPLE_GRID`] x [`SAMPLE_GRID`] pixels, and the [`Scene`] they imply (day, night,
//! or IR monochrome). Pure; no full-image pass. The night/IR filters
//! ([`crate::model::filters`]) and the benchmark read the scene.
//!
//! Rules (mirrored by `scripts/build_bench_sets.py`; keep the constants in sync):
//! - **luma** is BT.601 (`0.299 R + 0.587 G + 0.114 B`, 0..255); the frame is dark when the
//!   sample mean is below [`DARK_LUMA_MAX`];
//! - **chroma_spread** is `mean(|R-G| + |G-B|) / 2` (0..255): about 0 for a gray frame;
//! - **tint_spread** handles tinted monochrome (green or purple IR, a white-balance cast):
//!   there the channel differences are not 0 but a function of brightness, an offset
//!   (`R-G = a`) or a gain (`R-G = b * luma`). For each of `R-G` and `G-B` it fits
//!   `d = a + b * luma` by least squares over the sample and takes the mean absolute residual;
//!   `tint_spread = (residual(R-G) + residual(G-B)) / 2`. A colour scene has hues that do not
//!   follow brightness, so its residual stays large;
//!   When the sample's luma is exactly flat (variance below [`FLAT_LUMA_VAR`], e.g. a camera's
//!   solid blue "no video" screen) every difference is constant, the fit is exact and the residual
//!   says nothing, so `tint_spread = chroma_spread` there (a flat gray frame stays IR, a flat
//!   colour one does not);
//! - the frame is **IR** when `chroma_spread < IR_CHROMA_MAX` or `tint_spread < IR_CHROMA_MAX`.
//!
//! [`ImageStats::scene`]: IR wins over dark (IR illuminators make a night frame bright).

use crate::image::RgbImage;
use serde::{Deserialize, Serialize};

/// Below this colour spread (0..255) a frame counts as monochrome (IR night mode).
pub const IR_CHROMA_MAX: f32 = 3.0;
/// Below this mean BT.601 luma (0..255) a frame counts as dark (`NIGHT_LUMA` of the bench
/// build script).
pub const DARK_LUMA_MAX: f32 = 50.0;
/// Below this luma variance (0..255 scale, squared) the sample is flat and the tint fit is
/// degenerate: `tint_spread` falls back to `chroma_spread`. Only an exactly uniform luma is under
/// it; a dark, nearly flat tinted-IR frame still has its fit (QA-045).
pub const FLAT_LUMA_VAR: f64 = 1e-6;
/// Samples per axis (at most; a smaller image is sampled at every pixel).
pub const SAMPLE_GRID: u32 = 64;

/// What kind of frame a request carried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scene {
    #[default]
    Day,
    /// Dark colour frame.
    Night,
    /// Monochrome (IR-lit) frame, at any brightness.
    Ir,
}

impl Scene {
    /// Every scene, in the order the dataset's scene folders are listed
    /// ([`crate::collect::layout::SCENES`]).
    pub const ALL: [Scene; 3] = [Scene::Day, Scene::Night, Scene::Ir];

    pub const fn as_str(self) -> &'static str {
        match self {
            Scene::Day => "day",
            Scene::Night => "night",
            Scene::Ir => "ir",
        }
    }

    /// The scene [`Self::as_str`] spells `s`; None for anything else (exact, lowercase).
    pub fn parse(s: &str) -> Option<Scene> {
        Self::ALL.into_iter().find(|c| c.as_str() == s)
    }

    /// Night or IR: where the night rules apply.
    pub fn is_night(self) -> bool {
        matches!(self, Scene::Night | Scene::Ir)
    }
}

impl std::fmt::Display for Scene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Statistics of one frame (see the module docs for the definitions).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ImageStats {
    /// Mean BT.601 luma of the sample, 0..255.
    pub mean_luma: f32,
    /// `mean(|R-G| + |G-B|) / 2`, 0..255.
    pub chroma_spread: f32,
    /// Mean absolute residual of the channel differences after a linear fit on luma (see the
    /// module docs), 0..255.
    pub tint_spread: f32,
    pub is_ir: bool,
    pub is_dark: bool,
}

/// Mean absolute residual of `d ≈ a + b * l` fitted by least squares.
fn fit_residual(l: &[f32], d: &[f32]) -> f32 {
    let n = l.len().min(d.len());
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let (mut sl, mut sd) = (0f64, 0f64);
    for (x, y) in l.iter().zip(d) {
        sl += f64::from(*x);
        sd += f64::from(*y);
    }
    let (ml, md) = (sl / nf, sd / nf);
    let (mut sll, mut sld) = (0f64, 0f64);
    for (x, y) in l.iter().zip(d) {
        let dx = f64::from(*x) - ml;
        sll += dx * dx;
        sld += dx * (f64::from(*y) - md);
    }
    // A flat frame (no luma variance) fits the offset only.
    let b = if sll > 1e-9 { sld / sll } else { 0.0 };
    let a = md - b * ml;
    let mut res = 0f64;
    for (x, y) in l.iter().zip(d) {
        res += (f64::from(*y) - (a + b * f64::from(*x))).abs();
    }
    #[allow(
        clippy::cast_possible_truncation,
        reason = "a mean of 0..255 values fits f32"
    )]
    let r = (res / nf) as f32;
    r
}

/// Sample positions along one axis of `len` pixels: the centres of `min(len, SAMPLE_GRID)`
/// equal cells.
fn axis(len: u32) -> impl Iterator<Item = usize> {
    let n = len.clamp(1, SAMPLE_GRID);
    let len = u64::from(len);
    (0..u64::from(n)).map(move |i| {
        let p = (2 * i + 1) * len / (2 * u64::from(n));
        usize::try_from(p).unwrap_or(usize::MAX)
    })
}

impl ImageStats {
    /// Statistics of `img` from a strided sample of at most `SAMPLE_GRID`² pixels. A frame with
    /// no pixels (or a buffer shorter than its size) yields the default (day, not IR).
    pub fn compute(img: &RgbImage) -> Self {
        let w = img.width as usize;
        let cap = (SAMPLE_GRID * SAMPLE_GRID) as usize;
        let (mut lum, mut rg, mut gb) = (
            Vec::with_capacity(cap),
            Vec::with_capacity(cap),
            Vec::with_capacity(cap),
        );
        let mut spread = 0f64;
        for y in axis(img.height) {
            for x in axis(img.width) {
                let i = y.saturating_mul(w).saturating_add(x).saturating_mul(3);
                let Some(&[r, g, b]) = img.rgb.get(i..i.saturating_add(3)) else {
                    continue;
                };
                let (r, g, b) = (f32::from(r), f32::from(g), f32::from(b));
                lum.push(0.299 * r + 0.587 * g + 0.114 * b);
                rg.push(r - g);
                gb.push(g - b);
                spread += f64::from((r - g).abs() + (g - b).abs());
            }
        }
        if lum.is_empty() {
            return Self::default();
        }
        let n = lum.len() as f64;
        let mean_luma = lum.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let luma_var = lum
            .iter()
            .map(|&v| (f64::from(v) - mean_luma).powi(2))
            .sum::<f64>()
            / n;
        #[allow(
            clippy::cast_possible_truncation,
            reason = "means of 0..255 values fit f32"
        )]
        let (mean_luma, chroma_spread) = (mean_luma as f32, (spread / n / 2.0) as f32);
        let tint_spread = if luma_var < FLAT_LUMA_VAR {
            chroma_spread
        } else {
            (fit_residual(&lum, &rg) + fit_residual(&lum, &gb)) / 2.0
        };
        Self {
            mean_luma,
            chroma_spread,
            tint_spread,
            is_ir: chroma_spread < IR_CHROMA_MAX || tint_spread < IR_CHROMA_MAX,
            is_dark: mean_luma < DARK_LUMA_MAX,
        }
    }

    /// `Ir` when monochrome (even if bright), else `Night` when dark, else `Day`.
    pub fn scene(&self) -> Scene {
        if self.is_ir {
            Scene::Ir
        } else if self.is_dark {
            Scene::Night
        } else {
            Scene::Day
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> RgbImage {
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                rgb.extend_from_slice(&f(x, y));
            }
        }
        RgbImage {
            width: w,
            height: h,
            rgb,
        }
    }

    /// A smooth gray ramp with a little deterministic noise.
    fn gray_level(x: u32, y: u32) -> u8 {
        let base = 40 + (x * 3 + y * 2) % 180;
        u8::try_from(base).unwrap_or(255)
    }

    /// A colourful scene: hue varies across the frame independently of brightness.
    fn colour(x: u32, y: u32) -> [u8; 3] {
        let r = (x * 7) % 256;
        let g = (y * 5 + 60) % 256;
        let b = ((x + y) * 3 + 120) % 256;
        [r as u8, g as u8, b as u8]
    }

    #[test]
    fn gray_frame_is_ir_even_when_bright() {
        let img = image(320, 240, |x, y| {
            let v = gray_level(x, y);
            [v, v, v]
        });
        let s = ImageStats::compute(&img);
        assert_eq!(s.chroma_spread, 0.0);
        assert!(s.mean_luma > DARK_LUMA_MAX, "{s:?}");
        assert!(s.is_ir && !s.is_dark);
        assert_eq!(s.scene(), Scene::Ir);
    }

    #[test]
    fn colour_frame_is_day() {
        let img = image(640, 480, colour);
        let s = ImageStats::compute(&img);
        assert!(s.chroma_spread > 20.0, "{s:?}");
        assert!(s.tint_spread > 20.0, "{s:?}");
        assert!(!s.is_ir && !s.is_dark);
        assert_eq!(s.scene(), Scene::Day);
    }

    #[test]
    fn dark_colour_frame_is_night() {
        let img = image(640, 480, |x, y| {
            let [r, g, b] = colour(x, y);
            [r / 8, g / 8, b / 8]
        });
        let s = ImageStats::compute(&img);
        assert!(s.mean_luma < DARK_LUMA_MAX, "{s:?}");
        assert!(!s.is_ir, "{s:?}");
        assert_eq!(s.scene(), Scene::Night);
    }

    #[test]
    fn green_tinted_monochrome_is_ir() {
        // Constant offset tint: G is 30 above R and B.
        let img = image(320, 240, |x, y| {
            let v = gray_level(x, y).min(200);
            [v, v + 30, v]
        });
        let s = ImageStats::compute(&img);
        assert!(
            s.chroma_spread >= IR_CHROMA_MAX,
            "raw spread sees the tint: {s:?}"
        );
        assert!(s.tint_spread < 0.01, "{s:?}");
        assert!(s.is_ir);
        // Gain tint: G = 1.3 * v, B = 0.8 * v (a white-balance cast scales with brightness).
        let img = image(320, 240, |x, y| {
            let v = f32::from(gray_level(x, y).min(190));
            [v as u8, (v * 1.3) as u8, (v * 0.8) as u8]
        });
        let s = ImageStats::compute(&img);
        assert!(s.chroma_spread >= IR_CHROMA_MAX, "{s:?}");
        assert!(s.tint_spread < 1.0, "{s:?}");
        assert_eq!(s.scene(), Scene::Ir);
    }

    #[test]
    fn dark_gray_is_ir_not_night() {
        let img = image(100, 80, |_, _| [10, 10, 10]);
        let s = ImageStats::compute(&img);
        assert!(s.is_dark && s.is_ir);
        assert_eq!(s.scene(), Scene::Ir);
    }

    // QA-045: a flat colour frame (a camera's blue "no video" screen) fitted exactly, so its
    // tint residual was 0 and it was filed as IR.
    #[test]
    fn flat_colour_frame_is_not_ir() {
        let blue = ImageStats::compute(&image(640, 480, |_, _| [0, 0, 255]));
        assert!((blue.chroma_spread - 127.5).abs() < 1e-3, "{blue:?}");
        assert_eq!(blue.tint_spread, blue.chroma_spread, "{blue:?}");
        assert!(!blue.is_ir, "{blue:?}");
        // Mean luma 29: dark.
        assert_eq!(blue.scene(), Scene::Night);
        let green = ImageStats::compute(&image(1, 1, |_, _| [0, 255, 0]));
        assert!(!green.is_ir, "{green:?}");
        assert_eq!(green.scene(), Scene::Day);
        // A flat gray frame is still IR (its chroma spread is 0).
        let gray = ImageStats::compute(&image(64, 64, |_, _| [128, 128, 128]));
        assert_eq!(gray.scene(), Scene::Ir);
    }

    #[test]
    fn luma_is_bt601() {
        let img = image(10, 10, |_, _| [100, 200, 50]);
        let s = ImageStats::compute(&img);
        let want = 0.299 * 100.0 + 0.587 * 200.0 + 0.114 * 50.0;
        assert!((s.mean_luma - want).abs() < 1e-3, "{s:?}");
        assert!((s.chroma_spread - 125.0).abs() < 1e-3, "{s:?}");
    }

    #[test]
    fn sample_is_bounded_and_small_images_work() {
        assert_eq!(axis(4000).count(), SAMPLE_GRID as usize);
        assert_eq!(axis(3).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert!(axis(4000).all(|p| p < 4000));
        let one = image(1, 1, |_, _| [0, 255, 0]);
        let s = ImageStats::compute(&one);
        assert!((s.chroma_spread - 255.0).abs() < 1e-3);
        // A buffer shorter than its size: no panic, default stats.
        let bad = RgbImage {
            width: 10,
            height: 10,
            rgb: Vec::new(),
        };
        assert_eq!(ImageStats::compute(&bad), ImageStats::default());
        assert_eq!(ImageStats::default().scene(), Scene::Day);
    }

    #[test]
    fn scene_serde_and_display() {
        assert_eq!(serde_json::to_string(&Scene::Ir).unwrap(), "\"ir\"");
        assert_eq!(
            serde_json::from_str::<Scene>("\"night\"").unwrap(),
            Scene::Night
        );
        assert_eq!(Scene::Day.to_string(), "day");
        assert!(Scene::Ir.is_night() && Scene::Night.is_night() && !Scene::Day.is_night());
    }

    #[test]
    fn scene_parse_is_the_inverse_of_as_str() {
        for s in Scene::ALL {
            assert_eq!(Scene::parse(s.as_str()), Some(s));
            // The serde spelling is the same one.
            assert_eq!(
                serde_json::to_string(&s).unwrap(),
                format!("\"{}\"", s.as_str())
            );
        }
        for bad in ["", "Day", "NIGHT", "dusk", " ir", "ir "] {
            assert_eq!(Scene::parse(bad), None, "{bad:?}");
        }
        assert_eq!(
            Scene::ALL.map(Scene::as_str),
            crate::collect::layout::SCENES
        );
    }

    /// The bench builder's twin of these rules (`scripts/build_bench_sets.py`).
    const BENCH_SCRIPT: &str = include_str!("../scripts/build_bench_sets.py");

    /// The value of the script's top-level `NAME = <number>` (a trailing comment ignored).
    fn py_number(name: &str) -> f64 {
        let prefix = format!("{name} = ");
        let line = BENCH_SCRIPT
            .lines()
            .find_map(|l| l.strip_prefix(&prefix))
            .unwrap_or_else(|| panic!("{name} not in build_bench_sets.py"));
        let value = line.split('#').next().unwrap_or_default().trim();
        value
            .parse()
            .unwrap_or_else(|e| panic!("{name} = {value:?}: {e}"))
    }

    // QA-159: the bench sets' day / night / IR tags come from the script, the service's scene from
    // here: the constants must agree.
    #[test]
    fn the_bench_script_has_the_same_constants() {
        assert_eq!(py_number("NIGHT_LUMA"), f64::from(DARK_LUMA_MAX));
        assert_eq!(py_number("GRAY_SPREAD"), f64::from(IR_CHROMA_MAX));
        assert_eq!(py_number("GRAY_GRID"), f64::from(SAMPLE_GRID));
        assert_eq!(py_number("FLAT_LUMA_VAR"), FLAT_LUMA_VAR);
    }

    /// Runs the script's `chroma_spread` on `images` with a stub `PIL` (the script needs Pillow only
    /// to open files; the rule reads `convert`, `size` and `load`), so no Pillow is needed.
    const PY_DRIVER: &str = r#"
import json, sys, types
text = sys.stdin.read()
script, images = text.split("\n#--IMAGES--\n")
class Img:
    def __init__(self, w, h, px):
        self.size = (w, h)
        self.px = px
    def convert(self, mode):
        assert mode == "RGB"
        return self
    def load(self):
        return self
    def __getitem__(self, xy):
        x, y = xy
        i = (y * self.size[0] + x) * 3
        return tuple(self.px[i:i + 3])
pil = types.ModuleType("PIL")
pil.Image = types.ModuleType("PIL.Image")
pil.ImageStat = types.ModuleType("PIL.ImageStat")
sys.modules["PIL"] = pil
sys.modules["PIL.Image"] = pil.Image
sys.modules["PIL.ImageStat"] = pil.ImageStat
ns = {"__name__": "build_bench_sets", "__file__": "scripts/build_bench_sets.py"}
exec(compile(script, "build_bench_sets.py", "exec"), ns)
out = [list(ns["chroma_spread"](Img(w, h, px))) for w, h, px in json.loads(images)]
print("RESULT: " + json.dumps(out))
"#;

    // QA-159: the script's chroma and tint spreads equal `ImageStats::compute`'s on generated
    // frames (grey ramp, offset and gain tints, colour, dark colour, flat colour, one pixel). Luma
    // is not compared: the script's night test is Pillow's whole-image "L" mean, not this sample.
    #[test]
    fn the_bench_script_computes_the_same_spreads() {
        let Some(python) = crate::training::progress::test_python::python3("bench image stats")
        else {
            return;
        };
        let images = [
            image(100, 70, |x, y| {
                let v = gray_level(x, y);
                [v, v, v]
            }),
            image(90, 40, |x, y| {
                let v = gray_level(x, y).min(200);
                [v, v + 30, v]
            }),
            image(80, 66, |x, y| {
                let v = f32::from(gray_level(x, y).min(190));
                [v as u8, (v * 1.3) as u8, (v * 0.8) as u8]
            }),
            image(130, 50, colour),
            image(70, 65, |x, y| {
                let [r, g, b] = colour(x, y);
                [r / 8, g / 8, b / 8]
            }),
            image(12, 9, |_, _| [0, 0, 255]),
            image(1, 1, |_, _| [0, 255, 0]),
        ];
        let json: Vec<serde_json::Value> = images
            .iter()
            .map(|i| serde_json::json!([i.width, i.height, i.rgb]))
            .collect();
        let input = format!(
            "{BENCH_SCRIPT}\n#--IMAGES--\n{}",
            serde_json::to_string(&json).unwrap()
        );
        let out =
            crate::training::progress::test_python::run(python, &["-I", "-c", PY_DRIVER], &input);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let line = stdout
            .lines()
            .find_map(|l| l.strip_prefix("RESULT: "))
            .unwrap_or_else(|| panic!("no result in {stdout}"));
        let py: Vec<[f64; 2]> = serde_json::from_str(line).unwrap();
        assert_eq!(py.len(), images.len());
        for (img, [chroma, tint]) in images.iter().zip(py) {
            let s = ImageStats::compute(img);
            let at = (img.width, img.height);
            assert!(
                (f64::from(s.chroma_spread) - chroma).abs() < 1e-4,
                "{at:?}: {s:?} vs {chroma}"
            );
            assert!(
                (f64::from(s.tint_spread) - tint).abs() < 1e-4,
                "{at:?}: {s:?} vs {tint}"
            );
        }
    }
}
