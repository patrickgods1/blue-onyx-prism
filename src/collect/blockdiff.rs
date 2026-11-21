//! Near-duplicate test of two frames of one fixed camera that a small object still tells apart
//! (offline evaluation only for now: `tests/it/dedup_eval.rs`; the collector still uses
//! [`super::dhash`]). Pure: thumbnails in, a [`Verdict`] out.
//!
//! What the evaluation found on 3,598 reviewed frames (2,087 pairs within 10 minutes): at the
//! defaults it called none of 1,009 labelled object changes a duplicate, where dHash at 6 called
//! 56% of them one (76% in imported footage), but it also called every one of 1,078 same-label
//! pairs different: at night these cameras' frames differ in rain, insects, IR flicker, foliage and
//! the clock overlay, which a pixel test cannot tell from a small animal. So the defaults are a
//! sensitivity ceiling, not a dedup: as a filter they keep every frame. The sweep's best point
//! (`z_cell` 5, `d_min` 8, groups of 3, texture masking 1) missed 6.8% for 68% kept repeats, where
//! dHash at 2 missed 20.7% for 64%.
//!
//! Why: the collector's whole-frame dHash averages 2688 x 1520 into 9 x 8 cells of about 300 x 190
//! pixels, so a 30 x 30 animal or a distant person in an otherwise identical night scene moves a
//! cell mean by well under one grey level and the frame is dropped as a duplicate. Here the grid is
//! 128 x 72 cells (about 21 x 21 pixels on 2688 x 1520), the scene's slow light changes are taken
//! out, and a frame differs when a *connected* group of cells changed by more than its noise.
//!
//! Steps and constants:
//!
//! - **Thumbnail** ([`Thumb`]): [`LONG_CELLS`] x [`SHORT_CELLS`] (128 x 72, the 16:9 of most
//!   cameras; transposed for a portrait frame, so cells stay near square) exact box means of the
//!   luma plane, rounded to u8, with the cell edges of [`dhash::cell_edges`] (a frame smaller than
//!   the grid repeats pixels). At 2688 x 1520 a 30 x 30 object covers at least four cells by about
//!   half (one whole cell when aligned), so its contrast reaches the cells nearly undiluted. JPEGs
//!   decode straight to luma ([`dhash::luma_of_encoded`]).
//! - **Detail layer**: `D = T - box9x9(T)`, the box blur over [`BLUR_RADIUS`] cells each way from an
//!   integral image with coordinates clamped at the edges (border cells repeat). It removes what
//!   is smooth over about 190 pixels: exposure offsets, IR fall-off, a headlight wash, dawn. It is
//!   kept exactly as `81 D`, an integer.
//! - **Gain fit**: auto exposure and IR gain scale the scene's contrast, so the new detail is
//!   matched to the reference's by least squares, `a = Σ D_new D_ref / Σ D_ref²` (`a = 1` when the
//!   reference has no detail at all), fitted again once without the cells whose residual exceeds
//!   [`OUTLIER_SIGMAS`] σ (an object must not pull the gain). Residual `R = D_new - a D_ref`.
//! - **Noise**: per pair `σ = max(SIGMA_FLOOR, MAD_TO_SIGMA · MAD(R))`, the MAD read from a
//!   histogram of `R` in 1/16 grey levels (O(cells), deterministic). [`MAD_TO_SIGMA`] (1.4826) makes
//!   it the standard deviation of Gaussian noise; the MAD ignores the object's own cells.
//!   [`SIGMA_FLOOR`] (0.75 grey levels): rounding each mean to u8 alone gives 0.29 per frame and
//!   0.41 per pair, JPEG adds more, so a very clean pair does not turn a 2-level flicker into a
//!   large z. A per-camera, per-cell map ([`NoiseMap`], optional) adds what one pair cannot see:
//!   cells that are always noisy (leaves, water, an IR-lit insect spot). It is an exponential mean
//!   with [`NOISE_ALPHA`] (1/16: about the last 16 frames) of `|R|` against the best-matching
//!   reference, updated by [`update_noise`], and read as `σ_c = MEAN_ABS_TO_SIGMA · mean|R|`
//!   (`sqrt(π/2)`, the ratio of σ to the mean absolute value of Gaussian noise). Optional texture
//!   masking ([`Params::texture_beta`], default [`TEXTURE_BETA`] = 0, off): a cell's σ is at least
//!   `β · texture`, the smaller of the [`TEXTURE_RADIUS`] (5 x 5 cell) means of `|a D_ref|` and
//!   `|D_new|`, so swaying foliage or flicker inside strong texture counts less, while an object
//!   that appears on (or leaves) a plain area keeps its full weight (the smaller of the two sides
//!   is the plain one). Off by default because the offline evaluation found it also hides small
//!   objects (β = 0.5 missed 2 of the 4 pairs whose object covers 0.05-0.5% of the frame in
//!   imported IR footage: one distant person) for a modest cut in repeats (100% to 96% kept). Score `z = R / max(σ, σ_c, β · texture)`.
//! - **Decision** ([`Residual::verdict`], [`Params`]): a cell changed when `|z| >= Z_CELL` and
//!   `|R| >= d_min` ([`D_MIN_DARK`] grey levels for night and IR, [`D_MIN_DAY`] by day, where JPEG
//!   and texture noise are larger in absolute terms). [`Z_CELL`] = 5: Gaussian noise passes it in
//!   about one cell of 1.7 million, one false cell in ~190 frames of 9,216 cells. Changed cells are
//!   grouped 8-connected; the frames differ when a group has [`MIN_COMPONENT`] (2) cells or more
//!   (a real object of 21 px or more nearly always straddles two cells, a lone flicker does not),
//!   or one cell reaches [`Z_SINGLE`] (10, a strong change too small for two cells). Weather rule:
//!   when more than [`WEATHER_SHARE`] (3%) of the cells changed and no group reaches
//!   [`WEATHER_COMPONENT`] (4) cells, rain, snow or sensor noise is scattered over the frame and
//!   only a group of 4 or more would count, so the frames are duplicates.
//! - **Duplicate** ([`duplicate_of`], [`is_duplicate`]): the new frame is a duplicate when any
//!   reference, newest first, finds no qualifying group (the first match stops the search). The
//!   caller keeps references per camera and scene; thumbnails of another grid never match.

use super::dhash;
use crate::image_stats::Scene;
use anyhow::{Result, anyhow};

/// Cells along the long side of a [`Thumb`].
pub const LONG_CELLS: usize = 128;
/// Cells along the short side of a [`Thumb`].
pub const SHORT_CELLS: usize = 72;
/// Half-width of the box blur in cells: a 9 x 9 cell window.
pub const BLUR_RADIUS: usize = 4;
/// Side of the blur window.
const BLUR_SIDE: usize = 2 * BLUR_RADIUS + 1;
/// Cells in the blur window (`BLUR_SIDE²`): the detail layer is kept multiplied by it.
const BLUR_AREA: i32 = 81;
/// Smallest per-pair noise σ, in grey levels.
pub const SIGMA_FLOOR: f32 = 0.75;
/// MAD to σ of Gaussian noise.
pub const MAD_TO_SIGMA: f32 = 1.4826;
/// Cells with `|R|` above this many σ are left out of the second gain fit.
pub const OUTLIER_SIGMAS: f32 = 3.0;
/// Weight of a new residual in the per-cell noise map.
pub const NOISE_ALPHA: f32 = 1.0 / 16.0;
/// σ over the mean absolute value of Gaussian noise, `sqrt(π/2)`.
pub const MEAN_ABS_TO_SIGMA: f32 = 1.253_314_1;
/// `|z|` at or above which a cell changed.
pub const Z_CELL: f32 = 5.0;
/// `|z|` at or above which one changed cell alone makes the frames differ.
pub const Z_SINGLE: f32 = 10.0;
/// Smallest `|R|` of a changed cell at night and in IR (grey levels).
pub const D_MIN_DARK: f32 = 4.0;
/// Smallest `|R|` of a changed cell by day (grey levels).
pub const D_MIN_DAY: f32 = 6.0;
/// Cells of a group of changed cells that make the frames differ.
pub const MIN_COMPONENT: usize = 2;
/// Share of changed cells above which the weather rule applies.
pub const WEATHER_SHARE: f32 = 0.03;
/// Cells of a group that still count under the weather rule.
pub const WEATHER_COMPONENT: usize = 4;
/// Texture masking off by default (see the module docs).
pub const TEXTURE_BETA: f32 = 0.0;
/// Half-width of the texture window: 5 x 5 cells.
pub const TEXTURE_RADIUS: usize = 2;
/// Histogram bins per grey level of the MAD.
const HIST_STEPS: f32 = 16.0;
/// Bins on each side of zero: residuals within ±512 grey levels (the detail layer is within
/// ±255, a gain can stretch it; beyond is clamped into the end bins).
const HIST_HALF: usize = 512 * 16;

/// A frame reduced to the grid of exact luma box means (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Thumb {
    cols: usize,
    rows: usize,
    /// Row-major, `cols * rows` means.
    cells: Vec<u8>,
}

impl Thumb {
    /// The grid `(cols, rows)` of a `w` x `h` frame: 128 x 72, or 72 x 128 for a portrait one.
    pub fn grid(w: usize, h: usize) -> (usize, usize) {
        if h > w {
            (SHORT_CELLS, LONG_CELLS)
        } else {
            (LONG_CELLS, SHORT_CELLS)
        }
    }

    /// The thumbnail of a `w` x `h` luma plane (`luma.len() >= w * h`); None for an empty one.
    /// Each cell is the mean of its pixels rounded half up.
    pub fn of_luma(luma: &[u8], w: usize, h: usize) -> Option<Self> {
        if w == 0 || h == 0 || luma.len() < w.checked_mul(h)? {
            return None;
        }
        let (cols, rows) = Self::grid(w, h);
        let xs = dhash::cell_edges(w, cols);
        let ys = dhash::cell_edges(h, rows);
        let mut colsum = vec![0u64; w];
        let mut cells = Vec::with_capacity(cols * rows);
        for &(y0, y1) in &ys {
            colsum.fill(0);
            for y in y0..y1 {
                let row = luma.get(y * w..(y + 1) * w)?;
                for (s, &v) in colsum.iter_mut().zip(row) {
                    *s += u64::from(v);
                }
            }
            for &(x0, x1) in &xs {
                let sum: u64 = colsum.get(x0..x1)?.iter().sum();
                let n = u64::try_from((y1 - y0) * (x1 - x0)).ok()?.max(1);
                cells.push(u8::try_from((2 * sum + n) / (2 * n)).unwrap_or(u8::MAX));
            }
        }
        Some(Self { cols, rows, cells })
    }

    /// The thumbnail of a packed RGB8 image (BT.601 luma).
    pub fn of_rgb(img: &crate::image::RgbImage) -> Option<Self> {
        Self::of_luma(
            &dhash::luma_of_rgb(img),
            img.width as usize,
            img.height as usize,
        )
    }

    /// The thumbnail and size of an encoded image ([`dhash::luma_of_encoded`]).
    pub fn of_encoded(bytes: &[u8]) -> Result<(Self, u32, u32)> {
        let (luma, w, h) = dhash::luma_of_encoded(bytes)?;
        let thumb = Self::of_luma(&luma, w, h).ok_or_else(|| anyhow!("the image is empty"))?;
        Ok((
            thumb,
            u32::try_from(w).map_err(|_| anyhow!("width out of range"))?,
            u32::try_from(h).map_err(|_| anyhow!("height out of range"))?,
        ))
    }

    /// A thumbnail of given means (row-major); None unless `cells.len() == cols * rows > 0`.
    pub fn from_cells(cols: usize, rows: usize, cells: Vec<u8>) -> Option<Self> {
        (cols > 0 && rows > 0 && cells.len() == cols.checked_mul(rows)?).then_some(Self {
            cols,
            rows,
            cells,
        })
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The means, row-major.
    pub fn cells(&self) -> &[u8] {
        &self.cells
    }

    /// `81 · D`: each cell minus the mean of the 9 x 9 cells around it, edges clamped. Exact.
    #[allow(
        clippy::indexing_slicing,
        reason = "indices stay within the padded grid and the thumbnail, both checked at \
                  construction (cols, rows >= 1, cells.len() == cols * rows)"
    )]
    fn detail81(&self) -> Vec<i32> {
        let (c, r) = (self.cols, self.rows);
        let (pc, pr) = (c + 2 * BLUR_RADIUS, r + 2 * BLUR_RADIUS);
        let stride = pc + 1;
        // Integral image of the grid padded by BLUR_RADIUS repeated border cells.
        let mut ii = vec![0u32; stride * (pr + 1)];
        for y in 0..pr {
            let sy = y.saturating_sub(BLUR_RADIUS).min(r - 1);
            let mut run = 0u32;
            for x in 0..pc {
                let sx = x.saturating_sub(BLUR_RADIUS).min(c - 1);
                run += u32::from(self.cells[sy * c + sx]);
                ii[(y + 1) * stride + x + 1] = ii[y * stride + x + 1] + run;
            }
        }
        let mut d = Vec::with_capacity(c * r);
        for y in 0..r {
            for x in 0..c {
                // Padded window [x, x + 9) x [y, y + 9) is centred on cell (x, y).
                let (x1, y1) = (x + BLUR_SIDE, y + BLUR_SIDE);
                let s = (ii[y1 * stride + x1] - ii[y * stride + x1])
                    - (ii[y1 * stride + x] - ii[y * stride + x]);
                let s = i32::try_from(s).unwrap_or(i32::MAX);
                d.push(BLUR_AREA * i32::from(self.cells[y * c + x]) - s);
            }
        }
        d
    }
}

/// f64 to f32 for values within a few thousand.
#[allow(
    clippy::cast_possible_truncation,
    reason = "residuals and gains are small, well inside f32's range"
)]
fn narrow(x: f64) -> f32 {
    x as f32
}

/// Least-squares gain of `dn` on `dr` over the cells `keep` allows; 1 without reference detail.
fn gain_fit(dn: &[i32], dr: &[i32], keep: impl Fn(usize) -> bool) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (i, (&n, &r)) in dn.iter().zip(dr).enumerate() {
        if keep(i) {
            num += f64::from(n) * f64::from(r);
            den += f64::from(r) * f64::from(r);
        }
    }
    if den > 0.0 { num / den } else { 1.0 }
}

/// `R = (dn - a dr) / 81` in grey levels.
fn residuals(dn: &[i32], dr: &[i32], a: f64) -> Vec<f32> {
    dn.iter()
        .zip(dr)
        .map(|(&n, &r)| narrow((f64::from(n) - a * f64::from(r)) / f64::from(BLUR_AREA)))
        .collect()
}

/// The mean of each cell's `(2 radius + 1)²` window of a `cols` x `rows` grid, coordinates
/// clamped at the edges (an integral image in f64, so the result does not depend on the order of
/// the sums).
#[allow(
    clippy::indexing_slicing,
    reason = "indices stay within the padded grid and `v` (checked: v.len() == cols * rows)"
)]
fn box_mean(v: &[f32], cols: usize, rows: usize, radius: usize) -> Vec<f32> {
    if cols == 0 || rows == 0 || v.len() != cols * rows {
        return vec![0.0; v.len()];
    }
    let side = 2 * radius + 1;
    let (pc, pr) = (cols + 2 * radius, rows + 2 * radius);
    let stride = pc + 1;
    let mut ii = vec![0f64; stride * (pr + 1)];
    for y in 0..pr {
        let sy = y.saturating_sub(radius).min(rows - 1);
        let mut run = 0f64;
        for x in 0..pc {
            let sx = x.saturating_sub(radius).min(cols - 1);
            run += f64::from(v[sy * cols + sx]);
            ii[(y + 1) * stride + x + 1] = ii[y * stride + x + 1] + run;
        }
    }
    let area = (side * side) as f64;
    let mut out = Vec::with_capacity(v.len());
    for y in 0..rows {
        for x in 0..cols {
            let (x1, y1) = (x + side, y + side);
            let s = ii[y1 * stride + x1] - ii[y * stride + x1] - ii[y1 * stride + x]
                + ii[y * stride + x];
            out.push(narrow(s / area));
        }
    }
    out
}

/// The histogram bin of a value: 1/16 grey level, clamped to ±512 levels.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "clamped to 0..2 * HIST_HALF before the cast"
)]
fn bin(v: f32) -> usize {
    let half = HIST_HALF as f32;
    ((v * HIST_STEPS).round().clamp(-half, half - 1.0) + half) as usize
}

/// The lower median bin of a histogram of `n > 0` values.
fn hist_median(hist: &[u32], n: usize) -> usize {
    let k = n.div_ceil(2);
    let mut seen = 0usize;
    for (i, &c) in hist.iter().enumerate() {
        seen += c as usize;
        if seen >= k {
            return i;
        }
    }
    hist.len().saturating_sub(1)
}

/// `1.4826 · MAD(r)` in grey levels (histogram of 1/16 levels), at least [`SIGMA_FLOOR`].
pub fn robust_sigma(r: &[f32]) -> f32 {
    if r.is_empty() {
        return SIGMA_FLOOR;
    }
    let mut hist = vec![0u32; 2 * HIST_HALF];
    for &v in r {
        if let Some(h) = hist.get_mut(bin(v)) {
            *h += 1;
        }
    }
    let med = hist_median(&hist, r.len());
    let mut dev = vec![0u32; 2 * HIST_HALF];
    for (i, &c) in hist.iter().enumerate() {
        if c > 0
            && let Some(d) = dev.get_mut(i.abs_diff(med))
        {
            *d += c;
        }
    }
    let mad = hist_median(&dev, r.len()) as f32 / HIST_STEPS;
    (MAD_TO_SIGMA * mad).max(SIGMA_FLOOR)
}

/// The decision thresholds (defaults: the module's constants).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    pub z_cell: f32,
    pub z_single: f32,
    /// `d_min` at night and in IR.
    pub d_min_dark: f32,
    /// `d_min` by day.
    pub d_min_day: f32,
    /// Cells of a group that make the frames differ (0 counts as 1).
    pub min_component: usize,
    pub weather_share: f32,
    pub weather_component: usize,
    /// Texture masking: a cell's σ is at least this times its local texture (0 = off).
    pub texture_beta: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            z_cell: Z_CELL,
            z_single: Z_SINGLE,
            d_min_dark: D_MIN_DARK,
            d_min_day: D_MIN_DAY,
            min_component: MIN_COMPONENT,
            weather_share: WEATHER_SHARE,
            weather_component: WEATHER_COMPONENT,
            texture_beta: TEXTURE_BETA,
        }
    }
}

impl Params {
    /// `d_min` of a scene.
    pub fn d_min(&self, scene: Scene) -> f32 {
        if scene.is_night() {
            self.d_min_dark
        } else {
            self.d_min_day
        }
    }
}

/// What the comparison of two frames found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Verdict {
    /// The frames differ: a group qualified (see the module docs).
    pub different: bool,
    /// Groups of changed cells (8-connected).
    pub blobs: usize,
    /// Cells of the largest group.
    pub largest_blob: usize,
    /// Largest `|z|` of any cell, changed or not.
    pub max_z: f32,
    pub changed_cells: usize,
    /// The weather rule applied (many scattered changed cells, no group of
    /// [`WEATHER_COMPONENT`]).
    pub weather: bool,
}

impl Verdict {
    /// Thumbnails of different grids: never duplicates.
    pub const MISMATCH: Verdict = Verdict {
        different: true,
        blobs: 0,
        largest_blob: 0,
        max_z: 0.0,
        changed_cells: 0,
        weather: false,
    };
}

/// The residual detail of a new frame against a reference, with its per-pair noise.
#[derive(Debug, Clone, PartialEq)]
pub struct Residual {
    cols: usize,
    rows: usize,
    /// The fitted gain `a`.
    pub gain: f32,
    /// The per-pair noise σ (grey levels).
    pub sigma: f32,
    /// `R` per cell, row-major (grey levels).
    r: Vec<f32>,
    /// Local texture per cell: the smaller of the 5 x 5 means of `|a D_ref|` and `|D_new|`
    /// (grey levels), for [`Params::texture_beta`].
    texture: Vec<f32>,
}

impl Residual {
    /// `R` of `new` against `reference`; None when their grids differ.
    pub fn of(new: &Thumb, reference: &Thumb) -> Option<Self> {
        if new.cols != reference.cols || new.rows != reference.rows {
            return None;
        }
        let (dn, dr) = (new.detail81(), reference.detail81());
        let a0 = gain_fit(&dn, &dr, |_| true);
        let r0 = residuals(&dn, &dr, a0);
        let limit = OUTLIER_SIGMAS * robust_sigma(&r0);
        let a = gain_fit(&dn, &dr, |i| r0.get(i).is_some_and(|v| v.abs() <= limit));
        let r = residuals(&dn, &dr, a);
        let (c, rows) = (new.cols, new.rows);
        let scale = a.abs() / f64::from(BLUR_AREA);
        let abs_ref: Vec<f32> = dr
            .iter()
            .map(|&v| narrow(scale * f64::from(v.unsigned_abs())))
            .collect();
        let abs_new: Vec<f32> = dn
            .iter()
            .map(|&v| narrow(f64::from(v.unsigned_abs()) / f64::from(BLUR_AREA)))
            .collect();
        let texture = box_mean(&abs_ref, c, rows, TEXTURE_RADIUS)
            .into_iter()
            .zip(box_mean(&abs_new, c, rows, TEXTURE_RADIUS))
            .map(|(x, y)| x.min(y))
            .collect();
        Some(Self {
            cols: c,
            rows,
            gain: narrow(a),
            sigma: robust_sigma(&r),
            r,
            texture,
        })
    }

    /// `R` per cell, row-major.
    pub fn values(&self) -> &[f32] {
        &self.r
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The decision for a frame of `scene` (`sigma_c`: the camera's noise map, ignored when its
    /// grid differs).
    pub fn verdict(&self, scene: Scene, sigma_c: Option<&NoiseMap>, p: &Params) -> Verdict {
        let d_min = p.d_min(scene);
        let map = sigma_c.filter(|m| m.cols == self.cols && m.rows == self.rows);
        let mut changed = vec![false; self.r.len()];
        let (mut max_z, mut max_z_changed, mut count) = (0f32, 0f32, 0usize);
        for (i, (&r, c)) in self.r.iter().zip(changed.iter_mut()).enumerate() {
            let tex = p.texture_beta * self.texture.get(i).copied().unwrap_or(0.0);
            let s = map
                .and_then(|m| m.sigma_at(i))
                .map_or(self.sigma, |sc| sc.max(self.sigma))
                .max(tex);
            let z = (r / s).abs();
            max_z = max_z.max(z);
            if z >= p.z_cell && r.abs() >= d_min {
                *c = true;
                count += 1;
                max_z_changed = max_z_changed.max(z);
            }
        }
        let (blobs, largest) = components(&changed, self.cols, self.rows);
        let weather =
            count as f32 > p.weather_share * self.r.len() as f32 && largest < p.weather_component;
        let different = !weather
            && count > 0
            && (largest >= p.min_component.max(1) || max_z_changed >= p.z_single);
        Verdict {
            different,
            blobs,
            largest_blob: largest,
            max_z,
            changed_cells: count,
            weather,
        }
    }
}

/// Groups (8-connected) of the `true` cells of a `cols` x `rows` grid and the largest one's size.
#[allow(
    clippy::indexing_slicing,
    reason = "indices are cells of the cols x rows grid, changed.len() == cols * rows"
)]
fn components(changed: &[bool], cols: usize, rows: usize) -> (usize, usize) {
    if cols == 0 || rows == 0 || changed.len() != cols * rows {
        return (0, 0);
    }
    let mut seen = vec![false; changed.len()];
    let mut stack = Vec::new();
    let (mut blobs, mut largest) = (0usize, 0usize);
    for start in 0..changed.len() {
        if !changed[start] || seen[start] {
            continue;
        }
        blobs += 1;
        seen[start] = true;
        stack.push(start);
        let mut size = 0usize;
        while let Some(i) = stack.pop() {
            size += 1;
            let (x, y) = (i % cols, i / cols);
            for ny in y.saturating_sub(1)..=(y + 1).min(rows - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(cols - 1) {
                    let j = ny * cols + nx;
                    if changed[j] && !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        largest = largest.max(size);
    }
    (blobs, largest)
}

/// A camera's per-cell noise: the exponential mean of `|R|` (see the module docs).
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseMap {
    cols: usize,
    rows: usize,
    mean_abs: Vec<f32>,
    updates: u32,
}

impl NoiseMap {
    /// A map of zeros: until it learns, `σ_c` never raises the per-pair σ.
    pub fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            rows,
            mean_abs: vec![0.0; cols * rows],
            updates: 0,
        }
    }

    /// A map of `thumb`'s grid.
    pub fn for_thumb(thumb: &Thumb) -> Self {
        Self::new(thumb.cols, thumb.rows)
    }

    /// `σ_c` of cell `i` (row-major).
    pub fn sigma_at(&self, i: usize) -> Option<f32> {
        self.mean_abs.get(i).map(|m| m * MEAN_ABS_TO_SIGMA)
    }

    /// Residuals folded in so far.
    pub fn updates(&self) -> u32 {
        self.updates
    }
}

/// Fold a residual (against the best-matching reference) into the map:
/// `m += NOISE_ALPHA (|R| - m)` per cell. False (and nothing changed) when the grids differ.
pub fn update_noise(map: &mut NoiseMap, residual: &Residual) -> bool {
    if map.cols != residual.cols || map.rows != residual.rows {
        return false;
    }
    for (m, r) in map.mean_abs.iter_mut().zip(&residual.r) {
        *m += NOISE_ALPHA * (r.abs() - *m);
    }
    map.updates = map.updates.saturating_add(1);
    true
}

/// [`verdict_with`] at the default [`Params`].
pub fn verdict(
    new: &Thumb,
    reference: &Thumb,
    scene: Scene,
    sigma_c: Option<&NoiseMap>,
) -> Verdict {
    verdict_with(new, reference, scene, sigma_c, &Params::default())
}

/// Whether `new` differs from `reference` ([`Residual::verdict`]); [`Verdict::MISMATCH`] for
/// thumbnails of different grids.
pub fn verdict_with(
    new: &Thumb,
    reference: &Thumb,
    scene: Scene,
    sigma_c: Option<&NoiseMap>,
    params: &Params,
) -> Verdict {
    Residual::of(new, reference).map_or(Verdict::MISMATCH, |r| r.verdict(scene, sigma_c, params))
}

/// The index of the first reference (newest first) `new` does not differ from, if any.
pub fn duplicate_of<'a>(
    new: &Thumb,
    refs_newest_first: impl IntoIterator<Item = &'a Thumb>,
    scene: Scene,
    sigma_c: Option<&NoiseMap>,
    params: &Params,
) -> Option<usize> {
    refs_newest_first
        .into_iter()
        .position(|r| !verdict_with(new, r, scene, sigma_c, params).different)
}

/// `new` is a near duplicate of one of the references ([`duplicate_of`], default [`Params`]).
pub fn is_duplicate<'a>(
    new: &Thumb,
    refs_newest_first: impl IntoIterator<Item = &'a Thumb>,
    scene: Scene,
    sigma_c: Option<&NoiseMap>,
) -> bool {
    duplicate_of(new, refs_newest_first, scene, sigma_c, &Params::default()).is_some()
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;
    use crate::image::RgbImage;

    /// splitmix64.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform in [0, 1).
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
        /// Near-Gaussian, σ = 1: the sum of four 16-bit uniforms of one draw.
        fn gauss(&mut self) -> f32 {
            let v = self.next();
            let s: f32 = (0..4)
                .map(|k| ((v >> (16 * k)) & 0xFFFF) as f32 / 65535.0 - 0.5)
                .sum();
            // Var of U(-0.5, 0.5) is 1/12, of the sum 1/3.
            s * 3f32.sqrt()
        }
    }

    /// A scene of `w` x `h`: a ramp, 60 rectangles of grey in `lo..hi` and an 8 px block texture.
    fn make_scene(w: usize, h: usize, seed: u64, lo: f32, hi: f32) -> Vec<f32> {
        let mut rng = Rng(seed);
        let mut px: Vec<f32> = (0..w * h)
            .map(|i| lo + (hi - lo) * 0.3 * (i % w) as f32 / w as f32)
            .collect();
        for _ in 0..60 {
            let rw = ((0.05 + 0.2 * rng.unit()) * w as f32) as usize + 1;
            let rh = ((0.05 + 0.2 * rng.unit()) * h as f32) as usize + 1;
            let x0 = (rng.unit() * w as f32) as usize;
            let y0 = (rng.unit() * h as f32) as usize;
            let v = lo + (hi - lo) * rng.unit();
            for y in y0..(y0 + rh).min(h) {
                px[y * w + x0..y * w + (x0 + rw).min(w)].fill(v);
            }
        }
        for (i, p) in px.iter_mut().enumerate() {
            let (bx, by) = ((i % w) / 8, (i / w) / 8);
            let t = (bx.wrapping_mul(73_856_093) ^ by.wrapping_mul(19_349_663)) % 7;
            *p += t as f32 - 3.0;
        }
        px
    }

    /// `base` with per-pixel noise of `sigma`, rounded and clamped to u8.
    fn render(base: &[f32], sigma: f32, seed: u64) -> Vec<u8> {
        let mut rng = Rng(seed);
        base.iter()
            .map(|v| (v + sigma * rng.gauss()).round().clamp(0.0, 255.0) as u8)
            .collect()
    }

    /// `px` with a `bw` x `bh` rectangle at `(x0, y0)` shifted by `delta`.
    fn with_blob(
        px: &[f32],
        w: usize,
        (x0, y0, bw, bh): (usize, usize, usize, usize),
        delta: f32,
    ) -> Vec<f32> {
        let mut out = px.to_vec();
        for y in y0..y0 + bh {
            for x in x0..x0 + bw {
                out[y * w + x] += delta;
            }
        }
        out
    }

    fn thumb(px: &[u8], w: usize, h: usize) -> Thumb {
        Thumb::of_luma(px, w, h).unwrap()
    }

    const W: usize = 2688;
    const H: usize = 1520;

    #[test]
    fn thumb_cells_are_exact_rounded_box_means() {
        // 256 x 144: every cell is 2 x 2 pixels.
        let (w, h) = (256, 144);
        let mut px = vec![0u8; w * h];
        px[0] = 1;
        px[1] = 2;
        px[w] = 2;
        px[w + 1] = 2; // mean 1.75 -> 2
        px[2] = 1; // cell (1, 0): mean 0.25 -> 0
        px[4] = 2; // cell (2, 0): mean 0.5 -> 1 (half up)
        let t = thumb(&px, w, h);
        assert_eq!((t.cols(), t.rows()), (128, 72));
        assert_eq!(&t.cells()[..3], &[2, 0, 1]);
        assert_eq!(Thumb::grid(1520, 2688), (72, 128));
        assert_eq!(BLUR_AREA as usize, BLUR_SIDE * BLUR_SIDE);
        let img = RgbImage {
            width: 4,
            height: 2,
            rgb: vec![200; 24],
        };
        assert!(
            Thumb::of_rgb(&img)
                .unwrap()
                .cells()
                .iter()
                .all(|&v| v == 200)
        );
    }

    #[test]
    fn detail_of_a_flat_or_linear_grid_is_zero_inside() {
        let flat = Thumb::from_cells(128, 72, vec![90; 128 * 72]).unwrap();
        assert!(flat.detail81().iter().all(|&d| d == 0));
        let ramp: Vec<u8> = (0..128 * 72).map(|i| (i % 128) as u8).collect();
        let d = Thumb::from_cells(128, 72, ramp).unwrap().detail81();
        // Interior columns: the window mean equals the centre on a linear ramp.
        assert!((0..72).all(|y| (4..124).all(|x| d[y * 128 + x] == 0)));
        // Clamped edge: column 0 sees 0 four extra times: 81 * 0 - 9 * (0*5 + 1 + 2 + 3 + 4).
        assert_eq!(d[0], -90);
        assert!(Thumb::from_cells(2, 2, vec![0; 3]).is_none());
    }

    #[test]
    fn robust_sigma_reads_the_mad() {
        // -2, -1, 0, 1, 2 equally often and a few outliers: median 0, MAD 1 -> σ 1.4826.
        let mut r: Vec<f32> = (0..1000).map(|i| (i % 5) as f32 - 2.0).collect();
        r.extend([500.0, -300.0, 90.0]);
        let s = robust_sigma(&r);
        assert!((s - MAD_TO_SIGMA).abs() < 1e-4, "{s}");
        assert_eq!(robust_sigma(&[0.0; 50]), SIGMA_FLOOR);
        assert_eq!(robust_sigma(&[]), SIGMA_FLOOR);
        // Gaussian noise of σ 3: within 5%.
        let mut rng = Rng(7);
        let g: Vec<f32> = (0..9216).map(|_| 3.0 * rng.gauss()).collect();
        let s = robust_sigma(&g);
        assert!((s - 3.0).abs() < 0.15, "{s}");
    }

    #[test]
    fn static_scene_with_noise_is_a_duplicate() {
        for (scene, lo, hi, noise) in [(Scene::Day, 60.0, 200.0, 4.0), (Scene::Ir, 30.0, 70.0, 6.0)]
        {
            let base = make_scene(W, H, 1, lo, hi);
            let a = thumb(&render(&base, noise, 10), W, H);
            let b = thumb(&render(&base, noise, 11), W, H);
            let v = verdict(&b, &a, scene, None);
            assert!(!v.different, "{scene}: {v:?}");
            assert!(is_duplicate(&b, [&a], scene, None));
        }
    }

    #[test]
    fn a_global_gain_or_offset_is_a_duplicate() {
        let base = make_scene(W, H, 2, 50.0, 160.0);
        let a = thumb(&render(&base, 3.0, 20), W, H);
        for (gain, offset) in [(1.25, 12.0), (0.8, -5.0), (1.0, 30.0)] {
            let changed: Vec<f32> = base.iter().map(|v| v * gain + offset).collect();
            let b = thumb(&render(&changed, 3.0, 21), W, H);
            let r = Residual::of(&b, &a).unwrap();
            assert!((r.gain - gain).abs() < 0.03, "gain {} for {gain}", r.gain);
            for scene in [Scene::Day, Scene::Night] {
                let v = r.verdict(scene, None, &Params::default());
                assert!(!v.different, "{gain} {offset} {scene}: {v:?}");
            }
        }
    }

    #[test]
    fn a_30px_blob_is_different_at_night_and_by_day() {
        // Positions: inside one cell column, straddling four cells, at the frame's edge.
        let spots = [
            (1000, 700),
            (1013, 728),
            (0, 0),
            (W - 30, H - 30),
            (2100, 300),
        ];
        for (scene, lo, hi, noise, delta) in [
            (Scene::Ir, 40.0, 70.0, 4.0, 30.0),
            (Scene::Ir, 40.0, 70.0, 4.0, -30.0),
            (Scene::Night, 40.0, 70.0, 4.0, 30.0),
            (Scene::Day, 60.0, 200.0, 4.0, 40.0),
            (Scene::Day, 60.0, 200.0, 4.0, -40.0),
        ] {
            let base = make_scene(W, H, 3, lo, hi);
            let a = thumb(&render(&base, noise, 30), W, H);
            for (k, &(x, y)) in spots.iter().enumerate() {
                let b = thumb(
                    &render(
                        &with_blob(&base, W, (x, y, 30, 30), delta),
                        noise,
                        31 + k as u64,
                    ),
                    W,
                    H,
                );
                let v = verdict(&b, &a, scene, None);
                assert!(v.different, "{scene} {delta} at {x},{y}: {v:?}");
                assert!(!is_duplicate(&b, [&a], scene, None));
            }
        }
    }

    /// Snow: a 5 x 5 flake of `+amp` inside every third cell each way (no two flakes touch).
    fn snow(base: &[f32], amp: f32, seed: u64) -> Vec<f32> {
        let mut rng = Rng(seed);
        let mut out = base.to_vec();
        let xs = dhash::cell_edges(W, 128);
        let ys = dhash::cell_edges(H, 72);
        for &(y0, _) in ys.iter().step_by(3) {
            for &(x0, _) in xs.iter().step_by(3) {
                let fx = x0 + 3 + (rng.unit() * 11.0) as usize;
                let fy = y0 + 3 + (rng.unit() * 11.0) as usize;
                let a = amp * (0.7 + 0.6 * rng.unit());
                for y in fy..fy + 5 {
                    for x in fx..fx + 5 {
                        out[y * W + x] += a;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn scattered_speckles_are_a_duplicate_but_a_person_in_them_is_not() {
        let base = make_scene(W, H, 4, 40.0, 70.0);
        let a = thumb(&render(&base, 3.0, 40), W, H);
        let snowy = snow(&base, 110.0, 41);
        let b = thumb(&render(&snowy, 3.0, 42), W, H);
        let v = verdict(&b, &a, Scene::Ir, None);
        assert!(v.weather && !v.different, "{v:?}");
        assert!(v.changed_cells as f32 > WEATHER_SHARE * 9216.0, "{v:?}");
        // A 60 x 100 person in the snow: a group of 4+ cells.
        let person = with_blob(&snowy, W, (1500, 900, 60, 100), 35.0);
        let c = thumb(&render(&person, 3.0, 43), W, H);
        let v = verdict(&c, &a, Scene::Ir, None);
        assert!(v.different && v.largest_blob >= WEATHER_COMPONENT, "{v:?}");
    }

    #[test]
    fn a_headlight_wash_is_a_duplicate() {
        for scene in [Scene::Ir, Scene::Day] {
            let base = make_scene(W, H, 5, 40.0, 110.0);
            let a = thumb(&render(&base, 3.0, 50), W, H);
            let washed: Vec<f32> = base
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let x = (i % W) as f32 / W as f32;
                    v + 50.0 * x + 40.0 * (-((x - 0.6) / 0.15).powi(2)).exp()
                })
                .collect();
            let b = thumb(&render(&washed, 3.0, 51), W, H);
            let v = verdict(&b, &a, scene, None);
            assert!(!v.different, "{scene}: {v:?}");
        }
    }

    #[test]
    fn a_camera_shift_is_different() {
        let base = make_scene(W + 40, H + 20, 6, 50.0, 180.0);
        let crop = |dx: usize, dy: usize| -> Vec<f32> {
            (0..W * H)
                .map(|i| base[(i / W + dy) * (W + 40) + i % W + dx])
                .collect()
        };
        let a = thumb(&render(&crop(0, 0), 3.0, 60), W, H);
        let b = thumb(&render(&crop(16, 8), 3.0, 61), W, H);
        let v = verdict(&b, &a, Scene::Day, None);
        assert!(v.different && v.largest_blob > 20, "{v:?}");
    }

    #[test]
    fn verdicts_are_deterministic() {
        let base = make_scene(W, H, 7, 40.0, 90.0);
        let pa = render(&base, 4.0, 70);
        let pb = render(&with_blob(&base, W, (600, 400, 30, 30), 25.0), 4.0, 71);
        let runs: Vec<(Thumb, Thumb, Residual, Verdict)> = (0..2)
            .map(|_| {
                let (a, b) = (thumb(&pa, W, H), thumb(&pb, W, H));
                let r = Residual::of(&b, &a).unwrap();
                let v = verdict(&b, &a, Scene::Ir, None);
                (a, b, r, v)
            })
            .collect();
        assert_eq!(runs[0], runs[1]);
    }

    #[test]
    fn edge_sizes_work() {
        for (w, h) in [
            (1, 1),
            (2, 3),
            (9, 8),
            (127, 71),
            (129, 73),
            (1001, 777),
            (1920, 1080),
            (1520, 2688),
        ] {
            let base = make_scene(w, h, 8, 60.0, 160.0);
            let a = thumb(&render(&base, 2.0, 80), w, h);
            assert_eq!((a.cols(), a.rows()), Thumb::grid(w, h), "{w}x{h}");
            assert_eq!(a.cells().len(), a.cols() * a.rows());
            let v = verdict(&a, &a, Scene::Day, None);
            assert!(!v.different && v.changed_cells == 0, "{w}x{h}: {v:?}");
            if w >= 1000 {
                let b = thumb(&render(&base, 2.0, 81), w, h);
                assert!(!verdict(&b, &a, Scene::Day, None).different, "{w}x{h}");
                let blob = with_blob(&base, w, (w / 2, h / 3, 30, 30), 45.0);
                let c = thumb(&render(&blob, 2.0, 82), w, h);
                assert!(verdict(&c, &a, Scene::Day, None).different, "{w}x{h}");
            }
        }
        assert!(Thumb::of_luma(&[], 0, 0).is_none());
        assert!(Thumb::of_luma(&[1, 2], 2, 2).is_none());
        // Another grid never matches.
        let land = Thumb::from_cells(128, 72, vec![9; 9216]).unwrap();
        let port = Thumb::from_cells(72, 128, vec![9; 9216]).unwrap();
        assert_eq!(verdict(&land, &port, Scene::Day, None), Verdict::MISMATCH);
        assert!(!is_duplicate(&land, [&port], Scene::Day, None));
    }

    #[test]
    fn duplicate_of_stops_at_the_first_matching_reference() {
        let base = make_scene(W, H, 9, 40.0, 90.0);
        let plain = thumb(&render(&base, 3.0, 90), W, H);
        let with = thumb(
            &render(&with_blob(&base, W, (800, 800, 40, 40), 30.0), 3.0, 91),
            W,
            H,
        );
        let new = thumb(&render(&base, 3.0, 92), W, H);
        let p = Params::default();
        assert_eq!(
            duplicate_of(&new, [&with, &plain, &plain], Scene::Ir, None, &p),
            Some(1)
        );
        assert_eq!(duplicate_of(&new, [&with], Scene::Ir, None, &p), None);
        assert_eq!(
            duplicate_of(&new, std::iter::empty(), Scene::Ir, None, &p),
            None
        );
    }

    #[test]
    fn the_noise_map_raises_the_bar_where_cells_are_noisy() {
        let base = make_scene(W, H, 10, 40.0, 90.0);
        let a = thumb(&render(&base, 3.0, 100), W, H);
        let b = thumb(
            &render(&with_blob(&base, W, (1000, 600, 30, 30), 20.0), 3.0, 101),
            W,
            H,
        );
        let r = Residual::of(&b, &a).unwrap();
        assert!(r.verdict(Scene::Ir, None, &Params::default()).different);
        // A camera whose cells there flicker by about 20 levels: no longer a change.
        let mut map = NoiseMap::for_thumb(&a);
        let mut loud = r.clone();
        loud.r.iter_mut().for_each(|v| *v = 64.0);
        for _ in 0..40 {
            assert!(update_noise(&mut map, &loud));
        }
        assert_eq!(map.updates(), 40);
        let s = map.sigma_at(0).unwrap();
        let expected = 64.0 * (1.0 - (1.0 - NOISE_ALPHA).powi(40)) * MEAN_ABS_TO_SIGMA;
        assert!((s - expected).abs() < 1e-3, "{s} vs {expected}");
        assert!(
            !r.verdict(Scene::Ir, Some(&map), &Params::default())
                .different
        );
        // One update from zero: NOISE_ALPHA of |R|.
        let mut one = NoiseMap::for_thumb(&a);
        update_noise(&mut one, &loud);
        assert!((one.sigma_at(5).unwrap() - 4.0 * MEAN_ABS_TO_SIGMA).abs() < 1e-5);
        // A map of another grid is not used, nor updated.
        let mut other = NoiseMap::new(72, 128);
        assert!(!update_noise(&mut other, &r));
        assert!(
            r.verdict(Scene::Ir, Some(&other), &Params::default())
                .different
        );
    }

    #[test]
    fn box_mean_clamps_edges() {
        let v: Vec<f32> = (0..12).map(|i| (i % 4) as f32).collect(); // 4 x 3, rows 0 1 2 3
        let m = box_mean(&v, 4, 3, 1);
        // Column 0 sees 0, 0, 1; column 3 sees 2, 3, 3.
        assert!(
            (m[0] - 1.0 / 3.0).abs() < 1e-6 && (m[3] - 8.0 / 3.0).abs() < 1e-6,
            "{m:?}"
        );
        assert!((m[5] - 1.0).abs() < 1e-6);
        assert_eq!(box_mean(&[1.0; 3], 2, 2, 1), vec![0.0; 3]);
    }

    #[test]
    fn texture_masking_hides_changes_in_texture_only() {
        // Left half flat, right half a strong 2-cell checkerboard texture.
        let mut base = make_scene(W, H, 12, 60.0, 70.0);
        for (i, v) in base.iter_mut().enumerate() {
            let (x, y) = (i % W, i / W);
            if x >= W / 2 && ((x / 42) + (y / 42)) % 2 == 0 {
                *v += 120.0;
            }
        }
        let a = thumb(&render(&base, 3.0, 120), W, H);
        let masked = Params {
            texture_beta: 1.0,
            ..Params::default()
        };
        for (x, flat) in [(600, true), (2000, false)] {
            let b = thumb(
                &render(&with_blob(&base, W, (x, 700, 30, 30), 25.0), 3.0, 121),
                W,
                H,
            );
            let r = Residual::of(&b, &a).unwrap();
            assert!(
                r.verdict(Scene::Ir, None, &Params::default()).different,
                "{x}"
            );
            assert_eq!(r.verdict(Scene::Ir, None, &masked).different, flat, "{x}");
        }
    }

    #[test]
    fn an_encoded_jpeg_thumbs_like_its_pixels() {
        let (w, h) = (320usize, 180usize);
        let base = make_scene(w, h, 11, 40.0, 200.0);
        let px = render(&base, 0.0, 0);
        let img = RgbImage {
            width: w as u32,
            height: h as u32,
            rgb: px.iter().flat_map(|&v| [v, v, v]).collect(),
        };
        let jpeg = crate::image::encode_jpeg(&img, 95).unwrap();
        let (t, tw, th) = Thumb::of_encoded(&jpeg).unwrap();
        assert_eq!((tw, th), (320, 180));
        let direct = thumb(&px, w, h);
        let worst = t
            .cells()
            .iter()
            .zip(direct.cells())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        assert!(worst <= 3, "{worst}");
        assert!(!verdict(&t, &direct, Scene::Day, None).different);
        assert!(Thumb::of_encoded(b"nope").is_err());
    }
}
