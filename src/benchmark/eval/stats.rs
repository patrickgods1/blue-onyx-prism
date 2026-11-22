//! The statistics of the evaluation: Wilson score intervals for the targets, the seeded random
//! numbers of the bootstraps and their percentile intervals.
//!
//! - [`wilson`]: the 95% Wilson score interval of `k` successes out of `n` (two-sided,
//!   [`Z95`]). A target is met on its bound: precision's lower bound >= p, recall's lower
//!   bound >= r, the false alerts' upper bound x 100 <= N, the frame FPR's upper bound <= x.
//!   Wilson rather than the normal approximation: it stays inside [0, 1] and keeps its coverage
//!   at the proportions near 0 and 1 targets live at.
//! - [`Rng`]: SplitMix64 (Steele, Lea & Flood 2014), seeded with [`SEED`]: every bootstrap of
//!   the same number of groups draws the same resamples, so results are deterministic and two
//!   combos on the same frames are resampled alike (the paired comparison).
//! - [`interval`]: the 95% percentile interval of resampled values with symmetric ranks
//!   (`i = ceil(0.025 n)`, the `i`-th smallest and the `i`-th largest), so the interval of a
//!   negated difference is exactly the negated interval.

/// The standard normal quantile of 0.975: a two-sided 95% interval.
pub const Z95: f64 = 1.959_963_984_540_054;
/// The seed of every bootstrap (a fixed value: the same inputs give the same intervals).
pub const SEED: u64 = 0x5EED_0B0F_2026_1122;
/// Tail share of each side of a 95% percentile interval.
const TAIL: f64 = 0.025;

/// The 95% Wilson score interval of `k` successes in `n` trials; None for `n` 0.
pub fn wilson(k: usize, n: usize) -> Option<(f64, f64)> {
    if n == 0 {
        return None;
    }
    let n_f = n as f64;
    let p = k.min(n) as f64 / n_f;
    let z2 = Z95 * Z95;
    let d = 1.0 + z2 / n_f;
    let centre = (p + z2 / (2.0 * n_f)) / d;
    let half = Z95 * (p * (1.0 - p) / n_f + z2 / (4.0 * n_f * n_f)).sqrt() / d;
    Some(((centre - half).max(0.0), (centre + half).min(1.0)))
}

/// The lower bound of [`wilson`].
pub fn lower(k: usize, n: usize) -> Option<f64> {
    wilson(k, n).map(|w| w.0)
}

/// The upper bound of [`wilson`].
pub fn upper(k: usize, n: usize) -> Option<f64> {
    wilson(k, n).map(|w| w.1)
}

/// SplitMix64 (see the module docs).
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n` > 0; multiply-shift: the bias is below 2^-32 for any set size).
    #[allow(
        clippy::cast_possible_truncation,
        reason = "(x * n) >> 64 is below n, which is a usize"
    )]
    pub fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The resamples of a cluster bootstrap over `n` groups: `b` draws of `n` groups with
/// replacement, each as the times every group was drawn. Deterministic ([`SEED`]); the same `n`
/// and `b` give the same draws.
pub fn resamples(n: usize, b: usize) -> impl Iterator<Item = Vec<u32>> {
    let mut rng = Rng::new(SEED ^ (n as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
    (0..b).map(move |_| {
        let mut w = vec![0u32; n];
        if n > 0 {
            for _ in 0..n {
                if let Some(x) = w.get_mut(rng.below(n)) {
                    *x += 1;
                }
            }
        }
        w
    })
}

/// A value with its 95% interval.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Estimate {
    pub value: f64,
    /// None when too few resamples had a value.
    pub low: Option<f64>,
    pub high: Option<f64>,
}

impl Estimate {
    /// `value` with the percentile interval of `resampled` (values None are left out).
    pub fn of(value: f64, resampled: &[Option<f64>]) -> Self {
        let (low, high) = match interval(resampled) {
            Some((l, h)) => (Some(l), Some(h)),
            None => (None, None),
        };
        Self { value, low, high }
    }

    /// Half the interval's width ("± 0.02"), None without an interval.
    pub fn half_width(&self) -> Option<f64> {
        Some((self.high? - self.low?) / 2.0)
    }
}

/// Fewer resampled values than this: no interval.
pub const MIN_RESAMPLES: usize = 20;

/// The 95% percentile interval of the values of `resampled` that are Some (symmetric ranks, see
/// the module docs); None with fewer than [`MIN_RESAMPLES`] of them.
pub fn interval(resampled: &[Option<f64>]) -> Option<(f64, f64)> {
    let mut v: Vec<f64> = resampled
        .iter()
        .flatten()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    if v.len() < MIN_RESAMPLES {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a rank in 1..=n of a sample count"
    )]
    let i = ((TAIL * n as f64).ceil() as usize).clamp(1, n);
    Some((*v.get(i - 1)?, *v.get(n - i)?))
}

/// The paired difference `a - b` per resample (both Some) and its interval: "clearly better"
/// when the interval excludes 0.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Difference {
    pub value: f64,
    pub low: Option<f64>,
    pub high: Option<f64>,
}

impl Difference {
    /// `a - b` of two combos' values on the same frames and their aligned resamples.
    pub fn of(a: f64, b: f64, ra: &[Option<f64>], rb: &[Option<f64>]) -> Self {
        let d: Vec<Option<f64>> = ra
            .iter()
            .zip(rb)
            .map(|(x, y)| Some((*x)? - (*y)?))
            .collect();
        let (low, high) = match interval(&d) {
            Some((l, h)) => (Some(l), Some(h)),
            None => (None, None),
        };
        Self {
            value: a - b,
            low,
            high,
        }
    }

    /// `a` is clearly above `b`: the interval lies above 0.
    pub fn clearly_above(&self) -> bool {
        self.low.is_some_and(|l| l > 0.0)
    }

    /// `a` is clearly below `b`.
    pub fn clearly_below(&self) -> bool {
        self.high.is_some_and(|h| h < 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wilson_matches_known_values() {
        // Newcombe (1998) / the closed form, z = 1.96.
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        for (k, n, lo, hi) in [
            (8, 10, 0.490_162_471_5, 0.943_317_848_5),
            (0, 10, 0.0, 0.277_532_799_9),
            (10, 10, 0.722_467_200_1, 1.0),
            (81, 100, 0.722_211_546_2, 0.874_852_484_9),
            (45, 50, 0.786_397_685_6, 0.956_524_235_1),
            (1, 1, 0.206_549_314_4, 1.0),
            (950, 1000, 0.934_686_179_8, 0.961_869_737_6),
        ] {
            let (l, h) = wilson(k, n).unwrap();
            assert!(close(l, lo) && close(h, hi), "{k}/{n}: {l} {h}");
            assert_eq!(lower(k, n), Some(l));
            assert_eq!(upper(k, n), Some(h));
        }
        assert_eq!(wilson(0, 0), None);
    }

    #[test]
    fn resamples_are_deterministic_and_draw_n_groups() {
        let a: Vec<Vec<u32>> = resamples(7, 50).collect();
        let b: Vec<Vec<u32>> = resamples(7, 50).collect();
        assert_eq!(a, b);
        assert!(a.iter().all(|w| w.iter().sum::<u32>() == 7));
        // Different draws (not one repeated).
        assert!(a.iter().any(|w| *w != a[0]));
        assert_eq!(resamples(0, 3).count(), 3);
        // Every group is drawn about once per resample on average.
        let total: Vec<u32> = resamples(5, 2000).fold(vec![0; 5], |mut acc, w| {
            for (a, x) in acc.iter_mut().zip(w) {
                *a += x;
            }
            acc
        });
        assert!(
            total.iter().all(|&t| (1800..2200).contains(&t)),
            "{total:?}"
        );
    }

    #[test]
    fn intervals_are_symmetric_and_cover() {
        let v: Vec<Option<f64>> = (0..1000).map(|i| Some(f64::from(i))).collect();
        assert_eq!(interval(&v), Some((24.0, 975.0)));
        let neg: Vec<Option<f64>> = v.iter().map(|x| x.map(|x| -x)).collect();
        assert_eq!(interval(&neg), Some((-975.0, -24.0)));
        assert_eq!(interval(&v[..10]), None);
        // A paired difference is antisymmetric.
        let a: Vec<Option<f64>> = (0..200).map(|i| Some(f64::from(i % 17) / 10.0)).collect();
        let b: Vec<Option<f64>> = (0..200).map(|i| Some(f64::from(i % 13) / 9.0)).collect();
        let ab = Difference::of(1.0, 0.5, &a, &b);
        let ba = Difference::of(0.5, 1.0, &b, &a);
        assert_eq!(ab.value, -ba.value);
        assert_eq!(ab.low.map(|x| -x), ba.high);
        assert_eq!(ab.high.map(|x| -x), ba.low);
        assert_eq!(ab.clearly_above(), ba.clearly_below());
    }

    #[test]
    fn bootstrap_interval_of_a_mean_covers_the_truth() {
        // 40 groups of a Bernoulli(0.3), mean estimated 200 times: the 95% interval holds the
        // true mean in most trials (a sanity check, not an exact coverage).
        let mut rng = Rng::new(7);
        let mut covered = 0;
        let trials = 200;
        for _ in 0..trials {
            let x: Vec<f64> = (0..40)
                .map(|_| f64::from(u8::from(rng.unit() < 0.3)))
                .collect();
            let mean = |w: &[u32]| {
                let n: u32 = w.iter().sum();
                let s: f64 = x.iter().zip(w).map(|(v, k)| v * f64::from(*k)).sum();
                Some(s / f64::from(n))
            };
            let r: Vec<Option<f64>> = resamples(x.len(), 400).map(|w| mean(&w)).collect();
            let (l, h) = interval(&r).unwrap();
            covered += usize::from(l <= 0.3 && 0.3 <= h);
        }
        assert!(covered >= 175, "covered {covered} of {trials}");
    }
}
