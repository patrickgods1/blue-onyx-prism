//! Step 7 of [`super::plan`]: balance the training frames by class, scene, camera and / or
//! camera + day (docs/TRAINING.md "Balance"). Off by default; never touches the held-out or
//! validation frames.
//!
//! - **Cap** (optional, `cap_dominant = r`): a group above `r x` the median group size drops
//!   frames by seeded rank down to that size (camera + day: days of the same camera); a frame
//!   leaves the class axis only when every class in it is above the class limit. Hard negatives
//!   and a frame holding a class's last training box always stay.
//! - **Weights** (LVIS repeat-factor sampling, Gupta et al. 2019): `w(g) = (n_max / n_g)^strength`
//!   over training frames; a frame's class weight is the largest of its classes' (negatives 1);
//!   its factor is the product over the ticked axes, at most `max_repeat`.
//! - **Repeats**: `base x factor`, rounded by seeded systematic rounding (each frame rounds up
//!   with the probability of its fraction, the total within 1 of the exact sum), at least 1.
//! - **Negatives**: their repeats are scaled so their share of the images per epoch stays what
//!   it was before balancing (at most `max_repeat x base`).
//!
//! Pure: [`apply`] works on [`Item`]s, the plan maps frames onto them.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Default [`Balance::strength`]: the square root (LVIS).
pub const DEFAULT_STRENGTH: f32 = 0.5;
/// Default [`Balance::max_repeat`].
pub const DEFAULT_MAX_REPEAT: u32 = 4;
/// Most [`Balance::max_repeat`].
pub const MAX_REPEAT: u32 = 10;
/// The cap the page and CLI suggest when "cap dominant groups" is ticked.
pub const DEFAULT_CAP: f32 = 3.0;
/// Groups listed per axis in [`BalanceCounts::camera_days`].
pub const TOP_CAMERA_DAYS: usize = 20;

const CAP_SALT: u64 = 0xca9d_0b1a_5eed;
const ROUND_SALT: u64 = 0x0b1a_5ced_4ead;

/// What a run balances over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BalanceAxis {
    /// Every run class with a box in the frame (multi-label: the rarest one counts).
    Class,
    /// `day` / `night` / `ir` / `unknown`.
    Scene,
    /// The frame's camera (a public frame: its camera tag, else the set's id).
    Camera,
    /// `camera/day`, against the other days of the same camera.
    CameraDay,
}

impl BalanceAxis {
    pub const ALL: [BalanceAxis; 4] = [
        BalanceAxis::Class,
        BalanceAxis::Scene,
        BalanceAxis::Camera,
        BalanceAxis::CameraDay,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            BalanceAxis::Class => "class",
            BalanceAxis::Scene => "scene",
            BalanceAxis::Camera => "camera",
            BalanceAxis::CameraDay => "camera-day",
        }
    }

    /// `class`, `scene`, `camera`, `camera-day` (also `camera_day`, `cameraday`).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "class" | "classes" => Some(BalanceAxis::Class),
            "scene" | "scenes" => Some(BalanceAxis::Scene),
            "camera" | "cameras" => Some(BalanceAxis::Camera),
            "camera-day" | "cameraday" => Some(BalanceAxis::CameraDay),
            _ => None,
        }
    }

    /// A one-line hint (the page's checkbox, the CLI help).
    pub const fn describe(self) -> &'static str {
        match self {
            BalanceAxis::Class => "repeats frames of rare classes (e.g. animals beside people)",
            BalanceAxis::Scene => "evens out day / night / IR frame counts",
            BalanceAxis::Camera => "evens out frames per camera",
            BalanceAxis::CameraDay => "evens out the days of each camera",
        }
    }
}

impl std::fmt::Display for BalanceAxis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How a run balances its training frames (off = no axis).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Balance {
    /// Empty = off.
    pub axes: BTreeSet<BalanceAxis>,
    /// 0..=1: 0.5 the square root (gentle), 1 full equalisation.
    #[serde(serialize_with = "crate::config::thr::ser_clean")]
    pub strength: f32,
    /// 1..=10: most balancing repeats of a frame (the public-mix repeat comes on top).
    pub max_repeat: u32,
    /// Some(r): a group keeps at most `r x` the median group size (r >= 1).
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "crate::config::thr::ser_opt_clean"
    )]
    pub cap_dominant: Option<f32>,
}

impl Default for Balance {
    fn default() -> Self {
        Self {
            axes: BTreeSet::new(),
            strength: DEFAULT_STRENGTH,
            max_repeat: DEFAULT_MAX_REPEAT,
            cap_dominant: None,
        }
    }
}

impl Balance {
    /// No axis ticked: the plan is the unbalanced one.
    pub fn is_off(&self) -> bool {
        self.axes.is_empty()
    }

    /// Axes separated by `,`, `+` or spaces; empty, `off` or `none` = none. The error names the
    /// unknown word.
    pub fn parse_axes(s: &str) -> Result<BTreeSet<BalanceAxis>> {
        let mut out = BTreeSet::new();
        for t in s.split(|c: char| c == ',' || c == '+' || c.is_whitespace()) {
            let t = t.trim();
            if t.is_empty() || t.eq_ignore_ascii_case("off") || t.eq_ignore_ascii_case("none") {
                continue;
            }
            match BalanceAxis::parse(t) {
                Some(a) => {
                    out.insert(a);
                }
                None => bail!(
                    "balance: '{t}' is not an axis (class, scene, camera or camera-day, comma \
                     separated)"
                ),
            }
        }
        Ok(out)
    }

    /// Range-check; errors name the field (`balance.strength`, ...).
    pub fn validate(&self) -> Result<()> {
        if !self.strength.is_finite() || !(0.0..=1.0).contains(&self.strength) {
            bail!(
                "balance.strength must be between 0 and 1 (0.5 gentle, 1 full), got {}",
                self.strength
            );
        }
        if !(1..=MAX_REPEAT).contains(&self.max_repeat) {
            bail!(
                "balance.max_repeat must be between 1 and {MAX_REPEAT}, got {}",
                self.max_repeat
            );
        }
        if let Some(r) = self.cap_dominant
            && (!r.is_finite() || r < 1.0)
        {
            bail!("balance.cap_dominant must be at least 1 (x the median group size), got {r}");
        }
        Ok(())
    }

    /// `class+scene, strength 0.5, max 4x, cap 3x the median`; `off`.
    pub fn describe(&self) -> String {
        if self.is_off() {
            return "off".into();
        }
        let axes: Vec<&str> = self.axes.iter().map(|a| a.as_str()).collect();
        let mut s = format!(
            "{}, strength {}, max {}x",
            axes.join("+"),
            crate::config::clean_f32(self.strength),
            self.max_repeat
        );
        if let Some(r) = self.cap_dominant {
            s.push_str(&format!(
                ", cap {}x the median",
                crate::config::clean_f32(r)
            ));
        }
        s
    }
}

/// Training images per epoch by group (repeats counted).
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct BalanceCounts {
    pub images: usize,
    pub boxes_per_class: BTreeMap<String, usize>,
    /// Images with at least one box of the class.
    pub frames_per_class: BTreeMap<String, usize>,
    pub frames_per_scene: BTreeMap<String, usize>,
    pub frames_per_camera: BTreeMap<String, usize>,
    /// The [`TOP_CAMERA_DAYS`] largest `camera/day` groups before balancing, in that order (the
    /// same groups before and after).
    pub camera_days: Vec<(String, usize)>,
    /// Share of the images without a box of a run class.
    pub negative_share: f64,
}

/// What balancing did (`dataset.json`, the preview).
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct BalanceReport {
    pub balance: Balance,
    pub before: BalanceCounts,
    pub after: BalanceCounts,
    /// Unique training frames before and after the cap.
    pub frames_before: usize,
    pub frames_after: usize,
    /// Frames the cap dropped.
    pub dropped: usize,
    /// The largest repeat of a frame (the public-mix repeat included).
    pub max_repeat: u32,
}

impl BalanceReport {
    /// `209 → 412 images/epoch; person 443→443 boxes, raccoon 32→118`.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{} → {} images/epoch",
            self.before.images, self.after.images
        );
        if self.dropped > 0 {
            s.push_str(&format!(", {} frame(s) capped", self.dropped));
        }
        let classes: Vec<String> = self
            .before
            .boxes_per_class
            .iter()
            .map(|(c, b)| {
                let a = self.after.boxes_per_class.get(c).copied().unwrap_or(0);
                format!("{c} {b}→{a}")
            })
            .collect();
        if !classes.is_empty() {
            s.push_str(&format!("; {} boxes", classes.join(", ")));
        }
        s
    }
}

/// One training frame as balancing sees it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Item {
    /// [`super::frame_key`] (seeded ranks).
    pub key: String,
    /// Repeats before balancing (the public-mix `oversample`, else 1).
    pub base: u32,
    /// `day`, `night`, `ir` or `unknown`.
    pub scene: String,
    pub camera: String,
    pub day: String,
    /// Boxes per run class (index = class).
    pub boxes: Vec<usize>,
    pub hard_negative: bool,
}

impl Item {
    fn positive(&self) -> bool {
        self.boxes.iter().any(|&n| n > 0)
    }

    fn classes(&self) -> impl Iterator<Item = usize> + '_ {
        self.boxes
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 0)
            .map(|(i, _)| i)
    }

    /// `(partition, group)` of a frame-level axis: groups compare within their partition.
    fn group(&self, axis: BalanceAxis) -> (&str, String) {
        match axis {
            BalanceAxis::Scene => ("", self.scene.clone()),
            BalanceAxis::Camera => ("", self.camera.clone()),
            BalanceAxis::CameraDay => (&self.camera, format!("{}/{}", self.camera, self.day)),
            BalanceAxis::Class => ("", String::new()),
        }
    }
}

/// What [`apply`] decided, index-aligned with its items.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Outcome {
    /// False = dropped by the cap.
    pub keep: Vec<bool>,
    /// Final repeats (0 for a dropped frame).
    pub repeats: Vec<u32>,
    pub report: BalanceReport,
    pub warnings: Vec<String>,
}

fn unit(seed: u64, key: &str) -> f64 {
    super::rank(seed, key) as f64 / (u64::MAX as f64 + 1.0)
}

/// Median of `v` (mean of the middle two for an even count); 0 when empty.
fn median(v: &mut [usize]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_unstable();
    let n = v.len();
    let hi = v.get(n / 2).copied().unwrap_or(0) as f64;
    if n % 2 == 1 {
        hi
    } else {
        (v.get(n / 2 - 1).copied().unwrap_or(0) as f64 + hi) / 2.0
    }
}

/// `r x median`, rounded up (so a group at the median never shrinks), at least 1.
fn limit_of(r: f32, median: f64) -> usize {
    let l = super::super::estimate::to_u64((f64::from(r) * median).ceil());
    usize::try_from(l).unwrap_or(usize::MAX).max(1)
}

/// `(n_max / n)^strength` (1 for an empty group).
pub fn weight(n_max: usize, n: usize, strength: f32) -> f64 {
    if n == 0 || n_max <= n {
        return 1.0;
    }
    (n_max as f64 / n as f64).powf(f64::from(strength))
}

/// Seeded systematic rounding of `x` (each `>= 0`): every value rounds up with the probability of
/// its fraction, in seeded rank order of `keys`, so the total is within 1 of the exact sum.
pub fn round_repeats(x: &[f64], keys: &[&str], seed: u64) -> Vec<u32> {
    let mut order: Vec<(u64, usize)> = (0..x.len())
        .map(|i| {
            let k = keys.get(i).copied().unwrap_or("");
            (super::rank(seed ^ ROUND_SALT, k), i)
        })
        .collect();
    order.sort_unstable();
    let mut acc = unit(seed ^ ROUND_SALT, "\u{0}offset");
    let mut out = vec![0u32; x.len()];
    for (_, i) in order {
        let v = x.get(i).copied().unwrap_or(1.0).max(0.0);
        let fl = v.floor();
        let before = acc.floor();
        acc += v - fl;
        let up = acc.floor() - before;
        if let Some(o) = out.get_mut(i) {
            *o = u32::try_from(super::super::estimate::to_u64(fl + up)).unwrap_or(u32::MAX);
        }
    }
    out
}

/// Frames per class over the kept items.
fn class_frames(items: &[Item], keep: &[bool], n_classes: usize) -> Vec<usize> {
    let mut fc = vec![0usize; n_classes];
    for (it, _) in items.iter().zip(keep).filter(|(_, k)| **k) {
        for c in it.classes() {
            if let Some(n) = fc.get_mut(c) {
                *n += 1;
            }
        }
    }
    fc
}

/// Drop `it` unless it is a hard negative or holds a class's last box (`box_left`); true when
/// dropped.
fn try_drop(it: Option<&Item>, keep: Option<&mut bool>, box_left: &mut [usize]) -> bool {
    let (Some(it), Some(k)) = (it, keep) else {
        return false;
    };
    let last = it
        .boxes
        .iter()
        .zip(box_left.iter())
        .any(|(&n, &l)| n > 0 && l <= n);
    if !*k || it.hard_negative || last {
        return false;
    }
    *k = false;
    for (l, n) in box_left.iter_mut().zip(&it.boxes) {
        *l = l.saturating_sub(*n);
    }
    true
}

/// The cap of dominant groups (`r x` the median); returns the frames dropped.
fn cap(
    items: &[Item],
    n_classes: usize,
    b: &Balance,
    r: f32,
    seed: u64,
    keep: &mut [bool],
) -> usize {
    let mut box_left = vec![0usize; n_classes];
    for it in items {
        for (l, n) in box_left.iter_mut().zip(&it.boxes) {
            *l += n;
        }
    }
    let rank_of = |i: usize| {
        items
            .get(i)
            .map_or(0, |it| super::rank(seed ^ CAP_SALT, &it.key))
    };
    let mut dropped = 0usize;
    for axis in b.axes.iter().copied().filter(|a| *a != BalanceAxis::Class) {
        // partition -> group -> kept frames
        let mut parts: BTreeMap<String, BTreeMap<String, Vec<usize>>> = BTreeMap::new();
        for ((i, it), _) in items
            .iter()
            .enumerate()
            .zip(keep.iter())
            .filter(|(_, k)| **k)
        {
            let (p, g) = it.group(axis);
            parts
                .entry(p.to_string())
                .or_default()
                .entry(g)
                .or_default()
                .push(i);
        }
        for groups in parts.values() {
            let mut sizes: Vec<usize> = groups.values().map(Vec::len).collect();
            let limit = limit_of(r, median(&mut sizes));
            for g in groups.values().filter(|g| g.len() > limit) {
                let mut order: Vec<(u64, usize)> = g.iter().map(|&i| (rank_of(i), i)).collect();
                order.sort_unstable();
                let mut size = g.len();
                for (_, i) in order {
                    if size <= limit {
                        break;
                    }
                    if try_drop(items.get(i), keep.get_mut(i), &mut box_left) {
                        size -= 1;
                        dropped += 1;
                    }
                }
            }
        }
    }
    if b.axes.contains(&BalanceAxis::Class) {
        let mut fc = class_frames(items, keep, n_classes);
        let mut sizes: Vec<usize> = fc.iter().copied().filter(|&v| v > 0).collect();
        let limit = limit_of(r, median(&mut sizes));
        let mut order: Vec<(u64, usize)> = items
            .iter()
            .enumerate()
            .zip(keep.iter())
            .filter(|((_, it), k)| **k && it.positive())
            .map(|((i, _), _)| (rank_of(i), i))
            .collect();
        order.sort_unstable();
        for (_, i) in order {
            let Some(it) = items.get(i) else { continue };
            if !it.classes().all(|c| fc.get(c).is_some_and(|&n| n > limit)) {
                continue;
            }
            if try_drop(Some(it), keep.get_mut(i), &mut box_left) {
                for c in it.classes() {
                    if let Some(n) = fc.get_mut(c) {
                        *n -= 1;
                    }
                }
                dropped += 1;
            }
        }
    }
    dropped
}

/// Balance factors of the kept items (product of the ticked axes' weights, uncapped) and the
/// single-frame groups `(item, "axis group")`.
fn factors(
    items: &[Item],
    classes: &[String],
    b: &Balance,
    keep: &[bool],
) -> (Vec<f64>, Vec<(usize, String)>) {
    let mut factor = vec![1.0f64; items.len()];
    let mut singles: Vec<(usize, String)> = Vec::new();
    for axis in b.axes.iter().copied().filter(|a| *a != BalanceAxis::Class) {
        let mut sizes: BTreeMap<(String, String), usize> = BTreeMap::new();
        for (it, _) in items.iter().zip(keep).filter(|(_, k)| **k) {
            let (p, g) = it.group(axis);
            *sizes.entry((p.to_string(), g)).or_default() += 1;
        }
        let mut n_max: BTreeMap<&str, usize> = BTreeMap::new();
        for ((p, _), &s) in &sizes {
            let m = n_max.entry(p.as_str()).or_default();
            *m = (*m).max(s);
        }
        for (((i, it), _), f) in items
            .iter()
            .enumerate()
            .zip(keep)
            .zip(factor.iter_mut())
            .filter(|((_, k), _)| **k)
        {
            let (p, g) = it.group(axis);
            let key = (p.to_string(), g);
            let s = sizes.get(&key).copied().unwrap_or(1);
            *f *= weight(n_max.get(p).copied().unwrap_or(s), s, b.strength);
            if s == 1 {
                singles.push((i, format!("{} {}", axis.as_str(), key.1)));
            }
        }
    }
    if b.axes.contains(&BalanceAxis::Class) {
        let fc = class_frames(items, keep, classes.len());
        let n_max = fc.iter().copied().max().unwrap_or(0);
        for (((i, it), _), f) in items
            .iter()
            .enumerate()
            .zip(keep)
            .zip(factor.iter_mut())
            .filter(|((_, k), _)| **k)
        {
            let n_of = |c: usize| fc.get(c).copied().unwrap_or(0);
            *f *= it
                .classes()
                .map(|c| weight(n_max, n_of(c), b.strength))
                .fold(1.0f64, f64::max);
            for c in it.classes().filter(|&c| n_of(c) == 1) {
                if let Some(name) = classes.get(c) {
                    singles.push((i, format!("class {name}")));
                }
            }
        }
    }
    (factor, singles)
}

fn counts(items: &[Item], reps: &[u32], classes: &[String]) -> BalanceCounts {
    let mut c = BalanceCounts::default();
    for name in classes {
        c.boxes_per_class.insert(name.clone(), 0);
        c.frames_per_class.insert(name.clone(), 0);
    }
    let mut camera_days: BTreeMap<String, usize> = BTreeMap::new();
    let mut negatives = 0usize;
    for (it, &r) in items.iter().zip(reps) {
        let r = r as usize;
        if r == 0 {
            continue;
        }
        c.images += r;
        for (ci, &n) in it.boxes.iter().enumerate() {
            if n == 0 {
                continue;
            }
            if let Some(name) = classes.get(ci) {
                *c.boxes_per_class.entry(name.clone()).or_default() += n * r;
                *c.frames_per_class.entry(name.clone()).or_default() += r;
            }
        }
        if !it.positive() {
            negatives += r;
        }
        *c.frames_per_scene.entry(it.scene.clone()).or_default() += r;
        *c.frames_per_camera.entry(it.camera.clone()).or_default() += r;
        *camera_days
            .entry(format!("{}/{}", it.camera, it.day))
            .or_default() += r;
    }
    c.negative_share = if c.images > 0 {
        negatives as f64 / c.images as f64
    } else {
        0.0
    };
    c.camera_days = camera_days.into_iter().collect();
    c
}

/// Balance `items` (the training frames, not empty) over `classes` (see the module docs).
pub fn apply(items: &[Item], classes: &[String], b: &Balance, seed: u64) -> Outcome {
    let base: Vec<u32> = items.iter().map(|i| i.base.max(1)).collect();
    let mut keep = vec![true; items.len()];
    let dropped = match b.cap_dominant {
        Some(r) => cap(items, classes.len(), b, r, seed, &mut keep),
        None => 0,
    };
    let (mut factor, singles) = factors(items, classes, b, &keep);
    let max_f = f64::from(b.max_repeat.max(1));
    for f in &mut factor {
        *f = f.min(max_f);
    }
    let mut x: Vec<f64> = base
        .iter()
        .zip(&factor)
        .zip(&keep)
        .map(|((&n, f), k)| if *k { f64::from(n) * f } else { 0.0 })
        .collect();

    // Negatives keep their share of the images per epoch (the share before step 7).
    let before = counts(items, &base, classes);
    let share = before.negative_share;
    let (mut pos, mut neg) = (0.0f64, 0.0f64);
    for (it, v) in items.iter().zip(&x) {
        if it.positive() {
            pos += v;
        } else {
            neg += v;
        }
    }
    if share > 0.0 && share < 1.0 && neg > 0.0 && pos > 0.0 {
        let m = (share * pos / (1.0 - share)) / neg;
        for (((it, v), k), &n) in items.iter().zip(x.iter_mut()).zip(&keep).zip(&base) {
            if *k && !it.positive() {
                *v = (*v * m).clamp(1.0, (f64::from(n) * max_f).max(1.0));
            }
        }
    }

    let keys: Vec<&str> = items.iter().map(|i| i.key.as_str()).collect();
    let mut repeats = round_repeats(&x, &keys, seed);
    for (r, k) in repeats.iter_mut().zip(&keep) {
        *r = if *k { (*r).max(1) } else { 0 };
    }
    let after = counts(items, &repeats, classes);

    let mut warnings = Vec::new();
    let mut few: Vec<String> = singles
        .into_iter()
        .filter(|(i, _)| max_f > 1.0 && factor.get(*i).is_some_and(|f| *f >= max_f))
        .map(|(_, g)| g)
        .collect();
    few.sort();
    few.dedup();
    if !few.is_empty() {
        let shown: Vec<&str> = few.iter().take(5).map(String::as_str).collect();
        warnings.push(format!(
            "few unique frames: {}{} ha{} a single training frame, repeated {}x; balancing only \
             repeats it, so collect more frames of {}",
            shown.join(", "),
            if few.len() > shown.len() {
                format!(" and {} more", few.len() - shown.len())
            } else {
                String::new()
            },
            if few.len() == 1 { "s" } else { "ve" },
            b.max_repeat,
            if few.len() == 1 { "it" } else { "them" }
        ));
    }
    if after.images > before.images * 2 {
        warnings.push(format!(
            "balancing raises the training images per epoch from {} to {}: epochs get longer \
             (patience counts epochs, so a run may take more time)",
            before.images, after.images
        ));
    }

    // The same largest camera/day groups before and after.
    let mut top = before.camera_days.clone();
    top.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    top.truncate(TOP_CAMERA_DAYS);
    let after_days: BTreeMap<String, usize> = after.camera_days.iter().cloned().collect();
    let after = BalanceCounts {
        camera_days: top
            .iter()
            .map(|(k, _)| (k.clone(), after_days.get(k).copied().unwrap_or(0)))
            .collect(),
        ..after
    };
    let before = BalanceCounts {
        camera_days: top,
        ..before
    };

    let report = BalanceReport {
        balance: b.clone(),
        frames_before: items.len(),
        frames_after: keep.iter().filter(|k| **k).count(),
        dropped,
        max_repeat: repeats.iter().copied().max().unwrap_or(0),
        before,
        after,
    };
    Outcome {
        keep,
        repeats,
        report,
        warnings,
    }
}

#[cfg(test)]
mod tests;
