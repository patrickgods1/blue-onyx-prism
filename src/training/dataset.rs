//! Dataset assembly: review exports (YOLO layout) plus an optional public subset in, the
//! trainer's input out (docs/TRAINING.md "Dataset").
//!
//! 1. **Read** ([`read_export`], [`read_public_set`]): class list, frames with their boxes
//!    (normalized YOLO boxes), scene, camera and day. Two export layouts are read:
//!    - the Review tab's export (`src/collect/labels/export.rs`): `images/{train,val}/`,
//!      `labels/{train,val}/`, `classes.txt`, flat names `<scene>.<camera>.<date>.<stem>.jpg`;
//!      its `val` split is already held out by whole camera days, so it becomes the held-out set
//!      (an export with no `val` frame, such as one camera on one day, is held out here instead);
//!    - a flat YOLO folder: `images/` (or the folder itself), `labels/` (or `<stem>.txt` beside
//!      the image), `classes.txt` or `data.yaml`, optional `test.txt` / `heldout.txt` listing
//!      held-out images; scene / camera / date tags from a `manifest.json` (benchmark format) when
//!      there is one.
//!
//!    A missing label file means "no objects" (the Ultralytics convention), except in a review
//!    export, which writes one per image; a file that cannot be read is an error. `ignore.json`
//!    ([`IgnoreFile`]) lists real objects not to train on (unsure, too small) and the frames with
//!    nothing but parked-spot boxes; no file = none. `frames.json` ([`FrameNotes`]) names the
//!    frames confirmed empty or accepted in bulk and `hard_negatives.json` the rejected boxes,
//!    whichever formats the export wrote.
//! 2. **Plan** ([`plan`], pure): map the classes onto the run's class list (renames, drops);
//!    drop the copies of a frame that several exports hold (a held-out copy wins); hold out review
//!    frames by whole camera days (unless the exports already did; one camera day: frame by frame,
//!    with a warning) from **every** review frame, before any filter, so runs with other scene
//!    filters are benchmarked on the same frames ([`Plan::heldout_fingerprint`]); judge the size
//!    floor regions of the train / valid frames again at the run's input ([`restore_floor`]: one
//!    not under the export's floor there is a box again) and compare the exports' floors
//!    ([`floor_warnings`]); then filter the train / valid pool by scene
//!    ([`SceneFilter`]), handle its frames with ignore regions ([`IgnoreMode`]; no trainer here
//!    can leave a region out of its loss: by default a frame with a box of the run's classes
//!    trains with the region as background and a frame of nothing but regions is left out, never
//!    as a negative; [`IgnoreMode::DropFrame`] leaves every such frame out; validation never holds
//!    one, so one on a camera day validated on leaves both sides; the held-out set keeps them, with
//!    the regions `ignore` in its manifest) and, in [`Excluded::DropFrame`] mode, its frames with
//!    boxes of untrained labels; cap the frames with nothing but parked-spot boxes at
//!    [`PARKED_CAP`] per spot (the same crop again and again would dominate training), then, of
//!    the rest, the frames whose every box of the run's classes lies at a static spot (found by
//!    geometry over the train / valid pool's own frames, [`statics`]) at [`Options::static_cap`]
//!    per spot and class, kept across scenes and days ([`over_static_cap`]); held-out frames are
//!    never capped; keep about
//!    `negative_share` frames without objects in training (frames a reviewer confirmed empty or
//!    rejected a box on first), take a
//!    validation split of `val_share` of the review training frames as whole camera days
//!    ([`valid`], never a class's last training box; the positives that leave with a validated
//!    day for their ignore regions no longer count for the negatives, which are trimmed to
//!    match), and repeat the review frames
//!    `oversample` times when public frames are mixed in. Public frames are only ever trained on.
//!    Seeded choices rank a frame by its path relative to its source folder, so the plan is the
//!    same wherever the folders are (reproducible runs). Step 7, opt-in ([`Balance`],
//!    [`balance`]): repeat the training frames
//!    of rare classes / scenes / cameras / camera days (and optionally drop frames of dominant
//!    ones); validation and held-out frames are never balanced.
//! 3. **Write** ([`write`]): `images/` + `labels/` (hard links, else copies), `classes.txt`,
//!    `train.txt` / `valid.txt`, `data.yaml` (Ultralytics), `train.coco.json` /
//!    `valid.coco.json` (category ids 0..C-1, for RF-DETR and transformers), the held-out set as
//!    a benchmark `dir:` set (`heldout/`, every scene, tagged by [`heldout_tags`]),
//!    `dataset.json` (what went in, for provenance) and [`IMAGE_SOURCES_FILE`] (the source of
//!    each image in `images/`).
//!
//! [`inventory`] counts the boxes per class and scene of the sources (the class picker).

use crate::benchmark::images::{self, Manifest, ManifestObject};
use crate::collect::labels::bulk;
use crate::collect::labels::export::{
    FRAMES_FILE, FRAMES_VERSION, FrameNotes, HARD_NEGATIVES_FILE, IGNORE_FILE, IGNORE_VERSION,
    IgnoreBox, IgnoreFile,
};
use crate::collect::labels::{SizeFloor, statics};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub mod balance;
mod valid;
pub use balance::{Balance, BalanceAxis, BalanceCounts, BalanceReport};
pub use valid::valid_groups_text;

/// Which frames a run trains on, by scene: any combination of day, night and IR. None or all
/// three = every frame (unknown scenes included). Serialized as its canonical string
/// ([`SceneFilter::as_str`]); every older spelling still parses. `night` alone is night only and
/// `ir` alone IR only (before scene sets both meant night + IR).
#[derive(Debug, Clone, Copy)]
pub struct SceneFilter {
    pub day: bool,
    pub night: bool,
    pub ir: bool,
}

impl Default for SceneFilter {
    fn default() -> Self {
        Self::ALL_SCENES
    }
}

impl SceneFilter {
    /// Every frame.
    pub const ALL_SCENES: SceneFilter = SceneFilter::new(true, true, true);
    pub const DAY: SceneFilter = SceneFilter::new(true, false, false);
    /// Night and IR frames (a night model).
    pub const NIGHT_IR: SceneFilter = SceneFilter::new(false, true, true);
    /// The page's checkboxes, in order.
    pub const CHOICES: [&'static str; 3] = ["day", "night", "ir"];
    /// The usual picks: a general, a day and a night model.
    pub const PRESETS: [SceneFilter; 3] = [Self::ALL_SCENES, Self::DAY, Self::NIGHT_IR];

    pub const fn new(day: bool, night: bool, ir: bool) -> Self {
        Self { day, night, ir }
    }

    /// No scene restriction (all three or none ticked).
    pub const fn is_all(self) -> bool {
        self.day == self.night && self.night == self.ir
    }

    /// A night model: night and / or IR, no day.
    pub const fn is_night(self) -> bool {
        !self.is_all() && !self.day && (self.night || self.ir)
    }

    /// The canonical spelling: `all`, `day`, `night`, `ir`, `day+night`, `day+ir`, `night+ir`.
    pub const fn as_str(self) -> &'static str {
        if self.is_all() {
            return "all";
        }
        match (self.day, self.night, self.ir) {
            (true, false, false) => "day",
            (false, true, false) => "night",
            (false, false, true) => "ir",
            (true, true, false) => "day+night",
            (true, false, true) => "day+ir",
            _ => "night+ir",
        }
    }

    /// The model-name prefix: `custom` (all), `day`, `night` (night + IR), `night-only`, `ir`,
    /// `day-night`, `day-ir`.
    pub const fn label(self) -> &'static str {
        if self.is_all() {
            return "custom";
        }
        match (self.day, self.night, self.ir) {
            (true, false, false) => "day",
            (false, true, false) => "night-only",
            (false, false, true) => "ir",
            (true, true, false) => "day-night",
            (true, false, true) => "day-ir",
            _ => "night",
        }
    }

    /// A human description (the page's preset list).
    pub const fn describe(self) -> &'static str {
        if self.is_all() {
            return "all frames (a general custom model)";
        }
        match (self.day, self.night, self.ir) {
            (true, false, false) => "day frames only",
            (false, true, false) => "night frames only (no IR)",
            (false, false, true) => "IR frames only",
            (true, true, false) => "day + night frames (no IR)",
            (true, false, true) => "day + IR frames",
            _ => "night + IR frames (a night model)",
        }
    }

    /// Parse scene names separated by `+`, `,` or spaces (`day`, `night`, `ir`, `all`), and the
    /// old spellings `night-ir` / `nightir`. Empty = all.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        if matches!(s.as_str(), "night-ir" | "nightir") {
            return Some(Self::NIGHT_IR);
        }
        let mut f = Self::new(false, false, false);
        for t in s.split(|c: char| c == '+' || c == ',' || c.is_whitespace()) {
            match t {
                "" => {}
                "all" => f = Self::ALL_SCENES,
                "day" => f.day = true,
                "night" => f.night = true,
                "ir" => f.ir = true,
                _ => return None,
            }
        }
        Some(f.normalized())
    }

    /// All three ticked when none is (one spelling of "all").
    fn normalized(self) -> Self {
        if self.is_all() {
            Self::ALL_SCENES
        } else {
            self
        }
    }

    /// Whether a frame of `scene` (None = unknown) is in: everything for `all`, else only frames
    /// tagged with a chosen scene.
    pub fn keeps(self, scene: Option<&str>) -> bool {
        if self.is_all() {
            return true;
        }
        match scene {
            Some("day") => self.day,
            Some("night") => self.night,
            Some("ir") => self.ir,
            _ => false,
        }
    }
}

/// Equal by meaning: no scene ticked equals all three.
impl PartialEq for SceneFilter {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for SceneFilter {}

impl std::hash::Hash for SceneFilter {
    fn hash<H: std::hash::Hasher>(&self, h: &mut H) {
        self.as_str().hash(h);
    }
}

impl std::fmt::Display for SceneFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Serialize for SceneFilter {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for SceneFilter {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Self::parse(&s).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "scene '{s}' is not all or a combination of day, night and ir"
            ))
        })
    }
}

/// What happens to a train / valid frame holding a box of a label the run does not train (not in
/// its classes, not dropped by an empty rename).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Excluded {
    /// The frame stays; the object is background (a frame left without a box is a negative).
    #[default]
    Background,
    /// The frame is left out, so a look-alike of a trained class is never taught as background.
    DropFrame,
}

impl Excluded {
    pub const fn as_str(self) -> &'static str {
        match self {
            Excluded::Background => "background",
            Excluded::DropFrame => "drop-frame",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "background" | "" => Some(Excluded::Background),
            "drop-frame" | "drop" => Some(Excluded::DropFrame),
            _ => None,
        }
    }
}

/// What happens to a train / valid frame with ignore regions ([`Frame::ignore`]: objects a reviewer
/// marked unsure or too small, or training boxes under the export's size floor). No trainer here
/// can leave a region out of its loss: either the object trains as background or the frame goes.
/// Held-out frames keep their regions as ignore regions in both modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IgnoreMode {
    /// The frame trains, its regions unlabelled (background), when it holds at least one box of the
    /// run's classes; a frame with nothing but regions is left out (an image whose only object is
    /// unlabelled would teach "nothing here" exactly where something is), and so is a frame of a
    /// camera day validated on (validation cannot ignore a region, and the day is validation's).
    /// Never a negative: such a frame shows a real object.
    #[default]
    Background,
    /// Every such frame is left out of train and valid (the behaviour of earlier versions): one
    /// unsure or tiny object takes its whole frame out of training.
    DropFrame,
}

impl IgnoreMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            IgnoreMode::Background => "background",
            IgnoreMode::DropFrame => "drop-frame",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "background" => Some(IgnoreMode::Background),
            "drop-frame" | "drop" => Some(IgnoreMode::DropFrame),
            _ => None,
        }
    }

    /// The default (not written in the config).
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// A box, normalized to the image (YOLO: center, size).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct YoloBox {
    pub class: String,
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

/// Where a source put a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SplitHint {
    /// The source says nothing: the plan decides.
    None,
    Train,
    /// Held out by the source (the review export's `val`, a `test.txt`).
    Heldout,
}

/// One image of a source.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Frame {
    pub image: PathBuf,
    pub boxes: Vec<YoloBox>,
    /// Real objects not to train on (`ignore.json`): not in `boxes`, so the frame trains with them
    /// as background or not at all ([`IgnoreMode`]), never validates, and they are `ignore` when
    /// the frame is benchmarked.
    pub ignore: Vec<YoloBox>,
    /// Indices into `ignore` of the regions the export's size floor made ([`IgnoreBox::size_floor`],
    /// [`Source::size_floor`]): training boxes a run at a larger input may give back
    /// ([`Options::imgsz`]).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub floor_regions: Vec<usize>,
    /// `day`, `night` or `ir`; None = unknown.
    pub scene: Option<String>,
    pub camera: String,
    /// `YYYY-MM-DD` or `unknown`.
    pub day: String,
    pub split: SplitHint,
    /// Confirmed empty by the reviewer (a hard negative).
    pub hard_negative: bool,
    /// A reviewer rejected a box of a model or labeler here ([`read_rejected`]): a frame the model
    /// is known to get wrong, the most useful kind of negative.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub human_rejected: bool,
    /// Accepted in bulk ([`FrameNotes::bulk_accepted`]): nobody looked at this frame on its own.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub bulk_accepted: bool,
    /// `(width, height)` when the source knows it.
    pub size: Option<(u32, u32)>,
    /// The parked spots of a frame with nothing but parked-spot boxes ([`IgnoreFile::parked`]).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub parked: Vec<String>,
}

/// A public dataset mixed in (never held out, never oversampled).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Public {
    pub id: String,
    pub license: String,
}

/// One input of a run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Source {
    pub dir: PathBuf,
    pub public: Option<Public>,
    pub classes: Vec<String>,
    pub frames: Vec<Frame>,
    /// Problems that did not stop the read.
    pub warnings: Vec<String>,
    /// The folder says which frames were accepted in bulk ([`FRAMES_FILE`]); exports of older
    /// versions do not.
    pub notes_known: bool,
    /// The size floor the export applied ([`IgnoreFile::size_floor`]); None = none (or not a
    /// review export of this version).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_floor: Option<SizeFloor>,
}

/// Unknown camera / day.
pub const UNKNOWN: &str = "unknown";
const SCENES: [&str; 3] = ["day", "night", "ir"];
const SPLIT_DIRS: [(&str, SplitHint); 5] = [
    ("train", SplitHint::Train),
    ("val", SplitHint::Heldout),
    ("valid", SplitHint::Heldout),
    ("test", SplitHint::Heldout),
    ("heldout", SplitHint::Heldout),
];
/// Most frames read from one source.
pub const MAX_FRAMES: usize = 500_000;
/// Most frames with nothing but parked-spot boxes kept per spot, for training and validation.
pub const PARKED_CAP: usize = 20;

/// An image file by extension; hidden files (macOS AppleDouble `._x.jpg`) are not.
fn is_image(p: &Path) -> bool {
    p.is_file()
        && !p
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with('.'))
        && p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
            ["jpg", "jpeg", "png"]
                .iter()
                .any(|x| x.eq_ignore_ascii_case(e))
        })
}

fn images_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_image(p))
        .collect();
    v.sort();
    Ok(v)
}

fn stem(p: &Path) -> String {
    p.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Class names of a YOLO folder: `classes.txt`, else `data.yaml` / `dataset.yaml`.
pub fn read_classes(dir: &Path) -> Result<Vec<String>> {
    let txt = dir.join("classes.txt");
    if txt.is_file() {
        let names: Vec<String> = std::fs::read_to_string(&txt)
            .with_context(|| format!("reading {}", txt.display()))?
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect();
        if names.is_empty() {
            bail!("{} lists no class", txt.display());
        }
        return Ok(names);
    }
    for y in ["data.yaml", "dataset.yaml"] {
        let p = dir.join(y);
        if p.is_file() {
            return images::parse_yolo_yaml(&std::fs::read_to_string(&p)?)
                .with_context(|| format!("parsing {}", p.display()));
        }
    }
    bail!(
        "{} has no classes.txt or data.yaml: it is not a YOLO export (export from the Dataset tab \
         with the YOLO format)",
        dir.display()
    )
}

/// Parse one YOLO label file (`class cx cy w h` or a polygon `class x1 y1 x2 y2 ...`,
/// normalized). Errors name the line. Zero-size boxes are left out.
pub fn parse_labels(text: &str, classes: &[String]) -> Result<Vec<YoloBox>> {
    const TOL: f32 = 1e-3;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let n = n + 1;
        let f: Vec<&str> = line.split_whitespace().collect();
        let Some((c, rest)) = f.split_first() else {
            continue;
        };
        let id: usize = c
            .parse()
            .with_context(|| format!("line {n}: bad class '{c}'"))?;
        let Some(class) = classes.get(id) else {
            bail!(
                "line {n}: class {id} is not in the class list ({} classes)",
                classes.len()
            );
        };
        let v = rest
            .iter()
            .map(|s| {
                let x: f32 = s
                    .parse()
                    .with_context(|| format!("line {n}: bad number '{s}'"))?;
                if !x.is_finite() || !(-TOL..=1.0 + TOL).contains(&x) {
                    bail!("line {n}: '{s}' is not a normalized coordinate (0..1)");
                }
                Ok(x.clamp(0.0, 1.0))
            })
            .collect::<Result<Vec<f32>>>()?;
        let (cx, cy, w, h) = match v.as_slice() {
            [cx, cy, w, h] => (*cx, *cy, *w, *h),
            pts if pts.len() >= 6 && pts.len() % 2 == 0 => {
                let xs = pts.iter().step_by(2);
                let ys = pts.iter().skip(1).step_by(2);
                let (x0, x1) = xs.fold((1f32, 0f32), |(a, b), &x| (a.min(x), b.max(x)));
                let (y0, y1) = ys.fold((1f32, 0f32), |(a, b), &y| (a.min(y), b.max(y)));
                ((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0)
            }
            _ => bail!(
                "line {n}: expected `class cx cy w h` or a polygon, found {} values",
                v.len()
            ),
        };
        if w > 0.0 && h > 0.0 {
            out.push(YoloBox {
                class: class.clone(),
                cx,
                cy,
                w,
                h,
            });
        }
    }
    Ok(out)
}

/// Scene, camera, day and hard-negative of an image from benchmark-manifest tags.
#[derive(Debug, Clone, Default, PartialEq)]
struct Meta {
    scene: Option<String>,
    camera: Option<String>,
    day: Option<String>,
    hard_negative: bool,
    size: Option<(u32, u32)>,
}

fn meta_from_tags(tags: &[String]) -> Meta {
    let mut m = Meta::default();
    for t in tags {
        let t = t.trim();
        if let Some(c) = t.strip_prefix("camera:") {
            m.camera = Some(c.to_string());
        } else if let Some(d) = t.strip_prefix("date:") {
            m.day = Some(d.to_string());
        } else if SCENES.contains(&t) {
            // `night` + `ir` tags together: IR.
            if m.scene.as_deref() != Some("ir") {
                m.scene = Some(t.to_string());
            }
        } else if t == "hard-negative" {
            m.hard_negative = true;
        }
    }
    if m.scene.is_none() && tags.iter().any(|t| t == "grayscale") {
        m.scene = Some("ir".into());
    }
    m
}

/// Scene, camera and day from a review export's flat name (`<scene>.<camera>.<date>.<stem>`) or
/// a collector stem (`20261022T123456789Z_<id>`: the date only).
fn meta_from_name(stem: &str) -> Meta {
    let parts: Vec<&str> = stem.split('.').collect();
    if let [scene, camera, date, _] = parts.as_slice()
        && SCENES.contains(scene)
    {
        return Meta {
            scene: Some((*scene).to_string()),
            camera: Some((*camera).to_string()),
            day: Some((*date).to_string()),
            ..Meta::default()
        };
    }
    let b = stem.as_bytes();
    if b.len() >= 9 && b.get(8) == Some(&b'T') && b.iter().take(8).all(u8::is_ascii_digit) {
        let d = |r: std::ops::Range<usize>| stem.get(r).unwrap_or_default();
        return Meta {
            day: Some(format!("{}-{}-{}", d(0..4), d(4..6), d(6..8))),
            ..Meta::default()
        };
    }
    Meta::default()
}

/// Tags of every image in a `manifest.json` of the folder (or its `benchmark/`), by file name.
fn manifest_meta(dir: &Path) -> BTreeMap<String, Meta> {
    let mut out = BTreeMap::new();
    for p in [
        dir.join("manifest.json"),
        dir.join("benchmark").join("manifest.json"),
    ] {
        let Ok(text) = std::fs::read_to_string(&p) else {
            continue;
        };
        let Ok(m) = Manifest::parse(&text) else {
            continue;
        };
        for i in m.images {
            let mut meta = meta_from_tags(&i.tags);
            meta.size = i.width.zip(i.height);
            out.entry(i.file).or_insert(meta);
        }
    }
    out
}

fn list_file(dir: &Path, names: &[&str]) -> Option<BTreeSet<String>> {
    names.iter().find_map(|n| {
        let text = std::fs::read_to_string(dir.join(n)).ok()?;
        Some(
            text.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(|l| file_name(Path::new(l)))
                .collect(),
        )
    })
}

/// What a folder's [`IGNORE_FILE`] says, by image file name.
#[derive(Debug, Default)]
struct IgnoreRead {
    /// The regions, and which of them the export's size floor made ([`Frame::floor_regions`]).
    regions: BTreeMap<String, (Vec<YoloBox>, Vec<usize>)>,
    parked: BTreeMap<String, Vec<String>>,
    /// [`IgnoreFile::size_floor`].
    size_floor: Option<SizeFloor>,
}

/// The [`IgnoreRead`] of a folder; no file = nothing (exports of older versions). A file that
/// cannot be read is an error: going on without it would teach the objects it lists as background.
fn read_ignore(dir: &Path) -> Result<IgnoreRead> {
    let p = dir.join(IGNORE_FILE);
    let bytes = match std::fs::read(&p) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(IgnoreRead::default()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", p.display())),
    };
    let file: IgnoreFile = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "{} is damaged, so this run cannot tell which objects not to train on; export the \
             frames again from the Dataset tab",
            p.display()
        )
    })?;
    if file.version > IGNORE_VERSION {
        bail!(
            "{} has version {}, newer than this build understands ({IGNORE_VERSION}); update Blue \
             Onyx Prism or export the frames again",
            p.display(),
            file.version
        );
    }
    // Normalized coordinates, 6 decimals: an f32 holds them.
    #[allow(clippy::cast_possible_truncation)]
    let unit = |v: f64| (v as f32).clamp(0.0, 1.0);
    let to_box = |b: &IgnoreBox| YoloBox {
        class: b.label.clone(),
        cx: unit(b.cx),
        cy: unit(b.cy),
        w: unit(b.w),
        h: unit(b.h),
    };
    Ok(IgnoreRead {
        regions: file
            .images
            .iter()
            .map(|(name, list)| {
                let floor: Vec<usize> = list
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| b.size_floor)
                    .map(|(i, _)| i)
                    .collect();
                (name.clone(), (list.iter().map(to_box).collect(), floor))
            })
            .collect(),
        parked: file.parked,
        size_floor: file.size_floor,
    })
}

/// The [`FrameNotes`] of a folder; None = no file, or one this build cannot read (a warning: the
/// notes rank frames and inform the report, a frame is the same without them).
fn read_frame_notes(dir: &Path, warnings: &mut Vec<String>) -> Option<FrameNotes> {
    let p = dir.join(FRAMES_FILE);
    let bytes = std::fs::read(&p).ok()?;
    match serde_json::from_slice::<FrameNotes>(&bytes) {
        Ok(n) if n.version <= FRAMES_VERSION => Some(n),
        Ok(n) => {
            warnings.push(format!(
                "{} has version {}, newer than this build understands: frames confirmed empty are \
                 not preferred; update Blue Onyx Prism or export the frames again",
                p.display(),
                n.version
            ));
            None
        }
        Err(e) => {
            warnings.push(format!(
                "{} is damaged ({e}): frames confirmed empty are not preferred; export the frames \
                 again from the Dataset tab",
                p.display()
            ));
            None
        }
    }
}

/// The image file names of a folder's `hard_negatives.json` with a box a reviewer rejected. No
/// file = none. A dismissal ([`bulk::is_dismissal`]) is a labelers run's own filter, and a reject
/// for the class list ([`bulk::OTHER_CLASS_REASON`]) says the class is unwanted, not that the box
/// was wrong; exports of older versions may list both.
fn read_rejected(dir: &Path, warnings: &mut Vec<String>) -> BTreeSet<String> {
    #[derive(Deserialize)]
    struct Rejected {
        file: String,
        #[serde(default)]
        reason: Option<String>,
    }
    let p = dir.join(HARD_NEGATIVES_FILE);
    let Ok(bytes) = std::fs::read(&p) else {
        return BTreeSet::new();
    };
    match serde_json::from_slice::<Vec<Rejected>>(&bytes) {
        Ok(v) => v
            .into_iter()
            .filter(|b| {
                let reason = b.reason.as_deref();
                !bulk::is_dismissal(reason) && reason != Some(bulk::OTHER_CLASS_REASON)
            })
            .map(|b| b.file)
            .collect(),
        Err(e) => {
            warnings.push(format!(
                "{} is damaged ({e}): frames with rejected boxes are not preferred as negatives; \
                 export the frames again from the Dataset tab",
                p.display()
            ));
            BTreeSet::new()
        }
    }
}

/// Whether `dir` is a review export that holds a label file per image (an empty one for a frame
/// without objects): its `export.json` lists the YOLO format. A file that cannot be read says
/// nothing against it. Other YOLO folders may leave out the label files of background images.
fn wrote_every_label(dir: &Path) -> bool {
    let Ok(bytes) = std::fs::read(dir.join("export.json")) else {
        return false;
    };
    serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .and_then(|v| v.get("formats")?.as_array().cloned())
        .is_none_or(|formats| formats.iter().any(|f| f == "yolo"))
}

/// Read a review export / YOLO folder (see the module docs).
pub fn read_export(dir: &Path) -> Result<Source> {
    if !dir.is_dir() {
        bail!("dataset folder {} does not exist", dir.display());
    }
    let classes = read_classes(dir)?;
    let meta = manifest_meta(dir);
    let mut ignore = read_ignore(dir)?;
    let mut warnings = Vec::new();
    let notes = read_frame_notes(dir, &mut warnings);
    let notes_known = notes.is_some();
    let names_of = |pick: fn(&FrameNotes) -> &Vec<String>| -> BTreeSet<&str> {
        notes
            .iter()
            .flat_map(|n| pick(n).iter().map(String::as_str))
            .collect()
    };
    let empty_confirmed = names_of(|n| &n.empty_confirmed);
    let bulk_accepted = names_of(|n| &n.bulk_accepted);
    let rejected = read_rejected(dir, &mut warnings);
    // (image, label file, hint)
    let mut entries: Vec<(PathBuf, PathBuf, SplitHint)> = Vec::new();
    let split_layout = SPLIT_DIRS
        .iter()
        .any(|(d, _)| dir.join("images").join(d).is_dir());
    if split_layout {
        for (d, hint) in SPLIT_DIRS {
            let img_dir = dir.join("images").join(d);
            if !img_dir.is_dir() {
                continue;
            }
            for img in images_in(&img_dir)? {
                let label = dir
                    .join("labels")
                    .join(d)
                    .join(format!("{}.txt", stem(&img)));
                entries.push((img, label, hint));
            }
        }
    } else {
        let img_dir = if dir.join("images").is_dir() {
            dir.join("images")
        } else {
            dir.to_path_buf()
        };
        let heldout = list_file(dir, &["heldout.txt", "test.txt"]);
        for img in images_in(&img_dir)? {
            let s = stem(&img);
            let label = [
                dir.join("labels").join(format!("{s}.txt")),
                img.with_extension("txt"),
            ]
            .into_iter()
            .find(|p| p.is_file())
            .unwrap_or_else(|| dir.join("labels").join(format!("{s}.txt")));
            let hint = match &heldout {
                Some(h) if h.contains(&file_name(&img)) => SplitHint::Heldout,
                Some(_) => SplitHint::Train,
                None => SplitHint::None,
            };
            entries.push((img, label, hint));
        }
    }
    if entries.is_empty() {
        bail!(
            "no .jpg/.png images in {} (images/, images/train/ or the folder itself)",
            dir.display()
        );
    }
    if entries.len() > MAX_FRAMES {
        bail!(
            "{} has {} images; at most {MAX_FRAMES} are read from one folder",
            dir.display(),
            entries.len()
        );
    }
    let every_label_written = wrote_every_label(dir);
    let mut missing = 0usize;
    let mut frames = Vec::with_capacity(entries.len());
    let size_floor = ignore.size_floor;
    for (img, label, split) in entries {
        let boxes = match std::fs::read_to_string(&label) {
            Ok(t) => {
                parse_labels(&t, &classes).with_context(|| format!("in {}", label.display()))?
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && !every_label_written => {
                missing += 1;
                Vec::new()
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => bail!(
                "{} is missing, but this export wrote a label file for every image, so the \
                 objects of {} are lost and it would train as empty; export the frames again \
                 from the Dataset tab",
                label.display(),
                file_name(&img)
            ),
            // Unknown is not empty: its objects would train as background.
            Err(e) => {
                return Err(e).with_context(|| {
                    format!(
                        "reading {}, so the objects of {} are unknown; check the file and its \
                         permissions, or export the frames again from the Dataset tab",
                        label.display(),
                        file_name(&img)
                    )
                });
            }
        };
        let name = file_name(&img);
        let m = meta.get(&name).cloned().unwrap_or_default();
        let n = meta_from_name(&stem(&img));
        let (regions, floor_regions) = ignore.regions.remove(&name).unwrap_or_default();
        frames.push(Frame {
            boxes,
            ignore: regions,
            floor_regions,
            parked: ignore.parked.remove(&name).unwrap_or_default(),
            scene: m.scene.or(n.scene),
            camera: m.camera.or(n.camera).unwrap_or_else(|| UNKNOWN.into()),
            day: m.day.or(n.day).unwrap_or_else(|| UNKNOWN.into()),
            split,
            hard_negative: m.hard_negative || empty_confirmed.contains(name.as_str()),
            human_rejected: rejected.contains(&name),
            bulk_accepted: bulk_accepted.contains(name.as_str()),
            size: m.size,
            image: img,
        });
    }
    if missing > 0 {
        warnings.push(format!(
            "{missing} image(s) in {} have no label file and count as frames without objects",
            dir.display()
        ));
    }
    Ok(Source {
        dir: dir.to_path_buf(),
        public: None,
        classes,
        frames,
        warnings,
        notes_known,
        size_floor,
    })
}

/// Read a public subset installed in `dir` from its benchmark-format `manifest` (pixel boxes,
/// tags). Images that are missing, unannotated or carry `ignore` regions (an unlabelled object
/// would be trained as background) are left out.
pub fn read_public_set(dir: &Path, manifest: &Manifest) -> Result<Source> {
    let mut classes: Vec<String> = Vec::new();
    let mut frames = Vec::new();
    let mut skipped = 0usize;
    for i in &manifest.images {
        let img = dir.join(&i.file);
        let Some(objects) = i.objects.as_ref().filter(|_| img.is_file()) else {
            skipped += 1;
            continue;
        };
        if objects.iter().any(|o| o.ignore || o.crowd) {
            skipped += 1;
            continue;
        }
        let size = match i.width.zip(i.height) {
            Some(s) => s,
            None => match image::image_dimensions(&img) {
                Ok(s) => s,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            },
        };
        let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
        let mut boxes = Vec::new();
        for o in objects {
            if !classes.contains(&o.label) {
                classes.push(o.label.clone());
            }
            let [x0, y0, x1, y1] = o.bbox;
            let (x0, x1) = ((x0 / w).clamp(0.0, 1.0), (x1 / w).clamp(0.0, 1.0));
            let (y0, y1) = ((y0 / h).clamp(0.0, 1.0), (y1 / h).clamp(0.0, 1.0));
            if x1 > x0 && y1 > y0 {
                boxes.push(YoloBox {
                    class: o.label.clone(),
                    cx: (x0 + x1) / 2.0,
                    cy: (y0 + y1) / 2.0,
                    w: x1 - x0,
                    h: y1 - y0,
                });
            }
        }
        let m = meta_from_tags(&i.tags);
        frames.push(Frame {
            image: img,
            boxes,
            ignore: Vec::new(),
            floor_regions: Vec::new(),
            parked: Vec::new(),
            scene: m.scene,
            camera: m.camera.unwrap_or_else(|| manifest.id.clone()),
            day: m.day.unwrap_or_else(|| UNKNOWN.into()),
            split: SplitHint::Train,
            hard_negative: m.hard_negative || i.tags.iter().any(|t| t == "empty"),
            human_rejected: false,
            bulk_accepted: false,
            size: Some(size),
        });
    }
    let mut warnings = Vec::new();
    if skipped > 0 {
        warnings.push(format!(
            "{skipped} image(s) of {} left out (not downloaded, unannotated or with ignore regions)",
            manifest.id
        ));
    }
    Ok(Source {
        dir: dir.to_path_buf(),
        public: Some(Public {
            id: manifest.id.clone(),
            license: manifest.license.clone(),
        }),
        classes,
        frames,
        warnings,
        notes_known: true,
        size_floor: None,
    })
}

// ---------------------------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------------------------

/// How a run assembles its data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Options {
    /// The run's classes in head order; None = every class of the sources (after `rename`), in
    /// first-seen order.
    pub classes: Option<Vec<String>>,
    /// Label renames (`motorbike` -> `motorcycle`); an empty target drops the label.
    #[serde(default)]
    pub rename: BTreeMap<String, String>,
    /// Scenes of the train / valid frames (the held-out frames keep every scene).
    #[serde(default)]
    pub scene: SceneFilter,
    /// What a train / valid frame with a box of an untrained label becomes.
    #[serde(default)]
    pub excluded: Excluded,
    /// What a train / valid frame with ignore regions becomes (`training.ignore`).
    #[serde(default)]
    pub ignore: IgnoreMode,
    pub negative_share: f32,
    pub heldout_share: f32,
    /// Share of the training frames used for validation (checkpoint choice), at least one.
    pub val_share: f32,
    /// Smallest validation split wanted (capped at a quarter of the candidates): a 10% share of a
    /// small pool is too few frames to choose a checkpoint on.
    #[serde(default = "default_min_valid")]
    pub min_valid: usize,
    pub oversample: u32,
    pub seed: u64,
    /// Step 7 (off by default: the plan of older versions).
    #[serde(default, skip_serializing_if = "Balance::is_off")]
    pub balance: Balance,
    /// The static cap (step 5): most train / valid frames kept per static spot of those showing
    /// nothing but objects at static spots ([`statics`]); None or 0 = off.
    #[serde(default = "default_static_cap")]
    pub static_cap: Option<usize>,
    /// The run's input size (its recipe's `imgsz`): a size floor region of a train / valid frame is
    /// judged again at it ([`restore_floor`]). None = not known (`train prepare` without a model):
    /// the plan only warns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imgsz: Option<u32>,
    /// `collect.size_floor` now, to compare each export's floor with ([`floor_warnings`]); None =
    /// no comparison (the floor is off, or a caller that does not say).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_floor: Option<SizeFloor>,
}

/// Default `Options.min_valid`.
pub const MIN_VALID: usize = 50;

fn default_min_valid() -> usize {
    MIN_VALID
}

/// Default `Options.static_cap`: [`PARKED_CAP`].
pub fn default_static_cap() -> Option<usize> {
    Some(PARKED_CAP)
}

impl Default for Options {
    fn default() -> Self {
        Self {
            classes: None,
            rename: BTreeMap::new(),
            scene: SceneFilter::ALL_SCENES,
            excluded: Excluded::Background,
            ignore: IgnoreMode::Background,
            negative_share: super::config::DEFAULT_NEGATIVE_SHARE,
            heldout_share: super::config::DEFAULT_HELDOUT_SHARE,
            val_share: 0.1,
            min_valid: MIN_VALID,
            oversample: super::config::DEFAULT_OVERSAMPLE,
            seed: 0,
            balance: Balance::default(),
            static_cap: default_static_cap(),
            imgsz: None,
            size_floor: None,
        }
    }
}

/// A frame of a source: `(source index, frame index)`.
pub type FrameRef = (usize, usize);

/// What goes where, and why.
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct Plan {
    pub classes: Vec<String>,
    /// Training frames and how often each repeats.
    pub train: Vec<(FrameRef, u32)>,
    pub valid: Vec<FrameRef>,
    pub heldout: Vec<FrameRef>,
    /// Hex SHA-256 of the sorted held-out frame keys, a frame's ignore regions part of its key:
    /// runs with the same value were benchmarked on the same frames, scored the same way
    /// (whatever their scene filter). The class list plays no part. A frame without regions keeps
    /// the key it always had, so such a set compares with runs of older versions.
    pub heldout_fingerprint: String,
    pub stats: Stats,
    pub warnings: Vec<String>,
    /// What balancing did (None when off).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub balance: Option<BalanceReport>,
}

/// Counts of a plan.
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct Stats {
    pub frames_read: usize,
    /// Train / valid frames left out by the scene filter.
    pub scene_filtered: usize,
    /// Train / valid frames left out for a box of an untrained label ([`Excluded::DropFrame`]).
    pub excluded_frames: usize,
    /// Train / valid frames left out for ignore regions ([`Frame::ignore`]): every such frame with
    /// [`IgnoreMode::DropFrame`], else `ignore_only + ignore_valid_day`.
    #[serde(skip_serializing_if = "is_zero")]
    pub ignore_frames: usize,
    /// [`IgnoreMode::Background`]: frames with ignore regions and no box of the run's classes, left
    /// out (neither a positive nor a negative).
    #[serde(skip_serializing_if = "is_zero")]
    pub ignore_only: usize,
    /// [`IgnoreMode::Background`]: frames with ignore regions on camera days validated on, left out
    /// of both (validation cannot ignore a region; the day belongs to validation).
    #[serde(skip_serializing_if = "is_zero")]
    pub ignore_valid_day: usize,
    /// [`IgnoreMode::Background`]: training frames trained with an unlabelled object (their ignore
    /// regions as background), repeats counted once.
    #[serde(skip_serializing_if = "is_zero")]
    pub ignore_kept: usize,
    /// [`IgnoreMode::Background`]: the regions of those frames per class (renamed like the boxes;
    /// a label renamed to nothing keeps its own name: the object is just as real).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub ignore_boxes_background: BTreeMap<String, usize>,
    /// Size floor regions of the train and validation frames that are not under the export's
    /// floor at the run's input ([`Options::imgsz`] above the floor's): given back as boxes of their
    /// label (mapped through the run's classes like any box), per label renamed like the boxes (a
    /// label renamed to nothing keeps its own name).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub floor_restored: BTreeMap<String, usize>,
    /// Of `negatives_dropped`, training negatives left out because positives with ignore regions
    /// left for their validation day ([`Self::ignore_valid_day`]) after the negative target was set
    /// from them: the target counts the positives that train.
    #[serde(skip_serializing_if = "is_zero")]
    pub negatives_trimmed: usize,
    pub negatives_dropped: usize,
    /// Boxes dropped per label not in the class list.
    pub boxes_dropped: BTreeMap<String, usize>,
    /// Training boxes per class (repeats counted once).
    pub train_boxes: BTreeMap<String, usize>,
    pub train_positives: usize,
    pub train_negatives: usize,
    pub public_frames: usize,
    /// `camera/day` groups held out by the plan (empty when the source chose).
    pub heldout_groups: Vec<String>,
    /// `camera/day` groups validated on, each whole and sorted (empty when validation fell back
    /// to the frame split, see [`valid`]).
    pub valid_groups: Vec<String>,
    /// Training frames the balance cap left out ([`Balance::cap_dominant`]).
    #[serde(skip_serializing_if = "is_zero")]
    pub balance_dropped: usize,
    /// Train / valid frames left out over the cap per parked spot ([`PARKED_CAP`]).
    #[serde(skip_serializing_if = "is_zero")]
    pub parked_capped: usize,
    /// Train / valid frames showing nothing but objects at static spots left out over
    /// [`Options::static_cap`] per spot and class.
    #[serde(skip_serializing_if = "is_zero")]
    pub static_capped: usize,
    /// Static spots found in the train / valid pool (0 with the cap off).
    #[serde(skip_serializing_if = "is_zero")]
    pub static_spots: usize,
    /// The (spot, class) quotas with frames showing nothing but static objects, most such frames
    /// first.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub static_caps: Vec<StaticCap>,
    /// Boxes per class (the run's) on the frames the static cap left out: what the warnings of a
    /// class with few or no training boxes name.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub static_capped_boxes: BTreeMap<String, usize>,
    /// Copies of a frame several exports hold, left out ([`frame_identity`]).
    #[serde(skip_serializing_if = "is_zero")]
    pub duplicates_dropped: usize,
    /// Of `train_negatives`, frames a reviewer confirmed empty or rejected a box on.
    #[serde(skip_serializing_if = "is_zero")]
    pub hard_negatives: usize,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

/// SHA-256 of `seed` and `key` as a number, for deterministic choices.
fn rank(seed: u64, key: &str) -> u64 {
    use sha2::Digest;
    let d = sha2::Sha256::digest(format!("{seed}:{key}").as_bytes());
    let mut b = [0u8; 8];
    for (o, i) in b.iter_mut().zip(d.iter()) {
        *o = *i;
    }
    u64::from_be_bytes(b)
}

/// `n x share`, rounded.
fn share_of(n: usize, share: f32) -> usize {
    super::estimate::to_u64(n as f64 * f64::from(share))
        .try_into()
        .unwrap_or(usize::MAX)
}

/// The boxes of `f` in the run's classes (renamed; others dropped and counted).
pub fn mapped_boxes<'a>(
    f: &'a Frame,
    opts: &'a Options,
    classes: &'a [String],
) -> impl Iterator<Item = (usize, &'a YoloBox)> + 'a {
    mapped(&f.boxes, opts, classes)
}

/// `boxes` in the run's classes (renamed; others dropped), with their class index.
fn mapped<'a>(
    boxes: &'a [YoloBox],
    opts: &'a Options,
    classes: &'a [String],
) -> impl Iterator<Item = (usize, &'a YoloBox)> + 'a {
    boxes.iter().filter_map(move |b| {
        let name = opts.rename.get(&b.class).unwrap_or(&b.class);
        classes.iter().position(|c| c == name).map(|i| (i, b))
    })
}

/// The frames of `frames` to leave out so that at most [`PARKED_CAP`] of them stand for each parked
/// spot, seeded. A frame with several spots stays while one of them has room, and counts for each.
fn over_parked_cap(frames: Vec<(u64, FrameRef, &[String])>) -> BTreeSet<FrameRef> {
    let mut frames = frames;
    frames.sort_by_key(|(rank, r, _)| (*rank, *r));
    let mut kept: BTreeMap<&str, usize> = BTreeMap::new();
    let mut over = BTreeSet::new();
    for (_, r, spots) in frames {
        let room = spots
            .iter()
            .any(|s| kept.get(s.as_str()).copied().unwrap_or(0) < PARKED_CAP);
        if room {
            for s in spots {
                *kept.entry(s).or_default() += 1;
            }
        } else {
            over.insert(r);
        }
    }
    over
}

/// One quota of the static cap: a static spot of a plan's train / valid pool and one class of the
/// boxes at it, with frames showing nothing but static objects ([`Stats::static_caps`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticCap {
    /// [`statics::Spot::id`]: one spot has one entry per class of the boxes at it.
    pub id: String,
    pub camera: String,
    /// The class (the run's) of this quota: frames whose box at the spot is of it.
    pub label: String,
    /// Frames of the pool showing nothing but static objects with a box of `label` at the spot.
    pub frames: usize,
    /// Of those, frames kept (more than the cap when frames kept for another quota count here
    /// too; never fewer than the cap or `frames`, whichever is smaller).
    pub kept: usize,
}

/// A quota of the static cap: (spot id, class).
type CapKey = (String, String);

/// A pool frame showing nothing but objects at static spots.
struct StaticFrame<'a> {
    r: FrameRef,
    /// [`frame_key`].
    key: String,
    scene: &'a str,
    day: &'a str,
    /// The (spot, class of its box there) quotas its boxes count toward, sorted, each once.
    spots: Vec<CapKey>,
}

/// The key that orders a frame of a quota for the static cap: (round within its (scene, day)
/// group, the day's place among the scene's days, the scene's place, the frame's rank, its key).
type StaticKey = (usize, usize, usize, u64, String);

/// The frames of `frames` to leave out so that at most `cap` of them stand for each (static spot,
/// class) quota, and per quota (frames, kept). Each class at a place has a quota of its own, so a
/// pickup labelled `truck` on two days at a car's place keeps its frames whatever the car's
/// number. Chosen for variety: per quota, round-robin over its (scene, day) groups (the first
/// frame of every group before the second of any), the scenes alternating (seeded order of scenes,
/// then of each scene's days, then of each group's frames). A frame with several quotas is placed
/// by its best key over them and kept while any of them is under the cap; it then counts toward
/// all of them, so a quota whose frames also show a fuller one can end over the cap. Every quota
/// keeps at least `min(cap, its frames)`: each of its frames is looked at while it is under the
/// cap, and kept then.
fn over_static_cap(
    frames: &[StaticFrame<'_>],
    cap: usize,
    seed: u64,
) -> (BTreeSet<FrameRef>, BTreeMap<CapKey, (usize, usize)>) {
    let mut by_spot: BTreeMap<&CapKey, Vec<usize>> = BTreeMap::new();
    for (i, f) in frames.iter().enumerate() {
        for s in &f.spots {
            by_spot.entry(s).or_default().push(i);
        }
    }
    let at = |i: usize| frames.get(i);
    let mut best: BTreeMap<usize, StaticKey> = BTreeMap::new();
    for (key, idx) in &by_spot {
        let spot = format!("{}#{}", key.0, key.1);
        let mut scenes: Vec<&str> = idx.iter().filter_map(|&i| at(i)).map(|f| f.scene).collect();
        scenes.sort_unstable();
        scenes.dedup();
        scenes.sort_by_key(|s| (rank(seed, &format!("static-scene:{s}")), *s));
        let mut groups: BTreeMap<(&str, &str), Vec<usize>> = BTreeMap::new();
        for &i in idx {
            if let Some(f) = at(i) {
                groups.entry((f.scene, f.day)).or_default().push(i);
            }
        }
        let mut day_round: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        for scene in &scenes {
            let mut days: Vec<&str> = groups
                .keys()
                .filter(|(s, _)| s == scene)
                .map(|(_, d)| *d)
                .collect();
            days.sort_by_key(|d| (rank(seed, &format!("static-day:{spot}/{scene}/{d}")), *d));
            for (n, d) in days.into_iter().enumerate() {
                day_round.insert((scene, d), n);
            }
        }
        for (g, mut list) in groups {
            let frame_rank = |i: &usize| at(*i).map_or(0, |f| rank(seed, &f.key));
            list.sort_by_key(|i| (frame_rank(i), at(*i).map(|f| f.key.clone())));
            let scene_pos = scenes.iter().position(|s| *s == g.0).unwrap_or(0);
            let day_pos = day_round.get(&g).copied().unwrap_or(0);
            for (round, i) in list.into_iter().enumerate() {
                let Some(f) = at(i) else { continue };
                let k: StaticKey = (round, day_pos, scene_pos, rank(seed, &f.key), f.key.clone());
                match best.get(&i) {
                    Some(b) if *b <= k => {}
                    _ => {
                        best.insert(i, k);
                    }
                }
            }
        }
    }
    let mut order: Vec<(StaticKey, FrameRef, usize)> = best
        .into_iter()
        .filter_map(|(i, k)| at(i).map(|f| (k, f.r, i)))
        .collect();
    order.sort();
    let mut counts: BTreeMap<CapKey, (usize, usize)> = by_spot
        .iter()
        .map(|(s, idx)| ((*s).clone(), (idx.len(), 0)))
        .collect();
    let mut over = BTreeSet::new();
    for (_, r, i) in order {
        let Some(f) = at(i) else { continue };
        let room = f
            .spots
            .iter()
            .any(|s| counts.get(s).is_some_and(|(_, kept)| *kept < cap));
        if room {
            for s in &f.spots {
                if let Some((_, kept)) = counts.get_mut(s) {
                    *kept += 1;
                }
            }
        } else {
            over.insert(r);
        }
    }
    (over, counts)
}

/// The boxes of `f` in the run's classes as `(class, normalised [x0, y0, x1, y1])`.
fn static_boxes<'a>(
    f: &'a Frame,
    opts: &'a Options,
    classes: &'a [String],
) -> impl Iterator<Item = (&'a str, [f32; 4])> + 'a {
    mapped_boxes(f, opts, classes).filter_map(move |(i, b)| {
        let c = classes.get(i)?;
        let (hw, hh) = (b.w / 2.0, b.h / 2.0);
        Some((c.as_str(), [b.cx - hw, b.cy - hh, b.cx + hw, b.cy + hh]))
    })
}

/// The scene of a frame for the static cap's groups.
fn scene_name(f: &Frame) -> &str {
    f.scene.as_deref().unwrap_or(UNKNOWN)
}

/// The static cap's sentence for a pre-flight and `train prepare`: how many frames showing nothing
/// but static objects were left out, of how many spots, and the largest; None when none were.
pub fn static_cap_note(stats: &Stats, cap: Option<usize>) -> Option<String> {
    if stats.static_capped == 0 {
        return None;
    }
    let capped: Vec<&StaticCap> = stats
        .static_caps
        .iter()
        .filter(|s| s.frames > s.kept)
        .collect();
    let spots: BTreeSet<&str> = capped.iter().map(|s| s.id.as_str()).collect();
    let largest: Vec<String> = capped
        .iter()
        .take(3)
        .map(|s| {
            format!(
                "a {} on camera {} ({} frames, {} kept)",
                s.label, s.camera, s.frames, s.kept
            )
        })
        .collect();
    Some(format!(
        "{} frame(s) showing nothing but objects parked at {} static spot(s) were left out (at \
         most {} kept per spot and class, across scenes and days); the largest: {}. Set \
         training.static_cap to 0 (or --static-cap 0) to keep them all",
        stats.static_capped,
        spots.len(),
        cap.unwrap_or(PARKED_CAP),
        largest.join("; ")
    ))
}

/// What the warnings of class `class` (few or no training boxes) add when the static cap left out
/// frames with its boxes ([`Stats::static_capped_boxes`]); None when it took none.
pub fn static_cap_clause(stats: &Stats, class: &str) -> Option<String> {
    let n = stats
        .static_capped_boxes
        .get(class)
        .copied()
        .filter(|n| *n > 0)?;
    Some(format!(
        "the static cap left out {n} box(es) of it on frames showing nothing but parked objects \
         (set training.static_cap to 0, or --static-cap 0, to keep them)"
    ))
}

/// The pre-flight's note of a run with the static cap on whose classes can form no static spot
/// ([`statics::may_be_static`]: a class renamed into a custom merged name such as `vehicle` is
/// none), so the cap leaves nothing out; None otherwise.
pub fn static_cap_idle_note(classes: &[String], cap: Option<usize>) -> Option<String> {
    if cap.is_none_or(|c| c == 0) || classes.iter().any(|c| statics::may_be_static(c)) {
        return None;
    }
    Some(format!(
        "The static cap is on, but no class of this run ({}) is one parked objects are found for: \
         only vehicle classes (car, truck, bus, ...) and other inanimate COCO classes form static \
         spots, so a class renamed into a name of its own (such as 'vehicle') is never capped. \
         Keep the COCO names to cap frames of parked objects",
        classes.join(", ")
    ))
}

/// The stable identity of a frame for the plan's seeded choices: its image path relative to its
/// source folder, with `/` separators. Never the absolute path: the plan must not change when
/// the export folder moves (or a test runs in a fresh temporary folder), so that a run is
/// reproducible.
fn frame_key(s: &Source, f: &Frame) -> String {
    let rel = f.image.strip_prefix(&s.dir).unwrap_or(&f.image);
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

/// What identifies a frame across review exports: its flat name (`<scene>.<camera>.<date>.<stem>`,
/// [`crate::collect::labels::export::flat_name`]), the one thing of the frame's path that survives
/// into every export; the stem holds the capture time and an id. Any other name says nothing (two
/// folders may well both hold `0001.jpg`): None, never a duplicate.
fn frame_identity(f: &Frame) -> Option<String> {
    let stem = stem(&f.image);
    let parts: Vec<&str> = stem.split('.').collect();
    match parts.as_slice() {
        [scene, _, _, _] if SCENES.contains(scene) => Some(file_name(&f.image)),
        _ => None,
    }
}

/// `review` without the copies of a frame several sources hold, and how many went. A held-out copy
/// wins (else the frame would train and be measured on); among equals the first source does.
fn without_duplicates(sources: &[Source], review: Vec<FrameRef>) -> (Vec<FrameRef>, usize) {
    let frame = |r: FrameRef| sources.get(r.0).and_then(|s| s.frames.get(r.1));
    let mut keep: BTreeMap<String, FrameRef> = BTreeMap::new();
    for &r in &review {
        let Some(id) = frame(r).and_then(frame_identity) else {
            continue;
        };
        let held = |r: FrameRef| frame(r).is_some_and(|f| f.split == SplitHint::Heldout);
        match keep.get_mut(&id) {
            Some(kept) => {
                if held(r) && !held(*kept) {
                    *kept = r;
                }
            }
            None => {
                keep.insert(id, r);
            }
        }
    }
    let before = review.len();
    let out: Vec<FrameRef> = review
        .into_iter()
        .filter(|&r| match frame(r).and_then(frame_identity) {
            Some(id) => keep.get(&id) == Some(&r),
            None => true,
        })
        .collect();
    let dropped = before - out.len();
    (out, dropped)
}

/// The plan's sentence on the frames with ignore regions in [`IgnoreMode::Background`]: how many
/// train with an unlabelled object, and how many were left out and why; None when there were none.
pub fn background_warning(s: &Stats) -> Option<String> {
    if s.ignore_kept + s.ignore_only + s.ignore_valid_day == 0 {
        return None;
    }
    let mut parts = vec![format!(
        "{} training frame(s) show an unsure or too-small object and train with it unlabelled, as \
         background (no trainer can leave a region out of its loss; the other objects of the \
         frame still teach)",
        s.ignore_kept
    )];
    if s.ignore_only > 0 {
        parts.push(format!(
            "{} frame(s) with nothing else were left out: an image whose only object is unlabelled \
             would teach \"nothing here\" exactly where something is",
            s.ignore_only
        ));
    }
    if s.ignore_valid_day > 0 {
        parts.push(format!(
            "{} frame(s) of camera days used for validation were left out of both (validation \
             cannot ignore a region, and the day is validation's)",
            s.ignore_valid_day
        ));
    }
    Some(format!(
        "{}. Held-out frames keep them as ignore regions: detections there do not count. To leave \
         every such frame out instead, set training.ignore to drop-frame (--ignore drop-frame)",
        parts.join("; ")
    ))
}

/// The `camera/day` group of a frame (held-out and validation splits).
fn group_key(f: &Frame) -> String {
    format!("{}/{}", f.camera, f.day)
}

/// Boxes per run class of `f`.
fn class_boxes(f: &Frame, opts: &Options, classes: &[String]) -> Vec<usize> {
    let mut n = vec![0usize; classes.len()];
    for (i, _) in mapped_boxes(f, opts, classes) {
        if let Some(c) = n.get_mut(i) {
            *c += 1;
        }
    }
    n
}

/// A held-out frame's part of [`Plan::heldout_fingerprint`]: `key`, then its ignore regions as
/// written (sorted, not mapped to the run's classes), so no regions = `key` alone.
fn heldout_key(key: String, f: &Frame) -> String {
    let mut regions: Vec<String> = f
        .ignore
        .iter()
        .map(|b| format!("{} {:.6} {:.6} {:.6} {:.6}", b.class, b.cx, b.cy, b.w, b.h))
        .collect();
    regions.sort_unstable();
    regions.iter().fold(key, |k, r| format!("{k}\t{r}"))
}

/// Hex SHA-256 of `keys` sorted, one per line (the held-out set's identity; empty set = the hash
/// of nothing).
fn fingerprint(keys: impl Iterator<Item = String>) -> String {
    use sha2::Digest;
    let mut keys: Vec<String> = keys.collect();
    keys.sort_unstable();
    let mut h = sha2::Sha256::new();
    for k in &keys {
        h.update(k.as_bytes());
        h.update(b"\n");
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// A YOLO box's short side at a model input of `input` pixels: `min(w, h) * input` on its numbers
/// normalised to the image, the size floor's measure ([`crate::collect::labels::under_floor`]).
fn short_at(b: &YoloBox, input: u32) -> f32 {
    b.w.min(b.h) * input as f32
}

/// The sources as a run at its input size trains on them ([`restore_floor`]), and the labels given
/// back per frame.
struct AtInput {
    sources: Vec<Source>,
    restored: BTreeMap<FrameRef, Vec<String>>,
}

/// Every size floor region ([`Frame::floor_regions`]) of a frame not in `heldout`, judged again at
/// the run's input ([`Options::imgsz`]) when that is larger than the input its export's floor was
/// applied at ([`Source::size_floor`]): a region that is not under the floor's `px` there is given
/// back as an ordinary box of its label (mapped through the run's classes later, like any box). The
/// held-out frames keep theirs, so the held-out set and its fingerprint do not change. None when
/// nothing is given back (no input size, no floor, no larger input, nothing over the floor there).
fn restore_floor(
    sources: &[Source],
    heldout: &BTreeSet<FrameRef>,
    opts: &Options,
) -> Option<AtInput> {
    let imgsz = opts.imgsz?;
    let mut copy: Option<Vec<Source>> = None;
    let mut restored: BTreeMap<FrameRef, Vec<String>> = BTreeMap::new();
    for (si, s) in sources.iter().enumerate() {
        let Some(floor) = s
            .size_floor
            .filter(|f| s.public.is_none() && f.is_on() && imgsz > f.input)
        else {
            continue;
        };
        for (fi, f) in s.frames.iter().enumerate() {
            if f.floor_regions.is_empty() || heldout.contains(&(si, fi)) {
                continue;
            }
            let back: BTreeSet<usize> = f
                .floor_regions
                .iter()
                .copied()
                .filter(|&i| {
                    f.ignore
                        .get(i)
                        .is_some_and(|b| short_at(b, imgsz) >= floor.px)
                })
                .collect();
            if back.is_empty() {
                continue;
            }
            let mut g = f.clone();
            g.ignore.clear();
            g.floor_regions.clear();
            let mut labels = Vec::new();
            for (i, b) in f.ignore.iter().enumerate() {
                if back.contains(&i) {
                    labels.push(b.class.clone());
                    g.boxes.push(b.clone());
                } else {
                    if f.floor_regions.contains(&i) {
                        g.floor_regions.push(g.ignore.len());
                    }
                    g.ignore.push(b.clone());
                }
            }
            let all = copy.get_or_insert_with(|| sources.to_vec());
            if let Some(slot) = all.get_mut(si).and_then(|s| s.frames.get_mut(fi)) {
                *slot = g;
                restored.insert((si, fi), labels);
            }
        }
    }
    copy.map(|sources| AtInput { sources, restored })
}

/// `sources` as a run of `plan` trains on them: [`restore_floor`] for the plan's held-out set, as
/// [`plan`] itself did. What [`write`] and the pre-flight report read.
pub fn at_run_input<'a>(
    sources: &'a [Source],
    plan: &Plan,
    opts: &Options,
) -> std::borrow::Cow<'a, [Source]> {
    let held: BTreeSet<FrameRef> = plan.heldout.iter().copied().collect();
    match restore_floor(sources, &held, opts) {
        Some(a) => std::borrow::Cow::Owned(a.sources),
        None => std::borrow::Cow::Borrowed(sources),
    }
}

/// A size floor in words: `10 px at 384`, or `no size floor`.
fn floor_words(f: Option<SizeFloor>) -> String {
    f.map_or_else(
        || "no size floor".to_string(),
        |f| format!("{} px at {}", f.px, f.input),
    )
}

/// What the plan says about the size floors of the exports ([`Source::size_floor`]): exports made
/// with different floors (an object one leaves out, another trains on), an export whose floor is
/// not the current one ([`Options::size_floor`]; made before it was set, or with another), and an
/// export floor applied at another input than the run's ([`Options::imgsz`]): unknown, or smaller,
/// where the boxes the run cannot learn are still labels (a larger one is [`restore_floor`]'s).
/// `heldout`: the held-out frames, whose labels are not counted.
fn floor_warnings(sources: &[Source], heldout: &BTreeSet<FrameRef>, opts: &Options) -> Vec<String> {
    let mut out = Vec::new();
    let own: Vec<(usize, &Source)> = sources
        .iter()
        .enumerate()
        .filter(|(_, s)| s.public.is_none())
        .collect();
    let floors: Vec<Option<SizeFloor>> = own.iter().map(|(_, s)| s.size_floor).collect();
    if floors.iter().any(|f| Some(f) != floors.first()) {
        let list: Vec<String> = own
            .iter()
            .map(|(_, s)| format!("{} ({})", s.dir.display(), floor_words(s.size_floor)))
            .collect();
        out.push(format!(
            "the exports were made with different size floors: {}. An object one of them leaves \
             out as too small trains as a label from another (and a frame in two exports keeps \
             the first one's copy); export them again with the current floor",
            list.join(", ")
        ));
    }
    if let Some(now) = opts.size_floor.filter(SizeFloor::is_on) {
        for (_, s) in &own {
            match s.size_floor {
                None => out.push(format!(
                    "{} was exported without a size floor (before it was set, or by an older \
                     version), so its objects under {} still train as labels; export again to \
                     apply the floor",
                    s.dir.display(),
                    floor_words(Some(now))
                )),
                Some(f) if f != now => out.push(format!(
                    "{} was exported with a size floor of {}, and the current floor is {}; export \
                     again to apply the current one",
                    s.dir.display(),
                    floor_words(Some(f)),
                    floor_words(Some(now))
                )),
                Some(_) => {}
            }
        }
    }
    for (si, s) in &own {
        let Some(floor) = s.size_floor.filter(SizeFloor::is_on) else {
            continue;
        };
        match opts.imgsz {
            None => {
                let regions: usize = s.frames.iter().map(|f| f.floor_regions.len()).sum();
                if regions > 0 {
                    out.push(format!(
                        "{} left out {regions} object(s) under its size floor ({}), and this plan \
                         does not know the size the run trains at: at a larger input some of them \
                         could be learned, at a smaller one some labels are too small for it. A \
                         run of a model judges them at its own input",
                        s.dir.display(),
                        floor_words(Some(floor))
                    ));
                }
            }
            Some(n) if n < floor.input => {
                let small = s
                    .frames
                    .iter()
                    .enumerate()
                    .filter(|(fi, _)| !heldout.contains(&(*si, *fi)))
                    .flat_map(|(_, f)| &f.boxes)
                    .filter(|b| short_at(b, n) < floor.px)
                    .count();
                out.push(format!(
                    "this run trains at {n} px, but the size floor of {} was applied at {}: \
                     objects of {:.1} to {} px at this run's input are still labels ({small} on \
                     its training frames), though it cannot learn them; export again with the \
                     floor's input set to {n}",
                    s.dir.display(),
                    floor.input,
                    floor.px * n as f32 / floor.input as f32,
                    floor.px,
                ));
            }
            Some(_) => {}
        }
    }
    out
}

/// The plan's sentence on the size floor regions given back at the run's input
/// ([`Stats::floor_restored`]); None when there are none.
fn restored_note(s: &Stats, sources: &[Source], opts: &Options) -> Option<String> {
    let n: usize = s.floor_restored.values().sum();
    if n == 0 {
        return None;
    }
    let floors: BTreeSet<String> = sources
        .iter()
        .filter_map(|x| x.size_floor)
        .map(|f| floor_words(Some(f)))
        .collect();
    let per: Vec<String> = s
        .floor_restored
        .iter()
        .map(|(c, k)| format!("{c} {k}"))
        .collect();
    Some(format!(
        "{n} object(s) the export's size floor ({}) left out are not under it at this run's {} px \
         input, so they train as labels again ({}); the held-out frames keep them as ignore \
         regions, so the benchmark is the export's",
        floors.into_iter().collect::<Vec<_>>().join(", "),
        opts.imgsz.unwrap_or_default(),
        per.join(", ")
    ))
}

/// Decide the splits (see the module docs). Pure and deterministic for a given `seed`.
pub fn plan(sources: &[Source], opts: &Options) -> Result<Plan> {
    // 1. Classes.
    let classes = match &opts.classes {
        Some(c) => c.clone(),
        None => {
            let mut v: Vec<String> = Vec::new();
            for s in sources {
                for c in &s.classes {
                    let name = opts.rename.get(c).unwrap_or(c);
                    if !name.is_empty() && !v.contains(name) {
                        v.push(name.clone());
                    }
                }
            }
            v
        }
    };
    let mut p = Plan {
        classes,
        ..Plan::default()
    };
    if p.classes.is_empty() {
        bail!("no class to train: the class list is empty");
    }
    let unique: BTreeSet<&String> = p.classes.iter().collect();
    if unique.len() != p.classes.len() || p.classes.iter().any(|c| c.trim().is_empty()) {
        bail!(
            "the class list has duplicate or empty names: {:?}",
            p.classes
        );
    }
    // 2. Every frame, review and public apart.
    let mut review: Vec<FrameRef> = Vec::new();
    let mut public: Vec<FrameRef> = Vec::new();
    for (si, s) in sources.iter().enumerate() {
        for fi in 0..s.frames.len() {
            p.stats.frames_read += 1;
            if s.public.is_some() {
                public.push((si, fi));
            } else {
                review.push((si, fi));
            }
        }
    }
    let (review, duplicates) = without_duplicates(sources, review);
    p.stats.duplicates_dropped = duplicates;
    if duplicates > 0 {
        p.warnings.push(format!(
            "{duplicates} frame(s) are in more than one of the ticked exports: one copy of each \
             stays (the held-out copy when there is one, so no frame both trains and is measured \
             on). Tick one export per set of frames to avoid this"
        ));
    }
    let frame = |r: FrameRef| -> Option<&Frame> { sources.get(r.0)?.frames.get(r.1) };
    let key_of = |r: FrameRef| -> String {
        sources
            .get(r.0)
            .zip(frame(r))
            .map(|(s, f)| frame_key(s, f))
            .unwrap_or_default()
    };
    // 3. Held out, from every review frame before the scene filter, so that runs with other
    // scene filters are benchmarked on the same frames (every scene). The sources' split when it
    // leaves both sides; else the same held-out share as a folder without a split: an export of
    // one camera day has nothing to split, and its one group goes all to train or (by its hash)
    // all to val. All held out stays as it is when public frames train.
    let is_held = |r: FrameRef| frame(r).is_some_and(|f| f.split == SplitHint::Heldout);
    let hinted = review.iter().any(|&r| is_held(r))
        && (review.iter().any(|&r| !is_held(r)) || !public.is_empty());
    let (pool, heldout): (Vec<FrameRef>, Vec<FrameRef>) = if hinted {
        review
            .iter()
            .partition(|&&r| frame(r).is_none_or(|f| f.split != SplitHint::Heldout))
    } else if opts.heldout_share <= 0.0 || review.is_empty() {
        (review.clone(), Vec::new())
    } else {
        let mut groups: BTreeMap<String, Vec<FrameRef>> = BTreeMap::new();
        for &r in &review {
            if let Some(f) = frame(r) {
                groups.entry(group_key(f)).or_default().push(r);
            }
        }
        if groups.len() >= 2 {
            let mut keys: Vec<&String> = groups.keys().collect();
            keys.sort_by_key(|k| (rank(opts.seed, k), (*k).clone()));
            let want = share_of(review.len(), opts.heldout_share).max(1);
            let mut held = Vec::new();
            let mut held_groups = Vec::new();
            // Never hold out every group.
            for k in keys.iter().take(keys.len() - 1) {
                if held.len() >= want {
                    break;
                }
                held.extend(groups.get(*k).into_iter().flatten().copied());
                held_groups.push((*k).clone());
            }
            p.stats.heldout_groups = held_groups;
            let held_set: BTreeSet<FrameRef> = held.iter().copied().collect();
            let rest = review
                .iter()
                .copied()
                .filter(|r| !held_set.contains(r))
                .collect();
            (rest, held)
        } else {
            p.warnings.push(
                "every frame is from one camera on one day: the held-out frames are a share of \
                 that day, and frames of one day are similar, so the benchmark of this run reads \
                 better than the model is; review and export frames from more cameras or days \
                 for a fair measurement"
                    .into(),
            );
            let mut v = review.clone();
            v.sort_by_key(|r| rank(opts.seed, &key_of(*r)));
            let n = share_of(v.len(), opts.heldout_share)
                .max(1)
                .min(v.len().saturating_sub(1));
            let held = v.split_off(v.len() - n);
            (v, held)
        }
    };
    p.heldout = heldout;
    p.heldout.sort_unstable();
    p.heldout_fingerprint = fingerprint(p.heldout.iter().map(|&r| match frame(r) {
        Some(f) => heldout_key(key_of(r), f),
        None => key_of(r),
    }));
    if p.heldout.is_empty() {
        p.warnings.push(
            "no frame is held out: the trained model is not benchmarked on your own frames".into(),
        );
    }
    // The train / valid pool at the run's input size: size floor regions it can learn are given
    // back as boxes (held-out frames keep theirs), and the exports' floors are compared. From here
    // on the plan reads these sources; `write` and the pre-flight make the same (`at_run_input`).
    let held_set: BTreeSet<FrameRef> = p.heldout.iter().copied().collect();
    p.warnings.extend(floor_warnings(sources, &held_set, opts));
    let at_input = restore_floor(sources, &held_set, opts);
    let restored_labels: BTreeMap<FrameRef, Vec<String>> = at_input
        .as_ref()
        .map(|a| a.restored.clone())
        .unwrap_or_default();
    let sources: &[Source] = at_input.as_ref().map_or(sources, |a| a.sources.as_slice());
    let frame = |r: FrameRef| -> Option<&Frame> { sources.get(r.0)?.frames.get(r.1) };
    let key_of = |r: FrameRef| -> String {
        sources
            .get(r.0)
            .zip(frame(r))
            .map(|(s, f)| frame_key(s, f))
            .unwrap_or_default()
    };
    let positive = |r: FrameRef| -> bool {
        frame(r).is_some_and(|f| mapped_boxes(f, opts, &p.classes).next().is_some())
    };
    // 4. Train / valid pool: the rest of the review frames plus the public ones, by scene, then
    // without frames with ignore regions (every one with `IgnoreMode::DropFrame`, else those with
    // no box of the run's classes), and without frames of untrained labels when the run drops
    // them.
    let mut in_scope: Vec<FrameRef> = Vec::new();
    let mut unknown_left_out = 0usize;
    for r in pool.into_iter().chain(public.iter().copied()) {
        let scene = frame(r).and_then(|f| f.scene.as_deref());
        if opts.scene.keeps(scene) {
            in_scope.push(r);
        } else {
            p.stats.scene_filtered += 1;
            unknown_left_out += usize::from(scene.is_none());
        }
    }
    if unknown_left_out > 0 {
        p.warnings.push(format!(
            "{unknown_left_out} training frame(s) have no scene tag and were left out by the {} \
             filter",
            opts.scene
        ));
    }
    let untrained = |b: &YoloBox| -> bool {
        let name = opts.rename.get(&b.class).unwrap_or(&b.class);
        !name.is_empty() && !p.classes.contains(name)
    };
    let scoped = in_scope.len();
    let mut pool: Vec<FrameRef> = Vec::with_capacity(scoped);
    for r in in_scope {
        let Some(f) = frame(r) else { continue };
        if !f.ignore.is_empty() {
            match opts.ignore {
                IgnoreMode::DropFrame => {
                    p.stats.ignore_frames += 1;
                    continue;
                }
                IgnoreMode::Background if mapped_boxes(f, opts, &p.classes).next().is_none() => {
                    p.stats.ignore_only += 1;
                    continue;
                }
                IgnoreMode::Background => {}
            }
        }
        if opts.excluded == Excluded::DropFrame && f.boxes.iter().any(untrained) {
            p.stats.excluded_frames += 1;
            continue;
        }
        pool.push(r);
    }
    if p.stats.ignore_frames > 0 {
        p.warnings.push(format!(
            "{} of {scoped} training frame(s) were left out for unsure or too-small boxes: no \
             trainer can skip such a box, so the object would train as background. Held-out \
             frames keep them, and detections on those boxes do not count; to train on a frame \
             again, give its box a class or reject it with another reason on the Review tab, or \
             train such frames with the object as background (training.ignore background, \
             --ignore background)",
            p.stats.ignore_frames
        ));
    }
    if p.stats.excluded_frames * 10 > scoped * 3 {
        p.warnings.push(format!(
            "{} of {scoped} training frame(s) ({:.0}%) were left out for boxes of classes this \
             run does not train; add those classes, or treat their objects as background",
            p.stats.excluded_frames,
            100.0 * p.stats.excluded_frames as f64 / scoped.max(1) as f64
        ));
    }
    // Labels the run does not train, over the frames it uses.
    for &r in pool.iter().chain(&p.heldout) {
        for b in frame(r).map(|f| f.boxes.as_slice()).unwrap_or_default() {
            let name = opts.rename.get(&b.class).unwrap_or(&b.class);
            if !p.classes.contains(name) {
                *p.stats.boxes_dropped.entry(name.clone()).or_default() += 1;
            }
        }
    }
    // 5. Negatives: about `negative_share` of the training pool (a frame whose only boxes are
    // of untrained labels is one, in the background mode).
    let (mut pos, mut neg): (Vec<FrameRef>, Vec<FrameRef>) =
        pool.iter().partition(|&&r| positive(r));
    // Before the negatives' target, which follows the positive frames. Held-out frames stay.
    let over = over_parked_cap(
        pos.iter()
            .filter_map(|&r| {
                let f = frame(r).filter(|f| !f.parked.is_empty())?;
                Some((rank(opts.seed, &key_of(r)), r, f.parked.as_slice()))
            })
            .collect(),
    );
    pos.retain(|r| !over.contains(r));
    p.stats.parked_capped = over.len();
    if !over.is_empty() {
        p.warnings.push(format!(
            "{} training frame(s) with nothing but parked-spot boxes were left out, over the cap \
             of {PARKED_CAP} per spot: the same crop again and again would dominate training",
            over.len()
        ));
    }
    // The static cap, on the positive frames the parked cap left. The spots are found by geometry
    // over the whole train / valid pool's own review frames (exports of any version are capped):
    // also the frames the parked cap took out, which are looks at the place like any other, and
    // the frames with only untrained labels (their boxes are not the run's, so they give no
    // observation). Only frames whose every box of the run's classes lies at a spot are capped,
    // per (spot, class of the box there).
    if let Some(cap) = opts.static_cap.filter(|&c| c > 0) {
        let own = |r: &FrameRef| -> Option<&Frame> {
            sources
                .get(r.0)
                .filter(|s| s.public.is_none())
                .and_then(|s| s.frames.get(r.1))
                .filter(|f| f.camera != UNKNOWN)
        };
        let classes = &p.classes;
        let table = statics::Table::build(pool.iter().flat_map(|r| {
            let key = key_of(*r);
            own(r).into_iter().flat_map(move |f| {
                let key = key.clone();
                static_boxes(f, opts, classes).map(move |(c, bbox)| statics::Obs {
                    camera: f.camera.clone(),
                    frame: key.clone(),
                    day: f.day.clone(),
                    scene: scene_name(f).to_string(),
                    label: c.to_string(),
                    bbox,
                })
            })
        }));
        let candidates: Vec<StaticFrame<'_>> = pos
            .iter()
            .filter_map(|r| {
                let f = own(r)?;
                let mut keys: BTreeSet<CapKey> = BTreeSet::new();
                for (class, b) in static_boxes(f, opts, classes) {
                    let s = table.spot_at(&f.camera, class, &b)?;
                    keys.insert((s.id.clone(), class.to_string()));
                }
                (!keys.is_empty()).then(|| StaticFrame {
                    r: *r,
                    key: key_of(*r),
                    scene: scene_name(f),
                    day: &f.day,
                    spots: keys.into_iter().collect(),
                })
            })
            .collect();
        let (over, counts) = over_static_cap(&candidates, cap, opts.seed);
        pos.retain(|r| !over.contains(r));
        p.stats.static_capped = over.len();
        p.stats.static_spots = table.len();
        for r in &over {
            if let Some(f) = frame(*r) {
                for (i, _) in mapped_boxes(f, opts, classes) {
                    if let Some(c) = classes.get(i) {
                        *p.stats.static_capped_boxes.entry(c.clone()).or_default() += 1;
                    }
                }
            }
        }
        let mut caps: Vec<StaticCap> = counts
            .into_iter()
            .filter_map(|((id, class), (frames, kept))| {
                let s = table.get(&id)?;
                Some(StaticCap {
                    camera: s.camera.clone(),
                    label: class,
                    id,
                    frames,
                    kept,
                })
            })
            .collect();
        caps.sort_by(|a, b| {
            b.frames
                .cmp(&a.frames)
                .then_with(|| a.id.cmp(&b.id))
                .then_with(|| a.label.cmp(&b.label))
        });
        p.stats.static_caps = caps;
    }
    if pos.is_empty() {
        bail!(
            "no training frame has a box of the classes {:?} (after the held-out split, the \
             scene filter, the frames with unsure or too-small boxes and the excluded-class \
             rule); review more frames with objects",
            p.classes
        );
    }
    let share = f64::from(opts.negative_share.clamp(0.0, 0.9));
    let want_for = |positives: usize| -> usize {
        let n = super::estimate::to_u64(positives as f64 * share / (1.0 - share));
        usize::try_from(n).unwrap_or(usize::MAX)
    };
    let want_neg = want_for(pos.len());
    // Frames a reviewer confirmed empty or rejected a box on first, then by rank.
    let neg_key = |r: &FrameRef| {
        let f = frame(*r);
        (
            !f.is_some_and(|f| f.hard_negative || f.human_rejected),
            rank(opts.seed, &key_of(*r)),
        )
    };
    let positives = pos.len();
    if neg.len() > want_neg {
        neg.sort_by_key(neg_key);
        p.stats.negatives_dropped = neg.len() - want_neg;
        neg.truncate(want_neg);
    } else if opts.negative_share > 0.0 && neg.len() < want_neg {
        p.warnings.push(format!(
            "only {} frame(s) without objects for {} with objects ({:.0}%, target {:.0}%): add \
             empty frames (hard negatives) to cut false positives",
            neg.len(),
            pos.len(),
            100.0 * neg.len() as f64 / (pos.len() + neg.len()) as f64,
            share * 100.0
        ));
    }
    let mut pool: Vec<FrameRef> = pos.into_iter().chain(neg.iter().copied()).collect();
    pool.sort_unstable();
    // 6. Validation (the trainer's checkpoint choice).
    let split = valid::pick_valid_groups(sources, &pool, opts, &p.classes);
    let valid = split.frames;
    // Frames with ignore regions of the camera days validated on: out of both.
    let valid_day: BTreeSet<FrameRef> = split.dropped.iter().copied().collect();
    p.stats.ignore_valid_day = valid_day.len();
    p.stats.valid_groups = split.groups;
    p.warnings.extend(split.warning);
    if valid.is_empty() {
        p.warnings.push(
            "too few frames for a validation split: the trainer validates on its training frames"
                .into(),
        );
    }
    let val_set: BTreeSet<FrameRef> = valid.iter().copied().collect();
    // The positives that left with their validation day do not train, so the negative target
    // counts the others: the excess training negatives go, the least wanted first (validation's
    // stay). Only frames with ignore regions leave this way, so a plan without any (every
    // drop-frame plan) is the same as before. The caps above are not run again: a capped spot may
    // keep a frame fewer.
    let leaving = valid_day.iter().filter(|r| positive(**r)).count();
    let want_now = want_for(positives.saturating_sub(leaving));
    let mut trimmed: BTreeSet<FrameRef> = BTreeSet::new();
    if leaving > 0 && neg.len() > want_now {
        let mut spare: Vec<FrameRef> = neg
            .iter()
            .copied()
            .filter(|r| !val_set.contains(r))
            .collect();
        spare.sort_by_key(neg_key);
        let excess = neg.len() - want_now;
        trimmed.extend(spare.iter().rev().take(excess).copied());
        p.stats.negatives_trimmed = trimmed.len();
        p.stats.negatives_dropped += trimmed.len();
    }
    let mixed = !public.is_empty();
    let base = |r: FrameRef| -> u32 {
        let is_public = sources.get(r.0).is_some_and(|s| s.public.is_some());
        if mixed && !is_public {
            opts.oversample.max(1)
        } else {
            1
        }
    };
    let train: Vec<FrameRef> = pool
        .into_iter()
        .filter(|r| !val_set.contains(r) && !valid_day.contains(r) && !trimmed.contains(r))
        .collect();
    // 7. Balance (opt-in): repeats per training frame, after the cap.
    let mut repeats: Vec<u32> = train.iter().map(|&r| base(r)).collect();
    let mut keep = vec![true; train.len()];
    if !opts.balance.is_off() {
        let items: Vec<balance::Item> = train
            .iter()
            .map(|&r| {
                let f = frame(r);
                balance::Item {
                    key: key_of(r),
                    base: base(r),
                    scene: f
                        .and_then(|f| f.scene.clone())
                        .unwrap_or_else(|| UNKNOWN.into()),
                    camera: f.map(|f| f.camera.clone()).unwrap_or_default(),
                    day: f.map(|f| f.day.clone()).unwrap_or_default(),
                    boxes: f.map_or_else(
                        || vec![0; p.classes.len()],
                        |f| class_boxes(f, opts, &p.classes),
                    ),
                    hard_negative: f.is_some_and(|f| {
                        f.hard_negative
                            || (f.human_rejected
                                && mapped_boxes(f, opts, &p.classes).next().is_none())
                    }),
                }
            })
            .collect();
        let out = balance::apply(&items, &p.classes, &opts.balance, opts.seed);
        repeats = out.repeats;
        keep = out.keep;
        p.stats.balance_dropped = out.report.dropped;
        p.warnings.extend(out.warnings);
        p.balance = Some(out.report);
    }
    for ((r, repeats), _) in train.into_iter().zip(repeats).zip(keep).filter(|(_, k)| *k) {
        let is_public = sources.get(r.0).is_some_and(|s| s.public.is_some());
        if let Some(f) = frame(r) {
            let mut any = false;
            for (i, _) in mapped_boxes(f, opts, &p.classes) {
                if let Some(c) = p.classes.get(i) {
                    *p.stats.train_boxes.entry(c.clone()).or_default() += 1;
                }
                any = true;
            }
            if any {
                p.stats.train_positives += 1;
            } else {
                p.stats.train_negatives += 1;
                p.stats.hard_negatives += usize::from(f.hard_negative || f.human_rejected);
            }
            if !f.ignore.is_empty() {
                p.stats.ignore_kept += 1;
                for b in &f.ignore {
                    let name = opts
                        .rename
                        .get(&b.class)
                        .filter(|n| !n.is_empty())
                        .unwrap_or(&b.class);
                    *p.stats
                        .ignore_boxes_background
                        .entry(name.clone())
                        .or_default() += 1;
                }
            }
        }
        p.stats.public_frames += usize::from(is_public);
        p.train.push((r, repeats));
    }
    // Without a split of its own the trainer validates on its training frames, those with ignore
    // regions left out (validation cannot ignore a region) unless nothing else is left.
    p.valid = if valid.is_empty() {
        let all: Vec<FrameRef> = p.train.iter().map(|(r, _)| *r).collect();
        let plain: Vec<FrameRef> = all
            .iter()
            .copied()
            .filter(|r| frame(*r).is_none_or(|f| f.ignore.is_empty()))
            .collect();
        if plain.is_empty() { all } else { plain }
    } else {
        valid
    };
    if opts.ignore == IgnoreMode::Background {
        p.stats.ignore_frames = p.stats.ignore_only + p.stats.ignore_valid_day;
        if let Some(w) = background_warning(&p.stats) {
            p.warnings.push(w);
        }
    }
    // Size floor regions given back at the run's input, on the frames it trains or validates on.
    let used: BTreeSet<FrameRef> = p
        .train
        .iter()
        .map(|(r, _)| *r)
        .chain(p.valid.iter().copied())
        .collect();
    for label in used.iter().filter_map(|r| restored_labels.get(r)).flatten() {
        let name = opts
            .rename
            .get(label)
            .filter(|n| !n.is_empty())
            .unwrap_or(label);
        *p.stats.floor_restored.entry(name.clone()).or_default() += 1;
    }
    if let Some(w) = restored_note(&p.stats, sources, opts) {
        p.warnings.push(w);
    }
    let mut heldout_boxes: BTreeMap<&str, usize> = BTreeMap::new();
    for &r in &p.heldout {
        if let Some(f) = frame(r) {
            for (i, _) in mapped_boxes(f, opts, &p.classes) {
                if let Some(c) = p.classes.get(i) {
                    *heldout_boxes.entry(c.as_str()).or_default() += 1;
                }
            }
        }
    }
    for c in &p.classes {
        if p.stats.train_boxes.contains_key(c) {
            continue;
        }
        // Each (spot, class) keeps min(cap, its frames), so the cap alone never takes a class's
        // last frame; the validation split can take what it kept. Said all the same.
        let capped = static_cap_clause(&p.stats, c)
            .map(|clause| format!("; {clause}"))
            .unwrap_or_default();
        p.warnings.push(match heldout_boxes.get(c.as_str()) {
            Some(n) if hinted => format!(
                "class '{c}' has no training box: its {n} box(es) are only in the held-out split \
                 (the export's val split: camera days held out for the benchmark), so the model \
                 cannot learn it; export again and hold out other cameras or days (Dataset tab \
                 export: holdout cameras / days), or review frames with it from the camera days \
                 that train{capped}"
            ),
            Some(n) => format!(
                "class '{c}' has no training box: its {n} box(es) are only in the held-out \
                 frames{}, so the model cannot learn it; review frames with it from other \
                 cameras or days, or hold out fewer frames (training.heldout_share){capped}",
                if p.stats.heldout_groups.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", p.stats.heldout_groups.join(", "))
                }
            ),
            None => format!(
                "class '{c}' has no training box: no frame has a box of it (after the scene \
                 filter, the frames left out for unsure or too-small boxes and the caps on parked \
                 and static frames), so the model cannot learn it; review frames with it or drop \
                 the class{capped}"
            ),
        });
    }
    Ok(p)
}

// ---------------------------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------------------------

/// What [`write`] produced.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Prepared {
    pub dir: PathBuf,
    pub classes: Vec<String>,
    /// Training image entries (repeats counted).
    pub train_images: usize,
    /// Unique training frames (repeats counted once).
    pub train_frames: usize,
    pub valid_images: usize,
    /// [`Stats::valid_groups`].
    pub valid_groups: Vec<String>,
    /// The held-out benchmark set (`dir:<path>`), when frames were held out.
    pub heldout_dir: Option<PathBuf>,
    pub heldout_images: usize,
    /// [`Plan::heldout_fingerprint`].
    pub heldout_fingerprint: String,
    pub warnings: Vec<String>,
    /// [`BalanceReport::summary`] when the run balanced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub balance_summary: Option<String>,
}

/// The [`Prepared`] of the dataset already in `dir` (its `dataset.json`, `<dir>/heldout`), for a
/// run that exports again without assembling anything. A file that does not read is an error: the
/// export checks the class list against it.
pub fn read_prepared(dir: &Path) -> Result<Prepared> {
    #[derive(Deserialize, Default)]
    struct SavedStats {
        #[serde(default)]
        valid_groups: Vec<String>,
    }
    #[derive(Deserialize)]
    struct Saved {
        classes: Vec<String>,
        train_images: usize,
        #[serde(default)]
        train_frames: usize,
        valid_images: usize,
        heldout_images: usize,
        #[serde(default)]
        heldout_fingerprint: String,
        #[serde(default)]
        warnings: Vec<String>,
        #[serde(default)]
        stats: SavedStats,
    }
    let p = dir.join("dataset.json");
    let bytes = std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
    let s: Saved =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", p.display()))?;
    let heldout = dir.join("heldout");
    Ok(Prepared {
        dir: dir.to_path_buf(),
        classes: s.classes,
        train_images: s.train_images,
        train_frames: s.train_frames,
        valid_images: s.valid_images,
        valid_groups: s.stats.valid_groups,
        heldout_dir: heldout.is_dir().then_some(heldout),
        heldout_images: s.heldout_images,
        heldout_fingerprint: s.heldout_fingerprint,
        warnings: s.warnings,
        balance_summary: None,
    })
}

/// `dataset.json`: what went into a prepared dataset.
#[derive(Debug, Clone, Serialize)]
pub struct Summary<'a> {
    pub version: u32,
    pub created: String,
    pub classes: &'a [String],
    pub options: &'a Options,
    pub sources: Vec<SourceSummary>,
    pub stats: &'a Stats,
    pub train_images: usize,
    /// Unique training frames (repeats counted once).
    pub train_frames: usize,
    pub valid_images: usize,
    pub heldout_images: usize,
    pub heldout_fingerprint: &'a str,
    pub warnings: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub balance: Option<&'a BalanceReport>,
}

/// One source in `dataset.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceSummary {
    pub dir: PathBuf,
    /// `review` or the public set's id.
    pub kind: String,
    pub license: String,
    pub frames: usize,
    pub used_train: usize,
    pub used_heldout: usize,
}

/// Licence noted for the user's own frames.
pub const OWN_FRAMES_LICENSE: &str = "own camera footage (private)";

fn safe_name(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    out.truncate(120);
    out
}

fn link_or_copy(src: &Path, dst: &Path) -> Result<()> {
    if std::fs::hard_link(src, dst).is_ok() {
        return Ok(());
    }
    std::fs::copy(src, dst)
        .map(|_| ())
        .with_context(|| format!("copying {} to {}", src.display(), dst.display()))
}

fn yolo_line(class: usize, b: &YoloBox) -> String {
    format!("{class} {:.6} {:.6} {:.6} {:.6}\n", b.cx, b.cy, b.w, b.h)
}

/// Tag prefix of the held-out manifest's own scene tags. The bare `day` / `night` / `ir` tags of
/// the benchmark also match images of built-in sets run beside the held-out set; these do not.
pub const HELDOUT_TAG: &str = "heldout:";
/// Held-out tag of every image ([`HELDOUT_TAG`] + `all`).
pub const HELDOUT_ALL: &str = "heldout:all";
/// Held-out tag of the images in the run's scene filter.
pub const HELDOUT_IN_SCOPE: &str = "heldout:in-scope";

/// Benchmark tags of a held-out frame of `scene` for a run filtered by `filter`: the bare scene
/// (the benchmark's own scene views), `heldout:<scene>`, `heldout:all` and, when the run
/// trained on its scene, `heldout:in-scope` (so [`super::bench::summarize`] reads exact AP per
/// scene and over the run's scenes from `accuracy.by_tag`).
pub fn heldout_tags(scene: Option<&str>, filter: SceneFilter) -> Vec<String> {
    let mut tags: Vec<String> = Vec::new();
    if let Some(s) = scene {
        tags.push(s.to_string());
        tags.push(format!("{HELDOUT_TAG}{s}"));
    }
    tags.push(HELDOUT_ALL.into());
    if filter.keeps(scene) {
        tags.push(HELDOUT_IN_SCOPE.into());
    }
    tags
}

/// Write `plan` into `out` (created; must not exist or be empty). The frames as the plan read them
/// ([`at_run_input`]: size floor regions a run at `opts.imgsz` can learn are boxes).
pub fn write(sources: &[Source], plan: &Plan, opts: &Options, out: &Path) -> Result<Prepared> {
    let at_input = at_run_input(sources, plan, opts);
    let sources: &[Source] = &at_input;
    if out.exists()
        && std::fs::read_dir(out)
            .with_context(|| format!("reading {}", out.display()))?
            .next()
            .is_some()
    {
        bail!("{} is not empty", out.display());
    }
    for d in ["images", "labels"] {
        std::fs::create_dir_all(out.join(d))
            .with_context(|| format!("creating {}", out.join(d).display()))?;
    }
    let frame = |r: FrameRef| -> Result<&Frame> {
        sources
            .get(r.0)
            .and_then(|s| s.frames.get(r.1))
            .ok_or_else(|| anyhow::anyhow!("plan refers to a missing frame {r:?}"))
    };
    // Unique file names: `<source>-<name>`.
    let mut names: BTreeMap<FrameRef, String> = BTreeMap::new();
    let mut used: BTreeSet<String> = BTreeSet::new();
    let mut name_of = |r: FrameRef, f: &Frame| -> String {
        if let Some(n) = names.get(&r) {
            return n.clone();
        }
        let base = safe_name(&format!("s{}-{}", r.0, file_name(&f.image)));
        let mut n = base.clone();
        let mut k = 1;
        while !used.insert(n.clone()) {
            k += 1;
            let p = Path::new(&base);
            n = format!(
                "{}-{k}.{}",
                stem(p),
                p.extension()
                    .map(|e| e.to_string_lossy())
                    .unwrap_or_default()
            );
        }
        names.insert(r, n.clone());
        n
    };
    let mut sizes: BTreeMap<FrameRef, (u32, u32)> = BTreeMap::new();
    let mut size_of = |r: FrameRef, f: &Frame| -> Result<(u32, u32)> {
        if let Some(s) = sizes.get(&r) {
            return Ok(*s);
        }
        let s = match f.size {
            Some(s) => s,
            None => image::image_dimensions(&f.image)
                .with_context(|| format!("reading the size of {}", f.image.display()))?,
        };
        sizes.insert(r, s);
        Ok(s)
    };
    let classes = &plan.classes;
    let label_text = |f: &Frame| -> String {
        mapped_boxes(f, opts, classes)
            .map(|(i, b)| yolo_line(i, b))
            .collect()
    };
    // A held-out frame's ground truth: read back from the very lines `label_text` writes, as the
    // benchmark reads YOLO labels, so an ordinary box scores as it would from `labels/`. Ignore
    // regions follow, through the same class map.
    let ground_truth = |f: &Frame, (w, h): (u32, u32)| -> Result<Vec<ManifestObject>> {
        let regions: String = mapped(&f.ignore, opts, classes)
            .map(|(i, b)| yolo_line(i, b))
            .collect();
        let mut objects = Vec::new();
        for (text, ignore) in [(label_text(f), false), (regions, true)] {
            for g in images::parse_yolo(&text, classes, w, h)? {
                objects.push(ManifestObject {
                    label: g.label,
                    bbox: g.bbox,
                    ignore,
                    crowd: false,
                });
            }
        }
        Ok(objects)
    };
    let mut written: BTreeSet<FrameRef> = BTreeSet::new();
    let put = |r: FrameRef, f: &Frame, n: &str, written: &mut BTreeSet<FrameRef>| -> Result<()> {
        if written.insert(r) {
            link_or_copy(&f.image, &out.join("images").join(n))?;
            let txt = out
                .join("labels")
                .join(format!("{}.txt", stem(Path::new(n))));
            std::fs::write(&txt, label_text(f))
                .with_context(|| format!("writing {}", txt.display()))?;
        }
        Ok(())
    };
    // COCO image entries per split.
    let coco = |entries: &[(FrameRef, String, (u32, u32))]| -> Result<serde_json::Value> {
        let mut imgs = Vec::new();
        let mut anns = Vec::new();
        for (id, (r, n, (w, h))) in entries.iter().enumerate() {
            imgs.push(serde_json::json!({
                "id": id,
                "file_name": format!("images/{n}"),
                "width": w,
                "height": h,
            }));
            let f = frame(*r)?;
            let (wf, hf) = (f64::from(*w), f64::from(*h));
            for (ci, b) in mapped_boxes(f, opts, classes) {
                let bw = f64::from(b.w) * wf;
                let bh = f64::from(b.h) * hf;
                let x = f64::from(b.cx) * wf - bw / 2.0;
                let y = f64::from(b.cy) * hf - bh / 2.0;
                let r2 = |v: f64| (v * 100.0).round() / 100.0;
                anns.push(serde_json::json!({
                    "id": anns.len(),
                    "image_id": id,
                    "category_id": ci,
                    "bbox": [r2(x.max(0.0)), r2(y.max(0.0)), r2(bw), r2(bh)],
                    "area": r2(bw * bh),
                    "iscrowd": 0,
                }));
            }
        }
        let cats: Vec<serde_json::Value> = classes
            .iter()
            .enumerate()
            .map(|(i, c)| serde_json::json!({"id": i, "name": c, "supercategory": "object"}))
            .collect();
        Ok(serde_json::json!({"images": imgs, "annotations": anns, "categories": cats}))
    };
    // Where each image of `images/` came from ([`IMAGE_SOURCES_FILE`]).
    let mut image_sources: BTreeMap<String, PathBuf> = BTreeMap::new();
    let mut train_lines = String::new();
    let mut train_entries = Vec::new();
    for &(r, repeats) in &plan.train {
        let f = frame(r)?;
        let n = name_of(r, f);
        put(r, f, &n, &mut written)?;
        image_sources.insert(n.clone(), f.image.clone());
        let s = size_of(r, f)?;
        for _ in 0..repeats.max(1) {
            train_lines.push_str(&format!("./images/{n}\n"));
            train_entries.push((r, n.clone(), s));
        }
    }
    let mut valid_lines = String::new();
    let mut valid_entries = Vec::new();
    for &r in &plan.valid {
        let f = frame(r)?;
        let n = name_of(r, f);
        put(r, f, &n, &mut written)?;
        image_sources.insert(n.clone(), f.image.clone());
        valid_lines.push_str(&format!("./images/{n}\n"));
        valid_entries.push((r, n.clone(), size_of(r, f)?));
    }
    let write_text = |name: &str, text: &str| -> Result<()> {
        std::fs::write(out.join(name), text)
            .with_context(|| format!("writing {}", out.join(name).display()))
    };
    let mut class_text = classes.join("\n");
    class_text.push('\n');
    write_text("classes.txt", &class_text)?;
    write_text("train.txt", &train_lines)?;
    write_text("valid.txt", &valid_lines)?;
    write_text(
        IMAGE_SOURCES_FILE,
        &serde_json::to_string_pretty(&image_sources)?,
    )?;
    let abs = std::path::absolute(out).unwrap_or_else(|_| out.to_path_buf());
    let mut yaml = format!(
        "# Blue Onyx Prism training dataset (Ultralytics)\npath: {}\ntrain: train.txt\nval: valid.txt\nnames:\n",
        serde_json::to_string(&abs.to_string_lossy())?
    );
    for (i, c) in classes.iter().enumerate() {
        yaml.push_str(&format!("  {i}: {}\n", serde_json::to_string(c)?));
    }
    write_text("data.yaml", &yaml)?;
    write_text(
        "train.coco.json",
        &serde_json::to_string(&coco(&train_entries)?)?,
    )?;
    write_text(
        "valid.coco.json",
        &serde_json::to_string(&coco(&valid_entries)?)?,
    )?;
    // Held-out benchmark set.
    let mut unreadable = 0usize;
    let heldout_dir = (!plan.heldout.is_empty()).then(|| out.join("heldout"));
    if let Some(hd) = &heldout_dir {
        std::fs::create_dir_all(hd.join("images"))?;
        std::fs::create_dir_all(hd.join("labels"))?;
        let mut m = Manifest {
            id: "heldout".into(),
            title: "Held-out frames of this training run".into(),
            description: "Your own frames, held out of training. Whole camera days when there \
                          are several (other days of the same camera may train), else a share of \
                          the one camera day."
                .into(),
            license: "private: your own camera footage, do not share".into(),
            scored_labels: Some(classes.clone()),
            ..Manifest::default()
        };
        for &r in &plan.heldout {
            let f = frame(r)?;
            // Boxes cannot be placed without the size: left out, as the benchmark always did.
            let Some(size) = size_of(r, f).ok().filter(|s| s.0 > 0 && s.1 > 0) else {
                unreadable += 1;
                continue;
            };
            let n = name_of(r, f);
            link_or_copy(&f.image, &hd.join("images").join(&n))?;
            let txt = hd
                .join("labels")
                .join(format!("{}.txt", stem(Path::new(&n))));
            std::fs::write(&txt, label_text(f))?;
            let mut tags = heldout_tags(f.scene.as_deref(), opts.scene);
            tags.push(format!("camera:{}", f.camera));
            tags.push(format!("date:{}", f.day));
            if f.hard_negative {
                tags.push("hard-negative".into());
            }
            m.images.push(images::ManifestImage {
                file: n,
                url: None,
                sha256: None,
                size: None,
                width: Some(size.0),
                height: Some(size.1),
                tags,
                objects: Some(ground_truth(f, size)?),
            });
        }
        std::fs::write(hd.join("classes.txt"), &class_text)?;
        std::fs::write(hd.join("manifest.json"), serde_json::to_vec_pretty(&m)?)?;
    }
    let mut sources_summary = Vec::new();
    for (si, s) in sources.iter().enumerate() {
        sources_summary.push(SourceSummary {
            dir: s.dir.clone(),
            kind: s
                .public
                .as_ref()
                .map_or_else(|| "review".into(), |p| p.id.clone()),
            license: s
                .public
                .as_ref()
                .map_or_else(|| OWN_FRAMES_LICENSE.into(), |p| p.license.clone()),
            frames: s.frames.len(),
            used_train: plan.train.iter().filter(|((x, _), _)| *x == si).count(),
            used_heldout: plan.heldout.iter().filter(|(x, _)| *x == si).count(),
        });
    }
    let mut warnings: Vec<String> = sources.iter().flat_map(|s| s.warnings.clone()).collect();
    warnings.extend(plan.warnings.iter().cloned());
    if unreadable > 0 {
        warnings.push(format!(
            "{unreadable} held-out image(s) could not be read and were left out of the benchmark \
             set: their boxes cannot be placed without the image size. The files are probably \
             damaged; replace them."
        ));
    }
    let summary = Summary {
        version: 1,
        created: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        classes,
        options: opts,
        sources: sources_summary,
        stats: &plan.stats,
        train_images: train_entries.len(),
        train_frames: plan.train.len(),
        valid_images: valid_entries.len(),
        heldout_images: plan.heldout.len(),
        heldout_fingerprint: &plan.heldout_fingerprint,
        warnings: &warnings,
        balance: plan.balance.as_ref(),
    };
    write_text("dataset.json", &serde_json::to_string_pretty(&summary)?)?;
    Ok(Prepared {
        dir: out.to_path_buf(),
        classes: classes.clone(),
        train_images: train_entries.len(),
        train_frames: plan.train.len(),
        valid_images: valid_entries.len(),
        valid_groups: plan.stats.valid_groups.clone(),
        heldout_dir,
        heldout_images: plan.heldout.len(),
        heldout_fingerprint: plan.heldout_fingerprint.clone(),
        warnings,
        balance_summary: plan.balance.as_ref().map(BalanceReport::summary),
    })
}

/// The file of a prepared dataset that names, for every image in `images/` (the names
/// `train.txt` and `valid.txt` list), the source image it was linked or copied from: a JSON object
/// `{"<name>": "<source path>"}`. The names are not the sources' own (a source prefix, a 120-byte
/// cut, a `-2` for a name taken twice), so a reader that must know which frame an image is (the
/// dataset check's unseen check) reads this instead of guessing.
pub const IMAGE_SOURCES_FILE: &str = "image_sources.json";

/// [`IMAGE_SOURCES_FILE`] of the prepared dataset in `dir`.
pub fn read_image_sources(dir: &Path) -> Result<BTreeMap<String, PathBuf>> {
    let p = dir.join(IMAGE_SOURCES_FILE);
    let bytes = std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", p.display()))
}

/// Read `dataset.json` sources of a prepared dataset (provenance).
pub fn read_summary_sources(dir: &Path) -> Vec<SourceSummary> {
    #[derive(Deserialize)]
    struct S {
        #[serde(default)]
        sources: Vec<SourceSummary>,
    }
    std::fs::read(dir.join("dataset.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<S>(&b).ok())
        .map(|s| s.sources)
        .unwrap_or_default()
}

/// The held-out fingerprint in the `dataset.json` beside a held-out set (`<data>/heldout`); None
/// for datasets of older versions.
pub fn read_heldout_fingerprint(heldout_dir: &Path) -> Option<String> {
    #[derive(Deserialize)]
    struct S {
        #[serde(default)]
        heldout_fingerprint: Option<String>,
    }
    let b = std::fs::read(heldout_dir.parent()?.join("dataset.json")).ok()?;
    serde_json::from_slice::<S>(&b)
        .ok()?
        .heldout_fingerprint
        .filter(|f| !f.is_empty())
}

// ---------------------------------------------------------------------------------------------
// Class inventory
// ---------------------------------------------------------------------------------------------

/// Boxes per scene of a class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
pub struct SceneCounts {
    pub day: usize,
    pub night: usize,
    pub ir: usize,
    /// Frames without a scene tag.
    pub unknown: usize,
}

/// One class of the chosen datasets (the Train tab's class picker, `train classes`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ClassCount {
    pub name: String,
    pub boxes: usize,
    /// Frames with at least one box of it.
    pub frames: usize,
    pub by_scene: SceneCounts,
}

/// Every class of `sources` after `rename` (an empty target drops the label), listed classes
/// without a box included, by box count (most first) then name. Pure.
pub fn inventory(sources: &[Source], rename: &BTreeMap<String, String>) -> Vec<ClassCount> {
    let renamed = |c: &String| -> Option<String> {
        let n = rename.get(c).unwrap_or(c);
        (!n.is_empty()).then(|| n.clone())
    };
    fn slot<'a>(by: &'a mut BTreeMap<String, ClassCount>, n: &str) -> &'a mut ClassCount {
        by.entry(n.to_string()).or_insert_with(|| ClassCount {
            name: n.to_string(),
            boxes: 0,
            frames: 0,
            by_scene: SceneCounts::default(),
        })
    }
    let mut by: BTreeMap<String, ClassCount> = BTreeMap::new();
    for s in sources {
        for c in s.classes.iter().filter_map(renamed) {
            slot(&mut by, &c);
        }
        for f in &s.frames {
            let mut seen: BTreeSet<String> = BTreeSet::new();
            for b in &f.boxes {
                let Some(n) = renamed(&b.class) else { continue };
                let cc = slot(&mut by, &n);
                cc.boxes += 1;
                let sc = &mut cc.by_scene;
                *match f.scene.as_deref() {
                    Some("day") => &mut sc.day,
                    Some("night") => &mut sc.night,
                    Some("ir") => &mut sc.ir,
                    _ => &mut sc.unknown,
                } += 1;
                if seen.insert(n) {
                    cc.frames += 1;
                }
            }
        }
    }
    let mut v: Vec<ClassCount> = by.into_values().collect();
    v.sort_by(|a, b| b.boxes.cmp(&a.boxes).then_with(|| a.name.cmp(&b.name)));
    v
}

#[cfg(test)]
mod tests;
