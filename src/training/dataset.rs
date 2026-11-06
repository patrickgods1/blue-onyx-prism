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
//!    A missing label file means "no objects" (the Ultralytics convention). `ignore.json`
//!    ([`IgnoreFile`]) lists real objects not to train on (unsure, too small) and the frames with
//!    nothing but parked-spot boxes; no file = none. `frames.json` ([`FrameNotes`]) names the
//!    frames confirmed empty or accepted in bulk and `hard_negatives.json` the rejected boxes,
//!    whichever formats the export wrote.
//! 2. **Plan** ([`plan`], pure): map the classes onto the run's class list (renames, drops);
//!    drop the copies of a frame that several exports hold (a held-out copy wins); hold out review
//!    frames by whole camera days (unless the exports already did; one camera day: frame by frame,
//!    with a warning) from **every** review frame, before any filter, so runs with other scene
//!    filters are benchmarked on the same frames ([`Plan::heldout_fingerprint`]); then filter the
//!    train / valid pool by scene
//!    ([`SceneFilter`]), drop its frames with ignore regions (no trainer here can leave a region
//!    out of its loss, so the object would train as background; the held-out set keeps them, with
//!    the regions `ignore` in its manifest) and, in [`Excluded::DropFrame`] mode, its frames with
//!    boxes of untrained labels; cap the frames with nothing but parked-spot boxes at
//!    [`PARKED_CAP`] per spot (the same crop again and again would dominate training); keep about
//!    `negative_share` frames without objects in training (frames a reviewer confirmed empty or
//!    rejected a box on first), take a
//!    validation split of `val_share` of the review training frames as whole camera days
//!    ([`valid`], never a class's last training box), and repeat the review frames
//!    `oversample` times when public frames are mixed in. Public frames are only ever trained on.
//!    Seeded choices rank a frame by its path relative to its source folder, so the plan is the
//!    same wherever the folders are (reproducible runs). Step 7, opt-in ([`Balance`],
//!    [`balance`]): repeat the training frames
//!    of rare classes / scenes / cameras / camera days (and optionally drop frames of dominant
//!    ones); validation and held-out frames are never balanced.
//! 3. **Write** ([`write`]): `images/` + `labels/` (hard links, else copies), `classes.txt`,
//!    `train.txt` / `valid.txt`, `data.yaml` (Ultralytics), `train.coco.json` /
//!    `valid.coco.json` (category ids 0..C-1, for RF-DETR and transformers), the held-out set as
//!    a benchmark `dir:` set (`heldout/`, every scene, tagged by [`heldout_tags`]), and
//!    `dataset.json` (what went in, for provenance).
//!
//! [`inventory`] counts the boxes per class and scene of the sources (the class picker).

use crate::benchmark::images::{self, Manifest, ManifestObject};
use crate::collect::labels::bulk;
use crate::collect::labels::export::{
    FRAMES_FILE, FRAMES_VERSION, FrameNotes, HARD_NEGATIVES_FILE, IGNORE_FILE, IGNORE_VERSION,
    IgnoreBox, IgnoreFile,
};
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
    /// Real objects not to train on (`ignore.json`): not in `boxes`, so the frame cannot be
    /// trained on, and `ignore` when the frame is benchmarked.
    pub ignore: Vec<YoloBox>,
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
    regions: BTreeMap<String, Vec<YoloBox>>,
    parked: BTreeMap<String, Vec<String>>,
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
            .map(|(name, list)| (name.clone(), list.iter().map(to_box).collect()))
            .collect(),
        parked: file.parked,
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
    let mut missing = 0usize;
    let mut frames = Vec::with_capacity(entries.len());
    for (img, label, split) in entries {
        let boxes = match std::fs::read_to_string(&label) {
            Ok(t) => {
                parse_labels(&t, &classes).with_context(|| format!("in {}", label.display()))?
            }
            Err(_) => {
                missing += 1;
                Vec::new()
            }
        };
        let name = file_name(&img);
        let m = meta.get(&name).cloned().unwrap_or_default();
        let n = meta_from_name(&stem(&img));
        frames.push(Frame {
            boxes,
            ignore: ignore.regions.remove(&name).unwrap_or_default(),
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
}

/// Default `Options.min_valid`.
pub const MIN_VALID: usize = 50;

fn default_min_valid() -> usize {
    MIN_VALID
}

impl Default for Options {
    fn default() -> Self {
        Self {
            classes: None,
            rename: BTreeMap::new(),
            scene: SceneFilter::ALL_SCENES,
            excluded: Excluded::Background,
            negative_share: super::config::DEFAULT_NEGATIVE_SHARE,
            heldout_share: super::config::DEFAULT_HELDOUT_SHARE,
            val_share: 0.1,
            min_valid: MIN_VALID,
            oversample: super::config::DEFAULT_OVERSAMPLE,
            seed: 0,
            balance: Balance::default(),
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
    /// Train / valid frames left out for an ignore region ([`Frame::ignore`]).
    #[serde(skip_serializing_if = "is_zero")]
    pub ignore_frames: usize,
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
    let positive = |r: FrameRef| -> bool {
        frame(r).is_some_and(|f| mapped_boxes(f, opts, &p.classes).next().is_some())
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
    // 4. Train / valid pool: the rest of the review frames plus the public ones, by scene, then
    // without frames with ignore regions, and without frames of untrained labels when the run
    // drops them.
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
            p.stats.ignore_frames += 1;
            continue;
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
             again, give its box a class or reject it with another reason on the Review tab",
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
    if pos.is_empty() {
        bail!(
            "no training frame has a box of the classes {:?} (after the held-out split, the \
             scene filter, the frames with unsure or too-small boxes and the excluded-class \
             rule); review more frames with objects",
            p.classes
        );
    }
    let share = f64::from(opts.negative_share.clamp(0.0, 0.9));
    let want_neg = super::estimate::to_u64(pos.len() as f64 * share / (1.0 - share));
    let want_neg = usize::try_from(want_neg).unwrap_or(usize::MAX);
    if neg.len() > want_neg {
        // Frames a reviewer confirmed empty or rejected a box on first, then by rank.
        neg.sort_by_key(|r| {
            let f = frame(*r);
            (
                !f.is_some_and(|f| f.hard_negative || f.human_rejected),
                rank(opts.seed, &key_of(*r)),
            )
        });
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
    let mut pool: Vec<FrameRef> = pos.into_iter().chain(neg).collect();
    pool.sort_unstable();
    // 6. Validation (the trainer's checkpoint choice).
    let split = valid::pick_valid_groups(sources, &pool, opts, &p.classes);
    let valid = split.frames;
    p.stats.valid_groups = split.groups;
    p.warnings.extend(split.warning);
    if valid.is_empty() {
        p.warnings.push(
            "too few frames for a validation split: the trainer validates on its training frames"
                .into(),
        );
    }
    let val_set: BTreeSet<FrameRef> = valid.iter().copied().collect();
    let mixed = !public.is_empty();
    let base = |r: FrameRef| -> u32 {
        let is_public = sources.get(r.0).is_some_and(|s| s.public.is_some());
        if mixed && !is_public {
            opts.oversample.max(1)
        } else {
            1
        }
    };
    let train: Vec<FrameRef> = pool.into_iter().filter(|r| !val_set.contains(r)).collect();
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
        }
        p.stats.public_frames += usize::from(is_public);
        p.train.push((r, repeats));
    }
    p.valid = if valid.is_empty() {
        p.train.iter().map(|(r, _)| *r).collect()
    } else {
        valid
    };
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
        p.warnings.push(match heldout_boxes.get(c.as_str()) {
            Some(n) if hinted => format!(
                "class '{c}' has no training box: its {n} box(es) are only in the held-out split \
                 (the export's val split: camera days held out for the benchmark), so the model \
                 cannot learn it; export again and hold out other cameras or days (Dataset tab \
                 export: holdout cameras / days), or review frames with it from the camera days \
                 that train"
            ),
            Some(n) => format!(
                "class '{c}' has no training box: its {n} box(es) are only in the held-out \
                 frames{}, so the model cannot learn it; review frames with it from other \
                 cameras or days, or hold out fewer frames (training.heldout_share)",
                if p.stats.heldout_groups.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", p.stats.heldout_groups.join(", "))
                }
            ),
            None => format!(
                "class '{c}' has no training box: no frame has a box of it (after the scene \
                 filter and the frames left out for unsure or too-small boxes), so the model \
                 cannot learn it; review frames with it or drop the class"
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

/// Write `plan` into `out` (created; must not exist or be empty).
pub fn write(sources: &[Source], plan: &Plan, opts: &Options, out: &Path) -> Result<Prepared> {
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
    let mut train_lines = String::new();
    let mut train_entries = Vec::new();
    for &(r, repeats) in &plan.train {
        let f = frame(r)?;
        let n = name_of(r, f);
        put(r, f, &n, &mut written)?;
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
