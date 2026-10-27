//! Dataset assembly: review exports (YOLO layout) plus an optional public subset in, the
//! trainer's input out (docs/TRAINING.md "Dataset").
//!
//! 1. **Read** ([`read_export`], [`read_public_set`]): class list, frames with their boxes
//!    (normalized YOLO boxes), scene, camera and day. Two export layouts are read:
//!    - the Review page's export (`src/collect/labels/export.rs`): `images/{train,val}/`,
//!      `labels/{train,val}/`, `classes.txt`, flat names `<scene>.<camera>.<date>.<stem>.jpg`;
//!      its `val` split is already held out by camera + day, so it becomes the held-out set;
//!    - a flat YOLO folder: `images/` (or the folder itself), `labels/` (or `<stem>.txt` beside
//!      the image), `classes.txt` or `data.yaml`, optional `test.txt` / `heldout.txt` listing
//!      held-out images; scene / camera / date tags from a `manifest.json` (benchmark format) when
//!      there is one.
//!
//!    A missing label file means "no objects" (the Ultralytics convention).
//! 2. **Plan** ([`plan`], pure): map the classes onto the run's class list (renames, drops);
//!    hold out review frames by camera + day (unless the export already did) from **every**
//!    review frame, before any filter, so runs with other scene filters are benchmarked on the
//!    same frames ([`Plan::heldout_fingerprint`]); then filter the train / valid pool by scene
//!    ([`SceneFilter`]) and, in [`Excluded::DropFrame`] mode, drop its frames with boxes of
//!    untrained labels; keep about `negative_share` frames without objects in training, take a
//!    validation split of `val_share` of the review training frames (never a class's last
//!    training box), and repeat the review frames `oversample` times when public frames are
//!    mixed in. Public frames are only ever trained on. Seeded choices rank a frame by its path
//!    relative to its source folder, so the plan is the same wherever the folders are
//!    (reproducible runs).
//! 3. **Write** ([`write`]): `images/` + `labels/` (hard links, else copies), `classes.txt`,
//!    `train.txt` / `valid.txt`, `data.yaml` (Ultralytics), `train.coco.json` /
//!    `valid.coco.json` (category ids 0..C-1, for RF-DETR and transformers), the held-out set as
//!    a benchmark `dir:` set (`heldout/`, every scene, tagged by [`heldout_tags`]), and
//!    `dataset.json` (what went in, for provenance).
//!
//! [`inventory`] counts the boxes per class and scene of the sources (the class picker).

use crate::benchmark::images::{self, Manifest};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

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
    /// `day`, `night` or `ir`; None = unknown.
    pub scene: Option<String>,
    pub camera: String,
    /// `YYYY-MM-DD` or `unknown`.
    pub day: String,
    pub split: SplitHint,
    /// Confirmed empty by the reviewer (a hard negative).
    pub hard_negative: bool,
    /// `(width, height)` when the source knows it.
    pub size: Option<(u32, u32)>,
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

fn is_image(p: &Path) -> bool {
    p.is_file()
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
        "{} has no classes.txt or data.yaml: it is not a YOLO export (export from the Review page \
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

/// Read a review export / YOLO folder (see the module docs).
pub fn read_export(dir: &Path) -> Result<Source> {
    if !dir.is_dir() {
        bail!("dataset folder {} does not exist", dir.display());
    }
    let classes = read_classes(dir)?;
    let meta = manifest_meta(dir);
    let mut warnings = Vec::new();
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
            scene: m.scene.or(n.scene),
            camera: m.camera.or(n.camera).unwrap_or_else(|| UNKNOWN.into()),
            day: m.day.or(n.day).unwrap_or_else(|| UNKNOWN.into()),
            split,
            hard_negative: m.hard_negative,
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
            scene: m.scene,
            camera: m.camera.unwrap_or_else(|| manifest.id.clone()),
            day: m.day.unwrap_or_else(|| UNKNOWN.into()),
            split: SplitHint::Train,
            hard_negative: m.hard_negative || i.tags.iter().any(|t| t == "empty"),
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
    pub oversample: u32,
    pub seed: u64,
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
            oversample: super::config::DEFAULT_OVERSAMPLE,
            seed: 0,
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
    /// Hex SHA-256 of the sorted held-out frame keys: runs with the same value were benchmarked
    /// on the same frames (whatever their scene filter).
    pub heldout_fingerprint: String,
    pub stats: Stats,
    pub warnings: Vec<String>,
}

/// Counts of a plan.
#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct Stats {
    pub frames_read: usize,
    /// Train / valid frames left out by the scene filter.
    pub scene_filtered: usize,
    /// Train / valid frames left out for a box of an untrained label ([`Excluded::DropFrame`]).
    pub excluded_frames: usize,
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
    f.boxes.iter().filter_map(move |b| {
        let name = opts.rename.get(&b.class).unwrap_or(&b.class);
        classes.iter().position(|c| c == name).map(|i| (i, b))
    })
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
    // scene filters are benchmarked on the same frames (every scene).
    let hinted = review
        .iter()
        .any(|&r| frame(r).is_some_and(|f| f.split != SplitHint::None));
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
                groups
                    .entry(format!("{}/{}", f.camera, f.day))
                    .or_default()
                    .push(r);
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
                "every frame is from one camera on one day: the held-out frames share it with \
                 the training frames, so the benchmark of this run is optimistic; export frames \
                 from more cameras or days"
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
    p.heldout_fingerprint = fingerprint(p.heldout.iter().map(|&r| key_of(r)));
    if p.heldout.is_empty() {
        p.warnings.push(
            "no frame is held out: the trained model is not benchmarked on your own scenes".into(),
        );
    }
    // 4. Train / valid pool: the rest of the review frames plus the public ones, by scene, then
    // without frames of untrained labels when the run drops them.
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
        if opts.excluded == Excluded::DropFrame && f.boxes.iter().any(untrained) {
            p.stats.excluded_frames += 1;
            continue;
        }
        pool.push(r);
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
    let (pos, mut neg): (Vec<FrameRef>, Vec<FrameRef>) = pool.iter().partition(|&&r| positive(r));
    if pos.is_empty() {
        bail!(
            "no training frame has a box of the classes {:?} (after the held-out split, the \
             scene filter and the excluded-class rule); review more frames with objects",
            p.classes
        );
    }
    let share = f64::from(opts.negative_share.clamp(0.0, 0.9));
    let want_neg = super::estimate::to_u64(pos.len() as f64 * share / (1.0 - share));
    let want_neg = usize::try_from(want_neg).unwrap_or(usize::MAX);
    if neg.len() > want_neg {
        // Keep the reviewer's confirmed negatives first, then by rank.
        neg.sort_by_key(|r| {
            let f = frame(*r);
            (
                !f.is_some_and(|f| f.hard_negative),
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
    // 6. Validation (the trainer's checkpoint choice): `val_share` of the review frames of the
    // pool when there are any (else of the pool), never a frame holding the last training box of
    // a class (so at least one positive frame stays in training). The size is a share of the
    // frames it is taken from: a share of the whole pool, public frames included, once took
    // every review frame, so a class only the review frames have was never trained.
    let key = |r: &FrameRef| rank(opts.seed ^ 0x5eed, &key_of(*r));
    let mut cands: Vec<FrameRef> = pool
        .iter()
        .copied()
        .filter(|r| sources.get(r.0).is_some_and(|s| s.public.is_none()))
        .collect();
    if cands.is_empty() {
        cands = pool.clone();
    }
    cands.sort_by_key(key);
    let n_val = if cands.len() >= 2 {
        share_of(cands.len(), opts.val_share).clamp(1, cands.len() - 1)
    } else {
        0
    };
    let boxes_per_class = |r: FrameRef| -> Vec<usize> {
        let mut n = vec![0usize; p.classes.len()];
        if let Some(f) = frame(r) {
            for (i, _) in mapped_boxes(f, opts, &p.classes) {
                if let Some(c) = n.get_mut(i) {
                    *c += 1;
                }
            }
        }
        n
    };
    let mut left = vec![0usize; p.classes.len()];
    for &r in &pool {
        for (l, n) in left.iter_mut().zip(boxes_per_class(r)) {
            *l += n;
        }
    }
    let mut valid: Vec<FrameRef> = Vec::new();
    for r in cands {
        if valid.len() >= n_val {
            break;
        }
        let need = boxes_per_class(r);
        if need.iter().zip(&left).any(|(n, l)| *n > 0 && l <= n) {
            continue;
        }
        for (l, n) in left.iter_mut().zip(&need) {
            *l -= n;
        }
        valid.push(r);
    }
    valid.sort_unstable();
    if valid.is_empty() {
        p.warnings.push(
            "too few frames for a validation split: the trainer validates on its training frames"
                .into(),
        );
    }
    let val_set: BTreeSet<FrameRef> = valid.iter().copied().collect();
    let mixed = !public.is_empty();
    for r in pool.into_iter().filter(|r| !val_set.contains(r)) {
        let is_public = sources.get(r.0).is_some_and(|s| s.public.is_some());
        let repeats = if mixed && !is_public {
            opts.oversample.max(1)
        } else {
            1
        };
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
                 (the export's val split: cameras and days held out for the benchmark), so the \
                 model cannot learn it; export again and hold out other cameras or days (Review \
                 page export: holdout cameras / days), or review frames with it from the \
                 training cameras and days"
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
                 filter), so the model cannot learn it; review frames with it or drop the class"
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
    pub valid_images: usize,
    /// The held-out benchmark set (`dir:<path>`), when frames were held out.
    pub heldout_dir: Option<PathBuf>,
    pub heldout_images: usize,
    /// [`Plan::heldout_fingerprint`].
    pub heldout_fingerprint: String,
    pub warnings: Vec<String>,
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
    pub valid_images: usize,
    pub heldout_images: usize,
    pub heldout_fingerprint: &'a str,
    pub warnings: &'a [String],
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
    let heldout_dir = (!plan.heldout.is_empty()).then(|| out.join("heldout"));
    if let Some(hd) = &heldout_dir {
        std::fs::create_dir_all(hd.join("images"))?;
        std::fs::create_dir_all(hd.join("labels"))?;
        let mut m = Manifest {
            id: "heldout".into(),
            title: "Held-out frames of this training run".into(),
            description: "Your own frames from cameras and days the model did not train on.".into(),
            license: "private: your own camera footage, do not share".into(),
            scored_labels: Some(classes.clone()),
            ..Manifest::default()
        };
        for &r in &plan.heldout {
            let f = frame(r)?;
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
                width: f.size.map(|s| s.0),
                height: f.size.map(|s| s.1),
                tags,
                objects: None,
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
    let summary = Summary {
        version: 1,
        created: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        classes,
        options: opts,
        sources: sources_summary,
        stats: &plan.stats,
        train_images: train_entries.len(),
        valid_images: valid_entries.len(),
        heldout_images: plan.heldout.len(),
        heldout_fingerprint: &plan.heldout_fingerprint,
        warnings: &warnings,
    };
    write_text("dataset.json", &serde_json::to_string_pretty(&summary)?)?;
    Ok(Prepared {
        dir: out.to_path_buf(),
        classes: classes.clone(),
        train_images: train_entries.len(),
        valid_images: valid_entries.len(),
        heldout_dir,
        heldout_images: plan.heldout.len(),
        heldout_fingerprint: plan.heldout_fingerprint.clone(),
        warnings,
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

/// One class of the chosen datasets (the Training page's class picker, `train classes`).
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
