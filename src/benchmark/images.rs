//! Benchmark image sets ("datasets"): the built-in sets pinned by the manifests in
//! `assets/bench/<id>.json` (downloaded on demand to `<data_root>/bench/<id>/`), user directories
//! (with optional COCO JSON or YOLO txt ground truth and an optional `manifest.json` with tags),
//! and the embedded sample image (always available, works offline).
//!
//! Manifest (`assets/bench/<id>.json`, also accepted as `<dir>/manifest.json`):
//!
//! ```json
//! { "id": "coco-cctv", "title": "...", "description": "...", "license": "...",
//!   "attribution": "...", "source": "https://...", "revision": "...",
//!   "images": [ { "file": "000000123.jpg", "url": "https://...", "sha256": "<hex>",
//!                 "size": 123456, "width": 640, "height": 480,
//!                 "tags": ["day", "cctv", "complexity:easy", "res:sd", "small-objects"],
//!                 "objects": [ { "label": "person", "bbox": [x_min, y_min, x_max, y_max],
//!                                "ignore": false, "crowd": false } ] } ] }
//! ```
//!
//! Optional manifest fields:
//!
//! - `"transform": "grayscale"`: every image is converted after decode (before preprocessing)
//!   to its BT.601 luma replicated to three channels ([`crate::image::RgbImage::to_gray3`]),
//!   a rough proxy for night IR monochrome. Only the pixels the model sees change: downloads,
//!   SHA-256 checks and the files on disk are those of the source images.
//! - `"images_from": "<set id>"`: the set reuses another built-in set's images and download
//!   directory (`bench/<that id>/`, resource `bench:<that id>`). An empty `images` list inherits
//!   that set's images and ground truth (`exdark-ir` = `exdark-night` + `grayscale`).
//!
//! `crowd` marks a COCO `iscrowd` region (it implies `ignore`; detections inside it are
//! ignored by intersection over the detection's area, like pycocotools).
//!
//! A user directory may hold just a JSON array of `images` entries in `manifest.json` (tags, and
//! `objects` as ground truth). Without ground truth a set is scored against pseudo ground truth
//! from a reference model (see `super::pseudo_ground_truth`).

use super::metrics::{BBox, GtBox};
use crate::config::DatasetRef;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

/// Built-in set ids the project curates (`assets/bench/<id>.json`). An id listed here whose
/// manifest is not embedded in this build is skipped with a warning.
pub const KNOWN_SET_IDS: [&str; 5] = [
    "coco-cctv",
    "bmd45-cctv",
    "exdark-night",
    "exdark-ir",
    "nir-night",
];
/// Sets a config without a `benchmark.datasets` key runs. The night/IR sets (`exdark-ir`,
/// `nir-night`) are opt-in: they add ~68 MB of downloads and ~360 images per model.
pub const DEFAULT_SET_IDS: [&str; 3] = ["coco-cctv", "bmd45-cctv", "exdark-night"];
/// Id of the embedded sample image.
pub const SAMPLE_ID: &str = "sample";
/// Prefix of a user-directory dataset reference (`dir:/path/to/images`).
pub const DIR_PREFIX: &str = "dir:";
/// Subdirectory of the data root that holds downloaded sets.
pub const BENCH_DIR: &str = "bench";
/// Image file extensions picked up from directories.
pub const IMAGE_EXTENSIONS: [&str; 3] = ["jpg", "jpeg", "png"];

/// Manifests embedded in this build: `(id, json)`. The curated manifests are added here (with
/// `include_str!`) once they land in `assets/bench/`.
pub const BUILTIN_MANIFESTS: &[(&str, &str)] = &[
    (
        "coco-cctv",
        include_str!("../../assets/bench/coco-cctv.json"),
    ),
    (
        "bmd45-cctv",
        include_str!("../../assets/bench/bmd45-cctv.json"),
    ),
    (
        "exdark-night",
        include_str!("../../assets/bench/exdark-night.json"),
    ),
    (
        "exdark-ir",
        include_str!("../../assets/bench/exdark-ir.json"),
    ),
    (
        "nir-night",
        include_str!("../../assets/bench/nir-night.json"),
    ),
];

/// A per-set pixel transform applied after decode (manifest `transform`, see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageTransform {
    /// BT.601 luma replicated to three channels (an NIR monochrome proxy).
    Grayscale,
}

impl ImageTransform {
    pub fn apply(self, img: &mut crate::image::RgbImage) {
        match self {
            ImageTransform::Grayscale => img.to_gray3(),
        }
    }
}

/// One object of a manifest image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestObject {
    pub label: String,
    /// `[x_min, y_min, x_max, y_max]`, absolute pixels of the original image.
    pub bbox: BBox,
    #[serde(default)]
    pub ignore: bool,
    /// A COCO `iscrowd` region (implies `ignore`): a detection inside it is ignored by
    /// intersection over the detection's area, as pycocotools does, not by IoU.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub crowd: bool,
}

/// One image of a manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManifestImage {
    pub file: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size: Option<u64>,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Ground truth; None = unannotated.
    #[serde(default)]
    pub objects: Option<Vec<ManifestObject>>,
}

/// A set manifest (see the module docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Manifest {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub license: String,
    #[serde(default)]
    pub attribution: String,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub revision: String,
    /// Only these labels are scored on this set: ground truth of other labels is dropped and
    /// predictions of other labels are neither true nor false positives (a set annotated for
    /// vehicles only). None = every label.
    #[serde(default)]
    pub scored_labels: Option<Vec<String>>,
    /// Pixel transform applied to every image after decode (see the module docs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<ImageTransform>,
    /// Built-in set whose images (and download directory) this set reuses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images_from: Option<String>,
    #[serde(default)]
    pub images: Vec<ManifestImage>,
}

impl Manifest {
    /// A full manifest object, or a bare array of images.
    pub fn parse(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)?;
        if v.is_array() {
            return Ok(Self {
                images: serde_json::from_value(v)?,
                ..Self::default()
            });
        }
        Ok(serde_json::from_value(v)?)
    }

    /// Total download size of the images.
    pub fn size(&self) -> u64 {
        self.images.iter().filter_map(|i| i.size).sum()
    }

    /// Every image has `objects`.
    pub fn has_ground_truth(&self) -> bool {
        !self.images.is_empty() && self.images.iter().all(|i| i.objects.is_some())
    }

    /// Id of the set whose download directory holds this set's images: `images_from`, else
    /// the set itself.
    pub fn images_set_id(&self) -> &str {
        match self.images_from.as_deref() {
            Some(s) if !s.trim().is_empty() => s.trim(),
            _ => &self.id,
        }
    }
}

/// Parse the embedded manifests and resolve `images_from` (an empty `images` list inherits the
/// source set's images). A set whose source is missing, itself borrowed, or that borrows from
/// itself is dropped with an error log.
fn parse_builtins(list: &[(&str, &str)]) -> Vec<Manifest> {
    let mut out: Vec<Manifest> = list
        .iter()
        .filter_map(|(id, json)| match Manifest::parse(json) {
            Ok(mut m) => {
                if m.id.is_empty() {
                    m.id = id.to_string();
                }
                Some(m)
            }
            Err(e) => {
                tracing::error!("embedded benchmark manifest {id} is invalid: {e:#}");
                None
            }
        })
        .collect();
    let sources: Vec<(String, Option<Vec<ManifestImage>>)> = out
        .iter()
        .map(|m| {
            (
                m.id.clone(),
                m.images_from.is_none().then(|| m.images.clone()),
            )
        })
        .collect();
    out.retain_mut(|m| {
        let Some(src) = m.images_from.clone() else {
            return true;
        };
        let found = sources
            .iter()
            .find(|(id, _)| id.eq_ignore_ascii_case(src.trim()) && !id.eq_ignore_ascii_case(&m.id));
        match found {
            Some((id, Some(images))) => {
                m.images_from = Some(id.clone());
                if m.images.is_empty() {
                    m.images = images.clone();
                }
                true
            }
            _ => {
                tracing::error!(
                    "embedded benchmark manifest {} takes its images from '{src}', which is not \
                     an embedded set with its own images; the set is left out",
                    m.id
                );
                false
            }
        }
    });
    out
}

/// The embedded built-in manifests, parsed once.
pub fn builtin_manifests() -> &'static [Manifest] {
    static M: std::sync::OnceLock<Vec<Manifest>> = std::sync::OnceLock::new();
    M.get_or_init(|| parse_builtins(BUILTIN_MANIFESTS))
}

pub fn builtin_manifest(id: &str) -> Option<&'static Manifest> {
    builtin_manifests()
        .iter()
        .find(|m| m.id.eq_ignore_ascii_case(id.trim()))
}

/// Where a built-in set is downloaded to: `<data_root>/bench/<id>/`, or the directory of the
/// set its images come from (`images_from`).
pub fn builtin_dir(data_root: &Path, id: &str) -> PathBuf {
    let dir_id = builtin_manifest(id).map_or(id, Manifest::images_set_id);
    data_root.join(BENCH_DIR).join(dir_id)
}

/// Where the ground truth of a set came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", tag = "kind", content = "reference")]
pub enum GroundTruth {
    /// None: scored against pseudo ground truth, or not at all.
    None,
    /// `objects` of the manifest.
    Manifest,
    /// COCO instances JSON.
    Coco,
    /// YOLO txt labels.
    Yolo,
    /// Detections of a reference model ("rt-detrv2-x on openvino:cpu"): relative, not ground
    /// truth.
    Pseudo(String),
}

impl GroundTruth {
    pub fn is_real(&self) -> bool {
        matches!(
            self,
            GroundTruth::Manifest | GroundTruth::Coco | GroundTruth::Yolo
        )
    }

    pub fn describe(&self) -> String {
        match self {
            GroundTruth::None => "no ground truth".into(),
            GroundTruth::Manifest => "ground truth (manifest)".into(),
            GroundTruth::Coco => "ground truth (COCO JSON)".into(),
            GroundTruth::Yolo => "ground truth (YOLO labels)".into(),
            GroundTruth::Pseudo(r) => format!("relative to {r}, not ground truth"),
        }
    }
}

/// Largest dataset image file the benchmark reads (64 MiB).
pub const MAX_IMAGE_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// Read `path`, failing (with an error naming the file) when it is larger than `max` bytes.
fn read_limited(path: &Path, max: u64) -> Result<Vec<u8>> {
    use std::io::Read;
    let f = std::fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let len = f
        .metadata()
        .with_context(|| format!("reading {}", path.display()))?
        .len();
    if len > max {
        bail!(
            "{} is {len} bytes, larger than the {max}-byte limit for benchmark images",
            path.display()
        );
    }
    let mut buf = Vec::new();
    // `take` guards against a file that grows between the size check and the read.
    f.take(max.saturating_add(1))
        .read_to_end(&mut buf)
        .with_context(|| format!("reading {}", path.display()))?;
    if buf.len() as u64 > max {
        bail!(
            "{} is larger than the {max}-byte limit for benchmark images",
            path.display()
        );
    }
    Ok(buf)
}

/// One image of a set.
#[derive(Debug, Clone, PartialEq)]
pub struct SetImage {
    /// File name relative to the set directory.
    pub file: String,
    /// None = the embedded sample.
    pub path: Option<PathBuf>,
    pub width: u32,
    pub height: u32,
    /// Manifest tags plus the derived `res:` bucket.
    pub tags: Vec<String>,
    /// Ground truth (raw labels); None = unannotated.
    pub gt: Option<Vec<GtBox>>,
}

impl SetImage {
    /// Encoded image bytes. Files larger than [`MAX_IMAGE_FILE_BYTES`] are rejected before
    /// they are read; decoding goes through `crate::image::decode` (pixel ceiling).
    pub fn bytes(&self) -> Result<Cow<'static, [u8]>> {
        match &self.path {
            None => Ok(Cow::Borrowed(super::DEFAULT_IMAGE)),
            Some(p) => read_limited(p, MAX_IMAGE_FILE_BYTES).map(Cow::Owned),
        }
    }

    /// `res:` bucket of this image.
    /// `res:` bucket: the manifest's `res:` tag when it names a known bucket, else derived from
    /// the size.
    pub fn resolution(&self) -> &'static str {
        self.tags
            .iter()
            .filter_map(|t| t.strip_prefix("res:"))
            .find_map(|r| {
                RESOLUTION_BUCKETS
                    .iter()
                    .find(|b| b.eq_ignore_ascii_case(r))
            })
            .copied()
            .unwrap_or_else(|| resolution_bucket(self.width, self.height))
    }
}

/// Resolution bucket from the long side: `sd` (up to 960 px, e.g. 640x480, 640x640), `hd` (up
/// to 1280, 720p), `fhd` (up to 1920, 1080p), `4mp+` (larger: 2560x1440 and up). The manifests
/// use the same `res:` vocabulary.
pub fn resolution_bucket(w: u32, h: u32) -> &'static str {
    match w.max(h) {
        0..=960 => "sd",
        961..=1280 => "hd",
        1281..=1920 => "fhd",
        _ => "4mp+",
    }
}

/// Bucket order for tables.
pub const RESOLUTION_BUCKETS: [&str; 4] = ["sd", "hd", "fhd", "4mp+"];

fn add_resolution_tag(tags: &mut Vec<String>, w: u32, h: u32) {
    if w > 0 && h > 0 && !tags.iter().any(|t| t.starts_with("res:")) {
        tags.push(format!("res:{}", resolution_bucket(w, h)));
    }
}

/// Kind of a dataset source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SetKind {
    Builtin,
    Dir,
    Sample,
}

/// A loaded dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageSet {
    /// `coco-cctv`, `sample`, `dir:<path>` (or the user's `name`).
    pub id: String,
    pub title: String,
    pub kind: SetKind,
    /// Directory the files are in (None for the sample).
    pub dir: Option<PathBuf>,
    pub license: String,
    pub attribution: String,
    pub source: String,
    pub ground_truth: GroundTruth,
    /// See [`Manifest::scored_labels`].
    pub scored_labels: Option<Vec<String>>,
    /// See [`Manifest::transform`]: applied to every image after decode.
    pub transform: Option<ImageTransform>,
    pub images: Vec<SetImage>,
}

impl ImageSet {
    /// The embedded sample image (no ground truth).
    pub fn sample() -> Self {
        // Header only: no decode.
        let (width, height) = image::ImageReader::new(std::io::Cursor::new(super::DEFAULT_IMAGE))
            .with_guessed_format()
            .ok()
            .and_then(|r| r.into_dimensions().ok())
            .unwrap_or((0, 0));
        let mut tags = vec!["day".to_string(), "general".to_string()];
        add_resolution_tag(&mut tags, width, height);
        Self {
            id: SAMPLE_ID.into(),
            title: "Embedded sample (dog, bicycle, car)".into(),
            kind: SetKind::Sample,
            dir: None,
            license: "public domain test image".into(),
            attribution: String::new(),
            source: String::new(),
            ground_truth: GroundTruth::None,
            scored_labels: None,
            transform: None,
            images: vec![SetImage {
                file: super::DEFAULT_IMAGE_NAME.into(),
                path: None,
                width,
                height,
                tags,
                gt: None,
            }],
        }
    }

    /// One image file (`--image`), no ground truth.
    pub fn single_file(path: &Path) -> Result<Self> {
        let (w, h) =
            image::image_dimensions(path).with_context(|| format!("reading {}", path.display()))?;
        let mut tags = Vec::new();
        add_resolution_tag(&mut tags, w, h);
        Ok(Self {
            id: path.display().to_string(),
            title: path.display().to_string(),
            kind: SetKind::Dir,
            dir: path.parent().map(Path::to_path_buf),
            license: String::new(),
            attribution: String::new(),
            source: String::new(),
            ground_truth: GroundTruth::None,
            scored_labels: None,
            transform: None,
            images: vec![SetImage {
                file: path
                    .file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                path: Some(path.to_path_buf()),
                width: w,
                height: h,
                tags,
                gt: None,
            }],
        })
    }

    /// A built-in set from its manifest and the files present in `dir`. Missing files are
    /// skipped (`missing` counts them).
    pub fn from_manifest(m: &Manifest, dir: &Path) -> (Self, usize) {
        let mut images = Vec::new();
        let mut missing = 0;
        for i in &m.images {
            let path = dir.join(&i.file);
            if !safe_file_name(&i.file) || !path.is_file() {
                missing += 1;
                continue;
            }
            images.push(manifest_image(i, path));
        }
        let gt = if m.has_ground_truth() {
            GroundTruth::Manifest
        } else {
            GroundTruth::None
        };
        (
            Self {
                id: m.id.clone(),
                title: if m.title.is_empty() {
                    m.id.clone()
                } else {
                    m.title.clone()
                },
                kind: SetKind::Builtin,
                dir: Some(dir.to_path_buf()),
                license: m.license.clone(),
                attribution: m.attribution.clone(),
                source: m.source.clone(),
                ground_truth: gt,
                scored_labels: m.scored_labels.clone(),
                transform: m.transform,
                images,
            },
            missing,
        )
    }

    /// Keep at most `n` images (0 = all), evenly spread over the set so every part of it (and
    /// so every kind of scene, if the set is ordered) is represented.
    pub fn limit(&mut self, n: usize) {
        let len = self.images.len();
        if n == 0 || n >= len {
            return;
        }
        // i * len / n < len for i < n.
        let picked: Vec<SetImage> = (0..n)
            .filter_map(|i| self.images.get(i * len / n).cloned())
            .collect();
        self.images = picked;
    }

    /// Images with ground truth (real or pseudo).
    pub fn annotated(&self) -> bool {
        self.images.iter().all(|i| i.gt.is_some()) && !self.images.is_empty()
    }
}

/// A plain file name (`a.jpg`): exactly one normal path component on every platform, so no
/// directories, no `.`/`..`, no drive or stream syntax (`C:x.jpg`, `a.jpg:stream`; BQ-03) and
/// no NUL.
pub fn safe_file_name(f: &str) -> bool {
    use std::path::Component;
    if f.is_empty() || f.contains(['/', '\\', ':', '\0']) {
        return false;
    }
    let mut comps = Path::new(f).components();
    matches!(
        (comps.next(), comps.next()),
        (Some(Component::Normal(n)), None) if n == f
    )
}

fn manifest_image(i: &ManifestImage, path: PathBuf) -> SetImage {
    let (w, h) = match (i.width, i.height) {
        (Some(w), Some(h)) => (w, h),
        _ => image::image_dimensions(&path).unwrap_or((0, 0)),
    };
    let mut tags = i.tags.clone();
    add_resolution_tag(&mut tags, w, h);
    SetImage {
        file: i.file.clone(),
        path: Some(path),
        width: w,
        height: h,
        tags,
        gt: i.objects.as_ref().map(|objs| {
            objs.iter()
                .map(|o| GtBox {
                    label: o.label.clone(),
                    bbox: o.bbox,
                    ignore: o.ignore || o.crowd,
                    crowd: o.crowd,
                })
                .collect()
        }),
    }
}

/// An image by extension; hidden files (macOS AppleDouble `._x.jpg` beside every file copied
/// to a non-APFS drive, `.DS_Store`) are not.
fn is_image(p: &Path) -> bool {
    !p.file_name()
        .is_some_and(|n| n.to_string_lossy().starts_with('.'))
        && p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| IMAGE_EXTENSIONS.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// Image files directly in `dir`, sorted by name.
fn list_images(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && is_image(p))
        .collect();
    v.sort();
    Ok(v)
}

/// Load a user directory: the images in it (or in its `images/` subdirectory), tags and/or
/// ground truth from `manifest.json`, else ground truth from `gt` (a COCO JSON file, or a
/// directory of YOLO txt labels) or found automatically (`_annotations.coco.json`,
/// `annotations.json`, `instances*.json`; `labels/` or `<stem>.txt` beside the images).
/// Returns the set and warnings (images left out because the ground truth does not cover
/// them or their size is unknown, duplicate COCO file names).
pub fn load_dir(
    dir: &Path,
    gt: Option<&Path>,
    name: Option<&str>,
) -> Result<(ImageSet, Vec<String>)> {
    if !dir.is_dir() {
        bail!("benchmark image directory {} does not exist", dir.display());
    }
    let mut img_dir = dir.to_path_buf();
    let mut files = list_images(dir)?;
    if files.is_empty() && dir.join("images").is_dir() {
        img_dir = dir.join("images");
        files = list_images(&img_dir)?;
    }
    if files.is_empty() {
        bail!(
            "no .jpg/.jpeg/.png images in {} (or its images/ subdirectory)",
            dir.display()
        );
    }
    let manifest = match dir.join("manifest.json") {
        p if p.is_file() => Some(
            Manifest::parse(&std::fs::read_to_string(&p)?)
                .with_context(|| format!("parsing {}", p.display()))?,
        ),
        _ => None,
    };
    let mut images = Vec::new();
    for path in files {
        let file = path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        let entry = manifest
            .as_ref()
            .and_then(|m| m.images.iter().find(|i| i.file == file));
        images.push(match entry {
            Some(e) => manifest_image(e, path),
            None => {
                let (w, h) = image::image_dimensions(&path).unwrap_or((0, 0));
                let mut tags = Vec::new();
                add_resolution_tag(&mut tags, w, h);
                SetImage {
                    file,
                    path: Some(path),
                    width: w,
                    height: h,
                    tags,
                    gt: None,
                }
            }
        });
    }
    let mut warnings = Vec::new();
    let mut ground_truth = if images.iter().all(|i| i.gt.is_some()) {
        GroundTruth::Manifest
    } else {
        GroundTruth::None
    };
    if ground_truth == GroundTruth::None {
        let coco = match gt {
            Some(p) if p.is_file() => Some(p.to_path_buf()),
            Some(p) if p.is_dir() => None,
            Some(p) => bail!("ground truth {} does not exist", p.display()),
            None => find_coco(dir),
        };
        if let Some(p) = coco {
            warnings.extend(apply_coco(&mut images, &p)?);
            ground_truth = GroundTruth::Coco;
        } else {
            let labels = match gt {
                Some(p) if p.is_dir() => Some(p.to_path_buf()),
                _ => find_yolo_labels(dir, &img_dir, &images),
            };
            if let Some(l) = labels {
                warnings.extend(apply_yolo(&mut images, &l, dir)?);
                ground_truth = GroundTruth::Yolo;
            }
        }
    }
    let id = match name {
        Some(n) if !n.trim().is_empty() => n.trim().to_string(),
        _ => format!("{DIR_PREFIX}{}", dir.display()),
    };
    let m = manifest.unwrap_or_default();
    let set = ImageSet {
        title: if m.title.is_empty() {
            id.clone()
        } else {
            m.title.clone()
        },
        id,
        kind: SetKind::Dir,
        dir: Some(img_dir),
        license: m.license,
        attribution: m.attribution,
        source: m.source,
        ground_truth,
        scored_labels: m.scored_labels,
        transform: m.transform,
        images,
    };
    Ok((set, warnings))
}

fn find_coco(dir: &Path) -> Option<PathBuf> {
    for n in ["_annotations.coco.json", "annotations.json"] {
        let p = dir.join(n);
        if p.is_file() {
            return Some(p);
        }
    }
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("instances") && n.ends_with(".json"))
        })
        .collect();
    v.sort();
    v.into_iter().next()
}

#[derive(Deserialize)]
struct CocoFile {
    images: Vec<CocoImage>,
    #[serde(default)]
    annotations: Vec<CocoAnn>,
    categories: Vec<CocoCat>,
}
#[derive(Deserialize)]
struct CocoImage {
    id: u64,
    file_name: String,
}
#[derive(Deserialize)]
struct CocoAnn {
    image_id: u64,
    category_id: u64,
    bbox: [f32; 4],
    #[serde(default)]
    iscrowd: u8,
}
#[derive(Deserialize)]
struct CocoCat {
    id: u64,
    name: String,
}

/// Parse a COCO instances JSON into `file name -> boxes` (crowd annotations become `ignore`).
pub fn parse_coco(text: &str) -> Result<Vec<(String, Vec<GtBox>)>> {
    let f: CocoFile = serde_json::from_str(text).context("parsing COCO JSON")?;
    let cats: std::collections::HashMap<u64, &str> = f
        .categories
        .iter()
        .map(|c| (c.id, c.name.as_str()))
        .collect();
    let mut out: Vec<(String, Vec<GtBox>)> = Vec::new();
    let mut index = std::collections::HashMap::new();
    for img in &f.images {
        let base = Path::new(&img.file_name)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| img.file_name.clone());
        index.insert(img.id, out.len());
        out.push((base, Vec::new()));
    }
    for a in &f.annotations {
        let (Some(&i), Some(name)) = (index.get(&a.image_id), cats.get(&a.category_id)) else {
            continue;
        };
        let [x, y, w, h] = a.bbox;
        // `index` values are positions in `out`.
        let Some(entry) = out.get_mut(i) else {
            continue;
        };
        entry.1.push(GtBox {
            label: name.to_string(),
            bbox: [x, y, x + w, y + h],
            ignore: a.iscrowd != 0,
            crowd: a.iscrowd != 0,
        });
    }
    Ok(out)
}

/// Up to three of `files`, comma separated (warnings).
fn examples(files: &[String]) -> String {
    files.iter().take(3).cloned().collect::<Vec<_>>().join(", ")
}

/// Apply a COCO JSON to the images of a folder, matched by file name (the JSON's
/// `file_name` without folders). Images the JSON does not list are not part of the annotated
/// set and are left out (with a warning): counting their detections as false positives would
/// be wrong. A JSON that lists none of the images is an error (another split, or file names
/// that differ from the files on disk; BQ-08). Returns warnings.
fn apply_coco(images: &mut Vec<SetImage>, path: &Path) -> Result<Vec<String>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let parsed = parse_coco(&text).with_context(|| format!("in {}", path.display()))?;
    let mut warnings = Vec::new();
    let mut by_file: std::collections::HashMap<String, Vec<GtBox>> =
        std::collections::HashMap::with_capacity(parsed.len());
    let mut duplicates: Vec<String> = Vec::new();
    for (file, boxes) in parsed {
        if by_file.contains_key(&file) {
            if !duplicates.contains(&file) {
                duplicates.push(file);
            }
            continue;
        }
        by_file.insert(file, boxes);
    }
    if !duplicates.is_empty() {
        warnings.push(format!(
            "{} lists {} file name(s) more than once in different folders (e.g. {}). Images \
             are matched by file name only, so the first entry was used; give the images \
             unique file names.",
            path.display(),
            duplicates.len(),
            examples(&duplicates)
        ));
    }
    let total = images.len();
    let mut unlisted = Vec::new();
    images.retain_mut(|img| match by_file.get(&img.file) {
        Some(b) => {
            img.gt = Some(b.clone());
            true
        }
        None => {
            unlisted.push(img.file.clone());
            false
        }
    });
    if images.is_empty() {
        bail!(
            "the COCO annotations {} list none of the {total} images in the folder. They are \
             probably for another split or name the files differently (case, extension). Point \
             `gt` at the annotation file of these images.",
            path.display()
        );
    }
    if !unlisted.is_empty() {
        warnings.push(format!(
            "{} of {total} images are not listed in the COCO annotations {} (e.g. {}) and were \
             left out of the set. They may be unannotated or the file names differ; add them \
             to the annotations to score them.",
            unlisted.len(),
            path.display(),
            examples(&unlisted)
        ));
    }
    Ok(warnings)
}

fn find_yolo_labels(dir: &Path, img_dir: &Path, images: &[SetImage]) -> Option<PathBuf> {
    let has_txt = |d: &Path| {
        images.iter().any(|i| {
            Path::new(&i.file)
                .file_stem()
                .is_some_and(|s| d.join(format!("{}.txt", s.to_string_lossy())).is_file())
        })
    };
    [dir.join("labels"), img_dir.to_path_buf()]
        .into_iter()
        .find(|d| d.is_dir() && has_txt(d))
}

/// YOLO class names: `classes.txt` (one per line) or `data.yaml` / `dataset.yaml` (`names` list
/// or index map) in any of `dirs`; else COCO-80.
pub fn yolo_class_names(dirs: &[&Path]) -> Result<Vec<String>> {
    for d in dirs {
        let txt = d.join("classes.txt");
        if txt.is_file() {
            return Ok(std::fs::read_to_string(&txt)?
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_string)
                .collect());
        }
        for y in ["data.yaml", "dataset.yaml"] {
            let p = d.join(y);
            if p.is_file() {
                return parse_yolo_yaml(&std::fs::read_to_string(&p)?)
                    .with_context(|| format!("parsing {}", p.display()));
            }
        }
    }
    Ok(crate::model::classes::coco80())
}

/// Largest class index accepted in a YOLO `names:` map.
pub const MAX_YOLO_CLASS_INDEX: u64 = 10_000;

/// `names:` of a YOLO dataset YAML, as a list or an `index: name` map. A map index above
/// [`MAX_YOLO_CLASS_INDEX`] or above 4x the entry count (at least 16) is an error.
pub fn parse_yolo_yaml(text: &str) -> Result<Vec<String>> {
    let v: serde_yaml_ng::Value = serde_yaml_ng::from_str(text)?;
    let names = v.get("names").context("no `names` in the dataset YAML")?;
    if let Some(seq) = names.as_sequence() {
        return Ok(seq
            .iter()
            .map(|n| n.as_str().unwrap_or_default().to_string())
            .collect());
    }
    if let Some(map) = names.as_mapping() {
        let mut pairs: Vec<(u64, String)> = map
            .iter()
            .filter_map(|(k, v)| Some((k.as_u64()?, v.as_str()?.to_string())))
            .collect();
        pairs.sort();
        // Validate the largest index before allocating: the list is sized by it.
        let max_index = pairs.last().map_or(0, |p| p.0);
        let bound = MAX_YOLO_CLASS_INDEX.min((pairs.len() as u64).saturating_mul(4).max(16));
        if max_index > bound {
            bail!(
                "class index {max_index} in `names` is too large ({} entries; at most {bound} allowed)",
                pairs.len()
            );
        }
        let len = usize::try_from(max_index)
            .ok()
            .and_then(|m| m.checked_add(1))
            .context("class index out of range")?;
        let mut out = vec![String::new(); len];
        for (i, n) in pairs {
            if let Some(slot) = usize::try_from(i).ok().and_then(|i| out.get_mut(i)) {
                *slot = n;
            }
        }
        return Ok(out);
    }
    bail!("`names` must be a list or a map")
}

/// How far outside [0, 1] a normalized YOLO value may be (float noise of exporters) before it
/// is an error; values within it are clamped.
const YOLO_TOLERANCE: f32 = 1e-3;

/// Parse one YOLO label file into boxes for a `w x h` image: `class cx cy w h` (normalized),
/// or a YOLO segmentation polygon `class x1 y1 x2 y2 x3 y3 ...` (normalized, at least three
/// points), taken as its bounding box. Every value must be a finite number in [0, 1]; errors
/// name the line (BQ-09).
pub fn parse_yolo(text: &str, names: &[String], w: u32, h: u32) -> Result<Vec<GtBox>> {
    let (w, h) = (w as f32, h as f32);
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let n = n + 1;
        let f: Vec<&str> = line.split_whitespace().collect();
        let Some((c, values)) = f.split_first() else {
            continue;
        };
        let cls: usize = c
            .parse()
            .with_context(|| format!("line {n}: bad class '{c}'"))?;
        let v = values
            .iter()
            .map(|s| {
                let x: f32 = s
                    .parse()
                    .with_context(|| format!("line {n}: bad number '{s}'"))?;
                if !x.is_finite() || !(-YOLO_TOLERANCE..=1.0 + YOLO_TOLERANCE).contains(&x) {
                    bail!(
                        "line {n}: '{s}' is not a normalized coordinate (YOLO labels hold \
                         values in 0..1, relative to the image size)"
                    );
                }
                Ok(x.clamp(0.0, 1.0))
            })
            .collect::<Result<Vec<f32>>>()?;
        let bbox = match v.as_slice() {
            [cx, cy, bw, bh] => {
                let (cx, cy, bw, bh) = (cx * w, cy * h, bw * w, bh * h);
                [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]
            }
            pts if pts.len() >= 6 && pts.len() % 2 == 0 => {
                let xs = pts.iter().step_by(2);
                let ys = pts.iter().skip(1).step_by(2);
                let (x0, x1) = xs.fold((1f32, 0f32), |(a, b), &x| (a.min(x), b.max(x)));
                let (y0, y1) = ys.fold((1f32, 0f32), |(a, b), &y| (a.min(y), b.max(y)));
                [x0 * w, y0 * h, x1 * w, y1 * h]
            }
            _ => bail!(
                "line {n}: expected `class cx cy w h` or a polygon `class x1 y1 x2 y2 x3 y3 \
                 ...`, found {} values",
                v.len()
            ),
        };
        let label = names
            .get(cls)
            .cloned()
            .unwrap_or_else(|| format!("class{cls}"));
        out.push(GtBox {
            label,
            bbox,
            ignore: false,
            crowd: false,
        });
    }
    Ok(out)
}

/// Apply YOLO txt labels to the images of a folder. An image whose size could not be read
/// (its normalized boxes can not be placed) is left out with a warning (BQ-09). Returns
/// warnings.
fn apply_yolo(images: &mut Vec<SetImage>, labels: &Path, dir: &Path) -> Result<Vec<String>> {
    let names = yolo_class_names(&[labels, dir])?;
    let mut unreadable = Vec::new();
    images.retain(|img| {
        let ok = img.width > 0 && img.height > 0;
        if !ok {
            unreadable.push(img.file.clone());
        }
        ok
    });
    if images.is_empty() {
        bail!(
            "none of the images in {} could be read (e.g. {}). The files are probably damaged \
             or not JPEG/PNG; replace them.",
            dir.display(),
            unreadable.first().map_or("", String::as_str)
        );
    }
    for img in images.iter_mut() {
        let stem = Path::new(&img.file)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let p = labels.join(format!("{stem}.txt"));
        // No label file: an image without objects.
        let text = std::fs::read_to_string(&p).unwrap_or_default();
        img.gt = Some(
            parse_yolo(&text, &names, img.width, img.height)
                .with_context(|| format!("in {}", p.display()))?,
        );
    }
    let mut warnings = Vec::new();
    if !unreadable.is_empty() {
        warnings.push(format!(
            "{} image(s) in {} could not be read (e.g. {}) and were left out of the set: their \
             YOLO boxes can not be placed without the image size. The files are probably \
             damaged; replace them.",
            unreadable.len(),
            dir.display(),
            examples(&unreadable)
        ));
    }
    Ok(warnings)
}

/// Datasets resolved for a run.
#[derive(Debug, Clone, Default)]
pub struct Resolved {
    pub sets: Vec<ImageSet>,
    /// Problems that did not stop the run (sets skipped, files missing).
    pub warnings: Vec<String>,
    /// Built-in sets with no image on disk (download `bench:<id>`).
    pub not_downloaded: Vec<String>,
}

/// Every dataset id accepted besides `dir:<path>`: `sample` and the built-in sets.
pub fn valid_ids() -> Vec<String> {
    let mut v = vec![SAMPLE_ID.to_string()];
    for id in KNOWN_SET_IDS {
        v.push(id.to_string());
    }
    for m in builtin_manifests() {
        if !v.contains(&m.id) {
            v.push(m.id.clone());
        }
    }
    v
}

fn dir_of(d: &DatasetRef, data_root: &Path) -> Option<(PathBuf, Option<PathBuf>, Option<String>)> {
    let abs = |p: PathBuf| {
        if p.is_absolute() {
            p
        } else {
            data_root.join(p)
        }
    };
    d.dir().map(|(dir, gt, name)| (abs(dir), gt.map(abs), name))
}

/// Check dataset references without loading them: ids must be known, directories must exist
/// (relative ones are under `data_root`).
pub fn validate(refs: &[DatasetRef], data_root: &Path) -> Result<()> {
    let valid = valid_ids();
    for d in refs {
        match dir_of(d, data_root) {
            Some((dir, gt, _)) => {
                if !dir.is_dir() {
                    bail!(
                        "dataset {}: directory {} does not exist",
                        d.label(),
                        dir.display()
                    );
                }
                if let Some(g) = gt
                    && !g.exists()
                {
                    bail!(
                        "dataset {}: ground truth {} does not exist",
                        d.label(),
                        g.display()
                    );
                }
            }
            None => {
                let id = d.label();
                if !valid.iter().any(|v| v.eq_ignore_ascii_case(&id)) {
                    bail!(
                        "unknown dataset '{id}'; valid ids: {}, or dir:<path> for a folder of images",
                        valid.join(", ")
                    );
                }
            }
        }
    }
    Ok(())
}

/// Load the referenced datasets, each limited to `max_images` (0 = all). Built-in sets use the
/// files present under `<data_root>/bench/<id>/`; a set with none is listed in
/// [`Resolved::not_downloaded`]. A folder that can not be loaded (no images, ground truth
/// that does not parse or matches none of its images) is skipped with a warning like a
/// missing built-in set, unless that leaves no dataset at all: then its error fails the run
/// (benchmarking the sample instead of the user's own folder would mislead; BQ-10). A set
/// whose id repeats an earlier one is skipped with a warning. When nothing is left otherwise,
/// the embedded sample is used (with a warning), so a run always has an image.
pub fn resolve(refs: &[DatasetRef], data_root: &Path, max_images: usize) -> Result<Resolved> {
    validate(refs, data_root)?;
    let mut out = Resolved::default();
    let mut dir_error: Option<anyhow::Error> = None;
    for d in refs {
        let mut set = match dir_of(d, data_root) {
            Some((dir, gt, name)) => {
                // The id stays as configured (`dir:<path>` or the name).
                let id = name.unwrap_or_else(|| d.label());
                match load_dir(&dir, gt.as_deref(), Some(&id)) {
                    Ok((set, warnings)) => {
                        out.warnings
                            .extend(warnings.into_iter().map(|w| format!("dataset {id}: {w}")));
                        set
                    }
                    Err(e) => {
                        let e = e.context(format!("dataset {}", d.label()));
                        out.warnings.push(format!(
                            "{}. The folder was skipped; fix it or remove it from the \
                             benchmark datasets.",
                            format!("{e:#}").trim_end_matches('.')
                        ));
                        dir_error.get_or_insert(e);
                        continue;
                    }
                }
            }
            None => {
                let id = d.label();
                if id.eq_ignore_ascii_case(SAMPLE_ID) {
                    ImageSet::sample()
                } else if let Some(m) = builtin_manifest(&id) {
                    let (set, missing) = ImageSet::from_manifest(m, &builtin_dir(data_root, &m.id));
                    if set.images.is_empty() {
                        // A set borrowing another's images is fetched as that set.
                        let dl = m.images_set_id().to_string();
                        out.warnings.push(format!(
                            "dataset {} is not downloaded (resource bench:{dl})",
                            m.id
                        ));
                        if !out.not_downloaded.contains(&dl) {
                            out.not_downloaded.push(dl);
                        }
                        continue;
                    }
                    if missing > 0 {
                        out.warnings.push(format!(
                            "dataset {}: {missing} of {} images are not downloaded; using the {} present",
                            m.id,
                            m.images.len(),
                            set.images.len()
                        ));
                    }
                    set
                } else {
                    out.warnings.push(format!(
                        "dataset {id} is not available in this build (its manifest is not embedded)"
                    ));
                    continue;
                }
            }
        };
        if out.sets.iter().any(|s| s.id == set.id) {
            out.warnings.push(format!(
                "dataset {} is listed more than once (or two folders share the name {}); only \
                 the first was used. Give each folder dataset a distinct name.",
                set.id, set.id
            ));
            continue;
        }
        set.limit(max_images);
        out.sets.push(set);
    }
    if out.sets.is_empty()
        && let Some(e) = dir_error
    {
        return Err(e);
    }
    if out.sets.is_empty() {
        out.warnings
            .push("no dataset images available; using the embedded sample image".to_string());
        out.sets.push(ImageSet::sample());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/bench")
    }

    #[test]
    fn resolution_buckets() {
        assert_eq!(resolution_bucket(640, 480), "sd");
        assert_eq!(resolution_bucket(640, 640), "sd");
        assert_eq!(resolution_bucket(704, 576), "sd");
        // A manifest `res:` tag wins over the size.
        let mut img = ImageSet::sample().images.remove(0);
        img.width = 1280;
        img.height = 720;
        img.tags = vec!["res:sd".into()];
        assert_eq!(img.resolution(), "sd");
        img.tags = vec!["res:weird".into()];
        assert_eq!(img.resolution(), "hd");
        assert_eq!(resolution_bucket(1280, 720), "hd");
        assert_eq!(resolution_bucket(1920, 1080), "fhd");
        assert_eq!(resolution_bucket(2560, 1440), "4mp+");
        assert_eq!(resolution_bucket(3840, 2160), "4mp+");
    }

    #[test]
    fn sample_set_is_always_available() {
        let s = ImageSet::sample();
        assert_eq!(s.images.len(), 1);
        assert!(s.images[0].width > 0);
        assert!(s.images[0].tags.iter().any(|t| t.starts_with("res:")));
        assert_eq!(
            s.images[0].bytes().unwrap().len(),
            super::super::DEFAULT_IMAGE.len()
        );
        assert!(!s.annotated());
    }

    #[test]
    fn manifest_fixture_loads_present_files() {
        let dir = fixture().join("manifest-set");
        let m =
            Manifest::parse(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(m.id, "test-set");
        assert!(m.has_ground_truth());
        let (set, missing) = ImageSet::from_manifest(&m, &dir);
        assert_eq!(missing, 1, "the fixture lists one file that is not there");
        assert_eq!(set.images.len(), 2);
        assert_eq!(set.ground_truth, GroundTruth::Manifest);
        let a = &set.images[0];
        assert_eq!((a.width, a.height), (64, 48));
        assert!(a.tags.contains(&"night".to_string()));
        assert!(a.tags.contains(&"res:sd".to_string()));
        let gt = a.gt.as_ref().unwrap();
        assert_eq!(gt.len(), 2);
        assert!(gt[1].ignore);
        // Bare-array manifests work too.
        let m = Manifest::parse(r#"[{"file":"x.jpg","tags":["day"]}]"#).unwrap();
        assert_eq!(m.images.len(), 1);
        assert!(!m.has_ground_truth());
    }

    #[test]
    fn limit_spreads_evenly() {
        let mut s = ImageSet::sample();
        let img = s.images[0].clone();
        s.images = (0..10)
            .map(|i| SetImage {
                file: format!("{i}.jpg"),
                ..img.clone()
            })
            .collect();
        s.limit(3);
        let files: Vec<&str> = s.images.iter().map(|i| i.file.as_str()).collect();
        assert_eq!(files, ["0.jpg", "3.jpg", "6.jpg"]);
        s.limit(0);
        assert_eq!(s.images.len(), 3);
    }

    #[test]
    fn coco_parser() {
        let text = r#"{"images":[{"id":7,"file_name":"sub/a.jpg","width":100,"height":80}],
            "annotations":[{"image_id":7,"category_id":3,"bbox":[10,20,30,40],"iscrowd":0},
                           {"image_id":7,"category_id":1,"bbox":[0,0,50,50],"iscrowd":1},
                           {"image_id":99,"category_id":1,"bbox":[0,0,1,1]}],
            "categories":[{"id":1,"name":"person"},{"id":3,"name":"car"}]}"#;
        let p = parse_coco(text).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].0, "a.jpg");
        assert_eq!(p[0].1.len(), 2);
        assert_eq!(p[0].1[0].label, "car");
        assert_eq!(p[0].1[0].bbox, [10.0, 20.0, 40.0, 60.0]);
        assert!(p[0].1[1].ignore);
        assert!(
            p[0].1[1].crowd,
            "iscrowd regions match by intersection over detection"
        );
        assert!(!p[0].1[0].crowd);
        assert!(parse_coco("{").is_err());
    }

    #[test]
    fn yolo_parser_and_names() {
        let names = vec!["person".to_string(), "car".to_string()];
        let b = parse_yolo("1 0.5 0.5 0.2 0.4\n\n0 0.1 0.1 0.1 0.1\n", &names, 200, 100).unwrap();
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].label, "car");
        assert_eq!(b[0].bbox, [80.0, 30.0, 120.0, 70.0]);
        assert!(parse_yolo("0 0.1 0.1", &names, 1, 1).is_err());
        assert!(parse_yolo("x 0.1 0.1 0.1 0.1", &names, 1, 1).is_err());
        assert_eq!(
            parse_yolo("5 .5 .5 .1 .1", &names, 10, 10).unwrap()[0].label,
            "class5"
        );
        assert_eq!(parse_yolo_yaml("names: [a, b]").unwrap(), ["a", "b"]);
        assert_eq!(
            parse_yolo_yaml("names:\n  1: car\n  0: person\n").unwrap(),
            ["person", "car"]
        );
        assert!(parse_yolo_yaml("nc: 2").is_err());
        // Index maps are bounded before anything is allocated.
        assert_eq!(
            parse_yolo_yaml("names:\n  0: a\n  15: b\n").unwrap().len(),
            16
        );
        for huge in ["100000000000", "18446744073709551615", "10001"] {
            let e = parse_yolo_yaml(&format!("names:\n  {huge}: x\n")).unwrap_err();
            assert!(format!("{e:#}").contains("too large"), "{huge}: {e:#}");
        }
    }

    /// YOLO values must be finite and normalized (the error names the line); polygons give
    /// their bounding box; other field counts are errors (BQ-09).
    #[test]
    fn yolo_labels_are_validated() {
        let names = vec!["person".to_string()];
        for (text, line) in [
            ("0 nan 0.5 0.1 0.1", 1),
            ("0 0.5 0.5 0.1 0.1\n0 inf 0.5 0.1 0.1", 2),
            ("0 0.5 -0.5 0.1 0.1", 1),
            ("\n\n0 0.5 0.5 1.5 0.1", 3),
            ("0 0.5 0.5 0.1 0.1 0.2", 1),
            ("0 0.1 0.1 0.2 0.2", 0),
        ] {
            let r = parse_yolo(text, &names, 100, 100);
            if line == 0 {
                assert!(r.is_ok(), "{text}");
                continue;
            }
            let e = format!("{:#}", r.unwrap_err());
            assert!(e.contains(&format!("line {line}:")), "{text}: {e}");
        }
        // Exporter float noise just above 1 is clamped.
        let b = parse_yolo("0 0.5 0.5 1.0005 1.0", &names, 100, 100).unwrap();
        assert_eq!(b[0].bbox, [0.0, 0.0, 100.0, 100.0]);
        // A segmentation polygon (triangle) becomes its bounding box.
        let b = parse_yolo("0 0.1 0.2 0.5 0.9 0.3 0.4", &names, 200, 100).unwrap();
        assert_eq!(b[0].bbox, [20.0, 20.0, 100.0, 90.0]);
        assert!(
            parse_yolo("0 0.1 0.2 0.5 0.9", &names, 200, 100).is_ok(),
            "a plain box"
        );
        assert!(parse_yolo("0 0.1 0.2 0.5 0.9 0.3", &names, 200, 100).is_err());
    }

    /// A temp folder with `files` copied from the fixture image (or given bytes).
    fn temp_set(files: &[(&str, Option<&[u8]>)]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bop-set-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let jpg = std::fs::read(fixture().join("plain-dir/a.jpg")).unwrap();
        for (name, bytes) in files {
            std::fs::write(dir.join(name), bytes.unwrap_or(&jpg)).unwrap();
        }
        dir
    }

    /// A COCO JSON is matched by file name: unlisted images are left out with a warning,
    /// duplicate base names warn, and a JSON that lists none of the images is an error
    /// (BQ-08).
    #[test]
    fn coco_json_must_match_the_images() {
        let dir = temp_set(&[("a.jpg", None), ("b.jpg", None)]);
        let coco = |files: &[&str]| {
            let images: Vec<String> = files
                .iter()
                .enumerate()
                .map(|(i, f)| format!(r#"{{"id":{i},"file_name":"{f}"}}"#))
                .collect();
            format!(
                r#"{{"images":[{}],"annotations":[{{"image_id":0,"category_id":1,"bbox":[1,1,5,5]}}],
                   "categories":[{{"id":1,"name":"person"}}]}}"#,
                images.join(",")
            )
        };
        let ann = dir.join("annotations.json");
        std::fs::write(&ann, coco(&["x/a.jpg", "y/a.jpg"])).unwrap();
        let (set, warnings) = load_dir(&dir, None, Some("s")).unwrap();
        assert_eq!(set.ground_truth, GroundTruth::Coco);
        let files: Vec<&str> = set.images.iter().map(|i| i.file.as_str()).collect();
        assert_eq!(files, ["a.jpg"]);
        assert_eq!(
            set.images[0].gt.as_ref().unwrap().len(),
            1,
            "first entry wins"
        );
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(warnings.iter().any(|w| w.contains("more than once")));
        assert!(
            warnings
                .iter()
                .any(|w| w.contains("1 of 2 images") && w.contains("b.jpg"))
        );

        // Another split: nothing matches.
        std::fs::write(&ann, coco(&["000000000139.jpg"])).unwrap();
        let e = format!("{:#}", load_dir(&dir, None, None).unwrap_err());
        assert!(e.contains("list none of the 2 images"), "{e}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// macOS AppleDouble files (`._a.jpg`, not a JPEG) are not images of the set.
    #[test]
    fn hidden_files_are_not_images() {
        let dir = temp_set(&[("a.jpg", None), ("._a.jpg", Some(b"\0\x05\x16\x07"))]);
        let files: Vec<String> = list_images(&dir)
            .unwrap()
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(files, ["a.jpg"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// With YOLO labels, an image whose size can not be read is left out with a warning.
    #[test]
    fn yolo_images_of_unknown_size_are_left_out() {
        let dir = temp_set(&[("a.jpg", None), ("bad.jpg", Some(b"not an image"))]);
        std::fs::create_dir_all(dir.join("labels")).unwrap();
        std::fs::write(dir.join("labels/a.txt"), "0 0.5 0.5 0.5 0.5\n").unwrap();
        std::fs::write(dir.join("labels/bad.txt"), "0 0.5 0.5 0.5 0.5\n").unwrap();
        let (set, warnings) = load_dir(&dir, None, None).unwrap();
        assert_eq!(set.ground_truth, GroundTruth::Yolo);
        assert_eq!(set.images.len(), 1);
        assert_eq!(set.images[0].file, "a.jpg");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("bad.jpg"), "{warnings:?}");
        // A bad value fails the folder with the file and line.
        std::fs::write(dir.join("labels/a.txt"), "0 0.5 0.5 0.5 0.5\n0 nan 0 0 0\n").unwrap();
        let e = format!("{:#}", load_dir(&dir, None, None).unwrap_err());
        assert!(e.contains("a.txt") && e.contains("line 2"), "{e}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A folder that does not load is skipped with a warning while other sets remain, and
    /// fails the run when it was the only one; a repeated id warns (BQ-10).
    #[test]
    fn bad_folders_and_duplicate_ids() {
        let empty = std::env::temp_dir().join(format!("bop-empty-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&empty).unwrap();
        let bad = DatasetRef::Id(format!("dir:{}", empty.display()));
        let root = fixture();
        let r = resolve(&[DatasetRef::Id("sample".into()), bad.clone()], &root, 0).unwrap();
        assert_eq!(r.sets.len(), 1);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("no .jpg") && w.contains("skipped")),
            "{:?}",
            r.warnings
        );
        let e = format!("{:#}", resolve(&[bad], &root, 0).unwrap_err());
        assert!(e.contains("no .jpg"), "{e}");

        let named = |dir: &str| DatasetRef::Dir {
            dir: dir.into(),
            gt: None,
            name: Some("mine".into()),
        };
        let r = resolve(&[named("coco-dir"), named("yolo-dir")], &root, 0).unwrap();
        assert_eq!(r.sets.len(), 1);
        assert_eq!(r.sets[0].ground_truth, GroundTruth::Coco);
        assert!(
            r.warnings.iter().any(|w| w.contains("more than once")),
            "{:?}",
            r.warnings
        );
        std::fs::remove_dir_all(&empty).ok();
    }

    #[test]
    fn oversized_image_files_are_rejected() {
        let dir = std::env::temp_dir().join(format!("bop-img-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("big.jpg");
        std::fs::write(&p, vec![0u8; 100]).unwrap();
        assert_eq!(read_limited(&p, 100).unwrap().len(), 100);
        let e = read_limited(&p, 99).unwrap_err();
        assert!(format!("{e:#}").contains("big.jpg"), "{e:#}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn user_dirs_with_coco_yolo_or_nothing() {
        let coco = load_dir(&fixture().join("coco-dir"), None, Some("mine"))
            .unwrap()
            .0;
        assert_eq!(coco.id, "mine");
        assert_eq!(coco.ground_truth, GroundTruth::Coco);
        assert_eq!(coco.images.len(), 2);
        // b.jpg has no annotations: annotated with zero objects.
        assert_eq!(coco.images[1].gt.as_ref().unwrap().len(), 0);
        assert_eq!(coco.images[0].gt.as_ref().unwrap()[0].label, "person");

        let yolo = load_dir(&fixture().join("yolo-dir"), None, None).unwrap().0;
        assert_eq!(yolo.ground_truth, GroundTruth::Yolo);
        assert!(yolo.id.starts_with(DIR_PREFIX));
        let gt = yolo.images[0].gt.as_ref().unwrap();
        assert_eq!(gt[0].label, "dog");
        // 64x48 image, box centered, half size.
        assert_eq!(gt[0].bbox, [16.0, 12.0, 48.0, 36.0]);

        let plain = load_dir(&fixture().join("plain-dir"), None, None)
            .unwrap()
            .0;
        assert_eq!(plain.ground_truth, GroundTruth::None);
        assert!(!plain.annotated());

        let err = load_dir(&fixture().join("nope"), None, None).unwrap_err();
        assert!(format!("{err:#}").contains("does not exist"));
        let err = load_dir(
            &fixture().join("plain-dir"),
            Some(&fixture().join("x.json")),
            None,
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("does not exist"));
    }

    #[test]
    fn file_names_are_checked() {
        assert!(safe_file_name("a.jpg"));
        for bad in [
            "",
            ".",
            "..",
            "../a.jpg",
            "x/a.jpg",
            "x\\a.jpg",
            "C:x.jpg",
            "a.jpg:stream",
            "a\0.jpg",
            "/a.jpg",
        ] {
            assert!(!safe_file_name(bad), "{bad}");
        }
    }

    #[test]
    fn dataset_resolution_and_validation() {
        let root = fixture();
        let id = |s: &str| DatasetRef::Id(s.to_string());
        // Relative directories are under the data root; the sample always works.
        let r = resolve(
            &[
                id("sample"),
                id("dir:coco-dir"),
                DatasetRef::Dir {
                    dir: "yolo-dir".into(),
                    gt: None,
                    name: Some("yolo".into()),
                },
                id("sample"),
            ],
            &root,
            1,
        )
        .unwrap();
        let ids: Vec<&str> = r.sets.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["sample", "dir:coco-dir", "yolo"]);
        assert_eq!(r.sets[1].images.len(), 1, "limited to 1 image");
        // Known built-in ids without an embedded manifest are skipped with a warning, and the
        // sample stands in when nothing is left.
        let r = resolve(&[id("coco-cctv")], &root, 0).unwrap();
        assert_eq!(r.sets[0].id, "sample");
        assert!(r.warnings.len() >= 2, "{:?}", r.warnings);
        // Unknown ids and missing directories are errors naming the valid ids.
        let e = validate(&[id("nope")], &root).unwrap_err();
        let msg = format!("{e:#}");
        assert!(
            msg.contains("unknown dataset 'nope'") && msg.contains("coco-cctv"),
            "{msg}"
        );
        let e = validate(&[id("dir:missing")], &root).unwrap_err();
        assert!(format!("{e:#}").contains("does not exist"));
        let e = validate(
            &[DatasetRef::Dir {
                dir: "plain-dir".into(),
                gt: Some("nope.json".into()),
                name: None,
            }],
            &root,
        )
        .unwrap_err();
        assert!(format!("{e:#}").contains("ground truth"));
    }

    #[test]
    fn builtin_sets_are_catalog_resources() {
        let ids: Vec<&str> = builtin_manifests().iter().map(|m| m.id.as_str()).collect();
        for id in KNOWN_SET_IDS {
            assert!(ids.contains(&id), "{id} is embedded");
        }
        let coco: Vec<String> = crate::model::classes::coco80()
            .iter()
            .map(|l| super::super::metrics::canonical_label(l))
            .collect();
        for m in builtin_manifests() {
            let r = crate::resources::catalog::bench_set(&m.id).unwrap();
            assert_eq!(r.parts.len(), m.images.len(), "{}", m.id);
            assert_eq!(r.size(), m.size());
            // A set borrowing another's images is installed by that set's resource.
            assert_eq!(r.dest, format!("{BENCH_DIR}/{}", m.images_set_id()));
            assert_eq!(
                builtin_dir(Path::new("/d"), &m.id),
                Path::new("/d").join(r.dest)
            );
            assert!(
                r.parts
                    .iter()
                    .all(|p| p.url.starts_with("https://") && p.sha256.len() == 64)
            );
            assert!(m.has_ground_truth());
            assert_eq!(crate::resources::status::group_of(r), "bench-images");
            assert_eq!(
                crate::resources::catalog::find(&format!("bench:{}", m.id), "linux", "x86_64")
                    .map(|r| r.id),
                Some(r.id)
            );
            // Every label is a COCO-80 label (any spelling), so models map onto it, or
            // camera-trap wildlife (an animal ignore region for models without the class).
            for l in m.scored_labels.iter().flatten() {
                let c = super::super::metrics::canonical_label(l);
                assert!(
                    coco.contains(&c) || super::super::metrics::is_wildlife(l),
                    "{}: scored {l}",
                    m.id
                );
            }
            for i in &m.images {
                assert!(safe_file_name(&i.file));
                for o in i.objects.iter().flatten() {
                    let c = super::super::metrics::canonical_label(&o.label);
                    assert!(
                        coco.contains(&c) || super::super::metrics::is_wildlife(&o.label),
                        "{}: {}",
                        m.id,
                        o.label
                    );
                    assert!(!o.crowd || o.ignore, "a crowd region is an ignore region");
                }
            }
        }
        // coco-cctv's ignore regions are COCO `iscrowd` boxes: matched like pycocotools
        // (intersection over the detection, BQ-06).
        let coco_cctv = builtin_manifest("coco-cctv").unwrap();
        let ignored: Vec<&ManifestObject> = coco_cctv
            .images
            .iter()
            .flat_map(|i| i.objects.iter().flatten())
            .filter(|o| o.ignore)
            .collect();
        assert!(!ignored.is_empty());
        assert!(ignored.iter().all(|o| o.crowd));
    }

    /// exdark-ir is exdark-night's images and ground truth with the grayscale transform, in
    /// exdark-night's folder and fetched as `bench:exdark-night`; nir-night is real IR.
    #[test]
    fn exdark_ir_borrows_exdark_night() {
        let night = builtin_manifest("exdark-night").unwrap();
        let ir = builtin_manifest("exdark-ir").unwrap();
        assert_eq!(ir.transform, Some(ImageTransform::Grayscale));
        assert_eq!(ir.images_set_id(), "exdark-night");
        assert_eq!(ir.images, night.images);
        assert_eq!(ir.scored_labels, night.scored_labels);
        assert!(night.transform.is_none());
        let root = Path::new("/data");
        assert_eq!(
            builtin_dir(root, "exdark-ir"),
            builtin_dir(root, "exdark-night")
        );
        let r = crate::resources::catalog::bench_set("exdark-ir").unwrap();
        assert_eq!(r.id, "bench:exdark-night");
        assert!(
            crate::resources::catalog::bench_sets()
                .iter()
                .all(|r| r.id != "bench:exdark-ir"),
            "no resource of its own"
        );
        assert_eq!(
            crate::resources::catalog::find("bench:exdark-ir", "linux", "x86_64").map(|r| r.id),
            Some("bench:exdark-night")
        );
        // Not downloaded: reported as the set that installs it.
        let empty = std::env::temp_dir().join(format!("bop-ir-{}", uuid::Uuid::new_v4()));
        let r = resolve(&[DatasetRef::Id("exdark-ir".into())], &empty, 0).unwrap();
        assert_eq!(r.not_downloaded, ["exdark-night"]);
        let (set, missing) = ImageSet::from_manifest(ir, &builtin_dir(&empty, &ir.id));
        assert_eq!(set.transform, Some(ImageTransform::Grayscale));
        assert_eq!(missing, ir.images.len());

        let nir = builtin_manifest("nir-night").unwrap();
        assert!(nir.images_from.is_none() && nir.transform.is_none());
        assert!(nir.images.iter().all(|i| i.tags.iter().any(|t| t == "ir")));
        assert!(
            nir.images
                .iter()
                .any(|i| i.tags.iter().any(|t| t == "hard-negative"))
        );
        // coco-cctv tags its monochrome images.
        let coco = builtin_manifest("coco-cctv").unwrap();
        assert!(
            coco.images
                .iter()
                .any(|i| i.tags.iter().any(|t| t == "grayscale"))
        );
    }

    #[test]
    fn images_from_and_transform_are_resolved() {
        let base = r#"{"id":"a","images":[{"file":"x.jpg","objects":[]}]}"#;
        let borrow = r#"{"id":"b","images_from":"A","transform":"grayscale","images":[]}"#;
        let own = r#"{"id":"c","images_from":"a","images":[{"file":"y.jpg"}]}"#;
        let missing = r#"{"id":"d","images_from":"nope"}"#;
        let chained = r#"{"id":"e","images_from":"b"}"#;
        let selfref = r#"{"id":"f","images_from":"f","images":[{"file":"z.jpg"}]}"#;
        let ms = parse_builtins(&[
            ("a", base),
            ("b", borrow),
            ("c", own),
            ("d", missing),
            ("e", chained),
            ("f", selfref),
        ]);
        let ids: Vec<&str> = ms.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c"]);
        assert_eq!(ms[1].images_from.as_deref(), Some("a"), "canonical id");
        assert_eq!(ms[1].images, ms[0].images, "inherited");
        assert_eq!(ms[1].transform, Some(ImageTransform::Grayscale));
        assert_eq!(ms[2].images[0].file, "y.jpg", "own list kept");
        assert!(Manifest::parse(r#"{"transform":"sepia","images":[]}"#).is_err());

        let mut img = crate::image::RgbImage {
            width: 1,
            height: 1,
            rgb: vec![255, 0, 0],
        };
        ImageTransform::Grayscale.apply(&mut img);
        assert_eq!(img.rgb, [76, 76, 76]);
    }
}
