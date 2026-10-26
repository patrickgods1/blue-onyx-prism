//! Footage import (`blue-onyx-prism train import`, the Training page's Import card,
//! docs/TRAINING.md "Footage import"): frames from local videos, image folders (Blue Iris clip and
//! alert folders), a video URL (yt-dlp) or pinned open footage (MEVA, CC BY 4.0) into the dataset
//! folder, so they show on the Dataset and Review pages like collected frames.
//!
//! Frame selection, in order:
//! 1. FFmpeg's `select` scene-change filter at a bounded rate, plus one frame every few seconds
//!    ([`select::select_expr`], run by `scripts/import_footage.py` with PyAV);
//! 2. near duplicates dropped by dHash (Hamming distance <= 6 by default, per camera; the
//!    collector's [`crate::collect::dhash`]);
//! 3. SigLIP 2 embeddings and k-center greedy to the requested number of frames, balanced by scene
//!    (`day | night | ir` from [`crate::image_stats`] on the decoded frame, computed here); or
//!    k-center on the dHashes (`--select dhash`, no embeddings and, for image folders, no Python).
//!
//! Every frame gets a version-1 sidecar without detections, `source: "import:<id>"` and its
//! provenance (file or URL, the licence the user confirmed, the time offset): [`write`]. The job
//! runs on the training queue ([`super::service`], one GPU job at a time). Imports never touch
//! sites that list unsecured cameras ([`source::DENYLIST`]) and never read live streams.
//!
//! - [`source`]: what an import reads, the URL denylist, the licence confirmation.
//! - [`select`]: the pure selection steps.
//! - [`write`]: the dataset layout of the imported frames.
//! - [`preview`]: what an import of a path would read, before it is queued.
//! - [`parallel`]: the worker pools of the extraction (one helper process per video) and the
//!   analysis (decode, dHash, scene).
//! - [`job`]: the stage machine ([`ImportJob`], a [`super::task::Task`]).
//! - [`disk`]: the space an import needs, checked before it is queued ([`plan_import`]) and
//!   before each writing stage, and guarded while it runs (a full disk ends it with advice).
//!
//! Before anything is extracted, the candidate budget ([`MAX_CANDIDATES_TOTAL`]) is shared
//! fairly by the cameras, their videos and the images ([`select::budget`]); each video's share
//! is its `--max-frames`. Videos are extracted [`ImportRequest::workers`] at a time (default
//! half the physical cores, 1..=4), decoded on the GPU when PyAV's FFmpeg offers a hardware
//! decoder ([`HwAccel`], software otherwise or when it fails).
//!
//! Two passes: the candidates are stored downscaled ([`disk::CANDIDATE_MAX_SIDE`] px on the long
//! side, which is all the analysis needs) with their source time and frame index; only the frames
//! kept are then extracted again at full resolution (`import_footage.py frames`) and written to
//! the dataset, the same files as a single full-resolution pass would write. Image folders are
//! read in place in both.

pub mod disk;
pub mod job;
pub mod parallel;
pub mod preview;
pub mod select;
pub mod source;
pub mod write;

pub use job::ImportJob;
pub use source::ImportSource;

use super::capability::Capability;
use crate::config::Config;
use crate::resources::catalog::{self, Platform};
use crate::resources::export::train::{self as env, TrainPaths};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Default number of frames an import keeps.
pub const DEFAULT_FRAMES: usize = 200;
/// Most frames one import keeps.
pub const MAX_FRAMES: usize = 5000;
/// Default FFmpeg scene-change threshold (0..1). FFmpeg's score is the mean frame difference: a
/// person crossing a fixed 1080p camera's view scores about 0.002-0.015 (measured on MEVA clips),
/// a cut in edited footage 0.3 and more. 0.01 keeps the busy moments of surveillance footage; the
/// rate limit, the dedup and the k-center selection thin them out.
pub const DEFAULT_SCENE: f32 = 0.01;
/// Default most frames per second kept from scene changes.
pub const DEFAULT_MAX_FPS: f32 = 1.0;
/// Default dedup window (seconds): frames further apart are never duplicates.
pub const DEFAULT_DEDUP_WINDOW_S: f32 = 10.0;
/// Default interval of the periodic frame (seconds).
pub const DEFAULT_EVERY_S: f32 = 10.0;
/// Most frames extracted from one video before the selection.
pub const DEFAULT_MAX_CANDIDATES: usize = 1500;
/// Most candidate frames of one import (all videos and images).
pub const MAX_CANDIDATES_TOTAL: usize = 20_000;
/// Largest video downloaded from a URL.
pub const MAX_URL_BYTES: u64 = 2_000_000_000;
/// Most videos extracted at the same time (`--workers`).
pub const MAX_WORKERS: usize = 16;

/// Video decoding of the extraction (`import_footage.py extract --hwaccel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HwAccel {
    /// The platform's hardware decoder when PyAV's FFmpeg has it (D3D11VA on Windows, CUDA
    /// then VAAPI on Linux); software when there is none or it fails. macOS decodes in software:
    /// VideoToolbox measured slower there (docs/TRAINING.md).
    #[default]
    Auto,
    /// Software decoding only.
    Off,
}

impl HwAccel {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" | "on" | "hw" => Some(HwAccel::Auto),
            "off" | "none" | "sw" | "software" => Some(HwAccel::Off),
            _ => None,
        }
    }
    pub const fn as_str(self) -> &'static str {
        match self {
            HwAccel::Auto => "auto",
            HwAccel::Off => "off",
        }
    }
}

/// How the frames are picked after the dedup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Selection {
    /// SigLIP 2 image embeddings, k-center greedy on the cosine distance.
    #[default]
    Siglip,
    /// k-center greedy on the dHash Hamming distance (no embeddings).
    Dhash,
}

impl Selection {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "siglip" | "siglip2" | "embeddings" => Some(Selection::Siglip),
            "dhash" | "hash" => Some(Selection::Dhash),
            _ => None,
        }
    }
    pub const fn as_str(self) -> &'static str {
        match self {
            Selection::Siglip => "siglip",
            Selection::Dhash => "dhash",
        }
    }
}

/// What to import and how.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportRequest {
    pub source: ImportSource,
    /// Camera folder of the frames (default: from the file names, else `import`).
    #[serde(default)]
    pub camera: Option<String>,
    pub frames: usize,
    pub scene_threshold: f32,
    pub max_fps: f32,
    pub every_s: f32,
    pub dedup_hamming: u32,
    /// Seconds within which two frames of a camera can be duplicates (0 = any time).
    #[serde(default = "default_window")]
    pub dedup_window_s: f32,
    pub selection: Selection,
    /// Share the frames out equally by scene (day / night / IR).
    pub balance: bool,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub license_confirmed: bool,
    pub max_candidates: usize,
    /// Allow downloads over 500 MB (the environment, SigLIP 2).
    #[serde(default)]
    pub allow_large: bool,
    /// Keep the extracted candidates in the work folder.
    #[serde(default)]
    pub keep_work: bool,
    /// Videos extracted at the same time; 0 = automatic ([`parallel::extract_workers`]).
    #[serde(default)]
    pub workers: usize,
    /// Hardware video decoding.
    #[serde(default)]
    pub hwaccel: HwAccel,
    /// Dataset folder the frames are written to (relative = under the data root; default
    /// `collect.dir`, the folder the Dataset and Review pages show).
    #[serde(default)]
    pub out_dir: Option<PathBuf>,
}

fn default_window() -> f32 {
    DEFAULT_DEDUP_WINDOW_S
}

impl ImportRequest {
    /// A request with the defaults.
    pub fn new(source: ImportSource) -> Self {
        Self {
            source,
            camera: None,
            frames: DEFAULT_FRAMES,
            scene_threshold: DEFAULT_SCENE,
            max_fps: DEFAULT_MAX_FPS,
            every_s: DEFAULT_EVERY_S,
            dedup_hamming: crate::collect::config::DEFAULT_DEDUP_HAMMING,
            dedup_window_s: DEFAULT_DEDUP_WINDOW_S,
            selection: Selection::default(),
            balance: true,
            license: None,
            license_confirmed: false,
            max_candidates: DEFAULT_MAX_CANDIDATES,
            allow_large: false,
            keep_work: false,
            workers: 0,
            hwaccel: HwAccel::Auto,
            out_dir: None,
        }
    }

    /// Range checks; errors name the option.
    pub fn validate(&self) -> Result<()> {
        if !(1..=MAX_FRAMES).contains(&self.frames) {
            bail!(
                "frames must be between 1 and {MAX_FRAMES}, got {}",
                self.frames
            );
        }
        if !(self.scene_threshold.is_finite() && (0.0..=1.0).contains(&self.scene_threshold)) {
            bail!(
                "the scene threshold must be between 0 and 1, got {}",
                self.scene_threshold
            );
        }
        if !(self.max_fps.is_finite() && (0.0..=60.0).contains(&self.max_fps)) {
            bail!("max fps must be between 0 and 60, got {}", self.max_fps);
        }
        if !(self.every_s.is_finite() && (0.0..=3600.0).contains(&self.every_s)) {
            bail!(
                "the periodic frame interval must be between 0 and 3600 s, got {}",
                self.every_s
            );
        }
        if self.dedup_hamming > crate::collect::config::MAX_DEDUP_HAMMING {
            bail!(
                "the dedup distance must be at most {} bits",
                crate::collect::config::MAX_DEDUP_HAMMING
            );
        }
        if !(self.dedup_window_s.is_finite() && (0.0..=86_400.0).contains(&self.dedup_window_s)) {
            bail!(
                "the dedup window must be between 0 and 86400 s, got {}",
                self.dedup_window_s
            );
        }
        if !(1..=MAX_CANDIDATES_TOTAL).contains(&self.max_candidates) {
            bail!("max candidates must be between 1 and {MAX_CANDIDATES_TOTAL}");
        }
        if self.workers > MAX_WORKERS {
            bail!(
                "workers must be between 0 (automatic) and {MAX_WORKERS}, got {}",
                self.workers
            );
        }
        Ok(())
    }

    /// The FFmpeg `select` expression of the extraction.
    pub fn select_expr(&self) -> String {
        select::select_expr(self.scene_threshold, self.max_fps, self.every_s)
    }
}

/// Plan an import of `request` on `config` (checks that fail fast: the source, the URL
/// denylist, the licence confirmation, whether the training environment is needed and allowed
/// to download). The job then runs on the training queue.
pub fn plan_import(
    config: &Config,
    mut request: ImportRequest,
    cap: &Capability,
) -> Result<ImportJob> {
    request.validate()?;
    let license = source::check_license(
        &request.source,
        request.license.as_deref(),
        request.license_confirmed,
    )?;
    let mut needs_python = request.selection == Selection::Siglip;
    let mut plan_sources = PlanSources::default();
    match &mut request.source {
        ImportSource::Url(u) => {
            *u = source::check_url(u)?;
            needs_python = true;
        }
        ImportSource::Path(p) => {
            *p = config.data_path(p);
            let d = source::discover(p)?;
            needs_python |= !d.videos.is_empty();
            plan_sources.cameras = d
                .videos
                .iter()
                .map(|v| source::camera_of(v, request.camera.as_deref()))
                .collect();
            plan_sources.images = d.images;
        }
        ImportSource::Dataset(d) => {
            if catalog::import_dataset(d).is_none() {
                bail!(
                    "unknown dataset '{d}'; pinned footage: {}",
                    catalog::IMPORT_DATASETS
                        .iter()
                        .map(|r| catalog::install_key(r))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            needs_python = true;
        }
    }
    let accelerator = cap.accelerator.clone();
    if needs_python && accelerator.is_none() {
        bail!(
            "importing videos, URLs or with SigLIP selection needs the training environment, which \
             needs an NVIDIA GPU (Windows / Linux) or Apple silicon: {}; image folders import \
             without it with --select dhash",
            cap.summary()
        );
    }
    let paths = TrainPaths::for_config(config);
    let platform = Platform::current();
    let allow_large = request.allow_large || config.allow_large_downloads;
    if let Some(a) = &accelerator
        && needs_python
        && !env::train_env_ready(&paths, platform, a.lane())
        && !allow_large
    {
        let (dl, disk) = a.lane().estimate();
        bail!(
            "the import needs the training environment (PyTorch {}, PyAV, yt-dlp; about {} \
             download, {} on disk under {}); rerun with --allow-large (Training page: tick the \
             download confirmation)",
            a.lane().as_str(),
            catalog::format_size(dl),
            catalog::format_size(disk),
            paths.export.tools.display()
        );
    }
    request.allow_large = allow_large;
    let id = super::service::new_run_id();
    match &request.source {
        ImportSource::Path(_) => {}
        ImportSource::Url(_) => plan_sources.cameras.push("url".into()),
        ImportSource::Dataset(d) => {
            if let Some(res) = catalog::import_dataset(d) {
                plan_sources.cameras = res
                    .parts
                    .iter()
                    .map(|p| {
                        source::camera_of(
                            std::path::Path::new(p.file_name),
                            request.camera.as_deref(),
                        )
                    })
                    .collect();
            }
        }
    }
    let dataset_dir = match &request.out_dir {
        Some(d) => {
            let d = config.data_path(d);
            if d.exists() && !d.is_dir() {
                bail!(
                    "the output folder {} is a file; choose a folder for the frames",
                    d.display()
                );
            }
            d
        }
        None => crate::collect::CollectSettings::for_config(config).dir,
    };
    let work = super::training_dir(config).join("imports").join(&id);
    disk::check(
        disk::estimate(&plan_sources.plan(&request)),
        &work,
        &dataset_dir,
    )?;
    Ok(ImportJob {
        work,
        id,
        request,
        license,
        confirmed_at: crate::collect::utc_rfc3339(std::time::SystemTime::now()),
        dataset_dir,
        paths,
        platform,
        accelerator,
        root: config.data_root(),
        timeout: Duration::from_secs(u64::from(config.training.timeout_hours) * 3600),
        needs_python,
        finish: None,
    })
}

/// What [`plan_import`] knows of the source for the space estimate.
#[derive(Debug, Default)]
struct PlanSources {
    /// The camera of each video.
    cameras: Vec<String>,
    images: Vec<PathBuf>,
}

impl PlanSources {
    /// The [`disk::Plan`] of `r` (the same candidate budget as the job; the videos' size is not
    /// known yet: [`disk::ASSUMED_SOURCE`]; the images' mean size from up to 200 of them).
    fn plan(&self, r: &ImportRequest) -> disk::Plan {
        let cams: Vec<&str> = self.cameras.iter().map(String::as_str).collect();
        let b = select::budget(
            &cams,
            self.images.len(),
            MAX_CANDIDATES_TOTAL,
            r.max_candidates,
        );
        let used = select::stride(self.images.len(), b.images);
        let sample: Vec<u64> = select::stride(used.len(), used.len().min(200))
            .into_iter()
            .filter_map(|i| {
                let p = self.images.get(*used.get(i)?)?;
                std::fs::metadata(p).ok().map(|m| m.len())
            })
            .collect();
        disk::Plan {
            video_candidates: b.per_video.iter().sum(),
            image_candidates: b.images,
            frames: r.frames,
            source: None,
            image_bytes: if sample.is_empty() {
                0
            } else {
                sample.iter().sum::<u64>() / sample.len() as u64
            },
            download: 0,
        }
    }
}

/// `<data root>/<training dir>/imports`: the work folders of the imports.
pub fn imports_dir(config: &Config) -> PathBuf {
    super::training_dir(config).join("imports")
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::backend::detect::{HardwareInfo, apple_gpu};

    fn mac() -> Capability {
        super::super::capability::capability(
            &HardwareInfo::new("macos", "aarch64", vec![apple_gpu("Apple M1")]),
            Some(16.0),
        )
    }

    #[test]
    fn requests_validate_and_build_the_select_expression() {
        let r = ImportRequest::new(ImportSource::Dataset("meva-sample".into()));
        r.validate().unwrap();
        assert!(r.select_expr().contains("gt(scene,0.01)"));
        for f in [
            |r: &mut ImportRequest| r.frames = 0,
            |r: &mut ImportRequest| r.scene_threshold = 1.5,
            |r: &mut ImportRequest| r.max_fps = f32::NAN,
            |r: &mut ImportRequest| r.every_s = -1.0,
            |r: &mut ImportRequest| r.dedup_hamming = 40,
            |r: &mut ImportRequest| r.max_candidates = 0,
            |r: &mut ImportRequest| r.workers = 17,
        ] {
            let mut x = r.clone();
            f(&mut x);
            assert!(x.validate().is_err(), "{x:?}");
        }
        assert_eq!(Selection::parse("SigLIP2"), Some(Selection::Siglip));
        assert_eq!(Selection::parse("dhash"), Some(Selection::Dhash));
        assert_eq!(Selection::parse("x"), None);
        assert_eq!(HwAccel::parse(" OFF "), Some(HwAccel::Off));
        assert_eq!(HwAccel::parse("auto"), Some(HwAccel::Auto));
        assert_eq!(HwAccel::parse("cuda"), None);
        // Requests saved before the fields existed still load.
        let mut v = serde_json::to_value(&r).unwrap();
        let o = v.as_object_mut().unwrap();
        o.remove("workers");
        o.remove("hwaccel");
        o.remove("out_dir");
        let back: ImportRequest = serde_json::from_value(v).unwrap();
        assert_eq!((back.workers, back.hwaccel), (0, HwAccel::Auto));
        assert_eq!(back.out_dir, None);
    }

    #[test]
    fn planning_checks_the_source_licence_and_downloads() {
        let root = std::env::temp_dir().join(format!("bop-import-plan-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("clips")).unwrap();
        std::fs::write(root.join("clips/Drive.20240101_010101.mp4"), b"x").unwrap();
        std::fs::write(root.join("clips/a.jpg"), b"x").unwrap();
        let c = Config {
            download_dir: Some(root.clone()),
            ..Config::default()
        };
        let cap = mac();
        let mut r = ImportRequest::new(ImportSource::Path("clips".into()));
        // No confirmation.
        let e = plan_import(&c, r.clone(), &cap).unwrap_err().to_string();
        assert!(e.contains("confirm"), "{e}");
        r.license_confirmed = true;
        // The environment is not set up and large downloads are not allowed.
        let e = plan_import(&c, r.clone(), &cap).unwrap_err().to_string();
        assert!(e.contains("--allow-large"), "{e}");
        r.allow_large = true;
        let j = plan_import(&c, r.clone(), &cap).unwrap();
        assert!(j.needs_python);
        assert_eq!(j.license, "own footage");
        assert!(j.work.starts_with(root.join("training/imports")));
        assert_eq!(j.dataset_dir, root.join("dataset"));
        // Another output folder (relative = under the data root); a file is refused.
        let mut o = r.clone();
        o.out_dir = Some("picked".into());
        assert_eq!(
            plan_import(&c, o.clone(), &cap).unwrap().dataset_dir,
            root.join("picked")
        );
        o.out_dir = Some("clips/a.jpg".into());
        let e = plan_import(&c, o, &cap).unwrap_err().to_string();
        assert!(e.contains("is a file"), "{e}");
        // A denylisted URL fails before anything is queued.
        let mut u = ImportRequest::new(ImportSource::Url(
            "http://www.insecam.org/en/view/1/".into(),
        ));
        u.license = Some("CC0".into());
        u.license_confirmed = true;
        u.allow_large = true;
        assert!(
            plan_import(&c, u, &cap)
                .unwrap_err()
                .to_string()
                .contains("refused")
        );
        // Without a GPU: videos are refused, an image folder with dHash selection is fine.
        let none = super::super::capability::capability(
            &HardwareInfo::new("linux", "x86_64", vec![]),
            None,
        );
        let e = plan_import(&c, r.clone(), &none).unwrap_err().to_string();
        assert!(e.contains("--select dhash"), "{e}");
        std::fs::remove_file(root.join("clips/Drive.20240101_010101.mp4")).unwrap();
        r.selection = Selection::Dhash;
        let j = plan_import(&c, r, &none).unwrap();
        assert!(!j.needs_python);
        let e = plan_import(
            &c,
            ImportRequest::new(ImportSource::Dataset("nope".into())),
            &cap,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("meva-sample"), "{e}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_plan_counts_the_budget_and_the_images_mean_size() {
        let root = std::env::temp_dir().join(format!("bop-import-est-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let images: Vec<PathBuf> = (0..4)
            .map(|i| {
                let p = root.join(format!("i{i}.jpg"));
                std::fs::write(&p, vec![0u8; 1000 * (i + 1)]).unwrap();
                p
            })
            .collect();
        let s = PlanSources {
            cameras: vec!["a".into(), "a".into(), "b".into()],
            images,
        };
        let mut r = ImportRequest::new(ImportSource::Path(root.clone()));
        r.max_candidates = 100;
        let p = s.plan(&r);
        assert_eq!(p.video_candidates, 300);
        assert_eq!(p.image_candidates, 4);
        assert_eq!(p.image_bytes, 2500);
        let n = disk::estimate(&p);
        assert_eq!(
            n.work,
            300 * disk::candidate_frame_bytes(1920, 1080)
                + 200 * disk::full_frame_bytes(1920, 1080)
        );
        assert_eq!(n.dataset, 200 * disk::full_frame_bytes(1920, 1080));
        let _ = std::fs::remove_dir_all(&root);
    }
}
