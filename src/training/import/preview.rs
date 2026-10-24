//! What an import of a path would read, before it is queued (`GET /v1/training/import/preview`,
//! the line under the Training page's path field): the videos and images [`super::source::discover`]
//! finds, the cameras of the videos, their size, the skipped files, and the candidate budget
//! ([`super::select::budget`]) each video would get.
//!
//! The page previews while a path is typed, and a path may be a whole drive, so the walk of
//! [`preview_latest`] has a budget ([`PREVIEW_MAX_DIRS`] folders, [`PREVIEW_TIME`]) and is
//! single-flight: one walk runs at a time, process-wide. A newer request supersedes the
//! running one: the running walk stops at its next check (a folder, or [`source::CHECK_EVERY`]
//! entries) and answers [`Busy`], and the newer one starts as soon as the walk let go. A request
//! that is itself superseded while waiting answers [`Busy`] at once, and none waits longer than
//! [`PREVIEW_TIME`] plus [`WAIT_SLACK`].

use super::select;
use super::source;
use super::{DEFAULT_MAX_CANDIDATES, MAX_CANDIDATES_TOTAL};
use anyhow::{Result, bail};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, TryLockError};
use std::time::{Duration, Instant};

/// Most folders a preview reads.
pub const PREVIEW_MAX_DIRS: usize = 2_000;
/// Longest walk of a preview.
pub const PREVIEW_TIME: Duration = Duration::from_secs(2);
/// How much longer than [`PREVIEW_TIME`] a request waits for the running walk.
pub const WAIT_SLACK: Duration = Duration::from_millis(500);

/// The single-flight state of the previews (one process-wide instance, [`PREVIEWS`]).
#[derive(Debug)]
pub struct SingleFlight {
    /// The newest request (its number).
    latest: AtomicU64,
    /// Held by the running walk.
    walk: Mutex<()>,
}

impl SingleFlight {
    pub const fn new() -> SingleFlight {
        SingleFlight {
            latest: AtomicU64::new(0),
            walk: Mutex::new(()),
        }
    }

    /// Run `f` as the newest request: wait for the running walk (stopped by this request's
    /// arrival through its `cancel`), unless a newer request arrives or [`PREVIEW_TIME`] +
    /// [`WAIT_SLACK`] pass ([`Busy`]). `f` gets the cancel check of its own walk.
    pub fn run<T>(&self, f: impl FnOnce(&(dyn Fn() -> bool + Sync)) -> Result<T>) -> Result<T> {
        let me = self.latest.fetch_add(1, Ordering::SeqCst) + 1;
        let newer = || self.latest.load(Ordering::SeqCst) != me;
        let until = Instant::now() + PREVIEW_TIME + WAIT_SLACK;
        let _walk = loop {
            if newer() {
                bail!(Busy);
            }
            match self.walk.try_lock() {
                Ok(g) => break g,
                Err(TryLockError::Poisoned(p)) => break p.into_inner(),
                Err(TryLockError::WouldBlock) => {}
            }
            if Instant::now() >= until {
                bail!(Busy);
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        f(&newer)
    }
}

impl Default for SingleFlight {
    fn default() -> Self {
        SingleFlight::new()
    }
}

/// The previews of this process.
static PREVIEWS: SingleFlight = SingleFlight::new();

/// A preview that did not run or stopped: a newer one replaced it, or the running walk did not
/// let go in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Busy;

impl std::fmt::Display for Busy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "busy: a newer preview replaced this one")
    }
}

impl std::error::Error for Busy {}

/// Videos of one camera folder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CameraVideos {
    pub camera: String,
    pub videos: usize,
}

/// The preview of a path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Preview {
    pub videos: usize,
    pub images: usize,
    /// Videos per camera folder (from the file names or the camera given), by name.
    pub cameras: Vec<CameraVideos>,
    /// Size of the videos and images.
    pub bytes: u64,
    /// `bytes` for people (`3.1 GB`).
    pub size: String,
    /// Blue Iris `.bvr` clips skipped.
    pub bvr: usize,
    /// Other files skipped.
    pub other: usize,
    /// The listing stopped at [`source::MAX_FILES`].
    pub truncated: bool,
    /// The candidate budget of the import.
    pub budget_total: usize,
    /// Most candidate frames a video gives (the largest per-video quota).
    pub per_video: usize,
    /// The smallest quota of a video that is read.
    pub per_video_min: usize,
    pub videos_skipped: usize,
    /// Images used (an even spread when over their share).
    pub images_used: usize,
    pub warnings: Vec<String>,
    /// One line: "12 videos from 4 cameras, 240 images (3.1 GB); 2 .bvr skipped; up to 1,500
    /// frames per video".
    pub summary: String,
}

/// `n` with thousands separators (`1,500`).
pub fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{} {}", thousands(n), if n == 1 { one } else { many })
}

/// [`preview_with`] within the preview budget, one walk at a time (see the module doc); an
/// error that is [`Busy`] (`downcast_ref`) means a newer request replaced this one.
pub fn preview_latest(path: &Path, camera: Option<&str>, per_video_max: usize) -> Result<Preview> {
    PREVIEWS.run(|newer| {
        let budget = source::Budget {
            max_dirs: Some(PREVIEW_MAX_DIRS),
            deadline: Some(Instant::now() + PREVIEW_TIME),
            cancel: Some(newer),
            sizes: true,
            ..source::Budget::default()
        };
        preview_with(path, camera, per_video_max, &budget)
    })
}

/// The preview of an import of `path` (already checked by the caller) with the camera folder
/// `camera` (None = from the file names) and `per_video_max` candidates per video at most, with
/// no limit on the walk but [`source::MAX_FILES`].
pub fn preview(path: &Path, camera: Option<&str>, per_video_max: usize) -> Result<Preview> {
    preview_with(
        path,
        camera,
        per_video_max,
        &source::Budget {
            sizes: true,
            ..source::Budget::default()
        },
    )
}

/// [`preview`] with the walk limited by `budget` (its `sizes` should be set: the size comes from
/// the walk). A cancelled walk is [`Busy`].
pub fn preview_with(
    path: &Path,
    camera: Option<&str>,
    per_video_max: usize,
    budget: &source::Budget<'_>,
) -> Result<Preview> {
    let d = source::discover_with(path, budget)?;
    if d.stopped == Some(source::Stop::Cancelled) {
        bail!(Busy);
    }
    let per_video_max = if per_video_max == 0 {
        DEFAULT_MAX_CANDIDATES
    } else {
        per_video_max.min(MAX_CANDIDATES_TOTAL)
    };
    let cams: Vec<String> = d
        .videos
        .iter()
        .map(|v| source::camera_of(v, camera))
        .collect();
    let mut by_cam: BTreeMap<&str, usize> = BTreeMap::new();
    for c in &cams {
        *by_cam.entry(c.as_str()).or_default() += 1;
    }
    let refs: Vec<&str> = cams.iter().map(String::as_str).collect();
    let b = select::budget(&refs, d.images.len(), MAX_CANDIDATES_TOTAL, per_video_max);
    let bytes = d.bytes;
    let size = crate::resources::catalog::format_size(bytes);
    let mut warnings = d.warnings();
    warnings.extend(b.warnings(MAX_CANDIDATES_TOTAL));

    let mut parts: Vec<String> = Vec::new();
    if !d.videos.is_empty() {
        parts.push(format!(
            "{} from {}",
            plural(d.videos.len(), "video", "videos"),
            plural(by_cam.len(), "camera", "cameras")
        ));
    }
    if !d.images.is_empty() {
        parts.push(plural(d.images.len(), "image", "images"));
    }
    let listed = d.videos.len() + d.images.len();
    let mut summary = match d.stopped {
        // "large folder: 20,000+ videos and images, counted the first 20,000: ..."
        Some(source::Stop::Files) => format!(
            "large folder: {}+ videos and images, counted the first {}: ",
            thousands(listed),
            thousands(listed)
        ),
        Some(source::Stop::Folders | source::Stop::Time) => format!(
            "large folder: counted the first {} in {} (the preview stops there{}): ",
            plural(listed, "video or image", "videos and images"),
            plural(d.dirs, "folder", "folders"),
            if d.stopped == Some(source::Stop::Time) {
                format!(", after {} s", PREVIEW_TIME.as_secs())
            } else {
                String::new()
            }
        ),
        _ => String::new(),
    };
    if parts.is_empty() {
        summary.push_str("nothing to import");
    } else {
        summary.push_str(&format!("{} ({size})", parts.join(", ")));
    }
    if d.bvr > 0 {
        summary.push_str(&format!("; {} .bvr skipped", thousands(d.bvr)));
    }
    if !d.videos.is_empty() {
        let (lo, hi) = (b.min_per_video(), b.max_per_video());
        if lo == hi {
            summary.push_str(&format!("; up to {} frames per video", thousands(hi)));
        } else {
            summary.push_str(&format!(
                "; up to {}-{} frames per video",
                thousands(lo),
                thousands(hi)
            ));
        }
        if b.skipped_videos() > 0 {
            summary.push_str(&format!(
                " ({} skipped)",
                plural(b.skipped_videos(), "video", "videos")
            ));
        }
    }
    if b.images < b.image_total {
        summary.push_str(&format!("; {} images used", thousands(b.images)));
    }
    Ok(Preview {
        videos: d.videos.len(),
        images: d.images.len(),
        cameras: by_cam
            .into_iter()
            .map(|(camera, videos)| CameraVideos {
                camera: camera.to_string(),
                videos,
            })
            .collect(),
        bytes,
        size,
        bvr: d.bvr,
        other: d.other,
        truncated: d.truncated,
        budget_total: MAX_CANDIDATES_TOTAL,
        per_video: b.max_per_video(),
        per_video_min: b.min_per_video(),
        videos_skipped: b.skipped_videos(),
        images_used: b.images,
        warnings,
        summary,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn thousands_separators() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1500), "1,500");
        assert_eq!(thousands(20_000), "20,000");
        assert_eq!(thousands(1_234_567), "1,234,567");
    }

    #[test]
    fn a_folder_preview_counts_cameras_bytes_and_skips() {
        let dir = std::env::temp_dir().join(format!("bop-import-preview-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("Clips")).unwrap();
        for (f, n) in [
            ("Clips/Drive.20240101_010101.mp4", 1000),
            ("Clips/Drive.20240101_020202.mp4", 1000),
            ("Clips/Porch.20240101_010101.mkv", 500),
            ("Clips/Porch.20240101_020202.bvr", 10),
            ("a.jpg", 20),
            ("notes.txt", 5),
        ] {
            std::fs::write(dir.join(f), vec![0u8; n]).unwrap();
        }
        let p = preview(&dir, None, 0).unwrap();
        assert_eq!((p.videos, p.images, p.bvr, p.other), (3, 1, 1, 1));
        assert_eq!(
            p.cameras,
            [
                CameraVideos {
                    camera: "drive".into(),
                    videos: 2
                },
                CameraVideos {
                    camera: "porch".into(),
                    videos: 1
                }
            ]
        );
        assert_eq!(p.bytes, 2520);
        assert_eq!(p.per_video, DEFAULT_MAX_CANDIDATES);
        assert_eq!(p.images_used, 1);
        assert_eq!(
            p.summary,
            "3 videos from 2 cameras, 1 image (3 kB); 1 .bvr skipped; up to 1,500 frames per video"
        );
        assert!(p.warnings[0].contains(".bvr"));
        // One camera given: one folder; a smaller cap shows.
        let p = preview(&dir, Some("Yard"), 100).unwrap();
        assert_eq!(p.cameras.len(), 1);
        assert!(p.summary.contains("from 1 camera,"), "{}", p.summary);
        assert!(
            p.summary.ends_with("up to 100 frames per video"),
            "{}",
            p.summary
        );
        // JSON for the page.
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v["cameras"][0]["camera"], "yard");
        assert_eq!(v["per_video"], 100);
        assert!(preview(&dir.join("missing"), None, 0).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_large_folder_preview_says_what_it_counted() {
        let dir = std::env::temp_dir().join(format!("bop-import-large-{}", uuid::Uuid::new_v4()));
        for i in 0..6 {
            let sub = dir.join(format!("d{i}"));
            std::fs::create_dir_all(&sub).unwrap();
            std::fs::write(sub.join(format!("Cam.2024010{i}_010101.mp4")), b"v").unwrap();
        }
        let folders = source::Budget {
            max_dirs: Some(3),
            sizes: true,
            ..source::Budget::default()
        };
        let p = preview_with(&dir, None, 0, &folders).unwrap();
        assert!(p.truncated);
        assert!(
            p.summary.starts_with(
                "large folder: counted the first 2 videos and images in 3 folders (the preview \
                 stops there): 2 videos from 1 camera"
            ),
            "{}",
            p.summary
        );
        let files = source::Budget {
            max_files: Some(4),
            sizes: true,
            ..source::Budget::default()
        };
        let p = preview_with(&dir, None, 0, &files).unwrap();
        assert!(
            p.summary
                .starts_with("large folder: 4+ videos and images, counted the first 4: 4 videos"),
            "{}",
            p.summary
        );
        assert_eq!(p.bytes, 4);
        let late = source::Budget {
            deadline: Some(Instant::now()),
            ..source::Budget::default()
        };
        let e = preview_with(&dir, None, 0, &late).unwrap_err();
        assert!(e.to_string().contains("large folder"), "{e}");
        let stop = || true;
        let cancelled = source::Budget {
            cancel: Some(&stop),
            ..source::Budget::default()
        };
        let e = preview_with(&dir, None, 0, &cancelled).unwrap_err();
        assert_eq!(e.downcast_ref::<Busy>(), Some(&Busy));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One walk at a time: a newer request stops the running walk (which answers busy) and runs
    /// right after it; a request superseded while it waits answers busy at once.
    #[test]
    fn a_newer_preview_supersedes_the_running_one() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        let sf = Arc::new(SingleFlight::new());
        let walking = Arc::new(AtomicBool::new(false));
        // A walk that runs until it is cancelled.
        let (s1, w1) = (sf.clone(), walking.clone());
        let first = std::thread::spawn(move || {
            s1.run::<()>(|cancel| {
                w1.store(true, Ordering::SeqCst);
                let t = Instant::now();
                while !cancel() {
                    assert!(t.elapsed() < Duration::from_secs(10), "never cancelled");
                    std::thread::sleep(Duration::from_millis(1));
                }
                bail!(Busy)
            })
        });
        let t = Instant::now();
        while !walking.load(Ordering::SeqCst) {
            assert!(t.elapsed() < Duration::from_secs(10));
            std::thread::sleep(Duration::from_millis(1));
        }
        let started = Instant::now();
        let second = sf.run(|cancel| Ok(cancel()));
        assert!(!second.unwrap(), "ran, not cancelled");
        assert!(started.elapsed() < PREVIEW_TIME, "{:?}", started.elapsed());
        let e = first.join().unwrap().unwrap_err();
        assert_eq!(e.downcast_ref::<Busy>(), Some(&Busy));

        // Waiting behind a walk that never lets go: busy after the wait budget.
        let sf = SingleFlight::new();
        let held = sf.walk.lock().unwrap();
        let t = Instant::now();
        let e = sf.run(|_| Ok(())).unwrap_err();
        assert!(e.downcast_ref::<Busy>().is_some());
        assert!(t.elapsed() >= PREVIEW_TIME, "{:?}", t.elapsed());
        // Superseded while waiting: busy at once.
        std::thread::scope(|s| {
            let waiter = s.spawn(|| sf.run(|_| Ok::<(), anyhow::Error>(())));
            std::thread::sleep(Duration::from_millis(50));
            sf.latest.fetch_add(1, Ordering::SeqCst);
            let t = Instant::now();
            assert!(
                waiter
                    .join()
                    .unwrap()
                    .unwrap_err()
                    .downcast_ref::<Busy>()
                    .is_some()
            );
            assert!(t.elapsed() < Duration::from_secs(1));
        });
        drop(held);
        // Free again.
        assert!(sf.run(|_| Ok(1)).is_ok());
    }
}
