//! What an import of a path would read, before it is queued (`GET /v1/training/import/preview`,
//! the line under the Training page's path field): the videos and images [`super::source::discover`]
//! finds, the cameras of the videos, their size, the skipped files, and the candidate budget
//! ([`super::select::budget`]) each video would get.

use super::select;
use super::source;
use super::{DEFAULT_MAX_CANDIDATES, MAX_CANDIDATES_TOTAL};
use anyhow::Result;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

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

/// The preview of an import of `path` (already checked by the caller) with the camera folder
/// `camera` (None = from the file names) and `per_video_max` candidates per video at most.
pub fn preview(path: &Path, camera: Option<&str>, per_video_max: usize) -> Result<Preview> {
    let d = source::discover(path)?;
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
    let bytes: u64 = d
        .videos
        .iter()
        .chain(&d.images)
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();
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
    let mut summary = if parts.is_empty() {
        "nothing to import".to_string()
    } else {
        format!("{} ({size})", parts.join(", "))
    };
    if d.bvr > 0 {
        summary.push_str(&format!("; {} .bvr skipped", thousands(d.bvr)));
    }
    if d.truncated {
        summary.push_str(&format!(
            "; only the first {} files listed",
            thousands(source::MAX_FILES)
        ));
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
}
