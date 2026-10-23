//! What an import reads: local video files and image folders (Blue Iris clip and alert folders
//! included), a video URL, or pinned open footage. Pure planning plus a directory walk:
//! [`discover`] lists the videos and images under a path (Blue Iris `.bvr` clips are counted and
//! explained, never read), [`check_url`] refuses URLs of unsecured-camera aggregators and anything
//! but http(s), [`check_license`] enforces the per-import licence confirmation, and
//! [`blue_iris_time`] / [`camera_of`] read Blue Iris file names (`Driveway.20240315_083012.mp4`).

use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

/// Video extensions read by the import (PyAV / FFmpeg decodes them).
pub const VIDEO_EXTS: [&str; 4] = ["mp4", "mkv", "avi", "mov"];
/// Image extensions read by the import.
pub const IMAGE_EXTS: [&str; 4] = ["jpg", "jpeg", "png", "bmp"];
/// Most files a folder import lists.
pub const MAX_FILES: usize = 20_000;
/// Deepest folder level walked below the import path.
pub const MAX_DEPTH: usize = 6;

/// Hosts (and their subdomains) that list unsecured or private cameras without their owners'
/// consent. Footage from them is never imported, whatever licence is claimed.
pub const DENYLIST: [&str; 12] = [
    "insecam.org",
    "insecam.com",
    "insecam.net",
    "insecam.cc",
    "opentopia.com",
    "shodan.io",
    "camhacker.com",
    "camhackers.com",
    "lookcam.com",
    "unsecuredcams.com",
    "unsecuredcameras.com",
    "openipcam.com",
];

/// Where the frames come from.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase", tag = "kind", content = "value")]
pub enum ImportSource {
    /// A video file, an image, or a folder of them (a Blue Iris clip / alert folder).
    Path(PathBuf),
    /// A video page or file URL, downloaded with yt-dlp.
    Url(String),
    /// Pinned open footage (`meva-sample`).
    Dataset(String),
}

impl ImportSource {
    /// Short text for titles and the provenance (`clip.mp4`, `https://...`, `meva-sample`).
    pub fn describe(&self) -> String {
        match self {
            ImportSource::Path(p) => p.file_name().map_or_else(
                || p.display().to_string(),
                |n| n.to_string_lossy().into_owned(),
            ),
            ImportSource::Url(u) => u.clone(),
            ImportSource::Dataset(d) => d.clone(),
        }
    }
}

fn ext_of(p: &Path) -> String {
    p.extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default()
}

/// A video file the import reads.
pub fn is_video(p: &Path) -> bool {
    VIDEO_EXTS.contains(&ext_of(p).as_str())
}

/// An image file the import reads.
pub fn is_image(p: &Path) -> bool {
    IMAGE_EXTS.contains(&ext_of(p).as_str())
}

/// What a path holds.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct Discovered {
    pub videos: Vec<PathBuf>,
    pub images: Vec<PathBuf>,
    /// Blue Iris `.bvr` clips (not readable: export them to MP4 in Blue Iris).
    pub bvr: usize,
    /// Other files skipped.
    pub other: usize,
    /// The listing stopped at [`MAX_FILES`].
    pub truncated: bool,
}

impl Discovered {
    /// Warnings for the report.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if self.bvr > 0 {
            w.push(format!(
                "{} Blue Iris .bvr clip(s) skipped: .bvr is Blue Iris's own format and cannot be \
                 read; export the clips to MP4 in Blue Iris (select the clips, right-click, Export, \
                 format MP4) and import the exported files",
                self.bvr
            ));
        }
        if self.truncated {
            w.push(format!(
                "only the first {MAX_FILES} files were listed; import sub-folders one at a time"
            ));
        }
        w
    }
}

/// The videos and images at `path` (a file, or a folder walked [`MAX_DEPTH`] levels deep; hidden
/// entries and symlinked folders skipped), sorted by path.
pub fn discover(path: &Path) -> Result<Discovered> {
    let meta = std::fs::metadata(path)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?;
    let mut d = Discovered::default();
    if meta.is_file() {
        classify(path, &mut d);
        if d.videos.is_empty() && d.images.is_empty() {
            if d.bvr > 0 {
                bail!("{}", d.warnings().join("; "));
            }
            bail!(
                "{} is not a video ({}) or image ({}) file",
                path.display(),
                VIDEO_EXTS.join(", "),
                IMAGE_EXTS.join(", ")
            );
        }
        return Ok(d);
    }
    let mut stack = vec![(path.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut entries: Vec<_> = rd.flatten().collect();
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for e in entries {
            if e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() {
                if depth < MAX_DEPTH {
                    stack.push((e.path(), depth + 1));
                }
            } else if ft.is_file() {
                if d.videos.len() + d.images.len() >= MAX_FILES {
                    d.truncated = true;
                    continue;
                }
                classify(&e.path(), &mut d);
            }
        }
    }
    d.videos.sort();
    d.images.sort();
    if d.videos.is_empty() && d.images.is_empty() {
        let mut why = format!(
            "{} has no video ({}) or image ({}) files",
            path.display(),
            VIDEO_EXTS.join(", "),
            IMAGE_EXTS.join(", ")
        );
        for w in d.warnings() {
            why.push_str("; ");
            why.push_str(&w);
        }
        bail!("{why}");
    }
    Ok(d)
}

fn classify(p: &Path, d: &mut Discovered) {
    if is_video(p) {
        d.videos.push(p.to_path_buf());
    } else if is_image(p) {
        d.images.push(p.to_path_buf());
    } else if ext_of(p) == "bvr" {
        d.bvr += 1;
    } else {
        d.other += 1;
    }
}

/// The host of an http(s) URL, lowercase, without credentials, port or a trailing dot.
fn host_of(url: &str) -> Option<String> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let authority = rest.split(['/', '?', '#']).next()?;
    let host_port = authority.rsplit('@').next()?;
    let host = if let Some(v6) = host_port.strip_prefix('[') {
        v6.split(']').next()?.to_string()
    } else {
        host_port.split(':').next()?.to_string()
    };
    let h = host.trim_end_matches('.').to_ascii_lowercase();
    (!h.is_empty()).then_some(h)
}

/// Check a video URL: http(s) only, a host, and not on the [`DENYLIST`] of unsecured-camera
/// aggregators (the host or any parent domain). Returns the URL trimmed.
pub fn check_url(url: &str) -> Result<String> {
    let u = url.trim();
    if u.is_empty() {
        bail!("no URL given");
    }
    if u.chars().any(char::is_control) || u.len() > 2048 {
        bail!("the URL is not valid");
    }
    let lower = u.to_ascii_lowercase();
    if !(lower.starts_with("https://") || lower.starts_with("http://")) {
        bail!(
            "only http:// and https:// video URLs can be imported (got {}); for a camera stream, \
             record a clip and import the file",
            u.split(':').next().unwrap_or("?")
        );
    }
    let Some(host) = host_of(&lower) else {
        bail!("the URL has no host");
    };
    if let Some(bad) = DENYLIST
        .iter()
        .find(|d| host == **d || host.ends_with(&format!(".{d}")))
    {
        bail!(
            "{host} is refused: {bad} lists unsecured or private cameras whose owners did not \
             agree to be watched; footage from such sites is never imported. Use your own \
             cameras' clips or footage published under a licence that allows it"
        );
    }
    Ok(u.to_string())
}

/// The licence of an import: the stated one (or `default` when allowed) and the confirmation,
/// which a file or URL import must carry. Errors say what to pass.
pub fn check_license(
    source: &ImportSource,
    license: Option<&str>,
    confirmed: bool,
) -> Result<String> {
    let stated = license.map(str::trim).filter(|l| !l.is_empty());
    match source {
        ImportSource::Dataset(_) => Ok(stated.unwrap_or_default().to_string()),
        ImportSource::Url(_) => {
            let Some(l) = stated else {
                bail!(
                    "state the video's licence (--license, e.g. \"CC BY 4.0\" or \"own footage\") \
                     and confirm you may use it for training (--confirm-license)"
                );
            };
            if !confirmed {
                bail!(
                    "confirm that the licence \"{l}\" allows you to use this video for training \
                     (--confirm-license, or tick the box on the Training page)"
                );
            }
            if l.chars().count() > 200 || l.chars().any(char::is_control) {
                bail!("the licence text must be one line of at most 200 characters");
            }
            Ok(l.to_string())
        }
        ImportSource::Path(_) => {
            let l = stated.unwrap_or("own footage");
            if !confirmed {
                bail!(
                    "confirm that you may use this footage for training (licence \"{l}\"; \
                     --confirm-license, or tick the box on the Training page)"
                );
            }
            if l.chars().count() > 200 || l.chars().any(char::is_control) {
                bail!("the licence text must be one line of at most 200 characters");
            }
            Ok(l.to_string())
        }
    }
}

/// The recording time of a Blue Iris file name (`Driveway.20240315_083012.mp4`,
/// `Cam1.20240315_083012.123456.3-1.jpg`): the first `.YYYYMMDD_HHMMSS` part, as local time
/// is unknown taken as UTC.
pub fn blue_iris_time(name: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    let file = name.rsplit(['/', '\\']).next().unwrap_or(name);
    for part in file.split('.').skip(1) {
        let b = part.as_bytes();
        if b.len() >= 15
            && b.get(8) == Some(&b'_')
            && b.get(..8).is_some_and(|d| d.iter().all(u8::is_ascii_digit))
            && b.get(9..15)
                .is_some_and(|d| d.iter().all(u8::is_ascii_digit))
        {
            let s = part.get(..15)?;
            if let Ok(t) = chrono::NaiveDateTime::parse_from_str(s, "%Y%m%d_%H%M%S") {
                return Some(t.and_utc());
            }
        }
    }
    None
}

/// The recording time of a MEVA clip name (`2018-03-11.23-55-00.00-00-00.school.G639.r13.avi`).
pub fn meva_time(name: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    let file = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let mut parts = file.split('.');
    let (d, t) = (parts.next()?, parts.next()?);
    chrono::NaiveDateTime::parse_from_str(&format!("{d} {t}"), "%Y-%m-%d %H-%M-%S")
        .ok()
        .map(|x| x.and_utc())
}

/// The camera folder of a file: `camera` when given, else the file name's camera part the way
/// the collector names Blue Iris uploads (`Driveway.20240315_083012.mp4` -> `driveway`; a MEVA
/// clip -> `<site>-<camera>`), else `import`.
pub fn camera_of(file: &Path, camera: Option<&str>) -> String {
    let fallback = |s: &str| crate::collect::layout::camera_dir(&format!("import|cam:{s}"), 0, 0);
    if let Some(c) = camera.map(str::trim).filter(|c| !c.is_empty()) {
        return fallback(c);
    }
    let name = file
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if meva_time(&name).is_some() {
        let parts: Vec<&str> = name.split('.').collect();
        if let (Some(site), Some(cam)) = (parts.get(3), parts.get(4)) {
            return fallback(&format!("meva-{site}-{cam}"));
        }
    }
    let key = crate::static_memory::camera_key("import", &name, 0, 0);
    if key.contains("|cam:") {
        crate::collect::layout::camera_dir(&key, 0, 0)
    } else {
        "import".into()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn denylisted_hosts_and_other_schemes_are_refused() {
        for bad in [
            "http://www.insecam.org/en/view/123/",
            "https://INSECAM.com/cam",
            "https://images.shodan.io/x.jpg",
            "http://user:pw@opentopia.com:8080/webcam",
            "https://insecam.org./x",
        ] {
            let e = check_url(bad).unwrap_err().to_string();
            assert!(e.contains("is refused"), "{bad}: {e}");
            assert!(e.contains("unsecured"), "{bad}: {e}");
        }
        for bad in [
            "rtsp://192.168.1.10/stream",
            "file:///etc/passwd",
            "ftp://x/y.mp4",
            "",
            "https://",
        ] {
            assert!(check_url(bad).is_err(), "{bad}");
        }
        assert_eq!(
            check_url(" https://www.youtube.com/watch?v=abc ").unwrap(),
            "https://www.youtube.com/watch?v=abc"
        );
        // A host that only contains a denied name is fine.
        assert!(check_url("https://notinsecam.org.example.com/v.mp4").is_ok());
        assert!(check_url("https://example.com/insecam.org.mp4").is_ok());
        assert_eq!(
            host_of("https://[::1]:8080/x").as_deref(),
            Some("::1"),
            "IPv6 literal"
        );
    }

    #[test]
    fn licence_confirmation_is_required_for_files_and_urls() {
        let url = ImportSource::Url("https://example.com/v.mp4".into());
        let path = ImportSource::Path("clip.mp4".into());
        let ds = ImportSource::Dataset("meva-sample".into());
        assert!(
            check_license(&url, None, true)
                .unwrap_err()
                .to_string()
                .contains("--license")
        );
        assert!(
            check_license(&url, Some("CC BY 4.0"), false)
                .unwrap_err()
                .to_string()
                .contains("--confirm-license")
        );
        assert_eq!(
            check_license(&url, Some(" CC BY 4.0 "), true).unwrap(),
            "CC BY 4.0"
        );
        assert!(check_license(&path, None, false).is_err());
        assert_eq!(check_license(&path, None, true).unwrap(), "own footage");
        assert_eq!(check_license(&ds, None, false).unwrap(), "");
        assert!(check_license(&url, Some("a\nb"), true).is_err());
    }

    #[test]
    fn blue_iris_and_meva_names() {
        let t = blue_iris_time("Driveway.20240315_083012.mp4").unwrap();
        assert_eq!(t.to_rfc3339(), "2024-03-15T08:30:12+00:00");
        assert!(
            blue_iris_time("D:\\BlueIris\\Alerts\\Cam1.20240315_083012.123456.3-1.jpg").is_some()
        );
        assert!(
            blue_iris_time("Cam1.20241315_083012.mp4").is_none(),
            "month 13"
        );
        assert!(blue_iris_time("clip.mp4").is_none());
        let m = meva_time("2018-03-11.23-55-00.00-00-00.school.G639.r13.avi").unwrap();
        assert_eq!(m.to_rfc3339(), "2018-03-11T23:55:00+00:00");
        assert_eq!(
            camera_of(Path::new("/x/Driveway.20240315_083012.mp4"), None),
            "driveway"
        );
        assert_eq!(
            camera_of(
                Path::new("2018-03-11.23-55-00.00-00-00.school.G639.r13.avi"),
                None
            ),
            "meva-school-g639"
        );
        assert_eq!(camera_of(Path::new("image.jpg"), None), "import");
        assert_eq!(
            camera_of(Path::new("a.mp4"), Some("Back Yard")),
            "back_yard"
        );
    }

    #[test]
    fn discover_lists_videos_images_and_explains_bvr() {
        let dir = std::env::temp_dir().join(format!("bop-import-src-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("Clips/sub")).unwrap();
        std::fs::create_dir_all(dir.join(".hidden")).unwrap();
        for f in [
            "Clips/Drive.20240101_010101.mp4",
            "Clips/Drive.20240101_020202.bvr",
            "Clips/sub/b.MKV",
            "Alerts.jpg",
            "notes.txt",
            ".hidden/x.mp4",
        ] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        let d = discover(&dir).unwrap();
        assert_eq!(d.videos.len(), 2);
        assert_eq!(d.images, [dir.join("Alerts.jpg")]);
        assert_eq!((d.bvr, d.other), (1, 1));
        assert!(d.warnings()[0].contains("export the clips to MP4"));
        let one = discover(&dir.join("Clips/sub/b.MKV")).unwrap();
        assert_eq!(one.videos.len(), 1);
        let e = discover(&dir.join("Clips/Drive.20240101_020202.bvr")).unwrap_err();
        assert!(e.to_string().contains(".bvr"), "{e}");
        assert!(discover(&dir.join("notes.txt")).is_err());
        assert!(discover(&dir.join("missing")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
