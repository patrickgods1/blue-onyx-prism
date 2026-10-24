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

/// Why a listing stopped before the end of the walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Stop {
    /// [`MAX_FILES`] videos and images listed, with more entries left.
    Files,
    /// [`Budget::max_dirs`] folders read.
    Folders,
    /// [`Budget::deadline`] passed.
    Time,
    /// [`Budget::cancel`] said so (a newer preview).
    Cancelled,
}

/// Limits of a walk on top of [`MAX_FILES`] and [`MAX_DEPTH`] (the preview's budget; an import
/// lists with [`Budget::default`], no further limit).
#[derive(Default)]
pub struct Budget<'a> {
    /// Most videos and images listed (None = [`MAX_FILES`]; smaller in tests).
    pub max_files: Option<usize>,
    /// Most folders read (`read_dir` calls).
    pub max_dirs: Option<usize>,
    /// The walk stops when this passes (checked per folder and every [`CHECK_EVERY`] entries).
    pub deadline: Option<std::time::Instant>,
    /// Stop when this answers true (checked like the deadline).
    pub cancel: Option<&'a (dyn Fn() -> bool + Sync)>,
    /// Sum the sizes of the videos and images into [`Discovered::bytes`].
    pub sizes: bool,
}

/// Entries between two checks of the deadline and the cancel flag inside one folder.
pub const CHECK_EVERY: usize = 256;

impl std::fmt::Debug for Budget<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Budget")
            .field("max_files", &self.max_files)
            .field("max_dirs", &self.max_dirs)
            .field("deadline", &self.deadline)
            .field("cancel", &self.cancel.is_some())
            .field("sizes", &self.sizes)
            .finish()
    }
}

impl Budget<'_> {
    fn interrupted(&self) -> Option<Stop> {
        if self.cancel.is_some_and(|c| c()) {
            return Some(Stop::Cancelled);
        }
        if self
            .deadline
            .is_some_and(|d| std::time::Instant::now() >= d)
        {
            return Some(Stop::Time);
        }
        None
    }
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
    /// The listing stopped early ([`Discovered::stopped`] says why).
    pub truncated: bool,
    pub stopped: Option<Stop>,
    /// Folders read.
    pub dirs: usize,
    /// Size of the videos and images (only with [`Budget::sizes`]).
    pub bytes: u64,
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
        match self.stopped {
            Some(Stop::Files) => w.push(format!(
                "only the first {MAX_FILES} videos and images are listed; import sub-folders one \
                 at a time"
            )),
            Some(Stop::Folders | Stop::Time) => w.push(format!(
                "a large folder: the preview counted the first {} folders only; an import lists \
                 up to {MAX_FILES} videos and images",
                self.dirs
            )),
            Some(Stop::Cancelled) | None => {}
        }
        w
    }
}

/// The videos and images at `path` (a file, or a folder walked [`MAX_DEPTH`] levels deep; hidden
/// entries and symlinked folders skipped), sorted by path. The walk stops as soon as
/// [`MAX_FILES`] videos and images are listed.
pub fn discover(path: &Path) -> Result<Discovered> {
    discover_with(path, &Budget::default())
}

/// [`discover`] within `budget`: the walk stops (`truncated`, [`Discovered::stopped`]) at the
/// first limit reached and never reads another folder after it. A cancelled walk answers what
/// it found (maybe nothing) without an error; a walk stopped by the budget before finding
/// anything is an error that says so.
pub fn discover_with(path: &Path, budget: &Budget<'_>) -> Result<Discovered> {
    let meta = std::fs::metadata(path)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?;
    let mut d = Discovered::default();
    if meta.is_file() {
        classify(path, &mut d);
        if budget.sizes && (!d.videos.is_empty() || !d.images.is_empty()) {
            d.bytes = meta.len();
        }
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
    d.stopped = walk(path, budget, &mut d);
    d.truncated = d.stopped.is_some();
    d.videos.sort();
    d.images.sort();
    let empty = d.videos.is_empty() && d.images.is_empty();
    if d.stopped == Some(Stop::Cancelled) {
        return Ok(d);
    }
    if empty && matches!(d.stopped, Some(Stop::Folders | Stop::Time)) {
        bail!(
            "{} is a large folder: no video ({}) or image ({}) files in the first {} folders the \
             preview reads; choose the sub-folder that holds the clips",
            path.display(),
            VIDEO_EXTS.join(", "),
            IMAGE_EXTS.join(", "),
            d.dirs
        );
    }
    if empty {
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

/// The folder walk of [`discover_with`]: why it stopped early, None when it listed everything.
fn walk(path: &Path, budget: &Budget<'_>, d: &mut Discovered) -> Option<Stop> {
    let mut stack = vec![(path.to_path_buf(), 0usize)];
    let mut seen = 0usize;
    let mut tick = |budget: &Budget<'_>| -> Option<Stop> {
        seen += 1;
        if seen.is_multiple_of(CHECK_EVERY) {
            budget.interrupted()
        } else {
            None
        }
    };
    while let Some((dir, depth)) = stack.pop() {
        if let Some(stop) = budget.interrupted() {
            return Some(stop);
        }
        if budget.max_dirs.is_some_and(|m| d.dirs >= m) {
            return Some(Stop::Folders);
        }
        d.dirs += 1;
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut entries = Vec::new();
        for e in rd.flatten() {
            entries.push(e);
            if let Some(stop) = tick(budget) {
                return Some(stop);
            }
        }
        entries.sort_by_key(std::fs::DirEntry::file_name);
        let mut rest = entries.into_iter().peekable();
        while let Some(e) = rest.next() {
            if let Some(stop) = tick(budget) {
                return Some(stop);
            }
            if e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() {
                if depth < MAX_DEPTH {
                    stack.push((e.path(), depth + 1));
                }
            } else if ft.is_file() {
                let p = e.path();
                let media = is_video(&p) || is_image(&p);
                classify(&p, d);
                if media && budget.sizes {
                    d.bytes += e.metadata().map_or(0, |m| m.len());
                }
                if media && d.videos.len() + d.images.len() >= budget.max_files.unwrap_or(MAX_FILES)
                {
                    // Full: stop here; "more" when anything is left to look at.
                    let more = rest.peek().is_some() || !stack.is_empty();
                    return more.then_some(Stop::Files);
                }
            }
        }
    }
    None
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

    #[test]
    fn a_budget_stops_the_walk_without_reading_further_folders() {
        let dir = std::env::temp_dir().join(format!("bop-import-budget-{}", uuid::Uuid::new_v4()));
        // 30 folders of one clip each, plus a notes file.
        for i in 0..30 {
            let sub = dir.join(format!("d{i:02}"));
            std::fs::create_dir_all(&sub).unwrap();
            std::fs::write(sub.join("Cam.20240101_010101.mp4"), b"vid").unwrap();
        }
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        let all = discover_with(
            &dir,
            &Budget {
                sizes: true,
                ..Budget::default()
            },
        )
        .unwrap();
        assert_eq!((all.videos.len(), all.dirs, all.truncated), (30, 31, false));
        assert_eq!(all.stopped, None);
        assert_eq!(all.bytes, 90);
        // At most 5 folders read: the root and 4 sub-folders.
        let d = discover_with(
            &dir,
            &Budget {
                max_dirs: Some(5),
                sizes: true,
                ..Budget::default()
            },
        )
        .unwrap();
        assert_eq!((d.videos.len(), d.dirs), (4, 5));
        assert_eq!(d.stopped, Some(Stop::Folders));
        assert!(d.truncated);
        assert_eq!(d.bytes, 12);
        assert!(
            d.warnings()[0].contains("first 5 folders"),
            "{:?}",
            d.warnings()
        );
        // A deadline in the past reads nothing; with nothing found that is an error that says
        // the folder is large.
        let e = discover_with(
            &dir,
            &Budget {
                deadline: Some(std::time::Instant::now()),
                ..Budget::default()
            },
        )
        .unwrap_err();
        assert!(e.to_string().contains("is a large folder"), "{e}");
        // A cancelled walk answers what it has, without an error.
        let yes = || true;
        let c = discover_with(
            &dir,
            &Budget {
                cancel: Some(&yes),
                ..Budget::default()
            },
        )
        .unwrap();
        assert_eq!(
            (c.stopped, c.dirs, c.videos.len()),
            (Some(Stop::Cancelled), 0, 0)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_file_cap_stops_at_once_and_only_claims_more_when_there_is_more() {
        let dir = std::env::temp_dir().join(format!("bop-import-cap-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(dir.join("a")).unwrap();
        for i in 0..10 {
            std::fs::write(dir.join("a").join(format!("{i:02}.jpg")), b"").unwrap();
        }
        let cap = |n| Budget {
            max_files: Some(n),
            ..Budget::default()
        };
        // Exactly the cap: not truncated.
        let d = discover_with(&dir, &cap(10)).unwrap();
        assert_eq!((d.images.len(), d.truncated), (10, false));
        // Entries left in the folder: more.
        let d = discover_with(&dir, &cap(4)).unwrap();
        assert_eq!((d.images.len(), d.stopped), (4, Some(Stop::Files)));
        // Folders are read last-sorted first: `a`, then `0extra`, which is never read once `a`
        // fills the cap (the walk stops at once), yet makes it "more".
        std::fs::create_dir_all(dir.join("0extra/c")).unwrap();
        std::fs::write(dir.join("0extra/c/x.mp4"), b"").unwrap();
        let d = discover_with(&dir, &cap(10)).unwrap();
        assert_eq!((d.images.len(), d.videos.len()), (10, 0));
        assert_eq!(d.stopped, Some(Stop::Files));
        assert_eq!(d.dirs, 2, "the root and `a` only");
        assert!(
            d.warnings()[0].contains("only the first 20000"),
            "{:?}",
            d.warnings()
        );
        // Without the cap everything is listed.
        let d = discover(&dir).unwrap();
        assert_eq!((d.images.len(), d.videos.len(), d.dirs), (10, 1, 4));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
