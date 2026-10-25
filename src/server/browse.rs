//! `GET /v1/fs/browse`: a read-only, server-side file browser for the Config page's
//! "Add a model by path" form, the Training page's footage import and the Dataset page's
//! folder picker (`kind=folder`: folders only). Lists folders plus model
//! (`.onnx`/`.xml`), classes (`.yaml`/`.yml`) or, for the import picker only (`kind=video`),
//! video (`.mp4`/`.mkv`/`.avi`/`.mov`) files, and only inside the roots the page offers ([`roots`]: the
//! data root, the models and download directories, the user's home and, on Windows, the
//! drives), so the unauthenticated page discloses as little as possible (SV-10). A path is
//! canonicalized first, so a symlink that leaves every root is refused like any other outside
//! path.
//!
//! [`allowed_path`] is the check for the paths the forms submit ("Add a model by path", the
//! footage import): inside a root as above, or a path the native file dialog of
//! [`super::pick`] returned in the last hour, or anything under it (canonicalized too, so a
//! symlink out of a picked folder is still refused). A pick is trusted because the person at
//! the server's own screen chose it, in a dialog only a loopback peer can open; that lets a
//! folder on another volume or share be used without widening the roots. The listing itself
//! ([`list`]) never trusts picks: it stays inside the roots (SV-10 unchanged).

use super::AppState;
use crate::config::Config;
use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Entries returned at most.
const MAX_ENTRIES: usize = 2000;

#[derive(Debug, Deserialize)]
pub(super) struct BrowseQuery {
    path: Option<String>,
    kind: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Model,
    Classes,
    /// The footage import's picker: videos (folders of images are picked as folders).
    Video,
    /// The Dataset page's folder picker: folders only, no files.
    Folder,
}

impl Kind {
    fn parse(s: Option<&str>) -> Option<Kind> {
        match s.map(str::trim) {
            None | Some("") | Some("model") => Some(Kind::Model),
            Some("classes") => Some(Kind::Classes),
            Some("video") => Some(Kind::Video),
            Some("folder") => Some(Kind::Folder),
            _ => None,
        }
    }

    fn accepts(self, name: &str) -> bool {
        let ext = Path::new(name)
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        match self {
            Kind::Model => ext == "onnx" || ext == "xml",
            Kind::Classes => ext == "yaml" || ext == "yml",
            Kind::Video => crate::training::import::source::VIDEO_EXTS.contains(&ext.as_str()),
            Kind::Folder => false,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Root {
    label: String,
    path: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Entry {
    name: String,
    path: String,
    dir: bool,
    size: Option<u64>,
    size_text: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Listing {
    path: String,
    parent: Option<String>,
    selected: Option<String>,
    roots: Vec<Root>,
    entries: Vec<Entry>,
    truncated: bool,
}

#[derive(Debug)]
pub(super) struct BrowseError {
    status: StatusCode,
    message: String,
}

impl BrowseError {
    /// Status and message, for answers rendered elsewhere.
    pub(super) fn into_parts(self) -> (StatusCode, String) {
        (self.status, self.message)
    }
}

impl IntoResponse for BrowseError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({ "success": false, "message": self.message })),
        )
            .into_response()
    }
}

/// `GET /v1/fs/browse`. The listing (`read_dir` + `metadata` per entry, up to
/// [`MAX_ENTRIES`]) and the root probing (every drive letter on Windows) can block for seconds
/// on a slow or disconnected drive, so they run on the blocking pool, never on one of the two
/// runtime threads that serve detections (SV-07).
pub(super) async fn browse(
    State(state): State<Arc<AppState>>,
    Query(q): Query<BrowseQuery>,
) -> Response {
    super::blocking(move || {
        let cfg = state.config_read().clone();
        match list(&cfg, q.path.as_deref(), q.kind.as_deref()) {
            Ok(l) => Json(l).into_response(),
            Err(e) => e.into_response(),
        }
    })
    .await
}

/// Display form of a path: canonical, without the Windows `\\?\` verbatim prefix.
fn display(p: &Path) -> String {
    let s = p.to_string_lossy().into_owned();
    #[cfg(windows)]
    {
        if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
            return format!(r"\\{rest}");
        }
        if let Some(rest) = s.strip_prefix(r"\\?\") {
            return rest.to_string();
        }
    }
    s
}

fn human_size(n: u64) -> String {
    const UNITS: [&str; 4] = ["KB", "MB", "GB", "TB"];
    if n < 1024 {
        return format!("{n} B");
    }
    let mut v = n as f64 / 1024.0;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < UNITS.len() {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.1} {}", UNITS.get(i).copied().unwrap_or_default())
}

/// The browsable roots (label, canonical path), which are also the only places a listing or
/// an added model may come from.
fn roots(cfg: &Config) -> Vec<(String, PathBuf)> {
    let mut out: Vec<(String, PathBuf)> = Vec::new();
    let mut add = |label: &str, p: PathBuf| {
        if let Ok(c) = p.canonicalize()
            && !out.iter().any(|(_, r)| *r == c)
        {
            out.push((label.to_string(), c));
        }
    };
    add("Data root", cfg.data_root());
    let models = cfg.data_path(&cfg.models_dir);
    if models.is_dir() {
        add("Models", models);
    }
    if let Some(d) = &cfg.download_dir {
        add("Downloads", crate::resolve_path(d));
    }
    for var in ["HOME", "USERPROFILE"] {
        if let Some(h) = std::env::var_os(var).map(PathBuf::from)
            && h.is_dir()
        {
            add("Home", h);
            break;
        }
    }
    #[cfg(windows)]
    for letter in b'A'..=b'Z' {
        let d = PathBuf::from(format!("{}:\\", letter as char));
        if d.is_dir() {
            add(&format!("{}:\\", letter as char), d);
        }
    }
    out
}

fn root_views(roots: &[(String, PathBuf)]) -> Vec<Root> {
    roots
        .iter()
        .map(|(label, p)| Root {
            label: label.clone(),
            path: display(p),
        })
        .collect()
}

/// Lexically inside one of `roots` (the canonical roots, or the data root as configured).
/// Compared without the Windows verbatim prefix: a canonical root carries `\\?\`, a path the
/// client sent (or the configured data root) does not, and `Path::starts_with` treats the
/// prefix as a component of its own.
fn inside_any(p: &Path, roots: &[(String, PathBuf)], cfg: &Config) -> bool {
    let p = PathBuf::from(display(p));
    roots.iter().any(|(_, r)| p.starts_with(display(r))) || p.starts_with(display(&cfg.data_root()))
}

/// The rest of a UNC path (`\\\\server\\share...` or `//server/share...`) after its leading
/// separators; `None` for anything else, including the verbatim (`\\\\?\\`) and device
/// (`\\\\.\\`) namespaces.
fn unc_rest(raw: &str) -> Option<&str> {
    let rest = raw.strip_prefix(r"\\").or_else(|| raw.strip_prefix("//"))?;
    let mut chars = rest.chars();
    let first = chars.next()?;
    let second = chars.next();
    if matches!(first, '?' | '.') && matches!(second, None | Some('\\' | '/')) {
        return None;
    }
    if matches!(first, '\\' | '/') {
        return None;
    }
    Some(rest)
}

/// The canonical path of `raw` (quotes trimmed; relative to the data root) when it exists
/// and lies inside one of [`roots`] or, for the forms, is covered by a native pick
/// ([`super::pick::picked_covers`]; a picked UNC share passes the syntax check too); 400 for
/// bad syntax, 404 when it does not exist inside a root or pick, 403 otherwise (including a
/// symlink whose target leaves every root and pick). Error messages repeat what the client
/// sent, not the resolved path.
pub(super) fn allowed_path(cfg: &Config, raw: &str) -> Result<PathBuf, BrowseError> {
    resolve(cfg, raw, true)
}

/// [`allowed_path`] for a path a form saves for later (a model typed into the Config page's
/// models JSON whose download has not happened yet): also accepted when it does not exist,
/// provided it lies inside one of [`roots`] (or a native pick) and its deepest existing
/// ancestor, canonicalized, does too or is free of symlinks (so a symlinked folder cannot
/// lead out of the roots).
/// Returns the canonical path when it exists, else the resolved one.
pub(super) fn allowed_or_missing_inside_roots(
    cfg: &Config,
    raw: &str,
) -> Result<PathBuf, BrowseError> {
    match allowed_path(cfg, raw) {
        Err(e) if e.status == StatusCode::NOT_FOUND => {}
        other => return other,
    }
    // `allowed_path` answers 404 only after the lexical check passed.
    let raw = raw.trim().trim_matches(|c| c == '"' || c == '\'').trim();
    let start = cfg.data_path(Path::new(raw));
    let roots = roots(cfg);
    let mut ancestor = start.parent();
    while let Some(a) = ancestor {
        if let Ok(canon) = a.canonicalize() {
            // No symlink on the way (a data root not created yet): the lexical check holds.
            let plain = Path::new(&display(&canon)) == a;
            if plain || inside_any(&canon, &roots, cfg) || super::pick::picked_covers(&canon) {
                return Ok(start);
            }
            break;
        }
        ancestor = a.parent();
    }
    Err(BrowseError {
        status: StatusCode::FORBIDDEN,
        message: format!("{raw} leads outside the browsable folders"),
    })
}

/// [`allowed_path`], trusting native picks only when `trust_picked` (never for [`list`]).
fn resolve(cfg: &Config, raw: &str, trust_picked: bool) -> Result<PathBuf, BrowseError> {
    let raw = raw.trim().trim_matches(|c| c == '"' || c == '\'').trim();
    let picked = |p: &Path| trust_picked && super::pick::picked_covers(p);
    crate::config::check_path_syntax("path", raw)
        .or_else(|e| match unc_rest(raw) {
            // A share the person at the server's screen picked: the other checks still apply.
            Some(rest) if picked(Path::new(raw)) => crate::config::check_path_syntax("path", rest),
            _ => Err(e),
        })
        .map_err(|e| BrowseError {
            status: StatusCode::BAD_REQUEST,
            message: e.to_string(),
        })?;
    let roots = roots(cfg);
    let start = cfg.data_path(Path::new(raw));
    let outside = || BrowseError {
        status: StatusCode::FORBIDDEN,
        message: format!(
            "{raw} is outside the browsable folders ({})",
            roots
                .iter()
                .map(|(_, r)| display(r))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    if !inside_any(&start, &roots, cfg) && !picked(&start) {
        return Err(outside());
    }
    let canon = start.canonicalize().map_err(|e| io_error(raw, &e))?;
    if !inside_any(&canon, &roots, cfg) && !picked(&canon) {
        return Err(outside());
    }
    // Windows: `C:\x`, not the verbatim `\\?\C:\x` of `canonicalize` (the folder settings
    // refuse verbatim paths, and the forms store and show what this returns).
    Ok(plain(&canon))
}

/// `p` without the Windows verbatim prefix ([`display`]); unchanged elsewhere.
pub(super) fn plain(p: &Path) -> PathBuf {
    PathBuf::from(display(p))
}

/// `shown` is what the client asked for (never a path the server resolved).
fn io_error(shown: &str, e: &std::io::Error) -> BrowseError {
    match e.kind() {
        std::io::ErrorKind::NotFound => BrowseError {
            status: StatusCode::NOT_FOUND,
            message: format!("not found: {shown}"),
        },
        std::io::ErrorKind::PermissionDenied => BrowseError {
            status: StatusCode::FORBIDDEN,
            message: format!("permission denied: {shown}"),
        },
        _ => BrowseError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("cannot read {shown}: {}", e.kind()),
        },
    }
}

/// The pure listing behind the endpoint.
pub(super) fn list(
    cfg: &Config,
    path: Option<&str>,
    kind: Option<&str>,
) -> Result<Listing, BrowseError> {
    let kind = Kind::parse(kind).ok_or_else(|| BrowseError {
        status: StatusCode::BAD_REQUEST,
        message: "unknown kind (use model, classes, video or folder)".to_string(),
    })?;
    let raw = path
        .map(|p| p.trim().trim_matches(|c| c == '"' || c == '\'').trim())
        .unwrap_or("");
    let canon = if raw.is_empty() {
        let m = cfg.data_path(&cfg.models_dir);
        let start = if m.is_dir() { m } else { cfg.data_root() };
        start
            .canonicalize()
            .map_err(|e| io_error(&display(&start), &e))?
    } else {
        resolve(cfg, raw, false)?
    };
    let (dir, selected) = if canon.is_dir() {
        (canon, None)
    } else {
        let sel = canon.file_name().map(|n| n.to_string_lossy().into_owned());
        match canon.parent() {
            Some(p) => (p.to_path_buf(), sel),
            None => (canon, None),
        }
    };
    let shown = if raw.is_empty() {
        display(&dir)
    } else {
        raw.to_string()
    };
    let rd = std::fs::read_dir(&dir).map_err(|e| io_error(&shown, &e))?;
    let mut dirs: Vec<Entry> = Vec::new();
    let mut files: Vec<Entry> = Vec::new();
    let mut truncated = false;
    for item in rd.flatten() {
        let name = item.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let Ok(meta) = std::fs::metadata(item.path()) else {
            continue;
        };
        let is_dir = meta.is_dir();
        if !is_dir && !(meta.is_file() && kind.accepts(&name)) {
            continue;
        }
        if dirs.len() + files.len() >= MAX_ENTRIES {
            truncated = true;
            break;
        }
        let e = Entry {
            path: display(&item.path()),
            dir: is_dir,
            size: (!is_dir).then_some(meta.len()),
            size_text: if is_dir {
                String::new()
            } else {
                human_size(meta.len())
            },
            name,
        };
        if is_dir { &mut dirs } else { &mut files }.push(e);
    }
    let key = |e: &Entry| e.name.to_lowercase();
    dirs.sort_by_key(key);
    files.sort_by_key(key);
    dirs.extend(files);
    Ok(Listing {
        path: display(&dir),
        parent: dir.parent().map(display),
        selected,
        roots: root_views(&roots(cfg)),
        entries: dirs,
        truncated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (PathBuf, Config) {
        let root = std::env::temp_dir()
            .join(format!("bop-browse-{}", uuid::Uuid::new_v4()))
            .canonicalize_or_self();
        std::fs::create_dir_all(root.join("models/Zdir")).unwrap();
        std::fs::create_dir_all(root.join("models/adir")).unwrap();
        for f in [
            "models/b.ONNX",
            "models/a.xml",
            "models/c.yaml",
            "models/d.yml",
            "models/e.txt",
            "models/.hidden.onnx",
        ] {
            std::fs::write(root.join(f), b"x").unwrap();
        }
        let cfg = Config {
            download_dir: Some(root.clone()),
            ..Default::default()
        };
        (root, cfg)
    }

    trait CanonOrSelf {
        fn canonicalize_or_self(self) -> PathBuf;
    }
    impl CanonOrSelf for PathBuf {
        fn canonicalize_or_self(self) -> PathBuf {
            std::fs::create_dir_all(&self).ok();
            self.canonicalize().unwrap_or(self)
        }
    }

    fn names(l: &Listing) -> Vec<&str> {
        l.entries.iter().map(|e| e.name.as_str()).collect()
    }

    #[test]
    fn default_start_is_models_dir() {
        let (root, cfg) = setup();
        let l = list(&cfg, None, None).unwrap();
        assert_eq!(l.path, display(&root.join("models")));
        assert!(l.selected.is_none());
        let l2 = list(&cfg, Some("  "), None).unwrap();
        assert_eq!(l2.path, l.path);
        assert!(l.roots.iter().any(|r| r.label == "Data root"));
    }

    #[test]
    fn default_falls_back_to_data_root() {
        let (root, cfg) = setup();
        std::fs::remove_dir_all(root.join("models")).unwrap();
        assert_eq!(list(&cfg, None, None).unwrap().path, display(&root));
    }

    #[test]
    fn filters_sorts_and_hides_dotfiles() {
        let (_root, cfg) = setup();
        let l = list(&cfg, None, Some("model")).unwrap();
        assert_eq!(names(&l), ["adir", "Zdir", "a.xml", "b.ONNX"]);
        assert_eq!(l.entries[2].size, Some(1));
        assert_eq!(l.entries[2].size_text, "1 B");
        assert!(l.entries[0].dir && l.entries[0].size.is_none());
        let c = list(&cfg, None, Some("classes")).unwrap();
        assert_eq!(names(&c), ["adir", "Zdir", "c.yaml", "d.yml"]);
        // Videos only for the import picker.
        for f in ["clip.MP4", "b.mkv", "c.avi", "d.mov", "e.bvr"] {
            std::fs::write(_root.join("models").join(f), b"x").unwrap();
        }
        let v = list(&cfg, None, Some("video")).unwrap();
        assert_eq!(
            names(&v),
            ["adir", "Zdir", "b.mkv", "c.avi", "clip.MP4", "d.mov"]
        );
        let m = list(&cfg, None, Some("model")).unwrap();
        assert!(!names(&m).contains(&"clip.MP4"), "not in the model picker");
        assert!(!l.truncated);
    }

    #[test]
    fn folder_kind_lists_dirs_only() {
        let (_root, cfg) = setup();
        let l = list(&cfg, None, Some("folder")).unwrap();
        assert_eq!(names(&l), ["adir", "Zdir"]);
        assert!(l.entries.iter().all(|e| e.dir));
    }

    #[test]
    fn file_path_lists_parent_and_selects() {
        let (root, cfg) = setup();
        let f = root.join("models/a.xml");
        // `display` strips the Windows verbatim prefix the canonical temp dir carries (the UI
        // never sends one; the syntax check refuses it).
        let l = list(&cfg, Some(&format!("\"{}\"", display(&f))), None).unwrap();
        assert_eq!(l.path, display(&root.join("models")));
        assert_eq!(l.selected.as_deref(), Some("a.xml"));
    }

    #[test]
    fn relative_path_is_under_data_root() {
        let (root, cfg) = setup();
        let l = list(&cfg, Some("models/adir"), None).unwrap();
        assert_eq!(l.path, display(&root.join("models/adir")));
        assert_eq!(l.parent, Some(display(&root.join("models"))));
    }

    #[test]
    fn errors() {
        let (_root, cfg) = setup();
        assert_eq!(
            list(&cfg, Some("nope/zzz"), None).unwrap_err().status,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            list(&cfg, None, Some("bogus")).unwrap_err().status,
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn data_root_lists_with_its_parent_hidden_from_browsing() {
        let (root, cfg) = setup();
        let l = list(&cfg, Some(&display(&root)), None).unwrap();
        assert_eq!(l.path, display(&root));
        assert!(l.roots.iter().all(|r| r.label != "/"), "{:?}", l.roots);
        // The parent of a root is outside every root (not on Windows, where every drive is
        // a root of its own).
        #[cfg(unix)]
        {
            let parent = root.parent().unwrap();
            let e = list(&cfg, Some(&display(parent)), None).unwrap_err();
            assert_eq!(e.status, StatusCode::FORBIDDEN);
            assert!(
                e.message.contains("outside the browsable folders"),
                "{}",
                e.message
            );
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn outside_paths_are_refused() {
        let (root, cfg) = setup();
        // On Windows every drive is a browsable root, so nothing on a drive is "outside".
        #[cfg(unix)]
        {
            let top = root.ancestors().last().unwrap().to_path_buf();
            for p in [
                top.to_string_lossy().to_string(),
                "/etc".into(),
                "/no/such/dir".into(),
            ] {
                let e = list(&cfg, Some(&p), None).unwrap_err();
                assert_eq!(e.status, StatusCode::FORBIDDEN, "{p}: {}", e.message);
                // The resolved absolute path of the server is not in the message.
                assert!(!e.message.contains("not found"), "{p}: {}", e.message);
            }
        }
        // Missing inside a root: 404, naming only what was asked for.
        let e = list(&cfg, Some("models/zzz"), None).unwrap_err();
        assert_eq!(e.status, StatusCode::NOT_FOUND);
        assert_eq!(e.message, "not found: models/zzz");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escaping_a_root_is_refused() {
        let (root, cfg) = setup();
        let other = std::env::temp_dir().join(format!("bop-browse-out-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("m.onnx"), b"x").unwrap();
        std::os::unix::fs::symlink(&other, root.join("models/link")).unwrap();
        let e = list(&cfg, Some("models/link"), None).unwrap_err();
        assert_eq!(e.status, StatusCode::FORBIDDEN, "{}", e.message);
        let e = allowed_path(&cfg, "models/link/m.onnx").unwrap_err();
        assert_eq!(e.status, StatusCode::FORBIDDEN);
        // A symlink inside the root is fine.
        std::os::unix::fs::symlink(root.join("models/adir"), root.join("models/inner")).unwrap();
        assert!(list(&cfg, Some("models/inner"), None).is_ok());
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&other);
    }

    #[test]
    fn bad_path_syntax_is_rejected() {
        let (root, cfg) = setup();
        for (p, needle) in [
            (r"\\?\C:\x".to_string(), "verbatim"),
            (r"\\server\share".to_string(), "UNC"),
            ("C:models".to_string(), "drive-relative"),
            ("models/\u{0}".to_string(), "NUL"),
            ("x".repeat(300), "component"),
            (format!("models/{}", "a/".repeat(2100)), "longer than"),
            ("models/../etc".to_string(), "'..'"),
        ] {
            let e = list(&cfg, Some(&p), None).unwrap_err();
            assert_eq!(e.status, StatusCode::BAD_REQUEST, "{needle}: {}", e.message);
            assert!(e.message.contains(needle), "{needle}: {}", e.message);
            let e = allowed_path(&cfg, &p).unwrap_err();
            assert_eq!(e.status, StatusCode::BAD_REQUEST, "{needle}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(windows)]
    #[test]
    fn inside_any_ignores_the_verbatim_prefix() {
        let cfg = Config {
            download_dir: Some(PathBuf::from(r"C:\data")),
            ..Default::default()
        };
        let roots = vec![("r".to_string(), PathBuf::from(r"\\?\C:\root"))];
        assert!(inside_any(Path::new(r"C:\root\x"), &roots, &cfg));
        assert!(inside_any(Path::new(r"\\?\C:\root\x"), &roots, &cfg));
        assert!(inside_any(Path::new(r"\\?\C:\data\m"), &roots, &cfg));
        assert!(!inside_any(Path::new(r"C:\rootx"), &roots, &cfg));
        assert!(!inside_any(Path::new(r"D:\root\x"), &roots, &cfg));
    }

    /// A folder outside every root (another volume, in real use) is refused until the native
    /// dialog returns it; then it and its children are accepted by the forms' check, but the
    /// page still cannot list it, and a symlink out of it is still refused.
    #[cfg(unix)]
    #[test]
    fn natively_picked_paths_are_trusted_by_the_forms_only() {
        let (root, cfg) = setup();
        let other = std::env::temp_dir().join(format!("bop-browse-pick-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(other.join("clips")).unwrap();
        std::fs::write(other.join("clips/m.onnx"), b"x").unwrap();
        let sibling = PathBuf::from(format!("{}-2", other.display()));
        std::fs::create_dir_all(&sibling).unwrap();
        let outside = std::env::temp_dir().join(format!("bop-browse-esc-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, other.join("escape")).unwrap();
        let s = |p: &Path| p.to_string_lossy().into_owned();

        let e = allowed_path(&cfg, &s(&other)).unwrap_err();
        assert_eq!(e.status, StatusCode::FORBIDDEN, "{}", e.message);
        let e = allowed_path(&cfg, &s(&other.join("nope"))).unwrap_err();
        assert_eq!(e.status, StatusCode::FORBIDDEN, "no probing before a pick");

        super::super::pick::record_pick(&s(&other));
        let canon = other.canonicalize().unwrap();
        assert_eq!(allowed_path(&cfg, &s(&other)).unwrap(), canon);
        assert_eq!(
            allowed_path(&cfg, &format!("\"{}\"", s(&other.join("clips/m.onnx")))).unwrap(),
            canon.join("clips/m.onnx")
        );
        assert_eq!(
            allowed_path(&cfg, &s(&canon.join("clips"))).unwrap(),
            canon.join("clips")
        );
        assert_eq!(
            allowed_path(&cfg, &s(&other.join("nope")))
                .unwrap_err()
                .status,
            StatusCode::NOT_FOUND
        );
        for p in [
            sibling.clone(),
            other.parent().unwrap().to_path_buf(),
            other.join("escape"),
        ] {
            let e = allowed_path(&cfg, &s(&p)).unwrap_err();
            assert_eq!(
                e.status,
                StatusCode::FORBIDDEN,
                "{}: {}",
                p.display(),
                e.message
            );
        }
        // The page's listing keeps the roots rule.
        let e = list(&cfg, Some(&s(&other)), Some("folder")).unwrap_err();
        assert_eq!(e.status, StatusCode::FORBIDDEN);
        for d in [&root, &other, &sibling, &outside] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    #[test]
    fn unc_rest_takes_shares_only() {
        assert_eq!(unc_rest(r"\\nas\clips"), Some(r"nas\clips"));
        assert_eq!(unc_rest("//nas/clips"), Some("nas/clips"));
        for raw in [
            r"\\?\C:\x",
            r"\\.\pipe\x",
            "//?/C:/x",
            r"\\\x",
            "/x",
            r"C:\x",
            r"\\",
        ] {
            assert_eq!(unc_rest(raw), None, "{raw}");
        }
        // An unpicked share is still refused by the syntax check.
        let (root, cfg) = setup();
        let e = allowed_path(&cfg, r"\\nas-never-picked\share\m.onnx").unwrap_err();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        assert!(e.message.contains("UNC"), "{}", e.message);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A missing file is accepted when it would land inside a root (a model not downloaded
    /// yet), never outside one, nor through a symlink that leaves the roots.
    #[test]
    fn missing_paths_inside_roots_are_accepted_for_saving() {
        let (root, cfg) = setup();
        let ok = |raw: &str| allowed_or_missing_inside_roots(&cfg, raw);
        assert_eq!(ok("models/a.xml").unwrap(), root.join("models/a.xml"));
        assert_eq!(
            ok("models/later.onnx").unwrap(),
            root.join("models/later.onnx")
        );
        assert_eq!(
            ok("not-yet/sub/later.onnx").unwrap(),
            root.join("not-yet/sub/later.onnx")
        );
        let other = std::env::temp_dir()
            .join(format!("bop-browse-miss-{}", uuid::Uuid::new_v4()))
            .canonicalize_or_self();
        std::fs::create_dir_all(&other).unwrap();
        let in_home = ["HOME", "USERPROFILE"].iter().any(|v| {
            std::env::var_os(v)
                .and_then(|h| PathBuf::from(h).canonicalize().ok())
                .is_some_and(|h| other.starts_with(h))
        });
        if !in_home {
            let e = ok(&other.join("m.onnx").to_string_lossy()).unwrap_err();
            assert_eq!(e.status, StatusCode::FORBIDDEN, "{}", e.message);
            #[cfg(unix)]
            {
                std::os::unix::fs::symlink(&other, root.join("models/out")).unwrap();
                let e = ok("models/out/new.onnx").unwrap_err();
                assert_eq!(e.status, StatusCode::FORBIDDEN, "{}", e.message);
            }
        }
        let e = ok("models/../../x.onnx").unwrap_err();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&other);
    }

    #[test]
    fn allowed_path_canonicalizes_inside_roots() {
        let (root, cfg) = setup();
        assert_eq!(
            allowed_path(&cfg, " \"models/a.xml\" ").unwrap(),
            root.join("models/a.xml")
        );
        assert_eq!(
            allowed_path(&cfg, &display(&root.join("models"))).unwrap(),
            root.join("models")
        );
        assert_eq!(
            allowed_path(&cfg, "models/nope.onnx").unwrap_err().status,
            StatusCode::NOT_FOUND
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
