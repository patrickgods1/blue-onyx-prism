//! `GET /v1/fs/browse`: a read-only, server-side file browser for the Config page's
//! "Add a model by path" form. Lists folders plus model (`.onnx`/`.xml`) or classes
//! (`.yaml`/`.yml`) files only, so the unauthenticated page discloses as little as possible.

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
}

impl Kind {
    fn parse(s: Option<&str>) -> Option<Kind> {
        match s.map(str::trim) {
            None | Some("") | Some("model") => Some(Kind::Model),
            Some("classes") => Some(Kind::Classes),
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

impl IntoResponse for BrowseError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({ "success": false, "message": self.message })),
        )
            .into_response()
    }
}

pub(super) async fn browse(
    State(state): State<Arc<AppState>>,
    Query(q): Query<BrowseQuery>,
) -> Response {
    let cfg = state.config_read().clone();
    match list(&cfg, q.path.as_deref(), q.kind.as_deref()) {
        Ok(l) => Json(l).into_response(),
        Err(e) => e.into_response(),
    }
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
    format!("{v:.1} {}", UNITS[i])
}

fn roots(cfg: &Config) -> Vec<Root> {
    let mut out: Vec<Root> = Vec::new();
    let mut add = |label: &str, p: PathBuf| {
        if let Ok(c) = p.canonicalize() {
            let path = display(&c);
            if !out.iter().any(|r| r.path == path) {
                out.push(Root {
                    label: label.to_string(),
                    path,
                });
            }
        }
    };
    add("Data root", cfg.data_root());
    let models = cfg.data_path(&cfg.models_dir);
    if models.is_dir() {
        add("Models", models);
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
    #[cfg(not(windows))]
    {
        add("/", PathBuf::from("/"));
        #[cfg(target_os = "macos")]
        if Path::new("/Volumes").is_dir() {
            add("/Volumes", PathBuf::from("/Volumes"));
        }
    }
    out
}

fn io_error(path: &Path, e: &std::io::Error) -> BrowseError {
    match e.kind() {
        std::io::ErrorKind::NotFound => BrowseError {
            status: StatusCode::NOT_FOUND,
            message: format!("not found: {}", path.display()),
        },
        std::io::ErrorKind::PermissionDenied => BrowseError {
            status: StatusCode::FORBIDDEN,
            message: format!("permission denied: {}", path.display()),
        },
        _ => BrowseError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("cannot read {}: {e}", path.display()),
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
        message: "unknown kind (use model or classes)".to_string(),
    })?;
    let raw = path
        .map(|p| p.trim().trim_matches(|c| c == '"' || c == '\'').trim())
        .unwrap_or("");
    let start = if raw.is_empty() {
        let m = cfg.data_path(&cfg.models_dir);
        if m.is_dir() { m } else { cfg.data_root() }
    } else {
        cfg.data_path(Path::new(raw))
    };
    let canon = start.canonicalize().map_err(|e| io_error(&start, &e))?;
    let (dir, selected) = if canon.is_dir() {
        (canon, None)
    } else {
        let sel = canon.file_name().map(|n| n.to_string_lossy().into_owned());
        match canon.parent() {
            Some(p) => (p.to_path_buf(), sel),
            None => (canon, None),
        }
    };
    let rd = std::fs::read_dir(&dir).map_err(|e| io_error(&dir, &e))?;
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
        roots: roots(cfg),
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
        assert!(!l.truncated);
    }

    #[test]
    fn file_path_lists_parent_and_selects() {
        let (root, cfg) = setup();
        let f = root.join("models/a.xml");
        let l = list(&cfg, Some(&format!("\"{}\"", f.display())), None).unwrap();
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
    fn filesystem_root_has_no_parent() {
        let (_root, cfg) = setup();
        let top = std::env::current_dir()
            .unwrap()
            .ancestors()
            .last()
            .unwrap()
            .to_path_buf();
        let l = list(&cfg, Some(&top.to_string_lossy()), None).unwrap();
        assert!(l.parent.is_none());
    }
}
