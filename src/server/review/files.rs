//! The Review tab's files and settings: `POST /v1/review/export` (YOLO / COCO / X-AnyLabeling /
//! benchmark set), `POST /v1/review/import` (X-AnyLabeling files), `POST /v1/review/folder` (the
//! folder under review, `collect.review_dir`), `POST /v1/review/classes` and
//! `GET /v1/review/class-catalog`, with the checks of a folder a client names
//! ([`allowed_folder`]).

use super::{
    Answer, AppState, blocking, context, fail, folder_problem, folder_unavailable, hidden,
    json_body, now, success,
};
use crate::collect::CollectSettings;
use crate::collect::labels::export::{
    self, BenchSplit, ClassFilter, ExportRequest, Format, SplitConfig,
};
use crate::collect::labels::{self, bulk, xany};
use crate::server::{browse, save_config};
use axum::body::Bytes;
use axum::extract::State;
use axum::extract::rejection::BytesRejection;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;

/// The folder of the page's default exports, `<UTC time>` under it unless `out_dir` is given: where the Train tab looks for them
/// ([`crate::training::discover_exports`]), so a custom `training.dir` moves it too.
pub(super) fn exports_dir(state: &AppState) -> std::path::PathBuf {
    crate::training::training_dir(&state.config_read()).join("exports")
}

/// The dataset folder and every folder a finished import wrote to (newest first), once each.
pub(super) fn known_dirs(state: &AppState, dataset: &std::path::Path) -> Vec<String> {
    let mut v = vec![dataset.display().to_string()];
    for t in crate::server::training::tasks(state).iter().rev() {
        if let Some(d) = t
            .result
            .as_ref()
            .and_then(|r| r.get("dataset_dir"))
            .and_then(|d| d.as_str())
            && !v.iter().any(|k| k == d)
        {
            v.push(d.to_string());
        }
    }
    v
}

/// [`browse::allowed_path`] for a folder of frames, also accepting a folder this server
/// already works on: one a finished import wrote to ([`known_dirs`]) or the folder under review.
/// Both passed that check when chosen, often as a native pick, which it trusts for an hour only;
/// going back to one later must not need another pick. Exact folders only, canonicalized.
pub(in crate::server) fn allowed_folder(
    state: &AppState,
    config: &crate::config::Config,
    raw: &str,
) -> Result<PathBuf, (StatusCode, String)> {
    let err = match browse::allowed_path(config, raw) {
        Ok(p) => return Ok(p),
        Err(e) => e.into_parts(),
    };
    if err.0 != StatusCode::FORBIDDEN {
        return Err(err);
    }
    let trimmed = raw.trim().trim_matches(|c| c == '"' || c == '\'').trim();
    let Ok(canon) = config
        .data_path(std::path::Path::new(trimmed))
        .canonicalize()
    else {
        return Err(err);
    };
    let canon = browse::plain(&canon);
    let dataset = CollectSettings::for_config(config).dir;
    let mut trusted: Vec<PathBuf> = known_dirs(state, &dataset)
        .into_iter()
        .map(PathBuf::from)
        .collect();
    trusted.push(config.collect.resolved_review_dir(&config.data_root()));
    let known = trusted
        .iter()
        .filter_map(|k| k.canonicalize().ok())
        .any(|k| browse::plain(&k) == canon);
    if known { Ok(canon) } else { Err(err) }
}

/// One format or a list (`"yolo"`, `["yolo", "coco"]`, `"all"`).
#[derive(Deserialize)]
#[serde(untagged)]
enum Formats {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportBody {
    #[serde(alias = "format")]
    formats: Option<Formats>,
    out_dir: Option<String>,
    #[serde(default)]
    holdout_cameras: Vec<String>,
    #[serde(default)]
    holdout_days: Vec<String>,
    holdout_fraction: Option<f32>,
    #[serde(default)]
    benchmark_split: BenchSplit,
    #[serde(default)]
    include_unreviewed: bool,
    /// Only frames with a training box of these classes; empty = every class.
    #[serde(default)]
    labels: Vec<String>,
    /// Also frames confirmed empty.
    #[serde(default = "default_true")]
    include_empty: bool,
}

fn default_true() -> bool {
    true
}

/// The export's class filter: names trimmed, blanks and repeats dropped, every one in
/// `classes` (else the message for a 400).
fn class_filter(
    labels: Vec<String>,
    include_empty: bool,
    classes: &[String],
) -> Result<ClassFilter, String> {
    let mut out: Vec<String> = Vec::new();
    for l in labels.iter().map(|l| l.trim()).filter(|l| !l.is_empty()) {
        if !out.iter().any(|o| o == l) {
            out.push(l.to_string());
        }
    }
    let f = ClassFilter {
        labels: out,
        include_empty,
    };
    f.check(classes)
        .map_err(|e| format!("Not exported: {e:#}"))?;
    Ok(f)
}

fn parse_formats(f: Option<Formats>) -> Result<Vec<Format>, String> {
    let list = match f {
        None => return Ok(Format::ALL.to_vec()),
        Some(Formats::One(s)) => vec![s],
        Some(Formats::Many(v)) => v,
    };
    let mut out = Vec::new();
    for s in list {
        if s.trim().eq_ignore_ascii_case("all") {
            return Ok(Format::ALL.to_vec());
        }
        let f = Format::parse(&s).ok_or_else(|| {
            format!("format: '{s}' is not one of yolo, coco, xanylabeling, benchmark, all")
        })?;
        if !out.contains(&f) {
            out.push(f);
        }
    }
    if out.is_empty() {
        return Err("format: give at least one of yolo, coco, xanylabeling, benchmark".into());
    }
    Ok(out)
}

/// A folder a client named for this server to read or write: well formed and no system folder
/// ([`crate::config::validate_dir_setting`], 400), then [`allowed_folder`]. `Ok(None)` when
/// `allowed_folder` found nothing there.
fn named_folder(
    state: &AppState,
    key: &str,
    raw: &str,
    root: &std::path::Path,
) -> Result<Option<PathBuf>, (StatusCode, String)> {
    match crate::config::validate_dir_setting(key, raw, root) {
        Ok(Some(_)) => {}
        Ok(None) => return Err((StatusCode::BAD_REQUEST, format!("{key}: empty"))),
        Err(e) => return Err((StatusCode::BAD_REQUEST, format!("{e:#}"))),
    }
    let config = state.config_read().clone();
    match allowed_folder(state, &config, raw) {
        Ok(p) => Ok(Some(p)),
        Err((StatusCode::NOT_FOUND, _)) => Ok(None),
        Err(e) => Err(folder_refusal(key, e)),
    }
}

/// `refusal` of a folder under the setting's name, with what to do about it.
fn folder_refusal(key: &str, (status, message): (StatusCode, String)) -> (StatusCode, String) {
    let message = format!("{key}: {message}; choose the folder with the Browse button instead");
    (status, message)
}

/// Where a client-named export goes ([`named_folder`]). The export creates it, so a folder that
/// does not exist yet is fine inside the browsable folders or a native pick.
fn export_folder(
    state: &AppState,
    raw: &str,
    root: &std::path::Path,
) -> Result<PathBuf, (StatusCode, String)> {
    if let Some(p) = named_folder(state, "out_dir", raw, root)? {
        return Ok(p);
    }
    let config = state.config_read().clone();
    browse::allowed_or_missing_inside_roots(&config, raw)
        .map_err(|e| folder_refusal("out_dir", e.into_parts()))
}

/// `POST /v1/review/export` with `{format(s), out_dir?, holdout_cameras?, holdout_days?,
/// holdout_fraction?, benchmark_split?, include_unreviewed?, labels?, include_empty?}`.
pub(in crate::server) async fn export(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    let b: ExportBody = json_body(&headers, body)?;
    let formats = parse_formats(b.formats).map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    let fraction = b
        .holdout_fraction
        .unwrap_or(export::DEFAULT_HOLDOUT_FRACTION);
    if !(fraction.is_finite() && (0.0..=1.0).contains(&fraction)) {
        return Ok(fail(
            StatusCode::BAD_REQUEST,
            format!("holdout_fraction: must be between 0 and 1, got {fraction}"),
        ));
    }
    Ok(blocking(move || {
        let (dir, classes, root) = context(&state);
        if let Some(message) = folder_problem(&state, &dir) {
            return folder_unavailable(message);
        }
        let filter = match class_filter(b.labels, b.include_empty, &classes) {
            Ok(f) => f,
            Err(e) => return fail(StatusCode::BAD_REQUEST, e),
        };
        let out_dir = match b.out_dir.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            None => {
                exports_dir(&state).join(crate::collect::utc_file_ts(std::time::SystemTime::now()))
            }
            Some(raw) => match export_folder(&state, raw, &root) {
                Ok(p) => p,
                Err((s, m)) => return fail(s, m),
            },
        };
        let req = ExportRequest {
            formats,
            out_dir,
            split: SplitConfig {
                holdout_cameras: b.holdout_cameras,
                holdout_days: b.holdout_days,
                holdout_fraction: fraction,
            },
            benchmark_split: b.benchmark_split,
            include_unreviewed: b.include_unreviewed,
            hidden_classes: hidden(&state),
        };
        match export::run_filtered(&dir, &classes, &req, &filter) {
            Ok(r) => {
                tracing::info!(out = %r.out_dir.display(), frames = r.frames, "review export written");
                success(serde_json::json!({ "report": r }))
            }
            Err(e) => fail(StatusCode::UNPROCESSABLE_ENTITY, format!("Not exported: {e:#}")),
        }
    })
    .await)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImportBody {
    dir: String,
    #[serde(default)]
    treat_unchecked_as_reviewed: bool,
}

/// `POST /v1/review/import` with `{dir, treat_unchecked_as_reviewed?}`: X-AnyLabeling files of
/// exported frames back into their label files.
pub(in crate::server) async fn import(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    let b: ImportBody = json_body(&headers, body)?;
    Ok(blocking(move || {
        let (dir, classes, root) = context(&state);
        if let Some(message) = folder_problem(&state, &dir) {
            return folder_unavailable(message);
        }
        let src = match named_folder(&state, "dir", &b.dir, &root) {
            Ok(Some(p)) if p.is_dir() => p,
            Ok(Some(_)) => {
                return fail(
                    StatusCode::BAD_REQUEST,
                    format!("dir: '{}' is not a folder", b.dir),
                );
            }
            Ok(None) => {
                return fail(
                    StatusCode::NOT_FOUND,
                    format!("dir: '{}' does not exist; choose the folder again", b.dir),
                );
            }
            Err((s, m)) => return fail(s, m),
        };
        let hidden = hidden(&state);
        match xany::import_dir(
            &dir,
            &src,
            &classes,
            &hidden,
            b.treat_unchecked_as_reviewed,
            &now(),
        ) {
            Ok(r) => success(serde_json::json!({ "report": r })),
            Err(e) => fail(
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("Not imported: {e:#}"),
            ),
        }
    })
    .await)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FolderBody {
    dir: String,
}

/// `POST /v1/review/folder` with `{dir}`: the folder this page, its exports and the auto-labeler
/// work on (`collect.review_dir`), for frames an import saved outside the dataset folder. Empty or
/// the dataset folder itself = back to the dataset folder. Like the import's paths, it must be a
/// browsable folder, one picked in the native dialog or one already in use ([`allowed_folder`]).
pub(in crate::server) async fn folder(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    let b: FolderBody = json_body(&headers, body)?;
    Ok(blocking(move || {
        let config = state.config_read().clone();
        let dataset = CollectSettings::for_config(&config).dir;
        let raw = b.dir.trim();
        let chosen = if raw.is_empty() {
            None
        } else {
            let p = match allowed_folder(&state, &config, raw) {
                Ok(p) => p,
                Err((s, m)) => return fail(s, format!("Folder not changed: {m}")),
            };
            if !p.is_dir() {
                return fail(
                    StatusCode::BAD_REQUEST,
                    format!("Folder not changed: {raw} is not a folder"),
                );
            }
            let same = dataset.canonicalize().is_ok_and(|d| browse::plain(&d) == p);
            (!same).then_some(p)
        };
        let result = save_config(&state, "review page", |c| {
            crate::collect::config::set_review_dir(c, chosen.clone())
        });
        match result {
            Ok(msg) => success(serde_json::json!({
                "message": msg,
                "dir": chosen.as_ref().unwrap_or(&dataset),
            })),
            Err(e) => fail(
                StatusCode::BAD_REQUEST,
                format!("Folder not changed: {e:#}"),
            ),
        }
    })
    .await)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClassesBody {
    classes: Vec<String>,
    /// `collect.hidden_classes`; absent = unchanged (pruned to the names still listed).
    #[serde(default)]
    hidden: Option<Vec<String>>,
}

/// The classes `old` wanted (its list without its hidden ones) that `new` does not: removed
/// from the list or newly hidden, in the old list's order.
fn newly_unwanted(old: &bulk::Wanted<'_>, new: &bulk::Wanted<'_>) -> Vec<String> {
    old.classes
        .iter()
        .filter(|c| old.class_of(c).is_some() && new.class_of(c).is_none())
        .cloned()
        .collect()
}

/// `POST /v1/review/classes` with `{classes: [...], hidden?: [...]}`: saves `collect.classes`
/// and, when given, `collect.hidden_classes` (each must be listed; "don't want": hidden from
/// Review and their pending boxes rejected); hidden names no longer listed are dropped. Answers
/// both. When the save makes classes unwanted (removed or newly hidden: `newly_unwanted`), the
/// pending model and auto boxes of unwanted classes on the unreviewed frames of the folder under
/// review are rejected right away (`auto_rejected: {frames, boxes, relabelled, failed}`,
/// [`bulk::bulk_reject_unwanted`]) and `reviewed_with_unwanted: {frames, boxes, classes}` counts
/// the reviewed frames still holding accepted boxes of unwanted classes (the page asks before
/// `POST /v1/review/reject-unwanted-reviewed`); both zero otherwise.
pub(in crate::server) async fn classes(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    let b: ClassesBody = json_body(&headers, body)?;
    Ok(blocking(move || {
        let classes: Vec<String> = b
            .classes
            .into_iter()
            .map(|c| c.trim().to_string())
            .collect();
        let hidden_req: Option<Vec<String>> = b
            .hidden
            .map(|h| h.into_iter().map(|c| c.trim().to_string()).collect());
        let mut hidden = Vec::new();
        let (mut old_classes, mut old_hidden) = (Vec::new(), Vec::new());
        let mut dir = PathBuf::new();
        let result = save_config(&state, "review page", |c| {
            old_classes = c.collect.classes.clone();
            old_hidden = c.collect.hidden_classes.clone();
            c.collect.classes = classes.clone();
            match &hidden_req {
                Some(h) => c.collect.hidden_classes = h.clone(),
                None => c.collect.prune_hidden(),
            }
            c.collect.validate(&c.data_root())?;
            hidden = c.collect.hidden_classes.clone();
            dir = c.collect.resolved_review_dir(&c.data_root());
            Ok(())
        });
        let msg = match result {
            Ok(msg) => msg,
            Err(e) => return fail(StatusCode::BAD_REQUEST, format!("Not saved: {e:#}")),
        };
        let wanted = bulk::Wanted {
            classes: &classes,
            hidden: &hidden,
        };
        let old = bulk::Wanted {
            classes: &old_classes,
            hidden: &old_hidden,
        };
        let gone = newly_unwanted(&old, &wanted);
        let (rejected, reviewed) = if gone.is_empty() {
            (
                bulk::BulkOutcome::default(),
                bulk::ReviewedUnwanted::default(),
            )
        } else {
            let o = bulk::bulk_reject_unwanted(
                &dir,
                &wanted,
                bulk::Scope::Unreviewed,
                usize::MAX,
                false,
            );
            tracing::info!(
                classes = %gone.join(", "),
                boxes = o.boxes,
                frames = o.done_count,
                failed = o.failed_count,
                "review: classes no longer wanted; their pending boxes rejected"
            );
            (o, bulk::reviewed_with_unwanted(&dir, &wanted))
        };
        success(serde_json::json!({
            "message": msg,
            "classes": classes,
            "hidden": hidden,
            "classes_version": labels::classes_version(&classes),
            "newly_unwanted": gone,
            "auto_rejected": {
                "frames": rejected.done_count,
                "boxes": rejected.boxes,
                "relabelled": rejected.relabelled,
                "failed": rejected.failed,
            },
            "reviewed_with_unwanted": reviewed,
        }))
    })
    .await)
}

/// `GET /v1/review/class-catalog`: the class list, the hidden classes, the classes the enabled
/// models report, the models' classes and every class worth offering
/// ([`crate::training::autolabel::catalog::ClassCatalog`]; reads the models' class files).
pub(in crate::server) async fn class_catalog(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || {
        let config = state.config_read().clone();
        let cat = crate::training::autolabel::catalog::ClassCatalog::for_config(&config);
        match serde_json::to_value(&cat) {
            Ok(v) => success(v),
            Err(e) => fail(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
        }
    })
    .await
}
