//! The Review tab's files and settings: `POST /v1/review/export` (YOLO / COCO / X-AnyLabeling /
//! benchmark set), `POST /v1/review/import` (X-AnyLabeling files), `POST /v1/review/folder` (the
//! folder under review, `collect.review_dir`), `POST /v1/review/rescan` (read that folder again
//! and name the images no frame of the layout names, [`rescan`]), `POST /v1/review/classes` and
//! `GET /v1/review/class-catalog`, with the checks of a folder a client names
//! ([`allowed_folder`]).
//!
//! An export and an import run inside their request and answer when done; while they work they
//! are side jobs ([`crate::server::side_jobs`]) with their progress, so the Jobs tab lists them,
//! also after the browser left the page (the work goes on).
//!
//! Stopping (audit 2026-11 phase 1): both check the run's stop token (`AppState.shutdown`)
//! between two frames and answer 503 with what happened (an export keeps nothing of itself, an
//! import keeps the frames it did). An export cut off by a kill or a crash instead leaves its
//! scratch folder, which it records first in the training folder's
//! [`crate::storage::sweep::EXPORTS_PENDING_DIR`], so the next storage sweep removes it. While they work they hold only what they need, never the
//! `AppState`: a long export must not keep the stop's last drops (the provisioner, the log
//! guard) from happening.

use super::{
    Answer, AppState, blocking, context, fail, folder_problem, folder_unavailable, hidden,
    json_body, now, success,
};
use crate::collect::CollectSettings;
use crate::collect::labels::export::{
    self, BenchSplit, ClassFilter, ExportRequest, Format, SplitConfig,
};
use crate::collect::labels::heldout::HeldoutPolicy;
use crate::collect::labels::{self, bulk, xany};
use crate::server::side_jobs::{self, SideKind};
use crate::server::{browse, save_config};
use crate::sync::LockExt as _;
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

/// The Jobs tab's title of an export: its formats and where it goes.
fn export_title(req: &ExportRequest) -> String {
    let formats: Vec<&str> = req
        .formats
        .iter()
        .map(|f| match f {
            Format::Yolo => "YOLO",
            Format::Coco => "COCO",
            Format::XAnyLabeling => "X-AnyLabeling",
            Format::Benchmark => "benchmark set",
        })
        .collect();
    format!("{} to {}", formats.join(", "), req.out_dir.display())
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
                exports_dir(&state).join(crate::clock::frame_stamp(std::time::SystemTime::now()))
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
            size_floor: state.config_read().collect.size_floor,
            // The camera days held out for good: read, and grown by an export of the default
            // split, never with a day a training run has seen.
            heldout: Some(HeldoutPolicy {
                seen: crate::training::seen_days(&state.config_read()),
            }),
        };
        let (side, shutdown) = (state.side_jobs.clone(), state.shutdown.clone());
        // Where the export notes its scratch folder, so a sweep after a crash or a kill finds it
        // in a folder of the user's choice too.
        let pending = crate::training::training_dir(&state.config_read())
            .join(crate::storage::sweep::EXPORTS_PENDING_DIR);
        drop(state);
        let job = side.start(SideKind::DatasetExport, export_title(&req));
        let progress = |s| job.report(side_jobs::export_report(s));
        let stop = || shutdown.is_cancelled();
        let result = export::run_filtered_recorded(
            &dir,
            &classes,
            &req,
            &filter,
            (&progress, &stop),
            Some(&pending),
        );
        drop(job);
        crate::backend::memory::trim_soon("review export built");
        match result {
            Ok(r) => {
                tracing::info!(out = %r.out_dir.display(), frames = r.frames, "review export written");
                success(serde_json::json!({ "report": r }))
            }
            Err(e) if export::is_stopped(&e) => {
                tracing::info!("review export stopped: the service is stopping");
                fail(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Not exported: the service is stopping, so the export was cancelled and \
                     nothing of it was kept. Export again once the service is back."
                        .to_string(),
                )
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
        let (side, shutdown) = (state.side_jobs.clone(), state.shutdown.clone());
        drop(state);
        let job = side.start(SideKind::XanyImport, format!("From {}", src.display()));
        let result = xany::import_dir_until(
            &dir,
            &src,
            &classes,
            &hidden,
            b.treat_unchecked_as_reviewed,
            &now(),
            &|done, of| job.report(side_jobs::import_report(done, of)),
            &|| shutdown.is_cancelled(),
        );
        drop(job);
        match result {
            Ok(r) => success(serde_json::json!({ "report": r })),
            Err(e) if export::is_stopped(&e) => {
                tracing::info!("X-AnyLabeling import stopped: the service is stopping ({e:#})");
                fail(
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!(
                        "Not finished: {e:#}. Import the folder again once the service is back; \
                         the frames already imported come out the same."
                    ),
                )
            }
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

/// The folders being rescanned: a second rescan of one answers 409 while the first runs.
static RESCANNING: std::sync::Mutex<std::collections::BTreeSet<PathBuf>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

/// A rescan of its folder running; the folder leaves [`RESCANNING`] when this goes, panic
/// included.
pub(super) struct Rescanning(PathBuf);

impl Rescanning {
    /// The rescan of `dir`, None while another one of it runs.
    pub(super) fn start(dir: &std::path::Path) -> Option<Self> {
        RESCANNING
            .lock_recover()
            .insert(dir.to_path_buf())
            .then(|| Self(dir.to_path_buf()))
    }
}

impl Drop for Rescanning {
    fn drop(&mut self) {
        RESCANNING.lock_recover().remove(&self.0);
    }
}

/// `POST /v1/review/rescan` (no fields: an empty body or any JSON, such as the page's `{}`): read
/// the folder under review again now, for what changed outside this process. The review index
/// walks everything ([`labels::index::rescan`]; `indexed: false` when the folder has none, and
/// the next Next walks the files anyway), and the caches the Review and Train pages show beside
/// it are dropped: the reviewer's history ([`labels::repeats::invalidate`]), the parked-spot table
/// of the Recommended order ([`super::steer::forget_spots`]), the Train strip's counts
/// ([`crate::server::train_area::forget_tally`]) and the Refine tab's (open suggestions, coverage,
/// size floor, static spots). Then the images in the folder that no frame of the layout names
/// ([`labels::strays::find`]): `strays: {count, sample, unreadable}`. Writes nothing to the folder.
/// 503 with `folder_missing` when the folder is not there.
pub(in crate::server) async fn rescan(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    if !body.as_ref().is_ok_and(Bytes::is_empty) {
        let _: serde_json::Value = json_body(&headers, body)?;
    }
    Ok(blocking(move || {
        let (dir, _, _) = context(&state);
        if let Some(message) = folder_problem(&state, &dir) {
            return folder_unavailable(message);
        }
        let Some(_one) = Rescanning::start(&dir) else {
            return fail(
                StatusCode::CONFLICT,
                "A rescan is already running; its counts show on the page as it goes.",
            );
        };
        let start = std::time::Instant::now();
        let indexed = labels::index::rescan(&dir);
        labels::repeats::invalidate(&dir);
        super::steer::forget_spots(&dir);
        crate::server::train_area::forget_tally(&dir);
        crate::server::refine::forget_counts();
        crate::server::refine_coverage::forget();
        crate::server::refine_floor::forget();
        crate::server::refine_spots::forget();
        let strays = labels::strays::find(&dir);
        tracing::info!(
            "review: rescan of {} asked ({}); {} image(s) no frame of the layout names, {} \
             folder(s) not listable; {} ms",
            dir.display(),
            if indexed {
                "the index reads everything again"
            } else {
                "no index, the next Next walks the files"
            },
            strays.count,
            strays.unreadable,
            start.elapsed().as_millis()
        );
        success(serde_json::json!({
            "dir": dir,
            "indexed": indexed,
            "strays": {
                "count": strays.count,
                "sample": strays.sample,
                "unreadable": strays.unreadable,
            },
        }))
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::collect::labels::{BoxState, LabelBox, LabelFile, Source, Status};
    use crate::server::test_util::{call, test_state_with};
    use axum::body::Body;
    use axum::http::{Request, header};

    async fn post(state: &Arc<AppState>, uri: &str, body: String) -> (StatusCode, String) {
        let req = Request::post(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap();
        let (s, _, b) = call(state, req).await;
        let v: serde_json::Value = serde_json::from_slice(&b).unwrap();
        (s, v["message"].as_str().unwrap_or_default().to_string())
    }

    /// Audit 2026-11 phase 1: an export or import asked for while the service stops ends with a
    /// 503 that says so, and the export leaves nothing behind.
    #[tokio::test]
    async fn an_export_or_import_during_a_stop_answers_503_and_leaves_nothing() {
        let root = std::env::temp_dir().join(format!("bop-files-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let (d, r) = (root.join("dataset"), root.clone());
        let state = test_state_with(move |c| {
            c.download_dir = Some(r);
            c.collect.dir = Some(d);
        });
        let dir = root.join("dataset");
        let person = LabelBox {
            label: "person".into(),
            bbox: [2.0, 2.0, 30.0, 40.0],
            source: Source::Human,
            score: None,
            state: BoxState::Accepted,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: Default::default(),
        };
        let rel = bulk::test_support::put_frame(&dir, "a", None, vec![]);
        let frame = labels::Frame::resolve(&dir, &rel).unwrap();
        let reviewed = LabelFile {
            status: Status::Reviewed,
            boxes: vec![person.clone()],
            ..LabelFile::default()
        };
        labels::write(&frame.labels, &reviewed).unwrap();
        let src = root.join("xany");
        std::fs::create_dir_all(&src).unwrap();
        let name = export::flat_name(&crate::collect::layout::parse_rel(&rel).unwrap());
        let x = xany::to_xany(&name, 64, 48, [&person], true);
        std::fs::write(
            src.join(name.replace(".jpg", ".json")),
            serde_json::to_vec(&x).unwrap(),
        )
        .unwrap();
        let exports = exports_dir(&state);

        state.shutdown.cancel();
        let (s, m) = post(&state, "/v1/review/export", r#"{"format": "yolo"}"#.into()).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{m}");
        assert!(m.contains("the service is stopping"), "{m}");
        let left: Vec<_> = std::fs::read_dir(&exports)
            .map(|rd| rd.flatten().map(|e| e.file_name()).collect())
            .unwrap_or_default();
        assert!(left.is_empty(), "{left:?}");

        let body = serde_json::json!({ "dir": src }).to_string();
        let (s, m) = post(&state, "/v1/review/import", body).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{m}");
        assert!(m.contains("after 0 of 1 files"), "{m}");
        assert!(m.contains("Import the folder again"), "{m}");
        assert_eq!(frame.labels().unwrap().unwrap(), reviewed, "untouched");
        let _ = std::fs::remove_dir_all(&root);
    }
}
