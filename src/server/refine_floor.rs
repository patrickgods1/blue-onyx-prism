//! The Too small card of the Refine tab (admin only; `assets/train-refine.js`): what the size floor
//! (`collect.size_floor`, [`SizeFloor`]) does to the reviewed frames, the floor's and the ignore
//! mode's settings, and the one rule that writes boxes. Four routes:
//!
//! - `GET /v1/refine/floor?px=&input=`: what a floor would do, for the saved floor with no query or
//!   for the values in the query (a preview before saving; both are needed and checked with
//!   [`SizeFloor::validate`], 400 otherwise). `counts` ([`export::floor_counts`] over the reviewed
//!   frames: per class the training boxes, those under the floor and what is left) and `pending`
//!   ([`bulk::bulk_resolve_small`] as a dry run: the pending proposals the rule would hide) are
//!   null / zero when no floor is shown; `restorable` ([`bulk::bulk_restore_small`] as a dry run
//!   with the **saved** floor, or none when off) counts the size floor dismissals a restore gives
//!   back. The walks run off the async workers and are kept 60 s per folder, class list, shown and
//!   saved floor, at most [`KEEP`] answers; a floor save and the rule's apply drop them.
//!   `suggested` is the input size of the last dataset check's recipe, else the newest finished
//!   training run's, else 384, with 10 px (the measured knee: under 8 px such objects were found 6%
//!   of the time at 384).
//! - `POST /v1/refine/size-floor` `{px, input}` or `{off: true}`: saves `collect.size_floor`
//!   through [`save_config`], like the static cap. Accepted and added labels never change for it:
//!   it applies at the next export and in the dataset check; only the rule below writes.
//! - `POST /v1/refine/ignore-mode` `{mode: "background" | "drop-frame"}`: saves `training.ignore`
//!   (what a training frame with ignore regions becomes), read by the next training run.
//! - `POST /v1/refine/resolve-small` `{dry_run?, limit?, restore?}`: [`bulk::bulk_resolve_small`]
//!   with the **saved** floor over the unreviewed frames of the folder under review (the queue and
//!   the index files): pending model and auto-labeler boxes under it are dismissed (`reason:
//!   size_floor`, [`bulk::SIZE_FLOOR_REASON`]: hidden, no ignore region, no person's decision).
//!   Human, moved, unsure and audit boxes, reviewed and skipped frames are never touched, and a frame
//!   keeps its status. With `restore: true` it runs [`bulk::bulk_restore_small`] instead (no saved
//!   floor needed): the dismissals the saved floor (or none, when off) no longer covers go back to
//!   pending. A dry run counts what an apply on the unchanged folder does; at most `limit` frames
//!   (500) are written per call.

use super::refine::{fail, forget_counts, json_body, recall_report};
use super::review::{push_failed, push_past_limit};
use super::{AppState, blocking, save_config, training};
use crate::collect::labels::SizeFloor;
use crate::collect::labels::bulk;
use crate::collect::labels::export::{self, FloorCounts};
use crate::collect::labels::queue::{Filters, QueueSource};
use crate::collect::labels::{MAX_FLOOR_INPUT, MAX_FLOOR_PX, MIN_FLOOR_INPUT, MIN_FLOOR_PX};
use crate::sync::LockExt;
use crate::training::dataset::IgnoreMode;
use crate::training::service::RunState;
use axum::Json;
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Largest request body of the three POST routes.
pub(super) const BODY_LIMIT: usize = 4 * 1024;
/// Frames the rule writes at most per call (also its default).
pub(super) const RESOLVE_LIMIT: usize = 500;
/// How long a walk's answer is kept.
const KEEP_FOR: Duration = Duration::from_secs(60);
/// Most answers kept at once (folders x class lists x floors).
const KEEP: usize = 8;
/// The suggested floor's `px`: under about 10 px at the model's input no model finds an object
/// reliably (`SizeFloor` docs).
const SUGGESTED_PX: f32 = 10.0;
/// The suggested `input` when no check or training run says one.
const SUGGESTED_INPUT: u32 = 384;

/// `floor` as the page reads it: `px` rounded to two decimals (an f32 widened to f64 would print
/// as 7.300000190734863).
fn floor_json(f: SizeFloor) -> serde_json::Value {
    serde_json::json!({
        "px": (f64::from(f.px) * 100.0).round() / 100.0,
        "input": f.input,
    })
}

fn floor_opt(f: Option<SizeFloor>) -> serde_json::Value {
    f.map_or(serde_json::Value::Null, floor_json)
}

// ---------------------------------------------------------------------------------------------
// The walk and its cache

/// What one walk knows about a floor.
struct Computed {
    /// None with no floor shown.
    counts: Option<FloorCounts>,
    /// Pending proposals the rule would dismiss under the shown floor, and the unreviewed frames
    /// they are on.
    pending_boxes: usize,
    pending_frames: usize,
    /// Size floor dismissals the saved floor (or none) no longer covers: what a restore gives back.
    restorable_boxes: usize,
    restorable_frames: usize,
}

/// The filters of the rule: every unreviewed frame of the folder (the queue and the index files),
/// the wanted classes for a frame never opened (its pre-fill).
fn rule_filters(classes: &[String], hidden: &[String]) -> Filters {
    Filters {
        source: QueueSource::All,
        wanted: Some((classes.to_vec(), hidden.to_vec())),
        ..Filters::default()
    }
}

/// The walks of one answer: `shown` (the floor asked for or saved; None = none) and `saved` (what a
/// restore compares with; None = off).
fn compute(
    dir: &Path,
    classes: &[String],
    hidden: &[String],
    shown: Option<SizeFloor>,
    saved: Option<SizeFloor>,
) -> Computed {
    let filters = rule_filters(classes, hidden);
    let (counts, pending) = match shown {
        Some(floor) => (
            Some(export::floor_counts(dir, classes, floor)),
            bulk::bulk_resolve_small(dir, &filters, floor, true, 1),
        ),
        None => (None, bulk::BulkOutcome::default()),
    };
    let back = bulk::bulk_restore_small(dir, &filters, saved, true, 1);
    Computed {
        counts,
        pending_boxes: pending.boxes,
        pending_frames: pending.eligible,
        restorable_boxes: back.boxes,
        restorable_frames: back.eligible,
    }
}

/// A floor as part of a cache key (`-` for none).
fn key_of(f: Option<SizeFloor>) -> String {
    f.map_or_else(|| "-".into(), |f| format!("{}@{}", f.px.to_bits(), f.input))
}

struct Kept {
    at: Instant,
    value: Arc<Computed>,
}

/// The walks' answers per folder, class list and floor (one per process, [`CACHE`]; a test makes
/// its own, so other tests clearing the process's cannot touch it).
struct Cache {
    kept: Mutex<BTreeMap<String, Kept>>,
    /// One walk at a time: a second caller waits and finds the first one's answer.
    walking: Mutex<()>,
}

static CACHE: Cache = Cache::new();

/// Drop every kept answer of the process (a floor was saved or the rule wrote).
fn forget() {
    CACHE.forget();
}

impl Cache {
    const fn new() -> Self {
        Self {
            kept: Mutex::new(BTreeMap::new()),
            walking: Mutex::new(()),
        }
    }

    fn forget(&self) {
        self.kept.lock_recover().clear();
    }

    fn computed(
        &self,
        dir: &Path,
        classes: &[String],
        hidden: &[String],
        shown: Option<SizeFloor>,
        saved: Option<SizeFloor>,
    ) -> Arc<Computed> {
        let key = format!(
            "{}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
            dir.display(),
            classes.join("\u{1e}"),
            hidden.join("\u{1e}"),
            key_of(shown),
            key_of(saved)
        );
        let fresh = |k: &Kept| k.at.elapsed() < KEEP_FOR;
        if let Some(k) = self.kept.lock_recover().get(&key).filter(|k| fresh(k)) {
            return k.value.clone();
        }
        let _walking = self.walking.lock_recover();
        if let Some(k) = self.kept.lock_recover().get(&key).filter(|k| fresh(k)) {
            return k.value.clone();
        }
        let value = Arc::new(compute(dir, classes, hidden, shown, saved));
        let mut map = self.kept.lock_recover();
        map.retain(|k, v| *k != key && v.at.elapsed() < KEEP_FOR);
        while map.len() >= KEEP {
            let Some(oldest) = map.iter().min_by_key(|(_, v)| v.at).map(|(k, _)| k.clone()) else {
                break;
            };
            map.remove(&oldest);
        }
        map.insert(
            key,
            Kept {
                at: Instant::now(),
                value: value.clone(),
            },
        );
        value
    }
}

// ---------------------------------------------------------------------------------------------
// GET /v1/refine/floor

/// The input size and where it came from: the recipe of the check whose recall the cards show
/// (the full check's, else the quick check's, [`recall_report`]), else the newest finished training
/// run's, else [`SUGGESTED_INPUT`].
fn suggested_input(state: &AppState, dir: &Path) -> (u32, &'static str) {
    let ok = |n: u32| (MIN_FLOOR_INPUT..=MAX_FLOOR_INPUT).contains(&n);
    if let Some(r) = recall_report(dir)
        && ok(r.recipe.imgsz)
    {
        return (r.recipe.imgsz, "last_check");
    }
    let run = training::records(state)
        .into_iter()
        .filter(|r| r.state == RunState::Done && ok(r.request.recipe.imgsz))
        .max_by(|a, b| a.created.cmp(&b.created));
    match run {
        Some(r) => (r.request.recipe.imgsz, "last_run"),
        None => (SUGGESTED_INPUT, "default"),
    }
}

/// The floor the query names (None = no query: the saved one), or why it is refused.
fn floor_from_query(q: &BTreeMap<String, String>) -> Result<Option<SizeFloor>, String> {
    let get = |k: &str| q.get(k).map(|v| v.trim()).filter(|v| !v.is_empty());
    let (px, input) = (get("px"), get("input"));
    if px.is_none() && input.is_none() {
        return Ok(None);
    }
    let (Some(px), Some(input)) = (px, input) else {
        return Err("give both px and input, or neither (the saved floor)".into());
    };
    let floor = SizeFloor {
        px: px
            .parse::<f32>()
            .map_err(|_| format!("px: '{px}' is not a number"))?,
        input: input
            .parse::<u32>()
            .map_err(|_| format!("input: '{input}' is not a whole number"))?,
    };
    floor.validate("size floor").map_err(|e| format!("{e:#}"))?;
    Ok(Some(floor))
}

/// The answer of `GET /v1/refine/floor` for `state` and the floor asked for (None = the saved).
fn floor_of(state: &AppState, asked: Option<SizeFloor>) -> serde_json::Value {
    let config = state.config_read().clone();
    let dir = config.collect.resolved_review_dir(&config.data_root());
    let saved = config.collect.size_floor;
    let shown = asked.or(saved);
    let mut v = serde_json::json!({
        "success": true,
        "folder_missing": false,
        "saved": floor_opt(saved),
        "shown": floor_opt(shown),
        "suggested": { "px": SUGGESTED_PX, "input": SUGGESTED_INPUT, "from": "default" },
        "bounds": {
            "px_min": MIN_FLOOR_PX,
            "px_max": MAX_FLOOR_PX,
            "input_min": MIN_FLOOR_INPUT,
            "input_max": MAX_FLOOR_INPUT,
        },
        "counts": null,
        "pending": { "boxes": 0, "frames": 0 },
        "restorable": { "boxes": 0, "frames": 0 },
        "ignore_mode": config.training.ignore.as_str(),
    });
    let Some(o) = v.as_object_mut() else {
        return v;
    };
    // An unplugged drive is never an empty dataset: nothing is read from it.
    if let Some(why) = config.collect.review_folder_problem(&dir) {
        o.insert("folder_missing".into(), true.into());
        o.insert("message".into(), why.into());
        return v;
    }
    let (input, from) = suggested_input(state, &dir);
    o.insert(
        "suggested".into(),
        serde_json::json!({ "px": SUGGESTED_PX, "input": input, "from": from }),
    );
    let all = CACHE.computed(
        &dir,
        &config.collect.classes,
        &config.collect.hidden_classes,
        shown,
        saved,
    );
    o.insert(
        "restorable".into(),
        serde_json::json!({ "boxes": all.restorable_boxes, "frames": all.restorable_frames }),
    );
    let Some(counts) = &all.counts else {
        return v;
    };
    let per_class: Vec<serde_json::Value> = counts
        .boxes
        .iter()
        .map(|(class, boxes)| {
            let under = counts.under.get(class).copied().unwrap_or(0);
            serde_json::json!({
                "class": class,
                "boxes": boxes,
                "under": under,
                "left": boxes.saturating_sub(under),
            })
        })
        .collect();
    o.insert(
        "counts".into(),
        serde_json::json!({
            "frames": counts.frames,
            "frames_with": counts.frames_with,
            "frames_only": counts.frames_only,
            "per_class": per_class,
            "warnings": counts.warnings,
        }),
    );
    o.insert(
        "pending".into(),
        serde_json::json!({ "boxes": all.pending_boxes, "frames": all.pending_frames }),
    );
    v
}

/// `GET /v1/refine/floor`: see the module docs. 400 for a floor out of bounds.
pub(super) async fn floor(
    State(state): State<Arc<AppState>>,
    Query(q): Query<BTreeMap<String, String>>,
) -> Response {
    let asked = match floor_from_query(&q) {
        Ok(f) => f,
        Err(m) => return fail(StatusCode::BAD_REQUEST, format!("Not previewed: {m}")),
    };
    blocking(move || Json(floor_of(&state, asked)).into_response()).await
}

// ---------------------------------------------------------------------------------------------
// POST /v1/refine/size-floor and /v1/refine/ignore-mode

/// Body of `POST /v1/refine/size-floor`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FloorBody {
    #[serde(default)]
    px: Option<f32>,
    #[serde(default)]
    input: Option<u32>,
    #[serde(default)]
    off: bool,
}

/// `POST /v1/refine/size-floor` with `{px, input}` or `{off: true}`: see the module docs. 200
/// `{success, message, size_floor}` (`size_floor` null when off), 400 for a floor out of bounds or
/// a body that names both or neither.
pub(super) async fn size_floor(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: FloorBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((status, message)) => return fail(status, message),
    };
    let new = match (b.off, b.px, b.input) {
        (true, None, None) => None,
        (false, Some(px), Some(input)) => {
            let f = SizeFloor { px, input };
            if let Err(e) = f.validate("collect.size_floor") {
                return fail(StatusCode::BAD_REQUEST, format!("Not saved: {e:#}"));
            }
            Some(f)
        }
        _ => {
            return fail(
                StatusCode::BAD_REQUEST,
                "Not saved: send {\"px\": .., \"input\": ..} to set a floor or {\"off\": true} to \
                 turn it off, not both and not neither.",
            );
        }
    };
    blocking(move || {
        let r = save_config(&state, "refine page", |c| {
            c.collect.size_floor = new;
            c.collect.validate(&c.data_root())
        });
        match r {
            Ok(message) => {
                forget();
                Json(serde_json::json!({
                    "success": true,
                    "message": message,
                    "size_floor": floor_opt(new),
                }))
                .into_response()
            }
            Err(e) => fail(StatusCode::BAD_REQUEST, format!("Not saved: {e:#}")),
        }
    })
    .await
}

/// Body of `POST /v1/refine/ignore-mode`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModeBody {
    mode: IgnoreMode,
}

/// `POST /v1/refine/ignore-mode` with `{mode}`: saves `training.ignore`. 200 `{success, message,
/// mode}`, 400 for another word.
pub(super) async fn ignore_mode(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: ModeBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((status, message)) => return fail(status, message),
    };
    let mode = b.mode;
    blocking(move || {
        let r = save_config(&state, "refine page", |c| {
            c.training.ignore = mode;
            c.training.validate()
        });
        match r {
            Ok(message) => Json(serde_json::json!({
                "success": true,
                "message": message,
                "mode": mode.as_str(),
            }))
            .into_response(),
            Err(e) => fail(StatusCode::BAD_REQUEST, format!("Not saved: {e:#}")),
        }
    })
    .await
}

// ---------------------------------------------------------------------------------------------
// POST /v1/refine/resolve-small

/// Body of `POST /v1/refine/resolve-small`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResolveBody {
    #[serde(default)]
    dry_run: bool,
    #[serde(default)]
    limit: Option<usize>,
    /// Give back the size floor dismissals the saved floor (or none) no longer covers, instead.
    #[serde(default)]
    restore: bool,
}

/// `POST /v1/refine/resolve-small` with `{dry_run?, limit?, restore?}`: see the module docs. 200
/// `{success, dry_run, restore, boxes, frames, eligible, failed, failed_count, message}` (`boxes` and
/// `frames`: what a dry run would change over every eligible frame, what an apply changed), 400 for
/// no saved floor (not with `restore`) or a bad field, 503 for an unplugged review folder.
pub(super) async fn resolve_small(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: ResolveBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((status, message)) => return fail(status, message),
    };
    let limit = b.limit.unwrap_or(RESOLVE_LIMIT);
    if !(1..=RESOLVE_LIMIT).contains(&limit) {
        return fail(
            StatusCode::BAD_REQUEST,
            format!("limit: must be between 1 and {RESOLVE_LIMIT}, got {limit}"),
        );
    }
    let (dry_run, restore) = (b.dry_run, b.restore);
    blocking(move || {
        let config = state.config_read().clone();
        let saved = config.collect.size_floor;
        if saved.is_none() && !restore {
            return fail(
                StatusCode::BAD_REQUEST,
                "Not run: no size floor is saved. Set one in the Too small card and save it first.",
            );
        }
        let dir = config.collect.resolved_review_dir(&config.data_root());
        if let Some(why) = config.collect.review_folder_problem(&dir) {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "success": false, "folder_missing": true, "message": why,
                })),
            )
                .into_response();
        }
        let filters = rule_filters(&config.collect.classes, &config.collect.hidden_classes);
        let o = match saved {
            Some(floor) if !restore => {
                bulk::bulk_resolve_small(&dir, &filters, floor, dry_run, limit)
            }
            _ => bulk::bulk_restore_small(&dir, &filters, saved, dry_run, limit),
        };
        let frames = if dry_run { o.eligible } else { o.done_count };
        if !dry_run {
            // Label writes tell the Review index and the repeats cache themselves; the tab's own
            // answers are dropped here.
            forget_counts();
            forget();
            super::refine_coverage::forget();
        }
        let mut message = match (restore, o.boxes == 0, dry_run) {
            (false, true, _) => {
                "No pending proposals under the size floor on the unreviewed frames.".to_string()
            }
            (false, false, true) => format!(
                "Would hide {} pending proposal(s) under the size floor on {} unreviewed \
                 frame(s) (the frames stay unreviewed).",
                o.boxes, o.eligible
            ),
            (false, false, false) => format!(
                "Hid {} pending proposal(s) under the size floor on {} unreviewed frame(s): set \
                 aside unseen, not counted as your decisions (i shows them on the Review tab); \
                 the frames stay unreviewed.",
                o.boxes, o.done_count
            ),
            (true, true, _) => {
                "No hidden proposal is waiting for a lower size floor on the unreviewed frames."
                    .to_string()
            }
            (true, false, true) => format!(
                "Would restore {} proposal(s) the size floor no longer covers on {} unreviewed \
                 frame(s) (back to pending).",
                o.boxes, o.eligible
            ),
            (true, false, false) => format!(
                "Restored {} proposal(s) the size floor no longer covers on {} unreviewed \
                 frame(s): pending again, for you to decide on the Review tab.",
                o.boxes, o.done_count
            ),
        };
        push_failed(
            &mut message,
            o.failed_count,
            "frame(s) not changed",
            "review",
        );
        push_past_limit(&mut message, o.eligible, limit, "more frame(s)");
        if !dry_run {
            tracing::info!(
                boxes = o.boxes,
                frames = o.done_count,
                failed = o.failed_count,
                eligible = o.eligible,
                restore,
                "refine: pending proposals under the size floor {}",
                if restore { "restored" } else { "hidden" }
            );
        }
        Json(serde_json::json!({
            "success": true,
            "dry_run": dry_run,
            "restore": restore,
            "boxes": o.boxes,
            "frames": frames,
            "eligible": o.eligible,
            "failed": o.failed,
            "failed_count": o.failed_count,
            "message": message,
        }))
        .into_response()
    })
    .await
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::test_util::{call, test_state_with};
    use super::*;
    use crate::collect::labels::bulk::test_support::{auto_box, put_frame_at};
    use crate::collect::labels::{self, BoxState, Frame, LabelBox, Source, Status, Triage};
    use axum::body::Body;
    use axum::http::{Request, header};
    use std::path::PathBuf;

    fn scratch() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bo_floor_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn state_in(dir: &Path) -> Arc<AppState> {
        let dir = dir.to_path_buf();
        test_state_with(move |c| {
            c.download_dir = Some(dir.clone());
            c.collect.dir = Some(dir.join("dataset"));
            c.training.dir = Some(dir.join("training"));
            c.collect.classes = vec!["car".into(), "person".into()];
        })
    }

    async fn json(state: &Arc<AppState>, req: Request<Body>) -> (StatusCode, serde_json::Value) {
        let (s, _, b) = call(state, req).await;
        (
            s,
            serde_json::from_slice(&b).unwrap_or(serde_json::Value::Null),
        )
    }

    fn get(uri: &str) -> Request<Body> {
        Request::get(uri).body(Body::empty()).unwrap()
    }

    fn post_json(uri: &str, body: &str) -> Request<Body> {
        Request::post(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    const TINY: [f32; 4] = [2.0, 2.0, 3.0, 30.0];
    const BIG: [f32; 4] = [4.0, 4.0, 40.0, 40.0];

    /// A box of `label` at `bbox`, accepted (a training box) from the auto-labeler.
    fn accepted(label: &str, bbox: [f32; 4]) -> LabelBox {
        LabelBox {
            state: BoxState::Accepted,
            bbox,
            ..auto_box(label, 0.9, Triage::Accept)
        }
    }

    fn pending(label: &str, bbox: [f32; 4]) -> LabelBox {
        LabelBox {
            bbox,
            ..auto_box(label, 0.6, Triage::Review)
        }
    }

    /// A reviewed frame (64 x 48) of camera `camera` with `boxes`.
    fn reviewed(dir: &Path, camera: &str, stem: &str, boxes: Vec<LabelBox>) -> String {
        let rel = put_frame_at(dir, camera, stem, (64, 48), None, boxes);
        let path = Frame::resolve(dir, &rel).unwrap().labels;
        let mut f = labels::read(&path).unwrap().unwrap();
        f.status = Status::Reviewed;
        f.reviewed_at = Some("2026-10-22T00:00:00Z".into());
        labels::write(&path, &f).unwrap();
        rel
    }

    fn unreviewed(dir: &Path, camera: &str, stem: &str, boxes: Vec<LabelBox>) -> String {
        put_frame_at(dir, camera, stem, (64, 48), None, boxes)
    }

    fn file(dir: &Path, rel: &str) -> labels::LabelFile {
        labels::read(&Frame::resolve(dir, rel).unwrap().labels)
            .unwrap()
            .unwrap()
    }

    /// Reviewed: cars (3 big, 1 tiny), people (all 2 tiny) and a frame of only a tiny car;
    /// unreviewed: 2 frames with pending tiny proposals (3 boxes), one with a drawn tiny box.
    /// At 64 x 48, a box 1 px wide is 6 px at input 384 (under 10); one 36 px wide is 216.
    fn folder(dir: &Path) -> PathBuf {
        let review = dir.join("dataset");
        reviewed(
            &review,
            "yard",
            "r1",
            vec![accepted("car", BIG), accepted("car", TINY)],
        );
        reviewed(&review, "yard", "r2", vec![accepted("car", BIG)]);
        reviewed(&review, "yard", "r3", vec![accepted("car", BIG)]);
        reviewed(
            &review,
            "drive",
            "r4",
            vec![accepted("person", TINY), accepted("person", TINY)],
        );
        reviewed(&review, "drive", "r5", vec![accepted("car", TINY)]);
        unreviewed(
            &review,
            "yard",
            "u1",
            vec![
                pending("car", TINY),
                pending("person", TINY),
                pending("car", BIG),
            ],
        );
        unreviewed(&review, "yard", "u2", vec![pending("car", TINY)]);
        let mut drawn = pending("car", TINY);
        drawn.source = Source::Human;
        unreviewed(&review, "yard", "u3", vec![drawn]);
        review
    }

    #[tokio::test]
    async fn the_routes_need_a_sign_in_a_same_site_post_and_valid_input() {
        let dir = scratch();
        let state = state_in(&dir);
        for (method, uri, body) in [
            ("GET", "/v1/refine/floor", ""),
            ("POST", "/v1/refine/size-floor", r#"{"px":10,"input":384}"#),
            ("POST", "/v1/refine/ignore-mode", r#"{"mode":"drop-frame"}"#),
            ("POST", "/v1/refine/resolve-small", r#"{"dry_run":true}"#),
        ] {
            let mk = || {
                if method == "GET" {
                    get(uri)
                } else {
                    post_json(uri, body)
                }
            };
            let (s, _) = json(&state, super::super::test_util::anonymous(mk())).await;
            assert_eq!(s, StatusCode::UNAUTHORIZED, "{uri}");
            if method == "POST" {
                let cross = Request::post(uri)
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("host", "localhost:32168")
                    .header("origin", "http://evil.example")
                    .body(Body::from(body.to_string()))
                    .unwrap();
                let (s, _) = json(&state, cross).await;
                assert_eq!(s, StatusCode::FORBIDDEN, "{uri}");
                let form = super::super::test_util::post_form(uri, "px=10", false);
                let (s, _) = json(&state, form).await;
                assert_eq!(s, StatusCode::UNSUPPORTED_MEDIA_TYPE, "{uri}");
            }
        }
        // Bad input: refused with a message, nothing written.
        for (uri, bodies) in [
            (
                "/v1/refine/size-floor",
                vec![
                    "not json",
                    r#"{}"#,
                    r#"{"px":10}"#,
                    r#"{"input":384}"#,
                    r#"{"px":10,"input":384,"off":true}"#,
                    r#"{"off":true,"px":10}"#,
                    r#"{"px":10,"input":384,"x":1}"#,
                    r#"{"px":0.5,"input":384}"#,
                    r#"{"px":65,"input":384}"#,
                    r#"{"px":10,"input":63}"#,
                    r#"{"px":10,"input":4097}"#,
                    r#"{"px":"10","input":384}"#,
                    r#"{"px":10,"input":384.5}"#,
                ],
            ),
            (
                "/v1/refine/ignore-mode",
                vec![
                    r#"{}"#,
                    r#"{"mode":"skip"}"#,
                    r#"{"mode":""}"#,
                    r#"{"mode":"drop"}"#,
                    r#"{"mode":"background","x":1}"#,
                    r#"{"mode":7}"#,
                ],
            ),
            (
                "/v1/refine/resolve-small",
                vec![
                    r#"{"limit":0}"#,
                    r#"{"limit":501}"#,
                    r#"{"x":1}"#,
                    r#"{"dry_run":"yes"}"#,
                    r#"{"restore":"yes"}"#,
                ],
            ),
        ] {
            for body in bodies {
                let (s, v) = json(&state, post_json(uri, body)).await;
                assert_eq!(s, StatusCode::BAD_REQUEST, "{uri} {body}: {v}");
                assert_eq!(v["success"], false);
                assert!(v["message"].as_str().is_some_and(|m| !m.is_empty()));
            }
        }
        let cfg = state.config_read();
        assert_eq!(cfg.collect.size_floor, None);
        assert_eq!(cfg.training.ignore, IgnoreMode::Background);
        drop(cfg);
        // A body over 4 KiB.
        let big = format!(r#"{{"mode":"{}"}}"#, "x".repeat(5000));
        let (s, _) = json(&state, post_json("/v1/refine/ignore-mode", &big)).await;
        assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
        // The preview's query: both or neither, within bounds, numbers.
        for q in [
            "px=10",
            "input=384",
            "px=x&input=384",
            "px=10&input=3.5",
            "px=0&input=384",
            "px=10&input=5000",
            "px=NaN&input=384",
        ] {
            let (s, v) = json(&state, get(&format!("/v1/refine/floor?{q}"))).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{q}: {v}");
            assert_eq!(v["success"], false);
        }
    }

    #[tokio::test]
    async fn the_preview_counts_a_floor_per_class_saved_or_asked_for() {
        let dir = scratch();
        folder(&dir);
        let state = state_in(&dir);
        // No floor: nothing shown, the suggestion is the default, the mode the default.
        let (s, v) = json(&state, get("/v1/refine/floor")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["folder_missing"], false);
        assert!(v["saved"].is_null() && v["shown"].is_null() && v["counts"].is_null());
        assert_eq!(v["pending"], serde_json::json!({"boxes": 0, "frames": 0}));
        assert_eq!(
            v["suggested"],
            serde_json::json!({"px": 10.0, "input": 384, "from": "default"})
        );
        assert_eq!(v["ignore_mode"], "background");
        assert_eq!(v["bounds"]["input_max"], 4096);

        // A floor asked for, not saved.
        let (_, v) = json(&state, get("/v1/refine/floor?px=10&input=384")).await;
        assert!(v["saved"].is_null());
        assert_eq!(v["shown"], serde_json::json!({"px": 10.0, "input": 384}));
        let c = &v["counts"];
        assert_eq!(c["frames"], 5, "{v}");
        assert_eq!(c["frames_with"], 3, "r1, r4 and r5 hold a box under it");
        assert_eq!(c["frames_only"], 2, "r4 and r5 keep no label");
        assert_eq!(
            c["per_class"],
            serde_json::json!([
                {"class": "car", "boxes": 5, "under": 2, "left": 3},
                {"class": "person", "boxes": 2, "under": 2, "left": 0},
            ])
        );
        // 3 pending proposals under it on 2 unreviewed frames (the drawn box is a decision).
        assert_eq!(v["pending"], serde_json::json!({"boxes": 3, "frames": 2}));
        // A smaller floor leaves the same boxes (6 px) over it.
        let (_, v) = json(&state, get("/v1/refine/floor?px=6&input=384")).await;
        assert_eq!(v["counts"]["frames_with"], 0, "6 px is not under 6: {v}");
        assert_eq!(v["pending"], serde_json::json!({"boxes": 0, "frames": 0}));

        // Saved: the same answer with no query, and the query still wins.
        let (s, v) = json(
            &state,
            post_json("/v1/refine/size-floor", r#"{"px":10,"input":384}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let (_, v) = json(&state, get("/v1/refine/floor")).await;
        assert_eq!(v["saved"], serde_json::json!({"px": 10.0, "input": 384}));
        assert_eq!(v["shown"], v["saved"]);
        assert_eq!(v["counts"]["frames_with"], 3);
        let (_, v) = json(&state, get("/v1/refine/floor?px=20&input=640")).await;
        assert_eq!(v["saved"], serde_json::json!({"px": 10.0, "input": 384}));
        assert_eq!(v["shown"], serde_json::json!({"px": 20.0, "input": 640}));
        // 36 px wide at 640 / 64 = 360: over; 1 px wide = 10 px at 640: under 20.
        assert_eq!(v["counts"]["per_class"][0]["under"], 2, "{v}");
    }

    #[tokio::test]
    async fn the_suggestion_follows_the_last_check() {
        use crate::training::refine::report::RefineReport;
        let dir = scratch();
        let review = folder(&dir);
        let state = state_in(&dir);
        // A check's report of imgsz 512: its recipe wins over the default.
        let mut recipe = crate::training::recipe::TrainRecipe::new(
            crate::resources::catalog::TrainFamily::Rfdetr,
            "nano",
        )
        .unwrap();
        recipe.imgsz = 512;
        let report = RefineReport::new(
            "run1",
            4,
            crate::training::refine::recipe_info(&recipe, "test"),
        );
        let path = crate::training::refine::report::report_path(&review);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, serde_json::to_vec(&report).unwrap()).unwrap();
        let (_, v) = json(&state, get("/v1/refine/floor")).await;
        assert_eq!(
            v["suggested"],
            serde_json::json!({"px": 10.0, "input": 512, "from": "last_check"}),
            "{v}"
        );
    }

    #[test]
    fn the_walk_is_kept_until_forgotten_and_a_floor_has_its_own_answer() {
        let dir = scratch();
        let review = folder(&dir);
        let classes = vec!["car".to_string(), "person".to_string()];
        let none: Vec<String> = Vec::new();
        let f10 = SizeFloor {
            px: 10.0,
            input: 384,
        };
        let f20 = SizeFloor {
            px: 20.0,
            input: 384,
        };
        // A cache of its own: other tests clear the process's at any time.
        let cache = Cache::new();
        let first = cache.computed(&review, &classes, &none, Some(f10), None);
        let again = cache.computed(&review, &classes, &none, Some(f10), None);
        assert!(Arc::ptr_eq(&first, &again), "kept");
        forget();
        assert!(Arc::ptr_eq(
            &first,
            &cache.computed(&review, &classes, &none, Some(f10), None)
        ));
        let other = cache.computed(&review, &classes, &none, Some(f20), None);
        assert!(!Arc::ptr_eq(&first, &other), "another floor");
        let saved = cache.computed(&review, &classes, &none, Some(f10), Some(f10));
        assert!(!Arc::ptr_eq(&first, &saved), "another saved floor");
        let hid = vec!["car".to_string()];
        assert!(!Arc::ptr_eq(
            &first,
            &cache.computed(&review, &classes, &hid, Some(f10), None)
        ));
        cache.forget();
        assert!(!Arc::ptr_eq(
            &first,
            &cache.computed(&review, &classes, &none, Some(f10), None)
        ));
        // Bounded: many floors never keep more than KEEP answers.
        for px in 1..=20 {
            cache.computed(
                &review,
                &classes,
                &none,
                Some(SizeFloor {
                    px: px as f32,
                    input: 384,
                }),
                None,
            );
        }
        assert!(cache.kept.lock_recover().len() <= KEEP);
    }

    #[tokio::test]
    async fn the_floor_and_the_mode_are_saved_bounded_and_turned_off() {
        let dir = scratch();
        let state = state_in(&dir);
        let (s, v) = json(
            &state,
            post_json("/v1/refine/size-floor", r#"{"px":7.3,"input":640}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["success"], true);
        assert_eq!(
            v["size_floor"],
            serde_json::json!({"px": 7.3, "input": 640})
        );
        let f = state.config_read().collect.size_floor.unwrap();
        assert_eq!((f.px, f.input), (7.3, 640));
        // The bounds are inclusive.
        for (px, input) in [(1.0, 64), (64.0, 4096)] {
            let body = format!(r#"{{"px":{px},"input":{input}}}"#);
            let (s, _) = json(&state, post_json("/v1/refine/size-floor", &body)).await;
            assert_eq!(s, StatusCode::OK, "{body}");
        }
        let f = state.config_read().collect.size_floor.unwrap();
        assert_eq!((f.px, f.input), (64.0, 4096));
        // Off.
        let (s, v) = json(
            &state,
            post_json("/v1/refine/size-floor", r#"{"off":true}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert!(v["size_floor"].is_null());
        assert_eq!(state.config_read().collect.size_floor, None);
        let (_, v) = json(&state, get("/v1/refine/floor")).await;
        assert!(v["saved"].is_null());
        // Off again changes nothing and is no error.
        let (s, _) = json(
            &state,
            post_json("/v1/refine/size-floor", r#"{"off":true}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK);

        assert_eq!(state.config_read().training.ignore, IgnoreMode::Background);
        let (s, v) = json(
            &state,
            post_json("/v1/refine/ignore-mode", r#"{"mode":"drop-frame"}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["mode"], "drop-frame");
        assert_eq!(state.config_read().training.ignore, IgnoreMode::DropFrame);
        let (_, v) = json(&state, get("/v1/refine/floor")).await;
        assert_eq!(v["ignore_mode"], "drop-frame");
        let (s, _) = json(
            &state,
            post_json("/v1/refine/ignore-mode", r#"{"mode":"background"}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(state.config_read().training.ignore, IgnoreMode::Background);
    }

    #[tokio::test]
    async fn resolve_small_needs_a_saved_floor_and_an_unplugged_folder_is_a_503() {
        let dir = scratch();
        folder(&dir);
        let state = state_in(&dir);
        let (s, v) = json(
            &state,
            post_json("/v1/refine/resolve-small", r#"{"dry_run":true}"#),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
        assert_eq!(v["success"], false);
        assert!(
            v["message"]
                .as_str()
                .unwrap()
                .contains("no size floor is saved")
        );
        let d = dir.clone();
        let gone = test_state_with(move |c| {
            c.download_dir = Some(d.clone());
            c.collect.review_dir = Some(d.join("unplugged"));
            c.collect.size_floor = Some(SizeFloor {
                px: 10.0,
                input: 384,
            });
            c.training.dir = Some(d.join("training"));
        });
        let (s, v) = json(&gone, post_json("/v1/refine/resolve-small", "{}")).await;
        assert_eq!(
            (s, v["folder_missing"].as_bool()),
            (StatusCode::SERVICE_UNAVAILABLE, Some(true))
        );
        let (s, v) = json(&gone, get("/v1/refine/floor")).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["folder_missing"], true);
        assert!(v["counts"].is_null());
        assert_eq!(v["saved"], serde_json::json!({"px": 10.0, "input": 384}));
    }

    #[tokio::test]
    async fn the_rule_counts_what_it_does_then_has_nothing_left_to_do() {
        let dir = scratch();
        let review = folder(&dir);
        let state = state_in(&dir);
        let (s, _) = json(
            &state,
            post_json("/v1/refine/size-floor", r#"{"px":10,"input":384}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let names = ["r1", "r2", "r3", "r4", "r5", "u1", "u2", "u3"];
        let rel = |n: &str| {
            let cam = if matches!(n, "r4" | "r5") {
                "drive"
            } else {
                "yard"
            };
            format!("ir/{cam}/2026-10-21/{n}.jpg")
        };
        let before: Vec<labels::LabelFile> = names.iter().map(|n| file(&review, &rel(n))).collect();
        // The card's own count first.
        let (_, card) = json(&state, get("/v1/refine/floor")).await;
        assert_eq!(
            card["pending"],
            serde_json::json!({"boxes": 3, "frames": 2})
        );

        let (s, dry) = json(
            &state,
            post_json("/v1/refine/resolve-small", r#"{"dry_run":true}"#),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{dry}");
        assert_eq!(
            (
                dry["boxes"].as_u64(),
                dry["frames"].as_u64(),
                dry["eligible"].as_u64()
            ),
            (Some(3), Some(2), Some(2))
        );
        assert_eq!(dry["dry_run"], true);
        assert!(
            dry["message"].as_str().unwrap().starts_with(
                "Would hide 3 pending proposal(s) under the size floor on 2 unreviewed"
            )
        );
        let same: Vec<labels::LabelFile> = names.iter().map(|n| file(&review, &rel(n))).collect();
        assert_eq!(same, before, "a dry run writes nothing");

        let (s, done) = json(&state, post_json("/v1/refine/resolve-small", "{}")).await;
        assert_eq!(s, StatusCode::OK, "{done}");
        assert_eq!(done["dry_run"], false);
        assert_eq!(
            (
                done["boxes"].as_u64(),
                done["frames"].as_u64(),
                done["failed_count"].as_u64()
            ),
            (dry["boxes"].as_u64(), dry["frames"].as_u64(), Some(0)),
            "the dry run's count is what the apply does"
        );
        assert!(
            done["message"]
                .as_str()
                .unwrap()
                .starts_with("Hid 3 pending proposal(s)")
        );
        let after: Vec<labels::LabelFile> = names.iter().map(|n| file(&review, &rel(n))).collect();
        for (i, (b, a)) in before.iter().zip(&after).enumerate() {
            match names[i] {
                "u1" => {
                    assert_eq!(a.status, Status::Unreviewed);
                    let got: Vec<(BoxState, Option<&str>)> = a
                        .boxes
                        .iter()
                        .map(|x| (x.state, x.reason.as_deref()))
                        .collect();
                    assert_eq!(
                        got,
                        [
                            (BoxState::Rejected, Some("size_floor")),
                            (BoxState::Rejected, Some("size_floor")),
                            (BoxState::Pending, None),
                        ],
                        "the big proposal waits"
                    );
                    assert!(!a.boxes[0].is_ignore() && a.boxes[0].is_size_floor_dismissal());
                }
                "u2" => {
                    assert_eq!(a.boxes[0].state, BoxState::Rejected);
                    assert!(a.boxes[0].is_size_floor_dismissal());
                }
                // Reviewed frames, and a frame with a drawn box, are untouched.
                _ => assert_eq!(a, b, "{} is untouched", names[i]),
            }
        }
        // A second apply and a dry run find nothing; the card follows.
        for dry_run in [true, false] {
            let body = format!(r#"{{"dry_run":{dry_run}}}"#);
            let (_, again) = json(&state, post_json("/v1/refine/resolve-small", &body)).await;
            assert_eq!(
                (
                    again["boxes"].as_u64(),
                    again["frames"].as_u64(),
                    again["eligible"].as_u64()
                ),
                (Some(0), Some(0), Some(0)),
                "{again}"
            );
            assert!(
                again["message"]
                    .as_str()
                    .unwrap()
                    .starts_with("No pending proposals")
            );
        }
        let (_, card) = json(&state, get("/v1/refine/floor")).await;
        assert_eq!(
            card["pending"],
            serde_json::json!({"boxes": 0, "frames": 0}),
            "the kept answer was dropped by the apply"
        );
        assert_eq!(
            card["restorable"],
            serde_json::json!({"boxes": 0, "frames": 0}),
            "the saved floor still covers them"
        );
    }

    #[tokio::test]
    async fn a_lower_floor_or_none_gives_the_hidden_proposals_back() {
        let dir = scratch();
        let review = folder(&dir);
        let state = state_in(&dir);
        let save = |body: &'static str| {
            let state = state.clone();
            async move {
                let (s, v) = json(&state, post_json("/v1/refine/size-floor", body)).await;
                assert_eq!(s, StatusCode::OK, "{v}");
            }
        };
        save(r#"{"px":10,"input":384}"#).await;
        json(&state, post_json("/v1/refine/resolve-small", "{}")).await;
        let u1 = || file(&review, "ir/yard/2026-10-21/u1.jpg");
        let hidden = u1();
        assert!(hidden.boxes[0].is_size_floor_dismissal());
        // 6 px boxes: a 6 px floor no longer covers them (not under 6).
        save(r#"{"px":6,"input":384}"#).await;
        let (_, card) = json(&state, get("/v1/refine/floor")).await;
        assert_eq!(
            card["restorable"],
            serde_json::json!({"boxes": 3, "frames": 2}),
            "{card}"
        );
        let (s, dry) = json(
            &state,
            post_json(
                "/v1/refine/resolve-small",
                r#"{"restore":true,"dry_run":true}"#,
            ),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{dry}");
        assert_eq!(
            (
                dry["boxes"].as_u64(),
                dry["frames"].as_u64(),
                dry["restore"].as_bool()
            ),
            (Some(3), Some(2), Some(true))
        );
        assert_eq!(u1(), hidden, "a dry run writes nothing");
        let (_, done) = json(
            &state,
            post_json("/v1/refine/resolve-small", r#"{"restore":true}"#),
        )
        .await;
        assert_eq!(
            (done["boxes"].as_u64(), done["frames"].as_u64()),
            (dry["boxes"].as_u64(), dry["frames"].as_u64()),
            "dry run = apply: {done}"
        );
        assert!(
            done["message"]
                .as_str()
                .unwrap()
                .starts_with("Restored 3 proposal(s)")
        );
        let back = u1();
        assert_eq!(
            back.boxes
                .iter()
                .map(|b| (b.state, b.reason.as_deref()))
                .collect::<Vec<_>>(),
            [
                (BoxState::Pending, None),
                (BoxState::Pending, None),
                (BoxState::Pending, None)
            ]
        );
        // Hidden again under the old floor, then the floor off: a restore needs no saved floor.
        save(r#"{"px":10,"input":384}"#).await;
        json(&state, post_json("/v1/refine/resolve-small", "{}")).await;
        save(r#"{"off":true}"#).await;
        let (_, card) = json(&state, get("/v1/refine/floor")).await;
        assert_eq!(card["restorable"]["boxes"], 3, "{card}");
        let (s, done) = json(
            &state,
            post_json("/v1/refine/resolve-small", r#"{"restore":true}"#),
        )
        .await;
        assert_eq!((s, done["boxes"].as_u64()), (StatusCode::OK, Some(3)));
        let (_, card) = json(&state, get("/v1/refine/floor")).await;
        assert_eq!(card["restorable"]["boxes"], 0, "{card}");
    }

    #[tokio::test]
    async fn the_limit_caps_the_frames_written_and_says_so() {
        let dir = scratch();
        let review = folder(&dir);
        let state = state_in(&dir);
        json(
            &state,
            post_json("/v1/refine/size-floor", r#"{"px":10,"input":384}"#),
        )
        .await;
        let (_, one) = json(
            &state,
            post_json("/v1/refine/resolve-small", r#"{"limit":1}"#),
        )
        .await;
        assert_eq!(
            (one["frames"].as_u64(), one["eligible"].as_u64()),
            (Some(1), Some(2))
        );
        assert!(
            one["message"]
                .as_str()
                .unwrap()
                .contains("1 more frame(s) beyond the limit of 1; run it again."),
            "{one}"
        );
        let (_, rest) = json(&state, post_json("/v1/refine/resolve-small", "{}")).await;
        assert_eq!(
            (rest["frames"].as_u64(), rest["eligible"].as_u64()),
            (Some(1), Some(1))
        );
        let _ = review;
    }

    #[tokio::test]
    async fn the_labels_answer_carries_the_floor_for_the_page() {
        let dir = scratch();
        let review = folder(&dir);
        let state = state_in(&dir);
        let uri = "/v1/review/labels?frame=ir/yard/2026-10-21/u1.jpg";
        let (s, v) = json(&state, get(uri)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert!(v["size_floor"].is_null(), "off: null");
        json(
            &state,
            post_json("/v1/refine/size-floor", r#"{"px":10,"input":384}"#),
        )
        .await;
        let (_, v) = json(&state, get(uri)).await;
        assert_eq!(
            v["size_floor"],
            serde_json::json!({"px": 10.0, "input": 384})
        );
        assert_eq!(v["width"], 64);
        let _ = review;
    }
}
