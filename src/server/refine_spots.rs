//! The Static objects card of the Refine tab (admin only; `assets/train-refine.js`): the places
//! where one parked object sits across a camera's frames ([`statics`]), the missing boxes the
//! dataset check found there, and the training cap for them. Three routes:
//!
//! - `GET /v1/refine/spots`: the review folder's spots ([`statics::overview`] over its reviewed
//!   frames, off the async workers, kept 60 s per folder and class list and dropped when a check
//!   starts or ends or a bulk accept writes) with, per spot, its newest labelled frame (`sample`,
//!   `sample_size`, served by `/dataset/image`) and a padded `crop` of it around the box, so the page
//!   can show the place. `open_frames` is left out (the accept route finds the frames itself).
//! - `POST /v1/refine/accept-spot`: every live `add` suggestion lying at one spot (by geometry,
//!   [`statics::Table::spot_of_add`]: the spot id a check stored names a spot of that check's own
//!   table and is not used) scoring `min_score` or more, accepted frame by frame through
//!   [`suggestions::accept`] under the label write lock ([`labels::update`], which also tells the
//!   Review index and the repeats cache). Never a shortcut around `accept`: it replaces unjudged
//!   rejects, refuses a stale suggestion and keeps the frame reviewed; a frame that is no longer
//!   reviewed gets nothing, and a frame a reviewer confirmed empty is left for one-by-one review
//!   (`left_empty`: `accept` would clear the reviewer's "empty"). A dry run runs the same code on
//!   the frames as read and writes nothing, so on an unchanged folder its count is the applied
//!   count; at most `limit` frames (500) are written per call.
//! - `POST /v1/refine/static-cap` `{cap}`: `training.static_cap`, the most frames per spot a
//!   training run keeps of those showing nothing else (0 = off), saved through the one config
//!   writer like the other settings routes.

use super::refine::{fail, forget_counts, json_body, queue_stamp};
use super::review::{push_failed, push_past_limit};
use super::{AppState, blocking, save_config, training};
use crate::collect::labels::bulk::{LISTED, Wanted};
use crate::collect::labels::export::{self, ClassFilter, ExportFrame, SplitConfig};
use crate::collect::labels::refine as suggestions;
use crate::collect::labels::statics::{self, SPOT_IOU, SpotSummary};
use crate::collect::labels::{self, Frame, LabelFile, Status};
use crate::sync::LockExt;
use crate::training::config::{DEFAULT_STATIC_CAP, MAX_STATIC_CAP};
use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::extract::rejection::BytesRejection;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Largest request body of the two POST routes.
pub(super) const BODY_LIMIT: usize = 4 * 1024;
/// How long a walk's answer is kept.
const KEEP_FOR: Duration = Duration::from_secs(60);
/// Most answers kept at once (folders under review x class lists).
const KEEP: usize = 4;
/// Most spots the listing answers (the busiest first); `totals` always counts all of them.
const MAX_SPOTS: usize = 200;
/// Frames a bulk accept writes at most per call (also its default).
pub(super) const ACCEPT_LIMIT: usize = 500;
/// Frames the answer names to look at afterwards.
const SAMPLE: usize = 10;
/// The suggestion score from which a bulk accept takes an `add` by default.
const DEFAULT_MIN_SCORE: f32 = 0.5;
/// The crop around a spot's box: this share of the box's size on every side (at least
/// [`CROP_MIN_PAD`] of the frame).
const CROP_PAD: f32 = 0.6;
const CROP_MIN_PAD: f32 = 0.04;

// ---------------------------------------------------------------------------------------------
// The spots of the folder

/// A frame labelled at a spot, to show the place.
#[derive(Debug, Clone, PartialEq)]
struct Sample {
    rel: String,
    width: u32,
    height: u32,
}

/// What one walk of the folder knows.
struct Computed {
    /// The spots of the reviewed frames (what `spots` summarises).
    table: statics::Table,
    spots: Vec<SpotSummary>,
    samples: HashMap<String, Sample>,
}

/// The newest frame (by day, then name) with a training box at each spot of `spots`.
fn newest_samples(
    frames: &[ExportFrame],
    spots: &[SpotSummary],
    wanted: &Wanted<'_>,
) -> HashMap<String, Sample> {
    let mut by_camera: HashMap<&str, Vec<&statics::Spot>> = HashMap::new();
    for s in spots {
        by_camera.entry(&s.spot.camera).or_default().push(&s.spot);
    }
    // spot id -> (day, stem, index of the frame)
    let mut best: HashMap<&str, (&str, &str, usize)> = HashMap::new();
    for (i, f) in frames.iter().enumerate() {
        let Some(list) = by_camera.get(f.rel.camera.as_str()) else {
            continue;
        };
        for b in f.labels.training().filter(|b| !b.is_ignore()) {
            let (Some(class), Some(nb)) = (
                wanted.class_of(&b.label),
                statics::normalised(&b.bbox, f.width, f.height),
            ) else {
                continue;
            };
            for spot in list {
                if spot.iou_with(class, &nb, SPOT_IOU).is_none() {
                    continue;
                }
                let key = (f.rel.date.as_str(), f.rel.stem.as_str());
                let entry = best.entry(spot.id.as_str()).or_insert((key.0, key.1, i));
                if key > (entry.0, entry.1) {
                    *entry = (key.0, key.1, i);
                }
            }
        }
    }
    best.into_iter()
        .filter_map(|(id, (_, _, i))| {
            let f = frames.get(i)?;
            Some((
                id.to_string(),
                Sample {
                    rel: f.rel.jpg(),
                    width: f.width,
                    height: f.height,
                },
            ))
        })
        .collect()
}

/// A fresh walk of the reviewed frames of `dir`.
fn compute(dir: &Path, classes: &[String], hidden: &[String]) -> Computed {
    let wanted = Wanted { classes, hidden };
    let sel = export::reviewed_frames_filtered(
        dir,
        classes,
        &SplitConfig::default(),
        &ClassFilter::default(),
    );
    let (table, spots) = statics::overview_table(&sel.frames, &wanted);
    let samples = newest_samples(&sel.frames, &spots, &wanted);
    Computed {
        table,
        spots,
        samples,
    }
}

struct Kept {
    at: Instant,
    stamp: String,
    value: Arc<Computed>,
}

/// The walks' answers per folder and class list (one per process, [`CACHE`]; a test makes its
/// own, so other tests clearing the process's cannot touch it).
struct Cache {
    kept: Mutex<BTreeMap<String, Kept>>,
    /// One walk at a time: a second caller waits and finds the first one's answer.
    walking: Mutex<()>,
}

/// The process's walk answers.
static CACHE: Cache = Cache::new();

/// Drop every kept answer of the process (a label write changed what a walk finds).
fn forget() {
    CACHE.forget();
}

/// [`Cache::computed`] of the process.
fn computed(dir: &Path, classes: &[String], hidden: &[String], stamp: &str) -> Arc<Computed> {
    CACHE.computed(dir, classes, hidden, stamp)
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

    /// [`compute`], kept [`KEEP_FOR`]; `stamp` names the checks on the queue: when one starts,
    /// ends or is cancelled the kept answer is dropped.
    fn computed(
        &self,
        dir: &Path,
        classes: &[String],
        hidden: &[String],
        stamp: &str,
    ) -> Arc<Computed> {
        let key = format!(
            "{}\u{1f}{}\u{1f}{}",
            dir.display(),
            classes.join("\u{1e}"),
            hidden.join("\u{1e}")
        );
        let fresh = |k: &Kept| k.at.elapsed() < KEEP_FOR && k.stamp == stamp;
        if let Some(k) = self.kept.lock_recover().get(&key).filter(|k| fresh(k)) {
            return k.value.clone();
        }
        let _walking = self.walking.lock_recover();
        if let Some(k) = self.kept.lock_recover().get(&key).filter(|k| fresh(k)) {
            return k.value.clone();
        }
        let value = Arc::new(compute(dir, classes, hidden));
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
                stamp: stamp.to_string(),
                value: value.clone(),
            },
        );
        value
    }
}

/// The box `b` (normalised) with a margin, clamped to the frame: what the page shows of the place.
fn crop_of(b: &[f32; 4]) -> [f32; 4] {
    let pad = |size: f32| (size * CROP_PAD).max(CROP_MIN_PAD);
    let (w, h) = (b[2] - b[0], b[3] - b[1]);
    [
        (b[0] - pad(w)).max(0.0),
        (b[1] - pad(h)).max(0.0),
        (b[2] + pad(w)).min(1.0),
        (b[3] + pad(h)).min(1.0),
    ]
}

fn spot_json(s: &SpotSummary, sample: Option<&Sample>) -> serde_json::Value {
    let mut v = serde_json::to_value(s).unwrap_or_default();
    if let Some(o) = v.as_object_mut() {
        o.remove("open_frames");
        o.insert("crop".into(), serde_json::json!(crop_of(&s.spot.bbox)));
        o.insert(
            "sample".into(),
            sample
                .map(|x| x.rel.clone())
                .map_or(serde_json::Value::Null, Into::into),
        );
        o.insert(
            "sample_size".into(),
            sample.map_or(serde_json::Value::Null, |x| {
                serde_json::json!([x.width, x.height])
            }),
        );
    }
    v
}

/// `training.static_cap` and its bounds, for the card's setting.
fn cap_bits(state: &AppState) -> serde_json::Value {
    serde_json::json!({
        "static_cap": state.config_read().training.static_cap,
        "static_cap_default": DEFAULT_STATIC_CAP,
        "static_cap_max": MAX_STATIC_CAP,
    })
}

/// The answer of `GET /v1/refine/spots` for `state`.
fn spots_of(state: &AppState) -> serde_json::Value {
    let config = state.config_read().clone();
    let dir = config.collect.resolved_review_dir(&config.data_root());
    let mut v = serde_json::json!({
        "success": true,
        "folder_missing": false,
        "spots": [],
        "truncated": false,
        "totals": { "spots": 0, "labelled_boxes": 0, "open_adds": 0 },
    });
    if let (Some(o), Some(c)) = (v.as_object_mut(), cap_bits(state).as_object()) {
        o.extend(c.clone());
    }
    let Some(o) = v.as_object_mut() else {
        return v;
    };
    // An unplugged drive is never an empty dataset: nothing is read from it.
    if let Some(why) = config.collect.review_folder_problem(&dir) {
        o.insert("folder_missing".into(), true.into());
        o.insert("message".into(), why.into());
        return v;
    }
    let stamp = queue_stamp(&training::tasks(state));
    let all = computed(
        &dir,
        &config.collect.classes,
        &config.collect.hidden_classes,
        &stamp,
    );
    let mut order: Vec<&SpotSummary> = all.spots.iter().collect();
    // Where boxes are missing first, then the biggest places.
    order.sort_by(|a, b| {
        b.open_adds
            .cmp(&a.open_adds)
            .then(b.spot.frames.cmp(&a.spot.frames))
            .then(a.spot.id.cmp(&b.spot.id))
    });
    o.insert(
        "totals".into(),
        serde_json::json!({
            "spots": all.spots.len(),
            "labelled_boxes": all.spots.iter().map(|s| s.labelled).sum::<usize>(),
            "open_adds": all.spots.iter().map(|s| s.open_adds).sum::<usize>(),
        }),
    );
    o.insert("truncated".into(), (order.len() > MAX_SPOTS).into());
    o.insert(
        "spots".into(),
        order
            .into_iter()
            .take(MAX_SPOTS)
            .map(|s| spot_json(s, all.samples.get(&s.spot.id)))
            .collect::<Vec<_>>()
            .into(),
    );
    v
}

/// `GET /v1/refine/spots`: see the module docs.
pub(super) async fn spots(State(state): State<Arc<AppState>>) -> Response {
    blocking(move || Json(spots_of(&state)).into_response()).await
}

// ---------------------------------------------------------------------------------------------
// POST /v1/refine/accept-spot

/// Body of `POST /v1/refine/accept-spot`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AcceptBody {
    spot: String,
    #[serde(default)]
    min_score: Option<f32>,
    #[serde(default)]
    dry_run: bool,
    #[serde(default)]
    limit: Option<usize>,
}

/// What an accept did (or, in a dry run, would do).
#[derive(Debug, Default, PartialEq, Eq)]
struct Outcome {
    /// Boxes added (the frames' first `limit` with something to add).
    accepted: usize,
    /// Frames written (a dry run: that would be).
    frames: usize,
    /// Frames with something to add, also past the limit.
    eligible: usize,
    /// Live `add`s of the spot under the minimum score, left for one-by-one review.
    below_score: usize,
    /// Reviewed frames a reviewer confirmed empty with a live `add` at the spot: left for
    /// one-by-one review (an accept would end their "empty").
    left_empty: usize,
    /// Frames that changed between the look and the write (nothing left to add): passed over.
    stale: usize,
    /// The first frames not changed, `<frame>: <why>`.
    failed: Vec<String>,
    failed_count: usize,
    /// The frames written (a dry run: that would be), in the order of the walk.
    written: Vec<String>,
}

/// Why a spot accept did not start.
#[derive(Debug, PartialEq, Eq)]
enum Refusal {
    NoSuchSpot,
}

/// What [`accept_at_spot`] did to one frame.
#[derive(Debug, Default, PartialEq, Eq)]
struct AtSpot {
    /// Boxes added.
    added: usize,
    /// Live `add`s at the spot left: under `min`, or no longer fitting once the others were added.
    left: usize,
    /// Live `add`s at the spot on a frame confirmed empty, all left (nothing added there).
    on_empty: usize,
}

/// The spot of a bulk accept: its id in `table` (the folder's spots now), and the frame's place.
#[derive(Clone, Copy)]
struct Place<'a> {
    table: &'a statics::Table,
    id: &'a str,
}

/// Accept the live `add`s at spot `p` of `f` (a frame at `at`) that score `min` or more, through
/// [`suggestions::accept`]; only on a reviewed frame, never on one a reviewer confirmed empty
/// ([`AtSpot::on_empty`]: one by one, a reviewer looks at the frame; in bulk about 1 in 10 such
/// suggestions would put a wrong box on a frame somebody said was empty).
fn accept_at_spot(
    f: &mut LabelFile,
    p: Place<'_>,
    at: statics::FrameAt<'_>,
    wanted: &Wanted<'_>,
    min: f32,
) -> AtSpot {
    if f.status != Status::Reviewed {
        return AtSpot::default();
    }
    let live = |f: &LabelFile| statics::live_adds_at(f, p.table, at, p.id, wanted);
    if f.empty_confirmed {
        return AtSpot {
            on_empty: live(f).len(),
            ..AtSpot::default()
        };
    }
    let mut added = 0;
    // Highest index first: an accept removes its suggestion, so the lower ones stay valid.
    for i in live(f) {
        let strong = f
            .refine
            .get(i)
            .and_then(|s| s.score)
            .is_some_and(|s| s >= min);
        if strong && suggestions::accept(f, i, Some(wanted)) {
            added += 1;
        }
    }
    AtSpot {
        added,
        left: live(f).len(),
        on_empty: 0,
    }
}

/// [`accept_at_spot`] over the frames `spot` lists as open in a fresh walk of `dir`: see the
/// module docs. `limit` frames are written at most.
fn accept_spot(
    dir: &Path,
    wanted: &Wanted<'_>,
    spot: &str,
    min: f32,
    limit: usize,
    dry_run: bool,
) -> Result<Outcome, Refusal> {
    let all = compute(dir, wanted.classes, wanted.hidden);
    let Some(found) = all.spots.iter().find(|s| s.spot.id == spot) else {
        return Err(Refusal::NoSuchSpot);
    };
    let place = Place {
        table: &all.table,
        id: spot,
    };
    let camera = found.spot.camera.as_str();
    let mut out = Outcome::default();
    let mut todo: Vec<(&statics::OpenFrame, usize)> = Vec::new();
    for open in &found.open_frames {
        let rel = &open.rel;
        let at = statics::FrameAt {
            camera,
            width: open.width,
            height: open.height,
        };
        let read = Frame::resolve(dir, rel)
            .ok_or_else(|| "the frame is gone".to_string())
            .and_then(|fr| {
                labels::read(&fr.labels)
                    .map_err(|e| format!("{e:#}"))?
                    .ok_or_else(|| "the label file is gone".to_string())
            });
        let mut f = match read {
            Ok(f) => f,
            Err(why) => {
                out.failed_count += 1;
                if out.failed.len() < LISTED {
                    out.failed.push(format!("{rel}: {why}"));
                }
                continue;
            }
        };
        // On the file as read: the dry run's answer, and the frames worth a write.
        let did = accept_at_spot(&mut f, place, at, wanted, min);
        out.below_score += did.left;
        out.left_empty += usize::from(did.on_empty > 0);
        if did.added > 0 {
            todo.push((open, did.added));
        }
    }
    out.eligible = todo.len();
    todo.truncate(limit);
    if dry_run {
        out.accepted = todo.iter().map(|(_, n)| n).sum();
        out.frames = todo.len();
        out.written = todo.iter().map(|(o, _)| o.rel.clone()).collect();
        return Ok(out);
    }
    for (open, _) in todo {
        let rel = open.rel.as_str();
        let at = statics::FrameAt {
            camera,
            width: open.width,
            height: open.height,
        };
        let Some(fr) = Frame::resolve(dir, rel) else {
            out.failed_count += 1;
            if out.failed.len() < LISTED {
                out.failed.push(format!("{rel}: the frame is gone"));
            }
            continue;
        };
        let mut added = 0;
        // Judged again under the write lock: the file may have changed since it was read above.
        let r = labels::update(&fr.labels, |cur| {
            let Some(mut f) = cur else {
                return Ok(None);
            };
            added = accept_at_spot(&mut f, place, at, wanted, min).added;
            Ok((added > 0).then_some(f))
        });
        match r {
            Ok(true) => {
                out.accepted += added;
                out.frames += 1;
                out.written.push(rel.to_string());
            }
            Ok(false) => out.stale += 1,
            Err(e) => {
                out.failed_count += 1;
                if out.failed.len() < LISTED {
                    out.failed.push(format!("{rel}: {e:#}"));
                }
            }
        }
    }
    Ok(out)
}

/// Up to `n` of `list` spread over it (the walk lists frames by day, so over the days).
fn spread(list: &[String], n: usize) -> Vec<String> {
    if list.len() <= n {
        return list.to_vec();
    }
    (0..n)
        .filter_map(|i| list.get(i * list.len() / n).cloned())
        .collect()
}

fn score_text(min: f32) -> String {
    let t = format!("{min:.2}");
    t.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// `POST /v1/refine/accept-spot` with `{spot, min_score?, dry_run?, limit?}`: see the module docs.
/// 200 `{success, dry_run, spot, min_score, accepted, frames, eligible, below_score, left_empty,
/// stale, failed, failed_count, sample, message}`; 400 for a bad field, 404 for a spot the reviewed frames
/// do not have (it may have moved with a label change: the card reloads), 503 for an unplugged
/// review folder.
pub(super) async fn accept_spot_route(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: AcceptBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((status, message)) => return fail(status, message),
    };
    let min = b.min_score.unwrap_or(DEFAULT_MIN_SCORE);
    if !(0.0..=1.0).contains(&min) {
        return fail(
            StatusCode::BAD_REQUEST,
            format!("min_score: must be between 0 and 1, got {min}"),
        );
    }
    let limit = b.limit.unwrap_or(ACCEPT_LIMIT);
    if !(1..=ACCEPT_LIMIT).contains(&limit) {
        return fail(
            StatusCode::BAD_REQUEST,
            format!("limit: must be between 1 and {ACCEPT_LIMIT}, got {limit}"),
        );
    }
    if b.spot.trim().is_empty() || b.spot.len() > suggestions::MAX_SPOT_LEN {
        return fail(
            StatusCode::BAD_REQUEST,
            "spot: name a spot of the list (its id)",
        );
    }
    let dry_run = b.dry_run;
    blocking(move || {
        let config = state.config_read().clone();
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
        let wanted = Wanted {
            classes: &config.collect.classes,
            hidden: &config.collect.hidden_classes,
        };
        let o = match accept_spot(&dir, &wanted, &b.spot, min, limit, dry_run) {
            Ok(o) => o,
            Err(Refusal::NoSuchSpot) => {
                return fail(
                    StatusCode::NOT_FOUND,
                    "No static spot with this id on the reviewed frames: a label change can move \
                     or drop a spot. Reload the card and choose again.",
                );
            }
        };
        if !dry_run {
            // Label writes tell the Review index and the repeats cache themselves; the tab's own
            // answers are dropped here.
            forget_counts();
            forget();
        }
        let weak = score_text(min);
        let mut message = if dry_run {
            format!(
                "Would add {} box(es) on {} frame(s) at this spot (score {weak} or more).",
                o.accepted, o.frames
            )
        } else {
            format!(
                "Added {} box(es) on {} frame(s) at this spot (score {weak} or more; the frames \
                 stay reviewed).",
                o.accepted, o.frames
            )
        };
        if o.below_score > 0 {
            message.push_str(&format!(
                " {} weaker suggestion(s) stay for one-by-one review.",
                o.below_score
            ));
        }
        if o.left_empty > 0 {
            message.push_str(&format!(
                " {} frame(s) confirmed empty are left for one-by-one review.",
                o.left_empty
            ));
        }
        if o.stale > 0 {
            message.push_str(&format!(
                " {} frame(s) changed meanwhile and were passed over.",
                o.stale
            ));
        }
        push_failed(
            &mut message,
            o.failed_count,
            "frame(s) not changed",
            "review",
        );
        push_past_limit(&mut message, o.eligible, limit, "more frame(s)");
        if !dry_run {
            tracing::info!(
                spot = %b.spot,
                accepted = o.accepted,
                frames = o.frames,
                failed = o.failed_count,
                below = o.below_score,
                "refine: static spot suggestions accepted"
            );
        }
        Json(serde_json::json!({
            "success": true,
            "dry_run": dry_run,
            "spot": b.spot,
            "min_score": min,
            "accepted": o.accepted,
            "frames": o.frames,
            "eligible": o.eligible,
            "below_score": o.below_score,
            "left_empty": o.left_empty,
            "stale": o.stale,
            "failed": o.failed,
            "failed_count": o.failed_count,
            "sample": spread(&o.written, SAMPLE),
            "message": message,
        }))
        .into_response()
    })
    .await
}

// ---------------------------------------------------------------------------------------------
// POST /v1/refine/static-cap

/// Body of `POST /v1/refine/static-cap`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CapBody {
    cap: i64,
}

/// `POST /v1/refine/static-cap` with `{cap}` (0 = off, at most [`MAX_STATIC_CAP`]): saves
/// `training.static_cap` through [`save_config`]; the next training run reads it. 200 `{success,
/// message, cap}`, 400 for a number out of bounds.
pub(super) async fn static_cap(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let b: CapBody = match json_body(&headers, body) {
        Ok(b) => b,
        Err((status, message)) => return fail(status, message),
    };
    let Some(cap) = usize::try_from(b.cap).ok().filter(|c| *c <= MAX_STATIC_CAP) else {
        return fail(
            StatusCode::BAD_REQUEST,
            format!(
                "Not saved: the cap must be a whole number from 0 (off) to {MAX_STATIC_CAP}, got {}",
                b.cap
            ),
        );
    };
    blocking(move || {
        let r = save_config(&state, "refine page", |c| {
            c.training.static_cap = cap;
            c.training.validate()
        });
        match r {
            Ok(message) => Json(serde_json::json!({
                "success": true,
                "message": message,
                "cap": cap,
            }))
            .into_response(),
            Err(e) => fail(StatusCode::BAD_REQUEST, format!("Not saved: {e:#}")),
        }
    })
    .await
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::test_util::{call, test_state_with};
    use super::*;
    use crate::collect::labels::bulk::test_support::put_frame_at;
    use crate::collect::labels::refine::{Answer, Kind, Suggestion};
    use crate::collect::labels::{BoxState, LabelBox, Source};
    use axum::body::Body;
    use axum::http::{Request, header};
    use std::path::PathBuf;

    fn scratch() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("bo_spots_{}", uuid::Uuid::new_v4()));
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

    fn car(bbox: [f32; 4]) -> LabelBox {
        LabelBox {
            label: "car".into(),
            bbox,
            source: Source::Auto,
            score: Some(0.8),
            state: BoxState::Accepted,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: Default::default(),
        }
    }

    /// 64 x 48 frames: the parked car, where the checks found it missing on frames without its
    /// box (a little off, as a detection is), and a place elsewhere (no spot).
    const PARKED: [f32; 4] = [8.0, 24.0, 28.0, 40.0];
    const FOUND: [f32; 4] = [9.0, 24.0, 29.0, 40.0];
    const MISSING: [f32; 4] = [40.0, 4.0, 60.0, 16.0];

    fn add_at(bbox: [f32; 4], score: f32, spot: Option<&str>) -> Suggestion {
        Suggestion {
            kind: Kind::Add,
            target: None,
            target_label: None,
            bbox: Some(bbox),
            label: Some("car".into()),
            score: Some(score),
            why: "a fold model found it".into(),
            spot: spot.map(str::to_string),
            answer: None,
            run: "r1".into(),
        }
    }

    /// An add at the spot.
    fn add(score: f32, spot: Option<&str>) -> Suggestion {
        add_at(FOUND, score, spot)
    }

    /// A reviewed frame of camera `camera` on `day` with the parked car and `list`.
    fn frame(dir: &Path, camera: &str, day: &str, stem: &str, list: Vec<Suggestion>) -> String {
        frame_with(dir, camera, day, stem, vec![car(PARKED)], list)
    }

    /// A reviewed frame of camera `camera` on `day` with `boxes` and `list`.
    fn frame_with(
        dir: &Path,
        camera: &str,
        day: &str,
        stem: &str,
        boxes: Vec<LabelBox>,
        list: Vec<Suggestion>,
    ) -> String {
        // The test support writes day 2026-10-21; a second day is a moved copy.
        let rel = put_frame_at(dir, camera, stem, (64, 48), None, boxes);
        let rel = if day == "2026-10-21" {
            rel
        } else {
            let from = dir.join("ir").join(camera).join("2026-10-21");
            let to = dir.join("ir").join(camera).join(day);
            std::fs::create_dir_all(&to).unwrap();
            for ext in ["jpg", "json", "labels.json"] {
                std::fs::rename(
                    from.join(format!("{stem}.{ext}")),
                    to.join(format!("{stem}.{ext}")),
                )
                .unwrap();
            }
            format!("ir/{camera}/{day}/{stem}.jpg")
        };
        let path = Frame::resolve(dir, &rel).unwrap().labels;
        let mut f = labels::read(&path).unwrap().unwrap();
        f.status = Status::Reviewed;
        f.reviewed_at = Some("2026-10-22T00:00:00Z".into());
        f.refine = list;
        labels::write(&path, &f).unwrap();
        rel
    }

    /// The spot id the folder's frames make: the id is the walk's, so ask it.
    fn the_spot(dir: &Path) -> String {
        let classes = vec!["car".to_string(), "person".to_string()];
        let none: Vec<String> = Vec::new();
        let c = compute(dir, &classes, &none);
        assert_eq!(c.spots.len(), 1, "{:?}", c.spots);
        c.spots[0].spot.id.clone()
    }

    /// 12 frames on 2 days with a parked car (`c00`..`c11`: the spot), and 12 reviewed frames
    /// without its box (`a00`..`a11`, returned) where the check found it missing. The suggestions
    /// come with the spot's id once it is known (the first pass makes the spot, the second writes
    /// the suggestions). Strong adds at the spot on 6 frames of both days, a weak one on 2, an
    /// answered one on 1, a strong one on frame 10 that names another spot id (a check's table
    /// differs from the folder's now: still at this spot by geometry), and one on frame 11 that
    /// names this spot's id but lies elsewhere (at no spot).
    fn folder(dir: &Path) -> (PathBuf, String, Vec<String>) {
        let review = dir.join("dataset");
        let mut rels = Vec::new();
        for i in 0..12 {
            let day = if i < 6 { "2026-10-21" } else { "2026-10-22" };
            frame(&review, "yard", day, &format!("c{i:02}"), vec![]);
            rels.push(frame_with(
                &review,
                "yard",
                day,
                &format!("a{i:02}"),
                vec![],
                vec![],
            ));
        }
        let id = the_spot(&review);
        let write = |rel: &str, list: Vec<Suggestion>| {
            let p = Frame::resolve(&review, rel).unwrap().labels;
            labels::update(&p, |cur| {
                let mut f = cur.unwrap();
                f.refine = list;
                Ok(Some(f))
            })
            .unwrap();
        };
        for rel in [0, 1, 2, 6, 7, 8] {
            write(&rels[rel], vec![add(0.8, Some(&id))]);
        }
        for rel in [3, 9] {
            write(&rels[rel], vec![add(0.3, Some(&id))]);
        }
        let mut answered = add(0.9, Some(&id));
        answered.answer = Some(Answer::Dismissed);
        write(&rels[4], vec![answered]);
        write(&rels[10], vec![add(0.9, Some("yard/vehicle/4-16-14-26~7"))]);
        write(&rels[11], vec![add_at(MISSING, 0.9, Some(&id))]);
        (review, id, rels)
    }

    /// The frames of `folder` the bulk accept at 0.5 takes.
    const STRONG: [usize; 7] = [0, 1, 2, 6, 7, 8, 10];

    #[tokio::test]
    async fn the_routes_need_a_sign_in_a_same_site_post_and_valid_input() {
        let dir = scratch();
        let state = state_in(&dir);
        for (method, uri, body) in [
            ("GET", "/v1/refine/spots", ""),
            ("POST", "/v1/refine/accept-spot", r#"{"spot":"x"}"#),
            ("POST", "/v1/refine/static-cap", r#"{"cap":5}"#),
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
                let form = super::super::test_util::post_form(uri, "cap=5", false);
                let (s, _) = json(&state, form).await;
                assert_eq!(s, StatusCode::UNSUPPORTED_MEDIA_TYPE, "{uri}");
            }
        }
        // Bad input: refused with a message, nothing written.
        for body in [
            "not json",
            r#"{}"#,
            r#"{"spot":""}"#,
            r#"{"spot":"x","extra":1}"#,
            r#"{"spot":"x","min_score":1.5}"#,
            r#"{"spot":"x","min_score":-0.1}"#,
            r#"{"spot":"x","limit":0}"#,
            r#"{"spot":"x","limit":501}"#,
            r#"{"spot":7}"#,
        ] {
            let (s, v) = json(&state, post_json("/v1/refine/accept-spot", body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}: {v}");
            assert_eq!(v["success"], false);
            assert!(v["message"].as_str().is_some_and(|m| !m.is_empty()));
        }
        // A body over 4 KiB.
        let big = format!(r#"{{"spot":"{}"}}"#, "x".repeat(5000));
        let (s, _) = json(&state, post_json("/v1/refine/accept-spot", &big)).await;
        assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn a_spot_that_does_not_exist_is_a_404_and_an_unplugged_folder_a_503() {
        let dir = scratch();
        let (review, _, _) = folder(&dir);
        let _ = review;
        let state = state_in(&dir);
        let (s, v) = json(
            &state,
            post_json("/v1/refine/accept-spot", r#"{"spot":"yard/car/0-0-1-1"}"#),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{v}");
        assert_eq!(v["success"], false);
        let d = dir.clone();
        let gone = test_state_with(move |c| {
            c.download_dir = Some(d.clone());
            c.collect.review_dir = Some(d.join("unplugged"));
            c.training.dir = Some(d.join("training"));
        });
        let (s, v) = json(
            &gone,
            post_json("/v1/refine/accept-spot", r#"{"spot":"a"}"#),
        )
        .await;
        assert_eq!(
            (s, v["folder_missing"].as_bool()),
            (StatusCode::SERVICE_UNAVAILABLE, Some(true))
        );
        let (s, v) = json(&gone, get("/v1/refine/spots")).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["folder_missing"], true);
        assert!(v["spots"].as_array().unwrap().is_empty());
        assert_eq!(v["static_cap"], 20);
    }

    #[tokio::test]
    async fn the_listing_has_the_spot_with_its_counts_a_sample_and_a_crop() {
        let dir = scratch();
        let (_, id, _) = folder(&dir);
        let state = state_in(&dir);
        let (s, v) = json(&state, get("/v1/refine/spots")).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["folder_missing"], false);
        let list = v["spots"].as_array().unwrap();
        assert_eq!(list.len(), 1, "{v}");
        let sp = &list[0];
        assert_eq!(sp["id"], id.as_str());
        assert_eq!(
            (sp["camera"].as_str(), sp["label"].as_str()),
            (Some("yard"), Some("car"))
        );
        assert_eq!(
            (
                sp["frames"].as_u64(),
                sp["days"].as_u64(),
                sp["stay_days"].as_u64()
            ),
            (Some(12), Some(2), Some(2))
        );
        assert_eq!(sp["labelled"], 12);
        // 6 strong + 2 weak live adds at this spot, and the strong one naming another spot id
        // (geometry decides); not the answered one, nor the one naming this id elsewhere.
        assert_eq!(sp["open_adds"], 9, "{sp}");
        assert!(sp.get("open_frames").is_none());
        assert!(sp.get("day_frames").is_none());
        assert_eq!(sp["sample_size"], serde_json::json!([64, 48]));
        // The newest frame labelled there: the later day, the last stem.
        assert_eq!(sp["sample"], "ir/yard/2026-10-22/c11.jpg");
        let crop: Vec<f64> = sp["crop"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let b: Vec<f64> = sp["bbox"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        assert!(crop[0] <= b[0] && crop[1] <= b[1] && crop[2] >= b[2] && crop[3] >= b[3]);
        assert!(crop.iter().all(|c| (0.0..=1.0).contains(c)), "{crop:?}");
        assert_eq!(
            v["totals"],
            serde_json::json!({"spots": 1, "labelled_boxes": 12, "open_adds": 9})
        );
        assert_eq!(v["static_cap"], 20);
        assert_eq!(v["static_cap_default"], 20);
        assert_eq!(v["static_cap_max"], 100_000);
        // The image the page shows is served by the existing frame route.
        let (s, h, _) = call(
            &state,
            get(&format!(
                "/dataset/image?path={}",
                sp["sample"].as_str().unwrap()
            )),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(h[header::CONTENT_TYPE], "image/jpeg");
    }

    #[test]
    fn the_listing_is_kept_until_the_queue_changes_or_a_write_drops_it() {
        let dir = scratch();
        let (review, _, _) = folder(&dir);
        let classes = vec!["car".to_string(), "person".to_string()];
        let none: Vec<String> = Vec::new();
        // A cache of its own: other tests clear the process's at any time.
        let cache = Cache::new();
        let first = cache.computed(&review, &classes, &none, "");
        assert_eq!(first.spots.len(), 1);
        let again = cache.computed(&review, &classes, &none, "");
        assert!(Arc::ptr_eq(&first, &again), "kept");
        // The process's cache dropped meanwhile does not touch this one.
        forget();
        let kept = cache.computed(&review, &classes, &none, "");
        assert!(Arc::ptr_eq(&first, &kept), "another cache's forget");
        let other = cache.computed(&review, &classes, &none, "t1:running");
        assert!(!Arc::ptr_eq(&first, &other), "a check changed state");
        cache.forget();
        let third = cache.computed(&review, &classes, &none, "t1:running");
        assert!(!Arc::ptr_eq(&other, &third), "dropped");
    }

    fn files_of(dir: &Path, rels: &[String]) -> Vec<LabelFile> {
        rels.iter()
            .map(|r| {
                labels::read(&Frame::resolve(dir, r).unwrap().labels)
                    .unwrap()
                    .unwrap()
            })
            .collect()
    }

    /// [`accept_at_spot`] on a frame of `folder` (camera `yard`, 64 x 48) at spot `id` of `walk`.
    fn at_spot(
        f: &mut LabelFile,
        walk: &Computed,
        id: &str,
        wanted: &Wanted<'_>,
        min: f32,
    ) -> AtSpot {
        accept_at_spot(
            f,
            Place {
                table: &walk.table,
                id,
            },
            statics::FrameAt {
                camera: "yard",
                width: 64,
                height: 48,
            },
            wanted,
            min,
        )
    }

    #[test]
    fn a_dry_run_counts_what_the_apply_does_and_a_second_apply_does_nothing() {
        let dir = scratch();
        let (review, id, rels) = folder(&dir);
        let classes = vec!["car".to_string(), "person".to_string()];
        let none: Vec<String> = Vec::new();
        let wanted = Wanted {
            classes: &classes,
            hidden: &none,
        };
        let before = files_of(&review, &rels);
        let dry = accept_spot(&review, &wanted, &id, 0.5, 500, true).unwrap();
        assert_eq!(
            (
                dry.accepted,
                dry.frames,
                dry.eligible,
                dry.below_score,
                dry.left_empty
            ),
            (7, 7, 7, 2, 0),
            "{dry:?}"
        );
        assert_eq!(files_of(&review, &rels), before, "a dry run writes nothing");

        let done = accept_spot(&review, &wanted, &id, 0.5, 500, false).unwrap();
        assert_eq!(
            (done.accepted, done.frames, done.eligible, done.below_score),
            (dry.accepted, dry.frames, dry.eligible, dry.below_score)
        );
        assert_eq!(done.written, dry.written);
        assert_eq!((done.stale, done.failed_count), (0, 0));

        let after = files_of(&review, &rels);
        for (i, (b, a)) in before.iter().zip(&after).enumerate() {
            if STRONG.contains(&i) {
                // Reviewed still, one box more with reason refine, its suggestion gone. Frame 10's
                // suggestion named another spot id: at this spot by its place, so it is taken.
                assert_eq!(a.status, Status::Reviewed, "{i}");
                assert_eq!(a.reviewed_at, b.reviewed_at, "{i}");
                assert_eq!(a.boxes.len(), b.boxes.len() + 1, "{i}");
                assert_eq!(a.boxes[..b.boxes.len()], b.boxes[..], "{i}");
                let new = a.boxes.last().unwrap();
                assert_eq!(new.reason.as_deref(), Some("refine"));
                assert_eq!((new.state, new.source), (BoxState::Accepted, Source::Auto));
                assert_eq!(new.bbox, FOUND);
                assert!(a.refine.is_empty(), "{i}");
                let mut expect = b.clone();
                expect.boxes = a.boxes.clone();
                expect.refine = Vec::new();
                expect.empty_confirmed = a.empty_confirmed;
                assert_eq!(*a, expect, "nothing else changes ({i})");
            } else {
                // Frame 11's suggestion names this spot but lies elsewhere: left alone.
                assert_eq!(a, b, "frame {i} is untouched");
            }
        }
        // A second apply (and a dry run) find nothing to add; the weak ones are still open.
        for dry_run in [true, false] {
            let again = accept_spot(&review, &wanted, &id, 0.5, 500, dry_run).unwrap();
            assert_eq!(
                (
                    again.accepted,
                    again.frames,
                    again.eligible,
                    again.below_score
                ),
                (0, 0, 0, 2),
                "{again:?}"
            );
        }
        // A lower minimum takes the weak ones.
        let low = accept_spot(&review, &wanted, &id, 0.2, 500, false).unwrap();
        assert_eq!((low.accepted, low.frames, low.below_score), (2, 2, 0));
    }

    #[test]
    fn suggestions_are_placed_by_geometry_not_by_the_id_they_carry() {
        let dir = scratch();
        let (review, id, rels) = folder(&dir);
        let classes = vec!["car".to_string(), "person".to_string()];
        let none: Vec<String> = Vec::new();
        let wanted = Wanted {
            classes: &classes,
            hidden: &none,
        };
        let walk = compute(&review, &classes, &none);
        let read = |i: usize| {
            labels::read(&Frame::resolve(&review, &rels[i]).unwrap().labels)
                .unwrap()
                .unwrap()
        };
        let at = statics::FrameAt {
            camera: "yard",
            width: 64,
            height: 48,
        };
        // A foreign id at the spot: found there; the spot's own id elsewhere: not.
        assert_eq!(
            statics::live_adds_at(&read(10), &walk.table, at, &id, &wanted),
            [0]
        );
        assert!(statics::live_adds_at(&read(11), &walk.table, at, &id, &wanted).is_empty());
        // An add without an id lies at the spot just the same.
        let mut f = read(10);
        f.refine[0].spot = None;
        assert_eq!(
            statics::live_adds_at(&f, &walk.table, at, &id, &wanted),
            [0]
        );
        let spot = walk.spots.iter().find(|s| s.spot.id == id).unwrap();
        let open: Vec<&str> = spot.open_frames.iter().map(|o| o.rel.as_str()).collect();
        assert!(open.contains(&rels[10].as_str()), "{open:?}");
        assert!(!open.contains(&rels[11].as_str()), "{open:?}");
        assert!(
            spot.open_frames
                .iter()
                .all(|o| (o.width, o.height) == (64, 48))
        );
        // Its accept takes frame 10, never frame 11.
        let mut f = read(10);
        assert_eq!(at_spot(&mut f, &walk, &id, &wanted, 0.5).added, 1);
        let mut f = read(11);
        assert_eq!(at_spot(&mut f, &walk, &id, &wanted, 0.0), AtSpot::default());
    }

    #[test]
    fn frames_confirmed_empty_are_left_for_one_by_one_review() {
        let dir = scratch();
        let (review, id, rels) = folder(&dir);
        let classes = vec!["car".to_string(), "person".to_string()];
        let none: Vec<String> = Vec::new();
        let wanted = Wanted {
            classes: &classes,
            hidden: &none,
        };
        let path = Frame::resolve(&review, &rels[6]).unwrap().labels;
        labels::update(&path, |c| {
            let mut f = c.unwrap();
            f.empty_confirmed = true;
            Ok(Some(f))
        })
        .unwrap();
        let before = labels::read(&path).unwrap().unwrap();
        let dry = accept_spot(&review, &wanted, &id, 0.5, 500, true).unwrap();
        let done = accept_spot(&review, &wanted, &id, 0.5, 500, false).unwrap();
        for o in [&dry, &done] {
            assert_eq!(
                (o.accepted, o.frames, o.left_empty, o.below_score),
                (6, 6, 1, 2),
                "{o:?}"
            );
        }
        assert_eq!(done.written, dry.written);
        assert!(!done.written.contains(&rels[6]));
        assert_eq!(labels::read(&path).unwrap().unwrap(), before, "untouched");
        // Still there for one-by-one review.
        let walk = compute(&review, &classes, &none);
        let mut f = labels::read(&path).unwrap().unwrap();
        assert_eq!(
            at_spot(&mut f, &walk, &id, &wanted, 0.0),
            AtSpot {
                added: 0,
                left: 0,
                on_empty: 1
            }
        );
        assert_eq!(f, before);
    }

    #[test]
    fn the_limit_caps_the_frames_written_and_the_sample_spreads_over_the_days() {
        let dir = scratch();
        let (review, id, rels) = folder(&dir);
        let classes = vec!["car".to_string(), "person".to_string()];
        let none: Vec<String> = Vec::new();
        let wanted = Wanted {
            classes: &classes,
            hidden: &none,
        };
        let dry = accept_spot(&review, &wanted, &id, 0.5, 4, true).unwrap();
        let done = accept_spot(&review, &wanted, &id, 0.5, 4, false).unwrap();
        assert_eq!((dry.frames, dry.accepted, dry.eligible), (4, 4, 7));
        assert_eq!((done.frames, done.accepted, done.eligible), (4, 4, 7));
        assert_eq!(done.written, dry.written);
        // The rest is open: another call takes it.
        let rest = accept_spot(&review, &wanted, &id, 0.5, 4, false).unwrap();
        assert_eq!((rest.frames, rest.eligible), (3, 3));
        let _ = rels;

        let many: Vec<String> = (0..37).map(|i| format!("f{i}")).collect();
        let s = spread(&many, SAMPLE);
        assert_eq!(s.len(), SAMPLE);
        assert_eq!(s[0], "f0");
        assert!(s.windows(2).all(|w| w[0] < w[1] || w[0].len() < w[1].len()));
        assert_eq!(spread(&many[..3], SAMPLE).len(), 3);
    }

    #[test]
    fn stale_answered_and_unreviewed_frames_are_passed_over_cleanly() {
        let dir = scratch();
        let (review, id, rels) = folder(&dir);
        let classes = vec!["car".to_string(), "person".to_string()];
        let none: Vec<String> = Vec::new();
        let wanted = Wanted {
            classes: &classes,
            hidden: &none,
        };
        let path = |i: usize| Frame::resolve(&review, &rels[i]).unwrap().labels;
        // Frame 0 is no longer reviewed, frame 1's suggestion was answered, frame 2 got a box
        // that covers it: none of them is live, so a dry run does not count them.
        labels::update(&path(0), |c| {
            let mut f = c.unwrap();
            f.status = Status::Unreviewed;
            Ok(Some(f))
        })
        .unwrap();
        labels::update(&path(1), |c| {
            let mut f = c.unwrap();
            f.refine[0].answer = Some(Answer::Dismissed);
            Ok(Some(f))
        })
        .unwrap();
        labels::update(&path(2), |c| {
            let mut f = c.unwrap();
            f.boxes.push(car(FOUND));
            Ok(Some(f))
        })
        .unwrap();
        let dry = accept_spot(&review, &wanted, &id, 0.5, 500, true).unwrap();
        let done = accept_spot(&review, &wanted, &id, 0.5, 500, false).unwrap();
        // 7 strong, minus the three above: 4 left; frame 0 is not reviewed so it is no
        // candidate and its weak/strong add is not counted either.
        assert_eq!((dry.accepted, dry.frames), (4, 4), "{dry:?}");
        assert_eq!((done.accepted, done.frames), (4, 4));
        let f0 = labels::read(&path(0)).unwrap().unwrap();
        assert_eq!(
            (f0.status, f0.refine.len()),
            (Status::Unreviewed, 1),
            "untouched"
        );

        // A frame that changes between the look and the write is "stale", not an error.
        let walk = compute(&review, &classes, &none);
        let spot = walk.spots.iter().find(|s| s.spot.id == id).unwrap();
        assert_eq!(
            spot.open_adds, 2,
            "only the weak ones are live reviewed adds now: {spot:?}"
        );
        // The write path alone: accept_at_spot on a frame that lost its suggestion adds nothing.
        let mut f = labels::read(&path(6)).unwrap().unwrap();
        f.refine.clear();
        assert_eq!(at_spot(&mut f, &walk, &id, &wanted, 0.5), AtSpot::default());
        // Unreviewed and skipped frames get nothing.
        let mut f = labels::read(&path(0)).unwrap().unwrap();
        assert_eq!(at_spot(&mut f, &walk, &id, &wanted, 0.0), AtSpot::default());
        f.status = Status::Skipped;
        assert_eq!(at_spot(&mut f, &walk, &id, &wanted, 0.0), AtSpot::default());
    }

    #[tokio::test]
    async fn the_route_answers_dry_run_apply_and_a_second_apply() {
        let dir = scratch();
        let (review, id, rels) = folder(&dir);
        // Frame 3 (a weak add) is confirmed empty: named, left alone.
        let p3 = Frame::resolve(&review, &rels[3]).unwrap().labels;
        labels::update(&p3, |c| {
            let mut f = c.unwrap();
            f.empty_confirmed = true;
            Ok(Some(f))
        })
        .unwrap();
        let state = state_in(&dir);
        let body = |dry: bool| {
            serde_json::json!({"spot": id, "dry_run": dry, "min_score": 0.5}).to_string()
        };
        let (s, dry) = json(&state, post_json("/v1/refine/accept-spot", &body(true))).await;
        assert_eq!(s, StatusCode::OK, "{dry}");
        assert_eq!(
            (dry["accepted"].as_u64(), dry["frames"].as_u64()),
            (Some(7), Some(7))
        );
        assert_eq!(dry["below_score"], 1);
        assert_eq!(dry["left_empty"], 1);
        assert_eq!(dry["dry_run"], true);
        assert_eq!(dry["sample"].as_array().unwrap().len(), 7);
        let msg = dry["message"].as_str().unwrap();
        assert!(
            msg.starts_with("Would add 7 box(es) on 7 frame(s)"),
            "{msg}"
        );
        assert!(
            msg.contains("1 weaker suggestion(s) stay for one-by-one review"),
            "{msg}"
        );
        assert!(
            msg.contains("1 frame(s) confirmed empty are left for one-by-one review."),
            "{msg}"
        );
        // The card's own count: 9 open adds before.
        let (_, v) = json(&state, get("/v1/refine/spots")).await;
        assert_eq!(v["spots"][0]["open_adds"], 9);

        let (s, done) = json(&state, post_json("/v1/refine/accept-spot", &body(false))).await;
        assert_eq!(s, StatusCode::OK, "{done}");
        assert_eq!(done["accepted"], dry["accepted"]);
        assert_eq!(done["frames"], dry["frames"]);
        assert_eq!(done["sample"], dry["sample"]);
        assert_eq!(done["left_empty"], 1);
        assert!(
            done["message"]
                .as_str()
                .unwrap()
                .starts_with("Added 7 box(es) on 7 frame(s)")
        );
        assert_eq!(done["failed_count"], 0);
        // Those frames are reviewed with the new box.
        for rel in done["sample"].as_array().unwrap() {
            let f = labels::read(
                &Frame::resolve(&review, rel.as_str().unwrap())
                    .unwrap()
                    .labels,
            )
            .unwrap()
            .unwrap();
            assert_eq!(f.status, Status::Reviewed);
            assert_eq!(f.boxes.last().unwrap().reason.as_deref(), Some("refine"));
        }
        // The kept listing was dropped: the card sees the 2 weak ones only (one on the frame
        // confirmed empty), and 7 more frames labelled at the spot.
        let (_, v) = json(&state, get("/v1/refine/spots")).await;
        assert_eq!(v["spots"][0]["open_adds"], 2, "{v}");
        assert_eq!(v["spots"][0]["labelled"], 19);
        assert_eq!(v["spots"][0]["id"], id.as_str());
        let (_, again) = json(&state, post_json("/v1/refine/accept-spot", &body(false))).await;
        assert_eq!(
            (again["accepted"].as_u64(), again["frames"].as_u64()),
            (Some(0), Some(0))
        );
        assert_eq!(again["below_score"], 1);
    }

    #[tokio::test]
    async fn the_cap_is_saved_bounded_and_refused_out_of_range() {
        let dir = scratch();
        let state = state_in(&dir);
        assert_eq!(state.config_read().training.static_cap, 20);
        let (s, v) = json(&state, post_json("/v1/refine/static-cap", r#"{"cap":35}"#)).await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(
            (v["success"].as_bool(), v["cap"].as_u64()),
            (Some(true), Some(35))
        );
        assert_eq!(state.config_read().training.static_cap, 35);
        let (_, v) = json(&state, get("/v1/refine/spots")).await;
        assert_eq!(v["static_cap"], 35);
        // 0 = off, and the bounds are inclusive.
        for n in [0, 100_000] {
            let (s, _) = json(
                &state,
                post_json("/v1/refine/static-cap", &format!(r#"{{"cap":{n}}}"#)),
            )
            .await;
            assert_eq!(s, StatusCode::OK, "{n}");
            assert_eq!(state.config_read().training.static_cap, n);
        }
        for body in [
            r#"{"cap":-1}"#,
            r#"{"cap":100001}"#,
            r#"{"cap":2.5}"#,
            r#"{"cap":"7"}"#,
            r#"{}"#,
            r#"{"cap":1,"x":2}"#,
        ] {
            let (s, v) = json(&state, post_json("/v1/refine/static-cap", body)).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{body}: {v}");
            assert_eq!(v["success"], false);
            assert_eq!(state.config_read().training.static_cap, 100_000, "{body}");
        }
    }
}
