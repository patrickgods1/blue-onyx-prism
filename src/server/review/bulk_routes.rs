//! The Review tab's bulk tools: `POST /v1/review/accept-auto`, `.../reject-low`,
//! `.../accept-repeats`, `.../reject-stacked`, `.../restore-reopened`, `.../reject-other-class`,
//! `.../reject-unwanted-reviewed`, and the overlap check `.../suggest-overlaps`. Each walks the
//! folder under review ([`bulk`]); a `dry_run` only counts.

use super::{
    ACCEPT_AUTO_DEFAULT_LIMIT, ACCEPT_AUTO_MAX_LIMIT, Answer, AppState, blocking, bulk_filters,
    bulk_limit, context, fail, hidden, json_body, now, push_failed, push_past_limit, success,
};
use crate::collect::labels::bulk;
use crate::server::Shared;
use axum::body::Bytes;
use axum::extract::State;
use axum::extract::rejection::BytesRejection;
use axum::http::{HeaderMap, StatusCode};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AcceptAutoBody {
    #[serde(default)]
    bucket: Option<String>,
    #[serde(default)]
    scene: Option<String>,
    #[serde(default)]
    camera: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    dry_run: bool,
    #[serde(default)]
    limit: Option<usize>,
}

/// `POST /v1/review/accept-auto` with `{bucket?, scene?, camera?, label?, source?, dry_run,
/// limit?}` (empty = any): the unreviewed frames of those filters whose auto-label triage alone
/// settles them (`auto=accept`, [`labels::auto_accept_eligible`]) saved as reviewed, at most
/// `limit` (default 500; [`bulk::bulk_accept`]). `dry_run` only counts them.
pub(in crate::server) async fn accept_auto(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    let b: AcceptAutoBody = json_body(&headers, body)?;
    let limit = bulk_limit(b.limit, ACCEPT_AUTO_DEFAULT_LIMIT)?;
    let mut filters = bulk_filters(
        [b.bucket, b.scene, b.camera, b.label, b.source],
        &[("status", "unreviewed"), ("auto", "accept")],
    )?;
    let dry_run = b.dry_run;
    Ok(blocking(move || {
        let (dir, classes, _) = context(&state);
        filters.wanted = Some((classes.clone(), hidden(&state)));
        let o = bulk::bulk_accept(&dir, &filters, &classes, limit, &now(), dry_run, &|_| true);
        let eligible = o.eligible;
        if dry_run {
            return success(serde_json::json!({
                "eligible": eligible,
                "accepted": 0,
                "frames": o.done,
            }));
        }
        let mut message = format!(
            "Accepted {} frame(s) the auto-labelers settled (saved as reviewed).",
            o.done_count
        );
        push_failed(&mut message, o.failed_count, "not accepted", "review");
        push_past_limit(&mut message, eligible, limit, "more eligible");
        tracing::info!(
            accepted = o.done_count,
            failed = o.failed_count,
            eligible,
            "review: auto-labels bulk accepted"
        );
        success(serde_json::json!({
            "eligible": eligible,
            "accepted": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": message,
        }))
    })
    .await)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RejectLowBody {
    #[serde(default)]
    bucket: Option<String>,
    #[serde(default)]
    scene: Option<String>,
    #[serde(default)]
    camera: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    source: Option<String>,
    min_score: f32,
    #[serde(default)]
    dry_run: bool,
    #[serde(default)]
    limit: Option<usize>,
}

/// `POST /v1/review/reject-low` with `{bucket?, scene?, camera?, label?, source?, min_score,
/// dry_run, limit?}`: on the unreviewed frames of those filters, the pending auto boxes scoring
/// below `min_score` (0.01..=1) rejected with `reason: low_score` (kept as hard-negative
/// evidence; [`bulk::bulk_reject_below`]), at most `limit` frames (default 500). `dry_run`
/// only counts the frames and boxes.
pub(in crate::server) async fn reject_low(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    let b: RejectLowBody = json_body(&headers, body)?;
    if !(0.01..=1.0).contains(&b.min_score) {
        return Ok(fail(
            StatusCode::BAD_REQUEST,
            format!("min_score: must be between 0.01 and 1, got {}", b.min_score),
        ));
    }
    let limit = bulk_limit(b.limit, ACCEPT_AUTO_DEFAULT_LIMIT)?;
    let filters = bulk_filters(
        [b.bucket, b.scene, b.camera, b.label, b.source],
        &[("status", "unreviewed")],
    )?;
    let (dry_run, min) = (b.dry_run, b.min_score);
    Ok(blocking(move || {
        let (dir, _, _) = context(&state);
        let o = bulk::bulk_reject_below(&dir, &filters, min, limit, dry_run, &|_| true);
        let eligible = o.eligible;
        if dry_run {
            return success(serde_json::json!({
                "eligible": eligible,
                "boxes": o.boxes,
                "rejected": 0,
                "frames": o.done,
            }));
        }
        let mut message = format!(
            "Rejected {} auto box(es) scoring below {min} on {} frame(s) (kept as hard \
             negatives; the frames stay unreviewed).",
            o.boxes, o.done_count
        );
        push_failed(
            &mut message,
            o.failed_count,
            "frame(s) not changed",
            "review",
        );
        push_past_limit(&mut message, eligible, limit, "more frame(s)");
        tracing::info!(
            rejected = o.boxes,
            frames = o.done_count,
            failed = o.failed_count,
            eligible,
            min_score = min,
            "review: low-scoring auto boxes rejected"
        );
        success(serde_json::json!({
            "eligible": eligible,
            "boxes": o.boxes,
            "rejected": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": message,
        }))
    })
    .await)
}

/// `POST /v1/review/accept-repeats` with `{bucket?, scene?, camera?, label?, source?, dry_run,
/// limit?}`: the repeat-only frames of those filters (spot edits settled every box, none
/// pending; [`labels::repeats::repeats_only`]) saved as reviewed with `bulk_accepted`
/// ([`bulk::bulk_accept_repeats`]), at most `limit` (default 500). `dry_run` only counts.
pub(in crate::server) async fn accept_repeats(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    let b: AcceptAutoBody = json_body(&headers, body)?;
    let limit = bulk_limit(b.limit, ACCEPT_AUTO_DEFAULT_LIMIT)?;
    let filters = bulk_filters([b.bucket, b.scene, b.camera, b.label, b.source], &[])?;
    let dry_run = b.dry_run;
    Ok(blocking(move || {
        let (dir, classes, _) = context(&state);
        let o =
            bulk::bulk_accept_repeats(&dir, &filters, &classes, limit, &now(), dry_run, &|_| true);
        if dry_run {
            return success(serde_json::json!({
                "eligible": o.eligible,
                "accepted": 0,
                "frames": o.done,
            }));
        }
        let mut message = format!("Saved {} repeat-only frame(s) as reviewed.", o.done_count);
        push_failed(&mut message, o.failed_count, "not saved", "review");
        push_past_limit(&mut message, o.eligible, limit, "more");
        tracing::info!(
            saved = o.done_count,
            failed = o.failed_count,
            "review: repeat-only frames saved"
        );
        success(serde_json::json!({
            "eligible": o.eligible,
            "accepted": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": message,
        }))
    })
    .await)
}

/// `POST /v1/review/reject-stacked` with `{bucket?, scene?, camera?, label?, source?, dry_run,
/// limit?}`: on the reviewed and unreviewed frames with a label file that the bucket, scene, camera
/// and class filters select, the boxes stacked on a better box of the object (IoU >= 0.7) are
/// rejected (`reject_reason: duplicate`, kept as evidence; [`bulk::bulk_reject_stacked`]), at most
/// `limit` frames (default 500). The frames keep their status; drawn boxes and frames a reviewer
/// checked (`overlaps_checked`) are left alone. `source` is checked but does not narrow the walk: a
/// reviewed frame need not be in the review queue. `dry_run` only counts. `boxes` counts the boxes
/// rejected, `rejected` the frames changed (as reject-low), `overlaps_left` the frames that still
/// need a reviewer once the duplicates are gone (nested or overlapping boxes no rule can settle:
/// `auto=overlaps`), in a dry run too.
pub(in crate::server) async fn reject_stacked(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    let b: RejectOtherClassBody = json_body(&headers, body)?;
    let limit = bulk_limit(b.limit, ACCEPT_AUTO_DEFAULT_LIMIT)?;
    let filters = bulk_filters([b.bucket, b.scene, b.camera, b.label, b.source], &[])?;
    let dry_run = b.dry_run;
    Ok(blocking(move || {
        let (dir, classes, _) = context(&state);
        let hidden = hidden(&state);
        let wanted = bulk::Wanted {
            classes: &classes,
            hidden: &hidden,
        };
        let done = bulk::bulk_reject_stacked(&dir, &filters, &wanted, limit, dry_run, &|_| true);
        let (o, left, left_reviewed) = (
            done.outcome,
            done.overlaps_left,
            done.overlaps_left_reviewed,
        );
        let mut message = if o.boxes == 0 {
            "No stacked duplicates on the frames these filters select.".to_string()
        } else if dry_run {
            format!(
                "Would reject {} stacked duplicate box(es) on {} frame(s); the frames keep their \
                 status.",
                o.boxes, o.eligible
            )
        } else {
            format!(
                "Rejected {} stacked duplicate box(es) on {} frame(s) (kept as evidence, reason \
                 duplicate; reviewed frames stay reviewed).",
                o.boxes, o.done_count
            )
        };
        push_failed(
            &mut message,
            o.failed_count,
            "frame(s) not changed",
            "review",
        );
        if !dry_run {
            push_past_limit(&mut message, o.eligible, limit, "more frame(s)");
        }
        if left > 0 {
            message.push_str(&format!(
                " {left} frame(s) still have overlapping boxes no rule can settle, \
                 {left_reviewed} of them reviewed (those are the ones training uses): set Auto \
                 to overlapping boxes, Status to reviewed and Frames to all collected, then keep \
                 or reject them by hand."
            ));
        }
        if !dry_run {
            tracing::info!(
                boxes = o.boxes,
                frames = o.done_count,
                failed = o.failed_count,
                eligible = o.eligible,
                overlaps_left = left,
                "review: stacked duplicates rejected"
            );
        }
        success(serde_json::json!({
            "eligible": o.eligible,
            "boxes": o.boxes,
            "rejected": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": message,
            "overlaps_left": left,
            "overlaps_left_reviewed": left_reviewed,
        }))
    })
    .await)
}

/// `POST /v1/review/restore-reopened` with `{bucket?, scene?, camera?, label?, source?, dry_run,
/// limit?}`: the undo of a re-label that reopened reviewed frames. On the unreviewed frames of those
/// filters, a frame with a `disagreement` note, a `reviewed_at` and nothing pending goes back to
/// reviewed and loses the note ([`bulk::bulk_restore_reopened`], re-checked under the write lock),
/// at most `limit` frames (default 500). Frames with a pending box and frames never reviewed are
/// not touched. `dry_run` only counts. `eligible` counts the frames, `restored` those changed.
pub(in crate::server) async fn restore_reopened(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    let b: RejectOtherClassBody = json_body(&headers, body)?;
    let limit = bulk_limit(b.limit, ACCEPT_AUTO_DEFAULT_LIMIT)?;
    let filters = bulk_filters(
        [b.bucket, b.scene, b.camera, b.label, b.source],
        &[("status", "unreviewed")],
    )?;
    let dry_run = b.dry_run;
    Ok(blocking(move || {
        let (dir, _, _) = context(&state);
        let o = bulk::bulk_restore_reopened(&dir, &filters, limit, dry_run, &|_| true);
        let eligible = o.eligible;
        if dry_run {
            return success(serde_json::json!({
                "eligible": eligible,
                "restored": 0,
                "frames": o.done,
            }));
        }
        let mut message = format!(
            "Restored {} reopened frame(s) to reviewed (their boxes are unchanged; they are back \
             in exports).",
            o.done_count
        );
        push_failed(
            &mut message,
            o.failed_count,
            "frame(s) not changed",
            "review",
        );
        push_past_limit(&mut message, eligible, limit, "more frame(s)");
        tracing::info!(
            restored = o.done_count,
            failed = o.failed_count,
            eligible,
            "review: reopened frames restored"
        );
        success(serde_json::json!({
            "eligible": eligible,
            "restored": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": message,
        }))
    })
    .await)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SuggestOverlapsBody {
    #[serde(default)]
    bucket: Option<String>,
    #[serde(default)]
    scene: Option<String>,
    #[serde(default)]
    camera: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    gpu: Option<String>,
    #[serde(default)]
    allow_large: bool,
    /// Also pairs with an answer.
    #[serde(default)]
    again: bool,
    #[serde(default)]
    dry_run: bool,
}

/// `POST /v1/review/suggest-overlaps` with `{bucket?, scene?, camera?, label?, gpu?, allow_large?,
/// again?, dry_run?}`: queue the overlap check ([`crate::training::overlaps`]) on the label files
/// of the folder under review those filters select, reviewed frames too (no `source` or `status`
/// narrows it, a reviewed frame need not be in the review queue). It writes suggestions and nothing
/// else. `dry_run` only counts (`frames`, `pairs`, `crops`, `no_zoom`, `answered`, `passed_over`), and
/// needs no GPU. Otherwise `{success, message, id}`, the task id in `GET /v1/training` `tasks[]`.
pub(in crate::server) async fn suggest_overlaps(
    State(shared): State<Shared>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    use crate::training::overlaps::{OverlapsRequest, plan_overlaps, select};
    let b: SuggestOverlapsBody = json_body(&headers, body)?;
    Ok(blocking(move || {
        let state = shared.current();
        let mut req = match OverlapsRequest::parse(if b.again { "again=1" } else { "" }) {
            Ok(r) => r,
            Err(e) => return fail(StatusCode::BAD_REQUEST, format!("{e:#}")),
        };
        for (k, v) in [
            ("bucket", b.bucket),
            ("scene", b.scene),
            ("camera", b.camera),
            ("label", b.label),
        ] {
            if let Some(v) = v.filter(|v| !v.trim().is_empty()) {
                req.frames.query.insert(k.to_string(), v);
            }
        }
        req.gpu = b.gpu;
        req.allow_large = b.allow_large;
        if let Err(e) = req.frames.filters() {
            return fail(StatusCode::BAD_REQUEST, format!("{e:#}"));
        }
        if b.dry_run {
            let (dir, classes, _) = context(&state);
            let hidden = hidden(&state);
            let wanted = bulk::Wanted {
                classes: &classes,
                hidden: &hidden,
            };
            let s = match select::select(&dir, &req.frames, &wanted) {
                Ok(s) => s,
                Err(e) => return fail(StatusCode::BAD_REQUEST, format!("{e:#}")),
            };
            let message = (s.pairs == 0).then(|| {
                let again = if s.answered > 0 {
                    "; tick Ask again to check the answered ones again"
                } else {
                    ""
                };
                format!(
                    "No open overlap without an answer on these frames ({} pairs answered before, \
                     {} frames skipped or checked by a reviewer){again}.",
                    s.answered, s.passed_over
                )
            });
            return success(serde_json::json!({
                "frames": s.frame_count,
                "pairs": s.pairs,
                "no_zoom": s.no_zoom,
                "crops": s.crops,
                "answered": s.answered,
                "passed_over": s.passed_over,
                "message": message,
            }));
        }
        let config = state.config_read().clone();
        let cap = crate::training::capability_now();
        let job = match plan_overlaps(&config, req, &cap) {
            Ok(j) => j,
            Err(e) => return fail(StatusCode::BAD_REQUEST, format!("Not queued: {e:#}")),
        };
        let title = crate::training::task::Task::title(&job);
        match crate::server::training::service(&shared).enqueue_task(Box::new(job)) {
            Ok(id) => success(serde_json::json!({
                "id": id,
                "message": format!(
                    "Queued: {title}. It writes suggestions only: no box changes until you \
                     confirm one here."
                ),
            })),
            Err(e) => fail(StatusCode::BAD_REQUEST, format!("Not queued: {e}")),
        }
    })
    .await)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RejectOtherClassBody {
    #[serde(default)]
    bucket: Option<String>,
    #[serde(default)]
    scene: Option<String>,
    #[serde(default)]
    camera: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    dry_run: bool,
    #[serde(default)]
    limit: Option<usize>,
}

/// `POST /v1/review/reject-other-class` with `{bucket?, scene?, camera?, label?, source?,
/// dry_run, limit?}`: on the unreviewed frames of those filters, the pending model and auto boxes
/// of unwanted classes (not in `collect.classes`, or in `collect.hidden_classes`) rejected with
/// `reason: other_class` (kept as evidence), or renamed to the list's spelling when the label is
/// a wanted class under another name (`pottedplant` -> `potted plant`;
/// [`bulk::bulk_reject_other_class`]), at most `limit` frames (default 500). `dry_run` only
/// counts. `boxes` counts the boxes changed, `relabelled` the renamed ones among them,
/// `rejected` the frames changed (as reject-low); `reviewed_with_unwanted` the reviewed frames
/// that still hold training boxes of unwanted classes
/// ([`bulk::reviewed_with_unwanted`]; `POST /v1/review/reject-unwanted-reviewed` rejects them).
pub(in crate::server) async fn reject_other_class(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    let b: RejectOtherClassBody = json_body(&headers, body)?;
    let limit = bulk_limit(b.limit, ACCEPT_AUTO_DEFAULT_LIMIT)?;
    let filters = bulk_filters(
        [b.bucket, b.scene, b.camera, b.label, b.source],
        &[("status", "unreviewed")],
    )?;
    let dry_run = b.dry_run;
    Ok(blocking(move || {
        let (dir, classes, _) = context(&state);
        let hidden = hidden(&state);
        let wanted = bulk::Wanted {
            classes: &classes,
            hidden: &hidden,
        };
        let o = bulk::bulk_reject_other_class(&dir, &filters, &wanted, limit, dry_run, &|_| true);
        let eligible = o.eligible;
        let reviewed = bulk::reviewed_with_unwanted(&dir, &wanted);
        if dry_run {
            return success(serde_json::json!({
                "eligible": eligible,
                "boxes": o.boxes,
                "relabelled": o.relabelled,
                "rejected": 0,
                "frames": o.done,
                "reviewed_with_unwanted": reviewed,
            }));
        }
        let mut message = format!(
            "Rejected {} pending box(es) of unwanted classes on {} frame(s) (kept as evidence, \
             reason other_class; the frames stay unreviewed).",
            o.boxes - o.relabelled,
            o.done_count
        );
        if o.relabelled > 0 {
            message.push_str(&format!(
                " Renamed {} box(es) to the class list's spelling.",
                o.relabelled
            ));
        }
        push_failed(
            &mut message,
            o.failed_count,
            "frame(s) not changed",
            "review",
        );
        push_past_limit(&mut message, eligible, limit, "more frame(s)");
        tracing::info!(
            boxes = o.boxes,
            relabelled = o.relabelled,
            frames = o.done_count,
            failed = o.failed_count,
            eligible,
            "review: pending boxes of unwanted classes rejected"
        );
        success(serde_json::json!({
            "eligible": eligible,
            "boxes": o.boxes,
            "relabelled": o.relabelled,
            "rejected": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": message,
            "reviewed_with_unwanted": reviewed,
        }))
    })
    .await)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RejectUnwantedReviewedBody {
    #[serde(default)]
    dry_run: bool,
    #[serde(default)]
    limit: Option<usize>,
}

/// `POST /v1/review/reject-unwanted-reviewed` with `{dry_run, limit?}`: on the reviewed frames
/// of the folder under review, the training boxes (accepted or added) of unwanted classes (not in
/// `collect.classes`, or in `collect.hidden_classes`) rejected with `reason: other_class`; the
/// frames stay reviewed ([`bulk::bulk_reject_unwanted`] with [`bulk::Scope::Reviewed`]), at most
/// `limit` frames (default and most [`ACCEPT_AUTO_MAX_LIMIT`]). `dry_run` only counts: `eligible`
/// frames, `boxes`, `classes` (their labels). The Review tab asks before it calls this.
pub(in crate::server) async fn reject_unwanted_reviewed(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Answer {
    let b: RejectUnwantedReviewedBody = json_body(&headers, body)?;
    let limit = bulk_limit(b.limit, ACCEPT_AUTO_MAX_LIMIT)?;
    let dry_run = b.dry_run;
    Ok(blocking(move || {
        let (dir, classes, _) = context(&state);
        let hidden = hidden(&state);
        let wanted = bulk::Wanted {
            classes: &classes,
            hidden: &hidden,
        };
        if dry_run {
            let r = bulk::reviewed_with_unwanted(&dir, &wanted);
            return success(serde_json::json!({
                "eligible": r.frames,
                "boxes": r.boxes,
                "classes": r.classes,
                "rejected": 0,
            }));
        }
        let o = bulk::bulk_reject_unwanted(&dir, &wanted, bulk::Scope::Reviewed, limit, false);
        let eligible = o.eligible;
        let mut message = format!(
            "Rejected {} accepted box(es) of unwanted classes on {} reviewed frame(s) (kept as \
             evidence, reason other_class; the frames stay reviewed).",
            o.boxes, o.done_count
        );
        push_failed(&mut message, o.failed_count, "frame(s) not changed", "fix");
        push_past_limit(&mut message, eligible, limit, "more frame(s)");
        tracing::info!(
            boxes = o.boxes,
            frames = o.done_count,
            failed = o.failed_count,
            eligible,
            "review: accepted boxes of unwanted classes rejected on reviewed frames"
        );
        success(serde_json::json!({
            "eligible": eligible,
            "boxes": o.boxes,
            "rejected": o.done_count,
            "failed": o.failed,
            "frames": o.done,
            "message": message,
        }))
    })
    .await)
}
