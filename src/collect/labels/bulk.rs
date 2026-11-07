//! Bulk edits of the label files a Review queue selects: [`bulk_accept`] saves the frames the
//! auto-labelers' triage alone settles as reviewed (`POST /v1/review/accept-auto`, the
//! auto-label chain's accept step), [`bulk_reject_below`] rejects the pending auto boxes scoring
//! below a minimum on unreviewed frames (`reason: low_score`; the reject-low route and the
//! chain's reject step), [`bulk_reject_other_class`] rejects the pending model and auto boxes
//! of unwanted classes ([`Wanted`]: not in the class list, or in `collect.hidden_classes`) on
//! unreviewed frames (`reason: other_class`; an alias such as `pottedplant` is renamed to the
//! list's spelling instead; `POST /v1/review/reject-other-class`), [`bulk_reject_unwanted`]
//! does the same over every label file of the folder (after a class list change) or, with
//! [`Scope::Reviewed`], rejects the training boxes of unwanted classes on reviewed frames
//! (`POST /v1/review/reject-unwanted-reviewed`; [`reviewed_with_unwanted`] counts them),
//! [`bulk_spot`] rejects or accepts the boxes at one spot of one camera on its unreviewed frames
//! ("this spot everywhere", [`super::repeats`]), [`bulk_accept_repeats`] saves the frames such
//! edits settled (`POST /v1/review/reject-spot`, `accept-spot`, `accept-repeats`),
//! [`bulk_reject_stacked`] rejects the duplicates of stacked boxes on reviewed and unreviewed
//! frames (`reason: duplicate`, [`super::stacks`]; `POST /v1/review/reject-stacked`) and
//! [`bulk_restore_reopened`] puts frames a re-label reopened back to reviewed once nothing on them
//! is pending (`POST /v1/review/restore-reopened`). Each frame is re-checked under the label write
//! lock ([`super::update`]); counts are exact, frame lists capped at [`LISTED`].

use super::super::layout::Walker;
use super::queue::{self, AutoFilter, Filters, StatusFilter};
use super::{BoxState, Frame, LabelFile, Source, Status, stacks};
use serde::Serialize;
use std::path::Path;

/// Most frame names listed in a [`BulkOutcome`].
pub const LISTED: usize = 20;

/// The reason a box rejected by [`reject_low`] carries.
pub const LOW_SCORE_REASON: &str = "low_score";

/// The reason of a box a labelers run dismissed for its score
/// ([`crate::training::autolabel::triage::tier`], [`super::LabelBox::is_dismissed`]). Rejected like
/// [`LOW_SCORE_REASON`], which keeps it out of every reader of live boxes, but no reviewer's
/// decision: no hard negative, not in the repeats history, and a rerun may replace it.
pub const DISMISSED_REASON: &str = "auto_dismissed";

/// The reason of a box a labelers run dismissed because a reviewer rejected that spot of the
/// camera again and again ([`super::spots`]). Every reader treats it as [`DISMISSED_REASON`]
/// through [`is_dismissal`]; only the Review tab's wording differs.
pub const SPOT_DISMISSED_REASON: &str = "spot_dismissed";

/// Whether `reason` marks a dismissal ([`DISMISSED_REASON`], [`SPOT_DISMISSED_REASON`]): the one
/// place that lists them, so the two cannot drift apart. Twin of the page's `isDismissal`.
pub fn is_dismissal(reason: Option<&str>) -> bool {
    matches!(reason, Some(DISMISSED_REASON | SPOT_DISMISSED_REASON))
}

/// The reason a box rejected by [`reject_other_class`] carries.
pub const OTHER_CLASS_REASON: &str = "other_class";

/// What a bulk edit did (or, in a dry run, would do).
#[derive(Debug, Default, Clone, Serialize)]
pub struct BulkOutcome {
    /// Frames the edit applies to (all of them, also past the limit).
    pub eligible: usize,
    /// Boxes changed (a dry run of [`bulk_reject_below`] / [`bulk_reject_other_class`]: the
    /// boxes of every eligible frame).
    pub boxes: usize,
    /// Of `boxes`, those [`bulk_reject_other_class`] renamed to the class list's spelling rather
    /// than rejected (left out of the JSON when 0).
    #[serde(skip_serializing_if = "is_zero")]
    pub relabelled: usize,
    /// The first frames changed (a dry run: the first eligible frames), at most [`LISTED`].
    pub done: Vec<String>,
    /// Frames changed (0 in a dry run).
    pub done_count: usize,
    /// The first frames not changed, `<frame>: <why>`, at most [`LISTED`].
    pub failed: Vec<String>,
    pub failed_count: usize,
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

impl BulkOutcome {
    fn done(&mut self, rel: &str) {
        self.done_count += 1;
        if self.done.len() < LISTED {
            self.done.push(rel.to_string());
        }
    }

    fn failed(&mut self, rel: &str, why: &str) {
        self.failed_count += 1;
        if self.failed.len() < LISTED {
            self.failed.push(format!("{rel}: {why}"));
        }
    }
}

/// Whether `b` is a box [`reject_low`] rejects at `min_score`: a pending auto-labeler box
/// scoring below it (a box without a score is left alone).
fn is_low(b: &super::LabelBox, min_score: f32) -> bool {
    b.source == Source::Auto
        && b.state == BoxState::Pending
        && !b.auto.audit
        && b.score.is_some_and(|s| s < min_score)
}

/// Reject the pending auto-labeler boxes of `f` scoring below `min_score` (`reason:
/// low_score`, kept as hard-negative evidence), never an audit sample (it exists to be looked at);
/// only on an unreviewed frame. The frame's status
/// is unchanged. Returns the boxes rejected.
pub fn reject_low(f: &mut LabelFile, min_score: f32) -> usize {
    if f.status != Status::Unreviewed {
        return 0;
    }
    let mut n = 0;
    for b in f.boxes.iter_mut().filter(|b| is_low(b, min_score)) {
        b.state = BoxState::Rejected;
        b.reason = Some(LOW_SCORE_REASON.into());
        n += 1;
    }
    n
}

/// The boxes [`reject_low`] would reject.
fn count_low(f: &LabelFile, min_score: f32) -> usize {
    if f.status != Status::Unreviewed {
        return 0;
    }
    f.boxes.iter().filter(|b| is_low(b, min_score)).count()
}

/// Reject the low-scoring pending auto boxes ([`reject_low`]) of the unreviewed frames of `dir`
/// that `filters` (status forced to unreviewed) and `keep` (the frame's
/// `<scene>/<camera>/<date>/<stem>.jpg`) select, at most `limit` frames in queue order.
/// `dry_run` only counts. A frame that changed meanwhile (reviewed) is passed over.
pub fn bulk_reject_below(
    dir: &Path,
    filters: &Filters,
    min_score: f32,
    limit: usize,
    dry_run: bool,
    keep: &dyn Fn(&str) -> bool,
) -> BulkOutcome {
    edit_unreviewed(
        dir,
        filters,
        limit,
        dry_run,
        keep,
        &|l| (count_low(l, min_score), 0),
        &|l| (reject_low(l, min_score), 0),
    )
}

/// The classes the reviewer wants: `classes` (`collect.classes`) without `hidden`
/// (`collect.hidden_classes`). A label of no wanted class ([`super::wanted_class`]) is unwanted.
#[derive(Debug, Clone, Copy)]
pub struct Wanted<'a> {
    pub classes: &'a [String],
    pub hidden: &'a [String],
}

impl<'a> Wanted<'a> {
    /// The wanted class `label` is (the list's spelling), None when it is unwanted.
    pub fn class_of(&self, label: &str) -> Option<&'a str> {
        super::wanted_class(label, self.classes, self.hidden)
    }
}

/// Which boxes an unwanted-class reject looks at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pick {
    /// Pending model and auto-labeler boxes (proposals nobody decided yet).
    Proposals,
    /// Training boxes (accepted or added, any source).
    Training,
}

/// What an unwanted-class reject does to `b`: None = nothing (not a box of `pick`, or of a
/// wanted class in the list's spelling); `Some(Some(class))` = renamed to `class` (proposals
/// only: the list's spelling of an alias, [`super::map_class`]); `Some(None)` = rejected.
fn unwanted_action<'a>(
    b: &super::LabelBox,
    wanted: &Wanted<'a>,
    pick: Pick,
) -> Option<Option<&'a str>> {
    let picked = match pick {
        Pick::Proposals => b.state == BoxState::Pending && b.source != Source::Human,
        Pick::Training => b.is_training(),
    };
    if !picked {
        return None;
    }
    match wanted.class_of(&b.label) {
        Some(c) if c == b.label => None,
        Some(c) if pick == Pick::Proposals => Some(Some(c)),
        Some(_) => None,
        None => Some(None),
    }
}

/// Apply [`unwanted_action`] to the boxes of `f`: (rejected, renamed).
fn apply_unwanted(f: &mut LabelFile, wanted: &Wanted<'_>, pick: Pick) -> (usize, usize) {
    let (mut rejected, mut renamed) = (0, 0);
    for b in &mut f.boxes {
        match unwanted_action(b, wanted, pick) {
            None => {}
            Some(Some(class)) => {
                super::rename_to_class(b, class);
                renamed += 1;
            }
            Some(None) => {
                b.state = BoxState::Rejected;
                b.reason = Some(OTHER_CLASS_REASON.into());
                rejected += 1;
            }
        }
    }
    (rejected, renamed)
}

/// What [`apply_unwanted`] would do: (rejected, renamed).
fn count_unwanted(f: &LabelFile, wanted: &Wanted<'_>, pick: Pick) -> (usize, usize) {
    f.boxes
        .iter()
        .filter_map(|b| unwanted_action(b, wanted, pick))
        .fold((0, 0), |(r, n), a| match a {
            Some(_) => (r, n + 1),
            None => (r + 1, n),
        })
}

/// The pending model and auto-labeler boxes of `f` of an unwanted class rejected (`reason:
/// other_class`), those of a wanted class under another spelling renamed, whatever the frame's
/// status (a save of the Review tab, which does not show them, and the auto-label run's writes).
/// Returns (rejected, renamed).
pub fn reject_unwanted_proposals(f: &mut LabelFile, wanted: &Wanted<'_>) -> (usize, usize) {
    apply_unwanted(f, wanted, Pick::Proposals)
}

/// On an unreviewed frame, the pending model boxes (the serving model's, from the sidecar) and
/// auto-labeler boxes whose label is no wanted class ([`Wanted`]: not in the class list, or a
/// hidden class) rejected with `reason: other_class` (kept as evidence; a save would otherwise
/// accept a box the reviewer cannot see or does not want), those of a wanted class under another
/// spelling ([`super::map_class`]: `pottedplant` -> `potted plant`, the old label kept in
/// `model_label` unless only the case differs) renamed instead. Human boxes, decided boxes and
/// reviewed or skipped frames are untouched; the frame's status is unchanged. Returns
/// (rejected, renamed).
pub fn reject_other_class(f: &mut LabelFile, wanted: &Wanted<'_>) -> (usize, usize) {
    if f.status != Status::Unreviewed {
        return (0, 0);
    }
    apply_unwanted(f, wanted, Pick::Proposals)
}

/// What [`reject_other_class`] would do: (rejected, renamed).
fn count_other_class(f: &LabelFile, wanted: &Wanted<'_>) -> (usize, usize) {
    if f.status != Status::Unreviewed {
        return (0, 0);
    }
    count_unwanted(f, wanted, Pick::Proposals)
}

/// On a reviewed frame, the training boxes (accepted or added, any source) of an unwanted class
/// rejected with `reason: other_class`; the frame stays reviewed and its remaining training
/// boxes are all of wanted classes, so it still passes [`super::normalize`] (a rejected human
/// box is dropped at the next save). Other frames are untouched. Returns the boxes rejected.
pub fn reject_unwanted_reviewed(f: &mut LabelFile, wanted: &Wanted<'_>) -> usize {
    if f.status != Status::Reviewed {
        return 0;
    }
    apply_unwanted(f, wanted, Pick::Training).0
}

/// The training boxes [`reject_unwanted_reviewed`] would reject.
fn count_unwanted_reviewed(f: &LabelFile, wanted: &Wanted<'_>) -> usize {
    if f.status != Status::Reviewed {
        return 0;
    }
    count_unwanted(f, wanted, Pick::Training).0
}

/// [`reject_other_class`] on the unreviewed frames of `dir` that `filters` (status forced to
/// unreviewed) and `keep` select, at most `limit` frames in queue order; `boxes` counts the
/// boxes rejected or renamed, `relabelled` the renamed ones. `dry_run` only counts. A frame
/// that changed meanwhile (reviewed) is passed over.
pub fn bulk_reject_other_class(
    dir: &Path,
    filters: &Filters,
    wanted: &Wanted<'_>,
    limit: usize,
    dry_run: bool,
    keep: &dyn Fn(&str) -> bool,
) -> BulkOutcome {
    edit_unreviewed(
        dir,
        filters,
        limit,
        dry_run,
        keep,
        &|l| count_other_class(l, wanted),
        &|l| reject_other_class(l, wanted),
    )
}

/// Which label files [`bulk_reject_unwanted`] edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// Unreviewed frames: their pending model and auto boxes ([`reject_other_class`]).
    Unreviewed,
    /// Reviewed frames: their training boxes ([`reject_unwanted_reviewed`]).
    Reviewed,
}

/// The unwanted-class reject of `scope` over every label file of `dir` ([`super::label_files`],
/// queued or not; a frame without a label file is pre-filled without unwanted boxes anyway), at
/// most `limit` frames. `dry_run` only counts. Each frame is edited again under the write lock.
pub fn bulk_reject_unwanted(
    dir: &Path,
    wanted: &Wanted<'_>,
    scope: Scope,
    limit: usize,
    dry_run: bool,
) -> BulkOutcome {
    let count = |l: &LabelFile| match scope {
        Scope::Unreviewed => count_other_class(l, wanted),
        Scope::Reviewed => (count_unwanted_reviewed(l, wanted), 0),
    };
    let edit = |l: &mut LabelFile| match scope {
        Scope::Unreviewed => reject_other_class(l, wanted),
        Scope::Reviewed => (reject_unwanted_reviewed(l, wanted), 0),
    };
    let files = super::label_files(dir);
    let eligible = queue::over_days(&files, &|day| {
        day.iter()
            .filter_map(|(rel, path)| {
                let (n, renamed) = super::read(path)
                    .ok()
                    .flatten()
                    .map_or((0, 0), |l| count(&l));
                (n + renamed > 0).then(|| (rel.clone(), n, renamed))
            })
            .collect()
    });
    edit_listed(dir, eligible, limit, dry_run, &|_| None, &edit)
}

/// Reviewed frames holding training boxes of unwanted classes ([`reviewed_with_unwanted`]).
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub struct ReviewedUnwanted {
    pub frames: usize,
    pub boxes: usize,
    /// Their labels, sorted.
    pub classes: Vec<String>,
}

/// The reviewed frames of `dir` whose training boxes include some of an unwanted class (what
/// [`bulk_reject_unwanted`] with [`Scope::Reviewed`] would reject), with the boxes and labels.
pub fn reviewed_with_unwanted(dir: &Path, wanted: &Wanted<'_>) -> ReviewedUnwanted {
    let mut out = ReviewedUnwanted::default();
    let mut labels = std::collections::BTreeSet::new();
    let files = super::label_files(dir);
    // The unwanted labels of each reviewed frame with some.
    let frames = queue::over_days(&files, &|day| {
        day.iter()
            .filter_map(|(_, path)| {
                let l = super::read(path).ok().flatten()?;
                if l.status != Status::Reviewed {
                    return None;
                }
                let found: Vec<String> = l
                    .boxes
                    .iter()
                    .filter(|b| unwanted_action(b, wanted, Pick::Training) == Some(None))
                    .map(|b| b.label.clone())
                    .collect();
                (!found.is_empty()).then_some(found)
            })
            .collect()
    });
    for found in frames {
        out.frames += 1;
        out.boxes += found.len();
        labels.extend(found);
    }
    out.classes = labels.into_iter().collect();
    out
}

/// What [`bulk_reject_stacked`] did.
#[derive(Debug, Default, Clone)]
pub struct StackedOutcome {
    pub outcome: BulkOutcome,
    /// Frames (reviewed or unreviewed, of the same filters) that still need a reviewer once the
    /// duplicates are gone ([`stacks::needs_look_after`], the test of `auto=overlaps`); frames past
    /// the limit are counted as if done, so a dry run and the run answer the same.
    pub overlaps_left: usize,
    /// Of [`Self::overlaps_left`], the reviewed frames: the ones an export would train on.
    pub overlaps_left_reviewed: usize,
}

/// Reject the duplicates of stacked boxes ([`stacks::reject_stacked`]) on the reviewed and
/// unreviewed frames of `dir`, at most `limit` frames. Walks the label files ([`queue::labelled`]):
/// `filters` narrows by bucket, scene, camera and class, its `source`, `status` and `auto` are not
/// used (a reviewed frame need not be in the review queue). A frame without a label file is
/// passed over. The frame's status, `reviewed_at` and other boxes are unchanged. `dry_run` only
/// counts.
pub fn bulk_reject_stacked(
    dir: &Path,
    filters: &Filters,
    wanted: &Wanted<'_>,
    limit: usize,
    dry_run: bool,
    keep: &dyn Fn(&str) -> bool,
) -> StackedOutcome {
    let mut eligible = Vec::new();
    let (mut overlaps_left, mut overlaps_left_reviewed) = (0, 0);
    // The walk only reads; the edits below come after it, on this thread.
    let looked = queue::labelled_map(dir, filters, keep, &|rel, l| {
        (l.status != Status::Skipped).then(|| {
            let needs_look = stacks::needs_look_after(&l, wanted);
            let n = stacks::count_stacked(&l, wanted);
            (rel.to_string(), n, needs_look, l.status == Status::Reviewed)
        })
    });
    for (rel, n, needs_look, reviewed) in looked {
        if needs_look {
            overlaps_left += 1;
            overlaps_left_reviewed += usize::from(reviewed);
        }
        if n > 0 {
            eligible.push((rel, n, 0));
        }
    }
    let outcome = edit_listed(dir, eligible, limit, dry_run, &|_| None, &|l| {
        (stacks::reject_stacked(l, wanted), 0)
    });
    StackedOutcome {
        outcome,
        overlaps_left,
        overlaps_left_reviewed,
    }
}

/// Whether a re-label's reopening of `f` can be undone: it went back to review
/// ([`LabelFile::disagreement`]) from a review ([`LabelFile::reviewed_at`], which a save as
/// unreviewed clears) and nothing on it waits for a decision. A frame confirmed empty that has
/// gained a training box is left out: a save refuses that frame.
pub fn restorable(f: &LabelFile) -> bool {
    f.status == Status::Unreviewed
        && f.disagreement.is_some()
        && f.reviewed_at.is_some()
        && !f.boxes.iter().any(|b| b.state == BoxState::Pending)
        && !(f.empty_confirmed && f.training().next().is_some())
}

/// Put a [`restorable`] frame back to reviewed, its boxes and `reviewed_at` as they are, and clear
/// the note (and a spot edit's mark, as a save does). [`LabelFile::overlaps_checked`] comes back
/// from [`LabelFile::checked_before_reopen`]: nothing is pending and no new box was accepted (a spot
/// accept ends the mark, a save ends the reopening), so the frame's live boxes are the ones the
/// reviewer checked. Returns 1 when it did, else 0.
pub fn restore_reopened(f: &mut LabelFile) -> usize {
    if !restorable(f) {
        return 0;
    }
    f.status = Status::Reviewed;
    f.disagreement = None;
    f.repeat_spot = false;
    f.overlaps_checked |= std::mem::take(&mut f.checked_before_reopen);
    1
}

/// [`restore_reopened`] on the unreviewed frames of `dir` that `filters` (status forced to
/// unreviewed) and `keep` select, at most `limit` in queue order; `boxes` counts frames here.
/// `dry_run` only counts. Without it, a frame whose proposals were all rejected (by Reject auto
/// boxes below, say) would stay out of the exports. A frame that gained a pending box meanwhile
/// is passed over.
pub fn bulk_restore_reopened(
    dir: &Path,
    filters: &Filters,
    limit: usize,
    dry_run: bool,
    keep: &dyn Fn(&str) -> bool,
) -> BulkOutcome {
    edit_unreviewed(
        dir,
        filters,
        limit,
        dry_run,
        keep,
        &|l| (usize::from(restorable(l)), 0),
        &|l| (restore_reopened(l), 0),
    )
}

/// The shared run of the box edits of unreviewed frames: `count` and `edit` return (boxes
/// rejected, boxes renamed); only frames `count` finds something on are eligible, and each
/// is edited again under the write lock ([`super::update`]).
fn edit_unreviewed(
    dir: &Path,
    filters: &Filters,
    limit: usize,
    dry_run: bool,
    keep: &dyn Fn(&str) -> bool,
    count: &dyn Fn(&LabelFile) -> (usize, usize),
    edit: &dyn Fn(&mut LabelFile) -> (usize, usize),
) -> BulkOutcome {
    let mut f = filters.clone();
    f.status = StatusFilter::Is(Status::Unreviewed);
    let built = queue::build(dir, &f);
    let mut eligible = Vec::new();
    let mut walker = Walker::new(dir);
    for item in built.items.into_iter().filter(|i| keep(&i.frame)) {
        let (n, renamed) = Frame::resolve_in(&mut walker, &item.frame)
            .and_then(|fr| fr.labels().ok().flatten())
            .map_or((0, 0), |l| count(&l));
        if n + renamed > 0 {
            eligible.push((item.frame, n, renamed));
        }
    }
    edit_listed(dir, eligible, limit, dry_run, &|_| None, edit)
}

/// Edit the frames `eligible` lists (rel, boxes rejected, boxes renamed as counted), at most
/// `limit`, each re-checked under the write lock ([`super::update`]); a frame without a label
/// file is edited from `fresh` (None: passed over). `dry_run` only sums the counts.
fn edit_listed(
    dir: &Path,
    eligible: Vec<(String, usize, usize)>,
    limit: usize,
    dry_run: bool,
    fresh: &dyn Fn(&Frame) -> Option<LabelFile>,
    edit: &dyn Fn(&mut LabelFile) -> (usize, usize),
) -> BulkOutcome {
    let mut out = BulkOutcome {
        eligible: eligible.len(),
        ..BulkOutcome::default()
    };
    if dry_run {
        out.boxes = eligible.iter().map(|(_, n, r)| n + r).sum();
        out.relabelled = eligible.iter().map(|(_, _, r)| r).sum();
        out.done = eligible
            .iter()
            .take(LISTED)
            .map(|(r, _, _)| (*r).to_string())
            .collect();
        return out;
    }
    let mut walker = Walker::new(dir);
    for (rel, _, _) in eligible.into_iter().take(limit) {
        let rel = rel.as_str();
        let Some(frame) = Frame::resolve_in(&mut walker, rel) else {
            out.failed(rel, "the frame is gone");
            continue;
        };
        let (mut n, mut renamed) = (0, 0);
        let r = super::update(&frame.labels, |cur| {
            let Some(mut l) = cur.or_else(|| fresh(&frame)) else {
                return Ok(None);
            };
            (n, renamed) = edit(&mut l);
            Ok((n + renamed > 0).then_some(l))
        });
        match r {
            Ok(true) => {
                out.boxes += n + renamed;
                out.relabelled += renamed;
                out.done(rel);
            }
            Ok(false) => {}
            Err(e) => out.failed(rel, &format!("{e:#}")),
        }
    }
    out
}

/// Accept one frame the queue listed as eligible: re-checked under the write lock (it may have
/// changed since), its pending boxes of unwanted classes rejected first ([`reject_other_class`]:
/// a bulk accept must not turn them into labels), saved as reviewed ([`super::normalize`]) with
/// `bulk_accepted`; a pending box stacked on a better one is rejected as a duplicate rather than
/// accepted ([`stacks::reject_pending_stacked`]). `Ok(None)` when it is not eligible (any more),
/// also when the rejects left nothing to accept: those stay written, the frame unreviewed. Else
/// the boxes it accepted.
fn accept_frame(
    dir: &Path,
    rel: &str,
    wanted: &Wanted<'_>,
    now: &str,
) -> anyhow::Result<Option<usize>> {
    let frame = Frame::resolve(dir, rel).ok_or_else(|| anyhow::anyhow!("the frame is gone"))?;
    let rec = frame.record()?;
    let mut accepted = None;
    super::update(&frame.labels, |cur| {
        let Some(mut f) = cur.filter(super::auto_accept_eligible) else {
            return Ok(None);
        };
        let (rejected, renamed) = reject_other_class(&mut f, wanted);
        if !super::auto_accept_eligible(&f) {
            return Ok((rejected + renamed > 0).then_some(f));
        }
        stacks::reject_pending_stacked(&mut f, wanted);
        accepted = Some(
            f.boxes
                .iter()
                .filter(|b| {
                    b.state == BoxState::Pending && super::resolve_pending(b) == BoxState::Accepted
                })
                .count(),
        );
        f.status = Status::Reviewed;
        let mut n = super::normalize(f, wanted.classes, rec.width, rec.height, now)?;
        n.bulk_accepted = true;
        Ok(Some(n))
    })?;
    Ok(accepted)
}

/// Accept one eligible frame (see [`bulk_accept`]); `Ok(false)` when it is not eligible (any
/// more).
pub fn accept_one(dir: &Path, rel: &str, classes: &[String], now: &str) -> anyhow::Result<bool> {
    let wanted = Wanted {
        classes,
        hidden: &[],
    };
    accept_frame(dir, rel, &wanted, now).map(|n| n.is_some())
}

/// Save the unreviewed frames of `dir` whose auto-label triage alone settles them
/// ([`super::auto_accept_eligible`]; `filters` with status and `auto` forced to unreviewed /
/// accept) and that `keep` selects as reviewed with `bulk_accepted`, at most `limit` in queue
/// order, the labels checked against `classes` at time `now`. Pending boxes of classes that map
/// to none of `classes`, or of the hidden classes `filters.wanted` names (the caller's live
/// `collect.hidden_classes`), are rejected, not accepted ([`accept_frame`]). `dry_run` only counts.
pub fn bulk_accept(
    dir: &Path,
    filters: &Filters,
    classes: &[String],
    limit: usize,
    now: &str,
    dry_run: bool,
    keep: &dyn Fn(&str) -> bool,
) -> BulkOutcome {
    let mut f = filters.clone();
    f.status = StatusFilter::Is(Status::Unreviewed);
    f.auto = Some(AutoFilter::Accept);
    let built = queue::build(dir, &f);
    let eligible: Vec<&str> = built
        .items
        .iter()
        .map(|i| i.frame.as_str())
        .filter(|r| keep(r))
        .collect();
    let mut out = BulkOutcome {
        eligible: eligible.len(),
        ..BulkOutcome::default()
    };
    if dry_run {
        out.done = eligible
            .iter()
            .take(LISTED)
            .map(|r| (*r).to_string())
            .collect();
        return out;
    }
    let wanted = Wanted {
        classes,
        hidden: f
            .wanted
            .as_ref()
            .map_or(&[], |(_, hidden)| hidden.as_slice()),
    };
    for rel in eligible.into_iter().take(limit) {
        match accept_frame(dir, rel, &wanted, now) {
            Ok(Some(n)) => {
                out.boxes += n;
                out.done(rel);
            }
            Ok(None) => out.failed(
                rel,
                "it changed since the list was built, or holds nothing wanted to accept, and \
                 needs a look; review it by hand",
            ),
            Err(e) => out.failed(rel, &format!("{e:#}")),
        }
    }
    out
}

/// Apply `verdict` at `spot` ([`super::repeats::Spot`]) on the unreviewed frames of
/// `spot.camera` in `dir` that `filters` (camera and status forced) and `keep` select: every
/// pending model / auto box of the object there is rejected or accepted, the frame marked
/// ([`LabelFile::repeat_spot`]). A frame without a label file (never opened) is taken as the page
/// shows it, pre-filled from its sidecar with `classes` / `hidden` ([`super::prefill`]). Frames of
/// another size than the spot's are passed over; at most `limit` frames in queue order. `dry_run`
/// only counts.
#[allow(clippy::too_many_arguments)]
pub fn bulk_spot(
    dir: &Path,
    filters: &Filters,
    classes: &[String],
    hidden: &[String],
    spot: &super::repeats::Spot,
    verdict: &super::repeats::Verdict,
    limit: usize,
    dry_run: bool,
    keep: &dyn Fn(&str) -> bool,
) -> BulkOutcome {
    let mut f = filters.clone();
    f.status = StatusFilter::Is(Status::Unreviewed);
    f.camera = Some(spot.camera.clone());
    let built = queue::build(dir, &f);
    let fresh = |frame: &Frame| {
        let rec = frame.record().ok()?;
        Some(super::prefill(&rec, classes, hidden))
    };
    let mut eligible = Vec::new();
    let mut walker = Walker::new(dir);
    for item in built.items.into_iter().filter(|i| keep(&i.frame)) {
        let Some(frame) = Frame::resolve_in(&mut walker, &item.frame) else {
            continue;
        };
        let Ok(rec) = frame.record() else {
            continue;
        };
        if (rec.width, rec.height) != (spot.width, spot.height) {
            continue;
        }
        let l = match frame.labels() {
            Ok(Some(l)) => l,
            Ok(None) => super::prefill(&rec, classes, hidden),
            Err(_) => continue,
        };
        let n = spot.count(&l, verdict);
        if n > 0 {
            eligible.push((item.frame, n, 0));
        }
    }
    edit_listed(dir, eligible, limit, dry_run, &fresh, &|l| {
        (spot.apply(l, verdict), 0)
    })
}

/// Save the repeat-only frames ([`super::repeats::repeats_only`]: spot edits settled every box,
/// none pending) that `filters` (status and `auto` forced) and `keep` select as reviewed with
/// `bulk_accepted`, at most `limit` in queue order, checked against `classes` at time `now`. A
/// frame is re-checked under the write lock: one with a pending box is never saved.
pub fn bulk_accept_repeats(
    dir: &Path,
    filters: &Filters,
    classes: &[String],
    limit: usize,
    now: &str,
    dry_run: bool,
    keep: &dyn Fn(&str) -> bool,
) -> BulkOutcome {
    let mut f = filters.clone();
    f.status = StatusFilter::Is(Status::Unreviewed);
    f.auto = Some(AutoFilter::Repeats);
    let built = queue::build(dir, &f);
    let eligible: Vec<String> = built
        .items
        .into_iter()
        .map(|i| i.frame)
        .filter(|r| keep(r))
        .collect();
    let mut out = BulkOutcome {
        eligible: eligible.len(),
        ..BulkOutcome::default()
    };
    if dry_run {
        out.done = eligible.iter().take(LISTED).cloned().collect();
        return out;
    }
    for rel in eligible.iter().take(limit) {
        let saved = Frame::resolve(dir, rel)
            .ok_or_else(|| anyhow::anyhow!("the frame is gone"))
            .and_then(|frame| {
                let rec = frame.record()?;
                super::update(&frame.labels, |cur| {
                    let Some(mut l) = cur.filter(super::repeats::repeats_only) else {
                        return Ok(None);
                    };
                    l.status = Status::Reviewed;
                    // A repeat-only frame holds no box a reviewer did not settle, so a check
                    // from before a rerun reopened it still stands; `normalize` ends the memory.
                    let checked = l.checked_before_reopen;
                    let mut n = super::normalize(l, classes, rec.width, rec.height, now)?;
                    n.bulk_accepted = true;
                    n.overlaps_checked |= checked;
                    Ok(Some(n))
                })
            });
        match saved {
            Ok(true) => out.done(rel),
            Ok(false) => out.failed(
                rel,
                "it changed since the list was built and needs a look; review it by hand",
            ),
            Err(e) => out.failed(rel, &format!("{e:#}")),
        }
    }
    out
}

/// Frames on disk for the bulk-edit tests (also [`crate::training::finish`]'s).
#[cfg(test)]
#[allow(clippy::unwrap_used)]
pub(crate) mod test_support {
    use super::super::{AutoInfo, LabelBox, LabelFile, Source, Triage};
    use super::BoxState;
    use std::path::Path;

    /// A pending auto box of `label` at `score` triaged `triage`.
    pub(crate) fn auto_box(label: &str, score: f32, triage: Triage) -> LabelBox {
        LabelBox {
            label: label.into(),
            bbox: [2.0, 2.0, 30.0, 40.0],
            source: Source::Auto,
            score: Some(score),
            state: BoxState::Pending,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: AutoInfo {
                triage: Some(triage),
                ..AutoInfo::default()
            },
        }
    }

    /// Write frame `ir/cam/2026-10-21/<stem>.jpg` of `dir` (64 x 48, sidecar `source`, queued for
    /// review) with an unreviewed label file of `boxes`; returns its rel.
    pub(crate) fn put_frame(
        dir: &Path,
        stem: &str,
        source: Option<&str>,
        boxes: Vec<LabelBox>,
    ) -> String {
        put_frame_at(dir, "cam", stem, (64, 48), source, boxes)
    }

    /// [`put_frame`] of camera `camera` with a `w` x `h` frame.
    pub(crate) fn put_frame_at(
        dir: &Path,
        camera: &str,
        stem: &str,
        (w, h): (u32, u32),
        source: Option<&str>,
        boxes: Vec<LabelBox>,
    ) -> String {
        let day = dir.join("ir").join(camera).join("2026-10-21");
        std::fs::create_dir_all(&day).unwrap();
        image::RgbImage::from_pixel(w, h, image::Rgb([90, 90, 90]))
            .save(day.join(format!("{stem}.jpg")))
            .unwrap();
        let mut r = crate::collect::record::empty_record();
        r.id = stem.into();
        r.scene = "ir".into();
        r.camera = camera.into();
        r.timestamp = "2026-10-21T01:02:03.004Z".into();
        r.width = w;
        r.height = h;
        r.source = source.map(str::to_string);
        std::fs::write(
            day.join(format!("{stem}.json")),
            serde_json::to_vec(&r).unwrap(),
        )
        .unwrap();
        let rel = format!("ir/{camera}/2026-10-21/{stem}.jpg");
        crate::collect::layout::append_review(dir, std::slice::from_ref(&rel)).unwrap();
        let f = LabelFile {
            boxes,
            ..LabelFile::default()
        };
        let frame = super::Frame::resolve(dir, &rel).unwrap();
        super::super::write(&frame.labels, &f).unwrap();
        rel
    }

    /// The label file of `rel`.
    pub(crate) fn labels_of(dir: &Path, rel: &str) -> LabelFile {
        super::Frame::resolve(dir, rel)
            .unwrap()
            .labels()
            .unwrap()
            .unwrap()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::super::{Origin, Triage, auto_accept_eligible};
    use super::test_support::{auto_box, labels_of, put_frame, put_frame_at};
    use super::*;

    #[test]
    fn a_spot_everywhere_takes_only_that_cameras_unreviewed_frames_of_its_size() {
        use super::super::repeats::{ACCEPT_SPOT_IOU, REJECT_SPOT_IOU, Spot, Verdict};
        let dir = std::env::temp_dir().join(format!("bop-spot-{}", uuid::Uuid::new_v4()));
        let car = || auto_box("car", 0.4, Triage::Review); // at [2, 2, 30, 40]
        let mut shifted = car();
        shifted.bbox = [3.0, 2.0, 31.0, 40.0];
        let mut far = car();
        far.bbox = [34.0, 2.0, 62.0, 40.0];
        let a = put_frame(
            &dir,
            "a",
            None,
            vec![car(), auto_box("dog", 0.4, Triage::Review)],
        );
        let b = put_frame(&dir, "b", None, vec![shifted.clone()]);
        let c = put_frame(&dir, "c", None, vec![far]);
        let d = put_frame_at(&dir, "drive", "d", (64, 48), None, vec![car()]);
        let e = put_frame_at(&dir, "cam", "e", (32, 24), None, vec![car()]);
        let f = put_frame(&dir, "f", None, vec![car()]);
        let mut reviewed = labels_of(&dir, &f);
        reviewed.status = Status::Reviewed;
        super::super::write(&Frame::resolve(&dir, &f).unwrap().labels, &reviewed).unwrap();
        let spot = Spot {
            camera: "cam".into(),
            label: "car".into(),
            bbox: [2.0, 2.0, 30.0, 40.0],
            width: 64,
            height: 48,
            min_iou: REJECT_SPOT_IOU,
        };
        let reject = Verdict::Reject(Some(super::super::RejectReason::StaticObject));
        let all = Filters::default();
        let cls = ["car".to_string(), "dog".to_string()];
        let before: Vec<LabelFile> = [&a, &b, &c, &d, &e, &f]
            .iter()
            .map(|r| labels_of(&dir, r))
            .collect();
        let dry = bulk_spot(&dir, &all, &cls, &[], &spot, &reject, 100, true, &|_| true);
        assert_eq!((dry.eligible, dry.boxes, dry.done_count), (2, 2, 0));
        let after: Vec<LabelFile> = [&a, &b, &c, &d, &e, &f]
            .iter()
            .map(|r| labels_of(&dir, r))
            .collect();
        assert_eq!(before, after, "a dry run writes nothing");
        let o = bulk_spot(&dir, &all, &cls, &[], &spot, &reject, 100, false, &|_| true);
        assert_eq!((o.boxes, o.done_count), (2, 2));
        let la = labels_of(&dir, &a);
        assert_eq!(la.boxes[0].state, BoxState::Rejected);
        assert_eq!(
            la.boxes[0].reason.as_deref(),
            Some(super::super::repeats::REPEAT_SPOT_REASON)
        );
        assert_eq!(
            la.boxes[1].state,
            BoxState::Pending,
            "the dog is another object"
        );
        assert!(la.repeat_spot && labels_of(&dir, &b).repeat_spot);
        for (r, i) in [(&c, 2), (&d, 3), (&e, 4), (&f, 5)] {
            assert_eq!(
                labels_of(&dir, r),
                before[i],
                "{r}: another place, camera, size or reviewed"
            );
        }
        // Repeat-only: b (nothing pending) is saved; a (the dog) never is.
        let acc = bulk_accept_repeats(
            &dir,
            &all,
            &["car".to_string(), "dog".to_string()],
            100,
            "now",
            true,
            &|_| true,
        );
        assert_eq!((acc.eligible, acc.done.clone()), (1, vec![b.clone()]));
        let acc = bulk_accept_repeats(
            &dir,
            &all,
            &["car".to_string(), "dog".to_string()],
            100,
            "now",
            false,
            &|_| true,
        );
        assert_eq!(acc.done_count, 1);
        let lb = labels_of(&dir, &b);
        assert_eq!(lb.status, Status::Reviewed);
        assert!(lb.bulk_accepted && !lb.repeat_spot);
        assert_eq!(labels_of(&dir, &a).status, Status::Unreviewed);
        // An accept needs IoU 0.8: a box at 0.75 stays pending.
        let mut near = car();
        near.bbox = [2.0, 2.0, 30.0, 33.5]; // 31.5 / 38 high: IoU ~0.83
        let mut loose = car();
        loose.bbox = [2.0, 2.0, 30.0, 30.5]; // IoU 0.75
        let g = put_frame(&dir, "g", None, vec![near, loose]);
        let accept_spot = Spot {
            min_iou: ACCEPT_SPOT_IOU,
            ..spot
        };
        let o = bulk_spot(
            &dir,
            &all,
            &cls,
            &[],
            &accept_spot,
            &Verdict::Accept,
            100,
            false,
            &|_| true,
        );
        assert_eq!(o.boxes, 1);
        let lg = labels_of(&dir, &g);
        assert_eq!(
            (lg.boxes[0].state, lg.boxes[1].state),
            (BoxState::Accepted, BoxState::Pending)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_spot_everywhere_takes_frames_never_opened_as_the_page_shows_them() {
        use super::super::repeats::{REJECT_SPOT_IOU, Spot, Verdict};
        let dir = std::env::temp_dir().join(format!("bop-spot-new-{}", uuid::Uuid::new_v4()));
        // A frame with only its sidecar (no label file): the model's car at the spot.
        let rel = put_frame(&dir, "fresh", None, vec![]);
        let frame = Frame::resolve(&dir, &rel).unwrap();
        std::fs::remove_file(&frame.labels).unwrap();
        let mut rec = frame.record().unwrap();
        rec.predictions.push(crate::collect::record::Det {
            label: "car".into(),
            score: 0.7,
            bbox: [2.0, 2.0, 30.0, 40.0],
        });
        let json = frame.labels.with_file_name("fresh.json");
        std::fs::write(&json, serde_json::to_vec(&rec).unwrap()).unwrap();
        let spot = Spot {
            camera: "cam".into(),
            label: "car".into(),
            bbox: [2.0, 2.0, 30.0, 40.0],
            width: 64,
            height: 48,
            min_iou: REJECT_SPOT_IOU,
        };
        let cls = ["car".to_string()];
        let all = Filters::default();
        let reject = Verdict::Reject(None);
        let dry = bulk_spot(&dir, &all, &cls, &[], &spot, &reject, 100, true, &|_| true);
        assert_eq!((dry.eligible, dry.boxes), (1, 1));
        assert!(
            frame.labels().unwrap().is_none(),
            "a dry run writes nothing"
        );
        let o = bulk_spot(&dir, &all, &cls, &[], &spot, &reject, 100, false, &|_| true);
        assert_eq!((o.boxes, o.done_count), (1, 1));
        let l = labels_of(&dir, &rel);
        assert_eq!(l.status, Status::Unreviewed);
        assert_eq!(l.boxes[0].state, BoxState::Rejected);
        assert!(l.repeat_spot);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reject_low_leaves_an_audit_sample_to_be_looked_at() {
        let mut b = auto_box("car", 0.1, Triage::Review);
        b.auto.audit = true;
        let mut f = LabelFile {
            boxes: vec![b],
            ..LabelFile::default()
        };
        assert_eq!(reject_low(&mut f, 0.3), 0);
        assert_eq!(f.boxes[0].state, BoxState::Pending);
    }

    #[test]
    fn reject_low_touches_only_pending_auto_boxes_below_the_score() {
        let mut model = auto_box("car", 0.1, Triage::Review);
        model.source = Source::Model;
        model.origin = Some(Origin::Prediction);
        let mut human = auto_box("dog", 0.1, Triage::Review);
        human.source = Source::Human;
        let mut decided = auto_box("person", 0.1, Triage::Review);
        decided.state = BoxState::Accepted;
        let mut unscored = auto_box("person", 0.1, Triage::Review);
        unscored.score = None;
        let mut f = LabelFile {
            boxes: vec![
                auto_box("person", 0.9, Triage::Accept),
                auto_box("person", 0.29, Triage::Review),
                auto_box("car", 0.3, Triage::Review),
                model,
                human,
                decided,
                unscored,
            ],
            ..LabelFile::default()
        };
        let before = f.clone();
        assert_eq!(count_low(&f, 0.3), 1);
        assert_eq!(reject_low(&mut f, 0.3), 1);
        assert_eq!(f.boxes[1].state, BoxState::Rejected);
        assert_eq!(f.boxes[1].reason.as_deref(), Some(LOW_SCORE_REASON));
        for i in [0, 2, 3, 4, 5, 6] {
            assert_eq!(f.boxes[i], before.boxes[i], "box {i}");
        }
        assert_eq!(f.status, Status::Unreviewed);
        assert_eq!(reject_low(&mut f, 0.3), 0, "already rejected");
        // Reviewed and skipped frames are untouched.
        for s in [Status::Reviewed, Status::Skipped] {
            let mut r = before.clone();
            r.status = s;
            assert_eq!(count_low(&r, 0.5), 0);
            assert_eq!(reject_low(&mut r, 0.5), 0);
            assert_eq!(r.boxes, before.boxes);
        }
    }

    #[test]
    fn rejecting_the_low_review_boxes_leaves_a_frame_the_triage_settles() {
        let mut f = LabelFile {
            boxes: vec![
                auto_box("person", 0.9, Triage::Accept),
                auto_box("car", 0.2, Triage::Review),
            ],
            ..LabelFile::default()
        };
        assert!(!auto_accept_eligible(&f));
        assert_eq!(reject_low(&mut f, 0.3), 1);
        assert!(auto_accept_eligible(&f));
        // A review box above the minimum still needs a look.
        let mut g = LabelFile {
            boxes: vec![
                auto_box("person", 0.9, Triage::Accept),
                auto_box("car", 0.5, Triage::Review),
            ],
            ..LabelFile::default()
        };
        assert_eq!(reject_low(&mut g, 0.3), 0);
        assert!(!auto_accept_eligible(&g));
    }

    /// A pending model box of `label` from the sidecar part `origin`.
    fn model_box(label: &str, origin: Origin) -> super::super::LabelBox {
        let mut b = auto_box(label, 0.7, Triage::Review);
        b.source = Source::Model;
        b.origin = Some(origin);
        b.auto = Default::default();
        b
    }

    fn classes() -> Vec<String> {
        ["person", "car", "potted plant"].map(String::from).to_vec()
    }

    /// The wanted classes of `classes` without `hidden`.
    fn wanted<'a>(classes: &'a [String], hidden: &'a [String]) -> Wanted<'a> {
        Wanted { classes, hidden }
    }

    #[test]
    fn reject_other_class_touches_only_pending_proposals_off_the_list() {
        let mut decided = model_box("clock", Origin::Prediction);
        decided.state = BoxState::Accepted;
        let mut human = model_box("clock", Origin::Prediction);
        human.source = Source::Human;
        let mut f = LabelFile {
            boxes: vec![
                model_box("person", Origin::Prediction),
                model_box("parking meter", Origin::Prediction),
                model_box("pottedplant", Origin::BelowThreshold),
                model_box("Car", Origin::Suppressed),
                auto_box("clock", 0.9, Triage::Review),
                decided,
                human,
            ],
            ..LabelFile::default()
        };
        let before = f.clone();
        let cl = classes();
        let w = wanted(&cl, &[]);
        assert_eq!(count_other_class(&f, &w), (2, 2));
        assert_eq!(reject_other_class(&mut f, &w), (2, 2));
        assert_eq!(f.boxes[1].state, BoxState::Rejected);
        assert_eq!(f.boxes[1].reason.as_deref(), Some(OTHER_CLASS_REASON));
        assert_eq!(f.boxes[1].label, "parking meter");
        // An auto box of another class too.
        assert_eq!(
            (f.boxes[4].state, f.boxes[4].reason.as_deref()),
            (BoxState::Rejected, Some(OTHER_CLASS_REASON))
        );
        // An alias: renamed, still pending, the model's spelling kept.
        assert_eq!(
            (
                f.boxes[2].label.as_str(),
                f.boxes[2].model_label.as_deref(),
                f.boxes[2].state
            ),
            ("potted plant", Some("pottedplant"), BoxState::Pending)
        );
        // Another case only: renamed without a model label.
        assert_eq!(
            (f.boxes[3].label.as_str(), f.boxes[3].model_label.as_deref()),
            ("car", None)
        );
        for i in [0, 5, 6] {
            assert_eq!(f.boxes[i], before.boxes[i], "box {i}");
        }
        assert_eq!(f.status, Status::Unreviewed);
        assert_eq!(reject_other_class(&mut f, &w), (0, 0), "done");
        // Reviewed and skipped frames are untouched.
        for s in [Status::Reviewed, Status::Skipped] {
            let mut r = before.clone();
            r.status = s;
            assert_eq!(count_other_class(&r, &w), (0, 0));
            assert_eq!(reject_other_class(&mut r, &w), (0, 0));
            assert_eq!(r.boxes, before.boxes);
        }
    }

    #[test]
    fn pending_proposals_of_a_hidden_class_are_rejected_human_boxes_kept() {
        let mut human = auto_box("car", 0.9, Triage::Review);
        human.source = Source::Human;
        human.score = None;
        let mut decided = auto_box("car", 0.9, Triage::Accept);
        decided.state = BoxState::Accepted;
        let mut f = LabelFile {
            boxes: vec![
                model_box("person", Origin::Prediction),
                model_box("Car", Origin::Prediction),
                auto_box("car", 0.9, Triage::Accept),
                model_box("pottedplant", Origin::BelowThreshold),
                human,
                decided,
            ],
            ..LabelFile::default()
        };
        let before = f.clone();
        let cl = classes();
        let hidden = vec!["CAR".to_string()];
        let w = wanted(&cl, &hidden);
        assert_eq!(count_other_class(&f, &w), (2, 1));
        assert_eq!(reject_other_class(&mut f, &w), (2, 1));
        let got: Vec<(&str, BoxState)> = f
            .boxes
            .iter()
            .map(|b| (b.label.as_str(), b.state))
            .collect();
        assert_eq!(
            got,
            [
                ("person", BoxState::Pending),
                ("Car", BoxState::Rejected),
                ("car", BoxState::Rejected),
                ("potted plant", BoxState::Pending),
                ("car", BoxState::Pending),
                ("car", BoxState::Accepted),
            ]
        );
        assert_eq!(f.boxes[4], before.boxes[4], "a human box is the reviewer's");
        assert_eq!(f.boxes[5], before.boxes[5], "a decided box stays");
        // Whatever the status: the save of the Review tab.
        let mut r = before.clone();
        r.status = Status::Reviewed;
        assert_eq!(reject_unwanted_proposals(&mut r, &w), (2, 1));
    }

    #[test]
    fn reviewed_training_boxes_of_unwanted_classes_only_on_request() {
        let mut accepted_model = model_box("car", Origin::Prediction);
        accepted_model.state = BoxState::Accepted;
        let mut added = auto_box("dog", 0.9, Triage::Review);
        added.source = Source::Human;
        added.state = BoxState::Added;
        added.score = None;
        let mut person = model_box("person", Origin::Prediction);
        person.state = BoxState::Accepted;
        let mut rejected = model_box("car", Origin::Suppressed);
        rejected.state = BoxState::Rejected;
        let reviewed = LabelFile {
            status: Status::Reviewed,
            boxes: vec![person, accepted_model, added, rejected],
            ..LabelFile::default()
        };
        // `dog` was dropped from the list, `car` is hidden.
        let cl = classes();
        let hidden = vec!["car".to_string()];
        let w = wanted(&cl, &hidden);
        let mut untouched = reviewed.clone();
        assert_eq!(reject_other_class(&mut untouched, &w), (0, 0));
        assert_eq!(untouched, reviewed, "the unreviewed reject leaves it alone");
        assert_eq!(count_unwanted_reviewed(&reviewed, &w), 2);
        let mut f = reviewed.clone();
        assert_eq!(reject_unwanted_reviewed(&mut f, &w), 2);
        assert_eq!(f.status, Status::Reviewed);
        let got: Vec<(&str, BoxState)> = f
            .boxes
            .iter()
            .map(|b| (b.label.as_str(), b.state))
            .collect();
        assert_eq!(
            got,
            [
                ("person", BoxState::Accepted),
                ("car", BoxState::Rejected),
                ("dog", BoxState::Rejected),
                ("car", BoxState::Rejected),
            ]
        );
        assert_eq!(f.boxes[1].reason.as_deref(), Some(OTHER_CLASS_REASON));
        assert_eq!(f.boxes[3].reason, None, "already rejected: unchanged");
        assert_eq!(reject_unwanted_reviewed(&mut f, &w), 0, "done");
        // Still a valid reviewed file for the new list (the dog was dropped from it).
        let n = super::super::normalize(f, &cl, 64, 48, "t").unwrap();
        assert_eq!(n.status, Status::Reviewed);
        assert_eq!(
            n.training().map(|b| b.label.as_str()).collect::<Vec<_>>(),
            ["person"]
        );
        assert_eq!(
            n.boxes.len(),
            3,
            "the rejected human box is dropped at save"
        );
        // Unreviewed and skipped frames are not touched by the reviewed reject.
        for s in [Status::Unreviewed, Status::Skipped] {
            let mut r = reviewed.clone();
            r.status = s;
            assert_eq!(reject_unwanted_reviewed(&mut r, &w), 0);
        }
    }

    #[test]
    fn rejecting_an_off_list_prediction_leaves_a_frame_the_triage_settles() {
        let mut f = LabelFile {
            boxes: vec![
                auto_box("person", 0.9, Triage::Accept),
                model_box("clock", Origin::Prediction),
                model_box("dog", Origin::BelowThreshold),
            ],
            ..LabelFile::default()
        };
        assert!(!auto_accept_eligible(&f), "the clock would be accepted");
        let cl = classes();
        assert_eq!(reject_other_class(&mut f, &wanted(&cl, &[])), (2, 0));
        assert!(auto_accept_eligible(&f));
        let n = super::super::normalize(
            LabelFile {
                status: Status::Reviewed,
                ..f
            },
            &classes(),
            64,
            48,
            "t",
        )
        .unwrap();
        assert_eq!(n.training().count(), 1, "the save no longer fails");
    }

    /// A box of the "reviewed frame" shape at `bbox`.
    fn car_at(bbox: [f32; 4], score: f32) -> super::super::LabelBox {
        let mut b = auto_box("car", score, Triage::Accept);
        b.bbox = bbox;
        b.state = BoxState::Accepted;
        b
    }

    /// Make `rel` a reviewed frame as of `at`.
    fn mark_reviewed(dir: &Path, rel: &str, at: &str) {
        let frame = Frame::resolve(dir, rel).unwrap();
        let mut l = labels_of(dir, rel);
        l.status = Status::Reviewed;
        l.reviewed_at = Some(at.into());
        super::super::write(&frame.labels, &l).unwrap();
    }

    /// A frame reopened by a re-label: reviewed once, sent back with a note and `boxes`.
    fn reopened(dir: &Path, stem: &str, boxes: Vec<super::super::LabelBox>) -> String {
        let rel = put_frame(dir, stem, None, boxes);
        let frame = Frame::resolve(dir, &rel).unwrap();
        let mut l = labels_of(dir, &rel);
        l.reviewed_at = Some("2026-10-01T00:00:00Z".into());
        l.disagreement = Some("re-label by sam3: 1 new box".into());
        super::super::write(&frame.labels, &l).unwrap();
        rel
    }

    #[test]
    fn restore_takes_only_reopened_frames_with_nothing_pending() {
        let root = std::env::temp_dir().join(format!("bop-restore-{}", uuid::Uuid::new_v4()));
        let dir = root.as_path();
        let any = Filters {
            status: StatusFilter::Any,
            ..Filters::default()
        };
        let mut rejected = auto_box("car", 0.8, Triage::Review);
        rejected.state = BoxState::Rejected;
        let settled = car_at([1.0, 1.0, 20.0, 20.0], 0.8);
        let done = reopened(dir, "a", vec![settled.clone(), rejected.clone()]);
        let pending = reopened(
            dir,
            "b",
            vec![settled.clone(), auto_box("dog", 0.9, Triage::Review)],
        );
        // Never reviewed, whatever else it has.
        let never = put_frame(dir, "c", None, vec![settled.clone()]);
        let sent = put_frame(dir, "d", None, vec![settled.clone()]);
        let frame = Frame::resolve(dir, &sent).unwrap();
        let mut l = labels_of(dir, &sent);
        l.disagreement = Some("re-label by m: 1 new box".into());
        super::super::write(&frame.labels, &l).unwrap();
        // Reviewed once, reopened with no note (a proposal merge of an older build).
        let quiet = put_frame(dir, "e", None, vec![settled.clone()]);
        let frame = Frame::resolve(dir, &quiet).unwrap();
        let mut l = labels_of(dir, &quiet);
        l.reviewed_at = Some("2026-10-01T00:00:00Z".into());
        super::super::write(&frame.labels, &l).unwrap();
        let reviewed = put_frame(dir, "f", None, vec![settled.clone()]);
        mark_reviewed(dir, &reviewed, "2026-10-01T00:00:00Z");
        let empty = reopened(dir, "g", vec![settled.clone()]);
        let frame = Frame::resolve(dir, &empty).unwrap();
        let mut l = labels_of(dir, &empty);
        l.empty_confirmed = true;
        super::super::write(&frame.labels, &l).unwrap();

        let untouched = [&pending, &never, &sent, &quiet, &reviewed, &empty];
        let before: Vec<LabelFile> = untouched.iter().map(|r| labels_of(dir, r)).collect();
        let dry = bulk_restore_reopened(dir, &any, 500, true, &|_| true);
        assert_eq!((dry.eligible, dry.done_count), (1, 0));
        assert_eq!(dry.done, std::slice::from_ref(&done));
        assert_eq!(labels_of(dir, &done).status, Status::Unreviewed);

        let was = labels_of(dir, &done);
        let o = bulk_restore_reopened(dir, &any, 500, false, &|_| true);
        assert_eq!((o.eligible, o.done_count, o.failed_count), (1, 1, 0));
        let l = labels_of(dir, &done);
        assert_eq!(l.status, Status::Reviewed);
        assert_eq!(l.disagreement, None);
        assert_eq!(l.boxes, was.boxes, "boxes as they were");
        assert_eq!(l.reviewed_at, was.reviewed_at, "the review it goes back to");
        let after: Vec<LabelFile> = untouched.iter().map(|r| labels_of(dir, r)).collect();
        assert_eq!(after, before);
        // It exports again, and a rerun finds nothing.
        assert_eq!(
            bulk_restore_reopened(dir, &any, 500, false, &|_| true).eligible,
            0
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn restore_checks_again_under_the_lock_and_honours_the_scope() {
        let mut f = LabelFile {
            disagreement: Some("x".into()),
            reviewed_at: Some("t".into()),
            repeat_spot: true,
            ..LabelFile::default()
        };
        // A box that went pending between the list and the write.
        let mut pending = f.clone();
        pending.boxes.push(auto_box("dog", 0.9, Triage::Review));
        let frozen = pending.clone();
        assert_eq!(restore_reopened(&mut pending), 0);
        assert_eq!(pending, frozen);
        assert_eq!(restore_reopened(&mut f), 1);
        assert_eq!(f.status, Status::Reviewed);
        assert!(!f.repeat_spot, "as a save clears it");
        assert_eq!(restore_reopened(&mut f), 0, "reviewed already");

        let root = std::env::temp_dir().join(format!("bop-restore-{}", uuid::Uuid::new_v4()));
        let dir = root.as_path();
        let any = Filters::default();
        let a = reopened(dir, "a", vec![]);
        let b = reopened(dir, "b", vec![]);
        let o = bulk_restore_reopened(dir, &any, 1, false, &|_| true);
        assert_eq!((o.eligible, o.done_count), (2, 1));
        let o = bulk_restore_reopened(dir, &any, 500, false, &|r| r != b);
        assert_eq!(o.eligible, 0, "{a} went first, {b} is out of scope");
        assert_eq!(
            [labels_of(dir, &a).status, labels_of(dir, &b).status]
                .iter()
                .filter(|s| **s == Status::Reviewed)
                .count(),
            1
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn overlaps_left_reviewed_leaves_out_unreviewed_frames() {
        let dir = std::env::temp_dir().join(format!("bop-stacked-{}", uuid::Uuid::new_v4()));
        let (a, small) = ([0.0, 0.0, 40.0, 40.0], [5.0, 5.0, 25.0, 25.0]);
        let reviewed = put_frame(&dir, "r", None, vec![car_at(a, 0.9), car_at(small, 0.3)]);
        put_frame(&dir, "u", None, vec![car_at(a, 0.9), car_at(small, 0.3)]);
        mark_reviewed(&dir, &reviewed, "2026-10-01T00:00:00Z");
        let cl = classes();
        let w = wanted(&cl, &[]);
        let o = bulk_reject_stacked(&dir, &Filters::default(), &w, 100, true, &|_| true);
        assert_eq!((o.overlaps_left, o.overlaps_left_reviewed), (2, 1));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn stacked_duplicates_go_on_reviewed_frames_which_stay_reviewed() {
        use super::super::RejectReason;
        use super::super::stacks::DUPLICATE_REASON;
        let dir = std::env::temp_dir().join(format!("bop-stacked-{}", uuid::Uuid::new_v4()));
        let (a, near, small) = (
            [0.0, 0.0, 40.0, 40.0],
            [0.0, 0.0, 40.0, 38.0],
            [5.0, 5.0, 25.0, 25.0],
        );
        // Reviewed, stacked; reviewed on another camera; unreviewed, pending; nested only.
        let r = put_frame(&dir, "r", None, vec![car_at(a, 0.9), car_at(near, 0.4)]);
        let other = put_frame_at(
            &dir,
            "drive",
            "o",
            (64, 48),
            None,
            vec![car_at(a, 0.9), car_at(near, 0.4), car_at(small, 0.3)],
        );
        let mut p = car_at(near, 0.4);
        p.state = BoxState::Pending;
        let u = put_frame(&dir, "u", None, vec![car_at(a, 0.9), p]);
        let n = put_frame(&dir, "n", None, vec![car_at(a, 0.9), car_at(small, 0.3)]);
        for rel in [&r, &other, &n] {
            mark_reviewed(&dir, rel, "2026-10-01T00:00:00Z");
        }
        // Past the default filters: out of the review queue altogether.
        std::fs::remove_file(dir.join(crate::collect::layout::REVIEW_QUEUE_FILE)).unwrap();
        let cl = classes();
        let w = wanted(&cl, &[]);
        let all = Filters::default();
        let everything = |_: &str| true;
        let before: Vec<LabelFile> = [&r, &other, &u, &n]
            .iter()
            .map(|f| labels_of(&dir, f))
            .collect();
        let dry = bulk_reject_stacked(&dir, &all, &w, 100, true, &everything);
        assert_eq!(
            (
                dry.outcome.eligible,
                dry.outcome.boxes,
                dry.outcome.done_count
            ),
            (3, 3, 0)
        );
        assert_eq!(
            dry.overlaps_left, 2,
            "the frame with a nested box left and the nested frame, after the removal"
        );
        assert_eq!(dry.overlaps_left_reviewed, 2, "both of those are reviewed");
        let after: Vec<LabelFile> = [&r, &other, &u, &n]
            .iter()
            .map(|f| labels_of(&dir, f))
            .collect();
        assert_eq!(before, after, "a dry run writes nothing");
        // One camera first.
        let drive = Filters {
            camera: Some("drive".into()),
            ..Filters::default()
        };
        let o = bulk_reject_stacked(&dir, &drive, &w, 100, false, &everything);
        assert_eq!((o.outcome.boxes, o.outcome.done_count), (1, 1));
        assert_eq!(o.overlaps_left, 1);
        assert_eq!(
            labels_of(&dir, &r),
            before[0],
            "another camera is untouched"
        );
        assert_eq!(
            labels_of(&dir, &u),
            before[2],
            "another camera is untouched"
        );
        // The rest.
        let o = bulk_reject_stacked(&dir, &all, &w, 100, false, &everything);
        assert_eq!(
            (o.outcome.eligible, o.outcome.boxes, o.outcome.done_count),
            (2, 2, 2)
        );
        assert_eq!(o.overlaps_left, 2, "the same answer as the dry run");
        assert_eq!(o.overlaps_left_reviewed, dry.overlaps_left_reviewed);
        let lr = labels_of(&dir, &r);
        assert_eq!(lr.status, Status::Reviewed);
        assert_eq!(
            lr.reviewed_at.as_deref(),
            Some("2026-10-01T00:00:00Z"),
            "no re-save"
        );
        assert_eq!(lr.boxes[0], before[0].boxes[0], "the keeper is untouched");
        let dropped = &lr.boxes[1];
        assert_eq!(
            (
                dropped.state,
                dropped.reject_reason.clone(),
                dropped.reason.as_deref()
            ),
            (
                BoxState::Rejected,
                Some(RejectReason::Duplicate),
                Some(DUPLICATE_REASON)
            )
        );
        let lu = labels_of(&dir, &u);
        assert_eq!(lu.status, Status::Unreviewed);
        assert_eq!(lu.boxes[1].state, BoxState::Rejected);
        assert_eq!(
            labels_of(&dir, &n),
            before[3],
            "nested boxes are a reviewer's"
        );
        // A second run finds nothing and writes nothing.
        let settled: Vec<LabelFile> = [&r, &other, &u, &n]
            .iter()
            .map(|f| labels_of(&dir, f))
            .collect();
        let again = bulk_reject_stacked(&dir, &all, &w, 100, false, &everything);
        assert_eq!(
            (
                again.outcome.eligible,
                again.outcome.boxes,
                again.outcome.done_count
            ),
            (0, 0, 0)
        );
        assert_eq!(again.overlaps_left, 2);
        let now: Vec<LabelFile> = [&r, &other, &u, &n]
            .iter()
            .map(|f| labels_of(&dir, f))
            .collect();
        assert_eq!(now, settled);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_restored_frame_is_checked_again_only_if_it_was_before() {
        let root = std::env::temp_dir().join(format!("bop-restore-{}", uuid::Uuid::new_v4()));
        let dir = root.as_path();
        let (a, near) = ([0.0, 0.0, 40.0, 40.0], [0.0, 0.0, 40.0, 38.0]);
        let cars = || vec![car_at(a, 0.9), car_at(near, 0.4)];
        let kept = reopened(dir, "a", cars());
        let open = reopened(dir, "b", cars());
        let frame = Frame::resolve(dir, &kept).unwrap();
        let mut l = labels_of(dir, &kept);
        l.checked_before_reopen = true;
        super::super::write(&frame.labels, &l).unwrap();

        let all = Filters::default();
        let cl = classes();
        let w = wanted(&cl, &[]);
        // While the rerun has it open, the reviewer's check still holds the bulk back.
        let o = bulk_reject_stacked(dir, &all, &w, 100, true, &|_| true);
        assert_eq!(o.outcome.eligible, 1, "only the frame nobody checked");
        assert_eq!(o.outcome.done, vec![open.clone()]);

        let o = bulk_restore_reopened(dir, &all, 500, false, &|_| true);
        assert_eq!(o.done_count, 2);
        let l = labels_of(dir, &kept);
        assert!(l.overlaps_checked && !l.checked_before_reopen, "{l:?}");
        let l = labels_of(dir, &open);
        assert!(!l.overlaps_checked && !l.checked_before_reopen, "{l:?}");

        let o = bulk_reject_stacked(dir, &all, &w, 100, false, &|_| true);
        assert_eq!(o.outcome.done_count, 1, "only the frame nobody checked");
        assert_eq!(labels_of(dir, &kept).boxes[1].state, BoxState::Accepted);
        assert_eq!(labels_of(dir, &open).boxes[1].state, BoxState::Rejected);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn accepting_a_repeat_only_frame_keeps_a_check_from_before_the_reopen() {
        let root = std::env::temp_dir().join(format!("bop-repeat-check-{}", uuid::Uuid::new_v4()));
        let dir = root.as_path();
        let (a, near) = ([0.0, 0.0, 40.0, 40.0], [0.0, 0.0, 40.0, 38.0]);
        let cars = || vec![car_at(a, 0.9), car_at(near, 0.4)];
        let kept = reopened(dir, "a", cars());
        let open = reopened(dir, "b", cars());
        for (rel, checked) in [(&kept, true), (&open, false)] {
            let frame = Frame::resolve(dir, rel).unwrap();
            let mut l = labels_of(dir, rel);
            l.repeat_spot = true;
            l.checked_before_reopen = checked;
            super::super::write(&frame.labels, &l).unwrap();
        }
        let all = Filters::default();
        let o = bulk_accept_repeats(dir, &all, &classes(), 100, "now", false, &|_| true);
        assert_eq!(o.done_count, 2);
        let l = labels_of(dir, &kept);
        assert_eq!(l.status, Status::Reviewed);
        assert!(l.overlaps_checked && !l.checked_before_reopen, "{l:?}");
        let l = labels_of(dir, &open);
        assert_eq!(l.status, Status::Reviewed);
        assert!(!l.overlaps_checked && !l.checked_before_reopen, "{l:?}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn the_limit_and_a_reviewers_check_hold_the_stacked_reject_back() {
        let dir = std::env::temp_dir().join(format!("bop-stacked-{}", uuid::Uuid::new_v4()));
        let (a, near) = ([0.0, 0.0, 40.0, 40.0], [0.0, 0.0, 40.0, 38.0]);
        let first = put_frame(&dir, "a", None, vec![car_at(a, 0.9), car_at(near, 0.4)]);
        let second = put_frame(&dir, "b", None, vec![car_at(a, 0.9), car_at(near, 0.4)]);
        let kept = put_frame(&dir, "c", None, vec![car_at(a, 0.9), car_at(near, 0.4)]);
        let skipped = put_frame(&dir, "d", None, vec![car_at(a, 0.9), car_at(near, 0.4)]);
        let frame = Frame::resolve(&dir, &kept).unwrap();
        let mut l = labels_of(&dir, &kept);
        l.status = Status::Reviewed;
        l.overlaps_checked = true;
        super::super::write(&frame.labels, &l).unwrap();
        let frame = Frame::resolve(&dir, &skipped).unwrap();
        let mut l = labels_of(&dir, &skipped);
        l.status = Status::Skipped;
        super::super::write(&frame.labels, &l).unwrap();
        let cl = classes();
        let w = wanted(&cl, &[]);
        let all = Filters::default();
        let o = bulk_reject_stacked(&dir, &all, &w, 1, false, &|_| true);
        assert_eq!(
            (o.outcome.eligible, o.outcome.done_count, o.overlaps_left),
            (2, 1, 0),
            "a checked frame and a skipped one are neither eligible nor left to look at"
        );
        assert_eq!(labels_of(&dir, &first).boxes[1].state, BoxState::Rejected);
        assert_eq!(labels_of(&dir, &second).boxes[1].state, BoxState::Accepted);
        assert_eq!(labels_of(&dir, &kept).boxes[1].state, BoxState::Accepted);
        assert_eq!(labels_of(&dir, &skipped).boxes[1].state, BoxState::Accepted);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_stacked_reject_invalidates_the_cameras_history() {
        use super::super::repeats::history;
        let dir = std::env::temp_dir().join(format!("bop-stacked-{}", uuid::Uuid::new_v4()));
        let (a, near) = ([0.0, 0.0, 40.0, 40.0], [0.0, 0.0, 40.0, 38.0]);
        put_frame(&dir, "a", None, vec![car_at(a, 0.9), car_at(near, 0.4)]);
        assert_eq!(history(&dir, "cam").decisions.len(), 2);
        let cl = classes();
        let o = bulk_reject_stacked(
            &dir,
            &Filters::default(),
            &wanted(&cl, &[]),
            10,
            false,
            &|_| true,
        );
        assert_eq!(o.outcome.done_count, 1);
        // Seen at once, and the duplicate is no decision on the spot.
        assert_eq!(history(&dir, "cam").decisions.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_bulk_accept_rejects_pending_duplicates_and_keeps_explicit_accepts() {
        use super::super::RejectReason;
        let dir = std::env::temp_dir().join(format!("bop-stacked-{}", uuid::Uuid::new_v4()));
        let (a, near) = ([0.0, 0.0, 40.0, 40.0], [0.0, 0.0, 40.0, 38.0]);
        let pending = |bbox: [f32; 4], score: f32| {
            let mut b = auto_box("car", score, Triage::Accept);
            b.bbox = bbox;
            b
        };
        // Two proposals on one car, a far person; then an accepted box with a proposal on it.
        let mut person = auto_box("person", 0.9, Triage::Accept);
        person.bbox = [45.0, 2.0, 60.0, 40.0];
        let two = put_frame(
            &dir,
            "two",
            None,
            vec![pending(a, 0.6), pending(near, 0.8), person],
        );
        let mut explicit = pending(a, 0.5);
        explicit.state = BoxState::Accepted;
        let mut extra = pending(near, 0.9);
        extra.state = BoxState::Accepted;
        let decided = put_frame(
            &dir,
            "dec",
            None,
            vec![explicit.clone(), extra, pending(near, 0.95)],
        );
        let cl = vec!["person".to_string(), "car".to_string()];
        let filters = Filters {
            source: queue::QueueSource::All,
            ..Filters::default()
        };
        let o = bulk_accept(&dir, &filters, &cl, 10, "t", false, &|_| true);
        assert_eq!((o.done_count, o.failed_count), (2, 0), "{:?}", o.failed);
        assert_eq!(
            o.boxes, 2,
            "the car's best box and the person; no duplicate"
        );
        let f = labels_of(&dir, &two);
        assert_eq!(f.status, Status::Reviewed);
        assert!(f.bulk_accepted);
        let got: Vec<BoxState> = f.boxes.iter().map(|b| b.state).collect();
        assert_eq!(
            got,
            [BoxState::Rejected, BoxState::Accepted, BoxState::Accepted]
        );
        assert_eq!(f.boxes[0].reject_reason, Some(RejectReason::Duplicate));
        // Two explicit accepts stay: only a pending box is the rule's to reject.
        let f = labels_of(&dir, &decided);
        assert_eq!(
            f.boxes.iter().map(|b| b.state).collect::<Vec<_>>(),
            [BoxState::Accepted, BoxState::Accepted, BoxState::Rejected]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn outcome_lists_are_capped_and_counts_exact() {
        let mut o = BulkOutcome::default();
        for i in 0..(LISTED + 5) {
            o.done(&format!("f{i}"));
            o.failed(&format!("g{i}"), "x");
        }
        assert_eq!((o.done.len(), o.done_count), (LISTED, LISTED + 5));
        assert_eq!((o.failed.len(), o.failed_count), (LISTED, LISTED + 5));
        assert_eq!(o.failed[0], "g0: x");
    }

    #[test]
    fn bulk_reject_and_accept_on_disk_with_a_scope() {
        let dir = std::env::temp_dir().join(format!("bop-bulk-{}", uuid::Uuid::new_v4()));
        let a = put_frame(
            &dir,
            "a",
            Some("import:x"),
            vec![
                auto_box("person", 0.9, Triage::Accept),
                auto_box("car", 0.1, Triage::Review),
                auto_box("car", 0.2, Triage::Review),
            ],
        );
        let b = put_frame(
            &dir,
            "b",
            Some("import:y"),
            vec![
                auto_box("person", 0.9, Triage::Accept),
                auto_box("car", 0.1, Triage::Review),
            ],
        );
        let filters = Filters {
            source: queue::QueueSource::All,
            ..Filters::default()
        };
        let all = |_: &str| true;
        let dry = bulk_reject_below(&dir, &filters, 0.3, 10, true, &all);
        assert_eq!((dry.eligible, dry.boxes, dry.done_count), (2, 3, 0));
        assert_eq!(
            dry.done,
            [b.clone(), a.clone()],
            "fewer flagged boxes first"
        );
        assert_eq!(labels_of(&dir, &a).boxes[1].state, BoxState::Pending, "dry");
        // Only the frames of `keep`.
        let only_a = |r: &str| r == a;
        let o = bulk_reject_below(&dir, &filters, 0.3, 10, false, &only_a);
        assert_eq!((o.eligible, o.boxes, o.done_count), (1, 2, 1));
        let la = labels_of(&dir, &a);
        assert_eq!(la.status, Status::Unreviewed);
        assert_eq!(
            la.boxes.iter().map(|b| b.state).collect::<Vec<_>>(),
            [BoxState::Pending, BoxState::Rejected, BoxState::Rejected]
        );
        assert_eq!(labels_of(&dir, &b).boxes[1].state, BoxState::Pending);
        // `a` is now settled by the triage; `b` is not.
        let classes = vec!["person".to_string(), "car".into()];
        let dry = bulk_accept(&dir, &filters, &classes, 10, "t", true, &all);
        assert_eq!((dry.eligible, dry.done.clone()), (1, vec![a.clone()]));
        let o = bulk_accept(
            &dir,
            &filters,
            &classes,
            10,
            "2026-10-26T00:00:00Z",
            false,
            &all,
        );
        assert_eq!(
            (o.eligible, o.done_count, o.boxes, o.failed_count),
            (1, 1, 1, 0)
        );
        let la = labels_of(&dir, &a);
        assert_eq!(la.status, Status::Reviewed);
        assert!(la.bulk_accepted);
        assert_eq!(la.boxes[0].state, BoxState::Accepted);
        assert!(
            !accept_one(&dir, &a, &classes, "t").unwrap(),
            "not eligible any more"
        );
        // Limit: one frame of two.
        let o = bulk_reject_below(&dir, &filters, 0.3, 1, false, &all);
        assert_eq!(
            (o.eligible, o.done_count),
            (1, 1),
            "a is reviewed: left alone"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bulk_unwanted_rejects_on_disk_and_counts_reviewed_frames() {
        let dir = std::env::temp_dir().join(format!("bop-bulk-{}", uuid::Uuid::new_v4()));
        let u = put_frame(
            &dir,
            "u",
            None,
            vec![
                auto_box("person", 0.9, Triage::Accept),
                auto_box("car", 0.9, Triage::Review),
                model_box("clock", Origin::Prediction),
            ],
        );
        let keep = put_frame(
            &dir,
            "k",
            None,
            vec![auto_box("person", 0.9, Triage::Review)],
        );
        let r = put_frame(&dir, "r", None, vec![]);
        let frame = Frame::resolve(&dir, &r).unwrap();
        let mut accepted = auto_box("car", 0.9, Triage::Accept);
        accepted.state = BoxState::Accepted;
        let mut person = auto_box("person", 0.9, Triage::Accept);
        person.state = BoxState::Accepted;
        super::super::write(
            &frame.labels,
            &LabelFile {
                status: Status::Reviewed,
                boxes: vec![person, accepted.clone(), accepted],
                ..LabelFile::default()
            },
        )
        .unwrap();
        let cl = vec!["person".to_string(), "car".into()];
        let hidden = vec!["car".to_string()];
        let w = wanted(&cl, &hidden);
        let dry = bulk_reject_unwanted(&dir, &w, Scope::Unreviewed, usize::MAX, true);
        assert_eq!((dry.eligible, dry.boxes, dry.done_count), (1, 2, 0));
        assert_eq!(
            reviewed_with_unwanted(&dir, &w),
            ReviewedUnwanted {
                frames: 1,
                boxes: 2,
                classes: vec!["car".into()],
            }
        );
        let o = bulk_reject_unwanted(&dir, &w, Scope::Unreviewed, usize::MAX, false);
        assert_eq!((o.eligible, o.boxes, o.done_count), (1, 2, 1));
        assert_eq!(o.done, [u.as_str()]);
        let lu = labels_of(&dir, &u);
        assert_eq!(
            lu.boxes.iter().map(|b| b.state).collect::<Vec<_>>(),
            [BoxState::Pending, BoxState::Rejected, BoxState::Rejected]
        );
        assert!(
            auto_accept_eligible(&lu),
            "only the accept box is left pending"
        );
        assert_eq!(labels_of(&dir, &keep).boxes[0].state, BoxState::Pending);
        assert_eq!(labels_of(&dir, &r).boxes[1].state, BoxState::Accepted);
        // The reviewed frame, on request.
        let o = bulk_reject_unwanted(&dir, &w, Scope::Reviewed, usize::MAX, false);
        assert_eq!((o.eligible, o.boxes, o.done_count), (1, 2, 1));
        let lr = labels_of(&dir, &r);
        assert_eq!(lr.status, Status::Reviewed);
        assert_eq!(lr.training().count(), 1);
        assert_eq!(
            reviewed_with_unwanted(&dir, &w),
            ReviewedUnwanted::default()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn dismissed(label: &str) -> super::super::LabelBox {
        super::super::LabelBox {
            state: BoxState::Rejected,
            reason: Some(DISMISSED_REASON.into()),
            ..auto_box(label, 0.1, Triage::Review)
        }
    }

    #[test]
    fn the_bulk_edits_leave_dismissed_boxes_as_they_are() {
        let wanted_list = ["car".to_string()];
        let wanted = Wanted {
            classes: &wanted_list,
            hidden: &[],
        };
        let mut f = LabelFile {
            boxes: vec![dismissed("car"), dismissed("dog"), dismissed("car")],
            ..LabelFile::default()
        };
        let before = f.clone();
        assert_eq!(reject_low(&mut f, 0.5), 0);
        assert_eq!(reject_other_class(&mut f, &wanted), (0, 0));
        assert_eq!(stacks::reject_stacked(&mut f, &wanted), 0, "not live");
        assert_eq!(f, before);
        assert_eq!(f.hard_negatives().count(), 0);
    }

    #[test]
    fn no_bulk_save_passes_over_a_dismissed_box() {
        let dir = std::env::temp_dir().join(format!("bop-dismissed-{}", uuid::Uuid::new_v4()));
        let sure = auto_box("car", 0.9, Triage::Accept);
        let clean = put_frame(&dir, "a", None, vec![sure.clone()]);
        let held = put_frame(&dir, "b", None, vec![sure, dismissed("dog")]);
        let classes = ["car".to_string(), "dog".to_string()];
        let f = Filters {
            source: queue::QueueSource::All,
            ..Filters::default()
        };
        let o = bulk_accept(&dir, &f, &classes, 500, "t", false, &|_| true);
        assert_eq!((o.eligible, o.done_count), (1, 1));
        assert_eq!(labels_of(&dir, &clean).status, Status::Reviewed);
        assert_eq!(labels_of(&dir, &held).status, Status::Unreviewed);
        assert!(!accept_one(&dir, &held, &classes, "t").unwrap());
        // A repeat-only frame with one: not saved either.
        let mut l = labels_of(&dir, &held);
        l.repeat_spot = true;
        l.boxes[0].state = BoxState::Rejected;
        let frame = Frame::resolve(&dir, &held).unwrap();
        super::super::write(&frame.labels, &l).unwrap();
        let o = bulk_accept_repeats(&dir, &f, &classes, 500, "t", false, &|_| true);
        assert_eq!(o.eligible, 0);
        // Once a reviewer's decision clears the reason it is one.
        l.boxes[1].reason = None;
        super::super::write(&frame.labels, &l).unwrap();
        let o = bulk_accept_repeats(&dir, &f, &classes, 500, "t", false, &|_| true);
        assert_eq!(o.done_count, 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bulk_accept_rejects_hidden_and_unlisted_boxes_instead_of_accepting_them() {
        let dir = std::env::temp_dir().join(format!("bop-bulk-{}", uuid::Uuid::new_v4()));
        let classes = vec!["person".to_string(), "car".to_string(), "cat".to_string()];
        let hidden = vec!["cat".to_string()];
        // Far apart so that no box is a duplicate of another.
        let at = |label: &str, x: f32| {
            let mut b = auto_box(label, 0.9, Triage::Accept);
            b.bbox = [x, 2.0, x + 10.0, 40.0];
            b
        };
        let mixed = put_frame(
            &dir,
            "mixed",
            None,
            vec![at("person", 1.0), at("cat", 20.0), at("clock", 40.0)],
        );
        let only = put_frame(&dir, "only", None, vec![at("cat", 1.0)]);
        let filters = Filters {
            source: queue::QueueSource::All,
            wanted: Some((classes.clone(), hidden.clone())),
            ..Filters::default()
        };
        let o = bulk_accept(&dir, &filters, &classes, 10, "t", false, &|_| true);
        // Nothing wanted left to accept: the box is rejected, the frame waits for a reviewer.
        let f = labels_of(&dir, &only);
        assert_eq!(f.status, Status::Unreviewed);
        assert_eq!(f.boxes[0].state, BoxState::Rejected);
        assert_eq!(o.done_count, 1, "{:?}", o.failed);
        assert_eq!(o.boxes, 1, "the person only");
        assert_eq!(o.failed_count, 1, "{:?}", o.failed);
        let f = labels_of(&dir, &mixed);
        assert_eq!(f.status, Status::Reviewed);
        let got: Vec<(BoxState, Option<&str>)> = f
            .boxes
            .iter()
            .map(|b| (b.state, b.reason.as_deref()))
            .collect();
        assert_eq!(
            got,
            [
                (BoxState::Accepted, None),
                (BoxState::Rejected, Some(OTHER_CLASS_REASON)),
                (BoxState::Rejected, Some(OTHER_CLASS_REASON)),
            ]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
