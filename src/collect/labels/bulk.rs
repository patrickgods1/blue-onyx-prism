//! Bulk edits of the label files a Review queue selects: [`bulk_accept`] saves the frames the
//! auto-labelers' triage alone settles as reviewed (`POST /v1/review/accept-auto`, the
//! auto-label chain's accept step), [`bulk_reject_below`] rejects the pending auto boxes scoring
//! below a minimum on unreviewed frames (`reason: low_score`; the reject-low route and the
//! chain's reject step), [`bulk_reject_other_class`] rejects the pending model boxes of classes
//! not in the class list on unreviewed frames (`reason: other_class`; an alias such as
//! `pottedplant` is renamed to the list's spelling instead; `POST
//! /v1/review/reject-other-class`). Each frame is re-checked under the label write lock
//! ([`super::update`]); counts are exact, frame lists capped at [`LISTED`].

use super::queue::{self, AutoFilter, Filters, StatusFilter};
use super::{BoxState, Frame, LabelFile, Source, Status};
use serde::Serialize;
use std::path::Path;

/// Most frame names listed in a [`BulkOutcome`].
pub const LISTED: usize = 20;

/// The reason a box rejected by [`reject_low`] carries.
pub const LOW_SCORE_REASON: &str = "low_score";

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
        && b.score.is_some_and(|s| s < min_score)
}

/// Reject the pending auto-labeler boxes of `f` scoring below `min_score` (`reason:
/// low_score`, kept as hard-negative evidence); only on an unreviewed frame. The frame's status
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

/// What [`reject_other_class`] does to `b`: None = nothing (not a pending model box, or its
/// label is a class of `classes`); `Some(Some(class))` = renamed to `class` (the list's spelling
/// of an alias, [`super::map_class`]); `Some(None)` = rejected.
fn other_class_action<'a>(b: &super::LabelBox, classes: &'a [String]) -> Option<Option<&'a str>> {
    if b.source != Source::Model || b.state != BoxState::Pending || classes.contains(&b.label) {
        return None;
    }
    Some(super::map_class(&b.label, classes))
}

/// On an unreviewed frame, the pending model boxes (the serving model's, from the sidecar) whose
/// label is not a class of `classes`: renamed to the class when it is one under another
/// spelling ([`super::map_class`]: `pottedplant` -> `potted plant`, the old label kept in
/// `model_label` unless only the case differs), else rejected with `reason: other_class` (kept
/// as evidence; a save would otherwise accept a prediction of a class outside the list and
/// fail). Auto and human boxes, decided boxes and reviewed or skipped frames are untouched; the
/// frame's status is unchanged. Returns (rejected, renamed).
pub fn reject_other_class(f: &mut LabelFile, classes: &[String]) -> (usize, usize) {
    if f.status != Status::Unreviewed {
        return (0, 0);
    }
    let (mut rejected, mut renamed) = (0, 0);
    for b in &mut f.boxes {
        match other_class_action(b, classes) {
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

/// What [`reject_other_class`] would do: (rejected, renamed).
fn count_other_class(f: &LabelFile, classes: &[String]) -> (usize, usize) {
    if f.status != Status::Unreviewed {
        return (0, 0);
    }
    f.boxes
        .iter()
        .filter_map(|b| other_class_action(b, classes))
        .fold((0, 0), |(r, n), a| match a {
            Some(_) => (r, n + 1),
            None => (r + 1, n),
        })
}

/// [`reject_other_class`] on the unreviewed frames of `dir` that `filters` (status forced to
/// unreviewed) and `keep` select, at most `limit` frames in queue order; `boxes` counts the
/// boxes rejected or renamed, `relabelled` the renamed ones. `dry_run` only counts. A frame
/// that changed meanwhile (reviewed) is passed over.
pub fn bulk_reject_other_class(
    dir: &Path,
    filters: &Filters,
    classes: &[String],
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
        &|l| count_other_class(l, classes),
        &|l| reject_other_class(l, classes),
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
    let mut out = BulkOutcome::default();
    let mut eligible = Vec::new();
    for item in built.items.iter().filter(|i| keep(&i.frame)) {
        let (n, renamed) = Frame::resolve(dir, &item.frame)
            .and_then(|fr| fr.labels().ok().flatten())
            .map_or((0, 0), |l| count(&l));
        if n + renamed > 0 {
            eligible.push((item.frame.as_str(), n, renamed));
        }
    }
    out.eligible = eligible.len();
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
    for (rel, _, _) in eligible.into_iter().take(limit) {
        let Some(frame) = Frame::resolve(dir, rel) else {
            out.failed(rel, "the frame is gone");
            continue;
        };
        let (mut n, mut renamed) = (0, 0);
        let r = super::update(&frame.labels, |cur| {
            let Some(mut l) = cur else { return Ok(None) };
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
/// changed since), saved as reviewed ([`super::normalize`]) with `bulk_accepted`. `Ok(None)`
/// when it is not eligible any more, else the boxes it accepted.
fn accept_frame(
    dir: &Path,
    rel: &str,
    classes: &[String],
    now: &str,
) -> anyhow::Result<Option<usize>> {
    let frame = Frame::resolve(dir, rel).ok_or_else(|| anyhow::anyhow!("the frame is gone"))?;
    let rec = frame.record()?;
    let mut boxes = 0;
    let wrote = super::update(&frame.labels, |cur| {
        let Some(mut f) = cur.filter(super::auto_accept_eligible) else {
            return Ok(None);
        };
        boxes = f
            .boxes
            .iter()
            .filter(|b| {
                b.state == BoxState::Pending && super::resolve_pending(b) == BoxState::Accepted
            })
            .count();
        f.status = Status::Reviewed;
        let mut n = super::normalize(f, classes, rec.width, rec.height, now)?;
        n.bulk_accepted = true;
        Ok(Some(n))
    })?;
    Ok(wrote.then_some(boxes))
}

/// Accept one eligible frame (see [`bulk_accept`]); `Ok(false)` when it is not eligible (any
/// more).
pub fn accept_one(dir: &Path, rel: &str, classes: &[String], now: &str) -> anyhow::Result<bool> {
    accept_frame(dir, rel, classes, now).map(|n| n.is_some())
}

/// Save the unreviewed frames of `dir` whose auto-label triage alone settles them
/// ([`super::auto_accept_eligible`]; `filters` with status and `auto` forced to unreviewed /
/// accept) and that `keep` selects as reviewed with `bulk_accepted`, at most `limit` in queue
/// order, the labels checked against `classes` at time `now`. `dry_run` only counts.
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
    for rel in eligible.into_iter().take(limit) {
        match accept_frame(dir, rel, classes, now) {
            Ok(Some(n)) => {
                out.boxes += n;
                out.done(rel);
            }
            Ok(None) => out.failed(
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
        let day = dir.join("ir").join("cam").join("2026-10-21");
        std::fs::create_dir_all(&day).unwrap();
        image::RgbImage::from_pixel(64, 48, image::Rgb([90, 90, 90]))
            .save(day.join(format!("{stem}.jpg")))
            .unwrap();
        let mut r = crate::collect::record::empty_record();
        r.id = stem.into();
        r.scene = "ir".into();
        r.camera = "cam".into();
        r.timestamp = "2026-10-21T01:02:03.004Z".into();
        r.width = 64;
        r.height = 48;
        r.source = source.map(str::to_string);
        std::fs::write(
            day.join(format!("{stem}.json")),
            serde_json::to_vec(&r).unwrap(),
        )
        .unwrap();
        let rel = format!("ir/cam/2026-10-21/{stem}.jpg");
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
    use super::test_support::{auto_box, labels_of, put_frame};
    use super::*;

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

    #[test]
    fn reject_other_class_touches_only_pending_model_boxes_off_the_list() {
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
        assert_eq!(count_other_class(&f, &classes()), (1, 2));
        assert_eq!(reject_other_class(&mut f, &classes()), (1, 2));
        assert_eq!(f.boxes[1].state, BoxState::Rejected);
        assert_eq!(f.boxes[1].reason.as_deref(), Some(OTHER_CLASS_REASON));
        assert_eq!(f.boxes[1].label, "parking meter");
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
        for i in [0, 4, 5, 6] {
            assert_eq!(f.boxes[i], before.boxes[i], "box {i}");
        }
        assert_eq!(f.status, Status::Unreviewed);
        assert_eq!(reject_other_class(&mut f, &classes()), (0, 0), "done");
        // Reviewed and skipped frames are untouched.
        for s in [Status::Reviewed, Status::Skipped] {
            let mut r = before.clone();
            r.status = s;
            assert_eq!(count_other_class(&r, &classes()), (0, 0));
            assert_eq!(reject_other_class(&mut r, &classes()), (0, 0));
            assert_eq!(r.boxes, before.boxes);
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
        assert_eq!(reject_other_class(&mut f, &classes()), (2, 0));
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
        assert_eq!(dry.done, [a.clone(), b.clone()]);
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
}
