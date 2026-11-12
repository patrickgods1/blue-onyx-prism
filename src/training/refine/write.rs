//! A fold's suggestions into the label files. Through [`labels::update`] and judged afresh under its
//! lock ([`judge::judge`] on the file as it is now): a reviewer may have saved the frame since the
//! selection, so their boxes and status decide, and a frame no longer reviewed gets nothing. Only
//! [`crate::collect::labels::LabelFile::refine`] changes ([`refine::apply`] of the kinds this check
//! makes), and a frame whose suggestions are the same as before is not written (its revision stays,
//! so an open Review page can still save).

use super::judge::{self, Context, FrameInput};
use super::report::{KindCounts, RecallTally};
use super::tighten::{FrameCandidates, Looked, SavedDet};
use crate::collect::labels::export::ExportFrame;
use crate::collect::labels::refine::{self, Kind, Suggestion};
use crate::collect::labels::{self, Frame, LabelFile, iou};
use crate::training::autolabel::relabel::Detections;
use crate::training::autolabel::triage::Det;
use anyhow::{Result, anyhow};
use std::path::Path;

/// The kinds a fold writes (and replaces when not found again). `tighten` is the zoom stage's
/// ([`TIGHTEN_KINDS`], [`write_tighten`]) and a fold leaves it alone.
pub const KINDS: [Kind; 3] = [Kind::Add, Kind::Verify, Kind::Relabel];
/// The kinds the zoom stage writes, on the frames it looked at only.
pub const TIGHTEN_KINDS: [Kind; 1] = [Kind::Tighten];

/// What writing one fold did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Written {
    /// Frames judged (reviewed when written).
    pub checked: usize,
    /// Frames whose label file changed.
    pub frames: usize,
    /// Suggestions written now, per kind.
    pub written: KindCounts,
    /// Suggestions found and kept on the files, per kind: the judged ones [`refine::apply`] did
    /// not drop (a weaker copy of one finding, the same finding as an answered one, past the
    /// file's limit), whether new or the same as before.
    pub found: KindCounts,
    /// Of `found.add`, those at a static spot ([`Suggestion::spot`]).
    pub add_at_spots: usize,
    /// Loose labels.
    pub loose: usize,
    /// The first frames written (`<scene>/<camera>/<date>/<stem>.jpg`).
    pub sample: Vec<String>,
    /// Frames with a result from the fold model that were judged and written, or tried to be.
    pub attempted: usize,
    /// Of those, frames whose write failed.
    pub failed: usize,
    /// The first [`MAX_FAILURES`] of those failures, `<rel>: <why>`.
    pub failures: Vec<String>,
    /// Frames without a result from the fold model (it could not read them): not judged.
    pub missing: usize,
    /// Every frame whose label file changed (`<scene>/<camera>/<date>/<stem>.jpg`): a frame the
    /// zoom stage writes again counts once in the report.
    pub changed: Vec<String>,
    /// The judged frames with loose labels worth a closer look ([`judge::Judged::candidates`]).
    pub candidates: Vec<FrameCandidates>,
}

/// Most failure messages one fold keeps ([`Written::failures`]); the job keeps as many over the
/// whole check and counts the rest.
pub const MAX_FAILURES: usize = 20;

impl Written {
    /// Every frame with a result failed to write: nothing of the fold was saved.
    pub fn all_failed(&self) -> bool {
        self.attempted > 0 && self.failed == self.attempted
    }
}

fn bump(c: &mut KindCounts, kind: Kind) {
    match kind {
        Kind::Add => c.add += 1,
        Kind::Verify => c.verify += 1,
        Kind::Relabel => c.relabel += 1,
        Kind::Tighten => c.tighten += 1,
        Kind::Unknown => {}
    }
}

/// `s` without its run id, to compare a suggestion across runs.
fn without_run(s: &Suggestion) -> Suggestion {
    Suggestion {
        run: String::new(),
        ..s.clone()
    }
}

/// The suggestions of `after` that `before` did not hold (ignoring `run`), per kind.
fn new_ones(before: &[Suggestion], after: &[Suggestion]) -> KindCounts {
    let mut old: Vec<Suggestion> = before.iter().map(without_run).collect();
    let mut c = KindCounts::default();
    for s in after {
        let s = without_run(s);
        match old.iter().position(|o| *o == s) {
            Some(p) => {
                old.swap_remove(p);
            }
            None => bump(&mut c, s.kind),
        }
    }
    c
}

/// Judge and write one frame; whether its file changed.
fn write_frame(
    dir: &Path,
    f: &ExportFrame,
    dets: &[crate::training::autolabel::triage::Det],
    ctx: &Context<'_>,
    tally: &mut RecallTally,
    out: &mut Written,
) -> Result<bool> {
    let rel = f.rel.jpg();
    let frame = Frame::resolve(dir, &rel).ok_or_else(|| {
        anyhow!("the frame is gone (deleted or moved during the check); nothing was written for it")
    })?;
    let mut judged = None;
    let mut written = KindCounts::default();
    // The unanswered suggestions of this check's kinds on the file after `apply`: what it kept of
    // the judged ones (an unchanged file holds the same list, ignoring `run`).
    let mut kept: Vec<(Kind, bool)> = Vec::new();
    let changed = labels::update(&frame.labels, |cur| {
        let Some(mut file): Option<LabelFile> = cur else {
            return Ok(None);
        };
        // Saved back to unreviewed or skipped since the selection: not checked.
        if file.status != labels::Status::Reviewed {
            return Ok(None);
        }
        let j = judge::judge(
            &FrameInput {
                file: &file,
                width: f.width,
                height: f.height,
                dets,
                camera: &f.rel.camera,
                day: &f.rel.date,
            },
            ctx,
        );
        let before = file.refine.clone();
        let applied = refine::apply(&mut file, j.suggestions.clone(), &KINDS);
        written = new_ones(&before, &file.refine);
        kept = file
            .refine
            .iter()
            .filter(|s| !s.answered() && KINDS.contains(&s.kind))
            .map(|s| (s.kind, s.spot.is_some()))
            .collect();
        judged = Some(j);
        Ok(applied.changed().then_some(file))
    })?;
    if let Some(j) = judged {
        out.checked += 1;
        out.loose += j.loose;
        for (kind, at_spot) in kept {
            bump(&mut out.found, kind);
            out.add_at_spots += usize::from(kind == Kind::Add && at_spot);
        }
        out.written.add += written.add;
        out.written.verify += written.verify;
        out.written.relabel += written.relabel;
        tally.add(&f.rel.camera, &f.rel.scene, &j.boxes);
        if !j.candidates.is_empty() {
            out.candidates.push(FrameCandidates {
                rel: f.rel.jpg(),
                image: f.image.clone(),
                width: f.width,
                height: f.height,
                camera: f.rel.camera.clone(),
                day: f.rel.date.clone(),
                dets: dets.iter().map(SavedDet::from).collect(),
                loose: j.candidates,
            });
        }
    }
    Ok(changed)
}

/// Write the suggestions of a fold's `frames` from `found` (the fold model's detections; a frame
/// without a result is passed over and counted in [`Written::missing`], never judged as "nothing
/// found"), counting the recall into `tally`, calling `progress(done, total)` now and then. A frame
/// that fails is counted ([`Written::failed`], the first [`MAX_FAILURES`] messages kept): the
/// others go on.
pub fn write_fold(
    dir: &Path,
    frames: &[&ExportFrame],
    found: &Detections,
    ctx: &Context<'_>,
    tally: &mut RecallTally,
    progress: &mut dyn FnMut(usize, usize),
) -> Written {
    let mut out = Written::default();
    for (i, f) in frames.iter().enumerate() {
        if i % 25 == 0 {
            progress(i, frames.len());
        }
        let rel = f.rel.jpg();
        let Some(dets) = found.frames.get(&rel) else {
            out.missing += 1;
            continue;
        };
        out.attempted += 1;
        match write_frame(dir, f, dets, ctx, tally, &mut out) {
            Ok(true) => {
                out.frames += 1;
                if out.sample.len() < 20 {
                    out.sample.push(rel.clone());
                }
                out.changed.push(rel);
            }
            Ok(false) => {}
            Err(e) => {
                out.failed += 1;
                if out.failures.len() < MAX_FAILURES {
                    out.failures.push(format!("{rel}: {e:#}"));
                }
            }
        }
    }
    out
}

/// What [`write_tighten`] did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tightened {
    /// Frames whose label file changed.
    pub changed: Vec<String>,
    /// `tighten` suggestions written now (new since the file's last).
    pub written: usize,
    /// This look's proposals that [`refine::apply`] kept (new, or the same as an open one already
    /// there); never an earlier check's open `tighten` the write left alone.
    pub found: usize,
    /// Of those, per suggestion: IoU(`L`, `Z`) and `Z`'s area over `L`'s.
    pub shapes: Vec<(f32, f32)>,
    /// Per answered candidate that got no proposal: the frame, the candidate and why
    /// ([`judge::NoTighten`]).
    pub reasons: Vec<(String, judge::Loose, judge::NoTighten)>,
    /// Frames whose write failed, the first [`MAX_FAILURES`] reasons.
    pub failed: usize,
    pub failures: Vec<String>,
}

fn area(b: &[f32; 4]) -> f32 {
    (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
}

/// `a` and `b` name the same label: exact box, label alike.
fn same_target(a: Option<[f32; 4]>, label: Option<&str>, c: &judge::Loose) -> bool {
    a.map(|t| t.map(f32::to_bits)) == Some(c.target.map(f32::to_bits))
        && label.is_none_or(|l| l.trim().eq_ignore_ascii_case(c.target_label.trim()))
}

/// Write the `tighten` suggestions the closer look supports on the frames it answered (`looked`),
/// each judged afresh under the label write lock ([`labels::update`]): a frame no longer reviewed
/// gets nothing; its candidates are found again from the fold model's detections on the file as
/// it is now ([`judge::judge`], `ctx`), and each one the look answered is decided by
/// [`judge::tighten_verdict`]. Only `tighten` changes ([`refine::apply`] of [`TIGHTEN_KINDS`]),
/// and only on the labels the look answered: the open `tighten` of such a label is replaced by
/// what the answer supports (or goes, when it supports nothing); an open one on any other label
/// (no candidate now, its crop unanswered, left out of the look) is kept as it is, answered ones
/// stay, and an unchanged result leaves the file untouched. `progress(done, total)` now and then.
pub fn write_tighten(
    dir: &Path,
    looked: &[Looked<'_>],
    ctx: &Context<'_>,
    progress: &mut dyn FnMut(usize, usize),
) -> Tightened {
    let mut out = Tightened::default();
    for (i, l) in looked.iter().enumerate() {
        if i % 25 == 0 {
            progress(i, looked.len());
        }
        let f = l.frame;
        let dets: Vec<Det> = f.dets.iter().map(Det::from).collect();
        let mut kept: Vec<(f32, f32)> = Vec::new();
        let mut reasons: Vec<(judge::Loose, judge::NoTighten)> = Vec::new();
        let mut written = 0;
        let r = Frame::resolve(dir, &f.rel)
            .ok_or_else(|| {
                anyhow!(
                    "the frame is gone (deleted or moved during the check); nothing was written"
                )
            })
            .and_then(|frame| {
                labels::update(&frame.labels, |cur| {
                    let Some(mut file): Option<LabelFile> = cur else {
                        return Ok(None);
                    };
                    if file.status != labels::Status::Reviewed {
                        return Ok(None);
                    }
                    let input = FrameInput {
                        file: &file,
                        width: f.width,
                        height: f.height,
                        dets: &dets,
                        camera: &f.camera,
                        day: &f.day,
                    };
                    // This look's proposals, and the labels it answered (whose open `tighten`
                    // the proposals replace).
                    let mut proposed: Vec<Suggestion> = Vec::new();
                    let mut answered: Vec<judge::Loose> = Vec::new();
                    reasons.clear();
                    for c in judge::judge(&input, ctx).candidates {
                        if let Some((_, look)) = l.answers.iter().find(|(a, _)| a.same_pair(&c)) {
                            match judge::tighten_verdict(&input, &c, look, ctx) {
                                Ok(s) => proposed.push(s),
                                Err(why) => reasons.push((c.clone(), why)),
                            }
                            answered.push(c);
                        }
                    }
                    // Every other label keeps its open `tighten` as it is.
                    let mut found = proposed.clone();
                    found.extend(
                        file.refine
                            .iter()
                            .filter(|s| {
                                s.kind == Kind::Tighten
                                    && !s.answered()
                                    && !answered.iter().any(|c| {
                                        same_target(s.target, s.target_label.as_deref(), c)
                                    })
                            })
                            .cloned(),
                    );
                    let before = file.refine.clone();
                    let applied = refine::apply(&mut file, found, &TIGHTEN_KINDS);
                    written = new_ones(&before, &file.refine).tighten;
                    // What this look found: its proposals that `apply` kept (an identical open
                    // one already there counts; the earlier ones carried over do not).
                    let mine: Vec<Suggestion> = proposed.iter().map(without_run).collect();
                    kept = file
                        .refine
                        .iter()
                        .filter(|s| mine.contains(&without_run(s)))
                        .filter_map(|s| {
                            let (t, z) = (s.target?, s.bbox?);
                            let a = area(&t);
                            Some((iou(&t, &z), if a > 0.0 { area(&z) / a } else { 0.0 }))
                        })
                        .collect();
                    Ok(applied.changed().then_some(file))
                })
            });
        match r {
            Ok(changed) => {
                out.written += written;
                out.found += kept.len();
                out.shapes.extend(kept);
                out.reasons
                    .extend(reasons.into_iter().map(|(c, why)| (f.rel.clone(), c, why)));
                if changed {
                    out.changed.push(f.rel.clone());
                }
            }
            Err(e) => {
                out.failed += 1;
                if out.failures.len() < MAX_FAILURES {
                    out.failures.push(format!("{}: {e:#}", f.rel));
                }
            }
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::collect::labels::bulk::Wanted;
    use crate::collect::labels::bulk::test_support::put_frame_at;
    use crate::collect::labels::export::{self, ClassFilter, SplitConfig};
    use crate::collect::labels::statics;
    use crate::collect::labels::{BoxState, LabelBox, Source, Status};
    use crate::training::autolabel::triage::Det;
    use std::collections::BTreeMap;

    #[test]
    fn found_counts_what_apply_kept_not_the_copies_it_dropped() {
        let dir = std::env::temp_dir().join(format!("bo_refine_write_{}", uuid::Uuid::new_v4()));
        // A parked car at one place of camera `yard` on 12 other frames of 3 other days.
        let car_at = [100.0f32, 100.0, 300.0, 250.0];
        let nb = statics::normalised(&car_at, 640, 480).unwrap();
        let table = statics::Table::build((0..12).map(|i| statics::Obs {
            camera: "yard".into(),
            frame: format!("other/{i}"),
            day: format!("2026-10-0{}", i % 3 + 1),
            scene: "day".into(),
            label: "car".into(),
            bbox: nb,
        }));
        assert_eq!(table.len(), 1);
        // A reviewed frame without its box, and a person elsewhere.
        let person = LabelBox {
            label: "person".into(),
            bbox: [400.0, 100.0, 450.0, 300.0],
            source: Source::Auto,
            score: Some(0.9),
            state: BoxState::Accepted,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: Default::default(),
        };
        let rel = put_frame_at(&dir, "yard", "f1", (640, 480), None, vec![person]);
        let path = Frame::resolve(&dir, &rel).unwrap().labels;
        labels::update(&path, |c| {
            let mut f = c.unwrap();
            f.status = Status::Reviewed;
            Ok(Some(f))
        })
        .unwrap();
        let classes = vec!["car".to_string(), "truck".to_string(), "person".to_string()];
        let wanted = Wanted {
            classes: &classes,
            hidden: &[],
        };
        let sel = export::reviewed_frames_filtered(
            &dir,
            &classes,
            &SplitConfig::default(),
            &ClassFilter::default(),
        );
        assert_eq!(sel.frames.len(), 1, "{:?}", sel.warnings);
        // Two weak detections of the one parked car: both become the spot's car at the place, one
        // finding; apply keeps the stronger.
        let det = |label: &str, d: f32, score: f32| Det {
            label: label.into(),
            score,
            bbox: [car_at[0] + d, car_at[1], car_at[2] + d, car_at[3]],
        };
        let mut found = Detections {
            frames: BTreeMap::new(),
            device: String::new(),
            cancelled: false,
            load_s: None,
        };
        found.frames.insert(
            rel.clone(),
            vec![det("car", 0.0, 0.45), det("truck", 4.0, 0.4)],
        );
        // Persons too rare for a verify: the frame's one finding is the car's.
        let counts: BTreeMap<String, usize> = classes
            .iter()
            .map(|c| (c.clone(), if c == "person" { 0 } else { 500 }))
            .collect();
        let ctx = Context {
            model: None,
            wanted: &wanted,
            imgsz: 640,
            class_boxes: &counts,
            run: "r1",
            spots: Some(&table),
            floor: None,
        };
        let frames: Vec<&ExportFrame> = sel.frames.iter().collect();
        let mut tally = RecallTally::default();
        let w = write_fold(&dir, &frames, &found, &ctx, &mut tally, &mut |_, _| {});
        let file = labels::read(&path).unwrap().unwrap();
        assert_eq!(file.refine.len(), 1, "{:?}", file.refine);
        assert_eq!(
            (w.found.add, w.add_at_spots, w.written.add),
            (1, 1, 1),
            "{w:?}"
        );
        // A rerun finds the same: nothing written, still one found.
        let again = write_fold(&dir, &frames, &found, &ctx, &mut tally, &mut |_, _| {});
        assert_eq!(
            (
                again.found.add,
                again.add_at_spots,
                again.written.add,
                again.frames
            ),
            (1, 1, 0, 0)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn car(bbox: [f32; 4]) -> LabelBox {
        LabelBox {
            label: "car".into(),
            bbox,
            source: Source::Auto,
            score: Some(0.9),
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

    #[test]
    fn the_look_replaces_only_the_tighten_of_labels_it_answered_and_counts_its_own() {
        use super::super::judge::{Look, Loose};
        use crate::collect::labels::overlaps::Det as ZoomDet;
        let dir = std::env::temp_dir().join(format!("bo_refine_tight_{}", uuid::Uuid::new_v4()));
        // Two loose car labels on one reviewed frame. The fold model finds A's car at 0.45 (no
        // candidate) and B's at 0.9 (a candidate the look answers).
        let (la, da) = (
            [100.0f32, 100.0, 300.0, 180.0],
            [100.0f32, 100.0, 200.0, 180.0],
        );
        let (lb, db) = (
            [400.0f32, 100.0, 600.0, 180.0],
            [400.0f32, 100.0, 500.0, 180.0],
        );
        let zb = [401.0f32, 100.0, 501.0, 181.0];
        let rel = put_frame_at(&dir, "yard", "f1", (640, 480), None, vec![car(la), car(lb)]);
        let path = Frame::resolve(&dir, &rel).unwrap().labels;
        // An open better box on A from an earlier check (two looks agreed then).
        let earlier = |target: [f32; 4], bbox: [f32; 4]| Suggestion {
            kind: Kind::Tighten,
            target: Some(target),
            target_label: Some("car".into()),
            bbox: Some(bbox),
            label: None,
            score: Some(0.7),
            why: "an earlier look".into(),
            spot: None,
            answer: None,
            run: "old".into(),
        };
        let on_a = earlier(la, [102.0, 100.0, 202.0, 180.0]);
        labels::update(&path, |c| {
            let mut f = c.unwrap();
            f.status = Status::Reviewed;
            f.refine = vec![on_a.clone()];
            Ok(Some(f))
        })
        .unwrap();
        let classes = vec!["car".to_string(), "person".to_string()];
        let wanted = Wanted {
            classes: &classes,
            hidden: &[],
        };
        let counts: BTreeMap<String, usize> = classes.iter().map(|c| (c.clone(), 500)).collect();
        let ctx = Context {
            model: None,
            wanted: &wanted,
            imgsz: 640,
            class_boxes: &counts,
            run: "r2",
            spots: None,
            floor: None,
        };
        let det = |bbox: [f32; 4], score: f32| SavedDet {
            label: "car".into(),
            score,
            bbox,
        };
        let frame = FrameCandidates {
            rel: rel.clone(),
            image: dir.join("unused.jpg"),
            width: 640,
            height: 480,
            camera: "yard".into(),
            day: "2026-10-01".into(),
            dets: vec![det(da, 0.45), det(db, 0.9)],
            loose: vec![],
        };
        let cand_b = Loose {
            target: lb,
            target_label: "car".into(),
            class: "car".into(),
            det: db,
            det_score: 0.9,
        };
        let look = |dets: Vec<ZoomDet>| {
            vec![Looked {
                frame: &frame,
                answers: vec![(cand_b.clone(), Look { dets, place: None })],
            }]
        };
        let z = ZoomDet {
            label: "car".into(),
            bbox: zb.map(f64::from),
            score: 0.88,
        };
        let w = write_tighten(&dir, &look(vec![z.clone()]), &ctx, &mut |_, _| {});
        let file = labels::read(&path).unwrap().unwrap();
        let tight: Vec<&Suggestion> = file
            .refine
            .iter()
            .filter(|s| s.kind == Kind::Tighten)
            .collect();
        assert_eq!(tight.len(), 2, "{:?}", file.refine);
        assert!(
            file.refine.contains(&on_a),
            "A was not answered: kept as it was"
        );
        let mine = tight.iter().find(|s| s.target == Some(lb)).unwrap();
        assert_eq!((mine.bbox, mine.run.as_str()), (Some(zb), "r2"));
        // Found: this look's one proposal, not the earlier one on A.
        assert_eq!((w.found, w.written, w.shapes.len()), (1, 1, 1), "{w:?}");
        assert!(w.reasons.is_empty());
        // Again: the same proposal is found again (written 0), A's still kept.
        let again = write_tighten(&dir, &look(vec![z]), &ctx, &mut |_, _| {});
        assert_eq!((again.found, again.written, again.changed.len()), (1, 0, 0));
        // An answer that supports nothing now replaces B's open one with nothing, says why, and
        // still leaves A's alone.
        let none = write_tighten(&dir, &look(vec![]), &ctx, &mut |_, _| {});
        let file = labels::read(&path).unwrap().unwrap();
        assert_eq!(file.refine, vec![on_a.clone()]);
        assert_eq!((none.found, none.written), (0, 0));
        assert_eq!(
            none.reasons,
            vec![(rel.clone(), cand_b.clone(), judge::NoTighten::NoAgree)]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
