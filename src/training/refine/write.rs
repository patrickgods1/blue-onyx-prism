//! A fold's suggestions into the label files. Through [`labels::update`] and judged afresh under its
//! lock ([`judge::judge`] on the file as it is now): a reviewer may have saved the frame since the
//! selection, so their boxes and status decide, and a frame no longer reviewed gets nothing. Only
//! [`crate::collect::labels::LabelFile::refine`] changes ([`refine::apply`] of the kinds this check
//! makes), and a frame whose suggestions are the same as before is not written (its revision stays,
//! so an open Review page can still save).

use super::judge::{self, Context, FrameInput};
use super::report::{KindCounts, RecallTally};
use crate::collect::labels::export::ExportFrame;
use crate::collect::labels::refine::{self, Kind, Suggestion};
use crate::collect::labels::{self, Frame, LabelFile};
use crate::training::autolabel::relabel::Detections;
use anyhow::{Result, anyhow};
use std::path::Path;

/// The kinds a check writes (and replaces when not found again): `tighten` belongs to a later
/// stage and is left alone.
pub const KINDS: [Kind; 3] = [Kind::Add, Kind::Verify, Kind::Relabel];

/// What writing one fold did.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Written {
    /// Frames judged (reviewed when written).
    pub checked: usize,
    /// Frames whose label file changed.
    pub frames: usize,
    /// Suggestions written now, per kind.
    pub written: KindCounts,
    /// Suggestions found, per kind.
    pub found: KindCounts,
    /// Loose labels.
    pub loose: usize,
    /// The first frames written (`<scene>/<camera>/<date>/<stem>.jpg`).
    pub sample: Vec<String>,
}

fn bump(c: &mut KindCounts, kind: Kind) {
    match kind {
        Kind::Add => c.add += 1,
        Kind::Verify => c.verify += 1,
        Kind::Relabel => c.relabel += 1,
        Kind::Tighten | Kind::Unknown => {}
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
            },
            ctx,
        );
        let before = file.refine.clone();
        let applied = refine::apply(&mut file, j.suggestions.clone(), &KINDS);
        written = new_ones(&before, &file.refine);
        judged = Some(j);
        Ok(applied.changed().then_some(file))
    })?;
    if let Some(j) = judged {
        out.checked += 1;
        out.loose += j.loose;
        for s in &j.suggestions {
            bump(&mut out.found, s.kind);
        }
        out.written.add += written.add;
        out.written.verify += written.verify;
        out.written.relabel += written.relabel;
        tally.add(&f.rel.camera, &f.rel.scene, &j.boxes);
    }
    Ok(changed)
}

/// Write the suggestions of a fold's `frames` from `found` (the fold model's detections; a frame
/// without a result is passed over, never judged as "nothing found"), counting the recall into
/// `tally`, calling `progress(done, total)` now and then. A frame that fails is a warning: the
/// others go on.
pub fn write_fold(
    dir: &Path,
    frames: &[&ExportFrame],
    found: &Detections,
    ctx: &Context<'_>,
    tally: &mut RecallTally,
    warnings: &mut Vec<String>,
    progress: &mut dyn FnMut(usize, usize),
) -> Written {
    let mut out = Written::default();
    let mut missing = 0usize;
    for (i, f) in frames.iter().enumerate() {
        if i % 25 == 0 {
            progress(i, frames.len());
        }
        let rel = f.rel.jpg();
        let Some(dets) = found.frames.get(&rel) else {
            missing += 1;
            continue;
        };
        match write_frame(dir, f, dets, ctx, tally, &mut out) {
            Ok(true) => {
                out.frames += 1;
                if out.sample.len() < 20 {
                    out.sample.push(rel);
                }
            }
            Ok(false) => {}
            Err(e) => warnings.push(format!("{rel}: {e:#}")),
        }
    }
    if missing > 0 && !found.cancelled {
        warnings.push(format!(
            "{missing} frame(s) got no result from the fold model (unreadable images?); they \
             were not checked"
        ));
    }
    out
}
