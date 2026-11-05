//! Stage 4 of the overlap check: the zoom's answers become suggestions in the label files. Through
//! [`labels::update`] and judged afresh under its lock ([`overlaps::apply`]): a reviewer may be working
//! live, so a pair that moved or went is passed over, a frame they skipped or checked meanwhile is left
//! alone, and witnesses and guards are read from the file as it stands. The zoom only answers the
//! pair it ran for, keyed by both boxes. Only the suggestions change, and a frame that gained none
//! is not written (its revision stays, so an open page can still save).

use super::select::{Selected, pair_key};
use super::zoom::CropResult;
use crate::collect::labels::bulk::Wanted;
use crate::collect::labels::overlaps::{self, Applied, Zoom};
use crate::collect::labels::{self, Frame, LabelFile};
use anyhow::{Result, anyhow};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;

/// What the stage wrote.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Written {
    /// Frames that gained a suggestion.
    pub frames: usize,
    /// Drops suggested.
    pub drops: usize,
    /// Of those, in the weak band (the page warns).
    pub weak: usize,
    /// Of those, from the witnesses alone (lower confidence).
    pub low_confidence: usize,
    /// Pairs suggested as two objects.
    pub two_objects: usize,
    /// Pairs left to the reviewer with no suggestion.
    pub unclear: usize,
    /// Open pairs with no zoom answer: a crop not finished, or a box that moved under the job.
    pub no_result: usize,
    /// The first frames that gained a suggestion.
    pub sample: Vec<String>,
}

impl Written {
    fn add(&mut self, a: Applied) {
        self.drops += a.drops;
        self.weak += a.weak;
        self.low_confidence += a.low_confidence;
        self.two_objects += a.two_objects;
        self.unclear += a.unclear;
        self.no_result += a.no_result;
    }
}

/// The suggestions of one frame, `crops` being the finished crops of the run; whether it was written.
fn write_frame(
    dir: &Path,
    sel: &Selected,
    crops: &BTreeMap<usize, CropResult>,
    wanted: &Wanted<'_>,
    again: bool,
    out: &mut Written,
) -> Result<bool> {
    let frame =
        Frame::resolve(dir, &sel.rel).ok_or_else(|| anyhow!("the frame {} is gone", sel.rel))?;
    let zoom_of = |p: &overlaps::Pair| match *sel.pairs.get(&pair_key(p))? {
        None => Some(Zoom::Unavailable),
        Some(id) => crops.get(&id).map(|r| match r {
            CropResult::Boxes(b) => Zoom::Looked(b),
            CropResult::Unreadable => Zoom::Unavailable,
        }),
    };
    let mut applied = Applied::default();
    let written = labels::update(&frame.labels, |cur| {
        let Some(mut file): Option<LabelFile> = cur else {
            return Ok(None);
        };
        applied = overlaps::apply(&mut file, wanted, again, &zoom_of);
        Ok(applied.changed().then_some(file))
    })?;
    out.add(applied);
    Ok(written)
}

/// Write the suggestions of `frames` from the finished `crops`, calling `progress(done, total)` now
/// and then. A frame that fails is a warning: the others go on, and a rerun does it again.
pub fn write_frames(
    dir: &Path,
    frames: &[Selected],
    crops: &BTreeMap<usize, CropResult>,
    wanted: &Wanted<'_>,
    again: bool,
    warnings: &mut Vec<String>,
    progress: &mut dyn FnMut(usize, usize),
) -> Written {
    let mut out = Written::default();
    for (i, sel) in frames.iter().enumerate() {
        if i % 50 == 0 {
            progress(i, frames.len());
        }
        match write_frame(dir, sel, crops, wanted, again, &mut out) {
            Ok(true) => {
                out.frames += 1;
                if out.sample.len() < 20 {
                    out.sample.push(sel.rel.clone());
                }
            }
            Ok(false) => {}
            Err(e) => warnings.push(format!("{}: {e:#}", sel.rel)),
        }
    }
    out
}
