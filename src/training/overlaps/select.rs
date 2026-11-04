//! Stage 1 of the overlap check: walk the label files ([`queue::labelled`]: reviewed frames need not
//! be in the review queue) and list the open pairs that need an answer and the crops they share. The
//! frame may change after this; [`super::write`] judges again under the write lock.

use super::super::autolabel::FrameFilter;
use crate::collect::labels::bulk::Wanted;
use crate::collect::labels::{Frame, Status, overlaps, queue};
use anyhow::Result;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The exact bits of a box: pairs share a crop only at one union, and a moved box misses its pair.
pub type BoxKey = [u32; 4];

pub fn box_key(u: &[f32; 4]) -> BoxKey {
    u.map(f32::to_bits)
}

/// The key of an open pair: its two boxes as selected.
pub fn pair_key(p: &overlaps::Pair) -> (BoxKey, BoxKey) {
    (box_key(&p.a.bbox), box_key(&p.b.bbox))
}

/// One crop to run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Crop {
    /// Its line in the script's output.
    pub id: usize,
    pub union: [f32; 4],
}

/// A frame with work.
#[derive(Debug, Clone, PartialEq)]
pub struct Selected {
    pub rel: String,
    pub jpg: PathBuf,
    /// One per union; empty when the witnesses alone answer its pairs.
    pub crops: Vec<Crop>,
    /// The pairs selected, each with the crop that answers it (None: RF-DETR cannot see its class).
    pub pairs: BTreeMap<(BoxKey, BoxKey), Option<usize>>,
}

/// The work of a run.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Selection {
    #[serde(skip)]
    pub frames: Vec<Selected>,
    /// Frames with an open pair to answer.
    pub frame_count: usize,
    /// Open pairs to answer.
    pub pairs: usize,
    /// Of those, pairs of a class RF-DETR cannot see: the witnesses answer.
    pub no_zoom: usize,
    /// Crops to run.
    pub crops: usize,
    /// Open pairs left out: they have an answer already (`again=1` to redo).
    pub answered: usize,
    /// Frames with open pairs a reviewer skipped or checked.
    pub passed_over: usize,
}

/// The work under `dir`: frames of `filter`, at most its `limit` with work, at most
/// [`overlaps::MAX_SUGGESTIONS`] pairs per frame (a swarm of boxes costs one bound of crops, and a
/// rerun takes the pairs that did not fit).
pub fn select(dir: &Path, filter: &FrameFilter, wanted: &Wanted<'_>) -> Result<Selection> {
    let filters = filter.filters()?;
    let again = filter.again();
    let imported = filter.import().is_some();
    let keep = |_: &str| true;
    let mut out = Selection::default();
    let mut next = 0;
    for (rel, l) in queue::labelled(dir, &filters, &keep) {
        if out.frame_count >= filter.limit {
            break;
        }
        let opens = overlaps::open_pairs(&l, wanted);
        if opens.is_empty() {
            continue;
        }
        if l.status == Status::Skipped || l.overlaps_checked {
            out.passed_over += 1;
            continue;
        }
        let Some(frame) = Frame::resolve(dir, &rel) else {
            continue;
        };
        if imported
            && !frame
                .record()
                .is_ok_and(|r| filter.accepts_source(r.source.as_deref()))
        {
            continue;
        }
        let mut todo: Vec<&overlaps::Open> = opens
            .iter()
            .filter(|o| again || !overlaps::answered(&l, &o.pair))
            .collect();
        out.answered += opens.len() - todo.len();
        todo.truncate(overlaps::MAX_SUGGESTIONS);
        if todo.is_empty() {
            continue;
        }
        let mut crops: BTreeMap<BoxKey, Crop> = BTreeMap::new();
        let mut pairs = BTreeMap::new();
        for o in &todo {
            let crop = if overlaps::no_zoom(&o.pair) {
                out.no_zoom += 1;
                None
            } else {
                let union = o.pair.union();
                let c = crops.entry(box_key(&union)).or_insert_with(|| {
                    let id = next;
                    next += 1;
                    Crop { id, union }
                });
                Some(c.id)
            };
            pairs.insert(pair_key(&o.pair), crop);
        }
        out.frame_count += 1;
        out.pairs += todo.len();
        out.crops += crops.len();
        out.frames.push(Selected {
            rel,
            jpg: frame.jpg,
            crops: crops.into_values().collect(),
            pairs,
        });
    }
    Ok(out)
}
