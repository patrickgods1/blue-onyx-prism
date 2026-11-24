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
    /// Open pairs left out: their frame's suggestions are full ([`overlaps::MAX_SUGGESTIONS`]), so
    /// an answer could not be kept. They wait for room (a reviewer settles a pair) or `again=1`.
    pub no_room: usize,
    /// Frames with open pairs a reviewer skipped or checked.
    pub passed_over: usize,
}

/// The work under `dir`: frames of `filter`, at most its `limit` with work, at most
/// [`overlaps::MAX_SUGGESTIONS`] pairs per frame (a swarm of boxes costs one bound of crops). Without
/// `again`, a frame gets only as many new pairs as its suggestion list has room for: every pair
/// judged is then kept, and a rerun takes the pairs that did not fit once room frees (a pair a
/// reviewer settles stops being open). Judging more would be thrown away by the bound and, never
/// recorded, cropped again on every run (QA-070).
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
        // Without `again`, the pairs left out are those the list keeps an answer for.
        let kept = opens.len() - todo.len();
        out.answered += kept;
        let room = if again {
            overlaps::MAX_SUGGESTIONS
        } else {
            overlaps::MAX_SUGGESTIONS.saturating_sub(kept)
        };
        out.no_room += todo.len().saturating_sub(room);
        todo.truncate(room);
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::collect::labels::bulk::test_support::put_frame;
    use crate::collect::labels::{self, LabelBox};
    use crate::training::overlaps::OverlapsRequest;

    /// QA-070 ([41]): a frame with more open pairs than its list can hold gets only the pairs
    /// that fit; once its list is full, the rest are not cropped again on every run (they were
    /// judged, thrown away by the bound and never recorded) until room frees or `again`.
    #[test]
    fn a_full_frame_gets_no_pairs_its_list_cannot_keep() {
        let t = crate::test_dirs::TempDir::new("overlaps-select-room");
        let dir = t.join("dataset");
        // Nested boxes, each too much larger than the last to stack: every pair overlaps.
        let boxes: Vec<LabelBox> = (0..13)
            .map(|i| {
                let side = 30.0 * 1.4_f32.powi(i);
                let mut b = labels::tests_support::auto_box("car");
                b.bbox = [0.0, 0.0, side, side];
                b.auto.labelers.insert(format!("d{i}"), 0.5);
                b
            })
            .collect();
        let rel = put_frame(&dir, "s", None, boxes);
        let classes = vec!["car".to_string()];
        let wanted = Wanted {
            classes: &classes,
            hidden: &[],
        };
        let req = OverlapsRequest::parse("").unwrap();
        let frame = Frame::resolve(&dir, &rel).unwrap();
        let open =
            overlaps::open_pairs(&labels::read(&frame.labels).unwrap().unwrap(), &wanted).len();
        assert!(open > overlaps::MAX_SUGGESTIONS, "{open}");
        let s = select(&dir, &req.frames, &wanted).unwrap();
        assert_eq!(
            (s.pairs, s.no_room),
            (overlaps::MAX_SUGGESTIONS, open - overlaps::MAX_SUGGESTIONS)
        );
        // The run answers what it selected: the list is full.
        let picked = s.frames[0].pairs.clone();
        labels::update(&frame.labels, |cur| {
            let mut f = cur.unwrap();
            overlaps::apply(&mut f, &wanted, false, &|p| {
                picked
                    .contains_key(&pair_key(p))
                    .then_some(overlaps::Zoom::Unavailable)
            });
            Ok(Some(f))
        })
        .unwrap();
        let full = labels::read(&frame.labels).unwrap().unwrap();
        assert_eq!(full.overlap_suggestions.len(), overlaps::MAX_SUGGESTIONS);
        let s = select(&dir, &req.frames, &wanted).unwrap();
        assert_eq!(s.pairs, 0, "nothing judged that the list cannot keep");
        assert_eq!(s.no_room, open - overlaps::MAX_SUGGESTIONS);
        assert_eq!(s.answered, overlaps::MAX_SUGGESTIONS);
        // `again` looks at them afresh.
        let again = OverlapsRequest::parse("again=1").unwrap();
        let s = select(&dir, &again.frames, &wanted).unwrap();
        assert_eq!(s.pairs, overlaps::MAX_SUGGESTIONS);
    }
}
