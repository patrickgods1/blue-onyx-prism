//! What `GET /v1/review/next` answers, one shape for both ways of getting it: the walk over the
//! real files ([`Answer::walk`]) and the index ([`super::query`]). The route serializes this and
//! nothing else, so the two cannot differ in the JSON's fields.

use super::super::super::layout;
use super::super::queue::{self, BalanceTally, Counts, Filters, Item, Queue, QueueSource};
use serde::Serialize;
use std::path::Path;

/// How far a build or a walk of stale days has come: frames whose rows are current, of the frames
/// there are; `done < total`. Absent when nothing counted is behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Indexing {
    pub done: u64,
    pub total: u64,
}

/// The frame to show, where it stands and what the filter lists and counts need.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Answer {
    pub item: Option<Item>,
    /// 1-based place of [`Self::item`] in the sorted queue.
    pub position: Option<usize>,
    /// Past the last frame, so back at the first.
    pub wrapped: bool,
    /// Frames the filters list.
    pub total: usize,
    pub counts: Counts,
    pub cameras: Vec<String>,
    pub labels: Vec<String>,
    /// Sidecars and label files that do not parse, and damaged lines.
    pub unreadable: u64,
    /// The next frames after [`Self::item`], for the page to load ahead.
    pub upcoming: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexing: Option<Indexing>,
    /// With `balance=`: the reviewed frames per class, scene and camera the order weighed against.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub balance: Option<BalanceTally>,
}

/// How many frames [`Answer::upcoming`] lists.
pub(super) const UPCOMING: usize = 5;

impl Answer {
    /// The answer from the files: [`queue::build`], [`queue::next_after`] and
    /// [`layout::damaged_lines`]. What the index has to equal.
    pub fn walk(dir: &Path, f: &Filters, after: Option<&str>) -> Self {
        let damaged = layout::damaged_lines(dir, f.source == QueueSource::All);
        Self::of(queue::build(dir, f), damaged, after)
    }

    /// The answer of a built queue with `damaged` lines left out of it.
    pub(super) fn of(built: Queue, damaged: u64, after: Option<&str>) -> Self {
        let found = queue::next_after(&built.items, after);
        Self {
            item: found.map(|(_, i, _)| i.clone()),
            position: found.map(|(p, _, _)| p + 1),
            wrapped: found.is_some_and(|(_, _, w)| w),
            total: built.items.len(),
            upcoming: built
                .items
                .iter()
                .skip(found.map_or(0, |(p, _, _)| p + 1))
                .take(UPCOMING)
                .map(|i| i.frame.clone())
                .collect(),
            counts: built.counts,
            cameras: built.cameras,
            labels: built.labels,
            unreadable: built.unreadable + damaged,
            indexing: None,
            balance: built.balance,
        }
    }
}
