//! The index's rows: what [`super::super::queue::build`] reads of one frame, in about 108 bytes
//! plus its text and its need candidates. Names, bucket lists and class sets are interned
//! ([`Names`]), text and the Recommended order's candidate boxes ([`CellBox`], 12 bytes each, of
//! [`need::of_file`] or [`need::of_reply`]) sit in their day shard's arenas ([`Shard`]); no other
//! box geometry is kept: `auto=overlaps`, `auto=both` and `auto=refine` (per `refine_kind`,
//! [`Names::refine`]) are worked out with the real functions when a row is encoded, for the
//! index's wanted classes.
//!
//! A row has three parts because they change apart: [`Cand`] comes from the review line or index
//! line, [`Side`] from the sidecar, [`Part`] from the label file. Encoding goes through the real
//! functions ([`Uncertainty::of`], [`Uncertainty::with_labels`], [`Filters::auto_keeps`],
//! [`Filters::overlap_strength`], [`frame_labels`], [`LabelFile::training`], ...), so nothing about
//! what a box means is written twice.
//!
//! No field saturates: counts of what a file holds fit the types below because of the file size
//! caps, which the `const` checks state.
//!
//! The arenas only grow: a row's update appends its new text and candidates (unless they are the
//! ones it has) and leaves the old ones unreferenced. They are reclaimed when the thread walks the
//! day again (a scan of a changed day, the full pass every 30 minutes), which builds the shard anew.

use super::super::super::layout::{self, Walker};
use super::super::super::record::{Bucket, Record};
use super::super::bulk::Wanted;
use super::super::need::{self, CellBox};
use super::super::queue::{self, AutoFilter, Candidate, Filters, Uncertainty};
use super::super::recommend;
use super::super::refine::Kind as RefineKind;
use super::super::{Frame, LabelBox, LabelFile, MAX_LABEL_FILE_BYTES, MAX_SIDECAR_BYTES, Status};
use crate::sync::RwLockExt as _;
use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;
use std::sync::{Arc, RwLock};

/// An interned name ([`Names`]).
pub(super) type Name = u32;

/// Fewest bytes a box takes in a label file: it has four required keys, so this is far below the
/// truth. With it, a label file at its size cap holds fewer boxes than a `u16` counts.
const MIN_BOX_JSON_BYTES: u64 = 10;
const _: () = assert!(MAX_LABEL_FILE_BYTES / MIN_BOX_JSON_BYTES <= u16::MAX as u64);
// A count of anything in a sidecar is below its size in bytes.
const _: () = assert!(MAX_SIDECAR_BYTES <= u32::MAX as u64);

/// Need candidates of one file: at most two a box of a label file ([`need::of_file`]), one a
/// detection of a sidecar ([`need::of_reply`]); fewer than a `u32` counts either way.
const _: () = assert!(MAX_LABEL_FILE_BYTES / MIN_BOX_JSON_BYTES * 2 <= u32::MAX as u64);

/// Where a run of [`CellBox`]es of one file sits in its shard's arena ([`Shard::boxes`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Run {
    pub at: u32,
    pub len: u32,
}

/// A count of boxes of one label file, which fits (see [`MIN_BOX_JSON_BYTES`]).
fn box_count(n: usize) -> u16 {
    u16::try_from(n).unwrap_or(u16::MAX)
}

/// Fewest bytes a suggestion of the dataset check takes in a label file (`{"kind":"add"},`): with
/// it, a label file at its size cap holds fewer suggestions than a `u16` counts.
const MIN_SUGGESTION_JSON_BYTES: u64 = 15;
const _: () = assert!(MAX_LABEL_FILE_BYTES / MIN_SUGGESTION_JSON_BYTES <= u16::MAX as u64);
/// The strongest a suggestion can be: [`super::super::overlaps::strength`] has bands under 8,
/// [`super::super::refine::strength`] under 5, each times 1000 plus at most 999. So a frame
/// strength fits a `u16`.
const MAX_STRENGTH: u32 = 7 * 1000 + 999;
const _: () = assert!(MAX_STRENGTH <= u16::MAX as u32);

/// A strength ([`MAX_STRENGTH`]) or a count of suggestions ([`MIN_SUGGESTION_JSON_BYTES`]) of one
/// label file, which fits.
fn small(n: u32) -> u16 {
    u16::try_from(n).unwrap_or(u16::MAX)
}

/// The kinds `auto=refine` narrows to, each with its place in a frame's refine summary
/// ([`Names::refine`]); None = every kind.
const REFINE_KINDS: [Option<RefineKind>; 5] = [
    None,
    Some(RefineKind::Add),
    Some(RefineKind::Tighten),
    Some(RefineKind::Relabel),
    Some(RefineKind::Verify),
];

/// The auto filters, each with its bit in [`Part::auto`].
const AUTOS: [AutoFilter; 8] = [
    AutoFilter::Review,
    AutoFilter::Accept,
    AutoFilter::Nothing,
    AutoFilter::Never,
    AutoFilter::Repeats,
    AutoFilter::Overlaps,
    AutoFilter::Both,
    AutoFilter::Dismissed,
];

/// The bit of `auto` in [`Part::auto`].
pub(super) fn auto_bit(auto: AutoFilter) -> u8 {
    AUTOS
        .iter()
        .position(|a| *a == auto)
        .map_or(0, |at| 1 << at)
}

/// One interning table: each distinct value once, ids from 0.
#[derive(Debug)]
struct Table<K: ?Sized> {
    items: Vec<Arc<K>>,
    ids: HashMap<Arc<K>, u32>,
    /// Bytes behind the [`Arc`]s.
    heap: usize,
}

impl<K: ?Sized> Default for Table<K> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            ids: HashMap::new(),
            heap: 0,
        }
    }
}

impl<K: ?Sized + Hash + Eq> Table<K> {
    fn intern(&mut self, value: &K) -> u32
    where
        for<'a> Arc<K>: From<&'a K>,
    {
        if let Some(id) = self.ids.get(value) {
            return *id;
        }
        // Each entry takes memory, so the ceiling trips long before 4 billion of them.
        let id = u32::try_from(self.items.len()).unwrap_or(u32::MAX);
        let item: Arc<K> = Arc::from(value);
        self.heap += size_of_val(&*item) + 2 * size_of::<usize>();
        self.items.push(item.clone());
        self.ids.insert(item, id);
        id
    }

    fn get(&self, id: u32) -> Option<&K> {
        self.items.get(id as usize).map(|i| &**i)
    }

    fn find(&self, value: &K) -> Option<u32> {
        self.ids.get(value).copied()
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    fn iter(&self) -> impl Iterator<Item = &K> {
        self.items.iter().map(|i| &**i)
    }

    /// Bytes of the entries `used` (by id) does not mark: what each takes behind its [`Arc`], its
    /// slot in `items` and its map slot, as [`Self::bytes`] counts them.
    fn unused_bytes(&self, used: &[bool]) -> usize {
        let slots = size_of::<Arc<K>>() + (size_of::<(Arc<K>, u32)>() + 1) * 8 / 7;
        self.items
            .iter()
            .zip(used.iter().copied().chain(std::iter::repeat(false)))
            .filter(|(_, used)| !used)
            .map(|(item, _)| size_of_val(&**item) + 2 * size_of::<usize>() + slots)
            .sum()
    }

    /// Heap bytes, from capacities (a map slot is its entry and a control byte, at most 8/7 full).
    fn bytes(&self) -> usize {
        self.items.capacity() * size_of::<Arc<K>>()
            + self.ids.capacity() * 8 / 7 * (size_of::<(Arc<K>, u32)>() + 1)
            + self.heap
    }
}

/// Everything interned, for one index: it only grows, and goes with the index. Rows hold ids. A
/// walk of everything that leaves too much of it unused ([`Self::unused_bytes`]) drops the index
/// for a fresh one ([`super::COMPACT_UNUSED_BYTES`]).
#[derive(Debug, Default)]
pub(super) struct Names {
    names: Table<str>,
    /// Bucket lists as their line holds them: the order is shown, and a bucket can repeat.
    buckets: Table<[Bucket]>,
    /// Sets of class names ([`queue::frame_labels`]), sorted, each with how sure the frame is of
    /// it in hundredths ([`queue::frame_confidence`]).
    classes: Table<[(Name, u8)]>,
    /// The training boxes of a frame, per class with their count.
    training: Table<[(Name, u32)]>,
    /// What `auto=refine` reads of a frame: per entry of [`REFINE_KINDS`], its live suggestions and
    /// the strongest one's strength ([`queue::refine_open`]). Empty: none live, the common case.
    refine: Table<[(u16, u16)]>,
    /// The labels models and auto-labelers proposed on a frame and a reviewer decided on
    /// ([`queue::proposed_labels`]), in label order: what `balance=` weighs a camera's class by.
    proposed: Table<[Name]>,
}

impl Names {
    pub(super) fn name(&self, id: Name) -> &str {
        self.names.get(id).unwrap_or_default()
    }

    pub(super) fn find_name(&self, name: &str) -> Option<Name> {
        self.names.find(name)
    }

    pub(super) fn name_count(&self) -> usize {
        self.names.len()
    }

    pub(super) fn buckets(&self, id: u32) -> &[Bucket] {
        self.buckets.get(id).unwrap_or_default()
    }

    pub(super) fn bucket_lists(&self) -> impl Iterator<Item = &[Bucket]> {
        self.buckets.iter()
    }

    pub(super) fn class_sets(&self) -> impl Iterator<Item = &[(Name, u8)]> {
        self.classes.iter()
    }

    pub(super) fn class_set(&self, id: u32) -> &[(Name, u8)] {
        self.classes.get(id).unwrap_or_default()
    }

    pub(super) fn training_sets(&self) -> usize {
        self.training.len()
    }

    pub(super) fn training(&self, id: u32) -> &[(Name, u32)] {
        self.training.get(id).unwrap_or_default()
    }

    /// The proposed labels of [`Part::proposed`] (0: none).
    pub(super) fn proposed(&self, id: u32) -> &[Name] {
        id.checked_sub(1)
            .and_then(|id| self.proposed.get(id))
            .unwrap_or_default()
    }

    /// Per refine summary id ([`Part::refine`] minus one): the live suggestions of `kind` (None =
    /// every kind) and the strongest one's strength, as [`Filters::refine_open`] counts them.
    pub(super) fn refine_by_summary(&self, kind: Option<RefineKind>) -> Vec<(u32, u32)> {
        let at = REFINE_KINDS.iter().position(|k| *k == kind);
        self.refine
            .iter()
            .map(|summary| {
                at.and_then(|at| summary.get(at))
                    .map_or((0, 0), |&(n, s)| (u32::from(n), u32::from(s)))
            })
            .collect()
    }

    /// Bytes of the entries no row of `rows` refers to, directly or through an entry it uses
    /// (the names in a class set, a training set or a proposed list): what relabelled frames left
    /// behind. Rows of a frame that went still count: they keep their camera.
    pub(super) fn unused_bytes<'a>(&self, rows: impl IntoIterator<Item = &'a Row>) -> usize {
        fn mark(used: &mut [bool], id: u32) {
            if let Some(u) = used.get_mut(id as usize) {
                *u = true;
            }
        }
        let mut names = vec![false; self.names.len()];
        let mut buckets = vec![false; self.buckets.len()];
        let mut classes = vec![false; self.classes.len()];
        let mut training = vec![false; self.training.len()];
        let mut refine = vec![false; self.refine.len()];
        let mut proposed = vec![false; self.proposed.len()];
        for r in rows {
            mark(&mut names, r.cand.scene);
            mark(&mut names, r.cand.camera);
            mark(&mut buckets, r.cand.buckets);
            mark(&mut names, r.side.model);
            mark(&mut classes, r.side.reply);
            mark(&mut classes, r.part.classes);
            mark(&mut training, r.part.training);
            if let Some(id) = r.part.refine.checked_sub(1) {
                mark(&mut refine, id);
            }
            if let Some(id) = r.part.proposed.checked_sub(1) {
                mark(&mut proposed, id);
            }
        }
        fn live(used: &[bool]) -> impl Iterator<Item = bool> + '_ {
            used.iter().copied().chain(std::iter::repeat(false))
        }
        for (set, used) in self.classes.items.iter().zip(live(&classes)) {
            if used {
                set.iter().for_each(|(name, _)| mark(&mut names, *name));
            }
        }
        for (set, used) in self.training.items.iter().zip(live(&training)) {
            if used {
                set.iter().for_each(|(name, _)| mark(&mut names, *name));
            }
        }
        for (set, used) in self.proposed.items.iter().zip(live(&proposed)) {
            if used {
                set.iter().for_each(|name| mark(&mut names, *name));
            }
        }
        self.names.unused_bytes(&names)
            + self.buckets.unused_bytes(&buckets)
            + self.classes.unused_bytes(&classes)
            + self.training.unused_bytes(&training)
            + self.refine.unused_bytes(&refine)
            + self.proposed.unused_bytes(&proposed)
    }

    pub(super) fn bytes(&self) -> usize {
        self.names.bytes()
            + self.buckets.bytes()
            + self.classes.bytes()
            + self.training.bytes()
            + self.refine.bytes()
            + self.proposed.bytes()
    }
}

/// What the line a frame was found on says.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Cand {
    pub scene: Name,
    pub camera: Name,
    pub buckets: u32,
}

/// What the sidecar says.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct Side {
    pub model: Name,
    /// [`Uncertainty::margin`], exact: it is serialized. Meaningless without `has_margin`.
    pub margin: f32,
    pub suppressed: u32,
    /// The classes of the reply's predictions: what the frame shows without a label file.
    pub reply: u32,
    pub has_margin: bool,
    pub disagreement: bool,
    /// The frame's size ([`Record::width`], [`Record::height`]): where its boxes are in it.
    pub width: u32,
    pub height: u32,
    /// [`need::of_reply`]: the need candidates of a frame without a label file.
    pub boxes: Run,
    /// [`recommend::own_event`]: an event of its own in the Recommended order.
    pub alone: bool,
    /// False: it did not read, the frame counts as unreadable.
    pub ok: bool,
}

/// What the label file says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Part {
    /// The frame's classes, of its label file. Meaningless without `has_file`.
    pub classes: u32,
    pub training: u32,
    /// [`Shard::note`]'s number plus one, 0 = no re-label note.
    pub note: u32,
    /// The id of the frame's refine summary ([`Names::refine`]) plus one, 0 = no live suggestion
    /// of the dataset check.
    pub refine: u32,
    /// The id of the frame's proposed labels ([`Names::proposed`]) plus one, 0 = none.
    pub proposed: u32,
    /// [`need::of_file`]: the need candidates of the label file.
    pub boxes: Run,
    /// [`super::super::overlaps::frame_strength`] for the index's wanted classes
    /// ([`MAX_STRENGTH`]).
    pub strength: u16,
    pub auto_review: u16,
    pub likely: u16,
    /// Agreed boxes for the index's wanted classes.
    pub agreed: u16,
    pub negatives: u16,
    pub audit_accepted: u16,
    pub audit_rejected: u16,
    pub status: Status,
    /// One bit per auto filter the frame passes ([`auto_bit`]).
    pub auto: u8,
    pub has_file: bool,
    /// The label file did not read, the frame counts as unreadable.
    pub bad: bool,
    pub empty_confirmed: bool,
}

impl Default for Part {
    fn default() -> Self {
        Self {
            classes: 0,
            training: 0,
            note: 0,
            refine: 0,
            proposed: 0,
            boxes: Run::default(),
            strength: 0,
            auto_review: 0,
            likely: 0,
            agreed: 0,
            negatives: 0,
            audit_accepted: 0,
            audit_rejected: 0,
            status: Status::Unreviewed,
            auto: 0,
            has_file: false,
            bad: false,
            empty_confirmed: false,
        }
    }
}

/// One candidate of the queue.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct Row {
    /// Where the stem is in the arena; the timestamp follows it.
    text_at: u32,
    timestamp_len: u32,
    pub cand: Cand,
    pub side: Side,
    pub part: Part,
    /// At most 64 ([`layout::parse_rel`]).
    stem_len: u8,
    /// In `review-queue.jsonl`.
    pub queued: bool,
    /// A `.json` path on a review line: a candidate, but no frame can ever resolve from it.
    json: bool,
    /// Its image or sidecar is gone: only its camera still counts.
    pub gone: bool,
}

impl Row {
    /// Listed, or at least counted as there: it neither went away nor fails to read.
    pub(super) fn readable(&self) -> bool {
        self.side.ok && !self.part.bad
    }

    /// The set of [`queue::frame_labels`].
    pub(super) fn classes(&self) -> u32 {
        if self.part.has_file {
            self.part.classes
        } else {
            self.side.reply
        }
    }

    /// The row's re-derive needs the sidecar too: its side part is not known to be current.
    pub(super) fn needs_side(&self) -> bool {
        self.gone || !self.side.ok
    }

    /// The uncertainty of the row. `refine` is the row's live suggestions and their strength for
    /// the query's `refine_kind` (looked up by the caller in [`Names::refine_by_summary`]).
    /// `relabel` is the note's text; a caller that only sorts passes `Some(String::new())` when
    /// there is a note: the sort only asks whether there is one.
    pub(super) fn uncertainty(
        &self,
        auto: Option<AutoFilter>,
        refine: (u32, u32),
        relabel: Option<String>,
    ) -> Uncertainty {
        let (refine_open, refine_strength) = match auto {
            Some(AutoFilter::Refine) => refine,
            _ => (0, 0),
        };
        Uncertainty {
            margin: self.side.has_margin.then_some(self.side.margin),
            disagreement: self.side.disagreement,
            suppressed: self.side.suppressed as usize,
            relabel,
            auto_review: usize::from(self.part.auto_review),
            likely_boxes: u32::from(self.part.likely),
            repeats_only: self.part.auto & auto_bit(AutoFilter::Repeats) != 0,
            overlap_strength: match auto {
                Some(AutoFilter::Overlaps) => u32::from(self.part.strength),
                _ => 0,
            },
            agreed_boxes: match auto {
                Some(AutoFilter::Both) => u32::from(self.part.agreed),
                _ => 0,
            },
            refine_strength,
            refine_open,
            balance: None,
            need: None,
        }
    }
}

/// The rows of one `<scene>/<camera>/<date>`, by stem, and their text. A frame's day is that of
/// its path, whichever line named it.
#[derive(Debug, Default)]
pub(super) struct Shard {
    pub day: String,
    /// Changes with every row added or updated, unique over every shard of the process: what the
    /// query's per-shard cache ([`super::query::NeedCache`]) is keyed on.
    pub version: u64,
    pub rows: Vec<Row>,
    text: String,
    notes: Vec<Box<str>>,
    /// The rows' need candidates ([`Side::boxes`], [`Part::boxes`]): a row's update appends.
    boxes: Vec<CellBox>,
}

/// What the sidecar of a frame said: its row part, its timestamp and its need candidates.
#[derive(Debug, Clone, Default)]
pub(super) struct SideRead {
    pub side: Side,
    pub timestamp: String,
    pub boxes: Vec<CellBox>,
}

/// The next [`Shard::version`].
fn next_version() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// A frame as read from disk, to go into a row.
#[derive(Debug, Clone, Default)]
pub(super) struct Encoded {
    pub gone: bool,
    /// The sidecar part; None = not read, the row keeps what it has.
    pub side: Option<SideRead>,
    pub part: Part,
    /// The label file's need candidates ([`Part::boxes`]).
    pub part_boxes: Vec<CellBox>,
    pub note: Option<String>,
}

impl Shard {
    pub(super) fn new(day: String) -> Self {
        Self {
            day,
            version: next_version(),
            ..Self::default()
        }
    }

    fn text(&self, at: usize, len: usize) -> &str {
        at.checked_add(len)
            .and_then(|end| self.text.get(at..end))
            .unwrap_or_default()
    }

    fn stem(&self, r: &Row) -> &str {
        self.text(r.text_at as usize, usize::from(r.stem_len))
    }

    pub(super) fn timestamp(&self, r: &Row) -> &str {
        self.text(
            r.text_at as usize + usize::from(r.stem_len),
            r.timestamp_len as usize,
        )
    }

    pub(super) fn note(&self, r: &Row) -> Option<&str> {
        let at = r.part.note.checked_sub(1)?;
        self.notes.get(at as usize).map(|n| &**n)
    }

    fn run(&self, run: Run) -> &[CellBox] {
        let at = run.at as usize;
        at.checked_add(run.len as usize)
            .and_then(|end| self.boxes.get(at..end))
            .unwrap_or_default()
    }

    /// The need candidates of `r`: of its label file, else of its sidecar ([`Row::classes`]'s
    /// rule).
    pub(super) fn boxes_of(&self, r: &Row) -> &[CellBox] {
        if r.part.has_file {
            self.run(r.part.boxes)
        } else {
            self.run(r.side.boxes)
        }
    }

    /// Put `boxes` in the arena: where they are. Empty: no room taken.
    fn store_boxes(&mut self, boxes: &[CellBox]) -> Run {
        if boxes.is_empty() {
            return Run::default();
        }
        // The ceiling trips long before 4 billion boxes.
        let at = u32::try_from(self.boxes.len()).unwrap_or(u32::MAX);
        self.boxes.extend_from_slice(boxes);
        Run {
            at,
            len: u32::try_from(boxes.len()).unwrap_or(0),
        }
    }

    /// The path of `r`, `<scene>/<camera>/<date>/<stem>.jpg`.
    pub(super) fn rel(&self, r: &Row) -> String {
        let ext = if r.json { "json" } else { "jpg" };
        format!("{}/{}.{ext}", self.day, self.stem(r))
    }

    /// The bytes of [`Self::rel`], for a comparison that builds no string. `r` is a frame: `.jpg`.
    pub(super) fn rel_bytes<'a>(&'a self, r: &Row) -> impl Iterator<Item = u8> + 'a {
        self.day
            .bytes()
            .chain(std::iter::once(b'/'))
            .chain(self.stem(r).bytes())
            .chain(".jpg".bytes())
    }

    /// The row of `<stem>.jpg` (`.json` with `json`).
    pub(super) fn find(&self, stem: &str, json: bool) -> Option<usize> {
        self.rows
            .binary_search_by(|r| (self.stem(r), r.json).cmp(&(stem, json)))
            .ok()
    }

    /// Put `stem` and `timestamp` in the arena: where they are. None when the arena has outgrown
    /// what a row can point at; the ceiling trips long before.
    fn store(&mut self, stem: &str, timestamp: &str) -> Option<u32> {
        let at = u32::try_from(self.text.len()).ok()?;
        self.text.push_str(stem);
        self.text.push_str(timestamp);
        Some(at)
    }

    /// The number + 1 of the note `note`.
    fn store_note(&mut self, note: Option<&str>, old: u32) -> u32 {
        let Some(note) = note else { return 0 };
        if old != 0
            && let Some(same) = old.checked_sub(1)
            && self.notes.get(same as usize).is_some_and(|n| &**n == note)
        {
            return old;
        }
        self.notes.push(note.into());
        u32::try_from(self.notes.len()).unwrap_or(u32::MAX)
    }

    /// Add a row at the end: the caller adds them by stem, then by `.jpg` before `.json`.
    /// A row of this shard changes in place: a new [`Self::version`].
    pub(super) fn touch(&mut self) {
        self.version = next_version();
    }

    pub(super) fn push(&mut self, stem: &str, json: bool, queued: bool, cand: Cand, e: Encoded) {
        self.version = next_version();
        let SideRead {
            mut side,
            timestamp,
            boxes,
        } = e.side.unwrap_or_default();
        side.boxes = self.store_boxes(&boxes);
        let note = self.store_note(e.note.as_deref(), 0);
        let part_boxes = self.store_boxes(&e.part_boxes);
        let text_at = self.store(stem, &timestamp).unwrap_or(u32::MAX);
        self.rows.push(Row {
            text_at,
            timestamp_len: u32::try_from(timestamp.len()).unwrap_or(0),
            cand,
            side,
            part: Part {
                note,
                boxes: part_boxes,
                ..e.part
            },
            stem_len: u8::try_from(stem.len()).unwrap_or(u8::MAX),
            queued,
            json,
            gone: e.gone,
        });
    }

    /// Put what `e` says into row `at`.
    pub(super) fn update(&mut self, at: usize, e: Encoded) {
        self.version = next_version();
        let Some(mut row) = self.rows.get(at).copied() else {
            return;
        };
        row.gone = e.gone;
        if !e.gone {
            if let Some(SideRead {
                mut side,
                timestamp,
                boxes,
            }) = e.side
            {
                if self.timestamp(&row) != timestamp {
                    let stem = self.stem(&row).to_string();
                    if let Some(text_at) = self.store(&stem, &timestamp) {
                        row.text_at = text_at;
                        row.timestamp_len = u32::try_from(timestamp.len()).unwrap_or(0);
                    }
                }
                side.boxes = if self.run(row.side.boxes) == boxes.as_slice() {
                    row.side.boxes
                } else {
                    self.store_boxes(&boxes)
                };
                row.side = side;
            }
            let note = self.store_note(e.note.as_deref(), row.part.note);
            let part_boxes = if self.run(row.part.boxes) == e.part_boxes.as_slice() {
                row.part.boxes
            } else {
                self.store_boxes(&e.part_boxes)
            };
            row.part = Part {
                note,
                boxes: part_boxes,
                ..e.part
            };
        }
        if let Some(slot) = self.rows.get_mut(at) {
            *slot = row;
        }
    }

    /// Bytes of rows and text so far: a quick look during a walk.
    pub(super) fn used(&self) -> usize {
        self.rows.len() * size_of::<Row>()
            + self.text.len()
            + self.boxes.len() * size_of::<CellBox>()
    }

    /// Heap bytes, from capacities.
    pub(super) fn bytes(&self) -> usize {
        self.rows.capacity() * size_of::<Row>()
            + self.text.capacity()
            + self.boxes.capacity() * size_of::<CellBox>()
            + self.notes.capacity() * size_of::<Box<str>>()
            + self.notes.iter().map(|n| n.len()).sum::<usize>()
            + self.day.capacity()
    }

    /// Give back what a walk overallocated.
    pub(super) fn shrink(&mut self) {
        self.rows.shrink_to_fit();
        self.text.shrink_to_fit();
        self.boxes.shrink_to_fit();
    }
}

/// What the rows need of the class lists: the filters that ask for them, once per index.
#[derive(Debug)]
pub(super) struct Derive {
    /// One filter per auto filter, in [`AUTOS`]' order, with the index's wanted classes.
    filters: Vec<Filters>,
    /// `auto=refine` once per entry of [`REFINE_KINDS`], with the index's wanted classes.
    refine: Vec<Filters>,
    /// The index's wanted classes, for the need candidates.
    wanted: Option<(Vec<String>, Vec<String>)>,
}

impl Derive {
    pub(super) fn new(wanted: Option<&(Vec<String>, Vec<String>)>) -> Self {
        let with = |auto: AutoFilter, refine_kind: Option<RefineKind>| Filters {
            auto: Some(auto),
            wanted: wanted.cloned(),
            refine_kind,
            ..Filters::default()
        };
        Self {
            filters: AUTOS.iter().map(|a| with(*a, None)).collect(),
            refine: REFINE_KINDS
                .iter()
                .map(|k| with(AutoFilter::Refine, *k))
                .collect(),
            wanted: wanted.cloned(),
        }
    }

    /// The index's wanted classes as [`Wanted`].
    pub(super) fn wanted(&self) -> Option<Wanted<'_>> {
        self.wanted
            .as_ref()
            .map(|(classes, hidden)| Wanted { classes, hidden })
    }

    fn of(&self, auto: AutoFilter) -> Option<&Filters> {
        self.filters.iter().find(|f| f.auto == Some(auto))
    }

    /// The label part of a frame with `labels` (None = no label file), its re-label note and its
    /// need candidates.
    pub(super) fn part(
        &self,
        labels: Option<&LabelFile>,
        names: &RwLock<Names>,
    ) -> (Part, Option<String>, Vec<CellBox>) {
        let u = Uncertainty::default().with_labels(labels);
        let auto = self.filters.iter().enumerate().fold(0u8, |bits, (at, f)| {
            bits | u8::from(f.auto_keeps(labels)) << at
        });
        let refine: Vec<(u16, u16)> = self
            .refine
            .iter()
            .map(|f| {
                let (n, s) = f.refine_open(labels);
                (small(n), small(s))
            })
            .collect();
        let mut part = Part {
            strength: self
                .of(AutoFilter::Overlaps)
                .map_or(0, |f| small(f.overlap_strength(labels))),
            auto_review: box_count(u.auto_review),
            likely: box_count(u.likely_boxes as usize),
            agreed: self
                .of(AutoFilter::Both)
                .map_or(0, |f| box_count(f.agreed_boxes(labels) as usize)),
            auto,
            ..Part::default()
        };
        let Some(l) = labels else {
            return (part, u.relabel, Vec::new());
        };
        let boxes = need::of_file(l, self.wanted().as_ref());
        part.status = l.status;
        part.has_file = true;
        part.empty_confirmed = l.empty_confirmed;
        part.negatives = box_count(l.hard_negatives().count());
        let verdicts = || l.boxes.iter().filter_map(LabelBox::audit_verdict);
        part.audit_accepted = box_count(verdicts().filter(|v| *v).count());
        part.audit_rejected = box_count(verdicts().filter(|v| !*v).count());
        let classes = queue::file_confidence(l);
        let mut training: BTreeMap<&str, u32> = BTreeMap::new();
        for b in l.training() {
            *training.entry(b.label.as_str()).or_default() += 1;
        }
        let mut names = names.write_recover();
        // Every kind's count is 0 when the whole count is: a frame with nothing live interns
        // nothing.
        if refine.first().is_some_and(|(n, _)| *n > 0) {
            part.refine = names.refine.intern(&refine).saturating_add(1);
        }
        part.classes = names.class_set_of(classes.iter().map(|(c, s)| (c.as_str(), *s)));
        let per_class: Vec<(Name, u32)> = training
            .into_iter()
            .map(|(class, n)| (names.names.intern(class), n))
            .collect();
        part.training = names.training.intern(&per_class);
        let proposed = queue::proposed_labels(l);
        if !proposed.is_empty() {
            let set: Vec<Name> = proposed.iter().map(|p| names.names.intern(p)).collect();
            part.proposed = names.proposed.intern(&set).saturating_add(1);
        }
        (part, u.relabel, boxes)
    }
}

impl Names {
    /// The id of the class set of `classes`, sorted distinct names with their confidence.
    fn class_set_of<'a>(&mut self, classes: impl Iterator<Item = (&'a str, u8)>) -> u32 {
        let set: Vec<(Name, u8)> = classes.map(|(c, s)| (self.names.intern(c), s)).collect();
        self.classes.intern(&set)
    }
}

/// Reads frames into rows, on any thread.
#[derive(Debug, Clone, Copy)]
pub(super) struct Encoder<'a> {
    pub derive: &'a Derive,
    pub names: &'a RwLock<Names>,
}

impl Encoder<'_> {
    pub(super) fn cand(&self, c: &Candidate) -> Cand {
        let mut names = self.names.write_recover();
        Cand {
            scene: names.names.intern(&c.scene),
            camera: names.names.intern(&c.camera),
            buckets: names.buckets.intern(&c.buckets),
        }
    }

    pub(super) fn side(&self, rec: &Record) -> SideRead {
        let u = Uncertainty::of(rec);
        let reply = queue::reply_confidence(rec);
        let boxes = need::of_reply(rec, self.derive.wanted().as_ref());
        let mut names = self.names.write_recover();
        let side = Side {
            model: names.names.intern(&rec.model),
            margin: u.margin.unwrap_or(0.0),
            suppressed: u32::try_from(u.suppressed).unwrap_or(u32::MAX),
            reply: names.class_set_of(reply.iter().map(|(c, s)| (c.as_str(), *s))),
            width: rec.width,
            height: rec.height,
            boxes: Run::default(),
            has_margin: u.margin.is_some(),
            disagreement: u.disagreement,
            alone: recommend::own_event(rec),
            ok: true,
        };
        SideRead {
            side,
            timestamp: rec.timestamp.clone(),
            boxes,
        }
    }

    /// `frame` as a row: the sidecar too with `side`.
    pub(super) fn frame(&self, frame: Option<Frame>, side: bool) -> Encoded {
        let Some(frame) = frame else {
            return Encoded {
                gone: true,
                ..Encoded::default()
            };
        };
        let side = side.then(|| match frame.record() {
            Ok(rec) => self.side(&rec),
            Err(_) => SideRead::default(),
        });
        let (part, note, part_boxes) = match frame.labels() {
            Ok(labels) => self.derive.part(labels.as_ref(), self.names),
            Err(_) => (
                Part {
                    bad: true,
                    ..Part::default()
                },
                None,
                Vec::new(),
            ),
        };
        Encoded {
            gone: false,
            side,
            part,
            part_boxes,
            note,
        }
    }

    /// The row of `c` on a walk.
    pub(super) fn walked(&self, walker: &mut Walker<'_>, c: &Candidate) -> Encoded {
        self.frame(Frame::resolve_in(walker, &c.rel), true)
    }
}

/// `rel` as (stem, is a `.json` path).
pub(super) fn stem_of(rel: &str) -> Option<(String, bool)> {
    layout::parse_rel(rel).map(|r| (r.stem, rel.ends_with(".json")))
}
