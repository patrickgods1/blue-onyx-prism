//! The index's rows: what [`super::super::queue::build`] reads of one frame, in about 80 bytes
//! plus its text. Names, bucket lists and class sets are interned ([`Names`]), text sits in its day
//! shard's arena ([`Shard`]), and box geometry is not kept: `auto=overlaps` and `auto=both` are
//! worked out with the real functions when a row is encoded, for the index's wanted classes.
//!
//! A row has three parts because they change apart: [`Cand`] comes from the review line or index
//! line, [`Side`] from the sidecar, [`Part`] from the label file. Encoding goes through the real
//! functions ([`Uncertainty::of`], [`Uncertainty::with_labels`], [`Filters::auto_keeps`],
//! [`Filters::overlap_strength`], [`frame_labels`], [`LabelFile::training`], ...), so nothing about
//! what a box means is written twice.
//!
//! No field saturates: counts of what a file holds fit the types below because of the file size
//! caps, which the `const` checks state.

use super::super::super::layout::{self, Walker};
use super::super::super::record::{Bucket, Record};
use super::super::queue::{self, AutoFilter, Candidate, Filters, Uncertainty};
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

/// A count of boxes of one label file, which fits (see [`MIN_BOX_JSON_BYTES`]).
fn box_count(n: usize) -> u16 {
    u16::try_from(n).unwrap_or(u16::MAX)
}

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

    /// Heap bytes, from capacities (a map slot is its entry and a control byte, at most 8/7 full).
    fn bytes(&self) -> usize {
        self.items.capacity() * size_of::<Arc<K>>()
            + self.ids.capacity() * 8 / 7 * (size_of::<(Arc<K>, u32)>() + 1)
            + self.heap
    }
}

/// Everything interned, for one index: it only grows, and goes with the index. Rows hold ids.
#[derive(Debug, Default)]
pub(super) struct Names {
    names: Table<str>,
    /// Bucket lists as their line holds them: the order is shown, and a bucket can repeat.
    buckets: Table<[Bucket]>,
    /// Sets of class names ([`queue::frame_labels`]), sorted.
    classes: Table<[Name]>,
    /// The training boxes of a frame, per class with their count.
    training: Table<[(Name, u32)]>,
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

    pub(super) fn class_sets(&self) -> impl Iterator<Item = &[Name]> {
        self.classes.iter()
    }

    pub(super) fn class_set(&self, id: u32) -> &[Name] {
        self.classes.get(id).unwrap_or_default()
    }

    pub(super) fn training_sets(&self) -> usize {
        self.training.len()
    }

    pub(super) fn training(&self, id: u32) -> &[(Name, u32)] {
        self.training.get(id).unwrap_or_default()
    }

    pub(super) fn bytes(&self) -> usize {
        self.names.bytes() + self.buckets.bytes() + self.classes.bytes() + self.training.bytes()
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
    /// False: it did not read, the frame counts as unreadable.
    pub ok: bool,
}

/// What the label file says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Part {
    /// The frame's classes, of its label file. Meaningless without `has_file`.
    pub classes: u32,
    pub training: u32,
    /// [`super::super::overlaps::frame_strength`] for the index's wanted classes.
    pub strength: u32,
    /// [`Shard::note`]'s number plus one, 0 = no re-label note.
    pub note: u32,
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
            strength: 0,
            note: 0,
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

    /// The uncertainty of the row. `relabel` is the note's text; a caller that only sorts passes
    /// `Some(String::new())` when there is a note: the sort only asks whether there is one.
    pub(super) fn uncertainty(
        &self,
        auto: Option<AutoFilter>,
        relabel: Option<String>,
    ) -> Uncertainty {
        Uncertainty {
            margin: self.side.has_margin.then_some(self.side.margin),
            disagreement: self.side.disagreement,
            suppressed: self.side.suppressed as usize,
            relabel,
            auto_review: usize::from(self.part.auto_review),
            likely_boxes: u32::from(self.part.likely),
            repeats_only: self.part.auto & auto_bit(AutoFilter::Repeats) != 0,
            overlap_strength: match auto {
                Some(AutoFilter::Overlaps) => self.part.strength,
                _ => 0,
            },
            agreed_boxes: match auto {
                Some(AutoFilter::Both) => u32::from(self.part.agreed),
                _ => 0,
            },
        }
    }
}

/// The rows of one `<scene>/<camera>/<date>`, by stem, and their text. A frame's day is that of
/// its path, whichever line named it.
#[derive(Debug, Default)]
pub(super) struct Shard {
    pub day: String,
    pub rows: Vec<Row>,
    text: String,
    notes: Vec<Box<str>>,
}

/// A frame as read from disk, to go into a row.
#[derive(Debug, Clone, Default)]
pub(super) struct Encoded {
    pub gone: bool,
    /// The sidecar part and its timestamp; None = not read, the row keeps what it has.
    pub side: Option<(Side, String)>,
    pub part: Part,
    pub note: Option<String>,
}

impl Shard {
    pub(super) fn new(day: String) -> Self {
        Self {
            day,
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
    pub(super) fn push(&mut self, stem: &str, json: bool, queued: bool, cand: Cand, e: Encoded) {
        let (side, timestamp) = e.side.unwrap_or_default();
        let note = self.store_note(e.note.as_deref(), 0);
        let text_at = self.store(stem, &timestamp).unwrap_or(u32::MAX);
        self.rows.push(Row {
            text_at,
            timestamp_len: u32::try_from(timestamp.len()).unwrap_or(0),
            cand,
            side,
            part: Part { note, ..e.part },
            stem_len: u8::try_from(stem.len()).unwrap_or(u8::MAX),
            queued,
            json,
            gone: e.gone,
        });
    }

    /// Put what `e` says into row `at`.
    pub(super) fn update(&mut self, at: usize, e: Encoded) {
        let Some(mut row) = self.rows.get(at).copied() else {
            return;
        };
        row.gone = e.gone;
        if !e.gone {
            if let Some((side, timestamp)) = e.side {
                if self.timestamp(&row) != timestamp {
                    let stem = self.stem(&row).to_string();
                    if let Some(text_at) = self.store(&stem, &timestamp) {
                        row.text_at = text_at;
                        row.timestamp_len = u32::try_from(timestamp.len()).unwrap_or(0);
                    }
                }
                row.side = side;
            }
            let note = self.store_note(e.note.as_deref(), row.part.note);
            row.part = Part { note, ..e.part };
        }
        if let Some(slot) = self.rows.get_mut(at) {
            *slot = row;
        }
    }

    /// Bytes of rows and text so far: a quick look during a walk.
    pub(super) fn used(&self) -> usize {
        self.rows.len() * size_of::<Row>() + self.text.len()
    }

    /// Heap bytes, from capacities.
    pub(super) fn bytes(&self) -> usize {
        self.rows.capacity() * size_of::<Row>()
            + self.text.capacity()
            + self.notes.capacity() * size_of::<Box<str>>()
            + self.notes.iter().map(|n| n.len()).sum::<usize>()
            + self.day.capacity()
    }

    /// Give back what a walk overallocated.
    pub(super) fn shrink(&mut self) {
        self.rows.shrink_to_fit();
        self.text.shrink_to_fit();
    }
}

/// What the rows need of the class lists: the filters that ask for them, once per index.
#[derive(Debug)]
pub(super) struct Derive {
    /// One filter per auto filter, in [`AUTOS`]' order, with the index's wanted classes.
    filters: Vec<Filters>,
}

impl Derive {
    pub(super) fn new(wanted: Option<&(Vec<String>, Vec<String>)>) -> Self {
        Self {
            filters: AUTOS
                .iter()
                .map(|a| Filters {
                    auto: Some(*a),
                    wanted: wanted.cloned(),
                    ..Filters::default()
                })
                .collect(),
        }
    }

    fn of(&self, auto: AutoFilter) -> Option<&Filters> {
        self.filters.iter().find(|f| f.auto == Some(auto))
    }

    /// The label part of a frame with `labels` (None = no label file), and its re-label note.
    pub(super) fn part(
        &self,
        labels: Option<&LabelFile>,
        names: &RwLock<Names>,
    ) -> (Part, Option<String>) {
        let u = Uncertainty::default().with_labels(labels);
        let auto = self.filters.iter().enumerate().fold(0u8, |bits, (at, f)| {
            bits | u8::from(f.auto_keeps(labels)) << at
        });
        let mut part = Part {
            strength: self
                .of(AutoFilter::Overlaps)
                .map_or(0, |f| f.overlap_strength(labels)),
            auto_review: box_count(u.auto_review),
            likely: box_count(u.likely_boxes as usize),
            agreed: self
                .of(AutoFilter::Both)
                .map_or(0, |f| box_count(f.agreed_boxes(labels) as usize)),
            auto,
            ..Part::default()
        };
        let Some(l) = labels else {
            return (part, u.relabel);
        };
        part.status = l.status;
        part.has_file = true;
        part.empty_confirmed = l.empty_confirmed;
        part.negatives = box_count(l.hard_negatives().count());
        let verdicts = || l.boxes.iter().filter_map(LabelBox::audit_verdict);
        part.audit_accepted = box_count(verdicts().filter(|v| *v).count());
        part.audit_rejected = box_count(verdicts().filter(|v| !*v).count());
        let classes = queue::file_labels(l);
        let mut training: BTreeMap<&str, u32> = BTreeMap::new();
        for b in l.training() {
            *training.entry(b.label.as_str()).or_default() += 1;
        }
        let mut names = names.write_recover();
        part.classes = names.class_set_of(classes.iter().map(String::as_str));
        let per_class: Vec<(Name, u32)> = training
            .into_iter()
            .map(|(class, n)| (names.names.intern(class), n))
            .collect();
        part.training = names.training.intern(&per_class);
        (part, u.relabel)
    }
}

impl Names {
    /// The id of the class set of `classes`, sorted distinct names.
    fn class_set_of<'a>(&mut self, classes: impl Iterator<Item = &'a str>) -> u32 {
        let set: Vec<Name> = classes.map(|c| self.names.intern(c)).collect();
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

    pub(super) fn side(&self, rec: &Record) -> (Side, String) {
        let u = Uncertainty::of(rec);
        let reply = queue::frame_labels(rec, None);
        let mut names = self.names.write_recover();
        let side = Side {
            model: names.names.intern(&rec.model),
            margin: u.margin.unwrap_or(0.0),
            suppressed: u32::try_from(u.suppressed).unwrap_or(u32::MAX),
            reply: names.class_set_of(reply.iter().map(String::as_str)),
            has_margin: u.margin.is_some(),
            disagreement: u.disagreement,
            ok: true,
        };
        (side, rec.timestamp.clone())
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
            Err(_) => (Side::default(), String::new()),
        });
        let (part, note) = match frame.labels() {
            Ok(labels) => self.derive.part(labels.as_ref(), self.names),
            Err(_) => (
                Part {
                    bad: true,
                    ..Part::default()
                },
                None,
            ),
        };
        Encoded {
            gone: false,
            side,
            part,
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
