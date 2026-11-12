//! Suggestions of the dataset check (`train refine`, a cross-check by fold models that never saw
//! the frame), stored on the label file ([`LabelFile::refine`]) for a reviewer to confirm. Pure. A
//! check writes suggestions only ([`apply`] touches nothing but [`LabelFile::refine`]); a box
//! changes when a reviewer accepts one ([`accept`]), and a dismissal ([`dismiss`]) is remembered so
//! a rerun does not ask again.
//!
//! The kinds ([`Kind`]):
//!
//! - **add**: an object no box labels (`bbox`, `label`). Live while no box of its object that is
//!   not rejected, and no ignore box ([`LabelBox::is_ignore`]), covers the place ([`super::covers`])
//!   and no judged reject of its object lies on it at IoU >= [`MATCH_IOU`]. Unjudged rejects there
//!   ([`super::unjudged_reject`]) are replaced on accept ([`Live::replaces`]), so
//!   `hard_negatives.json` never calls the real object a false positive.
//! - **tighten**: a closer box (`bbox`) for a training box (`target`) no reviewer drew or moved.
//! - **relabel**: another class (`label`) for a training box no reviewer drew or relabelled; never
//!   between two vehicle classes ([`same_object`]), which name one object.
//! - **verify**: a training box no model found; accepting keeps the label ([`Answer::Kept`]).
//!
//! The target of `tighten`, `relabel` and `verify` is never an ignore region
//! ([`LabelBox::is_ignore`]): a box marked unsure (or rejected as too small) is no training box
//! for the check ([`crate::training::refine::judge`] never judges one), so marking it ends the
//! suggestion like any other edit of the box.
//!
//! **Keys.** `target` names an existing box by its exact `bbox`, and by its label too when
//! `target_label` is set (two boxes of different objects can share a `bbox`). A suggestion whose box
//! moved, went or changed so the rule above no longer holds is stale and ignored ([`live`]), never
//! an error. An answered suggestion ([`Suggestion::answer`]) is never live again; one whose box is
//! gone is dropped by the next check ([`apply`]); a kind or an answer this build does not know (a
//! newer build's) loads as `Unknown`, is never live, and stays in the file. A frame holds at most
//! [`MAX_SUGGESTIONS`], strongest first ([`strength`]).
//!
//! The Review tab will state [`live`], [`accept`] and [`dismiss`] in JavaScript: keep them in step.

use super::bulk::Wanted;
use super::repeats::same_object;
use super::stacks;
use super::{
    BoxState, LabelBox, LabelFile, MATCH_IOU, MAX_BOXES, Source, Status, covers, iou,
    rename_to_class, unjudged_reject,
};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeSet;

/// Most suggestions of one frame (like [`super::overlaps::MAX_SUGGESTIONS`]).
pub const MAX_SUGGESTIONS: usize = 64;
/// Longest [`Suggestion::why`], in characters.
pub const MAX_WHY_LEN: usize = 300;
/// IoU at or above which two `add` suggestions of one object are the same place ([`apply`]).
pub const ADD_SAME_IOU: f32 = 0.5;
/// Longest [`Suggestion::spot`], in bytes.
pub const MAX_SPOT_LEN: usize = 200;
/// Longest [`Suggestion::run`], in bytes.
pub const MAX_RUN_LEN: usize = 64;
/// The `reason` of a box an accepted `add` made ([`accept`]).
pub const REFINE_REASON: &str = "refine";

/// What a suggestion proposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// A box for an object no box labels.
    Add,
    /// A closer box for a training box.
    Tighten,
    /// Another class for a training box.
    Relabel,
    /// A training box no model found: a reviewer should look.
    Verify,
    /// A kind this build does not know (a newer build's): never live, kept in the file (written
    /// back as `unknown`).
    #[serde(other)]
    Unknown,
}

impl Kind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Kind::Add => "add",
            Kind::Tighten => "tighten",
            Kind::Relabel => "relabel",
            Kind::Verify => "verify",
            Kind::Unknown => "unknown",
        }
    }

    /// The band of [`strength`]: what a reviewer gains most from first.
    const fn band(self) -> u32 {
        match self {
            Kind::Add => 4,
            Kind::Relabel => 3,
            Kind::Tighten => 2,
            Kind::Verify => 1,
            Kind::Unknown => 0,
        }
    }
}

/// A reviewer's answer to a suggestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Answer {
    /// A `verify` the reviewer confirmed: the label stays.
    Kept,
    /// The reviewer said no: not asked again.
    Dismissed,
    /// An answer this build does not know (a newer build's): answered all the same (written back
    /// as `unknown`).
    #[serde(other)]
    Unknown,
}

/// A suggestion of the dataset check (see the module docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Suggestion {
    pub kind: Kind,
    /// The existing box (`tighten`, `relabel`, `verify`), by its exact `bbox`: the key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<[f32; 4]>,
    /// The label of the box `target` names, as it was when the check wrote it (trimmed,
    /// case-insensitive): with it the key is the `bbox` and the label, so of two boxes of
    /// different objects at the very same place the suggestion names the right one. None (files
    /// of the first checks): the `bbox` alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_label: Option<String>,
    /// The proposed box (`add`, `tighten`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[f32; 4]>,
    /// `add`: the class; `relabel`: the new class.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// The checking model's score, 0..=1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    /// The evidence, in plain words, shown to the reviewer.
    #[serde(default)]
    pub why: String,
    /// The static spot it belongs to, for grouping (opaque here).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spot: Option<String>,
    /// The reviewer's answer: an answered suggestion is never live again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<Answer>,
    /// Id of the check that wrote it.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub run: String,
}

impl Suggestion {
    /// The reviewer answered it ([`Answer`], an unknown answer too).
    pub fn answered(&self) -> bool {
        self.answer.is_some()
    }
}

/// A suggestion that still applies.
#[derive(Debug, Clone, PartialEq)]
pub struct Live<'a> {
    /// Its index in [`LabelFile::refine`].
    pub index: usize,
    pub suggestion: &'a Suggestion,
    /// The box `target` names (`tighten`, `relabel`, `verify`), an index into the frame's boxes.
    pub target: Option<usize>,
    /// `add`: the unjudged rejects of its object at IoU >= [`MATCH_IOU`], closest first (then by
    /// index); accepting puts the new box in the first one's place and removes the others.
    pub replaces: Vec<usize>,
}

/// Finite coordinates of a box with a positive size.
fn proper(b: &[f32; 4]) -> bool {
    b.iter().all(|v| v.is_finite()) && b[2] > b[0] && b[3] > b[1]
}

/// The class `label` names: with a class list, the list's spelling of a wanted class (None for an
/// unwanted label); without one, the label itself (None when blank).
fn class_of(label: &str, wanted: Option<&Wanted<'_>>) -> Option<String> {
    match wanted {
        Some(w) => w.class_of(label).map(str::to_string),
        None => {
            let t = label.trim();
            (!t.is_empty()).then(|| t.to_string())
        }
    }
}

/// The class of `b` as [`class_of`] names it, its own label when that is unwanted.
fn box_class(b: &LabelBox, wanted: Option<&Wanted<'_>>) -> String {
    class_of(&b.label, wanted).unwrap_or_else(|| b.label.trim().to_string())
}

/// Two class names of one object ([`same_object`], case-insensitive).
fn one_object(a: &str, b: &str) -> bool {
    same_object(
        &a.trim().to_ascii_lowercase(),
        &b.trim().to_ascii_lowercase(),
    )
}

/// Two labels written alike: trimmed, case-insensitive.
fn same_label(a: &str, b: &str) -> bool {
    a.trim().eq_ignore_ascii_case(b.trim())
}

/// Whether `b` is the box `s` names: its exact `bbox`, and its label when `s` has one
/// ([`Suggestion::target_label`]). False without a `target`.
fn names(s: &Suggestion, b: &LabelBox) -> bool {
    s.target.is_some_and(|t| b.bbox == t)
        && s.target_label
            .as_deref()
            .is_none_or(|l| same_label(&b.label, l))
}

/// The first box `s` names ([`names`]) that is a training box, no ignore region
/// ([`LabelBox::is_ignore`]: an unsure box, which the check never judges), no reviewer drew and
/// that `ok` takes.
fn target_box(f: &LabelFile, s: &Suggestion, ok: impl Fn(&LabelBox) -> bool) -> Option<usize> {
    f.boxes.iter().position(|b| {
        names(s, b) && b.is_training() && !b.is_ignore() && b.source != Source::Human && ok(b)
    })
}

/// The kinds that name an existing box by `target`.
const fn has_target(k: Kind) -> bool {
    matches!(k, Kind::Tighten | Kind::Relabel | Kind::Verify)
}

/// Where an `add` goes: None when it is not live, else the rejects it replaces ([`Live::replaces`]).
fn add_place(f: &LabelFile, s: &Suggestion, wanted: Option<&Wanted<'_>>) -> Option<Vec<usize>> {
    let bbox = s.bbox.filter(proper)?;
    let class = class_of(s.label.as_deref()?, wanted)?;
    let mut replaces: Vec<(usize, f32)> = Vec::new();
    for (i, b) in f.boxes.iter().enumerate() {
        if !one_object(&box_class(b, wanted), &class) {
            continue;
        }
        if b.state != BoxState::Rejected || b.is_ignore() {
            if covers(&b.bbox, &bbox) {
                return None;
            }
            continue;
        }
        let v = iou(&b.bbox, &bbox);
        if v >= MATCH_IOU {
            if !unjudged_reject(b) {
                return None;
            }
            replaces.push((i, v));
        }
    }
    if replaces.is_empty() && f.boxes.len() >= MAX_BOXES {
        return None;
    }
    replaces.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    Some(replaces.into_iter().map(|(i, _)| i).collect())
}

/// `s` on `f` now: None when it is not live, else ([`Live::target`], [`Live::replaces`]).
fn place(
    f: &LabelFile,
    s: &Suggestion,
    wanted: Option<&Wanted<'_>>,
) -> Option<(Option<usize>, Vec<usize>)> {
    if s.answered() {
        return None;
    }
    match s.kind {
        Kind::Add => add_place(f, s, wanted).map(|r| (None, r)),
        Kind::Tighten => {
            let bbox = s.bbox.filter(proper)?;
            target_box(f, s, |b| b.orig_bbox.is_none() && b.bbox != bbox).map(|t| (Some(t), vec![]))
        }
        Kind::Relabel => {
            let to = class_of(s.label.as_deref()?, wanted)?;
            target_box(f, s, |b| {
                b.model_label.is_none() && !one_object(&box_class(b, wanted), &to)
            })
            .map(|t| (Some(t), vec![]))
        }
        Kind::Verify => target_box(f, s, |b| b.orig_bbox.is_none()).map(|t| (Some(t), vec![])),
        Kind::Unknown => None,
    }
}

/// The suggestions of `f` a reviewer can act on (see the module docs): none on a skipped frame,
/// never an answered one or one of an unknown kind. `wanted` None: every label wanted, compared as
/// written (trimmed, case-insensitive). Computed at read time, never an error.
pub fn live<'a>(f: &'a LabelFile, wanted: Option<&Wanted<'_>>) -> Vec<Live<'a>> {
    if f.status == Status::Skipped || f.refine.is_empty() {
        return Vec::new();
    }
    f.refine
        .iter()
        .enumerate()
        .filter_map(|(index, s)| {
            let (target, replaces) = place(f, s, wanted)?;
            Some(Live {
                index,
                suggestion: s,
                target,
                replaces,
            })
        })
        .collect()
}

/// How strong a suggestion is, higher first; 0 for an answered one or one of an unknown kind. The
/// band by kind (add 4, relabel 3, tighten 2, verify 1), then the score:
/// `band * 1000 + round(score * 999)`.
pub fn strength(s: &Suggestion) -> u32 {
    let band = s.kind.band();
    if band == 0 || s.answered() {
        return 0;
    }
    let score = s
        .score
        .filter(|v| v.is_finite())
        .map_or(0.0, |v| (f64::from(v).clamp(0.0, 1.0) * 999.0).round());
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a whole number in 0..=999"
    )]
    let score = score as u32;
    band * 1000 + score
}

/// The strength of the strongest [`live`] suggestion of `f`, 0 for none.
pub fn frame_strength(f: &LabelFile, wanted: Option<&Wanted<'_>>) -> u32 {
    live(f, wanted)
        .iter()
        .map(|l| strength(l.suggestion))
        .max()
        .unwrap_or(0)
}

/// How many suggestions of `f` are [`live`].
pub fn open_count(f: &LabelFile, wanted: Option<&Wanted<'_>>) -> u32 {
    u32::try_from(live(f, wanted).len()).unwrap_or(u32::MAX)
}

/// Whether the box at `at` is in a stack or an open pair of `f` ([`stacks`]): a reviewer's earlier
/// look at the frame's overlaps did not cover it. `wanted` None: every label wanted.
fn in_open_pair(f: &LabelFile, at: usize, wanted: Option<&Wanted<'_>>) -> bool {
    stacks::groups(&f.boxes, wanted)
        .iter()
        .any(|s| s.members.contains(&at))
        || stacks::overlap_pairs(&f.boxes, wanted)
            .iter()
            .any(|&(i, j)| i == at || j == at)
}

/// What the reviewer's yes does to suggestion `index` of `f`; false, changing nothing, when it is
/// not [`live`]:
///
/// - add: a new `source: auto` box, accepted, with the suggestion's class (the list's spelling),
///   box and score and `reason` [`REFINE_REASON`], in the place of the first of [`Live::replaces`]
///   (the others removed, the other boxes in their order) or at the end; the frame is no longer
///   confirmed empty, and [`LabelFile::overlaps_checked`] ends when the new box is in a stack or an
///   open pair; the suggestion goes;
/// - tighten: `orig_bbox` keeps the old box (unless set), `bbox` becomes the proposal (clamping is
///   the caller's: no frame size here); `overlaps_checked` ends when the box is now in a stack or
///   an open pair; the suggestion goes;
/// - relabel: the new class, the old label kept in `model_label`; `overlaps_checked` ends as for
///   a tighten; the suggestion goes;
/// - verify: answered [`Answer::Kept`] and kept, so a rerun does not ask again.
pub fn accept(f: &mut LabelFile, index: usize, wanted: Option<&Wanted<'_>>) -> bool {
    let Some((target, replaces, s)) = live(f, wanted)
        .into_iter()
        .find(|l| l.index == index)
        .map(|l| (l.target, l.replaces, l.suggestion.clone()))
    else {
        return false;
    };
    match s.kind {
        Kind::Add => {
            let (Some(bbox), Some(label)) = (s.bbox, s.label.as_deref()) else {
                return false;
            };
            let Some(class) = class_of(label, wanted) else {
                return false;
            };
            let new = LabelBox {
                label: class,
                bbox,
                source: Source::Auto,
                score: s.score,
                state: BoxState::Accepted,
                origin: None,
                reason: Some(REFINE_REASON.to_string()),
                model_label: None,
                orig_bbox: None,
                difficult: false,
                reject_reason: None,
                auto: super::AutoInfo::default(),
            };
            let mut at = match replaces
                .first()
                .and_then(|&i| f.boxes.get_mut(i).map(|b| (i, b)))
            {
                Some((i, slot)) => {
                    *slot = new;
                    i
                }
                None => {
                    f.boxes.push(new);
                    f.boxes.len() - 1
                }
            };
            let gone: BTreeSet<usize> = replaces.iter().skip(1).copied().collect();
            if !gone.is_empty() {
                let mut i = 0;
                f.boxes.retain(|_| {
                    i += 1;
                    !gone.contains(&(i - 1))
                });
                let before = gone.iter().filter(|&&g| g < at).count();
                at -= before;
            }
            f.empty_confirmed = false;
            if f.overlaps_checked && in_open_pair(f, at, wanted) {
                f.overlaps_checked = false;
            }
        }
        Kind::Tighten => {
            let (Some(bbox), Some(t)) = (s.bbox, target) else {
                return false;
            };
            let Some(b) = f.boxes.get_mut(t) else {
                return false;
            };
            if b.orig_bbox.is_none() {
                b.orig_bbox = Some(b.bbox);
            }
            b.bbox = bbox;
            // The new box may now overlap another nobody looked at.
            if f.overlaps_checked && in_open_pair(f, t, wanted) {
                f.overlaps_checked = false;
            }
        }
        Kind::Relabel => {
            let Some(to) = s.label.as_deref().and_then(|l| class_of(l, wanted)) else {
                return false;
            };
            let Some(t) = target else {
                return false;
            };
            let Some(b) = f.boxes.get_mut(t) else {
                return false;
            };
            rename_to_class(b, &to);
            // Of another object now: it may stack with a box of that object.
            if f.overlaps_checked && in_open_pair(f, t, wanted) {
                f.overlaps_checked = false;
            }
        }
        Kind::Verify => {
            return match f.refine.get_mut(index) {
                Some(x) => {
                    x.answer = Some(Answer::Kept);
                    true
                }
                None => false,
            };
        }
        Kind::Unknown => return false,
    }
    if index < f.refine.len() {
        f.refine.remove(index);
    }
    true
}

/// The reviewer's no to suggestion `index`: answered [`Answer::Dismissed`], so a rerun does not ask
/// again. Never changes a box. False when there is no such suggestion or it is answered already.
pub fn dismiss(f: &mut LabelFile, index: usize) -> bool {
    match f.refine.get_mut(index) {
        Some(s) if !s.answered() => {
            s.answer = Some(Answer::Dismissed);
            true
        }
        _ => false,
    }
}

/// What [`apply`] did to a frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Applied {
    /// Found suggestions the file did not hold before (ignoring `run`), written now.
    pub written: usize,
    /// Answered suggestions in the file after the call.
    pub kept_answers: usize,
    /// Found suggestions not written: of a kind outside `kinds`, carrying an answer, the same as an
    /// answered one, a weaker copy of another found one, or past [`MAX_SUGGESTIONS`].
    pub dropped: usize,
    /// Suggestions the file held that went: unanswered ones of `kinds` not found again (ignoring
    /// `run`), answered ones of `kinds` whose box is gone, or past [`MAX_SUGGESTIONS`].
    pub removed: usize,
}

impl Applied {
    /// Whether the frame's suggestions changed: only then is it written.
    pub fn changed(&self) -> bool {
        self.written + self.removed > 0
    }
}

/// Two suggestions of one finding: the same kind and the same exact `target`, and the same
/// `target_label` when both have one (a suggestion without one, of the first checks, names the
/// place alone); for `add`, boxes of one object ([`same_object`]) at IoU >= [`ADD_SAME_IOU`].
pub fn same_finding(a: &Suggestion, b: &Suggestion) -> bool {
    if a.kind != b.kind {
        return false;
    }
    match a.kind {
        Kind::Add => match (a.bbox, b.bbox, a.label.as_deref(), b.label.as_deref()) {
            (Some(x), Some(y), Some(l), Some(m)) => one_object(l, m) && iou(&x, &y) >= ADD_SAME_IOU,
            _ => false,
        },
        _ => {
            a.target.is_some()
                && a.target == b.target
                && match (a.target_label.as_deref(), b.target_label.as_deref()) {
                    (Some(x), Some(y)) => same_label(x, y),
                    _ => true,
                }
        }
    }
}

/// Equal but for `run`; numbers by their bits, so a value equals itself whatever it is.
fn same_but_run(a: &Suggestion, b: &Suggestion) -> bool {
    let bits = |v: &Option<[f32; 4]>| v.map(|x| x.map(f32::to_bits));
    a.kind == b.kind
        && bits(&a.target) == bits(&b.target)
        && a.target_label == b.target_label
        && bits(&a.bbox) == bits(&b.bbox)
        && a.label == b.label
        && a.score.map(f32::to_bits) == b.score.map(f32::to_bits)
        && a.why == b.why
        && a.spot == b.spot
        && a.answer == b.answer
}

fn score_of(s: &Suggestion) -> f32 {
    s.score.filter(|v| v.is_finite()).unwrap_or(0.0)
}

fn cmp_box(a: &Option<[f32; 4]>, b: &Option<[f32; 4]>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(x), Some(y)) => x
            .iter()
            .zip(y)
            .map(|(p, q)| p.total_cmp(q))
            .find(|o| o.is_ne())
            .unwrap_or(Ordering::Equal),
    }
}

/// The file's order: strongest first, then by content (so a rerun that finds the same things in
/// another order writes the same list).
fn order(a: &Suggestion, b: &Suggestion) -> Ordering {
    strength(b)
        .cmp(&strength(a))
        .then_with(|| a.answered().cmp(&b.answered()))
        .then_with(|| a.kind.cmp(&b.kind))
        .then_with(|| cmp_box(&a.target, &b.target))
        .then_with(|| a.target_label.cmp(&b.target_label))
        .then_with(|| cmp_box(&a.bbox, &b.bbox))
        .then_with(|| a.label.cmp(&b.label))
        .then_with(|| a.answer.cmp(&b.answer))
        .then_with(|| a.spot.cmp(&b.spot))
        .then_with(|| a.why.cmp(&b.why))
        .then_with(|| a.run.cmp(&b.run))
}

/// Write what a check found on `f` (run under the label write lock: a reviewer may be working on
/// the frame), and nothing else: only [`LabelFile::refine`] changes. The unanswered suggestions of
/// `kinds` are replaced by `found`; answered suggestions and those of other kinds stay. A found
/// suggestion is dropped when it is of a kind outside `kinds`, carries an answer (only a reviewer
/// answers), is the same finding as an answered one
/// ([`same_finding`]: the reviewer answered it), the same as a stronger found one, or outside the
/// bounds a save checks ([`check`]: so a check never writes a file a Review save would refuse). An
/// answered suggestion of `kinds` whose `target` box is gone (moved, relabelled away, deleted) goes:
/// it can never be live again and would only take room; an answered `add` stays, it is what keeps
/// the question from being asked again. The list is sorted strongest first and holds at most
/// [`MAX_SUGGESTIONS`], answered ones kept first. When the result equals the old list ignoring
/// `run` (in any order), the file is left exactly as it was and [`Applied::changed`] is false: a
/// rerun that finds the same things writes nothing, and an open Review page keeps its revision.
pub fn apply(f: &mut LabelFile, found: Vec<Suggestion>, kinds: &[Kind]) -> Applied {
    let mut done = Applied::default();
    let target_gone = |s: &Suggestion| {
        s.answered()
            && has_target(s.kind)
            && kinds.contains(&s.kind)
            && !f.boxes.iter().any(|b| names(s, b))
    };
    let kept: Vec<Suggestion> = f
        .refine
        .iter()
        .filter(|s| (s.answered() || !kinds.contains(&s.kind)) && !target_gone(s))
        .cloned()
        .collect();
    let mut found = found;
    found.sort_by(|a, b| score_of(b).total_cmp(&score_of(a)));
    let mut fresh: Vec<Suggestion> = Vec::new();
    for s in found {
        if !kinds.contains(&s.kind)
            || s.answered()
            || check_one(0, &s).is_err()
            || kept.iter().any(|k| k.answered() && same_finding(k, &s))
            || fresh.iter().any(|k| same_finding(k, &s))
        {
            done.dropped += 1;
            continue;
        }
        fresh.push(s);
    }
    let mut list: Vec<(bool, Suggestion)> = kept
        .into_iter()
        .map(|s| (false, s))
        .chain(fresh.into_iter().map(|s| (true, s)))
        .collect();
    if list.len() > MAX_SUGGESTIONS {
        list.sort_by(|a, b| {
            b.1.answered()
                .cmp(&a.1.answered())
                .then_with(|| order(&a.1, &b.1))
        });
        done.dropped += list
            .drain(MAX_SUGGESTIONS..)
            .filter(|(is_fresh, _)| *is_fresh)
            .count();
    }
    list.sort_by(|a, b| order(&a.1, &b.1));
    let list: Vec<Suggestion> = list.into_iter().map(|(_, s)| s).collect();
    let mut old: Vec<&Suggestion> = f.refine.iter().collect();
    for s in &list {
        match old.iter().position(|o| same_but_run(o, s)) {
            Some(p) => {
                old.swap_remove(p);
            }
            None => done.written += 1,
        }
    }
    done.removed = old.len();
    done.kept_answers = list.iter().filter(|s| s.answered()).count();
    if done.changed() {
        f.refine = list;
    }
    done
}

/// Bounds of the suggestions a client sends back (they are only round-tripped): [`super::normalize`].
pub fn check(list: &[Suggestion]) -> Result<()> {
    if list.len() > MAX_SUGGESTIONS {
        bail!(
            "{} refine suggestions; at most {MAX_SUGGESTIONS} per frame",
            list.len()
        );
    }
    for (i, s) in list.iter().enumerate() {
        check_one(i + 1, s)?;
    }
    Ok(())
}

/// The bounds of [`check`] on one suggestion, the `n`-th of its list (1-based, for the message).
fn check_one(n: usize, s: &Suggestion) -> Result<()> {
    let text_ok = |t: &str, max: usize| t.len() <= max && !t.chars().any(char::is_control);
    if !s
        .target
        .iter()
        .chain(&s.bbox)
        .flatten()
        .all(|v| v.is_finite())
    {
        bail!("refine suggestion {n}: the coordinates are not finite numbers");
    }
    if s.why.chars().count() > MAX_WHY_LEN || s.why.chars().any(char::is_control) {
        bail!(
            "refine suggestion {n}: the reason is longer than {MAX_WHY_LEN} characters or holds \
             control characters"
        );
    }
    for l in s.label.iter().chain(&s.target_label) {
        if !super::label_ok(l) {
            bail!(
                "refine suggestion {n}: '{}' is not a valid label (1 to {} characters)",
                l.escape_debug(),
                super::MAX_LABEL_LEN
            );
        }
    }
    if let Some(v) = s.score
        && !(v.is_finite() && (0.0..=1.0).contains(&v))
    {
        bail!("refine suggestion {n}: score {v} is not between 0 and 1");
    }
    if s.spot.as_deref().is_some_and(|t| !text_ok(t, MAX_SPOT_LEN)) {
        bail!(
            "refine suggestion {n}: the spot is longer than {MAX_SPOT_LEN} bytes or holds control \
             characters"
        );
    }
    if !text_ok(&s.run, MAX_RUN_LEN) {
        bail!(
            "refine suggestion {n}: the run id is longer than {MAX_RUN_LEN} bytes or holds control \
             characters"
        );
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::tests_support::auto_box;
    use super::super::{Origin, RejectReason, Triage, normalize};
    use super::*;

    fn classes() -> Vec<String> {
        ["person", "car", "truck", "motorcycle", "dog"]
            .map(String::from)
            .to_vec()
    }

    fn wanted(c: &[String]) -> Wanted<'_> {
        Wanted {
            classes: c,
            hidden: &[],
        }
    }

    const CAR: [f32; 4] = [100.0, 100.0, 200.0, 180.0];
    /// IoU with [`CAR`] about 0.92.
    const NEAR: [f32; 4] = [102.0, 101.0, 201.0, 182.0];
    const TIGHT: [f32; 4] = [110.0, 105.0, 195.0, 175.0];
    const MAN: [f32; 4] = [300.0, 100.0, 340.0, 200.0];
    const FAR: [f32; 4] = [400.0, 300.0, 450.0, 400.0];

    /// A pending auto box.
    fn at(label: &str, bbox: [f32; 4]) -> LabelBox {
        LabelBox {
            bbox,
            ..auto_box(label)
        }
    }

    fn accepted(label: &str, bbox: [f32; 4]) -> LabelBox {
        LabelBox {
            state: BoxState::Accepted,
            ..at(label, bbox)
        }
    }

    fn drawn(label: &str, bbox: [f32; 4]) -> LabelBox {
        LabelBox {
            source: Source::Human,
            state: BoxState::Added,
            score: None,
            ..at(label, bbox)
        }
    }

    /// A rejected box with provenance `reason` and the reviewer's `judged` reason.
    fn rejected(
        label: &str,
        bbox: [f32; 4],
        reason: Option<&str>,
        judged: Option<RejectReason>,
    ) -> LabelBox {
        LabelBox {
            state: BoxState::Rejected,
            reason: reason.map(str::to_string),
            reject_reason: judged,
            ..at(label, bbox)
        }
    }

    fn sug(kind: Kind) -> Suggestion {
        Suggestion {
            kind,
            target: None,
            target_label: None,
            bbox: None,
            label: None,
            score: Some(0.8),
            why: "a fold model that never saw the frame found it at 0.80".into(),
            spot: None,
            answer: None,
            run: "r1".into(),
        }
    }

    fn add(label: &str, bbox: [f32; 4], score: f32) -> Suggestion {
        Suggestion {
            bbox: Some(bbox),
            label: Some(label.into()),
            score: Some(score),
            ..sug(Kind::Add)
        }
    }

    fn tighten(target: [f32; 4], bbox: [f32; 4]) -> Suggestion {
        Suggestion {
            target: Some(target),
            bbox: Some(bbox),
            ..sug(Kind::Tighten)
        }
    }

    fn relabel(target: [f32; 4], label: &str) -> Suggestion {
        Suggestion {
            target: Some(target),
            label: Some(label.into()),
            ..sug(Kind::Relabel)
        }
    }

    fn verify(target: [f32; 4]) -> Suggestion {
        Suggestion {
            target: Some(target),
            score: None,
            ..sug(Kind::Verify)
        }
    }

    fn answered(s: Suggestion, a: Answer) -> Suggestion {
        Suggestion {
            answer: Some(a),
            ..s
        }
    }

    fn file(boxes: Vec<LabelBox>, refine: Vec<Suggestion>) -> LabelFile {
        LabelFile {
            status: Status::Reviewed,
            boxes,
            refine,
            ..LabelFile::default()
        }
    }

    /// The indices of the live suggestions of `f`.
    fn live_at(f: &LabelFile, w: Option<&Wanted<'_>>) -> Vec<usize> {
        live(f, w).iter().map(|l| l.index).collect()
    }

    fn json(f: &LabelFile) -> String {
        serde_json::to_string_pretty(f).unwrap()
    }

    #[test]
    fn each_kind_is_live_on_its_happy_path() {
        let c = classes();
        let w = wanted(&c);
        let f = file(
            vec![accepted("car", CAR), accepted("person", MAN)],
            vec![
                add("dog", FAR, 0.9),
                tighten(CAR, TIGHT),
                relabel(MAN, "dog"),
                verify(CAR),
            ],
        );
        for wanted in [Some(&w), None] {
            let l = live(&f, wanted);
            let got: Vec<(usize, Option<usize>, Vec<usize>)> = l
                .iter()
                .map(|l| (l.index, l.target, l.replaces.clone()))
                .collect();
            assert_eq!(
                got,
                [
                    (0, None, vec![]),
                    (1, Some(0), vec![]),
                    (2, Some(1), vec![]),
                    (3, Some(0), vec![]),
                ]
            );
            assert_eq!(open_count(&f, wanted), 4);
            assert_eq!(frame_strength(&f, wanted), 4000 + 899);
        }
        // A pending box is no training box: nothing to tighten, relabel or verify.
        let mut pending = f.clone();
        pending.boxes = vec![at("car", CAR), at("person", MAN)];
        assert_eq!(live_at(&pending, Some(&w)), [0]);
    }

    #[test]
    fn an_add_is_covered_by_a_box_of_its_object_or_an_ignore_box() {
        let c = classes();
        let w = wanted(&c);
        let one = |boxes: Vec<LabelBox>, s: Suggestion| live_at(&file(boxes, vec![s]), Some(&w));
        // A training box, a pending one, a box of another vehicle class.
        assert!(one(vec![accepted("car", CAR)], add("car", NEAR, 0.9)).is_empty());
        assert!(one(vec![at("car", CAR)], add("car", NEAR, 0.9)).is_empty());
        assert!(one(vec![accepted("truck", CAR)], add("car", NEAR, 0.9)).is_empty());
        assert!(one(vec![accepted("Car", CAR)], add("car", NEAR, 0.9)).is_empty());
        // Another object at the same place covers nothing.
        assert_eq!(
            one(vec![accepted("person", CAR)], add("car", NEAR, 0.9)),
            [0]
        );
        // The detection mostly inside a larger box: covered at a low IoU.
        let inner = [110.0, 110.0, 140.0, 140.0];
        assert!(iou(&CAR, &inner) < COVER_IOU_FOR_TESTS);
        assert!(one(vec![accepted("car", CAR)], add("car", inner, 0.9)).is_empty());
        // Not the other way: a small box inside a large detection does not cover it.
        assert_eq!(one(vec![accepted("car", inner)], add("car", CAR, 0.9)), [0]);
        // An ignore box: rejected as too small, or unsure.
        let small = rejected("car", CAR, None, Some(RejectReason::TooSmall));
        assert!(small.is_ignore());
        assert!(one(vec![small], add("car", NEAR, 0.9)).is_empty());
        let unsure = LabelBox {
            difficult: true,
            ..accepted("car", CAR)
        };
        assert!(one(vec![unsure], add("car", NEAR, 0.9)).is_empty());
        // A plain judged reject does not cover (it says nothing about a nearby object) but blocks
        // the same place.
        let fp = rejected("car", CAR, None, Some(RejectReason::FalsePositive));
        assert!(one(vec![fp.clone()], add("car", NEAR, 0.9)).is_empty());
        let beside = [150.0, 100.0, 250.0, 180.0];
        assert!(iou(&CAR, &beside) < MATCH_IOU && iou(&CAR, &beside) >= COVER_IOU_FOR_TESTS);
        assert_eq!(one(vec![fp], add("car", beside, 0.9)), [0]);
        // A duplicate reject and a spot edit are judged too.
        let dup = rejected("car", CAR, Some(stacks::DUPLICATE_REASON), None);
        assert!(one(vec![dup], add("car", NEAR, 0.9)).is_empty());
        let spot = rejected(
            "car",
            CAR,
            Some(super::super::repeats::REPEAT_SPOT_REASON),
            None,
        );
        assert!(one(vec![spot], add("car", NEAR, 0.9)).is_empty());
        // An unwanted or blank class, no box: never live.
        assert!(one(vec![], add("zebra", FAR, 0.9)).is_empty());
        let mut no_box = add("car", FAR, 0.9);
        no_box.bbox = None;
        assert!(one(vec![], no_box).is_empty());
        let mut flat = add("car", FAR, 0.9);
        flat.bbox = Some([10.0, 10.0, 10.0, 40.0]);
        assert!(one(vec![], flat).is_empty());
        let hidden = ["dog".to_string()];
        let w2 = Wanted {
            classes: &c,
            hidden: &hidden,
        };
        assert!(live(&file(vec![], vec![add("dog", FAR, 0.9)]), Some(&w2)).is_empty());
        // Without a class list any label is wanted.
        assert_eq!(
            live_at(&file(vec![], vec![add("zebra", FAR, 0.9)]), None),
            [0]
        );
    }

    const COVER_IOU_FOR_TESTS: f32 = super::super::COVER_IOU;

    #[test]
    fn an_add_replaces_the_unjudged_rejects_of_its_place_closest_first() {
        let c = classes();
        let w = wanted(&c);
        let low = rejected("car", CAR, Some(super::super::bulk::LOW_SCORE_REASON), None);
        let gone = rejected(
            "car",
            NEAR,
            Some(super::super::bulk::DISMISSED_REASON),
            None,
        );
        let hint = LabelBox {
            source: Source::Model,
            origin: Some(Origin::BelowThreshold),
            ..rejected("car", [101.0, 100.0, 200.0, 181.0], None, None)
        };
        for b in [&low, &gone, &hint] {
            assert!(unjudged_reject(b), "{b:?}");
        }
        let f = file(
            vec![
                low.clone(),
                accepted("person", MAN),
                gone.clone(),
                hint.clone(),
            ],
            vec![add("car", NEAR, 0.9)],
        );
        let l = live(&f, Some(&w));
        assert_eq!(l.len(), 1);
        // NEAR itself (IoU 1), then the hint, then the low-score reject.
        assert_eq!(l[0].replaces, [2, 3, 0]);
        // A judged reject there as well: blocked, whatever else is there.
        let mut judged = f.clone();
        judged
            .boxes
            .push(rejected("truck", CAR, None, Some(RejectReason::Reflection)));
        assert!(live(&judged, Some(&w)).is_empty());
        // A reject of another object at that place is neither.
        let mut other = f.clone();
        other.boxes.push(rejected(
            "person",
            CAR,
            None,
            Some(RejectReason::Reflection),
        ));
        assert_eq!(live(&other, Some(&w))[0].replaces, [2, 3, 0]);
    }

    #[test]
    fn tighten_verify_and_relabel_keep_off_a_reviewers_boxes() {
        let c = classes();
        let w = wanted(&c);
        let all = |b: LabelBox| {
            let target = b.bbox;
            let f = file(
                vec![b],
                vec![
                    tighten(target, TIGHT),
                    verify(target),
                    relabel(target, "dog"),
                ],
            );
            live_at(&f, Some(&w))
        };
        assert_eq!(all(accepted("car", CAR)), [0, 1, 2]);
        // Drawn by the reviewer: none of them.
        assert!(all(drawn("car", CAR)).is_empty());
        // Moved by the reviewer: no tighten, no verify; another class is still a question.
        let moved = LabelBox {
            orig_bbox: Some([90.0, 90.0, 190.0, 170.0]),
            ..accepted("car", CAR)
        };
        assert_eq!(all(moved), [2]);
        // Relabelled by the reviewer (or mapped by a labelers run): no relabel.
        let relabelled = LabelBox {
            model_label: Some("truck".into()),
            ..accepted("car", CAR)
        };
        assert_eq!(all(relabelled), [0, 1]);
        // Rejected or pending: no training box.
        assert!(all(rejected("car", CAR, None, None)).is_empty());
        assert!(all(at("car", CAR)).is_empty());
        // An ignore region: marked unsure (accepted or added, any source) or rejected as too
        // small. The check never judges one, so marking it ends every suggestion about it.
        let unsure = LabelBox {
            difficult: true,
            ..accepted("car", CAR)
        };
        assert!(unsure.is_training() && unsure.is_ignore());
        assert!(all(unsure.clone()).is_empty());
        let model_unsure = LabelBox {
            source: Source::Model,
            origin: Some(Origin::Prediction),
            ..unsure.clone()
        };
        assert!(all(model_unsure).is_empty());
        assert!(all(rejected("car", CAR, None, Some(RejectReason::TooSmall))).is_empty());
        // ... and no accept goes through: nothing changes.
        let mut f = file(
            vec![unsure.clone()],
            vec![tighten(CAR, TIGHT), verify(CAR), relabel(CAR, "dog")],
        );
        let before = f.clone();
        for i in 0..3 {
            assert!(!accept(&mut f, i, Some(&w)), "{i}");
        }
        assert_eq!(f, before);
        // Unsure taken back: the suggestions are live again (none was answered).
        f.boxes[0].difficult = false;
        assert_eq!(live_at(&f, Some(&w)), [0, 1, 2]);
        // A tighten to the same box, or with no box, proposes nothing.
        let f = file(
            vec![accepted("car", CAR)],
            vec![tighten(CAR, CAR), {
                let mut t = tighten(CAR, TIGHT);
                t.bbox = None;
                t
            }],
        );
        assert!(live(&f, Some(&w)).is_empty());
        // A model box is not a reviewer's: it may be tightened.
        let model = LabelBox {
            source: Source::Model,
            origin: Some(Origin::Prediction),
            ..accepted("car", CAR)
        };
        assert_eq!(all(model), [0, 1, 2]);
    }

    #[test]
    fn relabel_needs_another_object_of_a_wanted_class() {
        let c = classes();
        let w = wanted(&c);
        let one = |to: &str, w: Option<&Wanted<'_>>| {
            live_at(&file(vec![accepted("car", CAR)], vec![relabel(CAR, to)]), w)
        };
        assert_eq!(one("person", Some(&w)), [0]);
        // Two vehicle classes name one object; the same class is no change.
        assert!(one("truck", Some(&w)).is_empty());
        assert!(one("car", Some(&w)).is_empty());
        assert!(one("Car", Some(&w)).is_empty());
        assert!(one("Truck", None).is_empty());
        // Not a class of the list, or a hidden one.
        assert!(one("zebra", Some(&w)).is_empty());
        assert_eq!(one("zebra", None), [0]);
        let hidden = ["person".to_string()];
        let w2 = Wanted {
            classes: &c,
            hidden: &hidden,
        };
        assert!(one("person", Some(&w2)).is_empty());
        // No label at all.
        let mut s = relabel(CAR, "person");
        s.label = None;
        assert!(live(&file(vec![accepted("car", CAR)], vec![s]), Some(&w)).is_empty());
    }

    #[test]
    fn stale_answered_unknown_and_skipped_are_not_live_and_no_error() {
        let c = classes();
        let w = wanted(&c);
        let f = file(
            vec![accepted("car", CAR)],
            vec![tighten(CAR, TIGHT), verify(CAR), relabel(CAR, "person")],
        );
        assert_eq!(live_at(&f, Some(&w)), [0, 1, 2]);
        // The box moved: the key names no box.
        let mut moved = f.clone();
        moved.boxes[0].bbox = [101.0, 100.0, 200.0, 180.0];
        assert!(live(&moved, Some(&w)).is_empty());
        // The box went.
        let mut gone = f.clone();
        gone.boxes.clear();
        assert!(live(&gone, Some(&w)).is_empty());
        // Skipped frame.
        let skipped = LabelFile {
            status: Status::Skipped,
            ..f.clone()
        };
        assert!(live(&skipped, Some(&w)).is_empty());
        assert_eq!(frame_strength(&skipped, Some(&w)), 0);
        // An unreviewed frame is checked like a reviewed one.
        let unreviewed = LabelFile {
            status: Status::Unreviewed,
            ..f.clone()
        };
        assert_eq!(open_count(&unreviewed, Some(&w)), 3);
        // Answered, or of a kind this build does not know.
        let mut done = f.clone();
        done.refine[0].answer = Some(Answer::Dismissed);
        done.refine[1].answer = Some(Answer::Kept);
        done.refine[2].kind = Kind::Unknown;
        assert!(live(&done, Some(&w)).is_empty());
        assert!(done.refine.iter().all(|s| strength(s) == 0));
    }

    #[test]
    fn unknown_kinds_and_answers_load_are_never_live_and_stay_in_the_file() {
        let c = classes();
        let w = wanted(&c);
        let s: Suggestion =
            serde_json::from_str(r#"{"kind":"split","target":[100,100,200,180],"why":"x"}"#)
                .unwrap();
        assert_eq!(s.kind, Kind::Unknown);
        let v: Suggestion = serde_json::from_str(
            r#"{"kind":"verify","target":[100,100,200,180],"answer":"later","run":"r0"}"#,
        )
        .unwrap();
        assert_eq!(v.answer, Some(Answer::Unknown));
        assert!(v.answered());
        let f = file(vec![accepted("car", CAR)], vec![s.clone(), v.clone()]);
        assert!(live(&f, Some(&w)).is_empty());
        assert_eq!((strength(&s), strength(&v)), (0, 0));
        // A save keeps both, and they are written back.
        let n = normalize(f.clone(), &c, 640, 480, "now").unwrap();
        assert_eq!(n.refine, [s.clone(), v.clone()]);
        let text = json(&n);
        assert!(text.contains("\"kind\": \"unknown\""), "{text}");
        assert!(text.contains("\"answer\": \"unknown\""), "{text}");
        let back: LabelFile = serde_json::from_str(&text).unwrap();
        assert_eq!(back.refine, n.refine);
        // A check of other kinds keeps them; one of every known kind keeps the answered one and
        // the unknown kind too.
        let mut g = f.clone();
        let a = apply(
            &mut g,
            vec![add("dog", FAR, 0.7)],
            &[Kind::Add, Kind::Tighten, Kind::Relabel, Kind::Verify],
        );
        assert_eq!((a.written, a.kept_answers, a.removed), (1, 1, 0));
        assert!(g.refine.contains(&s) && g.refine.contains(&v));
        // A dismissal of an answered one does nothing, an accept neither.
        let before = g.clone();
        let i = g.refine.iter().position(|x| *x == v).unwrap();
        assert!(!dismiss(&mut g, i));
        assert!(!accept(&mut g, i, Some(&w)));
        assert_eq!(g, before);
    }

    #[test]
    fn a_file_without_suggestions_serialises_as_before() {
        // The exact bytes of an empty file, as every build since 0.2.0 wrote them.
        assert_eq!(
            serde_json::to_string(&LabelFile::default()).unwrap(),
            r#"{"version":1,"status":"unreviewed","empty_confirmed":false,"boxes":[],"classes_version":""}"#
        );
        let plain = file(vec![accepted("car", CAR)], vec![]);
        let text = json(&plain);
        assert!(!text.contains("refine"), "{text}");
        // Suggestions added and taken away again: the same bytes.
        let mut with = plain.clone();
        with.refine = vec![add("dog", FAR, 0.5)];
        assert!(json(&with).contains("\"refine\""));
        with.refine.clear();
        assert_eq!(json(&with), text);
        // An old file loads without them.
        let old: LabelFile = serde_json::from_str(r#"{"status":"reviewed"}"#).unwrap();
        assert!(old.refine.is_empty());
        // A minimal suggestion is small: no empty optional field is written.
        let s = Suggestion {
            run: String::new(),
            ..verify(CAR)
        };
        assert_eq!(
            serde_json::to_string(&s).unwrap(),
            r#"{"kind":"verify","target":[100.0,100.0,200.0,180.0],"why":"a fold model that never saw the frame found it at 0.80"}"#
        );
        // With the target's label: after the target, and read back as written.
        let t = labelled(s.clone(), "car");
        let text = serde_json::to_string(&t).unwrap();
        assert_eq!(
            text,
            r#"{"kind":"verify","target":[100.0,100.0,200.0,180.0],"target_label":"car","why":"a fold model that never saw the frame found it at 0.80"}"#
        );
        assert_eq!(serde_json::from_str::<Suggestion>(&text).unwrap(), t);
        // A suggestion of the first checks (no target_label) loads as one.
        let old: Suggestion = serde_json::from_str(
            r#"{"kind":"verify","target":[100.0,100.0,200.0,180.0],"why":"w","run":"r0"}"#,
        )
        .unwrap();
        assert_eq!(old.target_label, None);
    }

    #[test]
    fn accepting_an_add_replaces_the_unjudged_rejects_and_ends_the_negative() {
        let c = classes();
        let w = wanted(&c);
        let low = rejected("car", CAR, Some(super::super::bulk::LOW_SCORE_REASON), None);
        let gone = rejected(
            "car",
            NEAR,
            Some(super::super::bulk::DISMISSED_REASON),
            None,
        );
        let person = accepted("person", MAN);
        let dog = accepted("dog", FAR);
        let mut f = file(
            vec![low, person.clone(), gone, dog.clone()],
            vec![verify(MAN), add("Car", NEAR, 0.875)],
        );
        let here = |f: &LabelFile| {
            f.hard_negatives()
                .filter(|b| iou(&b.bbox, &NEAR) >= MATCH_IOU)
                .count()
        };
        assert_eq!(
            here(&f),
            1,
            "the low-score reject calls the car a false positive"
        );
        assert!(accept(&mut f, 1, Some(&w)));
        assert_eq!(f.boxes.len(), 3);
        assert_eq!(f.boxes[0], person);
        // In the place of the closest reject, the others gone, the rest in their order.
        let new = &f.boxes[1];
        assert_eq!(
            (new.label.as_str(), new.bbox, new.source, new.state),
            ("car", NEAR, Source::Auto, BoxState::Accepted)
        );
        assert_eq!(
            (new.score, new.reason.as_deref()),
            (Some(0.875), Some(REFINE_REASON))
        );
        assert!(new.model_label.is_none() && new.orig_bbox.is_none() && new.auto.is_empty());
        assert_eq!(f.boxes[2], dog);
        assert_eq!(here(&f), 0);
        assert_eq!(
            f.refine,
            [verify(MAN)],
            "the suggestion went, the others stay"
        );
        // Not live any more (the box covers it), and a save keeps the box as a label.
        let again = add("car", NEAR, 0.9);
        f.refine.push(again);
        assert_eq!(live_at(&f, Some(&w)), [0]);
        let n = normalize(f, &c, 640, 480, "now").unwrap();
        assert_eq!(n.training().count(), 3);
        assert_eq!(n.boxes[1].reason.as_deref(), Some(REFINE_REASON));
    }

    #[test]
    fn accepting_an_add_ends_a_confirmed_empty_and_a_look_at_its_overlaps() {
        let c = classes();
        let w = wanted(&c);
        let mut f = file(vec![], vec![add("person", MAN, 0.9)]);
        f.empty_confirmed = true;
        assert!(accept(&mut f, 0, Some(&w)));
        assert!(!f.empty_confirmed);
        assert_eq!(f.training().count(), 1);
        normalize(f, &c, 640, 480, "now").unwrap();
        // The reviewer checked the frame's overlaps: the look covers a box far from the others...
        let inner = [110.0, 110.0, 140.0, 140.0];
        let mut far = file(vec![accepted("car", inner)], vec![add("person", MAN, 0.9)]);
        far.overlaps_checked = true;
        assert!(accept(&mut far, 0, Some(&w)));
        assert!(far.overlaps_checked);
        // ... but not a new box nested around one of its object.
        let mut nested = file(vec![accepted("car", inner)], vec![add("car", CAR, 0.9)]);
        nested.overlaps_checked = true;
        assert!(stacks::overlaps(&nested.boxes, &w).is_empty());
        assert!(accept(&mut nested, 0, Some(&w)));
        assert!(!nested.overlaps_checked);
        let mut nested = file(vec![accepted("car", inner)], vec![add("car", CAR, 0.9)]);
        nested.overlaps_checked = true;
        assert!(accept(&mut nested, 0, None));
        assert!(!nested.overlaps_checked);
    }

    #[test]
    fn accepting_tighten_relabel_and_verify() {
        let c = classes();
        let w = wanted(&c);
        let mut f = file(
            vec![accepted("car", CAR), accepted("person", MAN)],
            vec![tighten(CAR, TIGHT), relabel(MAN, "dog"), verify(CAR)],
        );
        assert!(accept(&mut f, 0, Some(&w)));
        assert_eq!((f.boxes[0].bbox, f.boxes[0].orig_bbox), (TIGHT, Some(CAR)));
        assert_eq!(f.refine, [relabel(MAN, "dog"), verify(CAR)]);
        // The box moved: the verify of its old place is stale now.
        assert_eq!(live_at(&f, Some(&w)), [0]);
        assert!(accept(&mut f, 0, Some(&w)));
        assert_eq!(f.boxes[1].label, "dog");
        assert_eq!(f.boxes[1].model_label.as_deref(), Some("person"));
        assert_eq!(f.refine, [verify(CAR)]);
        // A verify on the box as it is now: kept, answered, never asked again.
        let mut f = file(vec![accepted("car", CAR)], vec![verify(CAR)]);
        assert!(accept(&mut f, 0, Some(&w)));
        assert_eq!(f.refine, [answered(verify(CAR), Answer::Kept)]);
        assert_eq!(f.boxes, [accepted("car", CAR)]);
        assert!(live(&f, Some(&w)).is_empty());
        assert!(!accept(&mut f, 0, Some(&w)));
        // A saved frame keeps them.
        let n = normalize(f.clone(), &c, 640, 480, "now").unwrap();
        assert_eq!(n.refine, f.refine);
    }

    fn labelled(s: Suggestion, label: &str) -> Suggestion {
        Suggestion {
            target_label: Some(label.into()),
            ..s
        }
    }

    #[test]
    fn a_target_label_tells_two_boxes_at_one_place_apart() {
        let c = classes();
        let w = wanted(&c);
        // A car and a dog labelled with the very same box.
        let boxes = vec![accepted("car", CAR), accepted("dog", CAR)];
        let targets = |s: Suggestion| -> Vec<Option<usize>> {
            live(&file(boxes.clone(), vec![s]), Some(&w))
                .iter()
                .map(|l| l.target)
                .collect()
        };
        // Without a label (the files of the first checks): the first box at the place.
        assert_eq!(targets(verify(CAR)), [Some(0)]);
        assert_eq!(targets(labelled(verify(CAR), "dog")), [Some(1)]);
        assert_eq!(targets(labelled(verify(CAR), " Dog ")), [Some(1)]);
        assert_eq!(
            targets(labelled(verify(CAR), "person")),
            Vec::<Option<usize>>::new()
        );
        // A relabel of the dog to a person changes the dog, not the car.
        let mut f = file(boxes.clone(), vec![labelled(relabel(CAR, "person"), "dog")]);
        assert!(accept(&mut f, 0, Some(&w)));
        assert_eq!(
            (f.boxes[0].label.as_str(), f.boxes[1].label.as_str()),
            ("car", "person")
        );
        // A tighten too.
        let mut f = file(boxes.clone(), vec![labelled(tighten(CAR, TIGHT), "dog")]);
        assert!(accept(&mut f, 0, Some(&w)));
        assert_eq!((f.boxes[0].bbox, f.boxes[1].bbox), (CAR, TIGHT));
        // One finding per box: two verifies at one place are two findings, a labelled and an
        // unlabelled one of one place the same.
        let (vc, vd) = (labelled(verify(CAR), "car"), labelled(verify(CAR), "dog"));
        assert!(!same_finding(&vc, &vd));
        assert!(same_finding(&vc, &verify(CAR)) && same_finding(&verify(CAR), &vd));
        let mut f = file(boxes.clone(), vec![]);
        let a = apply(&mut f, vec![vc.clone(), vd.clone()], &ALL);
        assert_eq!((a.written, a.dropped), (2, 0));
        let got: Vec<Option<usize>> = live(&f, Some(&w)).iter().map(|l| l.target).collect();
        assert_eq!(got, [Some(0), Some(1)]);
        // A dismissed verify of the dog keeps the dog's question away, not the car's.
        let mut f = file(boxes, vec![answered(vd.clone(), Answer::Dismissed)]);
        let a = apply(&mut f, vec![vc.clone(), vd], &ALL);
        assert_eq!((a.written, a.dropped), (1, 1));
        assert!(f.refine.contains(&vc));
        // Bounded like `label`.
        let mut bad = vc;
        bad.target_label = Some(" ".into());
        assert!(check(std::slice::from_ref(&bad)).is_err());
    }

    #[test]
    fn answered_suggestions_whose_box_is_gone_go_at_the_next_check() {
        let c = classes();
        let w = wanted(&c);
        let kept = answered(verify(CAR), Answer::Kept);
        let moved = answered(relabel(MAN, "dog"), Answer::Dismissed);
        let other = answered(labelled(verify(CAR), "dog"), Answer::Dismissed);
        let add_no = answered(add("dog", FAR, 0.7), Answer::Dismissed);
        let unknown = Suggestion {
            kind: Kind::Unknown,
            ..answered(verify(MAN), Answer::Dismissed)
        };
        let tight = answered(tighten(MAN, TIGHT), Answer::Dismissed);
        let mut f = file(
            vec![accepted("car", CAR)],
            vec![
                kept.clone(),
                moved.clone(),
                other.clone(),
                add_no.clone(),
                unknown.clone(),
                tight.clone(),
            ],
        );
        let a = apply(&mut f, vec![], &write_kinds());
        // The person moved away and the dog never was at CAR: their answers go. The car's answer,
        // the add's (it has no box: it keeps the question away), an unknown kind and a kind the
        // check does not make stay.
        assert_eq!((a.removed, a.kept_answers, a.changed()), (2, 4, true));
        for s in [&kept, &add_no, &unknown, &tight] {
            assert!(f.refine.contains(s), "{s:?}");
        }
        assert!(!f.refine.contains(&moved) && !f.refine.contains(&other));
        // Nothing left to drop: the next check writes nothing.
        assert!(!apply(&mut f, vec![], &write_kinds()).changed());
        assert!(live(&f, Some(&w)).is_empty());
        // 64 answered ones on boxes that went do not block a new finding.
        let gone: Vec<Suggestion> = (0..MAX_SUGGESTIONS)
            .map(|i| {
                let x = 5.0 * i as f32;
                answered(verify([x, 0.0, x + 4.0, 4.0]), Answer::Kept)
            })
            .collect();
        let mut f = file(vec![accepted("car", CAR)], gone);
        let a = apply(&mut f, vec![add("dog", FAR, 0.9)], &write_kinds());
        assert_eq!((a.written, a.removed), (1, MAX_SUGGESTIONS));
        assert_eq!(f.refine, [add("dog", FAR, 0.9)]);
    }

    /// The kinds a check writes (`training::refine::write::KINDS`).
    fn write_kinds() -> [Kind; 3] {
        [Kind::Add, Kind::Verify, Kind::Relabel]
    }

    #[test]
    fn accepting_a_relabel_or_tighten_ends_a_look_at_the_overlaps_it_makes() {
        let c = classes();
        let w = wanted(&c);
        for wanted in [Some(&w), None] {
            // A person box on the car: relabelled to car, it stacks with the car.
            let mut f = file(
                vec![accepted("car", CAR), accepted("person", NEAR)],
                vec![relabel(NEAR, "car")],
            );
            f.overlaps_checked = true;
            assert!(accept(&mut f, 0, wanted));
            assert!(!f.overlaps_checked, "{wanted:?}");
            // A car far away, tightened onto the other car.
            let mut f = file(
                vec![accepted("car", CAR), accepted("car", FAR)],
                vec![tighten(FAR, NEAR)],
            );
            f.overlaps_checked = true;
            assert!(accept(&mut f, 0, wanted));
            assert!(!f.overlaps_checked, "{wanted:?}");
            // A relabel or tighten that makes no overlap keeps the look.
            let mut f = file(
                vec![accepted("car", CAR), accepted("person", MAN)],
                vec![relabel(MAN, "dog"), tighten(CAR, TIGHT)],
            );
            f.overlaps_checked = true;
            assert!(accept(&mut f, 0, wanted));
            assert!(accept(&mut f, 0, wanted));
            assert!(f.overlaps_checked, "{wanted:?}");
        }
    }

    #[test]
    fn a_check_never_writes_what_a_save_would_refuse() {
        let mut f = file(vec![], vec![]);
        let over = add("person", MAN, 1.5);
        let nan_box = Suggestion {
            bbox: Some([f32::NAN, 0.0, 1.0, 1.0]),
            ..add("dog", FAR, 0.5)
        };
        let long = Suggestion {
            why: "x".repeat(MAX_WHY_LEN + 1),
            ..add("car", CAR, 0.5)
        };
        let a = apply(
            &mut f,
            vec![over, nan_box, long, add("car", [0.0, 0.0, 20.0, 20.0], 0.6)],
            &write_kinds(),
        );
        assert_eq!((a.written, a.dropped), (1, 3));
        check(&f.refine).unwrap();
        normalize(f, &classes(), 640, 480, "now").unwrap();
    }

    #[test]
    fn accepting_what_is_not_live_changes_nothing() {
        let c = classes();
        let w = wanted(&c);
        let f = file(
            vec![drawn("car", CAR), accepted("car", FAR)],
            vec![
                tighten(CAR, TIGHT),
                add("car", NEAR, 0.9),
                relabel(FAR, "truck"),
                Suggestion {
                    kind: Kind::Unknown,
                    ..verify(FAR)
                },
                answered(verify(FAR), Answer::Dismissed),
            ],
        );
        assert!(live(&f, Some(&w)).is_empty());
        let text = json(&f);
        for i in 0..f.refine.len() + 2 {
            let mut g = f.clone();
            assert!(!accept(&mut g, i, Some(&w)), "{i}");
            assert_eq!(json(&g), text, "{i}");
        }
    }

    #[test]
    fn a_dismissal_is_remembered_and_changes_no_box() {
        let mut f = file(
            vec![accepted("car", CAR)],
            vec![verify(CAR), add("dog", FAR, 0.6)],
        );
        let boxes = f.boxes.clone();
        assert!(dismiss(&mut f, 1));
        assert_eq!(f.refine[1].answer, Some(Answer::Dismissed));
        assert!(!dismiss(&mut f, 1), "answered already");
        assert!(!dismiss(&mut f, 9));
        assert_eq!(f.boxes, boxes);
        assert_eq!(live_at(&f, None), [0]);
        // A rerun finding the same dog does not ask again.
        let a = apply(
            &mut f,
            vec![add("dog", [401.0, 300.0, 451.0, 400.0], 0.9)],
            &[Kind::Add],
        );
        assert_eq!((a.dropped, a.changed()), (1, false));
    }

    const ALL: [Kind; 4] = [Kind::Add, Kind::Tighten, Kind::Relabel, Kind::Verify];

    #[test]
    fn answers_survive_a_rerun_and_suppress_the_same_finding() {
        let mut f = file(
            vec![accepted("car", CAR), accepted("person", MAN)],
            vec![
                answered(add("car", FAR, 0.7), Answer::Dismissed),
                answered(verify(MAN), Answer::Kept),
                answered(relabel(CAR, "person"), Answer::Dismissed),
            ],
        );
        let answers = f.refine.clone();
        let far2 = [405.0, 305.0, 455.0, 405.0];
        assert!(iou(&FAR, &far2) >= ADD_SAME_IOU);
        let found = vec![
            add("car", far2, 0.9),                      // the dismissed add's place
            add("truck", far2, 0.95),                   // one object with it
            add("person", FAR, 0.8),                    // another object there: new
            verify(MAN),                                // answered
            relabel(CAR, "dog"),                        // same target as a dismissed relabel
            tighten(MAN, [305.0, 105.0, 335.0, 195.0]), // new
            add("car", [300.0, 0.0, 330.0, 20.0], 0.6), // new
        ];
        let a = apply(&mut f, found, &ALL);
        assert_eq!(
            (a.written, a.kept_answers, a.dropped, a.removed),
            (3, 3, 4, 0)
        );
        for s in &answers {
            assert!(f.refine.contains(s), "{s:?}");
        }
        assert_eq!(f.refine.len(), 6);
        // Strongest first: adds (by score), the tighten, then the answered ones.
        let kinds: Vec<(Kind, bool)> = f.refine.iter().map(|s| (s.kind, s.answered())).collect();
        assert_eq!(
            kinds,
            [
                (Kind::Add, false),
                (Kind::Add, false),
                (Kind::Tighten, false),
                (Kind::Add, true),
                (Kind::Relabel, true),
                (Kind::Verify, true),
            ]
        );
        assert_eq!(f.refine[0].label.as_deref(), Some("person"));
    }

    #[test]
    fn found_adds_of_one_place_keep_the_strongest() {
        let mut f = file(vec![], vec![]);
        let a = apply(
            &mut f,
            vec![
                add("car", CAR, 0.6),
                add("truck", NEAR, 0.9),
                add("person", NEAR, 0.5),
            ],
            &[Kind::Add],
        );
        assert_eq!((a.written, a.dropped), (2, 1));
        assert_eq!(
            f.refine,
            [add("truck", NEAR, 0.9), add("person", NEAR, 0.5)]
        );
    }

    #[test]
    fn a_rerun_that_finds_the_same_writes_nothing() {
        let mut f = file(vec![accepted("car", CAR)], vec![]);
        let found = vec![verify(CAR), add("dog", FAR, 0.7), add("person", MAN, 0.4)];
        assert!(apply(&mut f, found.clone(), &ALL).changed());
        let first = f.clone();
        // Another run, another order: the same findings.
        let mut again: Vec<Suggestion> = found
            .into_iter()
            .rev()
            .map(|s| Suggestion {
                run: "r2".into(),
                ..s
            })
            .collect();
        let a = apply(&mut f, again.clone(), &ALL);
        assert!(!a.changed(), "{a:?}");
        assert_eq!((a.written, a.removed), (0, 0));
        assert_eq!(f, first, "untouched: the old run ids stay");
        // A changed score is a change; the list is written with the new run.
        again[0].score = Some(0.45);
        let a = apply(&mut f, again, &ALL);
        assert_eq!((a.written, a.removed, a.changed()), (1, 1, true));
        assert!(f.refine.iter().any(|s| s.run == "r2"));
        // A rerun that finds nothing takes the unanswered ones of its kinds away.
        let a = apply(&mut f, vec![], &[Kind::Add]);
        assert_eq!((a.removed, a.changed()), (2, true));
        assert_eq!(
            f.refine.iter().map(|s| s.kind).collect::<Vec<_>>(),
            [Kind::Verify]
        );
    }

    #[test]
    fn a_check_of_some_kinds_leaves_the_others() {
        let mut f = file(
            vec![accepted("car", CAR)],
            vec![verify(CAR), tighten(CAR, TIGHT), add("dog", FAR, 0.3)],
        );
        let a = apply(
            &mut f,
            vec![add("person", MAN, 0.9), verify(CAR), relabel(CAR, "dog")],
            &[Kind::Add],
        );
        // The old add replaced, the found verify and relabel outside the kinds dropped.
        assert_eq!((a.written, a.removed, a.dropped), (1, 1, 2));
        assert_eq!(
            f.refine,
            [add("person", MAN, 0.9), tighten(CAR, TIGHT), verify(CAR)]
        );
        // A found suggestion carrying an answer is no finding.
        let a = apply(
            &mut f,
            vec![
                add("person", MAN, 0.9),
                answered(add("dog", FAR, 0.9), Answer::Kept),
            ],
            &[Kind::Add],
        );
        assert_eq!((a.dropped, a.changed()), (1, false));
    }

    #[test]
    fn the_cap_keeps_the_answers_first_then_the_strongest() {
        let answers: Vec<Suggestion> = (0..3)
            .map(|i| {
                let x = 600.0 + 10.0 * i as f32;
                answered(add("dog", [x, 0.0, x + 5.0, 5.0], 0.5), Answer::Dismissed)
            })
            .collect();
        let mut f = file(vec![], answers.clone());
        let found: Vec<Suggestion> = (0..70)
            .map(|i| {
                let x = 10.0 * i as f32;
                add("person", [x, 100.0, x + 5.0, 120.0], 0.01 * i as f32)
            })
            .collect();
        let a = apply(&mut f, found, &[Kind::Add]);
        assert_eq!(f.refine.len(), MAX_SUGGESTIONS);
        assert_eq!((a.written, a.kept_answers, a.dropped), (61, 3, 9));
        for s in &answers {
            assert!(f.refine.contains(s));
        }
        // The weakest found ones went.
        assert!(
            f.refine
                .iter()
                .filter(|s| !s.answered())
                .all(|s| s.score.unwrap() >= 0.085)
        );
        let s: Vec<u32> = f.refine.iter().map(strength).collect();
        assert!(s.windows(2).all(|p| p[0] >= p[1]), "{s:?}");
        check(&f.refine).unwrap();
    }

    #[test]
    fn strength_orders_kinds_then_scores() {
        let order = [
            add("car", CAR, 0.9),
            add("car", CAR, 0.1),
            relabel(CAR, "person"),
            tighten(CAR, TIGHT),
            verify(CAR),
        ];
        let got: Vec<u32> = order.iter().map(strength).collect();
        assert!(got.windows(2).all(|p| p[0] > p[1]), "{got:?}");
        assert_eq!(strength(&add("car", CAR, 1.0)), 4999);
        assert_eq!(strength(&add("car", CAR, 0.0)), 4000);
        assert_eq!(strength(&verify(CAR)), 1000, "no score");
        let mut nan = add("car", CAR, 0.5);
        nan.score = Some(f32::NAN);
        assert_eq!(strength(&nan), 4000);
        assert_eq!(strength(&answered(add("car", CAR, 0.9), Answer::Kept)), 0);
        assert_eq!(
            strength(&Suggestion {
                kind: Kind::Unknown,
                ..add("car", CAR, 0.9)
            }),
            0
        );
    }

    #[test]
    fn check_bounds_what_a_client_sends() {
        let c = classes();
        let good = add("car", CAR, 0.5);
        check(std::slice::from_ref(&good)).unwrap();
        let long_label = "x".repeat(super::super::MAX_LABEL_LEN + 1);
        let bad: [(&str, Suggestion); 11] = [
            (
                "bbox",
                Suggestion {
                    bbox: Some([f32::NAN, 0.0, 1.0, 1.0]),
                    ..good.clone()
                },
            ),
            (
                "target",
                Suggestion {
                    target: Some([0.0, f32::INFINITY, 1.0, 1.0]),
                    ..good.clone()
                },
            ),
            (
                "why",
                Suggestion {
                    why: "x".repeat(MAX_WHY_LEN + 1),
                    ..good.clone()
                },
            ),
            (
                "why control",
                Suggestion {
                    why: "a\nb".into(),
                    ..good.clone()
                },
            ),
            (
                "label",
                Suggestion {
                    label: Some(long_label),
                    ..good.clone()
                },
            ),
            (
                "label blank",
                Suggestion {
                    label: Some("  ".into()),
                    ..good.clone()
                },
            ),
            (
                "score",
                Suggestion {
                    score: Some(1.5),
                    ..good.clone()
                },
            ),
            (
                "score nan",
                Suggestion {
                    score: Some(f32::NAN),
                    ..good.clone()
                },
            ),
            (
                "spot",
                Suggestion {
                    spot: Some("x".repeat(MAX_SPOT_LEN + 1)),
                    ..good.clone()
                },
            ),
            (
                "run",
                Suggestion {
                    run: "x".repeat(MAX_RUN_LEN + 1),
                    ..good.clone()
                },
            ),
            (
                "run control",
                Suggestion {
                    run: "a\u{7}".into(),
                    ..good.clone()
                },
            ),
        ];
        for (what, s) in bad {
            assert!(check(std::slice::from_ref(&s)).is_err(), "{what}");
            // A save refuses it like any other field.
            let f = file(vec![], vec![s]);
            let e = normalize(f, &c, 640, 480, "now").unwrap_err();
            assert!(e.to_string().contains("refine suggestion 1"), "{what}: {e}");
        }
        // At the limits: fine.
        let mut s = good.clone();
        s.why = "é".repeat(MAX_WHY_LEN);
        s.spot = Some("x".repeat(MAX_SPOT_LEN));
        s.run = "x".repeat(MAX_RUN_LEN);
        s.score = Some(1.0);
        check(std::slice::from_ref(&s)).unwrap();
        assert!(check(&vec![good.clone(); MAX_SUGGESTIONS]).is_ok());
        assert!(check(&vec![good; MAX_SUGGESTIONS + 1]).is_err());
    }

    /// A small deterministic generator (xorshift) for the property test.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        fn chance(&mut self, pct: usize) -> bool {
            self.below(100) < pct
        }

        fn pick<T: Clone>(&mut self, v: &[T]) -> T {
            v[self.below(v.len())].clone()
        }

        /// A box near one of a few places, so boxes overlap often.
        fn bbox(&mut self) -> [f32; 4] {
            let places = [CAR, NEAR, TIGHT, MAN, FAR, [110.0, 110.0, 140.0, 140.0]];
            let mut b = self.pick(&places);
            if self.chance(40) {
                let d = self.below(30) as f32 - 15.0;
                b[0] += d;
                b[2] += d + self.below(20) as f32;
            }
            b
        }

        fn score(&mut self) -> f32 {
            self.below(101) as f32 / 100.0
        }
    }

    const LABELS: [&str; 7] = ["car", "truck", "person", "dog", "Car", "zebra", "motorbike"];

    fn gen_box(r: &mut Rng) -> LabelBox {
        let state = r.pick(&[
            BoxState::Pending,
            BoxState::Accepted,
            BoxState::Rejected,
            BoxState::Added,
        ]);
        let source = r.pick(&[Source::Model, Source::Auto, Source::Human]);
        let reasons = [
            None,
            Some(super::super::bulk::LOW_SCORE_REASON),
            Some(super::super::bulk::DISMISSED_REASON),
            Some(super::super::bulk::SPOT_DISMISSED_REASON),
            Some(super::super::repeats::REPEAT_SPOT_REASON),
            Some(stacks::DUPLICATE_REASON),
            Some(super::super::REST_REASON),
            Some(REFINE_REASON),
        ];
        let mut auto = super::super::AutoInfo::default();
        if r.chance(30) {
            auto.triage = Some(r.pick(&[Triage::Accept, Triage::Review]));
            auto.labelers.insert("gdino".into(), r.score());
        }
        LabelBox {
            label: r.pick(&LABELS).to_string(),
            bbox: r.bbox(),
            source,
            score: r.chance(80).then(|| r.score()),
            state,
            origin: r.pick(&[
                None,
                Some(Origin::Prediction),
                Some(Origin::Suppressed),
                Some(Origin::BelowThreshold),
                Some(Origin::VerifyFound),
            ]),
            reason: r.pick(&reasons).map(str::to_string),
            model_label: r.chance(15).then(|| r.pick(&LABELS).to_string()),
            orig_bbox: r.chance(15).then(|| r.bbox()),
            difficult: r.chance(10),
            reject_reason: r.pick(&[
                None,
                None,
                Some(RejectReason::FalsePositive),
                Some(RejectReason::Duplicate),
                Some(RejectReason::TooSmall),
                Some(RejectReason::Other("newer".into())),
            ]),
            auto,
        }
    }

    fn gen_suggestion(r: &mut Rng, boxes: &[LabelBox], answers: bool) -> Suggestion {
        let kind = r.pick(&[
            Kind::Add,
            Kind::Add,
            Kind::Tighten,
            Kind::Relabel,
            Kind::Verify,
            Kind::Unknown,
        ]);
        let target = match boxes.len() {
            0 => r.chance(50).then(|| r.bbox()),
            n if r.chance(80) => Some(boxes[r.below(n)].bbox),
            _ => r.chance(50).then(|| r.bbox()),
        };
        // Mostly the label of the box at the target, sometimes another one (two boxes at one place).
        let target_label = (kind != Kind::Add && r.chance(50)).then(|| {
            match boxes.iter().find(|b| Some(b.bbox) == target) {
                Some(b) if r.chance(80) => b.label.clone(),
                _ => r.pick(&LABELS).to_string(),
            }
        });
        Suggestion {
            kind,
            target: (kind != Kind::Add).then_some(target).flatten(),
            target_label,
            bbox: (kind != Kind::Verify && r.chance(90)).then(|| r.bbox()),
            label: r.chance(90).then(|| r.pick(&LABELS).to_string()),
            score: r.chance(90).then(|| r.score()),
            why: r
                .pick(&["seen by fold 2", "", "a person at 0.91"])
                .to_string(),
            spot: r.chance(20).then(|| format!("cam:{}", r.below(3))),
            answer: if answers {
                r.pick(&[
                    None,
                    None,
                    None,
                    Some(Answer::Kept),
                    Some(Answer::Dismissed),
                    Some(Answer::Unknown),
                ])
            } else {
                None
            },
            run: r.pick(&["r1", "r2", ""]).to_string(),
        }
    }

    fn gen_file(r: &mut Rng) -> LabelFile {
        let boxes: Vec<LabelBox> = (0..r.below(9)).map(|_| gen_box(r)).collect();
        let refine = (0..r.below(8))
            .map(|_| gen_suggestion(r, &boxes, true))
            .collect();
        let overlap_suggestions = if boxes.len() >= 2 && r.chance(30) {
            vec![super::super::overlaps::Suggestion {
                a: boxes[0].bbox,
                b: boxes[1].bbox,
                verdict: super::super::overlaps::Verdict::DropB,
                why: "zoom".into(),
                kept_score: Some(0.7),
                other_score: Some(0.0),
                low_confidence: false,
                weak: false,
            }]
        } else {
            Vec::new()
        };
        LabelFile {
            status: r.pick(&[Status::Unreviewed, Status::Reviewed, Status::Skipped]),
            empty_confirmed: r.chance(15),
            reviewed_at: r.chance(50).then(|| "2026-11-01T00:00:00Z".to_string()),
            classes_version: "abc".into(),
            disagreement: r.chance(15).then(|| "re-label by night".to_string()),
            auto_run: r.chance(30).then(|| super::super::AutoRun {
                at: "2026-11-01T00:00:00Z".into(),
                labeler: "mm-gdino-tiny + rfdetr-large".into(),
                classes: Vec::new(),
            }),
            bulk_accepted: r.chance(10),
            repeat_spot: r.chance(10),
            overlaps_checked: r.chance(20),
            checked_before_reopen: r.chance(10),
            boxes,
            overlap_suggestions,
            refine,
            ..LabelFile::default()
        }
    }

    fn without_refine(f: &LabelFile) -> LabelFile {
        LabelFile {
            refine: Vec::new(),
            ..f.clone()
        }
    }

    #[test]
    fn a_check_changes_nothing_but_refine_on_any_file() {
        let c = classes();
        let w = wanted(&c);
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        let mut changed = 0;
        for round in 0..1500 {
            let mut f = gen_file(&mut r);
            let before = f.clone();
            let found: Vec<Suggestion> = (0..r.below(12))
                .map(|_| gen_suggestion(&mut r, &before.boxes, round % 5 == 0))
                .collect();
            let kinds: Vec<Kind> = [
                Kind::Add,
                Kind::Tighten,
                Kind::Relabel,
                Kind::Verify,
                Kind::Unknown,
            ]
            .into_iter()
            .filter(|_| r.chance(60))
            .collect();
            let a = apply(&mut f, found.clone(), &kinds);
            assert_eq!(without_refine(&f), without_refine(&before), "round {round}");
            assert!(f.refine.len() <= MAX_SUGGESTIONS);
            check(&f.refine).unwrap();
            if a.changed() {
                changed += 1;
            } else {
                assert_eq!(
                    f.refine, before.refine,
                    "round {round}: unchanged = untouched"
                );
            }
            // Answered suggestions are never lost but for those of `kinds` whose box is gone;
            // unknown kinds neither unless replaced.
            for s in before.refine.iter().filter(|s| s.answered()) {
                let gone = has_target(s.kind)
                    && kinds.contains(&s.kind)
                    && !before.boxes.iter().any(|b| names(s, b));
                assert_eq!(!f.refine.contains(s), gone, "round {round}: {s:?}");
            }
            for s in before.refine.iter().filter(|s| !kinds.contains(&s.kind)) {
                assert!(f.refine.contains(s), "round {round}: lost {s:?}");
            }
            // No found suggestion of a kind outside `kinds`.
            let new: Vec<&Suggestion> = f
                .refine
                .iter()
                .filter(|s| !before.refine.contains(s))
                .collect();
            assert!(new.iter().all(|s| kinds.contains(&s.kind) && !s.answered()));
            // The same findings again, under another run id: nothing to write.
            let once = f.clone();
            let rerun: Vec<Suggestion> = found
                .into_iter()
                .map(|s| Suggestion {
                    run: "r9".into(),
                    ..s
                })
                .collect();
            let b = apply(&mut f, rerun, &kinds);
            assert!(!b.changed(), "round {round}: {b:?}");
            assert_eq!(f, once);
            // A reviewer's answers: an accept only where live, a dismissal never on a box.
            let live_now: Vec<usize> = live_at(&f, Some(&w));
            for i in 0..f.refine.len() + 1 {
                let mut g = f.clone();
                assert_eq!(
                    accept(&mut g, i, Some(&w)),
                    live_now.contains(&i),
                    "{round}"
                );
                if !live_now.contains(&i) {
                    assert_eq!(g, f);
                }
                assert!(g.boxes.len() <= super::super::MAX_BOXES);
                let mut g = f.clone();
                dismiss(&mut g, i);
                assert_eq!(g.boxes, f.boxes);
                assert!(!live_at(&g, Some(&w)).contains(&i));
            }
        }
        assert!(
            changed > 500,
            "{changed}: the generator rarely changes anything"
        );
    }

    #[test]
    fn suggestions_survive_a_save_and_a_file_without_them_reads_as_before() {
        let c = classes();
        let s = add("dog", FAR, 0.6);
        let f = file(vec![accepted("car", CAR)], vec![s.clone(), verify(CAR)]);
        for status in [Status::Unreviewed, Status::Reviewed, Status::Skipped] {
            let n = normalize(
                LabelFile {
                    status,
                    ..f.clone()
                },
                &c,
                640,
                480,
                "now",
            )
            .unwrap();
            assert_eq!(n.refine, f.refine, "{status:?}");
        }
        // The file's JSON round trip, exact floats included.
        let text = json(&f);
        assert!(text.contains("\"refine\""));
        assert_eq!(serde_json::from_str::<LabelFile>(&text).unwrap(), f);
        // Unbounded input is refused like any other field.
        let mut bad = f.clone();
        bad.refine[0].why = "x".repeat(MAX_WHY_LEN + 1);
        let e = normalize(bad, &c, 640, 480, "now").unwrap_err();
        assert!(e.to_string().contains("refine suggestion 1"), "{e}");
        // Off, the field is not written, and an old file loads without it.
        let off = file(vec![accepted("car", CAR)], vec![]);
        assert!(!json(&off).contains("refine"));
        let old: LabelFile = serde_json::from_str(r#"{"status":"reviewed"}"#).unwrap();
        assert!(old.refine.is_empty());
    }

    #[test]
    fn suggestions_survive_merges_triage_and_an_import() {
        let list = vec![add("dog", FAR, 0.6), verify(CAR)];
        let base = file(
            vec![accepted("car", CAR), accepted("car", MAN)],
            list.clone(),
        );
        // merge_proposals: the gained box reopens the frame.
        let mut f = base.clone();
        let r = super::super::merge_proposals(&mut f, vec![at("person", [0.0, 0.0, 20.0, 40.0])]);
        assert_eq!((r.added, f.status), (1, Status::Unreviewed));
        assert_eq!(f.refine, list);
        // Auto-label triage and a missing-label run.
        use crate::training::autolabel::triage;
        let mut f = base.clone();
        triage::apply(&mut f, vec![at("person", [0.0, 0.0, 20.0, 40.0])]);
        assert_eq!(f.refine, list);
        let mut f = base.clone();
        let m = triage::apply_missing(&mut f, vec![at("person", [0.0, 0.0, 20.0, 40.0])], "night");
        assert_eq!(m.added, 1);
        assert_eq!(f.refine, list);
        // The overlap check writes its own list only.
        let mut f = file(
            vec![
                accepted("car", CAR),
                accepted("car", [110.0, 110.0, 140.0, 140.0]),
            ],
            list.clone(),
        );
        let c = classes();
        let a = super::super::overlaps::apply(&mut f, &wanted(&c), false, &|_| {
            Some(super::super::overlaps::Zoom::Unavailable)
        });
        assert!(a.changed() || a.unclear > 0 || a.drops > 0, "{a:?}");
        assert_eq!(f.refine, list);
        // An X-AnyLabeling import rebuilds the file from its rectangles.
        let rects = [
            ("car".to_string(), CAR, None, false),
            ("car".to_string(), MAN, None, false),
        ];
        let f = super::super::xany::apply_import(base, &rects);
        assert_eq!(f.refine, list);
    }

    #[test]
    fn suggestions_survive_every_bulk_edit() {
        use super::super::bulk::test_support::{labels_of, put_frame};
        use super::super::bulk::{self, Scope};
        use super::super::queue::Filters;
        use super::super::repeats::{self, Spot};
        use super::super::{Frame, write};
        let dir = std::env::temp_dir().join(format!("bop-refine-bulk-{}", uuid::Uuid::new_v4()));
        let c = classes();
        let hidden = ["truck".to_string()];
        let w = Wanted {
            classes: &c,
            hidden: &hidden,
        };
        let list = vec![add("person", [40.0, 5.0, 50.0, 20.0], 0.6)];
        let filters = Filters::default();
        let (box_a, box_b) = ([0.0, 0.0, 30.0, 30.0], [2.0, 2.0, 20.0, 20.0]);
        let pair = |extra: Vec<LabelBox>| {
            let mut v = vec![at("car", box_a), at("car", box_b)];
            v.extend(extra);
            v
        };
        // Every frame carries the suggestions; each edit below must change its frame and keep them.
        let put = |stem: &str, boxes: Vec<LabelBox>, status: Status| {
            let rel = put_frame(&dir, stem, None, boxes);
            let frame = Frame::resolve(&dir, &rel).unwrap();
            let mut l = frame.labels().unwrap().unwrap();
            l.status = status;
            l.refine = list.clone();
            write(&frame.labels, &l).unwrap();
            rel
        };
        let low = LabelBox {
            score: Some(0.1),
            ..at("dog", [40.0, 30.0, 60.0, 45.0])
        };
        let check_kept = |rel: &str, what: &str, changed: bool| {
            assert!(changed, "{what} changed nothing");
            assert_eq!(labels_of(&dir, rel).refine, list, "{what}");
        };
        let rel = put("low", pair(vec![low]), Status::Unreviewed);
        let o = bulk::bulk_reject_below(&dir, &filters, 0.3, 500, false, &|r| r == rel);
        check_kept(&rel, "reject-low", o.boxes == 1);
        let clock = at("clock", [40.0, 30.0, 60.0, 45.0]);
        let rel = put("other", pair(vec![clock.clone()]), Status::Unreviewed);
        let o = bulk::bulk_reject_other_class(&dir, &filters, &w, 500, false, &|r| r == rel);
        check_kept(&rel, "reject-other-class", o.boxes == 1);
        let rel = put("unwanted", pair(vec![clock.clone()]), Status::Unreviewed);
        let o = bulk::bulk_reject_unwanted(&dir, &w, Scope::Unreviewed, 500, false);
        check_kept(&rel, "unwanted on class change", o.boxes >= 1);
        let truck = LabelBox {
            state: BoxState::Accepted,
            ..at("truck", [40.0, 30.0, 60.0, 45.0])
        };
        let rel = put("reviewed", vec![truck], Status::Reviewed);
        let o = bulk::bulk_reject_unwanted(&dir, &w, Scope::Reviewed, 500, false);
        check_kept(&rel, "unwanted on reviewed frames", o.boxes == 1);
        let near = [0.0, 0.0, 30.0, 29.0];
        let rel = put("stacked", pair(vec![at("car", near)]), Status::Unreviewed);
        let o = bulk::bulk_reject_stacked(&dir, &filters, &w, 500, false, &|r| r == rel);
        check_kept(&rel, "reject-stacked", o.outcome.boxes == 1);
        let sure = LabelBox {
            bbox: box_a,
            ..super::super::bulk::test_support::auto_box("car", 0.9, Triage::Accept)
        };
        let rel = put("accept", vec![sure], Status::Unreviewed);
        let ok = bulk::accept_one(&dir, &rel, &c, "now").unwrap();
        check_kept(&rel, "bulk accept", ok);
        let spot = Spot {
            camera: "cam".into(),
            label: "dog".into(),
            bbox: [40.0, 30.0, 60.0, 45.0],
            width: 64,
            height: 48,
            min_iou: repeats::REJECT_SPOT_IOU,
        };
        let v = repeats::Verdict::Reject(None);
        let dog = at("dog", spot.bbox);
        let rel = put("spot", vec![dog], Status::Unreviewed);
        let o = bulk::bulk_spot(&dir, &filters, &c, &[], &spot, &v, 500, false, &|r| {
            r == rel
        });
        check_kept(&rel, "reject-spot", o.boxes == 1);
        let o = bulk::bulk_accept_repeats(&dir, &filters, &c, 500, "now", false, &|r| r == rel);
        check_kept(&rel, "accept-repeats", o.done_count == 1);
        // A frame a re-label reopened, put back to reviewed.
        let kept = LabelBox {
            state: BoxState::Accepted,
            ..at("car", box_a)
        };
        let rel = put("reopened", vec![kept], Status::Unreviewed);
        let frame = Frame::resolve(&dir, &rel).unwrap();
        let mut l = frame.labels().unwrap().unwrap();
        l.disagreement = Some("re-label by night: 1 new box".into());
        l.reviewed_at = Some("2026-11-01T00:00:00Z".into());
        write(&frame.labels, &l).unwrap();
        let o = bulk::bulk_restore_reopened(&dir, &filters, 500, false, &|r| r == rel);
        check_kept(&rel, "restore-reopened", o.done_count == 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
