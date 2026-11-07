//! Several boxes on one object. Pure parts of one frame's boxes: [`stacks`] (near-identical boxes,
//! IoU >= [`STACK_IOU`], safe to resolve by rule: a keeper stays, [`drops`] go) and [`overlaps`]
//! (nested or partly overlapping pairs outside a stack, IoU >= [`OVERLAP_IOU`] or the smaller box
//! at least [`NESTED_SHARE`] inside the larger: often a fragment of a good box, as often a junk box
//! around the real one, so neither size nor score decides and these go to a reviewer, never a bulk
//! edit). The edits: [`reject_stacked`] (the bulk reject, [`super::bulk::bulk_reject_stacked`])
//! and [`reject_pending_stacked`] (a save must not accept new duplicates). [`needs_look`] is the
//! queue's `auto=overlaps` test. A box rejected here carries [`RejectReason::Duplicate`]: it sits on
//! a real object, so it is neither a hard negative ([`LabelFile::hard_negatives`]) nor a decision
//! about the spot ([`super::repeats`]).
//!
//! Twin of the Review tab's `dupGroups`, `rank` and `dupDrops` (`assets/review-rules.js`): keep the
//! thresholds, the keeper rank and the tie break in step (`tests/js_rules.rs` runs both on generated
//! frames). Three differences: a box of no listed class that is not rejected counts as unwanted
//! here, while the page keeps it (it shows such a box: a save refuses it until it is relabelled); an
//! alias of a hidden class is unwanted here, while the page, which knows no aliases, keeps it; and
//! this file computes in f32 and the page in f64, so an IoU or share that sits on a threshold to
//! within rounding (104.3 over 149 px, exactly 0.7 in the file's decimals) can fall on either side.

use super::bulk::Wanted;
use super::repeats::same_object;
use super::{BoxState, LabelBox, LabelFile, RejectReason, Source, Status};
use std::collections::BTreeMap;

/// IoU at or above which two boxes of one object are a stack (`dupGroups` on the page).
pub const STACK_IOU: f32 = 0.7;
/// IoU at or above which two boxes of one object, not stacked, need a reviewer.
pub const OVERLAP_IOU: f32 = super::MATCH_IOU;
/// Share of the smaller box inside the larger ([`super::inside_share`]) at or above which two
/// boxes of one object, not stacked, need a reviewer.
pub const NESTED_SHARE: f32 = 0.8;
/// The provenance (`reason`) a duplicate reject gives a box that had none.
pub const DUPLICATE_REASON: &str = "duplicate";

/// Boxes of one object (indices into the frame's boxes, ascending; at least 2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stack {
    pub members: Vec<usize>,
    /// The member that stays ([`rank`]; the first of equals).
    pub keeper: usize,
}

/// Whether `b` takes part: not rejected and, with a class list, not unwanted.
fn live(b: &LabelBox, wanted: Option<&Wanted<'_>>) -> bool {
    b.state != BoxState::Rejected && wanted.is_none_or(|w| w.class_of(&b.label).is_some())
}

/// The keeper order of a stack (the page's `rank`), highest first: drawn, reviewer-edited
/// (accepted and moved or relabelled), other accepted, pending, then a pending hint saving would
/// reject ([`super::resolve_pending`]: were it the keeper, saving would leave the object with no
/// box at all); within a tier by score.
fn rank(b: &LabelBox) -> (u8, f32) {
    let tier = match b.state {
        BoxState::Added => 4,
        BoxState::Accepted if b.orig_bbox.is_some() || b.model_label.is_some() => 3,
        BoxState::Accepted => 2,
        _ if super::resolve_pending(b) == BoxState::Rejected => 0,
        _ => 1,
    };
    (tier, b.score.unwrap_or(0.0))
}

/// The root of `x` in the disjoint sets of `parent`.
fn root(parent: &[usize], mut x: usize) -> usize {
    while let Some(&p) = parent.get(x)
        && p != x
    {
        x = p;
    }
    x
}

/// [`stacks`], with a class list or without one (every label wanted).
fn groups(boxes: &[LabelBox], wanted: Option<&Wanted<'_>>) -> Vec<Stack> {
    let items: Vec<(usize, &LabelBox)> = boxes
        .iter()
        .enumerate()
        .filter(|(_, b)| live(b, wanted))
        .collect();
    let mut parent: Vec<usize> = (0..boxes.len()).collect();
    for (n, (i, a)) in items.iter().enumerate() {
        for (j, b) in items.iter().skip(n + 1) {
            if same_object(&a.label, &b.label) && super::iou(&a.bbox, &b.bbox) >= STACK_IOU {
                let (ri, rj) = (root(&parent, *i), root(&parent, *j));
                if let Some(p) = parent.get_mut(rj) {
                    *p = ri;
                }
            }
        }
    }
    let mut by_root: BTreeMap<usize, Vec<(usize, &LabelBox)>> = BTreeMap::new();
    for (i, b) in items {
        by_root.entry(root(&parent, i)).or_default().push((i, b));
    }
    by_root
        .into_values()
        .filter(|m| m.len() >= 2)
        .filter_map(|m| {
            let keeper = m
                .iter()
                .copied()
                .reduce(|a, k| if rank(k.1) > rank(a.1) { k } else { a })?
                .0;
            Some(Stack {
                members: m.into_iter().map(|(i, _)| i).collect(),
                keeper,
            })
        })
        .collect()
}

/// The stacks of `boxes`: boxes that are neither rejected nor unwanted, of one object
/// ([`same_object`]), joined at IoU >= [`STACK_IOU`] (transitively: a chain is one stack).
pub fn stacks(boxes: &[LabelBox], wanted: &Wanted<'_>) -> Vec<Stack> {
    groups(boxes, Some(wanted))
}

/// [`overlaps`], with a class list or without one (every label wanted).
pub(super) fn overlap_pairs(
    boxes: &[LabelBox],
    wanted: Option<&Wanted<'_>>,
) -> Vec<(usize, usize)> {
    let stacked = groups(boxes, wanted);
    let stack_of = |i: usize| stacked.iter().position(|s| s.members.contains(&i));
    let items: Vec<(usize, &LabelBox)> = boxes
        .iter()
        .enumerate()
        .filter(|(_, b)| live(b, wanted))
        .collect();
    let mut out = Vec::new();
    for (n, (i, a)) in items.iter().enumerate() {
        for (j, b) in items.iter().skip(n + 1) {
            let one_stack = stack_of(*i).is_some() && stack_of(*i) == stack_of(*j);
            if !one_stack
                && same_object(&a.label, &b.label)
                && (super::iou(&a.bbox, &b.bbox) >= OVERLAP_IOU
                    || super::inside_share(&a.bbox, &b.bbox) >= NESTED_SHARE)
            {
                out.push((*i, *j));
            }
        }
    }
    out
}

/// Pairs (lower index first) of boxes of one object, neither rejected nor unwanted and not in one
/// stack, at IoU >= [`OVERLAP_IOU`] or with the smaller box >= [`NESTED_SHARE`] inside the larger.
/// A reviewer decides these: size and score do not tell a fragment from a junk box.
pub fn overlaps(boxes: &[LabelBox], wanted: &Wanted<'_>) -> Vec<(usize, usize)> {
    overlap_pairs(boxes, Some(wanted))
}

/// The members of `stacks` a rule may reject: not the keeper and not drawn by the reviewer (a
/// human box stays even when it is not the keeper), ascending per stack.
pub fn drops(stacks: &[Stack], boxes: &[LabelBox]) -> Vec<usize> {
    stacks
        .iter()
        .flat_map(|s| {
            s.members
                .iter()
                .copied()
                .filter(|&k| k != s.keeper)
                .filter(|&k| boxes.get(k).is_some_and(|b| b.source != Source::Human))
        })
        .collect()
}

/// Whether `f` belongs in `auto=overlaps`: it holds a stack or an overlap and no reviewer looked
/// at them ([`LabelFile::overlaps_checked`]). `wanted` None: every label is wanted.
pub fn needs_look(f: &LabelFile, wanted: Option<&Wanted<'_>>) -> bool {
    !f.overlaps_checked
        && (!groups(&f.boxes, wanted).is_empty() || !overlap_pairs(&f.boxes, wanted).is_empty())
}

/// Reject `b` as a duplicate, keeping its provenance (`reason`, [`DUPLICATE_REASON`] if none). A
/// parked accept taken back loses the rule's mark, as at a save (`markDuplicate` on the page).
pub(super) fn mark(b: &mut LabelBox) {
    b.state = BoxState::Rejected;
    b.reject_reason = Some(RejectReason::Duplicate);
    b.difficult = false;
    if b.reason.as_deref() == Some(super::spots::PARKED_REASON) {
        b.reason = None;
        b.auto.parked = None;
    }
    if b.reason.is_none() {
        b.reason = Some(DUPLICATE_REASON.into());
    }
}

fn mark_all(f: &mut LabelFile, which: &[usize]) -> usize {
    for &i in which {
        if let Some(b) = f.boxes.get_mut(i) {
            mark(b);
        }
    }
    which.len()
}

/// The boxes [`reject_stacked`] takes: the [`drops`] of a reviewed or unreviewed frame no reviewer
/// kept them on ([`LabelFile::overlaps_checked`], or [`LabelFile::checked_before_reopen`] while a
/// rerun has the frame open). A skipped frame is never exported.
fn stacked(f: &LabelFile, wanted: &Wanted<'_>) -> Vec<usize> {
    if f.status == Status::Skipped || f.overlaps_checked || f.checked_before_reopen {
        return Vec::new();
    }
    drops(&stacks(&f.boxes, wanted), &f.boxes)
}

/// How many boxes [`reject_stacked`] would reject.
pub fn count_stacked(f: &LabelFile, wanted: &Wanted<'_>) -> usize {
    stacked(f, wanted).len()
}

/// Reject the duplicates of the stacks of `f` (accepted ones too; never a human box or a frame a
/// reviewer checked), the frame's status unchanged. Returns the boxes rejected.
pub fn reject_stacked(f: &mut LabelFile, wanted: &Wanted<'_>) -> usize {
    let which = stacked(f, wanted);
    mark_all(f, &which)
}

/// Reject only the pending [`drops`] of `f`: what a save would otherwise accept as new duplicates.
/// A box a reviewer accepted stays. Returns the boxes rejected.
pub fn reject_pending_stacked(f: &mut LabelFile, wanted: &Wanted<'_>) -> usize {
    let which: Vec<usize> = drops(&stacks(&f.boxes, wanted), &f.boxes)
        .into_iter()
        .filter(|&i| f.boxes.get(i).is_some_and(|b| b.state == BoxState::Pending))
        .collect();
    mark_all(f, &which)
}

/// Whether `f` would still need a reviewer after [`reject_stacked`] ([`needs_look`] of the frame
/// without its duplicates: dropping a box can end an overlap it was part of, or leave two boxes a
/// chain had joined as an overlap).
pub fn needs_look_after(f: &LabelFile, wanted: &Wanted<'_>) -> bool {
    let mut simulated = f.clone();
    reject_stacked(&mut simulated, wanted);
    needs_look(&simulated, Some(wanted))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::super::tests_support::auto_box;
    use super::super::{Origin, Triage};
    use super::*;

    fn classes() -> Vec<String> {
        ["person", "car", "truck"].map(String::from).to_vec()
    }

    fn at(label: &str, bbox: [f32; 4], score: f32) -> LabelBox {
        LabelBox {
            bbox,
            score: Some(score),
            ..auto_box(label)
        }
    }

    fn accepted(b: LabelBox) -> LabelBox {
        LabelBox {
            state: BoxState::Accepted,
            ..b
        }
    }

    fn human(b: LabelBox) -> LabelBox {
        LabelBox {
            source: Source::Human,
            state: BoxState::Added,
            score: None,
            ..b
        }
    }

    const A: [f32; 4] = [0.0, 0.0, 100.0, 100.0];
    /// IoU with [`A`] 0.905.
    const NEAR: [f32; 4] = [0.0, 0.0, 100.0, 95.0];
    /// Nested in [`A`]: all of it inside, IoU 0.25.
    const SMALL: [f32; 4] = [10.0, 10.0, 60.0, 60.0];
    /// Far from everything.
    const FAR: [f32; 4] = [300.0, 300.0, 400.0, 400.0];

    #[test]
    fn a_near_identical_pair_is_a_stack_with_the_best_score_as_keeper() {
        let cl = classes();
        let w = Wanted {
            classes: &cl,
            hidden: &[],
        };
        let b = vec![
            at("car", A, 0.5),
            at("car", NEAR, 0.9),
            at("car", FAR, 0.9),
            at("person", A, 0.9),
        ];
        let s = stacks(&b, &w);
        assert_eq!(
            s,
            [Stack {
                members: vec![0, 1],
                keeper: 1
            }]
        );
        assert_eq!(drops(&s, &b), [0]);
        assert!(overlaps(&b, &w).is_empty(), "a stack is not an overlap");
    }

    #[test]
    fn equal_ranks_keep_the_first_box_like_the_page() {
        let cl = classes();
        let w = Wanted {
            classes: &cl,
            hidden: &[],
        };
        let b = vec![at("car", A, 0.5), at("car", NEAR, 0.5)];
        assert_eq!(stacks(&b, &w)[0].keeper, 0);
    }

    #[test]
    fn a_chain_is_one_stack_even_when_its_ends_are_not_near() {
        let cl = classes();
        let w = Wanted {
            classes: &cl,
            hidden: &[],
        };
        let (l, m, r) = (
            [0.0, 0.0, 100.0, 100.0],
            [15.0, 0.0, 115.0, 100.0],
            [30.0, 0.0, 130.0, 100.0],
        );
        assert!(super::super::iou(&l, &m) >= STACK_IOU);
        assert!(super::super::iou(&m, &r) >= STACK_IOU);
        assert!(super::super::iou(&l, &r) < STACK_IOU);
        let b = vec![at("car", l, 0.5), at("car", m, 0.9), at("car", r, 0.5)];
        let s = stacks(&b, &w);
        assert_eq!(
            s,
            [Stack {
                members: vec![0, 1, 2],
                keeper: 1
            }]
        );
        assert_eq!(drops(&s, &b), [0, 2]);
        assert!(overlaps(&b, &w).is_empty(), "one stack: not a pair");
        // Declared in the other order the root differs, the outcome does not.
        let rev = vec![at("car", r, 0.5), at("car", m, 0.9), at("car", l, 0.5)];
        assert_eq!(stacks(&rev, &w)[0].keeper, 1);
    }

    #[test]
    fn car_and_truck_stack_as_one_object_but_a_person_does_not() {
        let cl = classes();
        let w = Wanted {
            classes: &cl,
            hidden: &[],
        };
        let b = vec![
            accepted(at("car", A, 0.6)),
            accepted(at("truck", NEAR, 0.8)),
            accepted(at("person", NEAR, 0.9)),
        ];
        let s = stacks(&b, &w);
        assert_eq!(s.len(), 1);
        assert_eq!((s[0].members.clone(), s[0].keeper), (vec![0, 1], 1));
    }

    #[test]
    fn a_human_box_is_never_dropped_even_when_it_is_not_the_keeper() {
        let cl = classes();
        let w = Wanted {
            classes: &cl,
            hidden: &[],
        };
        // Two drawn boxes: one keeper, the other stays too.
        let b = vec![
            human(at("car", A, 0.0)),
            human(at("car", NEAR, 0.0)),
            accepted(at("car", NEAR, 0.99)),
        ];
        let s = stacks(&b, &w);
        assert_eq!(s[0].keeper, 0);
        assert_eq!(drops(&s, &b), [2]);
        // A human box still pending ranks as pending: an accepted keeper does not drop it.
        let pending = LabelBox {
            source: Source::Human,
            ..at("car", A, 0.1)
        };
        let b = vec![pending, accepted(at("car", NEAR, 0.5))];
        let s = stacks(&b, &w);
        assert_eq!(s[0].keeper, 1);
        assert!(drops(&s, &b).is_empty());
    }

    #[test]
    fn the_keeper_rank_goes_drawn_edited_accepted_pending_hint_then_score() {
        let cl = classes();
        let w = Wanted {
            classes: &cl,
            hidden: &[],
        };
        let moved = LabelBox {
            orig_bbox: Some([1.0, 1.0, 90.0, 90.0]),
            ..accepted(at("car", A, 0.2))
        };
        let relabelled = LabelBox {
            model_label: Some("truck".into()),
            ..accepted(at("car", A, 0.2))
        };
        let plain = accepted(at("car", NEAR, 0.99));
        let keeper = |b: Vec<LabelBox>| stacks(&b, &w)[0].keeper;
        assert_eq!(keeper(vec![plain.clone(), moved.clone()]), 1, "edited wins");
        assert_eq!(keeper(vec![plain.clone(), relabelled]), 1, "relabelled too");
        assert_eq!(keeper(vec![at("car", NEAR, 0.99), plain.clone()]), 1);
        assert_eq!(
            keeper(vec![moved, human(at("car", NEAR, 0.0))]),
            1,
            "drawn wins"
        );
        // A hint saving would reject never outranks a proposal saving would accept, or the
        // object would end with no box at all.
        let hint = LabelBox {
            source: Source::Model,
            origin: Some(Origin::Suppressed),
            ..at("car", NEAR, 0.95)
        };
        assert_eq!(super::super::resolve_pending(&hint), BoxState::Rejected);
        let proposal = LabelBox {
            source: Source::Model,
            origin: Some(Origin::Prediction),
            ..at("car", A, 0.4)
        };
        assert_eq!(keeper(vec![hint.clone(), proposal.clone()]), 1);
        // A hint the auto-labelers confirmed is accepted on save: a plain proposal again.
        let mut confirmed = hint.clone();
        confirmed.auto.triage = Some(Triage::Accept);
        assert_eq!(keeper(vec![proposal, confirmed]), 1, "by score");
        // Among hints, by score.
        let low = LabelBox {
            score: Some(0.1),
            ..hint.clone()
        };
        assert_eq!(keeper(vec![low, hint]), 1);
    }

    #[test]
    fn nested_and_partial_overlaps_go_to_a_reviewer_not_to_a_stack() {
        let cl = classes();
        let w = Wanted {
            classes: &cl,
            hidden: &[],
        };
        let b = vec![at("car", A, 0.9), at("truck", SMALL, 0.9)];
        assert!(stacks(&b, &w).is_empty());
        assert_eq!(overlaps(&b, &w), [(0, 1)]);
        assert!(drops(&stacks(&b, &w), &b).is_empty());
        // Partial, by IoU only (neither is inside the other by 0.8).
        let part = [[0.0, 0.0, 100.0, 100.0], [0.0, 25.0, 100.0, 125.0]];
        assert!(super::super::iou(&part[0], &part[1]) >= OVERLAP_IOU);
        assert!(super::super::inside_share(&part[0], &part[1]) < NESTED_SHARE);
        let b = vec![at("car", part[0], 0.9), at("car", part[1], 0.9)];
        assert_eq!(overlaps(&b, &w), [(0, 1)]);
        // Below both, or another object.
        let b = vec![
            at("car", A, 0.9),
            at("car", [50.0, 50.0, 150.0, 150.0], 0.9),
        ];
        assert!(overlaps(&b, &w).is_empty());
        let b = vec![at("car", A, 0.9), at("person", SMALL, 0.9)];
        assert!(overlaps(&b, &w).is_empty());
        // A box of a stack nested in another box of it is no pair; a third box inside both is.
        let b = vec![
            at("car", A, 0.9),
            at("car", NEAR, 0.8),
            at("car", SMALL, 0.7),
        ];
        assert_eq!(overlaps(&b, &w), [(0, 2), (1, 2)]);
    }

    #[test]
    fn rejected_and_unwanted_boxes_take_no_part() {
        let cl = classes();
        let hidden = ["truck".to_string()];
        let w = Wanted {
            classes: &cl,
            hidden: &hidden,
        };
        let rejected = LabelBox {
            state: BoxState::Rejected,
            ..at("car", NEAR, 0.9)
        };
        let b = vec![at("car", A, 0.5), rejected.clone()];
        assert!(stacks(&b, &w).is_empty());
        let b = vec![at("car", A, 0.5), at("truck", NEAR, 0.9)];
        assert!(stacks(&b, &w).is_empty(), "a hidden class");
        let b = vec![at("car", A, 0.5), at("truck", SMALL, 0.9)];
        assert!(overlaps(&b, &w).is_empty());
        let b = vec![at("clock", A, 0.5), at("clock", NEAR, 0.9)];
        assert!(stacks(&b, &w).is_empty(), "of no listed class");
        // An alias of the list is wanted.
        let plants = vec!["potted plant".to_string()];
        let aliases = Wanted {
            classes: &plants,
            hidden: &[],
        };
        let b = vec![at("pottedplant", A, 0.5), at("pottedplant", NEAR, 0.9)];
        assert_eq!(stacks(&b, &aliases).len(), 1);
    }

    #[test]
    fn the_reject_marks_duplicates_keeps_provenance_and_is_a_no_op_twice() {
        let cl = classes();
        let w = Wanted {
            classes: &cl,
            hidden: &[],
        };
        let mut unsure = accepted(at("car", NEAR, 0.5));
        unsure.difficult = true;
        unsure.reason = Some("night_threshold".into());
        let mut f = LabelFile {
            status: Status::Reviewed,
            reviewed_at: Some("t".into()),
            boxes: vec![
                accepted(at("car", A, 0.9)),
                unsure,
                accepted(at("car", NEAR, 0.4)),
            ],
            ..LabelFile::default()
        };
        assert_eq!(count_stacked(&f, &w), 2);
        assert_eq!(reject_stacked(&mut f, &w), 2);
        let b = &f.boxes[1];
        assert_eq!(
            (b.state, b.reject_reason.clone(), b.difficult),
            (BoxState::Rejected, Some(RejectReason::Duplicate), false)
        );
        assert_eq!(b.reason.as_deref(), Some("night_threshold"));
        assert_eq!(f.boxes[2].reason.as_deref(), Some(DUPLICATE_REASON));
        assert_eq!(f.boxes[0].state, BoxState::Accepted);
        assert_eq!(f.status, Status::Reviewed);
        assert_eq!(reject_stacked(&mut f, &w), 0, "done");
    }

    #[test]
    fn a_parked_box_rejected_as_a_duplicate_loses_the_rules_mark() {
        let mut b = accepted(at("car", A, 0.9));
        b.reason = Some(super::super::spots::PARKED_REASON.into());
        b.auto.parked = Some(super::super::ParkedMark {
            since: "2026-11-06".into(),
            bbox: A,
        });
        mark(&mut b);
        assert_eq!(b.reason.as_deref(), Some(DUPLICATE_REASON));
        assert_eq!(b.auto.parked, None);
    }

    #[test]
    fn a_reviewer_who_kept_boxes_and_a_skipped_frame_are_left_alone() {
        let cl = classes();
        let w = Wanted {
            classes: &cl,
            hidden: &[],
        };
        let boxes = vec![accepted(at("car", A, 0.9)), accepted(at("car", NEAR, 0.4))];
        let checked = LabelFile {
            status: Status::Reviewed,
            overlaps_checked: true,
            boxes: boxes.clone(),
            ..LabelFile::default()
        };
        assert_eq!(count_stacked(&checked, &w), 0);
        assert!(!needs_look(&checked, Some(&w)));
        let skipped = LabelFile {
            status: Status::Skipped,
            boxes,
            ..LabelFile::default()
        };
        assert_eq!(count_stacked(&skipped, &w), 0);
    }

    #[test]
    fn pending_duplicates_are_rejected_and_an_explicit_accept_is_not() {
        let cl = classes();
        let w = Wanted {
            classes: &cl,
            hidden: &[],
        };
        let mut f = LabelFile {
            boxes: vec![
                human(at("car", A, 0.0)),
                accepted(at("car", NEAR, 0.9)),
                at("car", NEAR, 0.95),
            ],
            ..LabelFile::default()
        };
        assert_eq!(reject_pending_stacked(&mut f, &w), 1);
        let got: Vec<BoxState> = f.boxes.iter().map(|b| b.state).collect();
        assert_eq!(
            got,
            [BoxState::Added, BoxState::Accepted, BoxState::Rejected]
        );
        assert_eq!(f.boxes[2].reject_reason, Some(RejectReason::Duplicate));
    }

    #[test]
    fn what_needs_a_look_is_judged_after_the_duplicates_go() {
        let cl = classes();
        let w = Wanted {
            classes: &cl,
            hidden: &[],
        };
        let file = |boxes: Vec<LabelBox>| LabelFile {
            status: Status::Reviewed,
            boxes,
            ..LabelFile::default()
        };
        // Only the dropped box overlaps the third one: nothing is left to look at.
        const SHIFT: [f32; 4] = [15.0, 0.0, 115.0, 100.0];
        let keeper = A;
        let inner = [90.0, 0.0, 110.0, 40.0]; // half inside the keeper, all inside SHIFT
        assert!(super::super::inside_share(&keeper, &inner) < NESTED_SHARE);
        assert!(super::super::inside_share(&SHIFT, &inner) >= NESTED_SHARE);
        assert!(super::super::iou(&keeper, &SHIFT) >= STACK_IOU);
        let f = file(vec![
            accepted(at("car", keeper, 0.9)),
            accepted(at("car", SHIFT, 0.4)),
            accepted(at("car", inner, 0.9)),
        ]);
        assert!(needs_look(&f, Some(&w)), "before");
        assert!(!needs_look_after(&f, &w));
        // A stack with no overlap: settled by the reject.
        let f = file(vec![at("car", A, 0.9), at("car", NEAR, 0.4)]);
        assert!(needs_look(&f, Some(&w)));
        assert!(!needs_look_after(&f, &w));
        // A nested pair stays.
        let f = file(vec![at("car", A, 0.9), at("car", SMALL, 0.4)]);
        assert!(needs_look_after(&f, &w));
        // Two drawn boxes the rule cannot touch stay flagged: the Auto filter lists them too.
        let f = file(vec![human(at("car", A, 0.0)), human(at("car", NEAR, 0.0))]);
        assert!(needs_look_after(&f, &w));
        // A chain's ends, once the middle box goes, are an overlap only a reviewer can settle.
        let (l, m, r) = (
            [0.0, 0.0, 100.0, 100.0],
            [15.0, 0.0, 115.0, 100.0],
            [30.0, 0.0, 130.0, 100.0],
        );
        let f = file(vec![
            human(at("car", l, 0.0)),
            at("car", m, 0.5),
            human(at("car", r, 0.0)),
        ]);
        assert_eq!(reject_stacked(&mut f.clone(), &w), 1);
        assert!(needs_look_after(&f, &w));
    }

    #[test]
    fn without_a_class_list_every_label_counts() {
        let f = LabelFile {
            boxes: vec![at("clock", A, 0.5), at("clock", NEAR, 0.9)],
            ..LabelFile::default()
        };
        assert!(needs_look(&f, None));
        let checked = LabelFile {
            overlaps_checked: true,
            ..f
        };
        assert!(!needs_look(&checked, None));
    }
}
