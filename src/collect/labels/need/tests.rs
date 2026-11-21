//! Tests of the need model ([`super`]) on synthetic frames and tallies: each factor alone, against
//! the formula worked out by hand.

#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use super::super::recall::{Run, Slice};
use super::super::tests_support::auto_box;
use super::super::{AutoInfo, VlmCheck};
use super::*;
use crate::benchmark::ClassRecall;
use crate::collect::record::{Suppressed, empty_record};

const PERSON: u16 = 0;
const CAR: u16 = 1;
const RACCOON: u16 = 2;
const CAT: u16 = 3;
const DOG: u16 = 4;
const MOTORCYCLE: u16 = 5;

fn classes() -> Vec<String> {
    ["person", "car", "raccoon", "cat", "dog", "motorcycle"]
        .map(String::from)
        .to_vec()
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "{a} != {b}");
}

/// A candidate of class `k` at `bbox` with evidence `e` (hundredths) and `flags`.
fn cb(k: u16, bbox: [u16; 4], e: u8, flags: u8) -> CellBox {
    CellBox {
        class: k,
        bbox,
        evidence: e,
        flags,
    }
}

fn trained(k: u16, bbox: [u16; 4]) -> CellBox {
    cb(k, bbox, 100, TRAINING | STRONG)
}

/// A reviewed frame's marker: class `k` was proposed there and judged.
fn proposed(k: u16) -> CellBox {
    cb(k, [0; 4], 0, PROPOSED)
}

/// A reviewed frame's hard negative of class `k`.
fn rejected(k: u16) -> CellBox {
    cb(k, LEFT, 0, REJECTED)
}

/// Region 0 and region 8 of a 640 x 480 frame.
const LEFT: [u16; 4] = [10, 10, 60, 60];
const RIGHT: [u16; 4] = [500, 300, 600, 400];

fn at<'a>(camera: &'a str, scene: &'a str, day: &'a str) -> Facts<'a> {
    Facts {
        camera,
        scene,
        day,
        width: 640,
        height: 480,
        heldout: false,
    }
}

/// `n` reviewed 640 x 480 frames of `camera` in `scene` on `day`, each with `boxes`.
fn review(
    need: &Need<'_>,
    t: &mut CellTally,
    day: (&str, &str, &str),
    n: usize,
    boxes: &[CellBox],
) {
    review_split(need, t, day, false, n, boxes);
}

/// [`review`] of a held-out day or not.
fn review_split(
    need: &Need<'_>,
    t: &mut CellTally,
    (camera, scene, day): (&str, &str, &str),
    heldout: bool,
    n: usize,
    boxes: &[CellBox],
) {
    for _ in 0..n {
        need.add_reviewed(
            t,
            &Reviewed {
                camera,
                scene,
                day,
                width: 640,
                height: 480,
                boxes,
                heldout,
            },
        );
    }
}

/// The score of a frame at `f` with candidates `boxes`.
fn score(need: &Need<'_>, t: &CellTally, f: &Facts<'_>, boxes: &[CellBox]) -> u32 {
    Scorer::new(need, t).score(f, boxes, &[]).score
}

/// v_k of a class with `n_ks` and `n_mks` reviewed boxes.
fn v(n_ks: f64, n_mks: f64) -> f64 {
    0.5 * T / (T + n_ks) + 0.5 * T_CAM / (T_CAM + n_mks)
}

/// The region factor of `n` reviewed boxes in a box's region.
fn rf(n: f64) -> f64 {
    (1.0 + W_REGION * T_R / (T_R + n)) / 2.0
}

/// The score the formula gives one box of weight `a` (evidence and its box factors) in a region
/// with `region` reviewed boxes, of a class with `n_ks` and `n_mks` reviewed boxes, times `rest`
/// (day, floor, recall, catch-up).
fn expect(a: f64, region: f64, n_ks: f64, n_mks: f64, rest: f64) -> u32 {
    score_of(crowd(a * rf(region)) * v(n_ks, n_mks) * rest)
}

fn wanted(c: &[String]) -> Wanted<'_> {
    Wanted {
        classes: c,
        hidden: &[],
    }
}

/// A new day of a class under the floor.
const FRESH: f64 = (1.0 + W_DAY) * RARE_GAIN;

#[test]
fn a_cell_needs_less_as_its_boxes_come_per_camera_and_overall() {
    let c = classes();
    let need = Need::new(Some(wanted(&c)), None);
    let mut t = CellTally::default();
    // 25 boxes of person on `back` (day d1), 40 on `front`: past the floor.
    review(
        &need,
        &mut t,
        ("back", "day", "d1"),
        25,
        &[trained(PERSON, LEFT)],
    );
    review(
        &need,
        &mut t,
        ("front", "day", "d1"),
        40,
        &[trained(PERSON, RIGHT)],
    );
    let one = [cb(PERSON, RIGHT, 100, STRONG)];
    // On `back`, at RIGHT (no reviewed box of it there), a day it has: 65 overall, 25 on camera.
    let got = score(&need, &t, &at("back", "day", "d1"), &one);
    assert_eq!(got, expect(1.0, 0.0, 65.0, 25.0, 1.0));
    // A camera with none: its own cell pulls fully (and the day is new to it).
    let fresh = score(&need, &t, &at("side", "day", "d1"), &one);
    assert_eq!(fresh, expect(1.0, 0.0, 65.0, 0.0, 1.0 + W_DAY));
    assert!(fresh > got);
    // Diminishing: more boxes on `back`, less need there.
    review(
        &need,
        &mut t,
        ("back", "day", "d1"),
        30,
        &[trained(PERSON, LEFT)],
    );
    let later = score(&need, &t, &at("back", "day", "d1"), &one);
    assert_eq!(later, expect(1.0, 0.0, 95.0, 55.0, 1.0));
    assert!(later < got);
    // Another scene is another cell.
    let night = score(&need, &t, &at("back", "night", "d1"), &one);
    assert_eq!(night, expect(1.0, 0.0, 0.0, 0.0, 1.0 + W_DAY));
}

/// A table with one parked spot of cars on `drive`: [0.1, 0.1, 0.3, 0.3] of the frame.
fn parked() -> statics::Table {
    let obs = (0..12).map(|i| statics::Obs {
        camera: "drive".into(),
        frame: format!("f{i}"),
        day: if i < 6 { "d1" } else { "d2" }.into(),
        scene: "day".into(),
        label: "car".into(),
        bbox: [0.1, 0.1, 0.3, 0.3],
    });
    let t = statics::Table::build(obs);
    assert_eq!(t.len(), 1, "{t:?}");
    t
}

/// [0.1, 0.1, 0.3, 0.3] of a 640 x 480 frame.
const SPOT: [u16; 4] = [64, 48, 192, 144];

#[test]
fn a_box_at_a_parked_spot_counts_a_quarter_but_whole_for_the_floor() {
    let c = classes();
    let steer = Steer {
        spots: Arc::new(parked()),
        ..Steer::default()
    };
    let need = Need::new(Some(wanted(&c)), Some(&steer));
    let mut t = CellTally::default();
    // 30 reviewed parked cars: 30 quarter-boxes, 7.5 boxes. A car elsewhere: 4.
    review(
        &need,
        &mut t,
        ("drive", "day", "d1"),
        30,
        &[trained(CAR, SPOT)],
    );
    review(
        &need,
        &mut t,
        ("drive", "day", "d1"),
        1,
        &[trained(CAR, RIGHT)],
    );
    assert_eq!(t.class_scene[&(CAR, 0)], 30 + 4);
    // Another camera has no spot there: a full box.
    review(
        &need,
        &mut t,
        ("gate", "day", "d1"),
        1,
        &[trained(CAR, SPOT)],
    );
    assert_eq!(t.class_scene[&(CAR, 0)], 38);
    // The floor counts whole boxes: 32 cars, none rare.
    assert_eq!(t.class_boxes[&CAR], 32);
    let s = Scorer::new(&need, &t);
    assert!(!s.below_floor(CAR));
    let car = |bbox| [cb(CAR, bbox, 100, STRONG)];
    let (n_ks, n_mks) = (38.0 / 4.0, 34.0 / 4.0);
    // The spot's region holds 7.5 boxes.
    let at_spot = score(&need, &t, &at("drive", "day", "d1"), &car(SPOT));
    assert_eq!(at_spot, expect(W_SPOT, 7.5, n_ks, n_mks, 1.0));
    let elsewhere = score(&need, &t, &at("drive", "day", "d1"), &car(RIGHT));
    assert_eq!(elsewhere, expect(1.0, 1.0, n_ks, n_mks, 1.0));
    let d = s.score(&at("drive", "day", "d1"), &car(SPOT), &[]);
    assert!(d.spot);
    // A parked box beside a car elsewhere carries under a quarter of the class: no spot cue.
    let both = [cb(CAR, RIGHT, 100, STRONG), cb(CAR, SPOT, 100, STRONG)];
    assert!(!s.score(&at("drive", "day", "d1"), &both, &[]).spot);
    // Without the table: neutral.
    let plain = Need::new(Some(wanted(&c)), None);
    let mut u = CellTally::default();
    review(
        &plain,
        &mut u,
        ("drive", "day", "d1"),
        30,
        &[trained(CAR, SPOT)],
    );
    assert_eq!(u.class_scene[&(CAR, 0)], 120);
    // A person never parks.
    let person = [cb(PERSON, SPOT, 100, STRONG)];
    assert_eq!(
        score(&need, &t, &at("drive", "day", "d1"), &person),
        expect(1.0, 0.0, 0.0, 0.0, FRESH)
    );
}

#[test]
fn a_new_region_and_a_new_day_add_need() {
    let c = classes();
    let need = Need::new(Some(wanted(&c)), None);
    let mut t = CellTally::default();
    review(
        &need,
        &mut t,
        ("back", "day", "d1"),
        25,
        &[trained(PERSON, LEFT)],
    );
    let f = at("back", "day", "d1");
    let old = score(&need, &t, &f, &[cb(PERSON, LEFT, 100, STRONG)]);
    let new = score(&need, &t, &f, &[cb(PERSON, RIGHT, 100, STRONG)]);
    assert_eq!(old, expect(1.0, 25.0, 25.0, 25.0, 1.0));
    assert_eq!(new, expect(1.0, 0.0, 25.0, 25.0, 1.0));
    assert!(new > old);
    let new_day = score(
        &need,
        &t,
        &at("back", "day", "d2"),
        &[cb(PERSON, LEFT, 100, STRONG)],
    );
    assert_eq!(new_day, expect(1.0, 25.0, 25.0, 25.0, 1.0 + W_DAY));
    // A frame of unknown size has one region of its own.
    assert_eq!(region(&cb(PERSON, LEFT, 0, 0), 0, 480), NO_REGION);
    assert_eq!(region(&cb(PERSON, LEFT, 0, 0), 640, 480), 0);
    assert_eq!(region(&cb(PERSON, RIGHT, 0, 0), 640, 480), 8);
    assert_eq!(region(&cb(PERSON, [300, 200, 340, 260], 0, 0), 640, 480), 4);
}

#[test]
fn a_missed_box_doubles_only_with_evidence_and_a_camera_that_keeps_the_class() {
    let c = classes();
    let need = Need::new(Some(wanted(&c)), None);
    let mut t = CellTally::default();
    review(
        &need,
        &mut t,
        ("back", "day", "d1"),
        25,
        &[trained(PERSON, LEFT)],
    );
    let f = at("back", "day", "d1");
    let plain = score(&need, &t, &f, &[cb(PERSON, RIGHT, 100, STRONG)]);
    let missed = score(&need, &t, &f, &[cb(PERSON, RIGHT, 100, STRONG | MISSED)]);
    assert_eq!(plain, expect(1.0, 0.0, 25.0, 25.0, 1.0));
    assert_eq!(missed, expect(W_MISSED, 0.0, 25.0, 25.0, 1.0));
    // A weak missed box (evidence under 0.5): no doubling.
    let weak = score(&need, &t, &f, &[cb(PERSON, RIGHT, 40, MISSED)]);
    assert_eq!(weak, expect(0.4, 0.0, 25.0, 25.0, 1.0));
    // A camera that rejected most person proposals: no doubling either.
    let mut u = CellTally::default();
    review(
        &need,
        &mut u,
        ("gate", "day", "d1"),
        25,
        &[trained(PERSON, LEFT)],
    );
    review(
        &need,
        &mut u,
        ("gate", "day", "d1"),
        25,
        &[proposed(PERSON)],
    );
    let s = Scorer::new(&need, &u);
    let d = s.score(
        &at("gate", "day", "d1"),
        &[cb(PERSON, RIGHT, 100, STRONG | MISSED)],
        &[],
    );
    assert!(!d.missed, "{d:?}");
    // Five strong boxes in a new place count as two (the first whole, the next two at a half).
    let five: Vec<CellBox> = (0..5).map(|_| cb(PERSON, RIGHT, 100, STRONG)).collect();
    assert_eq!(score(&need, &t, &f, &five), score_of(2.0 * v(25.0, 25.0)));
    let two: Vec<CellBox> = (0..2).map(|_| cb(PERSON, RIGHT, 100, STRONG)).collect();
    assert_eq!(score(&need, &t, &f, &two), score_of(1.5 * v(25.0, 25.0)));
}

#[test]
fn further_boxes_of_a_class_count_less() {
    assert_eq!(crowd(0.0), 0.0);
    assert_eq!(crowd(0.5), 0.5);
    assert_eq!(crowd(1.0), 1.0);
    assert_eq!(crowd(1.5), 1.25);
    assert_eq!(crowd(3.0), 2.0);
    assert_eq!(crowd(10.0), 2.0);
    assert_eq!((MORE_WEIGHT, MORE_BOXES), (0.5, 2.0));
}

fn vlm(ok: bool) -> VlmCheck {
    VlmCheck {
        model: "q".into(),
        answer: if ok { "yes" } else { "no" }.into(),
        expected: "yes".into(),
        ok,
    }
}

#[test]
fn the_evidence_table() {
    let auto = |edit: &dyn Fn(&mut AutoInfo)| {
        let mut a = AutoInfo::default();
        edit(&mut a);
        a
    };
    // Lone weak: sqrt(score) x W_LONE.
    assert_eq!(evidence(Some(0.64), None), (40, false, false));
    assert_eq!(evidence(None, None), (50, false, false));
    let lone = auto(&|a| {
        a.labelers.insert("gdino".into(), 0.64);
    });
    assert_eq!(evidence(Some(0.64), Some(&lone)), (40, false, false));
    // Strong: two labelers, agree, likely, a VLM yes.
    let strong = [
        auto(&|a| {
            a.labelers.insert("gdino".into(), 0.3);
            a.labelers.insert("rfdetr".into(), 0.3);
        }),
        auto(&|a| a.agree = Some(true)),
        auto(&|a| a.likely = true),
        auto(&|a| a.vlm = Some(vlm(true))),
    ];
    for a in &strong {
        assert_eq!(evidence(Some(0.1), Some(a)), (100, true, false), "{a:?}");
    }
    // SigLIP's species scores name an animal, they do not show there is one: no strong box.
    let named = auto(&|a| {
        a.labelers.insert("sam3".into(), 0.64);
        a.species_scores.insert("raccoon".into(), 0.9);
        a.species_scores.insert("cat".into(), 0.1);
    });
    assert_eq!(evidence(Some(0.64), Some(&named)), (40, false, false));
    // A VLM no: lower still, never zero; and two labelers agreeing on it make no strong box.
    let no = auto(&|a| a.vlm = Some(vlm(false)));
    assert_eq!(evidence(Some(0.64), Some(&no)), (10, false, true));
    let agreed_no = auto(&|a| {
        a.labelers.insert("gdino".into(), 0.3);
        a.labelers.insert("rfdetr".into(), 0.3);
        a.vlm = Some(vlm(false));
    });
    assert_eq!(evidence(Some(0.1), Some(&agreed_no)), (25, false, true));
    // An older run's category answer counts by its `ok`: `plant` for an animal is a no.
    let plant = auto(&|a| {
        a.vlm = Some(VlmCheck {
            model: "q".into(),
            answer: "plant".into(),
            expected: "animal".into(),
            ok: false,
        });
        a.species_scores.insert("raccoon".into(), 0.84);
    });
    assert_eq!(evidence(Some(0.64), Some(&plant)), (10, false, true));
    let animal = auto(&|a| {
        a.vlm = Some(VlmCheck {
            model: "q".into(),
            answer: "animal".into(),
            expected: "animal".into(),
            ok: true,
        });
    });
    assert_eq!(evidence(Some(0.1), Some(&animal)), (100, true, false));
    // A decided box is 1, and training.
    let mut b = auto_box("raccoon");
    b.state = BoxState::Accepted;
    let f = LabelFile {
        boxes: vec![b],
        ..LabelFile::default()
    };
    let c = classes();
    assert_eq!(
        of_file(&f, Some(&wanted(&c))),
        [cb(RACCOON, [10, 10, 50, 50], 100, TRAINING | STRONG)]
    );
}

/// `n` reviewed frames of `camera` at night that proposed class `k`, `kept` of them keeping it
/// (the others rejected it, a hard negative each).
fn proposals(need: &Need<'_>, t: &mut CellTally, camera: &str, k: u16, n: usize, kept: usize) {
    for i in 0..n {
        let boxes = if i < kept {
            vec![trained(k, LEFT), proposed(k)]
        } else {
            vec![rejected(k), proposed(k)]
        };
        review(need, t, (camera, "night", "d1"), 1, &boxes);
    }
}

#[test]
fn rejects_lower_weak_boxes_and_more_gently_strong_ones() {
    let c = classes();
    let need = Need::new(Some(wanted(&c)), None);
    let mut t = CellTally::default();
    // On `back`, cats were proposed on 9 reviewed frames and kept on 1.
    proposals(&need, &mut t, "back", CAT, 9, 1);
    let f = at("back", "night", "d1");
    let s = Scorer::new(&need, &t);
    let (kw, ks) = (2.0 / 10.0, 6.0 / 14.0);
    let vv = v(1.0, 1.0);
    // 8 hard negatives of cat on `back`.
    let hard = T_H / (T_H + 8.0);
    let weak = s.score(&f, &[cb(CAT, RIGHT, 40, 0)], &[]);
    let want = 0.4 * kw * vv * RARE_GAIN + W_HARDNEG * (1.0 - kw) * hard * vv;
    assert_eq!(weak.score, score_of(want), "{weak:?}");
    let strong = s.score(&f, &[cb(CAT, RIGHT, 100, STRONG)], &[]);
    let want = ks * vv * RARE_GAIN + W_HARDNEG * (1.0 - ks) * hard * vv;
    assert_eq!(strong.score, score_of(want), "{strong:?}");
    assert_eq!(
        strong.kept,
        Some(Kept {
            kept: 1,
            proposed: 9
        })
    );
    // Another camera's proposals say nothing here: no kept share, no hard negatives.
    let side = s.score(&at("side", "night", "d1"), &[cb(CAT, RIGHT, 40, 0)], &[]);
    assert_eq!(side.score, expect(0.4, 0.0, 1.0, 0.0, FRESH));
    assert_eq!((side.kept, side.hard), (None, 0.0));
    // A decided box is never lowered, and names no kept share.
    let decided = s.score(&f, &[trained(CAT, RIGHT)], &[]);
    assert_eq!(decided.kept, None);
    assert_eq!(decided.hard, 0.0);
}

#[test]
fn thirty_rejected_strong_proposals_lower_a_cameras_strong_need_fivefold() {
    let c = classes();
    let need = Need::new(Some(wanted(&c)), None);
    let mut t = CellTally::default();
    proposals(&need, &mut t, "back", MOTORCYCLE, 30, 0);
    let s = Scorer::new(&need, &t);
    let one = [cb(MOTORCYCLE, RIGHT, 100, STRONG)];
    let back = s.score(&at("back", "night", "d1"), &one, &[]);
    let side = s.score(&at("side", "night", "d1"), &one, &[]);
    // The need part (before the hard-negative share) of `back`: κ = 5/35.
    let need_back = f64::from(back.score) / 1000.0 - back.hard;
    let need_side = f64::from(side.score) / 1000.0 - side.hard;
    assert_eq!(side.hard, 0.0);
    assert!(need_side >= 5.0 * need_back, "{need_side} {need_back}");
    // `side` is weighed as a camera with no history of the class; a strong box keeps the floor's
    // gain although the class is mostly false (30 proposals, none kept).
    assert_eq!(side.score, expect(1.0, 0.0, 0.0, 0.0, FRESH));
    assert!(side.below_floor && back.below_floor);
    // A lone weak box of the mostly false class does not.
    let weak = s.score(
        &at("side", "night", "d1"),
        &[cb(MOTORCYCLE, RIGHT, 40, 0)],
        &[],
    );
    assert!(!weak.below_floor);
    assert_eq!(weak.score, expect(0.4, 0.0, 0.0, 0.0, 1.0 + W_DAY));
}

#[test]
fn a_class_under_the_floor_comes_first_until_it_is_mostly_false() {
    let c = classes();
    let need = Need::new(Some(wanted(&c)), None);
    let mut t = CellTally::default();
    // 19 raccoons over two cameras and scenes: rare. One more: not.
    review(
        &need,
        &mut t,
        ("back", "night", "d1"),
        10,
        &[trained(RACCOON, LEFT)],
    );
    review(
        &need,
        &mut t,
        ("side", "ir", "d1"),
        9,
        &[trained(RACCOON, LEFT)],
    );
    let f = at("yard", "day", "d1");
    let one = [cb(RACCOON, LEFT, 100, STRONG)];
    let d = Scorer::new(&need, &t).score(&f, &one, &[]);
    assert!(d.below_floor);
    assert_eq!(d.score, expect(1.0, 0.0, 0.0, 0.0, FRESH));
    review(
        &need,
        &mut t,
        ("side", "ir", "d1"),
        1,
        &[trained(RACCOON, LEFT)],
    );
    let d = Scorer::new(&need, &t).score(&f, &one, &[]);
    assert!(!d.below_floor);
    assert_eq!(d.score, expect(1.0, 0.0, 0.0, 0.0, 1.0 + W_DAY));
    // Cats proposed on 20 reviewed frames over two cameras, kept on 3 (under a fifth): mostly
    // false, so a lone weak cat loses the floor although 3 boxes are far under it, while a strong
    // cat keeps it. Kept on 4 (a fifth): rare for both.
    let mut u = CellTally::default();
    proposals(&need, &mut u, "back", CAT, 10, 3);
    proposals(&need, &mut u, "side", CAT, 10, 0);
    let s = Scorer::new(&need, &u);
    assert!(s.below_floor(CAT) && s.mostly_false(CAT));
    assert!(!s.floor_gain(CAT, false) && s.floor_gain(CAT, true));
    let yard = at("yard", "night", "d1");
    assert!(!s.score(&yard, &[cb(CAT, RIGHT, 40, 0)], &[]).below_floor);
    assert!(
        s.score(&yard, &[cb(CAT, RIGHT, 100, STRONG)], &[])
            .below_floor
    );
    let mut w = CellTally::default();
    proposals(&need, &mut w, "back", CAT, 10, 4);
    proposals(&need, &mut w, "side", CAT, 10, 0);
    let s = Scorer::new(&need, &w);
    assert!(s.floor_gain(CAT, false) && s.floor_gain(CAT, true));
    assert_eq!((CLASS_FLOOR, FALSE_MIN_PROPOSED), (20, 20));
}

#[test]
fn an_empty_frame_has_a_prior_raised_by_the_cameras_animals() {
    let c = classes();
    let need = Need::new(Some(wanted(&c)), None);
    let mut t = CellTally::default();
    // 50 empty reviewed night frames on `back`, and 25 with a raccoon: a third with an animal.
    review(&need, &mut t, ("back", "night", "d1"), 50, &[]);
    review(
        &need,
        &mut t,
        ("back", "night", "d1"),
        25,
        &[trained(RACCOON, LEFT)],
    );
    let s = Scorer::new(&need, &t);
    let d = s.score(&at("back", "night", "d1"), &[], &[]);
    let share = 25.0 / 75.0;
    assert_eq!(
        d.score,
        score_of(W_EMPTY * T / (T + 50.0) * (1.0 + ANIMAL_GAIN * share))
    );
    assert_eq!(d.empty, Some((50, share)));
    // A camera without animals: the prior alone.
    let gate = s.score(&at("gate", "night", "d1"), &[], &[]);
    assert_eq!(gate.score, score_of(W_EMPTY * T / (T + 50.0)));
    // Day has no reviewed empty frame.
    assert_eq!(
        s.score(&at("gate", "day", "d1"), &[], &[]).score,
        score_of(W_EMPTY)
    );
    let n = s.explain(&d, "back", "night", None);
    assert_eq!(n.class, None);
    assert_eq!(n.empty_reviewed, Some(50));
    assert!(!n.new_region);
}

/// A reviewed frame of [`the_quarter_box_tally_is_the_same_in_any_order`].
type Frame = (&'static str, &'static str, &'static str, Vec<CellBox>);

#[test]
fn the_quarter_box_tally_is_the_same_in_any_order() {
    let c = classes();
    let steer = Steer {
        spots: Arc::new(parked()),
        ..Steer::default()
    };
    let need = Need::new(Some(wanted(&c)), Some(&steer));
    let frames: Vec<Frame> = vec![
        (
            "drive",
            "day",
            "d1",
            vec![trained(CAR, SPOT), proposed(CAR)],
        ),
        (
            "drive",
            "night",
            "d2",
            vec![trained(CAR, RIGHT), trained(PERSON, LEFT)],
        ),
        ("back", "ir", "d1", vec![rejected(CAT), proposed(CAT)]),
        (
            "back",
            "ir",
            "d3",
            vec![trained(RACCOON, LEFT), proposed(RACCOON), proposed(CAT)],
        ),
        ("gate", "other", "d1", vec![trained(PERSON, RIGHT)]),
    ];
    let add = |t: &mut CellTally, f: &Frame| {
        need.add_reviewed(
            t,
            &Reviewed {
                camera: f.0,
                scene: f.1,
                day: f.2,
                width: 640,
                height: 480,
                boxes: &f.3,
                // The day d1 held out: its boxes count twice, apart.
                heldout: f.2 == "d1",
            },
        );
    };
    let mut forward = CellTally::default();
    for f in &frames {
        add(&mut forward, f);
    }
    let mut backward = CellTally::default();
    for f in frames.iter().rev() {
        add(&mut backward, f);
    }
    assert_eq!(forward, backward);
    // And in two parts merged, either way round.
    let (mut a, mut b) = (CellTally::default(), CellTally::default());
    for (i, f) in frames.iter().enumerate() {
        add(if i % 2 == 0 { &mut a } else { &mut b }, f);
    }
    let mut ab = a.clone();
    ab.merge(b.clone());
    let mut ba = b;
    ba.merge(a);
    assert_eq!(ab, forward);
    assert_eq!(ba, forward);
    assert_eq!(forward.empty, [0, 0, 1, 0]);
    assert_eq!(forward.class_boxes[&CAR], 2);
    assert_eq!(forward.cameras["back"].hard[&CAT], 1);
    assert_eq!(
        forward.cameras["back"].precision[&CAT],
        Kept {
            kept: 0,
            proposed: 2
        }
    );
    let counts = forward.counts(&c);
    assert_eq!(counts.boxes["car"]["day"], 0.25);
    assert_eq!(counts.boxes["person"]["other"], 1.0);
    assert_eq!(counts.empty["ir"], 1);
    // Held-out quarter-boxes, as the training ones (the parked car a quarter), d1 only.
    assert_eq!(forward.cameras["drive"].heldout[&(CAR, 0)], SPOT_QUARTERS);
    assert_eq!(forward.cameras["gate"].heldout[&(PERSON, 3)], BOX_QUARTERS);
    assert!(forward.cameras["back"].heldout.is_empty());
    assert!(!forward.cameras["drive"].heldout.contains_key(&(CAR, 1)));
}

/// A measured table of run `run-n` (person, car, cat; it lacks raccoon and motorcycle, and dropped
/// dog). Night: person 10 of 60 missed, 20 of 40 on `back`. Overall: person 30 of 100, cat 10 of
/// 30, car 5 of 50; `back` person 20 of 40 and cat 10 of 30, `side` 5 of 25 of each.
fn measured() -> Measured {
    let run = Run {
        id: "run-n".into(),
        classes: vec!["person".into(), "car".into(), "cat".into()],
        rename: [("dog".to_string(), String::new())].into_iter().collect(),
    };
    let cr = ClassRecall::new;
    let night = Slice::new(
        run.clone(),
        &[cr("person", 60, 10), cr("cat", 30, 10)],
        [("back", &[cr("person", 40, 20), cr("cat", 30, 10)][..])],
    );
    let overall = Slice::new(
        run,
        &[cr("person", 100, 30), cr("cat", 30, 10), cr("car", 50, 5)],
        [
            ("back", &[cr("person", 40, 20), cr("cat", 30, 10)][..]),
            ("side", &[cr("person", 25, 5), cr("cat", 25, 5)][..]),
        ],
    );
    Measured {
        scenes: [("night".to_string(), night)].into_iter().collect(),
        overall: Some(overall),
    }
}

#[test]
fn measured_recall_raises_need_and_a_class_the_model_lacks_needs_most() {
    let c = classes();
    let m = Arc::new(measured());
    let steer = Steer {
        recall: Some(m.clone()),
        ..Steer::default()
    };
    let need = Need::new(Some(wanted(&c)), Some(&steer));
    let t = CellTally::default();
    let s = Scorer::new(&need, &t);
    let f = at("back", "night", "d1");
    // Person on `back`: r = 1 + RECALL_GAIN x the shrunk miss rate; the camera's lowest class.
    let rate = m.miss("back", "night", "person").unwrap().rate.unwrap();
    let person = [cb(PERSON, RIGHT, 100, STRONG)];
    let d = s.score(&f, &person, &[]);
    assert!(d.catch_up);
    assert_eq!(
        d.score,
        expect(
            1.0,
            0.0,
            0.0,
            0.0,
            FRESH * (1.0 + RECALL_GAIN * rate) * CATCHUP_GAIN
        )
    );
    let n = s.explain(&d, "back", "night", None);
    close(n.recall.unwrap(), four(1.0 - rate));
    assert_eq!(n.heldout_gt, Some(40));
    assert_eq!(n.run.as_deref(), Some("run-n"));
    assert!(n.catch_up && !n.model_lacks);
    // Raccoon: the model has none. The most need, and catch-up.
    let raccoon = [cb(RACCOON, RIGHT, 100, STRONG)];
    let d = s.score(&f, &raccoon, &[]);
    assert!(d.catch_up);
    assert_eq!(
        d.score,
        expect(
            1.0,
            0.0,
            0.0,
            0.0,
            FRESH * (1.0 + RECALL_GAIN) * CATCHUP_GAIN
        )
    );
    let n = s.explain(&d, "back", "night", Some(RACCOON));
    assert!(n.model_lacks);
    assert_eq!((n.recall, n.heldout_gt), (None, None));
    assert_eq!(n.visit_class.as_deref(), Some("raccoon"));
    assert!(n.expanded);
    // Dog: dropped on purpose. No recall information.
    let d = s.score(&f, &[cb(DOG, RIGHT, 100, STRONG)], &[]);
    assert!(!d.catch_up);
    assert_eq!(d.score, expect(1.0, 0.0, 0.0, 0.0, FRESH));
    // Day has no measured run: r = 1 (catch-up is over every scene and still applies).
    let d = s.score(&at("back", "day", "d1"), &person, &[]);
    assert_eq!(d.score, expect(1.0, 0.0, 0.0, 0.0, FRESH * CATCHUP_GAIN));
    // No steer at all: r = U = 1.
    let plain = Need::new(Some(wanted(&c)), None);
    let d = Scorer::new(&plain, &t).score(&f, &raccoon, &[]);
    assert_eq!(d.score, expect(1.0, 0.0, 0.0, 0.0, FRESH));
}

#[test]
fn catch_up_fills_the_lowest_class_of_each_camera() {
    let m = measured();
    let classes = classes();
    let of = |camera: &str| -> Vec<bool> {
        let r: Vec<Option<CameraRecall>> = classes
            .iter()
            .map(|k| m.class_recall_on_camera(camera, k))
            .collect();
        catch_up_of(&r)
    };
    let recall = |camera: &str, class: &str| {
        m.class_recall_on_camera(camera, class)
            .unwrap()
            .recall
            .unwrap()
    };
    // `back`: person (40 held out) is lower than cat (30); car has no held-out car there (not
    // supported, excluded), raccoon and motorcycle are absent from the model, dog was dropped.
    assert!(recall("back", "person") < recall("back", "cat") - CATCHUP_TOL);
    assert_eq!(of("back"), [true, false, true, false, false, true]);
    // `side`: person and cat on 25 objects each, within the tolerance: both (a tie).
    assert!((recall("side", "person") - recall("side", "cat")).abs() <= CATCHUP_TOL);
    assert_eq!(of("side"), [true, false, true, true, false, true]);
    // A camera with nothing measured: only the absent classes.
    assert_eq!(of("gate"), [false, false, true, false, false, true]);
}

#[test]
fn candidates_of_a_file_and_of_a_reply() {
    let c = classes();
    let w = wanted(&c);
    let pred = |label: &str, bbox: [f32; 4]| LabelBox {
        source: Source::Model,
        origin: Some(Origin::Prediction),
        bbox,
        ..auto_box(label)
    };
    // A raccoon the camera model called a cat: the merge kept only the class suggestion.
    let mut cat = pred("cat", [100.0, 100.0, 200.0, 200.0]);
    cat.score = Some(0.81);
    cat.auto.auto_label = Some("raccoon".into());
    cat.auto.auto_score = Some(0.49);
    cat.auto.labelers.insert("gdino".into(), 0.49);
    // An auto box with a prediction on it, and one without.
    let mut seen = auto_box("person");
    seen.bbox = [300.0, 100.0, 340.0, 200.0];
    let person = pred("person", [302.0, 100.0, 340.0, 200.0]);
    let mut lost = auto_box("person");
    lost.bbox = [500.0, 100.0, 540.0, 200.0];
    // A below-threshold model box; a rejected box, an ignore region, an unwanted class.
    let below = LabelBox {
        origin: Some(Origin::BelowThreshold),
        ..pred("dog", [0.0, 300.0, 50.0, 350.0])
    };
    let rejected_dog = LabelBox {
        state: BoxState::Rejected,
        ..auto_box("dog")
    };
    let unsure = LabelBox {
        difficult: true,
        ..auto_box("dog")
    };
    let f = LabelFile {
        boxes: vec![
            cat,
            seen,
            person,
            lost,
            below,
            rejected_dog,
            unsure,
            auto_box("bird"),
        ],
        ..LabelFile::default()
    };
    // By class, in their order within one.
    assert_eq!(
        of_file(&f, Some(&w)),
        [
            cb(PERSON, [300, 100, 340, 200], 35, 0),
            cb(PERSON, [302, 100, 340, 200], 35, 0),
            cb(PERSON, [500, 100, 540, 200], 35, MISSED),
            // The raccoon: missed, sqrt(0.49) x 0.5.
            cb(RACCOON, [100, 100, 200, 200], 35, MISSED),
            // The cat itself: lone weak, sqrt(0.81) x 0.5; its findings are the raccoon's.
            cb(CAT, [100, 100, 200, 200], 45, 0),
            cb(DOG, [0, 300, 50, 350], 35, MISSED),
        ]
    );
    // Reviewed: the training boxes, the hard negatives and a marker per judged proposal.
    let mut r = f.clone();
    r.status = Status::Reviewed;
    r.boxes[3].state = BoxState::Accepted;
    r.boxes[0].state = BoxState::Rejected;
    assert_eq!(
        of_file(&r, Some(&w)),
        [
            // Pending boxes are left on it: no export's frame, so no parked-spot evidence.
            cb(PERSON, [0; 4], 0, UNSPOTTED),
            cb(PERSON, [500, 100, 540, 200], 100, TRAINING | STRONG),
            proposed(PERSON),
            cb(CAT, [100, 100, 200, 200], 0, REJECTED),
            proposed(CAT),
            cb(DOG, [10, 10, 50, 50], 0, REJECTED),
            proposed(DOG),
        ]
    );
    // No class list: nothing.
    assert!(of_file(&f, None).is_empty());
    // A frame without a label file: predictions, below-threshold and suppressed boxes.
    let det = |label: &str, score: f32, bbox: [f32; 4]| Det {
        label: label.into(),
        score,
        bbox,
    };
    let mut rec = empty_record();
    rec.predictions = vec![det("person", 0.81, [0.0, 0.0, 10.0, 10.0])];
    rec.below_threshold = vec![
        det("raccoon", 0.36, [100.0, 100.0, 150.0, 150.0]),
        det("person", 0.4, [0.0, 0.0, 10.0, 11.0]),
    ];
    rec.suppressed = vec![Suppressed {
        label: "cat".into(),
        score: 0.64,
        bbox: [200.0, 0.0, 260.0, 60.0],
        reason: "night_threshold".into(),
    }];
    assert_eq!(
        of_reply(&rec, Some(&w)),
        [
            cb(PERSON, [0, 0, 10, 10], 45, 0),
            // A prediction is on it: the camera model alerted, no miss.
            cb(PERSON, [0, 0, 10, 11], 32, 0),
            cb(RACCOON, [100, 100, 150, 150], 30, MISSED),
            cb(CAT, [200, 0, 260, 60], 40, MISSED),
        ]
    );
}

#[test]
fn weak_candidates_are_thinned_and_at_most_eight_of_a_class_kept() {
    let c = classes();
    let w = wanted(&c);
    // Twelve pending person boxes of rising score, and a dog of a VLM no (under a tenth: kept, it
    // may be the frame's hard negative).
    let mut boxes: Vec<LabelBox> = (0..12)
        .map(|i| LabelBox {
            score: Some(0.2 + 0.05 * i as f32),
            bbox: [i as f32 * 10.0, 0.0, i as f32 * 10.0 + 5.0, 5.0],
            ..auto_box("person")
        })
        .collect();
    let mut no = auto_box("dog");
    no.score = Some(0.1);
    no.auto.vlm = Some(vlm(false));
    boxes.push(no);
    let f = LabelFile {
        boxes,
        ..LabelFile::default()
    };
    let got = of_file(&f, Some(&w));
    assert_eq!(got.len(), MAX_PER_CLASS + 1, "{got:?}");
    let persons: Vec<&CellBox> = got.iter().filter(|b| b.class == PERSON).collect();
    assert_eq!(persons.len(), MAX_PER_CLASS);
    // The eight strongest, in their order.
    let xs: Vec<u16> = persons.iter().map(|b| b.bbox[0]).collect();
    assert_eq!(xs, [40, 50, 60, 70, 80, 90, 100, 110]);
    let dog = got.iter().find(|b| b.class == DOG).unwrap();
    assert!(dog.has(VLM_NO) && dog.e() < MIN_EVIDENCE, "{dog:?}");
}

#[test]
fn a_faint_box_counts_only_when_likely_false_and_then_for_its_hard_negative_alone() {
    let c = classes();
    let need = Need::new(Some(wanted(&c)), None);
    let t = CellTally::default();
    let s = Scorer::new(&need, &t);
    let f = at("back", "night", "d1");
    // A lone box of score 0.3 the VLM rejected: sqrt(0.3) · W_LONE · W_VLM_NO, under a tenth.
    let mut b = auto_box("dog");
    b.score = Some(0.3);
    b.auto.vlm = Some(vlm(false));
    let l = LabelFile {
        boxes: vec![b],
        ..LabelFile::default()
    };
    let w = wanted(&c);
    let boxes = of_file(&l, Some(&w));
    assert_eq!(boxes.len(), 1);
    assert!(boxes[0].e() < MIN_EVIDENCE && boxes[0].has(VLM_NO));
    let d = s.score(&f, &boxes, &[]);
    assert!(d.hard > 0.0 && d.hard_only, "{d:?}");
    assert!(
        d.empty.is_none() && d.top.is_none(),
        "not an empty frame: {d:?}"
    );
    // H alone: half a fresh box's v, times the VLM no's falseness, the region factor.
    let want = W_HARDNEG * crowd(VLM_NO_FALSE * v(0.0, 0.0) * rf(0.0));
    assert_eq!(d.score, score_of(want), "{d:?}");
    let n = s.explain(&d, "back", "night", None);
    assert!(
        n.hard_only && n.class.is_none() && n.empty_reviewed.is_none(),
        "{n:?}"
    );
    assert_eq!(n.hard_class.as_deref(), Some("dog"));
    // A faint box nobody calls false is no candidate: the frame weighs as an empty one.
    let faint = [cb(DOG, RIGHT, 5, 0)];
    let d = s.score(&f, &faint, &[]);
    assert!(d.empty.is_some() && !d.hard_only, "{d:?}");
    // On a camera that rejects the class (κ under a half), it is likely false: H only.
    let mut u = CellTally::default();
    proposals(&need, &mut u, "back", DOG, 9, 1);
    let d = Scorer::new(&need, &u).score(&f, &faint, &[]);
    assert!(d.hard_only && d.hard > 0.0, "{d:?}");
    // Beside a real box, a faint likely false one adds its share and no need.
    let real = [cb(CAT, LEFT, 100, STRONG)];
    let both = [cb(CAT, LEFT, 100, STRONG), cb(DOG, RIGHT, 6, VLM_NO)];
    let (a, ab) = (s.score(&f, &real, &[]), s.score(&f, &both, &[]));
    assert!(
        ab.score > a.score && !ab.hard_only && ab.top == Some(CAT),
        "{a:?} {ab:?}"
    );
}

#[test]
fn one_object_proposed_as_several_classes_scores_once() {
    let c = classes();
    let need = Need::new(Some(wanted(&c)), None);
    let t = CellTally::default();
    let s = Scorer::new(&need, &t);
    let f = at("back", "night", "d1");
    let one = expect(1.0, 0.0, 0.0, 0.0, FRESH);
    let animal = [
        cb(RACCOON, [100, 100, 200, 200], 100, STRONG),
        cb(CAT, [102, 100, 200, 200], 100, STRONG),
        cb(DOG, [100, 100, 198, 202], 100, STRONG),
    ];
    let d = s.score(&f, &animal, &[]);
    assert_eq!(d.score, one, "{d:?}");
    assert_eq!(d.top, Some(RACCOON), "ties to the first class");
    // A box inside a bigger one of another class is the same object too.
    let nested = [
        cb(CAT, [100, 100, 200, 200], 100, STRONG),
        cb(DOG, [110, 110, 180, 190], 100, STRONG),
    ];
    assert_eq!(s.score(&f, &nested, &[]).score, one);
    // Two animals apart are two.
    let two = [
        cb(CAT, [100, 100, 200, 200], 100, STRONG),
        cb(DOG, [400, 100, 500, 200], 100, STRONG),
    ];
    assert_eq!(
        s.score(&f, &two, &[]).score,
        score_of(2.0 * v(0.0, 0.0) * FRESH)
    );
    // A weak suggestion on a box: the object's best class only; a strong one wins it.
    let weak = [
        cb(CAT, [100, 100, 200, 200], 45, 0),
        cb(RACCOON, [100, 100, 200, 200], 35, MISSED),
    ];
    let d = s.score(&f, &weak, &[]);
    assert_eq!(
        (d.top, d.score),
        (Some(CAT), expect(0.45, 0.0, 0.0, 0.0, FRESH))
    );
    let strong = [
        cb(CAT, [100, 100, 200, 200], 45, 0),
        cb(RACCOON, [100, 100, 200, 200], 100, MISSED | STRONG),
    ];
    let d = s.score(&f, &strong, &[]);
    assert_eq!(d.top, Some(RACCOON));
    assert!(d.missed);
    assert_eq!(d.score, expect(W_MISSED, 0.0, 0.0, 0.0, FRESH));
}

#[test]
fn only_confusable_classes_of_similar_size_are_one_object() {
    let c = classes();
    let need = Need::new(Some(wanted(&c)), None);
    let t = CellTally::default();
    let s = Scorer::new(&need, &t);
    let f = at("back", "night", "d1");
    let one = expect(1.0, 0.0, 0.0, 0.0, FRESH);
    // A person on a motorcycle: two objects, two terms (different regions' factors aside, both
    // fresh), however much they overlap.
    let rider = [
        cb(PERSON, [100, 100, 200, 260], 100, STRONG),
        cb(MOTORCYCLE, [100, 150, 210, 270], 100, STRONG),
    ];
    let d = s.score(&f, &rider, &[]);
    assert!(d.score > one, "{d:?}");
    let alone = |b: CellBox| s.score(&f, &[b], &[]).score;
    let sum = |bs: &[CellBox]| bs.iter().map(|b| f64::from(alone(*b))).sum::<f64>();
    assert!((f64::from(d.score) - sum(&rider)).abs() <= 1.0, "{d:?}");
    // A decided person and a pending car on it: both kept.
    let by_car = [
        trained(PERSON, [100, 100, 200, 200]),
        cb(CAR, [100, 100, 200, 200], 100, STRONG),
    ];
    let d = s.score(&f, &by_car, &[]);
    assert!((f64::from(d.score) - sum(&by_car)).abs() <= 1.0, "{d:?}");
    // A cat, a dog and a raccoon proposed on one animal: one.
    let animal = [
        cb(CAT, [100, 100, 200, 200], 100, STRONG),
        cb(DOG, [101, 100, 200, 201], 100, STRONG),
        cb(RACCOON, [100, 99, 199, 200], 100, STRONG),
    ];
    assert_eq!(s.score(&f, &animal, &[]).score, one);
    // A cat inside a dog box of four times its area: two animals (a kitten by a dog).
    let sizes = [
        cb(CAT, [100, 100, 150, 150], 100, STRONG),
        cb(DOG, [100, 100, 200, 200], 100, STRONG),
    ];
    assert!(s.score(&f, &sizes, &[]).score > one);
    // Vehicles join: a car proposed as a truck-sized motorcycle is one vehicle.
    let vehicle = [
        cb(CAR, [100, 100, 300, 200], 100, STRONG),
        cb(MOTORCYCLE, [105, 100, 300, 205], 100, STRONG),
    ];
    assert_eq!(s.score(&f, &vehicle, &[]).score, one);
    // Two decided animals a reviewer drew on one place stay two; a pending one between them joins
    // one of them, and the decided class wins the object.
    let drawn = [
        trained(CAT, [100, 100, 200, 200]),
        trained(DOG, [102, 100, 200, 200]),
    ];
    assert_eq!(
        s.score(&f, &drawn, &[]).score,
        alone(drawn[0]) + alone(drawn[1])
    );
    let joined = [
        trained(DOG, [100, 100, 200, 200]),
        cb(RACCOON, [100, 100, 200, 200], 100, STRONG),
    ];
    let d = s.score(&f, &joined, &[]);
    assert_eq!(d.top, Some(DOG), "{d:?}");
    assert_eq!(d.score, alone(joined[0]));
}

#[test]
fn the_floor_follows_the_boxes_that_survive_their_objects() {
    let c = classes();
    let need = Need::new(Some(wanted(&c)), None);
    // Cats: mostly false (20 proposals, 3 kept) and under the floor.
    let mut t = CellTally::default();
    proposals(&need, &mut t, "back", CAT, 10, 3);
    proposals(&need, &mut t, "side", CAT, 10, 0);
    let s = Scorer::new(&need, &t);
    assert!(s.below_floor(CAT) && s.mostly_false(CAT));
    let f = at("yard", "night", "d1");
    // A strong cat that is one object with a strong raccoon (which wins it: the floor gains tie,
    // so the first class), and a weak cat elsewhere. The strong cat counts nowhere, so it gives
    // the weak one no floor: the frame scores as the raccoon and the weak cat alone.
    let with_strong = [
        cb(RACCOON, [100, 100, 200, 200], 100, STRONG),
        cb(CAT, [100, 100, 200, 200], 100, STRONG),
        cb(CAT, [400, 300, 500, 400], 40, 0),
    ];
    let without = [
        cb(RACCOON, [100, 100, 200, 200], 100, STRONG),
        cb(CAT, [400, 300, 500, 400], 40, 0),
    ];
    let (a, b) = (s.score(&f, &with_strong, &[]), s.score(&f, &without, &[]));
    assert_eq!(a.score, b.score, "{a:?} {b:?}");
    assert_eq!(a.top, Some(RACCOON));
}

#[test]
fn the_hard_negative_share_falls_as_rejections_come_and_a_real_object_outranks_it() {
    let c = classes();
    let need = Need::new(Some(wanted(&c)), None);
    // `bad` kept 1 of its cat proposals and rejected `rejects` more.
    let share = |rejects: usize| {
        let mut t = CellTally::default();
        proposals(&need, &mut t, "bad", CAT, rejects + 1, 1);
        let s = Scorer::new(&need, &t);
        s.score(&at("bad", "night", "d1"), &[cb(CAT, RIGHT, 40, 0)], &[])
            .hard
    };
    let (h5, h20, h80) = (share(5), share(20), share(80));
    assert!(h5 > 0.0 && h20 < h5 && h80 < h20, "{h5} {h20} {h80}");
    // A VLM no is likely false even on a camera that keeps the class.
    let mut t = CellTally::default();
    proposals(&need, &mut t, "good", CAT, 1, 1);
    let s = Scorer::new(&need, &t);
    let no = s.score(
        &at("good", "night", "d1"),
        &[cb(CAT, RIGHT, 40, VLM_NO)],
        &[],
    );
    assert!(no.hard > 0.0);
    let n = s.explain(&no, "good", "night", None);
    assert_eq!(n.hard_class.as_deref(), Some("cat"));
    assert!(n.hard_negative > 0.0);
    // At equal counts, a real object (a camera that keeps the class) outranks a likely false one.
    let mut t = CellTally::default();
    proposals(&need, &mut t, "good", CAT, 10, 10);
    proposals(&need, &mut t, "bad", CAT, 10, 0);
    review(
        &need,
        &mut t,
        ("bad", "night", "d1"),
        10,
        &[trained(CAT, LEFT)],
    );
    let s = Scorer::new(&need, &t);
    for (e, flags) in [(40, 0), (100, STRONG)] {
        let real = s.score(&at("good", "night", "d1"), &[cb(CAT, RIGHT, e, flags)], &[]);
        let fake = s.score(&at("bad", "night", "d1"), &[cb(CAT, RIGHT, e, flags)], &[]);
        assert!(fake.hard > 0.0 && real.hard == 0.0, "{real:?} {fake:?}");
        assert!(real.score > fake.score, "{real:?} {fake:?}");
    }
}

#[test]
fn a_gap_in_a_visit_is_found_and_the_visit_holds_its_decided_classes() {
    // One event: a raccoon decided at 0 s and 10 s, a strong person at 10 s only. At 5 s a frame
    // with nothing and one with the raccoon already boxed; one at 40 s, beyond GAP_S of both.
    let raccoon_at = |x: u16| trained(RACCOON, [x, 100, x + 50, 150]);
    let first: &'static [CellBox] = Box::leak(Box::new([raccoon_at(100)]));
    let last: &'static [CellBox] = Box::leak(Box::new([
        raccoon_at(200),
        cb(PERSON, [400, 100, 440, 200], 100, STRONG),
    ]));
    let covered: &'static [CellBox] =
        Box::leak(Box::new([cb(RACCOON, [150, 100, 200, 150], 30, 0)]));
    let frame = |ms: i64, boxes: &'static [CellBox]| DayFrame {
        ms: Some(ms),
        alone: false,
        boxes,
    };
    let frames = [
        frame(0, first),
        frame(5_000, &[]),
        frame(5_000, covered),
        frame(10_000, last),
        frame(40_000, &[]),
        // An import of its own: no gap, its own visit.
        DayFrame {
            alone: true,
            ..frame(5_000, &[])
        },
    ];
    let out = day(&frames);
    // Halfway between [100..150] and [200..250]: [150..200].
    assert_eq!(
        out[1].gaps,
        [cb(RACCOON, [150, 100, 200, 150], 50, GAP)],
        "{out:?}"
    );
    assert_eq!(out[1].visit_classes, [RACCOON]);
    // The raccoon is boxed there: no gap. The person is in one frame only.
    assert!(out[2].gaps.is_empty(), "{out:?}");
    assert!(out[4].gaps.is_empty(), "{out:?}");
    assert_eq!(out[4].visit_classes, [RACCOON]);
    assert!(out[5].gaps.is_empty() && out[5].visit_classes.is_empty());
    assert!(out[0].gaps.is_empty() && out[3].gaps.is_empty());
    // The order the frames come in changes nothing.
    let mut rev = frames.to_vec();
    rev.reverse();
    let mut back = day(&rev);
    back.reverse();
    assert_eq!(back, out);
    // A gap candidate weighs as a missed box.
    let c = classes();
    let need = Need::new(Some(wanted(&c)), None);
    let t = CellTally::default();
    let d = Scorer::new(&need, &t).score(&at("back", "night", "d1"), &[], &out[1].gaps);
    assert!(d.gap && d.top == Some(RACCOON));
    assert_eq!(d.score, expect(0.5 * W_MISSED, 0.0, 0.0, 0.0, FRESH));
}

#[test]
fn a_visit_expands_for_a_rare_or_catch_up_class() {
    let c = classes();
    let steer = Steer {
        recall: Some(Arc::new(measured())),
        ..Steer::default()
    };
    let need = Need::new(Some(wanted(&c)), Some(&steer));
    let mut t = CellTally::default();
    review(
        &need,
        &mut t,
        ("gate", "day", "d1"),
        25,
        &[trained(CAR, LEFT)],
    );
    review(
        &need,
        &mut t,
        ("gate", "day", "d1"),
        25,
        &[trained(PERSON, LEFT)],
    );
    let s = Scorer::new(&need, &t);
    // Raccoon: under the floor.
    assert_eq!(s.expands("gate", &[RACCOON]), Some(RACCOON));
    // Car: past the floor, no catch-up on `gate`.
    assert_eq!(s.expands("gate", &[CAR]), None);
    // Person: past the floor, but the catch-up class of `back`.
    assert_eq!(s.expands("gate", &[PERSON]), None);
    assert_eq!(s.expands("back", &[PERSON, CAR]), Some(PERSON));
    assert_eq!(s.expands("back", &[]), None);
    assert_eq!(visit_score(1000, 2, true), 666);
    assert_eq!(visit_score(1000, 2, false), 333);
    assert_eq!(visit_score(u32::MAX, 0, false), u32::MAX);
}

#[test]
fn a_candidate_is_twelve_bytes_and_its_parts_round() {
    assert_eq!(size_of::<CellBox>(), 12);
    assert_eq!(px(-3.0), 0);
    assert_eq!(px(f32::NAN), 0);
    assert_eq!(px(1e9), u16::MAX);
    assert_eq!(px(10.5), 11);
    assert_eq!(date_of("night/back/2026-10-21"), "2026-10-21");
    let c = classes();
    let w = wanted(&c);
    assert_eq!(class_index(&w, "Raccoon"), Some(RACCOON));
    assert_eq!(class_index(&w, "bird"), None);
    let mut set = ClassSet::default();
    for k in [0, 63, 64, 199] {
        set.insert(k);
    }
    assert_eq!(set.iter().collect::<Vec<_>>(), [0, 63, 64, 199]);
}

#[test]
fn a_steer_compares_by_identity_and_shows_a_summary() {
    let a = Steer {
        spots: Arc::new(parked()),
        ..Steer::default()
    };
    let b = a.clone();
    assert_eq!(a, b);
    let c = Steer {
        spots: Arc::new(parked()),
        ..Steer::default()
    };
    assert_ne!(a, c, "equal tables, but other snapshots");
    let shown = format!("{a:?}");
    assert!(
        shown.contains("spots: 1") && !shown.contains("bbox"),
        "{shown}"
    );
}

/// A steer with held-out steering on and `listed` camera days.
fn heldout_steer(listed: &[&str]) -> Steer {
    Steer {
        heldout_listed: Some(Arc::new(listed.iter().map(|s| (*s).to_string()).collect())),
        ..Steer::default()
    }
}

#[test]
fn a_day_is_held_out_when_listed_or_by_the_export_hash_with_steering_on() {
    use super::super::export::{DEFAULT_HOLDOUT_FRACTION, hash_heldout};
    let c = classes();
    let frac = f64::from(DEFAULT_HOLDOUT_FRACTION);
    let dates: Vec<String> = (0..60)
        .map(|d| format!("2026-{:02}-{:02}", 1 + d / 28, 1 + d % 28))
        .collect();
    let hashed = dates
        .iter()
        .find(|d| hash_heldout("back", d, frac))
        .unwrap();
    let mut open = dates.iter().filter(|d| !hash_heldout("back", d, frac));
    let (listed_day, other) = (open.next().unwrap(), open.next().unwrap());
    let listed = format!("back/{listed_day}");
    let steer = heldout_steer(&[&listed]);
    let on = Need::new(Some(wanted(&c)), Some(&steer));
    assert!(on.steers_heldout());
    // Per camera day, whatever the scene folder.
    for scene in ["day", "ir", "other"] {
        assert!(on.heldout_day(&format!("{scene}/back/{hashed}")), "hashed");
        assert!(
            on.heldout_day(&format!("{scene}/back/{listed_day}")),
            "listed"
        );
        assert!(!on.heldout_day(&format!("{scene}/back/{other}")));
    }
    // The listed key names a camera: another camera's same date is its own day.
    assert_eq!(
        on.heldout_day(&format!("day/gate/{listed_day}")),
        hash_heldout("gate", listed_day, frac)
    );
    // Not a day folder: no.
    assert!(!on.heldout_day(&format!("back/{hashed}")));
    assert!(!on.heldout_day(&format!("ir/back/{hashed}/x")));
    // Off without a steer, or with one that carries no list (the default): no day held out.
    let plain = Steer::default();
    for steer in [None, Some(&plain)] {
        let off = Need::new(Some(wanted(&c)), steer);
        assert!(!off.steers_heldout());
        assert!(!off.heldout_day(&format!("ir/back/{hashed}")));
        assert!(!off.heldout_day(&format!("ir/back/{listed_day}")));
    }
    // About the default share of days.
    let share = dates
        .iter()
        .filter(|d| hash_heldout("back", d, frac))
        .count();
    assert!((4..=24).contains(&share), "{share}");
}

#[test]
fn held_out_days_pull_until_the_cell_has_its_floor_then_training_days() {
    let c = classes();
    let steer = heldout_steer(&[]);
    let need = Need::new(Some(wanted(&c)), Some(&steer));
    let mut t = CellTally::default();
    // 25 cat boxes of back at IR (past the class floor), 7 of them on a held-out day.
    let cat = [trained(CAT, LEFT)];
    review_split(&need, &mut t, ("back", "ir", "d1"), true, 7, &cat);
    review_split(&need, &mut t, ("back", "ir", "d2"), false, 18, &cat);
    let held = Facts {
        heldout: true,
        ..at("back", "ir", "d1")
    };
    let train = at("back", "ir", "d2");
    let b = [cb(CAT, RIGHT, 100, STRONG)];
    let s = Scorer::new(&need, &t);
    let (dh, dt) = (s.score(&held, &b, &[]), s.score(&train, &b, &[]));
    assert_eq!(dh.score, expect(1.0, 0.0, 25.0, 25.0, HELDOUT_GAIN));
    assert_eq!(dt.score, expect(1.0, 0.0, 25.0, 25.0, 1.0));
    assert!(dh.heldout_gain && dh.heldout_day, "{dh:?}");
    assert!(!dt.heldout_gain && !dt.heldout_day, "{dt:?}");
    assert_eq!((dh.heldout_boxes, dt.heldout_boxes), (Some(7.0), Some(7.0)));
    let n = s.explain(&dh, "back", "ir", None);
    assert_eq!(
        (
            n.heldout_day,
            n.heldout_boxes,
            n.heldout_floor,
            n.heldout_gain
        ),
        (true, Some(7.0), Some(HELDOUT_FLOOR), true)
    );
    // The cell is per camera and scene: back's cats at night have no held-out box yet.
    let night = Facts {
        heldout: true,
        ..at("back", "night", "d1")
    };
    assert_eq!(s.score(&night, &b, &[]).heldout_boxes, Some(0.0));
    assert!(s.score(&night, &b, &[]).heldout_gain);
    // A parked box counts a quarter here too: 12 more held-out boxes at the spot are 3.
    let mut parked_t = t.clone();
    let spot_steer = Steer {
        spots: Arc::new(parked()),
        ..heldout_steer(&[])
    };
    let spot_need = Need::new(Some(wanted(&c)), Some(&spot_steer));
    review_split(
        &spot_need,
        &mut parked_t,
        ("drive", "day", "d1"),
        true,
        12,
        &[trained(CAR, SPOT)],
    );
    assert_eq!(
        parked_t.cameras["drive"].heldout[&(CAR, 0)],
        12 * SPOT_QUARTERS
    );
    // At the floor (13 more held-out boxes: 20), training days pull instead.
    review_split(&need, &mut t, ("back", "ir", "d1"), true, 13, &cat);
    let s = Scorer::new(&need, &t);
    let (dh, dt) = (s.score(&held, &b, &[]), s.score(&train, &b, &[]));
    assert_eq!(dh.score, expect(1.0, 0.0, 38.0, 38.0, 1.0));
    assert_eq!(dt.score, expect(1.0, 0.0, 38.0, 38.0, HELDOUT_GAIN));
    assert!(!dh.heldout_gain && dh.heldout_day && dt.heldout_gain);
    assert_eq!(dt.heldout_boxes, Some(20.0));
    // Without held-out steering: no gain either way, nothing shown.
    let plain = Steer::default();
    let off = Need::new(Some(wanted(&c)), Some(&plain));
    let s = Scorer::new(&off, &t);
    for f in [&held, &train] {
        let d = s.score(f, &b, &[]);
        assert_eq!(d.score, expect(1.0, 0.0, 38.0, 38.0, 1.0));
        assert!(!d.heldout_gain && d.heldout_boxes.is_none());
    }
}

#[test]
fn a_held_out_day_has_no_missed_bonus_and_no_gap() {
    let c = classes();
    let steer = heldout_steer(&[]);
    let need = Need::new(Some(wanted(&c)), Some(&steer));
    let mut t = CellTally::default();
    review(
        &need,
        &mut t,
        ("back", "day", "d1"),
        25,
        &[trained(PERSON, LEFT)],
    );
    let train = at("back", "day", "d1");
    let held = Facts {
        heldout: true,
        ..train
    };
    let missed = [cb(PERSON, RIGHT, 100, STRONG | MISSED)];
    let s = Scorer::new(&need, &t);
    // A training day of a cell short of held-out boxes: the miss doubles, no gain.
    let d = s.score(&train, &missed, &[]);
    assert_eq!(d.score, expect(W_MISSED, 0.0, 25.0, 25.0, 1.0));
    assert!(d.missed);
    // A held-out day: the gain, never the miss.
    let d = s.score(&held, &missed, &[]);
    assert_eq!(d.score, expect(1.0, 0.0, 25.0, 25.0, HELDOUT_GAIN));
    assert!(!d.missed && d.heldout_gain);
    // A gap: a candidate on a training day, none on a held-out one (an empty frame there).
    let gap = [cb(PERSON, RIGHT, 50, GAP)];
    let d = s.score(&train, &[], &gap);
    assert!(d.gap && d.top == Some(PERSON), "{d:?}");
    let d = s.score(&held, &[], &gap);
    assert!(
        d.top.is_none() && d.empty.is_some() && d.heldout_day,
        "{d:?}"
    ); // A likely false box: its hard-negative share on a training day, none on a held-out one.
    let false_box = [cb(PERSON, LEFT, 100, VLM_NO), cb(DOG, RIGHT, 6, VLM_NO)];
    let d = s.score(&train, &false_box, &[]);
    assert!(d.hard > 0.0, "{d:?}");
    let d = s.score(&held, &false_box, &[]);
    assert!(
        d.hard == 0.0 && d.hard_class.is_none() && !d.hard_only,
        "{d:?}"
    );
    // A faint likely false box alone: hard-negative only on a training day, an empty frame held out.
    let faint = [cb(DOG, RIGHT, 6, VLM_NO)];
    let d = s.score(&train, &faint, &[]);
    assert!(d.hard_only && d.hard > 0.0, "{d:?}");
    let d = s.score(&held, &faint, &[]);
    assert!(d.empty.is_some() && !d.hard_only && d.hard == 0.0, "{d:?}");
}
