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

fn classes() -> Vec<String> {
    ["person", "car", "raccoon", "cat", "dog"]
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
    }
}

/// `n` reviewed 640 x 480 frames of `camera` in `scene` on `day`, each with `boxes`.
fn review(
    need: &Need<'_>,
    t: &mut CellTally,
    (camera, scene, day): (&str, &str, &str),
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
                proposed: &[],
            },
        );
    }
}

/// The score of a frame at `f` with candidates `boxes`.
fn score(need: &Need<'_>, t: &CellTally, f: &Facts<'_>, boxes: &[CellBox]) -> u32 {
    Scorer::new(need, t).score(f, boxes, &[]).score
}

/// The score the formula gives one box of weight `a` (evidence and its box factors) in a region
/// with `region` reviewed boxes, of a class with `n_ks` and `n_mks` reviewed boxes, times `rest`
/// (day, floor, recall, catch-up).
fn expect(a: f64, region: f64, n_ks: f64, n_mks: f64, rest: f64) -> u32 {
    let v = 0.5 * T / (T + n_ks) + 0.5 * T_CAM / (T_CAM + n_mks);
    score_of(a * (1.0 + W_REGION * T_R / (T_R + region)) / 2.0 * v * rest)
}

fn wanted(c: &[String]) -> Wanted<'_> {
    Wanted {
        classes: c,
        hidden: &[],
    }
}

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
fn a_box_at_a_parked_spot_counts_a_quarter() {
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
    let car = |bbox| [cb(CAR, bbox, 100, STRONG)];
    let (n_ks, n_mks) = (38.0 / 4.0, 34.0 / 4.0);
    // Under the floor (9.5 boxes < 20): x RARE_GAIN. The spot's region holds 7.5 boxes.
    let at_spot = score(&need, &t, &at("drive", "day", "d1"), &car(SPOT));
    assert_eq!(at_spot, expect(W_SPOT, 7.5, n_ks, n_mks, RARE_GAIN));
    let elsewhere = score(&need, &t, &at("drive", "day", "d1"), &car(RIGHT));
    assert_eq!(elsewhere, expect(1.0, 1.0, n_ks, n_mks, RARE_GAIN));
    let d = Scorer::new(&need, &t).score(&at("drive", "day", "d1"), &car(SPOT), &[]);
    assert!(d.spot);
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
        expect(1.0, 0.0, 0.0, 0.0, (1.0 + W_DAY) * RARE_GAIN)
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
fn a_missed_box_doubles_and_a_frame_counts_up_to_the_cap() {
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
    // Five strong boxes in a new place count as BOX_CAP.
    let crowd: Vec<CellBox> = (0..5).map(|_| cb(PERSON, RIGHT, 100, STRONG)).collect();
    let v = 0.5 * T / (T + 25.0) + 0.5 * T_CAM / (T_CAM + 25.0);
    assert_eq!(score(&need, &t, &f, &crowd), score_of(BOX_CAP * v));
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
    assert_eq!(evidence(Some(0.64), None), (40, false));
    assert_eq!(evidence(None, None), (50, false));
    let lone = auto(&|a| {
        a.labelers.insert("gdino".into(), 0.64);
    });
    assert_eq!(evidence(Some(0.64), Some(&lone)), (40, false));
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
        assert_eq!(evidence(Some(0.1), Some(a)), (100, true), "{a:?}");
    }
    // SigLIP's species scores name an animal, they do not show there is one: no strong box.
    let named = auto(&|a| {
        a.labelers.insert("sam3".into(), 0.64);
        a.species_scores.insert("raccoon".into(), 0.9);
        a.species_scores.insert("cat".into(), 0.1);
    });
    assert_eq!(evidence(Some(0.64), Some(&named)), (40, false));
    // A VLM no: lower still, never zero; and two labelers agreeing on it make no strong box.
    let no = auto(&|a| a.vlm = Some(vlm(false)));
    assert_eq!(evidence(Some(0.64), Some(&no)), (10, false));
    let agreed_no = auto(&|a| {
        a.labelers.insert("gdino".into(), 0.3);
        a.labelers.insert("rfdetr".into(), 0.3);
        a.vlm = Some(vlm(false));
    });
    assert_eq!(evidence(Some(0.1), Some(&agreed_no)), (25, false));
    // An older run's category answer counts by its `ok`: `plant` for an animal is a no, SigLIP's
    // species score of the class (which sums to 1 over the animals whatever the crop) or not.
    let plant = auto(&|a| {
        a.vlm = Some(VlmCheck {
            model: "q".into(),
            answer: "plant".into(),
            expected: "animal".into(),
            ok: false,
        });
        a.species_scores.insert("raccoon".into(), 0.84);
    });
    assert_eq!(evidence(Some(0.64), Some(&plant)), (10, false));
    let animal = auto(&|a| {
        a.vlm = Some(VlmCheck {
            model: "q".into(),
            answer: "animal".into(),
            expected: "animal".into(),
            ok: true,
        });
    });
    assert_eq!(evidence(Some(0.1), Some(&animal)), (100, true));
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

#[test]
fn kappa_lowers_weak_boxes_only() {
    let c = classes();
    let need = Need::new(Some(wanted(&c)), None);
    let mut t = CellTally::default();
    // On `back`, cats were proposed on 9 reviewed frames and kept on 1: kappa = 2 / 10.
    for kept in [true, false, false, false, false, false, false, false, false] {
        let boxes = if kept {
            vec![trained(CAT, LEFT)]
        } else {
            vec![]
        };
        need.add_reviewed(
            &mut t,
            &Reviewed {
                camera: "back",
                scene: "night",
                day: "d1",
                width: 640,
                height: 480,
                boxes: &boxes,
                proposed: &[CAT],
            },
        );
    }
    let f = at("back", "night", "d1");
    let weak = score(&need, &t, &f, &[cb(CAT, RIGHT, 40, 0)]);
    let strong = score(&need, &t, &f, &[cb(CAT, RIGHT, 100, STRONG)]);
    assert_eq!(weak, expect(0.4 * 0.2, 0.0, 1.0, 1.0, RARE_GAIN));
    assert_eq!(strong, expect(1.0, 0.0, 1.0, 1.0, RARE_GAIN));
    // Another camera's proposals say nothing here.
    let side = score(
        &need,
        &t,
        &at("side", "night", "d1"),
        &[cb(CAT, RIGHT, 40, 0)],
    );
    assert_eq!(side, expect(0.4, 0.0, 1.0, 0.0, RARE_GAIN * (1.0 + W_DAY)));
    let d = Scorer::new(&need, &t).score(&f, &[cb(CAT, RIGHT, 40, 0)], &[]);
    assert_eq!(
        d.kept,
        Some(Kept {
            kept: 1,
            proposed: 9
        })
    );
}

#[test]
fn a_class_under_the_floor_comes_first() {
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
    assert_eq!(
        d.score,
        expect(1.0, 0.0, 0.0, 0.0, RARE_GAIN * (1.0 + W_DAY))
    );
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
    assert_eq!(CLASS_FLOOR, 20);
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
    let n = s.explain(&d, "back", "night", false);
    assert_eq!(n.class, None);
    assert_eq!(n.empty_reviewed, Some(50));
    assert!(!n.new_region);
}

/// A reviewed frame of [`the_quarter_box_tally_is_the_same_in_any_order`].
type Frame = (
    &'static str,
    &'static str,
    &'static str,
    Vec<CellBox>,
    Vec<u16>,
);

#[test]
fn the_quarter_box_tally_is_the_same_in_any_order() {
    let c = classes();
    let steer = Steer {
        spots: Arc::new(parked()),
        ..Steer::default()
    };
    let need = Need::new(Some(wanted(&c)), Some(&steer));
    let frames: Vec<Frame> = vec![
        ("drive", "day", "d1", vec![trained(CAR, SPOT)], vec![CAR]),
        (
            "drive",
            "night",
            "d2",
            vec![trained(CAR, RIGHT), trained(PERSON, LEFT)],
            vec![],
        ),
        ("back", "ir", "d1", vec![], vec![CAT]),
        (
            "back",
            "ir",
            "d3",
            vec![trained(RACCOON, LEFT)],
            vec![RACCOON, CAT],
        ),
        ("gate", "other", "d1", vec![trained(PERSON, RIGHT)], vec![]),
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
                proposed: &f.4,
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
    let counts = forward.counts(&c);
    assert_eq!(counts.boxes["car"]["day"], 0.25);
    assert_eq!(counts.boxes["person"]["other"], 1.0);
    assert_eq!(counts.empty["ir"], 1);
}

/// A measured table of run `run-n` (person, car, cat; it lacks raccoon and dropped dog). Night:
/// person 10 of 60 missed, 20 of 40 on `back`. Overall: person 30 of 100, cat 10 of 30, car 5 of
/// 50; `back` person 20 of 40 and cat 10 of 30, `side` 5 of 25 of each.
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
    let fresh = (1.0 + W_DAY) * RARE_GAIN;
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
            fresh * (1.0 + RECALL_GAIN * rate) * CATCHUP_GAIN
        )
    );
    let n = s.explain(&d, "back", "night", false);
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
            fresh * (1.0 + RECALL_GAIN) * CATCHUP_GAIN
        )
    );
    let n = s.explain(&d, "back", "night", false);
    assert!(n.model_lacks);
    assert_eq!((n.recall, n.heldout_gt), (None, None));
    // Dog: dropped on purpose. No recall information.
    let d = s.score(&f, &[cb(DOG, RIGHT, 100, STRONG)], &[]);
    assert!(!d.catch_up);
    assert_eq!(d.score, expect(1.0, 0.0, 0.0, 0.0, fresh));
    // Day has no measured run: r = 1 (catch-up is over every scene and still applies).
    let d = s.score(&at("back", "day", "d1"), &person, &[]);
    assert_eq!(d.score, expect(1.0, 0.0, 0.0, 0.0, fresh * CATCHUP_GAIN));
    // No steer at all: r = U = 1.
    let plain = Need::new(Some(wanted(&c)), None);
    let d = Scorer::new(&plain, &t).score(&f, &raccoon, &[]);
    assert_eq!(d.score, expect(1.0, 0.0, 0.0, 0.0, fresh));
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
    // supported, excluded), raccoon is absent from the model, dog was dropped.
    assert!(recall("back", "person") < recall("back", "cat") - CATCHUP_TOL);
    assert_eq!(of("back"), [true, false, true, false, false]);
    // `side`: person and cat on 25 objects each, within the tolerance: both (a tie).
    assert!((recall("side", "person") - recall("side", "cat")).abs() <= CATCHUP_TOL);
    assert_eq!(of("side"), [true, false, true, true, false]);
    // A camera with nothing measured: only the absent class.
    assert_eq!(of("gate"), [false, false, true, false, false]);
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
    let rejected = LabelBox {
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
            rejected,
            unsure,
            auto_box("bird"),
        ],
        ..LabelFile::default()
    };
    assert_eq!(
        of_file(&f, Some(&w)),
        [
            // The cat itself: lone weak, sqrt(0.81) x 0.5; its findings are the raccoon's.
            cb(CAT, [100, 100, 200, 200], 45, 0),
            // The raccoon: missed, sqrt(0.49) x 0.5.
            cb(RACCOON, [100, 100, 200, 200], 35, MISSED),
            cb(PERSON, [300, 100, 340, 200], 35, 0),
            cb(PERSON, [302, 100, 340, 200], 35, 0),
            cb(PERSON, [500, 100, 540, 200], 35, MISSED),
            cb(DOG, [0, 300, 50, 350], 35, MISSED),
        ]
    );
    // Reviewed: the training boxes only.
    let mut r = f.clone();
    r.status = Status::Reviewed;
    r.boxes[3].state = BoxState::Accepted;
    r.boxes[0].state = BoxState::Rejected;
    assert_eq!(
        of_file(&r, Some(&w)),
        [cb(PERSON, [500, 100, 540, 200], 100, TRAINING | STRONG)]
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
            cb(RACCOON, [100, 100, 150, 150], 30, MISSED),
            // A prediction is on it: the camera model alerted, no miss.
            cb(PERSON, [0, 0, 10, 11], 32, 0),
            cb(CAT, [200, 0, 260, 60], 40, MISSED),
        ]
    );
}

#[test]
fn a_suggested_class_counts_toward_its_class() {
    let c = classes();
    let need = Need::new(Some(wanted(&c)), None);
    let t = CellTally::default();
    // The raccoon suggestion of a "cat" box: its own class, missed.
    let boxes = [
        cb(CAT, [100, 100, 200, 200], 45, 0),
        cb(RACCOON, [100, 100, 200, 200], 35, MISSED),
    ];
    let d = Scorer::new(&need, &t).score(&at("back", "night", "d1"), &boxes, &[]);
    let fresh = (1.0 + W_DAY) * RARE_GAIN;
    let cat = expect(0.45, 0.0, 0.0, 0.0, fresh);
    let raccoon = expect(0.35 * W_MISSED, 0.0, 0.0, 0.0, fresh);
    assert!(
        d.score.abs_diff(cat + raccoon) <= 1,
        "{} {cat} {raccoon}",
        d.score
    );
    assert_eq!(d.top, Some(RACCOON));
    assert!(d.missed);
}

#[test]
fn a_gap_in_a_visit_is_found_and_the_visit_holds_its_decided_classes() {
    // One event: a raccoon decided at 0 s and 10 s, a strong person at 10 s only. At 5 s a frame
    // with nothing and one with the raccoon already boxed; one at 40 s, beyond GAP_S of both.
    let raccoon_at = |x: u16| trained(RACCOON, [x, 100, x + 50, 150]);
    let first = [raccoon_at(100)];
    let last = [
        raccoon_at(200),
        cb(PERSON, [400, 100, 440, 200], 100, STRONG),
    ];
    let covered = [cb(RACCOON, [150, 100, 200, 150], 30, 0)];
    let frame = |ms: i64, boxes: &'static [CellBox], listed: bool| DayFrame {
        ms: Some(ms),
        alone: false,
        boxes,
        listed,
    };
    let first: &'static [CellBox] = Box::leak(Box::new(first));
    let last: &'static [CellBox] = Box::leak(Box::new(last));
    let covered: &'static [CellBox] = Box::leak(Box::new(covered));
    let frames = [
        frame(0, first, false),
        frame(5_000, &[], true),
        frame(5_000, covered, true),
        frame(10_000, last, false),
        frame(40_000, &[], true),
        // An import of its own: no gap, its own visit.
        DayFrame {
            alone: true,
            ..frame(5_000, &[], true)
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
    assert_eq!(out[0], DayOut::default(), "not listed");
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
    assert_eq!(
        d.score,
        expect(0.5 * W_MISSED, 0.0, 0.0, 0.0, (1.0 + W_DAY) * RARE_GAIN)
    );
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
    assert!(s.expands("gate", &[RACCOON]));
    // Car: past the floor, no catch-up on `gate`.
    assert!(!s.expands("gate", &[CAR]));
    // Person: past the floor, but the catch-up class of `back`.
    assert!(!s.expands("gate", &[PERSON]));
    assert!(s.expands("back", &[PERSON, CAR]));
    assert!(!s.expands("back", &[]));
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
    assert_eq!(
        proposed_classes(["cat", "raccoon", "cat", "bird"], Some(&w)),
        [RACCOON, CAT]
    );
}
