//! Balancing on synthetic training frames.

use super::*;

fn classes(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

/// A frame `key` of `scene` on `camera`/`day` with `boxes` per class.
fn item(key: &str, scene: &str, camera: &str, day: &str, boxes: &[usize]) -> Item {
    Item {
        key: key.into(),
        base: 1,
        scene: scene.into(),
        camera: camera.into(),
        day: day.into(),
        boxes: boxes.to_vec(),
        hard_negative: false,
    }
}

fn on(axes: &[BalanceAxis], strength: f32, max_repeat: u32) -> Balance {
    Balance {
        axes: axes.iter().copied().collect(),
        strength,
        max_repeat,
        cap_dominant: None,
    }
}

/// `n` positive frames (one box of class 0) of `scene` on `camera`/`day`.
fn frames(tag: &str, n: usize, scene: &str, camera: &str, day: &str) -> Vec<Item> {
    (0..n)
        .map(|i| item(&format!("{tag}-{i}"), scene, camera, day, &[1]))
        .collect()
}

#[test]
fn the_weight_follows_the_formula_per_strength() {
    assert_eq!(weight(100, 25, 0.5), 2.0);
    assert_eq!(weight(100, 25, 1.0), 4.0);
    assert_eq!(weight(100, 25, 0.0), 1.0);
    assert_eq!(weight(10, 10, 1.0), 1.0);
    assert_eq!(weight(10, 0, 1.0), 1.0);
    assert!((weight(9, 2, 0.5) - 4.5f64.sqrt()).abs() < 1e-12);
}

#[test]
fn scene_axis_repeats_the_rare_scene_by_strength() {
    let mut items = frames("d", 16, "day", "a", "1");
    items.extend(frames("i", 4, "ir", "a", "1"));
    let c = classes(&["person"]);
    for (strength, want) in [(0.5, 2), (1.0, 4)] {
        let o = apply(&items, &c, &on(&[BalanceAxis::Scene], strength, 10), 7);
        for (it, r) in items.iter().zip(&o.repeats) {
            assert_eq!(*r, if it.scene == "ir" { want } else { 1 }, "{}", it.key);
        }
        assert_eq!(o.report.before.frames_per_scene["ir"], 4);
        assert_eq!(o.report.after.frames_per_scene["ir"], 4 * want as usize);
        assert_eq!(o.report.after.images, 16 + 4 * want as usize);
        assert_eq!(o.report.max_repeat, want);
    }
}

#[test]
fn the_class_weight_is_the_largest_of_a_frames_classes() {
    // person in 16 frames (one also has the only raccoon), dog in 4: n_max 16.
    let c = classes(&["person", "dog", "raccoon"]);
    let mut items: Vec<Item> = (0..15)
        .map(|i| item(&format!("p{i}"), "day", "a", "1", &[1, 0, 0]))
        .collect();
    items.push(item("pr", "day", "a", "1", &[2, 0, 1]));
    items.extend((0..4).map(|i| item(&format!("d{i}"), "day", "a", "1", &[0, 1, 0])));
    let o = apply(&items, &c, &on(&[BalanceAxis::Class], 0.5, 10), 1);
    let rep = |k: &str| o.repeats[items.iter().position(|i| i.key == k).unwrap()];
    assert_eq!(rep("p0"), 1);
    assert_eq!(rep("pr"), 4, "raccoon's weight (16/1)^0.5, not person's 1");
    assert_eq!(rep("d0"), 2);
    let after = &o.report.after;
    assert_eq!(after.boxes_per_class["raccoon"], 4);
    assert_eq!(after.boxes_per_class["person"], 15 + 2 * 4);
    assert_eq!(after.frames_per_class["dog"], 8);
    assert_eq!(o.report.before.boxes_per_class["raccoon"], 1);
    assert!(
        o.report.summary().starts_with("20 → 27 images/epoch;"),
        "{}",
        o.report.summary()
    );
    assert!(
        o.report.summary().contains("raccoon 1→4"),
        "{}",
        o.report.summary()
    );
}

#[test]
fn the_product_of_the_axes_is_capped_at_max_repeat() {
    // ir is 1/4 of the frames and camera b too: 3 x 3 at strength 1, capped at 6.
    let mut items = frames("da", 12, "day", "a", "1");
    items.extend(frames("ib", 4, "ir", "b", "1"));
    let c = classes(&["person"]);
    let o = apply(
        &items,
        &c,
        &on(&[BalanceAxis::Scene, BalanceAxis::Camera], 1.0, 6),
        3,
    );
    for (it, r) in items.iter().zip(&o.repeats) {
        assert_eq!(*r, if it.camera == "b" { 6 } else { 1 });
    }
    // Without the camera axis: 3x only.
    let o = apply(&items, &c, &on(&[BalanceAxis::Scene], 1.0, 6), 3);
    assert_eq!(o.report.max_repeat, 3);
}

#[test]
fn seeded_rounding_is_deterministic_and_keeps_the_total() {
    let x: Vec<f64> = (0..300).map(|i| 1.0 + f64::from(i % 17) * 0.137).collect();
    let keys: Vec<String> = (0..300).map(|i| format!("k{i}")).collect();
    let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
    let sum: f64 = x.iter().sum();
    for seed in [0, 1, 42, u64::MAX] {
        let a = round_repeats(&x, &keys, seed);
        assert_eq!(a, round_repeats(&x, &keys, seed), "deterministic");
        let total: f64 = a.iter().map(|&r| f64::from(r)).sum();
        assert!((total - sum).abs() <= 1.0, "{total} vs {sum}");
        for (r, v) in a.iter().zip(&x) {
            assert!(f64::from(*r) == v.floor() || f64::from(*r) == v.floor() + 1.0);
        }
    }
    assert_ne!(round_repeats(&x, &keys, 1), round_repeats(&x, &keys, 2));
    // Whole numbers stay as they are.
    assert_eq!(
        round_repeats(&[2.0, 1.0, 3.0], &["a", "b", "c"], 9),
        [2, 1, 3]
    );
    // apply: the images per epoch are within 1 of the exact expectation.
    let mut items = frames("d", 9, "day", "a", "1");
    items.extend(frames("i", 2, "ir", "a", "1"));
    let o = apply(
        &items,
        &classes(&["p"]),
        &on(&[BalanceAxis::Scene], 0.5, 10),
        5,
    );
    let want = 9.0 + 2.0 * 4.5f64.sqrt();
    assert!((o.report.after.images as f64 - want).abs() <= 1.0);
}

#[test]
fn the_cap_never_drops_hard_negatives_or_a_class_last_box() {
    let c = classes(&["person", "raccoon"]);
    let mut items = Vec::new();
    for i in 0..3 {
        let mut it = item(&format!("hn{i}"), "night", "a", "1", &[0, 0]);
        it.hard_negative = true;
        items.push(it);
    }
    items.push(item("raccoon", "night", "a", "1", &[0, 1]));
    items.push(item("pa0", "day", "a", "1", &[1, 0]));
    items.push(item("pa1", "day", "a", "1", &[1, 0]));
    items.push(item("pb", "day", "b", "1", &[1, 0]));
    items.push(item("pc", "day", "c", "1", &[1, 0]));
    let mut b = on(&[BalanceAxis::Camera], 0.5, 4);
    // Camera a has 7 frames, b and c 1: median 1, limit 1.
    b.cap_dominant = Some(1.0);
    let o = apply(&items, &c, &b, 11);
    let kept = |k: &str| o.keep[items.iter().position(|i| i.key == k).unwrap()];
    for k in ["hn0", "hn1", "hn2", "raccoon", "pb", "pc"] {
        assert!(kept(k), "{k} must stay");
    }
    assert!(!kept("pa0") && !kept("pa1"));
    assert_eq!(o.report.dropped, 2);
    assert_eq!(o.report.frames_after, 6);
    assert_eq!(o.repeats.iter().filter(|r| **r == 0).count(), 2);
    // The last person box of the training set stays even when its camera is over the cap.
    let items = vec![
        item("p", "day", "a", "1", &[1, 0]),
        item("r0", "day", "a", "1", &[0, 1]),
        item("r1", "day", "a", "1", &[0, 1]),
        item("x", "day", "b", "1", &[0, 1]),
        item("y", "day", "c", "1", &[0, 1]),
    ];
    let o = apply(&items, &c, &b, 11);
    assert!(o.keep[0], "the only person frame");
    assert_eq!(o.report.dropped, 2);
}

#[test]
fn the_class_cap_drops_only_frames_whose_classes_are_all_dominant() {
    let c = classes(&["person", "dog", "raccoon"]);
    let mut items: Vec<Item> = (0..9)
        .map(|i| item(&format!("p{i}"), "day", "a", "1", &[1, 0, 0]))
        .collect();
    items.push(item("pd", "day", "a", "1", &[1, 1, 0]));
    items.push(item("d", "day", "a", "1", &[0, 1, 0]));
    items.push(item("r", "day", "a", "1", &[0, 0, 1]));
    let mut b = on(&[BalanceAxis::Class], 0.5, 4);
    // Frames per class 10 / 2 / 1: median 2, limit 2.
    b.cap_dominant = Some(1.0);
    let o = apply(&items, &c, &b, 2);
    assert_eq!(o.report.dropped, 8);
    assert!(o.keep[9], "person + dog: dog is not dominant");
    assert!(o.keep[10] && o.keep[11]);
    let person_frames = (0..9).filter(|&i| o.keep[i]).count();
    assert_eq!(
        person_frames, 1,
        "down to the limit (with the person + dog frame)"
    );
}

#[test]
fn camera_days_compare_only_with_days_of_the_same_camera() {
    let mut items = frames("a1", 10, "day", "a", "1");
    items.extend(frames("a2", 10, "day", "a", "2"));
    items.extend(frames("b1", 2, "day", "b", "1"));
    items.extend(frames("b2", 2, "day", "b", "2"));
    items.extend(frames("c1", 8, "day", "c", "1"));
    items.extend(frames("c2", 2, "day", "c", "2"));
    let o = apply(
        &items,
        &classes(&["p"]),
        &on(&[BalanceAxis::CameraDay], 0.5, 4),
        0,
    );
    for (it, r) in items.iter().zip(&o.repeats) {
        let want = if it.key.starts_with("c2") { 2 } else { 1 };
        assert_eq!(*r, want, "{}", it.key);
    }
    let days = &o.report.after.camera_days;
    assert_eq!(days.first().map(|d| d.0.as_str()), Some("a/1"));
    assert!(days.contains(&("c/2".into(), 4)));
}

#[test]
fn the_negative_share_is_restored() {
    let c = classes(&["person", "raccoon"]);
    let mut items: Vec<Item> = (0..60)
        .map(|i| item(&format!("p{i}"), "day", "a", "1", &[1, 0]))
        .collect();
    items.extend((0..20).map(|i| item(&format!("r{i}"), "night", "a", "1", &[0, 1])));
    items.extend((0..20).map(|i| item(&format!("n{i}"), "day", "a", "1", &[0, 0])));
    let o = apply(&items, &c, &on(&[BalanceAxis::Class], 1.0, 4), 4);
    let (before, after) = (&o.report.before, &o.report.after);
    assert!((before.negative_share - 0.2).abs() < 1e-9);
    assert!(after.images > before.images);
    assert!(
        (after.negative_share - before.negative_share).abs() < 0.02,
        "{} vs {}",
        after.negative_share,
        before.negative_share
    );
    // raccoon 60/20 = 3x; the negatives follow to keep their 20%.
    assert_eq!(after.frames_per_class["raccoon"], 60);
}

#[test]
fn warnings_name_single_frame_groups_and_long_epochs() {
    let mut items = frames("d", 30, "day", "a", "1");
    items.push(item("ir", "ir", "a", "1", &[1]));
    let o = apply(
        &items,
        &classes(&["p"]),
        &on(&[BalanceAxis::Scene], 1.0, 4),
        0,
    );
    assert!(
        o.warnings
            .iter()
            .any(|w| w.contains("few unique frames") && w.contains("scene ir")),
        "{:?}",
        o.warnings
    );
    assert!(!o.warnings.iter().any(|w| w.contains("epochs get longer")));
    let mut items = frames("d", 4, "day", "a", "1");
    items.extend((0..10).map(|i| item(&format!("e{i}"), "day", &format!("e{i}"), "1", &[1])));
    let o = apply(
        &frames("x", 1, "day", "a", "1")
            .into_iter()
            .chain(frames("y", 10, "ir", "b", "1"))
            .collect::<Vec<_>>(),
        &classes(&["p"]),
        &on(&[BalanceAxis::Camera], 1.0, 10),
        0,
    );
    assert_eq!(o.report.after.images, 20);
    assert!(
        !o.warnings.iter().any(|w| w.contains("epochs get longer")),
        "11 -> 20 is not more than double"
    );
    let o = apply(
        &items,
        &classes(&["p"]),
        &on(&[BalanceAxis::Camera], 1.0, 4),
        0,
    );
    assert!(
        o.warnings.iter().any(|w| w.contains("epochs get longer")),
        "{:?}",
        o.warnings
    );
}

#[test]
fn axes_parse_and_balance_round_trips() {
    let a = Balance::parse_axes("class, scene+camera-day").unwrap();
    assert_eq!(
        a.into_iter().collect::<Vec<_>>(),
        [
            BalanceAxis::Class,
            BalanceAxis::Scene,
            BalanceAxis::CameraDay
        ]
    );
    assert!(Balance::parse_axes("").unwrap().is_empty());
    assert!(Balance::parse_axes("off").unwrap().is_empty());
    assert_eq!(Balance::parse_axes("camera_day camera").unwrap().len(), 2);
    let e = Balance::parse_axes("class,colour").unwrap_err().to_string();
    assert!(e.contains("'colour'"), "{e}");
    for a in BalanceAxis::ALL {
        assert_eq!(BalanceAxis::parse(a.as_str()), Some(a));
        let j = serde_json::to_string(&a).unwrap();
        assert_eq!(j, format!("\"{}\"", a.as_str()));
    }
    let b: Balance = serde_json::from_str(r#"{"axes":["scene"]}"#).unwrap();
    assert_eq!((b.strength, b.max_repeat, b.cap_dominant), (0.5, 4, None));
    assert_eq!(b.describe(), "scene, strength 0.5, max 4x");
    assert_eq!(Balance::default().describe(), "off");
    let b = Balance {
        cap_dominant: Some(3.0),
        strength: 0.3,
        ..b
    };
    let text = serde_json::to_string(&b).unwrap();
    assert_eq!(
        text,
        r#"{"axes":["scene"],"strength":0.3,"max_repeat":4,"cap_dominant":3.0}"#
    );
    assert_eq!(serde_json::from_str::<Balance>(&text).unwrap(), b);
}
