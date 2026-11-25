//! Unit tests of the plan's and the writer's step functions on their own (QA-255):
//! [`Frames`], [`split_heldout`], [`cap_static`], [`negatives_for`] / [`pick_negatives`],
//! [`class_warnings`], [`coco_json`] and [`label_text`] / [`ground_truth`].

use super::*;

fn bx(class: &str, cx: f32, cy: f32, w: f32, h: f32) -> YoloBox {
    YoloBox {
        class: class.into(),
        cx,
        cy,
        w,
        h,
    }
}

/// Camera `cam`, day `day`: `n` frames with a `car` box each (frame `i` at `f<i>.jpg`).
fn day_frames(dir: &Path, cam: &str, day: usize, n: usize, first: usize) -> Vec<Frame> {
    (first..first + n)
        .map(|i| {
            let mut f = patio_frame(
                dir,
                i,
                day,
                "day",
                &[("car", 0.1 + i as f32 * 0.01, 0.5, 0.1, 0.1)],
            );
            f.camera = cam.into();
            f.split = SplitHint::None;
            f
        })
        .collect()
}

fn source(frames: Vec<Frame>) -> Source {
    patio_source(Path::new("steps"), frames)
}

#[test]
fn frames_look_up_a_reference_and_its_key() {
    let s = [source(day_frames(Path::new("steps"), "a", 1, 2, 0))];
    let frames = Frames { sources: &s };
    assert_eq!(
        frames.frame((0, 1)).unwrap().image,
        Path::new("steps").join("f0001.jpg")
    );
    assert_eq!(frames.key((0, 1)), "f0001.jpg");
    assert!(frames.frame((0, 2)).is_none());
    assert_eq!(frames.key((1, 0)), "");
    let e = frames.planned((0, 9)).unwrap_err();
    assert_eq!(format!("{e:#}"), "plan refers to a missing frame (0, 9)");
}

#[test]
fn the_held_out_split_takes_whole_camera_days_and_never_all() {
    let dir = Path::new("steps");
    let mut frames = day_frames(dir, "a", 1, 10, 0);
    frames.extend(day_frames(dir, "a", 2, 10, 10));
    frames.extend(day_frames(dir, "b", 1, 10, 20));
    let s = [source(frames)];
    let review: Vec<FrameRef> = (0..30).map(|i| (0, i)).collect();
    let opts = Options {
        heldout_share: 0.2,
        ..Options::default()
    };
    let mut p = Plan::default();
    let (pool, hinted) = split_heldout(Frames { sources: &s }, &review, &[], &opts, &mut p);
    assert!(!hinted);
    assert_eq!(p.stats.heldout_groups.len(), 1, "{:?}", p.stats);
    assert_eq!(p.heldout.len(), 10);
    assert_eq!(pool.len() + p.heldout.len(), 30);
    let day = |r: &FrameRef| group_key(&s[0].frames[r.1]);
    let held: BTreeSet<String> = p.heldout.iter().map(day).collect();
    assert_eq!(held.len(), 1, "one whole camera day");
    assert!(pool.iter().all(|r| !held.contains(&day(r))));
    assert!(p.warnings.is_empty(), "{:?}", p.warnings);
    // Everything wanted held out: one group still trains.
    let opts = Options {
        heldout_share: 1.0,
        ..Options::default()
    };
    let mut p = Plan::default();
    let (pool, _) = split_heldout(Frames { sources: &s }, &review, &[], &opts, &mut p);
    assert_eq!((pool.len(), p.heldout.len()), (10, 20));
}

#[test]
fn the_held_out_split_follows_the_sources_split_and_frozen_days() {
    let dir = Path::new("steps");
    let mut frames = day_frames(dir, "a", 1, 4, 0);
    frames.extend(day_frames(dir, "a", 2, 4, 4));
    frames.extend(day_frames(dir, "b", 3, 4, 8));
    for f in &mut frames {
        f.split = if f.day == "2026-10-02" {
            SplitHint::Heldout
        } else {
            SplitHint::Train
        };
    }
    let s = [source(frames)];
    let review: Vec<FrameRef> = (0..12).map(|i| (0, i)).collect();
    let opts = Options {
        frozen_heldout: ["b/2026-10-03".to_string()].into_iter().collect(),
        ..Options::default()
    };
    let mut p = Plan::default();
    let (pool, hinted) = split_heldout(Frames { sources: &s }, &review, &[], &opts, &mut p);
    assert!(hinted);
    assert_eq!(p.heldout, (4..12).map(|i| (0, i)).collect::<Vec<_>>());
    assert_eq!(pool, (0..4).map(|i| (0, i)).collect::<Vec<_>>());
    assert_eq!(p.stats.frozen_moved, 4);
    assert!(
        p.warnings[0].starts_with("4 frame(s) of 1 camera day(s) held out for good (b/2026-10-03)"),
        "{:?}",
        p.warnings
    );
    assert_eq!(p.heldout_fingerprint.len(), 64);
    // Nothing to hold out: said so.
    let mut p = Plan::default();
    let none = Options {
        heldout_share: 0.0,
        ..Options::default()
    };
    let s = [source(day_frames(dir, "a", 1, 3, 0))];
    let (pool, _) = split_heldout(
        Frames { sources: &s },
        &[(0, 0), (0, 1)],
        &[],
        &none,
        &mut p,
    );
    assert_eq!(pool, [(0, 0), (0, 1)]);
    assert!(p.heldout.is_empty());
    assert_eq!(
        p.warnings,
        ["no frame is held out: the trained model is not benchmarked on your own frames"]
    );
}

#[test]
fn the_static_cap_keeps_the_cap_per_spot_and_counts() {
    let t = tmp("steps-static");
    let s = [static_source(&t.0)];
    let classes: Vec<String> = vec!["car".into(), "person".into()];
    let opts = Options::default();
    // The train / valid pool: every frame but the held-out day.
    let pool: Vec<FrameRef> = (0..s[0].frames.len())
        .filter(|&i| s[0].frames[i].split != SplitHint::Heldout)
        .map(|i| (0, i))
        .collect();
    let pos: Vec<FrameRef> = pool
        .iter()
        .copied()
        .filter(|r| !s[0].frames[r.1].boxes.is_empty())
        .collect();
    let mut stats = Stats::default();
    let over = cap_static(
        Frames { sources: &s },
        &pool,
        &pos,
        5,
        &opts,
        &classes,
        &mut stats,
    );
    // 60 car-only frames at the spot: 55 over the cap of 5.
    assert_eq!(over.len(), 55);
    assert!(over.iter().all(|r| car_only(&s[0].frames[r.1])));
    assert_eq!(stats.static_capped, 55);
    assert_eq!(stats.static_spots, 1);
    assert_eq!(stats.static_capped_boxes.get("car"), Some(&55));
    assert_eq!(stats.static_caps.len(), 1);
    assert_eq!(
        (stats.static_caps[0].frames, stats.static_caps[0].kept),
        (60, 5)
    );
}

#[test]
fn negatives_follow_the_share_and_the_wanted_ones_stay() {
    assert_eq!(negatives_for(85, 0.15), 15);
    assert_eq!(negatives_for(10, 0.0), 0);
    // Clamped at 0.9: 9 negatives per positive.
    assert_eq!(negatives_for(10, 1.0), 90);
    let dir = Path::new("steps");
    let mut frames = day_frames(dir, "a", 1, 6, 0);
    for f in &mut frames {
        f.boxes.clear();
    }
    frames[4].hard_negative = true;
    frames[5].human_rejected = true;
    let s = [source(frames)];
    let frames = Frames { sources: &s };
    let opts = Options {
        negative_share: 0.5,
        ..Options::default()
    };
    let mut neg: Vec<FrameRef> = (0..6).map(|i| (0, i)).collect();
    let (mut stats, mut warnings) = (Stats::default(), Vec::new());
    pick_negatives(frames, &mut neg, 2, &opts, &mut stats, &mut warnings);
    assert_eq!(neg.len(), 2);
    assert!(neg.contains(&(0, 4)) && neg.contains(&(0, 5)), "{neg:?}");
    assert_eq!(stats.negatives_dropped, 4);
    assert!(warnings.is_empty());
    // Too few: a warning, nothing dropped.
    let mut neg = vec![(0, 0)];
    let (mut stats, mut warnings) = (Stats::default(), Vec::new());
    pick_negatives(frames, &mut neg, 9, &opts, &mut stats, &mut warnings);
    assert_eq!(neg, [(0, 0)]);
    assert_eq!(stats.negatives_dropped, 0);
    assert_eq!(
        warnings,
        [
            "only 1 frame(s) without objects for 9 with objects (10%, target 50%): add empty \
          frames (hard negatives) to cut false positives"
        ]
    );
}

#[test]
fn a_class_without_training_boxes_is_named_with_where_its_boxes_are() {
    let dir = Path::new("steps");
    let mut frames = day_frames(dir, "a", 1, 2, 0);
    frames[1].boxes.push(bx("fox", 0.5, 0.5, 0.1, 0.1));
    let s = [source(frames)];
    let opts = Options::default();
    let mut p = Plan {
        classes: vec!["car".into(), "fox".into(), "owl".into()],
        heldout: vec![(0, 1)],
        ..Plan::default()
    };
    p.stats.train_boxes.insert("car".into(), 1);
    p.stats.heldout_groups = vec!["a/2026-10-01".into()];
    let w = class_warnings(Frames { sources: &s }, &p, &opts, false);
    assert_eq!(w.len(), 2, "{w:?}");
    assert!(
        w[0].starts_with(
            "class 'fox' has no training box: its 1 box(es) are only in the held-out frames \
             (a/2026-10-01)"
        ),
        "{w:?}"
    );
    assert!(w[1].starts_with("class 'owl' has no training box: no frame has a box of it"));
    let w = class_warnings(Frames { sources: &s }, &p, &opts, true);
    assert!(
        w[0].contains("only in the held-out split (the export's val split"),
        "{w:?}"
    );
}

#[test]
fn coco_entries_are_in_pixels_and_labels_in_the_run_classes() {
    let dir = Path::new("steps");
    let mut f = patio_frame(dir, 0, 1, "day", &[("car", 0.5, 0.5, 0.2, 0.1)]);
    f.boxes.push(bx("truck", 0.1, 0.1, 0.2, 0.2));
    f.ignore.push(bx("person", 0.9, 0.9, 0.05, 0.1));
    let s = [source(vec![f])];
    let opts = Options::default();
    let classes: Vec<String> = vec!["person".into(), "car".into()];
    let v = coco_json(
        Frames { sources: &s },
        &[((0, 0), "x.jpg".into(), (100, 50))],
        &opts,
        &classes,
    )
    .unwrap();
    assert_eq!(
        v,
        serde_json::json!({
            "images": [{"id": 0, "file_name": "images/x.jpg", "width": 100, "height": 50}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 1,
                "bbox": [40.0, 22.5, 20.0, 5.0], "area": 100.0, "iscrowd": 0}],
            "categories": [
                {"id": 0, "name": "person", "supercategory": "object"},
                {"id": 1, "name": "car", "supercategory": "object"},
            ],
        })
    );
    let missing = coco_json(
        Frames { sources: &s },
        &[((0, 3), "y.jpg".into(), (1, 1))],
        &opts,
        &classes,
    );
    assert!(missing.is_err());
    let f = &s[0].frames[0];
    assert_eq!(label_text(f, &opts, &classes), yolo_line(1, &f.boxes[0]));
    let truth = ground_truth(f, (100, 50), &opts, &classes).unwrap();
    assert_eq!(
        truth
            .iter()
            .map(|o| (o.label.as_str(), o.ignore))
            .collect::<Vec<_>>(),
        [("car", false), ("person", true)]
    );
}
