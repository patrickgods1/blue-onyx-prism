//! Validation split tests on in-memory sources (a plan never opens a file).

use super::super::*;
use super::VALID_SALT;

fn frame(camera: &str, day: &str, name: &str, classes: &[&str]) -> Frame {
    Frame {
        image: PathBuf::from("/export/images").join(format!("{camera}.{day}.{name}.jpg")),
        boxes: classes
            .iter()
            .map(|c| YoloBox {
                class: (*c).into(),
                cx: 0.5,
                cy: 0.5,
                w: 0.2,
                h: 0.2,
            })
            .collect(),
        ignore: Vec::new(),
        parked: Vec::new(),
        scene: Some("day".into()),
        camera: camera.into(),
        day: day.into(),
        split: SplitHint::None,
        hard_negative: false,
        human_rejected: false,
        bulk_accepted: false,
        size: Some((64, 48)),
    }
}

/// `n` frames of a camera day, each with one box of `class` ("" = no box).
fn frames(camera: &str, day: &str, n: usize, class: &str) -> Vec<Frame> {
    let boxes: &[&str] = if class.is_empty() { &[] } else { &[class] };
    (0..n)
        .map(|i| frame(camera, day, &format!("{class}{i}"), boxes))
        .collect()
}

/// `count` camera days (`cam<i>/2026-10-01`) of `n` person frames each.
fn days(count: usize, n: usize) -> Vec<Frame> {
    (0..count)
        .flat_map(|i| frames(&format!("cam{i}"), "2026-10-01", n, "person"))
        .collect()
}

fn source(frames: Vec<Frame>) -> Source {
    Source {
        dir: PathBuf::from("/export"),
        public: None,
        classes: ["person", "car", "truck", "dog", "cat"]
            .map(String::from)
            .to_vec(),
        frames,
        warnings: Vec::new(),
        notes_known: false,
    }
}

/// Everything goes to train / valid, unless a test says otherwise.
fn opts() -> Options {
    Options {
        heldout_share: 0.0,
        negative_share: 0.0,
        ..Options::default()
    }
}

fn group_of(s: &[Source], r: FrameRef) -> String {
    group_key(&s[r.0].frames[r.1])
}

fn groups_of(s: &[Source], refs: impl Iterator<Item = FrameRef>) -> BTreeSet<String> {
    refs.map(|r| group_of(s, r)).collect()
}

/// Validation's groups are the plan's list, and none of them trains.
fn assert_whole_days(s: &[Source], p: &Plan) {
    let valid = groups_of(s, p.valid.iter().copied());
    let train = groups_of(s, p.train.iter().map(|(r, _)| *r));
    assert!(valid.is_disjoint(&train), "{valid:?} / {train:?}");
    assert_eq!(valid, p.stats.valid_groups.iter().cloned().collect());
    let mut sorted = p.stats.valid_groups.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, p.stats.valid_groups);
}

fn split_warning(p: &Plan) -> Option<&String> {
    p.warnings
        .iter()
        .find(|w| w.contains("validation is split frame by frame"))
}

/// The frame split of a pool without guarded classes: the first `n` frames by seeded rank.
fn frame_split(s: &Source, o: &Options, n: usize) -> Vec<FrameRef> {
    let mut v: Vec<FrameRef> = (0..s.frames.len()).map(|i| (0, i)).collect();
    v.sort_by_cached_key(|r| rank(o.seed ^ VALID_SALT, &frame_key(s, &s.frames[r.1])));
    v.truncate(n);
    v.sort_unstable();
    v
}

/// Boxes of `class` in validation.
fn boxes_in_valid(s: &[Source], p: &Plan, class: &str) -> usize {
    p.valid
        .iter()
        .flat_map(|r| &s[r.0].frames[r.1].boxes)
        .filter(|b| b.class == class)
        .count()
}

#[test]
fn validation_is_whole_camera_days_unseen_in_training() {
    let s = vec![source(days(12, 20))];
    for seed in 0..20 {
        let o = Options { seed, ..opts() };
        let p = plan(&s, &o).unwrap();
        // 240 frames: 50 wanted, three days reach it.
        assert_eq!(p.valid.len(), 60);
        assert_eq!(p.stats.valid_groups.len(), 3);
        assert_whole_days(&s, &p);
        assert_eq!(p.train.len(), 180);
        assert_eq!(split_warning(&p), None);
        assert!(p.valid.is_sorted());
    }
    let t = valid_groups_text(&["a/2026-10-01".to_string(), "b/2026-10-02".to_string()]);
    assert_eq!(
        t.as_deref(),
        Some("validation on 2 camera day(s) kept out of training: a/2026-10-01, b/2026-10-02")
    );
    assert_eq!(valid_groups_text(&[]), None);
}

#[test]
fn empty_frames_stay_with_their_camera_day() {
    let mut f = Vec::new();
    for i in 0..10 {
        let cam = format!("cam{i}");
        f.extend(frames(&cam, "2026-10-01", 30, "person"));
        f.extend(frames(&cam, "2026-10-01", 5, ""));
    }
    let s = vec![source(f)];
    let o = Options {
        negative_share: 0.5,
        ..opts()
    };
    let p = plan(&s, &o).unwrap();
    assert_whole_days(&s, &p);
    assert!(
        p.valid.iter().any(|r| s[0].frames[r.1].boxes.is_empty()),
        "the negatives of a picked camera day validate"
    );
}

/// The same plan for every training seed: the seed reaches the trainer, not the options (see
/// `preview_plan_is_the_jobs_plan`), so here: the same options on the same frames.
#[test]
fn scene_filters_keep_the_plan_stable() {
    let mut f = Vec::new();
    for i in 0..6 {
        let cam = format!("cam{i}");
        for (day, scene) in [("2026-10-01", "day"), ("2026-10-02", "night")] {
            let mut g = frames(&cam, day, 12, "person");
            for x in &mut g {
                x.scene = Some(scene.into());
            }
            f.extend(g);
        }
        // One camera day of both scenes: the whole of it is one group.
        let mut g = frames(&cam, "2026-10-03", 8, "person");
        for (n, x) in g.iter_mut().enumerate() {
            x.scene = Some(if n % 2 == 0 { "day" } else { "night" }.into());
        }
        f.extend(g);
    }
    let s = vec![source(f)];
    let mut moved = s.clone();
    moved[0].dir = PathBuf::from("/moved");
    for x in &mut moved[0].frames {
        x.image = Path::new("/moved").join(x.image.strip_prefix("/export").unwrap());
    }
    let mut fingerprints = BTreeSet::new();
    for scene in SceneFilter::PRESETS {
        let o = Options {
            scene,
            heldout_share: 0.2,
            ..opts()
        };
        let p = plan(&s, &o).unwrap();
        assert_eq!(p, plan(&s, &o).unwrap());
        assert_eq!(p, plan(&moved, &o).unwrap());
        assert_whole_days(&s, &p);
        assert!(!p.stats.valid_groups.is_empty());
        fingerprints.insert(p.heldout_fingerprint);
    }
    assert_eq!(fingerprints.len(), 1);
}

#[test]
fn a_camera_day_holding_most_of_a_class_stays_in_training() {
    // The case that sent 3 of 4 dog boxes to validation: one day holds three of them.
    let mut f = days(10, 20);
    f.extend(frames("porch", "2026-10-01", 3, "dog"));
    f.extend(frames("porch", "2026-10-02", 1, "dog"));
    let s = vec![source(f)];
    for seed in 0..30 {
        let p = plan(&s, &Options { seed, ..opts() }).unwrap();
        assert!(!p.stats.valid_groups.contains(&"porch/2026-10-01".into()));
        assert!(boxes_in_valid(&s, &p, "dog") <= 1);
        assert_whole_days(&s, &p);
    }
}

#[test]
fn validation_never_holds_more_than_half_of_a_class() {
    // Four days of one dog box each: no single day is over half, together they would be.
    let mut f = days(10, 20);
    for i in 0..4 {
        f.extend(frames(&format!("dog{i}"), "2026-10-01", 1, "dog"));
    }
    let s = vec![source(f)];
    for seed in 0..30 {
        let o = Options {
            seed,
            val_share: 0.25,
            ..opts()
        };
        let p = plan(&s, &o).unwrap();
        assert!(boxes_in_valid(&s, &p, "dog") <= 2);
        assert!(p.stats.train_boxes["dog"] >= 2);
        assert_whole_days(&s, &p);
    }
}

#[test]
fn the_last_boxes_of_a_class_never_leave_training() {
    let mut f = days(10, 20);
    // One box, and two on the same camera day.
    f.extend(frames("porch", "2026-10-01", 1, "cat"));
    f.extend(frames("drive", "2026-10-01", 2, "dog"));
    let s = vec![source(f)];
    for seed in 0..30 {
        let p = plan(&s, &Options { seed, ..opts() }).unwrap();
        assert_eq!(p.stats.train_boxes["cat"], 1);
        assert_eq!(p.stats.train_boxes["dog"], 2);
    }
}

#[test]
fn days_adding_a_class_go_first() {
    // Most days have people only; dog and car are on two days each.
    let mut f = days(20, 10);
    for (cam, class) in [("a", "dog"), ("b", "dog"), ("c", "car"), ("d", "car")] {
        f.extend(frames(cam, "2026-10-01", 4, class));
    }
    let s = vec![source(f)];
    for seed in 0..30 {
        let p = plan(&s, &Options { seed, ..opts() }).unwrap();
        // Half of a class's boxes at most: exactly one of its two days.
        assert_eq!(boxes_in_valid(&s, &p, "dog"), 4, "seed {seed}");
        assert_eq!(boxes_in_valid(&s, &p, "car"), 4, "seed {seed}");
        assert_whole_days(&s, &p);
    }
}

#[test]
fn the_first_pass_stops_at_one_and_a_half_times_the_wanted_size() {
    // 74-frame days: 500 frames want 50, one day is 74 <= 75.
    let mut f = days(6, 74);
    f.extend(frames("small", "2026-10-01", 56, "person"));
    let s = vec![source(f)];
    for seed in 0..20 {
        let p = plan(&s, &Options { seed, ..opts() }).unwrap();
        assert!(
            p.valid.len() >= 50 && p.valid.len() <= 75,
            "{}",
            p.valid.len()
        );
        assert_whole_days(&s, &p);
        assert_eq!(split_warning(&p), None);
    }
}

#[test]
fn the_second_pass_goes_up_to_the_cap() {
    // 480 frames want 50: 80-frame days overshoot the first pass (75), fit the cap (120).
    let s = vec![source(days(6, 80))];
    let p = plan(&s, &opts()).unwrap();
    assert_eq!(p.valid.len(), 80);
    assert_eq!(p.stats.valid_groups.len(), 1);
    assert_whole_days(&s, &p);
    assert_eq!(split_warning(&p), None);

    // A day taken in the first pass, then a larger one in the second, within the cap: 530 frames
    // want 53 (first pass 79, cap 132).
    let mut f = days(6, 80);
    f.extend(frames("small", "2026-10-01", 50, "person"));
    let s = vec![source(f)];
    for seed in 0..20 {
        let p = plan(&s, &Options { seed, ..opts() }).unwrap();
        assert_eq!(p.valid.len(), 130, "seed {seed}");
        assert!(p.stats.valid_groups.contains(&"small/2026-10-01".into()));
        assert_whole_days(&s, &p);
    }
}

#[test]
fn one_whole_day_may_pass_a_quarter_of_a_small_pool() {
    // 210 frames want 50, a quarter is 52: the first pass still takes a 70-frame day.
    let s = vec![source(days(3, 70))];
    let p = plan(&s, &opts()).unwrap();
    assert_eq!(p.valid.len(), 70);
    assert_eq!(split_warning(&p), None);
    assert_whole_days(&s, &p);
    // The most it can be: 3/8 of the candidates.
    let s = vec![source(days(3, 76))];
    let p = plan(&s, &opts()).unwrap();
    assert!(split_warning(&p).is_some());
}

#[test]
fn a_share_above_a_quarter_is_not_capped() {
    let s = vec![source(days(10, 40))];
    let o = Options {
        val_share: 0.5,
        ..opts()
    };
    let p = plan(&s, &o).unwrap();
    assert_eq!(p.valid.len(), 200);
    assert_eq!(p.stats.valid_groups.len(), 5);
    assert_whole_days(&s, &p);
}

#[test]
fn under_three_camera_days_fall_back_to_the_frame_split() {
    for count in [1, 2] {
        let src = source(days(count, 100 / count));
        let s = vec![src.clone()];
        let o = opts();
        let p = plan(&s, &o).unwrap();
        assert_eq!(p.valid, frame_split(&src, &o, 25));
        assert!(p.stats.valid_groups.is_empty());
        let w = split_warning(&p).unwrap();
        assert!(w.contains(&format!("only {count} camera day(s)")), "{w}");
        assert!(w.contains("cam0/2026-10-01"), "{w}");
        assert!(w.contains("review frames from more cameras or days"), "{w}");
    }
    // Three are enough, and the held-out days do not count.
    let s = vec![source(days(3, 30))];
    let p = plan(&s, &opts()).unwrap();
    assert_eq!(split_warning(&p), None);
    assert_whole_days(&s, &p);
    let o = Options {
        heldout_share: 0.2,
        ..opts()
    };
    let p = plan(&s, &o).unwrap();
    assert!(!p.stats.heldout_groups.is_empty());
    assert!(split_warning(&p).unwrap().contains("only 2 camera day(s)"));
}

#[test]
fn days_reaching_under_half_of_the_wanted_size_fall_back() {
    // Every day is above the cap.
    let src = source(days(3, 300));
    let s = vec![src.clone()];
    let o = opts();
    let p = plan(&s, &o).unwrap();
    let w = split_warning(&p).unwrap();
    assert!(w.contains("reach only 0 of the 90"), "{w}");
    assert!(w.contains("review frames from more cameras or days"), "{w}");
    assert_eq!(p.valid, frame_split(&src, &o, 90));
    assert!(p.stats.valid_groups.is_empty());

    // 40 of 100 is under half, the others do not fit the cap (250).
    let mut f = days(2, 480);
    f.extend(frames("small", "2026-10-01", 40, "person"));
    let p = plan(&[source(f)], &opts()).unwrap();
    assert!(
        split_warning(&p)
            .unwrap()
            .contains("reach only 40 of the 100")
    );

    // 50 of 100 is half: kept.
    let mut f = days(2, 475);
    f.extend(frames("small", "2026-10-01", 50, "person"));
    let s = vec![source(f)];
    let p = plan(&s, &opts()).unwrap();
    assert_eq!(split_warning(&p), None);
    assert_eq!(p.valid.len(), 50);
    assert_whole_days(&s, &p);
}

#[test]
fn days_without_any_box_are_no_validation() {
    // The days with objects are above the cap (180), the empty ones reach the size.
    let mut f = days(2, 300);
    for i in 0..12 {
        f.extend(frames(&format!("empty{i}"), "2026-10-01", 10, ""));
    }
    let s = vec![source(f)];
    let o = Options {
        negative_share: 0.5,
        ..opts()
    };
    let p = plan(&s, &o).unwrap();
    let w = split_warning(&p).unwrap();
    assert!(w.contains("no camera day with objects"), "{w}");
    assert!(p.stats.valid_groups.is_empty());
    assert!(p.valid.iter().any(|r| !s[0].frames[r.1].boxes.is_empty()));
}

#[test]
fn frames_without_tags_fall_back_and_never_validate_as_a_group() {
    let src = source(frames(UNKNOWN, UNKNOWN, 100, "person"));
    let s = vec![src.clone()];
    let o = opts();
    let p = plan(&s, &o).unwrap();
    let w = split_warning(&p).unwrap();
    assert!(w.contains("no frame has a camera or day tag"), "{w}");
    assert!(w.contains("Dataset tab export"), "{w}");
    assert_eq!(p.valid, frame_split(&src, &o, 25));

    // Beside real days the untagged frames stay in training: they may be from any camera day.
    let mut f = days(8, 20);
    f.extend(frames(UNKNOWN, UNKNOWN, 30, "person"));
    let s = vec![source(f)];
    for seed in 0..20 {
        let p = plan(&s, &Options { seed, ..opts() }).unwrap();
        assert!(!p.stats.valid_groups.contains(&"unknown/unknown".into()));
        assert_whole_days(&s, &p);
    }
    // A camera without a day is a group like any other.
    let mut f = days(2, 30);
    f.extend(frames("cam9", UNKNOWN, 30, "person"));
    let p = plan(&[source(f)], &opts()).unwrap();
    assert_eq!(split_warning(&p), None);
}

#[test]
fn public_only_pools_fall_back_silently() {
    let mut src = source(days(10, 10));
    src.public = Some(Public {
        id: "pub".into(),
        license: "cc".into(),
    });
    let s = vec![src.clone()];
    let o = opts();
    let p = plan(&s, &o).unwrap();
    assert_eq!(split_warning(&p), None);
    assert!(p.stats.valid_groups.is_empty());
    assert_eq!(p.valid, frame_split(&src, &o, 25));
}

#[test]
fn public_frames_are_never_validated_on_or_counted() {
    let mut public = source(frames("pub", UNKNOWN, 400, "person"));
    public.public = Some(Public {
        id: "pub".into(),
        license: "cc".into(),
    });
    let s = vec![source(days(10, 30)), public];
    let p = plan(&s, &opts()).unwrap();
    assert!(p.valid.iter().all(|r| r.0 == 0));
    // 300 review frames want 50: two days are 60. The public frames do not grow that.
    assert_eq!(p.valid.len(), 60);
    assert_eq!(p.stats.public_frames, 400);
    assert_whole_days(&s, &p);
}

#[test]
fn too_few_frames_for_validation_say_only_that() {
    let s = vec![source(days(1, 1))];
    let p = plan(&s, &opts()).unwrap();
    assert_eq!(split_warning(&p), None);
    assert!(p.warnings.iter().any(|w| w.contains("too few frames")));
}

#[test]
fn dropped_frames_are_no_part_of_a_camera_day() {
    // Trucks are not trained: `drop` loses a whole day and part of another.
    let mut f = days(10, 20);
    f.extend(frames("trucks", "2026-10-01", 20, "truck"));
    let mut part = frames("mixed", "2026-10-01", 20, "person");
    part.extend(frames("mixed", "2026-10-01", 6, "truck"));
    f.extend(part);
    let s = vec![source(f)];
    for seed in 0..30 {
        let o = Options {
            seed,
            classes: Some(vec!["person".into()]),
            excluded: Excluded::DropFrame,
            ..opts()
        };
        let p = plan(&s, &o).unwrap();
        assert_eq!(p.stats.excluded_frames, 26);
        assert!(!p.stats.valid_groups.contains(&"trucks/2026-10-01".into()));
        let used = p
            .train
            .iter()
            .map(|(r, _)| *r)
            .chain(p.valid.iter().copied());
        assert!(
            used.into_iter()
                .all(|r| { s[0].frames[r.1].boxes.iter().all(|b| b.class == "person") })
        );
        assert_whole_days(&s, &p);
    }
}

#[test]
fn a_day_gone_to_validation_leaves_the_balance_axis() {
    let mut f = Vec::new();
    for i in 0..8 {
        // Days of the same camera, of different sizes.
        f.extend(frames(
            "yard",
            &format!("2026-10-0{}", i + 1),
            10 + 5 * i,
            "person",
        ));
    }
    let s = vec![source(f)];
    for seed in 0..10 {
        let o = Options {
            seed,
            balance: Balance {
                axes: Balance::parse_axes("camera-day").unwrap(),
                cap_dominant: Some(3.0),
                ..Balance::default()
            },
            ..opts()
        };
        let p = plan(&s, &o).unwrap();
        assert_whole_days(&s, &p);
        let report = p.balance.as_ref().unwrap();
        for g in &p.stats.valid_groups {
            assert!(!report.before.camera_days.iter().any(|(k, _)| k == g));
            assert!(!report.after.camera_days.iter().any(|(k, _)| k == g));
        }
        // Balancing runs on what is left and leaves the choice alone.
        let off = plan(
            &s,
            &Options {
                balance: Balance::default(),
                ..o.clone()
            },
        )
        .unwrap();
        assert_eq!(p.valid, off.valid);
        assert_eq!(p.stats.valid_groups, off.stats.valid_groups);
        assert_eq!(p.heldout_fingerprint, off.heldout_fingerprint);
        assert!(p.train.len() <= off.train.len());
    }
}

#[test]
fn hinted_exports_validate_on_their_train_split() {
    let mut f = days(8, 20);
    for x in &mut f {
        x.split = SplitHint::Train;
    }
    let mut held = frames("held", "2026-10-01", 20, "person");
    for x in &mut held {
        x.split = SplitHint::Heldout;
    }
    f.extend(held);
    let s = vec![source(f)];
    let p = plan(&s, &opts()).unwrap();
    assert_eq!(p.heldout.len(), 20);
    assert_eq!(groups_of(&s, p.heldout.iter().copied()).len(), 1);
    assert!(!p.stats.valid_groups.contains(&"held/2026-10-01".into()));
    assert_whole_days(&s, &p);
    assert_eq!(split_warning(&p), None);
}

#[test]
fn the_heldout_set_is_what_it_was() {
    // 12 days of 10 frames, seed 7, 20% held out. The value is from an independent Python
    // implementation of step 3 (SHA-256 of "7:<camera>/<day>", then of the sorted frame keys).
    let s = vec![source(days(12, 10))];
    let base = Options {
        seed: 7,
        heldout_share: 0.2,
        ..opts()
    };
    let p = plan(&s, &base).unwrap();
    assert_eq!(p.heldout_fingerprint, HELDOUT_GOLDEN);
    // Nothing about validation reaches it.
    for o in [
        Options {
            val_share: 0.3,
            ..base.clone()
        },
        Options {
            min_valid: 0,
            ..base.clone()
        },
        Options {
            scene: SceneFilter::DAY,
            ..base.clone()
        },
    ] {
        let q = plan(&s, &o).unwrap();
        assert_eq!(q.heldout, p.heldout);
        assert_eq!(q.heldout_fingerprint, HELDOUT_GOLDEN);
        assert_eq!(q.stats.heldout_groups, p.stats.heldout_groups);
    }
}

const HELDOUT_GOLDEN: &str = "5b17e6fb416ceb0a73649740a3aba71e85638f1de1c6f11a05d47c447f5b4398";
