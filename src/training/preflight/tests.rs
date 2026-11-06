//! The report's thresholds at their boundaries and the disk refusal, on synthetic plans.

use super::*;
use crate::training::dataset::{Frame, Options, SplitHint, YoloBox, plan};

fn cat() -> YoloBox {
    YoloBox {
        class: "cat".into(),
        cx: 0.5,
        cy: 0.5,
        w: 0.2,
        h: 0.2,
    }
}

fn frame(name: &str, scene: &str, boxes: Vec<YoloBox>, split: SplitHint) -> Frame {
    let mut parts = name.split('.');
    let _ = parts.next();
    let camera = parts.next().unwrap_or("c").to_string();
    let day = parts.next().unwrap_or("d").to_string();
    Frame {
        image: PathBuf::from("/export/images").join(format!("{name}.jpg")),
        boxes,
        ignore: Vec::new(),
        scene: Some(scene.into()),
        camera,
        day,
        split,
        hard_negative: false,
        human_rejected: false,
        bulk_accepted: false,
        size: Some((64, 48)),
        parked: Vec::new(),
    }
}

fn source(frames: Vec<Frame>) -> Source {
    Source {
        dir: PathBuf::from("/export"),
        public: None,
        classes: vec!["cat".into()],
        frames,
        warnings: Vec::new(),
        notes_known: true,
    }
}

/// `train` training frames with a cat of `scene` and `held` held-out ones.
fn export(train: usize, held: usize, scene: &str) -> Source {
    let mut v = Vec::new();
    for i in 0..train {
        let name = format!("{scene}.cam{}.2026-10-0{}.f{i}", i % 4, i % 3 + 1);
        v.push(frame(&name, scene, vec![cat()], SplitHint::Train));
    }
    for i in 0..held {
        let name = format!("{scene}.held.2026-11-01.h{i}");
        v.push(frame(&name, scene, vec![cat()], SplitHint::Heldout));
    }
    source(v)
}

fn opts() -> Options {
    Options {
        min_valid: 0,
        val_share: 0.0,
        ..Options::default()
    }
}

fn says(warnings: &[String], text: &str) -> bool {
    warnings.iter().any(|w| w.contains(text))
}

#[test]
fn few_boxes_of_a_class_warn_under_the_threshold_only() {
    let sources = [export(MIN_CLASS_BOXES, 20, "day")];
    let p = planned(&sources, &opts()).unwrap();
    let at = |boxes: usize| {
        let mut p = p.clone();
        p.stats.train_boxes.insert("cat".into(), boxes);
        advice(&p, &sources)
    };
    assert!(says(
        &at(MIN_CLASS_BOXES - 1),
        "Only 49 training box(es) of 'cat'"
    ));
    assert!(says(&at(MIN_CLASS_BOXES - 1), "Review more frames with it"));
    assert!(!says(&at(MIN_CLASS_BOXES), "training box(es)"));
    // No box at all is the plan's own warning, not a second one.
    assert!(!says(&at(0), "training box(es)"));
}

#[test]
fn few_held_out_frames_warn_none_refuse() {
    let sources = [export(60, MIN_HELDOUT_FRAMES, "day")];
    let p = planned(&sources, &opts()).unwrap();
    assert_eq!(p.heldout.len(), MIN_HELDOUT_FRAMES);
    assert!(!says(&p.warnings, "held-out frame(s)"));
    assert!(refusals(&p, &sources, &opts()).is_empty());

    let sources = [export(60, MIN_HELDOUT_FRAMES - 1, "day")];
    let p = planned(&sources, &opts()).unwrap();
    assert_eq!(p.heldout.len(), MIN_HELDOUT_FRAMES - 1);
    assert!(
        says(&p.warnings, "Only 9 held-out frame(s)"),
        "{:?}",
        p.warnings
    );
    assert!(
        refusals(&p, &sources, &opts()).is_empty(),
        "a warning, no refusal"
    );

    // Empty: refused with the next action.
    let mut p2 = p.clone();
    p2.heldout.clear();
    let r = refusals(&p2, &sources, &opts());
    assert_eq!(r.len(), 1, "{r:?}");
    assert!(
        r[0].contains("review more frames and export again"),
        "{r:?}"
    );
    assert!(check_plan(&p2, &sources, &opts()).is_err());
    // training.heldout_share 0 opts out: a warning stays, no refusal.
    let none = Options {
        heldout_share: 0.0,
        ..opts()
    };
    assert!(refusals(&p2, &sources, &none).is_empty());
}

#[test]
fn a_rare_scene_names_the_way_out() {
    let mut s = export(60, 12, "day");
    let night = export(12, 0, "night");
    s.frames.extend(night.frames);
    let sources = [s];
    let p = planned(&sources, &opts()).unwrap();
    assert!(
        says(
            &p.warnings,
            "night frame(s): review more, or train day only."
        ),
        "{:?}",
        p.warnings
    );
    // At the threshold, nothing.
    let at = |night: usize| {
        let mut s = export(60, 0, "day");
        s.frames.extend(export(night, 0, "night").frames);
        let p = Plan {
            classes: vec!["cat".into()],
            train: (0..s.frames.len()).map(|i| ((0, i), 1)).collect(),
            heldout: vec![(0, 0); MIN_HELDOUT_FRAMES],
            ..Plan::default()
        };
        advice(&p, &[s])
    };
    assert!(!says(&at(MIN_SCENE_FRAMES), "night frame(s)"));
    assert!(says(&at(MIN_SCENE_FRAMES - 1), "Only 49 night frame(s)"));
    // A night model has no other scene to suggest.
    let sources = [export(12, 12, "night")];
    let p = planned(&sources, &opts()).unwrap();
    assert!(
        says(
            &p.warnings,
            "night frame(s) to train on: review more of them."
        ),
        "{:?}",
        p.warnings
    );
}

#[test]
fn the_report_counts_splits_scenes_and_the_bulk_share() {
    let mut s = export(60, 10, "day");
    for f in s.frames.iter_mut().take(6) {
        f.bulk_accepted = true;
    }
    s.frames.push(frame(
        "night.cam9.2026-10-05.e1",
        "night",
        Vec::new(),
        SplitHint::Train,
    ));
    let sources = [s];
    let o = opts();
    let p = planned(&sources, &o).unwrap();
    let r = report(&p, &sources, &o, Some(7));
    assert_eq!(r.train_frames, p.train.len());
    assert_eq!(r.heldout_frames, 10);
    assert_eq!(r.classes.len(), 1);
    assert_eq!(r.classes[0].train + r.classes[0].valid, 60);
    assert_eq!(r.classes[0].heldout, 10);
    assert_eq!(r.classes[0].by_scene.day, r.classes[0].train);
    assert_eq!(r.scenes.day + r.scenes.night, r.train_frames);
    assert_eq!(r.bulk_accepted, Some(6));
    assert_eq!(r.own_frames, r.train_frames + r.valid_frames);
    assert!(r.notes.iter().any(|n| n.contains("6 of ")), "{:?}", r.notes);
    assert!(
        r.notes
            .iter()
            .any(|n| n.contains("7 frame(s) in the folder"))
    );
    assert!(text(&r).contains("cat: "), "{}", text(&r));
    // An export of an older version: said so, no number.
    let mut old = sources[0].clone();
    old.notes_known = false;
    let sources = [old];
    let r = report(&p, &sources, &o, None);
    assert_eq!(r.bulk_accepted, None);
    assert!(
        r.notes.iter().any(|n| n.contains("not known")),
        "{:?}",
        r.notes
    );
}

#[test]
fn a_negative_counts_as_hard_when_confirmed_or_rejected_on() {
    let mut s = export(60, 10, "day");
    for (i, (confirmed, rejected)) in [(true, false), (false, true), (false, false)]
        .into_iter()
        .enumerate()
    {
        let mut f = frame(
            &format!("day.cam0.2026-10-01.n{i}"),
            "day",
            Vec::new(),
            SplitHint::Train,
        );
        f.hard_negative = confirmed;
        f.human_rejected = rejected;
        s.frames.push(f);
    }
    let sources = [s];
    let p = plan(&sources, &opts()).unwrap();
    let in_train = |keep: fn(&Frame) -> bool| {
        p.train
            .iter()
            .filter(|((_, i), _)| {
                let f = &sources[0].frames[*i];
                f.boxes.is_empty() && keep(f)
            })
            .count()
    };
    assert_eq!(p.stats.train_negatives, in_train(|_| true));
    assert_eq!(
        p.stats.hard_negatives,
        in_train(|f| f.hard_negative || f.human_rejected)
    );
    assert!(p.stats.train_negatives >= 2);
}

const GB: u64 = 1 << 30;

fn need(dir: &str, bytes: u64) -> VolumeNeed {
    VolumeNeed {
        dir: PathBuf::from(dir),
        bytes,
        advice: TRAINING_ADVICE,
    }
}

#[test]
fn the_disk_refusal_is_exact_at_the_margin_and_names_the_numbers() {
    // 100 GB volume: it keeps 5 GB free.
    let s = Space {
        available: 20 * GB,
        total: 100 * GB,
    };
    assert_eq!(s.margin(), 5 * GB);
    assert_eq!(refusal(&[(need("/t", 15 * GB), Some(s))]), None);
    let m = refusal(&[(need("/t", 15 * GB + 1), Some(s))]).unwrap();
    assert!(m.contains("Not enough disk space"), "{m}");
    assert!(m.contains("16.1 GB under /t"), "{m}");
    assert!(m.contains("21.5 GB is free"), "{m}");
    assert!(m.contains("Training folder"), "{m}");
    // Space unknown: no refusal.
    assert_eq!(refusal(&[(need("/t", u64::MAX), None)]), None);
    // Two short volumes: both named.
    let m = refusal(&[
        (need("/a", 99 * GB), Some(s)),
        (need("/b", 99 * GB), Some(s)),
    ])
    .unwrap();
    assert!(m.contains("/a") && m.contains("/b"), "{m}");
    let line = disk_line(&need("/t", GB), Some(s));
    assert_eq!(line.text, "about 1.1 GB needed under /t, 21.5 GB free");
    assert_eq!(line.keep, Some(5 * GB));
}

#[test]
fn a_real_volume_refuses_what_cannot_fit_and_needs_add_up() {
    let dir = std::env::temp_dir();
    REAL_DISK.with(|r| r.set(true));
    let e = check_disk(vec![VolumeNeed {
        dir: dir.clone(),
        bytes: u64::MAX / 2,
        advice: TOOLS_ADVICE,
    }])
    .unwrap_err()
    .to_string();
    assert!(e.contains("Not enough disk space"), "{e}");
    assert!(
        check_disk(vec![VolumeNeed {
            dir: dir.clone(),
            bytes: 0,
            advice: TOOLS_ADVICE,
        }])
        .is_ok()
    );
    let merged = combine(vec![
        VolumeNeed {
            dir: dir.clone(),
            bytes: 5,
            advice: TOOLS_ADVICE,
        },
        VolumeNeed {
            dir: dir.join("not-there-yet"),
            bytes: 7,
            advice: MODELS_ADVICE,
        },
    ]);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].bytes, 12);
}

#[test]
fn frames_that_cannot_be_linked_count_as_copies() {
    let root = std::env::temp_dir().join(format!("bop-preflight-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(root.join("images")).unwrap();
    let img = root.join("images/day.a.2026-10-01.x.jpg");
    std::fs::write(&img, vec![0u8; 1000]).unwrap();
    let mut f = frame("day.a.2026-10-01.x", "day", vec![cat()], SplitHint::Train);
    f.image = img.clone();
    let mut s = source(vec![f]);
    s.dir = root.clone();
    let sources = [s];
    let p = Plan {
        classes: vec!["cat".into()],
        train: vec![((0, 0), 1)],
        ..Plan::default()
    };
    // Same volume, hard links work: nothing to copy, and the probe left nothing behind.
    assert_eq!(copied_bytes(&p, &sources, &root.join("run")), 0);
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
    // The base weights set the rest of the need.
    let recipe = TrainRecipe::new(crate::resources::catalog::TrainFamily::Yolo26, "n").unwrap();
    let n = run_needs(&p, &sources, &recipe, &root.join("run"), &root.join("m")).unwrap();
    let w = recipe.base().unwrap().size();
    assert_eq!(n[0].bytes, LABEL_BYTES_PER_FRAME + WORK_WEIGHT_FACTOR * w);
    assert_eq!(n[1].bytes, w);
    let _ = std::fs::remove_dir_all(&root);
}
