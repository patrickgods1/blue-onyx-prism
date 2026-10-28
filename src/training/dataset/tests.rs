//! Dataset assembly tests on synthetic YOLO exports (tiny generated JPEGs).

use super::*;

struct Tmp(PathBuf);

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn tmp(tag: &str) -> Tmp {
    let d = std::env::temp_dir().join(format!("bop-train-ds-{tag}-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&d).unwrap();
    Tmp(d)
}

fn jpg(path: &Path, w: u32, h: u32) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    image::RgbImage::from_pixel(w, h, image::Rgb([90, 90, 90]))
        .save(path)
        .unwrap();
}

/// A flat export: `classes`, and per frame (name, label lines, tags).
fn flat_export(dir: &Path, classes: &[&str], frames: &[(&str, &str, &[&str])]) {
    std::fs::write(dir.join("classes.txt"), classes.join("\n")).unwrap();
    let mut images = Vec::new();
    for (name, labels, tags) in frames {
        jpg(&dir.join("images").join(name), 64, 48);
        let s = Path::new(name)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        std::fs::create_dir_all(dir.join("labels")).unwrap();
        std::fs::write(dir.join("labels").join(format!("{s}.txt")), labels).unwrap();
        images.push(serde_json::json!({"file": name, "width": 64, "height": 48, "tags": tags}));
    }
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec(&serde_json::json!({"images": images})).unwrap(),
    )
    .unwrap();
}

/// 3 cameras x 2 days, 4 frames each: 3 with a cat or dog, 1 empty (or 2 empty on day 2).
fn camera_export(dir: &Path) -> usize {
    let mut frames: Vec<(String, String, Vec<String>)> = Vec::new();
    for cam in ["drive", "yard", "gate"] {
        for day in ["2026-10-01", "2026-10-02"] {
            for i in 0..4 {
                let name = format!("{cam}-{day}-{i}.jpg");
                let label = match i {
                    0 => "0 0.5 0.5 0.2 0.4\n".to_string(),
                    1 => "1 0.25 0.25 0.1 0.1\n0 0.7 0.7 0.2 0.2\n".to_string(),
                    2 => "1 0.5 0.5 0.5 0.5\n".to_string(),
                    _ => String::new(),
                };
                let scene = if day.ends_with('1') { "day" } else { "ir" };
                frames.push((
                    name,
                    label,
                    vec![scene.into(), format!("camera:{cam}"), format!("date:{day}")],
                ));
            }
        }
    }
    let refs: Vec<(&str, &str, Vec<&str>)> = frames
        .iter()
        .map(|(n, l, t)| {
            (
                n.as_str(),
                l.as_str(),
                t.iter().map(String::as_str).collect(),
            )
        })
        .collect();
    let refs2: Vec<(&str, &str, &[&str])> = refs
        .iter()
        .map(|(n, l, t)| (*n, *l, t.as_slice()))
        .collect();
    flat_export(dir, &["cat", "dog"], &refs2);
    frames.len()
}

fn group(s: &[Source], r: FrameRef) -> String {
    let f = &s[r.0].frames[r.1];
    format!("{}/{}", f.camera, f.day)
}

#[test]
fn reads_a_flat_export_with_manifest_tags() {
    let t = tmp("flat");
    let n = camera_export(&t.0);
    let s = read_export(&t.0).unwrap();
    assert_eq!(s.classes, ["cat", "dog"]);
    assert_eq!(s.frames.len(), n);
    let f = s
        .frames
        .iter()
        .find(|f| f.image.ends_with("yard-2026-10-02-1.jpg"))
        .unwrap();
    assert_eq!(f.camera, "yard");
    assert_eq!(f.day, "2026-10-02");
    assert_eq!(f.scene.as_deref(), Some("ir"));
    assert_eq!(f.boxes.len(), 2);
    assert_eq!(f.boxes[0].class, "dog");
    assert_eq!(f.split, SplitHint::None);
    assert_eq!(f.size, Some((64, 48)));
}

/// macOS AppleDouble files (`._x.jpg`, written beside every file copied to a non-APFS drive)
/// are not frames.
#[test]
fn hidden_files_are_not_frames() {
    let t = tmp("hidden");
    let n = camera_export(&t.0);
    std::fs::write(
        t.0.join("images/._yard-2026-10-02-1.jpg"),
        b"\0\x05\x16\x07",
    )
    .unwrap();
    let s = read_export(&t.0).unwrap();
    assert_eq!(s.frames.len(), n);
}

#[test]
fn heldout_frames_come_from_other_cameras_and_days() {
    let t = tmp("heldout");
    camera_export(&t.0);
    let s = vec![read_export(&t.0).unwrap()];
    let opts = Options {
        heldout_share: 0.3,
        ..Options::default()
    };
    let p = plan(&s, &opts).unwrap();
    assert!(!p.heldout.is_empty());
    let held: BTreeSet<String> = p.heldout.iter().map(|r| group(&s, *r)).collect();
    let trained: BTreeSet<String> = p
        .train
        .iter()
        .map(|(r, _)| group(&s, *r))
        .chain(p.valid.iter().map(|r| group(&s, *r)))
        .collect();
    assert!(held.is_disjoint(&trained), "{held:?} vs {trained:?}");
    assert_eq!(held, p.stats.heldout_groups.iter().cloned().collect());
    // About 30% of 24 frames: two groups of 4.
    assert_eq!(p.heldout.len(), 8);
    // Deterministic.
    assert_eq!(plan(&s, &opts).unwrap(), p);
    // Another seed may pick other groups but keeps the sizes.
    let other = plan(
        &s,
        &Options {
            seed: 7,
            ..opts.clone()
        },
    )
    .unwrap();
    assert_eq!(other.heldout.len(), 8);
}

#[test]
fn negatives_are_capped_near_the_target_share() {
    let t = tmp("neg");
    let mut frames = Vec::new();
    for i in 0..10 {
        frames.push((format!("p{i}.jpg"), "0 0.5 0.5 0.2 0.2\n"));
    }
    for i in 0..20 {
        frames.push((format!("n{i}.jpg"), ""));
    }
    let tags: &[&str] = &["day", "camera:a", "date:2026-01-01"];
    let refs: Vec<(&str, &str, &[&str])> =
        frames.iter().map(|(n, l)| (n.as_str(), *l, tags)).collect();
    flat_export(&t.0, &["person"], &refs);
    let s = vec![read_export(&t.0).unwrap()];
    let opts = Options {
        heldout_share: 0.0,
        negative_share: 0.2,
        val_share: 0.0,
        ..Options::default()
    };
    let p = plan(&s, &opts).unwrap();
    // 10 positives -> 10 x 0.2 / 0.8 = 2.5 -> 3 negatives (rounded), the other 17 dropped.
    assert_eq!(p.stats.negatives_dropped, 17);
    assert_eq!(
        p.stats.train_negatives + p.stats.train_positives,
        p.train.len()
    );
    // 13 frames: one validates, 12 train.
    assert_eq!(p.stats.train_positives + p.stats.train_negatives, 12);
    assert_eq!(p.valid.len(), 1);
    assert!(p.heldout.is_empty());
    assert!(
        p.warnings
            .iter()
            .any(|w| w.contains("no frame is held out"))
    );
    // Too few negatives: a warning, nothing dropped.
    let few = Options {
        negative_share: 0.9,
        ..opts
    };
    let p = plan(&s, &few).unwrap();
    assert_eq!(p.stats.negatives_dropped, 0);
    assert!(
        p.warnings.iter().any(|w| w.contains("hard negatives")),
        "{:?}",
        p.warnings
    );
}

#[test]
fn class_lists_are_mapped_by_name_across_sources() {
    let (a, b) = (tmp("cls-a"), tmp("cls-b"));
    let tags: &[&str] = &["night", "camera:c", "date:2026-02-02"];
    flat_export(
        &a.0,
        &["cat", "dog"],
        &[("a1.jpg", "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n", tags)],
    );
    flat_export(
        &b.0,
        &["dog", "person", "motorbike", "tree"],
        &[(
            "b1.jpg",
            "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.2 0.3\n2 0.6 0.6 0.1 0.1\n3 0.1 0.1 0.1 0.1\n",
            tags,
        )],
    );
    let s = vec![read_export(&a.0).unwrap(), read_export(&b.0).unwrap()];
    let opts = Options {
        rename: BTreeMap::from([
            ("motorbike".to_string(), "motorcycle".to_string()),
            ("tree".to_string(), String::new()),
        ]),
        heldout_share: 0.0,
        val_share: 0.0,
        ..Options::default()
    };
    let p = plan(&s, &opts).unwrap();
    assert_eq!(p.classes, ["cat", "dog", "person", "motorcycle"]);
    assert_eq!(p.stats.boxes_dropped.get(""), Some(&1));
    // An explicit list drops the others.
    let only = Options {
        classes: Some(vec!["dog".into(), "cat".into()]),
        ..opts.clone()
    };
    let p2 = plan(&s, &only).unwrap();
    assert_eq!(p2.classes, ["dog", "cat"]);
    assert_eq!(p2.stats.boxes_dropped.get("person"), Some(&1));
    assert_eq!(p2.stats.boxes_dropped.get("motorcycle"), Some(&1));
    let out = tmp("cls-out");
    let dir = out.0.join("data");
    let prep = write(&s, &p2, &only, &dir).unwrap();
    assert_eq!(prep.classes, ["dog", "cat"]);
    // Source a: cat (class 0 there) -> 1, dog -> 0; source b: dog -> 0, the rest dropped.
    let la = std::fs::read_to_string(dir.join("labels/s0-a1.txt")).unwrap();
    assert_eq!(
        la,
        "1 0.500000 0.500000 0.200000 0.200000\n0 0.200000 0.200000 0.100000 0.100000\n"
    );
    let lb = std::fs::read_to_string(dir.join("labels/s1-b1.txt")).unwrap();
    assert_eq!(lb, "0 0.500000 0.500000 0.200000 0.200000\n");
    let coco: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("train.coco.json")).unwrap()).unwrap();
    let cats: Vec<(i64, String)> = coco["categories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            (
                c["id"].as_i64().unwrap(),
                c["name"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    assert_eq!(cats, [(0, "dog".to_string()), (1, "cat".to_string())]);
    // One frame trains, the other validates (at least one validation frame): both files
    // together hold the three mapped boxes.
    let valid: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("valid.coco.json")).unwrap()).unwrap();
    let mut ids: Vec<i64> = [&coco, &valid]
        .iter()
        .flat_map(|c| c["annotations"].as_array().unwrap().clone())
        .map(|a| a["category_id"].as_i64().unwrap())
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, [0, 0, 1]);
}

#[test]
fn writes_yolo_coco_and_a_loadable_heldout_set() {
    let t = tmp("write");
    camera_export(&t.0);
    let s = vec![read_export(&t.0).unwrap()];
    let opts = Options::default();
    let p = plan(&s, &opts).unwrap();
    let out = tmp("write-out");
    let dir = out.0.join("data");
    let prep = write(&s, &p, &opts, &dir).unwrap();
    assert_eq!(prep.train_images, p.train.len());
    assert_eq!(prep.valid_images, p.valid.len());
    let train = std::fs::read_to_string(dir.join("train.txt")).unwrap();
    assert_eq!(train.lines().count(), p.train.len());
    for l in train.lines() {
        let img = dir.join(l.trim_start_matches("./"));
        assert!(img.is_file(), "{l}");
        let lbl = dir.join("labels").join(format!("{}.txt", stem(&img)));
        assert!(lbl.is_file(), "{}", lbl.display());
    }
    let yaml = std::fs::read_to_string(dir.join("data.yaml")).unwrap();
    assert!(yaml.contains("train: train.txt\nval: valid.txt"), "{yaml}");
    assert_eq!(images::parse_yolo_yaml(&yaml).unwrap(), ["cat", "dog"]);
    // COCO: pixels of a 64x48 frame; YOLO (0.5, 0.5, 0.2, 0.4) -> [25.6, 14.4, 12.8, 19.2].
    let coco: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("train.coco.json")).unwrap()).unwrap();
    let images_json = coco["images"].as_array().unwrap();
    assert_eq!(images_json.len(), p.train.len());
    // A specific frame: the first training frame `<cam>-<day>-0.jpg` (one cat box). Frames
    // `-1` also hold a cat box 12.8 px wide, so the box is looked up by its image, not its size.
    let (r0, _) = p
        .train
        .iter()
        .find(|(r, _)| file_name(&s[0].frames[r.1].image).ends_with("-0.jpg"))
        .expect("a frame 0 in training");
    let want = format!("images/s0-{}", file_name(&s[0].frames[r0.1].image));
    let image_id = images_json
        .iter()
        .find(|i| i["file_name"] == want.as_str())
        .unwrap_or_else(|| panic!("{want} in train.coco.json"))["id"]
        .clone();
    let anns: Vec<&serde_json::Value> = coco["annotations"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|a| a["image_id"] == image_id)
        .collect();
    assert_eq!(anns.len(), 1, "{anns:?}");
    assert_eq!(anns[0]["category_id"], 0);
    assert_eq!(anns[0]["bbox"], serde_json::json!([25.6, 14.4, 12.8, 19.2]));
    assert_eq!(anns[0]["area"], serde_json::json!(245.76));
    // Deterministic: the same plan and the same files on every run.
    assert_eq!(plan(&s, &opts).unwrap(), p);
    for img in images_json {
        assert!(dir.join(img["file_name"].as_str().unwrap()).is_file());
    }
    // The held-out set is a benchmark `dir:` set with YOLO ground truth.
    let hd = prep.heldout_dir.clone().unwrap();
    let (set, warnings) = images::load_dir(&hd, None, Some("heldout")).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(set.images.len(), p.heldout.len());
    assert_eq!(set.ground_truth, images::GroundTruth::Yolo);
    assert_eq!(
        set.scored_labels.as_deref(),
        Some(&["cat".to_string(), "dog".to_string()][..])
    );
    let boxes: usize = set
        .images
        .iter()
        .map(|i| i.gt.as_ref().unwrap().len())
        .sum();
    let expected: usize = p
        .heldout
        .iter()
        .map(|r| s[r.0].frames[r.1].boxes.len())
        .sum();
    assert_eq!(boxes, expected);
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("dataset.json")).unwrap()).unwrap();
    assert_eq!(summary["sources"][0]["kind"], "review");
    assert_eq!(read_summary_sources(&dir).len(), 1);
    // Not into a non-empty folder.
    assert!(
        write(&s, &p, &opts, &dir)
            .unwrap_err()
            .to_string()
            .contains("not empty")
    );
}

#[test]
fn the_review_export_layout_keeps_its_val_split_as_heldout() {
    let t = tmp("review");
    let d = &t.0;
    std::fs::write(d.join("classes.txt"), "person\nraccoon\n").unwrap();
    for (split, name, label) in [
        (
            "train",
            "ir.driveway.2026-10-20.20261020T010203004Z_aa.jpg",
            "1 0.5 0.5 0.3 0.3\n",
        ),
        (
            "train",
            "day.driveway.2026-10-21.20261021T120000000Z_bb.jpg",
            "0 0.4 0.5 0.1 0.4\n",
        ),
        (
            "train",
            "night.yard.2026-10-21.20261021T230000000Z_cc.jpg",
            "",
        ),
        (
            "val",
            "ir.gate.2026-10-22.20261022T020000000Z_dd.jpg",
            "1 0.5 0.5 0.2 0.2\n",
        ),
    ] {
        jpg(&d.join("images").join(split).join(name), 32, 32);
        let s = name.trim_end_matches(".jpg");
        std::fs::create_dir_all(d.join("labels").join(split)).unwrap();
        std::fs::write(d.join("labels").join(split).join(format!("{s}.txt")), label).unwrap();
    }
    let src = read_export(d).unwrap();
    let held: Vec<&Frame> = src
        .frames
        .iter()
        .filter(|f| f.split == SplitHint::Heldout)
        .collect();
    assert_eq!(held.len(), 1);
    assert_eq!(
        (held[0].camera.as_str(), held[0].day.as_str()),
        ("gate", "2026-10-22")
    );
    assert_eq!(held[0].scene.as_deref(), Some("ir"));
    let s = vec![src];
    let p = plan(&s, &Options::default()).unwrap();
    assert_eq!(p.heldout.len(), 1);
    assert!(p.stats.heldout_groups.is_empty(), "the export chose");
    // Night + IR: the day frame is out.
    let night = Options {
        scene: SceneFilter::NIGHT_IR,
        ..Options::default()
    };
    let p = plan(&s, &night).unwrap();
    assert_eq!(p.stats.scene_filtered, 1);
    let trained: Vec<&str> = p
        .train
        .iter()
        .map(|(r, _)| *r)
        .chain(p.valid.iter().copied())
        .map(|r| s[0].frames[r.1].scene.as_deref().unwrap())
        .collect();
    assert!(trained.iter().all(|sc| *sc != "day"), "{trained:?}");
}

#[test]
fn public_frames_train_only_and_review_frames_repeat() {
    let (r, pubd) = (tmp("mix-review"), tmp("mix-public"));
    camera_export(&r.0);
    jpg(&pubd.0.join("lila1.jpg"), 100, 50);
    jpg(&pubd.0.join("lila2.jpg"), 100, 50);
    jpg(&pubd.0.join("lila3.jpg"), 100, 50);
    let m = Manifest::parse(
        r#"{"id": "lila-night", "license": "CDLA-Permissive-1.0", "images": [
            {"file": "lila1.jpg", "width": 100, "height": 50, "tags": ["night", "ir"],
             "objects": [{"label": "raccoon", "bbox": [10, 10, 30, 40]}]},
            {"file": "lila2.jpg", "tags": ["ir", "empty"], "objects": []},
            {"file": "lila3.jpg", "tags": ["ir"], "objects": [{"label": "deer", "bbox": [1, 1, 5, 5], "ignore": true}]},
            {"file": "missing.jpg", "objects": []}
        ]}"#,
    )
    .unwrap();
    let public = read_public_set(&pubd.0, &m).unwrap();
    assert_eq!(
        public.frames.len(),
        2,
        "ignore regions and missing images are left out"
    );
    let b = &public.frames[0].boxes[0];
    for (got, want) in [(b.cx, 0.2), (b.cy, 0.5), (b.w, 0.2), (b.h, 0.6)] {
        assert!((got - want).abs() < 1e-6, "{b:?}");
    }
    assert!(public.frames[1].hard_negative);
    assert_eq!(public.frames[1].size, Some((100, 50)));
    let s = vec![read_export(&r.0).unwrap(), public];
    let opts = Options {
        negative_share: 0.5,
        ..Options::default()
    };
    let p = plan(&s, &opts).unwrap();
    assert_eq!(p.classes, ["cat", "dog", "raccoon"]);
    assert!(
        p.heldout.iter().all(|r| r.0 == 0),
        "public frames are never held out"
    );
    assert!(p.valid.iter().all(|r| r.0 == 0));
    for (r, rep) in &p.train {
        assert_eq!(*rep, if r.0 == 0 { 3 } else { 1 }, "{r:?}");
    }
    assert_eq!(p.stats.public_frames, 2);
    let out = tmp("mix-out");
    let prep = write(&s, &p, &opts, &out.0.join("data")).unwrap();
    let reviewed_train = p.train.iter().filter(|(r, _)| r.0 == 0).count();
    assert_eq!(prep.train_images, reviewed_train * 3 + 2);
    let summary = read_summary_sources(&out.0.join("data"));
    assert_eq!(summary[1].kind, "lila-night");
    assert_eq!(summary[1].license, "CDLA-Permissive-1.0");
    assert_eq!(summary[1].used_heldout, 0);
}

#[test]
fn label_parsing_names_the_problem() {
    let classes = vec!["a".to_string(), "b".to_string()];
    let ok = parse_labels(
        "1 0.5 0.5 0.2 0.2\n\n0 0.1 0.1 0.3 0.1 0.3 0.4\n0 0.5 0.5 0 0.1\n",
        &classes,
    )
    .unwrap();
    assert_eq!(ok.len(), 2, "the zero-size box is dropped");
    assert_eq!(ok[0].class, "b");
    // Polygon -> its bounding box.
    assert!(
        (ok[1].w - 0.2).abs() < 1e-6 && (ok[1].h - 0.3).abs() < 1e-6,
        "{:?}",
        ok[1]
    );
    let e = parse_labels("2 0.5 0.5 0.1 0.1", &classes)
        .unwrap_err()
        .to_string();
    assert!(e.contains("line 1") && e.contains("class 2"), "{e}");
    let e = parse_labels("0 0.5 1.5 0.1 0.1", &classes)
        .unwrap_err()
        .to_string();
    assert!(e.contains("normalized"), "{e}");
    let e = parse_labels("0 0.5 0.5", &classes).unwrap_err().to_string();
    assert!(e.contains("found 2 values"), "{e}");
}

#[test]
fn errors_say_what_is_missing() {
    let t = tmp("err");
    let e = read_export(&t.0).unwrap_err().to_string();
    assert!(e.contains("classes.txt"), "{e}");
    std::fs::write(t.0.join("classes.txt"), "a\n").unwrap();
    let e = read_export(&t.0).unwrap_err().to_string();
    assert!(e.contains("no .jpg"), "{e}");
    flat_export(&t.0, &["a"], &[("x.jpg", "", &["day"])]);
    let s = vec![read_export(&t.0).unwrap()];
    let e = plan(&s, &Options::default()).unwrap_err().to_string();
    assert!(e.contains("no training frame has a box"), "{e}");
    let e = plan(
        &s,
        &Options {
            classes: Some(vec!["a".into(), "a".into()]),
            ..Options::default()
        },
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("duplicate"), "{e}");
}

#[test]
fn scene_filters_parse() {
    let nf = SceneFilter::new;
    for (text, want) in [
        ("all", SceneFilter::ALL_SCENES),
        ("", SceneFilter::ALL_SCENES),
        ("day", SceneFilter::DAY),
        ("night+ir", SceneFilter::NIGHT_IR),
        ("night-ir", SceneFilter::NIGHT_IR),
        ("nightir", SceneFilter::NIGHT_IR),
        ("night ir", SceneFilter::NIGHT_IR),
        ("Night, IR", SceneFilter::NIGHT_IR),
        // Behaviour change: one scene alone is that scene only.
        ("night", nf(false, true, false)),
        ("IR", nf(false, false, true)),
        ("day+ir", nf(true, false, true)),
        ("ir+day", nf(true, false, true)),
        ("day night", nf(true, true, false)),
        ("day+night+ir", SceneFilter::ALL_SCENES),
    ] {
        assert_eq!(SceneFilter::parse(text), Some(want), "{text}");
        // Serde accepts every spelling.
        let j = serde_json::to_string(text).unwrap();
        assert_eq!(
            serde_json::from_str::<SceneFilter>(&j).unwrap(),
            want,
            "{text}"
        );
    }
    assert_eq!(SceneFilter::parse("dusk"), None);
    assert_eq!(SceneFilter::parse("day+dusk"), None);
    assert!(serde_json::from_str::<SceneFilter>(r#""dusk""#).is_err());
    // Canonical strings round-trip; none ticked equals all.
    let canon = [
        ("all", "custom"),
        ("day", "day"),
        ("night", "night-only"),
        ("ir", "ir"),
        ("day+night", "day-night"),
        ("day+ir", "day-ir"),
        ("night+ir", "night"),
    ];
    for (c, label) in canon {
        let f = SceneFilter::parse(c).unwrap();
        assert_eq!(f.as_str(), c);
        assert_eq!(f.to_string(), c);
        assert_eq!(f.label(), label);
        let j = serde_json::to_string(&f).unwrap();
        assert_eq!(j, format!("\"{c}\""));
        assert_eq!(serde_json::from_str::<SceneFilter>(&j).unwrap(), f);
    }
    assert_eq!(nf(false, false, false), SceneFilter::ALL_SCENES);
    assert_eq!(nf(false, false, false).as_str(), "all");
    assert_eq!(SceneFilter::default(), SceneFilter::ALL_SCENES);
    // keeps
    assert!(SceneFilter::NIGHT_IR.keeps(Some("ir")));
    assert!(SceneFilter::NIGHT_IR.keeps(Some("night")));
    assert!(!SceneFilter::NIGHT_IR.keeps(Some("day")));
    assert!(!SceneFilter::NIGHT_IR.keeps(None));
    assert!(nf(false, true, false).keeps(Some("night")));
    assert!(!nf(false, true, false).keeps(Some("ir")));
    assert!(SceneFilter::ALL_SCENES.keeps(None));
    assert!(nf(false, false, false).keeps(None));
    assert!(!SceneFilter::DAY.keeps(None));
    // is_night: night and / or IR without day.
    assert!(SceneFilter::NIGHT_IR.is_night());
    assert!(nf(false, true, false).is_night() && nf(false, false, true).is_night());
    assert!(!SceneFilter::DAY.is_night() && !SceneFilter::ALL_SCENES.is_night());
    assert!(!nf(true, true, false).is_night());
    assert_eq!(SceneFilter::CHOICES, ["day", "night", "ir"]);
    // Excluded
    assert_eq!(Excluded::default(), Excluded::Background);
    for e in [Excluded::Background, Excluded::DropFrame] {
        assert_eq!(Excluded::parse(e.as_str()), Some(e));
        let j = serde_json::to_string(&e).unwrap();
        assert_eq!(j, format!("\"{}\"", e.as_str()));
    }
    assert_eq!(Excluded::parse("drop_frame"), Some(Excluded::DropFrame));
    assert_eq!(Excluded::parse("x"), None);
    // Options of older versions (no `excluded`) still read.
    let o: Options = serde_json::from_str(
        r#"{"classes": null, "scene": "nightir", "negative_share": 0.2, "heldout_share": 0.15,
            "val_share": 0.1, "oversample": 3, "seed": 0}"#,
    )
    .unwrap();
    assert_eq!(o.scene, SceneFilter::NIGHT_IR);
    assert_eq!(o.excluded, Excluded::Background);
    let meta = meta_from_name("20261022T123456789Z_ab12");
    assert_eq!(meta.day.as_deref(), Some("2026-10-22"));
    assert_eq!(meta.camera, None);
}

/// 5 cameras x 3 days (day 1 day, day 2 night, day 3 IR), 4 frames each (cat, dog + cat, dog,
/// empty).
fn scene_export(dir: &Path) {
    let mut frames: Vec<(String, String, Vec<String>)> = Vec::new();
    for cam in ["a", "b", "c", "d", "e"] {
        for (day, scene) in [
            ("2026-10-01", "day"),
            ("2026-10-02", "night"),
            ("2026-10-03", "ir"),
        ] {
            for i in 0..4 {
                let label = match i {
                    0 => "0 0.5 0.5 0.2 0.4\n",
                    1 => "1 0.25 0.25 0.1 0.1\n0 0.7 0.7 0.2 0.2\n",
                    2 => "1 0.5 0.5 0.5 0.5\n",
                    _ => "",
                };
                frames.push((
                    format!("{cam}-{day}-{i}.jpg"),
                    label.into(),
                    vec![scene.into(), format!("camera:{cam}"), format!("date:{day}")],
                ));
            }
        }
    }
    let tags: Vec<Vec<&str>> = frames
        .iter()
        .map(|(_, _, t)| t.iter().map(String::as_str).collect())
        .collect();
    let refs: Vec<(&str, &str, &[&str])> = frames
        .iter()
        .zip(&tags)
        .map(|((n, l, _), t)| (n.as_str(), l.as_str(), t.as_slice()))
        .collect();
    flat_export(dir, &["cat", "dog"], &refs);
}

fn scenes_of(s: &[Source], refs: impl Iterator<Item = FrameRef>) -> BTreeSet<String> {
    refs.map(|r| s[r.0].frames[r.1].scene.clone().unwrap_or_default())
        .collect()
}

/// Runs with other scene filters hold out the same frames (every scene), so their benchmarks
/// compare; only the train / valid pool follows the filter.
#[test]
fn the_heldout_set_is_the_same_for_every_scene_filter() {
    let t = tmp("scene-held");
    scene_export(&t.0);
    let s = vec![read_export(&t.0).unwrap()];
    let base = Options {
        heldout_share: 0.2,
        ..Options::default()
    };
    let plans: Vec<(SceneFilter, Plan)> = [
        SceneFilter::ALL_SCENES,
        SceneFilter::DAY,
        SceneFilter::NIGHT_IR,
        SceneFilter::new(false, true, false),
    ]
    .into_iter()
    .map(|scene| {
        let o = Options {
            scene,
            ..base.clone()
        };
        (scene, plan(&s, &o).unwrap())
    })
    .collect();
    let (_, all) = &plans[0];
    assert_eq!(all.heldout.len(), 12, "3 of 15 camera/day groups");
    assert_eq!(all.heldout_fingerprint.len(), 64);
    assert!(
        all.heldout_fingerprint
            .chars()
            .all(|c| c.is_ascii_hexdigit())
    );
    for (scene, p) in &plans {
        assert_eq!(p.heldout, all.heldout, "{scene}");
        assert_eq!(p.heldout_fingerprint, all.heldout_fingerprint, "{scene}");
        assert_eq!(p.stats.heldout_groups, all.stats.heldout_groups, "{scene}");
        let trained = scenes_of(
            &s,
            p.train
                .iter()
                .map(|(r, _)| *r)
                .chain(p.valid.iter().copied()),
        );
        let want: BTreeSet<String> = ["day", "night", "ir"]
            .into_iter()
            .filter(|x| scene.keeps(Some(x)))
            .map(String::from)
            .collect();
        assert_eq!(trained, want, "{scene}");
        // Scene-filtered = train / valid frames of the other scenes.
        let pool = 60 - p.heldout.len();
        let kept = (0..60)
            .filter(|&i| !p.heldout.contains(&(0, i)))
            .filter(|&i| scene.keeps(s[0].frames[i].scene.as_deref()))
            .count();
        assert_eq!(p.stats.scene_filtered, pool - kept, "{scene}");
    }
    // Day and night + IR train on disjoint frames.
    let day: BTreeSet<FrameRef> = plans[1].1.train.iter().map(|(r, _)| *r).collect();
    let night: BTreeSet<FrameRef> = plans[2].1.train.iter().map(|(r, _)| *r).collect();
    assert!(day.is_disjoint(&night));
    // Another held-out set, another fingerprint.
    let other = plan(
        &s,
        &Options {
            seed: 9,
            ..base.clone()
        },
    )
    .unwrap();
    if other.heldout != all.heldout {
        assert_ne!(other.heldout_fingerprint, all.heldout_fingerprint);
    }
    // Written into dataset.json and the held-out manifest's tags.
    let o = Options {
        scene: SceneFilter::NIGHT_IR,
        ..base
    };
    let p = &plans[2].1;
    let out = tmp("scene-held-out");
    let dir = out.0.join("data");
    let prep = write(&s, p, &o, &dir).unwrap();
    assert_eq!(prep.heldout_fingerprint, all.heldout_fingerprint);
    let hd = prep.heldout_dir.clone().unwrap();
    assert_eq!(
        read_heldout_fingerprint(&hd).as_deref(),
        Some(all.heldout_fingerprint.as_str())
    );
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("dataset.json")).unwrap()).unwrap();
    assert_eq!(
        summary["heldout_fingerprint"],
        all.heldout_fingerprint.as_str()
    );
    assert_eq!(summary["options"]["scene"], "night+ir");
    assert_eq!(summary["options"]["excluded"], "background");
    let (set, _) = images::load_dir(&hd, None, Some("heldout")).unwrap();
    for i in &set.images {
        let scene = ["day", "night", "ir"]
            .into_iter()
            .find(|x| i.tags.iter().any(|t| t == x))
            .unwrap();
        assert!(i.tags.contains(&format!("heldout:{scene}")), "{:?}", i.tags);
        assert!(i.tags.contains(&HELDOUT_ALL.to_string()));
        assert_eq!(
            i.tags.contains(&HELDOUT_IN_SCOPE.to_string()),
            scene != "day",
            "{:?}",
            i.tags
        );
    }
}

#[test]
fn heldout_tags_name_the_scene_and_the_scope() {
    assert_eq!(
        heldout_tags(Some("ir"), SceneFilter::NIGHT_IR),
        ["ir", "heldout:ir", "heldout:all", "heldout:in-scope"]
    );
    assert_eq!(
        heldout_tags(Some("day"), SceneFilter::NIGHT_IR),
        ["day", "heldout:day", "heldout:all"]
    );
    assert_eq!(
        heldout_tags(None, SceneFilter::ALL_SCENES),
        ["heldout:all", "heldout:in-scope"]
    );
    assert_eq!(heldout_tags(None, SceneFilter::DAY), ["heldout:all"]);
}

/// A flat export with a `heldout.txt`: cats, cats with a dog, a dog alone, an empty frame; two
/// held-out frames with dogs. The run trains `cat` only.
fn dog_export(dir: &Path) -> Vec<Source> {
    let tags: &[&str] = &["day", "camera:a", "date:2026-01-01"];
    let cat = "0 0.5 0.5 0.2 0.2\n";
    let both = "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n";
    let dog = "1 0.2 0.2 0.1 0.1\n";
    flat_export(
        dir,
        &["cat", "dog"],
        &[
            ("c1.jpg", cat, tags),
            ("c2.jpg", cat, tags),
            ("c3.jpg", cat, tags),
            ("c4.jpg", cat, tags),
            ("d1.jpg", both, tags),
            ("d2.jpg", both, tags),
            ("o1.jpg", dog, tags),
            ("e1.jpg", "", tags),
            ("h1.jpg", dog, tags),
            ("h2.jpg", both, tags),
        ],
    );
    std::fs::write(
        dir.join("heldout.txt"),
        "h1.jpg
h2.jpg
",
    )
    .unwrap();
    vec![read_export(dir).unwrap()]
}

fn named(s: &[Source], refs: impl Iterator<Item = FrameRef>) -> BTreeSet<String> {
    refs.map(|r| file_name(&s[r.0].frames[r.1].image)).collect()
}

#[test]
fn drop_frame_leaves_out_train_frames_with_untrained_labels_only() {
    let t = tmp("drop");
    let s = dog_export(&t.0);
    let opts = Options {
        classes: Some(vec!["cat".into()]),
        excluded: Excluded::DropFrame,
        negative_share: 0.5,
        ..Options::default()
    };
    let p = plan(&s, &opts).unwrap();
    assert_eq!(p.stats.excluded_frames, 3, "d1, d2, o1");
    let used = named(
        &s,
        p.train
            .iter()
            .map(|(r, _)| *r)
            .chain(p.valid.iter().copied()),
    );
    for n in ["d1.jpg", "d2.jpg", "o1.jpg"] {
        assert!(!used.contains(n), "{n} in {used:?}");
    }
    assert!(used.contains("c1.jpg") && used.contains("e1.jpg"));
    // The held-out frames keep their dogs.
    assert_eq!(
        named(&s, p.heldout.iter().copied()),
        BTreeSet::from(["h1.jpg".to_string(), "h2.jpg".to_string()])
    );
    // 3 of 8 in-scope frames (38%) left out: a warning.
    assert!(
        p.warnings.iter().any(|w| w.contains("3 of 8")),
        "{:?}",
        p.warnings
    );
    // A label dropped by an empty rename is not an untrained label.
    let renamed = Options {
        rename: BTreeMap::from([("dog".to_string(), String::new())]),
        ..opts.clone()
    };
    assert_eq!(plan(&s, &renamed).unwrap().stats.excluded_frames, 0);
    // Background (the default) keeps them.
    let bg = plan(
        &s,
        &Options {
            excluded: Excluded::Background,
            ..opts
        },
    )
    .unwrap();
    assert_eq!(bg.stats.excluded_frames, 0);
    assert!(!bg.warnings.iter().any(|w| w.contains("do not train")));
}

/// In the background mode a frame whose only boxes are of untrained labels is a negative and
/// counts toward `negative_share`.
#[test]
fn background_mode_makes_frames_of_untrained_labels_only_negatives() {
    let t = tmp("bg");
    let s = dog_export(&t.0);
    let opts = Options {
        classes: Some(vec!["cat".into()]),
        negative_share: 0.5,
        ..Options::default()
    };
    let p = plan(&s, &opts).unwrap();
    let used: Vec<FrameRef> = p
        .train
        .iter()
        .map(|(r, _)| *r)
        .chain(p.valid.iter().copied())
        .collect();
    assert!(named(&s, used.iter().copied()).contains("o1.jpg"));
    let negatives: BTreeSet<String> = named(
        &s,
        used.iter().copied().filter(|r| {
            mapped_boxes(&s[0].frames[r.1], &opts, &p.classes)
                .next()
                .is_none()
        }),
    );
    assert_eq!(
        negatives,
        BTreeSet::from(["e1.jpg".to_string(), "o1.jpg".to_string()])
    );
    assert_eq!(
        p.stats.boxes_dropped.get("dog"),
        Some(&5),
        "d1, d2, o1, h1, h2"
    );
    // Toward the share: 6 positives at 0.1 -> 6 x 0.1 / 0.9 = 0.67 -> 1 negative kept; the
    // reviewer-unconfirmed o1 / e1 rank by seed.
    let few = plan(
        &s,
        &Options {
            negative_share: 0.1,
            ..opts
        },
    )
    .unwrap();
    assert_eq!(few.stats.negatives_dropped, 1);
}

#[test]
fn inventory_counts_boxes_per_class_and_scene() {
    let (a, b) = (tmp("inv-a"), tmp("inv-b"));
    flat_export(
        &a.0,
        &["cat", "dog", "tree"],
        &[
            (
                "1.jpg",
                "0 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1\n1 0.3 0.3 0.1 0.1\n",
                &["day"],
            ),
            ("2.jpg", "0 0.5 0.5 0.2 0.2\n", &["night"]),
            ("3.jpg", "2 0.5 0.5 0.2 0.2\n", &["ir"]),
        ],
    );
    flat_export(
        &b.0,
        &["motorbike", "cat", "person"],
        &[
            ("4.jpg", "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n", &[]),
            ("5.jpg", "1 0.5 0.5 0.2 0.2\n", &["ir"]),
        ],
    );
    let s = vec![read_export(&a.0).unwrap(), read_export(&b.0).unwrap()];
    let rename = BTreeMap::from([
        ("motorbike".to_string(), "motorcycle".to_string()),
        ("tree".to_string(), String::new()),
    ]);
    let inv = inventory(&s, &rename);
    let names: Vec<&str> = inv.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["cat", "dog", "motorcycle", "person"]);
    let cat = &inv[0];
    assert_eq!((cat.boxes, cat.frames), (5, 4));
    assert_eq!(
        cat.by_scene,
        SceneCounts {
            day: 2,
            night: 1,
            ir: 1,
            unknown: 1
        }
    );
    assert_eq!((inv[1].boxes, inv[1].by_scene.day), (1, 1));
    assert_eq!((inv[2].boxes, inv[2].by_scene.unknown), (1, 1));
    assert_eq!((inv[3].boxes, inv[3].frames), (0, 0), "listed, no box");
    assert!(inventory(&[], &BTreeMap::new()).is_empty());
}

/// The plan of a source does not depend on where its folder is (it once ranked frames by their
/// absolute path, so the temp folder's random name moved frames between train and valid).
#[test]
fn the_plan_does_not_depend_on_the_folder_location() {
    type Names = (Vec<(usize, String, u32)>, Vec<String>, Vec<String>);
    let plans: Vec<Names> = (0..6)
        .map(|i| {
            let t = tmp(&format!("where-{i}"));
            camera_export(&t.0);
            let s = vec![read_export(&t.0).unwrap()];
            let p = plan(&s, &Options::default()).unwrap();
            let name = |r: &FrameRef| file_name(&s[r.0].frames[r.1].image);
            (
                p.train.iter().map(|(r, n)| (r.0, name(r), *n)).collect(),
                p.valid.iter().map(name).collect(),
                p.heldout.iter().map(name).collect(),
            )
        })
        .collect();
    for p in &plans[1..] {
        assert_eq!(p, &plans[0]);
    }
}

/// Pixel boxes `(label, [x0, y0, x1, y1])` of a reviewed frame.
type PixelBoxes<'a> = &'a [(&'a str, [f32; 4])];
/// `(scene, camera, date, stem, boxes)` of a reviewed frame.
type ReviewedFrame<'a> = (&'a str, &'a str, &'a str, &'a str, PixelBoxes<'a>);

/// A collector dataset folder with real frames (64x48 JPEG, sidecar) reviewed with `boxes`
/// (`(label, [x0, y0, x1, y1])` in pixels), per frame `(scene, camera, date, stem, boxes)`.
fn reviewed_dataset(dir: &Path, frames: &[ReviewedFrame<'_>]) {
    use crate::collect::labels::{self as cl, BoxState, LabelBox, LabelFile, Status};
    for (scene, camera, date, stem, boxes) in frames {
        let day = dir.join(scene).join(camera).join(date);
        jpg(&day.join(format!("{stem}.jpg")), 64, 48);
        let mut r = crate::collect::record::empty_record();
        r.id = (*stem).into();
        r.scene = (*scene).into();
        r.camera = (*camera).into();
        r.timestamp = format!("{date}T01:02:03.004Z");
        r.width = 64;
        r.height = 48;
        std::fs::write(
            day.join(format!("{stem}.json")),
            serde_json::to_vec(&r).unwrap(),
        )
        .unwrap();
        let labels = LabelFile {
            status: Status::Reviewed,
            empty_confirmed: boxes.is_empty(),
            boxes: boxes
                .iter()
                .map(|(label, bbox)| LabelBox {
                    label: (*label).into(),
                    bbox: *bbox,
                    source: cl::Source::Human,
                    score: None,
                    state: BoxState::Added,
                    origin: None,
                    reason: None,
                    model_label: None,
                    orig_bbox: None,
                    auto: Default::default(),
                })
                .collect(),
            ..LabelFile::default()
        };
        cl::write(&day.join(format!("{stem}.labels.json")), &labels).unwrap();
    }
}

/// The review page's export of `dataset` into `out` (YOLO + benchmark, the page's split).
fn review_export(
    dataset: &Path,
    out: &Path,
    split: crate::collect::labels::export::SplitConfig,
) -> crate::collect::labels::export::ExportReport {
    use crate::collect::labels::export::{self, ExportRequest, Format};
    let classes: Vec<String> = ["person", "car", "dog", "cat", "raccoon"]
        .map(String::from)
        .to_vec();
    export::run(
        dataset,
        &classes,
        &ExportRequest {
            formats: vec![Format::Yolo, Format::Coco, Format::Benchmark],
            out_dir: out.to_path_buf(),
            split,
            benchmark_split: Default::default(),
            include_unreviewed: false,
            hidden_classes: Vec::new(),
        },
    )
    .unwrap()
}

/// A stand-in for the LILA night subset: `n` public frames, animals only (raccoon, dog, cat).
fn animal_public(dir: &Path, n: usize) -> Source {
    let labels = ["raccoon", "dog", "cat"];
    let images: Vec<serde_json::Value> = (0..n)
        .map(|i| {
            let file = format!("lila{i}.jpg");
            jpg(&dir.join(&file), 100, 50);
            serde_json::json!({"file": file, "width": 100, "height": 50, "tags": ["ir"],
                "objects": [{"label": labels[i % 3], "bbox": [10, 10, 30, 40]}]})
        })
        .collect();
    let m = Manifest::parse(
        &serde_json::json!({"id": "lila-night", "license": "CDLA-Permissive-1.0",
            "images": images})
        .to_string(),
    )
    .unwrap();
    read_public_set(dir, &m).unwrap()
}

/// Per frame `(camera, date, index, person?)` of a review dataset: every frame IR, a person box
/// or confirmed empty.
fn person_frames(groups: &[(&str, &str, usize)]) -> Vec<(String, String, String, bool)> {
    let mut v = Vec::new();
    for (cam, date, n) in groups {
        for i in 0..*n {
            let stem = format!("{}T01020{}004Z_{cam}{i}", date.replace('-', ""), i % 10);
            v.push(((*cam).to_string(), (*date).to_string(), stem, i % 3 != 2));
        }
    }
    v
}

fn put_person_frames(dir: &Path, frames: &[(String, String, String, bool)]) {
    let person: &[(&str, [f32; 4])] = &[("person", [4.0, 4.0, 20.0, 40.0])];
    let refs: Vec<ReviewedFrame<'_>> = frames
        .iter()
        .map(|(c, d, s, p)| {
            (
                "ir",
                c.as_str(),
                d.as_str(),
                s.as_str(),
                if *p { person } else { &[] },
            )
        })
        .collect();
    reviewed_dataset(dir, &refs);
}

/// The review agent's real export (`export::run`: `images/{train,val}/`, flat names) plus a
/// public subset ten times its size: the trainer's validation split once took every review
/// training frame (its size was a share of the whole pool, public frames included, but it only
/// took review frames), so `person`, a class only the review frames have, had no training box
/// and the review export showed `used_train: 0`.
#[test]
fn a_real_review_export_keeps_its_classes_in_training_beside_a_large_public_set() {
    use crate::collect::labels::export::SplitConfig;
    let (ds, out, pubd) = (tmp("rx-ds"), tmp("rx-out"), tmp("rx-pub"));
    let frames = person_frames(&[
        ("yard", "2026-10-01", 3),
        ("yard", "2026-10-02", 3),
        ("gate", "2026-10-01", 3),
        ("gate", "2026-10-02", 3),
        ("drive", "2026-10-01", 3),
        ("drive", "2026-10-02", 3),
    ]);
    put_person_frames(&ds.0, &frames);
    let x = out.0.join("export");
    let report = review_export(
        &ds.0,
        &x,
        SplitConfig {
            holdout_days: vec!["2026-10-02".into()],
            ..SplitConfig::default()
        },
    );
    assert_eq!((report.train, report.val), (9, 9), "{:?}", report.splits);
    assert!(x.join("images/train").is_dir() && x.join("images/val").is_dir());
    let review = read_export(&x).unwrap();
    assert_eq!(
        review
            .frames
            .iter()
            .filter(|f| f.split == SplitHint::Train)
            .count(),
        9
    );
    assert!(
        review
            .frames
            .iter()
            .all(|f| (f.split == SplitHint::Heldout) == (f.day == "2026-10-02")),
        "the export's val split is its held-out day"
    );
    let s = vec![review, animal_public(&pubd.0, 120)];
    let p = plan(&s, &Options::default()).unwrap();
    assert!(
        p.stats.train_boxes.get("person").copied().unwrap_or(0) > 0,
        "{:?} {:?}",
        p.stats,
        p.warnings
    );
    assert!(
        !p.warnings.iter().any(|w| w.contains("'person'")),
        "{:?}",
        p.warnings
    );
    // The export's val frames are the held-out set; its train frames train (and validate).
    assert_eq!(p.heldout.len(), 9);
    assert!(p.heldout.iter().all(|r| r.0 == 0));
    let review_train = p.train.iter().filter(|(r, _)| r.0 == 0).count();
    let review_valid = p.valid.iter().filter(|r| r.0 == 0).count();
    assert!(review_train >= 5, "{review_train} of 9 review frames train");
    assert!(review_valid >= 1);
    assert_eq!(
        review_train + review_valid,
        9,
        "no review frame is dropped (3 are hard negatives)"
    );
    let prep = write(&s, &p, &Options::default(), &out.0.join("data")).unwrap();
    let summary = read_summary_sources(&out.0.join("data"));
    assert_eq!(summary[0].used_train, review_train);
    assert_eq!(summary[0].used_heldout, 9);
    assert!(prep.warnings.iter().all(|w| !w.contains("'person'")));
}

/// When every frame of a class is in the export's val split (the held-out cameras and days),
/// no training box of it is right; the warning says so and how to change the hold-out.
#[test]
fn a_class_only_in_the_heldout_split_says_so() {
    use crate::collect::labels::export::SplitConfig;
    let (ds, out, pubd) = (tmp("rxh-ds"), tmp("rxh-out"), tmp("rxh-pub"));
    // Persons only on the held-out day; the training day has empty frames only.
    let mut frames = person_frames(&[("yard", "2026-10-02", 4), ("gate", "2026-10-02", 2)]);
    for i in 0..3 {
        frames.push((
            "yard".into(),
            "2026-10-01".into(),
            format!("20261001T01020{i}004Z_e{i}"),
            false,
        ));
    }
    put_person_frames(&ds.0, &frames);
    let x = out.0.join("export");
    let report = review_export(
        &ds.0,
        &x,
        SplitConfig {
            holdout_days: vec!["2026-10-02".into()],
            ..SplitConfig::default()
        },
    );
    assert_eq!((report.train, report.val), (3, 6));
    let s = vec![read_export(&x).unwrap(), animal_public(&pubd.0, 20)];
    let p = plan(&s, &Options::default()).unwrap();
    assert!(!p.stats.train_boxes.contains_key("person"));
    let w = p
        .warnings
        .iter()
        .find(|w| w.contains("'person'"))
        .unwrap_or_else(|| panic!("{:?}", p.warnings));
    assert!(w.contains("only in the held-out"), "{w}");
    assert!(w.contains("hold out"), "{w}");
    // `car` has no box anywhere: a different message.
    let car = p.warnings.iter().find(|w| w.contains("'car'")).unwrap();
    assert!(car.contains("no frame"), "{car}");
}

/// Skewed footage: 4 cameras, a busy day (8 person frames + 1 empty) and a short IR night
/// (2 frames, a raccoon on cameras a and b, + 1 empty) each.
fn skewed_export(dir: &Path) {
    let mut frames: Vec<(String, String, Vec<String>)> = Vec::new();
    for cam in ["a", "b", "c", "d"] {
        for (day, scene, n) in [("2026-10-01", "day", 8), ("2026-10-02", "ir", 2)] {
            for i in 0..=n {
                let label = match (scene, i) {
                    (_, i) if i == n => "",
                    ("ir", 0) if cam == "a" || cam == "b" => "1 0.5 0.5 0.2 0.2\n",
                    _ => "0 0.5 0.5 0.2 0.4\n",
                };
                frames.push((
                    format!("{cam}-{day}-{i}.jpg"),
                    label.into(),
                    vec![scene.into(), format!("camera:{cam}"), format!("date:{day}")],
                ));
            }
        }
    }
    let tags: Vec<Vec<&str>> = frames
        .iter()
        .map(|(_, _, t)| t.iter().map(String::as_str).collect())
        .collect();
    let refs: Vec<(&str, &str, &[&str])> = frames
        .iter()
        .zip(&tags)
        .map(|((n, l, _), t)| (n.as_str(), l.as_str(), t.as_slice()))
        .collect();
    flat_export(dir, &["person", "raccoon"], &refs);
}

fn balanced(axes: &str, cap: Option<f32>) -> Options {
    Options {
        balance: Balance {
            axes: Balance::parse_axes(axes).unwrap(),
            cap_dominant: cap,
            ..Balance::default()
        },
        ..Options::default()
    }
}

#[test]
fn balance_off_is_the_old_plan() {
    let t = tmp("bal-off");
    skewed_export(&t.0);
    let s = vec![read_export(&t.0).unwrap()];
    let old = plan(&s, &Options::default()).unwrap();
    // Every knob but the axes set: still off.
    let off = Options {
        balance: Balance {
            axes: BTreeSet::new(),
            strength: 1.0,
            max_repeat: 9,
            cap_dominant: Some(1.0),
        },
        ..Options::default()
    };
    assert_eq!(plan(&s, &off).unwrap(), old);
    assert_eq!(old.balance, None);
    assert_eq!(old.stats.balance_dropped, 0);
    assert!(old.train.iter().all(|(_, r)| *r == 1));
    // Options of older runs (no balance key) read as off and write no balance key.
    let mut v = serde_json::to_value(Options::default()).unwrap();
    assert!(v.get("balance").is_none(), "{v}");
    v.as_object_mut().unwrap().remove("balance");
    let back: Options = serde_json::from_value(v).unwrap();
    assert!(back.balance.is_off());
    assert_eq!(back, Options::default());
}

#[test]
fn balance_repeats_rare_training_frames_and_never_touches_heldout_or_valid() {
    let t = tmp("bal-on");
    skewed_export(&t.0);
    let s = vec![read_export(&t.0).unwrap()];
    let old = plan(&s, &Options::default()).unwrap();
    let opts = balanced("class,scene", None);
    let p = plan(&s, &opts).unwrap();
    assert_eq!(p.heldout, old.heldout);
    assert_eq!(p.heldout_fingerprint, old.heldout_fingerprint);
    assert_eq!(p.valid, old.valid);
    assert_eq!(
        p.train.iter().map(|(r, _)| *r).collect::<Vec<_>>(),
        old.train.iter().map(|(r, _)| *r).collect::<Vec<_>>(),
        "no cap: the same training frames"
    );
    let f = |r: FrameRef| &s[r.0].frames[r.1];
    let mut raccoon_reps = 0;
    let (mut ir, mut ir_reps) = (0, 0);
    for (r, rep) in &p.train {
        let fr = f(*r);
        let raccoon = fr.boxes.iter().any(|b| b.class == "raccoon");
        if raccoon {
            raccoon_reps += rep;
            assert!(*rep > 1, "raccoon frame {}", fr.image.display());
        } else if fr.scene.as_deref() == Some("ir") && !fr.boxes.is_empty() {
            // sqrt(day / ir) between 1 and 2: rounded up or down by rank.
            assert!((1..=2).contains(rep), "IR frame {}", fr.image.display());
            (ir, ir_reps) = (ir + 1, ir_reps + rep);
        } else if fr.scene.as_deref() == Some("day") && !fr.boxes.is_empty() {
            assert_eq!(*rep, 1, "day person frame {}", fr.image.display());
        }
    }
    assert!(
        ir > 0 && ir_reps > ir,
        "{ir_reps} repeats of {ir} IR frames"
    );
    let report = p.balance.as_ref().unwrap();
    assert_eq!(
        report.after.boxes_per_class["raccoon"],
        raccoon_reps as usize
    );
    assert!(report.after.images > report.before.images);
    assert_eq!(
        report.before.images,
        old.train.iter().map(|(_, r)| *r as usize).sum::<usize>()
    );
    assert_eq!(
        report.after.images,
        p.train.iter().map(|(_, r)| *r as usize).sum::<usize>()
    );
    assert_eq!(
        p.stats.train_boxes, old.stats.train_boxes,
        "repeats count once"
    );
    // The same plan again (seeded), another seed may round otherwise but keeps the splits.
    assert_eq!(plan(&s, &opts).unwrap(), p);
    // Written: repeated lines and the report in dataset.json.
    let out = tmp("bal-out");
    let prep = write(&s, &p, &opts, &out.0.join("data")).unwrap();
    assert_eq!(prep.train_images, report.after.images);
    let summary = prep.balance_summary.as_deref().unwrap();
    assert!(
        summary.starts_with(&format!(
            "{} → {} images/epoch",
            report.before.images, report.after.images
        )),
        "{summary}"
    );
    let lines = std::fs::read_to_string(out.0.join("data").join("train.txt")).unwrap();
    assert_eq!(lines.lines().count(), report.after.images);
    let j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.0.join("data").join("dataset.json")).unwrap())
            .unwrap();
    assert_eq!(
        j["options"]["balance"]["axes"],
        serde_json::json!(["class", "scene"])
    );
    assert_eq!(
        j["balance"]["after"]["images"].as_u64(),
        Some(report.after.images as u64)
    );
    // Unbalanced: no report, no summary.
    let out2 = tmp("bal-out-off");
    let prep = write(&s, &old, &Options::default(), &out2.0.join("data")).unwrap();
    assert_eq!(prep.balance_summary, None);
    let text = std::fs::read_to_string(out2.0.join("data").join("dataset.json")).unwrap();
    assert!(!text.contains("\"balance"), "{text}");
}

#[test]
fn the_balance_cap_drops_only_training_frames() {
    let t = tmp("bal-cap");
    skewed_export(&t.0);
    let s = vec![read_export(&t.0).unwrap()];
    let old = plan(&s, &Options::default()).unwrap();
    let p = plan(&s, &balanced("scene,camera-day", Some(1.0))).unwrap();
    assert_eq!(p.heldout_fingerprint, old.heldout_fingerprint);
    assert_eq!(p.valid, old.valid);
    assert!(p.stats.balance_dropped > 0);
    assert_eq!(
        p.train.len() + p.stats.balance_dropped,
        old.train.len(),
        "dropped frames leave training only"
    );
    assert!(p.stats.train_boxes.contains_key("raccoon"));
    assert_eq!(p.balance.as_ref().unwrap().dropped, p.stats.balance_dropped);
    let train: BTreeSet<FrameRef> = p.train.iter().map(|(r, _)| *r).collect();
    assert!(
        old.train
            .iter()
            .all(|(r, _)| train.contains(r) || s[r.0].frames[r.1].scene.as_deref() == Some("day"))
    );
}
