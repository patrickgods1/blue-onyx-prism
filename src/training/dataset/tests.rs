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
        min_valid: 0,
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
        min_valid: 0,
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
    // Every image the lists name, with the source it came from.
    let sources = read_image_sources(&dir).unwrap();
    let listed: BTreeSet<String> = ["train.txt", "valid.txt"]
        .iter()
        .flat_map(|l| {
            std::fs::read_to_string(dir.join(l))
                .unwrap()
                .lines()
                .map(|x| x.trim_start_matches("./images/").to_string())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(sources.keys().cloned().collect::<BTreeSet<_>>(), listed);
    assert_eq!(sources["s0-a1.jpg"].file_name().unwrap(), "a1.jpg");
    assert_eq!(sources["s1-b1.jpg"].file_name().unwrap(), "b1.jpg");
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
    assert_eq!(set.ground_truth, images::GroundTruth::Manifest);
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
                    difficult: false,
                    reject_reason: None,
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
            size_floor: None,
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

/// Validation frames of a plan over `n` frames with one person box each, default split options.
fn valid_count(n: usize) -> usize {
    let t = tmp(&format!("minval-{n}"));
    let tags: &[&str] = &["day", "camera:a", "date:2026-01-01"];
    let frames: Vec<(String, &str, &[&str])> = (0..n)
        .map(|i| (format!("p{i}.jpg"), "0 0.5 0.5 0.2 0.2\n", tags))
        .collect();
    let refs: Vec<(&str, &str, &[&str])> = frames
        .iter()
        .map(|(name, l, tg)| (name.as_str(), *l, *tg))
        .collect();
    flat_export(&t.0, &["person"], &refs);
    let s = vec![read_export(&t.0).unwrap()];
    let opts = Options {
        heldout_share: 0.0,
        negative_share: 0.0,
        ..Options::default()
    };
    plan(&s, &opts).unwrap().valid.len()
}

#[test]
fn validation_split_has_a_minimum_capped_at_a_quarter() {
    assert_eq!(valid_count(241), 50);
    assert_eq!(valid_count(100), 25);
    assert_eq!(valid_count(1000), 100);
    assert_eq!(valid_count(3), 1);
    assert_eq!(valid_count(2), 1);
}

/// A plan of `unique` frames (all one small image) with `lines` train.txt lines in all.
fn repeated_plan(s: &Source, unique: usize, lines: usize) -> Plan {
    let extra = lines - unique;
    Plan {
        classes: vec!["person".into()],
        train: (0..unique)
            .map(|i| {
                // The extra lines spread over the first frames, 2 each (one more when odd).
                let more = (extra.saturating_sub(2 * i)).min(2) as u32;
                ((0, i), 1 + more)
            })
            .collect(),
        valid: vec![(0, s.frames.len() - 1)],
        ..Plan::default()
    }
}

fn one_image_source(dir: &Path, n: usize) -> Source {
    jpg(&dir.join("x.jpg"), 8, 8);
    Source {
        dir: dir.to_path_buf(),
        public: None,
        classes: vec!["person".into()],
        frames: vec![
            Frame {
                image: dir.join("x.jpg"),
                boxes: vec![YoloBox {
                    class: "person".into(),
                    cx: 0.5,
                    cy: 0.5,
                    w: 0.2,
                    h: 0.2,
                }],
                ignore: Vec::new(),
                floor_regions: Vec::new(),
                parked: Vec::new(),
                scene: Some("day".into()),
                camera: "a".into(),
                day: "2026-10-01".into(),
                split: SplitHint::None,
                hard_negative: false,
                human_rejected: false,
                bulk_accepted: false,
                size: Some((8, 8)),
            };
            n
        ],
        warnings: Vec::new(),
        notes_known: false,
        size_floor: None,
    }
}

/// The run of 20261103-204935-f7d1: balancing took 423 frames to 1,019 lines, so RF-DETR left the
/// small-dataset rate it needed.
#[test]
fn the_learning_rate_counts_unique_frames_not_repeats() {
    use crate::resources::catalog::TrainFamily;
    let t = tmp("lr-unique");
    let s = vec![one_image_source(&t.0, 424)];
    let p = repeated_plan(&s[0], 423, 1019);
    let prep = write(&s, &p, &Options::default(), &t.0.join("data")).unwrap();
    assert_eq!((prep.train_images, prep.train_frames), (1019, 423));
    let text = std::fs::read(t.0.join("data").join("dataset.json")).unwrap();
    let j: serde_json::Value = serde_json::from_slice(&text).unwrap();
    assert_eq!(
        (j["train_images"].as_u64(), j["train_frames"].as_u64()),
        (Some(1019), Some(423))
    );
    let r = crate::training::recipe::TrainRecipe::new(TrainFamily::Rfdetr, "nano").unwrap();
    assert_eq!(r.effective_lr(prep.train_images), None, "the old count");
    let lr = r.effective_lr(prep.train_frames).unwrap();
    assert_eq!((lr.lr, lr.small_dataset), (5e-5, true));
    assert!(
        r.lr_line(prep.train_frames)
            .contains("(423 unique training frames < 1000: lower than"),
        "{}",
        r.lr_line(prep.train_frames)
    );
    // A large unique set keeps the default, repeats or not.
    let t = tmp("lr-large");
    let s = vec![one_image_source(&t.0, 1001)];
    let p = repeated_plan(&s[0], 1000, 1000);
    let prep = write(&s, &p, &Options::default(), &t.0.join("data")).unwrap();
    assert_eq!((prep.train_images, prep.train_frames), (1000, 1000));
    assert_eq!(r.effective_lr(prep.train_frames), None);
}

#[test]
fn public_frames_count_once_and_review_repeats_do_not() {
    let t = tmp("lr-public");
    // Eight cameras of 5 own frames, 30 public frames.
    let source = |dir: PathBuf, n: usize, camera: &dyn Fn(usize) -> String| {
        let mut s = one_image_source(&dir, n);
        for (i, f) in s.frames.iter_mut().enumerate() {
            f.image = dir.join(format!("{i}.jpg"));
            f.camera = camera(i);
            jpg(&f.image, 8, 8);
        }
        s
    };
    let own = source(t.0.join("own"), 40, &|i| format!("cam{}", i % 8));
    let mut public = source(t.0.join("pub"), 30, &|_| "pub".to_string());
    public.public = Some(Public {
        id: "pub".into(),
        license: "cc".into(),
    });
    let s = vec![own, public];
    let opts = Options {
        heldout_share: 0.0,
        negative_share: 0.0,
        oversample: 3,
        ..Options::default()
    };
    let p = plan(&s, &opts).unwrap();
    let review = p.train.iter().filter(|(r, _)| r.0 == 0).count();
    assert!(
        p.train
            .iter()
            .all(|((si, _), n)| *n == if *si == 0 { 3 } else { 1 })
    );
    let prep = write(&s, &p, &opts, &t.0.join("data")).unwrap();
    assert_eq!(prep.train_frames, review + 30);
    assert_eq!(prep.train_images, review * 3 + 30);
}

/// `ignore.json` beside an export: (image, label, cx, cy, w, h).
fn ignore_json(dir: &Path, regions: &[(&str, &str, f64, f64, f64, f64)]) {
    let mut images: BTreeMap<String, Vec<serde_json::Value>> = BTreeMap::new();
    for (name, label, cx, cy, w, h) in regions {
        images
            .entry((*name).to_string())
            .or_default()
            .push(serde_json::json!({"label": label, "cx": cx, "cy": cy, "w": w, "h": h}));
    }
    std::fs::write(
        dir.join("ignore.json"),
        serde_json::to_vec(&serde_json::json!({"version": 1, "images": images})).unwrap(),
    )
    .unwrap();
}

fn name_of(s: &Source, r: FrameRef) -> String {
    file_name(&s.frames[r.1].image)
}

/// The options of earlier versions for frames with ignore regions: each leaves train and valid.
fn drop_frame() -> Options {
    Options {
        ignore: IgnoreMode::DropFrame,
        ..Options::default()
    }
}

fn used(p: &Plan, r: FrameRef) -> bool {
    p.train.iter().any(|(t, _)| *t == r) || p.valid.contains(&r)
}

#[test]
fn an_old_export_without_ignore_json_reads_as_none() {
    let t = tmp("no-ignore");
    camera_export(&t.0);
    let s = read_export(&t.0).unwrap();
    assert!(s.frames.iter().all(|f| f.ignore.is_empty()));
    let p = plan(std::slice::from_ref(&s), &Options::default()).unwrap();
    assert_eq!(p.stats.ignore_frames, 0);
    assert!(p.warnings.iter().all(|w| !w.contains("too-small")));
    // Not written for a run that leaves nothing out: dataset.json stays as it was.
    assert!(
        serde_json::to_value(&p.stats)
            .unwrap()
            .get("ignore_frames")
            .is_none()
    );
}

#[test]
fn ignore_json_is_read_by_image_name_and_a_damaged_one_stops_the_read() {
    let t = tmp("ignore-read");
    camera_export(&t.0);
    ignore_json(
        &t.0,
        &[
            ("yard-2026-10-01-0.jpg", "cat", 0.5, 0.25, 0.1, 0.2),
            ("nothing-by-this-name.jpg", "cat", 0.5, 0.5, 0.1, 0.1),
        ],
    );
    let s = read_export(&t.0).unwrap();
    let with: Vec<&Frame> = s.frames.iter().filter(|f| !f.ignore.is_empty()).collect();
    assert_eq!(with.len(), 1);
    assert!(with[0].image.ends_with("yard-2026-10-01-0.jpg"));
    assert_eq!(
        with[0].ignore,
        [YoloBox {
            class: "cat".into(),
            cx: 0.5,
            cy: 0.25,
            w: 0.1,
            h: 0.2,
        }]
    );
    assert_eq!(with[0].boxes.len(), 1, "the frame's boxes are its own");
    std::fs::write(t.0.join("ignore.json"), "{").unwrap();
    let e = format!("{:#}", read_export(&t.0).unwrap_err());
    assert!(
        e.contains("damaged") && e.contains("export the frames again"),
        "{e}"
    );
    std::fs::write(t.0.join("ignore.json"), r#"{"version": 99, "images": {}}"#).unwrap();
    let e = format!("{:#}", read_export(&t.0).unwrap_err());
    assert!(e.contains("newer than this build"), "{e}");
}

/// The export of [`camera_export`] with ignore regions on one frame of the train / valid pool and
/// one held-out frame; returns them with the plan before the regions.
fn with_regions(dir: &Path, opts: &Options) -> (Plan, FrameRef, FrameRef) {
    camera_export(dir);
    let s = read_export(dir).unwrap();
    let before = plan(std::slice::from_ref(&s), opts).unwrap();
    let pooled = *before
        .train
        .iter()
        .map(|(r, _)| r)
        .chain(&before.valid)
        .next()
        .unwrap();
    let held = before.heldout[0];
    ignore_json(
        dir,
        &[
            (&name_of(&s, pooled), "dog", 0.2, 0.2, 0.1, 0.1),
            (&name_of(&s, held), "dog", 0.25, 0.25, 0.5, 0.5),
        ],
    );
    (before, pooled, held)
}

#[test]
fn a_frame_with_an_ignore_region_leaves_train_and_valid_and_is_counted() {
    let t = tmp("ignore-plan");
    let opts = drop_frame();
    let (before, pooled, held) = with_regions(&t.0, &opts);
    let s = read_export(&t.0).unwrap();
    let p = plan(std::slice::from_ref(&s), &opts).unwrap();
    assert!(used(&before, pooled) && !used(&p, pooled));
    assert_eq!(
        p.stats.ignore_frames, 1,
        "the held-out frame is not counted"
    );
    assert!(
        p.warnings
            .iter()
            .any(|w| w.starts_with("1 of ") && w.contains("too-small")),
        "{:?}",
        p.warnings
    );
    assert_eq!(p.heldout, before.heldout, "the held-out set does not move");
    assert!(p.heldout.contains(&held));
    assert!(!used(&p, held));
    assert_eq!(
        serde_json::to_value(&p.stats).unwrap()["ignore_frames"],
        serde_json::json!(1)
    );
}

#[test]
fn a_frame_left_out_by_the_scene_filter_is_not_counted_as_ignored() {
    let t = tmp("ignore-scene");
    let opts = drop_frame();
    let (_, pooled, _) = with_regions(&t.0, &opts);
    let s = read_export(&t.0).unwrap();
    let scene = s.frames[pooled.1].scene.clone().unwrap();
    let other = if scene == "day" { "ir" } else { "day" };
    let filtered = Options {
        scene: SceneFilter::parse(other).unwrap(),
        ..drop_frame()
    };
    let p = plan(std::slice::from_ref(&s), &filtered).unwrap();
    assert_eq!(p.stats.ignore_frames, 0);
}

#[test]
fn every_frame_of_a_class_with_a_region_explains_the_refusal() {
    let t = tmp("ignore-all");
    camera_export(&t.0);
    let s = read_export(&t.0).unwrap();
    let names: Vec<String> = s.frames.iter().map(|f| file_name(&f.image)).collect();
    let regions: Vec<(&str, &str, f64, f64, f64, f64)> = names
        .iter()
        .map(|n| (n.as_str(), "cat", 0.5, 0.5, 0.1, 0.1))
        .collect();
    ignore_json(&t.0, &regions);
    let s = read_export(&t.0).unwrap();
    let e = plan(&[s], &drop_frame()).unwrap_err().to_string();
    assert!(e.contains("unsure or too-small"), "{e}");
}

#[test]
fn held_out_frames_keep_their_regions_and_ordinary_frames_score_as_before() {
    let t = tmp("ignore-held");
    let opts = Options {
        rename: BTreeMap::from([("dog".to_string(), "hound".to_string())]),
        ..Options::default()
    };
    let (_, _, held) = with_regions(&t.0, &opts);
    let s = vec![read_export(&t.0).unwrap()];
    let p = plan(&s, &opts).unwrap();
    assert!(p.heldout.contains(&held));
    let dir = t.0.join("prepared");
    let prep = write(&s, &p, &opts, &dir).unwrap();
    let hd = prep.heldout_dir.unwrap();
    let (set, warnings) = images::load_dir(&hd, None, Some("heldout")).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(set.ground_truth, images::GroundTruth::Manifest);
    assert_eq!(set.images.len(), p.heldout.len());

    // The set as a dataset of older versions: `objects: None`, ground truth from the YOLO labels.
    let text = std::fs::read_to_string(hd.join("manifest.json")).unwrap();
    let mut old: Manifest = serde_json::from_str(&text).unwrap();
    for i in &mut old.images {
        i.objects = None;
    }
    let old_dir = t.0.join("old-heldout");
    std::fs::create_dir_all(old_dir.join("images")).unwrap();
    std::fs::create_dir_all(old_dir.join("labels")).unwrap();
    for e in std::fs::read_dir(hd.join("images")).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), old_dir.join("images").join(e.file_name())).unwrap();
    }
    for e in std::fs::read_dir(hd.join("labels")).unwrap() {
        let e = e.unwrap();
        std::fs::copy(e.path(), old_dir.join("labels").join(e.file_name())).unwrap();
    }
    std::fs::copy(hd.join("classes.txt"), old_dir.join("classes.txt")).unwrap();
    std::fs::write(
        old_dir.join("manifest.json"),
        serde_json::to_vec(&old).unwrap(),
    )
    .unwrap();
    let (old_set, _) = images::load_dir(&old_dir, None, Some("old")).unwrap();
    assert_eq!(old_set.ground_truth, images::GroundTruth::Yolo);

    let held_name = format!("s0-{}", name_of(&s[0], held));
    let mut empties = 0;
    for (new, old) in set.images.iter().zip(&old_set.images) {
        assert_eq!(new.file, old.file);
        assert_eq!((new.width, new.height), (old.width, old.height));
        assert_eq!(new.tags, old.tags, "{}", new.file);
        let (gt, was) = (new.gt.as_ref().unwrap(), old.gt.as_ref().unwrap());
        let ordinary: Vec<_> = gt.iter().filter(|g| !g.ignore).cloned().collect();
        assert_eq!(&ordinary, was, "{}: bit for bit", new.file);
        empties += usize::from(gt.is_empty());
        if new.file == held_name {
            let region: Vec<_> = gt.iter().filter(|g| g.ignore).collect();
            assert_eq!(region.len(), 1, "{gt:?}");
            assert_eq!(region[0].label, "hound", "renamed like the boxes");
            assert_eq!(region[0].bbox, [0.0, 0.0, 32.0, 24.0]);
            assert!(!region[0].crowd);
        } else {
            assert_eq!(gt.len(), was.len(), "{}: no region here", new.file);
        }
    }
    assert!(
        empties > 0,
        "an empty frame stays scored as empty: an empty list, not None"
    );
    assert_eq!(set.ground_truth.is_real(), old_set.ground_truth.is_real());
    let labels = std::fs::read_to_string(
        hd.join("labels")
            .join(format!("{}.txt", stem(Path::new(&held_name)))),
    )
    .unwrap();
    assert!(!labels.contains("0.250000 0.250000 0.500000"), "{labels}");
}

#[test]
fn a_region_of_a_label_the_run_does_not_score_is_left_out_of_the_held_out_set() {
    let t = tmp("ignore-unscored");
    let opts = Options::default();
    camera_export(&t.0);
    let s = read_export(&t.0).unwrap();
    let held = plan(std::slice::from_ref(&s), &opts).unwrap().heldout[0];
    ignore_json(&t.0, &[(&name_of(&s, held), "raccoon", 0.5, 0.5, 0.1, 0.1)]);
    let s = vec![read_export(&t.0).unwrap()];
    let p = plan(&s, &opts).unwrap();
    let dir = t.0.join("prepared");
    let hd = write(&s, &p, &opts, &dir).unwrap().heldout_dir.unwrap();
    let (set, _) = images::load_dir(&hd, None, None).unwrap();
    let want = format!("s0-{}", name_of(&s[0], held));
    let img = set.images.iter().find(|i| i.file == want).unwrap();
    assert!(img.gt.as_ref().unwrap().iter().all(|g| !g.ignore));
}

#[test]
fn a_held_out_image_of_no_readable_size_is_left_out_with_a_warning() {
    let t = tmp("ignore-size");
    let opts = Options::default();
    camera_export(&t.0);
    let mut s = read_export(&t.0).unwrap();
    let p = plan(std::slice::from_ref(&s), &opts).unwrap();
    let held = p.heldout[0];
    s.frames[held.1].size = None;
    std::fs::write(&s.frames[held.1].image, b"not a jpeg").unwrap();
    let dir = t.0.join("prepared");
    let prep = write(std::slice::from_ref(&s), &p, &opts, &dir).unwrap();
    assert!(
        prep.warnings
            .iter()
            .any(|w| w.contains("could not be read")),
        "{:?}",
        prep.warnings
    );
    let hd = prep.heldout_dir.unwrap();
    let (set, warnings) = images::load_dir(&hd, None, None).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(set.images.len(), p.heldout.len() - 1);
    assert_eq!(set.ground_truth, images::GroundTruth::Manifest);
}

#[test]
fn the_fingerprint_follows_the_ignore_regions() {
    let t = tmp("ignore-print");
    let opts = Options::default();
    let (before, _, held) = with_regions(&t.0, &opts);
    let s = read_export(&t.0).unwrap();
    let with = plan(std::slice::from_ref(&s), &opts).unwrap();
    assert_ne!(with.heldout_fingerprint, before.heldout_fingerprint);
    // Without regions: the plain hash of the frame keys, as older versions recorded it.
    let keys = before.heldout.iter().map(|r| frame_key(&s, &s.frames[r.1]));
    assert_eq!(before.heldout_fingerprint, fingerprint(keys));
    // Another region on the frame, or the same one elsewhere on it, is another scoring.
    let name = name_of(&s, held);
    ignore_json(&t.0, &[(&name, "dog", 0.3, 0.25, 0.5, 0.5)]);
    let s2 = read_export(&t.0).unwrap();
    let moved = plan(std::slice::from_ref(&s2), &opts).unwrap();
    assert_ne!(moved.heldout_fingerprint, with.heldout_fingerprint);
    // Regions on train frames do not change how the held-out set is scored.
    ignore_json(&t.0, &[]);
    let pooled = before.train[0].0;
    ignore_json(&t.0, &[(&name_of(&s, pooled), "dog", 0.3, 0.25, 0.5, 0.5)]);
    let s3 = read_export(&t.0).unwrap();
    let elsewhere = plan(std::slice::from_ref(&s3), &opts).unwrap();
    assert_eq!(elsewhere.heldout_fingerprint, before.heldout_fingerprint);
    // Nor does the order of a frame's regions, the class list or the scene filter.
    ignore_json(
        &t.0,
        &[
            (&name, "dog", 0.1, 0.1, 0.1, 0.1),
            (&name, "cat", 0.2, 0.2, 0.1, 0.1),
        ],
    );
    let a = plan(&[read_export(&t.0).unwrap()], &opts).unwrap();
    ignore_json(
        &t.0,
        &[
            (&name, "cat", 0.2, 0.2, 0.1, 0.1),
            (&name, "dog", 0.1, 0.1, 0.1, 0.1),
        ],
    );
    let other = Options {
        scene: SceneFilter::parse("day").unwrap(),
        classes: Some(vec!["cat".into()]),
        ..Options::default()
    };
    let b = plan(&[read_export(&t.0).unwrap()], &other).unwrap();
    assert_eq!(a.heldout_fingerprint, b.heldout_fingerprint);
}

#[test]
fn a_confirmed_empty_frame_stays_a_hard_negative() {
    let t = tmp("ignore-empty");
    camera_export(&t.0);
    let s = read_export(&t.0).unwrap();
    let mut m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(t.0.join("manifest.json")).unwrap()).unwrap();
    let opts = Options {
        negative_share: 0.5,
        ..drop_frame()
    };
    let p = plan(std::slice::from_ref(&s), &opts).unwrap();
    let empty = *p
        .train
        .iter()
        .map(|(r, _)| r)
        .find(|r| s.frames[r.1].boxes.is_empty())
        .expect("a negative in training");
    let name = name_of(&s, empty);
    for i in m["images"].as_array_mut().unwrap() {
        if i["file"] == name.as_str() {
            i["tags"]
                .as_array_mut()
                .unwrap()
                .push("hard-negative".into());
        }
    }
    std::fs::write(t.0.join("manifest.json"), serde_json::to_vec(&m).unwrap()).unwrap();
    // A region on some other frame changes nothing for it.
    let other = p
        .train
        .iter()
        .map(|(r, _)| *r)
        .find(|r| *r != empty)
        .unwrap();
    ignore_json(&t.0, &[(&name_of(&s, other), "cat", 0.5, 0.5, 0.1, 0.1)]);
    let s = read_export(&t.0).unwrap();
    assert!(s.frames[empty.1].hard_negative && s.frames[empty.1].ignore.is_empty());
    let p = plan(std::slice::from_ref(&s), &opts).unwrap();
    assert!(
        used(&p, empty),
        "still trained on as a frame without objects"
    );
    assert_eq!(p.stats.ignore_frames, 1);
}

fn region(class: &str) -> YoloBox {
    YoloBox {
        class: class.into(),
        cx: 0.9,
        cy: 0.9,
        w: 0.02,
        h: 0.03,
    }
}

/// `s` with an ignore region of `class` on frame `r`.
fn with_region(s: &Source, r: FrameRef, class: &str) -> Source {
    let mut s = s.clone();
    s.frames[r.1].ignore.push(region(class));
    s
}

#[test]
fn without_ignore_regions_both_modes_plan_the_same() {
    let t = tmp("ignore-modes-same");
    camera_export(&t.0);
    let s = [read_export(&t.0).unwrap()];
    let a = plan(&s, &Options::default()).unwrap();
    let b = plan(&s, &drop_frame()).unwrap();
    assert_eq!(a, b);
    let stats = serde_json::to_value(&a.stats).unwrap();
    for k in [
        "ignore_frames",
        "ignore_only",
        "ignore_valid_day",
        "ignore_kept",
        "ignore_boxes_background",
    ] {
        assert!(stats.get(k).is_none(), "{k} is not written when 0: {stats}");
    }
}

#[test]
fn a_frame_with_a_label_and_a_region_trains_with_the_region_as_background() {
    let t = tmp("ignore-background");
    camera_export(&t.0);
    let s = read_export(&t.0).unwrap();
    let opts = Options::default();
    let before = plan(std::slice::from_ref(&s), &opts).unwrap();
    let pool: Vec<FrameRef> = before
        .train
        .iter()
        .map(|(r, _)| *r)
        .chain(before.valid.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let (mut kept, mut valid_days, mut only) = (0, 0, 0);
    for r in pool {
        let one = [with_region(&s, r, "dog")];
        let p = plan(&one, &opts).unwrap();
        let st = &p.stats;
        assert!(
            !p.valid.contains(&r),
            "{r:?}: validation cannot ignore a region"
        );
        assert_eq!(p.heldout, before.heldout, "the held-out set does not move");
        assert_eq!(p.heldout_fingerprint, before.heldout_fingerprint);
        assert_eq!(st.ignore_frames, st.ignore_only + st.ignore_valid_day);
        if s.frames[r.1].boxes.is_empty() {
            // Nothing else on it: neither a positive nor a negative.
            assert!(!used(&p, r), "{r:?}");
            assert_eq!((st.ignore_only, st.ignore_kept), (1, 0));
            only += 1;
        } else if p.train.iter().any(|(x, _)| *x == r) {
            assert_eq!(
                (st.ignore_kept, st.ignore_only, st.ignore_valid_day),
                (1, 0, 0)
            );
            assert_eq!(
                st.ignore_boxes_background,
                BTreeMap::from([("dog".to_string(), 1)])
            );
            assert!(
                p.warnings
                    .iter()
                    .any(|w| w.starts_with("1 training frame(s) show an unsure or too-small")),
                "{:?}",
                p.warnings
            );
            kept += 1;
        } else {
            // Its camera day validates: out of both sides, counted.
            assert!(!used(&p, r));
            assert!(st.valid_groups.contains(&group(&one, r)), "{r:?}");
            assert_eq!((st.ignore_valid_day, st.ignore_kept), (1, 0));
            assert!(p.warnings.iter().any(|w| w.contains("used for validation")));
            valid_days += 1;
        }
    }
    assert!(
        kept > 0 && valid_days > 0 && only > 0,
        "every path is taken: kept {kept}, validation days {valid_days}, region only {only}"
    );
}

#[test]
fn a_frame_with_only_a_region_is_no_negative() {
    let t = tmp("ignore-no-negative");
    camera_export(&t.0);
    let s = read_export(&t.0).unwrap();
    let opts = Options {
        negative_share: 0.9,
        ..Options::default()
    };
    let before = plan(std::slice::from_ref(&s), &opts).unwrap();
    // A frame without boxes that trained as a negative, and one a reviewer confirmed empty.
    let empty = *before
        .train
        .iter()
        .map(|(r, _)| r)
        .find(|r| s.frames[r.1].boxes.is_empty())
        .expect("a negative in training");
    let mut with = with_region(&s, empty, "cat");
    with.frames[empty.1].hard_negative = true;
    let p = plan(&[with], &opts).unwrap();
    assert!(
        !used(&p, empty),
        "an unlabelled object is no 'nothing here', confirmed empty or not"
    );
    assert_eq!((p.stats.ignore_only, p.stats.ignore_kept), (1, 0));
}

#[test]
fn the_held_out_set_and_its_fingerprint_do_not_depend_on_the_ignore_mode() {
    let t = tmp("ignore-mode-print");
    let (_, pooled, held) = with_regions(&t.0, &Options::default());
    let s = [read_export(&t.0).unwrap()];
    assert!(!s[0].frames[pooled.1].ignore.is_empty() && !s[0].frames[held.1].ignore.is_empty());
    let a = plan(&s, &Options::default()).unwrap();
    let b = plan(&s, &drop_frame()).unwrap();
    assert_eq!(a.heldout, b.heldout);
    assert_eq!(a.heldout_fingerprint, b.heldout_fingerprint);
    assert!(a.heldout.contains(&held) && !used(&a, held) && !used(&b, held));
}

#[test]
fn a_trained_frame_writes_its_labels_without_the_region_and_never_validates() {
    let t = tmp("ignore-write");
    camera_export(&t.0);
    let s = read_export(&t.0).unwrap();
    let opts = Options::default();
    let before = plan(std::slice::from_ref(&s), &opts).unwrap();
    let r = before
        .train
        .iter()
        .map(|(r, _)| *r)
        .find(|r| s.frames[r.1].boxes.len() == 2)
        .unwrap();
    let s = [with_region(&s, r, "cat")];
    let p = plan(&s, &opts).unwrap();
    assert!(p.train.iter().any(|(x, _)| *x == r), "{:?}", p.stats);
    let dir = t.0.join("prepared");
    write(&s, &p, &opts, &dir).unwrap();
    let name = format!("s0-{}", name_of(&s[0], r));
    let label = std::fs::read_to_string(
        dir.join("labels")
            .join(format!("{}.txt", stem(Path::new(&name)))),
    )
    .unwrap();
    assert_eq!(
        label.lines().count(),
        2,
        "its two boxes, no region: {label}"
    );
    assert!(!label.contains("0.900000"), "{label}");
    let train = std::fs::read_to_string(dir.join("train.txt")).unwrap();
    let valid = std::fs::read_to_string(dir.join("valid.txt")).unwrap();
    assert!(train.contains(&name) && !valid.contains(&name));
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("dataset.json")).unwrap()).unwrap();
    assert_eq!(summary["options"]["ignore"], "background");
    assert_eq!(summary["stats"]["ignore_kept"], 1);
    assert_eq!(summary["stats"]["ignore_boxes_background"]["cat"], 1);
    let d = drop_frame();
    let dp = plan(&s, &d).unwrap();
    let ddir = t.0.join("prepared-drop");
    write(&s, &dp, &d, &ddir).unwrap();
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(ddir.join("dataset.json")).unwrap()).unwrap();
    assert_eq!(summary["options"]["ignore"], "drop-frame");
    assert_eq!(summary["stats"]["ignore_frames"], 1);
}

#[test]
fn neither_validation_fallback_takes_a_frame_with_a_region() {
    let t = tmp("ignore-fallback");
    let source = |n: usize| {
        let mut s = one_image_source(&t.0, n);
        for (i, f) in s.frames.iter_mut().enumerate() {
            f.image = t.0.join(format!("{i}.jpg"));
        }
        s.frames[0].ignore.push(region("person"));
        s
    };
    let opts = Options {
        heldout_share: 0.0,
        val_share: 0.0,
        min_valid: 0,
        negative_share: 0.0,
        static_cap: None,
        ..Options::default()
    };
    // One camera day: validation frame by frame, never the frame with the region; it trains.
    let p = plan(&[source(6)], &opts).unwrap();
    assert!(
        !p.valid.is_empty() && !p.valid.contains(&(0, 0)),
        "{:?}",
        p.valid
    );
    assert!(p.train.iter().any(|(r, _)| *r == (0, 0)));
    assert_eq!(p.stats.ignore_kept, 1);
    // Too few frames for a split: the trainer validates on the training frames without regions.
    let p = plan(&[source(2)], &opts).unwrap();
    assert_eq!(p.train.len(), 2);
    assert_eq!(p.valid, [(0, 1)]);
}

/// `n` training frames of camera `a` with only the parked spot `spot`, and `free` frames with a
/// normal box, over two days so that a day can be held out.
fn parked_source(dir: &Path, n: usize, free: usize, spots: &[&str]) -> Source {
    let mut s = one_image_source(dir, n + free);
    for (i, f) in s.frames.iter_mut().enumerate() {
        f.image = dir.join(format!("{i}.jpg"));
        f.day = format!("2026-10-{:02}", i % 2 + 1);
        f.split = SplitHint::Train;
        if i < n {
            f.parked = spots.iter().map(|s| (*s).to_string()).collect();
        }
    }
    s
}

/// Every frame in the pool: a split-hinted export with no held-out frame now gets a held-out share.
/// The static cap is off: these fixtures repeat one box on every frame of one camera, which is a
/// static spot, and they test the parked cap alone.
fn no_negatives() -> Options {
    Options {
        negative_share: 0.0,
        heldout_share: 0.0,
        static_cap: None,
        ..Options::default()
    }
}

fn parked_kept(s: &Source, p: &Plan) -> usize {
    p.train
        .iter()
        .map(|(r, _)| *r)
        .chain(p.valid.iter().copied())
        .collect::<BTreeSet<_>>()
        .iter()
        .filter(|r| !s.frames[r.1].parked.is_empty())
        .count()
}

#[test]
fn parked_frames_are_capped_per_spot_and_reported() {
    let t = tmp("parked-cap");
    let s = parked_source(&t.0, 50, 30, &["a/8x8/car/1,1,5,5"]);
    let sources = [s];
    let p = plan(&sources, &no_negatives()).unwrap();
    assert_eq!(parked_kept(&sources[0], &p), PARKED_CAP);
    assert_eq!(p.stats.parked_capped, 50 - PARKED_CAP);
    let warning = p.warnings.iter().find(|w| w.contains("parked-spot boxes"));
    assert!(
        warning.is_some_and(|w| w.contains("30 training frame(s)")),
        "{:?}",
        p.warnings
    );
    // The same choice on every run, another with another seed.
    let again = plan(&sources, &no_negatives()).unwrap();
    assert_eq!(p.train, again.train);
    let seeded = Options {
        seed: 7,
        ..no_negatives()
    };
    assert_ne!(p.train, plan(&sources, &seeded).unwrap().train);
}

#[test]
fn the_cap_counts_each_spot_and_leaves_other_frames_alone() {
    let t = tmp("parked-spots");
    // Under the cap nothing goes, and no word is said.
    let few = [parked_source(&t.0, 20, 10, &["a/8x8/car/1,1,5,5"])];
    let p = plan(&few, &no_negatives()).unwrap();
    assert_eq!(p.stats.parked_capped, 0);
    assert!(!p.warnings.iter().any(|w| w.contains("parked-spot")));
    // Two spots, half the frames each: each keeps its own 20.
    let mut two = parked_source(&t.0, 60, 10, &["a/8x8/car/1,1,5,5"]);
    for f in two.frames.iter_mut().take(30) {
        f.parked = vec!["a/8x8/car/9,9,15,15".into()];
    }
    let two = [two];
    let p = plan(&two, &no_negatives()).unwrap();
    assert_eq!(parked_kept(&two[0], &p), 2 * PARKED_CAP);
    // A frame with both spots stays while one of them has room.
    let both = [parked_source(
        &t.0,
        40,
        10,
        &["a/8x8/car/1,1,5,5", "a/8x8/car/9,9,15,15"],
    )];
    let p = plan(&both, &no_negatives()).unwrap();
    assert_eq!(parked_kept(&both[0], &p), PARKED_CAP);
}

#[test]
fn a_held_out_parked_frame_is_never_capped() {
    let t = tmp("parked-held");
    let mut s = parked_source(&t.0, 40, 30, &["a/8x8/car/1,1,5,5"]);
    for f in s.frames.iter_mut().take(10) {
        f.split = SplitHint::Heldout;
    }
    let p = plan(&[s], &no_negatives()).unwrap();
    assert_eq!(p.heldout.len(), 10);
    assert_eq!(p.stats.parked_capped, 10, "30 parked frames in the pool");
}

/// From a label file to the plan: the export names the frames, the reader passes them on.
#[test]
fn the_export_names_frames_with_only_parked_boxes() {
    use crate::collect::labels::export::{IGNORE_FILE, IGNORE_VERSION, IgnoreFile};
    let t = tmp("parked-read");
    let dir = &t.0;
    let tags: &[&str] = &["day"];
    flat_export(
        dir,
        &["person"],
        &[("a.jpg", "0 0.5 0.5 0.2 0.2\n", tags), ("b.jpg", "", tags)],
    );
    let file = IgnoreFile {
        version: IGNORE_VERSION,
        images: BTreeMap::new(),
        parked: [(
            "a.jpg".to_string(),
            vec!["cam/8x8/person/1,1,5,5".to_string()],
        )]
        .into(),
        size_floor: None,
    };
    std::fs::write(dir.join(IGNORE_FILE), serde_json::to_vec(&file).unwrap()).unwrap();
    let s = read_export(dir).unwrap();
    let parked: Vec<usize> = s.frames.iter().map(|f| f.parked.len()).collect();
    assert_eq!(parked.iter().sum::<usize>(), 1, "{parked:?}");
    // A file of an older version has no such list.
    std::fs::write(dir.join(IGNORE_FILE), r#"{"version": 1, "images": {}}"#).unwrap();
    assert!(
        read_export(dir)
            .unwrap()
            .frames
            .iter()
            .all(|f| f.parked.is_empty())
    );
}

/// A Dataset tab export of `frames` (flat names; `val` = held out), with its label files.
fn split_export(dir: &Path, train: &[&str], val: &[&str]) {
    std::fs::write(dir.join("classes.txt"), "cat\n").unwrap();
    for (split, list) in [("train", train), ("val", val)] {
        for stem in list {
            jpg(
                &dir.join("images").join(split).join(format!("{stem}.jpg")),
                16,
                12,
            );
            let l = dir.join("labels").join(split);
            std::fs::create_dir_all(&l).unwrap();
            let boxes = if stem.contains("-empty") {
                ""
            } else {
                "0 0.5 0.5 0.2 0.2\n"
            };
            std::fs::write(l.join(format!("{stem}.txt")), boxes).unwrap();
        }
    }
}

#[test]
fn a_single_camera_day_export_still_holds_frames_out() {
    let t = tmp("one-day");
    let names: Vec<String> = (0..20)
        .map(|i| format!("day.yard.2026-10-01.f{i:02}"))
        .collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    // What export.rs writes for one camera day: everything under train.
    split_export(&t.0, &refs, &[]);
    let s = [read_export(&t.0).unwrap()];
    let o = Options::default();
    let p = plan(&s, &o).unwrap();
    assert_eq!(p.heldout.len(), 4, "training.heldout_share of the frames");
    assert!(
        p.warnings
            .iter()
            .any(|w| w.contains("frames of one day are similar")),
        "{:?}",
        p.warnings
    );
    let held: BTreeSet<FrameRef> = p.heldout.iter().copied().collect();
    assert!(p.train.iter().all(|(r, _)| !held.contains(r)));
    assert!(p.valid.iter().all(|r| !held.contains(r)));
    // The same day whose hash sent it all to val: nothing would train.
    let t3 = tmp("one-day-val");
    split_export(&t3.0, &[], &refs);
    let s3 = [read_export(&t3.0).unwrap()];
    let p = plan(&s3, &o).unwrap();
    assert_eq!(p.heldout.len(), 4);
    assert!(!p.train.is_empty());
    // heldout_share 0 stays the explicit way out.
    let none = Options {
        heldout_share: 0.0,
        ..Options::default()
    };
    assert!(plan(&s, &none).unwrap().heldout.is_empty());
    // An explicit split wins: the export's val, nothing added.
    let t2 = tmp("explicit");
    split_export(&t2.0, &refs, &["day.gate.2026-10-02.h1"]);
    let s = [read_export(&t2.0).unwrap()];
    let p = plan(&s, &o).unwrap();
    assert_eq!(p.heldout.len(), 1);
}

#[test]
fn an_export_without_val_over_several_days_holds_whole_days_out() {
    let t = tmp("several-days");
    let names: Vec<String> = (0..24)
        .map(|i| format!("day.yard.2026-10-0{}.f{i:02}", i % 4 + 1))
        .collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    split_export(&t.0, &refs, &[]);
    let s = [read_export(&t.0).unwrap()];
    let p = plan(&s, &Options::default()).unwrap();
    assert_eq!(p.heldout.len(), 6, "one whole day");
    assert_eq!(p.stats.heldout_groups.len(), 1);
}

#[test]
fn a_frame_in_two_exports_counts_once_and_held_out_wins() {
    let (a, b) = (tmp("dup-a"), tmp("dup-b"));
    let train: Vec<String> = (0..12)
        .map(|i| format!("day.yard.2026-10-01.f{i:02}"))
        .collect();
    let refs: Vec<&str> = train.iter().map(String::as_str).collect();
    // `f00` trains in the first export and is held out in the second.
    split_export(&a.0, &refs, &["day.gate.2026-10-02.h1"]);
    split_export(&b.0, &refs[1..4], &[refs[0], "day.gate.2026-10-03.h2"]);
    let s = [read_export(&a.0).unwrap(), read_export(&b.0).unwrap()];
    let o = Options {
        min_valid: 0,
        val_share: 0.0,
        ..Options::default()
    };
    let p = plan(&s, &o).unwrap();
    // f00 (train copy) + f01..f03 (second copies) go.
    assert_eq!(p.stats.duplicates_dropped, 4);
    assert!(
        p.warnings
            .iter()
            .any(|w| w.contains("4 frame(s) are in more than one")),
        "{:?}",
        p.warnings
    );
    let name = |r: FrameRef| file_name(&s[r.0].frames[r.1].image);
    let held: Vec<String> = p.heldout.iter().map(|&r| name(r)).collect();
    assert!(held.contains(&format!("{}.jpg", refs[0])), "{held:?}");
    let trained: BTreeSet<String> = p.train.iter().map(|(r, _)| name(*r)).collect();
    assert_eq!(trained.len(), p.train.len(), "each frame once");
    assert!(
        !trained.contains(&format!("{}.jpg", refs[0])),
        "{trained:?}"
    );
    for h in &held {
        assert!(!trained.contains(h), "{h} trains and is held out");
    }
    // Other names say nothing: both 0001.jpg stay.
    let (c, d) = (tmp("dup-c"), tmp("dup-d"));
    flat_export(&c.0, &["cat"], &[("0001.jpg", "0 0.5 0.5 0.2 0.2\n", &[])]);
    flat_export(&d.0, &["cat"], &[("0001.jpg", "0 0.5 0.5 0.2 0.2\n", &[])]);
    let s = [read_export(&c.0).unwrap(), read_export(&d.0).unwrap()];
    let p = plan(&s, &Options::default()).unwrap();
    assert_eq!(p.stats.duplicates_dropped, 0);
}

#[test]
fn negatives_rank_reviewer_evidence_first_whatever_the_formats() {
    let t = tmp("negatives");
    let mut train: Vec<String> = (0..4)
        .map(|i| format!("day.yard.2026-10-01.p{i}"))
        .collect();
    let empties: Vec<String> = (0..30)
        .map(|i| format!("day.yard.2026-10-01.e{i:02}-empty"))
        .collect();
    train.extend(empties.iter().cloned());
    let refs: Vec<&str> = train.iter().map(String::as_str).collect();
    split_export(&t.0, &refs, &["day.gate.2026-10-02.h1"]);
    let jpg_of = |s: &str| format!("{s}.jpg");
    // No benchmark manifest: frames.json and hard_negatives.json say it all.
    std::fs::write(
        t.0.join(FRAMES_FILE),
        serde_json::to_vec(&FrameNotes {
            version: FRAMES_VERSION,
            empty_confirmed: vec![jpg_of(&empties[29])],
            bulk_accepted: Vec::new(),
        })
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        t.0.join(HARD_NEGATIVES_FILE),
        serde_json::to_vec(&serde_json::json!([
            {"file": jpg_of(&empties[28]), "reason": null},
            {"file": jpg_of(&empties[27]), "reason": bulk::DISMISSED_REASON},
            {"file": jpg_of(&empties[26]), "reason": bulk::SPOT_DISMISSED_REASON},
            {"file": jpg_of(&empties[25]), "reason": bulk::OTHER_CLASS_REASON},
        ]))
        .unwrap(),
    )
    .unwrap();
    let s = [read_export(&t.0).unwrap()];
    assert!(s[0].notes_known);
    let by_name = |n: &str| {
        s[0].frames
            .iter()
            .find(|f| file_name(&f.image) == jpg_of(n))
            .unwrap()
    };
    assert!(by_name(&empties[29]).hard_negative);
    assert!(by_name(&empties[28]).human_rejected);
    assert!(
        !by_name(&empties[27]).human_rejected,
        "a dismissal is no human reject"
    );
    assert!(!by_name(&empties[26]).human_rejected);
    assert!(
        !by_name(&empties[25]).human_rejected,
        "a class-list reject says nothing"
    );
    // Room for only two negatives: the two with evidence, whatever their seeded rank.
    let o = Options {
        negative_share: 2.0 / 6.0,
        min_valid: 0,
        val_share: 0.0,
        ..Options::default()
    };
    for seed in 0..5 {
        let p = plan(&s, &Options { seed, ..o.clone() }).unwrap();
        let negs: BTreeSet<String> = p
            .train
            .iter()
            .map(|(r, _)| r)
            .chain(p.valid.iter())
            .map(|(_, i)| &s[0].frames[*i])
            .filter(|f| f.boxes.is_empty())
            .map(|f| file_name(&f.image))
            .collect();
        assert_eq!(negs.len(), 2, "{negs:?}");
        assert!(negs.contains(&jpg_of(&empties[29])), "{negs:?}");
        assert!(negs.contains(&jpg_of(&empties[28])), "{negs:?}");
    }
    // A damaged file: a warning, the frames still read.
    std::fs::write(t.0.join(HARD_NEGATIVES_FILE), b"not json").unwrap();
    std::fs::write(t.0.join(FRAMES_FILE), b"not json").unwrap();
    let s = read_export(&t.0).unwrap();
    assert!(!s.notes_known);
    assert_eq!(s.warnings.len(), 2, "{:?}", s.warnings);
}

#[test]
fn a_prepared_dataset_reads_back() {
    let (t, out) = (tmp("prepared"), tmp("prepared-out"));
    camera_export(&t.0);
    let s = [read_export(&t.0).unwrap()];
    let o = Options::default();
    let p = plan(&s, &o).unwrap();
    let dir = out.0.join("data");
    let w = write(&s, &p, &o, &dir).unwrap();
    let r = read_prepared(&dir).unwrap();
    assert_eq!(r.classes, w.classes);
    assert_eq!(r.train_images, w.train_images);
    assert_eq!(r.train_frames, w.train_frames);
    assert_eq!(r.valid_images, w.valid_images);
    assert_eq!(r.valid_groups, w.valid_groups);
    assert_eq!(r.heldout_dir, w.heldout_dir);
    assert_eq!(r.heldout_images, w.heldout_images);
    assert_eq!(r.heldout_fingerprint, w.heldout_fingerprint);
    assert_eq!(r.warnings, w.warnings);
    assert!(read_prepared(&out.0).is_err(), "no dataset.json");
}

#[test]
fn a_label_file_that_cannot_be_read_never_means_no_objects() {
    let t = tmp("unreadable-label");
    flat_export(&t.0, &["person"], &[("a.jpg", "0 0.5 0.5 0.2 0.4\n", &[])]);
    // Invalid UTF-8: the file is there, and holds boxes nobody can read now.
    let label = t.0.join("labels/a.txt");
    std::fs::write(&label, b"0 0.5 0.5 0.2 0.4\n\xff\n").unwrap();
    let e = read_export(&t.0).unwrap_err();
    assert!(format!("{e:#}").contains("a.txt"), "{e:#}");
    // Not a file at all (the read fails with something else than NotFound).
    std::fs::remove_file(&label).unwrap();
    std::fs::create_dir(&label).unwrap();
    let e = read_export(&t.0).unwrap_err();
    assert!(format!("{e:#}").contains("a.txt"), "{e:#}");
}

#[test]
fn a_missing_label_file_is_no_objects_only_outside_our_own_exports() {
    let t = tmp("missing-label");
    flat_export(&t.0, &["person"], &[("a.jpg", "0 0.5 0.5 0.2 0.4\n", &[])]);
    std::fs::remove_file(t.0.join("labels/a.txt")).unwrap();
    // Somebody's YOLO folder: a background image, with a warning, as before.
    let s = read_export(&t.0).unwrap();
    assert!(s.frames[0].boxes.is_empty());
    assert!(s.warnings.iter().any(|w| w.contains("no label file")));
    // Our export wrote a label file for every image, so this one was lost.
    std::fs::write(t.0.join("export.json"), br#"{"formats":["yolo"]}"#).unwrap();
    let e = read_export(&t.0).unwrap_err();
    assert!(format!("{e:#}").contains("a.txt"), "{e:#}");
}

// ---------------------------------------------------------------------------------------------
// The static cap
// ---------------------------------------------------------------------------------------------

/// A box of the static cap's tests: (class, cx, cy, w, h).
type TBox = (&'static str, f32, f32, f32, f32);

/// A frame of camera `patio` for the static cap's tests.
fn patio_frame(dir: &Path, i: usize, day: usize, scene: &str, boxes: &[TBox]) -> Frame {
    Frame {
        image: dir.join(format!("f{i:04}.jpg")),
        boxes: boxes
            .iter()
            .map(|&(class, cx, cy, w, h)| YoloBox {
                class: class.into(),
                cx,
                cy,
                w,
                h,
            })
            .collect(),
        ignore: Vec::new(),
        floor_regions: Vec::new(),
        parked: Vec::new(),
        scene: Some(scene.into()),
        camera: "patio".into(),
        day: format!("2026-10-{day:02}"),
        split: SplitHint::Train,
        hard_negative: false,
        human_rejected: false,
        bulk_accepted: false,
        size: Some((640, 480)),
    }
}

/// The parked car of the static tests, jittered by frame.
fn parked_car(i: usize) -> TBox {
    let j = (i % 5) as f32 * 0.001;
    ("car", 0.3 + j, 0.6 - j, 0.2, 0.15)
}

/// A person walking by: at another place on each frame `k`, never a static spot.
fn person(k: usize) -> TBox {
    ("person", 0.5 + k as f32 * 0.04, 0.3, 0.05, 0.2)
}
const STATIC_SCENES: [&str; 3] = ["day", "night", "ir"];

fn patio_source(dir: &Path, frames: Vec<Frame>) -> Source {
    Source {
        dir: dir.to_path_buf(),
        public: None,
        classes: vec!["car".into(), "person".into()],
        frames,
        warnings: Vec::new(),
        notes_known: true,
        size_floor: None,
    }
}

/// Camera `patio`: 60 frames showing only the parked car (6 days, 3 scenes), 10 with the car and
/// a person, 10 with a car passing (each at another place), 20 empty, and day 7 held out (10
/// frames with the car only).
fn static_source(dir: &Path) -> Source {
    let mut frames = Vec::new();
    let mut push = |day: usize, scene: &str, boxes: &[TBox]| {
        let i = frames.len();
        frames.push(patio_frame(dir, i, day, scene, boxes));
    };
    for k in 0..60 {
        push(k % 6 + 1, STATIC_SCENES[k % 3], &[parked_car(k)]);
    }
    for k in 0..10 {
        push(k % 6 + 1, "day", &[parked_car(k), person(k)]);
    }
    for k in 0..10 {
        let x = 0.05 + k as f32 * 0.09;
        push(k % 6 + 1, "day", &[("car", x, 0.2, 0.05, 0.05)]);
    }
    for k in 0..20 {
        push(k % 6 + 1, "night", &[]);
    }
    for k in 0..10 {
        push(7, "day", &[parked_car(k)]);
    }
    for f in frames.iter_mut().filter(|f| f.day == "2026-10-07") {
        f.split = SplitHint::Heldout;
    }
    patio_source(dir, frames)
}

/// Train and valid frames of `p`, each once.
fn used_frames(p: &Plan) -> BTreeSet<FrameRef> {
    p.train
        .iter()
        .map(|(r, _)| *r)
        .chain(p.valid.iter().copied())
        .collect()
}

/// Whether a frame shows the parked car alone.
fn car_only(f: &Frame) -> bool {
    f.boxes.len() == 1 && f.boxes[0].class == "car" && (f.boxes[0].cy - 0.6).abs() < 0.01
}

fn cap_off() -> Options {
    Options {
        static_cap: None,
        ..Options::default()
    }
}

#[test]
fn frames_of_a_static_spot_are_capped_by_geometry_without_parked_info() {
    let t = tmp("static-cap");
    let s = [static_source(&t.0)];
    assert!(
        s[0].frames.iter().all(|f| f.parked.is_empty()),
        "no export info"
    );
    let p = plan(&s, &Options::default()).unwrap();
    assert_eq!(p.stats.static_spots, 1, "{:?}", p.stats);
    assert_eq!(p.stats.static_capped, 40, "{:?}", p.stats);
    assert_eq!(p.stats.parked_capped, 0);
    let used = used_frames(&p);
    let kept_static = used.iter().filter(|r| car_only(&s[0].frames[r.1])).count();
    assert_eq!(kept_static, PARKED_CAP);
    let caps = &p.stats.static_caps;
    assert_eq!(caps.len(), 1);
    assert_eq!((caps[0].frames, caps[0].kept), (60, PARKED_CAP));
    assert_eq!(
        (caps[0].camera.as_str(), caps[0].label.as_str()),
        ("patio", "car")
    );
    assert!(caps[0].id.starts_with("patio/vehicle/"), "{}", caps[0].id);
    // A frame with a person, and a passing car, are never left out.
    let other = |f: &Frame| f.day != "2026-10-07" && !f.boxes.is_empty() && !car_only(f);
    let others: BTreeSet<FrameRef> = (0..s[0].frames.len())
        .map(|i| (0, i))
        .filter(|r| other(&s[0].frames[r.1]))
        .collect();
    assert_eq!(others.len(), 20);
    assert!(others.is_subset(&used), "{:?}", others.difference(&used));
    // The held-out day is whole.
    assert_eq!(p.heldout.len(), 10);
    // The note says what went and how to keep it.
    let note = static_cap_note(&p.stats, Options::default().static_cap).unwrap();
    assert!(
        note.starts_with("40 frame(s) showing nothing but objects parked at 1 static spot(s)"),
        "{note}"
    );
    assert!(
        note.contains("a car on camera patio (60 frames, 20 kept)"),
        "{note}"
    );
    assert!(note.contains("training.static_cap"), "{note}");
    // dataset.json lists the spot.
    let v = serde_json::to_value(&p.stats).unwrap();
    assert_eq!(v["static_capped"], 40);
    assert_eq!(v["static_caps"][0]["kept"], 20);
}

#[test]
fn the_held_out_set_is_the_same_with_the_static_cap_on_and_off() {
    let t = tmp("static-held");
    let mut s = static_source(&t.0);
    // No split from the export: the plan holds out whole camera days itself.
    for f in &mut s.frames {
        f.split = SplitHint::None;
    }
    let s = [s];
    let on = plan(&s, &Options::default()).unwrap();
    let off = plan(&s, &cap_off()).unwrap();
    assert!(on.stats.static_capped > 0, "{:?}", on.stats);
    assert!(!on.heldout.is_empty());
    assert_eq!(on.heldout, off.heldout);
    assert_eq!(on.heldout_fingerprint, off.heldout_fingerprint);
    assert_eq!(on.stats.heldout_groups, off.stats.heldout_groups);
    // Off: nothing of the cap in the plan.
    assert_eq!(
        (
            off.stats.static_capped,
            off.stats.static_spots,
            off.stats.static_caps.len()
        ),
        (0, 0, 0)
    );
    assert!(static_cap_note(&off.stats, None).is_none());
}

#[test]
fn a_static_cap_of_zero_is_the_plan_without_it() {
    let t = tmp("static-zero");
    let s = [static_source(&t.0)];
    let none = plan(&s, &cap_off()).unwrap();
    let zero = plan(
        &s,
        &Options {
            static_cap: Some(0),
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(none, zero);
    let used = used_frames(&none);
    assert_eq!(
        used.iter().filter(|r| car_only(&s[0].frames[r.1])).count(),
        60
    );
    // Options of older runs (no key) read as the default cap; a written null stays off.
    let old: Options = serde_json::from_value(serde_json::json!({
        "classes": null, "negative_share": 0.15, "heldout_share": 0.2, "val_share": 0.1,
        "oversample": 3, "seed": 0
    }))
    .unwrap();
    assert_eq!(old.static_cap, Some(PARKED_CAP));
    let back: Options = serde_json::from_str(&serde_json::to_string(&cap_off()).unwrap()).unwrap();
    assert_eq!(back.static_cap, None);
}

#[test]
fn the_static_plan_does_not_depend_on_the_frame_order() {
    let t = tmp("static-order");
    let a = static_source(&t.0);
    let mut b = a.clone();
    b.frames.reverse();
    b.frames.rotate_left(13);
    let names = |s: &Source, p: &Plan| {
        let name = |r: &FrameRef| file_name(&s.frames[r.1].image);
        let mut train: Vec<(String, u32)> = p.train.iter().map(|(r, n)| (name(r), *n)).collect();
        train.sort();
        let valid: BTreeSet<String> = p.valid.iter().map(name).collect();
        let held: BTreeSet<String> = p.heldout.iter().map(name).collect();
        (train, valid, held)
    };
    let o = Options::default();
    let pa = plan(std::slice::from_ref(&a), &o).unwrap();
    let pb = plan(std::slice::from_ref(&b), &o).unwrap();
    assert!(pa.stats.static_capped > 0);
    assert_eq!(names(&a, &pa), names(&b, &pb));
    assert_eq!(pa.stats.static_caps, pb.stats.static_caps);
    assert_eq!(pa.heldout_fingerprint, pb.heldout_fingerprint);
    // Another seed keeps as many.
    let seeded = plan(std::slice::from_ref(&a), &Options { seed: 9, ..o }).unwrap();
    assert_eq!(seeded.stats.static_capped, pa.stats.static_capped);
}

#[test]
fn kept_static_frames_span_the_scenes_and_days_first() {
    let t = tmp("static-variety");
    let dir = &t.0;
    // 3 scenes x 6 days, 4 frames each: 18 groups, 72 frames of the car alone.
    let mut frames = Vec::new();
    for day in 1..=6 {
        for scene in STATIC_SCENES {
            for _ in 0..4 {
                let i = frames.len();
                frames.push(patio_frame(dir, i, day, scene, &[parked_car(i)]));
            }
        }
    }
    // Something else to train on.
    for k in 0..5 {
        let i = frames.len();
        frames.push(patio_frame(dir, i, k % 6 + 1, "day", &[person(k)]));
    }
    let s = [patio_source(dir, frames)];
    let o = Options {
        heldout_share: 0.0,
        ..Options::default()
    };
    for seed in 0..4 {
        let p = plan(&s, &Options { seed, ..o.clone() }).unwrap();
        let mut groups: BTreeMap<(String, String), usize> = BTreeMap::new();
        for r in used_frames(&p) {
            let f = &s[0].frames[r.1];
            if car_only(f) {
                *groups
                    .entry((f.scene.clone().unwrap(), f.day.clone()))
                    .or_default() += 1;
            }
        }
        assert_eq!(groups.values().sum::<usize>(), PARKED_CAP, "seed {seed}");
        assert_eq!(
            groups.len(),
            18,
            "every scene and day before a second frame of any: {groups:?}"
        );
        assert!(groups.values().all(|n| *n <= 2), "{groups:?}");
    }
    // Two days, three scenes: 6 groups of 10; the 20 kept are 3 or 4 of each, so 6 or 7 per scene.
    let mut frames: Vec<Frame> = (0..60)
        .map(|k| {
            patio_frame(
                dir,
                1000 + k,
                k % 2 + 1,
                STATIC_SCENES[k % 3],
                &[parked_car(k)],
            )
        })
        .collect();
    frames.push(patio_frame(dir, 2000, 1, "day", &[person(0)]));
    let s2 = [patio_source(dir, frames)];
    let p = plan(&s2, &o).unwrap();
    let mut per_scene: BTreeMap<String, usize> = BTreeMap::new();
    for r in used_frames(&p) {
        let f = &s2[0].frames[r.1];
        if car_only(f) {
            *per_scene.entry(f.scene.clone().unwrap()).or_default() += 1;
        }
    }
    assert_eq!(per_scene.values().sum::<usize>(), PARKED_CAP);
    assert!(
        per_scene.values().all(|n| (6..=7).contains(n)),
        "{per_scene:?}"
    );
}

#[test]
fn a_frame_of_an_unpicked_class_only_is_a_negative_not_static() {
    let t = tmp("static-excluded");
    let s = [static_source(&t.0)];
    // Only persons train: the car frames are negatives (background), never static.
    let p = plan(
        &s,
        &Options {
            classes: Some(vec!["person".into()]),
            negative_share: 0.0,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!((p.stats.static_capped, p.stats.static_spots), (0, 0));
    // The car frames are negatives: with room for them all, every one stays.
    let p = plan(
        &s,
        &Options {
            classes: Some(vec!["person".into()]),
            negative_share: 0.9,
            ..Options::default()
        },
    )
    .unwrap();
    assert_eq!(p.stats.static_capped, 0);
    let used = used_frames(&p);
    assert_eq!(
        used.iter().filter(|r| car_only(&s[0].frames[r.1])).count(),
        60
    );
}

#[test]
fn negatives_follow_the_frames_the_static_cap_left() {
    let t = tmp("static-negatives");
    let mut s = static_source(&t.0);
    for k in 0..100 {
        let f = patio_frame(&t.0, 5000 + k, k % 6 + 1, "night", &[]);
        s.frames.push(f);
    }
    let s = [s];
    let o = Options {
        negative_share: 0.5,
        ..Options::default()
    };
    let p = plan(&s, &o).unwrap();
    assert_eq!(p.stats.static_capped, 40);
    let used = used_frames(&p);
    let (neg, pos): (Vec<FrameRef>, Vec<FrameRef>) =
        used.iter().partition(|r| s[0].frames[r.1].boxes.is_empty());
    // 20 static + 10 with a person + 10 passing cars; as many negatives at a share of 0.5.
    assert_eq!(pos.len(), 40);
    assert_eq!(neg.len(), 40);
}

#[test]
fn the_parked_cap_runs_first_and_the_static_cap_on_the_rest() {
    let t = tmp("static-parked");
    let dir = t.0.clone();
    let mut frames: Vec<Frame> = (0..80)
        .map(|k| patio_frame(&dir, k, k % 4 + 1, STATIC_SCENES[k % 3], &[parked_car(k)]))
        .collect();
    // The export knew 50 of them as parked-spot frames.
    for f in frames.iter_mut().take(50) {
        f.parked = vec!["patio/640x480/car/160,250,290,330".into()];
    }
    frames.push(patio_frame(&dir, 100, 1, "day", &[person(0)]));
    let s = [patio_source(&dir, frames)];
    let o = Options {
        heldout_share: 0.0,
        negative_share: 0.0,
        ..Options::default()
    };
    let p = plan(&s, &o).unwrap();
    // 50 parked: 20 kept by the parked cap; then those 20 + 30 static-only: 20 kept.
    assert_eq!(p.stats.parked_capped, 30);
    assert_eq!(p.stats.static_capped, 30);
    assert_eq!(
        p.stats.static_caps[0].frames, 50,
        "what the parked cap took out is not counted again"
    );
    let used = used_frames(&p);
    assert_eq!(
        used.iter().filter(|r| car_only(&s[0].frames[r.1])).count(),
        PARKED_CAP
    );
}

#[test]
fn a_person_at_one_place_is_never_a_static_spot_nor_capped() {
    let t = tmp("static-person");
    let dir = t.0.clone();
    // Someone sitting at the same place on 40 frames over 4 days, and the parked car on 40
    // others: the car's frames are capped, the person's never.
    let sitting: TBox = ("person", 0.7, 0.4, 0.08, 0.25);
    let mut frames: Vec<Frame> = (0..40)
        .map(|k| patio_frame(&dir, k, k % 4 + 1, STATIC_SCENES[k % 3], &[sitting]))
        .collect();
    frames.extend((0..40).map(|k| patio_frame(&dir, 100 + k, k % 4 + 1, "day", &[parked_car(k)])));
    let s = [patio_source(&dir, frames)];
    let o = Options {
        heldout_share: 0.0,
        negative_share: 0.0,
        ..Options::default()
    };
    let p = plan(&s, &o).unwrap();
    assert_eq!(p.stats.static_spots, 1, "the car's only: {:?}", p.stats);
    assert_eq!(p.stats.static_capped, 20);
    assert_eq!(p.stats.static_caps[0].label, "car");
    let used = used_frames(&p);
    let persons = used
        .iter()
        .filter(|r| s[0].frames[r.1].boxes[0].class == "person")
        .count();
    assert_eq!(persons, 40, "every person frame trains");
}

#[test]
fn a_minority_class_at_a_spot_keeps_a_quota_of_its_own() {
    let t = tmp("static-truck");
    let dir = &t.0;
    // A car at its place on 60 days (2 frames a day), and a pickup labelled truck parked at the
    // same place on 2 of those days (2 frames each): one spot (vehicles are one object), two
    // quotas. With one quota of 20 for the place the truck's 4 frames were lost about half the
    // time; now they always stay.
    let mut frames = Vec::new();
    for day in 1..=60 {
        for _ in 0..2 {
            let i = frames.len();
            frames.push(patio_frame(
                dir,
                i,
                day,
                STATIC_SCENES[i % 3],
                &[parked_car(i)],
            ));
        }
    }
    for day in [10, 40] {
        for _ in 0..2 {
            let i = frames.len();
            let (_, cx, cy, w, h) = parked_car(i);
            frames.push(patio_frame(dir, i, day, "day", &[("truck", cx, cy, w, h)]));
        }
    }
    for k in 0..5 {
        let i = frames.len();
        frames.push(patio_frame(dir, i, k + 1, "day", &[person(k)]));
    }
    let s = [Source {
        classes: vec!["car".into(), "truck".into(), "person".into()],
        ..patio_source(dir, frames)
    }];
    let o = Options {
        heldout_share: 0.0,
        negative_share: 0.0,
        ..Options::default()
    };
    for seed in 0..20 {
        let p = plan(&s, &Options { seed, ..o.clone() }).unwrap();
        let used = used_frames(&p);
        let trucks = used
            .iter()
            .filter(|r| s[0].frames[r.1].boxes[0].class == "truck")
            .count();
        assert_eq!(trucks, 4, "seed {seed}: {:?}", p.stats.static_caps);
        assert_eq!(p.stats.static_spots, 1);
        // `static_caps`: one entry per (spot, class); `label` is the class of the quota.
        let caps = &p.stats.static_caps;
        assert_eq!(caps.len(), 2, "{caps:?}");
        assert_eq!(caps[0].id, caps[1].id, "one spot");
        assert_eq!(
            (caps[0].label.as_str(), caps[0].frames, caps[0].kept),
            ("car", 120, PARKED_CAP)
        );
        assert_eq!(
            (caps[1].label.as_str(), caps[1].frames, caps[1].kept),
            ("truck", 4, 4)
        );
        assert_eq!(p.stats.static_capped, 100);
        assert_eq!(
            p.stats.static_capped_boxes,
            [("car".to_string(), 100)].into()
        );
        // The note counts the place once.
        let note = static_cap_note(&p.stats, o.static_cap).unwrap();
        assert!(
            note.starts_with("100 frame(s) showing nothing but objects parked at 1 static spot(s)"),
            "{note}"
        );
        assert!(note.contains("per spot and class"), "{note}");
    }
}

#[test]
fn a_class_the_cap_took_frames_of_says_so_in_its_warnings() {
    let mut stats = Stats::default();
    assert_eq!(static_cap_clause(&stats, "truck"), None);
    stats.static_capped_boxes.insert("truck".into(), 7);
    let c = static_cap_clause(&stats, "truck").unwrap();
    assert!(c.contains("left out 7 box(es)"), "{c}");
    assert!(c.contains("training.static_cap to 0"), "{c}");
    assert!(c.contains("--static-cap 0"), "{c}");
    assert_eq!(static_cap_clause(&stats, "car"), None);
    // The note of a run whose classes can never be static: only with the cap on.
    let merged = vec!["vehicle".to_string(), "person".to_string()];
    assert!(static_cap_idle_note(&merged, Some(20)).is_some());
    assert!(static_cap_idle_note(&merged, None).is_none());
    assert!(static_cap_idle_note(&merged, Some(0)).is_none());
    assert!(static_cap_idle_note(&["car".to_string()], Some(20)).is_none());
}

// ---- the size floor at the run's input, and the exports' floors

const FLOOR_384: SizeFloor = SizeFloor {
    px: 10.0,
    input: 384,
};

/// A size floor region of `class`: `w` x `h` of the image (10 px at 384 = 0.026 of it).
fn floor_region(class: &str, w: f32, h: f32) -> YoloBox {
    YoloBox {
        class: class.into(),
        cx: 0.8,
        cy: 0.8,
        w,
        h,
    }
}

/// `s` exported with `floor`, and these size floor regions on their frames.
fn with_floor_regions(s: &Source, floor: SizeFloor, at: &[(FrameRef, YoloBox)]) -> Source {
    let mut s = s.clone();
    s.size_floor = Some(floor);
    for (r, b) in at {
        let f = &mut s.frames[r.1];
        f.floor_regions.push(f.ignore.len());
        f.ignore.push(b.clone());
    }
    s
}

fn at_imgsz(n: Option<u32>) -> Options {
    Options {
        imgsz: n,
        ..Options::default()
    }
}

#[test]
fn ignore_json_names_the_floor_and_its_regions() {
    let t = tmp("floor-read");
    camera_export(&t.0);
    let regions = serde_json::json!({
        "version": 1,
        "size_floor": {"px": 10, "input": 384},
        "images": {"yard-2026-10-01-0.jpg": [
            {"label": "cat", "cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.3},
            {"label": "dog", "cx": 0.1, "cy": 0.1, "w": 0.02, "h": 0.03, "size_floor": true},
        ]},
    });
    std::fs::write(
        t.0.join("ignore.json"),
        serde_json::to_vec(&regions).unwrap(),
    )
    .unwrap();
    let s = read_export(&t.0).unwrap();
    assert_eq!(s.size_floor, Some(FLOOR_384));
    let f = s.frames.iter().find(|f| !f.ignore.is_empty()).unwrap();
    assert_eq!(f.ignore.len(), 2);
    assert_eq!(
        f.floor_regions,
        [1],
        "only the floor's region; the unsure one is a reviewer's"
    );
    // An export of no floor, or from before floors: none.
    ignore_json(
        &t.0,
        &[("yard-2026-10-01-0.jpg", "cat", 0.5, 0.5, 0.3, 0.3)],
    );
    let s = read_export(&t.0).unwrap();
    assert_eq!(s.size_floor, None);
    assert!(s.frames.iter().all(|f| f.floor_regions.is_empty()));
}

#[test]
fn a_run_at_a_larger_input_trains_the_floor_regions_it_can_learn() {
    let t = tmp("floor-larger");
    camera_export(&t.0);
    let s = read_export(&t.0).unwrap();
    let before = plan(std::slice::from_ref(&s), &Options::default()).unwrap();
    let pooled: Vec<FrameRef> = before.train.iter().map(|(r, _)| *r).collect();
    let (grown, tiny) = (pooled[0], pooled[1]);
    let held = before.heldout[0];
    // At 384 both are under 10 px (7.7 and 3.8); at 640 the first is 12.8, the second 6.4.
    let s = with_floor_regions(
        &s,
        FLOOR_384,
        &[
            (grown, floor_region("dog", 0.02, 0.03)),
            (tiny, floor_region("cat", 0.01, 0.03)),
            (held, floor_region("dog", 0.02, 0.03)),
        ],
    );
    let s = [s];
    let at384 = plan(&s, &at_imgsz(Some(384))).unwrap();
    let at640 = plan(&s, &at_imgsz(Some(640))).unwrap();
    let unknown = plan(&s, &at_imgsz(None)).unwrap();
    // Held out exactly as exported, at any input.
    for p in [&at640, &unknown] {
        assert_eq!(p.heldout, at384.heldout);
        assert_eq!(p.heldout_fingerprint, at384.heldout_fingerprint);
    }
    assert!(at384.stats.floor_restored.is_empty() && unknown.stats.floor_restored.is_empty());
    assert_eq!(
        at640.stats.floor_restored,
        BTreeMap::from([("dog".to_string(), 1)]),
        "{:?}",
        at640.stats
    );
    assert!(
        at640.warnings.iter().any(|w| w
            .starts_with("1 object(s) the export's size floor (10 px at 384) left out")
            && w.contains("this run's 640 px input")),
        "{:?}",
        at640.warnings
    );
    // The given-back frame has no region left: it trains or validates as a plain frame.
    let view = at_run_input(&s, &at640, &at_imgsz(Some(640)));
    let g = &view[0].frames[grown.1];
    assert!(g.ignore.is_empty() && g.floor_regions.is_empty());
    assert_eq!(g.boxes.last().unwrap().class, "dog");
    assert!(used(&at640, grown));
    let k = &view[0].frames[tiny.1];
    assert_eq!((k.ignore.len(), k.floor_regions.as_slice()), (1, &[0][..]));
    assert_eq!(view[0].frames[held.1].ignore.len(), 1, "held out keeps it");
    // Written: the dog is a label line of the frame, the held-out one stays an ignore region.
    let out = t.0.join("prepared");
    let opts = at_imgsz(Some(640));
    write(&s, &at640, &opts, &out).unwrap();
    let name = format!("s0-{}", name_of(&s[0], grown));
    let label = std::fs::read_to_string(
        out.join("labels")
            .join(format!("{}.txt", stem(Path::new(&name)))),
    )
    .unwrap();
    assert!(
        label.lines().any(|l| l.starts_with("1 0.800000 0.800000")),
        "{label}"
    );
    let m: Manifest =
        serde_json::from_slice(&std::fs::read(out.join("heldout/manifest.json")).unwrap()).unwrap();
    assert!(
        m.images
            .iter()
            .flat_map(|i| i.objects.iter().flatten())
            .any(|o| o.ignore && o.label == "dog")
    );
    let summary: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("dataset.json")).unwrap()).unwrap();
    assert_eq!(summary["stats"]["floor_restored"]["dog"], 1);
    assert_eq!(summary["options"]["imgsz"], 640);
    // The pre-flight counts the frame's dog with the labels.
    let r = crate::training::preflight::report(&at640, &s, &opts, None);
    assert_eq!(r.floor_restored, BTreeMap::from([("dog".to_string(), 1)]));
    let r384 = crate::training::preflight::report(&at384, &s, &at_imgsz(Some(384)), None);
    let dogs = |r: &crate::training::preflight::Report| {
        let l = r.classes.iter().find(|c| c.name == "dog").unwrap();
        l.train + l.valid
    };
    assert_eq!(dogs(&r), dogs(&r384) + 1);
    // A renamed label is counted under its new name, as its box is mapped.
    let renamed = Options {
        rename: BTreeMap::from([("dog".to_string(), "canine".to_string())]),
        ..at_imgsz(Some(640))
    };
    let p = plan(&s, &renamed).unwrap();
    assert_eq!(
        p.stats.floor_restored,
        BTreeMap::from([("canine".to_string(), 1)])
    );
}

#[test]
fn a_run_at_a_smaller_or_unknown_input_says_what_the_floor_cannot_do() {
    let t = tmp("floor-smaller");
    camera_export(&t.0);
    let s = read_export(&t.0).unwrap();
    let before = plan(std::slice::from_ref(&s), &Options::default()).unwrap();
    let r = before.train[0].0;
    let at640 = SizeFloor {
        px: 10.0,
        input: 640,
    };
    let mut s = with_floor_regions(&s, at640, &[(r, floor_region("dog", 0.01, 0.01))]);
    // A label of 8 px at 384 (13.4 at 640): over the export's floor, under it at the run's input.
    s.frames[r.1].boxes.push(floor_region("cat", 0.021, 0.021));
    let s = [s];
    let p = plan(&s, &at_imgsz(Some(384))).unwrap();
    assert!(
        p.stats.floor_restored.is_empty(),
        "boxes never become regions"
    );
    let w = p
        .warnings
        .iter()
        .find(|w| w.contains("this run trains at 384 px"))
        .unwrap_or_else(|| panic!("{:?}", p.warnings));
    assert!(
        w.contains(
            "was applied at 640: objects of 6.0 to 10 px at this run's input are still labels \
             (1 on its training frames)"
        ) && w.contains("export again with the floor's input set to 384"),
        "{w}"
    );
    // Unknown: a warning that names the floor, nothing given back.
    let p = plan(&s, &at_imgsz(None)).unwrap();
    assert!(
        p.warnings.iter().any(|w| w
            .contains("left out 1 object(s) under its size floor (10 px at 640)")
            && w.contains("does not know the size the run trains at")),
        "{:?}",
        p.warnings
    );
    // The floor's own input: nothing to say.
    let p = plan(&s, &at_imgsz(Some(640))).unwrap();
    assert!(
        p.warnings.iter().all(|w| !w.contains("size floor")),
        "{:?}",
        p.warnings
    );
}

#[test]
fn exports_of_other_floors_than_each_other_or_the_current_one_are_named() {
    let t = tmp("floor-mixed");
    let (a, b) = (t.0.join("a"), t.0.join("b"));
    for d in [&a, &b] {
        std::fs::create_dir_all(d).unwrap();
        camera_export(d);
    }
    let sa = read_export(&a).unwrap();
    let sb = read_export(&b).unwrap();
    let s = [
        Source {
            size_floor: Some(FLOOR_384),
            ..sa.clone()
        },
        sb.clone(),
    ];
    let p = plan(&s, &Options::default()).unwrap();
    let mixed = p
        .warnings
        .iter()
        .find(|w| w.starts_with("the exports were made with different size floors"))
        .unwrap_or_else(|| panic!("{:?}", p.warnings));
    assert!(
        mixed.contains(&format!("{} (10 px at 384)", a.display()))
            && mixed.contains(&format!("{} (no size floor)", b.display())),
        "{mixed}"
    );
    // Against the current floor: the one without, and one of another floor.
    let now = Options {
        size_floor: Some(SizeFloor {
            px: 12.0,
            input: 384,
        }),
        ..Options::default()
    };
    let p = plan(&s, &now).unwrap();
    assert!(
        p.warnings.iter().any(|w| w.starts_with(&format!(
            "{} was exported without a size floor (before it was set",
            b.display()
        )) && w.contains("under 12 px at 384 still train as labels")),
        "{:?}",
        p.warnings
    );
    assert!(
        p.warnings.iter().any(|w| w.starts_with(&format!(
            "{} was exported with a size floor of 10 px at 384, and the current floor is 12 px \
             at 384",
            a.display()
        ))),
        "{:?}",
        p.warnings
    );
    // One floor everywhere, the current one: nothing to say; no current floor: no comparison.
    let same = [
        Source {
            size_floor: Some(FLOOR_384),
            ..sa.clone()
        },
        Source {
            size_floor: Some(FLOOR_384),
            ..sb.clone()
        },
    ];
    let p = plan(
        &same,
        &Options {
            size_floor: Some(FLOOR_384),
            ..Options::default()
        },
    )
    .unwrap();
    assert!(
        p.warnings.iter().all(|w| !w.contains("size floor")),
        "{:?}",
        p.warnings
    );
    let p = plan(&[sa, sb], &Options::default()).unwrap();
    assert!(p.warnings.iter().all(|w| !w.contains("size floor")));
}

/// 6 camera days of 2 cats (one with an ignore region) and 6 empty frames each.
fn many_negatives(dir: &Path) -> Vec<Source> {
    let mut frames: Vec<(String, String, Vec<String>)> = Vec::new();
    let mut regions: Vec<String> = Vec::new();
    for cam in ["drive", "yard", "gate"] {
        for day in ["2026-10-01", "2026-10-02"] {
            for i in 0..8 {
                let label = if i < 2 { "0 0.5 0.5 0.2 0.4\n" } else { "" };
                let name = format!("{cam}-{day}-{i}.jpg");
                if i == 0 {
                    regions.push(name.clone());
                }
                frames.push((
                    name,
                    label.to_string(),
                    vec!["day".into(), format!("camera:{cam}"), format!("date:{day}")],
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
    flat_export(dir, &["cat"], &refs2);
    let at: Vec<(&str, &str, f64, f64, f64, f64)> = regions
        .iter()
        .map(|n| (n.as_str(), "cat", 0.9, 0.9, 0.02, 0.02))
        .collect();
    ignore_json(dir, &at);
    vec![read_export(dir).unwrap()]
}

#[test]
fn positives_leaving_with_their_validation_day_do_not_count_for_the_negatives() {
    let t = tmp("valid-day-negatives");
    let s = many_negatives(&t.0);
    let positive = |r: &FrameRef| !s[0].frames[r.1].boxes.is_empty();
    // Half negatives: the target is one negative per positive that trains or validates.
    let opts = Options {
        negative_share: 0.5,
        static_cap: None,
        ..Options::default()
    };
    let p = plan(&s, &opts).unwrap();
    assert!(p.stats.ignore_valid_day > 0, "{:?}", p.stats);
    assert_eq!(
        p.stats.negatives_trimmed, p.stats.ignore_valid_day,
        "{:?}",
        p.stats
    );
    let used: BTreeSet<FrameRef> = p
        .train
        .iter()
        .map(|(r, _)| *r)
        .chain(p.valid.iter().copied())
        .collect();
    let (pos, neg): (Vec<&FrameRef>, Vec<&FrameRef>) = used.iter().partition(|r| positive(r));
    assert_eq!(neg.len(), pos.len(), "{:?}", p.stats);
    // Drop-frame plans never trim: no frame with a region is in the pool to leave.
    let d = plan(
        &s,
        &Options {
            ignore: IgnoreMode::DropFrame,
            ..opts.clone()
        },
    )
    .unwrap();
    assert_eq!(
        (d.stats.negatives_trimmed, d.stats.ignore_valid_day),
        (0, 0)
    );
}

#[test]
fn validation_on_public_frames_only_is_said() {
    let t = tmp("valid-public-only");
    let own = t.0.join("own");
    std::fs::create_dir_all(&own).unwrap();
    camera_export(&own);
    let mut s = read_export(&own).unwrap();
    // Every own frame shows an unsure object: none can validate.
    for f in &mut s.frames {
        f.ignore.push(region("dog"));
    }
    let public = animal_public(&t.0.join("lila"), 60);
    let p = plan(&[s, public], &Options::default()).unwrap();
    assert!(p.valid.iter().all(|r| r.0 == 1), "{:?}", p.valid);
    assert!(
        p.warnings.iter().any(|w| w
            .starts_with("every training frame of your own shows an unsure or too-small object")),
        "{:?}",
        p.warnings
    );
}
