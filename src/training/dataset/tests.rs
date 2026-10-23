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
    let first_cat = coco["annotations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["category_id"] == 0 && a["bbox"][2] == 12.8)
        .expect("a cat box of frame 0");
    assert_eq!(
        first_cat["bbox"],
        serde_json::json!([25.6, 14.4, 12.8, 19.2])
    );
    assert_eq!(first_cat["area"], serde_json::json!(245.76));
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
        scene: SceneFilter::NightIr,
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
    assert_eq!(SceneFilter::parse("night+ir"), Some(SceneFilter::NightIr));
    assert_eq!(SceneFilter::parse("IR"), Some(SceneFilter::NightIr));
    assert_eq!(SceneFilter::parse("day"), Some(SceneFilter::Day));
    assert_eq!(SceneFilter::parse("dusk"), None);
    assert!(SceneFilter::NightIr.keeps(Some("ir")));
    assert!(!SceneFilter::NightIr.keeps(None));
    assert!(SceneFilter::All.keeps(None));
    assert_eq!(
        serde_json::to_string(&SceneFilter::NightIr).unwrap(),
        r#""night+ir""#
    );
    let meta = meta_from_name("20261022T123456789Z_ab12");
    assert_eq!(meta.day.as_deref(), Some("2026-10-22"));
    assert_eq!(meta.camera, None);
}
