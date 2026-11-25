//! Characterisation of [`plan`] and [`write_reusing`] (QA-255): the whole plan (serialized) for
//! a synthetic world of three cameras over six days under many option sets, and the files a
//! written dataset holds. Pinned before `plan` and `write_reusing` were split into step
//! functions; a difference here is a change of behaviour, never a reason to update the pins
//! without one.

use super::*;
use sha2::Digest;

fn sha(bytes: &[u8]) -> String {
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn b(class: &str, cx: f32, cy: f32, w: f32, h: f32) -> YoloBox {
    YoloBox {
        class: class.into(),
        cx,
        cy,
        w,
        h,
    }
}

const CAMERAS: [&str; 3] = ["patio", "drive", "gate"];
const SCENE_CYCLE: [Option<&str>; 4] = [Some("day"), Some("night"), Some("ir"), None];

/// Frame `k` of the review source in `dir`: three cameras over six days, a car parked at one
/// place per camera, people and dogs walking by, cats (an extra class), empty frames confirmed
/// empty or with a rejected box, ignore regions (some the size floor made), parked-spot frames and
/// random picks.
fn world_frame(dir: &Path, k: usize) -> Frame {
    let cam = CAMERAS[k % 3];
    let day = (k / 3) % 6 + 1;
    let scene = SCENE_CYCLE[(k / 7) % 4];
    let j = (k % 5) as f32 * 0.001;
    let ci = (k % 3) as f32 * 0.1;
    let parked_car = b("car", 0.2 + ci + j, 0.6 - j, 0.2, 0.15);
    let walk = (k % 11) as f32 * 0.07;
    let mut f = Frame {
        image: dir.join(format!(
            "{}.{cam}.2026-10-{day:02}.f{k:04}.jpg",
            scene.unwrap_or("day")
        )),
        boxes: Vec::new(),
        ignore: Vec::new(),
        floor_regions: Vec::new(),
        scene: scene.map(str::to_string),
        camera: cam.into(),
        day: format!("2026-10-{day:02}"),
        split: SplitHint::None,
        hard_negative: false,
        human_rejected: false,
        bulk_accepted: k.is_multiple_of(13),
        random_pick: k.is_multiple_of(37),
        size: Some((640, 480)),
        parked: Vec::new(),
    };
    match k % 10 {
        0..=3 => f.boxes.push(parked_car),
        4 => f.boxes.push(b("person", 0.1 + walk, 0.4, 0.05 + j, 0.2)),
        5 => f.boxes.push(b("dog", 0.8 - walk / 2.0, 0.7, 0.1, 0.08)),
        6 => f.hard_negative = k % 20 == 6,
        7 => f.human_rejected = true,
        8 => f.boxes.push(b("cat", 0.4, 0.4 + walk / 4.0, 0.06, 0.05)),
        _ => {
            f.boxes.push(b("person", 0.6, 0.5 - walk / 4.0, 0.04, 0.12));
            // Under the export's floor at 384 (5.4 px), over it at 640 (9 px).
            f.ignore.push(b("person", 0.9, 0.1, 0.014, 0.03));
            f.ignore.push(b("dog", 0.1, 0.9, 0.05, 0.04));
            if k % 20 == 9 {
                f.floor_regions.push(0);
            }
        }
    }
    if k % 10 <= 1 {
        f.parked = vec!["lot".into()];
    }
    if k % 40 == 1 {
        f.parked.push(format!("{cam}-spot"));
    }
    // A fox, only ever seen on camera gate's last day.
    if cam == "gate" && day == 6 && f.boxes.is_empty() {
        f.boxes.push(b("fox", 0.3, 0.3, 0.1, 0.1));
    }
    if k % 17 == 3 {
        f.boxes.push(b("truck", 0.5, 0.5, 0.3, 0.2));
    }
    f
}

fn world(n: usize) -> Source {
    let dir = PathBuf::from("pins").join("review");
    Source {
        frames: (0..n).map(|k| world_frame(&dir, k)).collect(),
        dir,
        public: None,
        classes: vec![
            "car".into(),
            "person".into(),
            "dog".into(),
            "cat".into(),
            "truck".into(),
            "fox".into(),
        ],
        warnings: vec!["a warning of the read".into()],
        notes_known: true,
        size_floor: Some(SizeFloor {
            px: 8.0,
            input: 384,
        }),
    }
}

fn public(n: usize) -> Source {
    let dir = PathBuf::from("pins").join("public");
    let frames = (0..n)
        .map(|k| Frame {
            image: dir.join(format!("p{k:03}.jpg")),
            boxes: if k % 3 == 0 {
                Vec::new()
            } else {
                vec![b(
                    if k % 2 == 0 { "dog" } else { "person" },
                    0.5,
                    0.5,
                    0.2,
                    0.3,
                )]
            },
            ignore: Vec::new(),
            floor_regions: Vec::new(),
            scene: Some(if k % 2 == 0 { "night" } else { "ir" }.into()),
            camera: UNKNOWN.into(),
            day: UNKNOWN.into(),
            split: SplitHint::Train,
            hard_negative: false,
            human_rejected: false,
            bulk_accepted: false,
            random_pick: false,
            size: Some((800, 600)),
            parked: Vec::new(),
        })
        .collect();
    Source {
        dir,
        public: Some(Public {
            id: "lila-night".into(),
            license: "CDLA".into(),
        }),
        classes: vec!["person".into(), "dog".into()],
        frames,
        warnings: Vec::new(),
        notes_known: false,
        size_floor: None,
    }
}

/// The world with the export's held-out hint on camera `gate`'s days 5 and 6.
fn hinted(n: usize) -> Source {
    let mut s = world(n);
    for f in &mut s.frames {
        if f.camera == "gate" && (f.day == "2026-10-05" || f.day == "2026-10-06") {
            f.split = SplitHint::Heldout;
        } else {
            f.split = SplitHint::Train;
        }
    }
    s
}

/// The plan of every case as (name, sha256 of its JSON, its JSON).
fn cases() -> Vec<(&'static str, Result<String, String>)> {
    let plan_of = |sources: &[Source], opts: &Options| -> Result<String, String> {
        plan(sources, opts)
            .map(|p| serde_json::to_string(&p).unwrap())
            .map_err(|e| format!("{e:#}"))
    };
    let base = Options::default();
    let mut out = Vec::new();
    out.push(("default", plan_of(&[world(240)], &base)));
    out.push((
        "drop frames",
        plan_of(
            &[world(240)],
            &Options {
                classes: Some(vec!["car".into(), "person".into(), "dog".into()]),
                excluded: Excluded::DropFrame,
                ignore: IgnoreMode::DropFrame,
                ..base.clone()
            },
        ),
    ));
    out.push((
        "balanced",
        plan_of(
            &[world(240)],
            &Options {
                seed: 3,
                ..balanced("class,scene,camera", Some(2.0))
            },
        ),
    ));
    out.push((
        "tight caps",
        plan_of(
            &[world(240)],
            &Options {
                static_cap: Some(3),
                negative_share: 0.5,
                seed: 7,
                ..base.clone()
            },
        ),
    ));
    out.push((
        "few negatives",
        plan_of(
            &[world(240)],
            &Options {
                negative_share: 0.8,
                ..base.clone()
            },
        ),
    ));
    out.push((
        "floor restored",
        plan_of(
            &[world(240)],
            &Options {
                imgsz: Some(640),
                size_floor: Some(SizeFloor {
                    px: 10.0,
                    input: 640,
                }),
                ..base.clone()
            },
        ),
    ));
    out.push((
        "hinted and frozen",
        plan_of(
            &[hinted(240)],
            &Options {
                frozen_heldout: ["patio/2026-10-02".to_string(), "drive/2026-10-03".into()]
                    .into_iter()
                    .collect(),
                ..base.clone()
            },
        ),
    ));
    out.push((
        "frozen, no hint",
        plan_of(
            &[world(240)],
            &Options {
                frozen_heldout: ["patio/2026-10-02".to_string()].into_iter().collect(),
                ..base.clone()
            },
        ),
    ));
    out.push((
        "one camera day",
        plan_of(
            &[{
                let mut s = world(240);
                s.frames
                    .retain(|f| f.camera == "patio" && f.day == "2026-10-01");
                s
            }],
            &base,
        ),
    ));
    out.push((
        "with public",
        plan_of(
            &[world(240), public(30)],
            &Options {
                oversample: 3,
                scene: SceneFilter::NIGHT_IR,
                ..base.clone()
            },
        ),
    ));
    out.push((
        "duplicates",
        plan_of(
            &[world(120), {
                let mut s = hinted(120);
                s.dir = PathBuf::from("pins").join("again");
                s
            }],
            &base,
        ),
    ));
    out.push((
        "renamed, nothing held out",
        plan_of(
            &[world(240)],
            &Options {
                rename: [
                    ("cat".to_string(), "dog".to_string()),
                    ("truck".into(), String::new()),
                ]
                .into_iter()
                .collect(),
                heldout_share: 0.0,
                ..base.clone()
            },
        ),
    ));
    out.push((
        "a class only held out",
        plan_of(
            &[hinted(240)],
            &Options {
                classes: Some(vec!["car".into(), "person".into(), "bird".into()]),
                ..base.clone()
            },
        ),
    ));
    out.push((
        "a class only on held-out days",
        plan_of(
            &[world(240)],
            &Options {
                classes: Some(vec!["car".into(), "person".into(), "fox".into()]),
                ..base.clone()
            },
        ),
    ));
    out.push((
        "a class only in the held-out split",
        plan_of(
            &[{
                let mut s = hinted(240);
                for f in &mut s.frames {
                    if f.camera == "gate" && f.day == "2026-10-05" && f.boxes.is_empty() {
                        f.boxes.push(b("fox", 0.3, 0.3, 0.1, 0.1));
                    }
                }
                s
            }],
            &Options {
                classes: Some(vec!["car".into(), "person".into(), "fox".into()]),
                ..base.clone()
            },
        ),
    ));
    out.push((
        "only cars, the rest dropped",
        plan_of(
            &[world(240)],
            &Options {
                classes: Some(vec!["car".into()]),
                excluded: Excluded::DropFrame,
                ..base.clone()
            },
        ),
    ));
    out.push((
        "no positives",
        plan_of(
            &[world(240)],
            &Options {
                classes: Some(vec!["bird".into()]),
                ..base.clone()
            },
        ),
    ));
    out.push((
        "duplicate classes",
        plan_of(
            &[world(10)],
            &Options {
                classes: Some(vec!["car".into(), "car".into()]),
                ..base
            },
        ),
    ));
    out
}

/// Every case's plan pinned by its hash; a difference prints the plan.
#[test]
fn plans_are_pinned() {
    let mut got = Vec::new();
    for (name, r) in cases() {
        let line = match &r {
            Ok(json) => format!("{name}: {}", sha(json.as_bytes())),
            Err(e) => format!("{name}: error {e}"),
        };
        got.push(line);
    }
    let want: Vec<&str> = PLAN_PINS.lines().collect();
    for (i, g) in got.iter().enumerate() {
        if want.get(i) != Some(&g.as_str()) {
            let json = cases().into_iter().nth(i).map(|(_, r)| r);
            panic!("case {i} differs:\n{g}\n{json:?}\nall:\n{}", got.join("\n"));
        }
    }
    assert_eq!(got.len(), want.len(), "all:\n{}", got.join("\n"));
}

const PLAN_PINS: &str = "\
default: 8c2431dff4a5a88f20eb86d9ec39b07829cb840d9ebafd506b3431bb56f3515a
drop frames: c81a3c559368629533d9f2a2f62020b99c079897a9ad9bd1bd6772c8b92da026
balanced: ae6dd95758cf3a6bf8038a68ffe4f7ad3a37b148f0b6107c8f58fd1a254a9e70
tight caps: 47ec25b1060052fcdb71ed0707546815e698d9f1cd7ffbb3769766901b7beef1
few negatives: c32fcdd0356851c39f37e1f9e216f27969afe3d560deaf5c7f04ab41aa600ce8
floor restored: 7e6459bd63d92259d45f34f654891c9d1256d731a051ec750ef1a749a65fa696
hinted and frozen: 0cb606a8326256e8d4356620366e2fded3f7b369d4c00270866b9d03389b5183
frozen, no hint: 933cfc2ac2adeb68e4b1aa2b4e4f97f4cdea99b45d441125b65bec22423c7109
one camera day: f8b897bcb282359ad598273eb4a77d5e9e7c70e4bb82b93c2084398c251e71e5
with public: 66df917896d53f81956079bf020e62e58bd472aaf5a7e1371f8c88763bd30c9b
duplicates: db1e3e6edc9a3e2776a99aea44d84f576f1529005512b685ec3a8234d8af8bde
renamed, nothing held out: abd46685b162d3f946b9c28a3b1b3a4c1280bdbfd915e3db781936088edee857
a class only held out: 01187f03113903d3b89174dc1c2dffab06f3ed23b4d700175bf731463a07ff56
a class only on held-out days: d40586e3ecbfb1b04e3c3368112d1673538990728367efe7f6e570d575d1a073
a class only in the held-out split: 6c6dbd23a953e6703fd6ac02e8fd063a418d33ae625f8bbe7f96436295e87d3c
only cars, the rest dropped: e68d500445a63f1521f3e184f575f2fd7557ca138a51905d5468bc6a5140b0db
no positives: error no training frame has a box of the classes [\"bird\"] (after the held-out split, the scene filter, the frames with unsure or too-small boxes and the excluded-class rule); review more frames with objects
duplicate classes: error the class list has duplicate or empty names: [\"car\", \"car\"]
";

/// The cases reach every step of the plan, so the pins cover them.
#[test]
fn the_cases_reach_every_step() {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    let mut warned = String::new();
    for (_, r) in cases() {
        let Ok(json) = r else { continue };
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        for k in [
            "duplicates_dropped",
            "frozen_moved",
            "scene_filtered",
            "excluded_frames",
            "ignore_frames",
            "ignore_valid_day",
            "parked_capped",
            "static_capped",
            "negatives_dropped",
            "negatives_trimmed",
            "balance_dropped",
            "public_frames",
        ] {
            if v["stats"][k].as_u64().is_some_and(|n| n > 0) {
                seen.insert(k);
            }
        }
        if v["stats"]["floor_restored"]
            .as_object()
            .is_some_and(|m| !m.is_empty())
        {
            seen.insert("floor_restored");
        }
        warned.push_str(&v["warnings"].to_string());
    }
    assert_eq!(seen.len(), 13, "{seen:?}");
    for w in [
        "only in the held-out split",
        "only in the held-out frames (",
        "no frame has a box of it",
        "every frame is from one camera on one day",
        "no frame is held out",
        "frame(s) without objects for",
        "camera day(s) held out for good",
        "parked-spot boxes were left out",
        "for boxes of classes this run does not train",
        "have no scene tag",
        "are in more than one of the ticked exports",
    ] {
        assert!(warned.contains(w), "no case warns {w:?}");
    }
}

/// A written dataset's files, pinned by hash (`dataset.json` without its time, the folder's
/// own path as `<OUT>`, the sources' as `<SRC>`).
#[test]
fn a_written_dataset_is_pinned() {
    let t = tmp("pins-write");
    let src_dir = t.0.join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    let mut s = hinted(120);
    s.dir = src_dir.clone();
    for f in &mut s.frames {
        let name = f.image.file_name().unwrap().to_owned();
        f.image = src_dir.join(name);
        std::fs::write(&f.image, f.image.to_string_lossy().as_bytes()).unwrap();
    }
    // One held-out frame of no readable size.
    if let Some(f) = s
        .frames
        .iter_mut()
        .find(|f| f.split == SplitHint::Heldout && !f.boxes.is_empty())
    {
        f.size = None;
    }
    let sources = [s];
    let opts = Options {
        imgsz: Some(640),
        ..Options::default()
    };
    let p = plan(&sources, &opts).unwrap();
    let out = t.0.join("data");
    let prepared = write(&sources, &p, &opts, &out).unwrap();
    let n = |text: &str| -> String {
        let abs = std::path::absolute(&out).unwrap_or_else(|_| out.clone());
        let json_abs = serde_json::to_string(&abs.to_string_lossy()).unwrap();
        let json_src = serde_json::to_string(&src_dir.to_string_lossy()).unwrap();
        let json_out = serde_json::to_string(&out.to_string_lossy()).unwrap();
        text.replace(json_abs.trim_matches('"'), "<OUT>")
            .replace(json_out.trim_matches('"'), "<OUT>")
            .replace(json_src.trim_matches('"'), "<SRC>")
            .replace("\\\\", "/")
    };
    let mut got = Vec::new();
    let mut labels: BTreeMap<&str, String> = BTreeMap::new();
    let mut files: Vec<PathBuf> = Vec::new();
    let mut stack = vec![out.clone()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else {
                files.push(p);
            }
        }
    }
    files.sort();
    for f in files {
        let rel: Vec<String> = f
            .strip_prefix(&out)
            .unwrap()
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        let rel = rel.join("/");
        if rel.starts_with("images/") || rel.starts_with("heldout/images/") {
            continue;
        }
        let text = std::fs::read_to_string(&f).unwrap();
        let text = match rel.as_str() {
            "dataset.json" => {
                let mut v: serde_json::Value = serde_json::from_str(&text).unwrap();
                v["created"] = serde_json::Value::Null;
                n(&serde_json::to_string_pretty(&v).unwrap())
            }
            // The record of file identities (sizes, times) is the reuse tests' business.
            reuse::PREPARED_FILES_FILE => continue,
            _ => n(&text),
        };
        match ["labels/", "heldout/labels/"]
            .iter()
            .find(|d| rel.starts_with(**d))
        {
            Some(d) => {
                let all = labels.entry(*d).or_default();
                all.push_str(&format!("{rel}\n{text}\n"));
            }
            None => got.push(format!("{rel}: {}", sha(text.as_bytes()))),
        }
    }
    for (d, all) in &labels {
        got.push(format!(
            "{d}: {} files, {}",
            all.matches(".txt\n").count(),
            sha(all.as_bytes())
        ));
    }
    got.push(format!(
        "prepared: {} {} {} {} {} {:?} {}",
        prepared.train_images,
        prepared.train_frames,
        prepared.valid_images,
        prepared.heldout_images,
        prepared.heldout_fingerprint,
        prepared.valid_groups,
        prepared.warnings.len()
    ));
    let images = std::fs::read_dir(out.join("images")).unwrap().count();
    let held = std::fs::read_dir(out.join("heldout").join("images"))
        .unwrap()
        .count();
    got.push(format!("images: {images}, held-out images: {held}"));
    let got = got.join("\n");
    assert_eq!(got, WRITE_PINS.trim_end(), "all:\n{got}");
}

const WRITE_PINS: &str = "\
classes.txt: 4d412ee666f6b2614149853855eb6182614598a7c0f563077df1e5efb40865d3
data.yaml: f7c35738203f10a5aa261e1410bb7fd16584aa5c3f3f5372df560a09555906de
dataset.json: c6d85eb3579b5d11ac8d0587d3f4ca4abe43d66f2389c74acc18f2c3b61df74c
heldout/classes.txt: 4d412ee666f6b2614149853855eb6182614598a7c0f563077df1e5efb40865d3
heldout/manifest.json: 1b2b988fa25e8d72a245f4f753a84893c62cfb65ee661dcb3b8d78213f2ac335
image_sources.json: 2e3cbb95ae487ade1437aa645db5ea8f2dad2ed389f036b32ac6e1cef4a8e26c
train.coco.json: 541aed8381f0e862e4d082f7bff8ee7142200902d1073d4f29353270477c6b0e
train.txt: 6d8080ff1c1cee0a188424ebff993f5cad334c107f838806958ece889a4f56af
valid.coco.json: 78ade301da75e7d6ae9619726c624ffb2d2924cc88517d16862e27ddbec9d5b1
valid.txt: de9a1711c946bb258b23e6dfa30b37ba6a5fc4302e3d93d1b28b90bf7e5ffe2c
heldout/labels/: 11 files, bbbc8f56a793b974ee9f984c60c4e856d84bcf2b474f2a9c7e247f0581031b7e
labels/: 98 files, db314197dbedd8dcce7bfd0b6d21220fc05841be4aaef640bb8b38755742bc62
prepared: 70 70 28 12 f95edf557e6763634ca96d6b3a4a28b5d7dfdcc12c09a20fbc6dde9886a0b998 [\"drive/2026-10-01\", \"drive/2026-10-06\", \"gate/2026-10-02\", \"patio/2026-10-03\", \"patio/2026-10-05\"] 7
images: 98, held-out images: 11
";
