//! The camera days held out for good: the pure decision, then the list on disk through real
//! exports of a temporary review folder.

use super::super::export::{
    self, BenchSplit, ExportRequest, Format, Split, SplitConfig, assign_splits,
};
use super::super::{BoxState, LabelBox, LabelFile, Source, Status};
use super::*;
use std::path::PathBuf;

fn g(c: &str, d: &str) -> Group {
    (c.to_string(), d.to_string())
}

/// The hash of a group as a fraction of 2^64 (the default split holds out those under 0.2).
#[allow(clippy::cast_precision_loss, reason = "a fraction of 2^64")]
fn frac(group: &Group) -> f64 {
    group_hash(group) as f64 / 18_446_744_073_709_551_616.0
}

/// `n` days of `camera` in October to December 2026 whose hash `keep` takes, in date order.
fn dates(camera: &str, n: usize, keep: impl Fn(f64) -> bool) -> Vec<Group> {
    let mut out = Vec::new();
    for month in 10..=12 {
        for day in 1..=28 {
            let grp = g(camera, &format!("2026-{month:02}-{day:02}"));
            if keep(frac(&grp)) {
                out.push(grp);
            }
            if out.len() == n {
                return out;
            }
        }
    }
    panic!("not enough dates for {camera}");
}

/// Days the default split trains on.
fn train_days(camera: &str, n: usize) -> Vec<Group> {
    dates(camera, n, |x| x >= f64::from(DEFAULT_HOLDOUT_FRACTION))
}

/// Days the default split holds out.
fn val_days(camera: &str, n: usize) -> Vec<Group> {
    dates(camera, n, |x| x < f64::from(DEFAULT_HOLDOUT_FRACTION))
}

fn set(v: &[Group]) -> BTreeSet<Group> {
    v.iter().cloned().collect()
}

fn by_hash(mut v: Vec<Group>) -> Vec<Group> {
    v.sort_by_key(|x| (group_hash(x), x.clone()));
    v
}

#[test]
fn the_hash_days_are_kept_and_listed() {
    let mut all = val_days("back", 2);
    all.extend(train_days("back", 4));
    all.extend(val_days("drive", 1));
    all.extend(train_days("drive", 5));
    let groups = set(&all);
    let cfg = SplitConfig::default();
    let base = assign_splits(&groups, &cfg);
    let d = decide(&groups, &cfg, &BTreeSet::new(), Some(&BTreeSet::new()));
    for (grp, s) in &base {
        if *s == Split::Val {
            assert_eq!(d.splits[grp], Split::Val, "{grp:?}");
            assert!(d.added.contains(&(grp.clone(), Reason::Hash)), "{grp:?}");
        }
    }
    // drive has one hash day: one more, its lowest-hash training day.
    let drive_next = by_hash(train_days("drive", 5))[0].clone();
    assert_eq!(d.splits[&drive_next], Split::Val);
    assert!(
        d.added
            .contains(&(drive_next.clone(), Reason::CameraMinimum))
    );
    assert_eq!(d.added.len(), 4);
    assert!(d.short.is_empty());
    assert!(d.added.is_sorted());
    // The same input, the same answer.
    assert_eq!(
        d,
        decide(&groups, &cfg, &BTreeSet::new(), Some(&BTreeSet::new()))
    );
}

#[test]
fn a_listed_day_stays_held_out_at_fraction_zero_and_with_named_cameras() {
    let mut all = train_days("back", 3);
    all.extend(train_days("drive", 3));
    let groups = set(&all);
    let listed = set(&all[..1]);
    for cfg in [
        SplitConfig {
            holdout_fraction: 0.0,
            ..SplitConfig::default()
        },
        SplitConfig {
            holdout_cameras: vec!["drive".into()],
            ..SplitConfig::default()
        },
        SplitConfig {
            holdout_days: vec!["2027-01-01".into()],
            ..SplitConfig::default()
        },
        SplitConfig {
            holdout_fraction: 0.5,
            ..SplitConfig::default()
        },
    ] {
        let d = decide(&groups, &cfg, &listed, Some(&BTreeSet::new()));
        assert_eq!(d.splits[&all[0]], Split::Val, "{cfg:?}");
        // Not the default split: nothing added, nobody short.
        assert!(d.added.is_empty(), "{cfg:?}");
        assert!(d.short.is_empty(), "{cfg:?}");
        assert!(!is_default_split(&cfg));
    }
    // Fraction 0 alone: the listed day is all that is held out.
    let zero = SplitConfig {
        holdout_fraction: 0.0,
        ..SplitConfig::default()
    };
    let d = decide(&groups, &zero, &listed, None);
    assert_eq!(
        d.splits.values().filter(|s| **s == Split::Val).count(),
        1,
        "{d:?}"
    );
    assert!(is_default_split(&SplitConfig::default()));
}

#[test]
fn top_ups_take_unseen_days_lowest_hash_first_and_never_the_last_training_day() {
    // nook: five training days, its lowest-hash one seen by a run.
    let nook = by_hash(train_days("nook", 5));
    // alley: two training days, none seen: one may go, the other trains.
    let alley = train_days("alley", 2);
    // porch: three training days, all seen.
    let porch = train_days("porch", 3);
    // back: one held-out day by hash, so the split has one.
    let back = val_days("back", 1);
    // side: one held-out day by hash, two training days, both seen.
    let side_val = val_days("side", 1);
    let side = train_days("side", 2);
    let mut all = nook.clone();
    all.extend(alley.clone());
    all.extend(porch.clone());
    all.extend(back.clone());
    all.extend(side_val.clone());
    all.extend(side.clone());
    let groups = set(&all);
    let mut seen = set(&porch);
    seen.insert(nook[0].clone());
    seen.extend(side.iter().cloned());
    let d = decide(
        &groups,
        &SplitConfig::default(),
        &BTreeSet::new(),
        Some(&seen),
    );
    // nook: the second and third lowest hash, the seen one skipped.
    assert_eq!(d.splits[&nook[0]], Split::Train);
    assert_eq!(d.splits[&nook[1]], Split::Val);
    assert_eq!(d.splits[&nook[2]], Split::Val);
    assert_eq!(d.splits[&nook[3]], Split::Train);
    assert_eq!(d.splits[&nook[4]], Split::Train);
    // alley: one held out, the last training day kept.
    let alley_val: Vec<&Group> = alley
        .iter()
        .filter(|x| d.splits[*x] == Split::Val)
        .collect();
    assert_eq!(alley_val, [&by_hash(alley.clone())[0]]);
    // porch: nothing unseen.
    assert!(porch.iter().all(|x| d.splits[x] == Split::Train));
    let mut want = vec![
        (nook[1].clone(), Reason::CameraMinimum),
        (nook[2].clone(), Reason::CameraMinimum),
        (by_hash(alley.clone())[0].clone(), Reason::CameraMinimum),
        (back[0].clone(), Reason::Hash),
        (side_val[0].clone(), Reason::Hash),
    ];
    want.sort();
    assert_eq!(d.added, want);
    let short: BTreeMap<&str, &Short> = d.short.iter().map(|s| (s.camera.as_str(), s)).collect();
    assert_eq!(short.len(), 4, "{:?}", d.short);
    assert_eq!(short["side"].held, 1);
    assert_eq!(
        short["side"].why,
        "every other reviewed side day was trained on; review a newer day"
    );
    assert_eq!((short["alley"].held, short["alley"].wanted), (1, 2));
    assert_eq!(
        short["alley"].why,
        "its one training day of this export stays in training; review another day of alley"
    );
    assert_eq!(short["porch"].held, 0);
    assert_eq!(
        short["porch"].why,
        "every reviewed porch day was trained on; review a newer day"
    );
    assert_eq!(short["back"].held, 1);
    assert_eq!(
        short["back"].why,
        "only 1 reviewed back day(s), all held out; review more days of back"
    );
}

#[test]
fn without_the_runs_lists_no_day_is_added_for_a_camera() {
    let nook = train_days("nook", 4);
    let back = val_days("back", 1);
    let mut all = nook.clone();
    all.extend(back.clone());
    let groups = set(&all);
    let d = decide(&groups, &SplitConfig::default(), &BTreeSet::new(), None);
    assert_eq!(d.added, [(back[0].clone(), Reason::Hash)]);
    assert!(nook.iter().all(|x| d.splits[x] == Split::Train));
    assert_eq!(d.short.len(), 2);
    assert!(
        d.short.iter().all(|s| s
            .why
            .starts_with("the frame lists of past training runs could not")),
        "{:?}",
        d.short
    );
}

#[test]
fn the_guard_never_moves_a_listed_day_to_train() {
    // Two held-out days by hash: the split's guard trains the higher one. Listed, it stays held
    // out, and the other trains.
    let two = by_hash(val_days("cam", 2));
    let groups = set(&two);
    let cfg = SplitConfig::default();
    assert_eq!(assign_splits(&groups, &cfg)[&two[1]], Split::Train);
    let listed = set(&two[1..]);
    let d = decide(&groups, &cfg, &listed, Some(&BTreeSet::new()));
    assert_eq!(d.splits[&two[1]], Split::Val);
    assert_eq!(d.splits[&two[0]], Split::Train);
    assert!(d.added.is_empty(), "{:?}", d.added);
    // Every day listed: all held out, none moved.
    let d = decide(&groups, &cfg, &groups, Some(&BTreeSet::new()));
    assert!(d.splits.values().all(|s| *s == Split::Val));
    assert!(d.added.is_empty());
    // A listed day that hashes to train, beside one that hashes to val: both held out would
    // leave nothing to train, so the unlisted one trains.
    let mut mixed = train_days("yard", 1);
    mixed.extend(val_days("yard", 1));
    let groups = set(&mixed);
    let d = decide(&groups, &cfg, &set(&mixed[..1]), Some(&BTreeSet::new()));
    assert_eq!(d.splits[&mixed[0]], Split::Val);
    assert_eq!(d.splits[&mixed[1]], Split::Train);
}

// ---------------------------------------------------------------------------------------------
// On disk
// ---------------------------------------------------------------------------------------------

/// A temporary folder, removed on drop.
struct Tmp(PathBuf);

impl Tmp {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("bop-heldout-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&p).unwrap();
        Self(p)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn person() -> LabelBox {
    LabelBox {
        label: "person".into(),
        bbox: [4.0, 4.0, 20.0, 30.0],
        source: Source::Model,
        score: Some(0.9),
        state: BoxState::Accepted,
        origin: None,
        reason: None,
        model_label: None,
        orig_bbox: None,
        difficult: false,
        reject_reason: None,
        auto: Default::default(),
    }
}

/// One reviewed frame with a person on each of `days`.
fn put_days(dir: &Path, days: &[Group]) {
    for (camera, date) in days {
        let day = dir.join("day").join(camera).join(date);
        std::fs::create_dir_all(&day).unwrap();
        image::RgbImage::from_pixel(64, 48, image::Rgb([90, 90, 90]))
            .save(day.join("f.jpg"))
            .unwrap();
        let mut r = crate::collect::record::empty_record();
        r.id = "f".into();
        r.camera = camera.clone();
        r.width = 64;
        r.height = 48;
        std::fs::write(day.join("f.json"), serde_json::to_vec(&r).unwrap()).unwrap();
        let f = LabelFile {
            status: Status::Reviewed,
            boxes: vec![person()],
            ..LabelFile::default()
        };
        super::super::write(&day.join(format!("f{}", super::super::LABELS_SUFFIX)), &f).unwrap();
    }
}

fn classes() -> Vec<String> {
    vec!["person".into()]
}

fn request(out: PathBuf, split: SplitConfig, heldout: Option<HeldoutPolicy>) -> ExportRequest {
    ExportRequest {
        formats: vec![Format::Yolo],
        out_dir: out,
        split,
        benchmark_split: BenchSplit::All,
        include_unreviewed: false,
        hidden_classes: Vec::new(),
        size_floor: None,
        heldout,
    }
}

fn unseen() -> Option<HeldoutPolicy> {
    Some(HeldoutPolicy {
        seen: Some(BTreeSet::new()),
    })
}

/// The groups of an export's report with `split`.
fn groups_of(r: &export::ExportReport, split: Split) -> BTreeSet<Group> {
    r.splits
        .iter()
        .filter(|s| s.split == split)
        .map(|s| (s.camera.clone(), s.date.clone()))
        .collect()
}

#[test]
fn the_first_export_lists_its_picks_and_a_second_only_grows_the_list() {
    let t = Tmp::new("grow");
    let dir = t.0.join("dataset");
    let mut days = val_days("back", 1);
    days.extend(train_days("back", 3));
    days.extend(train_days("drive", 3));
    put_days(&dir, &days);
    assert!(!dir.join(FILE).exists());
    let r = export::run(
        &dir,
        &classes(),
        &request(t.0.join("one"), SplitConfig::default(), unseen()),
    )
    .unwrap();
    let h = r.heldout.clone().unwrap();
    let file = load(&dir).unwrap();
    assert_eq!(file.version, VERSION);
    assert_eq!(h.listed, file.days.len());
    assert_eq!(file.days, h.added);
    // back: its hash day and one more; drive: two more.
    assert_eq!(file.days.len(), 4, "{:?}", file.days);
    assert_eq!(file.groups(), groups_of(&r, Split::Val));
    assert!(h.short.is_empty());
    assert!(!h.custom_split);
    let raw: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join(FILE)).unwrap()).unwrap();
    assert_eq!(raw["version"], 1);
    assert!(
        raw["days"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["reason"] == "hash")
    );
    assert!(
        raw["days"]
            .as_array()
            .unwrap()
            .iter()
            .any(|d| d["reason"] == "camera-minimum")
    );
    assert!(raw["days"][0]["added"].as_str().unwrap().ends_with('Z'));
    let json = serde_json::to_value(&r).unwrap();
    assert_eq!(json["heldout"]["listed"], 4);
    assert!(json["heldout"].get("custom_split").is_none());
    // `export.json` records it too.
    let recorded: serde_json::Value =
        serde_json::from_slice(&std::fs::read(r.out_dir.join("export.json")).unwrap()).unwrap();
    assert_eq!(recorded["heldout"], json["heldout"]);

    // A new camera: the list grows by its days, the earlier ones stay as they were.
    let more = train_days("alley", 3);
    put_days(&dir, &more);
    let r = export::run(
        &dir,
        &classes(),
        &request(t.0.join("two"), SplitConfig::default(), unseen()),
    )
    .unwrap();
    let after = load(&dir).unwrap();
    assert_eq!(&after.days[..4], file.days.as_slice());
    assert_eq!(after.days.len(), 6);
    assert!(after.days[4..].iter().all(|d| d.camera == "alley"));
    assert_eq!(r.heldout.unwrap().added, after.days[4..]);
    // Again: nothing new, the file untouched.
    let bytes = std::fs::read(dir.join(FILE)).unwrap();
    let r = export::run(
        &dir,
        &classes(),
        &request(t.0.join("three"), SplitConfig::default(), unseen()),
    )
    .unwrap();
    assert!(r.heldout.unwrap().added.is_empty());
    assert_eq!(std::fs::read(dir.join(FILE)).unwrap(), bytes);
}

#[test]
fn a_listed_day_is_held_out_by_every_export_of_the_list() {
    let t = Tmp::new("listed");
    let dir = t.0.join("dataset");
    let days = train_days("back", 4);
    put_days(&dir, &days);
    let listed = HeldoutFile {
        version: VERSION,
        days: vec![HeldoutDay {
            camera: days[2].0.clone(),
            date: days[2].1.clone(),
            added: "2026-11-01T00:00:00.000Z".into(),
            reason: Reason::Hash,
        }],
    };
    std::fs::write(dir.join(FILE), serde_json::to_vec_pretty(&listed).unwrap()).unwrap();
    let before = std::fs::read(dir.join(FILE)).unwrap();
    // Other split settings: the listed day held out, nothing written.
    let r = export::run(
        &dir,
        &classes(),
        &request(
            t.0.join("custom"),
            SplitConfig {
                holdout_fraction: 0.0,
                ..SplitConfig::default()
            },
            unseen(),
        ),
    )
    .unwrap();
    assert_eq!(groups_of(&r, Split::Val), set(&days[2..3]));
    let h = r.heldout.unwrap();
    assert!(h.custom_split && h.added.is_empty() && h.listed == 1);
    assert_eq!(std::fs::read(dir.join(FILE)).unwrap(), before);
    // Without the policy (the dataset check's selections): the split alone.
    let r = export::run(
        &dir,
        &classes(),
        &request(
            t.0.join("plain"),
            SplitConfig {
                holdout_fraction: 0.0,
                ..SplitConfig::default()
            },
            None,
        ),
    )
    .unwrap();
    assert!(groups_of(&r, Split::Val).is_empty());
    assert!(r.heldout.is_none());
}

#[test]
fn an_export_of_other_split_settings_creates_no_list() {
    let t = Tmp::new("custom");
    let dir = t.0.join("dataset");
    let mut days = val_days("back", 1);
    days.extend(train_days("back", 3));
    put_days(&dir, &days);
    for (n, split) in [
        SplitConfig {
            holdout_fraction: 0.3,
            ..SplitConfig::default()
        },
        SplitConfig {
            holdout_cameras: vec!["back".into()],
            ..SplitConfig::default()
        },
    ]
    .into_iter()
    .enumerate()
    {
        let r = export::run(
            &dir,
            &classes(),
            &request(t.0.join(format!("out{n}")), split, unseen()),
        )
        .unwrap();
        assert!(r.heldout.unwrap().custom_split);
        assert!(!dir.join(FILE).exists());
    }
}

#[test]
fn a_newer_or_damaged_list_refuses_the_export_and_stays_as_it_is() {
    let t = Tmp::new("refuse");
    let dir = t.0.join("dataset");
    put_days(&dir, &train_days("back", 3));
    for (text, word) in [
        (r#"{"version":2,"days":[]}"#, "newer"),
        ("{\"version\":1,\"days\":[{\"camera\":", "damaged"),
        (r#"{"days":[]}"#, "damaged"),
        (r#"{"version":1,"days":[{"camera":"a"}]}"#, "damaged"),
        ("", "damaged"),
    ] {
        std::fs::write(dir.join(FILE), text).unwrap();
        let out = t.0.join("out");
        for policy in [unseen(), Some(HeldoutPolicy { seen: None })] {
            let e = export::run(
                &dir,
                &classes(),
                &request(out.clone(), SplitConfig::default(), policy),
            )
            .unwrap_err();
            let e = format!("{e:#}");
            assert!(e.contains(word), "{text}: {e}");
            assert!(e.contains(FILE), "{e}");
        }
        assert_eq!(std::fs::read(dir.join(FILE)).unwrap(), text.as_bytes());
        assert!(!out.exists(), "nothing exported");
        // Nothing moved aside beside it.
        let names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains("heldout"))
            .collect();
        assert_eq!(names.len(), 2, "{names:?}");
    }
    // A plain export (no policy) does not read it.
    let r = export::run(
        &dir,
        &classes(),
        &request(t.0.join("plain"), SplitConfig::default(), None),
    );
    assert!(r.is_ok());
}

#[test]
fn an_update_waits_for_a_lock_another_holder_has() {
    let t = Tmp::new("lock");
    let dir = t.0.clone();
    let held = std::fs::File::create(dir.join(LOCK_FILE)).unwrap();
    fs4::FileExt::try_lock(&held).unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let d = dir.clone();
    std::thread::spawn(move || {
        let r = update(&d, |_| {
            Ok((
                vec![HeldoutDay {
                    camera: "back".into(),
                    date: "2026-11-01".into(),
                    added: "2026-11-18T00:00:00.000Z".into(),
                    reason: Reason::Hash,
                }],
                (),
            ))
        });
        tx.send(r.map(|(f, ())| f.days.len()).map_err(|e| e.to_string()))
            .unwrap();
    });
    assert!(
        rx.recv_timeout(Duration::from_millis(300)).is_err(),
        "the update went past a held lock"
    );
    assert!(!dir.join(FILE).exists());
    fs4::FileExt::unlock(&held).unwrap();
    let n = rx.recv_timeout(Duration::from_secs(20)).unwrap().unwrap();
    assert_eq!(n, 1);
    assert_eq!(load(&dir).unwrap().days.len(), 1);
}

#[test]
fn a_lock_held_past_the_wait_refuses_and_writes_nothing() {
    let t = Tmp::new("busy");
    let dir = t.0.clone();
    let held = std::fs::File::create(dir.join(LOCK_FILE)).unwrap();
    fs4::FileExt::try_lock(&held).unwrap();
    let groups = set(&val_days("back", 1));
    let e = apply_with(
        &dir,
        &groups,
        &SplitConfig::default(),
        &HeldoutPolicy::default(),
        SystemTime::now(),
        Duration::from_millis(50),
    )
    .unwrap_err();
    assert!(e.to_string().contains("stayed locked"), "{e}");
    assert!(!dir.join(FILE).exists());
    fs4::FileExt::unlock(&held).unwrap();
}

#[test]
fn a_lock_that_cannot_work_refuses_only_days_to_add() {
    let t = Tmp::new("broken");
    let dir = t.0.clone();
    // Something at the lock's path that cannot be opened as a file.
    std::fs::create_dir(dir.join(LOCK_FILE)).unwrap();
    let back = val_days("back", 1);
    let policy = HeldoutPolicy {
        seen: Some(BTreeSet::new()),
    };
    let e = apply(
        &dir,
        &set(&back),
        &SplitConfig::default(),
        &policy,
        SystemTime::now(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("cannot be"), "{e}");
    assert!(!dir.join(FILE).exists());
    // Nothing to add (other split settings, or every day listed already): it goes on.
    let custom = SplitConfig {
        holdout_fraction: 0.5,
        ..SplitConfig::default()
    };
    let (splits, report) = apply(&dir, &set(&back), &custom, &policy, SystemTime::now()).unwrap();
    assert_eq!(splits.len(), 1);
    assert!(report.added.is_empty());
}

#[test]
fn a_day_listed_twice_counts_once() {
    let t = Tmp::new("twice");
    let day = |added: &str| HeldoutDay {
        camera: "back".into(),
        date: "2026-11-01".into(),
        added: added.into(),
        reason: Reason::Hash,
    };
    let f = HeldoutFile {
        version: VERSION,
        days: vec![
            day("2026-11-01T00:00:00.000Z"),
            day("2026-11-02T00:00:00.000Z"),
        ],
    };
    std::fs::write(t.0.join(FILE), serde_json::to_vec(&f).unwrap()).unwrap();
    let read = load(&t.0).unwrap();
    assert_eq!(read.days, [day("2026-11-01T00:00:00.000Z")]);
    assert_eq!(read.keys(), BTreeSet::from(["back/2026-11-01".to_string()]));
}
