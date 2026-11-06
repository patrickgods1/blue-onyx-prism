use super::*;
use crate::benchmark::metrics::{Counts, ThresholdPoint};
use crate::benchmark::report::test_support::model;
use crate::benchmark::threshold::{DevicePick, Pick, ThresholdAdvice};
use crate::model::filters::OverlapRule;
use crate::static_memory::StaticSuppression;
use crate::verify::VerifyConfig;

fn names(l: &[&str]) -> Vec<String> {
    l.iter().map(|s| (*s).to_string()).collect()
}

fn entry_named(name: &str) -> ModelConfig {
    ModelConfig {
        name: Some(name.into()),
        path: format!("models/{name}.onnx").into(),
        ..Default::default()
    }
}

/// `old` (the default, tuned), `new` and a third model the settings may name.
fn config() -> Config {
    let mut old = entry_named("old");
    old.confidence_threshold = Some(0.6);
    old.device_thresholds = BTreeMap::from([("openvino:cpu".to_string(), 0.7)]);
    old.object_filter = Some(names(&["person", "car", "truck"]));
    old.night_model = Some("night".into());
    old.night_threshold = Some(0.35);
    old.night_confidence_threshold = Some(0.65);
    old.label_thresholds = Some(BTreeMap::from([
        ("car".to_string(), 0.5),
        ("truck".to_string(), 0.55),
    ]));
    old.person_aspect = Some([0.2, 0.9]);
    old.min_box_area_frac = Some(0.001);
    old.suppress_overlaps = Some(vec![
        OverlapRule {
            drop: "truck".into(),
            if_overlaps: "car".into(),
            iou: 0.5,
            night_only: false,
        },
        OverlapRule {
            drop: "person".into(),
            if_overlaps: "car".into(),
            iou: 0.5,
            night_only: true,
        },
    ]);
    old.static_suppression = Some(StaticSuppression {
        enabled: true,
        ..Default::default()
    });
    old.verify = Some(VerifyConfig::with_model("night"));
    Config {
        default_model: Some("old".into()),
        models: vec![old, entry_named("new"), entry_named("night")],
        ..Config::default()
    }
}

const NEW_CLASSES: [&str; 2] = ["person", "car"];

fn switch_opts(c: &Config, copy: bool) -> SwitchOptions {
    SwitchOptions {
        from: current_default(c),
        copy,
        classes: names(&NEW_CLASSES),
        at: "2026-11-05T10:00:00Z".into(),
        ..Default::default()
    }
}

fn get<'a>(c: &'a Config, name: &str) -> &'a ModelConfig {
    entry(&c.models, name).unwrap()
}

#[test]
fn copy_lists_what_carried_over_and_what_stayed() {
    let c = config();
    let mut to = entry_named("new");
    to.device = Some("ort:cpu".into());
    let r = copy_settings(get(&c, "old"), &mut to, &names(&NEW_CLASSES), &c.models);
    // Thresholds, filters, night routing, verifier, static suppression travel.
    assert_eq!(to.confidence_threshold, Some(0.6));
    assert_eq!(to.night_model.as_deref(), Some("night"));
    assert_eq!(to.night_threshold, Some(0.35));
    assert_eq!(to.night_confidence_threshold, Some(0.65));
    assert_eq!(to.person_aspect, Some([0.2, 0.9]));
    assert_eq!(to.min_box_area_frac, Some(0.001));
    assert!(to.static_suppression.as_ref().unwrap().enabled);
    assert_eq!(to.verify.as_ref().unwrap().model, "night");
    // What names a class it lacks is narrowed.
    assert_eq!(to.object_filter, Some(names(&["person", "car"])));
    assert_eq!(
        to.label_thresholds,
        Some(BTreeMap::from([("car".to_string(), 0.5)]))
    );
    let rules = to.suppress_overlaps.as_ref().unwrap();
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].drop, "person");
    // Classes, device and the old model's measurements stay.
    assert_eq!(to.device.as_deref(), Some("ort:cpu"));
    assert!(to.classes.is_none() && to.device_thresholds.is_empty());
    let label = |l: &[Copied], x: &str| l.iter().any(|c| c.label == x);
    for x in [
        "Confidence threshold",
        "Object filter",
        "Night model",
        "Night threshold (same model)",
        "Verifier",
        "Static-box suppression",
    ] {
        assert!(label(&r.copied, x), "{x}");
    }
    let no = |x: &str| r.not_copied.iter().find(|n| n.label == x);
    assert!(no("Classes").is_some() && no("Device").is_some());
    assert!(no("Confidence per device").unwrap().why.contains("old"));
    assert!(no("Object filter entries").unwrap().why.contains("truck"));
    assert!(no("Label thresholds").unwrap().why.contains("truck"));
    assert!(no("Overlap rules").is_some());
}

#[test]
fn copy_leaves_out_what_makes_no_sense_for_the_new_model() {
    let mut c = config();
    // No shared class at all: nothing that names one travels.
    let mut to = entry_named("new");
    let cats = names(&["cat"]);
    let r = copy_settings(get(&c, "old"), &mut to, &cats, &c.models);
    assert_eq!(
        to.object_filter, None,
        "an empty filter would mean every class"
    );
    assert_eq!(to.person_aspect, None);
    assert_eq!(to.verify, None);
    assert!(
        r.not_copied
            .iter()
            .any(|n| n.label == "Object filter" && n.why.contains("none"))
    );
    assert!(r.not_copied.iter().any(|n| n.label == "Person aspect"));
    // Pointing the model at itself.
    c.models[0].night_model = Some("new".into());
    c.models[0].verify = Some(VerifyConfig::with_model("New"));
    let mut to = entry_named("new");
    let r = copy_settings(get(&c, "old"), &mut to, &names(&NEW_CLASSES), &c.models);
    assert_eq!((to.night_model.clone(), to.verify.clone()), (None, None));
    assert_eq!(
        r.not_copied
            .iter()
            .filter(|n| n.why.contains("itself"))
            .count(),
        2
    );
    // A night model of another entry cannot get one itself: the config would not validate.
    let mut to = entry_named("new");
    c.models[0].night_model = Some("night".into());
    c.models[2].night_model = Some("new".into());
    let r = copy_settings(get(&c, "old"), &mut to, &names(&NEW_CLASSES), &c.models);
    assert_eq!(to.night_model, None);
    assert!(r.not_copied.iter().any(|n| n.why.contains("do not chain")));
    // A mapped spelling is a class it has, in its own spelling.
    let mut c = config();
    c.models[0].object_filter = Some(names(&["motorbike"]));
    let mut to = entry_named("new");
    copy_settings(get(&c, "old"), &mut to, &names(&["motorcycle"]), &c.models);
    assert_eq!(to.object_filter, Some(names(&["motorcycle"])));
}

fn side<'a>(classes: &'a [String], device: &'a str) -> Side<'a> {
    Side {
        classes,
        unknown: None,
        device,
    }
}

#[test]
fn the_diff_names_what_changes_on_each_outcome() {
    let c = config();
    let (old_c, new_c) = (names(&["person", "car", "truck"]), names(&NEW_CLASSES));
    let d = diff(
        &c,
        get(&c, "old"),
        get(&c, "new"),
        side(&old_c, "openvino:cpu"),
        side(&new_c, "ort:cuda:0"),
    );
    assert_eq!((d.current.as_str(), d.new.as_str()), ("old", "new"));
    let row = |k: &str| d.rows.iter().find(|r| r.key == k).unwrap();
    let t = row("confidence_threshold");
    assert_eq!(t.current, "0.70 (profile for openvino:cpu)");
    assert_eq!(t.new, "0.50 (global default)");
    assert_eq!(t.with_copy, "0.60 (this model's own)");
    let n = row("night_model");
    assert_eq!(
        (n.current.as_str(), n.new.as_str(), n.with_copy.as_str()),
        ("night", "none", "night")
    );
    assert_eq!(row("night_threshold").new, "off");
    assert_eq!(row("object_filter").new, "every class of the model");
    assert_eq!(row("verify").current, "night checks person at night");
    assert!(row("static_suppression").current.starts_with("on ("));
    assert_eq!(row("static_suppression").new, "off");
    assert_eq!(row("min_box_area_frac").with_copy, "0.10% of the image");
    let dev = row("device");
    assert_eq!(
        (
            dev.current.as_str(),
            dev.new.as_str(),
            dev.with_copy.as_str()
        ),
        ("openvino:cpu", "ort:cuda:0", "ort:cuda:0")
    );
    // Settings equal on both sides are not rows; the threshold always is (see `diff`).
    assert!(d.rows.iter().all(|r| r.key != "preprocess"));
    let same = diff(
        &c,
        get(&c, "new"),
        get(&c, "night"),
        side(&new_c, "ort:cpu"),
        side(&new_c, "ort:cpu"),
    );
    assert_eq!(same.rows.len(), 1);
    assert_eq!(same.rows[0].key, "confidence_threshold");
    assert_eq!(
        d.copy
            .copied
            .iter()
            .filter(|x| x.label == "Verifier")
            .count(),
        1
    );
}

#[test]
fn a_lost_class_is_flagged() {
    let mut c = config();
    let (old_c, new_c) = (
        names(&["person", "car", "truck", "dog"]),
        names(&["person", "car", "ghost"]),
    );
    let view = |c: &Config| {
        diff(
            c,
            get(c, "old"),
            get(c, "new"),
            side(&old_c, "ort:cpu"),
            side(&new_c, "ort:cpu"),
        )
        .classes
    };
    // `dog` is filtered out today: never reported, so nothing is lost there.
    let d = view(&c);
    assert_eq!(d.lost, names(&["truck"]));
    assert_eq!(d.gained, names(&["ghost"]));
    assert!(
        d.gained_with_copy.is_empty(),
        "the copied filter narrows it"
    );
    assert_eq!(d.unknown, None);
    // The global filter counts when a model has none of its own.
    c.models[0].object_filter = None;
    c.object_filter = names(&["car", "dog"]);
    let d = view(&c);
    assert_eq!(d.lost, names(&["dog"]));
    // A mapped spelling is no loss.
    c.object_filter = vec![];
    let (a, b) = (names(&["motorbike"]), names(&["motorcycle"]));
    let d = diff(
        &c,
        get(&c, "old"),
        get(&c, "new"),
        side(&a, "x"),
        side(&b, "x"),
    )
    .classes;
    assert!(d.lost.is_empty());
    // Unknown classes say so instead of claiming nothing is lost.
    let none: Vec<String> = vec![];
    let d = diff(
        &c,
        get(&c, "old"),
        get(&c, "new"),
        Side {
            classes: &none,
            unknown: Some("old.yaml is missing"),
            device: "x",
        },
        side(&b, "x"),
    )
    .classes;
    assert!(d.lost.is_empty());
    assert_eq!(d.unknown.as_deref(), Some("old.yaml is missing"));
}

#[test]
fn a_switch_records_what_it_changed_and_revert_restores_it() {
    let mut c = config();
    let start = c.clone();
    let opts = switch_opts(&c, true);
    let o = apply_switch(&mut c, "new", opts).unwrap();
    assert_eq!(c.default_model.as_deref(), Some("new"));
    assert_eq!(get(&c, "new").confidence_threshold, Some(0.6));
    assert!(
        o.message("new")
            .contains("Copied: confidence threshold 0.60")
    );
    assert!(!o.already_default);
    let p = c.previous_default.clone().unwrap();
    assert_eq!((p.model.as_str(), p.switched_to.as_str()), ("old", "new"));
    assert_eq!(p.default_model.as_deref(), Some("old"));
    assert!(p.changed.contains(&"confidence_threshold".to_string()));
    assert!(!p.changed.contains(&"enabled".to_string()));
    assert_eq!(p.settings.as_ref().unwrap().confidence_threshold, None);
    // The JSON loads back as it was (a save writes it next to the models).
    let back: Config = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
    assert_eq!(back.previous_default, c.previous_default);

    let r = revert(&mut c, "2026-11-05T11:00:00Z").unwrap();
    assert_eq!((r.to.as_str(), r.from.as_str()), ("old", "new"));
    assert!(r.restored.contains(&"threshold".to_string()));
    assert!(r.message().contains("Back to old"));
    assert_eq!(c.default_model, start.default_model);
    assert_eq!(c.models, start.models);
}

#[test]
fn reverting_twice_loses_nothing() {
    let mut c = config();
    let opts = switch_opts(&c, true);
    apply_switch(&mut c, "new", opts).unwrap();
    let switched = c.clone();
    // An edit after the switch to a field it changed, and to one it did not.
    entry_mut(&mut c.models, "new")
        .unwrap()
        .confidence_threshold = Some(0.42);
    entry_mut(&mut c.models, "new").unwrap().lazy = true;
    let edited = c.clone();

    revert(&mut c, "t1").unwrap();
    assert_eq!(c.default_model.as_deref(), Some("old"));
    assert_eq!(get(&c, "new").confidence_threshold, None);
    assert!(
        get(&c, "new").lazy,
        "a field the switch did not touch stays"
    );
    revert(&mut c, "t2").unwrap();
    assert_eq!(c.default_model, edited.default_model);
    assert_eq!(
        c.models, edited.models,
        "the second revert gives it all back"
    );
    revert(&mut c, "t3").unwrap();
    revert(&mut c, "t4").unwrap();
    assert_eq!(c.models, edited.models);
    assert_ne!(switched.models, edited.models);
    assert_eq!(c.previous_default.as_ref().unwrap().model, "old");
}

#[test]
fn a_model_the_switch_enabled_is_disabled_again_and_comes_back() {
    let mut c = config();
    c.models[1].enabled = false;
    let from = current_default(&c);
    // As `apply_use` does: read, enable, switch.
    let before = get(&c, "new").clone();
    entry_mut(&mut c.models, "new").unwrap().enabled = true;
    let opts = SwitchOptions {
        from,
        before: Some(before),
        ..switch_opts(&c, false)
    };
    apply_switch(&mut c, "new", opts).unwrap();
    let p = c.previous_default.as_ref().unwrap();
    assert_eq!(p.changed, ["enabled"]);
    revert(&mut c, "t1").unwrap();
    assert!(!get(&c, "new").enabled);
    assert_eq!(current_default(&c).as_deref(), Some("old"));
    revert(&mut c, "t2").unwrap();
    assert!(get(&c, "new").enabled, "the redo needs it loaded");
    assert_eq!(current_default(&c).as_deref(), Some("new"));
}

#[test]
fn a_default_left_unset_goes_back_unset() {
    let mut c = config();
    c.default_model = None;
    let opts = switch_opts(&c, false);
    apply_switch(&mut c, "new", opts).unwrap();
    assert_eq!(c.previous_default.as_ref().unwrap().settings, None);
    revert(&mut c, "t").unwrap();
    assert_eq!(c.default_model, None);
    assert_eq!(current_default(&c).as_deref(), Some("old"));
    // Unset would now name another model: the name is written instead.
    let opts = switch_opts(&c, false);
    apply_switch(&mut c, "new", opts).unwrap();
    c.models.swap(0, 1);
    revert(&mut c, "t").unwrap();
    assert_eq!(c.default_model.as_deref(), Some("old"));
}

#[test]
fn revert_after_the_old_model_was_removed_changes_nothing() {
    let mut c = config();
    c.models[1].night_model = None;
    let opts = switch_opts(&c, true);
    apply_switch(&mut c, "new", opts).unwrap();
    c.models.retain(|m| m.effective_name() != "old");
    // The stale record does not stop the config from loading.
    c.validate().unwrap();
    crate::config::validate_config(&c).unwrap();
    let before = c.clone();
    let e = revert(&mut c, "t").unwrap_err().to_string();
    assert!(e.contains("old is no longer in the config"), "{e}");
    assert!(e.contains("Add it again"), "{e}");
    assert_eq!(c, before, "nothing changed");
    // The page offers to drop the record instead of a button that cannot work.
    let v = revert_view(&c).unwrap();
    assert!(v.blocked.unwrap().contains("no longer in the config"));
}

#[test]
fn revert_refuses_a_default_changed_by_hand_and_a_disabled_model() {
    let mut c = config();
    let opts = switch_opts(&c, false);
    apply_switch(&mut c, "new", opts).unwrap();
    // Changed by hand since: Revert is no longer offered, and refused.
    c.default_model = Some("night".into());
    assert_eq!(revert_view(&c), None);
    let before = c.clone();
    let e = revert(&mut c, "t").unwrap_err().to_string();
    assert!(e.contains("now night"), "{e}");
    assert_eq!(c, before);
    // The old model disabled since: it cannot serve.
    c.default_model = Some("new".into());
    c.models[0].enabled = false;
    let before = c.clone();
    let e = revert(&mut c, "t").unwrap_err().to_string();
    assert!(e.contains("old is not enabled"), "{e}");
    assert_eq!(c, before);
    // Nothing recorded.
    let mut plain = config();
    assert!(revert(&mut plain, "t").is_err());
    assert_eq!(revert_view(&plain), None);
}

#[test]
fn switching_to_the_default_writes_no_record() {
    let mut c = config();
    let opts = switch_opts(&c, true);
    apply_switch(&mut c, "new", opts).unwrap();
    let record = c.previous_default.clone();
    let opts = SwitchOptions {
        threshold: Some(Accepted {
            threshold: 0.33,
            device: Some("ort:cpu".into()),
        }),
        ..switch_opts(&c, true)
    };
    let o = apply_switch(&mut c, "new", opts).unwrap();
    assert!(o.already_default);
    assert_eq!(c.previous_default, record, "Revert still means the switch");
    assert_eq!(get(&c, "new").device_thresholds["ort:cpu"], 0.33);
    assert!(o.message("new").contains("threshold was saved"));
    let opts = switch_opts(&c, true);
    let o = apply_switch(&mut c, "new", opts).unwrap();
    assert!(o.message("new").contains("nothing changed"));
}

#[test]
fn an_accepted_threshold_is_part_of_the_switch() {
    let mut c = config();
    let opts = SwitchOptions {
        threshold: Some(Accepted {
            threshold: 0.44,
            device: Some("ort:cpu".into()),
        }),
        ..switch_opts(&c, false)
    };
    apply_switch(&mut c, "new", opts).unwrap();
    assert_eq!(get(&c, "new").device_thresholds["ort:cpu"], 0.44);
    revert(&mut c, "t").unwrap();
    assert!(get(&c, "new").device_thresholds.is_empty());
    revert(&mut c, "t").unwrap();
    // Without a device: the model's own threshold.
    let mut c = config();
    let opts = SwitchOptions {
        threshold: Some(Accepted {
            threshold: 0.45,
            device: None,
        }),
        ..switch_opts(&c, false)
    };
    apply_switch(&mut c, "new", opts).unwrap();
    assert_eq!(get(&c, "new").confidence_threshold, Some(0.45));
}

#[test]
fn the_view_says_what_a_revert_restores() {
    let mut c = config();
    let opts = switch_opts(&c, true);
    apply_switch(&mut c, "new", opts).unwrap();
    let v = revert_view(&c).unwrap();
    assert_eq!((v.to.as_str(), v.from.as_str()), ("old", "new"));
    assert_eq!(v.at, "2026-11-05");
    assert!(v.restores.contains("threshold") && v.restores.contains("night model"));
    assert_eq!(v.blocked, None);
}

#[test]
fn a_different_verifier_needs_a_confirm_naming_it() {
    let mut c = config();
    let target = get(&c, "old").clone();
    // The same model: nothing to ask.
    check_verifier(&c, &target, "Night", None).unwrap();
    let e = check_verifier(&c, &target, "new", None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("night already verifies old"), "{e}");
    assert!(e.contains("confirm"), "{e}");
    // A confirm of what the page showed is not a confirm of something else.
    assert!(check_verifier(&c, &target, "new", Some("other")).is_err());
    check_verifier(&c, &target, "new", Some("night")).unwrap();
    // No verifier, or a disabled one: nothing to replace.
    let plain = get(&c, "new").clone();
    check_verifier(&c, &plain, "night", None).unwrap();
    let mut off = target.clone();
    off.verify.as_mut().unwrap().enabled = false;
    check_verifier(&c, &off, "new", None).unwrap();
    // The global verifier counts for a model without its own.
    c.verify = Some(VerifyConfig::with_model("night"));
    assert!(check_verifier(&c, &plain, "new", None).is_err());
    assert_eq!(verifiers(&c)["new"], "night");
}

fn point(t: f32, precision: f64, recall: f64) -> ThresholdPoint {
    ThresholdPoint {
        threshold: t,
        counts: Counts {
            tp: 1,
            ..Default::default()
        },
        precision: Some(precision),
        recall: Some(recall),
        f1: Some(0.5),
        f2: Some(0.5),
        fp_per_image: 0.0,
        frame: None,
    }
}

fn pick(t: f32, precision: f64, recall: f64) -> Pick {
    Pick {
        objective: "f1".into(),
        threshold: t,
        point: point(t, precision, recall),
        met: true,
        note: None,
        exact_threshold: None,
        exact_score: None,
        plateau: None,
    }
}

fn advised(datasets: &[&str]) -> crate::benchmark::report::ModelResult {
    let mut m = model(vec![], None);
    m.model = "new".into();
    m.images = 251;
    m.datasets = names(datasets);
    m.threshold = Some(ThresholdAdvice {
        objective: "f1".into(),
        device: "ort:cuda:0".into(),
        relative: false,
        configured: point(0.5, 0.8, 0.8),
        best: Some(pick(0.42, 0.88, 0.91)),
        alternatives: vec![],
        curve: vec![],
        exact: vec![],
        by_dataset: vec![],
        by_tag: vec![],
        per_class: vec![],
        by_device: vec![DevicePick {
            device: "ort:cuda:0".into(),
            best: Some(pick(0.42, 0.88, 0.91)),
        }],
        roc: None,
        now: None,
    });
    m
}

#[test]
fn the_suggestion_comes_from_the_heldout_result_in_words() {
    let r = advised(&["heldout-new"]);
    let s = suggest(&r, "new", Some("ort:cuda:0")).unwrap();
    assert!((s.threshold - 0.42).abs() < 1e-6);
    assert_eq!(s.device.as_deref(), Some("ort:cuda:0"));
    assert!(s.words.contains("At 0.42"), "{}", s.words);
    assert!(
        s.words.contains("found 91%") && s.words.contains("88% of what"),
        "{}",
        s.words
    );
    assert!(s.words.contains("251 frames"), "{}", s.words);
    assert!(s.notes.is_empty());
    // Another device: the model's own threshold, and it says so.
    let s = suggest(&r, "new", Some("ort:cpu")).unwrap();
    assert_eq!(s.device, None);
    assert!(s.notes[0].contains("not on ort:cpu"), "{:?}", s.notes);
    // Other sets and its own frames in the figures are named.
    let mut r = advised(&["heldout-new", "nir-night"]);
    r.trained_on_share = Some(0.4);
    let s = suggest(&r, "new", Some("ort:cuda:0")).unwrap();
    assert!(
        s.notes.iter().any(|n| n.contains("nir-night")),
        "{:?}",
        s.notes
    );
    assert!(s.notes.iter().any(|n| n.contains("40%")), "{:?}", s.notes);
}

#[test]
fn no_suggestion_without_heldout_frames_or_a_usable_result() {
    let r = advised(&["export-1"]);
    let e = suggest(&r, "new", None).unwrap_err();
    assert!(e.contains("did not use its held-out frames"), "{e}");
    let mut r = advised(&["heldout-new"]);
    r.threshold.as_mut().unwrap().relative = true;
    assert!(
        suggest(&r, "new", None)
            .unwrap_err()
            .contains("human labels")
    );
    let mut r = advised(&["heldout-new"]);
    r.threshold = None;
    assert!(suggest(&r, "new", None).is_err());
    // A goal it did not reach writes nothing.
    let mut r = advised(&["heldout-new"]);
    let a = r.threshold.as_mut().unwrap();
    a.best.as_mut().unwrap().met = false;
    a.by_device[0].best.as_mut().unwrap().met = false;
    assert!(
        suggest(&r, "new", Some("ort:cuda:0"))
            .unwrap_err()
            .contains("goal")
    );
}
