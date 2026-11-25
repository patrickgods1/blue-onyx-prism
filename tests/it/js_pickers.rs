//! The class and model pickers' pure parts (`ClassPicker.util` in `assets/class-picker.js`,
//! `ModelPicker.util` in `assets/model-picker.js`), loaded with `require` in `node` without a
//! document (dom.js first, which both read at load): what a class list change does to class ids
//! (the Dataset tab warns before ids change; exported labels are numbered by the list), the name
//! rules, the auto-label form's starting classes and quick sets, the "Find missing labels"
//! defaults (twins of `training::autolabel`'s `MAX_LIMIT`, `MISSING_THRESHOLD` and
//! `models::DEFAULT_THRESHOLD`) and the pre-accept score check.
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI).
#![allow(clippy::unwrap_used)]

use crate::js_harness::{asset, have_node, run_node_json};
use blue_onyx_prism::training::autolabel::{MAX_LIMIT, MISSING_THRESHOLD, models};
use serde_json::{Value, json};

/// `process.argv`: this script, dom.js, class-picker.js, model-picker.js, the request file. Prints
/// one JSON answer.
const DRIVER: &str = r#"
const fs = require('fs');
require(process.argv[2]);
const CU = require(process.argv[3]).util;
const MU = require(process.argv[4]).util;
const req = JSON.parse(fs.readFileSync(process.argv[5], 'utf8'));
const change = (a, b) => CU.classChange(a, b);
const cat = {
  classes: ['person', 'car', 'dog', 'sign', 'bird'], hidden: ['sign'], relevant: ['dog', 'person', 'cat'],
  entries: [{name: 'person', category: 'person'}, {name: 'car', category: 'vehicle'}, {name: 'dog', category: 'animal'},
    {name: 'bird', category: 'animal'}],
  models: [{name: 'm1', enabled: true, classes: ['person', 'dog', 'sign']}, {name: 'off', enabled: false, classes: ['car']},
    {name: 'none', enabled: true, classes: []}],
};
const t = req.defaults;
const on = MU.missingDefaults(true, t.threshold, t.limit, '');
const off = MU.missingDefaults(false, on.threshold, on.limit, on.savedLimit);
process.stdout.write(JSON.stringify({
  change: {
    append: change(['a', 'b'], ['a', 'b', 'c']),
    insert: change(['a', 'b'], ['a', 'x', 'b']),
    reorder: change(['a', 'b'], ['b', 'a']),
    remove: change(['a', 'b', 'c'], ['a', 'c']),
    same: change(['a', 'b'], ['a', 'b']),
  },
  valid: ['  person ', '', '   ', 'a'.repeat(64), 'a'.repeat(65), 'bad\u0007name', 'del\u007f', '\u{1F600}'.repeat(64),
    '\u{1F600}'.repeat(65)].map((s) => [CU.validName(s), CU.nameProblem(s)]),
  start: CU.defaultSelection(cat),
  visible: CU.visibleClasses(['person', 'car', 'Dog'], ['dog', 'car']),
  quick: CU.quickSetsFor(cat),
  missing: {on, off,
    kept: MU.missingDefaults(true, '0.4', '500', ''),
    keptOff: MU.missingDefaults(false, '0.6', '700', '200')},
  accept: [['0.3', ''], ['0.3', '  '], ['0.3', '0'], ['0.3', '1.01'], ['0.3', 'abc'], ['0.3', '0.2'], ['0.3', '0.3'],
    ['0.3', '1'], ['', '0.2'], ['0', '0.1']].map((p) => MU.acceptProblem(p[0], p[1])),
}));
"#;

fn node(request: &Value) -> Value {
    run_node_json(
        DRIVER,
        &[
            asset("dom.js").as_os_str(),
            asset("class-picker.js").as_os_str(),
            asset("model-picker.js").as_os_str(),
        ],
        &[("request.json", &serde_json::to_vec(request).unwrap())],
    )
}

/// The Rust defaults the model picker mirrors, as the form's text: the default threshold and limit
/// a "Find missing labels" tick changes.
fn defaults() -> Value {
    json!({
        "threshold": models::DEFAULT_THRESHOLD.to_string(),
        "limit": blue_onyx_prism::training::autolabel::DEFAULT_LIMIT.to_string(),
    })
}

/// Appending keeps every class id; inserting, reordering or removing moves later ids, which the
/// Dataset tab must warn about before saving.
#[test]
fn class_list_changes_say_when_ids_move() {
    if !have_node() {
        return;
    }
    let v = &node(&json!({ "defaults": defaults() }))["change"];
    let ids = |k: &str| v[k]["idsChange"].as_bool().unwrap();
    assert!(!ids("append"), "{}", v["append"]);
    assert_eq!(v["append"]["appended"], json!(["c"]));
    assert_eq!(v["append"]["reordered"], false);
    assert!(
        ids("insert"),
        "a name in the middle moves the ones after it: {}",
        v["insert"]
    );
    assert_eq!(v["insert"]["reordered"], true);
    assert!(ids("reorder"), "{}", v["reorder"]);
    assert!(ids("remove"), "{}", v["remove"]);
    assert_eq!(v["remove"]["removed"], json!(["b"]));
    assert_eq!(
        v["same"],
        json!({"removed": [], "appended": [], "reordered": false, "idsChange": false})
    );
}

/// Class names: trimmed, 1 to 64 code points (as the server counts `chars()`), no control
/// characters; the reason for a refused one.
#[test]
fn class_names_follow_the_server_rules() {
    if !have_node() {
        return;
    }
    let v = &node(&json!({ "defaults": defaults() }))["valid"];
    let a64 = "a".repeat(64);
    let e64 = "\u{1F600}".repeat(64);
    assert_eq!(
        v,
        &json!([
            ["person", ""],
            [null, "Type a class name first."],
            [null, "Type a class name first."],
            [a64, ""],
            [null, "A class name has at most 64 characters."],
            [null, "A class name cannot contain control characters."],
            [null, "A class name cannot contain control characters."],
            [e64, ""],
            [null, "A class name has at most 64 characters."]
        ])
    );
}

/// The auto-label form starts with what the enabled models detect, in list order, without the
/// hidden ones; the quick sets offer the visible list, person + vehicles, animals and each enabled
/// model's visible classes. Hidden classes are compared exactly here (Review ignores case: QA-112
/// left this open), so a hidden `dog` keeps `Dog`.
#[test]
fn starting_classes_and_quick_sets() {
    if !have_node() {
        return;
    }
    let v = node(&json!({ "defaults": defaults() }));
    assert_eq!(
        v["start"],
        json!(["person", "dog", "cat"]),
        "relevant in list order, then the rest of relevant, minus hidden"
    );
    assert_eq!(v["visible"], json!(["person", "Dog"]));
    assert_eq!(
        v["quick"],
        json!([
            {"label": "All classes", "names": ["person", "car", "dog", "bird"]},
            {"label": "Person + vehicles", "names": ["person", "car"]},
            {"label": "Animals", "names": ["dog", "bird"]},
            {"label": "What m1 detects", "names": ["person", "dog"]}
        ])
    );
}

/// "Find missing labels": ticked, the server's default threshold and limit become the
/// missing-only ones (the Rust constants), unticked they come back; values the user changed stay.
#[test]
fn missing_labels_defaults_are_the_rust_ones() {
    if !have_node() {
        return;
    }
    let d = defaults();
    let v = &node(&json!({ "defaults": d }))["missing"];
    assert_eq!(
        v["on"],
        json!({
            "threshold": MISSING_THRESHOLD.to_string(),
            "limit": MAX_LIMIT.to_string(),
            "savedLimit": d["limit"],
        })
    );
    assert_eq!(
        v["off"],
        json!({"threshold": d["threshold"], "limit": d["limit"], "savedLimit": ""}),
        "on then off restores the form"
    );
    assert_eq!(
        v["kept"],
        json!({"threshold": "0.4", "limit": MAX_LIMIT.to_string(), "savedLimit": "500"})
    );
    assert_eq!(
        v["keptOff"],
        json!({"threshold": "0.6", "limit": "700", "savedLimit": ""}),
        "a limit changed while ticked stays"
    );
}

/// The pre-accept score: empty is off; otherwise above 0, at most 1 and at least the threshold
/// (the server refuses the same).
#[test]
fn the_pre_accept_score_check() {
    if !have_node() {
        return;
    }
    let v = &node(&json!({ "defaults": defaults() }))["accept"];
    let range = "Pre-accept score must be between 0 and 1 (or empty for off).";
    assert_eq!(
        v,
        &json!([
            "",
            "",
            range,
            range,
            range,
            "Pre-accept score must be at least the threshold (0.3).",
            "",
            "",
            "",
            ""
        ])
    );
}
