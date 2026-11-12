//! How the Review tab offers the dataset check's suggestions (`assets/review-rules.js`, the page
//! decisions after the twins: `refineCurrent`, `refineLater`, `suggestionKey`, `refineCard`,
//! `refineFocus`, `answerRefine`, `sendsFilter`, `refineForSave`). `assets/review.js` applies these
//! answers to its canvas, card and keys, so they are checked here in Node: which suggestion is
//! current, what `y` and `w` act on, what the card says per kind, what an answer does to the file
//! and the page flags, and that the reviewer's own edits end a suggestion by themselves.
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI), which
//! makes a missing `node` a failure: a skip must not read as a pass there.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use serde_json::{Value, json};
use std::path::PathBuf;
use std::process::Command;

/// Runs each request through `ReviewRules` and prints the answers as one JSON array.
/// `process.argv`: this script, the rules file, the request file.
const DRIVER: &str = r#"
const fs = require('fs');
require(process.argv[2]);
const R = globalThis.ReviewRules;
const req = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const clone = (v) => JSON.parse(JSON.stringify(v));
// What a save strips (assets/review.js `save`).
const saved = (labels) => {
  labels.boxes.forEach((b) => { delete b._checked; delete b._relabelled; delete b._n; });
  return R.refineForSave(labels);
};
const brief = (c) => (c ? { index: c.l.index, k: c.k, n: c.n } : null);
// The page's own edits of a box (assets/review.js `rejectWith`, `tooSmall`, `toggleUnsure`,
// `markEdited` + a move, `relabel`).
const edit = (labels, e) => {
  const b = labels.boxes[e.box];
  if (e.op === 'reject') R.markRejected(b);
  else if (e.op === 'too_small') R.markRejected(b, 'too_small');
  else if (e.op === 'unsure') { b.difficult = true; delete b.reject_reason; b.state = 'accepted'; b._checked = true; }
  else if (e.op === 'sure') delete b.difficult;
  else if (e.op === 'move') { if (!b.orig_bbox) b.orig_bbox = b.bbox.slice(); b.state = 'accepted'; b.bbox = b.bbox.map((v) => v + 5); }
  else if (e.op === 'relabel') { if (!b.model_label) b.model_label = b.label; b.label = e.label; b._relabelled = true; b.state = 'accepted'; }
  else throw new Error('unknown edit ' + e.op);
};
console.log(JSON.stringify(req.map((r) => {
  if (r.op === 'current') {
    // `later_of`: the indices put off with Later, as their keys.
    const later = (r.later_of || []).map((i) => R.refineKey(r.live.find((l) => l.index === i)));
    return brief(R.refineCurrent(r.live, r.overlap, later));
  }
  if (r.op === 'later') {
    // Later pressed `steps` times: the current suggestion's index after each press.
    let later = [];
    const out = [R.refineCurrent(r.live, false, later).l.index];
    for (let i = 0; i < r.steps; i++) {
      later = R.refineLater(r.live, false, later);
      out.push(R.refineCurrent(r.live, false, later).l.index);
    }
    return out;
  }
  if (r.op === 'key') return R.suggestionKey(r.key, r.overlap, r.refine ? {} : null);
  if (r.op === 'live') {
    const labels = clone(r.labels);
    (r.edits || []).forEach((e) => edit(labels, e));
    return R.liveRefine(labels, r.classes, r.hidden).map((l) => ({ index: l.index, kind: l.kind, target_box: l.target_box, replaces: l.replaces }));
  }
  if (r.op === 'cards') {
    return R.liveRefine(r.labels, r.classes, r.hidden).map((l) => R.refineCard(l, r.labels.boxes, r.touch));
  }
  if (r.op === 'focus') {
    return R.liveRefine(r.labels, r.classes, r.hidden).map((l) => R.refineFocus(l, r.labels.boxes));
  }
  if (r.op === 'answer') {
    const labels = clone(r.labels);
    const before = JSON.stringify(labels);   // the page's undo snapshot
    const l = R.liveRefine(labels, r.classes, r.hidden).find((x) => x.index === r.index);
    const res = R.answerRefine(labels, l, r.yes, r.classes, r.hidden);
    // The twin alone, for comparison (what reaches the file must be the same).
    const plain = clone(r.labels);
    if (r.yes) R.acceptRefine(plain, r.index, r.classes, r.hidden); else R.dismissRefine(plain, r.index);
    const flags = labels.boxes.map((b) => ({ checked: !!b._checked, relabelled: !!b._relabelled }));
    return {
      ok: res.ok,
      box: res.box,
      selected: labels.boxes[res.box] || null,
      flags: flags,
      live: R.liveRefine(labels, r.classes, r.hidden).map((x) => x.index),
      undone: R.liveRefine(JSON.parse(before), r.classes, r.hidden).length,
      same_as_twin: JSON.stringify(saved(clone(labels))) === JSON.stringify(plain),
      body: saved(clone(labels)),
    };
  }
  if (r.op === 'answer_any') {
    // acceptRefine of an index whether it is live or not: refused = the file unchanged.
    const labels = clone(r.labels);
    const ok = R.acceptRefine(labels, r.index, r.classes, r.hidden);
    return { ok: ok, same: JSON.stringify(labels) === JSON.stringify(r.labels) };
  }
  if (r.op === 'overlap_first') {
    // An overlap suggestion on screen holds the dataset check's back until it is answered.
    const labels = clone(r.labels);
    R.markServed(labels);
    const cur = { labels: labels, overlap_live_boxes: r.overlap_live_boxes };
    const sugs = () => R.liveSuggestions(cur, R.dupGroups(labels.boxes, r.classes, r.hidden));
    const live = () => R.liveRefine(labels, r.classes, r.hidden);
    const before = { overlaps: sugs().length, current: brief(R.refineCurrent(live(), sugs().length > 0, [])), y: R.suggestionKey('y', sugs().length > 0, R.refineCurrent(live(), sugs().length > 0, [])) };
    R.acceptOverlap(labels, sugs()[0]);
    const after = { overlaps: sugs().length, current: brief(R.refineCurrent(live(), sugs().length > 0, [])), y: R.suggestionKey('y', sugs().length > 0, R.refineCurrent(live(), sugs().length > 0, [])) };
    return { before: before, after: after };
  }
  if (r.op === 'relabel_ends_overlap') {
    const labels = clone(r.labels);
    R.markServed(labels);
    const cur = { labels: labels, overlap_live_boxes: r.overlap_live_boxes };
    const sugs = () => R.liveSuggestions(cur, R.dupGroups(labels.boxes, r.classes, r.hidden)).length;
    const before = sugs();
    const l = R.liveRefine(labels, r.classes, r.hidden).find((x) => x.kind === 'relabel');
    R.answerRefine(labels, l, true, r.classes, r.hidden);
    return [before, sugs()];
  }
  if (r.op === 'sends') return R.sendsFilter(r.name, r.value, r.auto);
  if (r.op === 'for_save') return R.refineForSave(clone(r.labels));
  throw new Error('unknown op ' + r.op);
})));
"#;

/// Whether `node` runs here. Missing: a skip, or with `BOP_REQUIRE_NODE=1` a failure.
fn have_node() -> bool {
    let ok = Command::new("node")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !ok {
        assert!(
            std::env::var("BOP_REQUIRE_NODE").as_deref() != Ok("1"),
            "BOP_REQUIRE_NODE=1 but `node` does not run: install Node or unset the variable"
        );
        println!("skipped: no node");
    }
    ok
}

fn rules_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/review-rules.js")
}

/// Run [`DRIVER`] on `requests`, in a folder of its own, and read its answers.
fn node(requests: &Value) -> Vec<Value> {
    let dir = std::env::temp_dir().join(format!("bop-js-refine-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let (driver, cases) = (dir.join("driver.js"), dir.join("request.json"));
    std::fs::write(&driver, DRIVER).unwrap();
    std::fs::write(&cases, serde_json::to_vec(requests).unwrap()).unwrap();
    let out = Command::new("node")
        .arg(&driver)
        .arg(rules_path())
        .arg(&cases)
        .output()
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        out.status.success(),
        "node failed ({}): {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice::<Value>(&out.stdout)
        .unwrap()
        .as_array()
        .unwrap()
        .clone()
}

fn classes() -> Value {
    json!(["person", "car", "truck", "dog", "cat"])
}

/// A reviewed frame with one suggestion of each kind (file order: verify, add, relabel, tighten;
/// strength order: add 4899, relabel 3849, tighten 2699, verify 1200).
///
/// Boxes: 0 person (relabel target), 1 person rejected low_score near the add, 2 car (verify
/// target), 3 person rejected low_score exactly at the add, 4 person (tighten target).
fn frame() -> Value {
    json!({
        "version": 1,
        "status": "reviewed",
        "boxes": [
            {"label": "person", "bbox": [300, 300, 400, 500], "source": "model", "state": "accepted", "score": 0.8},
            {"label": "person", "bbox": [10, 10, 110, 210], "source": "auto", "state": "rejected", "reason": "low_score", "score": 0.1},
            {"label": "car", "bbox": [500, 0, 700, 100], "source": "model", "state": "accepted", "score": 0.6},
            {"label": "person", "bbox": [11, 11, 111, 211], "source": "auto", "state": "rejected", "reason": "low_score", "score": 0.12},
            {"label": "person", "bbox": [600, 300, 700, 500], "source": "model", "state": "accepted", "score": 0.7}
        ],
        "refine": [
            {"kind": "verify", "target": [500, 0, 700, 100], "target_label": "car", "score": 0.2, "why": "No fold model found this car."},
            {"kind": "add", "bbox": [11, 11, 111, 211], "label": "person", "score": 0.9, "why": "Two fold models found a person here."},
            {"kind": "relabel", "target": [300, 300, 400, 500], "target_label": "person", "label": "dog", "score": 0.85, "why": "The fold model sees a dog."},
            {"kind": "tighten", "target": [600, 300, 700, 500], "target_label": "person", "bbox": [610, 310, 690, 490], "score": 0.7, "why": "A closer box."}
        ]
    })
}

/// A live entry as `liveRefine` gives it (only what the page decisions read).
fn live(index: u64, kind: &str, score: f64) -> Value {
    json!({"index": index, "kind": kind, "score": score, "target": null, "target_label": null, "bbox": [index as f64, 0, 10, 10], "label": "person", "target_box": null, "replaces": []})
}

#[test]
fn the_current_suggestion_is_the_strongest_after_any_overlap() {
    if !have_node() {
        return;
    }
    let list = json!([
        live(0, "verify", 0.9),
        live(1, "add", 0.5),
        live(2, "relabel", 0.95),
        live(3, "add", 0.5),
        live(4, "tighten", 0.99),
    ]);
    let got = node(&json!([
        {"op": "current", "live": list, "overlap": false},
        // An overlap suggestion on screen comes first: none is current.
        {"op": "current", "live": list, "overlap": true},
        {"op": "current", "live": [], "overlap": false},
        // The first of two equal adds put off (Later): the second, place 2.
        {"op": "current", "live": list, "overlap": false, "later_of": [1]},
        // All put off: the strongest again.
        {"op": "current", "live": list, "overlap": false, "later_of": [0, 1, 2, 3, 4]},
    ]));
    // Adds first (band 4), equal scores by index; then relabel, tighten, verify.
    assert_eq!(got[0], json!({"index": 1, "k": 1, "n": 5}));
    assert_eq!(got[1], Value::Null);
    assert_eq!(got[2], Value::Null);
    assert_eq!(got[3], json!({"index": 3, "k": 2, "n": 5}));
    assert_eq!(got[4], json!({"index": 1, "k": 1, "n": 5}));
    // Later walks the strength order and wraps after the last one.
    let walk = node(&json!([{"op": "later", "live": list, "steps": 6}]));
    assert_eq!(walk[0], json!([1, 3, 2, 4, 0, 1, 3]));
}

#[test]
fn y_stays_with_an_overlap_and_w_only_answers_the_check() {
    if !have_node() {
        return;
    }
    let cases = [
        ("y", true, true, "overlap"),
        ("y", true, false, "overlap"),
        ("y", false, true, "accept"),
        // Nothing at all: the overlap handler says there is no suggestion, as it always did.
        ("y", false, false, "overlap"),
        ("w", true, true, "none"),
        ("w", false, true, "dismiss"),
        // No suggestion: `w` does nothing (the page leaves the key alone).
        ("w", false, false, "none"),
        ("a", false, true, "none"),
    ];
    let req: Vec<Value> = cases
        .iter()
        .map(|(k, o, r, _)| json!({"op": "key", "key": k, "overlap": o, "refine": r}))
        .collect();
    let got = node(&Value::Array(req));
    for (c, g) in cases.iter().zip(&got) {
        assert_eq!(g, c.3, "{c:?}");
    }
}

#[test]
fn an_overlap_on_screen_holds_the_check_back_until_answered() {
    if !have_node() {
        return;
    }
    // Two dogs overlapping at IoU 0.6 (an open pair, no stack), the suggestion drops box 1; a
    // verify on box 0.
    let labels = json!({
        "version": 1, "status": "reviewed",
        "boxes": [
            {"label": "dog", "bbox": [0, 0, 100, 100], "source": "model", "state": "accepted"},
            {"label": "dog", "bbox": [0, 0, 60, 100], "source": "model", "state": "accepted"}
        ],
        "overlap_suggestions": [{"a": [0, 0, 100, 100], "b": [0, 0, 60, 100], "verdict": "drop_b"}],
        "refine": [{"kind": "verify", "target": [0, 0, 100, 100], "target_label": "dog", "score": 0.3}]
    });
    let got = node(&json!([{
        "op": "overlap_first", "labels": labels, "classes": classes(), "hidden": [],
        "overlap_live_boxes": [{"index": 0, "a": 0, "b": 1}]
    }]));
    assert_eq!(
        got[0]["before"],
        json!({"overlaps": 1, "current": null, "y": "overlap"})
    );
    assert_eq!(
        got[0]["after"],
        json!({"overlaps": 0, "current": {"index": 0, "k": 1, "n": 1}, "y": "accept"})
    );
}

#[test]
fn the_card_says_what_each_kind_suggests() {
    if !have_node() {
        return;
    }
    let req = |touch: bool| json!({"op": "cards", "labels": frame(), "classes": classes(), "hidden": [], "touch": touch});
    let got = node(&json!([req(false), req(true)]));
    let keys = got[0].as_array().unwrap();
    let touch = got[1].as_array().unwrap();
    // liveRefine is in file order: verify, add, relabel, tighten.
    assert_eq!(keys.len(), 4);
    let verify = &keys[0];
    assert_eq!(
        verify["head"],
        "No model found box 3 (car): is the label right?"
    );
    assert_eq!(verify["yes"], "Keep the label");
    assert_eq!(verify["no"], "Dismiss");
    assert_eq!(verify["no_bad"], false);
    assert_eq!(verify["short"], "Box 3: no model found it");
    assert_eq!(verify["score"], "score 0.20");
    let note = verify["note"].as_str().unwrap();
    assert!(
        note.contains("r rejects box 3")
            && note.contains("Too small or d (unsure) marks it")
            && note.contains("ends the suggestion")
            && note.contains("w dismisses"),
        "{note}"
    );
    // An unsure box ends the suggestion by itself: no "then y" after d.
    assert!(
        !note.contains("then y") && !note.contains("d keeps"),
        "{note}"
    );
    // On a touch screen the note names the buttons, no key.
    let tnote = touch[0]["note"].as_str().unwrap();
    assert!(
        tnote.contains("No, reject")
            && tnote.contains("Too small or Unsure marks it")
            && tnote.contains("ends the suggestion")
            && !tnote.contains("then Keep the label")
            && !tnote.contains(" r "),
        "{tnote}"
    );

    let add = &keys[1];
    assert_eq!(add["head"], "Missing box: is the dashed box a person?");
    assert_eq!(add["yes"], "Accept: add this box");
    assert_eq!(add["no"], "Not an object");
    assert_eq!(add["no_bad"], true);
    assert_eq!(add["short"], "Missing person?");
    assert_eq!(add["score"], "score 0.90");
    // It takes the place of the two rejects nobody judged there.
    assert_eq!(
        add["note"],
        "Accepting puts it in the place of 2 rejected boxes here that nobody judged."
    );

    let relabel = &keys[2];
    assert_eq!(relabel["head"], "Box 1 is labelled person: is it a dog?");
    assert_eq!(relabel["yes"], "Accept: it is a dog");
    assert_eq!(relabel["no"], "Keep person");
    assert_eq!(relabel["note"], "");

    let tighten = &keys[3];
    assert_eq!(
        tighten["head"],
        "Box 5 (person): take the tighter box (dashed)?"
    );
    assert_eq!(tighten["yes"], "Accept the tighter box");
    assert_eq!(tighten["no"], "Keep my box");
    assert_eq!(tighten["short"], "Box 5: tighter box?");
}

#[test]
fn the_frame_shows_a_proposed_box_or_marks_the_box() {
    if !have_node() {
        return;
    }
    let got =
        node(&json!([{"op": "focus", "labels": frame(), "classes": classes(), "hidden": []}]));
    let f = got[0].as_array().unwrap();
    // verify: box 2 marked, no ghost, framed alone.
    assert_eq!(
        f[0],
        json!({"ghost": null, "mark": 2, "focus": [500, 0, 700, 100]})
    );
    // add: a ghost, no box selected, framed on the ghost.
    assert_eq!(
        f[1],
        json!({"ghost": [11, 11, 111, 211], "mark": -1, "focus": [11, 11, 111, 211]})
    );
    // relabel: box 0 marked.
    assert_eq!(
        f[2],
        json!({"ghost": null, "mark": 0, "focus": [300, 300, 400, 500]})
    );
    // tighten: box 4 marked, its ghost, both framed together.
    assert_eq!(
        f[3],
        json!({"ghost": [610, 310, 690, 490], "mark": 4, "focus": [600, 300, 700, 500]})
    );
}

fn answer(index: u64, yes: bool) -> Value {
    json!({"op": "answer", "labels": frame(), "classes": classes(), "hidden": [], "index": index, "yes": yes})
}

#[test]
fn an_answer_changes_the_file_as_the_twins_do_and_sets_the_page_flags() {
    if !have_node() {
        return;
    }
    let got = node(&json!([
        answer(1, true),
        answer(2, true),
        answer(3, true),
        answer(0, true),
        answer(1, false),
        answer(2, false),
        answer(0, false),
    ]));
    for (i, g) in got.iter().enumerate() {
        assert_eq!(g["ok"], true, "{i}: {g}");
        // The same file as acceptRefine / dismissRefine alone, once the page flags are off.
        assert_eq!(g["same_as_twin"], true, "{i}: {g}");
        // The undo snapshot (JSON) brings every suggestion back.
        assert_eq!(g["undone"], 4, "{i}");
        // Nothing private reaches the save body.
        let body = serde_json::to_string(&g["body"]).unwrap();
        assert!(!body.contains("\"_"), "{i}: {body}");
    }
    // add yes: the new box replaces the closest reject (box 3), the other (box 1) goes, so it is
    // now box 2 (0-based), selected, an accepted auto box marked looked at; the add leaves the
    // list (relabel, tighten, verify stay).
    let add = &got[0];
    assert_eq!(add["box"], 2);
    assert_eq!(add["selected"]["reason"], "refine");
    assert_eq!(add["selected"]["state"], "accepted");
    assert_eq!(add["selected"]["source"], "auto");
    assert_eq!(add["selected"]["label"], "person");
    assert_eq!(add["selected"]["bbox"], json!([11, 11, 111, 211]));
    assert_eq!(
        add["flags"][2],
        json!({"checked": true, "relabelled": false})
    );
    assert_eq!(add["body"]["boxes"].as_array().unwrap().len(), 4);
    assert_eq!(add["body"]["refine"].as_array().unwrap().len(), 3);
    // relabel yes: box 0 is a dog, the old class kept, flagged as relabelled like a relabel by
    // hand (an overlap suggestion that would drop it ends).
    let relabel = &got[1];
    assert_eq!(relabel["box"], 0);
    assert_eq!(relabel["selected"]["label"], "dog");
    assert_eq!(relabel["selected"]["model_label"], "person");
    assert_eq!(
        relabel["flags"][0],
        json!({"checked": true, "relabelled": true})
    );
    // tighten yes: box 4 takes the proposal, keeping its original.
    let tighten = &got[2];
    assert_eq!(tighten["box"], 4);
    assert_eq!(tighten["selected"]["bbox"], json!([610, 310, 690, 490]));
    assert_eq!(
        tighten["selected"]["orig_bbox"],
        json!([600, 300, 700, 500])
    );
    assert_eq!(
        tighten["flags"][4],
        json!({"checked": true, "relabelled": false})
    );
    // verify yes: answered kept, the box looked at, nothing else changes.
    let verify = &got[3];
    assert_eq!(verify["box"], 2);
    assert_eq!(verify["body"]["refine"][0]["answer"], "kept");
    assert_eq!(
        verify["flags"][2],
        json!({"checked": true, "relabelled": false})
    );
    assert_eq!(verify["live"], json!([1, 2, 3]));
    // No: answered dismissed, no box and no flag changes.
    for (g, idx) in [(&got[4], 1), (&got[5], 2), (&got[6], 0)] {
        assert_eq!(g["body"]["refine"][idx]["answer"], "dismissed");
        assert_eq!(g["body"]["boxes"], frame()["boxes"]);
        assert!(
            g["flags"]
                .as_array()
                .unwrap()
                .iter()
                .all(|f| f == &json!({"checked": false, "relabelled": false})),
            "{g}"
        );
        assert!(!g["live"].as_array().unwrap().contains(&json!(idx)));
    }
    // A no to the add leaves the boxes where they are: the selection goes nowhere.
    assert_eq!(got[4]["box"], -1);
}

#[test]
fn the_reviewers_own_edits_end_a_suggestion_by_themselves() {
    if !have_node() {
        return;
    }
    let live = |edits: Value| json!({"op": "live", "labels": frame(), "classes": classes(), "hidden": [], "edits": edits});
    let kinds = |v: &Value| -> Vec<String> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|l| l["kind"].as_str().unwrap().to_string())
            .collect()
    };
    let got = node(&json!([
        live(json!([])),
        // verify on box 2: rejected, or rejected as too small, ends it.
        live(json!([{"op": "reject", "box": 2}])),
        live(json!([{"op": "too_small", "box": 2}])),
        // Unsure makes the box an ignore region, which the check never judges: it ends the
        // verify too (no y needed after d).
        live(json!([{"op": "unsure", "box": 2}])),
        // relabel on box 0: a relabel by hand ends it; tighten on box 4: a move ends it.
        live(json!([{"op": "relabel", "box": 0, "label": "cat"}])),
        live(json!([{"op": "move", "box": 4}])),
        // Unsure ends a relabel and a tighten of its box the same way.
        live(json!([{"op": "unsure", "box": 0}])),
        live(json!([{"op": "unsure", "box": 4}])),
        // Too small on the relabel's box: rejected, so it ends as well.
        live(json!([{"op": "too_small", "box": 0}])),
    ]));
    assert_eq!(kinds(&got[0]), ["verify", "add", "relabel", "tighten"]);
    assert_eq!(kinds(&got[1]), ["add", "relabel", "tighten"]);
    assert_eq!(kinds(&got[2]), ["add", "relabel", "tighten"]);
    assert_eq!(kinds(&got[3]), ["add", "relabel", "tighten"]);
    assert_eq!(kinds(&got[4]), ["verify", "add", "tighten"]);
    assert_eq!(kinds(&got[5]), ["verify", "add", "relabel"]);
    assert_eq!(kinds(&got[6]), ["verify", "add", "tighten"]);
    assert_eq!(kinds(&got[7]), ["verify", "add", "relabel"]);
    assert_eq!(kinds(&got[8]), ["verify", "add", "tighten"]);
}

#[test]
fn an_unsure_box_cannot_be_answered_and_taking_it_back_reopens_the_suggestion() {
    if !have_node() {
        return;
    }
    // The page holds a verify on box 2 as current; the reviewer presses d: a y now finds no live
    // suggestion of that index (acceptRefine refuses, the file stays as it is), and taking the
    // mark back makes it live again (nothing was answered).
    let mut labels = frame();
    labels["boxes"][2]["difficult"] = json!(true);
    let got = node(&json!([
        {"op": "answer_any", "labels": labels, "classes": classes(), "hidden": [], "index": 0},
        {"op": "live", "labels": frame(), "classes": classes(), "hidden": [],
         "edits": [{"op": "unsure", "box": 2}, {"op": "sure", "box": 2}]},
    ]));
    assert_eq!(got[0], json!({"ok": false, "same": true}));
    assert_eq!(
        got[1].as_array().unwrap()[0]["kind"],
        "verify",
        "{}",
        got[1]
    );
}

#[test]
fn a_relabel_accepted_ends_the_overlap_suggestion_on_that_box() {
    if !have_node() {
        return;
    }
    let labels = json!({
        "version": 1, "status": "reviewed",
        "boxes": [
            {"label": "dog", "bbox": [0, 0, 100, 100], "source": "model", "state": "accepted"},
            {"label": "dog", "bbox": [0, 0, 60, 100], "source": "model", "state": "accepted"}
        ],
        "overlap_suggestions": [{"a": [0, 0, 100, 100], "b": [0, 0, 60, 100], "verdict": "drop_b"}],
        "refine": [{"kind": "relabel", "target": [0, 0, 60, 100], "target_label": "dog", "label": "cat", "score": 0.9}]
    });
    let got = node(&json!([{
        "op": "relabel_ends_overlap", "labels": labels, "classes": classes(), "hidden": [],
        "overlap_live_boxes": [{"index": 0, "a": 0, "b": 1}]
    }]));
    assert_eq!(got[0], json!([1, 0]));
}

#[test]
fn refine_kind_goes_to_the_queue_only_with_auto_refine() {
    if !have_node() {
        return;
    }
    let cases = [
        ("refine_kind", "add", "refine", true),
        ("refine_kind", "add", "overlaps", false),
        ("refine_kind", "add", "", false),
        ("refine_kind", "", "refine", false),
        ("auto", "refine", "refine", true),
        ("status", "any", "refine", true),
        ("camera", "", "", false),
    ];
    let req: Vec<Value> = cases
        .iter()
        .map(|(n, v, a, _)| json!({"op": "sends", "name": n, "value": v, "auto": a}))
        .collect();
    let got = node(&Value::Array(req));
    for (c, g) in cases.iter().zip(&got) {
        assert_eq!(g, &json!(c.3), "{c:?}");
    }
}

#[test]
fn a_save_keeps_answers_and_drops_page_fields_of_suggestions() {
    if !have_node() {
        return;
    }
    let labels = json!({
        "version": 1, "status": "reviewed", "boxes": [],
        "refine": [
            {"kind": "verify", "target": [0, 0, 10, 10], "answer": "kept", "_seen": true, "why": "x"},
            {"kind": "add", "bbox": [0, 0, 10, 10], "label": "person", "answer": "dismissed", "_k": 1}
        ]
    });
    let got = node(&json!([
        {"op": "for_save", "labels": labels},
        {"op": "for_save", "labels": {"version": 1, "boxes": []}},
    ]));
    assert_eq!(
        got[0]["refine"],
        json!([
            {"kind": "verify", "target": [0, 0, 10, 10], "answer": "kept", "why": "x"},
            {"kind": "add", "bbox": [0, 0, 10, 10], "label": "person", "answer": "dismissed"}
        ])
    );
    // A file without suggestions goes as it was.
    assert_eq!(got[1], json!({"version": 1, "boxes": []}));
}
