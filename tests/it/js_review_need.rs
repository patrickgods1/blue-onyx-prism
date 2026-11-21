//! The Review tab's need sentences (`assets/review-need.js`): what the details say about why the
//! Recommended order put a frame where it is, from the item's `need` exactly as the server
//! serializes it ([`RecallNeed`], [`CellTally::counts`]), so a renamed field fails here.
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI), which
//! makes a missing `node` a failure: a skip must not read as a pass there.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use blue_onyx_prism::collect::labels::need::{NeedCounts, RecallNeed};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::process::Command;

/// Runs each request through `ReviewNeed` and prints the answers as one JSON array.
/// `process.argv`: this script, the module, the request file.
const DRIVER: &str = r#"
const fs = require('fs');
require(process.argv[2]);
const N = globalThis.ReviewNeed;
const req = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
console.log(JSON.stringify(req.map((r) => r.op === 'counts' ? N.counts(r.c) : N.note(r.item))));
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

/// Run [`DRIVER`] on `requests`, in a folder of its own, and read its answers.
fn node(requests: &Value) -> Vec<String> {
    let dir = std::env::temp_dir().join(format!("bop-js-need-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let (driver, cases) = (dir.join("driver.js"), dir.join("request.json"));
    std::fs::write(&driver, DRIVER).unwrap();
    std::fs::write(&cases, serde_json::to_vec(requests).unwrap()).unwrap();
    let module = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/review-need.js");
    let out = Command::new("node")
        .arg(&driver)
        .arg(module)
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
    serde_json::from_slice(&out.stdout).unwrap()
}

/// A queue item as `GET /v1/review/next` sends it: its `camera`, `scene`, `event_rank` and need.
fn item(n: &RecallNeed, scene: &str, rank: u32) -> Value {
    json!({ "op": "note", "item": {
        "camera": "back", "scene": scene, "event_rank": rank, "need": serde_json::to_value(n).unwrap(),
    }})
}

#[test]
fn the_need_reads_as_the_reasons_for_the_frame() {
    if !have_node() {
        return;
    }
    let raccoon = RecallNeed {
        score: 9000,
        visit_score: 7200,
        class: Some("raccoon".into()),
        boxes: 0.7,
        camera_reviewed: 3.0,
        camera_target: 30.0,
        reviewed: 3.0,
        target: 100.0,
        class_reviewed: 3.25,
        floor: 20,
        recall: Some(0.22),
        heldout_gt: Some(9),
        run: Some("r1".into()),
        below_floor: true,
        catch_up: true,
        missed: true,
        new_region: true,
        expanded: true,
        visit_class: Some("raccoon".into()),
        evidence: 0.35,
        ..RecallNeed::default()
    };
    // A visit a decided fox expanded, and a likely false box of the frame.
    let cat_in_fox_visit = RecallNeed {
        class: Some("cat".into()),
        camera_reviewed: 25.0,
        reviewed: 40.0,
        class_reviewed: 60.0,
        strong: true,
        evidence: 1.0,
        expanded: true,
        visit_class: Some("fox".into()),
        hard_negative: 0.21,
        hard_class: Some("person".into()),
        ..RecallNeed::default()
    };
    let person = RecallNeed {
        class: Some("person".into()),
        camera_reviewed: 1.0,
        reviewed: 150.5,
        class_reviewed: 400.0,
        strong: true,
        evidence: 1.0,
        spot: true,
        new_day: true,
        gap: true,
        kept: Some(1),
        proposed: Some(9),
        recall: Some(0.9),
        heldout_gt: Some(0),
        ..RecallNeed::default()
    };
    let lacking = RecallNeed {
        class: Some("fox".into()),
        camera_reviewed: 0.0,
        class_reviewed: 0.0,
        below_floor: true,
        floor: 20,
        model_lacks: true,
        catch_up: true,
        evidence: 1.0,
        strong: true,
        ..RecallNeed::default()
    };
    let empty = RecallNeed {
        empty_reviewed: Some(1),
        animal_share: Some(0.333),
        ..RecallNeed::default()
    };
    // The top class's own likely false box: not named twice.
    let person_maybe_false = RecallNeed {
        hard_class: Some("cat".into()),
        ..cat_in_fox_visit.clone()
    };
    // A frame with nothing but a likely false box: its hard negative, never "empty".
    let only_false = RecallNeed {
        hard_negative: 0.12,
        hard_class: Some("dog".into()),
        hard_only: true,
        ..RecallNeed::default()
    };
    let counts = NeedCounts {
        boxes: [
            (
                "person".to_string(),
                [("day".to_string(), 8.0), ("night".to_string(), 4.25)]
                    .into_iter()
                    .collect(),
            ),
            (
                "raccoon".to_string(),
                [("ir".to_string(), 3.0)].into_iter().collect(),
            ),
        ]
        .into_iter()
        .collect(),
        empty: [("day".to_string(), 20)].into_iter().collect(),
        floor: 20,
        target: 100.0,
        camera_target: 30.0,
    };
    let got = node(&json!([
        item(&raccoon, "ir", 2),
        item(&raccoon, "ir", 0),
        item(&person, "day", 1),
        item(&lacking, "night", 0),
        item(&empty, "night", 0),
        item(&cat_in_fox_visit, "day", 4),
        item(&person_maybe_false, "day", 0),
        item(&only_false, "night", 0),
        { "op": "note", "item": { "camera": "back", "scene": "day" } },
        { "op": "counts", "c": serde_json::to_value(&counts).unwrap() },
        { "op": "counts", "c": serde_json::to_value(NeedCounts::default()).unwrap() },
    ]));
    assert_eq!(
        got,
        [
            "raccoon · back · IR: 3 boxes on this camera (3.25 in all, floor 20), recall 22% on 9 \
             held-out boxes (lowest on this camera), camera model missed it, new place in view, \
             weak evidence (0.35), frame 3 of a raccoon visit",
            // The first frame of its visit: no visit to speak of.
            "raccoon · back · IR: 3 boxes on this camera (3.25 in all, floor 20), recall 22% on 9 \
             held-out boxes (lowest on this camera), camera model missed it, new place in view, \
             weak evidence (0.35)",
            "person · back · day: 1 box on this camera (150.5 on every camera in this scene), \
             recall 90% (estimated: no held-out box on this camera), lost for a frame of its \
             visit, at a parked spot (counts a quarter), new day, 1 of 9 proposals kept on this \
             camera",
            "fox · back · night: 0 boxes on this camera (0 in all, floor 20), the trained model has \
             no fox",
            "nothing of a wanted class · back · night: 1 empty frame reviewed in this scene, 33% \
             of this camera's reviewed frames here had an animal",
            "cat · back · day: 25 boxes on this camera (40 on every camera in this scene), frame 5 \
             of a fox visit, likely false person: rejecting it teaches the model",
            "cat · back · day: 25 boxes on this camera (40 on every camera in this scene), some \
             may be false: rejecting those teaches the model",
            "likely false dog · back · night: nothing else to label here; rejecting it teaches \
             the model",
            "",
            "person 12.25 (day 8, night 4.25) · raccoon 3 (IR 3) · empty: day 20",
            "none yet",
        ]
    );
}
