//! The Review tab's finger rules (`assets/review-touch.js`): pinch zoom and pan, the two-tap box,
//! what one finger's drag does, where a tag sits in the zoomed view, which box an overview tap
//! picks and which corner the overview takes. The page applies these answers to its view and boxes
//! (`assets/review.js`), so they are checked here on exact numbers, in Node.
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI), which
//! makes a missing `node` a failure: a skip must not read as a pass there.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use serde_json::{Value, json};
use std::path::PathBuf;
use std::process::Command;

/// Runs each request through `ReviewTouch` and prints the answers as one JSON array.
/// `process.argv`: this script, the module, the request file.
const DRIVER: &str = r#"
const fs = require('fs');
require(process.argv[2]);
const T = globalThis.ReviewTouch;
const req = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
console.log(JSON.stringify(req.map((r) => {
  if (r.op === 'pinch') return T.pinch(r.start, r.now, r.limits);
  if (r.op === 'twoTapRect') return T.twoTapRect(r.a, r.b, r.minBox);
  if (r.op === 'touchIntent') return T.touchIntent(r.s);
  if (r.op === 'tagTop') return T.tagTop(r.bb, r.top, r.bottom, r.th);
  if (r.op === 'pickBox') return T.pickBox(r.boxes, r.x, r.y, r.pad);
  if (r.op === 'overviewCorner') return T.overviewCorner(r.current, r.frame, r.size, r.inset, r.keep, r.others);
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

fn module_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/review-touch.js")
}

/// Run [`DRIVER`] on `requests`, in a folder of its own, and read its answers.
fn node(requests: &Value) -> Vec<Value> {
    let dir = std::env::temp_dir().join(format!("bop-js-touch-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let (driver, cases) = (dir.join("driver.js"), dir.join("request.json"));
    std::fs::write(&driver, DRIVER).unwrap();
    std::fs::write(&cases, serde_json::to_vec(requests).unwrap()).unwrap();
    let out = Command::new("node")
        .arg(&driver)
        .arg(module_path())
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

const MAX_ZOOM: f64 = 8.0;

fn pinch_req(a: [f64; 2], b: [f64; 2], view: [f64; 3], na: [f64; 2], nb: [f64; 2]) -> Value {
    json!({
        "op": "pinch",
        "start": { "a": a, "b": b, "view": { "z": view[0], "ox": view[1], "oy": view[2] } },
        "now": { "a": na, "b": nb },
        "limits": { "min": 1, "max": MAX_ZOOM },
    })
}

fn view_of(v: &Value) -> [f64; 3] {
    [
        v["z"].as_f64().unwrap(),
        v["ox"].as_f64().unwrap(),
        v["oy"].as_f64().unwrap(),
    ]
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

/// The frame point under canvas point `c` in view `v` (review.js `framePoint` before clamping).
fn frame_point(v: [f64; 3], c: [f64; 2]) -> [f64; 2] {
    [c[0] / v[0] + v[1], c[1] / v[0] + v[2]]
}

#[test]
fn equal_distances_and_a_moved_midpoint_pan_exactly() {
    if !have_node() {
        return;
    }
    // Zoomed 2x at (100, 50): both fingers move by (+30, -20) canvas pixels, the zoom stays and the
    // view moves by the opposite of that in frame pixels (half, at 2x).
    let got = node(&json!([pinch_req(
        [200.0, 200.0],
        [400.0, 300.0],
        [2.0, 100.0, 50.0],
        [230.0, 180.0],
        [430.0, 280.0]
    )]));
    let v = view_of(&got[0]);
    assert!(close(v[0], 2.0), "{v:?}");
    assert!(close(v[1], 100.0 - 15.0), "{v:?}");
    assert!(close(v[2], 50.0 + 10.0), "{v:?}");
    // At zoom 1 a pan is canvas pixels one to one (the page then clamps it to the frame).
    let got = node(&json!([pinch_req(
        [10.0, 10.0],
        [110.0, 10.0],
        [1.0, 0.0, 0.0],
        [5.0, 13.0],
        [105.0, 13.0]
    )]));
    let v = view_of(&got[0]);
    assert_eq!(
        (v[0], v[1], v[2]),
        (1.0, 5.0, -3.0),
        "zoom 1 pans 1:1: {v:?}"
    );
}

#[test]
fn a_pinch_keeps_the_frame_point_under_the_midpoint() {
    if !have_node() {
        return;
    }
    let cases = [
        // Spread from 100 to 300 px about a fixed midpoint: 3x.
        (
            [300.0, 200.0],
            [400.0, 200.0],
            [1.0, 0.0, 0.0],
            [200.0, 200.0],
            [500.0, 200.0],
        ),
        // Already zoomed, fingers spread and the midpoint moves too.
        (
            [100.0, 100.0],
            [200.0, 200.0],
            [2.0, 40.0, 30.0],
            [90.0, 120.0],
            [240.0, 270.0],
        ),
        // Fingers pinch in: zoom out from 4x to 2x.
        (
            [500.0, 300.0],
            [700.0, 300.0],
            [4.0, 300.0, 100.0],
            [550.0, 300.0],
            [650.0, 300.0],
        ),
    ];
    let req: Vec<Value> = cases
        .iter()
        .map(|c| pinch_req(c.0, c.1, c.2, c.3, c.4))
        .collect();
    let got = node(&json!(req));
    for (c, g) in cases.iter().zip(&got) {
        let (a, b, v0, na, nb) = *c;
        let v = view_of(g);
        let d0 = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        let d1 = ((nb[0] - na[0]).powi(2) + (nb[1] - na[1]).powi(2)).sqrt();
        assert!(close(v[0], v0[0] * d1 / d0), "zoom by the ratio: {v:?}");
        let m0 = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
        let m1 = [(na[0] + nb[0]) / 2.0, (na[1] + nb[1]) / 2.0];
        let (before, after) = (frame_point(v0, m0), frame_point(v, m1));
        assert!(
            close(before[0], after[0]) && close(before[1], after[1]),
            "{before:?} -> {after:?} ({v:?})"
        );
    }
    // The first case exactly: 3x about canvas (350, 200).
    let v = view_of(&got[0]);
    assert!(close(v[0], 3.0));
    assert!(close(v[1], 350.0 - 350.0 / 3.0) && close(v[2], 200.0 - 200.0 / 3.0));
}

#[test]
fn the_zoom_is_clamped_to_one_and_the_maximum() {
    if !have_node() {
        return;
    }
    let got = node(&json!([
        // 20x spread from 2x: stops at the maximum.
        pinch_req(
            [300.0, 300.0],
            [310.0, 300.0],
            [2.0, 10.0, 10.0],
            [205.0, 300.0],
            [405.0, 300.0]
        ),
        // Pinched to a tenth at 2x: stops at 1.
        pinch_req(
            [100.0, 300.0],
            [500.0, 300.0],
            [2.0, 50.0, 20.0],
            [280.0, 300.0],
            [320.0, 300.0]
        ),
        // Two fingers down on one point give no ratio: the zoom stays.
        pinch_req(
            [300.0, 300.0],
            [300.0, 300.0],
            [3.0, 0.0, 0.0],
            [100.0, 300.0],
            [500.0, 300.0]
        ),
    ]));
    let top = view_of(&got[0]);
    assert!(close(top[0], MAX_ZOOM), "{top:?}");
    // The midpoint's frame point is still kept at the clamped zoom.
    let (before, after) = (
        frame_point([2.0, 10.0, 10.0], [305.0, 300.0]),
        frame_point(top, [305.0, 300.0]),
    );
    assert!(close(before[0], after[0]) && close(before[1], after[1]));
    assert!(close(view_of(&got[1])[0], 1.0), "{:?}", got[1]);
    assert!(close(view_of(&got[2])[0], 3.0), "{:?}", got[2]);
}

#[test]
fn two_taps_make_a_normalised_box_or_none() {
    if !have_node() {
        return;
    }
    let got = node(&json!([
        { "op": "twoTapRect", "a": [10, 20], "b": [110, 220], "minBox": 3 },
        // Reversed corners (bottom right first, then top left), and the other diagonal.
        { "op": "twoTapRect", "a": [110, 220], "b": [10, 20], "minBox": 3 },
        { "op": "twoTapRect", "a": [110, 20], "b": [10, 220], "minBox": 3 },
        // Too thin, too short, one spot.
        { "op": "twoTapRect", "a": [10, 20], "b": [12, 220], "minBox": 3 },
        { "op": "twoTapRect", "a": [10, 20], "b": [110, 22.5], "minBox": 3 },
        { "op": "twoTapRect", "a": [50, 50], "b": [50, 50], "minBox": 3 },
        // Exactly the minimum is a box.
        { "op": "twoTapRect", "a": [0, 0], "b": [3, 3], "minBox": 3 },
    ]));
    let want = json!([10, 20, 110, 220]);
    assert_eq!(got[0], want);
    assert_eq!(got[1], want);
    assert_eq!(got[2], want);
    assert_eq!(got[3], Value::Null);
    assert_eq!(got[4], Value::Null);
    assert_eq!(got[5], Value::Null);
    assert_eq!(got[6], json!([0, 0, 3, 3]));
}

#[test]
fn one_finger_intent_over_every_combination() {
    if !have_node() {
        return;
    }
    let mut req = Vec::new();
    let mut want = Vec::new();
    for bits in 0..16u8 {
        let (draw, zoomed, handle, inside) =
            (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0);
        req.push(json!({ "op": "touchIntent", "s": {
            "drawMode": draw, "zoomed": zoomed, "onHandle": handle, "insideSelected": inside,
        }}));
        // A handle resizes and the selected box moves in either mode; then Draw draws, a zoomed
        // frame pans, and an unzoomed one does nothing (a tap still selects).
        want.push(if handle {
            "resize"
        } else if inside {
            "move"
        } else if draw {
            "draw"
        } else if zoomed {
            "pan"
        } else {
            "none"
        });
    }
    let got = node(&json!(req));
    assert_eq!(got, json!(want).as_array().unwrap().clone());
    // The table names each answer at least once.
    for w in ["resize", "move", "draw", "pan", "none"] {
        assert!(want.contains(&w), "{w}");
    }
}

#[test]
fn a_tag_sits_above_then_below_then_inside_within_the_view() {
    if !have_node() {
        return;
    }
    let tag = |bb: [f64; 4], top: f64, bottom: f64| json!({ "op": "tagTop", "bb": bb, "top": top, "bottom": bottom, "th": 20.0 });
    let got = node(&json!([
        // The whole frame (zoom 1, 0..360): above with room, at the frame's top below, a box
        // filling the frame inside its top, exactly the tag's height above is still above.
        tag([10.0, 100.0, 50.0, 200.0], 0.0, 360.0),
        tag([10.0, 5.0, 50.0, 200.0], 0.0, 360.0),
        tag([0.0, 5.0, 640.0, 350.0], 0.0, 360.0),
        tag([10.0, 20.0, 50.0, 200.0], 0.0, 360.0),
        // Zoomed to rows 100..190: a box whose top is at the view's top gets its tag under it,
        // above the view's bottom it fits above again, one cut by both edges keeps it in view.
        tag([10.0, 105.0, 50.0, 150.0], 100.0, 190.0),
        tag([10.0, 140.0, 50.0, 185.0], 100.0, 190.0),
        tag([10.0, 60.0, 50.0, 250.0], 100.0, 190.0),
        // Too tall for either: inside its own top when that is in view.
        tag([10.0, 110.0, 50.0, 185.0], 100.0, 190.0),
    ]));
    let want = [80.0, 200.0, 5.0, 0.0, 150.0, 120.0, 100.0, 110.0];
    for (g, w) in got.iter().zip(want) {
        assert!(close(g.as_f64().unwrap(), w), "{got:?} want {want:?}");
    }
    // At zoom 1 the rule is the frame's old one (above when y >= th, else below when it fits,
    // else inside the top), over a grid of boxes.
    let (h, th) = (360.0, 20.0);
    let mut req = Vec::new();
    let mut old = Vec::new();
    for y0 in [0.0, 5.0, 19.0, 20.0, 21.0, 150.0, 339.0] {
        for y1 in [y0 + 3.0, y0 + 40.0, 339.0, 340.0, 341.0, 360.0] {
            if y1 > h || y1 <= y0 {
                continue;
            }
            req.push(tag([0.0, y0, 10.0, y1], 0.0, h));
            old.push(if y0 >= th {
                y0 - th
            } else if y1 + th <= h {
                y1
            } else {
                y0
            });
        }
    }
    let got = node(&json!(req));
    for (g, w) in got.iter().zip(&old) {
        assert!(close(g.as_f64().unwrap(), *w), "{got:?} want {old:?}");
    }
}

#[test]
fn an_overview_tap_picks_the_smallest_box_holding_it() {
    if !have_node() {
        return;
    }
    let boxes = json!([
        [0, 0, 100, 100],
        [20, 20, 60, 60],
        [30, 30, 50, 50],
        [200, 200, 210, 210]
    ]);
    let pick = |x: f64, y: f64, pad: f64| json!({ "op": "pickBox", "boxes": boxes, "x": x, "y": y, "pad": pad });
    let got = node(&json!([
        pick(40.0, 40.0, 0.0),
        pick(25.0, 25.0, 0.0),
        pick(90.0, 90.0, 0.0),
        pick(150.0, 150.0, 0.0),
        // On the edge counts; just past it only with a pad, which never changes the size.
        pick(210.0, 205.0, 0.0),
        pick(212.0, 205.0, 0.0),
        pick(212.0, 205.0, 3.0),
        pick(102.0, 50.0, 3.0),
        // No boxes: none.
        { "op": "pickBox", "boxes": [], "x": 1, "y": 1 },
        // Equal boxes: the first.
        { "op": "pickBox", "boxes": [[0, 0, 10, 10], [0, 0, 10, 10]], "x": 5, "y": 5 },
    ]));
    assert_eq!(
        got,
        json!([2, 1, 0, -1, 3, -1, 3, 0, -1, 0])
            .as_array()
            .unwrap()
            .clone()
    );
}

#[test]
fn the_overview_leaves_the_selected_box_in_sight() {
    if !have_node() {
        return;
    }
    // A 1000 x 600 frame, a 200 x 120 overview 8 px in: top right covers x 792..992, y 8..128.
    let corner = |current: &str, keep: Value, others: Value| {
        json!({ "op": "overviewCorner", "current": current, "frame": [1000, 600],
                "size": [200, 120], "inset": 8, "keep": keep, "others": others })
    };
    let got = node(&json!([
        // Nothing to keep in sight: it stays where it is.
        corner("tr", json!([]), json!([])),
        corner("bl", json!([]), json!([[0, 400, 300, 600]])),
        // The selected box in the top right corner: the next free corner, top left.
        corner("tr", json!([[850, 20, 950, 100]]), json!([])),
        // Top left would cover another box: bottom right, which covers nothing.
        corner(
            "tr",
            json!([[850, 20, 950, 100]]),
            json!([[0, 0, 100, 100]])
        ),
        // Where it is covers nothing of the box: it does not move for the other boxes.
        corner(
            "tl",
            json!([[850, 20, 950, 100]]),
            json!([[0, 0, 100, 100]])
        ),
        // Only a handle's margin reaches the corner: still covered, still moves.
        corner("tr", json!([[600, 120, 795, 300]]), json!([])),
        // A box over every corner and another over the top ones: the bottom corners cover least,
        // and of those bottom right comes first.
        corner(
            "tr",
            json!([[0, 0, 1000, 600], [0, 0, 1000, 200]]),
            json!([])
        ),
        // An unknown current corner is never kept.
        corner("middle", json!([]), json!([])),
    ]));
    assert_eq!(
        got,
        json!(["tr", "bl", "tl", "br", "tl", "tl", "br", "tr"])
            .as_array()
            .unwrap()
            .clone()
    );
}

/// Needs no `node`: the page loads the module before review.js, and it stays pure.
#[test]
fn the_page_loads_the_touch_rules_first_and_they_touch_no_page() {
    let page = include_str!("../../templates/train/review.html");
    let (touch, own) = (
        page.find("/static/review-touch.js?v=").unwrap(),
        page.find("/static/review.js?v=").unwrap(),
    );
    assert!(touch < own, "review.js reads ReviewTouch when it starts");
    let js = std::fs::read_to_string(module_path()).unwrap();
    for word in [
        "document",
        "window",
        "fetch(",
        "localStorage",
        "navigator",
        "location",
    ] {
        assert!(!js.contains(word), "review-touch.js is pure: {word}");
    }
}
