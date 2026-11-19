//! The Review tab's finger rules (`assets/review-touch.js`): pinch zoom and pan, the two-tap box,
//! what one finger's drag does, where a tag sits in the zoomed view, which box an overview tap
//! picks, which corner the overview takes and the quarters of the "Anything missing?" look; and
//! the editor's rules: which edge or corner of a box a press grabs (`edgeAt`), what the view does
//! when the selection moves (`nextView`) and a drag of the overview's view rect (`overviewPan`).
//! The page applies these answers to its view and boxes (`assets/review.js`), so they are checked
//! here on exact numbers, in Node.
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
  if (r.op === 'quarterView') return T.quarterView(r.i, r.w, r.h, r.share);
  if (r.op === 'edgeAt') return T.edgeAt(r.bbox, r.p, r.grabOut, r.grabIn);
  if (r.op === 'nextView') return T.nextView(r.o);
  if (r.op === 'overviewPan') return T.overviewPan(r.view, r.frame, r.delta);
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

/// The page's zoom limit (review.js `MAX_ZOOM`), pinned in
/// `the_pinch_limit_is_the_pages_zoom_limit`.
const MAX_ZOOM: f64 = 16.0;

#[test]
fn the_pinch_limit_is_the_pages_zoom_limit() {
    // Needs no `node`: the limits these tests pass to the pinch are the page's own.
    assert!(include_str!("../../assets/review.js").contains("var MAX_ZOOM = 16;"));
}

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
        // 6x spread from 2x: 12x, under the maximum, so not clamped.
        pinch_req(
            [300.0, 300.0],
            [320.0, 300.0],
            [2.0, 10.0, 10.0],
            [250.0, 300.0],
            [370.0, 300.0]
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
    assert!(close(view_of(&got[3])[0], 12.0), "{:?}", got[3]);
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

fn quarter(i: f64, w: f64, h: f64, share: f64) -> Value {
    json!({ "op": "quarterView", "i": i, "w": w, "h": h, "share": share })
}

fn rect_of(v: &Value) -> [f64; 4] {
    let a = v.as_array().unwrap();
    assert_eq!(a.len(), 4, "{v}");
    [
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
        a[3].as_f64().unwrap(),
    ]
}

fn within(r: [f64; 4], b: [f64; 4]) -> bool {
    r[0] >= b[0] && r[1] >= b[1] && r[2] <= b[2] && r[3] <= b[3]
}

#[test]
fn the_quarter_look_steps_through_overlapping_corners_then_the_whole_frame() {
    // The page's share, checked below on an even frame and on odd ones.
    assert!(include_str!("../../assets/review.js").contains("var QUARTER_SHARE = 0.55;"));
    if !have_node() {
        return;
    }
    let share = 0.55;
    for (w, h) in [
        (1000.0, 600.0),
        (641.0, 359.0),
        (1921.0, 1081.0),
        (3.0, 7.0),
    ] {
        let got = node(&json!(
            (0..6)
                .map(|i| quarter(f64::from(i), w, h, share))
                .collect::<Vec<_>>()
        ));
        let q: Vec<[f64; 4]> = got.iter().map(rect_of).collect();
        let (sw, sh) = (w * share, h * share);
        // Corners in order: top left, top right, bottom left, bottom right, each anchored to its
        // own corner exactly (the far edges are the frame's own, never past it).
        assert_eq!(q[0][0], 0.0, "{w}x{h}: {:?}", q[0]);
        assert_eq!(q[0][1], 0.0, "{w}x{h}: {:?}", q[0]);
        assert_eq!(q[1][2], w, "{w}x{h}: {:?}", q[1]);
        assert_eq!(q[1][1], 0.0, "{w}x{h}: {:?}", q[1]);
        assert_eq!(q[2][0], 0.0, "{w}x{h}: {:?}", q[2]);
        assert_eq!(q[2][3], h, "{w}x{h}: {:?}", q[2]);
        assert_eq!(q[3][2], w, "{w}x{h}: {:?}", q[3]);
        assert_eq!(q[3][3], h, "{w}x{h}: {:?}", q[3]);
        for r in &q[..4] {
            // Inside the frame, `share` of each side, so one zoom keeps the frame's aspect.
            assert!(within(*r, [0.0, 0.0, w, h]), "{w}x{h}: {r:?}");
            assert!(
                close(r[2] - r[0], sw) && close(r[3] - r[1], sh),
                "{w}x{h}: {r:?}"
            );
            assert!(
                close(w / (r[2] - r[0]), h / (r[3] - r[1])),
                "{w}x{h}: {r:?}"
            );
        }
        // Neighbours overlap by (2 x share - 1) of the side: 10% at 0.55.
        let (ox, oy) = ((2.0 * share - 1.0) * w, (2.0 * share - 1.0) * h);
        assert!(close(q[0][2] - q[1][0], ox), "{w}x{h}: {q:?}");
        assert!(close(q[2][2] - q[3][0], ox), "{w}x{h}: {q:?}");
        assert!(close(q[0][3] - q[2][1], oy), "{w}x{h}: {q:?}");
        assert!(close(q[1][3] - q[3][1], oy), "{w}x{h}: {q:?}");
        // Step 4 (and on) is the whole frame.
        assert_eq!(q[4], [0.0, 0.0, w, h], "{w}x{h}");
        assert_eq!(q[5], [0.0, 0.0, w, h], "{w}x{h}");
    }
    // An object on a centre line, up to the overlap wide, is whole in one quarter.
    let got = node(&json!([quarter(0.0, 1000.0, 600.0, share)]));
    assert!(within([455.0, 100.0, 545.0, 150.0], rect_of(&got[0])));
    // Steps that are no quarter show the whole frame; the share is held to 0.5..1 (0.5: the
    // quarters meet exactly, 1: each is the whole frame).
    let got = node(&json!([
        quarter(-1.0, 800.0, 400.0, share),
        quarter(1.5, 800.0, 400.0, share),
        quarter(3.0, 800.0, 400.0, 0.2),
        quarter(3.0, 800.0, 400.0, 1.5),
    ]));
    let r: Vec<[f64; 4]> = got.iter().map(rect_of).collect();
    assert_eq!(r[0], [0.0, 0.0, 800.0, 400.0]);
    assert_eq!(r[1], [0.0, 0.0, 800.0, 400.0]);
    assert_eq!(r[2], [400.0, 200.0, 800.0, 400.0]);
    assert_eq!(r[3], [0.0, 0.0, 800.0, 400.0]);
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

/// A box 100..200 x 50..250 with a 4 unit band outside each edge and 10 inside.
const BOX: [f64; 4] = [100.0, 50.0, 200.0, 250.0];

fn edge(bbox: [f64; 4], p: [f64; 2], out: f64, inside: f64) -> Value {
    json!({ "op": "edgeAt", "bbox": bbox, "p": p, "grabOut": out, "grabIn": inside })
}

/// Runs `cases` (point, want) on [`BOX`] with bands 4 out and 10 in, naming the point on a miss.
fn edges_on_box(cases: &[([f64; 2], Value)]) {
    let got = node(&json!(
        cases
            .iter()
            .map(|(p, _)| edge(BOX, *p, 4.0, 10.0))
            .collect::<Vec<_>>()
    ));
    for ((p, want), g) in cases.iter().zip(&got) {
        assert_eq!(g, want, "point {p:?}");
    }
}

#[test]
fn a_press_grabs_each_side_and_corner_and_none_inside_or_off_the_box() {
    if !have_node() {
        return;
    }
    edges_on_box(&[
        // Sides at their middles, just inside each edge.
        ([105.0, 150.0], json!("w")),
        ([195.0, 150.0], json!("e")),
        ([150.0, 55.0], json!("n")),
        ([150.0, 245.0], json!("s")),
        // Corners, inside both bands.
        ([105.0, 55.0], json!("nw")),
        ([195.0, 55.0], json!("ne")),
        ([195.0, 245.0], json!("se")),
        ([105.0, 245.0], json!("sw")),
        // Well inside: no edge, a move. Far outside on every side: nothing.
        ([150.0, 150.0], Value::Null),
        ([0.0, 150.0], Value::Null),
        ([300.0, 150.0], Value::Null),
        ([150.0, 0.0], Value::Null),
        ([150.0, 400.0], Value::Null),
        ([0.0, 0.0], Value::Null),
    ]);
}

#[test]
fn the_bands_reach_grab_out_outside_and_grab_in_inside_each_edge() {
    if !have_node() {
        return;
    }
    edges_on_box(&[
        // Left edge x 100: the band is 96..110.
        ([96.0, 150.0], json!("w")),
        ([95.9, 150.0], Value::Null),
        ([110.0, 150.0], json!("w")),
        ([110.1, 150.0], Value::Null),
        // Right edge x 200: 190..204.
        ([204.0, 150.0], json!("e")),
        ([204.1, 150.0], Value::Null),
        ([190.0, 150.0], json!("e")),
        ([189.9, 150.0], Value::Null),
        // Top edge y 50: 46..60.
        ([150.0, 46.0], json!("n")),
        ([150.0, 45.9], Value::Null),
        ([150.0, 60.0], json!("n")),
        ([150.0, 60.1], Value::Null),
        // Bottom edge y 250: 240..254.
        ([150.0, 254.0], json!("s")),
        ([150.0, 254.1], Value::Null),
        ([150.0, 240.0], json!("s")),
        ([150.0, 239.9], Value::Null),
    ]);
    // The same point is in or out by which band is wider: 6 outside the left edge needs a
    // 6 unit outer band, 6 inside it a 6 unit inner one.
    let got = node(&json!([
        edge(BOX, [94.0, 150.0], 4.0, 10.0),
        edge(BOX, [94.0, 150.0], 6.0, 2.0),
        edge(BOX, [106.0, 150.0], 6.0, 2.0),
        edge(BOX, [106.0, 150.0], 4.0, 10.0),
        // No bands at all: only the edge line itself.
        edge(BOX, [100.0, 150.0], 0.0, 0.0),
        edge(BOX, [100.5, 150.0], 0.0, 0.0),
    ]));
    assert_eq!(
        got,
        json!([null, "w", null, "w", "w", null])
            .as_array()
            .unwrap()
            .clone()
    );
}

#[test]
fn where_bands_cross_a_corner_wins_and_side_bands_end_grab_out_past_the_box() {
    if !have_node() {
        return;
    }
    edges_on_box(&[
        // Inside both the left and the top band, anywhere in their crossing (inner corner of
        // the crossing, outer corner, and mixed): the corner, never a side.
        ([110.0, 60.0], json!("nw")),
        ([96.0, 46.0], json!("nw")),
        ([96.0, 60.0], json!("nw")),
        ([110.0, 46.0], json!("nw")),
        // Just past the crossing on one axis: the side only.
        ([110.1, 55.0], json!("n")),
        ([105.0, 60.1], json!("w")),
        ([204.0, 254.0], json!("se")),
        ([189.9, 254.0], json!("s")),
        // Beside the left edge, the band runs y 46..254; beyond that, nothing (and a point far
        // below the box in line with its edge is no grab).
        ([98.0, 254.0], json!("sw")),
        ([98.0, 254.1], Value::Null),
        ([98.0, 45.9], Value::Null),
        ([105.0, 400.0], Value::Null),
        ([195.0, 400.0], Value::Null),
        // Beside the top edge, the band runs x 96..204.
        ([95.9, 55.0], Value::Null),
        ([204.1, 55.0], Value::Null),
        ([400.0, 55.0], Value::Null),
    ]);
}

#[test]
fn on_a_tiny_box_the_nearer_edge_wins_and_a_tie_goes_right_and_down() {
    if !have_node() {
        return;
    }
    // Bands 4 out and 10 in. On a 6 x 6 box at 100..106 every point from 96 to 110 is in both
    // opposite bands on both axes (the caller caps grabIn, but the rule must hold either way), so
    // each axis picks its nearer edge and a press there is always a corner.
    let tiny = [100.0, 100.0, 106.0, 106.0];
    // A flat box (100 x 6) and a thin one (6 x 100): one axis has both bands, the other none.
    let (flat, thin) = ([100.0, 100.0, 200.0, 106.0], [100.0, 100.0, 106.0, 200.0]);
    let got = node(&json!([
        edge(tiny, [101.0, 101.0], 4.0, 10.0),
        edge(tiny, [105.0, 101.0], 4.0, 10.0),
        edge(tiny, [98.0, 105.0], 4.0, 10.0),
        edge(tiny, [108.0, 105.0], 4.0, 10.0),
        // The exact middle on an axis: east across, south up and down.
        edge(tiny, [103.0, 103.0], 4.0, 10.0),
        edge(tiny, [103.0, 101.0], 4.0, 10.0),
        edge(tiny, [101.0, 103.0], 4.0, 10.0),
        edge(flat, [150.0, 101.0], 4.0, 10.0),
        edge(flat, [150.0, 105.0], 4.0, 10.0),
        edge(flat, [150.0, 103.0], 4.0, 10.0),
        edge(thin, [101.0, 150.0], 4.0, 10.0),
        edge(thin, [105.0, 150.0], 4.0, 10.0),
        edge(thin, [103.0, 150.0], 4.0, 10.0),
        // grabIn capped at a third of the side (2): the bands no longer meet and the middle moves.
        edge(tiny, [103.0, 103.0], 4.0, 2.0),
        edge(tiny, [101.0, 103.0], 4.0, 2.0),
        // A zero-size box with no inner band: a press on it is a tie on both axes, so a drag out
        // of it grows it right and down; a press beside it takes the side it is on.
        edge([50.0, 50.0, 50.0, 50.0], [50.0, 50.0], 4.0, 0.0),
        edge([50.0, 50.0, 50.0, 50.0], [47.0, 47.0], 4.0, 0.0),
        edge([50.0, 50.0, 50.0, 50.0], [53.0, 50.0], 4.0, 0.0),
        // A zero-width box with height: on its line east, beside it west, past its top end the
        // corner.
        edge([50.0, 50.0, 50.0, 90.0], [50.0, 70.0], 4.0, 0.0),
        edge([50.0, 50.0, 50.0, 90.0], [49.0, 70.0], 4.0, 0.0),
        edge([50.0, 50.0, 50.0, 90.0], [50.0, 48.0], 4.0, 0.0),
    ]));
    assert_eq!(
        got,
        json!([
            "nw", "ne", "sw", "se", "se", "ne", "sw", "n", "s", "s", "w", "e", "e", null, "w",
            "se", "nw", "se", "e", "w", "ne"
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[test]
fn an_edge_needs_finite_numbers_and_takes_the_corners_in_either_order() {
    if !have_node() {
        return;
    }
    // JSON has no NaN or Infinity: a null, a string and a missing value stand for a value that
    // is no number. JavaScript reads null as 0 and "4" as 4 in arithmetic, so each case is one
    // that would grab an edge if the value got through (x0 as 0: the east edge; y as 0: the
    // north-west corner; grabOut "4": the west edge).
    let got = node(&json!([
        { "op": "edgeAt", "bbox": [null, 50, 200, 250], "p": [195, 150], "grabOut": 4, "grabIn": 10 },
        { "op": "edgeAt", "bbox": [100, 0, 200, 250], "p": [105, null], "grabOut": 4, "grabIn": 10 },
        { "op": "edgeAt", "bbox": BOX, "p": [105, 150], "grabOut": "4", "grabIn": 10 },
        { "op": "edgeAt", "bbox": BOX, "p": [105, 150], "grabOut": 4 },
        { "op": "edgeAt", "bbox": BOX, "p": [105, 150], "grabOut": 4, "grabIn": 10 },
        // Reversed corners are the same box.
        edge([200.0, 250.0, 100.0, 50.0], [105.0, 55.0], 4.0, 10.0),
        edge([200.0, 50.0, 100.0, 250.0], [195.0, 245.0], 4.0, 10.0),
    ]));
    assert_eq!(
        got,
        json!([null, null, null, null, "w", "nw", "se"])
            .as_array()
            .unwrap()
            .clone()
    );
}

#[test]
fn the_next_view_over_every_combination() {
    if !have_node() {
        return;
    }
    let mut req = Vec::new();
    let mut want = Vec::new();
    for bits in 0..8u8 {
        let (follow, custom, same) = (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0);
        req.push(json!({ "op": "nextView", "o": {
            "follow": follow, "custom": custom, "sameBox": same,
        }}));
        // The same box keeps the reviewer's view; after a zoom or pan by hand the next box is
        // framed; else the last Fit / Zoom-to-box choice.
        want.push(if same {
            "keep"
        } else if custom || follow {
            "box"
        } else {
            "fit"
        });
    }
    let got = node(&json!(req));
    assert_eq!(got, json!(want).as_array().unwrap().clone());
    // Written out: only no follow, no custom zoom and another box gives the whole frame.
    assert_eq!(
        want,
        ["fit", "box", "box", "box", "keep", "keep", "keep", "keep"]
    );
}

fn pan(view: [f64; 3], frame: [f64; 2], delta: [f64; 2]) -> Value {
    json!({ "op": "overviewPan", "view": { "z": view[0], "ox": view[1], "oy": view[2] },
            "frame": frame, "delta": delta })
}

fn offsets(v: &Value) -> [f64; 2] {
    let o = v.as_object().unwrap();
    assert_eq!(o.len(), 2, "only ox and oy: {v}");
    [v["ox"].as_f64().unwrap(), v["oy"].as_f64().unwrap()]
}

#[test]
fn an_overview_drag_moves_the_view_and_stays_in_the_frame() {
    if !have_node() {
        return;
    }
    // A 1000 x 600 frame at 4x shows 250 x 150: the offsets run 0..750 and 0..450.
    let f = [1000.0, 600.0];
    let got = node(&json!([
        pan([4.0, 100.0, 100.0], f, [30.0, -20.0]),
        // Past each edge: held at it.
        pan([4.0, 100.0, 100.0], f, [-500.0, 0.0]),
        pan([4.0, 100.0, 100.0], f, [5000.0, 0.0]),
        pan([4.0, 100.0, 100.0], f, [0.0, -500.0]),
        pan([4.0, 100.0, 100.0], f, [0.0, 5000.0]),
        // Exactly onto the far corner.
        pan([4.0, 100.0, 100.0], f, [650.0, 350.0]),
        // A view already outside the frame comes back in, even with no move.
        pan([4.0, -40.0, 900.0], f, [0.0, 0.0]),
        // Zoom 1 and just under 1.001 are the whole frame; just over pans a little.
        pan([1.0, 30.0, 30.0], f, [10.0, 10.0]),
        pan([1.0009, 0.0, 0.0], f, [10.0, 10.0]),
        pan([1.25, 0.0, 0.0], f, [100.0, 100.0]),
        // A tall frame (360 x 640) at 2x: 0..180 across, 0..320 down.
        pan([2.0, 0.0, 0.0], [360.0, 640.0], [500.0, 500.0]),
        pan([2.0, 100.0, 100.0], [360.0, 640.0], [50.0, 200.0]),
    ]));
    let o: Vec<[f64; 2]> = got.iter().map(offsets).collect();
    let want = [
        [130.0, 80.0],
        [0.0, 100.0],
        [750.0, 100.0],
        [100.0, 0.0],
        [100.0, 450.0],
        [750.0, 450.0],
        [0.0, 450.0],
        [0.0, 0.0],
        [0.0, 0.0],
        // 1000 - 1000 / 1.25 = 200 across, 600 - 480 = 120 down.
        [100.0, 100.0],
        [180.0, 320.0],
        [150.0, 300.0],
    ];
    for (i, (g, w)) in o.iter().zip(want).enumerate() {
        assert!(
            close(g[0], w[0]) && close(g[1], w[1]),
            "case {i}: {g:?} want {w:?}"
        );
    }
    // Clamped to 1.25x's limits too, not just its zoom.
    let got = node(&json!([pan([1.25, 0.0, 0.0], f, [300.0, 300.0])]));
    let g = offsets(&got[0]);
    assert!(close(g[0], 200.0) && close(g[1], 120.0), "{g:?}");
}

#[test]
fn an_overview_drag_survives_values_that_are_no_numbers() {
    if !have_node() {
        return;
    }
    let f = json!([1000, 600]);
    let got = node(&json!([
        // No zoom or frame: the whole frame.
        { "op": "overviewPan", "view": { "z": null, "ox": 100, "oy": 100 }, "frame": f, "delta": [10, 10] },
        { "op": "overviewPan", "view": { "z": 4, "ox": 100, "oy": 100 }, "frame": [null, 600], "delta": [10, 10] },
        // No delta or offset: that one counts as 0, the rest still applies (a string delta
        // would otherwise join the offset as text, "100" + "5", and pan to the far edge).
        { "op": "overviewPan", "view": { "z": 4, "ox": 100, "oy": 100 }, "frame": f, "delta": [null, 10] },
        { "op": "overviewPan", "view": { "z": 4, "ox": 100, "oy": 100 }, "frame": f, "delta": ["5", 10] },
        { "op": "overviewPan", "view": { "z": 4, "ox": "x", "oy": 100 }, "frame": f, "delta": [10, 10] },
        { "op": "overviewPan", "view": { "z": 4, "ox": 100, "oy": 100 }, "frame": f },
    ]));
    let o: Vec<[f64; 2]> = got.iter().map(offsets).collect();
    assert_eq!(
        o,
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [100.0, 110.0],
            [100.0, 110.0],
            [10.0, 110.0],
            [100.0, 100.0]
        ]
    );
}
