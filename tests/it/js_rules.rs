//! The Review tab's rules (`assets/review-rules.js`) against their Rust twins, on the same frames.
//!
//! The JavaScript is what runs when a reviewer decides a box, and what it keeps reaches the exports;
//! the Rust twins run in the bulk tools and the queue. So the two are compared rather than each
//! tested alone: a seeded generator builds frames (stacks, chains, nested pairs, two boxes at one
//! position, hints, parked and dismissed boxes, unwanted classes, stale suggestions), they go to
//! `node` as JSON, exactly as `GET /v1/review/labels` shapes them, and every answer must match.
//!
//! Needs `node`. Without it these tests say so and pass, unless `BOP_REQUIRE_NODE=1` (CI), which
//! makes a missing `node` a failure: a skip must not read as a pass there.
#![allow(clippy::unwrap_used, clippy::indexing_slicing)]

use blue_onyx_prism::collect::labels::bulk::Wanted;
use blue_onyx_prism::collect::labels::overlaps::{self, Suggestion, Verdict};
use blue_onyx_prism::collect::labels::{
    self, AutoInfo, BoxState, LabelBox, LabelFile, Origin, ParkedMark, RejectReason, Source,
    Status, Triage, refine, repeats, resolve_pending, spots, stacks,
};
use blue_onyx_prism::model::coco;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

/// The frame's rules run in Node; prints one JSON answer per request. `process.argv`: this script,
/// the rules file, the request file.
const DRIVER: &str = r#"
const fs = require('fs');
require(process.argv[2]);
const R = globalThis.ReviewRules;
const req = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const clone = (v) => JSON.parse(JSON.stringify(v));
// What a save strips from the file (assets/review.js `save`).
const saved = (labels) => {
  (labels.overlap_suggestions || []).forEach((s) => { delete s._done; });
  labels.boxes.forEach((b) => { delete b._checked; delete b._relabelled; delete b._n; });
  return labels;
};
// `served`: what GET /v1/review/labels says is live (`overlap_live`, `overlap_live_boxes`).
const liveOf = (labels, served, classes, hidden) => {
  const d = R.dupGroups(labels.boxes, classes, hidden);
  return R.liveSuggestions(Object.assign({ labels: labels }, served), d);
};
const brief = (l) => l.map((g) => ({ index: g.index, a: g.a, b: g.b, drop: g.drop, keep: g.keep }));
// `relabel` of assets/review.js, the part that decides.
const relabel = (b, label) => {
  if (b.source !== 'human' && !b.model_label) b.model_label = b.label;
  b.label = label;
  b._relabelled = true;
  b._checked = true;
  R.endDismissal(b);
  if (b.source !== 'human') b.state = 'accepted';
};
const frame = (c, classes, hidden) => {
  const labels = c.labels;
  // What the page does with every answer (assets/review.js `loadFrame`).
  R.markServed(labels);
  const d = R.dupGroups(labels.boxes, classes, hidden);
  const served = { overlap_live: c.overlap_live, overlap_live_boxes: c.overlap_live_boxes };
  const live = R.liveSuggestions(Object.assign({ labels: labels }, served), d);
  const pending = clone(labels);
  R.pendingDupDrops(d, pending.boxes).forEach((k) => { R.markDuplicate(pending.boxes[k]); });
  return {
    resolved: labels.boxes.map(R.resolved),
    groups: d.groups.map((g) => ({ members: g.members, keeper: g.keeper })),
    overlaps: d.overlaps,
    drops: R.dupDrops(d, labels.boxes),
    pending_drops: R.pendingDupDrops(d, labels.boxes),
    pending_marked: saved(pending),
    live: brief(live),
    own_live: brief(liveOf(c.clean, { overlap_live: c.all_live }, classes, hidden)),
    accepted: live.map((g) => {
      const copy = clone(labels);
      const own = liveOf(copy, served, classes, hidden).find((x) => x.index === g.index);
      R.acceptOverlap(copy, own);
      const after = brief(liveOf(copy, served, classes, hidden));
      return { index: g.index, labels: saved(copy), after: after };
    }),
    relabelled: c.relabel.map((r) => {
      const copy = clone(labels);
      relabel(copy.boxes[r.box], r.label);
      return brief(liveOf(copy, served, classes, hidden));
    }),
  };
};
console.log(JSON.stringify({
  constants: {
    STACK_IOU: R.STACK_IOU,
    OVERLAP_IOU: R.OVERLAP_IOU,
    NESTED_SHARE: R.NESTED_SHARE,
    VEHICLES: R.VEHICLES,
  },
  frames: (req.frames || []).map((c) => frame(c, req.classes, req.hidden)),
  groups: (req.groups || []).map((c) => R.dupGroups(c.boxes, c.classes, c.hidden).groups.length),
}));
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

/// Run [`DRIVER`] on `request`, in a folder of its own, and read its answer.
fn node(request: &Value) -> Value {
    node_with(DRIVER, request)
}

/// Run `driver` on `request`, in a folder of its own, and read its answer.
fn node_with(driver_js: &str, request: &Value) -> Value {
    let dir = std::env::temp_dir().join(format!("bop-js-rules-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let (driver, cases) = (dir.join("driver.js"), dir.join("request.json"));
    std::fs::write(&driver, driver_js).unwrap();
    std::fs::write(&cases, serde_json::to_vec(request).unwrap()).unwrap();
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
    serde_json::from_slice(&out.stdout).unwrap()
}

/// xorshift64*: the generator's frames are the same on every run and OS.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() >> 33).unwrap() % n
    }

    /// True `percent` of the time.
    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    /// Uniform in `lo..hi`, to 0.001.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        let thousandths = u16::try_from(self.below(1000)).unwrap();
        lo + (hi - lo) * f32::from(thousandths) / 1000.0
    }

    fn pick<T: Copy>(&mut self, of: &[T]) -> T {
        of[self.below(of.len())]
    }
}

/// `v` to 0.1 px, as an auto-label run stores a coordinate.
fn tenth(v: f32) -> f32 {
    (v * 10.0).round() / 10.0
}

fn rounded(b: [f32; 4]) -> [f32; 4] {
    b.map(tenth)
}

fn fresh_bbox(r: &mut Rng) -> [f32; 4] {
    let (x, y) = (r.range(0.0, 1600.0), r.range(0.0, 800.0));
    let (w, h) = (r.range(30.0, 300.0), r.range(30.0, 250.0));
    rounded([x, y, x + w, y + h])
}

/// A box placed against `b`, so that every band of the rules is reached.
fn bbox_near(r: &mut Rng, b: [f32; 4]) -> [f32; 4] {
    let (w, h) = (b[2] - b[0], b[3] - b[1]);
    let out = match r.below(9) {
        // The same position: two boxes behind one suggestion key.
        0 => b,
        // A stack.
        1 => {
            let d = r.range(0.0, 0.06);
            [b[0] + w * d, b[1], b[2], b[3] - h * d]
        }
        // At a threshold, as near as tenths of a pixel get.
        2 => [b[0], b[1], b[2], b[1] + h * 0.7],
        3 => [b[0], b[1], b[2], b[1] + h * 0.5],
        // Nested, around the share that counts.
        4 => {
            let s = r.range(0.2, 0.7);
            [b[0], b[1], b[0] + w * s, b[1] + h * s]
        }
        5 => [b[2] - w * 0.3, b[1], b[2] + w * 0.05, b[1] + h * 0.5],
        // Partly over it, or the next link of a chain.
        6 => {
            let s = r.range(0.1, 0.45);
            [b[0] + w * s, b[1], b[2] + w * s, b[3]]
        }
        // An empty box.
        7 => [b[0], b[1], b[0], b[3]],
        _ => return fresh_bbox(r),
    };
    rounded(out)
}

const CLASSES: [&str; 7] = ["person", "car", "truck", "bus", "van", "dog", "cat"];
const HIDDEN: [&str; 1] = ["CAT"];

fn classes() -> Vec<String> {
    CLASSES.map(String::from).to_vec()
}

fn hidden() -> Vec<String> {
    HIDDEN.map(String::from).to_vec()
}

fn fresh_label(r: &mut Rng) -> &'static str {
    let labels = [
        "car", "car", "car", "car", "truck", "bus", "van", "person", "person", "person", "dog",
        "cat",
    ];
    r.pick(&labels)
}

fn fresh_box(r: &mut Rng, bbox: [f32; 4], label: &str) -> LabelBox {
    let source = r.pick(&[
        Source::Model,
        Source::Model,
        Source::Model,
        Source::Auto,
        Source::Auto,
        Source::Human,
    ]);
    let state = match (source, r.below(10)) {
        (_, 0..=1) => BoxState::Rejected,
        (Source::Human, 2) => BoxState::Pending,
        (Source::Human, _) => BoxState::Added,
        (_, 2..=4) => BoxState::Accepted,
        _ => BoxState::Pending,
    };
    let origin = (source == Source::Model).then(|| {
        r.pick(&[
            Origin::Prediction,
            Origin::Prediction,
            Origin::Suppressed,
            Origin::BelowThreshold,
            Origin::VerifyFound,
        ])
    });
    let mut auto = AutoInfo {
        triage: r.pick(&[None, None, Some(Triage::Accept), Some(Triage::Review)]),
        ..AutoInfo::default()
    };
    if r.chance(10) {
        auto.auto_label = Some("truck".to_string());
    }
    let mut reason = None;
    match r.below(12) {
        0 => reason = Some("night_threshold".to_string()),
        1 if state == BoxState::Accepted => {
            reason = Some(spots::PARKED_REASON.to_string());
            auto.parked = Some(ParkedMark {
                since: "2026-11-06".to_string(),
                bbox,
            });
        }
        2 if state == BoxState::Rejected => {
            reason = Some("auto_dismissed".to_string());
        }
        _ => {}
    }
    LabelBox {
        label: label.to_string(),
        bbox,
        source,
        // Tenths: ties are common, and a tie goes to the first box.
        score: r.chance(85).then(|| tenth(r.range(0.0, 1.0))),
        state,
        origin,
        reason,
        model_label: r.chance(15).then(|| r.pick(&["truck", "car"]).to_string()),
        orig_bbox: r.chance(10).then_some([1.0, 2.0, 3.0, 4.0]),
        difficult: r.chance(10),
        reject_reason: (state == BoxState::Rejected && r.chance(20))
            .then_some(RejectReason::IrGlare),
        auto,
    }
}

fn suggestion(r: &mut Rng, a: [f32; 4], b: [f32; 4]) -> Suggestion {
    let (a, b) = if r.chance(50) { (a, b) } else { (b, a) };
    Suggestion {
        a,
        b,
        verdict: r.pick(&[
            Verdict::DropA,
            Verdict::DropA,
            Verdict::DropA,
            Verdict::DropB,
            Verdict::DropB,
            Verdict::DropB,
            Verdict::TwoObjects,
            Verdict::TwoObjects,
            Verdict::Unclear,
            Verdict::Unknown,
        ]),
        why: "generated".to_string(),
        kept_score: Some(0.8),
        other_score: Some(0.0),
        low_confidence: false,
        weak: false,
    }
}

fn frame(r: &mut Rng, w: &Wanted<'_>) -> LabelFile {
    let n = 2 + r.below(7);
    let mut boxes: Vec<LabelBox> = Vec::with_capacity(n);
    for _ in 0..n {
        let near = (!boxes.is_empty() && r.chance(75)).then(|| boxes[r.below(boxes.len())].clone());
        let (bbox, label) = match &near {
            Some(b) => {
                let label = if r.chance(70) {
                    b.label.clone()
                } else {
                    fresh_label(r).to_string()
                };
                (bbox_near(r, b.bbox), label)
            }
            None => (fresh_bbox(r), fresh_label(r).to_string()),
        };
        boxes.push(fresh_box(r, bbox, &label));
    }
    // Of no class of the list: it takes part only as a rejected box (see
    // `the_differences_are_the_documented_ones`).
    if r.chance(15) {
        let bbox = boxes[0].bbox;
        let mut clock = fresh_box(r, bbox, "clock");
        clock.state = BoxState::Rejected;
        boxes.push(clock);
    }
    let mut list = Vec::new();
    for (i, j) in stacks::overlaps(&boxes, w) {
        if r.chance(70) {
            list.push(suggestion(r, boxes[i].bbox, boxes[j].bbox));
        }
    }
    // Pairs that are not open, and a key no box has.
    for _ in 0..2 {
        let (i, j) = (r.below(boxes.len()), r.below(boxes.len()));
        if i != j && r.chance(40) {
            list.push(suggestion(r, boxes[i].bbox, boxes[j].bbox));
        }
    }
    if r.chance(20) {
        list.push(suggestion(r, [9.0, 9.0, 10.0, 10.0], boxes[0].bbox));
    }
    LabelFile {
        status: r.pick(&[
            Status::Unreviewed,
            Status::Unreviewed,
            Status::Unreviewed,
            Status::Reviewed,
            Status::Reviewed,
            Status::Skipped,
        ]),
        empty_confirmed: r.chance(25),
        overlaps_checked: r.chance(10),
        boxes,
        overlap_suggestions: list,
        ..LabelFile::default()
    }
}

/// `f` without any `model_label`: what the page can judge alone (telling a mapped spelling from a
/// relabel is the server's part, `overlap_live`).
fn cleaned(f: &LabelFile) -> LabelFile {
    let mut c = f.clone();
    for b in &mut c.boxes {
        b.model_label = None;
    }
    c
}

/// Indices the server would let the page judge if it did not look at `model_label`.
fn all_live(f: &LabelFile) -> Vec<usize> {
    if f.status == Status::Skipped {
        return Vec::new();
    }
    (0..f.overlap_suggestions.len())
        .filter(|&i| overlaps::strength(&f.overlap_suggestions[i]) > 0)
        .collect()
}

/// How near a threshold an f64 value (the page's) can be with the f32 one (the server's) on its other
/// side: both start from the same decimal text, so only the rounding of a few operations separates
/// them. Seeded frames: wider than needed changes nothing, too narrow fails on every run.
const FLOAT_BAND: f64 = 5e-5;

/// `b`'s coordinates as the page reads them: the decimal text of the file, as f64.
fn wide(b: &[f32; 4]) -> [f64; 4] {
    b.map(|v| v.to_string().parse().unwrap())
}

/// Two boxes of one object whose IoU or inside share is on a threshold, to within [`FLOAT_BAND`]: the
/// third documented difference, so such a frame says nothing about the rules.
fn on_a_threshold(f: &LabelFile) -> bool {
    let near = |v: f64, t: f32| {
        let t: f64 = t.to_string().parse().unwrap();
        (v - t).abs() < FLOAT_BAND
    };
    f.boxes.iter().enumerate().any(|(i, x)| {
        f.boxes[..i].iter().any(|y| {
            if !repeats::same_object(&x.label, &y.label) {
                return false;
            }
            let (a, b) = (wide(&x.bbox), wide(&y.bbox));
            let inter = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0)
                * (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
            let (s, t) = ((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]));
            let union = s + t - inter;
            let iou = if union > 0.0 { inter / union } else { 0.0 };
            let small = s.min(t);
            let share = if small > 0.0 { inter / small } else { 0.0 };
            near(iou, stacks::STACK_IOU)
                || near(iou, stacks::OVERLAP_IOU)
                || near(share, stacks::NESTED_SHARE)
        })
    })
}

/// Two boxes at one position, one with a `model_label`: the server may pass over the relabelled box
/// for its twin at the same key, and the page must take the same box (RR-T9).
fn key_shared_with_a_relabel(f: &LabelFile) -> bool {
    f.boxes.iter().enumerate().any(|(i, x)| {
        f.boxes[..i]
            .iter()
            .any(|y| x.bbox == y.bbox && (x.model_label.is_some() || y.model_label.is_some()))
    })
}

/// `(index, a, b, drop, keep)` of a live suggestion, the page's -1 for no drop.
type Tuple = (usize, i64, i64, i64, i64);

/// The [`Tuple`] of `l`.
fn tuple(l: &overlaps::Live<'_>) -> Tuple {
    let (a, b) = (i64::try_from(l.a).unwrap(), i64::try_from(l.b).unwrap());
    let (drop, keep) = match l.suggestion.verdict {
        Verdict::DropA => (a, b),
        Verdict::DropB => (b, a),
        _ => (-1, -1),
    };
    (l.index, a, b, drop, keep)
}

fn rust_live(f: &LabelFile, w: &Wanted<'_>) -> Vec<Tuple> {
    overlaps::live(f, Some(w)).iter().map(tuple).collect()
}

fn js_live(v: &Value) -> Vec<Tuple> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|g| {
            (
                usize::try_from(g["index"].as_u64().unwrap()).unwrap(),
                g["a"].as_i64().unwrap(),
                g["b"].as_i64().unwrap(),
                g["drop"].as_i64().unwrap(),
                g["keep"].as_i64().unwrap(),
            )
        })
        .collect()
}

/// The page after an edit (`page`) against [`overlaps::live`] on the edited file (`rust`), for the
/// suggestions it was told of (`served`, as `overlap_live_boxes` named them). The page holds each one
/// to the boxes named: where `live` keeps them, so does the page; where `live` drops the suggestion,
/// so does the page; where `live` now takes another pair at the same key (a twin of an edited box),
/// the page offers the named pair or nothing, and this answers false (the documented difference).
fn held_to_named(page: &[Tuple], rust: &[Tuple], served: &[Tuple], what: &str) -> bool {
    let mut same = true;
    for p in page {
        assert!(
            served.iter().any(|t| t.0 == p.0),
            "{what}: the page offers {p:?}, which it was not told of"
        );
    }
    for t in served {
        let (p, r) = (
            page.iter().find(|x| x.0 == t.0),
            rust.iter().find(|x| x.0 == t.0),
        );
        match r {
            Some(r) if r == t => assert_eq!(p, Some(t), "{what}: {page:?} against {rust:?}"),
            None => assert_eq!(p, None, "{what}: {page:?} against {rust:?}"),
            Some(_) => {
                assert!(
                    p.is_none() || p == Some(t),
                    "{what}: {page:?} against {rust:?}"
                );
                same = false;
            }
        }
    }
    same
}

fn indices(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|i| usize::try_from(i.as_u64().unwrap()).unwrap())
        .collect()
}

fn file_of(v: &Value) -> LabelFile {
    serde_json::from_value(v.clone()).unwrap()
}

/// A page relabel of box `box_i` to `label`: the file a save would leave.
fn relabelled(f: &LabelFile, box_i: usize, label: &str) -> LabelFile {
    let mut c = f.clone();
    let b = &mut c.boxes[box_i];
    if b.source != Source::Human && b.model_label.is_none() {
        b.model_label = Some(b.label.clone());
    }
    b.label = label.to_string();
    if b.source != Source::Human {
        b.state = BoxState::Accepted;
    }
    c
}

/// The drop suggestions on a vehicle: relabelling the box to drop keeps the pair open, so only the
/// `relabelled` guard can end it. `{box, label}` for the driver.
fn relabels(f: &LabelFile, w: &Wanted<'_>) -> Vec<(usize, &'static str)> {
    overlaps::live(f, Some(w))
        .iter()
        .filter_map(|l| {
            let drop = match l.suggestion.verdict {
                Verdict::DropA => l.a,
                Verdict::DropB => l.b,
                _ => return None,
            };
            let label = f.boxes[drop].label.as_str();
            repeats::VEHICLES
                .contains(&label)
                .then_some((drop, if label == "bus" { "van" } else { "bus" }))
        })
        .collect()
}

/// What the driver gets for frame `f`: `labels`, `overlap_live` and `overlap_live_boxes` as
/// `GET /v1/review/labels` answers.
fn request_of(f: &LabelFile, w: &Wanted<'_>) -> Value {
    let moves: Vec<Value> = relabels(f, w)
        .iter()
        .map(|(i, l)| json!({"box": i, "label": l}))
        .collect();
    let live = overlaps::live(f, Some(w));
    let page: Vec<usize> = live.iter().map(|l| l.index).collect();
    let boxes: Vec<Value> = live
        .iter()
        .map(|l| json!({"index": l.index, "a": l.a, "b": l.b}))
        .collect();
    json!({
        "labels": f,
        "overlap_live": page,
        "overlap_live_boxes": boxes,
        "clean": cleaned(f),
        "all_live": all_live(f),
        "relabel": moves,
    })
}

#[test]
fn the_rules_agree_with_their_rust_twins() {
    if !have_node() {
        return;
    }
    let (c, h) = (classes(), hidden());
    let w = Wanted {
        classes: &c,
        hidden: &h,
    };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let frames: Vec<LabelFile> = (0..3000).map(|_| frame(&mut rng, &w)).collect();
    let request: Vec<Value> = frames.iter().map(|f| request_of(f, &w)).collect();
    let answer = node(&json!({"classes": c, "hidden": h, "frames": request}));
    let answers = answer["frames"].as_array().unwrap();
    assert_eq!(answers.len(), frames.len());

    // What the frames reached: a generator that stopped reaching a rule would prove nothing.
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    for (n, (f, a)) in frames.iter().zip(answers).enumerate() {
        if on_a_threshold(f) {
            *seen.entry("on a threshold").or_default() += 1;
            continue;
        }
        let got: Vec<Value> = f
            .boxes
            .iter()
            .map(|b| serde_json::to_value(resolve_pending(b)).unwrap())
            .collect();
        assert_eq!(a["resolved"], Value::Array(got), "frame {n}: resolved");

        let found = stacks::stacks(&f.boxes, &w);
        let want: Vec<Value> = found
            .iter()
            .map(|s| json!({"members": s.members, "keeper": s.keeper}))
            .collect();
        assert_eq!(a["groups"], Value::Array(want), "frame {n}: stacks {f:?}");

        let open: Vec<[usize; 2]> = stacks::overlaps(&f.boxes, &w)
            .iter()
            .map(|&(i, j)| [i, j])
            .collect();
        assert_eq!(a["overlaps"], json!(open), "frame {n}: overlaps {f:?}");

        let drops = stacks::drops(&found, &f.boxes);
        assert_eq!(indices(&a["drops"]), drops, "frame {n}: drops");

        let mut marked = f.clone();
        let rejected = stacks::reject_pending_stacked(&mut marked, &w);
        assert_eq!(
            indices(&a["pending_drops"]).len(),
            rejected,
            "frame {n}: pending drops"
        );
        assert_eq!(
            file_of(&a["pending_marked"]),
            marked,
            "frame {n}: a pending duplicate"
        );

        let live = rust_live(f, &w);

        // The page alone, handed every candidate, must reach the server's answer by itself.
        let clean = cleaned(f);
        assert_eq!(
            js_live(&a["own_live"]),
            rust_live(&clean, &w),
            "frame {n}: the page's own checks {clean:?}"
        );

        for (key, hit) in [
            ("stack", !found.is_empty()),
            ("chain", found.iter().any(|s| s.members.len() >= 3)),
            ("overlap", !open.is_empty()),
            ("pending duplicate", rejected > 0),
            ("live", !live.is_empty()),
            ("a live key shared", {
                f.boxes
                    .iter()
                    .enumerate()
                    .any(|(i, b)| f.boxes[..i].iter().any(|o| o.bbox == b.bbox))
                    && !live.is_empty()
            }),
            ("checked", f.overlaps_checked && !open.is_empty()),
        ] {
            if hit {
                *seen.entry(key).or_default() += 1;
            }
        }

        if key_shared_with_a_relabel(f) && !live.is_empty() {
            *seen.entry("live key shared with a relabel").or_default() += 1;
        }
        assert_eq!(js_live(&a["live"]), live, "frame {n}: as served {f:?}");

        for (accepted, l) in a["accepted"]
            .as_array()
            .unwrap()
            .iter()
            .zip(overlaps::live(f, Some(&w)))
        {
            let index = usize::try_from(accepted["index"].as_u64().unwrap()).unwrap();
            assert_eq!(index, l.index, "frame {n}");
            let mut mine = f.clone();
            assert!(overlaps::accept(&mut mine, l.suggestion, &w));
            assert_eq!(
                file_of(&accepted["labels"]),
                mine,
                "frame {n}: accepting suggestion {index}"
            );
            // What stays on offer: of what the page was told, minus a two-object answer it took
            // (`_done`; on the server that is `overlaps_checked` once saved).
            let two_objects = l.suggestion.verdict == Verdict::TwoObjects;
            let offered: Vec<_> = live
                .iter()
                .filter(|t| !(two_objects && t.0 == index))
                .copied()
                .collect();
            if !held_to_named(
                &js_live(&accepted["after"]),
                &rust_live(&mine, &w),
                &offered,
                &format!("frame {n}: left after {index}"),
            ) {
                *seen.entry("twin after an edit").or_default() += 1;
            }
            *seen
                .entry(if two_objects { "two objects" } else { "drop" })
                .or_default() += 1;
            if f.empty_confirmed {
                *seen.entry("confirmed empty cleared").or_default() += 1;
            }
            let promoted = f
                .boxes
                .iter()
                .zip(&mine.boxes)
                .any(|(x, y)| x.state == BoxState::Pending && y.state == BoxState::Accepted);
            if promoted {
                *seen.entry("hint promoted").or_default() += 1;
            }
            let parked_drop = f.boxes.iter().zip(&mine.boxes).any(|(x, y)| {
                x.reason.as_deref() == Some(spots::PARKED_REASON) && y.state == BoxState::Rejected
            });
            if parked_drop {
                *seen.entry("parked box dropped").or_default() += 1;
            }
        }

        let moves = relabels(f, &w);
        for (page, (i, label)) in a["relabelled"].as_array().unwrap().iter().zip(&moves) {
            let moved = relabelled(f, *i, label);
            if !held_to_named(
                &js_live(page),
                &rust_live(&moved, &w),
                &live,
                &format!("frame {n}: relabel of box {i}"),
            ) {
                *seen.entry("twin after an edit").or_default() += 1;
            }
            *seen.entry("relabel").or_default() += 1;
        }
    }
    println!("{seen:?}");
    let skipped = seen.get("on a threshold").copied().unwrap_or(0);
    assert!(
        skipped < frames.len() / 4,
        "{skipped} of {} frames left out",
        frames.len()
    );
    for (key, least) in [
        ("stack", 500),
        ("chain", 50),
        ("overlap", 500),
        ("pending duplicate", 100),
        ("live", 300),
        ("a live key shared", 20),
        ("checked", 20),
        ("drop", 100),
        ("two objects", 30),
        ("confirmed empty cleared", 30),
        ("hint promoted", 10),
        ("parked box dropped", 3),
        ("relabel", 20),
        ("live key shared with a relabel", 20),
        ("twin after an edit", 20),
    ] {
        let n = seen.get(key).copied().unwrap_or(0);
        assert!(n >= least, "the generator reached \"{key}\" only {n} times");
    }
}

#[test]
fn the_constants_are_the_rust_ones() {
    if !have_node() {
        return;
    }
    let got = &node(&json!({}))["constants"];
    // By their decimal text: `0.7_f32` and JavaScript's `0.7` are not one value, but one constant.
    assert_eq!(
        got["STACK_IOU"].to_string(),
        stacks::STACK_IOU.to_string(),
        "STACK_IOU"
    );
    assert_eq!(
        got["OVERLAP_IOU"].to_string(),
        stacks::OVERLAP_IOU.to_string(),
        "OVERLAP_IOU"
    );
    assert_eq!(
        got["NESTED_SHARE"].to_string(),
        stacks::NESTED_SHARE.to_string(),
        "NESTED_SHARE"
    );
    assert_eq!(got["VEHICLES"], json!(repeats::VEHICLES));
}

/// The twins differ in three places (`stacks.rs`), and the page holds a suggestion to the boxes the
/// server named (`overlaps.rs`, checked by `held_to_named`). Pinned, so that a change on either side
/// is a decision.
#[test]
fn the_differences_are_the_documented_ones() {
    if !have_node() {
        return;
    }
    let plain = |label: &str, bbox: [f32; 4]| LabelBox {
        label: label.to_string(),
        bbox,
        source: Source::Auto,
        score: Some(0.5),
        state: BoxState::Pending,
        origin: None,
        reason: None,
        model_label: None,
        orig_bbox: None,
        difficult: false,
        reject_reason: None,
        auto: AutoInfo::default(),
    };
    let (a, near) = ([0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 95.0]);

    // A live box of no listed class: the page shows it, a save refuses it until it is relabelled.
    let (listed, none) = (vec!["car".to_string()], Vec::<String>::new());
    let clocks = vec![plain("clock", a), plain("clock", near)];
    let w = Wanted {
        classes: &listed,
        hidden: &none,
    };
    assert!(stacks::stacks(&clocks, &w).is_empty());

    // An alias of a hidden class: the page knows no aliases.
    let plants = vec!["potted plant".to_string()];
    let aliased = vec![plain("pottedplant", a), plain("pottedplant", near)];
    let w = Wanted {
        classes: &plants,
        hidden: &plants,
    };
    assert!(stacks::stacks(&aliased, &w).is_empty());

    // Exactly on a threshold, by the file's decimals: 104.3 / 149 is 0.7, f32 lands on its side of
    // it and f64 under it.
    let tall = [427.2, 5.6, 669.4, 154.6];
    let short = [427.2, 5.6, 669.4, 109.9];
    let on_it = vec![plain("car", tall), plain("car", short)];
    let w = Wanted {
        classes: &listed,
        hidden: &none,
    };
    assert_eq!(stacks::stacks(&on_it, &w).len(), 1);

    let got = node(&json!({"groups": [
        {"boxes": clocks, "classes": listed, "hidden": none},
        {"boxes": aliased, "classes": plants, "hidden": plants},
        {"boxes": on_it, "classes": listed, "hidden": none},
    ]}));
    assert_eq!(got["groups"], json!([1, 1, 0]));
}

/// Two boxes at one position, the first relabelled by a reviewer before the page opened, and a
/// drop keyed to that position (RR-T9).
fn twin_frame() -> LabelFile {
    let plain = |bbox: [f32; 4]| LabelBox {
        label: "car".to_string(),
        bbox,
        source: Source::Auto,
        score: Some(0.5),
        state: BoxState::Pending,
        origin: None,
        reason: None,
        model_label: None,
        orig_bbox: None,
        difficult: false,
        reject_reason: None,
        auto: AutoInfo::default(),
    };
    let keep = [0.0, 0.0, 100.0, 100.0];
    let inner = [10.0, 10.0, 60.0, 60.0];
    let mut relabelled = plain(inner);
    relabelled.model_label = Some("truck".to_string());
    let mut hit = plain(inner);
    hit.state = BoxState::Accepted;
    LabelFile {
        boxes: vec![relabelled, hit, plain(keep)],
        overlap_suggestions: vec![Suggestion {
            a: keep,
            b: inner,
            verdict: Verdict::DropB,
            why: String::new(),
            kept_score: Some(0.8),
            other_score: Some(0.0),
            low_confidence: false,
            weak: false,
        }],
        ..LabelFile::default()
    }
}

/// RR-T9: the server passes over the relabelled box for its twin, and names the boxes it chose
/// (`overlap_live_boxes`); the page takes those, never the relabelled box. Told only "suggestion 0 is
/// live" (an old server), it still takes the first box at the key.
#[test]
fn the_page_never_drops_a_relabelled_twin() {
    if !have_node() {
        return;
    }
    let f = twin_frame();
    assert!(key_shared_with_a_relabel(&f));
    let (classes, none) = (classes(), Vec::<String>::new());
    let w = Wanted {
        classes: &classes,
        hidden: &none,
    };
    assert_eq!(rust_live(&f, &w), [(0, 2, 1, 1, 2)]);
    let served = request_of(&f, &w);
    let mut old = served.clone();
    old.as_object_mut().unwrap().remove("overlap_live_boxes");
    let got = node(&json!({
        "classes": classes,
        "hidden": none,
        "frames": [served, old],
    }));
    assert_eq!(js_live(&got["frames"][0]["live"]), [(0, 2, 1, 1, 2)]);
    assert_eq!(js_live(&got["frames"][1]["live"]), [(0, 2, 0, 0, 2)]);
}

/// A box removed on the page before the pair (a drawn box deleted, `removeBox`) shifts the indices:
/// the suggestion follows the boxes the server named, through an undo too, and a box drawn at the
/// same position later is not one of them.
#[test]
fn a_suggestion_follows_its_boxes_when_the_page_removes_one_before_them() {
    if !have_node() {
        return;
    }
    let mut f = twin_frame();
    let mut drawn = f.boxes[2].clone();
    drawn.source = Source::Human;
    drawn.state = BoxState::Added;
    drawn.bbox = [300.0, 300.0, 340.0, 340.0];
    f.boxes.insert(0, drawn);
    let (classes, none) = (classes(), Vec::<String>::new());
    let w = Wanted {
        classes: &classes,
        hidden: &none,
    };
    assert_eq!(rust_live(&f, &w), [(0, 3, 2, 2, 3)]);
    let driver = r#"
const fs = require('fs');
require(process.argv[2]);
const R = globalThis.ReviewRules;
const req = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const c = req.frame;
const live = (labels) => R.liveSuggestions(
  { labels: labels, overlap_live: c.overlap_live, overlap_live_boxes: c.overlap_live_boxes },
  R.dupGroups(labels.boxes, req.classes, req.hidden),
).map((g) => [g.index, g.a, g.b, g.drop, g.keep]);
const labels = c.labels;
R.markServed(labels);
const out = { served: live(labels) };
labels.boxes.splice(0, 1);
out.removed = live(labels);
out.undone = live(JSON.parse(JSON.stringify(labels)));
// The named box goes too; one drawn at its place is not it.
const twin = labels.boxes.splice(1, 1)[0];
labels.boxes.push({ label: twin.label, bbox: twin.bbox, source: 'human', state: 'added' });
out.redrawn = live(labels);
process.stdout.write(JSON.stringify(out));
"#;
    let got = node_with(
        driver,
        &json!({"classes": classes, "hidden": none, "frame": request_of(&f, &w)}),
    );
    assert_eq!(got["served"], json!([[0, 3, 2, 2, 3]]));
    assert_eq!(got["removed"], json!([[0, 2, 1, 1, 2]]));
    assert_eq!(got["undone"], json!([[0, 2, 1, 1, 2]]));
    assert_eq!(got["redrawn"], json!([]));
}

/// What a new `loadNext` does in every state (`assets/review-rules.js` `nextPlan`): at most one real
/// walk, the newest intent wins, a counts refresh never cancels a real Next.
#[test]
fn a_new_next_never_piles_up_and_a_refresh_never_cancels_a_real_next() {
    if !have_node() {
        return;
    }
    let driver = r#"
const fs = require('fs');
require(process.argv[2]);
const req = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
process.stdout.write(JSON.stringify(
  req.map(([r, w, want]) => globalThis.ReviewRules.nextPlan(r, w, want))));
"#;
    let table = [
        (None, None, "real", "start"),
        (None, None, "counts", "start"),
        // A refresh runs: a real Next does not wait for its walk.
        (Some("counts"), None, "real", "jump"),
        (Some("counts"), Some("counts"), "real", "jump"),
        // The running answer may predate what the refresh is for.
        (Some("counts"), None, "counts", "queue"),
        (Some("counts"), Some("counts"), "counts", "queue"),
        // A real walk runs: the newest real intent waits for it, one place.
        (Some("real"), None, "real", "queue"),
        (Some("real"), Some("real"), "real", "queue"),
        (Some("real"), Some("counts"), "real", "queue"),
        (Some("real"), None, "counts", "queue"),
        (Some("real"), Some("counts"), "counts", "queue"),
        // A real Next waits: its answer carries the counts, and it stays.
        (Some("real"), Some("real"), "counts", "drop"),
    ];
    let request: Vec<Value> = table
        .iter()
        .map(|(r, w, want, _)| json!([r, w, want]))
        .collect();
    let got = node_with(driver, &json!(request));
    let want: Vec<&str> = table.iter().map(|t| t.3).collect();
    assert_eq!(got, json!(want));
}

/// What a landed `loadNext` answer shows (`assets/review-rules.js` `nextLanding`): a counts refresh
/// keeps the frame on screen, but with nothing on screen it brings the frame it found, which is how
/// the 2 s refresh while a big folder is indexed shows the first frame to come in.
#[test]
fn a_refresh_keeps_the_frame_on_screen_and_brings_one_when_there_is_none() {
    if !have_node() {
        return;
    }
    let driver = r#"
const fs = require('fs');
require(process.argv[2]);
const req = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
process.stdout.write(JSON.stringify(
  req.map(([kind, replaced, shown]) => globalThis.ReviewRules.nextLanding(kind, replaced, shown))));
"#;
    let table = [
        // A real Next shows its frame, whatever was there.
        ("real", false, true, "frame"),
        ("real", false, false, "frame"),
        // A newer real intent replaced it: that one lands.
        ("real", true, true, "counts"),
        ("real", true, false, "counts"),
        // A counts refresh keeps the frame on screen and its edits.
        ("counts", false, true, "counts"),
        // Nothing on screen (indexing, or no frame matched): it shows what it found.
        ("counts", false, false, "frame"),
        ("counts", true, false, "counts"),
        ("counts", true, true, "counts"),
    ];
    let request: Vec<Value> = table.iter().map(|(k, r, s, _)| json!([k, r, s])).collect();
    let got = node_with(driver, &json!(request));
    let want: Vec<&str> = table.iter().map(|t| t.3).collect();
    assert_eq!(got, json!(want));
}

/// The dataset check's twins on one frame each: `liveRefine`, `refineStrength`, `acceptRefine` of
/// every live suggestion (with what is live after it), `acceptRefine` of every other index, and
/// `dismissRefine` of every index and one past the end. `process.argv` as [`DRIVER`].
const REFINE_DRIVER: &str = r#"
const fs = require('fs');
require(process.argv[2]);
const R = globalThis.ReviewRules;
const req = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const clone = (v) => JSON.parse(JSON.stringify(v));
const out = req.cases.map((c) => {
  const classes = c.wanted ? req.classes : null;
  const hidden = c.wanted ? req.hidden : null;
  const labels = c.labels;
  const list = labels.refine || [];
  const live = R.liveRefine(labels, classes, hidden);
  const accepted = live.map((l) => {
    const copy = clone(labels);
    const ok = R.acceptRefine(copy, l.index, classes, hidden);
    return { index: l.index, ok: ok, labels: copy, after: R.liveRefine(copy, classes, hidden) };
  });
  const refused = list.map((_, i) => i).filter((i) => !live.some((l) => l.index === i)).map((i) => {
    const copy = clone(labels);
    const ok = R.acceptRefine(copy, i, classes, hidden);
    return ok || JSON.stringify(copy) !== JSON.stringify(labels);
  });
  const dismissed = list.concat([null]).map((_, i) => {
    const copy = clone(labels);
    const ok = R.dismissRefine(copy, i);
    return { ok: ok, labels: copy };
  });
  return {
    live: live,
    strengths: list.map(R.refineStrength),
    accepted: accepted,
    refused: refused,
    dismissed: dismissed,
  };
});
console.log(JSON.stringify({
  constants: {
    MATCH_IOU: R.MATCH_IOU,
    COVER_IOU: R.COVER_IOU,
    COVER_INSIDE: R.COVER_INSIDE,
    MAX_BOXES: R.MAX_BOXES,
    MAX_LABEL_LEN: R.MAX_LABEL_LEN,
    REFINE_REASON: R.REFINE_REASON,
    COCO_NAMES: R.COCO_NAMES,
    COCO_ALIASES: R.COCO_ALIASES,
  },
  cases: out,
}));
"#;

/// One entry of `refine_live` (`GET /v1/review/labels`) and of `liveRefine`, numbers as the f32
/// they are in the file. Every field named: [`same_fields`] holds the key sets equal, so a field
/// renamed or dropped on either side fails rather than reading as null.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RefineLive {
    index: usize,
    kind: String,
    target: Option<[f32; 4]>,
    target_label: Option<String>,
    bbox: Option<[f32; 4]>,
    label: Option<String>,
    score: Option<f32>,
    why: String,
    spot: Option<String>,
    run: String,
    target_box: Option<usize>,
    replaces: Vec<usize>,
}

/// [`refine::live`] as the route serves it: [`refine::live_json`], the builder
/// `GET /v1/review/labels` calls for `refine_live`.
fn rust_refine_live(f: &LabelFile, w: Option<&Wanted<'_>>) -> Vec<RefineLive> {
    serde_json::from_value(Value::Array(refine::live_json(f, w))).unwrap()
}

/// The keys of each entry of `refine_live` for `f` equal those of the page's `liveRefine`
/// (`js`), entry by entry.
fn same_fields(f: &LabelFile, w: Option<&Wanted<'_>>, js: &Value, what: &str) {
    let keys = |v: &Value| -> Vec<String> {
        let mut k: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
        k.sort();
        k
    };
    let route = refine::live_json(f, w);
    let page = js.as_array().unwrap();
    assert_eq!(route.len(), page.len(), "{what}");
    for (r, p) in route.iter().zip(page) {
        assert_eq!(
            keys(r),
            keys(p),
            "{what}: the fields of refine_live and liveRefine"
        );
    }
}

fn js_refine_live(v: &Value) -> Vec<RefineLive> {
    serde_json::from_value(v.clone()).unwrap()
}

const REFINE_CLASSES: [&str; 9] = [
    "person",
    "car",
    "truck",
    "bus",
    "van",
    "dog",
    "cat",
    "motorcycle",
    "potted plant",
];

/// Box labels: the list's spellings, other cases, white space, aliases, a hidden class, a class of
/// no list.
const REFINE_LABELS: [&str; 16] = [
    "car",
    "car",
    "Car",
    " truck",
    "bus",
    "van",
    "person",
    "person",
    "PERSON",
    "dog",
    "cat",
    "motorbike",
    "motorcycle",
    "pottedplant",
    "clock",
    "dog\u{2003}",
];

fn refine_box(r: &mut Rng, bbox: [f32; 4], label: &str) -> LabelBox {
    let mut b = fresh_box(r, bbox, label);
    if b.state == BoxState::Rejected {
        // Every kind of reject: unjudged ones a found object may replace, judged ones that block
        // the place, ignore regions that cover it.
        b.reason = r
            .pick(&[
                None,
                Some("low_score"),
                Some("low_score"),
                Some("auto_dismissed"),
                Some("spot_dismissed"),
                Some("size_floor"),
                Some("duplicate"),
                Some("repeat_spot"),
                Some("night_threshold"),
            ])
            .map(String::from);
        b.reject_reason = match r.below(6) {
            3 => Some(RejectReason::TooSmall),
            4 => Some(RejectReason::FalsePositive),
            5 => Some(RejectReason::Duplicate),
            _ => None,
        };
    }
    b
}

/// `label` as a check may have written the target's label: as is, in another case, padded.
fn respelled(r: &mut Rng, label: &str) -> String {
    match r.below(3) {
        0 => label.to_string(),
        1 => label.to_ascii_uppercase(),
        _ => format!(" {label} "),
    }
}

fn refine_suggestion(r: &mut Rng, boxes: &[LabelBox]) -> refine::Suggestion {
    let kind = r.pick(&[
        refine::Kind::Add,
        refine::Kind::Add,
        refine::Kind::Add,
        refine::Kind::Tighten,
        refine::Kind::Tighten,
        refine::Kind::Relabel,
        refine::Kind::Relabel,
        refine::Kind::Verify,
        refine::Kind::Verify,
        refine::Kind::Unknown,
    ]);
    let rejected: Vec<&LabelBox> = boxes
        .iter()
        .filter(|b| b.state == BoxState::Rejected)
        .collect();
    // An add mostly where a box was rejected: a place it may take, or one a reviewer judged.
    let on = if kind == refine::Kind::Add && !rejected.is_empty() && r.chance(60) {
        Some(rejected[r.below(rejected.len())].clone())
    } else {
        (!boxes.is_empty() && r.chance(85)).then(|| boxes[r.below(boxes.len())].clone())
    };
    let target = on.as_ref().map_or_else(|| fresh_bbox(r), |b| b.bbox);
    let target_label = match (&on, r.below(10)) {
        (Some(b), 0..=3) => Some(respelled(r, &b.label)),
        (_, 4) => Some(r.pick(&REFINE_LABELS).to_string()),
        _ => None,
    };
    let add = kind == refine::Kind::Add;
    refine::Suggestion {
        kind,
        target: (!add).then_some(target),
        target_label: if add { None } else { target_label },
        bbox: match kind {
            refine::Kind::Add | refine::Kind::Tighten | refine::Kind::Unknown => {
                Some(bbox_near(r, target))
            }
            _ => None,
        },
        label: match (kind, &on) {
            (refine::Kind::Add, Some(b)) if r.chance(60) => Some(respelled(r, &b.label)),
            (refine::Kind::Add | refine::Kind::Relabel, _) => {
                Some(r.pick(&REFINE_LABELS).to_string())
            }
            _ => None,
        },
        score: r.chance(85).then(|| tenth(r.range(0.0, 1.0))),
        why: "generated".to_string(),
        spot: r.chance(20).then(|| "gate:12".to_string()),
        answer: r
            .chance(12)
            .then(|| r.pick(&[refine::Answer::Dismissed, refine::Answer::Kept])),
        run: "r1".to_string(),
    }
}

fn refine_frame(r: &mut Rng) -> LabelFile {
    let n = r.below(8);
    let mut boxes: Vec<LabelBox> = Vec::with_capacity(n + 1);
    for _ in 0..n {
        let near = (!boxes.is_empty() && r.chance(70)).then(|| boxes[r.below(boxes.len())].clone());
        let (bbox, label) = match &near {
            Some(b) => {
                let label = if r.chance(60) {
                    b.label.clone()
                } else {
                    r.pick(&REFINE_LABELS).to_string()
                };
                (bbox_near(r, b.bbox), label)
            }
            None => (fresh_bbox(r), r.pick(&REFINE_LABELS).to_string()),
        };
        boxes.push(refine_box(r, bbox, &label));
    }
    // Rejects no reviewer judged, stacked on one object: an add takes the closest one's place.
    if r.chance(30) {
        let base = fresh_bbox(r);
        let label = r.pick(&["car", "person", "dog", "truck"]);
        for _ in 0..2 + r.below(2) {
            let bbox = bbox_near(r, base);
            let mut b = refine_box(r, bbox, label);
            b.state = BoxState::Rejected;
            b.reject_reason = None;
            b.reason = Some(r.pick(&["low_score", "auto_dismissed"]).to_string());
            let at = r.below(boxes.len() + 1);
            boxes.insert(at, b);
        }
    }
    // Two boxes of different objects at one position: only `target_label` tells them apart.
    if !boxes.is_empty() && r.chance(20) {
        let mut twin = boxes[r.below(boxes.len())].clone();
        twin.label = r.pick(&["dog", "person", "truck"]).to_string();
        twin.state = BoxState::Accepted;
        twin.source = Source::Auto;
        twin.orig_bbox = None;
        twin.model_label = None;
        let at = r.below(boxes.len() + 1);
        boxes.insert(at, twin);
    }
    let refine: Vec<refine::Suggestion> = (0..1 + r.below(5))
        .map(|_| refine_suggestion(r, &boxes))
        .collect();
    LabelFile {
        status: r.pick(&[
            Status::Reviewed,
            Status::Reviewed,
            Status::Reviewed,
            Status::Unreviewed,
            Status::Skipped,
        ]),
        empty_confirmed: r.chance(25),
        overlaps_checked: r.chance(40),
        boxes,
        refine,
        ..LabelFile::default()
    }
}

/// A frame at [`MAX_BOXES`](blue_onyx_prism::collect::labels::MAX_BOXES): an add has no room
/// unless it takes a reject's place.
fn full_frame(with_reject: bool) -> LabelFile {
    let tiny = |i: usize| {
        let x = f32::from(u16::try_from(i % 40).unwrap()) * 10.0;
        let y = f32::from(u16::try_from(i / 40).unwrap()) * 10.0;
        LabelBox {
            label: "person".to_string(),
            bbox: [x, y, x + 5.0, y + 5.0],
            source: Source::Auto,
            score: Some(0.5),
            state: BoxState::Accepted,
            origin: None,
            reason: None,
            model_label: None,
            orig_bbox: None,
            difficult: false,
            reject_reason: None,
            auto: AutoInfo::default(),
        }
    };
    let mut boxes: Vec<LabelBox> = (0..blue_onyx_prism::collect::labels::MAX_BOXES)
        .map(tiny)
        .collect();
    let car = [600.0, 600.0, 700.0, 680.0];
    if with_reject {
        boxes[0] = LabelBox {
            label: "car".to_string(),
            bbox: car,
            state: BoxState::Rejected,
            reason: Some("low_score".to_string()),
            ..tiny(0)
        };
    }
    LabelFile {
        status: Status::Reviewed,
        boxes,
        refine: vec![refine::Suggestion {
            kind: refine::Kind::Add,
            target: None,
            target_label: None,
            bbox: Some([601.0, 600.0, 700.0, 680.0]),
            label: Some("car".to_string()),
            score: Some(0.9),
            why: String::new(),
            spot: None,
            answer: None,
            run: String::new(),
        }],
        ..LabelFile::default()
    }
}

#[test]
fn the_refine_rules_agree_with_their_rust_twins() {
    if !have_node() {
        return;
    }
    let classes: Vec<String> = REFINE_CLASSES.map(String::from).to_vec();
    let hidden: Vec<String> = vec!["CAT".to_string()];
    let w = Wanted {
        classes: &classes,
        hidden: &hidden,
    };
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut cases: Vec<(LabelFile, bool)> = (0..3000)
        .map(|_| {
            let f = refine_frame(&mut rng);
            let wanted = rng.chance(80);
            (f, wanted)
        })
        .collect();
    cases.push((full_frame(false), true));
    cases.push((full_frame(true), true));
    let request: Vec<Value> = cases
        .iter()
        .map(|(f, wanted)| json!({"labels": f, "wanted": wanted}))
        .collect();
    let answer = node_with(
        REFINE_DRIVER,
        &json!({"classes": classes, "hidden": hidden, "cases": request}),
    );
    let answers = answer["cases"].as_array().unwrap();
    assert_eq!(answers.len(), cases.len());

    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    for (n, ((f, wanted), a)) in cases.iter().zip(answers).enumerate() {
        let w = wanted.then_some(&w);
        let live = rust_refine_live(f, w);
        assert_eq!(js_refine_live(&a["live"]), live, "frame {n}: live {f:?}");
        same_fields(f, w, &a["live"], &format!("frame {n}"));
        let strengths: Vec<u32> = f.refine.iter().map(refine::strength).collect();
        assert_eq!(
            a["strengths"],
            json!(strengths),
            "frame {n}: strengths {:?}",
            f.refine
        );

        for (accepted, l) in a["accepted"].as_array().unwrap().iter().zip(&live) {
            let mut mine = f.clone();
            assert!(refine::accept(&mut mine, l.index, w), "frame {n}");
            assert_eq!(accepted["ok"], true, "frame {n}: accepting {}", l.index);
            assert_eq!(
                file_of(&accepted["labels"]),
                mine,
                "frame {n}: accepting {} ({}) of {f:?}",
                l.index,
                l.kind
            );
            assert_eq!(
                js_refine_live(&accepted["after"]),
                rust_refine_live(&mine, w),
                "frame {n}: left after accepting {}",
                l.index
            );
            *seen
                .entry(match l.kind.as_str() {
                    "add" => "add accepted",
                    "tighten" => "tighten accepted",
                    "relabel" => "relabel accepted",
                    _ => "verify kept",
                })
                .or_default() += 1;
            if l.replaces.len() >= 2 {
                *seen.entry("add replaces two or more").or_default() += 1;
            } else if l.replaces.len() == 1 {
                *seen.entry("add replaces one").or_default() += 1;
            }
            if f.overlaps_checked && !mine.overlaps_checked {
                *seen.entry("overlaps_checked ended").or_default() += 1;
            }
            if f.overlaps_checked && mine.overlaps_checked && l.kind != "verify" {
                *seen.entry("overlaps_checked kept").or_default() += 1;
            }
            if f.empty_confirmed && l.kind == "add" {
                *seen.entry("empty_confirmed ended").or_default() += 1;
            }
            if l.kind == "relabel"
                && l.label.as_deref() != Some(mine.boxes[l.target_box.unwrap()].label.as_str())
            {
                *seen.entry("relabel to the list's spelling").or_default() += 1;
            }
            if l.index + 1 < f.refine.len() && l.kind != "verify" {
                *seen.entry("indices shift after an accept").or_default() += 1;
            }
        }
        assert_eq!(
            a["accepted"].as_array().unwrap().len(),
            live.len(),
            "frame {n}"
        );
        // Every other index: refused, nothing changed, as in Rust.
        let refused: Vec<usize> = (0..f.refine.len())
            .filter(|i| !live.iter().any(|l| l.index == *i))
            .collect();
        for i in &refused {
            let mut mine = f.clone();
            assert!(!refine::accept(&mut mine, *i, w), "frame {n}: {i}");
            assert_eq!(&mine, f);
        }
        assert_eq!(
            a["refused"],
            json!(vec![false; refused.len()]),
            "frame {n}: an accept of a suggestion that is not live"
        );
        if !refused.is_empty() {
            *seen.entry("not live").or_default() += 1;
        }
        for (i, d) in a["dismissed"].as_array().unwrap().iter().enumerate() {
            let mut mine = f.clone();
            let ok = refine::dismiss(&mut mine, i);
            assert_eq!(d["ok"], ok, "frame {n}: dismissing {i}");
            assert_eq!(file_of(&d["labels"]), mine, "frame {n}: dismissing {i}");
            if ok {
                *seen.entry("dismissed").or_default() += 1;
            }
        }

        // What the frames reached.
        for (key, hit) in [
            ("live", !live.is_empty()),
            (
                "skipped",
                f.status == Status::Skipped && !f.refine.is_empty(),
            ),
            ("no class list", w.is_none() && !live.is_empty()),
            (
                "answered",
                f.refine.iter().any(refine::Suggestion::answered),
            ),
            (
                "unknown kind",
                f.refine.iter().any(|s| s.kind == refine::Kind::Unknown),
            ),
            (
                "a target_label picks the second box",
                live.iter().any(|l| {
                    l.target_box
                        .is_some_and(|t| f.boxes[..t].iter().any(|b| Some(b.bbox) == l.target))
                }),
            ),
            (
                "an alias or another spelling",
                live.iter().any(|l| {
                    l.label.as_deref().is_some_and(|x| {
                        [
                            "Car",
                            " truck",
                            "PERSON",
                            "motorbike",
                            "pottedplant",
                            "dog\u{2003}",
                        ]
                        .contains(&x)
                    })
                }),
            ),
        ] {
            if hit {
                *seen.entry(key).or_default() += 1;
            }
        }
        // A suggestion about an ignore region (an unsure box, a box rejected as too small) is
        // never live: the check never judges one. An unsure box that is otherwise a target ends a
        // suggestion that is live again once the mark is taken back.
        for (i, s) in f.refine.iter().enumerate() {
            if s.answered()
                || !matches!(
                    s.kind,
                    refine::Kind::Tighten | refine::Kind::Relabel | refine::Kind::Verify
                )
            {
                continue;
            }
            let names = |b: &LabelBox| {
                s.target == Some(b.bbox)
                    && s.target_label
                        .as_deref()
                        .is_none_or(|l| l.trim().eq_ignore_ascii_case(b.label.trim()))
            };
            let Some(at) = f.boxes.iter().position(|b| names(b) && b.is_ignore()) else {
                continue;
            };
            if f.boxes.iter().any(|b| names(b) && !b.is_ignore()) {
                continue; // another box it names may hold it live
            }
            assert!(
                !live.iter().any(|l| l.index == i),
                "frame {n}: suggestion {i} about an ignore region is live"
            );
            let b = &f.boxes[at];
            if b.state == BoxState::Rejected && b.reject_reason == Some(RejectReason::TooSmall) {
                *seen.entry("a target rejected as too small").or_default() += 1;
            }
            if b.difficult && b.is_training() {
                *seen.entry("a target marked unsure").or_default() += 1;
                let mut sure = f.clone();
                sure.boxes[at].difficult = false;
                if rust_refine_live(&sure, w).iter().any(|l| l.index == i) {
                    *seen.entry("unsure ends a live suggestion").or_default() += 1;
                }
            }
        }
        // A hidden class or one of no list is never live with the list, and live without one.
        if w.is_some()
            && f.refine.iter().any(|s| {
                !s.answered()
                    && s.kind == refine::Kind::Add
                    && s.label
                        .as_deref()
                        .is_some_and(|l| ["cat", "clock"].contains(&l))
            })
        {
            *seen.entry("an add of an unwanted class").or_default() += 1;
        }
    }
    // The fixed frames at the box cap: no room, then a reject's place.
    let full = &answers[cases.len() - 2];
    assert_eq!(full["live"], json!([]));
    let full = &answers[cases.len() - 1];
    assert_eq!(full["live"][0]["replaces"], json!([0]));
    println!("{seen:?}");
    for (key, least) in [
        ("live", 800),
        ("add accepted", 150),
        ("tighten accepted", 100),
        ("relabel accepted", 100),
        ("verify kept", 100),
        ("add replaces one", 20),
        ("add replaces two or more", 5),
        ("overlaps_checked ended", 20),
        ("overlaps_checked kept", 50),
        ("empty_confirmed ended", 30),
        ("relabel to the list's spelling", 20),
        ("indices shift after an accept", 100),
        ("not live", 500),
        ("dismissed", 1000),
        ("skipped", 100),
        ("no class list", 100),
        ("answered", 200),
        ("unknown kind", 100),
        ("a target_label picks the second box", 5),
        ("an alias or another spelling", 100),
        ("an add of an unwanted class", 50),
        ("a target rejected as too small", 40),
        ("a target marked unsure", 60),
        ("unsure ends a live suggestion", 20),
    ] {
        let n = seen.get(key).copied().unwrap_or(0);
        assert!(n >= least, "the generator reached \"{key}\" only {n} times");
    }

    // The constants are the Rust ones.
    let got = &answer["constants"];
    for (name, v) in [
        ("MATCH_IOU", labels::MATCH_IOU),
        ("COVER_IOU", labels::COVER_IOU),
        ("COVER_INSIDE", labels::COVER_INSIDE),
    ] {
        assert_eq!(got[name].to_string(), v.to_string(), "{name}");
    }
    assert_eq!(got["MAX_BOXES"], json!(labels::MAX_BOXES));
    assert_eq!(got["MAX_LABEL_LEN"], json!(labels::MAX_LABEL_LEN));
    assert_eq!(got["REFINE_REASON"], json!(refine::REFINE_REASON));
    assert_eq!(got["COCO_NAMES"], json!(coco::COCO_NAMES.to_vec()));
    let aliases: Vec<[&str; 2]> = coco::COCO_ALIASES.iter().map(|(a, c)| [*a, *c]).collect();
    assert_eq!(got["COCO_ALIASES"], json!(aliases));
}

/// Needs no `node`: the page loads the rules first, and they stay pure.
#[test]
fn the_page_loads_the_rules_first_and_they_touch_no_page() {
    let page = include_str!("../../templates/train/review.html");
    let (rules, own) = (
        page.find("/static/review-rules.js?v=").unwrap(),
        page.find("/static/review.js?v=").unwrap(),
    );
    assert!(rules < own, "review.js reads ReviewRules when it starts");
    let js = std::fs::read_to_string(rules_path()).unwrap();
    for word in [
        "document",
        "window",
        "fetch(",
        "localStorage",
        "navigator",
        "location",
    ] {
        assert!(!js.contains(word), "review-rules.js is pure: {word}");
    }
}

/// One case of the size floor: box, frame width and height, floor.
type FloorCase = ([f32; 4], u32, u32, Option<labels::SizeFloor>);

/// `v` moved by `k` f32 steps (`k` < 0: down); `v` positive.
fn ulps(v: f32, k: i32) -> f32 {
    f32::from_bits(v.to_bits().checked_add_signed(k).unwrap())
}

/// The cases of [`the_size_floor_agrees_with_its_rust_twin`]: the boundary of real floors on
/// non-square frames (exactly there and up to two f32 steps either side, on either axis), boxes at
/// and past the frame edge, inverted, empty and off-frame boxes, a floor that is off, a frame of
/// unknown size, and seeded random boxes.
fn floor_cases() -> Vec<FloorCase> {
    let floor = |px: f32, input: u32| Some(labels::SizeFloor { px, input });
    let mut v: Vec<FloorCase> = Vec::new();
    // A 512 px frame at input 512 maps a side 1:1 exactly: 10 px is exactly at a 10 px floor.
    for k in -2..=2 {
        let s = ulps(10.0, k);
        v.push(([0.0, 0.0, 300.0, s], 512, 512, floor(10.0, 512)));
        v.push(([0.0, 0.0, s, 300.0], 512, 512, floor(10.0, 512)));
    }
    let frames = [
        (1920, 1080),
        (640, 480),
        (2560, 1440),
        (1280, 720),
        (704, 576),
        (480, 640),
    ];
    let floors = [
        (10.0, 384),
        (8.0, 640),
        (12.5, 384),
        (1.0, 64),
        (64.0, 4096),
        (9.7, 448),
    ];
    for (w, h) in frames {
        let (wf, hf) = (w as f32, h as f32);
        for (px, input) in floors {
            let fl = floor(px, input);
            // The side that is exactly at the floor on each axis.
            let (sw, sh) = (px * wf / input as f32, px * hf / input as f32);
            for k in -2..=2 {
                let (bw, bh) = (ulps(sw, k), ulps(sh, k));
                v.push(([0.0, 0.0, wf, bh], w, h, fl));
                v.push(([0.0, 0.0, bw, hf], w, h, fl));
                // At the frame's far edge, and past it (clamped to the frame).
                v.push(([wf - bw, hf - 2.0 * bh, wf, hf], w, h, fl));
                v.push(([wf - bw, hf - bh, wf + 50.0, hf + 50.0], w, h, fl));
                v.push(([-30.0, -30.0, bw - 30.0, hf], w, h, fl));
                // Inverted corners.
                v.push(([bw, hf, 0.0, 0.0], w, h, fl));
            }
            v.push(([5.0, 5.0, 5.0, 50.0], w, h, fl));
            v.push(([wf + 10.0, 0.0, wf + 50.0, 50.0], w, h, fl));
            v.push(([0.0, 0.0, 1.0, 1.0], w, h, None));
            v.push(([0.0, 0.0, 1.0, 1.0], 0, h, fl));
            v.push(([0.0, 0.0, 1.0, 1.0], w, 0, fl));
            v.push(([0.0, 0.0, 1.0, 1.0], w, h, floor(0.0, input)));
            v.push(([0.0, 0.0, 1.0, 1.0], w, h, floor(px, 0)));
        }
    }
    let mut r = Rng(0x5153_e0f1_002d_9a3b);
    for _ in 0..4000 {
        let (w, h) = r.pick(&frames);
        let (px, input) = r.pick(&floors);
        let (x, y) = (r.range(-40.0, w as f32), r.range(-40.0, h as f32));
        let (bw, bh) = (r.range(0.0, 60.0), r.range(0.0, 60.0));
        v.push(([x, y, x + bw, y + bh], w, h, floor(px, input)));
    }
    v
}

/// `labels::under_floor` and the page's `underFloor` agree on every case, f32 for f32 (numbers go to
/// node as their f32 bits, so no decimal rounding stands between the two).
#[test]
fn the_size_floor_agrees_with_its_rust_twin() {
    if !have_node() {
        return;
    }
    let cases = floor_cases();
    let rust: Vec<bool> = cases
        .iter()
        .map(|(b, w, h, f)| labels::under_floor(b, *w, *h, *f))
        .collect();
    let request: Vec<Value> = cases
        .iter()
        .map(|(b, w, h, f)| {
            json!({
                "b": b.map(f32::to_bits),
                "w": w,
                "h": h,
                "floor": f.map(|f| json!({"px": f.px.to_bits(), "input": f.input})),
            })
        })
        .collect();
    let driver = r#"
const fs = require('fs');
require(process.argv[2]);
const R = globalThis.ReviewRules;
const req = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const f32 = (u) => { const d = new DataView(new ArrayBuffer(4)); d.setUint32(0, u); return d.getFloat32(0); };
const fl = { px: 10, input: 384 };
process.stdout.write(JSON.stringify({
  cases: req.map((c) => R.underFloor(c.b.map(f32), c.w, c.h,
    c.floor === null ? null : { px: f32(c.floor.px), input: c.floor.input })),
  // What JSON cannot carry: never under.
  odd: [
    R.underFloor([NaN, 0, 1, 1], 640, 480, fl),
    R.underFloor([0, 0, Infinity, 1], 640, 480, fl),
    R.underFloor([0, 0, 1e39, 1], 640, 480, fl),
    R.underFloor([0, 0, '1', 1], 640, 480, fl),
    R.underFloor([0, 0, 1, 1], 640, 480, { px: NaN, input: 384 }),
    R.underFloor([0, 0, 1, 1], 640, 480, undefined),
    R.underFloor([0, 0, 1], 640, 480, fl),
    R.underFloor([0, 0, 1, 1], 640, 480, fl),
  ],
}));
"#;
    let got = node_with(driver, &json!(request));
    let js: Vec<bool> = serde_json::from_value(got["cases"].clone()).unwrap();
    assert_eq!(js.len(), rust.len());
    let differ: Vec<String> = cases
        .iter()
        .zip(rust.iter().zip(&js))
        .filter(|(_, (r, j))| r != j)
        .take(10)
        .map(|(c, (r, j))| format!("{c:?}: rust {r}, page {j}"))
        .collect();
    assert!(differ.is_empty(), "{differ:#?}");
    // Both answers occur, at the boundary too: the cases test something.
    let under = rust.iter().filter(|u| **u).count();
    assert!(
        under > 500 && rust.len() - under > 500,
        "{under} of {}",
        rust.len()
    );
    assert_eq!(
        got["odd"],
        json!([false, false, false, false, false, false, false, true])
    );
    // Exactly at the floor is not under; one f32 step below is.
    let at = |s: f32| labels::under_floor(&[0.0, 0.0, 300.0, s], 512, 512, cases[0].3);
    assert!(!at(10.0) && at(ulps(10.0, -1)) && !at(ulps(10.0, 1)));
}

/// What the Review page says of a box under the size floor (`floorNote`): a chip and one line when
/// a training box is under it, nothing with no floor, a rejected or unsure box or a box over it.
/// The decision itself is `underFloor`, tested against its Rust twin above.
#[test]
fn a_box_under_the_size_floor_gets_a_chip_and_a_line_and_no_floor_gets_none() {
    if !have_node() {
        return;
    }
    let driver = r#"
const fs = require('fs');
require(process.argv[2]);
const R = globalThis.ReviewRules;
const floor = { px: 10, input: 384 };
// 640 x 480: a box 10 px wide is 6 px at 384, one 200 px wide is 120.
const tiny = (extra) => Object.assign({ bbox: [100, 100, 110, 300], state: 'accepted', difficult: false }, extra || {});
const big = { bbox: [100, 100, 300, 300], state: 'accepted', difficult: false };
process.stdout.write(JSON.stringify({
  tiny: R.floorNote(tiny(), 640, 480, floor),
  pending: R.floorNote(tiny({ state: 'pending' }), 640, 480, floor),
  drawn: R.floorNote(tiny({ state: 'added' }), 640, 480, floor),
  big: R.floorNote(big, 640, 480, floor),
  off: R.floorNote(tiny(), 640, 480, null),
  undef: R.floorNote(tiny(), 640, 480, undefined),
  badFloor: R.floorNote(tiny(), 640, 480, { px: 0, input: 384 }),
  rejected: R.floorNote(tiny({ state: 'rejected' }), 640, 480, floor),
  unsure: R.floorNote(tiny({ difficult: true }), 640, 480, floor),
  noFrame: R.floorNote(tiny(), 0, 0, floor),
  noBox: R.floorNote(null, 640, 480, floor),
  widened: R.floorNote(tiny(), 640, 480, { px: 7.300000190734863, input: 384 }),
  // Resized to 100 px wide while drawing: 60 px at 384, no longer under.
  resized: R.floorNote(tiny({ bbox: [100, 100, 200, 300] }), 640, 480, floor),
}));
"#;
    let got = node_with(driver, &json!({}));
    let line = "Under the size floor (10 px at 384): left out of the labels at export.";
    for k in ["tiny", "pending", "drawn"] {
        assert_eq!(got[k], json!({"chip": "too small", "note": line}), "{k}");
    }
    for k in [
        "big", "off", "undef", "badFloor", "rejected", "unsure", "noFrame", "noBox", "resized",
    ] {
        assert!(got[k].is_null(), "{k}: {}", got[k]);
    }
    // A box 10 px wide is 6 px at 384: under 7.3 too; the f32 floor prints without its tail.
    assert_eq!(
        got["widened"]["note"],
        "Under the size floor (7.3 px at 384): left out of the labels at export."
    );
}

/// The Refine tab's pure sentences (`RefineFloor` in `assets/train-refine.js`), loaded in Node
/// without a page: which classes the floor leaves too few labels, and the question before the rule
/// writes.
#[test]
fn the_too_small_cards_sentences_flag_a_class_the_floor_empties() {
    if !have_node() {
        return;
    }
    let driver = r#"
const fs = require('fs');
require(process.argv[2]);
const req = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
require(req.refine);
const F = globalThis.RefineFloor;
process.stdout.write(JSON.stringify({
  all: F.classNote({ class: 'motorcycle', boxes: 12, under: 12, left: 0 }),
  few: F.classNote({ class: 'person', boxes: 300, under: 270, left: 30 }),
  one: F.classNote({ class: 'bus', boxes: 3, under: 2, left: 1 }),
  enough: F.classNote({ class: 'car', boxes: 900, under: 100, left: 800 }),
  edge: F.classNote({ class: 'car', boxes: 100, under: 50, left: 50 }),
  edge49: F.classNote({ class: 'car', boxes: 99, under: 50, left: 49 }),
  untouched: F.classNote({ class: 'dog', boxes: 5, under: 0, left: 5 }),
  nothing: F.classNote(null),
  phrase: F.phrase({ px: 10, input: 384 }),
  phraseWide: F.phrase({ px: 7.300000190734863, input: 640 }),
  one_box: F.resolveText(1, 1),
  many: F.resolveText(12, 5),
  back_one: F.restoreText(1, 1),
  back_many: F.restoreText(7, 3),
  measured_floor: F.measuredWith({ folds_every_label: true, size_floor: { px: 10, input: 384 } }),
  measured_none: F.measuredWith({ folds_every_label: true }),
  measured_old: F.measuredWith({ recipe: { imgsz: 384 } }),
}));
"#;
    let refine = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/train-refine.js");
    let got = node_with(driver, &json!({"refine": refine}));
    assert_eq!(
        got["all"],
        "every motorcycle label would be left out: the model cannot learn the class from these frames"
    );
    assert_eq!(
        got["few"],
        "only 30 person labels would be left (under 50): too few to learn the class from"
    );
    assert_eq!(
        got["one"],
        "only 1 bus label would be left (under 50): too few to learn the class from"
    );
    for k in ["enough", "edge", "untouched", "nothing"] {
        assert_eq!(got[k], "", "{k}");
    }
    assert!(
        got["edge49"]
            .as_str()
            .unwrap()
            .starts_with("only 49 car labels")
    );
    assert_eq!(got["phrase"], "under 10 px at 384");
    assert_eq!(got["phraseWide"], "under 7.3 px at 640");
    assert_eq!(
        got["many"],
        "Hide 12 pending auto-label proposals on 5 unreviewed frames as under the floor. \
         They are set aside unseen like the auto-labelers' hidden boxes: no label, no negative, not counted as your decisions. \
         i on the Review tab shows them, and they can be restored here if the floor goes down or off."
    );
    assert!(got["one_box"].as_str().unwrap().starts_with(
        "Hide 1 pending auto-label proposal on 1 unreviewed frame as under the floor"
    ));
    assert_eq!(
        got["back_many"],
        "Restore 7 hidden proposals on 3 unreviewed frames that the size floor no longer covers: \
         they become pending again, for you to decide on the Review tab."
    );
    assert!(
        got["back_one"]
            .as_str()
            .unwrap()
            .starts_with("Restore 1 hidden proposal on 1 unreviewed frame that")
    );
    assert_eq!(
        got["measured_floor"],
        "Measured with fold models trained on every label; the check treated labels under 10 px \
         at 384 as ignore regions (counted in this table only)."
    );
    assert_eq!(
        got["measured_none"],
        "Measured with fold models trained on every label; no size floor was set."
    );
    assert!(
        got["measured_old"]
            .as_str()
            .unwrap()
            .starts_with("Measured by a check of an earlier version")
    );
}

/// The Findings card's line on loose labels (`RefineFloor.looseLine`): with the closer look, how
/// many got a proposed box; without it (`--no-tighten`, the weights missing, an earlier report),
/// that none was looked at.
#[test]
fn the_findings_line_says_how_many_loose_labels_got_a_proposed_box() {
    if !have_node() {
        return;
    }
    let driver = r#"
const fs = require('fs');
require(process.argv[2]);
const req = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
require(req.refine);
const F = globalThis.RefineFloor;
process.stdout.write(JSON.stringify({
  looked: F.looseLine({ loose: 148, tighten: true, found: { add: 165, tighten: 37 } }),
  none_found: F.looseLine({ loose: 3, tighten: true, found: { add: 1 } }),
  no_look: F.looseLine({ loose: 148, tighten: false, found: { add: 165, tighten: 0 } }),
  earlier: F.looseLine({ loose: 12, found: { add: 2, verify: 1, relabel: 0 } }),
}));
"#;
    let refine = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/train-refine.js");
    let got = node_with(driver, &json!({"refine": refine}));
    assert_eq!(
        got["looked"],
        "Loose labels counted: 148, 37 with a proposed box."
    );
    assert_eq!(
        got["none_found"],
        "Loose labels counted: 3, 0 with a proposed box."
    );
    assert_eq!(
        got["no_look"],
        "Loose labels counted: 148 (no closer look in this check)."
    );
    assert_eq!(
        got["earlier"],
        "Loose labels counted: 12 (no closer look in this check)."
    );
}
