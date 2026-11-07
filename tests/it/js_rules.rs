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
    AutoInfo, BoxState, LabelBox, LabelFile, Origin, ParkedMark, RejectReason, Source, Status,
    Triage, repeats, resolve_pending, spots, stacks,
};
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
