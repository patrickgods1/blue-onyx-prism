// Review tab rules (/train/review; loaded before review.js): what decides which boxes stay, which
// go as duplicates and which suggestions stand, so what reaches the exports. Pure: no page, no
// browser API, no state of its own; everything comes in as an argument. `tests/it/js_rules.rs` runs
// this file in Node against its Rust twins (labels/stacks.rs, labels/overlaps.rs, labels/repeats.rs,
// labels/refine.rs, labels::resolve_pending), so a change here needs the same change there.
(function () {
  'use strict';

  // What saving makes of a box (mirrors labels::resolve_pending).
  function resolved(b) {
    if (b.state !== 'pending') return b.state;
    if (b.source === 'human') return 'added';
    if (b.source === 'auto') return 'accepted';
    if (b.origin === 'prediction') return 'accepted';
    return (b.triage === 'accept' && !b.auto_label) ? 'accepted' : 'rejected';
  }

  // A reason that marks a dismissal (labels::bulk::is_dismissal).
  function isDismissal(reason) { return reason === 'auto_dismissed' || reason === 'spot_dismissed' || reason === 'size_floor'; }

  // A reviewer's decision ends a dismissal: then the box is a label or a negative like any other
  // (labels::normalize does the same).
  function endDismissal(b) { if (isDismissal(b.reason)) delete b.reason; }

  // A "don't want" class (collect.hidden_classes; case-insensitive).
  function isHiddenClass(label, hidden) {
    var l = String(label || '').trim().toLowerCase();
    return hidden.some(function (h) { return h.toLowerCase() === l; });
  }

  // A box of an unwanted class: of a hidden class, or rejected and of no class of the list (the
  // server's other_class rejects). Not drawn, listed or stepped through (display only; the file
  // keeps it). Other boxes of no listed class stay visible: a save would refuse them.
  function isUnwanted(b, classes, hidden) {
    return isHiddenClass(b.label, hidden) || (b.state === 'rejected' && classes.indexOf(b.label) < 0);
  }

  function iou(a, b) {
    var x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]), x1 = Math.min(a[2], b[2]), y1 = Math.min(a[3], b[3]);
    var inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
    var u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return u > 0 ? inter / u : 0;
  }

  var VEHICLES = ['car', 'truck', 'bus', 'van'];

  // Two labels that can name one object: the same class, or two vehicle classes.
  function sameObject(a, b) {
    return a === b || (VEHICLES.indexOf(a) >= 0 && VEHICLES.indexOf(b) >= 0);
  }

  // How much of the smaller box lies inside the other (labels::inside_share).
  function insideShare(a, b) {
    var inter = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    var small = Math.min(Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]), Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]));
    return small > 0 ? inter / small : 0;
  }

  var STACK_IOU = 0.7, OVERLAP_IOU = 0.5, NESTED_SHARE = 0.8;   // labels/stacks.rs

  // Stacked boxes: the boxes (not rejected, not unwanted) of one object, IoU >= STACK_IOU and a
  // label of the same object, grouped as NMS would. A group of 2 or more keeps one: a drawn box,
  // else a reviewer-edited one, else an accepted one, else a pending one (a hint saving rejects
  // last: as keeper it would leave the object with no box), then the highest score. `overlaps`:
  // pairs of live boxes of one object outside a stack, IoU >= OVERLAP_IOU or the smaller
  // >= NESTED_SHARE inside the larger; those are the reviewer's, no rule picks.
  // {groups: [{members, keeper}], of: {index: group}, overlaps: [[i, j]], near: {index: partner}}.
  // Twin of labels/stacks.rs (`stacks`, `rank`, `overlaps`): keep the two in step. Three differences:
  // a live box of no listed class counts here (the page shows it, a save refuses it until it is
  // relabelled), the server skips it; the page knows no aliases, so a box whose label is only an
  // alias of a hidden class counts here too; and numbers here are f64, the server's f32, so a value
  // on a threshold to within rounding can fall on either side.
  function dupGroups(bs, classes, hidden) {
    var n = bs.length, parent = [], i, j;
    var live = function (b) { return b.state !== 'rejected' && !isUnwanted(b, classes, hidden); };
    var root = function (x) { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    for (i = 0; i < n; i++) parent.push(i);
    for (i = 0; i < n; i++) {
      if (!live(bs[i])) continue;
      for (j = i + 1; j < n; j++) {
        if (live(bs[j]) && sameObject(bs[i].label, bs[j].label) && iou(bs[i].bbox, bs[j].bbox) >= STACK_IOU) parent[root(j)] = root(i);
      }
    }
    var byRoot = {}, out = { groups: [], of: {}, overlaps: [], near: {} };
    for (i = 0; i < n; i++) {
      if (live(bs[i])) (byRoot[root(i)] = byRoot[root(i)] || []).push(i);
    }
    var rank = function (k) {
      var b = bs[k], tier;
      if (b.state === 'added') tier = 40;
      else if (b.state === 'accepted') tier = (b.orig_bbox || b.model_label) ? 30 : 20;
      else tier = resolved(b) === 'rejected' ? 0 : 10;
      return tier + (b.score != null ? b.score : 0);
    };
    Object.keys(byRoot).forEach(function (r) {
      var m = byRoot[r];
      if (m.length < 2) return;
      var g = { members: m, keeper: m.reduce(function (a, k) { return rank(k) > rank(a) ? k : a; }, m[0]) };
      out.groups.push(g);
      m.forEach(function (k) { out.of[k] = g; });
    });
    var closest = {};
    var note = function (a, b, v) {
      if (closest[a] == null || v > closest[a]) { closest[a] = v; out.near[a] = b; }
    };
    for (i = 0; i < n; i++) {
      if (!live(bs[i])) continue;
      for (j = i + 1; j < n; j++) {
        if (!live(bs[j]) || !sameObject(bs[i].label, bs[j].label) || (out.of[i] && out.of[i] === out.of[j])) continue;
        var v = iou(bs[i].bbox, bs[j].bbox);
        if (v >= OVERLAP_IOU || insideShare(bs[i].bbox, bs[j].bbox) >= NESTED_SHARE) {
          out.overlaps.push([i, j]);
          note(i, j, v);
          note(j, i, v);
        }
      }
    }
    return out;
  }

  // The non-drawn boxes of the stacks that are not their keeper (labels/stacks.rs `drops`).
  function dupDrops(d, bs) {
    var out = [];
    d.groups.forEach(function (g) {
      g.members.forEach(function (k) { if (k !== g.keeper && bs[k].source !== 'human') out.push(k); });
    });
    return out;
  }

  // Of those, the pending ones: what saving or Shift+A would otherwise accept as new duplicates
  // (a box the reviewer accepted stays; labels/stacks.rs `reject_pending_stacked`).
  function pendingDupDrops(d, bs) {
    return dupDrops(d, bs).filter(function (k) { return bs[k].state === 'pending'; });
  }

  function sameBbox(a, b) {
    return a[0] === b[0] && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
  }

  // Mark every box of `labels` with its index in the file as served (`_n`), so that a suggestion
  // keeps to the boxes the server named (`overlap_live_boxes`) when the page removes a box before
  // them. Called on every answer of GET /v1/review/labels; `_n` survives undo (JSON) and a save
  // strips it. A box drawn on the page has none.
  function markServed(labels) {
    ((labels && labels.boxes) || []).forEach(function (b, i) { b._n = i; });
  }

  // Where the box served at index `n` is now, -1 if it went.
  function servedAt(bs, n) {
    for (var k = 0; k < bs.length; k++) if (bs[k]._n === n) return k;
    return -1;
  }

  // The suggestions a reviewer can act on now, strongest first (as stored): of those the server
  // calls live (labels/overlaps.rs `live`, which also knows a mapped spelling from a relabel), the
  // ones still true after what happened on this page. The pair must still be open under its two
  // bboxes (`dupGroups`: a moved, relabelled or rejected box ends it), a box to drop must not have
  // become a drawn, moved or relabelled one, and a two-object suggestion not be accepted yet.
  // `cur`: the frame as GET /v1/review/labels answers it (its boxes marked by `markServed`), `dups`:
  // its `dupGroups`. [{index, s, a, b, drop, keep}]: box indices; drop and keep -1 for two objects.
  // Twin of labels/overlaps.rs `live`: keep the two in step. `overlap_live_boxes` names the boxes the
  // server chose for each suggestion, so where two boxes share a position and the server passed over
  // the first for a relabel, this takes the same box (RR-T9). It holds a suggestion to those boxes:
  // where an edit here ends their pair, a twin at the same key waits for the next load. An older
  // server sends only `overlap_live` (the suggestion's index): then this takes the first open pair
  // under the suggestion's bboxes.
  function liveSuggestions(cur, dups) {
    var out = [];
    var list = cur && cur.labels.overlap_suggestions;
    var named = cur && cur.overlap_live_boxes, ok = cur && cur.overlap_live;
    if (!list || !(named || ok) || cur.labels.overlaps_checked) return out;
    var bs = cur.labels.boxes, open = dups.overlaps || [];
    // Box `a` for the suggestion's `a`, `b` for its `b`: offered unless the box to drop is guarded.
    var take = function (n, s, a, b) {
      var drop = s.verdict === 'drop_a' ? a : (s.verdict === 'drop_b' ? b : -1);
      var d = bs[drop];
      if (d && (d.source === 'human' || d.orig_bbox || d._relabelled)) return false;
      out.push({ index: n, s: s, a: a, b: b, drop: drop, keep: drop < 0 ? -1 : (drop === a ? b : a) });
      return true;
    };
    if (named) {
      named.forEach(function (e) {
        var s = list[e.index];
        if (!s || s._done) return;
        var a = servedAt(bs, e.a), b = servedAt(bs, e.b);
        if (a < 0 || b < 0 || !sameBbox(bs[a].bbox, s.a) || !sameBbox(bs[b].bbox, s.b)) return;
        var still = open.some(function (o) {
          return (o[0] === a && o[1] === b) || (o[0] === b && o[1] === a);
        });
        if (still) take(e.index, s, a, b);
      });
      return out;
    }
    ok.forEach(function (n) {
      var s = list[n];
      if (!s || s._done) return;
      open.some(function (o) {
        var i = o[0], j = o[1];
        if (sameBbox(bs[i].bbox, s.a) && sameBbox(bs[j].bbox, s.b)) return take(n, s, i, j);
        if (sameBbox(bs[j].bbox, s.a) && sameBbox(bs[i].bbox, s.b)) return take(n, s, j, i);
        return false;
      });
    });
    return out;
  }

  // A hint saving would reject stays through the save: accepted, so the object keeps a box.
  function keepThrough(b) {
    if (b.state === 'pending' && resolved(b) === 'rejected') b.state = 'accepted';
    b._checked = true;
  }

  // Every rejection by the reviewer goes through here (no undo step, no move: the callers).
  // `why`: the box's `reject_reason` ('duplicate', ...); none keeps an earlier one. A rejected
  // box is not "unsure".
  function markRejected(b, why) {
    b.state = 'rejected';
    b._checked = true;
    delete b.difficult;
    endDismissal(b);
    // A parked accept taken back is a look like any other (labels::normalize does the same).
    if (b.reason === 'parked_spot') { delete b.reason; delete b.parked; }
    if (why) b.reject_reason = why;
  }

  // A duplicate rejected by rule rather than by hand: the provenance says so, an older one stays.
  // Twin of labels/stacks.rs `mark`.
  function markDuplicate(b) {
    markRejected(b, 'duplicate');
    if (!b.reason) b.reason = 'duplicate';
  }

  // Accept suggestion `g` of `liveSuggestions` on `labels` (the file as the page holds it). A drop
  // rejects the box as a duplicate and keeps the other through the save; two objects keeps both
  // (the save then records `overlaps_checked`). Either way the frame is no longer confirmed empty.
  // Twin of labels/overlaps.rs `accept`: keep the two in step.
  function acceptOverlap(labels, g) {
    var bs = labels.boxes;
    if (g.drop >= 0) {
      keepThrough(bs[g.keep]);
      markDuplicate(bs[g.drop]);
    } else {
      keepThrough(bs[g.a]);
      keepThrough(bs[g.b]);
      g.s._done = true;
    }
    if (labels.empty_confirmed) labels.empty_confirmed = false;
  }

  // ---- The dataset check's suggestions (`labels.refine`; twin of labels/refine.rs) ----
  //
  // Unlike the rules above these compute as the server does, so the twins differ nowhere: numbers
  // in f32 (`Math.fround` after every step, which gives exactly the f32 result
  // of + - * /), labels trimmed of Unicode white space as Rust's `trim`, compared ignoring ASCII case
  // only, and class names mapped onto the list with the COCO / VOC aliases (model/coco.rs). The
  // class list and hidden classes come as the other rules take them (`classes`, `hidden`); a null
  // `classes` means every label is wanted, compared as written.

  var f32 = Math.fround;
  var MATCH_IOU = 0.5, COVER_IOU = 0.3, COVER_INSIDE = 0.5;   // labels/mod.rs
  var MAX_BOXES = 500, MAX_LABEL_LEN = 64, REFINE_REASON = 'refine';
  var MATCH32 = f32(MATCH_IOU), COVER_IOU32 = f32(COVER_IOU), COVER_INSIDE32 = f32(COVER_INSIDE);
  var OVERLAP32 = f32(OVERLAP_IOU), NESTED32 = f32(NESTED_SHARE);

  // model/coco.rs COCO_NAMES and COCO_ALIASES.
  var COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
  ];
  var COCO_ALIASES = [
    ['motorbike', 'motorcycle'], ['people', 'person'], ['pedestrian', 'person'], ['bike', 'bicycle'],
    ['sofa', 'couch'], ['tvmonitor', 'tv'], ['aeroplane', 'airplane'],
    ['pottedplant', 'potted plant'], ['diningtable', 'dining table']
  ];

  // Rust's `char::is_whitespace` (Unicode White_Space), which `str::trim` removes.
  var WS = '[\\u0009-\\u000D\\u0020\\u0085\\u00A0\\u1680\\u2000-\\u200A\\u2028\\u2029\\u202F\\u205F\\u3000]';
  var TRIM = new RegExp('^' + WS + '+|' + WS + '+$', 'g');
  function rtrim(s) { return String(s).replace(TRIM, ''); }
  function asciiLower(s) { return String(s).replace(/[A-Z]/g, function (c) { return c.toLowerCase(); }); }
  // `eq_ignore_ascii_case`.
  function sameAscii(a, b) { return asciiLower(a) === asciiLower(b); }

  // The COCO name of a class, aliases applied (model/coco.rs `coco_class`), or null.
  function cocoClass(label) {
    var l = String(label).toLowerCase();
    COCO_ALIASES.some(function (a) { if (a[0] === l) { l = a[1]; return true; } return false; });
    return COCO_NAMES.indexOf(l) >= 0 ? l : null;
  }

  function sameClass(a, b) {
    var c = cocoClass(a);
    return sameAscii(a, b) || (c !== null && cocoClass(b) === c);
  }

  // The class of `classes` a label is: the same name, else ignoring case, else through the aliases
  // (model/coco.rs `map_label`); null when none.
  function mapLabel(label, classes) {
    var l = rtrim(label), i;
    for (i = 0; i < classes.length; i++) if (classes[i] === l) return classes[i];
    for (i = 0; i < classes.length; i++) if (sameAscii(classes[i], l)) return classes[i];
    for (i = 0; i < classes.length; i++) if (sameClass(classes[i], l)) return classes[i];
    return null;
  }

  // The wanted class a label is, in the list's spelling, null for an unwanted one
  // (labels::wanted_class; `hidden` compared ignoring ASCII case).
  function wantedClass(label, classes, hidden) {
    var c = mapLabel(label, classes);
    if (c === null) return null;
    var t = rtrim(c);
    return (hidden || []).some(function (h) { return sameAscii(h, t); }) ? null : c;
  }

  // refine.rs `class_of`: with a list, the wanted class; without, the trimmed label (null if blank).
  function refineClassOf(label, classes, hidden) {
    if (classes) return wantedClass(label, classes, hidden);
    var t = rtrim(label);
    return t ? t : null;
  }

  function boxClass(b, classes, hidden) {
    var c = refineClassOf(b.label, classes, hidden);
    return c === null ? rtrim(b.label) : c;
  }

  // Two class names of one object (refine.rs `one_object`).
  function oneObject(a, b) { return sameObject(asciiLower(rtrim(a)), asciiLower(rtrim(b))); }

  function sameLabel(a, b) { return sameAscii(rtrim(a), rtrim(b)); }

  function box32(b) { return [f32(b[0]), f32(b[1]), f32(b[2]), f32(b[3])]; }

  function area32(r) { return f32(Math.max(f32(r[2] - r[0]), 0) * Math.max(f32(r[3] - r[1]), 0)); }

  // The overlap's width and height of two f32 boxes.
  function overlap32(a, b) {
    return [
      Math.max(f32(Math.min(a[2], b[2]) - Math.max(a[0], b[0])), 0),
      Math.max(f32(Math.min(a[3], b[3]) - Math.max(a[1], b[1])), 0)
    ];
  }

  // labels::iou in f32.
  function iou32(a, b) {
    var A = box32(a), B = box32(b), o = overlap32(A, B);
    var inter = f32(o[0] * o[1]);
    var union = f32(f32(area32(A) + area32(B)) - inter);
    return union > 0 ? f32(inter / union) : 0;
  }

  // labels::inside_share in f32.
  function insideShare32(a, b) {
    var A = box32(a), B = box32(b), o = overlap32(A, B);
    var smaller = Math.min(area32(A), area32(B));
    return smaller > 0 ? f32(f32(o[0] * o[1]) / smaller) : 0;
  }

  // labels::covers: box `b` covers the detection `p`.
  function covers32(b, p) {
    return iou32(b, p) >= COVER_IOU32 ||
      (area32(box32(p)) <= area32(box32(b)) && insideShare32(p, b) >= COVER_INSIDE32);
  }

  function sameBbox32(a, b) {
    return f32(a[0]) === f32(b[0]) && f32(a[1]) === f32(b[1]) && f32(a[2]) === f32(b[2]) && f32(a[3]) === f32(b[3]);
  }

  function isTraining(b) { return b.state === 'accepted' || b.state === 'added'; }

  // LabelBox::is_ignore: unsure, or rejected as too small.
  function isIgnore(b) { return !!b.difficult || (b.state === 'rejected' && b.reject_reason === 'too_small'); }

  // labels::under_floor: whether `bbox` ([x0, y0, x1, y1], pixels of a `w` x `h` frame) is under the
  // size floor `floor` ({px, input}, `collect.size_floor`): its short side at the floor's model input,
  // min(w / W, h / H) * input of the box clamped to the frame, is under `px`, in f32 as the server
  // computes it. Never under: a frame of unknown size, a non-finite box or one with nothing inside
  // the frame, a floor that is null or off.
  function underFloor(bbox, w, h, floor) {
    if (!floor || typeof floor.px !== 'number' || typeof floor.input !== 'number') return false;
    var px = f32(floor.px), input = f32(floor.input);
    if (!(isFinite(px) && px > 0 && input > 0)) return false;
    if (!(typeof w === 'number' && typeof h === 'number' && w > 0 && h > 0)) return false;
    if (!bbox || bbox.length !== 4 || !bbox.every(function (v) { return typeof v === 'number'; })) return false;
    var b = box32(bbox);
    if (!b.every(function (v) { return isFinite(v); })) return false;
    var W = f32(w), H = f32(h);
    var clamp = function (v, hi) { return Math.min(Math.max(v, 0), hi); };
    var x0 = clamp(Math.min(b[0], b[2]), W), x1 = clamp(Math.max(b[0], b[2]), W);
    var y0 = clamp(Math.min(b[1], b[3]), H), y1 = clamp(Math.max(b[1], b[3]), H);
    var bw = f32(x1 - x0), bh = f32(y1 - y0);
    if (!(bw > 0 && bh > 0)) return false;
    return f32(Math.min(f32(bw / W), f32(bh / H)) * input) < px;
  }

  // What the page says of a box under the size floor, or null: a box a reviewer rejected or marked
  // unsure is no training box (nothing is left out for it), and with no floor (null) or a box over
  // it nothing is said. {chip, note}: the box list's and tag's word, and the selected box's line.
  // Information only: it changes no box and is never saved.
  function floorNote(b, w, h, floor) {
    if (!b || b.state === 'rejected' || b.difficult) return null;
    if (!underFloor(b.bbox, w, h, floor)) return null;
    var px = String(Number(Number(floor.px).toFixed(2)));
    return {
      chip: 'too small',
      note: 'Under the size floor (' + px + ' px at ' + floor.input + '): left out of the labels at export.'
    };
  }

  // labels::unjudged_reject: a reject no reviewer judged (a bulk low-score reject, a dismissal, a
  // model hint the save default rejected).
  function unjudgedReject(b) {
    var reason = b.reason == null ? null : b.reason;
    return b.state === 'rejected' && b.reject_reason == null &&
      reason !== 'repeat_spot' && reason !== 'duplicate' &&
      (reason === 'low_score' || isDismissal(reason) || (b.source === 'model' && b.origin !== 'prediction'));
  }

  function proper(b) {
    return !!b && b.length === 4 && b.every(function (v) { return typeof v === 'number' && isFinite(v); }) &&
      f32(b[2]) > f32(b[0]) && f32(b[3]) > f32(b[1]);
  }

  // Whether `b` is the box suggestion `s` names: its exact bbox, and its label when `s` has one.
  function names(s, b) {
    return s.target != null && sameBbox32(b.bbox, s.target) &&
      (s.target_label == null || sameLabel(b.label, s.target_label));
  }

  // The first box `s` names that is a training box, no ignore region (an unsure box: the check
  // never judges one), no reviewer drew and that `ok` takes; -1 if none.
  function targetBox(f, s, ok) {
    var bs = f.boxes || [];
    for (var i = 0; i < bs.length; i++) {
      var b = bs[i];
      if (names(s, b) && isTraining(b) && !isIgnore(b) && b.source !== 'human' && ok(b)) return i;
    }
    return -1;
  }

  // Where an `add` goes: null when it is not live, else the rejects it replaces, closest first.
  function addPlace(f, s, classes, hidden) {
    if (!proper(s.bbox) || s.label == null) return null;
    var cls = refineClassOf(s.label, classes, hidden);
    if (cls === null) return null;
    var bs = f.boxes || [], rep = [];
    for (var i = 0; i < bs.length; i++) {
      var b = bs[i];
      if (!oneObject(boxClass(b, classes, hidden), cls)) continue;
      if (b.state !== 'rejected' || isIgnore(b)) {
        if (covers32(b.bbox, s.bbox)) return null;
        continue;
      }
      var v = iou32(b.bbox, s.bbox);
      if (v >= MATCH32) {
        if (!unjudgedReject(b)) return null;
        rep.push([i, v]);
      }
    }
    if (!rep.length && bs.length >= MAX_BOXES) return null;
    rep.sort(function (x, y) { return x[1] !== y[1] ? y[1] - x[1] : x[0] - y[0]; });
    return rep.map(function (x) { return x[0]; });
  }

  // refine.rs `place`: null when `s` is not live, else {target, replaces} (target -1 for an add).
  function refinePlace(f, s, classes, hidden) {
    if (s.answer != null) return null;
    var t;
    switch (s.kind) {
      case 'add': {
        var r = addPlace(f, s, classes, hidden);
        return r === null ? null : { target: -1, replaces: r };
      }
      case 'tighten':
        if (!proper(s.bbox)) return null;
        t = targetBox(f, s, function (b) { return b.orig_bbox == null && !sameBbox32(b.bbox, s.bbox); });
        break;
      case 'relabel': {
        if (s.label == null) return null;
        var to = refineClassOf(s.label, classes, hidden);
        if (to === null) return null;
        t = targetBox(f, s, function (b) {
          return b.model_label == null && !oneObject(boxClass(b, classes, hidden), to);
        });
        break;
      }
      case 'verify':
        t = targetBox(f, s, function (b) { return b.orig_bbox == null; });
        break;
      default:
        return null;
    }
    return t < 0 ? null : { target: t, replaces: [] };
  }

  function orNull(v) { return v == null ? null : v; }
  function copyBox(v) { return v == null ? null : v.slice(); }

  // The suggestions of the dataset check on `labels` a reviewer can act on: none on a skipped
  // frame, never an answered one or one of a kind this page does not know, never one whose box
  // moved, went or changed so it no longer applies. Each {index (into labels.refine), kind,
  // target, target_label, bbox, label, score, why, spot, run, target_box, replaces} (target_box:
  // the index of the box `target` names, null for an add; replaces: the unjudged rejects an add
  // takes the place of, closest first). Equal to `refine_live` of GET /v1/review/labels for the
  // same file. Twin of labels/refine.rs `live`: keep the two in step.
  function liveRefine(labels, classes, hidden) {
    var list = labels && labels.refine;
    if (!list || !list.length || labels.status === 'skipped') return [];
    var out = [];
    list.forEach(function (s, index) {
      if (!s) return;
      var p = refinePlace(labels, s, classes, hidden);
      if (p === null) return;
      out.push({
        index: index,
        kind: s.kind,
        target: copyBox(s.target),
        target_label: orNull(s.target_label),
        bbox: copyBox(s.bbox),
        label: orNull(s.label),
        score: orNull(s.score),
        why: s.why == null ? '' : s.why,
        spot: orNull(s.spot),
        run: s.run == null ? '' : s.run,
        target_box: p.target < 0 ? null : p.target,
        replaces: p.replaces
      });
    });
    return out;
  }

  // How strong a suggestion is, higher first (labels/refine.rs `strength`): add 4, relabel 3,
  // tighten 2, verify 1, times 1000, plus the score in thousandths; 0 when answered or unknown.
  function refineStrength(s) {
    var band = s.kind === 'add' ? 4 : s.kind === 'relabel' ? 3 : s.kind === 'tighten' ? 2 : s.kind === 'verify' ? 1 : 0;
    if (!band || s.answer != null) return 0;
    var v = (typeof s.score === 'number' && isFinite(s.score)) ? f32(s.score) : 0;
    return band * 1000 + Math.round(Math.min(Math.max(v, 0), 1) * 999);
  }

  // Whether box `at` is in a stack or an open pair (labels/stacks.rs `groups` / `overlap_pairs`,
  // refine.rs `in_open_pair`): it and another live box of one object at IoU >= OVERLAP_IOU, or the
  // smaller >= NESTED_SHARE inside the larger. (A stack's link is such a pair too, and a pair in one
  // stack is a stack, so this is the two rules together.)
  function inOpenPair(bs, at, classes, hidden) {
    var live = function (b) {
      return b.state !== 'rejected' && (!classes || wantedClass(b.label, classes, hidden || []) !== null);
    };
    var a = bs[at];
    if (!a || !live(a)) return false;
    return bs.some(function (b, j) {
      return j !== at && live(b) && sameObject(a.label, b.label) &&
        (iou32(a.bbox, b.bbox) >= OVERLAP32 || insideShare32(a.bbox, b.bbox) >= NESTED32);
    });
  }

  // labels::rename_to_class: the class list's spelling, the old label kept in `model_label`.
  function renameToClass(b, cls) {
    if (cls === b.label) return;
    if (b.model_label == null && !sameAscii(cls, rtrim(b.label))) {
      b.model_label = Array.from(String(b.label)).slice(0, MAX_LABEL_LEN).join('');
    }
    b.label = cls;
  }

  // The reviewer's yes to suggestion `index` of `labels` (the file as the page holds it); false,
  // changing nothing, when it is not live. add: a new accepted `auto` box (the class in the list's
  // spelling, reason 'refine') in the place of the first box of `replaces`, the others removed, the
  // frame no longer confirmed empty; tighten: `orig_bbox` keeps the old box, `bbox` the proposal
  // (unclamped: the save clamps); relabel: the new class, the old one in `model_label`. Those three
  // remove the suggestion (later indices move down by one) and end `overlaps_checked` when the box
  // is now in a stack or an open pair. verify: answered 'kept', it stays. Page flags (`_checked`,
  // `_relabelled`) are the caller's. Twin of labels/refine.rs `accept`: keep the two in step.
  function acceptRefine(labels, index, classes, hidden) {
    var l = liveRefine(labels, classes, hidden).filter(function (x) { return x.index === index; })[0];
    if (!l) return false;
    var s = labels.refine[index], bs = labels.boxes, at = l.target_box, b;
    switch (s.kind) {
      case 'add': {
        var cls = refineClassOf(s.label, classes, hidden);
        if (cls === null) return false;
        var nb = { label: cls, bbox: s.bbox.slice(), source: 'auto', state: 'accepted', reason: REFINE_REASON };
        if (s.score != null) nb.score = s.score;
        if (l.replaces.length) {
          at = l.replaces[0];
          bs[at] = nb;
        } else {
          bs.push(nb);
          at = bs.length - 1;
        }
        var gone = l.replaces.slice(1).sort(function (x, y) { return y - x; });
        gone.forEach(function (g) { bs.splice(g, 1); });
        at -= gone.filter(function (g) { return g < at; }).length;
        labels.empty_confirmed = false;
        break;
      }
      case 'tighten':
        b = bs[at];
        if (b.orig_bbox == null) b.orig_bbox = b.bbox.slice();
        b.bbox = s.bbox.slice();
        break;
      case 'relabel': {
        var to = refineClassOf(s.label, classes, hidden);
        if (to === null) return false;
        renameToClass(bs[at], to);
        break;
      }
      case 'verify':
        s.answer = 'kept';
        return true;
      default:
        return false;
    }
    if (labels.overlaps_checked && inOpenPair(bs, at, classes, hidden)) labels.overlaps_checked = false;
    labels.refine.splice(index, 1);
    return true;
  }

  // The reviewer's no to suggestion `index`: answered 'dismissed', so a rerun does not ask again;
  // no box changes. False when there is no such suggestion or it is answered already. Twin of
  // labels/refine.rs `dismiss`.
  function dismissRefine(labels, index) {
    var s = labels && labels.refine && labels.refine[index];
    if (!s || s.answer != null) return false;
    s.answer = 'dismissed';
    return true;
  }

  // ---- How the Review page offers those suggestions (page decisions: no Rust twin;
  // tests/it/js_review_refine.rs) ----
  //
  // `l` below is an entry of `liveRefine` (or `refine_live`): {index, kind, target_box, replaces,
  // bbox, label, score, why, ...}.

  // Strongest first (`refineStrength`), ties by index: the order the page offers them in.
  function refineOrder(live) {
    return (live || []).slice().sort(function (a, b) {
      var d = refineStrength(b) - refineStrength(a);
      return d !== 0 ? d : a.index - b.index;
    });
  }

  // What names a suggestion across edits: its index moves down when an accepted one before it
  // leaves the list.
  function refineKey(l) {
    var n = function (v) { return v == null ? null : v; };
    return JSON.stringify([l.kind, n(l.target), n(l.target_label), n(l.bbox), n(l.label)]);
  }

  // The suggestion the page shows, one at a time: none while an overlap suggestion is on screen
  // (`overlapShown`: those come first and keep `y`), else the strongest live one the reviewer did
  // not put off with Later (`later`: keys, refineKey), else the strongest. {l, k, n}: the live
  // entry, its place in the order (from 1), how many are live; null for none.
  function refineCurrent(live, overlapShown, later) {
    if (overlapShown || !live || !live.length) return null;
    var order = refineOrder(live), skip = later || [];
    for (var i = 0; i < order.length; i++) {
      if (skip.indexOf(refineKey(order[i])) < 0) return { l: order[i], k: i + 1, n: order.length };
    }
    return { l: order[0], k: 1, n: order.length };
  }

  // Later: the keys put off once the current suggestion is put off too; past the last one, none
  // (the strongest again).
  function refineLater(live, overlapShown, later) {
    var c = refineCurrent(live, overlapShown, later);
    if (!c) return (later || []).slice();
    var out = (later || []).concat([refineKey(c.l)]);
    var all = live.every(function (l) { return out.indexOf(refineKey(l)) >= 0; });
    return all ? [] : out;
  }

  // What key `y` or `w` acts on: 'overlap' (`y` while an overlap suggestion is on screen, and `y`
  // with no suggestion at all: the overlap handler says there is none, as it always did), 'accept'
  // / 'dismiss' (the current dataset check suggestion `refine`, refineCurrent), 'done' (`y` while
  // the box of an accepted add is being adjusted, `adjusting`: adjustingBox; `w` does nothing
  // then), 'none' (`w` with nothing to answer: the key does nothing), 'held' (`repeat`: the key's
  // auto-repeat while held down; nothing is done, as for `n` and `s`: every answer is remembered,
  // so a held key must not answer one suggestion after another, nor accept the adjusted box the
  // moment it was added).
  function suggestionKey(key, overlapShown, refine, repeat, adjusting) {
    var route = 'none';
    if (key === 'y') route = overlapShown ? 'overlap' : (adjusting ? 'done' : (refine ? 'accept' : 'overlap'));
    else if (key === 'w') route = !overlapShown && !adjusting && refine ? 'dismiss' : 'none';
    return repeat && route !== 'none' ? 'held' : route;
  }

  // ---- Adjusting the box of an accepted add (page decision) ----
  //
  // A yes to an `add` does not move on at once: answerRefine marks the new box `_adjusting`
  // (page-only, never saved), the page keeps it selected for the reviewer to move or resize, and
  // `y` again (Accept box, endAdjust) brings the next suggestion. The flag lives on the box, so
  // the undo snapshots carry it: undoing past the accept takes the box and the step away and the
  // suggestion is current again; rejecting the box ends the step too.

  // The box being adjusted on boxes `bs`: the index of a box marked `_adjusting` and not rejected;
  // -1 for none.
  function adjustingBox(bs) {
    for (var i = 0; i < (bs || []).length; i++) {
      if (bs[i] && bs[i]._adjusting && bs[i].state !== 'rejected') return i;
    }
    return -1;
  }

  // The adjusting is done: the mark off every box (a rejected one's too).
  function endAdjust(bs) {
    (bs || []).forEach(function (b) { if (b) delete b._adjusting; });
  }

  // Whether a reviewed save records `overlaps_checked` (assets/review.js `save`): stacks or
  // overlapping pairs left on boxes `bs` as saved (hints resolved as the save resolves them). The
  // save is the look: the page shows every pair it holds (the `dup` / `overlap` chips, the
  // progress line's "Overlap: a+b", the selected box's "Overlaps box N" with `u`) and the save's
  // check pass steps through the decided boxes not looked at on this visit. What made a pair on
  // this visit does not matter: a box drawn, moved or relabelled by hand and an accepted dataset
  // check suggestion (acceptRefine, which ends the flag on the file as its Rust twin does) count
  // alike.
  function overlapsCheckedOnSave(bs, classes, hidden) {
    var left = dupGroups(bs.map(function (b) { return Object.assign({}, b, { state: resolved(b) }); }), classes, hidden);
    return left.groups.length > 0 || left.overlaps.length > 0;
  }

  function score2(v) { return typeof v === 'number' && isFinite(v) ? v.toFixed(2) : ''; }

  // Share of a `tighten`'s proposed box inside the box it is about from which it is "the tighter
  // box": a few pixels past an edge still read as tighter. Below it the proposal is larger than
  // the label or shifted, so the card calls it "the better-fitting box".
  var TIGHTER_INSIDE = 0.95;

  // 'tighter' when the proposed box of live `tighten` `l` lies (almost) inside the box it is about
  // (on boxes `bs`), else 'better-fitting'.
  function tightenWord(l, bs) {
    var at = l.target_box == null ? -1 : l.target_box;
    var t = (bs || [])[at] ? bs[at].bbox : l.target;
    var z = l.bbox;
    if (!t || !z) return 'better-fitting';
    var inter = Math.max(0, Math.min(t[2], z[2]) - Math.max(t[0], z[0])) * Math.max(0, Math.min(t[3], z[3]) - Math.max(t[1], z[1]));
    var az = Math.max(0, z[2] - z[0]) * Math.max(0, z[3] - z[1]);
    return az > 0 && inter / az >= TIGHTER_INSIDE ? 'tighter' : 'better-fitting';
  }

  // What the card says about live suggestion `l` on boxes `bs`: {head (the question), short (the
  // one-column bar's line), yes, no (the buttons, keys y and w), no_bad (the no answer is a
  // rejection: drawn red), note ('' or a line under the evidence), score ('' or "score 0.82"),
  // said_yes, said_no (what the status line says after each answer)}. `touch`: the note names
  // buttons, not keys.
  function refineCard(l, bs, touch) {
    var at = l.target_box == null ? -1 : l.target_box;
    var b = (bs || [])[at];
    var now = b ? b.label : (l.target_label || '');
    var n = 'Box ' + (at + 1);
    var label = l.label == null ? '' : l.label;
    var c = { head: '', short: '', yes: '', no: '', no_bad: false, note: '', score: '', said_yes: '', said_no: '' };
    if (score2(l.score)) c.score = 'score ' + score2(l.score);
    switch (l.kind) {
      case 'add': {
        var r = (l.replaces || []).length;
        c.head = 'Missing box: is the dashed box a ' + label + '?';
        c.short = 'Missing ' + label + '?';
        c.yes = 'Accept: add this box';
        c.no = 'Not an object';
        c.no_bad = true;
        if (r) c.note = 'Accepting puts it in the place of ' + r + ' rejected box' + (r === 1 ? '' : 'es') + ' here that nobody judged.';
        c.said_yes = 'Added a ' + label + ' box';
        c.said_no = 'Not an object: suggestion dismissed';
        break;
      }
      case 'relabel':
        c.head = n + ' is labelled ' + now + ': is it a ' + label + '?';
        c.short = n + ': ' + label + '?';
        c.yes = 'Accept: it is a ' + label;
        c.no = 'Keep ' + now;
        c.said_yes = n + ' is a ' + label + ' now';
        c.said_no = n + ' stays a ' + now;
        break;
      case 'tighten': {
        // The closer look's box can also be larger than the label or shifted (tightenWord).
        var w = tightenWord(l, bs);
        c.head = n + ' (' + now + '): take the ' + w + ' box (dashed)?';
        c.short = n + ': ' + w + ' box?';
        c.yes = 'Accept the ' + w + ' box';
        c.no = 'Keep my box';
        c.said_yes = n + ' takes the ' + w + ' box';
        c.said_no = n + ' keeps its box';
        break;
      }
      case 'verify':
        c.head = 'No model found ' + n.toLowerCase() + ' (' + now + '): is the label right?';
        c.short = n + ': no model found it';
        c.yes = 'Keep the label';
        c.no = 'Dismiss';
        c.note = touch
          ? 'If it is wrong: No, reject rejects ' + n.toLowerCase() + '; Too small or Unsure marks it. Each ends the suggestion. Dismiss changes nothing and is not asked again.'
          : 'If it is wrong: r rejects ' + n.toLowerCase() + '; Too small or d (unsure) marks it. Each ends the suggestion. w dismisses: nothing changes, and it is not asked again.';
        c.said_yes = n + ': label kept';
        c.said_no = n + ': suggestion dismissed';
        break;
      default:
        return null;
    }
    return c;
  }

  // What the frame shows for live suggestion `l` on boxes `bs`: {ghost: the proposed box (add,
  // tighten; else null), mark: the box it is about, marked and selected (-1 for an add), focus:
  // the part of the frame to show (the ghost, the box, or both together for a tighter box)}.
  function refineFocus(l, bs) {
    var at = l.target_box == null ? -1 : l.target_box;
    var t = (bs || [])[at] ? bs[at].bbox : null;
    var ghost = (l.kind === 'add' || l.kind === 'tighten') && l.bbox ? l.bbox.slice() : null;
    var focus = ghost && t
      ? [Math.min(ghost[0], t[0]), Math.min(ghost[1], t[1]), Math.max(ghost[2], t[2]), Math.max(ghost[3], t[3])]
      : (ghost || (t ? t.slice() : null));
    return { ghost: ghost, mark: t ? at : -1, focus: focus };
  }

  // The reviewer's answer to live suggestion `l` on `labels` (the file as the page holds it): yes
  // = acceptRefine, no = dismissRefine. Then the page flags on the box a yes is about, as the
  // page's own edits set them: `_checked` (looked at on this visit), for a relabel `_relabelled`
  // (an overlap suggestion that would drop the box ends, as after a relabel by hand), and for an
  // add `_adjusting` (the reviewer fits the new box before the next suggestion: adjustingBox).
  // A no changes no box and sets no flag. {ok, box}: whether anything changed, and the box
  // to select after (the new box of an add, else the box the suggestion is about; -1 for none).
  function answerRefine(labels, l, yes, classes, hidden) {
    var at = l.target_box == null ? -1 : l.target_box;
    if (!yes) return { ok: dismissRefine(labels, l.index), box: at };
    if (l.kind === 'add') {
      var r = l.replaces || [];
      at = r.length ? r[0] - r.slice(1).filter(function (g) { return g < r[0]; }).length : (labels.boxes || []).length;
    }
    if (!acceptRefine(labels, l.index, classes, hidden)) return { ok: false, box: -1 };
    var b = labels.boxes[at];
    if (b) {
      b._checked = true;
      if (l.kind === 'relabel') b._relabelled = true;
      if (l.kind === 'add') b._adjusting = true;
    }
    return { ok: true, box: b ? at : -1 };
  }

  // Whether the Review queue sends filter `name` = `value` to GET /v1/review/next: a value given,
  // and `refine_kind` only with `auto=refine` (the server refuses it otherwise).
  function sendsFilter(name, value, auto) {
    return !!value && (name !== 'refine_kind' || auto === 'refine');
  }

  // A save's copy of the file: page-only fields (`_...`) off the dataset check's suggestions.
  function refineForSave(labels) {
    ((labels && labels.refine) || []).forEach(function (s) {
      if (!s) return;
      Object.keys(s).forEach(function (k) { if (k.charAt(0) === '_') delete s[k]; });
    });
    return labels;
  }

  // What a new call of the page's `loadNext` does while `running` ('real', 'counts' or null) and
  // `waiting` (the same, or null) stand; `want` is 'real' or 'counts'. GET /v1/review/next walks the
  // whole folder, seconds on a big one, so at most one real walk runs and the newest intent wins:
  // 'start'  nothing runs: go;
  // 'jump'   only a counts refresh runs: a real Next must not wait for it, and its answer, which
  //          carries newer counts, replaces that refresh and any waiting one;
  // 'queue'  take the one waiting place, after the running walk; a real intent also stops a real
  //          walk from landing (a filter changed, or a save must move on from its own frame);
  // 'drop'   a real Next already waits, and its answer will carry the counts.
  // A refresh never cancels a real Next: it only ever waits, or is dropped for one.
  function nextPlan(running, waiting, want) {
    if (!running) return 'start';
    if (want === 'real') return running === 'counts' ? 'jump' : 'queue';
    return waiting === 'real' ? 'drop' : 'queue';
  }

  // What an answer of the page's `loadNext` shows once it lands, for a walk of `kind` ('real' or
  // 'counts'): 'frame' its frame (or "no frame matches"), 'counts' the counts and lists only. A walk
  // a newer real intent replaced (`replaced`) leaves the frame to that one. A counts refresh keeps
  // the frame on screen and its edits; with none on screen or loading (`shown` false) it shows what
  // it found: that is how the 2 s refresh while a big folder is indexed brings the first frame, and
  // swaps "Indexing..." for "No frame matches" once the index is done with none.
  function nextLanding(kind, replaced, shown) {
    if (replaced) return 'counts';
    if (kind === 'real') return 'frame';
    return shown ? 'counts' : 'frame';
  }

  globalThis.ReviewRules = {
    STACK_IOU: STACK_IOU,
    OVERLAP_IOU: OVERLAP_IOU,
    NESTED_SHARE: NESTED_SHARE,
    VEHICLES: VEHICLES,
    resolved: resolved,
    isDismissal: isDismissal,
    endDismissal: endDismissal,
    isHiddenClass: isHiddenClass,
    isUnwanted: isUnwanted,
    iou: iou,
    sameObject: sameObject,
    insideShare: insideShare,
    dupGroups: dupGroups,
    dupDrops: dupDrops,
    pendingDupDrops: pendingDupDrops,
    markServed: markServed,
    liveSuggestions: liveSuggestions,
    markRejected: markRejected,
    markDuplicate: markDuplicate,
    acceptOverlap: acceptOverlap,
    MATCH_IOU: MATCH_IOU,
    COVER_IOU: COVER_IOU,
    COVER_INSIDE: COVER_INSIDE,
    MAX_BOXES: MAX_BOXES,
    MAX_LABEL_LEN: MAX_LABEL_LEN,
    REFINE_REASON: REFINE_REASON,
    COCO_NAMES: COCO_NAMES,
    COCO_ALIASES: COCO_ALIASES,
    wantedClass: wantedClass,
    underFloor: underFloor,
    floorNote: floorNote,
    liveRefine: liveRefine,
    refineStrength: refineStrength,
    acceptRefine: acceptRefine,
    dismissRefine: dismissRefine,
    refineOrder: refineOrder,
    refineKey: refineKey,
    refineCurrent: refineCurrent,
    refineLater: refineLater,
    suggestionKey: suggestionKey,
    overlapsCheckedOnSave: overlapsCheckedOnSave,
    refineCard: refineCard,
    TIGHTER_INSIDE: TIGHTER_INSIDE,
    tightenWord: tightenWord,
    refineFocus: refineFocus,
    answerRefine: answerRefine,
    adjustingBox: adjustingBox,
    endAdjust: endAdjust,
    sendsFilter: sendsFilter,
    refineForSave: refineForSave,
    nextPlan: nextPlan,
    nextLanding: nextLanding
  };
}());
