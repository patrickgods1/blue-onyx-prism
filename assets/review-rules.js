// Review tab rules (/train/review; loaded before review.js): what decides which boxes stay, which
// go as duplicates and which suggestions stand, so what reaches the exports. Pure: no page, no
// browser API, no state of its own; everything comes in as an argument. `tests/it/js_rules.rs` runs
// this file in Node against its Rust twins (labels/stacks.rs, labels/overlaps.rs, labels/repeats.rs,
// labels::resolve_pending), so a change here needs the same change there.
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
  function isDismissal(reason) { return reason === 'auto_dismissed' || reason === 'spot_dismissed'; }

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

  // The suggestions a reviewer can act on now, strongest first (as stored): of those the server
  // calls live (`overlap_live`: labels/overlaps.rs `live`, which also knows a mapped spelling from a
  // relabel), the ones still true after what happened on this page. The pair must still be open
  // under its two bboxes (`dupGroups`: a moved, relabelled or rejected box ends it), a box to drop must
  // not have become a drawn, moved or relabelled one, and a two-object suggestion not be accepted yet.
  // `cur`: the frame as GET /v1/review/labels answers it, `dups`: its `dupGroups`.
  // [{index, s, a, b, drop, keep}]: box indices; drop and keep -1 for two objects.
  // Twin of labels/overlaps.rs `live`: keep the two in step. One difference: `overlap_live` names a
  // suggestion, not its boxes, so where two boxes share a position and the server passed over the
  // first for a relabel made before this page opened, this takes the first.
  function liveSuggestions(cur, dups) {
    var out = [];
    var list = cur && cur.labels.overlap_suggestions, ok = cur && cur.overlap_live;
    if (!list || !ok || cur.labels.overlaps_checked) return out;
    var bs = cur.labels.boxes;
    ok.forEach(function (n) {
      var s = list[n];
      if (!s || s._done) return;
      (dups.overlaps || []).some(function (o) {
        var i = o[0], j = o[1], a, b;
        if (sameBbox(bs[i].bbox, s.a) && sameBbox(bs[j].bbox, s.b)) { a = i; b = j; }
        else if (sameBbox(bs[j].bbox, s.a) && sameBbox(bs[i].bbox, s.b)) { a = j; b = i; }
        else return false;
        var drop = s.verdict === 'drop_a' ? a : (s.verdict === 'drop_b' ? b : -1);
        var d = bs[drop];
        if (d && (d.source === 'human' || d.orig_bbox || d._relabelled)) return false;
        out.push({ index: n, s: s, a: a, b: b, drop: drop, keep: drop < 0 ? -1 : (drop === a ? b : a) });
        return true;
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
    liveSuggestions: liveSuggestions,
    markRejected: markRejected,
    markDuplicate: markDuplicate,
    acceptOverlap: acceptOverlap
  };
}());
