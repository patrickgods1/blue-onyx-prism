// Review tab, fingers (/train/review; loaded before review.js): the gesture rules of the frame on
// a touchscreen, where a box's tag sits in the zoomed view, which box a tap on the overview
// means, which corner of the zoomed frame the overview takes and the quarters of the
// "Anything missing?" look. Pure: no page, no browser API, no state of its own; review.js passes everything in
// and applies the answers. `tests/it/js_review_touch.rs` runs this file in Node.
//
// Coordinates are the canvas's buffer pixels (review.js `canvasPoint`: the frame at zoom 1), and a
// view is {z, ox, oy}: the zoom and the frame point at the canvas's top left, so the frame point
// under the canvas point c is c / z + o (review.js `zoomAt`, `framePoint`).
(function () {
  'use strict';

  function dist(a, b) {
    var dx = a[0] - b[0], dy = a[1] - b[1];
    return Math.sqrt(dx * dx + dy * dy);
  }

  function mid(a, b) { return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]; }

  // Two fingers: `start` = {a, b, view} when the second finger came down, `now` = {a, b}; `limits`
  // = {min (default 1), max}. The zoom follows the ratio of the fingers' distances (clamped), and
  // the frame point under the starting midpoint lands under the current midpoint, so moving both
  // fingers together pans. The caller clamps the offsets to the frame (review.js `clampView`).
  function pinch(start, now, limits) {
    var v = start.view;
    var lo = limits && limits.min != null ? limits.min : 1;
    var hi = limits && limits.max != null ? limits.max : Infinity;
    var d0 = dist(start.a, start.b), d1 = dist(now.a, now.b);
    // Fingers starting on one point give no ratio: the zoom stays, the pan still follows.
    var z = d0 >= 1 ? v.z * d1 / d0 : v.z;
    z = Math.min(hi, Math.max(lo, z));
    var m0 = mid(start.a, start.b), m1 = mid(now.a, now.b);
    var fx = m0[0] / v.z + v.ox, fy = m0[1] / v.z + v.oy;
    return { z: z, ox: fx - m1[0] / z, oy: fy - m1[1] / z };
  }

  // Two taps as opposite corners: the box [x0, y0, x1, y1] in either order of the corners, or null
  // when a side is under `minBox` (two taps on one spot are no box).
  function twoTapRect(a, b, minBox) {
    var r = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
    if (r[2] - r[0] < minBox || r[3] - r[1] < minBox) return null;
    return r;
  }

  // What one finger dragging on the frame does: `resize` (it started on a handle of the selected
  // box), `move` (inside the selected box), `draw` (Draw is on), `pan` (the frame is zoomed) or
  // `none`. The selected box's handles and inside win over Draw: a reviewer who drags a handle
  // means to resize, whichever mode the toggle is in, and Draw stays useful for the next box.
  // A mouse or a pen gets `drawMode: true` (a drag on empty frame draws, as it always did).
  function touchIntent(s) {
    if (s.onHandle) return 'resize';
    if (s.insideSelected) return 'move';
    if (s.drawMode) return 'draw';
    if (s.zoomed) return 'pan';
    return 'none';
  }

  // Where a box's tag goes, against the part of the frame the canvas shows (frame rows `viewTop`
  // .. `viewBottom`; the whole frame at zoom 1): the top of a tag `th` high above the box when it
  // fits in view, else under the box when that fits, else inside the box's top (kept in view, so a
  // box cut by the view's top edge still shows its tag). `bb` = [x0, y0, x1, y1]. At zoom 1 this
  // is the rule the frame always had: above, at the frame's top below, else inside.
  function tagTop(bb, viewTop, viewBottom, th) {
    if (bb[1] - th >= viewTop) return bb[1] - th;
    if (bb[3] + th <= viewBottom) return bb[3];
    return Math.max(bb[1], viewTop);
  }

  // The overview's tap: the index into `bboxes` (each [x0, y0, x1, y1]) of the smallest box
  // holding the point (x, y), the first of equal ones; -1 for none. `pad` (default 0) widens each
  // box on every side for the test only (a finger on a small overview), never for the size.
  function pickBox(bboxes, x, y, pad) {
    var m = pad || 0, best = -1, area = Infinity;
    for (var i = 0; i < bboxes.length; i++) {
      var b = bboxes[i];
      if (x < b[0] - m || x > b[2] + m || y < b[1] - m || y > b[3] + m) continue;
      var a = (b[2] - b[0]) * (b[3] - b[1]);
      if (a < area) { best = i; area = a; }
    }
    return best;
  }

  // The overview's corner over the zoomed frame (two columns): 'tr', 'tl', 'br' or 'bl'. In CSS
  // pixels of the frame as shown: `frame` = [w, h], `size` = the overview's [w, h], `inset` its
  // distance from the frame's edges; `keep` = rects [x0, y0, x1, y1] it must not cover (the
  // selected box with its handles and tag, a proposed or half-drawn box), `others` = rects it
  // would rather not cover (the other boxes). It stays at `current` while that covers nothing of
  // `keep`, so it never moves for nothing; else the corner covering least of `keep`, then least
  // of `others`, ties in the order tr, tl, br, bl.
  var CORNERS = ['tr', 'tl', 'br', 'bl'];

  function cornerRect(corner, frame, size, inset) {
    var x0 = corner.charAt(1) === 'l' ? inset : frame[0] - inset - size[0];
    var y0 = corner.charAt(0) === 't' ? inset : frame[1] - inset - size[1];
    return [x0, y0, x0 + size[0], y0 + size[1]];
  }

  function coveredArea(r, rects) {
    var a = 0;
    for (var i = 0; i < rects.length; i++) {
      var b = rects[i];
      var w = Math.min(r[2], b[2]) - Math.max(r[0], b[0]), h = Math.min(r[3], b[3]) - Math.max(r[1], b[1]);
      if (w > 0 && h > 0) a += w * h;
    }
    return a;
  }

  function overviewCorner(current, frame, size, inset, keep, others) {
    var k = keep || [], o = others || [];
    if (CORNERS.indexOf(current) >= 0 && coveredArea(cornerRect(current, frame, size, inset), k) === 0) return current;
    var best = 'tr', bk = Infinity, bo = Infinity;
    for (var i = 0; i < CORNERS.length; i++) {
      var r = cornerRect(CORNERS[i], frame, size, inset);
      var ck = coveredArea(r, k), co = coveredArea(r, o);
      if (ck < bk || (ck === bk && co < bo)) { best = CORNERS[i]; bk = ck; bo = co; }
    }
    return best;
  }

  // "Anything missing?" in quarters (`q`): the part of a `w` x `h` frame step `i` shows, as
  // [x0, y0, x1, y1] in frame pixels. Steps 0..3 are the top-left, top-right, bottom-left and
  // bottom-right corners, each `share` of the frame's width and height (clamped to 0.5..1), so
  // two neighbours overlap by (2 x share - 1) of the side and an object on a centre line is whole
  // in one of them; any other step is the whole frame. The same share both ways keeps the frame's
  // aspect, so the page shows a rect at zoom w / (x1 - x0) from its top-left corner.
  function quarterView(i, w, h, share) {
    if (!(i >= 0 && i <= 3) || i !== Math.floor(i)) return [0, 0, w, h];
    var s = Math.min(1, Math.max(0.5, share));
    var sw = w * s, sh = h * s;
    var right = i % 2 === 1, bottom = i >= 2;
    // The far edges are the frame's own (never an ulp past it).
    return [right ? w - sw : 0, bottom ? h - sh : 0, right ? w : sw, bottom ? h : sh];
  }

  globalThis.ReviewTouch = {
    quarterView: quarterView,
    pinch: pinch,
    twoTapRect: twoTapRect,
    touchIntent: touchIntent,
    tagTop: tagTop,
    pickBox: pickBox,
    overviewCorner: overviewCorner,
    cornerRect: cornerRect
  };
}());
