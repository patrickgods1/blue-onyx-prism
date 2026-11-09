// Review tab, fingers (/train/review; loaded before review.js): the gesture rules of the frame on
// a touchscreen. Pure: no page, no browser API, no state of its own; review.js passes everything in
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

  globalThis.ReviewTouch = {
    pinch: pinch,
    twoTapRect: twoTapRect,
    touchIntent: touchIntent
  };
}());
