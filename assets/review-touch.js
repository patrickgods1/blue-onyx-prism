// Review tab, fingers (/train/review; loaded before review.js): the gesture rules of the frame on
// a touchscreen, where a box's tag sits in the zoomed view, which box a tap on the overview
// means, which corner of the frame the overview takes (corner layout) and the quarters of the
// "Anything missing?" look, plus the editor's pointer and view rules: which edge or corner of a
// box a press grabs (`edgeAt`), how the view follows a new selection (`nextView`) and a drag of
// the overview's view rect (`overviewPan`); the view-only image setting (`viewSettings`, the tone
// table `viewLut`), the box tags' size (`tagSize`) and the side panel's width (`sideWidth`).
// Pure: no page, no browser API, no state of its own; review.js passes everything in and applies
// the answers. `tests/it/js_review_touch.rs` runs this file in Node.
//
// Coordinates are frame pixels, a canvas point being the frame point at zoom 1 (review.js
// `canvasPoint`; the canvas's buffer has its own, device pixel size), and a view is {z, ox, oy}:
// the zoom and the frame point at the canvas's top left, so the frame point under the canvas
// point c is c / z + o (review.js `zoomAt`, `framePoint`).
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

  // The overview's corner over the frame (two columns, corner layout, at any zoom): 'tr', 'tl',
  // 'br' or 'bl'. In CSS pixels of the frame as shown: `frame` = [w, h], `size` = the overview's
  // [w, h], `inset` its distance from the frame's edges; `keep` = rects [x0, y0, x1, y1] it must
  // not cover (the selected box with its grab band and tag, a proposed or half-drawn box),
  // `others` = rects it would rather not cover (the other boxes). It stays at `current` while
  // that covers nothing of `keep`, so it never moves for nothing; else the corner covering least
  // of `keep`, then least of `others`, ties in the order tr, tl, br, bl.
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

  function finite(v) { return typeof v === 'number' && isFinite(v); }

  // Which edge or corner of the box `bbox` = [x0, y0, x1, y1] a press at `p` = [x, y] grabs, all
  // in frame units: 'nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w', or null (no edge: inside, that is
  // a move, or off the box). Each edge has a band `grabOut` outside it and `grabIn` inside it (the
  // caller caps `grabIn` to a third of the shorter side, so a small box keeps a middle to move
  // by); a side band spans the box's length plus `grabOut` at both ends, so a press just past a
  // corner still takes it. A vertical and a horizontal band together are that corner. On a box
  // narrower than its two bands both opposite bands hold the point: the nearer edge wins, and a
  // tie (a press on a zero-width box, or the exact middle) goes to the right and the bottom, so a
  // drag from a degenerate box grows it right and down, the way a drawn box grows (a press beside
  // it is in one band only and takes that side). A non-finite input is no edge.
  function edgeAt(bbox, p, grabOut, grabIn) {
    if (!bbox || !p) return null;
    var vals = [bbox[0], bbox[1], bbox[2], bbox[3], p[0], p[1], grabOut, grabIn];
    for (var i = 0; i < vals.length; i++) if (!finite(vals[i])) return null;
    // Either order of the corners (a box mid-resize may be inside out); bands are never negative.
    var x0 = Math.min(bbox[0], bbox[2]), x1 = Math.max(bbox[0], bbox[2]);
    var y0 = Math.min(bbox[1], bbox[3]), y1 = Math.max(bbox[1], bbox[3]);
    var go = Math.max(0, grabOut), gi = Math.max(0, grabIn), x = p[0], y = p[1];
    var alongY = y >= y0 - go && y <= y1 + go, alongX = x >= x0 - go && x <= x1 + go;
    var w = alongY && x >= x0 - go && x <= x0 + gi, e = alongY && x >= x1 - gi && x <= x1 + go;
    var n = alongX && y >= y0 - go && y <= y0 + gi, s = alongX && y >= y1 - gi && y <= y1 + go;
    if (w && e) { if (Math.abs(x - x0) < Math.abs(x - x1)) e = false; else w = false; }
    if (n && s) { if (Math.abs(y - y0) < Math.abs(y - y1)) s = false; else n = false; }
    var side = (n ? 'n' : s ? 's' : '') + (w ? 'w' : e ? 'e' : '');
    return side || null;
  }

  // What the zoom does when the selection moves to another box or another frame: 'keep' (the
  // same box stays selected: leave the view as the reviewer has it), 'box' (zoom to the box) or
  // 'fit' (the whole frame). `o` = {follow, custom, sameBox}: `follow` is the reviewer's last
  // Fit / Zoom-to-box choice, remembered; `custom` says they zoomed or panned by hand since (wheel,
  // pinch, pan, an overview drag), which means they want a close look, so the next box is framed
  // by Zoom to box rather than thrown back to the whole frame or left at a view made for another box.
  function nextView(o) {
    if (o.sameBox) return 'keep';
    if (o.custom) return 'box';
    return o.follow ? 'box' : 'fit';
  }

  // Dragging the view rect on the overview: `view` = {z, ox, oy}, `frame` = [W, H] and `delta` =
  // [dx, dy], all in frame units. The new {ox, oy} is moved by the delta and held in the frame,
  // [0, W - W / z] x [0, H - H / z], as review.js `clampView` holds it; a view under zoom 1.001
  // is the whole frame there (it snaps to 1), so it is {0, 0} here. Non-finite: a zoom or frame
  // gives {0, 0}, an offset or delta counts as 0.
  function overviewPan(view, frame, delta) {
    var z = view && view.z, W = frame && frame[0], H = frame && frame[1];
    if (!finite(z) || !finite(W) || !finite(H) || z < 1.001) return { ox: 0, oy: 0 };
    var ox = finite(view.ox) ? view.ox : 0, oy = finite(view.oy) ? view.oy : 0;
    var dx = delta && finite(delta[0]) ? delta[0] : 0, dy = delta && finite(delta[1]) ? delta[1] : 0;
    var hx = Math.max(0, W - W / z), hy = Math.max(0, H - H / z);
    return { ox: Math.min(hx, Math.max(0, ox + dx)), oy: Math.min(hy, Math.max(0, oy + dy)) };
  }

  // ---- the view-only image setting (review.js viewSource) ----

  // A level of the Brightness or Contrast slider: an integer from -100 to 100; anything else
  // (not a number, NaN, Infinity) is 0.
  function viewLevel(v) {
    return finite(v) ? Math.max(-100, Math.min(100, Math.round(v))) : 0;
  }

  // The setting kept in storage (`bop.review.view`), checked: `mode` auto | off | bright | eq
  // (else auto), `brightness` and `contrast` levels (viewLevel). An older page kept a `gamma`
  // (0.4 .. 2.5, from `[` / `]`) instead of a brightness: it becomes the brightness whose gamma
  // it is (viewLut: g = 2.5^(b / 100), so b = 100 ln g / ln 2.5, rounded and clamped).
  function viewSettings(raw) {
    var o = raw && typeof raw === 'object' ? raw : {};
    var mode = typeof o.mode === 'string' && /^(auto|off|bright|eq)$/.test(o.mode) ? o.mode : 'auto';
    var b = viewLevel(o.brightness);
    if (o.brightness == null && finite(o.gamma) && o.gamma > 0) b = viewLevel(100 * Math.log(o.gamma) / Math.log(2.5));
    return { mode: mode, brightness: b, contrast: viewLevel(o.contrast) };
  }

  // viewLut's step 1 for level `x` (0 .. 255) whose cumulative share is `cx`: 0 .. 1.
  function toneStep1(x, cx, mode, lo, span) {
    var v = x / 255;
    if (mode === 'bright' || mode === 'eq') v = Math.max(0, Math.min(1, (x - lo) / span));
    if (mode === 'eq') v = 0.75 * cx + 0.25 * v;
    return mode === 'bright' ? Math.pow(v, 1 / 1.5) : v;
  }

  // The 256-entry tone table of the view-only image setting, from the frame's luma histogram
  // `hist` (256 counts, `total` of them), in this order:
  //   1. the mode: `off` leaves the levels; `bright` stretches them (the 1% .. 99.5% levels to
  //      0 .. 1) and lifts them (gamma 1.5); `eq` mostly equalises them (3/4 the histogram's
  //      cumulative share, 1/4 the stretch);
  //   2. contrast `c` (-100 .. 100): a slope of k = 2^(c / 50) (1/4 .. 4) about the level the
  //      frame's median luma has after step 1, clamped to 0 .. 1; the median stays where it is,
  //      so more contrast on a dark frame does not push it to black;
  //   3. brightness `b` (-100 .. 100) as a gamma, v^(1 / g) with g = 2.5^(b / 100) (0.4 .. 2.5):
  //      it lifts the shadows without clipping the highlights.
  // Off, 0 and 0 is the identity. `out` (optional, 256 entries) is filled and returned, so a
  // slider moved over and over makes no new table; without it a Uint8ClampedArray is made. No
  // other array is made: the cumulative shares are summed as the loops go.
  function viewLut(hist, total, mode, brightness, contrast, out) {
    var lut = out || new Uint8ClampedArray(256);
    var n = finite(total) && total > 0 ? total : 1;
    var b = viewLevel(brightness), c = viewLevel(contrast);
    // The levels holding 1%, 50% and 99.5% of the pixels (first of the cumulative share), and
    // the cumulative share at the median.
    var acc = 0, lo = -1, med = -1, top = -1, cMed = 0, i, cdf;
    for (i = 0; i < 256; i++) {
      acc += hist[i] || 0;
      cdf = acc / n;
      if (lo < 0 && cdf >= 0.01) lo = i;
      if (med < 0 && cdf >= 0.5) { med = i; cMed = cdf; }
      if (top < 0 && cdf >= 0.995) top = i;
    }
    if (lo < 0 || lo > 254) lo = 254;
    if (med < 0) med = 255;
    var hi = Math.max(lo + 1, top < 0 ? 255 : top), span = hi - lo;
    var k = Math.pow(2, c / 50), pivot = toneStep1(med, cMed, mode, lo, span), gi = 1 / Math.pow(2.5, b / 100);
    acc = 0;
    for (i = 0; i < 256; i++) {
      acc += hist[i] || 0;
      var v = toneStep1(i, acc / n, mode, lo, span);
      if (c !== 0) v = Math.max(0, Math.min(1, pivot + k * (v - pivot)));
      if (b !== 0) v = Math.pow(v, gi);
      lut[i] = Math.round(255 * v);
    }
    return lut;
  }

  // ---- the box tags' size (View's Label size; review.js draw) ----

  // The tags' font size in CSS px: a whole number from TAG_MIN to TAG_MAX; anything that is no
  // finite number (nothing kept, a damaged value) is TAG_SIZE.
  var TAG_MIN = 8, TAG_MAX = 24, TAG_SIZE = 12;
  function tagSize(v) {
    return finite(v) ? Math.max(TAG_MIN, Math.min(TAG_MAX, Math.round(v))) : TAG_SIZE;
  }

  // ---- the side panel's width (two columns; review.js) ----

  // The width in CSS px the panel takes for a wanted width `want`: rounded and held in
  // [min, max]; a `max` under `min` is `min`, one that is no finite number (Infinity: the page
  // before it knows the grid's width) no top. A non-finite `want` is `dflt` (held too).
  function sideWidth(want, min, max, dflt) {
    var w = finite(want) ? want : dflt;
    var hi = !finite(max) ? Infinity : (max > min ? max : min);
    return Math.round(Math.max(min, Math.min(hi, w)));
  }

  globalThis.ReviewTouch = {
    viewLevel: viewLevel,
    viewSettings: viewSettings,
    viewLut: viewLut,
    tagSize: tagSize,
    sideWidth: sideWidth,
    quarterView: quarterView,
    pinch: pinch,
    twoTapRect: twoTapRect,
    touchIntent: touchIntent,
    tagTop: tagTop,
    pickBox: pickBox,
    overviewCorner: overviewCorner,
    cornerRect: cornerRect,
    edgeAt: edgeAt,
    nextView: nextView,
    overviewPan: overviewPan
  };
}());
