// Blue Onyx Prism: Benchmark charts (inline SVG; PrismBench.charts). Loaded after bench-core.js.
// - lineChart(series, opts): lines over a linear x axis, a second (right) axis for series marked
//   `right`, shaded x bands, labelled x marks, interval bars and per-point dots (hollow ones for
//   points that do not compare), and a readout of every series at the pointer. `svg.setCursor(x)`
//   moves a cursor line with a dot on each series (Tune's slider) without drawing again.
// - block(o): a chart with a legend whose buttons hide or show a series (aria-pressed) and a
//   "Show as table" toggle (the Train area's run metrics, train-metrics.js).
// - scatter(points, opts): the goal metric against p95 on a log axis (Compare): each point with
//   its interval bar, its number, filled (GPU) or hollow (CPU), a dashed budget line; each point
//   is a focusable button whose label shows on a click / Enter.
// Every chart has <title>/<desc> (role="img" or "group") and the pages give a table alternative.
// Colors come from classes (style.css `.bn-chart`), never style attributes (CSP style-src 'self').
// Touch: a line chart's readout (what a mouse shows while it moves over the plot) shows on a tap
// and stays (`data-pinned` on the svg; live.js holds a re-render of the section meanwhile) until
// a tap elsewhere or Esc; a sideways finger drag scrubs it (the svg is `touch-action: pan-y
// pinch-zoom` in style.css, so an up-down drag still scrolls the page and a pinch zooms it). In a
// window under 600 px the charts draw a narrower viewBox (fitSize), so their 11-12 px text stays
// about that size on a phone. Pure helpers (fitSize, niceTicks, logTicks) are tested in node by
// tests/it/js_charts.rs with a stub DOM.
(function () {
  var PB = window.PrismBench, has = PB.has, fmt = PB.fmt;
  var SVGNS = "http://www.w3.org/2000/svg", chartId = 0;
  // Dash patterns by series index (color is not the only difference).
  var DASHES = [null, "7 3", "2 3", "9 3 2 3", "4 4", "1 3", "12 4", "6 2 2 2"];
  function dashOf(i) { return DASHES[i % DASHES.length]; }
  function sv(tag, attrs, text) { var e = document.createElementNS(SVGNS, tag); Object.keys(attrs || {}).forEach(function (k) { if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]); }); if (text !== undefined) e.textContent = text; return e; }
  function pct0(v) { return has(v) ? (v * 100).toFixed(0) : "-"; }
  // About n round tick values covering [lo, hi].
  function niceTicks(lo, hi, n) {
    var span = hi - lo; if (!(span > 0)) return [lo];
    var step = Math.pow(10, Math.floor(Math.log(span / n) / Math.LN10)), err = span / n / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var out = [], v = Math.ceil(lo / step - 1e-9) * step;
    for (; v <= hi + step * 1e-6; v += step) out.push(Math.round(v / step) * step);
    return out;
  }
  function niceMax(v) { if (!(v > 0)) return 1; var t = niceTicks(0, v, 5); var last = t[t.length - 1]; return last >= v ? last : last + (t[1] - t[0]); }
  // The 1-2-5 values of a log axis from lo to hi (ms).
  function logTicks(lo, hi) {
    var out = [];
    for (var e = Math.floor(Math.log(lo) / Math.LN10); e <= Math.ceil(Math.log(hi) / Math.LN10); e++) {
      [1, 2, 5].forEach(function (m) { var v = m * Math.pow(10, e); if (v >= lo * 0.999 && v <= hi * 1.001) out.push(Math.round(v * 1e6) / 1e6); });
    }
    return out;
  }

  // The viewBox [width, height] of a chart asked for at W x H that shows `avail` CSS px wide. An
  // svg scales its text with it, so a 560 wide viewBox shown 330 px wide has its 11 px ticks at
  // about 6.5 px: drawn as wide as it shows (at least 240), never wider than asked, the text keeps
  // its size; the height shrinks less, so the plot keeps its height. `avail` 0 = as asked.
  function fitSize(W, H, avail) {
    if (!(avail > 0)) return [W, H];
    var w = Math.max(240, Math.round(avail));
    if (w >= W) return [W, H];
    return [w, Math.round(H * Math.max(w / W, 0.72))];
  }
  // The width a chart drawn now gets: only in a window under NARROW px (a phone; wider windows
  // draw as asked). `o.avail` (the holder's width, which the page passes) wins, then the holder of
  // a block() redrawing one in the page (a legend toggle), else the window less PAD.
  var NARROW = 600, PAD = 100, holderWidth = 0;
  function windowWidth() {
    var d = typeof document !== "undefined" && document.documentElement;
    return (d && d.clientWidth) || (typeof window !== "undefined" && window.innerWidth) || 0;
  }
  function availWidth(o) {
    var vw = windowWidth();
    if (!(vw > 0) || vw >= NARROW) return 0;
    return (o && o.avail > 0) ? o.avail : holderWidth || vw - PAD;
  }
  // A finger (not a mouse): the scatter's hit rings double.
  function coarse() { return typeof window !== "undefined" && !!window.matchMedia && window.matchMedia("(pointer: coarse)").matches; }

  // The one readout kept after a tap (a chart at a time): {svg, clear}. A tap outside its chart
  // or Esc clears it.
  var pinned = null;
  function unpin() { if (pinned) { var p = pinned; pinned = null; p.clear(); } }
  if (typeof document !== "undefined" && document.addEventListener) {
    document.addEventListener("pointerdown", function (e) { if (pinned && !(pinned.svg.contains && pinned.svg.contains(e.target))) unpin(); }, true);
    document.addEventListener("keydown", function (e) { if (e.key === "Escape") unpin(); });
  }

  // Tooltip box inside an SVG: a <g> with a rect sized to its lines.
  function svgTip(svg) {
    var g = sv("g", {class: "thr-tip", visibility: "hidden"}), bg = sv("rect", {rx: 4, x: -6, y: -13}), tx = sv("text", {});
    g.appendChild(bg); g.appendChild(tx); svg.appendChild(g);
    return {
      show: function (lines, x, y, W, L, pw) {
        tx.replaceChildren(); var w = 0;
        lines.forEach(function (l, i) { tx.appendChild(sv("tspan", {x: 0, dy: i ? 15 : 0}, l)); w = Math.max(w, l.length); });
        var bw = Math.max(80, w * 6.3 + 12); bg.setAttribute("width", bw); bg.setAttribute("height", lines.length * 15 + 8);
        var tx0 = x + bw + 10 > L + pw ? x - bw - 4 : x + 10;
        g.setAttribute("transform", "translate(" + Math.max(8, Math.min(tx0, W - bw + 4)) + "," + y + ")"); g.setAttribute("visibility", "visible");
      },
      hide: function () { g.setAttribute("visibility", "hidden"); }
    };
  }

  // series: [{id, label, cls, dash, right, points: [{x, y, low, high, hollow}]}] (y null = a gap);
  // opts: title, desc, width, height, left, xMin/xMax/yMin/yMax, xTicks/yTicks, xFmt/yFmt,
  // xTickText(v), xLabel, yLabel, xName, xHover/yHover, right {max, ticks, fmt, label, hover}
  // (the scale of `right` series), bands [{x1, x2, cls}], marks [{x, cls, label}] (vertical lines),
  // hidden {id: true} (series left out), hlines [{y, right, cls}], dots [{x, y, cls, label, lines}] (above the hit rect; tap = readout),
  // pointDots (a dot per point), bars (each point's low..high), hover "x" | "xy", tip(p, s) ->
  // lines, cursor {x} (and svg.setCursor(x)), avail (the holder's width).
  function lineChart(series, o) {
    o = o || {};
    var size = fitSize(o.width || 560, o.height || 290, availWidth(o));
    var right = o.right || null;
    var W = size[0], H = size[1], L = o.left || 52, R = right ? 44 : 16, T = o.top || 18, B = 44, pw = W - L - R, ph = H - T - B;
    var x0 = has(o.xMin) ? o.xMin : 0, x1 = has(o.xMax) ? o.xMax : 1, y0 = has(o.yMin) ? o.yMin : 0, y1 = has(o.yMax) ? o.yMax : 1;
    var X = function (v) { return L + (v - x0) / ((x1 - x0) || 1) * pw; }, Y = function (v) { return T + (1 - (v - y0) / ((y1 - y0) || 1)) * ph; };
    var rmax = right ? (right.max || 1) : 1, Y2 = function (v) { return T + (1 - Math.min(v, rmax) / rmax) * ph; };
    var scaleOf = function (s) { return s && s.right ? Y2 : Y; };
    var id = "pb-chart-" + (++chartId), hidden = o.hidden || {};
    series = series.filter(function (s) { return !hidden[s.id]; });
    var svg = sv("svg", {viewBox: "0 0 " + W + " " + H, class: "thr-chart bn-chart " + (o.cls || ""), role: "img", "aria-labelledby": id + "-t " + id + "-d"});
    svg.appendChild(sv("title", {id: id + "-t"}, o.title || "Chart"));
    svg.appendChild(sv("desc", {id: id + "-d"}, o.desc || ""));
    var yf = o.yFmt || pct0, xf = o.xFmt || function (v) { return v.toFixed(1); };
    (o.bands || []).forEach(function (b) {
      if (!has(b.x1) || !has(b.x2)) return;
      svg.appendChild(sv("rect", {x: X(b.x1), y: T, width: Math.max(1, X(b.x2) - X(b.x1)), height: ph, class: b.cls || "bn-band"}));
    });
    (o.yTicks || niceTicks(y0, y1, 5)).forEach(function (v) {
      svg.appendChild(sv("line", {x1: L, x2: L + pw, y1: Y(v), y2: Y(v), class: "thr-grid"}));
      svg.appendChild(sv("text", {x: L - 6, y: Y(v) + 4, "text-anchor": "end", class: "thr-tick"}, yf(v)));
    });
    if (right) {
      (right.ticks || niceTicks(0, rmax, 4)).forEach(function (v) {
        svg.appendChild(sv("text", {x: L + pw + 6, y: Y2(v) + 4, class: "thr-tick bn-right"}, (right.fmt || String)(v)));
      });
      if (right.label) svg.appendChild(sv("text", {x: W - 2, y: 11, "text-anchor": "end", class: "thr-tick bn-right"}, right.label));
    }
    (o.xTicks || niceTicks(x0, x1, 5)).forEach(function (v) {
      if (o.xGrid) svg.appendChild(sv("line", {x1: X(v), x2: X(v), y1: T, y2: T + ph, class: "thr-grid"}));
      svg.appendChild(sv("line", {x1: X(v), x2: X(v), y1: T + ph, y2: T + ph + 4, class: "thr-axis"}));
      svg.appendChild(sv("text", {x: X(v), y: T + ph + 16, "text-anchor": "middle", class: "thr-tick"}, o.xTickText ? o.xTickText(v) : xf(v)));
    });
    svg.appendChild(sv("line", {x1: L, x2: L + pw, y1: T + ph, y2: T + ph, class: "thr-axis"}));
    svg.appendChild(sv("line", {x1: L, x2: L, y1: T, y2: T + ph, class: "thr-axis"}));
    if (o.xLabel) svg.appendChild(sv("text", {x: L + pw, y: H - 6, "text-anchor": "end", class: "thr-label"}, o.xLabel));
    if (o.yLabel) svg.appendChild(sv("text", {x: 12, y: T + ph / 2, "text-anchor": "middle", class: "thr-label", transform: "rotate(-90 12 " + (T + ph / 2) + ")"}, o.yLabel));
    (o.hlines || []).forEach(function (h) {
      if (!has(h.y)) return;
      var yy = (h.right ? Y2 : Y)(h.y);
      svg.appendChild(sv("line", {x1: L, x2: L + pw, y1: yy, y2: yy, class: h.cls || "bn-target"}));
    });
    series.forEach(function (s, i) {
      var d = "", pen = false, Ys = scaleOf(s);
      s.points.forEach(function (p) {
        if (!has(p.y) || !has(p.x)) { pen = false; return; }
        d += (pen ? "L" : "M") + X(p.x).toFixed(1) + " " + Ys(p.y).toFixed(1); pen = true;
      });
      var cls = s.cls || "ser ser-" + ((i % 8) + 1);
      if (d && !s.noLine) svg.appendChild(sv("path", {d: d, class: "thr-line " + cls, "stroke-dasharray": s.dash || null}));
      if (o.bars) s.points.forEach(function (p) {
        if (!has(p.x) || !has(p.low) || !has(p.high)) return;
        svg.appendChild(sv("line", {x1: X(p.x), x2: X(p.x), y1: Ys(p.high), y2: Ys(p.low), class: "bn-bar " + cls}));
      });
      if (o.pointDots) s.points.forEach(function (p) {
        if (!has(p.y) || !has(p.x)) return;
        var c = sv("circle", {cx: X(p.x), cy: Ys(p.y), r: 4.5, class: "pb-dot " + cls + (p.hollow ? " hollow" : "")});
        if (p.label) c.appendChild(sv("title", {}, p.label));
        svg.appendChild(c);
      });
    });
    (o.marks || []).forEach(function (m, k) {
      if (!has(m.x)) return;
      svg.appendChild(sv("line", {x1: X(m.x), x2: X(m.x), y1: T, y2: T + ph, class: m.cls || "bn-mark"}));
      if (m.label) {
        var lx = X(m.x) + 4, anchor = "start";
        if (lx + m.label.length * 6.4 > L + pw) { lx = X(m.x) - 4; anchor = "end"; }
        svg.appendChild(sv("text", {x: lx, y: T - 5 + 13 * k, "text-anchor": anchor, class: "thr-tick " + (m.textCls || "")}, m.label));
      }
    });
    // The cursor (Tune's slider): a line and a dot per series.
    var cursor = null;
    if (o.cursor) {
      cursor = {line: sv("line", {y1: T, y2: T + ph, class: "bn-cursor"}), dots: []};
      svg.appendChild(cursor.line);
      series.forEach(function (s, i) { var c = sv("circle", {r: 4.5, class: "bn-cursor-dot " + (s.cls || "ser ser-" + ((i % 8) + 1))}); cursor.dots.push(c); svg.appendChild(c); });
    }
    function at(s, x) { var b = null; s.points.forEach(function (p) { if (has(p.y) && (!b || Math.abs(p.x - x) < Math.abs(b.x - x))) b = p; }); return b; }
    svg.setCursor = function (x) {
      if (!cursor || !has(x)) return;
      cursor.line.setAttribute("x1", X(x)); cursor.line.setAttribute("x2", X(x));
      series.forEach(function (s, i) {
        var p = at(s, x), c = cursor.dots[i];
        if (!p || Math.abs(p.x - x) > (x1 - x0) * 0.02) { c.setAttribute("visibility", "hidden"); return; }
        c.setAttribute("visibility", "visible"); c.setAttribute("cx", X(p.x)); c.setAttribute("cy", scaleOf(s)(p.y));
      });
    };
    if (o.cursor) svg.setCursor(o.cursor.x);
    // Hover: the nearest point(s) and their exact values. The hit rect takes the pointer; the
    // dots above it keep their <title> (desktop hover) and show it as the readout on a tap; the
    // guide, the hover dot and the readout are on top and let the pointer through (style.css).
    var hit = sv("rect", {x: L, y: T, width: pw, height: ph, class: "thr-hit"}); svg.appendChild(hit);
    var dotEls = [];
    (o.dots || []).forEach(function (p) {
      if (!has(p.x) || !has(p.y)) return;
      var c = sv("circle", {cx: X(p.x), cy: Y(p.y), r: 5, class: p.cls}); if (p.label) c.appendChild(sv("title", {}, p.label)); svg.appendChild(c);
      dotEls.push([c, p]);
    });
    var guide = sv("line", {y1: T, y2: T + ph, class: "thr-guide", visibility: "hidden"}), hov = sv("circle", {r: 3.5, class: "bn-hover", visibility: "hidden"});
    svg.appendChild(guide); svg.appendChild(hov);
    var tipBox = svgTip(svg);
    function nearest(evt) {
      var r = svg.getBoundingClientRect(); if (!r.width) return null;
      var mx = (evt.clientX - r.left) * W / r.width, my = (evt.clientY - r.top) * H / r.height, best = null;
      series.forEach(function (s) {
        var Ys = scaleOf(s);
        s.points.forEach(function (p) {
          if (!has(p.x) || !has(p.y)) return;
          var dist = o.hover === "xy" ? Math.pow(X(p.x) - mx, 2) + Math.pow(Ys(p.y) - my, 2) : Math.abs(X(p.x) - mx);
          if (!best || dist < best.dist) best = {dist: dist, p: p, s: s};
        });
      });
      return best;
    }
    // The readout at the pointer of `evt`; false when there is nothing to show.
    function readout(evt) {
      var h = nearest(evt); if (!h) return false;
      var lines;
      if (o.tip) lines = o.tip(h.p, h.s);
      else if (o.hover === "xy") lines = [h.s.label, (o.xName || "x") + " " + (o.xHover || xf)(h.p.x) + ", " + (o.yName || "y") + " " + (o.yHover || yf)(h.p.y)];
      else {
        lines = [(o.xName || "") + " " + (o.xHover || xf)(h.p.x)];
        series.slice(0, 8).forEach(function (s) {
          var q = at(s, h.p.x);
          if (q && Math.abs(q.x - h.p.x) <= (x1 - x0) * 0.03) lines.push(s.label + ": " + (s.right ? ((right && right.hover) || String) : (o.yHover || yf))(q.y));
        });
      }
      if (!lines || !lines.length) return false;
      var Ys = scaleOf(h.s);
      if (o.hover === "xy") { hov.setAttribute("cx", X(h.p.x)); hov.setAttribute("cy", Ys(h.p.y)); hov.setAttribute("visibility", "visible"); guide.setAttribute("visibility", "hidden"); }
      else { guide.setAttribute("x1", X(h.p.x)); guide.setAttribute("x2", X(h.p.x)); guide.setAttribute("visibility", "visible"); }
      var ty = o.hover === "xy" ? Math.min(Ys(h.p.y) + 18, T + ph - lines.length * 15) : T + 30;
      tipBox.show(lines, X(h.p.x), ty, W, L, pw);
      return true;
    }
    function clear() {
      guide.setAttribute("visibility", "hidden"); hov.setAttribute("visibility", "hidden"); tipBox.hide();
      svg.removeAttribute("data-pinned");
      if (pinned && pinned.svg === svg) pinned = null;
    }
    // Keep what shows until a tap elsewhere, Esc or a re-render.
    function pin() {
      if (pinned && pinned.svg !== svg) unpin();
      pinned = {svg: svg, clear: clear}; svg.setAttribute("data-pinned", "");
    }
    // A finger on the plot: {x, y} where it went down, `scrub` once it moved sideways.
    var touch = null;
    hit.addEventListener("pointerdown", function (evt) {
      if (evt.pointerType === "mouse") return;
      touch = {x: evt.clientX, y: evt.clientY, scrub: false};
    });
    hit.addEventListener("pointermove", function (evt) {
      if (evt.pointerType === "mouse" || !touch) { if (evt.pointerType === "mouse" || !evt.buttons) readout(evt); return; }
      var dx = Math.abs(evt.clientX - touch.x), dy = Math.abs(evt.clientY - touch.y);
      if (!touch.scrub && dx > 6 && dx > dy) touch.scrub = true;
      if (touch.scrub && readout(evt)) pin();
    });
    hit.addEventListener("pointerup", function (evt) {
      if (evt.pointerType === "mouse") { if (o.onPick) { var h = nearest(evt); if (h) o.onPick(h.p.x); } return; }
      var wasScrub = touch && touch.scrub; touch = null;
      if (readout(evt)) pin(); else clear();
      if (o.onPick && !wasScrub) { var n = nearest(evt); if (n) o.onPick(n.p.x); }
    });
    // The page took the gesture (an up-down scroll): a scrub keeps its last readout.
    hit.addEventListener("pointercancel", function () {
      var scrubbed = touch && touch.scrub; touch = null;
      if (!scrubbed) clear();
    });
    hit.addEventListener("pointerleave", function (evt) {
      if (evt.pointerType !== "mouse" && svg.hasAttribute("data-pinned")) return;
      clear();
    });
    // A dot: its text as the readout on a tap (a mouse gets the <title>).
    dotEls.forEach(function (d) {
      var c = d[0], p = d[1];
      c.addEventListener("pointerup", function (evt) {
        if (evt.pointerType === "mouse") return;
        var lines = p.lines || (p.label ? [p.label] : []); if (!lines.length) return;
        guide.setAttribute("visibility", "hidden"); hov.setAttribute("visibility", "hidden");
        tipBox.show(lines, X(p.x), Math.min(Y(p.y) + 18, T + ph - lines.length * 15), W, L, pw);
        pin();
      });
    });
    return svg;
  }

  // The goal metric against speed. points: [{key, n (the number drawn), x (ms), y, low, high,
  // color (1..8), hollow (a CPU device), label, sub}], opts: title, desc, xLabel, yLabel, yFmt,
  // budget (ms), yMin/yMax, onPick(key), avail. A log x axis (p95 spans 10x and more).
  function scatter(points, o) {
    o = o || {};
    var size = fitSize(o.width || 620, o.height || 300, availWidth(o)), ring = coarse() ? 22 : 11;
    var W = size[0], H = size[1], L = 48, R = 16, T = 18, B = 44, pw = W - L - R, ph = H - T - B;
    var pts = points.filter(function (p) { return has(p.x) && p.x > 0 && has(p.y); });
    var xs = pts.map(function (p) { return p.x; }).concat(has(o.budget) && o.budget > 0 ? [o.budget] : []);
    var lo = xs.length ? Math.min.apply(null, xs) : 10, hi = xs.length ? Math.max.apply(null, xs) : 1000;
    var lx0 = Math.floor(Math.log(lo / 1.3) / Math.LN10 * 2) / 2, lx1 = Math.ceil(Math.log(hi * 1.3) / Math.LN10 * 2) / 2;
    if (lx1 - lx0 < 1) { lx0 -= 0.5; lx1 += 0.5; }
    var ys = [];
    pts.forEach(function (p) { ys.push(p.y); if (has(p.low)) ys.push(p.low); if (has(p.high)) ys.push(p.high); });
    var y0 = has(o.yMin) ? o.yMin : (ys.length ? Math.floor((Math.min.apply(null, ys) - 0.02) * 20) / 20 : 0);
    var y1 = has(o.yMax) ? o.yMax : (ys.length ? Math.ceil((Math.max.apply(null, ys) + 0.02) * 20) / 20 : 1);
    if (!(y1 > y0)) { y0 -= 0.05; y1 += 0.05; }
    var X = function (v) { return L + (Math.log(v) / Math.LN10 - lx0) / (lx1 - lx0) * pw; }, Y = function (v) { return T + (1 - (v - y0) / (y1 - y0)) * ph; };
    var id = "pb-scatter-" + (++chartId);
    var svg = sv("svg", {viewBox: "0 0 " + W + " " + H, class: "thr-chart bn-chart pb-scatter", role: "group", "aria-labelledby": id + "-t " + id + "-d"});
    svg.appendChild(sv("title", {id: id + "-t"}, o.title || "Goal metric against speed"));
    svg.appendChild(sv("desc", {id: id + "-d"}, o.desc || ""));
    var yf = o.yFmt || function (v) { return v.toFixed(2); };
    niceTicks(y0, y1, 5).forEach(function (v) {
      svg.appendChild(sv("line", {x1: L, x2: L + pw, y1: Y(v), y2: Y(v), class: "thr-grid"}));
      svg.appendChild(sv("text", {x: L - 6, y: Y(v) + 4, "text-anchor": "end", class: "thr-tick"}, yf(v)));
    });
    var ticks = logTicks(Math.pow(10, lx0), Math.pow(10, lx1));
    if (pw < 360) ticks = ticks.filter(function (v) { var m = Math.round(v / Math.pow(10, Math.floor(Math.log(v) / Math.LN10 + 1e-9))); return m !== 2; });
    ticks.forEach(function (v) {
      svg.appendChild(sv("line", {x1: X(v), x2: X(v), y1: T, y2: T + ph, class: "thr-grid"}));
      svg.appendChild(sv("text", {x: X(v), y: T + ph + 16, "text-anchor": "middle", class: "thr-tick"}, fmt.ms(v)));
    });
    svg.appendChild(sv("line", {x1: L, x2: L + pw, y1: T + ph, y2: T + ph, class: "thr-axis"}));
    svg.appendChild(sv("line", {x1: L, x2: L, y1: T, y2: T + ph, class: "thr-axis"}));
    svg.appendChild(sv("text", {x: L + pw, y: H - 6, "text-anchor": "end", class: "thr-label"}, o.xLabel || "p95 (log scale) →"));
    if (o.yLabel) svg.appendChild(sv("text", {x: L + 4, y: T - 6, class: "thr-label"}, o.yLabel));
    if (has(o.budget) && o.budget > 0) {
      var bx = X(o.budget);
      svg.appendChild(sv("line", {x1: bx, x2: bx, y1: T, y2: T + ph, class: "bn-budget"}));
      svg.appendChild(sv("text", {x: bx + 4, y: T + 12, class: "thr-tick bn-budget-text"}, "budget " + fmt.ms(o.budget)));
    }
    var label = sv("text", {class: "pb-pt-label", visibility: "hidden"}), labelBg = sv("rect", {class: "pb-pt-label-bg", rx: 3, visibility: "hidden"});
    function showLabel(p) {
      var text = (p.n ? p.n + ". " : "") + p.label + ": " + yf(p.y) + ", " + fmt.ms(p.x);
      var w = text.length * 6.3 + 10, x = X(p.x) + w + 12 > L + pw ? X(p.x) - w - 8 : X(p.x) + 10, y = Math.max(T + 12, Y(p.y) - 14);
      x = Math.max(2, Math.min(x, W - w - 2));
      label.textContent = text; label.setAttribute("x", x + 5); label.setAttribute("y", y + 4); label.setAttribute("visibility", "visible");
      labelBg.setAttribute("x", x); labelBg.setAttribute("y", y - 9); labelBg.setAttribute("width", w); labelBg.setAttribute("height", 18); labelBg.setAttribute("visibility", "visible");
    }
    // Numbers placed clear of each other and of the points (BenchCore.placeLabels).
    var BCx = window.BenchCore, spots = BCx && BCx.placeLabels ? BCx.placeLabels(pts.map(function (p) { return {x: X(p.x), y: Y(p.y), text: String(p.n || "")}; })) : null;
    pts.forEach(function (p, i) {
      var cx = X(p.x), cy = Y(p.y), color = "ser-" + (p.color || 1), spot = spots ? spots[i] : {x: cx, y: cy - 10, anchor: "middle"};
      var g = sv("g", {class: "pb-pt " + color + (p.hollow ? " hollow" : ""), "data-key": p.key, tabindex: 0, role: "button", "aria-label": (p.n ? p.n + ". " : "") + p.label + ": " + yf(p.y) + ", p95 " + fmt.ms(p.x) + (p.hollow ? ", CPU" : "")});
      if (has(p.low) && has(p.high)) g.appendChild(sv("line", {x1: cx, x2: cx, y1: Y(p.high), y2: Y(p.low), class: "bn-bar " + color}));
      g.appendChild(sv("circle", {cx: cx, cy: cy, r: ring, class: "pb-pt-ring"}));
      g.appendChild(sv("circle", {cx: cx, cy: cy, r: 6, class: "pb-pt-mark"}));
      if (p.n) g.appendChild(sv("text", {x: spot.x, y: spot.y, "text-anchor": spot.anchor, class: "pb-pt-n"}, String(p.n)));
      g.appendChild(sv("title", {}, (p.n ? p.n + ". " : "") + p.label + ": " + yf(p.y) + ", " + fmt.ms(p.x)));
      var pick = function () { showLabel(p); if (o.onPick) o.onPick(p.key); };
      g.addEventListener("click", pick);
      g.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      svg.appendChild(g);
    });
    svg.appendChild(labelBg); svg.appendChild(label);
    return svg;
  }

  // Hidden series per chart key: survives re-renders.
  var hiddenBy = {};
  // A chart with a legend (series toggles) and "Show as table". o: key, series (legend items
  // {id, label, cls, dash}), draw(hidden) -> svg, table() -> {head: [...], rows: [[...]]}, legend
  // (default true when more than one series), note (shown under the legend), cls.
  function block(o) {
    var el = PB.el, box = el("div", "chart-block " + (o.cls || "")), hidden = hiddenBy[o.key] = hiddenBy[o.key] || {};
    box.dataset.keep = "c:" + o.key;
    var holder = el("div", "chart-holder");
    function redraw() {
      holderWidth = holder.isConnected ? holder.clientWidth : 0;
      try { holder.replaceChildren(o.draw(hidden)); } finally { holderWidth = 0; }
    }
    if (o.legend !== false && o.series && o.series.length > 1) {
      var lg = el("div", "legend"); lg.setAttribute("role", "group"); lg.setAttribute("aria-label", "Show or hide series");
      o.series.forEach(function (s) {
        var b = el("button", "legend-item"); b.type = "button"; b.setAttribute("aria-pressed", hidden[s.id] ? "false" : "true");
        var sw = sv("svg", {viewBox: "0 0 22 10", class: "lg-swatch", "aria-hidden": "true"});
        sw.appendChild(sv("line", {x1: 1, x2: 21, y1: 5, y2: 5, class: "thr-line " + (s.cls || ""), "stroke-dasharray": s.dash || null}));
        b.appendChild(sw); b.appendChild(document.createTextNode(s.label));
        b.title = "Show or hide " + s.label;
        b.addEventListener("click", function () { if (hidden[s.id]) delete hidden[s.id]; else hidden[s.id] = true; b.setAttribute("aria-pressed", hidden[s.id] ? "false" : "true"); redraw(); });
        lg.appendChild(b);
      });
      box.appendChild(lg);
    }
    if (o.note) box.appendChild(el("p", "muted chart-note", o.note));
    redraw(); box.appendChild(holder);
    if (o.table) {
      var d = el("details", "chart-table"); d.dataset.keep = "d:ct:" + o.key; d.appendChild(el("summary", null, "Show as table"));
      var built = false;
      d.addEventListener("toggle", function () {
        if (!d.open || built) return; built = true;
        var t = o.table(), tb = el("table", "bench chart-data"), th = el("thead"), tr = el("tr");
        t.head.forEach(function (h, i) { tr.appendChild(el("th", i ? "num" : null, h)); }); th.appendChild(tr); tb.appendChild(th);
        var body = el("tbody"); t.rows.forEach(function (r) { var row = el("tr"); r.forEach(function (c, i) { row.appendChild(el("td", i ? "num" : null, c)); }); body.appendChild(row); }); tb.appendChild(body);
        var w = el("div", "scroll"); w.tabIndex = 0; w.setAttribute("role", "region"); w.setAttribute("aria-label", ((holder.querySelector("svg > title") || {}).textContent || o.title || "Chart") + ": data");
        w.appendChild(tb); d.appendChild(w);
        if (window.PrismSort) PrismSort.attach(tb, {key: "chart:" + String(o.key).split(":")[0]});
      });
      box.appendChild(d);
    }
    return box;
  }

  PB.charts = {sv: sv, lineChart: lineChart, scatter: scatter, block: block, dashOf: dashOf, niceTicks: niceTicks, niceMax: niceMax, logTicks: logTicks, fitSize: fitSize, unpin: unpin, hidden: hiddenBy};
})();
