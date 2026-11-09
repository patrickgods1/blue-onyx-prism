// Blue Onyx Prism: Benchmark charts (inline SVG; PrismBench.charts).
// lineChart(series, opts) and scatter(points, opts) draw the charts; block(o) wraps one with a
// legend whose buttons hide or show a series (aria-pressed) and a "Show as table" toggle.
// chartSvg (precision / recall / F1 vs confidence) and rocSvg (frame-level ROC) are the
// threshold charts built on them. Every chart has <title>/<desc>; series differ in dash pattern
// as well as color (--series-1..8 in style.css). Colors come from classes, never style
// attributes (CSP style-src 'self').
// Touch: a line chart's readout (what a mouse shows while it moves over the plot) shows on a tap
// and stays (`data-pinned` on the svg; live.js holds a re-render of the section meanwhile) until
// a tap elsewhere or Esc; a sideways finger drag scrubs it (the svg is `touch-action: pan-y
// pinch-zoom` in style.css, so an up-down drag still scrolls the page and a pinch zooms it). In a
// window under 600 px the charts draw a narrower viewBox (fitSize), so their 11-12 px text stays
// about that size on a phone.
(function () {
  var PB = window.PrismBench, el = PB.el, has = PB.has, fmt = PB.fmt;
  var SVGNS = "http://www.w3.org/2000/svg", chartId = 0;
  function sv(tag, attrs, text) { var e = document.createElementNS(SVGNS, tag); Object.keys(attrs || {}).forEach(function (k) { if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]); }); if (text !== undefined) e.textContent = text; return e; }
  // Dash patterns by series index (color is not the only difference).
  var DASHES = [null, "7 3", "2 3", "9 3 2 3", "4 4", "1 3", "12 4", "6 2 2 2"];
  function dashOf(i) { return DASHES[i % DASHES.length]; }
  function pct0(v) { return has(v) ? (v * 100).toFixed(0) : "-"; }
  function pct1(v) { return has(v) ? (v * 100).toFixed(1) : "-"; }
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

  // The viewBox [width, height] of a chart asked for at W x H that shows `avail` CSS px wide. An
  // svg scales its text with it, so a 560 wide viewBox shown 330 px wide has its 11 px ticks at
  // about 6.5 px: drawn as wide as it shows (at least 240), never wider than asked, the text keeps
  // its size; the height shrinks less, so the plot keeps its height. `avail` 0 = as asked.
  // Pure (node test in tests/it/js_charts.rs).
  function fitSize(W, H, avail) {
    if (!(avail > 0)) return [W, H];
    var w = Math.max(240, Math.round(avail));
    if (w >= W) return [W, H];
    return [w, Math.round(H * Math.max(w / W, 0.72))];
  }
  // The width a chart drawn now gets: only in a window under NARROW px (a phone; wider windows
  // draw as before). The holder's own width when block() redraws one in the page (a legend
  // toggle), else the window less PAD (page, card and details padding: a phone's charts show
  // 290..335 px wide in a 390 px window; a little narrower than shown only makes the text a
  // little larger).
  var NARROW = 600, PAD = 100, holderWidth = 0;
  function windowWidth() {
    var d = typeof document !== "undefined" && document.documentElement;
    return (d && d.clientWidth) || (typeof window !== "undefined" && window.innerWidth) || 0;
  }
  function availWidth() {
    var vw = windowWidth();
    if (!(vw > 0) || vw >= NARROW) return 0;
    return holderWidth || vw - PAD;
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
        // The box starts 6 left of its text: keep all of it inside the chart (narrow on a phone).
        g.setAttribute("transform", "translate(" + Math.max(8, Math.min(tx0, W - bw + 4)) + "," + y + ")"); g.setAttribute("visibility", "visible");
      },
      hide: function () { g.setAttribute("visibility", "hidden"); }
    };
  }

  // series: [{id, label, cls, dash, points: [{x, y, ...}]}] (y null = a gap in the line);
  // opts: title, desc, width, height, xMin/xMax/yMin/yMax, xTicks/yTicks, xFmt/yFmt (tick
  // text), xLabel, yLabel, marks [{x, cls, label}], dots [{x, y, cls, label, lines}], diag (chance
  // line), hover "x" (nearest x, every series' value) | "xy" (nearest point), tip(hit) -> lines,
  // hidden {id: true}, pointDots (draw each point), xTickText(v) for categorical ticks.
  function lineChart(series, o) {
    o = o || {};
    var size = fitSize(o.width || 560, o.height || 290, availWidth());
    var W = size[0], H = size[1], L = o.left || 52, R = 16, T = o.top || 18, B = 44, pw = W - L - R, ph = H - T - B;
    var x0 = has(o.xMin) ? o.xMin : 0, x1 = has(o.xMax) ? o.xMax : 1, y0 = has(o.yMin) ? o.yMin : 0, y1 = has(o.yMax) ? o.yMax : 1;
    var X = function (v) { return L + (v - x0) / ((x1 - x0) || 1) * pw; }, Y = function (v) { return T + (1 - (v - y0) / ((y1 - y0) || 1)) * ph; };
    var id = "pb-chart-" + (++chartId), hidden = o.hidden || {};
    var svg = sv("svg", {viewBox: "0 0 " + W + " " + H, class: "thr-chart " + (o.cls || ""), role: "img", "aria-labelledby": id + "-t " + id + "-d"});
    svg.appendChild(sv("title", {id: id + "-t"}, o.title || "Chart"));
    svg.appendChild(sv("desc", {id: id + "-d"}, o.desc || ""));
    var yf = o.yFmt || pct0, xf = o.xFmt || function (v) { return v.toFixed(1); };
    (o.yTicks || niceTicks(y0, y1, 5)).forEach(function (v) {
      svg.appendChild(sv("line", {x1: L, x2: L + pw, y1: Y(v), y2: Y(v), class: "thr-grid"}));
      svg.appendChild(sv("text", {x: L - 6, y: Y(v) + 4, "text-anchor": "end", class: "thr-tick"}, yf(v)));
    });
    (o.xTicks || niceTicks(x0, x1, 5)).forEach(function (v) {
      if (o.xGrid) svg.appendChild(sv("line", {x1: X(v), x2: X(v), y1: T, y2: T + ph, class: "thr-grid"}));
      svg.appendChild(sv("line", {x1: X(v), x2: X(v), y1: T + ph, y2: T + ph + 4, class: "thr-axis"}));
      svg.appendChild(sv("text", {x: X(v), y: T + ph + 16, "text-anchor": "middle", class: "thr-tick"}, o.xTickText ? o.xTickText(v) : xf(v)));
    });
    svg.appendChild(sv("line", {x1: L, x2: L + pw, y1: T + ph, y2: T + ph, class: "thr-axis"}));
    svg.appendChild(sv("line", {x1: L, x2: L, y1: T, y2: T + ph, class: "thr-axis"}));
    if (o.diag) svg.appendChild(sv("line", {x1: X(x0), y1: Y(y0), x2: X(x1), y2: Y(y1), class: "roc-diag"}));
    if (o.xLabel) svg.appendChild(sv("text", {x: L + pw / 2, y: H - 6, "text-anchor": "middle", class: "thr-label"}, o.xLabel));
    if (o.yLabel) svg.appendChild(sv("text", {x: 12, y: T + ph / 2, "text-anchor": "middle", class: "thr-label", transform: "rotate(-90 12 " + (T + ph / 2) + ")"}, o.yLabel));
    var shown = series.filter(function (s) { return !hidden[s.id]; });
    shown.forEach(function (s, i) {
      var d = "", pen = false, dash = s.dash !== undefined ? s.dash : null;
      s.points.forEach(function (p) {
        if (!has(p.y) || !has(p.x)) { pen = false; return; }
        d += (pen ? "L" : "M") + X(p.x).toFixed(1) + " " + Y(p.y).toFixed(1); pen = true;
      });
      if (d) svg.appendChild(sv("path", {d: d, class: "thr-line " + (s.cls || "ser ser-" + ((i % 8) + 1)), "stroke-dasharray": dash}));
      if (o.pointDots) s.points.forEach(function (p) { if (has(p.y) && has(p.x)) svg.appendChild(sv("circle", {cx: X(p.x), cy: Y(p.y), r: 3, class: "pb-dot " + (s.cls || "")})); });
    });
    (o.marks || []).forEach(function (m, k) {
      if (!has(m.x)) return;
      svg.appendChild(sv("line", {x1: X(m.x), x2: X(m.x), y1: T, y2: T + ph, class: m.cls}));
      svg.appendChild(sv("text", {x: Math.min(X(m.x) + 4, L + pw - 96), y: T + 12 + 14 * k, class: "thr-tick"}, m.label));
    });
    if (o.corner) svg.appendChild(sv("text", {x: L + pw - 4, y: T + ph - 8, "text-anchor": "end", class: "thr-label"}, o.corner));
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
    var guide = sv("line", {y1: T, y2: T + ph, class: "thr-guide", visibility: "hidden"}), hov = sv("circle", {r: 3.5, class: "roc-hover", visibility: "hidden"});
    svg.appendChild(guide); svg.appendChild(hov);
    var tipBox = svgTip(svg);
    function nearest(evt) {
      var r = svg.getBoundingClientRect(); if (!r.width) return null;
      var mx = (evt.clientX - r.left) * W / r.width, my = (evt.clientY - r.top) * H / r.height, best = null;
      shown.forEach(function (s) {
        s.points.forEach(function (p) {
          if (!has(p.x) || !has(p.y)) return;
          var dist = o.hover === "xy" ? Math.pow(X(p.x) - mx, 2) + Math.pow(Y(p.y) - my, 2) : Math.abs(X(p.x) - mx);
          if (!best || dist < best.dist) best = {dist: dist, p: p, s: s};
        });
      });
      return best;
    }
    function at(s, x) { var b = null; s.points.forEach(function (p) { if (has(p.y) && (!b || Math.abs(p.x - x) < Math.abs(b.x - x))) b = p; }); return b; }
    // The readout at the pointer of `evt`; false when there is nothing to show.
    function readout(evt) {
      var h = nearest(evt); if (!h) return false;
      var lines;
      if (o.tip) lines = o.tip(h.p, h.s);
      else if (o.hover === "xy") lines = [h.s.label, (o.xName || "x") + " " + (o.xHover || xf)(h.p.x) + ", " + (o.yName || "y") + " " + (o.yHover || yf)(h.p.y)];
      else {
        lines = [(o.xName || "") + " " + (o.xHover || xf)(h.p.x)];
        shown.slice(0, 8).forEach(function (s) { var q = at(s, h.p.x); if (q && Math.abs(q.x - h.p.x) <= (x1 - x0) * 0.03) lines.push(s.label + ": " + (o.yHover || yf)(q.y)); });
      }
      if (!lines || !lines.length) return false;
      if (o.hover === "xy") { hov.setAttribute("cx", X(h.p.x)); hov.setAttribute("cy", Y(h.p.y)); hov.setAttribute("visibility", "visible"); guide.setAttribute("visibility", "hidden"); }
      else { guide.setAttribute("x1", X(h.p.x)); guide.setAttribute("x2", X(h.p.x)); guide.setAttribute("visibility", "visible"); }
      var ty = o.hover === "xy" ? Math.min(Y(h.p.y) + 18, T + ph - lines.length * 15) : T + 30;
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
      if (evt.pointerType === "mouse") return;
      touch = null;
      if (readout(evt)) pin(); else clear();
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

  // Speed vs accuracy. points: [{key, x, y, color (1..8), shape "circle"|"square"|"triangle",
  // label, best}], opts: title, desc, xLabel, yLabel, xFmt, yFmt, frontier (join the best
  // points), selected (key), onPick(key). Each point is a focusable button (click / Enter).
  function scatter(points, o) {
    o = o || {};
    var size = fitSize(o.width || 620, o.height || 340, availWidth()), ring = coarse() ? 22 : 11;
    var W = size[0], H = size[1], L = 52, R = 18, T = 16, B = 46, pw = W - L - R, ph = H - T - B;
    var pts = points.filter(function (p) { return has(p.x) && has(p.y); });
    var xmax = niceMax(Math.max.apply(null, pts.map(function (p) { return p.x; }).concat([1])) * 1.08);
    var ys = pts.map(function (p) { return p.y; }), ymin = ys.length ? Math.min.apply(null, ys) : 0, ymax = ys.length ? Math.max.apply(null, ys) : 1;
    var y0 = Math.max(0, Math.floor((ymin - 0.04) * 10) / 10), y1 = Math.min(1, Math.ceil((ymax + 0.03) * 10) / 10); if (y1 - y0 < 0.1) { y0 = Math.max(0, y1 - 0.2); }
    var X = function (v) { return L + v / xmax * pw; }, Y = function (v) { return T + (1 - (v - y0) / (y1 - y0)) * ph; };
    var id = "pb-scatter-" + (++chartId);
    var svg = sv("svg", {viewBox: "0 0 " + W + " " + H, class: "thr-chart pb-scatter", role: "group", "aria-labelledby": id + "-t " + id + "-d"});
    svg.appendChild(sv("title", {id: id + "-t"}, o.title || "Speed vs accuracy"));
    svg.appendChild(sv("desc", {id: id + "-d"}, o.desc || ""));
    var yf = o.yFmt || pct0, xf = o.xFmt || function (v) { return v.toFixed(0); };
    niceTicks(y0, y1, 5).forEach(function (v) {
      svg.appendChild(sv("line", {x1: L, x2: L + pw, y1: Y(v), y2: Y(v), class: "thr-grid"}));
      svg.appendChild(sv("text", {x: L - 6, y: Y(v) + 4, "text-anchor": "end", class: "thr-tick"}, yf(v)));
    });
    niceTicks(0, xmax, 6).forEach(function (v) {
      svg.appendChild(sv("line", {x1: X(v), x2: X(v), y1: T, y2: T + ph, class: "thr-grid"}));
      svg.appendChild(sv("text", {x: X(v), y: T + ph + 16, "text-anchor": "middle", class: "thr-tick"}, xf(v)));
    });
    svg.appendChild(sv("line", {x1: L, x2: L + pw, y1: T + ph, y2: T + ph, class: "thr-axis"}));
    svg.appendChild(sv("line", {x1: L, x2: L, y1: T, y2: T + ph, class: "thr-axis"}));
    svg.appendChild(sv("text", {x: L + pw / 2, y: H - 6, "text-anchor": "middle", class: "thr-label"}, o.xLabel || "Time per image (ms, lower is better)"));
    svg.appendChild(sv("text", {x: 12, y: T + ph / 2, "text-anchor": "middle", class: "thr-label", transform: "rotate(-90 12 " + (T + ph / 2) + ")"}, o.yLabel || "Accuracy (%)"));
    svg.appendChild(sv("text", {x: L + 6, y: T + 12, class: "thr-tick"}, "↖ better"));
    if (o.frontier) {
      var fr = pts.filter(function (p) { return p.best; }).sort(function (a, b) { return a.x - b.x; });
      if (fr.length > 1) svg.appendChild(sv("path", {d: fr.map(function (p, i) { return (i ? "L" : "M") + X(p.x).toFixed(1) + " " + Y(p.y).toFixed(1); }).join(""), class: "pb-frontier"}));
    }
    var label = sv("text", {class: "pb-pt-label", visibility: "hidden"}); var labelBg = sv("rect", {class: "pb-pt-label-bg", rx: 3, visibility: "hidden"});
    function showLabel(p) {
      var text = p.label + ": " + (o.xHover || fmt.ms)(p.x) + ", " + (o.yHover || function (v) { return fmt.pct(v, 1); })(p.y);
      var w = text.length * 6.3 + 10, x = X(p.x) + w + 12 > L + pw ? X(p.x) - w - 8 : X(p.x) + 10, y = Math.max(T + 12, Y(p.y) - 10);
      x = Math.max(2, Math.min(x, W - w - 2));
      label.textContent = text; label.setAttribute("x", x + 5); label.setAttribute("y", y + 4); label.setAttribute("visibility", "visible");
      labelBg.setAttribute("x", x); labelBg.setAttribute("y", y - 9); labelBg.setAttribute("width", w); labelBg.setAttribute("height", 18); labelBg.setAttribute("visibility", "visible");
    }
    pts.forEach(function (p) {
      var g = sv("g", {class: "pb-pt ser-" + (p.color || 1) + (p.key === o.selected ? " selected" : "") + (p.best ? " best" : ""), "data-key": p.key, tabindex: 0, role: "button", "aria-label": p.label + ": " + fmt.ms(p.x) + ", " + fmt.pct(p.y, 1) + (p.best ? ", best trade-off" : "")});
      var cx = X(p.x), cy = Y(p.y), shape;
      if (p.shape === "square") shape = sv("rect", {x: cx - 5, y: cy - 5, width: 10, height: 10});
      else if (p.shape === "triangle") shape = sv("path", {d: "M" + cx + " " + (cy - 6) + "L" + (cx + 6) + " " + (cy + 5) + "L" + (cx - 6) + " " + (cy + 5) + "Z"});
      else shape = sv("circle", {cx: cx, cy: cy, r: 5.5});
      g.appendChild(sv("circle", {cx: cx, cy: cy, r: ring, class: "pb-pt-ring"}));
      g.appendChild(shape); g.appendChild(sv("title", {}, p.label + ": " + fmt.ms(p.x) + ", " + fmt.pct(p.y, 1) + (p.best ? " (best trade-off)" : "")));
      var pick = function () { showLabel(p); if (o.onPick) o.onPick(p.key); };
      g.addEventListener("click", pick);
      g.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      svg.appendChild(g);
      if (p.key === o.selected) setTimeout(function () { showLabel(p); }, 0);
    });
    svg.appendChild(labelBg); svg.appendChild(label);
    return svg;
  }

  // Hidden series per chart key: survives re-renders.
  var hiddenBy = {};
  // A chart with a legend (series toggles) and "Show as table". o: key, series (legend items
  // {id, label, cls, dash, color}), draw(hidden) -> svg, table() -> {head: [...], rows: [[...]]},
  // legend (default true when more than one series), note (shown under the legend), cls.
  function block(o) {
    var box = el("div", "chart-block " + (o.cls || "")), hidden = hiddenBy[o.key] = hiddenBy[o.key] || {};
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
        if (s.shape) sw.appendChild(s.shape === "square" ? sv("rect", {x: 7, y: 1, width: 8, height: 8, class: "pb-pt-swatch " + (s.cls || "")}) : sv("circle", {cx: 11, cy: 5, r: 4, class: "pb-pt-swatch " + (s.cls || "")}));
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
      });
      box.appendChild(d);
    }
    return box;
  }

  // Confidence threshold: precision / recall / F1 (and F2) vs threshold. o: title, desc,
  // points (ascending thresholds), configured (now), best, running (the loaded model's
  // threshold when it differs), f2, key.
  function chartSvg(o) {
    var defs = [["precision", "thr-p", "Correct (precision)", null], ["recall", "thr-r", "Finds (recall)", "6 3"], ["f1", "thr-f1", "Balance (F1)", null]];
    if (o.f2) defs.push(["f2", "thr-f2", "Miss-fewer (F2)", "2 3"]);
    var series = defs.map(function (d) { return {id: d[0], label: d[2], cls: d[1], dash: d[3], points: (o.points || []).map(function (p) { return {x: p.threshold, y: p[d[0]], raw: p}; })}; });
    var marks = [{x: o.configured, cls: "thr-mark-cfg", label: "now " + fmt.thr(o.configured)}, {x: o.best, cls: "thr-mark-best", label: "best " + fmt.thr(o.best)}];
    if (has(o.running) && Math.abs(o.running - o.configured) > 1e-6) marks.push({x: o.running, cls: "thr-mark-cfg", label: "running " + fmt.thr(o.running)});
    return block({
      key: o.key || ("thr:" + o.title), series: series,
      draw: function (hidden) {
        return lineChart(series, {
          title: o.title, desc: o.desc, hidden: hidden, marks: marks, xLabel: "Confidence threshold (\u00d7 100 = Blue Iris %)", yLabel: "%",
          xTicks: [0, 0.2, 0.4, 0.6, 0.8, 1], xFmt: function (v) { return v.toFixed(1); }, yTicks: [0, 0.2, 0.4, 0.6, 0.8, 1],
          tip: function (p) { var r = p.raw; return ["confidence " + r.threshold.toFixed(3) + " (" + Math.round(r.threshold * 100) + "%)", "correct " + pct1(r.precision) + "%  finds " + pct1(r.recall) + "%", "F1 " + pct1(r.f1) + "%  F2 " + pct1(r.f2) + "%", "TP " + r.tp + "  FP " + r.fp + "  FN " + r["fn"]]; }
        });
      },
      table: function () { return {head: ["Confidence", "Correct (precision)", "Finds (recall)", "F1", "F2", "TP", "FP", "FN"], rows: (o.points || []).map(function (p) { return [p.threshold.toFixed(3), pct1(p.precision) + "%", pct1(p.recall) + "%", pct1(p.f1) + "%", pct1(p.f2) + "%", String(p.tp), String(p.fp), String(p["fn"])]; })}; }
    });
  }
  // Frame-level ROC: false-alert rate (x) vs alert rate (y), the chance diagonal, markers at
  // the configured and the best threshold. o: title, desc, curve [{t, tpr, fpr}], auc,
  // configured / best (threshold points with .frame), key.
  function rocSvg(o) {
    var pts = [{t: null, tpr: 0, fpr: 0}].concat(o.curve || []);
    var series = [{id: "roc", label: "ROC", cls: "roc-line", points: pts.map(function (p) { return {x: p.fpr, y: p.tpr, raw: p}; })}];
    var dot = function (p, cls, name) {
      var f = p && p.frame; if (!(f && has(f.tpr) && has(f.fpr))) return null;
      var head = name + " " + fmt.thr(p.threshold), rates = "alert rate " + pct1(f.tpr) + "%, false alerts " + pct1(f.fpr) + "%";
      return {x: f.fpr, y: f.tpr, cls: cls, label: head + ": " + rates, lines: [head, rates]};
    };
    var dots = [dot(o.configured, "roc-dot-cfg", "now"), dot(o.best, "roc-dot-best", "best")].filter(Boolean);
    return block({
      key: o.key || ("roc:" + o.title), series: series, legend: false, cls: "roc-block",
      note: "● best · ○ now · dashed: chance",
      draw: function () {
        return lineChart(series, {
          title: o.title, desc: o.desc, width: 330, height: 300, left: 44, hover: "xy", diag: true, dots: dots, xGrid: true,
          xLabel: "False alerts on empty frames (%)", yLabel: "Alerts on frames with objects (%)", corner: "AUC " + pct1(o.auc) + "%",
          xTicks: [0, 0.2, 0.4, 0.6, 0.8, 1], yTicks: [0, 0.2, 0.4, 0.6, 0.8, 1], xFmt: pct0,
          tip: function (p) { var r = p.raw; return [r.t > 0 ? "confidence ≥ " + r.t.toFixed(3) : "every frame (no detection ≥ 0.05)", "alerts " + pct1(r.tpr) + "%  false " + pct1(r.fpr) + "%"]; }
        });
      },
      table: function () { return {head: ["Confidence", "Alert rate (TPR)", "False-alert rate (FPR)"], rows: (o.curve || []).map(function (p) { return [p.t > 0 ? p.t.toFixed(3) : "-", pct1(p.tpr) + "%", pct1(p.fpr) + "%"]; })}; }
    });
  }

  PB.charts = {sv: sv, lineChart: lineChart, scatter: scatter, block: block, chartSvg: chartSvg, rocSvg: rocSvg, dashOf: dashOf, niceTicks: niceTicks, niceMax: niceMax, fitSize: fitSize, unpin: unpin, hidden: hiddenBy};
})();
