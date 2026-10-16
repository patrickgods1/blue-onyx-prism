// Blue Onyx Prism: Benchmark page, Compare tab (PrismBench.compare).
// One filter bar (models, devices, runtimes, datasets, which runs; remembered per browser)
// drives "Side by side" (speed vs accuracy scatter, comparison table, overlaid quality-vs-
// confidence and ROC curves) and "Across runs" (run history of one model: A/B comparison with
// the server's verdicts, pin / delete, trend lines). The history is fetched only when the tab
// opens. benchmark.js calls render(status, models) on every poll; each section re-renders only
// when its data or the filters changed (PrismLive.keep keeps scroll, open details, sort).
(function () {
  var PB = window.PrismBench, L = window.PrismLive, el = PB.el, has = PB.has, fmt = PB.fmt, C = PB.charts;
  var $ = function (id) { return document.getElementById(id); };
  var MAX_SERIES = 8;
  var saved = PB.store.get("compare", null) || {};
  var F = {
    off: saved.off || {model: {}, device: {}, runtime: {}, dataset: {}},
    onlyRec: !!saved.onlyRec, runs: saved.runs || "latest", picked: saved.picked || [],
    y: saved.y || "score", curve: saved.curve || "f1"
  };
  ["model", "device", "runtime", "dataset"].forEach(function (k) { if (!F.off[k]) F.off[k] = {}; });
  function remember() { PB.store.set("compare", F); }
  var S = {j: null, models: [], selected: null, hist: {}, histErr: {}, histAll: null, histAllErr: null, cmp: {}, runModel: saved.runModel || null, a: null, b: null, trend: saved.trend || "p50", trendOff: {}, visible: false};

  // ---- rows: one per model x device ----------------------------------------------------------
  function serverFlags(j) {
    var flags = {};
    (j && (j.points || []) || []).forEach(function (r) { if (r && r.model && r.device && r.best_tradeoff !== undefined) flags[r.model.toLowerCase() + "|" + r.device] = !!r.best_tradeoff; });
    return flags;
  }
  function rowsFromModels(models, j) {
    var out = [];
    models.forEach(function (m) {
      if (F.runs === "this" && m.this_run === false) return;
      (m.devices || []).forEach(function (d) {
        var r = d.run; if (!r) return;
        var acc = r.accuracy, ov = acc && acc.overall, st = r.stages_ms || {};
        var sweep = acc && acc.sweep, adv = m.threshold && m.threshold.device === d.device ? m.threshold : null;
        out.push({
          key: m.model.toLowerCase() + "|" + d.device, model: m.model, device: d.device, devLabel: d.label, runtime: PB.runtimeOf(d.device, d.runtime),
          label: m.model + " · " + d.device, p50: st.total ? st.total.p50 : null, p95: st.total ? st.total.p95 : null, infer: st.infer ? st.infer.p50 : null,
          fps: r.throughput_fps, ap50: ov ? ov.ap50 : null, ap50_95: ov ? ov.ap50_95 : null, roc_auc: acc && acc.frame_roc ? acc.frame_roc.auc : null,
          precision: ov ? ov.precision : null, recall: ov ? ov.recall : null, f1: ov ? ov.f1 : null,
          score: r.grades ? r.grades.accuracy_score : null, grade: r.grades ? r.grades.overall : null, relative: !!(acc && acc.relative),
          trust: d.trust, recommended: m.recommended === d.device, fellBack: !!r.fell_back, datasets: m.datasets || [], timestamp: m.timestamp, thisRun: m.this_run,
          curve: sweep && sweep.overall ? sweep.overall.points : adv ? adv.curve : null,
          roc: sweep && sweep.roc && sweep.roc.curve && sweep.roc.curve.length ? sweep.roc.curve : adv && adv.roc ? adv.roc.curve : null,
          threshold: acc ? acc.threshold : null
        });
      });
    });
    return out;
  }
  function rowsFromHistory() {
    var out = [];
    if (!S.histAll) return out;
    Object.keys(S.histAll).forEach(function (name) {
      (S.histAll[name] || []).forEach(function (e) {
        if (F.picked.indexOf(name.toLowerCase() + "|" + e.run_id) < 0) return;
        (e.devices || []).forEach(function (dv) {
          if (dv.error) return;
          out.push({
            key: name.toLowerCase() + "|" + dv.device + "|" + e.run_id, model: e.model || name, device: dv.device, devLabel: dv.label, runtime: PB.runtimeOf(dv.device, dv.runtime),
            label: (e.model || name) + " · " + dv.device + " · " + fmt.day(e.timestamp), p50: dv.total_p50_ms, p95: dv.total_p95_ms, infer: dv.infer_p50_ms, fps: dv.fps,
            ap50: dv.ap50, ap50_95: dv.ap50_95, roc_auc: dv.roc_auc, precision: dv.precision, recall: dv.recall, f1: dv.f1, score: null, grade: dv.grade,
            trust: null, recommended: !!dv.recommended, fellBack: !!dv.fell_back, datasets: (e.settings && e.settings.datasets) || [], timestamp: e.timestamp, curve: null, roc: null
          });
        });
      });
    });
    return out;
  }
  function allRows() { return F.runs === "pick" ? rowsFromHistory() : rowsFromModels(S.models, S.j); }
  function values(rows, k) {
    var seen = {}, out = [];
    rows.forEach(function (r) { (k === "dataset" ? r.datasets : [r[k]]).forEach(function (v) { if (v !== undefined && v !== null && !seen[v]) { seen[v] = 1; out.push(v); } }); });
    return out.sort();
  }
  function passes(r) {
    if (F.off.model[r.model]) return false; if (F.off.device[r.device]) return false; if (F.off.runtime[r.runtime]) return false;
    if (r.datasets.length && r.datasets.every(function (d) { return F.off.dataset[d]; })) return false;
    if (F.onlyRec && !r.recommended) return false;
    return true;
  }
  // Rows no other shown row beats on both speed and accuracy (`y`).
  function pareto(rows, y) {
    return rows.map(function (r) {
      if (!has(r.p50) || !has(r[y]) || r.fellBack) return false;
      return !rows.some(function (o) { return o !== r && has(o.p50) && has(o[y]) && !o.fellBack && o.p50 <= r.p50 && o[y] >= r[y] && (o.p50 < r.p50 || o[y] > r[y]); });
    });
  }

  // ---- filter bar ------------------------------------------------------------------------------
  function checkGroup(title, kind, vals, labelOf) {
    var fs = el("fieldset", "cmp-group"); fs.appendChild(el("legend", null, title));
    if (!vals.length) fs.appendChild(el("p", "muted", "none"));
    vals.forEach(function (v) {
      var l = el("label", "check"), i = el("input"); i.type = "checkbox"; i.value = v; i.checked = !F.off[kind][v];
      i.addEventListener("change", function () { if (i.checked) delete F.off[kind][v]; else F.off[kind][v] = true; remember(); draw(true); });
      l.appendChild(i);
      if (kind === "model") { var sw = el("span", "swatch ser-" + PB.colorOf(v)); sw.setAttribute("aria-hidden", "true"); l.appendChild(sw); }
      l.appendChild(document.createTextNode(" " + (labelOf ? labelOf(v) : v))); fs.appendChild(l);
    });
    return fs;
  }
  function buildFilters(rows) {
    var box = $("cmp-filters"); box.replaceChildren();
    var form = el("form", "cmp-filters card"); form.addEventListener("submit", function (e) { e.preventDefault(); });
    var top = el("div", "cmp-quick");
    var runs = el("fieldset", "cmp-runs-mode"); runs.appendChild(el("legend", null, "Runs"));
    [["latest", "Latest of each model"], ["this", "This run only"], ["pick", "Pick runs…"]].forEach(function (o) {
      var l = el("label", "check"), i = el("input"); i.type = "radio"; i.name = "cmp-runs"; i.value = o[0]; i.checked = F.runs === o[0];
      i.addEventListener("change", function () { if (!i.checked) return; F.runs = o[0]; remember(); if (F.runs === "pick") loadAllHistory(); draw(true); });
      l.appendChild(i); l.appendChild(document.createTextNode(" " + o[1])); runs.appendChild(l);
    });
    top.appendChild(runs);
    var qb = el("div", "actions");
    [["All", function () { F.off = {model: {}, device: {}, runtime: {}, dataset: {}}; F.onlyRec = false; }],
     ["None", function () { F.onlyRec = false; values(rows, "model").forEach(function (v) { F.off.model[v] = true; }); }],
     ["Only recommended", function () { F.off = {model: {}, device: {}, runtime: {}, dataset: {}}; F.onlyRec = true; }]].forEach(function (b) {
      var x = el("button", "small secondary" + (b[0] === "Only recommended" && F.onlyRec ? " active" : ""), b[0]); x.type = "button";
      if (b[0] === "Only recommended") x.setAttribute("aria-pressed", F.onlyRec ? "true" : "false");
      x.addEventListener("click", function () { b[1](); remember(); draw(true); }); qb.appendChild(x);
    });
    top.appendChild(qb);
    form.appendChild(top);
    if (F.runs === "pick") form.appendChild(pickRuns());
    var grid = el("div", "cmp-grid");
    grid.appendChild(checkGroup("Models", "model", values(rows, "model")));
    grid.appendChild(checkGroup("Devices", "device", values(rows, "device")));
    grid.appendChild(checkGroup("Runtimes", "runtime", values(rows, "runtime"), function (v) { return (PB.RUNTIME_NAMES[v] || v) + " (" + (v === "onnxruntime" ? "ort:*" : "openvino:*") + ")"; }));
    grid.appendChild(checkGroup("Datasets", "dataset", values(rows, "dataset")));
    form.appendChild(grid);
    box.appendChild(form);
  }
  function pickRuns() {
    var fs = el("fieldset", "cmp-pick"); fs.appendChild(el("legend", null, "Runs to show"));
    if (S.histAllErr) { fs.appendChild(el("p", "warn", S.histAllErr)); return fs; }
    if (!S.histAll) { fs.appendChild(el("p", "muted", "loading the run history…")); return fs; }
    var names = Object.keys(S.histAll).sort(); if (!names.length) fs.appendChild(el("p", "muted", "No runs in the history yet."));
    names.forEach(function (name) {
      var first = (S.histAll[name] || [])[0], row = el("div", "pick-model"); row.appendChild(el("strong", null, (first && first.model || name) + ": "));
      (S.histAll[name] || []).forEach(function (e) {
        var k = name.toLowerCase() + "|" + e.run_id, l = el("label", "check inline"), i = el("input"); i.type = "checkbox"; i.checked = F.picked.indexOf(k) >= 0;
        i.addEventListener("change", function () { var at = F.picked.indexOf(k); if (i.checked && at < 0) F.picked.push(k); if (!i.checked && at >= 0) F.picked.splice(at, 1); remember(); draw(true); });
        l.appendChild(i); l.appendChild(document.createTextNode(" " + fmt.day(e.timestamp) + (e.pinned ? " ★" : ""))); row.appendChild(l);
      });
      fs.appendChild(row);
    });
    return fs;
  }

  // ---- side by side ---------------------------------------------------------------------------
  var Y_METRICS = [["score", "Accuracy score (ranking)", "accuracy_score"], ["ap50", "Detection quality (AP50)", "ap50"], ["ap50_95", "Box accuracy (AP50-95)", "ap50_95"], ["roc_auc", "Alert separation (ROC AUC)", "roc_auc"], ["recall", "Finds objects (recall)", "recall"], ["precision", "Correct detections (precision)", "precision"], ["f1", "Balance (F1)", "f1"]];
  var CURVE_METRICS = [["f1", "Balance (F1)"], ["f2", "Miss-fewer (F2)"], ["precision", "Correct detections (precision)"], ["recall", "Finds objects (recall)"]];
  function picker(labelText, opts, cur, onChange) {
    var l = el("label", "inline-label"); l.appendChild(document.createTextNode(labelText + " "));
    var s = el("select"); opts.forEach(function (o) { var x = el("option", null, o[1]); x.value = o[0]; if (o[0] === cur) x.selected = true; s.appendChild(x); });
    s.addEventListener("change", function () { onChange(s.value); }); l.appendChild(s); return l;
  }
  function dashWithin(rows) {
    var perModel = {}, out = {};
    rows.forEach(function (r) { var k = r.model.toLowerCase(); perModel[k] = (perModel[k] || 0); out[r.key] = C.dashOf(perModel[k]); perModel[k]++; });
    return out;
  }
  function shapeOf(rt) { return rt === "onnxruntime" ? "square" : rt === "openvino" ? "circle" : "triangle"; }
  function buildSide(rows) {
    var box = $("cmp-side"); box.replaceChildren();
    box.appendChild(el("h2", null, "Side by side"));
    var y = F.y; if (y === "score" && !rows.some(function (r) { return has(r.score); })) y = "ap50";
    var yDef = Y_METRICS.find(function (m) { return m[0] === y; }) || Y_METRICS[1];
    var tools = el("div", "chart-tools");
    tools.appendChild(picker("Accuracy axis", Y_METRICS.map(function (m) { return [m[0], m[1]]; }), y, function (v) { F.y = v; remember(); draw(true); }));
    tools.appendChild(PB.term("What is this?", yDef[2], "cmp"));
    box.appendChild(tools);
    // Best trade-offs: the server's (over every point, by the ranking's accuracy score) when the
    // view shows all latest results on that score; otherwise computed over the shown rows.
    var flags = serverFlags(S.j), useServer = y === "score" && F.runs === "latest" && rows.length === allRows().length && rows.every(function (r) { return flags[r.key] !== undefined; });
    var front = useServer ? rows.map(function (r) { return flags[r.key]; }) : pareto(rows, y);
    rows.forEach(function (r, i) { r.front = front[i]; });
    var drawn = rows.filter(function (r) { return has(r.p50) && has(r[y]) && !r.fellBack; });
    var sc = el("div", "scatter-wrap");
    sc.appendChild(C.block({
      key: "cmp-scatter", legend: false,
      note: "Color = model, shape = runtime (● OpenVINO, ■ ONNX Runtime). The line joins the best trade-offs: no other shown result is both faster and more accurate. Click a point to find its row.",
      draw: function () {
        return C.scatter(drawn.map(function (r) { return {key: r.key, x: r.p50, y: r[y], color: PB.colorOf(r.model), shape: shapeOf(r.runtime), label: r.label, best: r.front}; }), {
          title: "Speed vs accuracy", desc: drawn.length + " results: time per image (lower is better) against " + yDef[1] + " (higher is better). Best trade-offs: " + drawn.filter(function (r) { return r.front; }).map(function (r) { return r.label; }).join(", ") + ".",
          yLabel: yDef[1] + " (%)", frontier: true, selected: S.selected,
          onPick: function (k) { S.selected = k; highlight(); }
        });
      },
      table: function () { return {head: ["Result", "Time per image", yDef[1], "Best trade-off"], rows: drawn.map(function (r) { return [r.label, fmt.ms(r.p50), fmt.pct(r[y], 1), r.front ? "yes" : ""]; })}; }
    }));
    box.appendChild(sc);
    var missing = rows.length - drawn.length;
    if (missing > 0) box.appendChild(el("p", "muted", missing + " result" + (missing === 1 ? " is" : "s are") + " not drawn: no " + yDef[1].toLowerCase() + " or a device that fell back (they are in the table)."));
    box.appendChild(compareTable(rows, y));
    curves(box, rows);
  }
  // The comparison table: best value per column in bold, the gap to it on the others.
  var COLS = [
    ["p50", "p50", "lower", "ms"], ["p95", "p95", "lower", "ms"], ["infer", "infer_p50", "lower", "ms"], ["fps", "fps", "higher", "fps"],
    ["score", "accuracy_score", "higher", "pct"], ["ap50", "ap50", "higher", "pct"], ["ap50_95", "ap50_95", "higher", "pct"], ["roc_auc", "roc_auc", "higher", "pct"],
    ["recall", "recall", "higher", "pct"], ["precision", "precision", "higher", "pct"]
  ];
  function compareTable(rows, y) {
    var cols = COLS.filter(function (c) { return rows.some(function (r) { return has(r[c[0]]); }); });
    var best = {};
    cols.forEach(function (c) {
      var vs = rows.filter(function (r) { return !r.fellBack; }).map(function (r) { return r[c[0]]; }).filter(has);
      best[c[0]] = vs.length ? (c[2] === "lower" ? Math.min.apply(null, vs) : Math.max.apply(null, vs)) : null;
    });
    var columns = [
      {id: "model", head: "@model", value: function (r) { return r.model.toLowerCase(); }, render: function (r) {
        var s = el("span"), sw = el("span", "swatch ser-" + PB.colorOf(r.model)); sw.setAttribute("aria-hidden", "true"); s.appendChild(sw); s.appendChild(document.createTextNode(" " + r.model + (r.relative ? " *" : "")));
        if (r.recommended) { var t0 = el("span", "tag rec", "best"); t0.title = "recommended device for this model"; s.appendChild(t0); } s.className = "nowrap"; return s; }},
      {id: "device", head: "@device", value: function (r) { return r.device; }, render: function (r, td) { td.title = r.devLabel || ""; var s = el("span", "nowrap", r.device); if (r.fellBack) s.appendChild(el("span", "tag restart", "fell back")); return s; }},
      {id: "runtime", head: "@runtime", value: function (r) { return r.runtime; }, render: function (r) { return PB.RUNTIME_NAMES[r.runtime] || r.runtime; }}
    ].concat(cols.map(function (c) {
      return {id: c[0], head: "@" + c[1], num: true, first: c[2] === "lower" ? "asc" : "desc", value: function (r) { return r[c[0]]; }, render: function (r, td) {
        var v = r[c[0]], b = best[c[0]], s = el("span", null, c[3] === "ms" ? fmt.ms(v) : c[3] === "fps" ? (has(v) ? v.toFixed(1) + "/s" : "-") : fmt.pct(v, 1));
        if (has(v) && has(b) && !r.fellBack) {
          if (Math.abs(v - b) < 1e-9) { td.classList.add("best"); td.title = "best in this column"; }
          else s.appendChild(el("span", "gap", c[3] === "pct" ? fmt.points(v - b) : c[3] === "ms" ? "+" + (v - b).toFixed(1) + " ms" : "−" + (b - v).toFixed(1) + "/s"));
        }
        return s; }};
    })).concat([
      {id: "grade", head: "@grade", value: function (r) { return r.grade || null; }, render: function (r) { return el("span", "grade grade-" + (r.grade || "none"), r.grade || "-"); }},
      {id: "trust", head: "@trust", value: function (r) { return r.trust ? {high: 0, medium: 1, low: 2}[r.trust.level] : null; }, render: function (r) { return PB.trustBadge(r.trust, "cmp"); }},
      {id: "front", head: "@tradeoff", value: function (r) { return r.front ? 0 : 1; }, render: function (r) { return r.front ? PB.tip("★ best trade-off", PB.tipText("best_tradeoff"), "tag best-tradeoff", "cmp") : ""; }}
    ]);
    var d = el("div"); d.appendChild(el("h3", null, "Comparison table")); d.appendChild(el("p", "muted", "Bold = best in its column; the small number is the gap to the best. Accuracy at the run's confidence threshold."));
    d.appendChild(PB.dataTable({
      key: "compare", cls: "stack cmp-table", rows: rows, columns: columns,
      filters: {model: function (r) { return r.model; }, device: function (r) { return r.device; }, runtime: function (r) { return PB.RUNTIME_NAMES[r.runtime] || r.runtime; }, thisRun: function (r) { return r.thisRun; }},
      text: function (r) { return r.model + " " + r.device + " " + r.runtime; },
      rowAttrs: function (r, tr) {
        tr.dataset.key = r.key; if (r.key === S.selected) tr.classList.add("hl");
        tr.addEventListener("click", function (e) { if (e.target.closest("button")) return; S.selected = r.key; highlight(); });
      }
    }));
    return d;
  }
  function highlight() {
    document.querySelectorAll(".cmp-table tbody tr").forEach(function (r) { var on = r.dataset.key === S.selected; r.classList.toggle("hl", on); if (on) r.scrollIntoView({block: "nearest"}); });
    document.querySelectorAll(".pb-scatter .pb-pt").forEach(function (g) { g.classList.toggle("selected", g.getAttribute("data-key") === S.selected); });
  }
  function curves(box, rows) {
    var withCurve = rows.filter(function (r) { return r.curve && r.curve.length; }), withRoc = rows.filter(function (r) { return r.roc && r.roc.length; });
    box.appendChild(el("h3", null, "Detection quality vs confidence"));
    if (!withCurve.length) { box.appendChild(el("p", "empty", F.runs === "pick" ? "Curves come from the latest results only; switch Runs to “Latest of each model”." : "No curves: the shown results have no ground truth or reference labels.")); return; }
    var dash = dashWithin(withCurve), m = F.curve, mDef = CURVE_METRICS.find(function (c) { return c[0] === m; }) || CURVE_METRICS[0];
    var tools = el("div", "chart-tools");
    tools.appendChild(picker("Show", CURVE_METRICS, m, function (v) { F.curve = v; remember(); draw(true); }));
    tools.appendChild(PB.term("What is this?", m, "cmp")); box.appendChild(tools);
    var shown = withCurve.slice(0, MAX_SERIES);
    var series = shown.map(function (r) { return {id: r.key, label: r.label, cls: "ser ser-" + PB.colorOf(r.model), dash: dash[r.key], points: r.curve.map(function (p) { return {x: p.threshold, y: p[m]}; })}; });
    var note = withCurve.length > MAX_SERIES ? "Showing " + MAX_SERIES + " of " + withCurve.length + " results; narrow the filters to see the others." : null;
    var row = el("div", "chart-row");
    var c1 = el("div", "thr-chart-wrap");
    c1.appendChild(C.block({
      key: "cmp-curves", series: series, note: note,
      draw: function (hidden) {
        return C.lineChart(series, {title: mDef[1] + " vs confidence threshold", desc: "One line per result: " + series.map(function (s) { return s.label; }).join(", ") + ".", hidden: hidden, xLabel: "Confidence threshold (\u00d7 100 = Blue Iris %)", yLabel: "%", xName: "confidence", xHover: function (v) { return v.toFixed(2); }, yHover: function (v) { return fmt.pct(v, 1); }, xTicks: [0, 0.2, 0.4, 0.6, 0.8, 1], yTicks: [0, 0.2, 0.4, 0.6, 0.8, 1]});
      },
      table: function () {
        var th = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        return {head: ["Result"].concat(th.map(function (t) { return "at " + t.toFixed(1); })), rows: shown.map(function (r) { return [r.label].concat(th.map(function (t) { var p = r.curve.reduce(function (a, q) { return !a || Math.abs(q.threshold - t) < Math.abs(a.threshold - t) ? q : a; }, null); return p ? fmt.pct(p[m], 1) : "-"; })); })};
      }
    }));
    row.appendChild(c1);
    if (withRoc.length) {
      var rshown = withRoc.slice(0, MAX_SERIES), rdash = dashWithin(rshown);
      var rs = rshown.map(function (r) { return {id: r.key, label: r.label, cls: "ser ser-" + PB.colorOf(r.model), dash: rdash[r.key], points: [{x: 0, y: 0}].concat(r.roc.map(function (p) { return {x: p.fpr, y: p.tpr, t: p.t}; }))}; });
      var c2 = el("div", "roc-chart-wrap wide");
      c2.appendChild(C.block({
        key: "cmp-roc", series: rs, note: withRoc.length > MAX_SERIES ? "Showing " + MAX_SERIES + " of " + withRoc.length + "; narrow the filters." : "Frame level: up and to the left is better; dashed diagonal = chance.",
        draw: function (hidden) {
          return C.lineChart(rs, {title: "Frame-level ROC", desc: "Alert rate against false-alert rate for " + rs.map(function (s) { return s.label; }).join(", ") + ".", hidden: hidden, width: 360, height: 320, left: 44, hover: "xy", diag: true, xGrid: true, xLabel: "False alerts on empty frames (%)", yLabel: "Alerts on frames with objects (%)", xFmt: function (v) { return (v * 100).toFixed(0); }, xName: "false alerts", yName: "alerts", xHover: function (v) { return fmt.pct(v, 1); }, yHover: function (v) { return fmt.pct(v, 1); }, xTicks: [0, 0.2, 0.4, 0.6, 0.8, 1], yTicks: [0, 0.2, 0.4, 0.6, 0.8, 1]});
        },
        table: function () {
          return {head: ["Result", "ROC AUC"], rows: rshown.map(function (r) { return [r.label, fmt.pct(r.roc_auc, 1)]; })};
        }
      }));
      row.appendChild(c2);
    }
    box.appendChild(row);
  }

  // ---- across runs ----------------------------------------------------------------------------
  function histUrl(model) { return "/v1/benchmark/history" + (model ? "?model=" + encodeURIComponent(model) : ""); }
  function entriesOf(j, model) {
    var ms = (j && j.models) || {}; if (Array.isArray(ms)) return ms;
    if (ms[model]) return ms[model];
    var k = Object.keys(ms).find(function (x) { return x.toLowerCase() === String(model).toLowerCase(); });
    return k ? ms[k] : [];
  }
  function loadHistory(model, force) {
    if (!model || (!force && (S.hist[model] || S.histErr[model]))) return;
    S.histErr[model] = null;
    PB.getJSON(histUrl(model)).then(function (j) {
      if (j.success === false || !j.models) { S.histErr[model] = j.message || j.error || "the run history is not available"; S.hist[model] = null; }
      else {
        // The server's comparability with the newest entry, per run.
        var comp = entriesOf({models: j.comparability || {}}, model) || [];
        S.hist[model] = entriesOf(j, model).map(function (e) {
          var c = comp.find(function (x) { return x.run_id === e.run_id; });
          return c ? Object.assign({}, e, {comparable: c.comparable, reasons: c.reasons || []}) : e;
        });
      }
      if (S.runModel === model && S.hist[model]) defaults(S.hist[model], force);
      drawRuns(true);
    }, function () { S.histErr[model] = "the server did not answer"; drawRuns(true); });
  }
  function loadAllHistory(force) {
    if (!force && (S.histAll || S.histAllErr)) return;
    PB.getJSON(histUrl(null)).then(function (j) {
      if (j.success === false || !j.models) { S.histAllErr = j.message || j.error || "the run history is not available on this server"; S.histAll = null; }
      else { S.histAll = j.models; S.histAllErr = null; PB.colorsFor(Object.keys(j.models)); }
      draw(true);
    }, function () { S.histAllErr = "the server did not answer"; draw(true); });
  }
  // Why runs a and b are not comparable (the server's reasons when an entry carries them).
  function reasonsBetween(a, b) {
    var out = [], sa = a.settings || {}, sb = b.settings || {};
    var ds = function (s) { return (s.datasets || []).slice().sort().join(", "); };
    if (ds(sa) !== ds(sb)) out.push("different datasets: " + (ds(sa) || "none") + " → " + (ds(sb) || "none"));
    if (has(sa.images) && has(sb.images) && sa.images !== sb.images) out.push("different image count: " + sa.images + " → " + sb.images);
    if ((sa.reference || null) !== (sb.reference || null)) out.push("different reference model: " + (sa.reference || "none") + " → " + (sb.reference || "none"));
    if (a.hardware_key && b.hardware_key && a.hardware_key !== b.hardware_key) out.push("different hardware: " + a.hardware_key + " → " + b.hardware_key);
    var ra = a.runtimes || {}, rb = b.runtimes || {};
    ["openvino", "onnxruntime"].forEach(function (k) { if (ra[k] && rb[k] && ra[k] !== rb[k]) out.push("different " + (PB.RUNTIME_NAMES[k] || k) + " version: " + ra[k] + " → " + rb[k]); });
    return out;
  }
  function defaults(list, force) {
    if (!list.length) { S.a = S.b = null; return; }
    var ids = list.map(function (e) { return e.run_id; });
    if (force || ids.indexOf(S.b) < 0) S.b = list[0].run_id;
    if (force || ids.indexOf(S.a) < 0 || S.a === S.b) {
      var b = list.find(function (e) { return e.run_id === S.b; }), pin = list.find(function (e) { return e.pinned && e.run_id !== S.b; });
      var prev = list.find(function (e) { return e.run_id !== S.b && !reasonsBetween(e, b).length && new Date(e.timestamp) <= new Date(b.timestamp); });
      var older = list.find(function (e) { return e.run_id !== S.b; });
      S.a = (pin || prev || older || {}).run_id || null;
    }
  }
  function runModels() {
    var names = S.models.map(function (m) { return m.model; });
    var lower = names.map(function (n) { return n.toLowerCase(); });
    if (S.histAll) Object.keys(S.histAll).forEach(function (n) { var e = (S.histAll[n] || [])[0], shown = (e && e.model) || n; if (lower.indexOf(shown.toLowerCase()) < 0) { names.push(shown); lower.push(shown.toLowerCase()); } });
    return names;
  }
  function entryLabel(e) {
    var s = e.settings || {};
    return fmt.day(e.timestamp) + " · " + ((s.datasets || []).join(", ") || "-") + " · " + (has(s.images) ? s.images + " images" : "") + (e.pinned ? " · ★ baseline" : "");
  }
  function buildRuns() {
    var box = $("cmp-runs"); box.replaceChildren();
    box.appendChild(el("h2", null, "Across runs"));
    var names = runModels();
    if (!names.length) { box.appendChild(el("p", "empty", "No results yet: run a benchmark (twice, to compare runs).")); return; }
    if (!S.runModel || names.indexOf(S.runModel) < 0) S.runModel = names[0];
    var tools = el("div", "chart-tools");
    tools.appendChild(picker("Model", names.map(function (n) { return [n, n]; }), S.runModel, function (v) { S.runModel = v; saved.runModel = v; F.runModel = v; remember(); S.a = S.b = null; loadHistory(v); drawRuns(true); }));
    box.appendChild(tools);
    var list = S.hist[S.runModel];
    if (S.histErr[S.runModel]) { box.appendChild(el("p", "warn", "Run history: " + S.histErr[S.runModel] + ".")); return; }
    if (!list) { box.appendChild(el("p", "muted", S.visible ? "loading the run history…" : "The run history loads when this tab opens.")); return; }
    if (!list.length) { box.appendChild(el("p", "empty", "No runs of " + S.runModel + " in the history yet. Each finished benchmark adds one.")); return; }
    if (!S.a || !S.b) defaults(list);
    if (list.length === 1) box.appendChild(el("p", "muted", "Only one run of " + S.runModel + " so far: run the benchmark again (same datasets and images) to compare."));
    // A / B pickers.
    var ab = el("div", "ab-pickers");
    [["a", "Before (A)"], ["b", "After (B)"]].forEach(function (p) {
      var wrapL = el("div", "ab");
      wrapL.appendChild(picker(p[1], list.map(function (e) { return [e.run_id, entryLabel(e)]; }), S[p[0]], function (v) { S[p[0]] = v; drawRuns(true); }));
      var e = list.find(function (x) { return x.run_id === S[p[0]]; });
      if (e) {
        var pin = el("button", "small secondary", e.pinned ? "Unpin baseline" : "Pin as baseline"); pin.type = "button";
        pin.title = e.pinned ? "Stop comparing new runs with this one" : "Compare new runs of this model with this one (instead of the previous run)";
        pin.addEventListener("click", function () { act("/v1/benchmark/history/pin", {model: S.runModel, run_id: e.run_id, pinned: e.pinned ? "0" : "1"}); });
        var del = el("button", "small danger", "Delete run"); del.type = "button";
        del.addEventListener("click", function () { if (confirm("Delete the run of " + fmt.when(e.timestamp) + " of " + S.runModel + " from the history? The current results are not changed.")) act("/v1/benchmark/history/delete", {model: S.runModel, run_id: e.run_id}); });
        wrapL.appendChild(pin); wrapL.appendChild(del);
      }
      ab.appendChild(wrapL);
    });
    box.appendChild(ab);
    var msg = el("p", "muted"); msg.id = "cmp-run-msg"; msg.setAttribute("aria-live", "polite"); box.appendChild(msg);
    if (S.a && S.b && S.a !== S.b) box.appendChild(abTable(list));
    else box.appendChild(el("p", "muted", "Pick two different runs to compare."));
    box.appendChild(trends(list));
  }
  function act(url, fields) {
    PB.post(url, new URLSearchParams(fields)).then(function (j) {
      var m = $("cmp-run-msg"); if (m) { m.className = j.success ? "ok-text" : "error-text"; m.textContent = j.message || j.error || (j.success ? "done" : "failed"); }
      S.cmp = {}; S.histAll = null; S.histAllErr = null; loadHistory(S.runModel, true); if (F.runs === "pick") loadAllHistory(true);
      if (PB.refresh) PB.refresh();
    });
  }
  function abTable(list) {
    var key = S.runModel + "|" + S.a + "|" + S.b, d = el("div", "ab-result"), c = S.cmp[key];
    if (c === undefined) {
      S.cmp[key] = null;
      PB.getJSON("/v1/benchmark/history/compare?model=" + encodeURIComponent(S.runModel) + "&a=" + encodeURIComponent(S.a) + "&b=" + encodeURIComponent(S.b)).then(function (j) {
        S.cmp[key] = j.success === false || !j.changes ? {error: j.message || j.error || "comparison not available"} : j.changes; drawRuns(true);
      }, function () { S.cmp[key] = {error: "the server did not answer"}; drawRuns(true); });
    }
    if (!c) { d.appendChild(el("p", "muted", "comparing…")); return d; }
    if (c.error) { d.appendChild(el("p", "warn", c.error)); return d; }
    if (!c.comparable) {
      var p = el("div", "warn"); p.appendChild(PB.chip("not_comparable")); p.appendChild(document.createTextNode(" These runs are not compared, so no differences are shown:"));
      var ul = el("ul"); (c.reasons || []).forEach(function (r) { ul.appendChild(el("li", null, r)); }); p.appendChild(ul); d.appendChild(p); return d;
    }
    if (c.summary) d.appendChild(el("p", "notice", c.summary));
    var t = el("table", "bench stack ab-table"), th = el("thead"), tr = el("tr");
    ["What", "Device", "A (before)", "B (after)", "Change", "Why"].forEach(function (h) { tr.appendChild(el("th", null, h)); });
    th.appendChild(tr); t.appendChild(th);
    var tb = el("tbody");
    (c.metrics || []).forEach(function (mc) {
      var row = el("tr");
      var cells = [[PB.METRIC_NAMES[mc.metric] || mc.metric, "What"], [mc.device || "model", "Device"]];
      cells.forEach(function (x) { var td = el("td", null, x[0]); td.dataset.label = x[1]; row.appendChild(td); });
      [[PB.metricValue(mc.metric, mc.before), "A (before)"], [PB.metricValue(mc.metric, mc.after), "B (after)"]].forEach(function (x) { var td = el("td", "num", x[0]); td.dataset.label = x[1]; row.appendChild(td); });
      var ch = el("td", "num"); ch.dataset.label = "Change"; ch.appendChild(PB.metricChip(mc)); if (has(mc.delta)) ch.appendChild(el("span", "gap", PB.metricDelta(mc.metric, mc.delta) + (has(mc.rel) ? " (" + (mc.rel > 0 ? "+" : mc.rel < 0 ? "−" : "") + Math.abs(mc.rel * 100).toFixed(0) + "%)" : ""))); row.appendChild(ch);
      var why = el("td", "why", mc.why || ""); why.dataset.label = "Why"; row.appendChild(why);
      tb.appendChild(row);
    });
    t.appendChild(tb);
    var w = el("div", "scroll"); w.dataset.keep = "w:ab"; w.appendChild(t); d.appendChild(w);
    return d;
  }
  var TRENDS = [["p50", "Time per image (p50)"], ["ap50", "Detection quality (AP50)"], ["threshold", "Best confidence"]];
  function trends(list) {
    var d = el("div", "trends"); d.appendChild(el("h3", null, "Trend over runs"));
    var ordered = list.slice(0, 20).reverse(), latest = list[0];
    var tools = el("div", "chart-tools");
    tools.appendChild(picker("Show", TRENDS, S.trend, function (v) { S.trend = v; F.trend = v; remember(); drawRuns(true); }));
    var devs = []; ordered.forEach(function (e) { (e.devices || []).forEach(function (x) { if (devs.indexOf(x.device) < 0) devs.push(x.device); }); });
    if (S.trend !== "threshold") devs.forEach(function (dv) {
      var l = el("label", "check inline"), i = el("input"); i.type = "checkbox"; i.checked = !S.trendOff[dv];
      i.addEventListener("change", function () { if (i.checked) delete S.trendOff[dv]; else S.trendOff[dv] = true; drawRuns(true); });
      l.appendChild(i); l.appendChild(document.createTextNode(" " + dv)); tools.appendChild(l);
    });
    d.appendChild(tools);
    var bad = ordered.map(function (e) { return e.comparable === undefined ? reasonsBetween(e, latest) : e.comparable ? [] : (e.reasons && e.reasons.length ? e.reasons : ["not comparable"]); });
    var val = function (e, dv) {
      if (S.trend === "threshold") return e.threshold ? e.threshold.best : null;
      var x = (e.devices || []).find(function (q) { return q.device === dv; }); if (!x || x.error) return null;
      return S.trend === "p50" ? x.total_p50_ms : x.ap50;
    };
    var keys = S.trend === "threshold" ? ["model"] : devs.filter(function (dv) { return !S.trendOff[dv]; });
    var series = keys.map(function (dv, i) { return {id: dv, label: S.trend === "threshold" ? "best confidence" : dv, cls: "ser ser-" + ((i % 8) + 1), dash: C.dashOf(i), points: ordered.map(function (e, k) { return {x: k, y: bad[k].length ? null : val(e, dv)}; })}; });
    var ys = []; series.forEach(function (s) { s.points.forEach(function (p) { if (has(p.y)) ys.push(p.y); }); });
    if (!ys.length) { d.appendChild(el("p", "empty", "Nothing to draw: no comparable runs with this value.")); }
    else {
      var isMs = S.trend === "p50", lo = isMs ? 0 : Math.max(0, Math.floor((Math.min.apply(null, ys) - 0.05) * 10) / 10), hi = isMs ? C.niceMax(Math.max.apply(null, ys) * 1.1) : Math.min(1, Math.ceil((Math.max.apply(null, ys) + 0.05) * 10) / 10);
      var ticks = ordered.length <= 6 ? ordered.map(function (e, k) { return k; }) : C.niceTicks(0, ordered.length - 1, 5).filter(function (k) { return k === Math.round(k); });
      d.appendChild(C.block({
        key: "cmp-trend", series: series, legend: false,
        draw: function () {
          return C.lineChart(series, {
            title: (TRENDS.find(function (t) { return t[0] === S.trend; }) || TRENDS[0])[1] + " over the last " + ordered.length + " runs of " + S.runModel,
            desc: "One line per device; runs not comparable with the latest are gaps.", pointDots: true,
            xMin: ordered.length > 1 ? 0 : -1, xMax: Math.max(1, ordered.length - 1), yMin: lo, yMax: hi, xTicks: ticks, xTickText: function (k) { var e = ordered[k]; return e ? fmt.day(e.timestamp).replace(/,.*$/, "") : ""; },
            yFmt: isMs ? function (v) { return v.toFixed(0); } : S.trend === "threshold" ? function (v) { return v.toFixed(2); } : function (v) { return (v * 100).toFixed(0); },
            yLabel: isMs ? "ms (lower is better)" : S.trend === "threshold" ? "confidence" : "%", xName: "run",
            xHover: function (k) { var e = ordered[Math.round(k)]; return e ? fmt.day(e.timestamp) : ""; },
            yHover: isMs ? fmt.ms : S.trend === "threshold" ? fmt.thrBi : function (v) { return fmt.pct(v, 1); }
          });
        },
        table: function () { return {head: ["Run"].concat(series.map(function (s) { return s.label; })), rows: ordered.map(function (e, k) { return [fmt.when(e.timestamp)].concat(series.map(function (s) { var v = s.points[k].y; return bad[k].length ? "not comparable" : isMs ? fmt.ms(v) : S.trend === "threshold" ? fmt.thr(v) : fmt.pct(v, 1); })); })}; }
      }));
    }
    var nc = ordered.filter(function (e, k) { return bad[k].length; });
    if (nc.length) {
      var box = el("div", "nc-list"); box.appendChild(el("p", "muted", "Not comparable with the latest run (drawn as gaps):"));
      var ul = el("ul"); nc.forEach(function (e) { var k = ordered.indexOf(e); ul.appendChild(el("li", null, fmt.when(e.timestamp) + ": " + bad[k].join("; "))); }); box.appendChild(ul); d.appendChild(box);
    }
    return d;
  }

  // ---- wiring ---------------------------------------------------------------------------------
  function drawRuns(force) { var b = $("cmp-runs"); if (b) L.keep(b, force ? undefined : JSON.stringify([runModels(), S.runModel, S.hist[S.runModel] || null, S.histErr[S.runModel] || null, S.a, S.b, S.visible]), buildRuns); }
  function draw(force) {
    if (!$("cmp-filters")) return;
    var rows = allRows(), shown = rows.filter(passes), empty = $("cmp-empty");
    var sigF = JSON.stringify([values(rows, "model"), values(rows, "device"), values(rows, "runtime"), values(rows, "dataset"), F, !!S.histAll, S.histAllErr]);
    L.keep($("cmp-filters"), force ? undefined : sigF, function () { buildFilters(rows); });
    var none = !S.models.length && F.runs !== "pick";
    empty.hidden = !none && shown.length > 0;
    empty.textContent = none ? "Nothing to compare yet: run a benchmark under Run (two or more models or devices)." : rows.length ? "Nothing matches the filters: check at least one model, device, runtime and dataset, or press All." : F.runs === "pick" ? "Pick one or more runs above." : F.runs === "this" ? "The latest run has no results yet; choose “Latest of each model”." : "No results to compare.";
    var side = $("cmp-side"); side.hidden = !shown.length;
    if (shown.length) L.keep(side, force ? undefined : JSON.stringify([shown, F.y, F.curve, S.selected]), function () { buildSide(shown); });
    drawRuns(force);
  }
  function render(j, models) {
    S.j = j; S.models = models;
    PB.colorsFor(models.map(function (m) { return m.model; }));
    if (F.runs === "pick" && S.visible) loadAllHistory();
    draw(false);
  }
  // The tab became visible: fetch the run history once.
  function shown() {
    S.visible = true;
    var names = runModels(); if (!S.runModel || names.indexOf(S.runModel) < 0) S.runModel = names[0] || null;
    if (S.runModel) loadHistory(S.runModel);
    if (F.runs === "pick") loadAllHistory();
    drawRuns(false);
  }
  // A new run finished: the history has new entries.
  function invalidate() { S.hist = {}; S.histErr = {}; S.cmp = {}; S.histAll = null; S.histAllErr = null; if (S.visible) shown(); }
  PB.compare = {render: render, shown: shown, invalidate: invalidate, pareto: pareto};
})();
