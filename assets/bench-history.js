// Blue Onyx Prism: Benchmark History (templates/benchmark/history.html; after bench-charts.js and
// bench-shell.js). One model across runs, from GET /v1/benchmark/history[?model=]: the runs (pin a
// baseline: POST /v1/benchmark/history/pin; delete one, asked in the page:
// POST /v1/benchmark/history/delete), two runs compared (GET /v1/benchmark/history/compare:
// A = the later run, B = the earlier; Change = A - B with the server's verdict and why) and the
// trend of the goal metric (with its interval), p95 or the saved confidence on one device; runs on
// other images (not comparable with the newest) are hollow.
(function () {
  "use strict";
  var BC = window.BenchCore, PB = window.PrismBench, S = window.BenchShell, C = PB.charts, el = PB.el, fmt = BC.fmt;
  if (!S || !C) return;
  var $ = function (id) { return document.getElementById(id); };
  var params = new URLSearchParams(location.search);
  var st = {model: params.get("model") || "", entries: [], comp: [], a: null, b: null, device: params.get("device") || "", models: []};
  function say(text, kind) { var m = $("h-msg"); m.className = "bn-msg small" + (kind === "error" ? " bn-error" : kind === "ok" ? " notice" : ""); m.textContent = text || ""; }
  function when(ts) { return window.PrismTime && PrismTime.when ? PrismTime.when(ts) : String(ts || "").replace("T", " ").slice(0, 16); }
  function day(ts) { return window.PrismTime && PrismTime.date ? PrismTime.date(ts) : String(ts || "").slice(0, 10); }
  function devOf(e) {
    var d = (e.devices || []).filter(function (x) { return BC.sameDevice(x.device, st.device); })[0];
    return d || (e.devices || []).filter(function (x) { return x.goal; })[0] || (e.devices || [])[0] || null;
  }
  function comparable(e) { var c = st.comp.filter(function (x) { return x.run_id === e.run_id; })[0]; return c ? c : {comparable: true, reasons: []}; }

  function load() {
    $("h-error").hidden = true;
    return PB.getJSON("/v1/benchmark/history").then(function (j) {
      if (!j.success) { $("h-error").textContent = (j.message || j.error || "The history did not load.") + " Reload the page to try again."; $("h-error").hidden = false; return; }
      st.all = j; st.models = Object.keys(j.models || {});
      var key = st.models.filter(function (n) { return n.toLowerCase() === String(st.model).toLowerCase(); })[0];
      st.model = key || st.models[0] || "";
      choose();
    });
  }
  // The chosen model's runs out of the last answer.
  function choose() {
    var j = st.all || {models: {}, comparability: {}};
    st.entries = st.model ? (j.models[st.model] || []) : [];
    st.comp = st.model ? ((j.comparability || {})[st.model] || []) : [];
    var devs = [];
    st.entries.forEach(function (e) { (e.devices || []).forEach(function (d) { if (!devs.some(function (x) { return BC.sameDevice(x, d.device); })) devs.push(d.device); }); });
    if (!st.device || !devs.some(function (x) { return BC.sameDevice(x, st.device); })) st.device = devs[0] || "";
    st.devices = devs;
    var pinned = st.entries.filter(function (e) { return e.pinned; })[0];
    if (!st.entries.some(function (e) { return e.run_id === st.a; })) st.a = st.entries[0] ? st.entries[0].run_id : null;
    if (!st.entries.some(function (e) { return e.run_id === st.b; }) || st.b === st.a) st.b = pinned && pinned.run_id !== st.a ? pinned.run_id : st.entries[1] ? st.entries[1].run_id : null;
    render();
  }
  function render() {
    var ms = $("h-model");
    ms.replaceChildren.apply(ms, (st.models.length ? st.models : [st.model]).filter(Boolean).map(function (m) { var o = el("option", {value: m, text: m}); o.selected = m === st.model; return o; }));
    if (!st.model) ms.replaceChildren(el("option", {value: "", text: "No history yet"}));
    var dsel = $("h-device");
    dsel.replaceChildren.apply(dsel, (st.devices || []).map(function (d) { var o = el("option", {value: d, text: d}); o.selected = BC.sameDevice(d, st.device); return o; }));
    renderRuns(); renderAB(); renderTrend();
    if (history.replaceState) { var p = new URLSearchParams(location.search); if (st.model) p.set("model", st.model); if (st.device) p.set("device", st.device); history.replaceState(null, "", location.pathname + "?" + p.toString()); }
  }
  function renderRuns() {
    var t = $("h-runs");
    if (!st.entries.length) { t.replaceChildren(el("tbody", {}, [el("tr", {}, [el("td", {text: "No runs recorded for this model yet: a benchmark run records one."})])])); return; }
    var goalName = "Goal metric";
    var g0 = st.entries.map(devOf).filter(function (d) { return d && d.goal; })[0];
    if (g0 && g0.goal.optimize_for) goalName = "Goal (" + g0.goal.optimize_for + ")";
    var head = el("tr", {}, [el("th", {scope: "col", "data-sort-col": "when", text: "When"}), el("th", {scope: "col", "data-sort-col": "images", text: "Images"}), el("th", {scope: "col", "data-sort-col": "devices", text: "Devices"}),
      el("th", {scope: "col", class: "num", "data-sort-col": "goal", "data-sort-first": "desc", text: goalName + " (" + (st.device || "device") + ")"}), el("th", {scope: "col", class: "num", "data-sort-col": "p95", text: "p95"}),
      el("th", {scope: "col", "data-sort-off": ""}, [el("span", {class: "sr-only", text: "Actions"})])]);
    var newest = st.entries[0] && st.entries[0].run_id;
    var rows = st.entries.map(function (e, i) {
      var d = devOf(e), c = comparable(e), g = d && d.goal;
      var chips = [];
      if (e.run_id === newest) chips.push(el("span", {class: "bn-chip same", text: "newest"}));
      if (e.pinned) chips.push(el("span", {class: "bn-chip ok", text: "pinned baseline"}));
      if (!c.comparable) { var b = el("button", {type: "button", class: "term bn-chip warn", "data-tip": "Not comparable with the newest run: " + (c.reasons || []).join("; "), "aria-expanded": "false"}); b.textContent = "other images"; chips.push(b); }
      var lead = el("td", {class: "bn-lead rc-title", "data-label": "When", "data-v": e.timestamp}, [el("time", {datetime: e.timestamp, text: when(e.timestamp)})]);
      chips.forEach(function (x) { lead.appendChild(document.createTextNode(" ")); lead.appendChild(x); });
      var pin = el("button", {type: "button", class: "small secondary", text: e.pinned ? "Unpin" : "Pin", "aria-label": (e.pinned ? "Unpin" : "Pin") + " the run of " + when(e.timestamp)});
      pin.addEventListener("click", function () { act(pin, "/v1/benchmark/history/pin", {model: st.model, run_id: e.run_id, pinned: e.pinned ? "0" : "1"}); });
      var del = el("button", {type: "button", class: "small danger", text: "Delete", "aria-label": "Delete the run of " + when(e.timestamp)});
      del.addEventListener("click", function () { askDelete(e, del); });
      var s = e.settings || {};
      return el("tr", {"data-best": String(i + 1)}, [lead,
        el("td", {"data-label": "Images", text: (s.datasets || []).join(", ") + (s.images ? " · " + fmt.int(s.images) : "")}),
        el("td", {"data-label": "Devices", text: (e.devices || []).map(function (x) { return x.device; }).join(", ")}),
        el("td", {class: "num", "data-label": goalName, "data-v": g ? String(g.value) : ""}, [fmt.n2(g && g.value), g && BC.has(g.low) ? el("span", {class: "bn-ci", text: fmt.ci(g)}) : null]),
        el("td", {class: "num", "data-label": "p95", "data-v": d && BC.has(d.total_p95_ms) ? String(d.total_p95_ms) : "", text: fmt.ms(d && d.total_p95_ms)}),
        el("td", {class: "rc-action", "data-label": ""}, [pin, " ", del])]);
    });
    t.replaceChildren(el("thead", {}, [head]), el("tbody", {}, rows));
    if (window.PrismSort) PrismSort.apply(t);
  }
  function act(button, url, fields) {
    button.disabled = true; say("Saving…");
    PB.post(url, new URLSearchParams(fields)).then(function (j) {
      say(j.message || j.error || (j.success ? "Done." : "Not changed."), j.success ? "ok" : "error");
      button.disabled = false;
      if (j.success) load();
    });
  }
  function askDelete(e, button) {
    var box = $("h-confirm");
    $("h-confirm-text").textContent = "Delete the run of " + when(e.timestamp) + " of " + st.model + " from the history? The current results do not change.";
    box.hidden = false;
    $("h-confirm-yes").onclick = function () { box.hidden = true; act(button, "/v1/benchmark/history/delete", {model: st.model, run_id: e.run_id}); };
    $("h-confirm-no").onclick = function () { box.hidden = true; button.focus(); };
    $("h-confirm-yes").focus();
  }

  // ---- A / B -------------------------------------------------------------------------------------
  var VERDICT = {better: ["ok", "clearly better"], worse: ["bad", "worse"], same: ["same", "within noise"], not_comparable: ["warn", "not comparable"]};
  var NAMES = {goal: "Goal metric", p50_ms: "p50", infer_p50_ms: "Inference p50", fps: "Images per second", ap50: "AP50", ap50_95: "AP50-95", roc_auc: "ROC AUC", precision: "Precision", recall: "Recall", f1: "F1", person_false_alerts: "Person false alerts", threshold: "Saved confidence"};
  function valueText(metric, v) {
    if (!BC.has(v)) return "-";
    if (/_ms$/.test(metric)) return fmt.ms(v);
    if (metric === "fps") return fmt.n1(v) + " /s";
    if (metric === "person_false_alerts") return fmt.pct(v);
    return fmt.n2(v);
  }
  function renderAB() {
    var a = $("h-a"), b = $("h-b"), opts = function (sel, on) {
      sel.replaceChildren.apply(sel, st.entries.map(function (e) { var o = el("option", {value: e.run_id, text: when(e.timestamp) + (e.pinned ? " (pinned)" : "")}); o.selected = e.run_id === on; return o; }));
    };
    opts(a, st.a); opts(b, st.b);
    var t = $("h-ab"), sum = $("h-ab-summary");
    if (!st.a || !st.b || st.a === st.b) { t.replaceChildren(); sum.textContent = st.entries.length < 2 ? "One run so far: a second run compares here." : "Choose two different runs."; return; }
    var ea = st.entries.filter(function (e) { return e.run_id === st.a; })[0], eb = st.entries.filter(function (e) { return e.run_id === st.b; })[0];
    var later = ea.timestamp >= eb.timestamp ? ea : eb, earlier = later === ea ? eb : ea;
    sum.textContent = "Loading…";
    PB.getJSON("/v1/benchmark/history/compare?model=" + encodeURIComponent(st.model) + "&a=" + encodeURIComponent(earlier.run_id) + "&b=" + encodeURIComponent(later.run_id)).then(function (j) {
      if (!j.success) { sum.textContent = (j.message || j.error || "Not compared.") + " Reload the page."; return; }
      var ch = j.changes, flip = later !== ea;
      sum.textContent = (ch.summary || "") + (ch.comparable ? "" : " Not directly comparable: " + (ch.reasons || []).join("; ") + ".");
      var head = el("tr", {}, ["What", "Device", "A", "B", "Change", "Why"].map(function (h, i) { return el("th", {scope: "col", class: i >= 2 && i <= 4 ? "num" : null, text: h}); }));
      var rows = (ch.metrics || []).map(function (m) {
        var av = flip ? m.before : m.after, bv = flip ? m.after : m.before, d = BC.has(av) && BC.has(bv) ? av - bv : null;
        var v = VERDICT[m.verdict] || ["same", m.verdict];
        if (flip && m.verdict === "better") v = VERDICT.worse; else if (flip && m.verdict === "worse") v = VERDICT.better;
        var dtext = d === null ? "-" : /_ms$/.test(m.metric) ? (d > 0 ? "+" : d < 0 ? "−" : "±") + fmt.ms(Math.abs(d)) : fmt.signed(d);
        return el("tr", {}, [el("td", {class: "bn-lead rc-title", "data-label": "What", text: NAMES[m.metric] || m.metric}), el("td", {"data-label": "Device"}, [m.device ? el("code", {text: m.device}) : "model"]),
          el("td", {class: "num", "data-label": "A", text: valueText(m.metric, av)}), el("td", {class: "num", "data-label": "B", text: valueText(m.metric, bv)}),
          el("td", {class: "num", "data-label": "Change"}, [dtext + " ", el("span", {class: "bn-chip " + v[0], text: v[1]})]), el("td", {class: "small", "data-label": "Why", text: m.why || "-"})]);
      });
      t.replaceChildren(el("thead", {}, [head]), el("tbody", {}, rows));
    });
  }
  $("h-a").addEventListener("change", function (e) { st.a = e.target.value; renderAB(); });
  $("h-b").addEventListener("change", function (e) { st.b = e.target.value; renderAB(); });

  // ---- trend -------------------------------------------------------------------------------------
  function renderTrend() {
    var kind = $("h-metric").value, box = $("h-trend"), list = st.entries.slice().reverse();
    var pts = list.map(function (e, i) {
      var d = (e.devices || []).filter(function (x) { return BC.sameDevice(x.device, st.device); })[0], c = comparable(e);
      var y = null, lo = null, hi = null;
      if (kind === "goal" && d && d.goal) { y = d.goal.value; lo = d.goal.low; hi = d.goal.high; }
      else if (kind === "p95" && d) y = d.total_p95_ms;
      else if (kind === "conf" && e.threshold) y = e.threshold.configured;
      return {x: i, y: BC.has(y) ? y : null, low: lo, high: hi, hollow: !c.comparable, label: when(e.timestamp) + ": " + (kind === "p95" ? fmt.ms(y) : fmt.n2(y)) + (c.comparable ? "" : " (other images)"), ts: e.timestamp};
    });
    var have = pts.filter(function (p) { return BC.has(p.y); });
    if (!have.length) { box.replaceChildren(el("p", {class: "bn-empty", text: "Nothing to draw for " + (st.device || "this device") + " yet."})); $("h-trend-table").replaceChildren(); return; }
    var ys = []; have.forEach(function (p) { ys.push(p.y); if (BC.has(p.low)) ys.push(p.low); if (BC.has(p.high)) ys.push(p.high); });
    var lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys), pad = (hi - lo) * 0.15 || (kind === "p95" ? hi * 0.2 || 10 : 0.05);
    var y0 = kind === "p95" ? Math.max(0, lo - pad) : Math.max(0, lo - pad), y1 = kind === "p95" ? hi + pad : Math.min(1, hi + pad);
    var name = kind === "goal" ? "Goal metric" : kind === "p95" ? "p95" : "Saved confidence";
    var svg = C.lineChart([{id: "t", label: name, cls: "ser-1", points: pts.filter(function (p) { return !p.hollow; })}, {id: "o", label: name + " (other images)", cls: "ser-1", noLine: true, points: pts.filter(function (p) { return p.hollow; })}], {
      title: name + " of " + st.model + " on " + st.device + " over its runs", desc: "One point per run, oldest left; bars are 95% intervals; hollow points ran on other images. The table below lists the values.",
      xMin: -0.4, xMax: Math.max(0.6, pts.length - 0.6), xTicks: pts.map(function (p) { return p.x; }), xTickText: function (v) { var p = pts[Math.round(v)]; return p ? day(p.ts).slice(5) : ""; },
      yMin: y0, yMax: y1, yTicks: C.niceTicks(y0, y1, 4), yFmt: kind === "p95" ? function (v) { return fmt.ms(v); } : function (v) { return v.toFixed(2); },
      pointDots: true, bars: true, hover: "xy", xName: "run", yName: name, xHover: function (v) { var p = pts[Math.round(v)]; return p ? when(p.ts) : ""; }, yHover: kind === "p95" ? fmt.ms : fmt.n2,
      height: 220, avail: box.clientWidth, left: 58, width: Math.min(1000, Math.max(300, box.clientWidth || 560))
    });
    box.replaceChildren(svg);
    var rows = pts.map(function (p) { return el("tr", {}, [el("td", {"data-label": "Run", text: when(p.ts)}), el("td", {class: "num", "data-label": name, text: kind === "p95" ? fmt.ms(p.y) : fmt.n2(p.y)}), el("td", {class: "num", "data-label": "95% interval", text: BC.has(p.low) ? fmt.n2(p.low) + "–" + fmt.n2(p.high) : "-"}), el("td", {"data-label": "Images", text: p.hollow ? "other" : "same"})]); });
    $("h-trend-table").replaceChildren(el("table", {class: "bn-table rcards"}, [el("thead", {}, [el("tr", {}, ["Run", name, "95% interval", "Images"].map(function (h, i) { return el("th", {scope: "col", class: i === 1 || i === 2 ? "num" : null, text: h}); }))]), el("tbody", {}, rows)]));
  }
  $("h-metric").addEventListener("change", renderTrend);
  $("h-device").addEventListener("change", function (e) { st.device = e.target.value; renderRuns(); renderTrend(); });
  $("h-model").addEventListener("change", function (e) { st.model = e.target.value; st.a = st.b = null; choose(); });
  var rt = null;
  window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(renderTrend, 150); });
  S.onView(function (why) { if (why === "run") load(); });
  load();
})();
