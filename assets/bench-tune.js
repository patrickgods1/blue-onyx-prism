// Blue Onyx Prism: Benchmark step 3, Tune & save (templates/benchmark/tune.html; after
// bench-charts.js, bench-shell.js and bench-frames.js).
// One combo: GET /v1/benchmark/tune?model=&device=&set= (the 0.01 grid, the suggested pick, its
// cross-validated figure and interval, the target, the saved value, savable or why not). The chart
// draws the goal metric, recall and precision (left axis) and FA/100 dashed (right axis), the
// suggested line with the equally good band, the saved line and the slider's cursor; the slider
// moves over the grid (input: the cursor and the numbers, no request; release: the example strips
// from GET /v1/benchmark/tune/images, debounced, the previous request aborted). Save writes the
// slider's value: POST /v1/benchmark/apply-profiles `profile=<device>|<value>|<model>` (and
// `use=<device>|<model>` when Run on names another device); a low-confidence pick asks first, in
// the page; a pick on another model's detections (relative) or on a flat curve, or a target not
// met, cannot be saved (the server's `notSavable`).
// All models and devices: the Compare view's rows of the same set (saved, suggested, the goal and
// FA/100 at the saved and the suggested value: `atSaved` / `atSuggested`) with Run on radios;
// Save all / Save confidences only post explicit values. Every save answers with Undo (the
// server's `undo` pairs) and, when a device change waits for one, Restart now.
(function () {
  "use strict";
  var BC = window.BenchCore, PB = window.PrismBench, S = window.BenchShell, C = PB.charts, F = window.BenchFrames, el = PB.el, fmt = BC.fmt;
  if (!S || !C || !F) return;
  var $ = function (id) { return document.getElementById(id); };
  var params = new URLSearchParams(location.search);
  var st = {model: params.get("model") || "", device: params.get("device") || "", set: params.get("set") || "", compare: null, tune: null, t: null, useDev: {}, cfg: null, strips: null, err: null};
  // One Compare and one tune request at a time, the latest choice wins (BenchCore.latest).
  var compareLoads = BC.latest(), tuneLoads = BC.latest();
  var bodyEl = $("tune-body"), emptyEl = $("tune-empty");
  function say(box, text, kind, extra) {
    var p = el("div", {class: kind === "error" ? "bn-error" : "notice"}, [el("span", {text: text})].concat(extra || []));
    box.replaceChildren(p);
  }
  function cap(t) { t = String(t || ""); return t.charAt(0).toUpperCase() + t.slice(1); }
  function cpu(d) { return /(^|:)cpu$/.test(String(d)); }

  // ---- the objective --------------------------------------------------------------------------------
  // What the chart's goal line shows: the tuning objective's value at each grid row.
  function objectiveOf(t) { return BC.parseGoal(t.objective) || {kind: "f1", target: null}; }
  function valueAt(row, o) {
    if (!row) return null;
    switch (o.kind) {
      case "f1": return row.f1; case "f2": return row.f2; case "f05": return row.f05;
      case "precision": case "recall_at_fa": return row.recall;
      case "recall": return row.precision;
      case "fpr": return row.tpr;
      case "youden": return BC.has(row.tpr) && BC.has(row.fpr) ? row.tpr - row.fpr : null;
    }
    return row.f1;
  }
  function rowAt(t, x) {
    if (!t || !BC.has(x)) return null;
    var best = null;
    t.grid.forEach(function (r) { if (!best || Math.abs(r.threshold - x) < Math.abs(best.threshold - x)) best = r; });
    return best && Math.abs(best.threshold - x) < 0.006 ? best : null;
  }

  // ---- loading ------------------------------------------------------------------------------------
  function api(path, q) {
    var parts = [];
    Object.keys(q).forEach(function (k) { if (BC.has(q[k]) && q[k] !== "") parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(q[k])); });
    return S.withGoal(path + (parts.length ? "?" + parts.join("&") : ""));
  }
  function remember() {
    var p = new URLSearchParams(location.search);
    ["model", "device", "set"].forEach(function (k) { if (st[k]) p.set(k, st[k]); else p.delete(k); });
    var q = p.toString();
    if (history.replaceState) history.replaceState(null, "", location.pathname + (q ? "?" + q : ""));
  }
  // The Compare rows of the chosen set (the table and Save all): true once they are in.
  function loadCompare() {
    var t = compareLoads.begin();
    bodyEl.setAttribute("aria-busy", "true"); // until the set's rows (and its tune view) show
    return PB.getJSON(api("/v1/benchmark/compare", {set: st.set}), t.init).then(function (j) {
      if (!t.wins(j)) return false; // a newer choice took its place
      if (!tuneLoads.busy()) bodyEl.removeAttribute("aria-busy");
      if (j._status === 404) { bodyEl.hidden = true; emptyEl.hidden = false; return false; }
      if (!j.success) { showError((j.message || "The results did not load.") + " Reload the page to try again.", "compare"); return false; }
      if (st.err === "compare") showError("");
      st.compare = j.compare;
      if (!st.set && j.compare.set) st.set = j.compare.set.id;
      S.setGoal(j.compare.goal);
      return true;
    });
  }
  // The Compare rows Save all would post: only those of the set the page shows.
  function tableSet() { return st.compare && st.compare.set ? st.compare.set.id : null; }
  // `from`: which load failed ("compare" | "tune"), so a later success of the same clears it.
  function showError(text, from) {
    var e = $("t-error"); e.textContent = text; e.hidden = !text; st.err = text ? from || "tune" : null;
    if (!text) return;
    // Skeletons give way to a line saying nothing loaded (what did load stays).
    var none = function () { return el("p", {class: "muted small", text: "Not loaded."}); };
    if (!st.tune) { $("t-chart").replaceChildren(none()); $("t-side").replaceChildren(none()); $("t-examples").replaceChildren(el("h3", {text: "What you would miss and what would alert"}), none()); }
    if (!st.compare) $("t-all").replaceChildren(el("tbody", {}, [el("tr", {}, [el("td", {text: "Not loaded."})])]));
  }
  function loadTune() {
    var t = tuneLoads.begin();
    bodyEl.setAttribute("aria-busy", "true");
    return PB.getJSON(api("/v1/benchmark/tune", {model: st.model, device: st.device, set: st.set}), t.init).then(function (j) {
      if (!t.wins(j)) return; // a newer choice took its place
      if (!compareLoads.busy()) bodyEl.removeAttribute("aria-busy");
      if (!j.success) { showError((j.message || "This combo did not load.") + " Choose another model or device.", "tune"); return; }
      showError("");
      st.tune = j.tune; st.model = j.tune.model; st.device = j.tune.device; st.set = j.tune.set;
      resultBox.replaceChildren();
      st.t = j.tune.suggested ? j.tune.suggested.threshold : (j.tune.saved ? j.tune.saved.threshold : 0.5);
      S.setGoal(j.tune.goal);
      S.setSet(st.set);
      remember(); renderSelects(); renderTune(); loadStrips();
    });
  }
  // The set's rows, then the combo's tune view when the one shown is of another set (or none),
  // or always with `fresh` (the saved results changed). Whichever call wins does both, so a
  // refresh that replaces a set change's load still loads that set's tune view.
  function loadSet(fresh) {
    return loadCompare().then(function (ok) {
      if (!ok) return;
      if (!st.model) { var r0 = st.compare.rows.filter(function (r) { return r.block === "ok"; })[0] || st.compare.rows[0]; if (r0) { st.model = r0.model; st.device = r0.device; } }
      if (!st.model) { bodyEl.hidden = true; emptyEl.hidden = false; return; }
      renderAll();
      if (fresh || !st.tune || st.tune.set !== st.set || tuneLoads.busy()) loadTune();
    });
  }
  function start() { loadSet(false); }

  // ---- selects ------------------------------------------------------------------------------------
  function option(v, text, on) { var o = el("option", {value: v, text: text}); o.selected = !!on; return o; }
  function renderSelects() {
    var rows = (st.compare && st.compare.rows) || [], models = [];
    rows.forEach(function (r) { if (models.indexOf(r.model) < 0) models.push(r.model); });
    if (st.model && models.indexOf(st.model) < 0) models.push(st.model);
    $("t-model").replaceChildren.apply($("t-model"), models.map(function (m) { return option(m, m, m === st.model); }));
    var devs = (st.tune && st.tune.devices) || rows.filter(function (r) { return r.model === st.model; }).map(function (r) { return r.device; });
    if (st.device && !devs.some(function (d) { return BC.sameDevice(d, st.device); })) devs = devs.concat([st.device]);
    var label = function (d) { var r = rows.filter(function (x) { return x.model === st.model && BC.sameDevice(x.device, d); })[0]; return d + (r && r.deviceLabel ? " (" + r.deviceLabel + ")" : ""); };
    $("t-device").replaceChildren.apply($("t-device"), devs.map(function (d) { return option(d, label(d), BC.sameDevice(d, st.device)); }));
    var titles = {};
    ((st.compare && st.compare.sets) || []).forEach(function (s) { titles[s.id] = s.title + " · " + fmt.int(s.frames) + " frames"; });
    var sets = (st.tune && st.tune.sets) || [st.set];
    $("t-set").replaceChildren.apply($("t-set"), sets.map(function (s) { return option(s, titles[s] || s, s === st.set); }));
  }
  $("t-model").addEventListener("change", function (e) { st.model = e.target.value; st.device = ""; loadTune(); });
  $("t-device").addEventListener("change", function (e) { st.device = e.target.value; loadTune(); });
  // A set change: the tune answer in flight is of the set before (it would pull the page back).
  $("t-set").addEventListener("change", function (e) { st.set = e.target.value; tuneLoads.stop(); loadSet(false); });

  // ---- chart ----------------------------------------------------------------------------------------
  var chartSvg = null;
  function drawChart() {
    var t = st.tune, box = $("t-chart"), o = objectiveOf(t);
    var faMax = C.niceMax(Math.max(5, Math.max.apply(null, t.grid.map(function (r) { return r.fa100 || 0; }))));
    var name = t.objectiveName || "Goal";
    var series = [
      {id: "goal", label: name, cls: "c-f", points: t.grid.map(function (r) { return {x: r.threshold, y: valueAt(r, o)}; })},
      {id: "recall", label: "Recall", cls: "c-r", points: t.grid.map(function (r) { return {x: r.threshold, y: r.recall}; })},
      {id: "precision", label: "Precision", cls: "c-p", points: t.grid.map(function (r) { return {x: r.threshold, y: r.precision}; })},
      {id: "fa", label: "FA/100 empty", cls: "c-fa", right: true, points: t.grid.map(function (r) { return {x: r.threshold, y: r.fa100}; })}
    ];
    var sug = t.suggested, saved = t.saved, marks = [], hl = [];
    if (sug) marks.push({x: sug.threshold, cls: "bn-mark sug", label: "suggested " + fmt.n2(sug.threshold), textCls: "t-sug"});
    if (saved && (!sug || Math.abs(saved.threshold - sug.threshold) > 0.004)) marks.push({x: saved.threshold, cls: "bn-mark saved", label: "saved " + fmt.n2(saved.threshold)});
    if (t.target) {
      if (o.kind === "recall_at_fa") hl.push({y: o.target, right: true});
      else hl.push({y: o.target});
    }
    var first = t.grid.length ? t.grid[0].threshold : 0.05, last = t.grid.length ? t.grid[t.grid.length - 1].threshold : 1;
    chartSvg = C.lineChart(series, {
      title: name + ", recall, precision and false alerts against confidence for " + t.model + " on " + t.device,
      desc: "Left axis 0 to 1: " + name + ", recall, precision. Right axis: false alerts per 100 empty frames (dashed). The table below the slider lists every value.",
      xMin: first, xMax: last, xTicks: [0.05, 0.25, 0.5, 0.75, 1].filter(function (v) { return v >= first - 1e-9 && v <= last + 1e-9; }), xFmt: function (v) { return v.toFixed(2); },
      yTicks: [0, 0.25, 0.5, 0.75, 1], yFmt: function (v) { return v.toFixed(2); }, yHover: fmt.n2, xName: "confidence", xHover: fmt.n2,
      right: {max: faMax, label: "FA/100", fmt: function (v) { return String(Math.round(v)); }, hover: fmt.n1},
      bands: sug && sug.plateau ? [{x1: sug.plateau[0], x2: sug.plateau[1]}] : [], marks: marks, hlines: hl, top: 30,
      cursor: {x: st.t}, xLabel: "confidence →", avail: box.clientWidth,
      width: Math.min(900, Math.max(300, box.clientWidth || 560)), height: box.clientWidth < 500 ? 260 : 300,
      onPick: function (x) { setT(x, true); }
    });
    box.replaceChildren(chartSvg);
    var lg = $("t-legend");
    lg.replaceChildren.apply(lg, [["c-f", name, false], ["c-r", "Recall", false], ["c-p", "Precision", false], ["c-fa", "False alerts /100 empty (right axis)", true]].map(function (l) {
      return el("li", {class: l[0]}, [el("span", {class: "ln" + (l[2] ? " dash" : ""), "aria-hidden": "true"}), el("span", {text: l[1]})]);
    }));
    renderGridTable();
  }
  function renderGridTable() {
    var t = st.tune, o = objectiveOf(t), d = $("t-table-d");
    if (!d.open) return;
    var rows = t.grid.map(function (r) {
      return el("tr", {}, [el("td", {class: "num", "data-label": "Confidence", text: fmt.n2(r.threshold)}), el("td", {class: "num", "data-label": t.objectiveName, text: fmt.n2(valueAt(r, o))}),
        el("td", {class: "num", "data-label": "Recall", text: fmt.n2(r.recall)}), el("td", {class: "num", "data-label": "Precision", text: fmt.n2(r.precision)}),
        el("td", {class: "num", "data-label": "FA/100 empty", text: fmt.n1(r.fa100)}), el("td", {class: "num", "data-label": "TP / FP / FN", text: r.tp + " / " + r.fp + " / " + r["fn"]})]);
    });
    var tb = el("table", {class: "bn-table rcards"}, [el("thead", {}, [el("tr", {}, ["Confidence", t.objectiveName, "Recall", "Precision", "FA/100 empty", "TP / FP / FN"].map(function (h) { return el("th", {class: "num", scope: "col", text: h}); }))]), el("tbody", {}, rows)]);
    $("t-table").replaceChildren(tb);
  }
  $("t-table-d").addEventListener("toggle", function () { if (st.tune) renderGridTable(); });

  // ---- slider and numbers ---------------------------------------------------------------------------
  var slider = $("t-slider");
  function setT(x, release) {
    var r = rowAt(st.tune, x); if (!r) return;
    st.t = r.threshold;
    slider.value = String(Math.round(st.t * 100));
    if (chartSvg) chartSvg.setCursor(st.t);
    renderSide(); updateSave();
    if (release) loadStripsSoon();
  }
  slider.addEventListener("input", function () { setT(parseInt(slider.value, 10) / 100, false); });
  slider.addEventListener("change", function () { setT(parseInt(slider.value, 10) / 100, true); });
  $("t-reset").addEventListener("click", function () { if (st.tune && st.tune.suggested) setT(st.tune.suggested.threshold, true); });

  function big(k, v, d) { return el("div", {}, [el("span", {class: "k", text: k}), el("span", {class: "v", text: v}), d]); }
  function delta(a, b, better, dp) {
    if (!BC.has(b)) return el("span", {class: "d", text: "nothing saved for this device"});
    if (!BC.has(a)) return el("span", {class: "d", text: "-"});
    var d = a - b;
    if (Math.abs(d) < (dp === 1 ? 0.05 : 0.005)) return el("span", {class: "d", text: "same as saved"});
    var good = better === "up" ? d > 0 : d < 0;
    return el("span", {class: "d " + (good ? "up" : "down"), text: fmt.signed(d, dp === undefined ? 2 : dp) + " vs saved " + fmt.n2(st.tune.saved.threshold)});
  }
  function renderSide() {
    var t = st.tune, box = $("t-side"), o = objectiveOf(t), sug = t.suggested, kids = [];
    var at = rowAt(t, st.t), sv = t.saved ? rowAt(t, t.saved.threshold) : null;
    $("t-out").textContent = fmt.n2(st.t);
    slider.setAttribute("aria-valuetext", fmt.n2(st.t) + (sug && Math.abs(st.t - sug.threshold) < 0.004 ? ", suggested" : ""));
    if (sug) {
      kids.push(el("h3", {text: "Suggested " + fmt.n2(sug.threshold)}));
      // How it was validated; the server's low-confidence reason (too little data, or a fold that
      // borrowed the pick on every frame) follows as its own line.
      var cv = BC.cvText(t);
      var range = BC.rangeText(sug.plateau);
      kids.push(el("p", {class: "small muted", text: cv + " " + (range ? "Equally good: " + range + "." : "No other value scores as well.")}));
    } else kids.push(el("h3", {text: "No suggestion"}));
    if (t.lowConfidence) kids.push(el("p", {class: "warn small", text: cap(t.lowConfidence)}));
    if (t.flatNote) kids.push(el("p", {class: "warn small", text: t.flatNote}));
    if (t.target) kids.push(el("p", {class: (t.target.met ? "notice" : "warn") + " small", text: "Target " + t.target.text + (t.target.met ? " on frames the pick never saw." : ". Saving is off.")}));
    if (t.relative) kids.push(el("p", {class: "warn small", text: "These images have no labels: the curve is against another model's detections, so it cannot be saved."}));
    kids.push(el("p", {class: "small", text: "At " + fmt.n2(st.t) + (sug && Math.abs(st.t - sug.threshold) < 0.004 ? " (suggested)" : "") + ":"}));
    var sv0 = function (f) { return sv ? f(sv) : null; };
    kids.push(el("div", {class: "bn-big"}, [
      big(t.objectiveName, fmt.n2(valueAt(at, o)), delta(valueAt(at, o), sv0(function (r) { return valueAt(r, o); }), "up")),
      big("Recall (found)", fmt.n2(at && at.recall), delta(at && at.recall, sv0(function (r) { return r.recall; }), "up")),
      big("Precision (right)", fmt.n2(at && at.precision), delta(at && at.precision, sv0(function (r) { return r.precision; }), "up")),
      big("False alerts /100 empty", fmt.n1(at && at.fa100), delta(at && at.fa100, sv0(function (r) { return r.fa100; }), "down", 1))
    ]));
    if (at) {
      var objects = at.tp + at["fn"], alerts = BC.has(at.fa100) ? Math.round(at.fa100 * t.emptyFrames / 100) : null;
      kids.push(el("p", {class: "small muted", text: fmt.int(at.tp) + " of " + fmt.int(objects) + " objects found · " + fmt.int(at["fn"]) + " missed · " + (BC.has(alerts) ? fmt.int(alerts) + " of " + fmt.int(t.emptyFrames) + " empty frames alert" : "no empty frames") + " · " + fmt.int(at.fp) + " wrong detections."}));
    }
    box.replaceChildren.apply(box, kids);
  }

  // ---- example strips ---------------------------------------------------------------------------------
  var stripTimer = null, stripCtl = null;
  function loadStripsSoon() { clearTimeout(stripTimer); stripTimer = setTimeout(loadStrips, 250); }
  function loadStrips() {
    var t = st.tune; if (!t) return;
    if (stripCtl) stripCtl.abort();
    var ctl = window.AbortController ? new AbortController() : null; stripCtl = ctl;
    var thr = st.t, q = {model: t.model, device: t.device, set: t.set, threshold: thr.toFixed(2), limit: 6};
    var box = $("t-examples"); box.setAttribute("aria-busy", "true");
    Promise.all(["miss", "fa"].map(function (k) { return PB.getJSON(api("/v1/benchmark/tune/images", Object.assign({kind: k}, q)), ctl ? {signal: ctl.signal} : {}); })).then(function (js) {
      if (js.some(function (j) { return j.aborted; })) return;
      stripCtl = null; box.removeAttribute("aria-busy");
      var kids = [el("h3", {text: "At " + fmt.n2(thr) + ": what you would miss and what would alert"})];
      ["miss", "fa"].forEach(function (k, i) {
        var j = js[i], name = k === "miss" ? "Misses" : "False alerts";
        if (!j.success) { kids.push(el("p", {class: "bn-error small", text: name + ": " + (j.message || "did not load") + "."})); return; }
        var ex = j.images, link = "/benchmark/images?model=" + encodeURIComponent(t.model) + "&device=" + encodeURIComponent(t.device) + "&set=" + encodeURIComponent(t.set) + "&threshold=" + thr.toFixed(2) + "&kind=" + k;
        kids.push(el("p", {class: "small"}, [el("strong", {text: name + " (" + fmt.int(ex.total) + " frame" + (ex.total === 1 ? "" : "s") + ")"}), " ", ex.total ? el("a", {class: "bn-seeall", href: S.withGoal(link), text: "see all"}) : null]));
        if (!ex.frames.length) kids.push(el("p", {class: "muted small", text: k === "miss" ? "Nothing missed at this confidence." : "No false alert at this confidence."}));
        else kids.push(el("div", {class: "bn-thumbs strip"}, ex.frames.map(function (f) { return F.thumb(f, {threshold: thr, model: t.model, device: t.device}); })));
      });
      box.replaceChildren.apply(box, kids);
    });
  }

  // ---- saving -----------------------------------------------------------------------------------------
  function configModel(name) {
    var p = st.cfg; if (!p || !p.models) return null;
    return p.models.filter(function (m) { return String(m.name).toLowerCase() === String(name).toLowerCase(); })[0] || null;
  }
  // The device the model runs on now (loaded), else the one the config resolves to.
  function configuredDevice(model) {
    var m = configModel(model);
    return m ? (m.running && m.running.device) || m.resolvedDevice || null : null;
  }
  function runsOn(model) { return st.useDev[model] || configuredDevice(model); }
  // The single save's answer: one node, kept through the card's re-renders.
  var resultBox = el("div", {class: "bn-result", role: "status", "aria-live": "polite"});
  function post(pairs, label, box, button) {
    var b = new URLSearchParams(); pairs.forEach(function (p) { b.append(p[0], p[1]); });
    if (button) button.disabled = true;
    box.replaceChildren(el("p", {class: "muted small", text: "Saving…"}));
    return PB.post("/v1/benchmark/apply-profiles", b).then(function (j) {
      if (button) button.disabled = false;
      if (!j.success) { say(box, (j.message || "Nothing was saved.") + " Nothing changed.", "error"); return j; }
      var extra = [];
      if (j.undo && j.undo.length) {
        var u = el("button", {type: "button", class: "small secondary", text: "Undo"});
        u.addEventListener("click", function () {
          u.disabled = true;
          PB.post("/v1/benchmark/apply-profiles", new URLSearchParams(j.undo.map(function (p) { return [p[0], p[1]]; }))).then(function (k) {
            say(box, k.success ? "Undone: the previous values are back." : (k.message || "Not undone.") + " Try again.", k.success ? "ok" : "error");
            if (k.success) refresh();
          });
        });
        extra.push(u);
      }
      if (j.needsRestart && !j.restarting) {
        var r = el("button", {type: "button", class: "small danger", text: "Restart now"});
        r.addEventListener("click", function () { confirmRestart(box, r); });
        extra.push(r);
      }
      say(box, (j.message || label) + (j.needsRestart ? " The device change waits for a restart." : ""), "ok", extra);
      refresh();
      return j;
    });
  }
  function confirmRestart(box, r) {
    var c = el("div", {class: "bn-confirm", role: "alertdialog"}, [el("p", {text: "Restart the server now? It reloads the config file and recompiles the enabled models; Blue Iris gets no answers for a few seconds."})]);
    var yes = el("button", {type: "button", class: "danger", text: "Restart"}), no = el("button", {type: "button", class: "secondary", text: "Cancel"});
    yes.addEventListener("click", function () { c.remove(); window.PrismLive.restart(r, {confirmed: true}); });
    no.addEventListener("click", function () { c.remove(); r.focus(); });
    c.appendChild(el("div", {class: "actions"}, [yes, no]));
    box.appendChild(c); yes.focus();
  }
  function renderSave() {
    var t = st.tune, box = $("t-save"), kids = [el("h3", {text: "Save"})];
    kids.push(el("p", {class: "small muted", text: "Saved now: " + (t.saved ? fmt.n2(t.saved.threshold) + " (" + t.saved.source + ")" : "nothing")}));
    var btn = el("button", {type: "button", id: "t-save-one", text: "Save " + fmt.n2(st.t) + " for " + t.model + " on " + t.device});
    var why = el("p", {class: "small warn", id: "t-why"});
    if (!t.savable) { btn.disabled = true; why.textContent = t.notSavable || "This pick cannot be saved."; } else why.hidden = true;
    var conf = el("div", {class: "bn-confirm", role: "alertdialog", hidden: true});
    kids.push(el("div", {class: "actions"}, [btn]), why, conf);
    if (t.confirm && t.savable) kids.push(el("p", {class: "small muted", text: "Low confidence: saving asks you to confirm first."}));
    var devs = t.devices || [t.device], on = runsOn(t.model);
    var radios = el("div", {role: "radiogroup", "aria-label": "Run " + t.model + " on"});
    devs.forEach(function (d) {
      var r = el("input", {type: "radio", name: "t-runon", value: d}); r.checked = BC.sameDevice(d, on || t.device);
      r.addEventListener("change", function () { if (r.checked) { st.useDev[t.model] = d; renderAllTable(); } });
      radios.appendChild(el("label", {class: "check"}, [r, " ", el("code", {text: d}), BC.has(on) && BC.sameDevice(d, on) && !st.useDev[t.model] ? el("span", {class: "muted small", text: " in use"}) : null]));
    });
    kids.push(el("h3", {text: "Run " + t.model + " on"}), radios);
    kids.push(el("p", {class: "small muted", id: "t-bi"}));
    kids.push(resultBox);
    box.replaceChildren.apply(box, kids);
    updateSave();
    btn.addEventListener("click", function () {
      var pairs = [["profile", t.device + "|" + st.t.toFixed(2) + "|" + t.model]];
      var choice = st.useDev[t.model], now = configuredDevice(t.model);
      if (choice && !(now && BC.sameDevice(choice, now))) pairs.push(["use", choice + "|" + t.model]);
      if (t.confirm) {
        conf.hidden = false;
        conf.replaceChildren(el("p", {text: cap(t.lowConfidence || "This pick rests on little data.") + " Save " + fmt.n2(st.t) + " anyway?"}));
        var yes = el("button", {type: "button", text: "Save anyway"}), no = el("button", {type: "button", class: "secondary", text: "Cancel"});
        yes.addEventListener("click", function () { conf.hidden = true; post(pairs, "Saved.", resultBox, btn); });
        no.addEventListener("click", function () { conf.hidden = true; btn.focus(); });
        conf.appendChild(el("div", {class: "actions"}, [yes, no])); yes.focus();
        return;
      }
      post(pairs, "Saved.", resultBox, btn);
    });
  }

  // What follows the slider on the card: the button's value and the Blue Iris line.
  function updateSave() {
    var t = st.tune, btn = $("t-save-one"), bi = $("t-bi");
    if (!t || !btn) return;
    btn.textContent = "Save " + fmt.n2(st.t) + " for " + t.model + " on " + t.device;
    bi.textContent = "Blue Iris: set the cameras that use " + t.model + " to a minimum confidence of " + Math.round(st.t * 100) + "% or lower.";
  }

  // ---- all models and devices ------------------------------------------------------------------------
  function savableRow(r) {
    if (!BC.has(r.confidence)) return "no suggestion";
    if (r.flags && r.flags.relative) return "labels are another model's detections";
    if (r.flags && r.flags.flat) return "every confidence scores the same";
    if (r.flags && r.flags.unmet) return "target not met";
    if (r.flags && r.flags.trainedHere) return "trained on these frames";
    return null;
  }
  function renderAllTable() {
    var rows = (st.compare && st.compare.rows) || [], tbl = $("t-all");
    var head = el("tr", {}, [["Model", "model"], ["Device", "device"], ["Saved", "saved", 1], ["Suggested", "sug", 1], ["Goal now → new", "goal", 1], ["FA/100 now → new", "fa", 1], ["Run on", "-"]].map(function (h) {
      var c = el("th", {scope: "col", class: h[2] ? "num" : null, text: h[0], "data-sort-col": h[1]}); if (h[1] === "-") c.setAttribute("data-sort-off", ""); return c;
    }));
    var groups = {};
    var body = el("tbody", {}, rows.map(function (r) {
      var a = r.atSaved || {}, b = r.atSuggested || {}, no = savableRow(r);
      var name = el("span", {}, [el("button", {type: "button", class: "link-like", text: r.model})]);
      name.firstChild.addEventListener("click", function () { st.model = r.model; st.device = r.device; loadTune(); window.scrollTo({top: 0, behavior: window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"}); });
      BC.rowChips(r).filter(function (c) { return /low confidence|target not met|relative|flat|trained here|training unknown/.test(c.text); }).forEach(function (c) { var t = el("button", {type: "button", class: "term " + c.cls, "data-tip": c.tip, "aria-expanded": "false"}); t.textContent = c.text; name.appendChild(document.createTextNode(" ")); name.appendChild(t); });
      var radio = el("input", {type: "radio", name: "use-" + r.model, value: r.device, "aria-label": "Run " + r.model + " on " + r.device});
      var on = runsOn(r.model); radio.checked = on ? BC.sameDevice(on, r.device) : !groups[r.model];
      groups[r.model] = true;
      radio.addEventListener("change", function () { if (radio.checked) { st.useDev[r.model] = r.device; if (st.tune && st.tune.model === r.model) renderSave(); } });
      var current = st.tune && st.tune.model === r.model && BC.sameDevice(st.tune.device, r.device);
      return el("tr", {class: current ? "top" : null, "data-best": String(r.position || 0)}, [
        el("td", {class: "bn-lead rc-title", "data-label": "Model"}, [name]),
        el("td", {"data-label": "Device"}, [el("code", {text: r.device})]),
        el("td", {class: "num", "data-label": "Saved", text: BC.has(r.servedThreshold) ? fmt.n2(r.servedThreshold) : "-"}),
        el("td", {class: "num", "data-label": "Suggested", text: BC.has(r.confidence) ? fmt.n2(r.confidence) + (BC.rangeText(r.plateau) ? " (" + BC.rangeText(r.plateau) + ")" : "") + (no ? " · not savable: " + no : "") : "-"}),
        el("td", {class: "num", "data-label": "Goal now → new", text: fmt.n2(a.value) + " → " + fmt.n2(b.value)}),
        el("td", {class: "num", "data-label": "FA/100 now → new", text: fmt.n1(a.fa100) + " → " + fmt.n1(b.fa100)}),
        el("td", {"data-label": "Run on"}, [el("label", {class: "bn-radio-cell"}, [radio, el("span", {class: "sr-only", text: "Run " + r.model + " on " + r.device})])])
      ]);
    }));
    tbl.replaceChildren(el("thead", {}, [head]), body);
    if (window.PrismSort) PrismSort.apply(tbl);
  }
  function saveAll(useDevices, includeLow) {
    var rows = (st.compare && st.compare.rows) || [], pairs = [], skipped = [], low = [];
    rows.forEach(function (r) {
      var no = savableRow(r);
      if (no) { skipped.push(r.model + " on " + r.device + " (" + no + ")"); return; }
      if (r.flags && r.flags.lowConfidence && !includeLow) { low.push(r); return; }
      pairs.push(["profile", r.device + "|" + r.confidence.toFixed(2) + "|" + r.model]);
    });
    if (useDevices) {
      var models = {};
      rows.forEach(function (r) { models[r.model] = true; });
      Object.keys(models).forEach(function (m) {
        var radio = document.querySelector('#t-all input[type=radio][name="use-' + (window.CSS && CSS.escape ? CSS.escape(m) : m) + '"]:checked');
        if (radio) pairs.push(["use", radio.value + "|" + m]);
      });
    }
    return {pairs: pairs, skipped: skipped, low: low};
  }
  function doSaveAll(useDevices, button) {
    var box = $("t-result"), conf = $("t-all-confirm");
    if (tableSet() !== st.set) { say(box, "The table of these images is still loading: save once it shows.", "error"); return; }
    var plan = saveAll(useDevices, false);
    var go = function (includeLow) {
      conf.hidden = true;
      var p = saveAll(useDevices, includeLow);
      if (!p.pairs.length) { say(box, "Nothing to save: " + (p.skipped.join("; ") || "no suggestion") + ".", "error"); return; }
      post(p.pairs, "Saved.", box, button).then(function (j) {
        if (j && j.success && p.skipped.length) box.firstChild.appendChild(el("span", {class: "small", text: " Not saved: " + p.skipped.join("; ") + "."}));
      });
    };
    if (plan.low.length) {
      $("t-all-confirm-text").textContent = plan.low.length + " suggestion" + (plan.low.length === 1 ? " rests" : "s rest") + " on too little data to cross-validate (" + plan.low.map(function (r) { return r.model + " on " + r.device; }).join(", ") + "). Save " + (plan.low.length === 1 ? "it" : "them") + " too?";
      conf.hidden = false;
      $("t-all-yes").onclick = function () { go(true); };
      $("t-all-skip").onclick = function () { go(false); };
      $("t-all-no").onclick = function () { conf.hidden = true; button.focus(); };
      $("t-all-yes").focus();
      return;
    }
    go(false);
  }
  $("t-save-all").addEventListener("click", function () { doSaveAll(true, $("t-save-all")); });
  $("t-save-conf").addEventListener("click", function () { doSaveAll(false, $("t-save-conf")); });

  // ---- the whole view ---------------------------------------------------------------------------------
  function renderTune() { if (!st.tune) return; slider.min = String(Math.round(st.tune.grid[0].threshold * 100)); slider.max = String(Math.round(st.tune.grid[st.tune.grid.length - 1].threshold * 100)); drawChart(); setT(st.t, false); renderSave(); }
  function renderAll() { renderSelects(); renderAllTable(); }
  function refresh() {
    // Saved values changed: the tune view's saved line and the table follow.
    loadSet(false);
    PB.getJSON(api("/v1/benchmark/tune", {model: st.model, device: st.device, set: st.set})).then(function (j) {
      // Only onto the same combo and set, and never over a choice still loading.
      if (j.success && st.tune && !tuneLoads.busy() && j.tune.model === st.tune.model && j.tune.device === st.tune.device && j.tune.set === st.tune.set) { st.tune = j.tune; drawChart(); setT(st.t, false); renderSave(); }
    });
  }
  if (window.PrismLive) window.PrismLive.onConfig(function (p) {
    var first = !st.cfg; st.cfg = p;
    if (!first) return;
    if (st.tune) renderSave();
    if (st.compare) renderAllTable();
  });
  var rt = null, lastW = 0;
  window.addEventListener("resize", function () {
    clearTimeout(rt);
    rt = setTimeout(function () { var w = $("t-chart").clientWidth; if (st.tune && Math.abs(w - lastW) > 8) { lastW = w; drawChart(); } }, 150);
  });
  S.onView(function (why) { if (why === "run" || why === "rescore") loadSet(true); });
  if (!bodyEl.hidden) start();
})();
