// Blue Onyx Prism: Benchmark page (served from /static/; no inline scripts under the CSP).
// Overview (what to use, what changed), Confidence, Details, Run and the page wiring. The
// vocabulary and tooltips are in bench-glossary.js, the charts in bench-charts.js and the
// Compare tab in bench-compare.js, shared through window.PrismBench (loaded before this file).
(function () {
  var PB = window.PrismBench, C = PB.charts, F = PB.fmt, has = PB.has;
  var $ = function (id) { return document.getElementById(id); };
  var form = $("bench-form"), msg = $("msg"), bar = $("bar"), ptext = $("ptext");
  var polling = null, lastState = null, lastConfig = null, lastModels = [], lastStatus = null;
  // Old links: #tab-results is the Overview now, #tab-models the Details tab. Rewritten before
  // the tabs read the URL (and on later hash changes, before their listener runs).
  var ALIAS = {"#tab-results": "#tab-overview", "#tab-models": "#tab-details"};
  function alias() { var h = ALIAS[location.hash]; if (!h) return; if (history.replaceState) history.replaceState(null, "", h); else location.hash = h; }
  alias(); window.addEventListener("hashchange", alias);
  // The forms are driven by their buttons; Enter in a field must not navigate.
  ["bench-form", "search-form"].forEach(function (id) { var f = $(id); if (f) f.addEventListener("submit", function (e) { e.preventDefault(); }); });
  var el = PB.el;
  function pct(v) { return v === null || v === undefined ? "-" : (v * 100).toFixed(1); }
  function ms(v) { return v === null || v === undefined ? "-" : v.toFixed(1); }
  function gradeEl(g, rel) { var s = el("span", "grade grade-" + (g || "none"), (g || "-") + (rel ? "*" : "")); if (rel) s.title = "relative to a reference model, not ground truth"; return s; }
  var post = PB.post;
  function formBody(extra) {
    var b = new URLSearchParams();
    // Hidden "0" before its checkbox: an unchecked box still sends its value (the last one wins).
    form.querySelectorAll("input[type=hidden][name]").forEach(function (h) { b.append(h.name, h.value); });
    form.querySelectorAll("input[type=checkbox]:checked").forEach(function (c) { b.append(c.name, c.value); });
    ["max_images", "repeat", "warmup", "accuracy_weight", "threshold_objective", "accuracy_metric"].forEach(function (n) { b.append(n, form.elements[n].value); });
    b.append("reference_model", form.elements.reference_model.value);
    var f = $("add-folder").value.trim();
    if (f) b.append("dataset", f.indexOf("dir:") === 0 ? f : "dir:" + f);
    if (extra) Object.keys(extra).forEach(function (k) { b.append(k, extra[k]); });
    return b;
  }
  // The body of POST /v1/benchmark/settings: the Run form ("Save as default" sends the same).
  function settingsBody() {
    var b = formBody(); b.delete("dataset");
    form.querySelectorAll("input[name=dataset]:checked").forEach(function (c) { b.append("dataset", c.value); });
    var f = $("add-folder").value.trim(); if (f) b.append("add_folder", f);
    return b;
  }
  var runMsg = $("run-msg");
  $("start").addEventListener("click", function () {
    msg.textContent = "starting…"; runMsg.textContent = "";
    post("/v1/benchmark", formBody()).then(function (j) { msg.textContent = j.message; if (j.success) poll(); else runMsg.textContent = j.message; });
  });
  $("cancel").addEventListener("click", function () { post("/v1/benchmark/cancel", "").then(function (j) { msg.textContent = j.message; }); });
  $("save-defaults").addEventListener("click", function () {
    var f = $("add-folder").value.trim();
    post("/v1/benchmark/settings", settingsBody()).then(function (j) { runMsg.textContent = j.message; if (j.success && f) L.reloadKeep(); else if (j.success) refresh(); });
  });
  form.querySelectorAll("[data-download]").forEach(function (b) {
    b.addEventListener("click", function () {
      var id = b.dataset.download, body = new URLSearchParams({id: id});
      b.disabled = true;
      post("/v1/resources/download", body).then(function (j) {
        runMsg.textContent = j.message;
        if (!j.success) { b.disabled = false; return; }
        // Follow the download in the dataset's state text, then reload in place once it is done.
        var row = b.closest("label"), state = row && row.querySelector(".ds-state");
        (function tick() {
          fetch("/v1/resources", {cache: "no-store"}).then(function (r) { return r.json(); }).then(function (j2) {
            var r = (j2.resources || []).find(function (x) { return x.id === id; });
            if (r && state) state.textContent = r.state_text;
            if (r && r.busy) setTimeout(tick, 1500); else L.reloadKeep();
          }).catch(function () { setTimeout(tick, 3000); });
        })();
      });
    });
  });
  form.querySelectorAll("[data-remove-folder]").forEach(function (b) {
    b.addEventListener("click", function () {
      var v = b.dataset.removeFolder, body = formBody(); body.delete("dataset");
      form.querySelectorAll("input[name=dataset]:checked").forEach(function (c) { if (c.value !== v) body.append("dataset", c.value); });
      post("/v1/benchmark/settings", body).then(function (j) { runMsg.textContent = j.message; if (j.success) L.reloadKeep(); });
    });
  });

  // Sortable tables: click a header; numeric when the cell has data-v. The sort survives
  // re-renders (PrismLive.keep / patch carry it over by the table's key). A term tooltip
  // button inside a header does not sort (bench-glossary.js stops its events).
  var L = window.PrismLive, T = L && L.tabs(document.querySelector("nav.tabs"));
  function sortable(table) {
    table.querySelectorAll("th").forEach(function (th, i) {
      th.classList.add("sortable"); th.tabIndex = 0; th.setAttribute("role", "button");
      if (th.textContent) th.title = "Sort by " + th.textContent;
      var sort = function () { L.sortTable(table, i, th.dataset.dir !== "asc"); };
      th.addEventListener("click", sort);
      th.addEventListener("keydown", function (e) { if (e.target !== th) return; if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sort(); } });
    });
    return table;
  }
  PB.sortable = sortable;
  var when = F.when;
  // Headers: plain text, or "@id" for a LABELS column with its glossary tooltip.
  function table(cls, head) {
    var t = el("table", "bench " + (cls || "")), tr = el("tr"); var th = el("thead");
    head.forEach(function (h) { var c = el("th"); if (typeof h === "string" && h.charAt(0) === "@") c.appendChild(PB.head(h.slice(1))); else c.textContent = h; tr.appendChild(c); }); th.appendChild(tr); t.appendChild(th); t.appendChild(el("tbody")); return t;
  }
  function cell(tr, text, v, label) { var c = el("td", v !== undefined ? "num" : null, text); if (v !== undefined) c.dataset.v = v === null ? "" : v; if (label) c.dataset.label = label; tr.appendChild(c); return c; }
  // A horizontally scrolling wrapper; `key` names it (and its table) so its scroll position
  // and sort order survive re-renders.
  function wrap(t, key) { var d = el("div", "scroll"); if (key) { d.dataset.keep = "w:" + key; t.dataset.keep = "t:" + key; } d.appendChild(t); return d; }
  // A <details> whose open state survives re-renders.
  function det(key, summary, open) { var d = el("details", "bd"); d.dataset.keep = "d:" + key; if (open) d.open = true; d.appendChild(el("summary", null, summary)); return d; }
  function near(x, y) { return x !== null && x !== undefined && y !== null && y !== undefined && Math.abs(x - y) < 1e-6; }
  // The threshold configured now for an advice (the server fills `now` from the live config;
  // results without it fall back to the run's threshold): {threshold, running, point, source}.
  function nowOf(a) { return a.now || {threshold: a.configured.threshold, running: null, point: a.configured, source: "run"}; }
  function isRunning(n) { return n.running !== null && n.running !== undefined && !near(n.running, n.threshold); }
  // "0.35" or "0.35 (configured; running 0.50 until restart)".
  function nowText(n) { return thr(n.threshold) + (isRunning(n) ? " (configured; running " + thr(n.running) + " until restart)" : ""); }
  // A group's counts at the threshold configured now (null: not evaluated).
  function groupNow(a, g) { return a.now ? (g.now || null) : g.configured; }
  var NOT_EVALUATED = "not evaluated at this threshold: the stored predictions (benchmark-preds.json) of this run are missing";

  function verdictText(a) {
    if (!a) return "-";
    var total = a.matched + Math.max(a.only_device, a.only_reference);
    return {reference: "reference", agrees: "agrees", minor: "minor (" + a.matched + "/" + total + ")", disagrees: "DISAGREES (" + a.matched + "/" + total + ")"}[a.verdict] || a.verdict;
  }

  // ---- apply, with before -> after and Undo ---------------------------------------------------
  // Each apply answers {changes, previous}; one message lists every change of one action (a
  // device and its confidence together, or every model) and one Undo reverts them all.
  function applyDevice(model, device) { return post("/v1/benchmark/apply", new URLSearchParams({model: model, device: device})); }
  function applyThreshold(model, t) { return post("/v1/benchmark/apply-threshold", new URLSearchParams({model: model, threshold: typeof t === "number" ? t.toFixed(2) : String(t)})); }
  function useDevice(model, device) { return applyDevice(model, device).then(function (j) { showApply([{j: j, kind: "device"}]); }); }
  function useThreshold(model, t) { return applyThreshold(model, t).then(function (j) { showApply([{j: j, kind: "threshold"}]); }); }
  // A device and that device's best confidence together (the confidence only when the device
  // was saved; `t` null = the device only).
  function usePair(model, device, t) {
    var steps = [];
    // Applying a pair is also that model's device pick (step 2).
    choose(model, device); redrawSelection();
    return applyDevice(model, device).then(function (j) {
      steps.push({j: j, kind: "device"});
      if (!j.success || !has(t)) return null;
      return applyThreshold(model, t).then(function (k) { steps.push({j: k, kind: "threshold"}); });
    }).then(function () { showApply(steps); });
  }
  // The value before an apply: the answer's `previous` (newer servers), else the change's `from`
  // (null = the global setting: "global" for a device, the global value for a threshold).
  function previousOf(j, c, kind) {
    var p = j.previous && Object.prototype.hasOwnProperty.call(j.previous, c.model) ? j.previous[c.model] : undefined;
    if (p === undefined) p = c.from;
    // Newer servers answer "global" (follow the global setting) or "0.40"; both post back as is.
    if (kind === "threshold") { if (p === null || p === undefined) p = c.global; return typeof p === "number" ? p.toFixed(2) : String(p); }
    return p === null || p === undefined ? "global" : p;
  }
  function describeChange(c, kind) {
    if (kind === "threshold") {
      var from = has(c.from) ? F.thrBi(c.from) : (has(c.global) ? F.thrBi(c.global) + " (global)" : "global");
      return c.model + ": confidence " + from + " → " + F.thrBi(c.to);
    }
    return c.model + ": device " + (c.from || "global") + " → " + (c.to || "global");
  }
  // steps: [{j, kind: "device" | "threshold"}] (or the old (j, kind, undone) form).
  function showApply(steps, kind, undone) {
    if (!Array.isArray(steps)) steps = [{j: steps, kind: kind}]; else undone = kind;
    var box = $("apply-msg"), ok = steps.every(function (s) { return s.j && s.j.success; });
    box.hidden = false; box.className = ok ? "notice apply-box" : "error apply-box"; box.replaceChildren();
    var all = [];
    steps.forEach(function (s) { (s.j.changes || []).forEach(function (c) { if (c && c.model) all.push({c: c, kind: s.kind, j: s.j}); }); });
    if (all.length) {
      box.appendChild(el("strong", null, undone ? "Undone: " : "Applied now (no restart): "));
      var ul = el("ul", "apply-list"); all.forEach(function (x) { ul.appendChild(el("li", null, describeChange(x.c, x.kind))); }); box.appendChild(ul);
    }
    steps.forEach(function (s) { if (s.j.message) box.appendChild(el("span", "muted", s.j.message + " ")); });
    if (all.length && !undone) {
      var u = el("button", "small secondary", "Undo"); u.type = "button";
      u.addEventListener("click", function () { u.disabled = true; undoAll(all); });
      box.appendChild(u);
    }
    if (steps.some(function (s) { return s.j.success && /Restart the server to apply/.test(s.j.message || "") && !s.j.restarting; })) {
      var b = el("button", "small danger", "Restart now"); b.type = "button";
      b.addEventListener("click", restart); box.appendChild(b);
    }
    if (steps.some(function (s) { return s.j.restarting; })) waitForRestart();
    if (window.PrismLive) PrismLive.poll();
    refresh();
    refreshSearch();
  }
  // Revert every change of one action: the thresholds in one request, then each device.
  function undoAll(all) {
    var steps = [], thrs = all.filter(function (x) { return x.kind === "threshold"; }), devs = all.filter(function (x) { return x.kind === "device"; });
    var p = Promise.resolve();
    if (thrs.length) p = p.then(function () {
      var b = new URLSearchParams(); thrs.forEach(function (x) { b.append("model", x.c.model); b.append("threshold", previousOf(x.j, x.c, "threshold")); });
      return post("/v1/benchmark/apply-threshold", b).then(function (k) { steps.push({j: k, kind: "threshold"}); });
    });
    devs.forEach(function (x) { p = p.then(function () { return applyDevice(x.c.model, previousOf(x.j, x.c, "device")).then(function (k) { steps.push({j: k, kind: "device"}); }); }); });
    return p.then(function () { showApply(steps, true); });
  }
  function restart() { L.restarting(); fetch("/config/restart", {method: "POST"}).then(waitForRestart, waitForRestart); }
  // Reload in place (same scroll position) once the next generation answers.
  function waitForRestart() {
    msg.textContent = "restarting…";
    L.restarting();
    L.afterRestart();
  }
  $("apply-all-thr").addEventListener("click", function () {
    if (!confirm("Write the best confidence threshold of every benchmarked model into the config? It applies at once.")) return;
    post("/v1/benchmark/apply-threshold", new URLSearchParams({all: "1"})).then(function (j) { showApply(j, "threshold"); });
  });
  function thr(v) { return v === null || v === undefined ? "-" : v.toFixed(2); }
  function applicable(a) { return a && a.best && a.best.met && a.best.point.tp > 0; }
  // A best pick at which the model finds no object: no confidence setting helps.
  function findsNothing(b) { return !!(b && b.point && (b.point.tp === 0 || b.point.recall === 0)); }
  var FINDS_NOTHING = "This model finds no objects on these images, so no confidence setting helps; check the model, its classes or the dataset.";
  // Models whose stored advice was found for another objective than the configured one.
  function staleAdvice(j, models) {
    var cfg = String((j && j.config && j.config.benchmark && j.config.benchmark.threshold_objective) || "f1");
    return models.filter(function (m) { return m.threshold && m.threshold.objective && String(m.threshold.objective) !== cfg; });
  }
  function staleNotice(j, models) {
    var st = staleAdvice(j, models); if (!st.length) return null;
    var cfg = String(j.config.benchmark.threshold_objective || "f1"), objs = [];
    st.forEach(function (m) { var o = String(m.threshold.objective); if (objs.indexOf(o) < 0) objs.push(o); });
    var p = el("div", "warn stale-notice");
    p.appendChild(document.createTextNode("The saved results were found for " + objs.map(PB.objectiveName).join(", ") + " (" + st.length + " model" + (st.length === 1 ? "" : "s") + "); your setting is " + PB.objectiveName(cfg) + ". The confidences shown here are searched for your setting; values marked “found for” could not be. Save and re-rank to store them and re-grade (no new run). "));
    var b = el("button", "small", "Save and re-rank"); b.type = "button";
    b.disabled = !!j.running;
    b.addEventListener("click", function () { b.disabled = true; presetUi.msg = "saving\u2026"; saveSettings({threshold_objective: cfg}); });
    p.appendChild(b); return p;
  }

  // ---- threshold detail (Details cards, Confidence > More detail) ------------------------------
  // Frame-level rates of a threshold point ("TPR / FPR" in %).
  function tf(p) { return p && p.frame ? pct(p.frame.tpr) + " / " + pct(p.frame.fpr) : "-"; }
  function rocWrap(o) { var w = el("div", "thr-chart-wrap roc-chart-wrap"); w.appendChild(C.rocSvg(o)); return w; }
  // Per-class frame ROC table (frames with / without the class, AUC).
  function rocTable(roc) {
    var t = sortable(table("", ["Class", "Frames with", "Frames without", "@roc_auc"]));
    (roc.per_class || []).forEach(function (c) { var tr = el("tr"); cell(tr, c["class"]); cell(tr, c.positives, c.positives); cell(tr, c.negatives, c.negatives); cell(tr, pct(c.auc), c.auc === null ? null : -c.auc); t.tBodies[0].appendChild(tr); });
    var tr = el("tr"); cell(tr, "macro (micro " + pct(roc.micro_auc) + ")"); cell(tr, roc.positives, roc.positives); cell(tr, roc.negatives, roc.negatives); cell(tr, pct(roc.auc), roc.auc === null ? null : -roc.auc); t.tBodies[0].appendChild(tr);
    return t;
  }
  function thresholdChart(m, a) {
    var best = a.best, n = nowOf(a), np = n.point || {};
    return C.chartSvg({
      key: "thr:" + m.model.toLowerCase(),
      title: m.model + ": precision, recall and F1 vs confidence threshold on " + a.device,
      desc: "Best threshold " + (best ? thr(best.threshold) + " (" + a.objective + "): precision " + pct(best.point.precision) + "%, recall " + pct(best.point.recall) + "%, F1 " + pct(best.point.f1) + "%" : "none") + ". Configured " + nowText(n) + ": precision " + pct(np.precision) + "%, recall " + pct(np.recall) + "%, F1 " + pct(np.f1) + "%.",
      points: a.exact && a.exact.length > 1 ? a.exact : a.curve,
      configured: n.threshold, best: best ? best.threshold : null, running: isRunning(n) ? n.running : null, f2: a.objective === "f2"
    });
  }
  // Apply button for the best threshold, or "in use" when the config already has it.
  function thrAction(model, a, label) {
    var b = a.best, n = nowOf(a);
    if (!applicable(a)) return null;
    if (!near(n.threshold, b.threshold)) { var u = el("button", "small", label + thr(b.threshold)); u.type = "button"; u.addEventListener("click", function () { useThreshold(model, b.threshold); }); return u; }
    return el("span", "muted", isRunning(n) ? "in config (restart to use)" : "in use");
  }
  function thresholdBlock(m) {
    var a = m.threshold, box = el("div", "thr-block"), key = m.model.toLowerCase();
    if (!a) return box;
    box.appendChild(el("h3", null, "Confidence threshold"));
    var b = a.best, n = nowOf(a), np = n.point, p = el("p", b && b.met ? "notice" : "warn");
    if (b) {
      var cfgText = near(n.threshold, b.threshold) ? "the configured threshold" + (isRunning(n) ? "; running " + thr(n.running) + " until restart" : "")
        : "configured " + nowText(n) + ": " + (np ? "P " + pct(np.precision) + " R " + pct(np.recall) + " F1 " + pct(np.f1) + (np.frame ? ", frame TPR / FPR " + tf(np) : "") : "not evaluated");
      p.textContent = "Best " + F.thrBi(b.threshold) + " for " + a.objective + " on " + a.device + ": P " + pct(b.point.precision) + " R " + pct(b.point.recall) + " F1 " + pct(b.point.f1) + (b.point.frame ? ", frame TPR / FPR " + tf(b.point) : "") + " (" + cfgText + ")" + (b.note ? ". " + b.note : "") + (b.plateau && b.exact_threshold !== null && b.exact_threshold !== undefined ? ". Exact optimum at confidence " + b.exact_threshold.toFixed(3) + "; near-optimal range " + b.plateau[0].toFixed(3) + "–" + b.plateau[1].toFixed(3) + ", rounded down to 0.01." : ".");
    } else p.textContent = "No ground truth for a threshold curve.";
    box.appendChild(p);
    var row = el("div", "chart-row"), wrapC = el("div", "thr-chart-wrap"); wrapC.appendChild(thresholdChart(m, a)); row.appendChild(wrapC);
    if (a.roc && a.roc.curve && a.roc.curve.length) row.appendChild(rocWrap({
      key: "roc:" + key,
      title: m.model + ": frame-level ROC on " + a.device,
      desc: "Frame ROC AUC " + pct(a.roc.auc) + "% (macro over classes). Configured " + thr(n.threshold) + ": TPR / FPR " + tf(np) + "%" + (b ? "; best " + thr(b.threshold) + ": " + tf(b.point) + "%" : "") + ".",
      curve: a.roc.curve, auc: a.roc.auc, configured: np ? Object.assign({}, np, {threshold: n.threshold}) : null, best: b ? b.point : null
    }));
    box.appendChild(row);
    if (a.roc && a.roc.per_class && a.roc.per_class.length) {
      var rd = det("roc:" + key, "Frame-level ROC AUC by class (macro " + pct(a.roc.auc) + ", micro " + pct(a.roc.micro_auc) + ")"); rd.appendChild(wrap(rocTable(a.roc), "roc:" + key)); box.appendChild(rd);
    }
    var act = thrAction(m.model, a, "Apply threshold "); if (act && act.tagName === "BUTTON") box.appendChild(act);
    var groups = (a.by_dataset || []).concat(a.by_tag || [], (a.per_class || []).map(function (g) { return Object.assign({}, g, {key: "class " + g.key}); }), (a.by_device || []).map(function (d) { return {key: "device " + d.device, best: d.best, configured: null, now: null}; }));
    if (groups.length) {
      var t = sortable(table("", ["Group", "Images", "Objects", "Best", "P", "R", "F1", "F1 at configured " + thr(n.threshold), "ROC AUC"]));
      groups.forEach(function (g) {
        var tr = el("tr"), gb = g.best, gn = g.configured === null ? null : groupNow(a, g); cell(tr, g.key + (g.relative ? " (relative)" : "")); cell(tr, g.images === undefined ? "-" : g.images, g.images); cell(tr, g.gt === undefined ? "-" : g.gt, g.gt);
        cell(tr, gb ? thr(gb.threshold) + (gb.met ? "" : " !") : "-", gb ? gb.threshold : null);
        [gb && gb.point.precision, gb && gb.point.recall, gb && gb.point.f1, gn && gn.f1, g.roc_auc].forEach(function (v) { cell(tr, pct(v), v === null || v === undefined ? null : -v); });
        t.tBodies[0].appendChild(tr);
      });
      var d = det("thrg:" + key, "Best threshold by dataset, tag, class and device"); d.appendChild(wrap(t, "thrg:" + key)); box.appendChild(d);
    }
    return box;
  }
  // Columns hidden until "More columns" (the table keeps them for sorting and copying).
  var thrExpanded = false;
  function markExtra(t, idx) {
    var rows = [t.tHead.rows[0]].concat(Array.prototype.slice.call(t.tBodies[0].rows));
    rows.forEach(function (r) { idx.forEach(function (i) { if (r.cells[i]) r.cells[i].classList.add("extra"); }); });
    t.classList.toggle("compact", !thrExpanded);
    return t;
  }
  function moreColumns(t, extraLabel) {
    var tools = el("div", "table-tools"), b = el("button", "small secondary", thrExpanded ? "Fewer columns" : "More columns"); b.type = "button";
    b.addEventListener("click", function () { thrExpanded = !thrExpanded; t.classList.toggle("compact", !thrExpanded); b.textContent = thrExpanded ? "Fewer columns" : "More columns"; });
    tools.appendChild(b); tools.appendChild(el("span", "muted", extraLabel)); return tools;
  }
  function renderThresholds(models) {
    var box = $("thresholds"); box.replaceChildren();
    var rows = models.filter(function (m) { return m.threshold; });
    if (!rows.length) { box.appendChild(el("h2", null, "Best confidence threshold")); box.appendChild(el("p", "empty", "No threshold results yet: a benchmark with ground truth (or a reference model) fills this table.")); return; }
    box.appendChild(el("h2", null, "Best confidence threshold"));
    box.appendChild(el("p", "muted", "Exact best threshold per model on its recommended device, compared with the threshold configured now. Gain is the objective's value at the best minus at the configured threshold, in points."));
    box.appendChild(el("details", "help")); var hd = box.lastChild; hd.appendChild(el("summary", null, "How to read it")); hd.appendChild(el("p", null, "The best threshold is the middle of the widest near-optimal range, rounded down to 0.01; P / R / F1 at the configured threshold are exact, from the stored predictions. F2 favors recall; precision:0.9 limits false alerts; youden maximizes the frame alert rate minus the frame false-alert rate. Frame TPR / FPR: alert rate on frames with an object / on frames without one (a frame alerts when its top confidence reaches the threshold). Blue Iris may send its own min_confidence per request (it then replaces the server threshold) and each camera's minimum confidence filters again."));
    var t = sortable(table("", ["Model", "@objective", "Now", "@precision", "@recall", "@f1", "Best", "@precision", "@recall", "@f1", "Gain", "F2 pick", "P≥90% pick", "Youden pick", "@roc_auc", "TPR / FPR now", "TPR / FPR best", ""]));
    rows.forEach(function (m) {
      var a = m.threshold, b = a.best, tr = el("tr"), n = nowOf(a), np = n.point;
      tr.dataset.model = m.model;
      var alt = function (o) { var x = (a.alternatives || []).find(function (p) { return p.objective === o; }); return x ? thr(x.threshold) + (x.met ? "" : " !") : "-"; };
      cell(tr, m.model + (a.relative ? " *" : "")); cell(tr, a.objective);
      var nc = cell(tr, thr(n.threshold), n.threshold); nc.classList.add("thr-now");
      if (isRunning(n)) nc.appendChild(el("span", "muted thr-run", "(configured; running " + thr(n.running) + " until restart)"));
      nc.title = isRunning(n) ? "Configured " + thr(n.threshold) + "; the loaded model runs with " + thr(n.running) + " until a restart" : "The model's confidence threshold in the config now";
      [np && np.precision, np && np.recall, np && np.f1].forEach(function (v) { var x = cell(tr, pct(v), v === null || v === undefined ? null : -v); if (!np) x.title = NOT_EVALUATED; });
      var bc = cell(tr, b ? thr(b.threshold) : "-", b ? b.threshold : null); if (applicable(a)) bc.appendChild(el("span", "tag rec", "best"));
      [b && b.point.precision, b && b.point.recall, b && b.point.f1].forEach(function (v) { cell(tr, pct(v), v === null || v === undefined ? null : -v); });
      var gain = null;
      if (b && near(n.threshold, b.threshold)) gain = 0;
      else { var sb = b && objScore(a.objective, b.point), sc = objScore(a.objective, np); if (sb !== null && sb !== undefined && sc !== null && sc !== undefined) gain = (sb - sc) * 100; }
      var gc = cell(tr, gain === null ? "-" : (gain >= 0 ? "+" : "") + gain.toFixed(1) + " " + gainLabel(a.objective), gain === null ? null : -gain);
      gc.title = gainLabel(a.objective) + " at the best minus at the configured threshold (" + thr(n.threshold) + "), in points";
      cell(tr, alt("f2")); cell(tr, alt("precision:0.9")); cell(tr, alt("youden"));
      var auc = a.roc ? a.roc.auc : null; cell(tr, pct(auc), auc === null ? null : -auc);
      cell(tr, tf(np)); cell(tr, tf(b && b.point));
      var act = el("td", "actions-cell"), x = thrAction(m.model, a, "Apply "); if (x) act.appendChild(x);
      tr.appendChild(act);
      t.tBodies[0].appendChild(tr);
    });
    markExtra(t, [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16]);
    box.appendChild(moreColumns(t, "precision / recall / F1 at both thresholds, the F2, P≥90% and Youden picks, frame TPR / FPR"));
    box.appendChild(PB.tableFilter(t, "thr"));
    box.appendChild(wrap(t, "thr"));
  }

  // ---- confidence search over the stored predictions -----------------------------------------
  var search = {data: null, opts: null, thresholds: {}, selected: null, body: null};
  // Checkboxes for `values`; a refresh keeps what the user checked (new values get `checked`).
  function checks(box, name, values, checked) {
    var prev = {}; box.querySelectorAll("input[type=checkbox]").forEach(function (i) { prev[i.value] = i.checked; });
    box.replaceChildren();
    values.forEach(function (v) {
      var l = el("label", "check"), i = el("input"); i.type = "checkbox"; i.name = name; i.value = v; i.checked = v in prev ? prev[v] : checked(v);
      l.appendChild(i); l.append(" " + v); box.appendChild(l);
    });
    if (!values.length) box.appendChild(el("p", "muted", "none"));
  }
  function loadSearch() {
    return fetch("/v1/benchmark/threshold-search", {cache: "no-store"}).then(function (r) { return r.json(); }).then(function (j) {
      if (!j.success || j.available === false) { $("s-empty").hidden = false; $("s-empty").textContent = j.message; $("search-form").hidden = true; return; }
      $("s-empty").hidden = true; $("search-form").hidden = false;
      var first = !search.opts;
      search.opts = j.options; search.thresholds = j.thresholds || {};
      var ms = j.options.models.map(function (m) { return m.model; });
      checks($("s-models"), "s-model", ms, function () { return true; });
      var devs = []; j.options.models.forEach(function (m) { m.devices.forEach(function (d) { if (devs.indexOf(d) < 0) devs.push(d); }); });
      var sel = $("s-device"), keepDev = sel.value; sel.querySelectorAll("option:not(:first-child)").forEach(function (o) { o.remove(); });
      devs.forEach(function (d) { var o = el("option", null, d); o.value = d; sel.appendChild(o); });
      if (devs.indexOf(keepDev) >= 0) sel.value = keepDev;
      checks($("s-datasets"), "s-dataset", j.options.datasets, function () { return true; });
      checks($("s-tags"), "s-tag", j.options.tags.filter(function (t) { return t.indexOf("res:") !== 0; }), function () { return false; });
      checks($("s-classes"), "s-class", j.options.classes, function () { return false; });
      if (!first) return; // a refresh keeps the user's objective
      var obj = String(j.objective || "f1"), kind = obj.split(":")[0];
      $("s-objective").value = kind; if (obj.indexOf(":") > 0) $("s-target").value = obj.split(":")[1];
      $("s-target-label").hidden = !hasTarget(kind);
    }).catch(function () { /* older server or no results: the card keeps its empty state */ });
  }
  // The config changed (apply, another tab): refresh the options and redo the last search, so
  // its "configured" columns are at the threshold configured now.
  function refreshSearch() {
    return loadSearch().then(function () { if (search.body) return runSearch(search.body, true); });
  }
  function hasTarget(k) { return k === "precision" || k === "recall" || k === "fpr"; }
  $("s-objective").addEventListener("change", function () {
    var k = $("s-objective").value, t = $("s-target"), v = parseFloat(t.value);
    $("s-target-label").hidden = !hasTarget(k);
    // Sensible starting targets: few false alerts for fpr, a high bar for precision / recall.
    if (k === "fpr" && !(v < 0.5)) t.value = "0.05";
    if ((k === "precision" || k === "recall") && !(v >= 0.5)) t.value = "0.9";
  });
  // The value an objective maximizes, for its gain column.
  function objScore(obj, p) {
    if (!p) return null;
    var k = obj.split(":")[0], t = parseFloat(obj.split(":")[1]);
    if (k === "f2") return p.f2;
    if (k === "precision") return p.precision !== null && p.precision >= t - 1e-12 ? p.recall : null;
    if (k === "recall") return p.recall !== null && p.recall >= t - 1e-12 ? p.precision : null;
    if (k === "youden") return p.frame && p.frame.tpr !== null && p.frame.fpr !== null ? p.frame.tpr - p.frame.fpr : null;
    if (k === "fpr") return p.frame && p.frame.fpr !== null && p.frame.fpr <= t + 1e-12 ? p.frame.tpr : null;
    return p.f1;
  }
  function gainLabel(obj) { var k = obj.split(":")[0]; return k === "f2" ? "F2" : k === "precision" ? "recall" : k === "recall" ? "precision" : k === "youden" ? "J" : k === "fpr" ? "frame TPR" : "F1"; }
  function searchBody() {
    var b = new URLSearchParams(), checked = function (n) { return Array.prototype.map.call(document.querySelectorAll("input[name=" + n + "]:checked"), function (c) { return c.value; }); };
    var ms = checked("s-model"); if (!ms.length) ms = ["all"]; ms.forEach(function (m) { b.append("model", m); });
    if ($("s-device").value) b.append("device", $("s-device").value);
    checked("s-dataset").forEach(function (d) { b.append("dataset", d); });
    checked("s-tag").forEach(function (t) { b.append("tag", t); });
    checked("s-class").forEach(function (c) { b.append("class", c); });
    var k = $("s-objective").value; b.append("objective", hasTarget(k) ? k + ":" + $("s-target").value : k);
    b.append("iou", $("s-iou").value);
    return b;
  }
  function runSearch(body, quiet) {
    if (!quiet) $("s-msg").textContent = "searching…";
    search.body = body.toString();
    return post("/v1/benchmark/threshold-search", new URLSearchParams(search.body)).then(function (j) {
      if (!j.success) { $("s-msg").textContent = j.message; return; }
      search.data = j; if (!search.selected || !j.results.some(function (r) { return r.model === search.selected; })) search.selected = j.results[0].model;
      $("s-msg").textContent = j.results.length + " model(s) in " + j.elapsedMs.toFixed(1) + " ms" + (j.errors.length ? " · " + j.errors.join("; ") : "");
      drawSearch();
    });
  }
  $("s-run").addEventListener("click", function () { runSearch(searchBody()); });
  function applicableSearch(r) { return r.best && r.best.met && r.best.point.tp > 0; }
  $("s-apply-all").addEventListener("click", function () {
    var rs = search.data.results.filter(applicableSearch); if (!rs.length) return;
    if (!confirm("Write the searched best threshold of " + rs.length + " model(s) into the config? It applies at once.")) return;
    var b = new URLSearchParams(); rs.forEach(function (r) { b.append("model", r.model); b.append("threshold", r.best.threshold.toFixed(2)); });
    post("/v1/benchmark/apply-threshold", b).then(function (j) { showApply(j, "threshold"); });
  });
  $("s-csv-copy").addEventListener("click", function () {
    if (navigator.clipboard) navigator.clipboard.writeText(search.data.csv).then(function () { $("s-msg").textContent = "CSV copied"; }, function () { $("s-msg").textContent = "copy failed; use Download CSV"; });
  });
  $("s-csv-dl").addEventListener("click", function () {
    var a = el("a"); a.href = URL.createObjectURL(new Blob([search.data.csv], {type: "text/csv"})); a.download = "confidence-search.csv"; document.body.appendChild(a); a.click(); a.remove();
  });
  // The loaded model's threshold of a search result while a restart is pending (else null).
  function searchRunning(r) {
    var c = lastConfig && (lastConfig.models || []).find(function (x) { return x.name.toLowerCase() === r.model.toLowerCase(); });
    var t = c ? c.runningThreshold : null;
    return t !== null && t !== undefined && r.now && !near(t, r.now.threshold) ? t : null;
  }
  function drawSearch() { L.keep($("s-results"), undefined, drawSearchNow); }
  function drawSearchNow() {
    var j = search.data, box = $("s-results"); box.replaceChildren();
    $("s-csv-copy").hidden = $("s-csv-dl").hidden = !j;
    $("s-apply-all").hidden = !j || !j.results.some(applicableSearch);
    if (!j) return;
    var obj = j.results.length ? j.results[0].objective : "f1";
    var t = sortable(table("", ["Model", "Device", "Images", "Objects", "Configured", "@precision", "@recall", "@f1", "Best", "@precision", "@recall", "@f1", gainLabel(obj) + " gain", "@roc_auc", "TPR / FPR configured", "TPR / FPR best", ""]));
    j.results.forEach(function (r) {
      // "Configured": the threshold in the config now (r.now), not the run's.
      var tr = el("tr"), c = r.now || r.configured, b = r.best, run = searchRunning(r);
      tr.dataset.model = r.model;
      if (r.model === search.selected) tr.classList.add("selected");
      var name = cell(tr, r.model + (r.relative ? " *" : "")); name.title = r.ground_truth;
      cell(tr, r.device); cell(tr, r.images, r.images); cell(tr, r.gt, r.gt);
      var cc = cell(tr, thr(c.threshold), c.threshold); cc.classList.add("thr-now");
      if (run !== null) cc.appendChild(el("span", "muted thr-run", "(configured; running " + thr(run) + " until restart)"));
      [c.precision, c.recall, c.f1].forEach(function (v) { cell(tr, pct(v), v === null ? null : -v); });
      cell(tr, b ? thr(b.threshold) + (b.met ? "" : " !") : "-", b ? b.threshold : null);
      [b && b.point.precision, b && b.point.recall, b && b.point.f1].forEach(function (v) { cell(tr, pct(v), v === null || v === undefined ? null : -v); });
      var sb = b && objScore(r.objective, b.point), sc = objScore(r.objective, c);
      var gain = b && near(b.threshold, c.threshold) ? 0 : sb !== null && sb !== undefined && sc !== null && sc !== undefined ? (sb - sc) * 100 : null;
      var gc = cell(tr, gain === null ? (sb !== null && sb !== undefined ? "target met only at best" : "-") : (gain >= 0 ? "+" : "") + gain.toFixed(1), gain === null ? null : -gain);
      gc.title = gainLabel(r.objective) + " at the best minus at the configured threshold (" + thr(c.threshold) + "), in points";
      var auc = r.roc ? r.roc.auc : null, ac = cell(tr, pct(auc), auc === null ? null : -auc); if (r.roc) ac.title = "micro " + pct(r.roc.micro_auc);
      cell(tr, tf(c)); cell(tr, tf(b && b.point));
      var act = el("td", "actions-cell"), show = el("button", "small", "Chart"); show.type = "button";
      show.addEventListener("click", function () { search.selected = r.model; drawSearch(); }); act.appendChild(show);
      if (applicableSearch(r) && !near(c.threshold, b.threshold)) { var u = el("button", "small", "Apply " + thr(b.threshold)); u.type = "button"; u.addEventListener("click", function () { useThreshold(r.model, b.threshold); }); act.appendChild(u); }
      else if (applicableSearch(r)) act.appendChild(el("span", "muted", run !== null ? "in config (restart to use)" : "in use"));
      tr.appendChild(act); t.tBodies[0].appendChild(tr);
    });
    box.appendChild(PB.tableFilter(t, "s-sum"));
    box.appendChild(wrap(t, "s-sum"));
    var r = j.results.find(function (x) { return x.model === search.selected; }); if (!r) return;
    var b = r.best, f2 = r.objective === "f2", c = r.now || r.configured, run = searchRunning(r);
    box.appendChild(el("h3", null, r.model + " on " + r.device + " (" + r.ground_truth + ", IoU " + r.iou + ")"));
    if (b) box.appendChild(el("p", b.met ? "notice" : "warn", "Best " + F.thrBi(b.threshold) + " for " + r.objective + ": P " + pct(b.point.precision) + " R " + pct(b.point.recall) + " F1 " + pct(b.point.f1) + " F2 " + pct(b.point.f2) + " · " + b.point.fp_per_image.toFixed(2) + " false positives per image" + (b.point.frame ? " · frame TPR / FPR " + tf(b.point) + " (configured " + thr(c.threshold) + ": " + tf(c) + ")" : "") + ". Exact optimum at " + (b.exact_threshold !== null && b.exact_threshold !== undefined ? b.exact_threshold.toFixed(3) : "-") + (b.plateau ? ", near-optimal range " + b.plateau[0].toFixed(3) + "–" + b.plateau[1].toFixed(3) : "") + "." + (b.note ? " " + b.note + "." : "")));
    var crow = el("div", "chart-row"), cw = el("div", "thr-chart-wrap");
    cw.appendChild(C.chartSvg({
      key: "s-thr",
      title: r.model + ": precision, recall, F1" + (f2 ? ", F2" : "") + " vs confidence threshold on " + r.device,
      desc: "Configured " + thr(c.threshold) + ": F1 " + pct(c.f1) + "%. Best " + (b ? thr(b.threshold) + ": F1 " + pct(b.point.f1) + "%" : "none") + ".",
      points: r.exact.length > 1 ? r.exact : r.grid, configured: c.threshold, best: b ? b.threshold : null,
      running: run, f2: f2
    }));
    crow.appendChild(cw);
    if (r.roc && r.roc.curve && r.roc.curve.length) crow.appendChild(rocWrap({
      key: "s-roc",
      title: r.model + ": frame-level ROC on " + r.device,
      desc: "Frame ROC AUC " + pct(r.roc.auc) + "%. Configured " + thr(c.threshold) + ": TPR / FPR " + tf(c) + "%" + (b ? "; best " + thr(b.threshold) + ": " + tf(b.point) + "%" : "") + ".",
      curve: r.roc.curve, auc: r.roc.auc, configured: c, best: b ? b.point : null
    }));
    box.appendChild(crow);
    if (r.roc && r.roc.per_class && r.roc.per_class.length) {
      var rd = det("s-roc", "Frame-level ROC AUC by class (macro " + pct(r.roc.auc) + ", micro " + pct(r.roc.micro_auc) + ")"); rd.appendChild(wrap(rocTable(r.roc), "s-roc")); box.appendChild(rd);
    }
    var g = sortable(table("", ["Confidence", "@precision", "@recall", "@f1", "@f2", "@fp_per_image", "TP", "FP", "FN", "@tpr", "@fpr", ""]));
    var rows = r.grid.slice(); [c, b && b.point].forEach(function (p) { if (p && !rows.some(function (q) { return near(q.threshold, p.threshold); })) rows.push(p); });
    rows.sort(function (x, y) { return x.threshold - y.threshold; });
    rows.forEach(function (p) {
      var tr = el("tr"), marks = [];
      if (b && near(b.threshold, p.threshold)) { marks.push("best"); tr.classList.add("best-row"); }
      if (near(c.threshold, p.threshold)) marks.push("configured");
      if (run !== null && near(run, p.threshold)) marks.push("running");
      cell(tr, p.threshold.toFixed(2), p.threshold); [p.precision, p.recall, p.f1, p.f2].forEach(function (v) { cell(tr, pct(v), v === null ? null : -v); });
      cell(tr, p.fp_per_image.toFixed(2), p.fp_per_image); cell(tr, p.tp, p.tp); cell(tr, p.fp, p.fp); cell(tr, p["fn"], p["fn"]);
      var ft = p.frame ? p.frame.tpr : null, ff = p.frame ? p.frame.fpr : null; cell(tr, pct(ft), ft === null ? null : -ft); cell(tr, pct(ff), ff);
      cell(tr, marks.join(", "));
      g.tBodies[0].appendChild(tr);
    });
    var d = det("s-grid", "Every 0.05 step", true); d.appendChild(wrap(g, "s-grid")); box.appendChild(d);
    var groups = r.by_dataset.concat(r.per_class.map(function (x) { return Object.assign({}, x, {key: "class " + x.key}); }));
    if (groups.length) {
      var gt = sortable(table("", ["Group", "Images", "Objects", "Best", "P", "R", "F1", "F1 at configured " + thr(c.threshold), "ROC AUC"]));
      groups.forEach(function (x) {
        var tr = el("tr"), xb = x.best, xn = x.now || x.configured; cell(tr, x.key + (x.relative ? " (relative)" : "")); cell(tr, x.images, x.images); cell(tr, x.gt, x.gt);
        cell(tr, xb ? thr(xb.threshold) + (xb.met ? "" : " !") : "-", xb ? xb.threshold : null);
        [xb && xb.point.precision, xb && xb.point.recall, xb && xb.point.f1, xn.f1, x.roc_auc].forEach(function (v) { cell(tr, pct(v), v === null || v === undefined ? null : -v); });
        gt.tBodies[0].appendChild(tr);
      });
      var d2 = det("s-groups", "Best threshold by dataset and class"); d2.appendChild(wrap(gt, "s-groups")); box.appendChild(d2);
    }
  }

  // ---- results ---------------------------------------------------------------------------------
  // Saved results overlaid with the models of the current run.
  function merged(j) {
    var list = (j.results && j.results.models ? j.results.models.slice() : []);
    ((j.run && j.run.models) || []).forEach(function (m) {
      var i = list.findIndex(function (x) { return x.model.toLowerCase() === m.model.toLowerCase(); });
      // A model of the running benchmark is part of "this run" (the server marks it once saved).
      var mm = j.running && m.this_run === undefined ? Object.assign({}, m, {this_run: true}) : m;
      if (i >= 0) list[i] = mm; else list.push(mm);
    });
    return list;
  }
  function recDevice(m) { return (m.devices || []).find(function (d) { return d.device === m.recommended; }); }
  function recRun(m) { var d = recDevice(m); return d && d.run; }
  function cfgOf(cfg, name) { return ((cfg && cfg.models) || []).find(function (c) { return c.name.toLowerCase() === String(name).toLowerCase(); }) || {}; }
  function modelOf(models, name) { return models.find(function (m) { return m.model.toLowerCase() === String(name).toLowerCase(); }); }
  // " (AUC)" / " (AP50)": which metric graded the accuracy.
  function metricTag(g) { return g && g.metric === "roc_auc" ? " (AUC)" : " (AP50)"; }
  // The accuracy weight (0..1) of the config's benchmark settings.
  function weightOf(b) { var w = (b && b.weights) || {}; return (w.accuracy || 0) + (w.speed || 0) > 0 ? w.accuracy / (w.accuracy + w.speed) : 0.6; }

  // ---- Overview: run summary banner --------------------------------------------------------------
  function renderBanner(j, models) {
    var box = $("run-banner"); box.replaceChildren();
    var p = j.run && j.run.progress;
    if (j.running) {
      // run.models holds the finished models plus the one in progress.
      var total = p && p.modelCount ? p.modelCount : null, done = p && p.modelIndex ? p.modelIndex - 1 : Math.max(0, ((j.run && j.run.models) || []).length - 1);
      box.appendChild(el("p", "notice banner", "Benchmark running: partial results" + (total ? ", " + Math.min(done, total) + " of " + total + " models done" : "") + "; changes appear as each model finishes."));
      return;
    }
    var mine = models.filter(function (m) { return m.this_run; });
    if (!mine.length) return;
    var b = el("div", "card banner run-summary"), ts = j.latest_run_id || mine[0].timestamp;
    b.appendChild(el("h2", null, "Latest run: " + when(ts) + " · " + mine.length + " model" + (mine.length === 1 ? "" : "s")));
    var ul = el("ul", "change-list");
    mine.forEach(function (m) {
      var li = el("li"); li.appendChild(el("strong", null, m.model + ": "));
      var ch = m.changes;
      if (m.error) li.appendChild(el("span", "error-text", "failed: " + m.error));
      else if (ch === undefined) li.appendChild(el("span", "muted", m.recommendation || ""));
      else {
        var rk = (j.ranking || []).find(function (r) { return r.model === m.model; }), metric = rk && rk.metric === "roc_auc" ? "roc_auc" : "ap50";
        li.appendChild(document.createTextNode((ch && ch.summary) || (ch && ch.baseline ? "" : "first run of this model; the next run is compared with it") + " "));
        li.appendChild(PB.changeChips(ch, m.recommended, metric));
        var tc = ch && (ch.metrics || []).find(function (x) { return x.metric === "threshold"; });
        if (tc && has(tc.before) && has(tc.after) && !near(tc.before, tc.after)) { li.appendChild(document.createTextNode(" confidence " + thr(tc.before) + " → " + thr(tc.after) + " ")); li.appendChild(PB.metricChip(tc)); }
      }
      ul.appendChild(li);
    });
    b.appendChild(ul);
    var old = models.length - mine.length;
    if (old > 0) b.appendChild(el("p", "muted", old + " other model" + (old === 1 ? " keeps its" : "s keep their") + " results from an earlier run (marked “earlier”)."));
    box.appendChild(b);
  }

  // ---- Overview: "What matters most" -----------------------------------------------------------
  var PRESETS = [
    {id: "balanced", name: "Balanced", text: "Finding objects and avoiding false alerts count the same; speed counts half of the grade.", obj: "f1", w: 0.5},
    {id: "miss", name: "Miss fewer objects", text: "Catching every person or car matters most; some extra false alerts are fine.", obj: "f2", w: 0.7},
    {id: "false", name: "Fewer false alerts", text: "At least 9 in 10 detections must be right; then find as many objects as possible.", obj: "precision:0.9", w: 0.5},
    {id: "fast", name: "Fastest / lightest", text: "Keep the CPU or GPU free for Blue Iris and other work: speed counts 80%.", obj: "f1", w: 0.2},
    {id: "custom", name: "Custom", text: "Set the accuracy weight, the confidence objective and the accuracy metric yourself.", obj: null, w: null}
  ];
  var presetUi = {custom: false, msg: "", busy: false};
  function presetOf(b) {
    if (!b) return "custom";
    var w = weightOf(b), o = String(b.threshold_objective || "f1");
    var p = PRESETS.find(function (x) { return x.obj === o && Math.abs(x.w - w) < 0.011; });
    return p ? p.id : "custom";
  }
  function saveSettings(fields, done) {
    Object.keys(fields).forEach(function (k) { var c = form.elements[k]; if (c) { c.value = fields[k]; c.classList.remove("edited"); } });
    presetUi.busy = true;
    return post("/v1/benchmark/settings", settingsBody()).then(function (j) {
      presetUi.busy = false;
      presetUi.msg = j.success ? (j.regraded === true ? "Saved: the ranking and the recommended confidence are updated from the stored results (no new run)." : j.regraded === false ? "Saved. The ranking changes with the next run (the stored results cannot be regraded)." : "Saved as the benchmark default.") : (j.message || "not saved");
      presetUi.ok = !!j.success;
      if (done) done(j);
      refresh(); refreshSearch();
    });
  }
  function renderPreset(j) {
    var box = $("preset"); box.replaceChildren();
    if (!lastModels.length) return; // nothing to rank yet
    var b = j.config && j.config.benchmark, cur = presetUi.custom ? "custom" : presetOf(b);
    var card = el("div", "card preset-card");
    var h = stepHead(1, "What matters most to you?", PB.term("?", "grade", "preset"));
    var sn = staleNotice(j, lastModels); card.appendChild(h);
    card.appendChild(el("p", "muted", "Ranks the results and picks the recommended confidence; it changes no inference, so no new run is needed."));
    var fs = el("fieldset", "presets"); fs.appendChild(el("legend", "visually-hidden", "What matters most"));
    PRESETS.forEach(function (p) {
      var l = el("label", "preset" + (cur === p.id ? " on" : "")), i = el("input"); i.type = "radio"; i.name = "preset"; i.value = p.id; i.checked = cur === p.id; i.disabled = presetUi.busy || !!(j.running);
      i.addEventListener("change", function () {
        if (!i.checked) return;
        if (p.id === "custom") { presetUi.custom = true; renderPreset(lastStatus); return; }
        presetUi.custom = false; presetUi.msg = "saving…";
        saveSettings({accuracy_weight: p.w.toFixed(2), threshold_objective: p.obj});
        renderPreset(lastStatus);
      });
      var t = el("span", "preset-text"); t.appendChild(el("strong", null, p.name)); t.appendChild(el("span", "muted", p.text));
      l.appendChild(i); l.appendChild(t); fs.appendChild(l);
    });
    card.appendChild(fs);
    if (sn) card.appendChild(sn);
    if (cur === "custom") card.appendChild(customFields(b));
    if (j.running) card.appendChild(el("p", "muted", "Available when the benchmark finishes."));
    var m = el("p", presetUi.ok === false ? "error-text" : "muted", presetUi.msg); m.setAttribute("aria-live", "polite"); card.appendChild(m);
    box.appendChild(card);
  }
  function customFields(b) {
    var d = el("div", "custom-fields bench-grid");
    var wl = el("label", null, "Accuracy weight (0..1; speed gets the rest) "), w = el("input"); w.type = "number"; w.min = "0"; w.max = "1"; w.step = "0.05"; w.value = weightOf(b).toFixed(2); wl.appendChild(w); d.appendChild(wl);
    var clone = function (name, text) { var l = el("label", null, text + " "), s = el("select"); Array.prototype.forEach.call(form.elements[name].options, function (o) { var x = el("option", null, o.textContent); x.value = o.value; s.appendChild(x); }); s.value = name === "threshold_objective" ? String((b && b.threshold_objective) || form.elements[name].value) : String((b && b[name]) || form.elements[name].value); l.appendChild(s); d.appendChild(l); return s; };
    var o = clone("threshold_objective", "Best confidence optimizes"), m = clone("accuracy_metric", "Grade accuracy by");
    var act = el("div", "actions"), save = el("button", "small", "Save and re-rank"); save.type = "button";
    save.addEventListener("click", function () { presetUi.msg = "saving…"; saveSettings({accuracy_weight: w.value, threshold_objective: o.value, accuracy_metric: m.value}, function () { presetUi.custom = presetOf(lastStatus && lastStatus.config && lastStatus.config.benchmark) === "custom"; }); });
    act.appendChild(save); d.appendChild(act);
    return d;
  }

  // ---- the shared device selection (Overview steps 2-3, recommendation, Confidence) -------------
  // "rec" = the recommended device of each model, "all" = every device, else a device spec.
  // Remembered per browser.
  // "chosen" = the device picked for each model in step 2 (the default).
  var selDevice = PB.store.get("view", "chosen") || "chosen";
  function selName() { return selDevice === "chosen" ? "the chosen device of each model" : selDevice === "rec" ? "the recommended device of each model" : selDevice === "all" ? "every device" : selDevice; }
  // Step 2: the device picked for each model ({model (lower case): device spec}), remembered per
  // browser. Without a pick (or when the model did not run on it): the device it uses now when it
  // ran there, else its recommended device, else the first device it ran on.
  var choice = PB.store.get("choice", {}) || {};
  function ranOn(m, device) { return (m.devices || []).some(function (d) { return d.device === device && d.run && !d.run.fell_back; }); }
  function chosenDevice(j, m) {
    var c = choice[m.model.toLowerCase()];
    if (c && ranOn(m, c)) return c;
    var now = nowDevice(j, m);
    if (now && ranOn(m, now)) return now;
    if (m.recommended && ranOn(m, m.recommended)) return m.recommended;
    var d = (m.devices || []).find(function (x) { return x.run && !x.run.fell_back; });
    return d ? d.device : null;
  }
  function choose(model, device) { choice[model.toLowerCase()] = device; PB.store.set("choice", choice); }
  function choiceKey() { return JSON.stringify(choice); }
  function devicesOf(models) {
    var out = []; models.forEach(function (m) { (m.devices || []).forEach(function (d) { if (d.run && out.indexOf(d.device) < 0) out.push(d.device); }); });
    return out.sort();
  }
  // "Compare models on: [...]": changing it redraws everything that follows the selection.
  function deviceSelector(models, scope, text) {
    var l = el("label", "inline-label device-select"), s = el("select");
    l.appendChild(document.createTextNode((text || "Compare models on") + " "));
    var opts = [["chosen", "Chosen device of each model (step 2)"], ["rec", "Recommended device of each model"], ["all", "All devices"]].concat(devicesOf(models).map(function (d) { return [d, d + " (" + (PB.RUNTIME_NAMES[PB.runtimeOf(d)] || "") + ")"]; }));
    if (!opts.some(function (o) { return o[0] === selDevice; })) opts.push([selDevice, selDevice + " (no results)"]);
    opts.forEach(function (o) { var x = el("option", null, o[1]); x.value = o[0]; if (o[0] === selDevice) x.selected = true; s.appendChild(x); });
    s.name = "device-select-" + scope;
    s.addEventListener("change", function () { selDevice = s.value; PB.store.set("view", selDevice); redrawSelection(); });
    l.appendChild(s); return l;
  }
  function redrawSelection() {
    if (!lastStatus) return;
    var j = lastStatus, models = lastModels;
    L.keep($("recommend"), undefined, function () { renderRecommend(j, models); });
    L.keep($("devices"), undefined, function () { renderDevices(j, models); });
    L.keep($("ranking"), undefined, function () { renderRanking(j, models); });
    L.keep($("conf-pick"), undefined, function () { renderConfPick(j); });
    L.keep($("conf-best"), undefined, function () { renderConfBest(j, models); });
  }
  // The best pick of model m on `device` (the advice's own device, else its by_device entry).
  function devPick(m, device) {
    var a = m.threshold; if (!a) return null;
    if (a.device === device && a.best) return a.best;
    var x = (a.by_device || []).find(function (z) { return z.device === device; });
    return x ? x.best : null;
  }
  // The device the config resolves model m to now.
  function nowDevice(j, m) { var c = cfgOf(j.config, m.model); return c.device || c.configuredDevice || m.configured || null; }
  function pointOf(j, model, device) { return (j.points || []).find(function (p) { return p.model === model && p.device === device; }) || null; }
  function rankOf(j, model) { return (j.ranking || []).find(function (r) { return r.model === model; }) || null; }
  // The best confidence of model m on `device` for the configured objective, shared by every
  // view: the stored pick when it was found for that objective, else the threshold search for it
  // on that device (fetched once into conf.cache, cleared when the results or the objective
  // change). {pick, state: "ok" | "loading" | "fallback" | "none", foundFor}.
  function bestPick(j, m, device) {
    var a = m.threshold; if (!a) return {pick: null, state: "none"};
    var cfg = savedObjective(j);
    if (String(a.objective) === cfg) return {pick: devPick(m, device), state: "ok"};
    var s = conf.cache[cfg + "|" + device];
    if (!s) { confSearch(cfg, device); return {pick: null, state: "loading"}; }
    var x = s.error ? null : (s.results || []).find(function (q) { return q.model.toLowerCase() === m.model.toLowerCase() && q.device === device; });
    if (x) return {pick: x.best, state: "ok"};
    var own = devPick(m, device);
    return own ? {pick: own, state: "fallback", foundFor: a.objective} : {pick: null, state: "none"};
  }
  // "(found for F2: miss fewer objects)" after a value that is not for the configured objective.
  function foundFor(r) { return r.pickState === "fallback" ? el("span", "muted found-for", " (found for " + PB.objectiveName(r.foundFor).split(":")[0] + ")") : null; }
  function withFound(r, text) { var f = foundFor(r); if (!f) return text; var s = el("span", "nowrap"); s.appendChild(document.createTextNode(text)); s.appendChild(f); s.title = "Found for " + PB.objectiveName(r.foundFor) + "; the search for your setting (" + PB.objectiveName(savedObjective(lastStatus)) + ") is not available."; return s; }
  // One row per model x device that ran: speed, accuracy, the device's best confidence, trust.
  function devRows(j, models) {
    var out = [];
    models.forEach(function (m) {
      var now = nowDevice(j, m), rk = rankOf(j, m.model), picked = chosenDevice(j, m);
      (m.devices || []).forEach(function (d) {
        var r = d.run; if (!r) return;
        var p = pointOf(j, m.model, d.device), g = r.grades || {}, acc = r.accuracy, ov = acc && acc.overall, bp = bestPick(j, m, d.device), pk = bp.pick;
        var pt = pk && pk.point, rec = m.recommended === d.device, isRk = rk && rk.device === d.device;
        out.push({
          key: m.model.toLowerCase() + "|" + d.device, m: m, d: d, model: m.model, device: d.device, runtime: PB.runtimeOf(d.device, d.runtime),
          p50: p ? p.p50_ms : has(g.p50_ms) && g.p50_ms < 1e300 ? g.p50_ms : (r.stages_ms && r.stages_ms.total ? r.stages_ms.total.p50 : null),
          score: p ? p.accuracy_score : g.accuracy_score, overall: p ? p.overall : g.overall, overallPoints: p ? p.overall_points : g.overall_points,
          accGrade: p ? p.accuracy : g.accuracy, speedGrade: p ? p.speed : g.speed, relative: !!(p ? p.relative : g.relative),
          ap50: ov ? ov.ap50 : null, rocAuc: acc && acc.frame_roc ? acc.frame_roc.auc : null,
          pick: pk, thr: pk ? pk.threshold : null, recall: pt ? pt.recall : ov ? ov.recall : null, precision: pt ? pt.precision : ov ? ov.precision : null,
          fpr: pt && pt.frame ? pt.frame.fpr : null, atBest: !!pt, pickState: bp.state, foundFor: bp.foundFor,
          trust: (p && p.trust) || d.trust || (rec ? m.trust : null), rec: rec, now: now === d.device, chosen: picked === d.device, fellBack: !!r.fell_back, thisRun: m.this_run,
          // One rank order per view (as step 4 picks): the model ranking for the recommended devices,
          // else the rank among every model x device point.
          rank: selDevice === "rec" && isRk ? rk.rank : p ? p.rank : isRk ? rk.rank : null, pointRank: p ? p.rank : null,
          best: selDevice === "rec" ? (isRk && rk.best_tradeoff !== undefined ? !!rk.best_tradeoff : !!(p && p.best_tradeoff)) : !!(p && p.best_tradeoff)
        });
      });
    });
    return out;
  }
  function selectedRow(r) { return selDevice === "all" ? true : selDevice === "chosen" ? r.chosen : selDevice === "rec" ? r.rec : r.device === selDevice; }
  // The tags of a device cell: recommended, now, fell back.
  function deviceCell(r) {
    var s = el("span", "nowrap"); s.appendChild(document.createTextNode(r.device));
    if (r.rec) { var t0 = el("span", "tag rec", "best"); t0.title = "recommended device for this model"; s.appendChild(t0); }
    if (r.now) s.appendChild(el("span", "tag cfg", "now"));
    if (r.fellBack) s.appendChild(el("span", "tag restart", "fell back"));
    s.title = (PB.RUNTIME_NAMES[r.runtime] || r.runtime) + (r.d.label ? " · " + r.d.label : "");
    return s;
  }
  function stepHead(n, text, extra) {
    var h = el("h2", "step-h"); h.appendChild(el("span", "step-n", String(n))); h.appendChild(document.createTextNode(" " + text + " "));
    if (extra) h.appendChild(extra); return h;
  }
  var FILTERS = {model: function (r) { return r.model; }, device: function (r) { return r.device; }, runtime: function (r) { return PB.RUNTIME_NAMES[r.runtime] || r.runtime; }, thisRun: function (r) { return r.thisRun; }};
  // Confidence in a table cell: "0.54 (54%)" (the header tooltip says the % is Blue Iris's).
  function thrShort(v) { return has(v) ? v.toFixed(2) + " (" + Math.round(v * 100) + "%)" : "-"; }
  function confText(r) { return r.pickState === "loading" ? "\u2026" : findsNothing(r.pick) ? "finds no objects" : has(r.thr) ? withFound(r, thrShort(r.thr)) : "-"; }
  function usableT(r) { return r.pick && r.pick.met && r.pick.point && r.pick.point.tp > 0 && !findsNothing(r.pick) ? r.pick.threshold : null; }

  // ---- Overview: recommendation ------------------------------------------------------------------
  // "9 in 10" for a precision.
  function inTen(p) { if (!has(p)) return null; var n = Math.round(p * 10); return n >= 10 ? "nearly all detections are" : n + " in 10 detections are"; }
  function nowThr(j, r) { var a = r.m.threshold; return a ? nowOf(a).threshold : cfgOf(j.config, r.model).confidenceThreshold; }
  function inUseTag() { return el("span", "tag now", "✓ in use"); }
  // The pair action of a row: "Use <device> + 0.17", "Use confidence 0.17", or in use.
  function pairAction(j, r, small) {
    var t = usableT(r), thrNow = nowThr(j, r);
    if (r.pickState === "loading") return el("span", "muted", "\u2026");
    if (r.now && (!has(t) || near(thrNow, t))) { var s = el("span"); s.appendChild(inUseTag()); return s; }
    var label = r.now ? "Use confidence " + thr(t) : "Use " + r.device + (has(t) ? " + " + thr(t) : "");
    var b = el("button", (small ? "small " : "") + "nowrap", small ? (r.now ? "Apply " + thr(t) : "Use this") : label); b.type = "button";
    b.title = label + (has(t) ? " (= " + Math.round(t * 100) + "% in Blue Iris)" : "") + " for " + r.model;
    b.setAttribute("aria-label", b.title);
    b.disabled = !!j.running || r.fellBack;
    b.addEventListener("click", function () {
      b.disabled = true;
      if (r.now) useThreshold(r.model, t); else usePair(r.model, r.device, t);
    });
    return b;
  }
  function renderRecommend(j, models) {
    var box = $("recommend"); box.replaceChildren();
    var rows = devRows(j, models).filter(function (r) { return !r.fellBack; }), pick = null;
    if (selDevice === "rec" || selDevice === "all") {
      var top = (j.ranking || []).find(function (r) { return r.rank === 1; }) || (j.ranking || [])[0];
      pick = top ? rows.find(function (r) { return r.model === top.model && r.device === top.device; }) : null;
    } else {
      // The best-ranked of the chosen pairs (step 2), or of the models on one device.
      pick = rows.filter(selectedRow).sort(function (a, b) { return (has(a.pointRank) ? a.pointRank : 1e9) - (has(b.pointRank) ? b.pointRank : 1e9) || (b.overallPoints || 0) - (a.overallPoints || 0); })[0] || null;
    }
    if (!pick) { var rl = $("rec-line"); if (rl) rl.replaceChildren(); if (models.length && !j.running) { var w = el("div", "card"); w.appendChild(el("h2", null, "No recommendation yet" + (selDevice !== "rec" && selDevice !== "all" ? " for " + selName() : ""))); models.forEach(function (x) { w.appendChild(el("p", "muted", x.model + ": " + (x.error || x.recommendation || "no result"))); }); box.appendChild(w); } return; }
    var card = el("div", "card primary recommend");
    var anyDev = selDevice === "rec" || selDevice === "all";
    var h = stepHead(4, "Your pick: " + pick.model + " on " + pick.device, PB.trustBadge(pick.trust, "rec")); card.appendChild(h);
    var fl = el("p", "muted"), cl = el("a", null, "choose a device in step 2"); cl.href = "#devices";
    cl.textContent = "step 2";
    fl.appendChild(document.createTextNode(selDevice === "chosen" ? "The best model of your choices, each model on the device you chose for it in " : anyDev ? "The best model on any device (each model on its recommended device); choose a device per model in " : "The best model on " + selDevice + " (as compared in step 3); choose a device per model in "));
    fl.appendChild(cl); fl.appendChild(document.createTextNode("."));
    card.appendChild(fl);
    var parts = ["About " + F.ms(pick.p50) + " per image"];
    if (findsNothing(pick.pick)) parts.push("but it finds no objects on these images");
    else { if (has(pick.recall)) parts.push("finds " + F.pct(pick.recall) + " of objects"); if (has(pick.precision)) parts.push(inTen(pick.precision) + " correct"); }
    var rt = el("p", "rec-text", parts.join(", ") + (pick.pickState === "loading" ? " (finding the best confidence\u2026)" : has(pick.thr) && !findsNothing(pick.pick) ? " at confidence " + F.thrBi(pick.thr) : "") + "."); var ff = foundFor(pick); if (ff) rt.appendChild(ff);
    card.appendChild(rt);
    var line = $("rec-line"); if (line) { line.replaceChildren(); line.appendChild(document.createTextNode("Recommended: " + pick.model + " on " + pick.device + (has(pick.thr) && !findsNothing(pick.pick) ? ", confidence " + F.thrBi(pick.thr) : "") + " \u2014 ")); var ln = el("a", null, "see step 4"); ln.href = "#recommend"; line.appendChild(ln); }
    if (pick.relative) card.appendChild(el("p", "muted", "* Accuracy is relative to a reference model's detections (no human labels on these images), so it shows agreement, not true accuracy."));
    var nRank = (j.ranking || []).length, metric = ((j.ranking || [])[0] || {}).metric;
    var why = el("p", "muted");
    why.appendChild(document.createTextNode((selDevice === "rec" || selDevice === "all" ? (nRank > 1 ? "Ranked first of " + nRank + " models" : "The only ranked model") : "The best-ranked model on " + selName()) + " by " + (metric === "roc_auc" ? "alert separation (ROC AUC)" : "detection quality (AP50)") + " and speed, with accuracy counting " + Math.round(weightOf(j.config && j.config.benchmark) * 100) + "%. "));
    why.appendChild(PB.runTag(pick.m)); card.appendChild(why);
    var act = el("div", "actions"); act.appendChild(pairAction(j, pick, false)); card.appendChild(act);
    if (has(pick.thr) && !findsNothing(pick.pick)) card.appendChild(el("p", "bi-line", "In Blue Iris, set each camera's minimum confidence (AI settings) to " + Math.round(pick.thr * 100) + "% or lower; a higher value there filters the detections again."));
    card.appendChild(el("p", "muted small-print", "Device and confidence changes apply at once: the model reloads on its new device, and the next detections use the new confidence (no restart)."));
    box.appendChild(card);
  }

  // ---- Overview step 2: pick a device ------------------------------------------------------------
  function renderDevices(j, models) {
    var box = $("devices"); box.replaceChildren();
    var rows = devRows(j, models); if (!rows.length) return;
    var card = el("div", "card");
    card.appendChild(stepHead(2, "Pick a device for each model", PB.term("?", "p50", "dev")));
    card.appendChild(el("p", "muted", "Each model on every device it ran on, at that device's best confidence. Choose one device per model under \u201cRun on\u201d (each model can use a different device or runtime); steps 3 and 4 and the Confidence tab use your choices. Then apply them: each model gets its chosen device and the best confidence for that model and device together (both apply at once; Undo reverts them)."));
    // Quick picks for every model at once.
    var tools = el("div", "chart-tools quick-pick");
    tools.appendChild(el("span", "muted", "Choose for every model: "));
    var pickAll = function (label, title, fn) {
      var b = el("button", "small secondary", label); b.type = "button"; b.title = title;
      b.addEventListener("click", function () {
        models.forEach(function (m) { var own = rows.filter(function (r) { return r.m === m && !r.fellBack; }), r = fn(own, m); if (r) choose(m.model, r.device); });
        redrawSelection();
      });
      tools.appendChild(b);
    };
    pickAll("recommended", "Each model's recommended device (the best grade that agrees with the CPU)", function (own) { return own.find(function (r) { return r.rec; }); });
    pickAll("in use now", "The device each model uses now", function (own) { return own.find(function (r) { return r.now; }); });
    pickAll("fastest", "Each model's fastest device", function (own) { return own.filter(function (r) { return has(r.p50); }).sort(function (a, b) { return a.p50 - b.p50; })[0]; });
    devicesOf(models).forEach(function (d) { pickAll(d, "Every model on " + d + " where it ran there (the others keep their choice)", function (own) { return own.find(function (r) { return r.device === d; }); }); });
    card.appendChild(tools);
    card.appendChild(PB.dataTable({
      key: "devices", cls: "stack dev-table", rows: rows, filters: FILTERS, sort: {col: "model", dir: "asc"},
      text: function (r) { return r.model + " " + r.device + " " + r.runtime; },
      rowAttrs: function (r, tr) { if (r.chosen) tr.classList.add("hl"); if (r.fellBack) tr.classList.add("failed"); },
      columns: [
        {id: "pick", head: "Run on", label: "Run on", value: function (r) { return r.chosen ? 0 : 1; }, render: function (r) {
          if (r.fellBack) return "-";
          var i = el("input"); i.type = "radio"; i.name = "run-on-" + r.model.toLowerCase(); i.checked = r.chosen;
          i.setAttribute("aria-label", "Run " + r.model + " on " + r.device);
          i.addEventListener("change", function () { if (i.checked) { choose(r.model, r.device); redrawSelection(); } });
          return i; }},
        {id: "model", head: "@model", value: function (r) { return r.model.toLowerCase(); }, render: function (r) { return el("span", "nowrap", r.model + (r.relative ? " *" : "")); }},
        {id: "device", head: "@device", value: function (r) { return r.device; }, render: deviceCell},
        {id: "p50", head: "@p50", num: true, value: function (r) { return r.fellBack ? null : r.p50; }, render: function (r) { return F.ms(r.p50); }},
        {id: "score", head: "@accuracy_score", num: true, first: "desc", value: function (r) { return r.score; }, render: function (r) { return F.pct(r.score, 1); }},
        {id: "recall", head: "@recall", num: true, first: "desc", value: function (r) { return r.recall; }, render: function (r) { return F.pct(r.recall); }},
        {id: "precision", head: "@precision", num: true, first: "desc", value: function (r) { return r.precision; }, render: function (r) { return F.pct(r.precision); }},
        {id: "thr", head: "@best_conf", num: true, value: function (r) { return r.thr; }, render: function (r, td) { if (findsNothing(r.pick)) td.title = FINDS_NOTHING; return confText(r); }},
        {id: "now", head: "@now_conf", num: true, value: function (r) { return nowThr(j, r); }, render: function (r) { return thrShort(nowThr(j, r)); }},
        {id: "trust", head: "@trust", value: function (r) { return r.trust ? {high: 0, medium: 1, low: 2}[r.trust.level] : null; }, render: function (r) { return PB.trustBadge(r.trust, "dev"); }},
        {id: "act", head: "", label: "Action", render: function (r) { return r.fellBack || !r.chosen ? "" : pairAction(j, r, true); }}
      ]
    }));
    card.appendChild(choicesPanel(j, rows.filter(function (r) { return r.chosen; })));
    box.appendChild(card);
  }
  // What applying the step-2 choices changes: per model its device and/or its confidence (the
  // best one of the chosen model x device), with one button for all of them and one Undo.
  function choicesPanel(j, chosen) {
    var d = el("div", "choices"), todo = [], loading = false;
    chosen.forEach(function (r) {
      if (r.pickState === "loading") { loading = true; return; }
      var t = usableT(r), n = nowThr(j, r), dev = !r.now, th = has(t) && !near(n, t);
      if (dev || th) todo.push({r: r, t: has(t) ? t : null, dev: dev, th: th, n: n});
    });
    d.appendChild(el("h3", null, "Apply your choices"));
    if (loading) { d.appendChild(el("p", "muted", "Finding the best confidence of each chosen device…")); return d; }
    if (!todo.length) { d.appendChild(el("p", "muted", "Every model already runs on its chosen device with that pair's best confidence: nothing to apply.")); return d; }
    var ul = el("ul", "change-list");
    todo.forEach(function (x) {
      var parts = [];
      if (x.dev) parts.push("device " + (nowDevice(j, x.r.m) || "auto") + " → " + x.r.device);
      if (x.th) parts.push("confidence " + F.thrBi(x.n) + " → " + F.thrBi(x.t));
      else if (x.dev && !has(x.t)) parts.push("confidence unchanged (no usable best confidence on " + x.r.device + ")");
      var li = el("li"); li.appendChild(el("strong", null, x.r.model + ": ")); li.appendChild(document.createTextNode(parts.join("; "))); ul.appendChild(li);
    });
    d.appendChild(ul);
    var act = el("div", "actions"), b = el("button", null, "Apply " + todo.length + " change" + (todo.length === 1 ? "" : "s")); b.type = "button"; b.id = "apply-choices"; b.disabled = !!j.running;
    b.addEventListener("click", function () { b.disabled = true; applyChoices(todo); });
    act.appendChild(b);
    act.appendChild(el("span", "muted", "Both apply at once: a model on a new device reloads there; the next detections use the new confidence (no restart)."));
    d.appendChild(act);
    return d;
  }
  // Applies each model's chosen device and confidence in turn; one message and one Undo for all.
  function applyChoices(todo) {
    var steps = [];
    return todo.reduce(function (p, x) {
      return p.then(function () {
        var dev = x.dev ? applyDevice(x.r.model, x.r.device).then(function (j) { steps.push({j: j, kind: "device"}); return j.success; }) : Promise.resolve(true);
        return dev.then(function (ok) { if (ok && x.th) return applyThreshold(x.r.model, x.t).then(function (k) { steps.push({j: k, kind: "threshold"}); }); });
      });
    }, Promise.resolve()).then(function () { showApply(steps); });
  }

  // ---- Overview step 3: compare models on the selected device ------------------------------------
  // A ranking row's best trade-off flag: its own, else the matching top-level point.
  function bestTradeoff(j, r) {
    if (r.best_tradeoff !== undefined) return !!r.best_tradeoff;
    var p = (j.points || []).find(function (x) { return x.model === r.model && x.device === r.device; });
    return !!(p && p.best_tradeoff);
  }
  function renderRanking(j, models) {
    var box = $("ranking"); box.replaceChildren();
    var all = devRows(j, models); if (!all.length) return;
    var rows = all.filter(selectedRow);
    // Shown rank = the position among the rows of this view (1 = best on the chosen device).
    var rankKey = function (r) { return has(r.rank) ? r.rank : has(r.overallPoints) ? 1000 - r.overallPoints : 1e9; };
    rows.slice().sort(function (a, b) { return rankKey(a) - rankKey(b); }).forEach(function (r, i) { r.pos = i + 1; });
    var byAuc = (((j.ranking || [])[0] || (j.points || [])[0] || {}).metric) === "roc_auc", metricKey = byAuc ? "roc_auc" : "ap50";
    var card = el("div", "card");
    card.appendChild(stepHead(3, "Compare models on " + selName()));
    var tools = el("div", "chart-tools"); tools.appendChild(deviceSelector(models, "rank"));
    tools.appendChild(el("span", "muted", "★ = best trade-off. Best first by the overall grade. Accuracy graded by " + (byAuc ? "alert separation (frame ROC AUC)" : "detection quality (AP50, with small-object recall)") + "; finds / correct at each device's best confidence."));
    card.appendChild(tools);
    card.appendChild(PB.dataTable({
      key: "ranking", cls: "stack rank-table", rows: rows, filters: FILTERS, sort: {col: "rank", dir: "asc"},
      text: function (r) { return r.model + " " + r.device + " " + r.runtime; }, empty: "No results on " + selName() + ": choose another device above.",
      columns: [
        {id: "rank", head: "@rank", num: true, value: function (r) { return r.pos; }, render: function (r) { return String(r.pos); }},
        {id: "model", head: "@model", value: function (r) { return r.model.toLowerCase(); }, render: function (r) {
          var s = el("span", "nowrap"); s.appendChild(document.createTextNode(r.model + (r.relative ? " *" : "")));
          if (r.best) { var bt = PB.tip("★", PB.tipText("best_tradeoff"), "tag best-tradeoff", "rank"); bt.setAttribute("aria-label", "best trade-off"); s.appendChild(bt); } return s; }},
        {id: "device", head: "@device", value: function (r) { return r.device; }, render: deviceCell},
        {id: "overall", head: "@grade", value: function (r) { return has(r.overallPoints) ? -r.overallPoints : null; }, render: function (r) {
          var s = el("span", "nowrap grade-cell"); s.appendChild(gradeEl(r.overall));
          s.appendChild(el("span", "sub-grades muted", " acc " + (r.accGrade || "-") + (r.relative ? "*" : "") + " \u00b7 speed " + (r.speedGrade || "-")));
          s.title = "Overall " + (r.overall || "-") + ": accuracy " + (r.accGrade || "not scored") + (r.relative ? " (relative to a reference model)" : "") + ", speed " + (r.speedGrade || "-") + ", combined by the accuracy weight (\u201cWhat matters most\u201d).";
          return s; }},
        {id: "p50", head: "@p50", num: true, value: function (r) { return r.p50; }, render: function (r) { return F.ms(r.p50); }},
        {id: "recall", head: "@recall", num: true, first: "desc", value: function (r) { return r.recall; }, render: function (r) { return F.pct(r.recall); }},
        {id: "precision", head: "@precision", num: true, first: "desc", value: function (r) { return r.precision; }, render: function (r) { return F.pct(r.precision); }},
        {id: "thr", head: "@threshold", num: true, value: function (r) { return r.thr; }, render: function (r) { return r.pickState === "loading" ? "\u2026" : findsNothing(r.pick) ? "finds none" : has(r.thr) ? withFound(r, thr(r.thr) + " (" + Math.round(r.thr * 100) + "%)") : "-"; }},
        {id: "acc2", head: byAuc ? "@roc_auc" : "@ap50", num: true, first: "desc", value: function (r) { return byAuc ? r.rocAuc : r.ap50; }, render: function (r) { return F.pct(byAuc ? r.rocAuc : r.ap50, 1); }},
        {id: "ran", head: "@ran", value: function (r) { return r.m.timestamp ? -new Date(r.m.timestamp).getTime() : null; }, render: function (r) { return PB.runTag(r.m); }},
        {id: "change", head: "@change", render: function (r) { return r.m.changes ? PB.changeChips(r.m.changes, r.device, metricKey) : "-"; }},
        {id: "trust", head: "@trust", value: function (r) { return r.trust ? {high: 0, medium: 1, low: 2}[r.trust.level] : null; }, render: function (r) { return PB.trustBadge(r.trust, "rank"); }}
      ]
    }));
    box.appendChild(card);
  }

  // ---- Confidence tab ------------------------------------------------------------------------
  var USES = [
    ["f1", "Balanced", "Finds objects and avoids false alerts equally (F1)."],
    ["f2", "Miss fewer objects", "Finds more objects and accepts more false alerts (F2)."],
    ["precision:0.9", "Fewer false alerts", "At least 9 in 10 detections right, then finds as many objects as possible (precision ≥ 90%)."],
    ["fpr:0.05", "Fewer false-alert frames", "At most 5 in 100 empty frames alert, then alerts on as many frames with objects as possible (false-alert rate ≤ 5%)."]
  ];
  var conf = {obj: null, cache: {}, pending: {}};
  function savedObjective(j) { return String((j && j.config && j.config.benchmark && j.config.benchmark.threshold_objective) || "f1"); }
  // The threshold search for `obj` on `device` ("" = each model's recommended device).
  function confSearch(obj, device) {
    var k = obj + "|" + (device || "");
    if (conf.cache[k] || conf.pending[k]) return;
    conf.pending[k] = true;
    var b = new URLSearchParams(); b.append("model", "all"); if (device) b.append("device", device); b.append("objective", obj); b.append("iou", "0.5");
    post("/v1/benchmark/threshold-search", b).then(function (j) {
      conf.pending[k] = false; conf.cache[k] = j.success ? j : {error: j.message || "the search failed"};
      redrawSelection();
    }, function () { conf.pending[k] = false; conf.cache[k] = {error: "the server did not answer"}; redrawSelection(); });
  }
  function renderConfPick(j) {
    var box = $("conf-pick"); box.replaceChildren();
    var saved = savedObjective(j); if (!conf.obj) conf.obj = saved;
    var card = el("div", "card");
    var h = el("h2", null, "Best confidence for your cameras "); h.appendChild(PB.term("?", "threshold", "conf")); card.appendChild(h);
    card.appendChild(el("p", "muted", "Pick what matters for your cameras: the best confidence comes from the stored predictions of the last run, so switching is instant (no inference)."));
    var fs = el("fieldset", "presets"); fs.appendChild(el("legend", "visually-hidden", "Use case"));
    var list = USES.slice(); if (!list.some(function (u) { return u[0] === saved; })) list.push([saved, "Saved setting", "The objective saved in the config: " + saved + "."]);
    list.forEach(function (u) {
      var l = el("label", "preset" + (conf.obj === u[0] ? " on" : "")), i = el("input"); i.type = "radio"; i.name = "conf-use"; i.value = u[0]; i.checked = conf.obj === u[0];
      i.addEventListener("change", function () { if (!i.checked) return; conf.obj = u[0]; renderConfPick(lastStatus); L.keep($("conf-best"), undefined, function () { renderConfBest(lastStatus, lastModels); }); });
      var t = el("span", "preset-text"); t.appendChild(el("strong", null, u[1] + (u[0] === saved ? " (saved)" : ""))); t.appendChild(el("span", "muted", u[2]));
      l.appendChild(i); l.appendChild(t); fs.appendChild(l);
    });
    card.appendChild(fs);
    var tools = el("div", "chart-tools"); tools.appendChild(deviceSelector(lastModels, "conf")); card.appendChild(tools);
    if (conf.obj !== saved) {
      var act = el("div", "actions"), sb = el("button", "small secondary", "Make this the default"); sb.type = "button";
      var m = el("span", "muted"); m.setAttribute("aria-live", "polite");
      sb.title = "Saves it as the benchmark's confidence objective: the Overview and the Apply buttons then use it";
      sb.addEventListener("click", function () { sb.disabled = true; saveSettings({threshold_objective: conf.obj}, function (r) { m.textContent = r.message || ""; }); });
      act.appendChild(sb); act.appendChild(m); card.appendChild(act);
    }
    var sn = lastStatus ? staleNotice(lastStatus, lastModels) : null; if (sn) card.appendChild(sn);
    card.appendChild(el("p", "bi-line", "In Blue Iris, set each camera's minimum confidence (AI settings) at or below the percentage shown for its model; a higher value there filters the detections again."));
    box.appendChild(card);
  }
  // The pick of `obj` for a row: the stored per-device pick (same objective) or a search result.
  // {best, saved, other (why the search is not used), pending}
  function confPick(r, obj) {
    var a = r.m.threshold;
    if (a && String(a.objective) === obj) return {best: r.pick, saved: true};
    if (obj === savedObjective(lastStatus)) return r.pickState === "loading" ? {pending: true} : r.pickState === "fallback" ? {best: r.pick, saved: false, other: "not available", objective: r.foundFor} : {best: r.pick, saved: false};
    var k = obj + "|" + r.device, s = conf.cache[k];
    if (!s) return {pending: true};
    if (s.error) return {best: r.pick, saved: false, other: s.error, objective: a ? a.objective : null};
    var x = (s.results || []).find(function (q) { return q.model.toLowerCase() === r.model.toLowerCase() && q.device === r.device; });
    return x ? {best: x.best, saved: false} : {best: null, saved: false, missing: true};
  }
  function renderConfBest(j, models) {
    var box = $("conf-best"); box.replaceChildren();
    var saved = savedObjective(j), obj = conf.obj || saved;
    var all = devRows(j, models).filter(function (r) { return r.m.threshold && !r.fellBack; }), rows = all.filter(selectedRow);
    if (!all.length) { box.appendChild(el("p", "empty", models.length ? "No confidence results: the benchmark needs datasets with labels (or a reference model) to find the best confidence." : "No results yet: run a benchmark under Run; this tab then shows the best confidence for each model and device.")); return; }
    // Searches for an objective the stored advice was not found for (per device when chosen).
    if (rows.some(function (r) { return String(r.m.threshold.objective) !== obj; })) {
      rows.forEach(function (r) { confSearch(obj, r.device); });
    }
    var picked = rows.map(function (r) { var c = confPick(r, obj); return Object.assign({}, r, {cp: c, b: c.best || null}); });
    var other = picked.find(function (r) { return r.cp.other; });
    if (other) box.appendChild(el("p", "muted", "The search for " + PB.objectiveName(obj) + " is not available (" + other.cp.other + "); showing the confidences found for " + PB.objectiveName(other.cp.objective) + "."));
    var usable = function (b) { return b && b.met && b.point && b.point.tp > 0 && !findsNothing(b); };
    box.appendChild(PB.dataTable({
      key: "confidence", cls: "stack conf-table", rows: picked, filters: FILTERS, sort: {col: "model", dir: "asc"},
      text: function (r) { return r.model + " " + r.device + " " + r.runtime; }, empty: "No confidence results on " + selName() + ": choose another device.",
      columns: [
        {id: "model", head: "@model", value: function (r) { return r.model.toLowerCase(); }, render: function (r) { return el("span", "nowrap", r.model + (r.relative ? " *" : "")); }},
        {id: "device", head: "@device", value: function (r) { return r.device; }, render: deviceCell},
        {id: "best", head: "@best_conf", num: true, value: function (r) { return r.b ? r.b.threshold : null; }, render: function (r, td) {
          if (r.cp.pending) return "searching…";
          if (!r.b) return r.cp.missing ? "not searched" : "-";
          if (r.cp.other) { var fb = el("span", "nowrap"); fb.appendChild(document.createTextNode(thr(r.b.threshold) + " = " + Math.round(r.b.threshold * 100) + "%")); fb.appendChild(el("span", "muted found-for", " (found for " + PB.objectiveName(r.cp.objective).split(":")[0] + ")")); if (findsNothing(r.b)) { td.title = FINDS_NOTHING; return "finds no objects"; } return fb; }
          if (findsNothing(r.b)) { td.title = FINDS_NOTHING; td.classList.add("warn-cell"); return "finds no objects"; }
          var s = el("span"); s.appendChild(el("strong", null, thr(r.b.threshold))); s.appendChild(document.createTextNode(" = " + Math.round(r.b.threshold * 100) + "%"));
          if (!r.b.met) s.appendChild(PB.tip("target not met", "The target is not reached at any confidence; this is the closest one." + (r.b.note ? " " + r.b.note : ""), "tag restart", "conf"));
          return s; }},
        {id: "range", head: "@range", value: function (r) { return r.b && r.b.plateau ? r.b.plateau[1] - r.b.plateau[0] : null; }, render: function (r) {
          var b = r.b; if (!b || !b.plateau || findsNothing(b)) return "-";
          var lo = b.plateau[0].toFixed(2), hi = b.plateau[1].toFixed(2), s = el("span", null, lo === hi ? lo : lo + "–" + hi);
          if (lo === hi) s.appendChild(PB.tip("sensitive", "Only a narrow range scores within " + PB.noise.plateau_points + " points of the best: small changes of the confidence matter for this model.", "tag restart", "conf"));
          return s; }},
        {id: "recall", head: "@recall", num: true, first: "desc", value: function (r) { return r.b && r.b.point ? r.b.point.recall : null; }, render: function (r) { return r.b && !findsNothing(r.b) ? F.pct(r.b.point.recall) : "-"; }},
        {id: "precision", head: "@precision", num: true, first: "desc", value: function (r) { return r.b && r.b.point ? r.b.point.precision : null; }, render: function (r) { return r.b && !findsNothing(r.b) ? F.pct(r.b.point.precision) : "-"; }},
        {id: "fpr", head: "@fpr", num: true, value: function (r) { return r.b && r.b.point && r.b.point.frame ? r.b.point.frame.fpr : null; }, render: function (r) { return r.b && r.b.point && r.b.point.frame && !findsNothing(r.b) ? F.pct(r.b.point.frame.fpr) : "-"; }},
        {id: "now", head: "@now_conf", num: true, value: function (r) { return nowOf(r.m.threshold).threshold; }, render: function (r) { return thrShort(nowOf(r.m.threshold).threshold); }},
        {id: "change", head: "@change", render: function (r) {
          var ch = r.m.changes, a = r.m.threshold;
          if (!ch || !r.cp.saved || a.device !== r.device) return "-";
          if (!ch.baseline) return PB.chip("first", null, "No earlier run of this model to compare with; the next run is compared with this one.");
          if (!ch.comparable) return PB.chip("not_comparable", null, "Not compared with the run of " + F.when(ch.baseline.timestamp) + ": " + (ch.reasons || []).join("; "));
          var tc = (ch.metrics || []).find(function (x) { return x.metric === "threshold"; });
          if (!tc) return PB.chip("same", null, "No change recorded since " + F.when(ch.baseline.timestamp) + ".");
          var c = PB.metricChip(tc); c.dataset.tip = "Since the run of " + F.when(ch.baseline.timestamp) + ": " + thr(tc.before) + " → " + thr(tc.after) + ". " + (tc.why || ""); return c; }},
        {id: "trust", head: "@trust", value: function (r) { return r.trust ? {high: 0, medium: 1, low: 2}[r.trust.level] : null; }, render: function (r) { return PB.trustBadge(r.trust, "conf"); }},
        {id: "act", head: "", label: "Action", render: function (r) {
          if (!usable(r.b)) return "-";
          var n = nowOf(r.m.threshold).threshold, t = r.b.threshold;
          if (r.now && near(n, t)) return inUseTag();
          var label = r.now ? "Apply " + thr(t) : "Switch + apply", b = el("button", "small nowrap", label); b.type = "button";
          b.title = (r.now ? "Apply confidence " + thr(t) : "Use " + r.device + " and confidence " + thr(t)) + " (= " + Math.round(t * 100) + "% in Blue Iris) for " + r.model;
          b.setAttribute("aria-label", b.title);
          b.disabled = !!j.running;
          b.addEventListener("click", function () { b.disabled = true; if (r.now) useThreshold(r.model, t); else usePair(r.model, r.device, t); });
          return b; }}
      ]
    }));
  }

  // ---- Details tab (one collapsible card per model) ---------------------------------------------
  function breakdown(key, title, rows) {
    var d = el("div"); if (!rows || !rows.length) return d;
    d.appendChild(el("h4", null, title));
    var t = sortable(table("", ["", "Images", "Objects", "@ap50", "@ap50_95", "@roc_auc", "@precision", "@recall", "@f1"]));
    rows.forEach(function (b) {
      var tr = el("tr"); cell(tr, b.key + (b.relative ? " (relative)" : "")); cell(tr, b.images, b.images); cell(tr, b.gt, b.gt);
      [b.ap50, b.ap50_95, b.roc_auc, b.precision, b.recall, b.f1].forEach(function (v) { cell(tr, pct(v), v === undefined ? null : v); });
      t.tBodies[0].appendChild(tr);
    });
    d.appendChild(wrap(t, key + ":" + title)); return d;
  }
  function details(key, run) {
    var box = det("brk:" + key, "Breakdown on " + run.requested_device + (run.accuracy ? " (" + run.accuracy.ground_truth + ")" : ""));
    if (run.accuracy) {
      var a = run.accuracy;
      box.appendChild(el("p", "muted", "Scored classes: " + a.classes.join(", ") + ". Precision/recall at confidence " + a.threshold.toFixed(2) + ", IoU 0.5. Small/medium/large recall: " + a.overall.by_size.map(function (s) { return s.bucket + " " + pct(s.recall) + " (" + s.gt + ")"; }).join(", ")));
      box.appendChild(breakdown("bd:" + key, "By dataset", a.by_dataset));
      box.appendChild(breakdown("bd:" + key, "By tag", a.by_tag));
      var classAuc = function (k) { var x = a.frame_roc && (a.frame_roc.per_class || []).find(function (c) { return c["class"] === k; }); return x ? x.auc : null; };
      box.appendChild(breakdown("bd:" + key, "By class", a.overall.per_class.map(function (c) { return {key: c.class, images: a.overall.images, gt: c.gt, ap50: c.ap50, ap50_95: c.ap50_95, roc_auc: classAuc(c.class), precision: c.precision, recall: c.recall, f1: null}; })));
      if (a.frame_roc && a.frame_roc.per_class && a.frame_roc.per_class.length) { box.appendChild(el("h4", null, "Frame-level ROC by class")); box.appendChild(wrap(rocTable(a.frame_roc), "roc-run:" + key)); }
    } else if (run.accuracy_note) box.appendChild(el("p", "muted", "Accuracy not scored: " + run.accuracy_note + "."));
    if (run.by_resolution && run.by_resolution.length) {
      box.appendChild(el("h4", null, "Speed by resolution (ms)"));
      var t = sortable(table("", ["Resolution", "Images", "Decode+pre p50", "Infer p50", "Post p50", "Total p50", "Total p95"]));
      ["sd", "hd", "fhd", "4mp+"].forEach(function (k) {
        var b = run.by_resolution.find(function (x) { return x.bucket === k; }), tr = el("tr"); cell(tr, k);
        if (!b) { cell(tr, "0", 0); for (var i = 0; i < 5; i++) cell(tr, "n/a", null); tr.className = "muted"; }
        else { cell(tr, b.images, b.images); [b.pre_p50, b.infer_p50, b.post_p50, b.total_p50, b.total_p95].forEach(function (v) { cell(tr, ms(v), v); }); }
        t.tBodies[0].appendChild(tr);
      });
      box.appendChild(wrap(t, "res:" + key));
    }
    return box;
  }
  // One collapsible card per model, patched in place: only cards whose data changed are rebuilt
  // (keeping their scroll offsets, open <details> and sort order). The first card (or the one
  // the URL names) starts open; a rebuilt card keeps its open state.
  function renderModels(models, cfg, ranking) {
    var box = $("models");
    $("no-models").hidden = models.length > 0;
    var cfgByName = {}; (cfg.models || []).forEach(function (c) { cfgByName[c.name.toLowerCase()] = c; });
    var hashId = location.hash.slice(1), named = models.some(function (m) { return "m-" + encodeURIComponent(m.model) === hashId; });
    L.patch(box, models.map(function (m, i) {
      var c = cfgByName[m.model.toLowerCase()] || {}, id = "m-" + encodeURIComponent(m.model);
      var rk = (ranking || []).find(function (r) { return r.model === m.model; }), metric = rk && rk.metric === "roc_auc" ? "roc_auc" : "ap50";
      var was = document.getElementById(id), open = was ? was.open : (named ? id === hashId : i === 0);
      return {key: m.model.toLowerCase(), sig: JSON.stringify([m, c, cfg.globalLabel, metric]), build: function () { var d = modelCard(m, c, cfg, metric); d.open = open; return d; }};
    }));
    if (location.hash && !box.dataset.scrolled) { var t2 = document.getElementById(hashId); if (t2) { if (t2.tagName === "DETAILS") t2.open = true; t2.scrollIntoView(); box.dataset.scrolled = "1"; } }
  }
  function modelCard(m, c, cfg, metric) {
    var det2 = el("details", "card model"), key = m.model.toLowerCase(); det2.id = "m-" + encodeURIComponent(m.model);
    var card = el("div", "body");
    // The device the config resolves to now (not when the benchmark ran) and the loaded one.
    var configured = c.configuredDevice || m.configured, running = c.runningDevice;
    var sum = el("summary"), rr0 = recRun(m);
    sum.appendChild(el("span", "mname", m.model));
    if (rr0 && rr0.grades) { sum.appendChild(gradeEl(rr0.grades.overall)); sum.appendChild(el("span", "muted", (m.recommended || "-") + " · " + ms(rr0.grades.p50_ms) + " ms")); }
    else if (m.error) sum.appendChild(el("span", "error-text", "failed"));
    else sum.appendChild(el("span", "muted", m.recommended || m.recommendation || "running…"));
    if (c.restartNeeded) { var rb = el("span", "tag restart", "restart needed"); rb.title = (c.pending || []).join("; "); sum.appendChild(rb); }
    sum.appendChild(PB.runTag(m));
    var rd = recDevice(m); if ((rd && rd.trust) || m.trust) sum.appendChild(PB.trustBadge((rd && rd.trust) || m.trust, "card"));
    if (m.changes !== undefined) sum.appendChild(PB.changeChips(m.changes, m.recommended, metric));
    sum.appendChild(el("span", "muted", (m.datasets || []).join(", ") + " · " + (m.images || 0) + " image" + (m.images === 1 ? "" : "s") + " · " + when(m.timestamp)));
    det2.appendChild(sum); det2.appendChild(card);
    card.appendChild(el("p", "muted cfg-line", "Configured: " + (c.device ? c.device : "global (" + cfg.globalLabel + ")") + (configured ? " · resolves to " + configured : "") + (c.provider ? " · now: " + c.provider : "") + (c.restartNeeded ? " · restart needed (" + c.pending.join("; ") + ")" : "")));
    if (m.error) { card.appendChild(el("p", "error", m.error)); return det2; }
    card.appendChild(el("p", m.recommended ? "notice" : "warn", m.recommended ? "Recommended: " + m.recommended + " — " + m.recommendation : "No recommendation: " + m.recommendation));
    if (m.changes && m.changes.baseline && m.changes.comparable && (m.changes.metrics || []).length) {
      var cd = det("chg:" + key, "What changed since the run of " + when(m.changes.baseline.timestamp) + (m.changes.baseline.pinned ? " (pinned baseline)" : "") + (m.changes.summary ? ": " + m.changes.summary : ""));
      var ct = table("stack", ["What", "Device", "Before", "After", "Change", "Why"]);
      m.changes.metrics.forEach(function (mc) {
        var tr = el("tr"); cell(tr, PB.METRIC_NAMES[mc.metric] || mc.metric, undefined, "What"); cell(tr, mc.device || "model", undefined, "Device");
        cell(tr, PB.metricValue(mc.metric, mc.before), undefined, "Before").className = "num"; cell(tr, PB.metricValue(mc.metric, mc.after), undefined, "After").className = "num";
        var cc = cell(tr, "", undefined, "Change"); cc.appendChild(PB.metricChip(mc)); cell(tr, mc.why || "", undefined, "Why");
        ct.tBodies[0].appendChild(tr);
      });
      cd.appendChild(wrap(ct, "chg:" + key)); card.appendChild(cd);
    } else if (m.changes && m.changes.baseline && !m.changes.comparable) {
      card.appendChild(el("p", "muted", "⚠ Not compared with the run of " + when(m.changes.baseline.timestamp) + ": " + ((m.changes.reasons || []).join("; ") || "the runs differ") + "."));
    }
    var t = sortable(table("devices", ["Device", "@load_ms", "Infer p50", "Infer mean", "Total p50", "Total p95", "@fps", "@ap50", "@ap50_95", "@roc_auc", "@precision", "@recall", "@grade", "@agreement", "@trust", ""]));
    var okRuns = m.devices.filter(function (d) { return d.run && !d.run.fell_back; });
    var fastest = okRuns.reduce(function (a, d) { return !a || d.run.stages_ms.total.p50 < a.run.stages_ms.total.p50 ? d : a; }, null);
    m.devices.forEach(function (d) {
      var tr = el("tr"), r = d.run, name = cell(tr, d.device, undefined, "Device");
      name.title = d.label;
      if (d === fastest && okRuns.length > 1) { tr.classList.add("fastest"); name.appendChild(el("span", "tag", "fastest")); }
      if (d.device === m.recommended) name.appendChild(el("span", "tag rec", "recommended"));
      if (d.device === configured) name.appendChild(el("span", "tag cfg", "configured"));
      if (running && running !== configured && d.device === running) { var rt = el("span", "tag", "running"); rt.title = "loaded on this device until a restart"; name.appendChild(rt); }
      if (!r || r.fell_back) {
        var e = el("td", "error-text", d.error || ("fell back to " + (r && r.device))); e.colSpan = 15; tr.classList.add("failed"); tr.appendChild(e);
      } else {
        var acc = r.accuracy ? r.accuracy.overall : null;
        cell(tr, r.compile_ms.toFixed(0), r.compile_ms, "Load ms"); cell(tr, ms(r.stages_ms.infer.p50), r.stages_ms.infer.p50, "Infer p50"); cell(tr, ms(r.stages_ms.infer.mean), r.stages_ms.infer.mean, "Infer mean");
        cell(tr, ms(r.stages_ms.total.p50), r.stages_ms.total.p50, "Total p50"); cell(tr, ms(r.stages_ms.total.p95), r.stages_ms.total.p95, "Total p95"); cell(tr, r.throughput_fps.toFixed(1), -r.throughput_fps, "req/s");
        var rauc = r.accuracy && r.accuracy.frame_roc ? r.accuracy.frame_roc.auc : null;
        [["AP50", acc && acc.ap50], ["AP50-95", acc && acc.ap50_95], ["ROC AUC", rauc], ["P", acc && acc.precision], ["R", acc && acc.recall]].forEach(function (p) { cell(tr, pct(p[1]), p[1] === null || p[1] === undefined ? null : -p[1], p[0]); });
        var g = cell(tr, "", r.grades ? 4 - r.grades.overall_points : null, "Grade");
        if (r.grades) { g.append(gradeEl(r.grades.overall), " ", gradeEl(r.grades.accuracy, r.grades.relative), " ", gradeEl(r.grades.speed)); g.title = "overall / accuracy" + metricTag(r.grades) + " / speed"; }
        var ag = cell(tr, verdictText(r.agreement), undefined, "Agreement");
        if (r.agreement && r.agreement.verdict === "disagrees") { tr.classList.add("bad"); ag.title = "detections differ from " + r.agreement.reference + "; never recommended"; }
        var tc = cell(tr, "", undefined, "Trust"); tc.appendChild(PB.trustBadge(d.trust, "dev"));
        var act = el("td", "actions-cell");
        if (!(c.device && c.device === d.device)) { var u = el("button", "small", "Use " + d.device); u.type = "button"; u.addEventListener("click", function () { useDevice(m.model, d.device); }); act.appendChild(u); }
        if (r.images) { var i = el("button", "small", "Images"); i.type = "button"; i.addEventListener("click", function () { drill(m.model, d.device); }); act.appendChild(i); }
        tr.appendChild(act);
      }
      t.tBodies[0].appendChild(tr);
    });
    card.appendChild(wrap(t, "dev:" + key));
    if (m.skipped && m.skipped.length) card.appendChild(el("p", "muted", "Not runnable here: " + m.skipped.join("; ")));
    card.appendChild(thresholdBlock(m));
    var rr = recRun(m) || (fastest && fastest.run); if (rr) card.appendChild(details(key, rr));
    return det2;
  }
  // A link to #m-<model> (the Config page) opens that model's card.
  window.addEventListener("hashchange", function () {
    var t2 = document.getElementById(location.hash.slice(1));
    if (t2 && t2.tagName === "DETAILS" && t2.classList.contains("model")) { t2.open = true; t2.scrollIntoView(); }
  });

  // Drill-down: draw ground truth and predictions on each image.
  var drillData = null;
  function drill(model, device) {
    $("drill-title").textContent = model + " on " + device;
    $("drill-body").replaceChildren(el("p", "muted", "loading…"));
    $("drill").showModal();
    fetch("/v1/benchmark/images?model=" + encodeURIComponent(model) + "&device=" + encodeURIComponent(device)).then(function (r) { return r.json(); }).then(function (j) { drillData = j; drawDrill(); });
  }
  function drawDrill() {
    var body = $("drill-body"), f = $("drill-filter").value; body.replaceChildren();
    if (!drillData || !drillData.success) { body.appendChild(el("p", "error", drillData ? drillData.message : "no data")); return; }
    if (drillData.groundTruth) body.appendChild(el("p", "muted thumbs-note", "Scored against " + drillData.groundTruth + "."));
    drillData.images.forEach(function (im) {
      var c = im.counts || {tp: 0, fp: 0, fn: 0};
      if (f === "miss" && !c.fn) return; if (f === "fp" && !c.fp) return; if (f === "err" && !c.fn && !c.fp) return;
      var fig = el("figure", "thumb"), cv = el("canvas"); fig.appendChild(cv);
      fig.appendChild(el("figcaption", null, im.file + " · " + ms(im.totalMs) + " ms · TP " + c.tp + " FP " + c.fp + " FN " + c["fn"] + (im.tags.length ? " · " + im.tags.join(", ") : "")));
      body.appendChild(fig);
      var img = new Image();
      img.onload = function () {
        var scale = Math.min(1, 640 / img.width); cv.width = Math.round(img.width * scale); cv.height = Math.round(img.height * scale);
        var g = cv.getContext("2d"); g.drawImage(img, 0, 0, cv.width, cv.height); g.lineWidth = 2; g.font = "12px system-ui";
        function box(b, color, dash, text) { g.setLineDash(dash ? [5, 4] : []); g.strokeStyle = color; g.strokeRect(b[0] * scale, b[1] * scale, (b[2] - b[0]) * scale, (b[3] - b[1]) * scale); if (text) { g.fillStyle = color; g.fillText(text, b[0] * scale + 2, Math.max(10, b[1] * scale - 3)); } }
        (im.objects || []).forEach(function (o, k) { var mt = im.gtMatched ? im.gtMatched[k] : null; if (mt === null || mt === undefined) { if (!o.ignore) box(o.bbox, "#9aa3b2", true, null); return; } box(o.bbox, mt ? "#2e9e4f" : "#e53935", !mt, mt ? null : "missed " + o.label); });
        im.preds.forEach(function (p) { var col = p.tp === true ? "#1f6fe0" : p.tp === false ? "#f08c00" : "#8a93a5"; box(p.bbox, col, false, p.label + " " + p.confidence.toFixed(2)); });
      };
      img.src = "/v1/benchmark/image?set=" + encodeURIComponent(im.set) + "&file=" + encodeURIComponent(im.file);
    });
    if (!body.children.length) body.appendChild(el("p", "muted", "No images match."));
  }
  $("drill-filter").addEventListener("change", drawDrill);
  $("drill-close").addEventListener("click", function () { $("drill").close(); });

  // ---- Run tab: estimated run time ---------------------------------------------------------------
  // From the last run's time per image of each selected model x device (load + warm-up + timed
  // runs); pairs without an earlier run are not counted.
  function estimate() {
    var out = $("estimate"); if (!out || !form) return;
    var models = Array.prototype.map.call(form.querySelectorAll("input[name=model]:checked"), function (c) { return c.value; });
    var devices = Array.prototype.map.call(form.querySelectorAll("input[name=device]:checked"), function (c) { return c.value; });
    var nsets = form.querySelectorAll("input[name=dataset]:checked").length + ($("add-folder").value.trim() ? 1 : 0);
    var maxImages = parseInt(form.elements.max_images.value, 10) || 0, repeat = Math.max(1, parseInt(form.elements.repeat.value, 10) || 1), warmup = Math.max(0, parseInt(form.elements.warmup.value, 10) || 0);
    var total = 0, known = 0, pairs = models.length * devices.length;
    models.forEach(function (name) {
      var m = modelOf(lastModels, name); if (!m) return;
      devices.forEach(function (dv) {
        var d = (m.devices || []).find(function (x) { return x.device === dv; }), r = d && d.run; if (!r || !r.stages_ms || !r.stages_ms.total) return;
        var n = maxImages > 0 ? maxImages * Math.max(1, nsets) : m.images; if (!n) return;
        var per = r.stages_ms.total.mean || r.stages_ms.total.p50;
        total += (r.compile_ms || 0) / 1000 + (warmup + repeat * n) * per / 1000; known++;
      });
    });
    if (!pairs) { out.textContent = ""; return; }
    if (!known) { out.textContent = "Estimated run time: unknown (no earlier run of these models on these devices)."; return; }
    out.textContent = "Estimated run time: about " + F.duration(total) + (known < pairs ? " for " + known + " of " + pairs + " model × device pairs (the others have no earlier run)" : "") + ", from the last run's time per image" + (maxImages > 0 ? "" : " and image count") + ". Downloads and the reference model add to it.";
  }
  if (form) { form.addEventListener("input", estimate); form.addEventListener("change", estimate); }

  // ---- step strip ------------------------------------------------------------------------------
  function steps(j, models) {
    var box = $("steps"); if (!box) return;
    var hasResults = models.some(function (m) { return m.recommended || m.threshold; });
    var pending = models.some(function (m) {
      var c = cfgOf(j.config, m.model), configured = c.configuredDevice || m.configured;
      var devOk = !m.recommended || c.device === m.recommended || (!c.device && configured === m.recommended);
      var thrOk = !applicable(m.threshold) || near(nowOf(m.threshold).threshold, m.threshold.best.threshold);
      return !(devOk && thrOk);
    });
    var cur = j.running || !hasResults ? "run" : pending ? "review" : "done";
    var order = ["run", "review", "apply"];
    box.querySelectorAll("li").forEach(function (li) {
      var s = li.dataset.step, i = order.indexOf(s), ci = cur === "done" ? 3 : order.indexOf(cur);
      li.classList.toggle("done", i < ci); li.classList.toggle("current", i === ci);
      if (i === ci) li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current");
    });
  }

  // ---- render loop ------------------------------------------------------------------------------
  // Each section re-renders only when its data changed, keeping the user's scroll offsets,
  // open <details>, sort order and focus (PrismLive.keep / patch); progress updates in place.
  var firstRender = true;
  // A run finished: say so and offer the results (the page stays on the tab the user is on).
  function finished(j) {
    msg.replaceChildren(document.createTextNode((j.run.message || j.run.error || "Benchmark finished.") + " "));
    if (j.run.error) return;
    var b = el("button", "small", "See results"); b.type = "button";
    b.addEventListener("click", function () { if (T) T.show("tab-overview"); if (history.replaceState) history.replaceState(null, "", "#tab-overview"); var o = $("tab-overview"); if (o) o.scrollIntoView(); });
    msg.appendChild(b);
  }
  function compareShown() { if (T && T.current() === T.indexOf("tab-compare") && PB.compare) PB.compare.shown(); }
  var nav = document.querySelector("nav.tabs");
  if (nav) nav.addEventListener("click", function () { setTimeout(compareShown, 0); });
  window.addEventListener("hashchange", function () { setTimeout(compareShown, 0); });
  function render(j) {
    PB.setNoise(j.noise);
    var models = merged(j), sm = JSON.stringify(models), sc = JSON.stringify(j.config), sr = JSON.stringify(j.ranking || []);
    lastConfig = j.config; lastModels = models; lastStatus = j;
    $("no-results").hidden = models.length > 0 || j.running;
    if (firstRender) {
      firstRender = false;
      // Without a #fragment the page opens on Run (the first tab); a link to a tab opens it.
      if (T && location.hash) { renderModels(models, j.config, j.ranking); T.show(location.hash.slice(1)); }
    }
    var running = j.running;
    L.keep($("run-banner"), JSON.stringify([models.map(function (m) { return [m.model, m.this_run, m.changes, m.timestamp, m.error, m.recommendation]; }), running, j.run && j.run.progress && [j.run.progress.modelIndex, j.run.progress.modelCount], (j.run && j.run.models || []).length, j.latest_run_id, sr]), function () { renderBanner(j, models); });
    var objs = JSON.stringify(models.map(function (m) { return m.threshold ? m.threshold.objective : null; }));
    L.keep($("preset"), JSON.stringify([j.config && j.config.benchmark, presetUi, running, models.length > 0, objs]), function () { renderPreset(j); });
    var sp = JSON.stringify(j.points || []), sel = "|" + selDevice + "|" + choiceKey() + "|" + Object.keys(conf.cache).join(",");
    // The shared best-confidence searches belong to these results and this objective.
    var stamp = savedObjective(j) + "|" + (j.latest_run_id || "") + "|" + ((j.results && j.results.timestamp) || "") + "|" + sm.length;
    if (stamp !== conf.stamp) { conf.stamp = stamp; conf.cache = {}; conf.pending = {}; }
    L.keep($("recommend"), sm + sc + sr + sp + sel + running, function () { renderRecommend(j, models); });
    L.keep($("devices"), sm + sc + sp + sel + running, function () { renderDevices(j, models); });
    L.keep($("ranking"), sm + sr + sp + sel, function () { renderRanking(j, models); });
    L.keep($("glossary"), JSON.stringify(PB.noise), function () { var g = $("glossary"); g.replaceChildren(PB.glossaryDetails("overview")); });
    L.keep($("conf-pick"), JSON.stringify([savedObjective(j), conf.obj, objs, running, devicesOf(models), selDevice, choiceKey()]), function () { renderConfPick(j); });
    L.keep($("conf-best"), sm + sc + sp + conf.obj + sel + running + Object.keys(conf.cache).join(","), function () { renderConfBest(j, models); });
    L.keep($("thresholds"), sm + sc, function () { renderThresholds(models); });
    renderModels(models, j.config, j.ranking);
    if (PB.compare) PB.compare.render(j, models);
    steps(j, models);
    estimate();
    $("apply-thr-bar").hidden = running || !models.some(function (m) { return applicable(m.threshold); });
    var p = j.run.progress;
    $("start").disabled = running; $("cancel").hidden = !running; $("progress").hidden = !running;
    if (running && p) {
      if (p.stage === "benchmark") {
        bar.value = p.percent;
        ptext.textContent = p.model + " (" + p.modelIndex + "/" + p.modelCount + ") on " + p.device + " · " + p.phase + (p.total ? " " + p.done + "/" + p.total : "") + (p.image ? " · " + p.image : "") + " · " + p.percent.toFixed(0) + "%" + (p.etaSecs ? " · about " + Math.ceil(p.etaSecs) + " s left" : "");
      } else { bar.removeAttribute("value"); ptext.textContent = p.text || p.stage; }
    } else if (running) { bar.removeAttribute("value"); ptext.textContent = "starting…"; }
    if (!running && lastState === "running") { finished(j); refreshSearch(); if (PB.compare) PB.compare.invalidate(); }
    else if (!running && !msg.textContent && j.run.state !== "idle") msg.textContent = j.run.message || j.run.error || "";
    var warnings = j.run.warnings || [];
    L.keep($("warnings"), JSON.stringify(warnings), function () { var w = $("warnings"); w.replaceChildren(); warnings.forEach(function (t) { w.appendChild(el("li", null, t)); }); });
    lastState = j.state;
    compareShown();
    PB.refreshTip();
    return running;
  }
  function refresh() {
    return fetch("/v1/benchmark", {cache: "no-store"}).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    }).then(render);
  }
  PB.refresh = refresh;
  function poll() {
    if (polling) return;
    polling = setInterval(function () {
      refresh().then(function (running) { if (!running) { clearInterval(polling); polling = null; } }).catch(function (e) { msg.textContent = "server not answering" + (e && /^HTTP /.test(e.message) ? " (" + e.message + ")" : ""); });
    }, 1000);
  }
  refresh().then(function (running) { if (running) poll(); }, function (e) { msg.textContent = "server not answering (" + e.message + ")"; });
  loadSearch();

  // Live config: another tab, the Config page or the CLI changed the config -> redraw the
  // configured / in-use markers and Apply buttons, and refresh the settings the user has not
  // edited (edited ones are kept and marked).
  if (window.PrismLive) {
    var U = PrismLive.ui, seenRev = null, seenEpoch = null;
    U.markEdits(form);
    PrismLive.onConfig(function (cfg) {
      if (seenRev === null) { seenRev = cfg.revision; seenEpoch = cfg.epoch; return; }
      if (cfg.revision === seenRev && cfg.epoch === seenEpoch) return;
      seenRev = cfg.revision; seenEpoch = cfg.epoch;
      var b = cfg.config.benchmark, w = b.weights || {}, share = (w.accuracy || 0) + (w.speed || 0) > 0 ? w.accuracy / (w.accuracy + w.speed) : 0.6;
      var show = function (v) { return v === "" || v === null ? "automatic" : String(v); };
      [["max_images", b.max_images_per_dataset], ["repeat", Math.max(1, b.repeat_per_image)], ["warmup", b.warmup], ["accuracy_weight", share.toFixed(2)],
       ["threshold_objective", b.threshold_objective], ["accuracy_metric", b.accuracy_metric], ["reference_model", b.reference_model || ""]].forEach(function (p) {
        var c = form.elements[p[0]]; if (c) U.refresh(c, p[1], show);
      });
      refresh();
      refreshSearch();
    });
  }
})();
