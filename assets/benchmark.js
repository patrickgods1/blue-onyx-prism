// Blue Onyx Prism: Benchmark page (served from /static/; no inline scripts under the CSP).
// The wizard (1 Run, 2 Profiles, 3 Apply, Advanced): the run loop and polling, the goal chooser
// (the only place that sets the goal), the Advanced sections (Confidence, Details; Compare and
// History are in bench-compare.js) and the page wiring. The profile cards and the Apply step are
// in bench-profiles.js, the vocabulary in bench-glossary.js, the charts in bench-charts.js, all
// shared through window.PrismBench (loaded before this file).
(function () {
  var PB = window.PrismBench, C = PB.charts, F = PB.fmt, has = PB.has;
  var $ = function (id) { return document.getElementById(id); };
  var form = $("bench-form"), msg = $("msg"), bar = $("bar"), ptext = $("ptext");
  var polling = null, lastState = null, lastConfig = null, lastModels = [], lastStatus = null;
  // Old links: the Overview (and the older Results) is step 2 Profiles now; Compare, Confidence
  // and Details (and the older Models) are sections of Advanced. Rewritten before the tabs read
  // the URL (and on later hash changes, before their listener runs); a link into a section of
  // Advanced opens that section.
  var ALIAS = {"#tab-overview": "#tab-profiles", "#tab-results": "#tab-profiles", "#tab-compare": "#adv-compare", "#tab-confidence": "#adv-confidence", "#tab-details": "#adv-details", "#tab-models": "#adv-details", "#tab-history": "#adv-history"};
  function openTo(id) { for (var t = id ? document.getElementById(id) : null; t; t = t.parentElement) if (t.tagName === "DETAILS") t.open = true; }
  function alias() {
    var h = ALIAS[location.hash];
    if (h) { if (history.replaceState) history.replaceState(null, "", h); else location.hash = h; }
    openTo(location.hash.slice(1));
  }
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
    // No device field would make the server fall back to the saved devices, unseen here.
    if (!form.querySelector("input[name=device]:checked")) { runMsg.textContent = "Tick at least one device to benchmark on."; return; }
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

  // ---- apply from Advanced, with before -> after and Undo --------------------------------------
  // Details "Use <device>" sets a model's device (its confidence then follows the device through
  // the saved profiles); the confidence actions write the profile of one model on one device.
  // Each answer is shown in the status box with one Undo.
  function applyDevice(model, device) { return post("/v1/benchmark/apply", new URLSearchParams({model: model, device: device})); }
  function useDevice(model, device) { return applyDevice(model, device).then(function (j) { showApply([{j: j, kind: "device"}]); }); }
  // The profile of `model` on `device` (the per-device confidence).
  function useProfile(model, device, t) { return PB.profiles.setProfiles([[model, device, t]], $("apply-msg")); }
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
      return c.model + ": default confidence " + from + " → " + F.thrBi(c.to);
    }
    return c.model + ": device " + (c.from || "global") + " → " + (c.to || "global");
  }
  // steps: [{j, kind: "device" | "threshold"}] (or the old (j, kind, undone) form).
  function showApply(steps, kind, undone) {
    if (!Array.isArray(steps)) steps = [{j: steps, kind: kind}]; else undone = kind;
    var box = $("apply-msg"), ok = steps.every(function (s) { return s.j && s.j.success; });
    box.hidden = false; box.className = ok ? "notice apply-box" : "error apply-box"; box.replaceChildren();
    var all = [];
    steps.forEach(function (s) { (Array.isArray(s.j.changes) ? s.j.changes : []).forEach(function (c) { if (c && c.model) all.push({c: c, kind: s.kind, j: s.j}); }); });
    if (all.length) {
      box.appendChild(el("strong", null, undone ? "Undone: " : "Saved: "));
      var ul = el("ul", "apply-list"); all.forEach(function (x) { ul.appendChild(el("li", null, describeChange(x.c, x.kind))); }); box.appendChild(ul);
    }
    steps.forEach(function (s) { if (s.j.message) box.appendChild(el("span", "muted", s.j.message + " ")); });
    if (all.length && !undone) {
      var u = el("button", "small secondary", "Undo"); u.type = "button";
      u.addEventListener("click", function () { u.disabled = true; undoAll(all); });
      box.appendChild(u);
    }
    // One restart rule for every answer: the server's needsRestart (older servers: the message).
    var last = steps[steps.length - 1] && steps[steps.length - 1].j;
    if (last) PB.profiles.restartControls({success: ok, restarting: steps.some(function (s) { return s.j.restarting; }),
      needsRestart: steps.some(function (s) { return s.j.success && (s.j.needsRestart === true || (s.j.needsRestart === undefined && /Restart the server to apply/.test(s.j.message || ""))); })}, box);
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
  // After a profile save (bench-profiles.js): the confidence search compares with the new values.
  PB.afterApply = function () { refreshSearch(); };
  $("apply-all-thr").addEventListener("click", function () {
    if (!confirm("Write the best confidence of every benchmarked model (on its recommended device) as the model's default confidence in the config? Devices with a saved profile keep using their profile. It applies at once.")) return;
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
  // One notice when the stored confidences were found for another goal than the saved one
  // (a goal change re-ranks the stored results; this stays only when they cannot be).
  var STALE = "Run again to get confidences for this goal.";
  function renderStale(j, models) {
    var box = $("stale-goal"); if (!box) return;
    box.replaceChildren();
    if (j.running || !staleAdvice(j, models).length) return;
    var p = el("p", "warn stale-notice", STALE + " ");
    var b = el("button", "small secondary", "Re-rank the stored results"); b.type = "button";
    b.title = "Saves the goal again and re-ranks the stored results (no new run); confidences that need a new run stay as they are";
    b.addEventListener("click", function () { b.disabled = true; presetUi.msg = "saving\u2026"; saveSettings({}); });
    p.appendChild(b); box.appendChild(p);
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
  // Save the best confidence as the profile of the advice's device, or "saved" when it is.
  function thrAction(model, a, label) {
    var b = a.best, n = nowOf(a), saved = PB.profiles.saved(model, a.device);
    if (!applicable(a)) return null;
    if (!near(has(saved) ? saved : n.threshold, b.threshold)) {
      var u = el("button", "small", label + thr(b.threshold)); u.type = "button";
      u.title = "Save " + thr(b.threshold) + " as the profile of " + model + " on " + a.device;
      u.addEventListener("click", function () { u.disabled = true; useProfile(model, a.device, b.threshold); }); return u;
    }
    return el("span", "muted", "saved");
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
      var bc = cell(tr, b ? thr(b.threshold) : "-", b ? b.threshold : null); if (applicable(a)) bc.appendChild(el("span", "tag rec", "usable"));
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
    if (!confirm("Save the searched best confidence of " + rs.length + " model(s) as the profile of the device it was searched on? It applies at once.")) return;
    PB.profiles.setProfiles(rs.map(function (r) { return [r.model, r.device, r.best.threshold]; }), $("apply-msg"));
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
      var savedT = PB.profiles.saved(r.model, r.device);
      if (applicableSearch(r) && !near(has(savedT) ? savedT : c.threshold, b.threshold)) { var u = el("button", "small", "Apply " + thr(b.threshold)); u.type = "button"; u.title = "Save " + thr(b.threshold) + " as the profile of " + r.model + " on " + r.device; u.addEventListener("click", function () { u.disabled = true; useProfile(r.model, r.device, b.threshold); }); act.appendChild(u); }
      else if (applicableSearch(r)) act.appendChild(el("span", "muted", "saved"));
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

  // ---- step 2: the goal (the only place that sets it) --------------------------------------
  var PRESETS = [
    {id: "balanced", name: "Balanced", text: "Finding objects and avoiding false alerts count the same; speed counts 40% (the default).", obj: "f1", w: 0.6},
    {id: "miss", name: "Miss fewer objects", text: "Catching every person or car matters most; some extra false alerts are fine.", obj: "f2", w: 0.7},
    {id: "false", name: "Fewer false alerts", text: "At least 9 in 10 detections must be right; then find as many objects as possible.", obj: "precision:0.9", w: 0.5},
    {id: "fast", name: "Fastest / lightest", text: "Keep the CPU or GPU free for Blue Iris and other work: speed counts 80%.", obj: "f1", w: 0.2},
    {id: "custom", name: "Custom", text: "Set the accuracy weight, what the confidence optimizes and the accuracy measure yourself.", obj: null, w: null}
  ];
  var presetUi = {custom: false, msg: "", busy: false};
  function presetOf(b) {
    if (!b) return "custom";
    var w = weightOf(b), o = String(b.threshold_objective || "f1");
    var p = PRESETS.find(function (x) { return x.obj === o && Math.abs(x.w - w) < 0.011; });
    return p ? p.id : "custom";
  }
  // Saves the goal fields (and the rest of the Run form, as "Save as default" does) and
  // re-ranks the stored results.
  function saveSettings(fields, done) {
    Object.keys(fields).forEach(function (k) { var c = form.elements[k]; if (c) { c.value = fields[k]; c.classList.remove("edited"); } });
    presetUi.busy = true; renderGoalMsg();
    return post("/v1/benchmark/settings", settingsBody()).then(function (j) {
      presetUi.busy = false;
      presetUi.msg = j.success ? (j.regraded === true ? "Saved: the profiles and the ranking are updated from the stored results (no new run)." : j.regraded === false ? "Saved. The profiles change with the next run (the stored results cannot be re-ranked)." : "Saved.") : (j.message || "not saved");
      presetUi.ok = !!j.success;
      if (done) done(j);
      refresh(); refreshSearch();
    });
  }
  function renderGoalMsg() { var m = $("goal-msg"); if (m) { m.className = presetUi.ok === false ? "error-text" : "muted"; m.textContent = presetUi.busy ? "saving…" : presetUi.msg; } }
  function renderPreset(j) {
    var box = $("goal-presets"); if (!box) return;
    box.replaceChildren();
    var b = j.config && j.config.benchmark, cur = presetUi.custom ? "custom" : presetOf(b);
    var fs = el("fieldset", "presets"); fs.appendChild(el("legend", "visually-hidden", "What matters most"));
    PRESETS.forEach(function (p) {
      var l = el("label", "preset" + (cur === p.id ? " on" : "")), i = el("input"); i.type = "radio"; i.name = "preset"; i.value = p.id; i.checked = cur === p.id; i.disabled = presetUi.busy || !!(j.running);
      i.addEventListener("change", function () {
        if (!i.checked) return;
        if (p.id === "custom") { presetUi.custom = true; renderPreset(lastStatus); return; }
        presetUi.custom = false; presetUi.msg = "";
        saveSettings({accuracy_weight: p.w.toFixed(2), threshold_objective: p.obj});
        renderPreset(lastStatus);
      });
      var t = el("span", "preset-text"); t.appendChild(el("strong", null, p.name)); t.appendChild(el("span", "muted", p.text));
      l.appendChild(i); l.appendChild(t); fs.appendChild(l);
    });
    box.appendChild(fs);
    if (j.running) box.appendChild(el("p", "muted", "The goal can change when the benchmark finishes."));
    var custom = $("goal-custom"); if (custom) custom.hidden = cur !== "custom";
    var gs = $("goal-save"); if (gs) gs.disabled = presetUi.busy || !!j.running;
    renderGoalMsg();
  }
  // The Custom panel's controls belong to the Run form (form="bench-form"): Save goal sends them.
  var goalSave = $("goal-save");
  if (goalSave) goalSave.addEventListener("click", function () {
    presetUi.msg = "";
    saveSettings({}, function () { presetUi.custom = presetOf(lastStatus && lastStatus.config && lastStatus.config.benchmark) === "custom"; });
  });

  // ---- Advanced: the shared device selection (Confidence) ------------------------------------
  // "chosen" = the device picked for each model under Run on (step 2, PrismBench.choice), "rec"
  // = the recommended device of each model, "all" = every device, else a device spec.
  // Remembered per browser.
  var selDevice = PB.store.get("view", "chosen") || "chosen";
  function selName() { return selDevice === "chosen" ? "the chosen device of each model" : selDevice === "rec" ? "the recommended device of each model" : selDevice === "all" ? "every device" : selDevice; }
  function ranOn(m, device) { return (m.devices || []).some(function (d) { return d.device === device && d.run && !d.run.fell_back; }); }
  // The device chosen in step 2; without a choice (or when the model did not run there): the
  // device it uses now when it ran there, else its recommended device, else the first one.
  function chosenDevice(j, m) {
    var p = ((j && j.profiles) || []).find(function (x) { return x.model.toLowerCase() === m.model.toLowerCase(); });
    var c = p ? PB.profiles.chosen(p) : PB.choice.get(m.model);
    if (c && ranOn(m, c)) return c;
    var now = nowDevice(j, m);
    if (now && ranOn(m, now)) return now;
    if (m.recommended && ranOn(m, m.recommended)) return m.recommended;
    var d = (m.devices || []).find(function (x) { return x.run && !x.run.fell_back; });
    return d ? d.device : null;
  }
  function devicesOf(models) {
    var out = []; models.forEach(function (m) { (m.devices || []).forEach(function (d) { if (d.run && out.indexOf(d.device) < 0) out.push(d.device); }); });
    return out.sort();
  }
  // "Show: [...]": changing it redraws the Confidence views.
  function deviceSelector(models, scope, text) {
    var l = el("label", "inline-label device-select"), s = el("select");
    l.appendChild(document.createTextNode((text || "Show") + " "));
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
    L.keep($("conf-pick"), undefined, function () { renderConfPick(j); });
    L.keep($("conf-best"), undefined, function () { renderConfBest(j, models); });
  }
  PB.choice.onChange(redrawSelection);
  // The best pick of model m on `device` (the advice's own device, else its by_device entry).
  function devPick(m, device) {
    var a = m.threshold; if (!a) return null;
    if (a.device === device && a.best) return a.best;
    var x = (a.by_device || []).find(function (z) { return z.device === device; });
    return x ? x.best : null;
  }
  // The device the config resolves model m to now.
  function nowDevice(j, m) { var c = cfgOf(j.config, m.model); return c.effective || c.device || c.configuredDevice || m.configured || null; }
  function pointOf(j, model, device) { return (j.points || []).find(function (p) { return p.model === model && p.device === device; }) || null; }
  // The best confidence of model m on `device` for the saved goal: the stored pick when it was
  // found for that goal, else the threshold search for it on that device (fetched once into
  // conf.cache, cleared when the results or the goal change), else the stored pick for another
  // goal ("fallback": the section then shows one "Run again" notice).
  // {pick, state: "ok" | "loading" | "fallback" | "none"}.
  function bestPick(j, m, device) {
    var a = m.threshold; if (!a) return {pick: null, state: "none"};
    var cfg = savedObjective(j);
    if (String(a.objective) === cfg) return {pick: devPick(m, device), state: "ok"};
    var s = conf.cache[cfg + "|" + device];
    if (!s) { confSearch(cfg, device); return {pick: null, state: "loading"}; }
    var x = s.error ? null : (s.results || []).find(function (q) { return q.model.toLowerCase() === m.model.toLowerCase() && q.device === device; });
    if (x) return {pick: x.best, state: "ok"};
    var own = devPick(m, device);
    return own ? {pick: own, state: "fallback"} : {pick: null, state: "none"};
  }
  // One row per model x device that ran: speed, accuracy, the device's best confidence, trust.
  function devRows(j, models) {
    var out = [];
    models.forEach(function (m) {
      var now = nowDevice(j, m), picked = chosenDevice(j, m);
      (m.devices || []).forEach(function (d) {
        var r = d.run; if (!r) return;
        var p = pointOf(j, m.model, d.device), g = r.grades || {}, acc = r.accuracy, ov = acc && acc.overall, bp = bestPick(j, m, d.device), pk = bp.pick;
        var pt = pk && pk.point, rec = m.recommended === d.device;
        out.push({
          key: m.model.toLowerCase() + "|" + d.device, m: m, d: d, model: m.model, device: d.device, runtime: PB.runtimeOf(d.device, d.runtime),
          p50: p ? p.p50_ms : has(g.p50_ms) && g.p50_ms < 1e300 ? g.p50_ms : (r.stages_ms && r.stages_ms.total ? r.stages_ms.total.p50 : null),
          relative: !!(p ? p.relative : g.relative),
          pick: pk, thr: pk ? pk.threshold : null, recall: pt ? pt.recall : ov ? ov.recall : null, precision: pt ? pt.precision : ov ? ov.precision : null,
          pickState: bp.state,
          trust: (p && p.trust) || d.trust || (rec ? m.trust : null), rec: rec, now: now === d.device, chosen: picked === d.device, fellBack: !!r.fell_back, thisRun: m.this_run
        });
      });
    });
    return out;
  }
  function selectedRow(r) { return selDevice === "all" ? true : selDevice === "chosen" ? r.chosen : selDevice === "rec" ? r.rec : r.device === selDevice; }
  // The tags of a device cell: recommended, now, fell back (the same words as step 2).
  function deviceCell(r) {
    var s = el("span", "nowrap"); s.appendChild(document.createTextNode(r.device));
    if (r.rec) { var t0 = el("span", "tag rec", "Recommended"); t0.title = "The best device for this model and your goal"; s.appendChild(t0); }
    if (r.now) s.appendChild(el("span", "tag now", "Now"));
    if (r.fellBack) s.appendChild(el("span", "tag restart", "fell back"));
    s.title = (PB.RUNTIME_NAMES[r.runtime] || r.runtime) + (r.d.label ? " · " + r.d.label : "");
    return s;
  }
  var FILTERS = {model: function (r) { return r.model; }, device: function (r) { return r.device; }, runtime: function (r) { return PB.RUNTIME_NAMES[r.runtime] || r.runtime; }, thisRun: function (r) { return r.thisRun; }};
  // Confidence in a table cell: "0.54" with "= 54% in Blue Iris" as its tooltip.
  function thrShort(v) { return has(v) ? PB.conf(v) : "-"; }
  function savedTag() { return el("span", "tag now", "saved"); }

  // ---- Advanced: Confidence ------------------------------------------------------------------
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
  // The goals to try here: the same list as the goal's Custom panel (the saved one first).
  function goalOptions(saved) {
    var sel = form.elements.threshold_objective, out = [];
    if (sel && sel.options) Array.prototype.forEach.call(sel.options, function (o) { out.push([o.value, o.textContent]); });
    if (!out.some(function (o) { return o[0] === saved; })) out.unshift([saved, PB.objectiveName(saved)]);
    return out;
  }
  function renderConfPick(j) {
    var box = $("conf-pick"); box.replaceChildren();
    var saved = savedObjective(j); if (!conf.obj) conf.obj = saved;
    var card = el("div", "conf-tools");
    var h = el("h2", null, "Best confidence by device "); h.appendChild(PB.term("?", "best_conf", "conf")); card.appendChild(h);
    card.appendChild(el("p", "muted", "From the stored predictions of the last run, so switching is instant (no inference). Apply saves the profile of that model on that device."));
    var tools = el("div", "chart-tools");
    var gl = el("label", "inline-label"), gs = el("select"); gs.name = "conf-goal";
    gl.appendChild(document.createTextNode("Try another goal (doesn't change your saved goal) "));
    goalOptions(saved).forEach(function (o) { var x = el("option", null, o[1] + (o[0] === saved ? " (your goal)" : "")); x.value = o[0]; if (o[0] === conf.obj) x.selected = true; gs.appendChild(x); });
    gs.addEventListener("change", function () { conf.obj = gs.value; L.keep($("conf-best"), undefined, function () { renderConfBest(lastStatus, lastModels); }); });
    gl.appendChild(gs); tools.appendChild(gl);
    tools.appendChild(deviceSelector(lastModels, "conf"));
    card.appendChild(tools);
    box.appendChild(card);
  }
  // The pick of `obj` for a row: the stored per-device pick (same objective) or a search result.
  // {best, saved, other (why the search is not used), pending}
  function confPick(r, obj) {
    var a = r.m.threshold;
    if (a && String(a.objective) === obj) return {best: r.pick, saved: true};
    if (obj === savedObjective(lastStatus)) return r.pickState === "loading" ? {pending: true} : r.pickState === "fallback" ? {best: r.pick, saved: false, other: true} : {best: r.pick, saved: false};
    var k = obj + "|" + r.device, s = conf.cache[k];
    if (!s) return {pending: true};
    if (s.error) return {best: devPick(r.m, r.device), saved: false, other: true};
    var x = (s.results || []).find(function (q) { return q.model.toLowerCase() === r.model.toLowerCase() && q.device === r.device; });
    return x ? {best: x.best, saved: false} : {best: null, saved: false, missing: true};
  }
  function renderConfBest(j, models) {
    var box = $("conf-best"); box.replaceChildren();
    var saved = savedObjective(j), obj = conf.obj || saved;
    var all = devRows(j, models).filter(function (r) { return r.m.threshold && !r.fellBack; }), rows = all.filter(selectedRow);
    if (!all.length) { box.appendChild(el("p", "empty", models.length ? "No confidence results: the benchmark needs datasets with labels (or a reference model) to find the best confidence." : "No results yet: run a benchmark in 1 Run; this section then shows the best confidence for each model and device.")); return; }
    // Searches for an objective the stored advice was not found for (per device when chosen).
    if (rows.some(function (r) { return String(r.m.threshold.objective) !== obj; })) {
      rows.forEach(function (r) { confSearch(obj, r.device); });
    }
    var picked = rows.map(function (r) { var c = confPick(r, obj); return Object.assign({}, r, {cp: c, b: c.best || null}); });
    // Values found for another goal: one notice, no per-value marks.
    if (picked.some(function (r) { return r.cp.other; })) box.appendChild(el("p", "warn stale-notice", STALE));
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
          if (findsNothing(r.b)) { td.title = FINDS_NOTHING; td.classList.add("warn-cell"); return "finds no objects"; }
          var s = el("span"); s.appendChild(PB.conf(r.b.threshold, "best"));
          if (!r.b.met) s.appendChild(PB.tip("target not met", "The target is not reached at any confidence; this is the closest one." + (r.b.note ? " " + r.b.note : ""), "tag restart", "conf"));
          return s; }},
        {id: "range", head: "@range", value: function (r) { return r.b && r.b.plateau ? r.b.plateau[1] - r.b.plateau[0] : null; }, render: function (r) {
          var b = r.b; if (!b || !b.plateau || findsNothing(b)) return "-";
          var lo = b.plateau[0].toFixed(2), hi = b.plateau[1].toFixed(2), s = el("span", null, lo === hi ? lo : lo + "–" + hi);
          if (lo === hi) s.appendChild(PB.tip("sensitive", "Only a narrow range scores within " + PB.noise.plateau_points + " points of the best: small changes of the confidence matter for this model.", "tag restart", "conf"));
          return s; }},
        {id: "recall", head: "@finds", num: true, first: "desc", value: function (r) { return r.b && r.b.point ? r.b.point.recall : null; }, render: function (r) { return r.b && !findsNothing(r.b) ? F.pct(r.b.point.recall) : "-"; }},
        {id: "precision", head: "@correct", num: true, first: "desc", value: function (r) { return r.b && r.b.point ? r.b.point.precision : null; }, render: function (r) { return r.b && !findsNothing(r.b) ? F.pct(r.b.point.precision) : "-"; }},
        {id: "fpr", head: "@fpr", num: true, value: function (r) { return r.b && r.b.point && r.b.point.frame ? r.b.point.frame.fpr : null; }, render: function (r) { return r.b && r.b.point && r.b.point.frame && !findsNothing(r.b) ? F.pct(r.b.point.frame.fpr) : "-"; }},
        {id: "saved", head: "@saved_conf", num: true, value: function (r) { return PB.profiles.saved(r.model, r.device); }, render: function (r) { var v = PB.profiles.saved(r.model, r.device); return has(v) ? thrShort(v) : "not saved"; }},
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
          var t = r.b.threshold;
          if (near(PB.profiles.saved(r.model, r.device), t)) return savedTag();
          var b = el("button", "small nowrap", "Apply " + thr(t)); b.type = "button";
          b.title = "Save " + thr(t) + " as the profile of " + r.model + " on " + r.device + " (= " + Math.round(t * 100) + "% in Blue Iris); the model uses it whenever it runs on " + r.device;
          b.setAttribute("aria-label", b.title);
          b.disabled = !!j.running;
          b.addEventListener("click", function () { b.disabled = true; useProfile(r.model, r.device, t); });
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
  // One collapsible card per model (id "d-<model>"; "m-<model>" is its step-2 profile card),
  // patched in place: only cards whose data changed are rebuilt (keeping their scroll offsets,
  // open <details> and sort order). The first card (or the one the URL names) starts open; a
  // rebuilt card keeps its open state.
  function renderModels(models, cfg, ranking) {
    var box = $("models");
    $("no-models").hidden = models.length > 0;
    var cfgByName = {}; (cfg.models || []).forEach(function (c) { cfgByName[c.name.toLowerCase()] = c; });
    var hashId = location.hash.slice(1), named = models.some(function (m) { return "d-" + encodeURIComponent(m.model) === hashId; });
    L.patch(box, models.map(function (m, i) {
      var c = cfgByName[m.model.toLowerCase()] || {}, id = "d-" + encodeURIComponent(m.model);
      var rk = (ranking || []).find(function (r) { return r.model === m.model; }), metric = rk && rk.metric === "roc_auc" ? "roc_auc" : "ap50";
      var was = document.getElementById(id), open = was ? was.open : (named ? id === hashId : i === 0);
      return {key: m.model.toLowerCase(), sig: JSON.stringify([m, c, cfg.globalLabel, metric]), build: function () { var d = modelCard(m, c, cfg, metric); d.open = open; return d; }};
    }));
    if (named && !box.dataset.scrolled) { var t2 = document.getElementById(hashId); if (t2) { openTo(hashId); t2.scrollIntoView(); box.dataset.scrolled = "1"; } }
  }
  function modelCard(m, c, cfg, metric) {
    var det2 = el("details", "card model"), key = m.model.toLowerCase(); det2.id = "d-" + encodeURIComponent(m.model);
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
      if (d.device === m.recommended) name.appendChild(el("span", "tag rec", "Recommended"));
      if (d.device === configured) name.appendChild(el("span", "tag cfg", "Now"));
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
        if (!(c.device && c.device === d.device)) { var u = el("button", "small", "Use " + d.device); u.type = "button"; u.title = "Run " + m.model + " on " + d.device + "; its confidence follows the device through the saved profiles (save them in 3 Apply)"; u.addEventListener("click", function () { u.disabled = true; useDevice(m.model, d.device); }); act.appendChild(u); }
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
  // A link to #d-<model> opens that model's Details card (and the Details section).
  window.addEventListener("hashchange", function () {
    var t2 = document.getElementById(location.hash.slice(1));
    if (t2 && t2.tagName === "DETAILS" && t2.classList.contains("model")) { openTo(t2.id); t2.scrollIntoView(); }
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

  // ---- Save report (Advanced) -------------------------------------------------------------------
  // Writes the HTML report of the saved results into the chosen folder; the last folder used is
  // remembered in this browser (PB.store wraps localStorage in try/catch).
  (function () {
    var rf = $("report-form"), dir = $("report-dir"), save = $("report-save"), out = $("report-msg");
    if (!rf || !dir || !save) return;
    rf.addEventListener("submit", function (e) { e.preventDefault(); });
    if (!dir.value) dir.value = PB.store.get("reportDir", "") || "";
    save.addEventListener("click", function () {
      var d = dir.value.trim();
      if (!d) { out.textContent = "Choose a folder first (Browse…)."; return; }
      save.disabled = true; out.textContent = "saving…";
      post("/v1/benchmark/report", new URLSearchParams({dir: d})).then(function (j) {
        out.textContent = j.message || (j.success ? "Report saved to " + j.path + "." : "The report was not saved.");
        if (j.success) PB.store.set("reportDir", d);
      }).catch(function () { out.textContent = "Cannot reach the server."; }).then(function () { save.disabled = false; });
    });
  })();

  // ---- render loop ------------------------------------------------------------------------------
  // Each section re-renders only when its data changed, keeping the user's scroll offsets,
  // open <details>, sort order and focus (PrismLive.keep / patch); progress updates in place.
  var firstRender = true;
  // A run finished: say so and offer the results (the page stays on the tab the user is on).
  function finished(j) {
    msg.replaceChildren(document.createTextNode((j.run.message || j.run.error || "Benchmark finished.") + " "));
    if (j.run.error) return;
    var b = el("button", "small", "See results"); b.type = "button";
    b.addEventListener("click", function () { if (T) T.show("tab-profiles"); if (history.replaceState) history.replaceState(null, "", "#tab-profiles"); var o = $("tab-profiles"); if (o) o.scrollIntoView(); });
    msg.appendChild(b);
  }
  // Compare and History (bench-compare.js) load the run history once either is visible.
  function compareShown() {
    if (!T || !PB.compare || T.current() !== T.indexOf("tab-advanced")) return;
    var c = $("adv-compare"), h = $("adv-history");
    if ((c && c.open) || (h && h.open)) PB.compare.shown();
  }
  var nav = document.querySelector("nav.tabs");
  if (nav) nav.addEventListener("click", function () { setTimeout(compareShown, 0); });
  window.addEventListener("hashchange", function () { setTimeout(compareShown, 0); });
  ["adv-compare", "adv-history"].forEach(function (id) { var d = $(id); if (d) d.addEventListener("toggle", compareShown); });
  function render(j) {
    PB.setNoise(j.noise);
    var models = merged(j), sm = JSON.stringify(models), sc = JSON.stringify(j.config), sr = JSON.stringify(j.ranking || []);
    lastConfig = j.config; lastModels = models; lastStatus = j;
    var hasResults = models.length > 0 || ((j.profiles || []).length > 0);
    $("no-results").hidden = hasResults || j.running;
    var gl = $("goal-label"); if (gl && j.goal && j.goal.label) gl.textContent = j.goal.label;
    if (PB.profiles) PB.profiles.render(j);
    if (firstRender) {
      firstRender = false;
      // A link to a tab or into one (#m-<model> = a profile card, #d-<model> = a Details card)
      // opens it; without one the page opens on Profiles when there are results, else on Run.
      var target = location.hash.slice(1);
      if (target) renderModels(models, j.config, j.ranking);
      if (T && target && T.show(target)) { openTo(target); var te = $(target); if (te && te.getAttribute("role") !== "tabpanel") te.scrollIntoView(); }
      else if (T && hasResults) T.show("tab-profiles");
    }
    var running = j.running;
    L.keep($("run-banner"), JSON.stringify([models.map(function (m) { return [m.model, m.this_run, m.changes, m.timestamp, m.error, m.recommendation]; }), running, j.run && j.run.progress && [j.run.progress.modelIndex, j.run.progress.modelCount], (j.run && j.run.models || []).length, j.latest_run_id, sr]), function () { renderBanner(j, models); });
    var objs = JSON.stringify(models.map(function (m) { return m.threshold ? m.threshold.objective : null; }));
    L.keep($("goal-presets"), JSON.stringify([j.config && j.config.benchmark, presetUi, running]), function () { renderPreset(j); });
    L.keep($("stale-goal"), JSON.stringify([savedObjective(j), objs, running]), function () { renderStale(j, models); });
    var sp = JSON.stringify(j.points || []), sel = "|" + selDevice + "|" + PB.choice.key() + "|" + JSON.stringify(j.profiles || null) + "|" + Object.keys(conf.cache).join(",");
    // The shared best-confidence searches belong to these results and this objective.
    var stamp = savedObjective(j) + "|" + (j.latest_run_id || "") + "|" + ((j.results && j.results.timestamp) || "") + "|" + sm.length;
    if (stamp !== conf.stamp) { conf.stamp = stamp; conf.cache = {}; conf.pending = {}; }
    L.keep($("glossary"), "plain", function () { $("glossary").replaceChildren(PB.glossaryDetails("profiles", "plain")); });
    L.keep($("adv-glossary"), JSON.stringify(PB.noise), function () { $("adv-glossary").replaceChildren(PB.glossaryDetails("advanced")); });
    L.keep($("conf-pick"), JSON.stringify([savedObjective(j), conf.obj, objs, running, devicesOf(models), selDevice]), function () { renderConfPick(j); });
    L.keep($("conf-best"), sm + sc + sp + conf.obj + sel + running + Object.keys(conf.cache).join(","), function () { renderConfBest(j, models); });
    L.keep($("thresholds"), sm + sc + JSON.stringify(j.profiles || null), function () { renderThresholds(models); });
    renderModels(models, j.config, j.ranking);
    if (PB.compare) PB.compare.render(j, models);
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
