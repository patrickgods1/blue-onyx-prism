// Blue Onyx Prism: Benchmark step 2, Compare (templates/benchmark/compare.html; after
// bench-charts.js and bench-shell.js). Draws GET /v1/benchmark/compare?set=&scene= (plus a goal
// tried without saving, BenchShell.withGoal): the left-out line, the headline, the ranked table
// (PrismSort; `data-best` = the server's order, so "Best first" is the server's rank), optional
// columns (remembered in this browser), expandable details, Save report / Download CSV, the
// combos not run on these images (with Run on these images), the scatter (goal metric against
// p95, log scale) and the breakdown tabs (Day / Night / IR, Classes, Cameras, Speed).
// The address keeps `set` and `scene` (history.replaceState); `?model=` opens that model's rows.
(function () {
  "use strict";
  var BC = window.BenchCore, PB = window.PrismBench, S = window.BenchShell, C = PB.charts, el = PB.el, fmt = BC.fmt;
  if (!S || !C) return;
  var $ = function (id) { return document.getElementById(id); };
  var params = new URLSearchParams(location.search);
  var state = {set: params.get("set") || "", scene: params.get("scene") || "all", open: {}, tab: "scenes", pair: null, view: null, loadError: false};
  var loads = BC.latest();
  var focusModel = params.get("model");
  var columns = BC.columnsOf(BC.store.get("columns", []));
  var table = $("c-table"), body = $("cmp-body"), empty = $("cmp-empty");
  if (["all", "day", "night", "ir"].indexOf(state.scene) < 0) state.scene = "all";
  $("c-scene").value = state.scene;

  function keyOf(r) { return r.model + "\u0000" + r.device; }
  function isCpu(d) { return /(^|:)cpu$/.test(String(d)); }
  function colors(rows) {
    var by = {}, n = 0;
    rows.forEach(function (r) { if (!by[r.model]) by[r.model] = (n++ % 8) + 1; });
    return by;
  }
  function chipsOf(r) {
    return BC.rowChips(r).map(function (c) {
      var b = el("button", {type: "button", class: "term " + c.cls, "data-tip": c.tip, "aria-expanded": "false"});
      b.textContent = c.text; return b;
    });
  }
  function metricCell(r, goal) {
    if (goal.optimizeFor === "grade") return el("span", {text: r.grade || "-"});
    var m = r.metric;
    return el("span", {}, [fmt.n2(m && m.value), m && m.low !== null ? el("span", {class: "bn-ci", text: fmt.ci(m)}) : null]);
  }
  function td(label, content, cls, v) {
    var c = el("td", {class: cls || null, "data-label": label});
    if (content && typeof content === "object") c.appendChild(content); else c.textContent = content === undefined || content === null ? "-" : content;
    if (v !== undefined) c.setAttribute("data-v", BC.has(v) ? String(v) : "");
    return c;
  }
  function th(text, col, num, first, term) {
    var h = el("th", {scope: "col", class: num ? "num" : null, "data-sort-col": col});
    if (first) h.setAttribute("data-sort-first", first);
    if (term) h.appendChild(PB.term(text, term)); else h.textContent = text;
    return h;
  }
  function speedOf(r) { var s = r.speed || {}; return BC.has(s.burstP95Ms) ? s.burstP95Ms : s.p95Ms; }

  // ---- the table ---------------------------------------------------------------------------------
  function renderTable(v) {
    var goal = v.goal, rows = v.rows, color = colors(rows);
    var metricName = goal.metric + (goal.rankOnly ? "" : " (cross-validated)");
    var head = el("tr", {}, [th("#", "rank", true, null, "rank"), th("Model", "model"), th("Device", "device"), th(metricName, "metric", true, "desc", goal.rankOnly ? null : "oof"),
      th("Recall", "recall", true, "desc", "recall"), th("Precision", "precision", true, "desc", "precision"), th("False alerts /100 empty", "fa100", true, "asc", "fa100"),
      th("Confidence", "conf", true, null, "confidence"), th(rows.some(function (r) { return r.speed && BC.has(r.speed.burstP95Ms); }) ? "p95 (burst when run)" : "p95", "p95", true, "asc", "p95")]);
    columns.forEach(function (k) { var c = BC.COLUMNS.filter(function (x) { return x.key === k; })[0]; head.appendChild(th(c.label, "x-" + k, ["first_load", "cpu"].indexOf(k) < 0, null, ({f1: "f1", f05: "f05", ap50: "ap50", ap50_95: "ap50_95", roc_auc: "roc_auc", first_load: "first_load", cpu: "cpu", burst: "burst", last_answer: "last_answer"})[k])); });
    var det = el("th", {scope: "col", "data-sort-off": ""}); det.appendChild(el("span", {class: "sr-only", text: "Details"})); head.appendChild(det);
    var tbody = el("tbody");
    rows.forEach(function (r) {
      var k = keyOf(r), open = !!state.open[k], s = r.speed || {}, ok = r.block === "ok";
      var tr = el("tr", {class: [r.rank === 1 && ok ? "top" : "", r.flags && r.flags.tooSlow ? "slow" : "", r.block === "unfair" ? "lineup-out" : ""].join(" ").trim() || null, "data-best": String(r.position || 0), "data-key": k});
      var name = el("span", {}, [el("span", {class: "bn-dot ser-c" + color[r.model] + (isCpu(r.device) ? " hollow" : ""), "aria-hidden": "true"}), r.model]);
      chipsOf(r).forEach(function (c) { name.appendChild(document.createTextNode(" ")); name.appendChild(c); });
      tr.appendChild(td("Rank", el("span", {class: "bn-rank", text: ok ? String(r.rank) : "–"}), "num", ok ? r.rank : null));
      tr.appendChild(td("Model", name, "bn-lead rc-title", r.model));
      tr.appendChild(td("Device", el("code", {text: r.device, title: r.deviceLabel || null}), null, r.device));
      tr.appendChild(td(metricName, metricCell(r, goal), "num", r.metric ? r.metric.value : null));
      tr.appendChild(td("Recall", fmt.n2(r.recall), "num", r.recall));
      tr.appendChild(td("Precision", fmt.n2(r.precision), "num", r.precision));
      tr.appendChild(td("False alerts /100 empty", fmt.n1(r.fa100), "num", r.fa100));
      tr.appendChild(td("Confidence", fmt.n2(r.confidence), "num", r.confidence));
      tr.appendChild(td(BC.has(s.burstP95Ms) ? "p95 (burst)" : "p95", fmt.ms(speedOf(r)), "num", speedOf(r)));
      columns.forEach(function (key) {
        var c = BC.COLUMNS.filter(function (x) { return x.key === key; })[0];
        var raw = {f1: r.f1, f05: r.f05, ap50: r.ap50, ap50_95: r.ap5095, roc_auc: r.rocAuc, p50: s.p50Ms, p99: s.p99Ms, burst: s.burstP95Ms, last_answer: s.burst ? s.burst.lastAnswerP95Ms : null}[key];
        tr.appendChild(td(c.label, BC.columnValue(r, key), ["first_load", "cpu"].indexOf(key) < 0 ? "num" : null, raw === undefined ? undefined : raw));
      });
      var b = el("button", {type: "button", class: "bn-expand", "aria-expanded": open ? "true" : "false", "aria-controls": "d-" + r.position, "aria-label": (open ? "Less about " : "More about ") + r.model + " on " + r.device, text: open ? "Less" : "More"});
      b.addEventListener("click", function () { state.open[k] = !state.open[k]; render(); var nb = table.querySelector('tr[data-key="' + cssEsc(k) + '"] button.bn-expand'); if (nb) nb.focus(); });
      tr.appendChild(td("Details", b, "rc-action"));
      tbody.appendChild(tr);
      if (open) {
        var cell = el("td", {colspan: String(head.children.length), id: "d-" + r.position});
        cell.appendChild(details(r, v));
        tbody.appendChild(el("tr", {class: "bn-detail", "data-sort-follow": ""}, [cell]));
      }
    });
    table.replaceChildren(el("thead", {}, [head]), tbody);
    if (window.PrismSort) PrismSort.apply(table);
    if (window.PrismTip) PrismTip.refresh();
  }
  function cssEsc(s) { return window.CSS && CSS.escape ? CSS.escape(s) : String(s).replace(/["\\]/g, "\\$&"); }
  function fact(k, v) { return el("div", {}, [el("span", {class: "k", text: k}), el("span", {class: "v", text: v})]); }
  function details(r, v) {
    var s = r.speed || {}, f = r.flags || {}, box = el("div");
    var facts = [
      fact("Device", r.deviceLabel || r.device),
      fact("F1 / F2 / F0.5", fmt.n2(r.f1) + " / " + fmt.n2(r.f2) + " / " + fmt.n2(r.f05)),
      fact("AP50 · AP50-95", fmt.n2(r.ap50) + " · " + fmt.n2(r.ap5095)),
      fact("ROC AUC (frames)", fmt.n2(r.rocAuc)),
      fact("p50 · p95 · p99", fmt.ms(s.p50Ms) + " · " + fmt.ms(s.p95Ms) + " · " + fmt.ms(s.p99Ms)),
      fact("Burst p95", BC.has(s.burstP95Ms) ? fmt.ms(s.burstP95Ms) : s.burst ? "no answer" : "not run"),
      fact("First load", BC.firstLoadText(s.firstLoad)),
      fact("CPU check", BC.agreementText(r.agreement)),
      fact("Saved confidence", BC.has(r.servedThreshold) ? fmt.n2(r.servedThreshold) : "none"),
      fact("Equally good range", BC.rangeText(r.plateau) || (r.plateau ? "only " + fmt.n2(r.plateau[0]) : "-")),
      fact("Cross-validation", BC.cvFact(r.folds, f.lowConfidence))
    ];
    var vb = BC.vsBestText(r.vsBest);
    if (vb) facts.push(fact("Against the best", vb.charAt(0).toUpperCase() + vb.slice(1)));
    var bf = BC.burstFacts(s.burst);
    if (bf && !bf.error) {
      facts.push(fact("Burst load", bf.load));
      facts.push(fact("Burst p50 · p95 · p99", bf.latency));
      facts.push(fact("Last answer p50 · p95", bf.lastAnswer));
      facts.push(fact("Burst refused", bf.refused));
      facts.push(fact("Burst queue", bf.queue));
    }
    if (r.target) facts.push(fact("Target", (r.target.met ? "met" : "not met") + (BC.has(r.target.bound) ? ", 95% bound " + fmt.n2(r.target.bound) : "")));
    if (r.grade) facts.push(fact("Grade", r.grade));
    box.appendChild(el("div", {class: "bn-facts"}, facts));
    var notes = [];
    if (BC.has(f.liveLoad) && f.liveLoad > 0) notes.push("Measured while the service answered " + fmt.n1(f.liveLoad) + " requests a minute.");
    if (BC.has(f.videoDecode) && f.videoDecode > 1) notes.push("The GPU's video decode was " + Math.round(f.videoDecode) + "% busy during the run (Blue Iris Quick Sync).");
    notes.push(f.served ? "As served: night model, night threshold, filters, enhancement and verifier applied." : "Bare model: serving rules not applied.");
    if (bf && bf.error) notes.push("Burst test not measured: " + bf.error + ".");
    if (bf && !bf.error) notes.push("Burst test (" + bf.short + ", " + s.burst.bursts + " bursts): each request timed from when it was due, so queue wait counts; the speed budget applies to its p95." + (s.burst.notes.length ? " " + s.burst.notes.join("; ") + "." : ""));
    if (r.leader && r.vsLeader) notes.push("Against " + r.leader + ": " + fmt.signed(r.vsLeader.value) + (BC.has(r.vsLeader.low) ? " (95% interval " + fmt.signed(r.vsLeader.low) + " to " + fmt.signed(r.vsLeader.high) + ")" : "") + ".");
    box.appendChild(el("p", {class: "small muted", text: notes.join(" ")}));
    var set = v.set ? v.set.id : "";
    var tune = "/benchmark/tune?model=" + encodeURIComponent(r.model) + "&device=" + encodeURIComponent(r.device) + (set ? "&set=" + encodeURIComponent(set) : "");
    var imgs = "/benchmark/images?model=" + encodeURIComponent(r.model) + "&device=" + encodeURIComponent(r.device) + (set ? "&set=" + encodeURIComponent(set) : "") + (BC.has(r.confidence) ? "&threshold=" + r.confidence.toFixed(2) : "") + "&kind=miss";
    box.appendChild(el("div", {class: "actions"}, [el("a", {class: "btn small", href: S.withGoal(tune), text: "Tune this"}), el("a", {class: "btn small secondary", href: S.withGoal(imgs), text: "See its misses"})]));
    return box;
  }

  // ---- columns chooser ----------------------------------------------------------------------------
  function renderColumns() {
    var box = $("c-cols-list"); box.replaceChildren();
    BC.COLUMNS.forEach(function (c) {
      var cb = el("input", {type: "checkbox", value: c.key}); cb.checked = columns.indexOf(c.key) >= 0;
      cb.addEventListener("change", function () {
        columns = BC.columnsOf(BC.COLUMNS.map(function (x) { return x.key; }).filter(function (k) { return k === c.key ? cb.checked : columns.indexOf(k) >= 0; }));
        BC.store.set("columns", columns);
        if (state.view) renderTable(state.view);
      });
      box.appendChild(el("label", {class: "check"}, [cb, " " + c.label]));
    });
  }

  // ---- not run here -------------------------------------------------------------------------------
  function renderMissing(v) {
    var box = $("c-missing"), list = v.missing || [];
    box.hidden = !list.length;
    if (!list.length) { box.replaceChildren(); return; }
    var kids = [el("h3", {text: "Not run on these images"})];
    list.forEach(function (m) {
      var b = el("button", {type: "button", class: "small secondary", text: "Run on these images"});
      if (m.blocked) { b.disabled = true; b.title = m.blocked; }
      b.addEventListener("click", function () { runOn(m, v, b); });
      kids.push(el("div", {class: "bn-row between"}, [el("span", {}, [el("strong", {text: m.model}), " ", el("span", {class: "muted small", text: m.why + (m.blocked ? " " + m.blocked : "")})]), b]));
    });
    box.replaceChildren.apply(box, kids);
  }
  function runOn(m, v, b) {
    var body = new URLSearchParams(), run = (v.set && v.set.run) || {datasets: [], maxImages: 0};
    body.append("model", m.model);
    if (m.device) body.append("device", m.device);
    run.datasets.forEach(function (d) { body.append("dataset", d); });
    body.append("max_images", String(run.maxImages || 0));
    b.disabled = true; say("Starting " + m.model + "…");
    PB.post("/v1/benchmark", body).then(function (j) {
      say(j.success ? (j.message || "Started.") + " Follow it on the Run step." : (j.message || "Not started."), j.success ? "ok" : "error");
      if (j.success) S.kick(); else b.disabled = false;
    });
  }

  // ---- scatter ------------------------------------------------------------------------------------
  var lastScatterWidth = 0;
  function renderScatter(v) {
    var box = $("c-scatter"), legend = $("c-scatter-legend"), color = colors(v.rows), goal = v.goal;
    var pts = [], items = [];
    v.rows.forEach(function (r, i) {
      var x = speedOf(r), y = r.metric && r.metric.value;
      var n = i + 1, label = r.model + " on " + r.device;
      items.push(el("li", {}, [el("span", {class: "bn-dot ser-c" + color[r.model] + (isCpu(r.device) ? " hollow" : ""), "aria-hidden": "true"}), n + ". " + r.model + " · " + r.device.replace(/^openvino:/, "ov:") + (BC.has(y) && BC.has(x) ? "" : " (no point)")]));
      if (BC.has(x) && BC.has(y)) pts.push({key: keyOf(r), n: n, x: x, y: y, low: r.metric.low, high: r.metric.high, color: color[r.model], hollow: isCpu(r.device), label: label});
    });
    items.push(el("li", {class: "muted", text: "filled = GPU or accelerator, hollow = CPU; bars = 95% interval"}));
    legend.replaceChildren.apply(legend, items);
    $("sc-title").textContent = goal.metric + " against " + (v.rows.some(function (r) { return r.speed && BC.has(r.speed.burstP95Ms); }) ? "burst p95" : "p95 latency");
    lastScatterWidth = box.clientWidth;
    box.replaceChildren(C.scatter(pts, {title: goal.metric + " against p95", desc: "One point per model and device: across, p95 latency on a log scale; up, " + goal.metric + ". The table above lists the same numbers.",
      yLabel: goal.metric, budget: goal.speedBudgetMs, avail: box.clientWidth,
      width: Math.min(1000, Math.max(300, box.clientWidth || 620)), height: box.clientWidth < 500 ? 260 : 320,
      onPick: function (k) { state.open = {}; state.open[k] = true; render(); var tr = table.querySelector('tr[data-key="' + cssEsc(k) + '"]'); if (tr) tr.scrollIntoView({block: "center", behavior: "smooth"}); }}));
  }

  // ---- breakdowns ---------------------------------------------------------------------------------
  var TABS = ["scenes", "classes", "cameras", "speed"];
  function tabButtons() { return Array.prototype.slice.call(document.querySelectorAll("#c-tabs [role=tab]")); }
  function selectTab(name, focus) {
    state.tab = name;
    tabButtons().forEach(function (b) { var on = b.dataset.bt === name; b.setAttribute("aria-selected", on ? "true" : "false"); b.tabIndex = on ? 0 : -1; if (on && focus) b.focus(); });
    $("c-breakdown").setAttribute("aria-labelledby", "bt-" + name);
    if (state.view) renderBreakdown(state.view);
  }
  tabButtons().forEach(function (b) {
    b.addEventListener("click", function () { selectTab(b.dataset.bt, false); });
    b.addEventListener("keydown", function (e) {
      var i = TABS.indexOf(state.tab), n = TABS.length, j = null;
      if (e.key === "ArrowRight") j = (i + 1) % n; else if (e.key === "ArrowLeft") j = (i + n - 1) % n; else if (e.key === "Home") j = 0; else if (e.key === "End") j = n - 1;
      if (j !== null) { e.preventDefault(); selectTab(TABS[j], true); }
    });
  });
  function bTable(head, rows, key) {
    var t = el("table", {class: "bn-table rcards", "data-sort": key});
    t.appendChild(el("thead", {}, [el("tr", {}, head.map(function (h, i) { return el("th", {scope: "col", class: i && h.num !== false ? "num" : null, text: h.text || h}); }))]));
    t.appendChild(el("tbody", {}, rows));
    var w = el("div", {class: "scroll"}, [t]);
    if (window.PrismSort) setTimeout(function () { if (t.isConnected) PrismSort.attach(t); }, 0);
    return w;
  }
  function pairSelect(v) {
    if (!state.pair || !v.rows.some(function (r) { return keyOf(r) === state.pair; })) state.pair = v.rows.length ? keyOf(v.rows[0]) : null;
    var sel = el("select", {id: "c-pair"});
    v.rows.forEach(function (r) { var o = el("option", {value: keyOf(r), text: r.model + " · " + r.device}); o.selected = keyOf(r) === state.pair; sel.appendChild(o); });
    sel.addEventListener("change", function () { state.pair = sel.value; renderBreakdown(v); var s2 = $("c-pair"); if (s2) s2.focus(); });
    return el("label", {for: "c-pair"}, ["Model ", sel]);
  }
  function renderBreakdown(v) {
    var p = $("c-breakdown"), goal = v.goal, kids = [], color = colors(v.rows);
    var lead = function (r) { return el("td", {class: "bn-lead rc-title", "data-label": "Model · device"}, [el("span", {class: "bn-dot ser-c" + color[r.model] + (isCpu(r.device) ? " hollow" : ""), "aria-hidden": "true"}), r.model + " · " + r.device]); };
    var cellLow = function (label, text, low, n) { var c = el("td", {class: "num" + (low ? " cell-low" : ""), "data-label": label, text: text}); if (low) c.title = n + " labelled objects: too few to judge"; return c; };
    if (state.tab === "scenes") {
      var scenes = ["day", "night", "ir"], names = {day: "Day", night: "Night", ir: "IR"};
      kids.push(el("p", {class: "muted small", text: "Each pair at its confidence for your goal. Grey cells rest on fewer than 50 labelled objects: too few to judge."}));
      var head = ["Model · device"];
      scenes.forEach(function (s) { head.push(names[s] + " " + goal.metric, names[s] + " FA/100"); });
      var rows = v.rows.map(function (r) {
        var tr = el("tr", {"data-best": String(r.position || 0)}, [lead(r)]);
        scenes.forEach(function (s) {
          var c = (r.scenes || []).filter(function (x) { return x.scene === s; })[0];
          tr.appendChild(cellLow(names[s] + " " + goal.metric, c ? fmt.n2(c.metric) : "-", c && c.fewObjects, c && c.objects));
          tr.appendChild(el("td", {class: "num", "data-label": names[s] + " FA/100", text: c ? fmt.n1(c.fa100) : "-"}));
        });
        return tr;
      });
      kids.push(bTable(head, rows, "bench-scenes"));
      var sc = (v.set && v.set.scenes) || [];
      kids.push(el("p", {class: "bn-foot", text: sc.map(function (s) { return names[s.scene] + " " + fmt.int(s.frames) + " frames"; }).join(" · ") + (sc.length ? ". " : "") + "Scenes are the service's own day / night / IR decision per frame."}));
    } else if (state.tab === "classes") {
      var r = v.rows.filter(function (x) { return keyOf(x) === state.pair; })[0];
      kids.push(el("div", {class: "bn-row"}, [pairSelect(v), el("span", {class: "muted small", text: "Exact classes: a truck called a car is a miss and a false alert."})]));
      r = v.rows.filter(function (x) { return keyOf(x) === state.pair; })[0];
      var counted = (v.set && v.set.classes) || [], named = (goal.classes || []).length > 0;
      var crow = ((r && r.classes) || []).map(function (c) {
        var low = c.objects < 50;
        var tag = named ? el("span", {class: "bn-chip " + (counted.indexOf(c["class"]) >= 0 ? "ok" : "same"), text: counted.indexOf(c["class"]) >= 0 ? "counts" : "detail only"}) : null;
        return el("tr", {}, [el("td", {class: "bn-lead rc-title", "data-label": "Class"}, [c["class"], tag ? " " : null, tag]),
          el("td", {class: "num", "data-label": "Objects", text: fmt.int(c.objects)}),
          cellLow("Recall", fmt.n2(c.recall), low, c.objects), cellLow("Precision", fmt.n2(c.precision), low, c.objects),
          cellLow("F2", fmt.n2(c.f2), low, c.objects), cellLow("AP50", fmt.n2(c.ap50), low, c.objects),
          el("td", {class: "num", "data-label": "FA/100 frames without it", text: fmt.n1(c.fa100)})]);
      });
      kids.push(crow.length ? bTable(["Class", "Objects", "Recall", "Precision", "F2", "AP50", "FA/100 frames without it"], crow, "bench-classes") : el("p", {class: "bn-empty", text: "No class breakdown for this pair."}));
      kids.push(el("p", {class: "bn-foot", text: "Grey: under 50 labelled objects, too few to judge. Choose which classes count in the goal."}));
    } else if (state.tab === "cameras") {
      if (!v.set || v.set.id !== "cameras") {
        kids.push(el("p", {class: "bn-empty", text: "Per camera on Your cameras only: choose Your cameras under Images."}));
      } else {
        kids.push(el("div", {class: "bn-row"}, [pairSelect(v), el("span", {class: "muted small", text: "Worst camera first."})]));
        var rr = v.rows.filter(function (x) { return keyOf(x) === state.pair; })[0];
        var cams = ((rr && rr.cameras) || []).map(function (c) {
          return el("tr", {}, [el("td", {class: "bn-lead rc-title", "data-label": "Camera", text: c.camera}), el("td", {class: "num", "data-label": "Frames", text: fmt.int(c.frames)}),
            el("td", {class: "num", "data-label": "Objects", text: fmt.int(c.objects)}), el("td", {class: "num", "data-label": "Recall", text: fmt.n2(c.recall)}),
            el("td", {class: "num", "data-label": "FA/100 empty", text: fmt.n1(c.fa100)}), el("td", {"data-label": "Worst scene", text: c.worstScene || "-"})]);
        });
        kids.push(cams.length ? bTable(["Camera", "Frames", "Objects", "Recall", "FA/100 empty", {text: "Worst scene", num: false}], cams, "bench-cameras") : el("p", {class: "bn-empty", text: "No camera breakdown for this pair."}));
      }
    } else {
      kids.push(el("p", {class: "muted small", text: "Time per image, one at a time, on these frames: decode, prepare, inference and post-processing (the run's medians), then the whole request's percentiles."}));
      var srows = v.rows.map(function (r) {
        var s = r.speed || {};
        return el("tr", {"data-best": String(r.position || 0)}, [lead(r),
          el("td", {class: "num", "data-label": "Decode", text: fmt.ms(s.decodeMs)}), el("td", {class: "num", "data-label": "Prepare", text: fmt.ms(s.preprocessMs)}),
          el("td", {class: "num", "data-label": "Inference", text: fmt.ms(s.inferMs)}), el("td", {class: "num", "data-label": "Post", text: fmt.ms(s.postprocessMs)}),
          el("td", {class: "num", "data-label": "p50", text: fmt.ms(s.p50Ms)}), el("td", {class: "num", "data-label": "p95", text: fmt.ms(s.p95Ms)}), el("td", {class: "num", "data-label": "p99", text: fmt.ms(s.p99Ms)}),
          el("td", {class: "num", "data-label": "Burst p95", text: BC.has(s.burstP95Ms) ? fmt.ms(s.burstP95Ms) : "not run"}),
          el("td", {"data-label": "First load", text: BC.firstLoadText(s.firstLoad)})]);
      });
      kids.push(bTable(["Model · device", "Decode", "Prepare", "Inference", "Post", "p50", "p95", "p99", "Burst p95", {text: "First load", num: false}], srows, "bench-speed"));
      var bursts = v.rows.filter(function (r) { return r.speed && r.speed.burst; });
      if (bursts.length) {
        kids.push(el("h3", {text: "Burst test"}));
        kids.push(el("p", {class: "muted small", text: "Cameras triggering together, each sending its images a fixed time apart, three bursts per combo on a temporary worker of the serving code. Each request is timed from when it was due (queue wait included); a trigger lasts until its last answer."}));
        var brows = v.rows.map(function (r) {
          var b = r.speed && r.speed.burst, f = BC.burstFacts(b);
          if (!b) return el("tr", {"data-best": String(r.position || 0)}, [lead(r), el("td", {colspan: "8", "data-label": "Burst", text: "not run"})]);
          if (f.error) return el("tr", {"data-best": String(r.position || 0)}, [lead(r), el("td", {"data-label": "Load", text: f.short}), el("td", {colspan: "7", "data-label": "Burst", text: "not measured: " + f.error})]);
          return el("tr", {"data-best": String(r.position || 0)}, [lead(r),
            el("td", {"data-label": "Load", text: f.short}),
            el("td", {class: "num", "data-label": "p50", text: fmt.ms(b.p50Ms)}), el("td", {class: "num", "data-label": "p95", text: fmt.ms(b.p95Ms)}), el("td", {class: "num", "data-label": "p99", text: fmt.ms(b.p99Ms)}),
            el("td", {class: "num", "data-label": "Last answer p50", text: fmt.ms(b.lastAnswerP50Ms)}), el("td", {class: "num", "data-label": "Last answer p95", text: fmt.ms(b.lastAnswerP95Ms)}),
            el("td", {"data-label": "Refused", text: f.refused}), el("td", {class: "num", "data-label": "Most waiting", text: fmt.int(b.maxQueueDepth)})]);
        });
        kids.push(bTable(["Model · device", {text: "Load", num: false}, "p50", "p95", "p99", "Last answer p50", "Last answer p95", {text: "Refused", num: false}, "Most waiting"], brows, "bench-burst"));
      }
      var loaded = v.rows.filter(function (r) { return r.flags && ((r.flags.liveLoad || 0) > 0 || (r.flags.videoDecode || 0) > 1); });
      if (loaded.length) kids.push(el("p", {class: "warn small", text: "Measured while the service answered live requests or the GPU decoded video (" + loaded.map(function (r) { return r.model + " on " + r.device; }).join(", ") + "): live speed under that load is what Blue Iris sees; an idle machine reads faster."}));
    }
    p.replaceChildren.apply(p, kids);
  }

  // ---- the whole view ----------------------------------------------------------------------------
  function msgEl() { return $("c-msg"); }
  function say(text, kind) { state.loadError = false; var m = msgEl(); m.className = "bn-msg small" + (kind === "error" ? " bn-error" : kind === "ok" ? " notice" : ""); m.textContent = text || ""; }
  function render() {
    var v = state.view; if (!v) return;
    S.setGoal(v.goal);
    var sel = $("c-set"); sel.replaceChildren();
    (v.sets || []).forEach(function (s) { var o = el("option", {value: s.id, text: s.title + " · " + fmt.int(s.frames) + " frames"}); o.selected = v.set && v.set.id === s.id; sel.appendChild(o); });
    if (!(v.sets || []).length) sel.appendChild(el("option", {value: "", text: "No image set yet"}));
    var left = $("c-left"), set = v.set;
    if (set && set.framesLeftOut > 0) {
      left.replaceChildren(fmt.int(set.framesLeftOut) + " frames left out: " + (set.leftOutBy || []).join(", ") + " trained on them. ",
        PB.term("Why", "Scores on a model's own training frames say little about new frames, so every model here is compared on the frames none of them trained on."));
      left.hidden = false;
    } else left.hidden = true;
    $("c-headline").textContent = v.headline || "";
    if (set && set.relative) $("c-headline").appendChild(el("span", {class: "muted small", text: " These images have no labels: scored against " + (set.reference || "the reference model") + "'s detections (relative)."}));
    if (set && set.ignoredClasses && set.ignoredClasses.length) $("c-headline").appendChild(el("span", {class: "muted small", text: " Not scored here: " + set.ignoredClasses.join(", ") + "."}));
    if (!v.rows.length) {
      table.replaceChildren(el("tbody", {}, [el("tr", {}, [el("td", {text: v.headline || "Nothing to compare on these images."})])]));
    } else renderTable(v);
    $("c-foot").textContent = (v.notes || []).join(" ") + " p95 = the whole request, one image at a time.";
    renderMissing(v); renderScatter(v); renderBreakdown(v);
    if (focusModel) {
      var first = v.rows.filter(function (r) { return r.model.toLowerCase() === focusModel.toLowerCase(); })[0];
      focusModel = null;
      if (first) { state.open[keyOf(first)] = true; renderTable(v); var tr = table.querySelector('tr[data-key="' + cssEsc(keyOf(first)) + '"]'); if (tr) tr.scrollIntoView({block: "center"}); }
    }
  }
  function url() {
    var q = [];
    if (state.set) q.push("set=" + encodeURIComponent(state.set));
    if (state.scene && state.scene !== "all") q.push("scene=" + encodeURIComponent(state.scene));
    return "/v1/benchmark/compare" + (q.length ? "?" + q.join("&") : "");
  }
  function remember() {
    var p = new URLSearchParams(location.search);
    if (state.set) p.set("set", state.set); else p.delete("set");
    if (state.scene !== "all") p.set("scene", state.scene); else p.delete("scene");
    p.delete("model");
    var q = p.toString();
    if (history.replaceState) history.replaceState(null, "", location.pathname + (q ? "?" + q : "") + location.hash);
  }
  // A failed load: the error, and in place of the skeletons (first load) a line saying so; the
  // view of an earlier load stays as it was.
  function failed(text) {
    say(text + " Reload the page to try again.", "error"); state.loadError = true;
    if (state.view) return;
    $("c-headline").textContent = "The comparison did not load.";
    table.replaceChildren(el("tbody", {}, [el("tr", {}, [el("td", {text: "Nothing to show: " + text})])]));
  }
  function load() {
    var t = loads.begin();
    body.setAttribute("aria-busy", "true");
    return PB.getJSON(S.withGoal(url()), t.init).then(function (j) {
      if (!t.wins(j)) return; // a newer load took its place
      body.removeAttribute("aria-busy");
      if (j._status === 404) { body.hidden = true; empty.hidden = false; return; }
      if (!j.success) { failed(j.message || "The comparison did not load."); return; }
      if (state.loadError) { say(""); state.loadError = false; }
      body.hidden = false; empty.hidden = true;
      state.view = j.compare;
      if (j.compare.set) state.set = j.compare.set.id;
      S.setSet(state.set);
      remember(); render();
    });
  }
  $("c-set").addEventListener("change", function (e) { state.set = e.target.value; state.open = {}; load(); });
  $("c-scene").addEventListener("change", function (e) { state.scene = e.target.value; load(); });

  // ---- report and CSV ----------------------------------------------------------------------------
  var rdlg = $("report-dlg"), rdir = $("report-dir"), rmsg = $("report-msg");
  $("c-report").addEventListener("click", function () { rdir.value = rdir.value || BC.store.get("reportDir", "") || ""; rmsg.textContent = ""; rdlg.showModal(); rdir.focus(); });
  $("report-x").addEventListener("click", function () { rdlg.close(); });
  $("report-cancel").addEventListener("click", function () { rdlg.close(); });
  rdlg.addEventListener("close", function () { $("c-report").focus(); });
  $("report-save").addEventListener("click", function () {
    var d = rdir.value.trim(), b = $("report-save");
    if (!d) { rmsg.className = "bn-msg small bn-error"; rmsg.textContent = "Choose a folder first (Browse…)."; return; }
    b.disabled = true; rmsg.className = "bn-msg small"; rmsg.textContent = "Saving…";
    PB.post("/v1/benchmark/report", new URLSearchParams({dir: d})).then(function (j) {
      rmsg.className = "bn-msg small" + (j.success ? " notice" : " bn-error");
      rmsg.textContent = j.success ? "Report saved to " + j.path + "." : (j.message || "The report was not saved.") + " Check the folder and try again.";
      if (j.success) BC.store.set("reportDir", d);
      b.disabled = false;
    });
  });
  $("c-csv").addEventListener("click", function () {
    if (!state.view) return;
    var blob = new Blob([BC.compareCsv(state.view)], {type: "text/csv"}), a = el("a", {href: URL.createObjectURL(blob), download: "benchmark-compare-" + (state.set || "set") + (state.scene !== "all" ? "-" + state.scene : "") + ".csv"});
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
    say("CSV downloaded.", "ok");
  });

  // ---- redraws -----------------------------------------------------------------------------------
  var rt = null;
  window.addEventListener("resize", function () {
    clearTimeout(rt);
    rt = setTimeout(function () { var w = $("c-scatter").clientWidth; if (state.view && Math.abs(w - lastScatterWidth) > 8) renderScatter(state.view); }, 150);
  });
  S.onView(function () { load(); });
  renderColumns();
  if (!body.hidden) load();
})();
