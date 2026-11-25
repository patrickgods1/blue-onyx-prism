// Blue Onyx Prism: Benchmark pages, the shared part (served from /static/; ES5, no build step,
// no inline scripts under the CSP). Loaded first by every /benchmark/* page.
// Two parts, like live.js: a pure first part (no DOM; tested in node by tests/it/js_bench.rs)
// and helpers that need a document (PrismBench.el, post, getJSON), defined but only called in a
// browser.
//
// Pure part (globalThis.BenchCore unless noted):
// - fmt: n2 / n1 (two / one decimals, "-" for none), ms ("420 ms", "1.4 s"), pct ("87%"),
//   ci(estimate) ("±0.02": the larger half of the 95% interval), signed(d, dp), duration(s).
// - GOALS: the goal catalog of the Goal dialog (kind, label, long, target default and step,
//   `rankOnly`, `more` = under "More goals"); parseGoal("precision:0.9") -> {kind, target};
//   goalSpec(kind, target) -> "precision:0.9"; goalPairs(g) / tryParams(search): the `goal`,
//   `tune_for`, `classes`, `budget` query of a goal tried without saving (the views and the
//   Goal POST read the same fields); withTry(href, params) keeps them on a link.
// - rowChips(row): the chips of a Compare / Tune row (default, trained here, too slow, target not
//   met, served, stale rules, low confidence, relative, flat), each {cls, text, tip}.
// - COLUMNS, columnsOf(stored): the Compare table's optional columns and their stored choice
//   (localStorage `prism.bench.columns`, via store); burstFacts(b): the texts of a
//   row's burst test (load, request p50 · p95 · p99, last answer, refused, queue).
// - comboStates(run): the run poll's per-combo list ({model, device, state, text, now}).
// - cpuReference(devices, dismissed): the CPU device to tick when 2+ devices and no CPU one are
//   ticked (without a CPU reference nothing is recommended); null when none is needed.
// - estimate(o): seconds a run takes from the last run's time per image ({seconds, known, pairs});
//   with o.burst ({cameras, images, interval_ms}) each pair adds burstSeconds(b, ms per image)
//   and `burst` is the share of it.
// - burstSeconds(b, perMs): about how long one combo's burst test takes (src/benchmark/burst.rs:
//   BURSTS bursts, the next one K x I + 2 s after the one before and 2 s after its last answer;
//   the worker's load counted as BURST_LOAD_S).
// - mapHash(hash): the new address of an old /benchmark#tab-... link (null: none).
// - disagree(a, b): the frames on which two combos' examples disagree (one misses or alerts where
//   the other does not).
// - labelSize(w, h, boxW, boxH, px): the frame dialog's label size in image pixels;
//   placeBoxLabels(items, o): where its labels go (inside the image, clear of each other).
// - latest(): "latest request wins" for a view a select change fetches (ticket.wins(answer));
//   PrismPoll.latest of live.js under this name.
// - labelsText(lc) / rescoredText(r): the shell's "labels changed" banner and Re-score result;
//   vsBestText(d): a Compare row against the best; cvText(tune) / cvFact(folds, low): how a pick
//   was validated (cross-validated, partly in-sample, in-sample).
// - compareCsv(view): the Compare table as CSV.
// - cellTag(tag) (also globalThis.BenchTags.cellTag): held-out per-camera tags the tag lists leave
//   out (twin: `training::dataset::is_cell_tag`).
// - globalThis.BenchPoll: runUri(modelsTag), step(last, answer): what a run poll's answer means
//   (GET /v1/benchmark?only=run; src/server/benchmark/poll.rs checks it in node).
// - TERMS / globalThis.PrismBench.tipText(key): the glossary behind `button.term[data-term]`
//   (PrismTip, live.js).
(function (root) {
  "use strict";
  function has(v) { return v !== null && v !== undefined && !(typeof v === "number" && isNaN(v)); }

  // ---- formats ---------------------------------------------------------------------------------
  var fmt = {
    n2: function (v) { return has(v) ? Number(v).toFixed(2) : "-"; },
    n1: function (v) { return has(v) ? Number(v).toFixed(1) : "-"; },
    ms: function (v) { if (!has(v)) return "-"; return v >= 1000 ? (v / 1000).toFixed(1) + " s" : (v >= 100 ? Math.round(v) : Number(v).toFixed(v >= 10 ? 0 : 1)) + " ms"; },
    pct: function (v) { return has(v) ? Math.round(v * 100) + "%" : "-"; },
    // "±0.02": the larger half of a 95% interval {value, low, high}; "" without one.
    ci: function (e) {
      if (!e || !has(e.value) || !has(e.low) || !has(e.high)) return "";
      return "±" + Math.max(e.value - e.low, e.high - e.value, 0).toFixed(2);
    },
    signed: function (d, dp) { if (!has(d)) return "-"; var p = dp === undefined ? 2 : dp; var t = Math.abs(d).toFixed(p); return (Number(t) === 0 ? "±" : d > 0 ? "+" : "−") + t; },
    duration: function (s) { if (!has(s)) return "-"; s = Math.max(1, Math.round(s)); if (s < 90) return s + " s"; var m = Math.round(s / 60); if (m < 90) return m + " min"; return (m / 60).toFixed(1) + " h"; },
    int: function (n) { return has(n) ? String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ",") : "-"; }
  };

  // ---- goals -----------------------------------------------------------------------------------
  // `prefix`: the target's name in the dialog ("N = ").
  var GOALS = [
    {kind: "f1", label: "F1", long: "Balanced: find objects and avoid false alerts equally"},
    {kind: "f2", label: "F2", long: "Miss fewer objects: recall counts more than precision"},
    {kind: "f05", label: "F0.5", long: "Fewer false alerts: precision counts more than recall"},
    {kind: "recall_at_fa", label: "Recall at ≤ N false alerts", long: "Most objects found with at most N false alerts per 100 empty frames", target: 5, step: 0.5, min: 0, max: 100, prefix: "N"},
    {kind: "precision", label: "Recall with precision ≥ p", long: "Most objects found while at least p of the detections are right", target: 0.9, step: 0.01, min: 0.01, max: 0.99, prefix: "p"},
    {kind: "recall", label: "Precision with recall ≥ r", long: "Fewest wrong detections while finding at least r of the objects", target: 0.9, step: 0.01, min: 0.01, max: 0.99, prefix: "r"},
    {kind: "youden", label: "Youden's J", long: "Alert rate on frames with an object minus the false-alert rate on empty frames", more: true},
    {kind: "fpr", label: "Alert rate with FPR ≤ x", long: "Most frames with an object alerted, at most x of empty frames alerting", target: 0.05, step: 0.01, min: 0.01, max: 0.99, prefix: "x", more: true},
    {kind: "ap50", label: "AP50", long: "Average precision at IoU 0.5 over every confidence", rankOnly: true, more: true},
    {kind: "ap50_95", label: "AP50-95", long: "COCO average precision, IoU 0.5 to 0.95: also rewards tight boxes", rankOnly: true, more: true},
    {kind: "roc_auc", label: "ROC AUC", long: "How well a frame's top confidence separates frames with and without objects", rankOnly: true, more: true},
    {kind: "grade", label: "Grade", long: "The A-F grade: accuracy and speed, weighted", rankOnly: true, more: true}
  ];
  var TUNE_FOR = ["f1", "f2", "f05"];
  function goalOf(kind) { for (var i = 0; i < GOALS.length; i++) if (GOALS[i].kind === kind) return GOALS[i]; return null; }
  function parseGoal(spec) {
    var s = String(spec || "").trim().toLowerCase(), i = s.indexOf(":");
    var kind = i < 0 ? s : s.slice(0, i), g = goalOf(kind);
    if (!g) return null;
    if (g.target === undefined) return {kind: kind, target: null};
    var t = i < 0 ? g.target : parseFloat(s.slice(i + 1));
    return {kind: kind, target: isFinite(t) ? t : g.target};
  }
  // The target as the server writes it ("0.9", "5", "2.5"): up to 4 decimals, no trailing zeros.
  function num(t) { return String(Number(Number(t).toFixed(4))); }
  function goalSpec(kind, target) {
    var g = goalOf(kind); if (!g) return null;
    if (g.target === undefined) return kind;
    return kind + ":" + num(has(target) && isFinite(target) ? target : g.target);
  }
  // Why a target cannot be used (null: fine).
  function targetProblem(kind, target) {
    var g = goalOf(kind); if (!g || g.target === undefined) return null;
    if (!has(target) || !isFinite(target)) return g.label + ": enter a number.";
    if (target < g.min || target > g.max) return g.label + ": use a value from " + g.min + " to " + g.max + ".";
    return null;
  }
  // The query fields of goal `g` ({optimizeFor, tuneFor, classes [], budget (ms or null)}), as
  // [[key, value]] pairs: what the views take to try it and POST /v1/benchmark/goal saves.
  function goalPairs(g) {
    return [
      ["goal", g.optimizeFor],
      ["tune_for", g.tuneFor || "f1"],
      ["classes", g.classes && g.classes.length ? g.classes.join(",") : "all"],
      ["budget", has(g.budget) ? String(g.budget) : "none"]
    ];
  }
  function encodePairs(pairs) { return pairs.map(function (p) { return encodeURIComponent(p[0]) + "=" + encodeURIComponent(p[1]); }).join("&"); }
  var TRY_KEYS = ["goal", "tune_for", "classes", "budget"];
  // The goal fields of a page address's query (`?goal=f2&budget=500`): [[key, value]] (empty:
  // nothing tried).
  function tryParams(search) {
    var out = [], q = String(search || "").replace(/^\?/, "");
    if (!q) return out;
    q.split("&").forEach(function (kv) {
      var i = kv.indexOf("="), k, v;
      try { k = decodeURIComponent((i < 0 ? kv : kv.slice(0, i)).replace(/\+/g, " ")); v = i < 0 ? "" : decodeURIComponent(kv.slice(i + 1).replace(/\+/g, " ")); } catch (e) { return; }
      if (TRY_KEYS.indexOf(k) >= 0 && v !== "") out.push([k, v]);
    });
    return out;
  }
  // `href` with the tried goal's fields (replacing any it has).
  function withTry(href, pairs) {
    if (!pairs || !pairs.length) return href;
    var h = String(href), hash = "", qi;
    var hi = h.indexOf("#"); if (hi >= 0) { hash = h.slice(hi); h = h.slice(0, hi); }
    qi = h.indexOf("?");
    var path = qi < 0 ? h : h.slice(0, qi), rest = qi < 0 ? "" : h.slice(qi + 1);
    var kept = rest ? rest.split("&").filter(function (kv) { var k = kv.split("=")[0]; return kv && TRY_KEYS.indexOf(decodeURIComponent(k)) < 0; }) : [];
    var q = kept.concat([encodePairs(pairs)]).filter(Boolean).join("&");
    return path + (q ? "?" + q : "") + hash;
  }
  // The address without the tried goal's fields.
  function withoutTry(href) {
    var h = String(href), hash = "", hi = h.indexOf("#");
    if (hi >= 0) { hash = h.slice(hi); h = h.slice(0, hi); }
    var qi = h.indexOf("?"); if (qi < 0) return h + hash;
    var kept = h.slice(qi + 1).split("&").filter(function (kv) { return kv && TRY_KEYS.indexOf(decodeURIComponent(kv.split("=")[0])) < 0; });
    return h.slice(0, qi) + (kept.length ? "?" + kept.join("&") : "") + hash;
  }

  // ---- row chips -------------------------------------------------------------------------------
  function rowChips(row) {
    var f = (row && row.flags) || {}, out = [];
    if (row && row.isDefault) out.push({cls: "tag", text: "default", tip: "The model Blue Iris's default detection route uses."});
    if (f.trainedHere && f.unknownSeen) {
      out.push({cls: "bn-chip warn", text: "training unknown", tip: "Which of these frames it trained on cannot be told (" + f.unknownSeen + "), so it is never ranked among the fair rows. Its training run's lists must be readable (Train area) for it to be judged on the frames it never saw."});
    } else if (f.trainedHere) {
      var share = has(f.trainedOnShare) ? Math.round(f.trainedOnShare * 100) + "% of" : "some of";
      out.push({cls: "bn-chip warn", text: "trained here", tip: "It trained or validated on " + share + " these frames: its scores here say little about new frames, so it is never ranked among the fair rows."});
    }
    if (f.tooSlow) out.push({cls: "bn-chip bad", text: "too slow", tip: "Its p95 is over the goal's speed budget."});
    if (f.unmet) out.push({cls: "bn-chip bad", text: "target not met", tip: "The goal's target does not hold on the 95% bound of its cross-validated counts."});
    if (f.served) out.push({cls: "bn-chip same", text: "served", tip: "Scored as Blue Iris gets it: night model, night threshold, filters, enhancement and verifier applied."});
    if (f.staleRules) out.push({cls: "bn-chip warn", text: "stale rules", tip: "Its serving settings changed since this run: run it again for current numbers."});
    if (f.lowConfidence) out.push({cls: "bn-chip warn", text: "low confidence", tip: f.lowConfidence});
    if (f.relative) out.push({cls: "bn-chip same", text: "relative", tip: "These images have no labels: scored against another model's detections, so its confidence cannot be saved."});
    if (f.flat) out.push({cls: "bn-chip same", text: "flat curve", tip: "Every confidence scores about the same here: the pick says nothing about the confidence to use."});
    return out;
  }

  // ---- Compare columns ------------------------------------------------------------------------
  var COLUMNS = [
    {key: "f1", label: "F1"}, {key: "f05", label: "F0.5"}, {key: "ap50", label: "AP50"}, {key: "ap50_95", label: "AP50-95"},
    {key: "roc_auc", label: "ROC AUC"}, {key: "p50", label: "p50"}, {key: "p99", label: "p99"}, {key: "burst", label: "Burst p95"},
    {key: "last_answer", label: "Last answer"}, {key: "first_load", label: "First load"}, {key: "cpu", label: "CPU check"}
  ];
  function columnsOf(stored) {
    var list = stored;
    if (typeof stored === "string") { try { list = JSON.parse(stored); } catch (e) { list = null; } }
    if (!Array.isArray(list)) return [];
    var known = COLUMNS.map(function (c) { return c.key; });
    return known.filter(function (k) { return list.indexOf(k) >= 0; });
  }
  var AGREEMENT = {match: "agrees with CPU", agrees: "agrees with CPU", minor: "minor differences", differs: "differs from CPU", reference: "CPU reference", not_checked: "not checked against CPU"};
  function agreementText(v) { return has(v) ? (AGREEMENT[v] || String(v).replace(/_/g, " ")) : "not checked against CPU"; }
  // A run's first load (compile + request creation) as shown: "2.1 s, cached", "24 s, compiled",
  // "3.0 s, cache off", "3.0 s, cache unknown", "0.8 s". Twin: `report::first_load_text`
  // (tests/it/js_bench.rs).
  function loadText(compileMs, cache) {
    var s = Math.max(0, compileMs || 0) / 1e3, secs = (s >= 10 ? s.toFixed(0) : s.toFixed(1)) + " s";
    return ({hit: secs + ", cached", miss: secs + ", compiled", disabled: secs + ", cache off", unknown: secs + ", cache unknown"})[cache] || secs;
  }
  function firstLoadText(fl) {
    if (!fl || !has(fl.compileMs) || (!fl.compileMs && !fl.warmupMs && !fl.cache)) return "-";
    return loadText(fl.compileMs, fl.cache);
  }
  // b: a Compare row's speed.burst (src/benchmark/eval/compare.rs BurstSpeed). The texts of the
  // row details and the Speed tab: load ("8 cameras × 4 images, 500 ms apart"), request
  // percentiles, time to a trigger's last answer, refusals, the queue; `error` when it measured
  // nothing.
  function burstFacts(b) {
    if (!b) return null;
    var plural = function (n, w) { return n + " " + w + (n === 1 ? "" : "s"); };
    var out = {
      load: plural(b.cameras, "camera") + " × " + plural(b.images, "image") + ", " + b.intervalMs + " ms apart",
      short: b.cameras + " × " + b.images + " × " + b.intervalMs + " ms",
      error: b.error || null,
      latency: fmt.ms(b.p50Ms) + " · " + fmt.ms(b.p95Ms) + " · " + fmt.ms(b.p99Ms),
      lastAnswer: fmt.ms(b.lastAnswerP50Ms) + " · " + fmt.ms(b.lastAnswerP95Ms),
      refused: b.refused ? fmt.int(b.refused) + " of " + fmt.int(b.requests) + " (" + b.refusedBy + ")" : "none of " + fmt.int(b.requests),
      queue: "up to " + fmt.int(b.maxQueueDepth) + " waiting (queue " + fmt.int(b.queueCapacity) + ", " + fmt.int(b.inflight) + " in flight)"
    };
    if (b.error) { out.latency = out.lastAnswer = "-"; out.refused = "-"; out.queue = "-"; }
    return out;
  }
  function columnValue(row, key) {
    var s = row.speed || {};
    switch (key) {
      case "f1": return fmt.n2(row.f1);
      case "f05": return fmt.n2(row.f05);
      case "ap50": return fmt.n2(row.ap50);
      case "ap50_95": return fmt.n2(row.ap5095);
      case "roc_auc": return fmt.n2(row.rocAuc);
      case "p50": return fmt.ms(s.p50Ms);
      case "p99": return fmt.ms(s.p99Ms);
      case "burst": return has(s.burstP95Ms) ? fmt.ms(s.burstP95Ms) : s.burst ? "no answer" : "not run";
      case "last_answer": return s.burst && has(s.burst.lastAnswerP95Ms) ? fmt.ms(s.burst.lastAnswerP95Ms) : s.burst ? "-" : "not run";
      case "first_load": return firstLoadText(s.firstLoad);
      case "cpu": return agreementText(row.agreement);
    }
    return "-";
  }

  // ---- the run's combos ------------------------------------------------------------------------
  function key(s) { return String(s || "").trim().toLowerCase(); }
  function sameDevice(a, b) { var x = key(a).replace(/\.0$/, ""), y = key(b).replace(/\.0$/, ""); return x === y; }
  var PHASE = {loading: "compiling", warmup: "warming up", timing: "scoring", burst: "burst test"};
  // run: the `run` of GET /v1/benchmark?only=run ({request, progress, models, running, state}).
  function comboStates(run) {
    if (!run) return [];
    var req = run.request || {}, prog = run.progress || null, done = run.models || [];
    var names = (req.models || []).slice();
    done.forEach(function (m) { if (!names.some(function (n) { return key(n) === key(m.model); })) names.push(m.model); });
    if (prog && prog.model && !names.some(function (n) { return key(n) === key(prog.model); })) names.push(prog.model);
    var out = [];
    names.forEach(function (name) {
      var m = null;
      done.forEach(function (x) { if (key(x.model) === key(name)) m = x; });
      var devs = (req.devices || []).slice();
      if (!devs.length) {
        if (m) m.devices.forEach(function (d) { devs.push(d.device); });
        if (prog && key(prog.model) === key(name) && prog.device && !devs.some(function (d) { return sameDevice(d, prog.device); })) devs.push(prog.device);
        if (!devs.length) devs.push("");
      }
      devs.forEach(function (dev) {
        var d = null;
        if (m) (m.devices || []).forEach(function (x) { if (sameDevice(x.device, dev)) d = x; });
        var item = {model: (m && m.model) || name, device: dev, state: "queued", text: "queued", now: false};
        var current = prog && prog.stage === "benchmark" && key(prog.model) === key(name) && (!dev || sameDevice(prog.device, dev));
        if (d && d.error) { item.state = "failed"; item.text = "failed: " + d.error; }
        else if (d && d.run) { item.state = "done"; item.text = "done"; }
        else if (m && m.error) { item.state = "failed"; item.text = "failed: " + m.error; }
        else if (current && run.running) {
          item.now = true; item.state = PHASE[prog.phase] ? prog.phase : "timing";
          item.text = prog.phase === "timing" && prog.total ? "scoring " + fmt.int(prog.done) + " of " + fmt.int(prog.total) + " images"
            : prog.phase === "burst" && prog.total ? "burst test: " + fmt.int(prog.done) + " of " + fmt.int(prog.total) + " answered" : (PHASE[prog.phase] || "running");
          if (!dev) item.device = prog.device;
        } else if (!run.running) { item.state = "skipped"; item.text = run.state === "cancelled" ? "not run (cancelled)" : "not run"; }
        out.push(item);
      });
    });
    return out;
  }

  // ---- Run form rules ----------------------------------------------------------------------------
  // devices: [{spec, checked, runnable, cpu}]; dismissed: the spec the user unticked after it was
  // ticked for them (never ticked again). The CPU device to tick, else null.
  function cpuReference(devices, dismissed) {
    var ticked = (devices || []).filter(function (d) { return d.checked; });
    if (ticked.length < 2 || ticked.some(function (d) { return d.cpu; })) return null;
    var cpus = devices.filter(function (d) { return d.cpu && d.runnable && d.spec !== dismissed; });
    var pick = null;
    cpus.forEach(function (d) { if (!pick || (d.spec.indexOf("openvino:") === 0 && pick.spec.indexOf("openvino:") !== 0)) pick = d; });
    return pick ? pick.spec : null;
  }
  // The burst test (src/benchmark/burst.rs): BURSTS bursts; one lasts until its last send's answer
  // ((K - 1) x I + one request) or until the work queued is done (C x K requests one at a time:
  // the slowest case), whichever is later; the next starts K x I + 2 s after it began and 2 s
  // after its last answer. The
  // worker's compile (from the cache the timed run just filled) and warm-up: BURST_LOAD_S.
  var BURSTS = 3, BURST_SETTLE_S = 2, BURST_LOAD_S = 3;
  function burstSeconds(b, perMs) {
    if (!b) return 0;
    var c = Math.max(1, b.cameras || 1), k = Math.max(1, b.images || 1), i = Math.max(0, b.interval_ms || 0) / 1000;
    var per = has(perMs) && perMs > 0 ? perMs / 1000 : 0;
    var one = Math.max((k - 1) * i + per, c * k * per), total = BURST_LOAD_S;
    for (var n = 0; n < BURSTS; n++) total += n < BURSTS - 1 ? Math.max(k * i + BURST_SETTLE_S, one + BURST_SETTLE_S) : one;
    return total;
  }
  // o: {models: [name], devices: [spec], timing: {model: {device: ms per image}}, images (total per
  // run, after the per-set limit), repeat, warmup, burst (null: off)}. The warm-up lasts 2 to 20 s
  // once on.
  function estimate(o) {
    var total = 0, known = 0, burst = 0, pairs = (o.models || []).length * (o.devices || []).length;
    var repeat = Math.max(1, o.repeat || 1), warmup = Math.max(0, o.warmup || 0), n = o.images || 0;
    (o.models || []).forEach(function (m) {
      var t = (o.timing || {})[m] || {};
      (o.devices || []).forEach(function (d) {
        var per = null;
        Object.keys(t).forEach(function (k) { if (sameDevice(k, d)) per = t[k]; });
        if (!has(per) || !(per > 0) || !n) return;
        var warm = warmup > 0 ? Math.min(20, Math.max(2, warmup * per / 1000)) : 0;
        var b = o.burst ? burstSeconds(o.burst, per) : 0;
        total += warm + repeat * n * per / 1000 + b; burst += b; known++;
      });
    });
    var out = {seconds: known ? total : null, known: known, pairs: pairs};
    if (o.burst) out.burst = known ? burst : null;
    return out;
  }

  // ---- old addresses ---------------------------------------------------------------------------
  var OLD = {
    "tab-compare": "compare", "tab-run": "run", "tab-tune": "tune", "tab-history": "history",
    "tab-details": "compare", "tab-overview": "tune", "tab-results": "tune", "tab-profiles": "tune",
    "tab-apply": "tune", "tune-apply": "tune", "goal": "tune", "tab-advanced": "compare",
    "tab-confidence": "tune", "adv-confidence": "tune", "search-card": "tune", "tab-models": "compare",
    "adv-details": "compare", "adv-compare": "compare", "adv-history": "history", "cmp-runs": "history",
    "adv-report": "compare"
  };
  function mapHash(hash) {
    var h = String(hash || "").replace(/^#/, "");
    if (!h) return null;
    if (OLD[h]) return "/benchmark/" + OLD[h];
    var m = /^(m|d)-(.+)$/.exec(h);
    if (!m) return null;
    var name;
    try { name = decodeURIComponent(m[2]); } catch (e) { return null; }
    return (m[1] === "m" ? "/benchmark/tune?model=" : "/benchmark/compare?model=") + encodeURIComponent(name);
  }

  // ---- images ----------------------------------------------------------------------------------
  // a, b: frames ({set, file, missed, falsePositives}) of two combos; the frames of `a` on which
  // they disagree, each with `other` = b's frame.
  function disagree(a, b) {
    var byKey = {};
    (b || []).forEach(function (f) { byKey[f.set + "\u0000" + f.file] = f; });
    return (a || []).filter(function (f) {
      var o = byKey[f.set + "\u0000" + f.file]; if (!o) return false;
      return (f.missed > 0) !== (o.missed > 0) || (f.falsePositives > 0) !== (o.falsePositives > 0);
    }).map(function (f) { return Object.assign({}, f, {other: byKey[f.set + "\u0000" + f.file]}); });
  }

  // The frame dialog's box labels (bench-frames.js) are SVG text in the image's own pixels
  // (viewBox = w x h, fitted into a box of boxW x boxH CSS px with "meet"), so a fixed size would
  // shrink with the frame: 1920 px wide on a 390 px phone is 2-7 px. labelSize gives the size in
  // image pixels that renders at `px` CSS px (default 13), its halo (stroke) width (3 px at 13 px)
  // and the gap above its box; without a measured box (0), about 1/32 of the shorter side.
  var LABEL_PX = 13;
  function labelSize(w, h, boxW, boxH, px) {
    px = px || LABEL_PX;
    var scale = w > 0 && h > 0 && boxW > 0 && boxH > 0 ? Math.min(boxW / w, boxH / h) : 0;
    var size = scale ? px / scale : Math.max(12, Math.round(Math.min(w || 1, h || 1) / 32));
    return {size: size, halo: size * 3 / LABEL_PX, gap: size * 4 / LABEL_PX, scale: scale};
  }

  // Where the frame dialog's box labels go (bench-frames.js), in image pixels: items [{x1, y1, x2,
  // y2 (the box), text, width (the text's advance, measured; else about 0.6 x size per
  // character)}], o {w, h (the image), size (font size), gap, halo (stroke width)}. Each label
  // tries, in order: above its box, inside its top edge (no room above), lower inside the box a
  // line at a time, below its box, inside its bottom edge, then further above and below a line
  // at a time; x starts at the box's left and shifts left to end inside the image. The first spot
  // inside the image that covers no label placed before wins; with none free, the first inside
  // the image. Returns [{x, y (baseline), x1, y1, x2, y2 (the text's box, halo included)}]: from
  // the baseline 0.95 x size up and a quarter down, as browsers measure a line of text.
  function placeBoxLabels(items, o) {
    var w = o.w, h = o.h, size = o.size, gap = has(o.gap) ? o.gap : size * 0.3, hh = (o.halo || 0) / 2, pad = gap / 2, placed = [];
    var up = size * 0.95 + hh, down = size * 0.25 + hh, line = up + down + gap / 2;
    function hit(r) { return placed.some(function (q) { return r.x1 < q.x2 && r.x2 > q.x1 && r.y1 < q.y2 && r.y2 > q.y1; }); }
    function at(x, y, tw) { return {x: x, y: y, x1: x - hh, x2: x + tw + hh, y1: y - up, y2: y + down}; }
    return (items || []).map(function (it) {
      var tw = has(it.width) && it.width > 0 ? it.width : String(it.text || "").length * size * 0.6;
      var x = Math.max(hh, Math.min(Math.min(it.x1, it.x2) + pad, w - tw - hh));
      var top = Math.min(it.y1, it.y2), bottom = Math.max(it.y1, it.y2), k;
      var above = top - gap - down, inside = top + pad + up, below = bottom + gap + up;
      var ys = [above, inside];
      for (k = 1; k <= 4 && inside + k * line + down < bottom - pad; k++) ys.push(inside + k * line);
      ys.push(below, bottom - pad - down);
      for (k = 1; k <= 3; k++) ys.push(above - k * line, below + k * line);
      var pick = null, first = null;
      ys.forEach(function (y) {
        if (pick) return;
        var r = at(x, y, tw);
        if (r.y1 < 0 || r.y2 > h) return;
        if (!first) first = r;
        if (!hit(r)) pick = r;
      });
      if (!pick) pick = first || at(x, Math.max(up, Math.min(h - down, inside)), tw);
      placed.push(pick);
      return pick;
    });
  }

  // ---- CSV -------------------------------------------------------------------------------------
  function csvCell(v) { var s = has(v) ? String(v) : ""; return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }
  function compareCsv(view) {
    var goal = (view && view.goal) || {}, rows = (view && view.rows) || [];
    var head = ["rank", "model", "device", goal.metric || "metric", "low", "high", "recall", "precision", "f1", "f2", "f05", "fa_per_100_empty", "confidence", "ap50", "ap50_95", "roc_auc", "p50_ms", "p95_ms", "p99_ms", "flags"];
    var lines = [head.join(",")];
    rows.forEach(function (r) {
      var m = r.metric || {}, s = r.speed || {};
      var flags = rowChips(r).map(function (c) { return c.text; }).join("; ");
      lines.push([r.block === "ok" ? r.rank : "", r.model, r.device, m.value, m.low, m.high, r.recall, r.precision, r.f1, r.f2, r.f05, r.fa100, r.confidence, r.ap50, r.ap5095, r.rocAuc, s.p50Ms, s.p95Ms, s.p99Ms, flags].map(csvCell).join(","));
    });
    return lines.join("\r\n") + "\r\n";
  }

  // Where each point's number goes so none covers another number or a point: items [{x, y, text}]
  // (px, the point's centre), tried above, below, right, left, then the corners, a step further
  // out each round; the first free spot wins (else above). Returns [{x, y, anchor}] (`y` is the
  // text's baseline). `r`: the point's radius; `w`: width per character; `h`: the text's height.
  function placeLabels(items, o) {
    o = o || {}; var r = o.r || 7, cw = o.w || 7, h = o.h || 11, gap = 3;
    var boxes = items.map(function (p) { return {x1: p.x - r, x2: p.x + r, y1: p.y - r, y2: p.y + r}; });
    function hit(b) { return boxes.some(function (q) { return b.x1 < q.x2 && b.x2 > q.x1 && b.y1 < q.y2 && b.y2 > q.y1; }); }
    return items.map(function (p) {
      var w = String(p.text).length * cw, pick = null;
      for (var k = 0; k < 3 && !pick; k++) {
        var d = r + gap + k * (h + gap);
        var tries = [[0, -d, "middle"], [0, d + h, "middle"], [d, h / 2 - 1, "start"], [-d, h / 2 - 1, "end"], [d, -d, "start"], [-d, -d, "end"], [d, d + h, "start"], [-d, d + h, "end"]];
        for (var i = 0; i < tries.length; i++) {
          var t = tries[i], x = p.x + t[0], y = p.y + t[1];
          var x1 = t[2] === "middle" ? x - w / 2 : t[2] === "start" ? x : x - w;
          var b = {x1: x1, x2: x1 + w, y1: y - h, y2: y};
          if (!hit(b)) { pick = {x: x, y: y, anchor: t[2]}; boxes.push(b); break; }
        }
      }
      return pick || {x: p.x, y: p.y - r - gap, anchor: "middle"};
    });
  }

  // The equally good range of a pick ([first, last]): "0.20–0.24", or null when it is one value.
  function rangeText(p) {
    if (!p || !has(p[0]) || !has(p[1])) return null;
    var a = fmt.n2(p[0]), b = fmt.n2(p[1]);
    return a === b ? null : a + "–" + b;
  }
  function cellTag(k) { return String(k).indexOf("heldout:cam:") === 0 || String(k).indexOf("heldout:camscene:") === 0; }

  // ---- the run poll ----------------------------------------------------------------------------
  function runUri(modelsTag) {
    return "/v1/benchmark?only=run" + (modelsTag ? "&models=" + encodeURIComponent(modelsTag) : "");
  }
  // last: {viewTag, modelsTag, models, running} (null: first poll); s: the answer. Returns
  // {error} (not an answer), {refetch} (the models were left out under a tag the page does not
  // have: ask again without one) or {running, state, run (models filled in), models, modelsTag,
  // viewTag, viewChanged (the saved results or the config changed: reload the page's view),
  // ended (a run just finished), started, labelsChanged}.
  function step(last, s) {
    if (!s || typeof s !== "object" || s.success === false || s.viewTag === undefined) return {error: true};
    var run = s.run || {}, models = run.models;
    if (models === undefined) {
      if (!last || !last.modelsTag || last.modelsTag !== s.modelsTag) return {refetch: true};
      models = last.models || [];
    }
    var wasRunning = !!(last && last.running);
    return {
      running: !!s.running, state: s.state, run: Object.assign({}, run, {models: models}), models: models,
      modelsTag: s.modelsTag || null, viewTag: s.viewTag || null,
      viewChanged: !!(last && last.viewTag && s.viewTag && last.viewTag !== s.viewTag),
      ended: wasRunning && !s.running, started: !!last && !wasRunning && !!s.running,
      labelsChanged: s.labelsChanged === undefined ? null : s.labelsChanged
    };
  }

  // ---- texts of the server's answers -----------------------------------------------------------
  function cap(t) { t = String(t || ""); return t.charAt(0).toUpperCase() + t.slice(1); }
  function stop(t) { t = String(t || "").trim(); return t && !/[.!?]$/.test(t) ? t + "." : t; }
  function frames(n) { return fmt.int(n) + " frame" + (n === 1 ? "" : "s"); }
  // lc: `labelsChanged` of the run poll ({frames, changed, left, joined, unknown, stale,
  // message}). The shell's banner (with Re-score) as text, or null when it hides: nothing
  // differs, or not known yet (computing, the folder unreadable: no Re-score to offer). `stale`:
  // the results were saved apart from their labels snapshot, the server's message says so.
  // `unknown`: scored frames the walk could not read this time (neither changed nor left).
  function labelsText(lc) {
    if (!lc) return null;
    var n = has(lc.frames) ? lc.frames : 0, u = has(lc.unknown) ? lc.unknown : 0;
    var unread = u ? " " + frames(u) + " could not be read this time; a re-score keeps their figures." : "";
    if (lc.stale) return stop(cap(lc.message || "these results were saved with other labels than the ones beside them")) + unread;
    if (!n) return null;
    var bits = [];
    if (lc.changed) bits.push(fmt.int(lc.changed) + " changed");
    if (lc.left) bits.push(fmt.int(lc.left) + " no longer reviewed");
    if (lc.joined) bits.push(fmt.int(lc.joined) + " newly reviewed (a run scores those)");
    return "Labels of " + frames(n) + " of Your cameras differ from this run's: " + bits.join(", ") + "." + unread;
  }
  // r: `rescored` of POST /v1/benchmark/rescore ({frames, runs, needRun, notRescored, unknown
  // [file names]}): the result line.
  function rescoredText(r) {
    r = r || {};
    var t = "Re-scored " + frames(r.frames || 0) + " of " + fmt.int(r.runs || 0) + " run" + (r.runs === 1 ? "" : "s") + " against the current labels.";
    if (r.needRun) t += " Frames reviewed since the run need a new run.";
    var un = r.unknown || [];
    if (un.length) t += " " + frames(un.length) + " could not be read (" + un.slice(0, 3).join(", ") + (un.length > 3 ? ", and " + (un.length - 3) + " more" : "") + ") and keep their old figures; re-score again once they read.";
    var nr = r.notRescored || [];
    if (nr.length) t += " Not re-scored (no stored detections): " + nr.join(", ") + ".";
    return t;
  }
  // d: a Compare row's `vsBest` ({value, low, high}: the row minus the best row, paired on the
  // same frames; null for the best itself). "Not clearly different" while the 95% interval of
  // the difference holds 0.
  function vsBestText(d) {
    if (!d || !has(d.value)) return null;
    // Three decimals: differences between close rows are often under 0.01.
    var sg = function (x) { return fmt.signed(x, 3); }, n3 = function (x) { return Number(x).toFixed(3); };
    var iv = has(d.low) && has(d.high) ? " (95% interval of the difference " + sg(d.low) + " to " + sg(d.high) + ")" : "";
    if (!iv) return sg(d.value) + " against the best (no interval: too few groups to tell)";
    if (d.high < 0) return "clearly below the best by " + n3(-d.value) + iv;
    if (d.low > 0) return "clearly above the best by " + n3(d.value) + iv;
    return "not clearly different from the best: " + sg(d.value) + iv;
  }
  // How a pick was validated: folds (0 = in-sample), low (the server's low-confidence reason,
  // shown beside it). Partly in-sample = folds with a low-confidence reason (a fold borrowed the
  // pick on every frame).
  function cvKind(folds, low) { return !(folds > 0) ? "in-sample" : low ? "partly" : "validated"; }
  // The Tune view's first line: t {folds, groups, set, lowConfidence, objectiveName, suggested}.
  function cvText(t) {
    var k = cvKind(t.folds, t.lowConfidence), sug = t.suggested || {};
    if (k === "in-sample") return "In-sample pick: counted on the frames it was picked on.";
    var over = "Cross-validated over " + fmt.int(t.groups) + (t.set === "cameras" ? " camera days" : " groups") + " (" + t.folds + " folds)";
    var fig = sug.metric ? ": " + t.objectiveName + " " + fmt.n2(sug.metric.value) + " " + fmt.ci(sug.metric) : "";
    if (k === "partly") return over + ", partly in-sample" + fig + ".";
    return over + (fig ? fig + " on frames the pick never saw." : ".");
  }
  // The Compare details' "Cross-validation" fact.
  function cvFact(folds, low) {
    var k = cvKind(folds, low);
    return k === "in-sample" ? "in-sample (too little data)" : folds + " folds" + (k === "partly" ? ", partly in-sample" : "");
  }

  // ---- latest request wins --------------------------------------------------------------------
  // PrismPoll.latest of live.js (every page loads it first), kept under this name for the
  // Benchmark pages. Looked up on each call: node tests load this file without live.js.
  function latest(make) { return root.PrismPoll.latest(make); }

  // ---- glossary --------------------------------------------------------------------------------
  var TERMS = {
    oof: "Cross-validated: the confidence is picked on the other camera days (single images on public sets) and counted on the frames it never saw, so the number holds on new frames.",
    ci: "± is the 95% interval of a cluster bootstrap over camera days: the range the true value is likely in.",
    rank: "Rows that share a rank are not clearly different: the 95% interval of their paired difference includes zero.",
    fa100: "False alerts per 100 empty frames: frames with no labelled object of the counted classes on which the model reports one.",
    confidence: "The suggested confidence for this model on this device: the pick on every frame for your goal. 0.34 = 34% in Blue Iris.",
    p95: "95th percentile of the whole request (decode, prepare, inference, post) one image at a time: 19 in 20 frames are this fast or faster.",
    recall: "Share of the labelled objects found at the confidence shown.",
    precision: "Share of the detections that were a labelled object at the confidence shown.",
    f1: "Precision and recall with equal weight.", f2: "Recall counts more than precision: fewer missed objects.", f05: "Precision counts more than recall: fewer false alerts.",
    ap50: "Average precision over every confidence, a box counting at IoU 0.5.", ap50_95: "Average precision over IoU 0.5 to 0.95: rewards tight boxes too.",
    roc_auc: "How well a frame's top confidence separates frames with an object from empty ones (0.5 = chance).",
    first_load: "Compile (or cache load) plus warm-up of this model on this device when it starts.",
    cpu: "Detections compared image by image with the CPU device's, at this device's served confidence: catches a broken device path.",
    burst: "p95 under a burst of cameras triggering together (the burst test): each request timed from when it was due, queue wait included.",
    last_answer: "Burst test: from a trigger's first image being due to its last answer (p95 over the triggers), what Blue Iris waits for all of a trigger's answers.",
    low_confidence: "Too little data to cross-validate (fewer than 4 camera days or 50 labelled objects): the figures are in-sample and the interval is wide."
  };
  function tipText(k) { return TERMS[k] || ""; }

  var store = {
    get: function (k, dflt) { try { var v = root.localStorage.getItem("prism.bench." + k); return v === null ? dflt : JSON.parse(v); } catch (e) { return dflt; } },
    set: function (k, v) { try { root.localStorage.setItem("prism.bench." + k, JSON.stringify(v)); } catch (e) { /* no storage: the choice lasts for this page */ } }
  };

  root.BenchCore = {
    has: has, fmt: fmt, GOALS: GOALS, TUNE_FOR: TUNE_FOR, goalOf: goalOf, parseGoal: parseGoal, goalSpec: goalSpec,
    targetProblem: targetProblem, goalPairs: goalPairs, tryParams: tryParams, withTry: withTry,
    withoutTry: withoutTry, rowChips: rowChips, COLUMNS: COLUMNS, columnsOf: columnsOf,
    columnValue: columnValue, agreementText: agreementText, loadText: loadText, firstLoadText: firstLoadText, comboStates: comboStates,
    sameDevice: sameDevice, cpuReference: cpuReference, estimate: estimate, burstSeconds: burstSeconds, burstFacts: burstFacts, mapHash: mapHash, disagree: disagree,
    compareCsv: compareCsv, rangeText: rangeText, placeLabels: placeLabels, labelSize: labelSize, placeBoxLabels: placeBoxLabels, cellTag: cellTag, latest: latest, labelsText: labelsText, rescoredText: rescoredText, vsBestText: vsBestText, cvText: cvText, cvFact: cvFact, TERMS: TERMS, store: store
  };
  root.BenchTags = {cellTag: cellTag};
  root.BenchPoll = {runUri: runUri, step: step};
  var PB = root.PrismBench = root.PrismBench || {};
  PB.tipText = tipText; PB.has = has;
  PB.fmt = {ms: fmt.ms}; // bench-charts.js

  // ---- document helpers (called in a browser only) -----------------------------------------------
  // el(tag, cls, text) or el(tag, {attr: value, text, class}, children).
  PB.el = function (tag, a, b) {
    var e = document.createElement(tag);
    if (a && typeof a === "object") {
      Object.keys(a).forEach(function (k) {
        var v = a[k];
        if (v === null || v === undefined || v === false) return;
        if (k === "text") e.textContent = v; else if (k === "class") e.className = v; else e.setAttribute(k, v === true ? "" : v);
      });
      (b || []).forEach(function (c) { if (c !== null && c !== undefined && c !== false) e.appendChild(typeof c === "object" ? c : document.createTextNode(String(c))); });
    } else {
      if (a) e.className = a;
      if (b !== undefined && b !== null) e.textContent = b;
    }
    return e;
  };
  // A `button.term` with a tooltip (PrismTip): `key` from TERMS, else `text` itself.
  PB.term = function (label, keyOrText) {
    var b = document.createElement("button"); b.type = "button"; b.className = "term";
    b.textContent = label;
    if (TERMS[keyOrText]) b.setAttribute("data-term", keyOrText); else b.setAttribute("data-tip", keyOrText);
    b.setAttribute("aria-expanded", "false");
    return b;
  };
  function why(r) {
    return r.status === 403 ? " (refused: admin actions must come from this server's own HTTPS pages)" : r.status === 404 ? "" : r.status >= 500 ? " (see the Logs page)" : "";
  }
  // A request given up by its caller (AbortController): before the headers or while the body is
  // read. The caller ignores it (a newer request took its place).
  function aborted(e) { return !!(e && e.name === "AbortError"); }
  var ABORTED = {success: false, aborted: true, _status: 0, message: "cancelled"};
  function answer(r) {
    return r.json().then(function (j) { if (j && typeof j === "object") j._status = r.status; return j; }, function (e) {
      if (aborted(e)) return Object.assign({}, ABORTED);
      return {success: false, _status: r.status, message: "The server answered HTTP " + r.status + why(r) + "."};
    });
  }
  function offline() { return {success: false, _status: 0, message: "No answer from the server: check that it is running, then try again."}; }
  PB.post = function (url, body) {
    return fetch(url, {method: "POST", body: body, headers: {"Accept": "application/json"}}).then(answer, offline);
  };
  PB.getJSON = function (url, init) {
    return fetch(url, Object.assign({cache: "no-store", headers: {"Accept": "application/json"}}, init || {})).then(answer, function (e) {
      return aborted(e) ? Object.assign({}, ABORTED) : offline();
    });
  };
})(typeof globalThis !== "undefined" ? globalThis : window);
